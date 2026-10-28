"""FSDP2 layouts for the same semantic Program, not a model-specific launcher.

Declared trainable roots and explicit immutable forward roots are sharded.
Other frozen-only models stay resident. Cross-root immutable aliasing is not combined with this capability.
Initial dense optimizer moments are redistributed, never reset. Current CPU
execution is tested on Torch 2.10; CUDA source paths require separate validation.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import replace
import torch
import torch.distributed as dist
from distillschedule.core.contracts import require
from .replicated import ReplicaLifecycle, checked, gather, identity, forward_order_admitted, BACKEND
from .replica_contract import execution_graph
from .torch_backend import storage_key, physical_tensor, signature, unwrap_module
from distillschedule.workloads.best_known.registry import fingerprint

MODES = ('fsdp2_reshard', 'fsdp2_keep_forward')
HSDP_MODES = ('hsdp2_reshard', 'hsdp2_keep_forward')


class ParameterShadowOwnership:
    """Local identity guard for the explicitly installed native shadow owners."""
    def __init__(self, session, bindings):
        self.bindings = bindings
        self.installed = self._capture(session.state)
        self.workload = self._capture(session.workload.state)

    def _capture(self, state):
        return ({key:state[key] for key in self.bindings},
                {root:dict(unwrap_module(state[root]).named_parameters(remove_duplicate=False))
                 for root in set(self.bindings.values())})

    @staticmethod
    def _matches(state, owners):
        shadows, roots = owners
        if any(state.get(key) is not value for key, value in shadows.items()):
            return False
        for root, installed in roots.items():
            model = unwrap_module(state.get(root))
            if not isinstance(model, torch.nn.Module):
                return False
            current = dict(model.named_parameters(remove_duplicate=False))
            if current.keys() != installed.keys() or any(current[name] is not p for name, p in installed.items()):
                return False
        return True

    def validate(self, session):
        require(self._matches(session.state, self.installed),
                'installed parameter shadow session object/parameter identity changed', path='open.fsdp')
        if self._matches(session.workload.state, self.installed):
            # A copied trial may explicitly adopt its installed state on the
            # next bind_cycle. Do not retain the original dense trial owners.
            self.workload = self.installed
        else:
            require(self._matches(session.workload.state, self.workload),
                    'installed parameter shadow workload object/parameter identity changed', path='open.fsdp')


def rejection(workload, module_keys=None):
    if workload.replica_semantics is None:
        return 'REPLICA_SEMANTICS_REQUIRED'
    if workload.replica_semantics.forward_interface not in ('module_forward', 'registered_methods'):
        return 'PRIVATE_METHODS_REQUIRE_POST_BACKWARD'
    try:
        from torch.distributed.fsdp import fully_shard, FSDPModule
        from torch.distributed.tensor import DTensor
    except ImportError: return 'FSDP2_UNAVAILABLE'
    if workload.replica_semantics.forward_interface == 'registered_methods':
        try:
            workload.replica_semantics.validate(workload)
        except (ValueError, TypeError): return 'INVALID_FORWARD_METHOD_CONTRACT'
        try:
            from torch.distributed.fsdp import register_fsdp_forward_method
        except ImportError: return 'FSDP_FORWARD_METHOD_API_UNAVAILABLE'
        if any(hasattr(FSDPModule, name) for name in workload.replica_semantics.forward_methods.values()):
            return 'FORWARD_METHOD_COLLIDES_WITH_FSDP_API'
    version=tuple(int(v) for v in torch.__version__.split('+')[0].split('.')[:2])
    if workload.device.type=='cpu' and version<(2,10):return 'CPU_FSDP2_REQUIRES_TORCH_2_10'
    keys=set(workload.replica_semantics.forward_module_bindings.values()) if module_keys is None else set(module_keys)
    for key in keys:
        params=list(workload.state[key].parameters())
        if len({p.dtype for p in params})!=1:return 'MIXED_PARAMETER_DTYPES_REQUIRE_SEPARATE_GROUPING'
        if any(isinstance(p,DTensor) or p.ndim==0 or not p.is_contiguous() for p in params):
            return 'DENSE_CONTIGUOUS_PARAMETER_FACTORY_REQUIRED'
    seen={}
    for key,value in workload.state.items():
        if not isinstance(value,torch.nn.Module):continue
        for p in value.parameters():
            storage=storage_key(p)
            if storage in seen and seen[storage]!=key and (key in keys or seen[storage] in keys):
                return 'CROSS_ROOT_ALIASES_NEED_A_SHARED_FSDP_GROUP'
            seen[storage]=key
    for value in workload.state.values():
        if not isinstance(value,torch.optim.Optimizer):continue
        if type(value) not in (torch.optim.Adam,torch.optim.AdamW,torch.optim.SGD):
            return 'OPTIMIZER_STATE_LAYOUT_NOT_SUPPORTED'
        for p,state in value.state.items():
            if any(isinstance(t,torch.Tensor) and t.ndim!=0 and t.shape!=p.shape for t in state.values()):
                return 'NON_POINTWISE_OPTIMIZER_STATE'
    return None


def shard_roots(session, module_keys, mesh, *, reshard_after_forward, forward_methods=None,
                parameter_shadows=None):
    """Common native installation; untouched optimizer groups/moments stay intact."""
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.tensor import DTensor, distribute_tensor
    from distillschedule.training.tensor_state import validate_parameter_shadows
    shadows = {} if parameter_shadows is None else parameter_shadows
    validate_parameter_shadows(session.state, shadows,
                               {key for key, value in session.state.items() if isinstance(value, torch.nn.Module)})
    # A leaf/subgroup installation cannot silently migrate only part of a full
    # root's shadow map. Existing role-pool callers omit this opt-in entirely.
    for root in shadows.values():
        owned = {id(p) for p in session.state[root].parameters()}
        require(all(key == root or not any(id(p) in owned for p in session.state[key].parameters())
                    for key in module_keys),
                'parameter shadows require whole-root sharding, not aliased subroots', path='open.fsdp')
    replacements = {}; parameter_bytes = 0; count = 0
    for key in module_keys:
        model = session.state[key]
        original = dict(model.named_parameters())
        parameter_bytes += sum(p.numel()*p.element_size() for p in original.values())
        fully_shard(model, mesh=mesh, reshard_after_forward=reshard_after_forward)
        if forward_methods and key in forward_methods:
            from torch.distributed.fsdp import register_fsdp_forward_method
            # Only the declared outer entrypoint is hooked once per root. A
            # rollout may call denoise internally; registering both would nest
            # pre/post hooks and reshard the parameters before rollout finishes.
            register_fsdp_forward_method(model, forward_methods[key])
        for name, parameter in model.named_parameters():
            require(isinstance(parameter, DTensor), 'FSDP did not install distributed parameters', path='open.fsdp')
            replacements[id(original[name])] = parameter
            count += 1
        for shadow_key, owner in sorted(shadows.items()):
            if owner != key: continue
            # Preserve the explicit map shared by workload/component/session
            # owners. Replacing only session.state would retain a dense copy
            # in the original owner and break subsequent persistent rebinds.
            for name, parameter in model.named_parameters():
                session.state[shadow_key][name] = distribute_tensor(
                    session.state[shadow_key][name], parameter.device_mesh, parameter.placements)
    migrated = 0
    for optimizer in session.state.values():
        if not isinstance(optimizer, torch.optim.Optimizer): continue
        state = defaultdict(dict)
        for old, values in optimizer.state.items():
            if id(old) not in replacements:
                state[old] = values
                continue
            new = replacements[id(old)]; state[new] = {}
            for name, value in values.items():
                if isinstance(value, torch.Tensor) and value.ndim > 0:
                    state[new][name] = distribute_tensor(value.detach().to(session.device),
                                                        new.device_mesh, new.placements)
                    migrated += 1
                else: state[new][name] = value
        optimizer.state = state
        for group in optimizer.param_groups:
            group['params'] = [replacements.get(id(p), p) for p in group['params']]
    return parameter_bytes, count, migrated


class ShardedLifecycle(ReplicaLifecycle):
    def prepare(self,session):
        options=gather(self.option)
        require(all(x==options[0] for x in options),'ranks selected different implementations',path='open.fsdp')
        hybrid = self.option['reduction'] in HSDP_MODES
        def validate():
            keys = {'backend','local','reduction','bucket_cap_mb'} | ({'node_groups'} if hybrid else set())
            require(set(self.option)==keys
                    and self.option['backend']==BACKEND and self.option['reduction'] in (*MODES, *HSDP_MODES)
                    and type(self.option['bucket_cap_mb']) is int and self.option['bucket_cap_mb']==0
                    and not self.option['local']['share_immutable'],
                    'unadmitted sharded combination',path='open.fsdp')
            require(rejection(self.workload) is None, str(rejection(self.workload)),path='open.fsdp')
        checked(validate,'FSDP admission'); identity(self.workload)
        self.module_keys=tuple(sorted(set(self.workload.replica_semantics.forward_module_bindings.values())))
        session.graph=checked(lambda:execution_graph(self.workload.program,
            self.workload.replica_semantics,'fsdp_forward'),'FSDP call boundaries')
        from torch.distributed.device_mesh import init_device_mesh, DeviceMesh
        if hybrid:
            from .topology import discover
            topology = discover()
            require(topology['hsdp_admitted'] and self.option['node_groups'] == topology['node_groups'],
                    'HSDP plan does not match current allocated nodes', path='open.fsdp')
            mesh = DeviceMesh(session.device.type, torch.tensor(topology['node_groups']),
                              mesh_dim_names=('replicate', 'shard'))
        else:
            topology = None
            mesh = init_device_mesh(session.device.type,(self.world,))
        semantics = self.workload.replica_semantics
        methods = {semantics.forward_module_bindings[binding]: name for binding, name in semantics.forward_methods.items()}
        self.forward_interface = semantics.forward_interface
        self.forward_methods = methods
        self.forward_contract = semantics.to_dict()
        self.parameter_shadows = dict(semantics.parameter_shadows)
        self.full_parameter_bytes, sharded_count, migrated = shard_roots(
            session, self.module_keys, mesh,
            reshard_after_forward=self.option['reduction'].endswith('_reshard'), forward_methods=methods,
            parameter_shadows=self.parameter_shadows)
        self.shadow_ownership = ParameterShadowOwnership(session, self.parameter_shadows)
        if methods: session.effects['fsdp_forward_methods'] = methods
        if self.parameter_shadows: session.effects['parameter_shadows'] = self.parameter_shadows
        session.effects.update(distributed=True,world_size=self.world,reduction=self.option['reduction'],
            fsdp_sharded_parameters=sharded_count,optimizer_moments_migrated=migrated,
            parameter_layout='HSDP2_Replicate_Shard0' if hybrid else 'FSDP2_Shard0',
            mesh_shape=list(mesh.shape), allocated_topology=topology, frozen_only_models='resident',
            reshard_after_forward=self.option['reduction'].endswith('_reshard'))
        if semantics.frozen_module_bindings:
            frozen_roots = set(semantics.frozen_module_bindings.values())
            resident = sorted(key for key, value in session.state.items() if isinstance(value, torch.nn.Module)
                              and not any(p.requires_grad for p in value.parameters()) and key not in frozen_roots)
            session.effects['frozen_only_models'] = {'sharded': sorted(frozen_roots), 'resident': resident}

    def before_run(self,session,order):
        orders=gather(tuple(order))
        require(all(x==orders[0] for x in orders),'rank execution orders differ',path='open.fsdp')
        def validate():
            semantics = self.workload.replica_semantics
            if semantics.frozen_module_bindings:
                semantics.validate(replace(self.workload, state=session.state, feeds=session.feeds))
            require(semantics.forward_interface == self.forward_interface and
                    semantics.to_dict() == self.forward_contract,
                    'installed FSDP method contract changed between cycles', path='open.fsdp')
            require(semantics.parameter_shadows == self.parameter_shadows,
                    'installed FSDP parameter shadow bindings changed between cycles', path='open.fsdp')
            self.shadow_ownership.validate(session)
            from distillschedule.training.tensor_state import validate_parameter_shadows
            validate_parameter_shadows(session.state, self.parameter_shadows, self.module_keys, feeds=session.feeds)
            require(forward_order_admitted(self.workload,order),
                    'unsupported outstanding FSDP forward',path='open.fsdp')
        checked(validate, 'FSDP cycle admission')
        dist.barrier()

    def after_call(self,session,call):
        if call.id in self.workload.replica_semantics.backward_calls:self.backward_count+=1

    def finish(self,session):
        # Inference-only final calls may retain full parameters in keep-forward
        # mode. Complete the plan with sharded state before inspecting moments.
        for key in self.module_keys:session.state[key].reshard()
        session.synchronize()
        digests=gather(fingerprint(signature(session.state)))
        require(len(set(digests))==1,'FSDP model/optimizer state diverged',path='open.fsdp')
        session.effects.update(parameter_backward_calls=self.backward_count,
            replica_state_consensus=True,reduction_completed=True,
            gradient_collectives=None,communicated_gradient_bytes=None,
            collective_count_scope='native_FSDP_collectives_not_instrumented')
