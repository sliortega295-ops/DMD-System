"""FSDP2 layouts for the same semantic Program, not a model-specific launcher.

Declared trainable roots and explicit immutable forward roots are sharded.
Other frozen-only models stay resident. Cross-root immutable aliasing is not combined with this capability.
Initial dense optimizer moments are redistributed, never reset. Current CPU
execution is tested on Torch 2.10; CUDA source paths require separate validation.
"""
from __future__ import annotations
from collections import defaultdict
from copy import deepcopy
from dataclasses import replace
import inspect
import hashlib
import uuid
import ast
from pathlib import Path
from weakref import WeakKeyDictionary
import torch
import torch.distributed as dist
from distillschedule.core.contracts import require
from .replicated import ReplicaLifecycle, checked, gather, identity, forward_order_admitted, BACKEND
from .replica_contract import execution_graph, shard_boundary_modules
from .torch_backend import storage_key, physical_tensor, signature, unwrap_module
from distillschedule.workloads.best_known.registry import fingerprint

MODES = ('fsdp2_reshard', 'fsdp2_keep_forward')
HSDP_MODES = ('hsdp2_reshard', 'hsdp2_keep_forward')
_CONSTRUCTION_HANDOFFS = WeakKeyDictionary()
_CONSTRUCTION_MESHES = WeakKeyDictionary()


def frozen_construction_mesh():
    """One exact mesh object per live group, including sequential rebuilds.

    Native DTensor caches may retain an equivalent mesh's original object.
    Recreating equal-but-distinct meshes would break R82's identity check;
    reuse the actual construction mesh rather than weakening that check.
    """
    mesh = _CONSTRUCTION_MESHES.get(dist.group.WORLD)
    flags = gather(mesh is not None)
    require(all(flag == flags[0] for flag in flags), 'rank construction mesh lifetimes differ', path='open.fsdp')
    if mesh is None:
        from torch.distributed.device_mesh import init_device_mesh
        mesh = init_device_mesh('cpu', (dist.get_world_size(),))
        _CONSTRUCTION_MESHES[dist.group.WORLD] = mesh
    checked(lambda: require(mesh.device_type == 'cpu' and mesh.ndim == 1
            and mesh.mesh.tolist() == list(range(dist.get_world_size())),
            'construction mesh changed during group lifetime', path='open.fsdp'),
            'construction cached mesh integrity')
    return mesh


class _FrozenRootHandoff:
    """One newly materialized frozen child, adopted once by the native owner.

    The issuer must construct the native child through load(), not submit a
    caller module. Identity checks are local and precede tensor collectives.
    This is an ownership capability, not an authentication/security boundary.
    """
    def __init__(self, mesh, option):
        self.mesh = self.mesh_owner = mesh
        self.option = deepcopy(option)
        self.option_digest = fingerprint(self.option)
        self.group = dist.group.WORLD
        self.mesh_group = mesh.get_group()
        self.mesh_tensor = mesh.mesh
        self.ranks = tuple(mesh.mesh.tolist())
        self.native = self.root = self.session = None
        self.status = 'new'
        self.handle = None
        self.cohort = gather(uuid.uuid4().hex if dist.get_rank() == 0 else None)[0]

    @staticmethod
    def _local_identity(parameter):
        local = physical_tensor(parameter)
        return (tuple(parameter.shape), parameter.dtype, tuple(local.shape),
                tuple(parameter.stride()), tuple(local.stride()),
                local.untyped_storage().data_ptr(),
                hashlib.sha256(local.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())

    def load(self, cls, base, spec, *, config_transform=None):
        checked(lambda: require(self.status == 'new',
            'frozen construction can load only one new root', path='open.fsdp'), 'frozen construction load ownership')
        from distillschedule.training.flow_image import _load_native_frozen_shards, _native_shard_constants
        self.status = 'loading'
        try:
            self.native, config = _load_native_frozen_shards(cls, base, spec, self.mesh,
                                                            config_transform=config_transform)
            from torch.distributed.fsdp import FSDPModule, fully_shard
            def bind_loaded():
                require(type(self.native).__bases__ == (FSDPModule, cls)
                        and FSDPModule._get_fsdp_state(self.native) is fully_shard.state(self.native),
                        'unexpected generated native FSDP class/state', path='open.fsdp')
                self.native_class, self.generated_class = cls, type(self.native)
                self.native_mro, self.native_forward = type(self.native).__mro__, type(self.native).forward
                self.native_state = fully_shard.state(self.native)
                self.loaded_parameters = {name: (p, self._local_identity(p))
                                          for name, p in self.native.named_parameters(remove_duplicate=False)}
                self.native_constants = []
                for child, name, buffer, _ in _native_shard_constants(self.native):
                    value = child._buffers[name] if buffer else vars(child)[name]
                    self.native_constants.append((child, name, buffer, value, self._local_identity(value)))
                self.status = 'loaded'
                # Installed after the native hook, prepended before ANY all-gather.
                self.handle = self.native.register_forward_pre_hook(self._guard, prepend=True)
            checked(bind_loaded, 'frozen construction native class/payload binding')
            return self.native, config
        except BaseException:
            self.close()
            raise

    def _guard(self, module, inputs):
        if self.status != 'installed':
            self.status = 'expired'
            raise ValueError('construction-owned frozen root cannot forward before handoff or after close')

    @staticmethod
    def _method(root, name):
        method = inspect.unwrap(getattr(root, name, None))
        return getattr(method, '__func__', method)

    @staticmethod
    def _forward_binding(module):
        """Capture the issued module's actual callable and instance slot.

        Bound method objects are recreated on access, so retain their exact
        receiver/function instead. Native outer FSDP installation may change
        the class, but must not replace the original forward callable.
        """
        forward = module.forward
        return (type(module).forward, 'forward' in vars(module), vars(module).get('forward'),
                getattr(forward, '__self__', None), getattr(forward, '__func__', forward))

    def bind(self, root, *, key, child_path, method):
        from torch.distributed.fsdp import FSDPModule
        from torch.distributed.tensor import DTensor
        require(self.status == 'loaded' and isinstance(root, torch.nn.Module)
                and not isinstance(root, FSDPModule) and root.get_submodule(child_path) is self.native,
                'handoff requires its newly constructed child and an unsharded outer root', path='open.fsdp')
        slots = dict(root.named_parameters(remove_duplicate=False))
        require(slots and all(isinstance(p, DTensor) and not p.requires_grad and p.grad is None for p in slots.values())
                and {id(p) for p in slots.values()} == {id(p) for p in self.native.parameters()},
                'frozen construction cannot adopt trainable, LoRA or foreign root parameters', path='open.fsdp')
        require(inspect.ismethod(getattr(root, method, None)) and method not in vars(root)
                and not hasattr(FSDPModule, method), 'outer method must be owned by its model class', path='open.fsdp')
        self.root, self.key, self.child_path, self.method = root, key, child_path, method
        self.root_class = type(root)
        self.method_function = self._method(root, method)
        self.modules = dict(root.named_modules())
        self.forward_bindings = {name: self._forward_binding(module) for name, module in self.modules.items()}
        self.training_flags = {name: module.training for name, module in self.modules.items()}
        self.parameters = slots
        self.buffers = dict(root.named_buffers(remove_duplicate=False))
        self.buffer_layouts = {name: self._local_identity(value) for name, value in self.buffers.items()}
        self.status = 'bound'
        require(root not in _CONSTRUCTION_HANDOFFS, 'root already has a construction owner', path='open.fsdp')
        _CONSTRUCTION_HANDOFFS[root] = self

    def validate(self, state, semantics, option, *, session=None):
        from torch.distributed.fsdp import FSDPModule
        from torch.distributed.tensor import DTensor, Shard
        require(self.status in ('bound', 'installed') and _CONSTRUCTION_HANDOFFS.get(self.root) is self
                and state.get(self.key) is self.root and option == self.option
                and fingerprint(self.option) == self.option_digest and self.mesh is self.mesh_owner
                and dist.group.WORLD is self.group and self.mesh.get_group() is self.mesh_group
                and self.mesh.mesh.untyped_storage().data_ptr() == self.mesh_tensor.untyped_storage().data_ptr()
                and tuple(self.mesh.mesh.tolist()) == self.ranks
                and self.ranks == tuple(range(dist.get_world_size())),
                'frozen construction handoff expired or root/mesh/implementation changed', path='open.fsdp')
        require(self.status == 'bound' or self.session is session,
                'frozen construction root already belongs to another session', path='open.fsdp')
        require((self.status == 'bound' and type(self.root) is self.root_class)
                or (self.status == 'installed' and type(self.root) is self.root_generated
                    and type(self.root).__bases__ == (FSDPModule, self.root_class)
                    and type(self.root).__mro__ == self.root_mro
                    and FSDPModule._get_fsdp_state(self.root) is self.root_state),
                'construction outer class/MRO/native state changed', path='open.fsdp')
        require(self.native._native_shard_load_status == 'ready'
                and self.root.get_submodule(self.child_path) is self.native
                and isinstance(self.native, FSDPModule)
                and self._method(self.root, self.method) is self.method_function,
                'frozen construction child or outer method was rebound', path='open.fsdp')
        from torch.distributed.fsdp import fully_shard
        require(type(self.native) is self.generated_class and type(self.native).__mro__ == self.native_mro
                and type(self.native).__bases__ == (FSDPModule, self.native_class)
                and type(self.native).forward is self.native_forward
                and FSDPModule._get_fsdp_state(self.native) is self.native_state is fully_shard.state(self.native),
                'construction native FSDP class/MRO/method/state identity changed', path='open.fsdp')
        require(semantics.forward_interface == 'registered_methods'
                and set(semantics.frozen_module_bindings.values()) == {self.key}
                and all(semantics.forward_methods[binding] == self.method
                        for binding, key in semantics.frozen_module_bindings.items() if key == self.key)
                and tuple(semantics.shard_boundaries.get(self.key, ())) == (self.child_path,),
                'frozen construction requires its exact registered root/child boundary', path='open.fsdp')
        actual = dict(self.root.named_parameters(remove_duplicate=False))
        require(actual.keys() == self.parameters.keys() and all(actual[name] is p and isinstance(p, DTensor)
                and not p.requires_grad and p.grad is None and p.device_mesh is self.mesh
                and len(p.placements) == 1 and type(p.placements[0]) is Shard and p.placements[0].dim == 0
                for name, p in self.parameters.items()), 'frozen construction Parameter/mesh identity changed', path='open.fsdp')
        loaded = dict(self.native.named_parameters(remove_duplicate=False))
        require(loaded.keys() == self.loaded_parameters.keys() and all(loaded[name] is p
                and self._local_identity(p) == layout for name, (p, layout) in self.loaded_parameters.items()),
                'frozen construction local payload/storage changed after loading', path='open.fsdp')
        require(dict(self.root.named_modules()).keys() == self.modules.keys()
                and all(self.root.get_submodule(name) is module for name, module in self.modules.items())
                and all(module.training == self.training_flags[name] for name, module in self.modules.items())
                and dict(self.root.named_buffers(remove_duplicate=False)).keys() == self.buffers.keys()
                and all(self.root.get_buffer(name) is tensor for name, tensor in self.buffers.items()),
                'frozen construction module/buffer objects changed', path='open.fsdp')
        require(all(all(actual is original for actual, original in
                        zip(self._forward_binding(module), self.forward_bindings[name]))
                    for name, module in self.modules.items()),
                'frozen construction module forward binding/instance slot changed', path='open.fsdp')
        require(all(self._local_identity(value) == self.buffer_layouts[name]
                    for name, value in self.buffers.items())
                and all((child._buffers.get(name) if buffer else vars(child).get(name)) is value
                        and self._local_identity(value) == layout
                        for child, name, buffer, value, layout in self.native_constants),
                'frozen construction constant values/layout changed after loading', path='open.fsdp')

    def installed(self, session):
        from torch.distributed.fsdp import FSDPModule
        require(self.status == 'bound', 'frozen construction can be adopted only once', path='open.fsdp')
        self.root_generated, self.root_mro = type(self.root), type(self.root).__mro__
        require(type(self.root).__bases__ == (FSDPModule, self.root_class),
                'unexpected generated outer FSDP class', path='open.fsdp')
        self.root_state = FSDPModule._get_fsdp_state(self.root)
        self.session, self.status = session, 'installed'

    def close(self):
        if self.root is not None and _CONSTRUCTION_HANDOFFS.get(self.root) is self:
            del _CONSTRUCTION_HANDOFFS[self.root]
        self.status = 'closed'
        if self.native is not None and self.handle is None:
            self.handle = self.native.register_forward_pre_hook(self._guard, prepend=True)
        # Keep the noncollective guard on expired objects. No parameter,
        # optimizer, EMA or already-published tensor is rebound or poisoned.


def construction_handoffs(workload, option, *, copy_state=False, session=None):
    """Collectively reject foreign/asymmetric owners BEFORE Session signatures."""
    def local():
        from torch.distributed.tensor import DTensor
        semantics = workload.replica_semantics
        keys = set(semantics.forward_module_bindings.values()) if semantics else set()
        owners = {key: _CONSTRUCTION_HANDOFFS[workload.state[key]] for key in keys
                  if workload.state[key] in _CONSTRUCTION_HANDOFFS}
        distributed_roots = {key for key, value in workload.state.items() if isinstance(value, torch.nn.Module)
                             and any(isinstance(p, DTensor) for p in value.parameters())}
        require(distributed_roots <= owners.keys(), 'DENSE_CONTIGUOUS_PARAMETER_FACTORY_REQUIRED', path='open.fsdp')
        require(not owners or not copy_state, 'construction-owned roots cannot be cloned for trials', path='open.fsdp')
        for owner in owners.values():
            owner.validate(workload.state, semantics, option, session=session)
        require(rejection(workload, construction_owners=owners) is None,
                str(rejection(workload, construction_owners=owners)), path='open.fsdp')
        return owners
    owners = checked(local, 'FSDP construction handoff admission')
    descriptors = gather({key: dict(cohort=owner.cohort, child=owner.child_path, method=owner.method,
        ranks=list(owner.ranks), layout={name: (list(p.shape), str(p.dtype)) for name, p in owner.parameters.items()})
        for key, owner in owners.items()})
    require(all(value == descriptors[0] for value in descriptors),
            'ranks differ in frozen construction handoff', path='open.fsdp')
    return owners


def manual_construction_handoff(workload, option):
    """Skip unused inventory ONLY after a real live native-shard certificate."""
    from torch.distributed.tensor import DTensor
    present = checked(lambda: any(isinstance(value, torch.nn.Module)
        and any(isinstance(p, DTensor) for p in value.parameters()) for value in workload.state.values()),
        'manual construction layout')
    flags = gather(present)
    require(all(flag == flags[0] for flag in flags), 'ranks differ in manual construction layout', path='open.fsdp')
    if not present:
        return False
    return bool(construction_handoffs(workload, option))


def construction_model_code(workload, classes):
    """Source-less native subclasses require the LIVE construction certificate."""
    from torch.distributed.fsdp import FSDPModule
    from .torch_backend import code_signature
    certified = {}
    for value in workload.state.values():
        if not isinstance(value, torch.nn.Module):
            continue
        owner = _CONSTRUCTION_HANDOFFS.get(value)
        if owner is None:
            continue
        owner.validate(workload.state, workload.replica_semantics, owner.option, session=owner.session)
        certified[owner.generated_class] = owner.native_class
    generated = {cls for cls in classes if issubclass(cls, FSDPModule)}
    require(generated <= certified.keys(),
            'opaque/foreign FSDP model classes require a live construction certificate', path='open.fsdp')
    actual = (classes - generated) | {certified[cls] for cls in generated}
    result = {cls.__module__ + '.' + cls.__qualname__: code_signature(cls) for cls in actual}
    # Torch exports FSDPModule under a public module that may not contain its
    # definition. Resolve its REAL method source, verify that module's class
    # object, then extract the class AST. Never catch arbitrary source failures.
    method = FSDPModule.reshard
    require(method.__globals__.get('FSDPModule') is FSDPModule,
            'FSDP source method is not bound to the native class', path='open.fsdp')
    source = Path(inspect.getsourcefile(method)).read_text()
    nodes = [node for node in ast.parse(source).body if isinstance(node, ast.ClassDef) and node.name == 'FSDPModule']
    require(len(nodes) == 1, 'exact native FSDP class source required', path='open.fsdp')
    result[FSDPModule.__module__ + '.' + FSDPModule.__qualname__] = fingerprint(
        {'source': ast.get_source_segment(source, nodes[0])})
    return result


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


def rejection(workload, module_keys=None, *, construction_owners=None):
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
        owner = (construction_owners or {}).get(key)
        owned = owner is not None and _CONSTRUCTION_HANDOFFS.get(workload.state[key]) is owner
        if any((isinstance(p,DTensor) and not owned) or p.ndim==0 or not p.is_contiguous() for p in params):
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
                parameter_shadows=None, shard_boundaries=None, construction_owners=None):
    """Common native installation; untouched optimizer groups/moments stay intact."""
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.tensor import DTensor, distribute_tensor
    from distillschedule.training.tensor_state import validate_parameter_shadows
    shadows = {} if parameter_shadows is None else parameter_shadows
    boundaries = {} if shard_boundaries is None else shard_boundaries
    require(not boundaries or forward_methods,
            'shard boundaries require registered outer forward methods', path='open.fsdp')
    children = shard_boundary_modules(session.state, boundaries, set(forward_methods or ()))
    require(set(children) <= set(module_keys), 'shard boundaries reference an uninstalled root', path='open.fsdp')
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
        # Capture original IDs before any child replacement, then migrate the
        # full root's optimizer and shadows exactly once after all groups exist.
        for _, child in children.get(key, ()):
            if key not in (construction_owners or {}):
                fully_shard(child, mesh=mesh, reshard_after_forward=reshard_after_forward)
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


def close_sharded_forward(root):
    """Release actual native groups inside one owned root, including the root.

    ``reshard`` is not recursive. Reverse module order closes nested children
    before their parent; unrelated roles and unsharded modules are untouched.
    This also closes keep-forward inference when no backward will follow.
    """
    from torch.distributed.fsdp import FSDPModule
    for module in reversed(tuple(root.modules())):
        if isinstance(module, FSDPModule):
            module.reshard()


class ShardedLifecycle(ReplicaLifecycle):
    def prepare(self,session):
        self.construction_owners = construction_handoffs(self.workload, self.option, session=session)
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
            require(rejection(self.workload, construction_owners=self.construction_owners) is None,
                    str(rejection(self.workload, construction_owners=self.construction_owners)),path='open.fsdp')
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
        elif self.construction_owners:
            topology = None
            mesh = next(iter(self.construction_owners.values())).mesh
        else:
            topology = None
            mesh = init_device_mesh(session.device.type,(self.world,))
        semantics = self.workload.replica_semantics
        methods = {semantics.forward_module_bindings[binding]: name for binding, name in semantics.forward_methods.items()}
        self.forward_interface = semantics.forward_interface
        self.forward_methods = methods
        self.forward_contract = semantics.to_dict()
        self.parameter_shadows = dict(semantics.parameter_shadows)
        self.shard_boundaries = {key: tuple(paths) for key, paths in semantics.shard_boundaries.items()}
        self.full_parameter_bytes, sharded_count, migrated = shard_roots(
            session, self.module_keys, mesh,
            reshard_after_forward=self.option['reduction'].endswith('_reshard'), forward_methods=methods,
            parameter_shadows=self.parameter_shadows, shard_boundaries=self.shard_boundaries,
            construction_owners=self.construction_owners)
        for owner in self.construction_owners.values():
            checked(lambda: owner.installed(session), 'frozen construction native adoption')
        if self.construction_owners:
            session.effects['construction_owned_frozen_roots'] = sorted(self.construction_owners)
        self.shadow_ownership = ParameterShadowOwnership(session, self.parameter_shadows)
        self.boundary_owners = {key: {path: (session.state[key].get_submodule(path),
            dict(session.state[key].get_submodule(path).named_parameters(remove_duplicate=False))) for path in paths}
            for key, paths in self.shard_boundaries.items()}
        if self.shard_boundaries:
            session.effects['fsdp_shard_boundaries'] = {key: list(paths) for key, paths in self.shard_boundaries.items()}
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
            for owner in self.construction_owners.values():
                owner.validate(session.state, self.workload.replica_semantics, self.option, session=session)
            semantics = self.workload.replica_semantics
            if semantics.frozen_module_bindings or semantics.shard_boundaries:
                semantics.validate(replace(self.workload, state=session.state, feeds=session.feeds))
            require(semantics.forward_interface == self.forward_interface and
                    semantics.to_dict() == self.forward_contract,
                    'installed FSDP method contract changed between cycles', path='open.fsdp')
            require(semantics.parameter_shadows == self.parameter_shadows,
                    'installed FSDP parameter shadow bindings changed between cycles', path='open.fsdp')
            for key, owners in self.boundary_owners.items():
                for path, (installed, parameters) in owners.items():
                    current = session.state[key].get_submodule(path)
                    actual = dict(current.named_parameters(remove_duplicate=False))
                    require(current is installed and actual.keys() == parameters.keys()
                            and all(actual[name] is parameter for name, parameter in parameters.items()),
                            'installed shard boundary object/parameter identity changed', path='open.fsdp')
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
        for key in self.module_keys:
            close_sharded_forward(session.state[key])
        session.synchronize()
        digests=gather(fingerprint(signature(session.state)))
        require(len(set(digests))==1,'FSDP model/optimizer state diverged',path='open.fsdp')
        session.effects.update(parameter_backward_calls=self.backward_count,
            replica_state_consensus=True,reduction_completed=True,
            gradient_collectives=None,communicated_gradient_bytes=None,
            collective_count_scope='native_FSDP_collectives_not_instrumented')
