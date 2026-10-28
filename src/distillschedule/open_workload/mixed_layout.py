"""Different native layouts for different declared model roots.

The factory supplies model interfaces, not the mapping chosen here. Existing
DDP hooks and FSDP installation/moment migration are reused. Calls remain serial
across the allocated data-parallel world; this is NOT a role-pool pipeline.
"""
from __future__ import annotations
import itertools
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from distillschedule.core.contracts import require
from distillschedule.workloads.best_known.registry import fingerprint
from .replicated import ReplicaLifecycle, checked, gather, identity, ddp_order_admitted, BACKEND
from .replica_contract import execution_graph
from .sharded import rejection, shard_roots, MODES, HSDP_MODES, ParameterShadowOwnership
from .torch_backend import signature, unwrap_module

MODE = 'per_module'


def domains(workload, topology=None):
    result = {}
    order = tuple(c.id for c in workload.program.calls) + ('__open_finish__',)
    for key in sorted(set(workload.replica_semantics.module_bindings.values())):
        values = ['post_backward']
        if ddp_order_admitted(workload, order, {key}):
            values.append('ddp_overlap')
            if rejection(workload, (key,)) is None:
                values.extend(MODES)
                if topology and topology['hsdp_admitted']: values.extend(HSDP_MODES)
        result[key] = tuple(values)
    return result


def enumerate_assignments(domain):
    """Prioritize one-root changes; then all other nonuniform assignments lazily."""
    keys = tuple(domain)
    if len(keys) < 2: return
    seen = set()
    for base in ('post_backward', 'ddp_overlap'):
        if not all(base in domain[key] for key in keys): continue
        for key in keys:
            for value in domain[key]:
                assignment = tuple(value if k == key else base for k in keys)
                if len(set(assignment)) > 1 and assignment not in seen:
                    seen.add(assignment); yield dict(zip(keys, assignment))
    for assignment in itertools.product(*(domain[k] for k in keys)):
        if len(set(assignment)) > 1 and assignment not in seen:
            yield dict(zip(keys, assignment))


def assignment_count(domain):
    if len(domain) < 2: return 0
    count = 1
    for values in domain.values(): count *= len(values)
    return count - len(set.intersection(*(set(values) for values in domain.values())))


def option(local, assignment, topology):
    result = {'backend':BACKEND, 'local':local, 'reduction':MODE,
              'bucket_cap_mb':128, 'module_layouts':assignment}
    if any(value in HSDP_MODES for value in assignment.values()):
        result['node_groups'] = topology['node_groups']
    return result


def graph(workload, implementation):
    context = {key for key, value in implementation['module_layouts'].items() if value != 'post_backward'}
    return execution_graph(workload.program, workload.replica_semantics, 'ddp_overlap', context)


class MixedLifecycle(ReplicaLifecycle):
    def prepare(self, session):
        rows = gather(self.option)
        require(all(x == rows[0] for x in rows), 'rank mixed layouts differ', path='open.mixed')
        from .topology import discover
        topology = discover()
        def validate():
            mapping = self.option.get('module_layouts')
            require(isinstance(mapping, dict), 'per-module layout mapping required', path='open.mixed')
            admitted = domains(self.workload, topology)
            hybrid = any(value in HSDP_MODES for value in mapping.values())
            keys = {'backend','local','reduction','bucket_cap_mb','module_layouts'} | ({'node_groups'} if hybrid else set())
            require(set(self.option) == keys and self.option['backend'] == BACKEND
                    and self.option['reduction'] == MODE and self.option['bucket_cap_mb'] == 128
                    and type(self.option['bucket_cap_mb']) is int
                    and not self.option['local']['share_immutable']
                    and set(mapping) == set(admitted) and len(set(mapping.values())) > 1
                    and all(value in admitted[key] for key,value in mapping.items()),
                    'unsupported mixed layout combination', path='open.mixed')
            require(not hybrid or self.option['node_groups'] == topology['node_groups'],
                    'mixed HSDP topology changed', path='open.mixed')
        checked(validate, 'mixed layout admission'); identity(self.workload)
        self.mapping = dict(self.option['module_layouts'])
        self.parameter_shadows = dict(self.workload.replica_semantics.parameter_shadows)
        self.module_keys = tuple(sorted(self.mapping))
        session.graph = checked(lambda:graph(self.workload,self.option), 'mixed native completion edges')
        self.reference = ReplicaLifecycle(self.workload, {'reduction':'post_backward'})
        self.reference.module_keys = tuple(key for key in self.module_keys if self.mapping[key] == 'post_backward')
        self.full_parameter_bytes = 0; self.bucket_storage_bound_bytes = 0
        migrated = count = 0
        from distillschedule.runtime.wan13b_fair import _DDPCommState, ddp_sum_then_divide_hook
        from torch.distributed.device_mesh import init_device_mesh, DeviceMesh
        meshes = {}
        for key in self.module_keys:
            mode = self.mapping[key]
            if mode == 'ddp_overlap':
                model = session.state[key]
                self.bucket_storage_bound_bytes += 2*sum(p.numel()*p.element_size() for p in model.parameters() if p.requires_grad)
                wrapped = DDP(model, device_ids=[session.device.index] if session.device.type=='cuda' else None,
                              process_group=dist.group.WORLD, broadcast_buffers=False, bucket_cap_mb=128,
                              find_unused_parameters=True, gradient_as_bucket_view=False)
                state = _DDPCommState(dist.group.WORLD,self.world)
                wrapped.register_comm_hook(state,ddp_sum_then_divide_hook)
                self.hooks.append(state); session.state[key] = wrapped
            elif mode in (*MODES, *HSDP_MODES):
                kind = 'hsdp' if mode in HSDP_MODES else 'fsdp'
                if kind not in meshes:
                    meshes[kind] = (DeviceMesh(session.device.type,torch.tensor(topology['node_groups']),
                                    mesh_dim_names=('replicate','shard')) if kind=='hsdp' else
                                    init_device_mesh(session.device.type,(self.world,)))
                size, number, moments = shard_roots(session, (key,), meshes[kind],
                    reshard_after_forward=mode.endswith('_reshard'),
                    parameter_shadows=self.parameter_shadows)
                self.full_parameter_bytes += size; count += number; migrated += moments
        self.shadow_ownership = ParameterShadowOwnership(session, self.parameter_shadows)
        session.effects.update(reduction=MODE,module_layouts=self.mapping,
            fsdp_sharded_parameters=count, optimizer_moments_migrated=migrated,
            world_size=self.world, allocated_topology=topology, cross_role_pipeline=False)

    def before_run(self, session, order):
        orders = gather(tuple(order))
        require(all(x == orders[0] for x in orders), 'rank execution orders differ', path='open.mixed')
        context = {key for key,value in self.mapping.items() if value != 'post_backward'}
        def validate_shadows():
            from distillschedule.training.tensor_state import validate_parameter_shadows
            require(self.workload.replica_semantics.parameter_shadows == self.parameter_shadows,
                    'installed parameter shadow bindings changed between cycles', path='open.mixed')
            self.shadow_ownership.validate(session)
            validate_parameter_shadows(session.state, self.parameter_shadows, self.module_keys, feeds=session.feeds)
        checked(validate_shadows, 'mixed parameter shadow admission')
        require(ddp_order_admitted(self.workload,order,context),
                'selected order overwrites a native reducer context', path='open.mixed')
        dist.barrier()

    def after_call(self, session, call):
        if call.id in self.workload.replica_semantics.backward_calls: self.backward_count += 1
        if self.reference.module_keys: self.reference.after_call(session,call)

    def finish(self, session):
        for key, mode in self.mapping.items():
            if mode in (*MODES, *HSDP_MODES): session.state[key].reshard()
        session.synchronize()
        digests = gather(fingerprint(signature({k:unwrap_module(v) for k,v in session.state.items()})))
        require(len(set(digests)) == 1, 'mixed-layout state diverged', path='open.mixed')
        self._drain_ddp_telemetry()
        session.effects.update(parameter_backward_calls=self.backward_count, replica_state_consensus=True,
            reduction_completed=True, gradient_collectives=None,
            collective_count_scope='mixed native collectives not fully instrumented',
            reference_collectives=self.reference.collectives,
            ddp_bucket_completions=self.collectives)
