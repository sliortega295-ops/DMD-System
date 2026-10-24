"""Executable, name-independent replicated training capabilities.

Reference: post-backward parameter SUM/world_size. Alternative: PyTorch DDP
with the EXISTING DMD-System asynchronous SUM/divide hook. Only parameters are
averaged; input/latent gradients remain on their original autograd path. Each
rank executes the same program with its own feeds. No FSDP/pool claim is made.
"""
from __future__ import annotations
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from distillschedule.core.contracts import require
from distillschedule.workloads.best_known.registry import fingerprint
from .torch_backend import Session, candidates as local_candidates, signature, tensors, unwrap_module

BACKEND = 'torch-callable-replicated-v1'
_CONTROL_GROUP = ContextVar('distillschedule_control_group', default=None)


@contextmanager
def control_plane(group):
    """Route only metadata through an explicitly owned full-world Gloo group.

    Tensor/FSDP/reducer collectives retain their original NCCL group, algorithm
    and math. This avoids sending Python objects through a Tree-only NCCL path.
    """
    require(dist.is_initialized() and group is not None and dist.get_backend(group) == 'gloo'
            and dist.get_world_size(group) == dist.get_world_size()
            and dist.get_rank(group) == dist.get_rank(),
            'control plane requires matching full-world Gloo rank order', path='open.replicated')
    token = _CONTROL_GROUP.set(group)
    try:
        yield
    finally:
        _CONTROL_GROUP.reset(token)


def gather(value):
    require(dist.is_available() and dist.is_initialized() and dist.get_world_size() >= 2,
            'explicit initialized group with at least two ranks required', path='open.replicated')
    rows = [None] * dist.get_world_size()
    dist.all_gather_object(rows, value, group=_CONTROL_GROUP.get())
    return rows


def broadcast(value):
    box = [value]
    dist.broadcast_object_list(box, src=0, group=_CONTROL_GROUP.get())
    return box[0]


def metadata_timeout_seconds():
    """Actual control-group timeout, including externally owned groups.

    PyTorch exposes this on the backend options, not the configured trainer.
    Read the metadata group's backend so an NCCL data timeout cannot mask a
    shorter Gloo control timeout. No group or device is created here.
    """
    group = _CONTROL_GROUP.get()
    if group is None:
        group = dist.group.WORLD
    require(dist.is_initialized() and group is not None,
            'initialized metadata group required', path='open.replicated')
    backend = str(dist.get_backend(group))
    require(backend in ('gloo', 'nccl'), 'unsupported metadata backend', path='open.replicated')
    device = torch.device('cpu' if backend == 'gloo' else 'cuda')
    native = group._get_backend(device)
    if dist.is_gloo_available():
        # DETAIL wraps the backend on Torch 2.10; only the underlying real
        # backend exposes its configured options. Do not guess a timeout.
        from torch.distributed.distributed_c10d import _ProcessGroupWrapper
        while isinstance(native, _ProcessGroupWrapper):
            native = native.wrapped_pg
    seconds = native.options._timeout.total_seconds()
    require(0 < seconds < float('inf'), 'invalid metadata-group timeout', path='open.replicated')
    return seconds


def checked(fn: Callable, label: str):
    """Make preflight/validation failures visible to every rank before collectives."""
    value = None
    try:
        value = fn(); error = None
    except Exception as exc:
        error = type(exc).__name__ + ': ' + str(exc)
    errors = gather(error)
    require(not any(errors), f'{label}: rank failures {errors}', path='open.replicated')
    return value


def identity(workload):
    def local():
        workload.validate()
        require(workload.replica_semantics is not None, 'replica semantics are required', path='open.replicated')
        backend = str(dist.get_backend())
        require((workload.device.type == 'cpu' and backend == 'gloo') or
                (workload.device.type == 'cuda' and backend == 'nccl'),
                'use CPU/Gloo or CUDA/NCCL with one device per rank', path='open.replicated')
        if workload.device.type == 'cuda':
            require(workload.device.index == torch.cuda.current_device(), 'wrong local CUDA device', path='open.replicated')
        value = workload.identity
        schema = [(list(t.shape), str(t.dtype), list(t.stride())) for t in tensors(workload.feeds)]
        common = {k:v for k,v in value.items() if k not in ('feeds','device')}
        return {'common': common, 'feeds':value['feeds'], 'feed_schema':schema,
                'device_type':workload.device.type, 'device':str(workload.device), 'backend':backend}
    record = checked(local, 'workload admission')
    rows = gather(record)
    require(all(r['common'] == rows[0]['common'] and r['feed_schema'] == rows[0]['feed_schema']
                and r['device_type'] == rows[0]['device_type'] and r['backend'] == rows[0]['backend']
                for r in rows), 'replica state, program, code, batch or geometry differs', path='open.replicated')
    return {'world_size':dist.get_world_size(), 'ranks':rows, 'digest':fingerprint(rows)}


def ddp_order_admitted(workload, order, context_modules=None):
    """Reducer contexts cannot be overwritten by another outstanding forward.

    Generic post-backward reduction remains available for valid programs outside
    this DDP execution family; do not weaken the mathematical Program contract.
    """
    if workload.replica_semantics.forward_interface != 'module_forward': return False
    calls = {c.id:c for c in workload.program.calls}
    ancestors = {}
    for c in workload.program.calls:
        parents = set(c.inputs) & set(calls)
        ancestors[c.id] = set(parents)
        for p in parents: ancestors[c.id].update(ancestors[p])
    active = {}
    for key in order:
        if key == '__open_finish__': continue
        c = calls[key]
        module = workload.replica_semantics.module_bindings.get(c.binding)
        if context_modules is not None and module not in context_modules: module = None
        if module is not None:
            if module in active: return False
            if c.grad == 'grad': active[module] = key
        if key in workload.replica_semantics.backward_calls:
            active = {m:f for m,f in active.items() if f not in ancestors[key]}
        if set(c.writes) & set(active): return False
    return not active


def inventory(workload, *, limit=16, topology=None):
    workload.validate()
    require(workload.replica_semantics is not None, 'replica semantics required', path='open.replicated')
    local = local_candidates(workload, limit=limit)
    original = tuple(c.id for c in workload.program.calls) + ('__open_finish__',)
    modes = [('post_backward', 0)]
    if ddp_order_admitted(workload, original): modes += [('ddp_overlap', 1), ('ddp_overlap', 128)]
    from .sharded import rejection as sharding_rejection, MODES as sharded_modes, HSDP_MODES
    reason = sharding_rejection(workload)
    admit_sharding = reason is None and len(modes) > 1
    hybrid_modes = HSDP_MODES if topology and topology['hsdp_admitted'] else ()
    from .mixed_layout import domains, enumerate_assignments, assignment_count, option as mixed_option
    domain = domains(workload, topology)
    mixed_count = assignment_count(domain)
    result = []
    # Uniform layouts are kept first for compatibility; nonuniform root layouts
    # then precede further local policy combinations. Budget truncation is explicit.
    for local_option in local['candidates']:
        choices = list(modes)
        if admit_sharding and not local_option['share_immutable']:
            choices.extend((mode, 0) for mode in (*sharded_modes, *hybrid_modes))
        for mode, mb in choices:
            if len(result) < limit:
                item = {'backend':BACKEND, 'local':local_option, 'reduction':mode, 'bucket_cap_mb':mb}
                if mode in hybrid_modes: item['node_groups'] = topology['node_groups']
                result.append(item)
        if not local_option['share_immutable']:
            for assignment in enumerate_assignments(domain):
                if len(result) >= limit: break
                result.append(mixed_option(local_option, assignment, topology))
        if len(result) >= limit: break
    from .torch_backend import sharing_inventory
    unshared_count = local['total_candidates'] // (2 if sharing_inventory(workload) else 1)
    total = local['total_candidates'] * len(modes) + unshared_count * (mixed_count +
                (len(sharded_modes)+len(hybrid_modes) if admit_sharding else 0))
    return {'candidates':result, 'total_candidates':total, 'exhaustive':len(result)==total,
            'capabilities':local['capabilities']+['post_backward_mean']+(['ddp_async_mean'] if len(modes) > 1 else [])+
                           (list(sharded_modes)+list(hybrid_modes) if admit_sharding else [])+(['per_module'] if mixed_count else []),
            'layout':'data_parallel_replicated_or_sharded', 'cross_call_overlap':False,
            'reused_hook':'distillschedule.runtime.wan13b_fair.ddp_sum_then_divide_hook',
            'sharding_rejection':reason if reason else (None if admit_sharding else 'FORWARD_BACKWARD_CONTEXT'),
            'topology':topology,
            'unsupported':['arbitrary mesh repartition','role pool partition','unequal rank batches','higher-order parameter gradients']}


class ReplicaLifecycle:
    def __init__(self, workload, implementation):
        self.workload=workload; self.option=implementation; self.hooks=[]
        self.world=dist.get_world_size(); self.backward_count=0; self.collectives=0; self.gradient_bytes=0

    def prepare(self, session):
        # Check the full selection before a DDP constructor can issue collectives.
        options=gather(self.option)
        require(all(x==options[0] for x in options), 'ranks selected different implementations', path='open.replicated')
        def validate():
            require(set(self.option)=={'backend','local','reduction','bucket_cap_mb'}
                    and self.option['backend']==BACKEND, 'invalid replica backend', path='open.replicated')
            mode,mb=self.option['reduction'],self.option['bucket_cap_mb']
            require((mode=='post_backward' and type(mb) is int and mb==0) or
                    (mode=='ddp_overlap' and type(mb) is int and mb in (1,128)),
                    'unknown executable reduction capability', path='open.replicated')
        checked(validate, 'implementation admission')
        identity(self.workload)
        from .replica_contract import execution_graph
        session.graph = checked(lambda: execution_graph(self.workload.program,
            self.workload.replica_semantics, self.option['reduction']), 'execution graph')
        self.module_keys=tuple(sorted(set(self.workload.replica_semantics.module_bindings.values())))
        if self.option['reduction']=='ddp_overlap':
            from distillschedule.runtime.wan13b_fair import _DDPCommState, ddp_sum_then_divide_hook
            for key in self.module_keys:
                module=session.state[key]
                device_ids=[session.device.index] if session.device.type=='cuda' else None
                wrapper=DDP(module, device_ids=device_ids, process_group=dist.group.WORLD,
                            broadcast_buffers=False, bucket_cap_mb=self.option['bucket_cap_mb'],
                            find_unused_parameters=True, gradient_as_bucket_view=False)
                hook=_DDPCommState(dist.group.WORLD,self.world)
                wrapper.register_comm_hook(hook, ddp_sum_then_divide_hook)
                self.hooks.append(hook);session.state[key]=wrapper
        session.effects.update(replicated=True, world_size=self.world,
                               reduction=self.option['reduction'], bucket_cap_mb=self.option['bucket_cap_mb'])

    def before_run(self, session, order):
        orders=gather(tuple(order))
        require(all(o==orders[0] for o in orders), 'rank execution orders differ', path='open.replicated')
        if self.option['reduction']=='ddp_overlap':
            require(ddp_order_admitted(self.workload,order), 'selected order overwrites a DDP context', path='open.replicated')
        dist.barrier()

    def before_call(self,session,call):
        pass

    def after_call(self, session, call):
        if call.id not in self.workload.replica_semantics.backward_calls: return
        self.backward_count+=1
        if self.option['reduction']!='post_backward': return
        parameters=[p for key in self.module_keys for p in unwrap_module(session.state[key]).parameters() if p.requires_grad]
        def validate_gradients():
            for parameter in parameters:
                grad = parameter.grad
                require(grad is None or (grad.layout == torch.strided and not grad.requires_grad
                        and grad.shape == parameter.shape and grad.dtype == parameter.dtype
                        and bool(torch.isfinite(grad).all())),
                        'only finite dense first-order parameter gradients admitted', path='open.replicated')
        checked(validate_gradients, 'parameter gradients')
        # None on one replica means zero contribution, not an omitted collective.
        flags=torch.tensor([int(p.grad is not None) for p in parameters],device=session.device,dtype=torch.int64)
        dist.all_reduce(flags,op=dist.ReduceOp.SUM)
        for p,used in zip(parameters,flags.tolist()):
            if not used: continue
            grad=p.grad
            if grad is None: grad=torch.zeros_like(p)
            require(grad.layout==torch.strided and not grad.requires_grad,
                    'only dense first-order parameter gradients admitted',path='open.replicated')
            dist.all_reduce(grad,op=dist.ReduceOp.SUM);grad.mul_(1.0/self.world)
            p.grad=grad; self.collectives+=1; self.gradient_bytes+=grad.numel()*grad.element_size()

    def _drain_ddp_telemetry(self):
        # Backward futures have completed at this cycle boundary. Retain only
        # cumulative scalars, not every bucket ever seen by a persistent hook.
        for hook in self.hooks:
            self.collectives += len(hook.bucket_bytes)
            self.gradient_bytes += sum(hook.bucket_bytes)
            hook.bucket_bytes.clear()
            hook.bucket_indices.clear()

    def finish(self, session):
        # DDP backward completion waits for registered futures; CUDA synchronization
        # then closes the physical lifetime before the next update/version boundary.
        session.synchronize()
        state=fingerprint(signature({k:unwrap_module(v) for k,v in session.state.items()}))
        digests=gather(state)
        require(len(set(digests))==1, 'replicated model/optimizer state diverged', path='open.replicated')
        self._drain_ddp_telemetry()
        session.effects.update(parameter_backward_calls=self.backward_count,
            gradient_collectives=self.collectives,
            communicated_gradient_bytes=self.gradient_bytes,
            replica_state_consensus=True, reduction_completed=True)


def make_session(workload, implementation, *, copy_state=True, persistent=False):
    checked(lambda: workload.validate(), 'session preflight')
    require(isinstance(implementation,dict) and 'local' in implementation, 'replica option required',path='open.replicated')
    checked(lambda: require(workload.replica_semantics is not None and
        (workload.replica_semantics.forward_interface == 'module_forward' or implementation.get('reduction') == 'post_backward'),
        'private methods require post_backward reduction, not forward wrappers', path='open.replicated'), 'forward interface')
    # Standard Session retains all original functions, optimizer objects, version
    # updates and RNG scopes. Only a generic execution lifecycle is attached.
    from .mixed_layout import MixedLifecycle, MODE as MIXED
    if implementation.get('reduction') == MIXED:
        return Session(workload, implementation['local'], lifecycle=MixedLifecycle(workload, implementation),
                       copy_state=copy_state,persistent=persistent)
    from .sharded import ShardedLifecycle, MODES, HSDP_MODES
    lifecycle = ShardedLifecycle if implementation.get('reduction') in (*MODES, *HSDP_MODES) else ReplicaLifecycle
    return Session(workload,implementation['local'],lifecycle=lifecycle(workload,implementation),
                   copy_state=copy_state,persistent=persistent)
