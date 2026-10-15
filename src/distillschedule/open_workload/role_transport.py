"""Original-autograd-preserving G/Score tensor/VJP transport across real processes.

Each role has a disjoint resident model pool. The score consumes a detached
transported latent leaf and returns the actual d(score_loss)/d(latent) VJP;
the producer applies it to its OWN retained forward graph before any parameter
step. This does not pretend PyTorch autograd spans send/recv implicitly.

The supported lowering is FIFO per (producer, score) pair, with bounded producer
in-flight windows. Sends use tag-free point-to-point operations (NCCL does not
support arbitrary message tags). Score and generator optimizer versions cannot
advance until all their dependent VJPs are finished.
"""
from __future__ import annotations
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable
import time

import torch
from torch import nn
import torch.distributed as dist
from distillschedule.core.contracts import require
from distillschedule.workloads.best_known.registry import fingerprint
from .role_program import RoleTask, PoolLayout, AllocatedWorld, transport_bytes_by_rank
from .role_offload import HostBudget, ActivationOffloader


@dataclass(frozen=True)
class RoleFunctions:
    """User-supplied math and inputs; never placement, timing or schedule hints."""
    make_model: Callable[[str, torch.device], nn.Module]
    make_optimizer: Callable[[str, nn.Module], torch.optim.Optimizer | None]
    sample: Callable[[int, str, torch.device], Any]
    generator_forward: Callable[[nn.Module, Any], torch.Tensor]
    score_loss: Callable[[nn.Module, torch.Tensor, Any], torch.Tensor] | None
    score_train_loss: Callable[[nn.Module, torch.Tensor, Any], torch.Tensor] | None = None
    data_identity: str = ''  # source/cohort SHA, NOT a preselected layout
    role_storage_estimate_bytes: dict[str,int] | None = None
    # Alternative to a scalar loss: the ORIGINAL algorithm constructs the
    # unnormalized per-microbatch latent gradient under no_grad (e.g. DMD).
    score_input_gradient: Callable[[nn.Module, torch.Tensor, Any], torch.Tensor] | None = None
    # Semantic dataset access by cycle and local logical microbatch, not a
    # scheduler hint. Required to advance beyond a single diagnostic cohort.
    sample_at: Callable[[int, int, str, torch.device], Any] | None = None
    # Training fact, not an optimization recipe. Verified against the model's
    # actual requires_grad flags before a frozen Score FSDP2 option can run.
    score_parameters_frozen: bool = False
    # Required for >1 Score update: original algorithm supplies an explicit
    # cycle/update/sample binding instead of silently replaying the old cohort.
    score_update_input: Callable[[Any], tuple[torch.Tensor, Any]] | None = None

    def validate(self, task):
        require(all(callable(x) for x in (self.make_model, self.make_optimizer,
                    self.sample, self.generator_forward)),
                'role workload must supply callable math and data', path='role.bindings')
        require((callable(self.score_loss) and self.score_input_gradient is None) or
                (self.score_loss is None and callable(self.score_input_gradient)),
                'declare exactly one scalar loss or constructed latent gradient',path='role.bindings')
        require(self.sample_at is None or callable(self.sample_at),
                'sample_at must be a semantic data callable',path='role.bindings')
        require(self.score_update_input is None or callable(self.score_update_input),
                'score_update_input must be a semantic data callable',path='role.bindings')
        require(task.score_updates<=1 or self.score_update_input is not None,
                'repeated score updates require explicit score_update_input semantics',
                path='role.bindings')
        require(self.score_update_input is None or task.score_updates>0,
                'score_update_input declared without Score updates',path='role.bindings')
        require((self.score_train_loss is None) == (task.score_updates==0),
                'score update semantics and function mismatch', path='role.bindings')
        require(type(self.score_parameters_frozen) is bool and
                (not self.score_parameters_frozen or task.score_updates==0),
                'frozen Score cannot have optimizer updates',path='role.bindings')
        if self.role_storage_estimate_bytes is not None:
            require(set(self.role_storage_estimate_bytes)=={'generator','score'} and
                    all(type(v) is int and v>0 for v in self.role_storage_estimate_bytes.values()),
                    'role model residency estimates must be positive for both roles',path='role.bindings')


def _parameters(module):
    return tuple(p for p in module.parameters() if p.requires_grad)


def _validate_tensor(t, task, name):
    require(isinstance(t,torch.Tensor) and tuple(t.shape)==task.output_shape and
            t.dtype==getattr(torch,task.dtype) and t.is_floating_point() and
            bool(torch.isfinite(t).all()),
            f'{name} does not satisfy the declared latent/dtype/finite contract',path='role.transport')


def _sync_gradients(module, group, world):
    """Sum already globally normalized per-microbatch gradients across one role."""
    params=_parameters(module)
    require(params,'selected role must have actual trainable parameters',path='role.grad')
    missing=torch.tensor([int(p.grad is None) for p in params],device=params[0].device,dtype=torch.int64)
    if world>1:dist.all_reduce(missing,op=dist.ReduceOp.SUM,group=group)
    for p,was_missing in zip(params,missing.tolist()):
        if was_missing==world:continue
        g=p.grad if p.grad is not None else torch.zeros_like(p)
        require(g.layout==torch.strided and not g.requires_grad,
                'only first-order dense parameter gradients supported',path='role.grad')
        if world>1:dist.all_reduce(g,op=dist.ReduceOp.SUM,group=group)
        p.grad=g


def _model_identity(module):
    """Small-model correctness gate; full tensor digests are expensive for 14B."""
    from .torch_backend import signature
    return fingerprint(signature(module))


class RolePoolExecutor:
    def __init__(self, task:RoleTask, layout:PoolLayout, hardware:AllocatedWorld,
                 bindings:RoleFunctions, *, device=None, profile_components=False):
        require(dist.is_available() and dist.is_initialized() and dist.get_world_size()==hardware.world,
                'explicit initialized world process group required',path='role.runtime')
        layout.validate(task,hardware);bindings.validate(task)
        self.task,self.layout,self.hardware,self.bindings=task,layout,hardware,bindings
        self.rank=dist.get_rank()
        selected = device if device is not None else (
            torch.device('cuda', torch.cuda.current_device()) if torch.cuda.is_available() else 'cpu')
        self.device=torch.device(selected)
        backend=str(dist.get_backend())
        require((self.device.type=='cpu' and backend=='gloo') or
                (self.device.type=='cuda' and backend=='nccl' and
                 self.device.index==torch.cuda.current_device()),
                'CPU/Gloo or one CUDA/NCCL device per rank required',path='role.runtime')
        require(type(profile_components) is bool and (not profile_components or (
                    layout.window==1 and layout.activation_policy=='resident' and
                    layout.generator_parallelism=='replicated' and
                    layout.score_parallelism=='replicated')),
                'component profiling requires a replicated resident window-one anchor',
                path='role.components')
        self.profile_components=profile_components
        self.role='generator' if self.rank in layout.generator else 'score'
        self.group=None
        self.events=[]
        self.module=None;self.optimizer=None
        self._generator_mesh=None;self._score_mesh=None;self._sharding={'role':'replicated'}
        self._prepared=False;self._closed=False;self._failed=False;self._busy=False
        self._cycle=0
        self._versions={'generator':0,'score':0,'optimizer_g':0,'gradient_g':0,'optimizer_s':0}
        self._admission=(task.digest,fingerprint(layout.to_dict()),hardware)
        self._bound_bindings=bindings

    def _event(self, action, i=None):
        self.events.append({'action':action,'microbatch':i,'rank':self.rank,'cycle':self._cycle})

    def _sample(self, index, role):
        if self.bindings.sample_at is not None:
            return self.bindings.sample_at(self._cycle,index,role,self.device)
        require(self._cycle==0,'continuous execution requires sample_at',path='role.data')
        return self.bindings.sample(index,role,self.device)

    def _prepare(self):
        if self._prepared:return
        task,layout=self.task,self.layout
        # Collective group creation order MUST be identical on every rank.
        gen_group=dist.new_group(ranks=list(layout.generator))
        score_group=dist.new_group(ranks=list(layout.score))
        self.gen_group,self.score_group=gen_group,score_group
        group=gen_group if self.role=='generator' else score_group
        group_size=len(layout.generator if self.role=='generator' else layout.score)
        module=self.bindings.make_model(self.role,self.device)
        require(isinstance(module,nn.Module),'model factory must return a torch Module',path='role.runtime')
        module=module.to(self.device)
        # Frozen guidance still differentiates its INPUT, not its parameters.
        # Do not fabricate trainable parameters/an optimizer to admit that role.
        updating=self.role=='generator' or task.score_updates>0
        optimizer=self.bindings.make_optimizer(self.role,module) if updating else None
        if updating:
            require(isinstance(optimizer,torch.optim.Optimizer),
                    'updating role needs a torch Optimizer',path='role.runtime')
            require(_parameters(module),'updating role has no trainable parameters',path='role.runtime')
        # No parameter/model sharing across roles is assumed. Within each role,
        # fail closed if replicas did not start from identical weights.
        identities=[None for _ in range(group_size)]
        dist.all_gather_object(identities,_model_identity(module),group=group)
        from .replicated import checked
        checked(lambda: require(len(set(identities))==1,
              'role replica model initializations differ',path='role.runtime'),
              'role replica identity admission')
        def native_admission():
            if self.role=='score' and self.bindings.score_parameters_frozen:
                require(all(not p.requires_grad for p in module.parameters()),
                        'declared frozen Score has trainable parameters',path='role.runtime')
            if self.role=='score' and layout.score_parallelism=='fsdp2_frozen':
                require(self.bindings.score_parameters_frozen,
                        'Score pool FSDP2 requires verified frozen Score semantics',path='role.runtime')
                from .role_sharding import admit_frozen_score
                admit_frozen_score(module)
            if self.role=='score' and layout.score_parallelism=='fsdp2_train':
                require(task.score_updates>0 and not self.bindings.score_parameters_frozen
                        and self.bindings.score_input_gradient is not None,
                        'trainable Score FSDP2 needs original no-grad constructed gradient semantics',
                        path='role.runtime')
                from .role_sharding import admit
                admit(module,optimizer)
            if self.role=='generator' and layout.generator_parallelism=='fsdp2_keep_forward':
                from .role_sharding import admit
                admit(module,optimizer)
        checked(native_admission,'role native layout admission')
        if layout.score_parallelism=='fsdp2_frozen' and self.role=='score':
            from .role_sharding import install_frozen_score
            module,self._score_mesh,self._sharding=install_frozen_score(
                module,self.device,layout.score,score_group)
        if layout.score_parallelism=='fsdp2_train' and self.role=='score':
            from .role_sharding import install_trainable_score
            module,optimizer,self._score_mesh,self._sharding=install_trainable_score(
                module,optimizer,self.device,layout.score,score_group)
        if layout.generator_parallelism=='fsdp2_keep_forward' and self.role=='generator':
            from .role_sharding import install_generator
            module,optimizer,self._generator_mesh,self._sharding=install_generator(
                module,optimizer,self.device,layout.generator,gen_group)
        self.module,self.optimizer=module,optimizer
        self.gen_group,self.score_group=gen_group,score_group
        self.group=group
        self._prepared=True

    def run(self):
        """Run one real cycle; repeated calls reuse models, optimizers and groups."""
        from .replicated import checked
        def admission():
            require(not (self._closed or self._failed or self._busy),
                    'closed, failed or active role executor cannot run',path='role.runtime')
            require(self._admission==(self.task.digest,fingerprint(self.layout.to_dict()),self.hardware),
                    'task/layout changed in live executor',path='role.runtime')
            require(self.bindings is self._bound_bindings,
                    'bindings changed in live executor',path='role.runtime')
            require(self._cycle==0 or self.bindings.sample_at is not None,
                    'continuous execution requires explicit sample_at',path='role.data')
            return (self._cycle,self._versions,self.task.digest,self.layout.to_dict())
        local=checked(admission,'role cycle admission')
        entries=[None]*self.hardware.world;dist.all_gather_object(entries,local)
        require(all(x==entries[0] for x in entries),'rank cycle/version mismatch',path='role.runtime')
        self._busy=True
        try:
            self._prepare()
            result=self._run_cycle()
            self._versions=dict(result[0]['final_versions']);self._cycle+=1
            return result
        except BaseException:
            # A partly committed iteration is never eligible for blind retry.
            self._failed=True
            raise
        finally:self._busy=False

    def close(self):
        """Release only groups owned here, after a completed cycle has drained."""
        require(not self._busy,'cannot close an active role executor',path='role.runtime')
        if self._closed:return
        if dist.is_initialized():
            for group in (getattr(self,'gen_group',None),getattr(self,'score_group',None)):
                # FSDP2 parameter metadata retains the named mesh communicator.
                # Native Torch 2.10 may cache that mesh between bounded trials.
                # Keep the Generator subgroup alive until the parent torchrun
                # world closes; destroy ordinary replica/Score groups normally.
                if group is not None and group!=dist.GroupMember.NON_GROUP_MEMBER and not (
                    (group is self.gen_group and self._generator_mesh is not None) or
                    (group is self.score_group and self._score_mesh is not None)):
                    dist.destroy_process_group(group)
        self._closed=True
        # The DeviceMesh shares gen_group (no second subgroup).
        self._generator_mesh=None;self._score_mesh=None
        self.module=None;self.optimizer=None;self.group=None

    def _run_cycle(self):
        task,layout=self.task,self.layout
        module,optimizer=self.module,self.optimizer
        gen_group,score_group=self.gen_group,self.score_group
        group_size=len(layout.generator if self.role=='generator' else layout.score)
        self.events=[]
        from .role_profiling import ComponentRecorder
        components=ComponentRecorder(self.device,self.profile_components)
        graph=task.program(initial_versions=self._versions).compile()
        from .torch_backend import storage_bytes
        initial_bytes=storage_bytes((module,optimizer))
        require(all(p.grad is None for p in module.parameters()),
                'cycle begins with pending parameter gradients',path='role.runtime')
        if optimizer is not None:optimizer.zero_grad(set_to_none=True)
        if self.device.type=='cuda':torch.cuda.reset_peak_memory_stats(self.device)
        dist.barrier()
        started=time.perf_counter_ns()
        send_work=[]
        host_budget=HostBudget(self.hardware.host_capacities[self.rank])
        offload_records=[]
        local_score_data=[]
        peak=0;transport_peak=0
        if self.role=='generator':
            inflight=deque()
            def flush():
                i, latent, grad, recv_work, sent, source, offloader = inflight.popleft()
                if offloader is not None:offloader.prefetch()
                recv_work.wait();sent.wait()
                _validate_tensor(grad,task,'returned latent gradient')
                # FSDP's reduce-scatter averages across its shard group. The
                # input VJP is already divided by GLOBAL logical microbatch
                # count; compensate once before invoking native autograd.
                vjp=(grad*group_size if layout.generator_parallelism=='fsdp2_keep_forward' else grad)
                with components.span('backward',i):
                    with torch.enable_grad():latent.backward(vjp)
                self._event('generator_vjp_backward',i)
                if offloader is not None:
                    offload_records.append(offloader.statistics)
                    offloader.close()
            for i in range(task.microbatches):
                owner,consumer=layout.owners(i)
                if owner!=self.rank:continue
                while len(inflight)>=layout.window:flush()
                offloader=(ActivationOffloader(self.device,module,host_budget)
                           if layout.activation_policy=='cpu_offload_prefetch' else None)
                with components.saved(module,i), components.span('forward',i):
                    with torch.enable_grad():
                        with offloader.hooks() if offloader is not None else nullcontext():
                            latent=self.bindings.generator_forward(module,self._sample(i,'generator'))
                    _validate_tensor(latent,task,'generator output')
                    require(latent.requires_grad,'generator must produce an autograd-carrying latent',path='role.runtime')
                # Pre-post response receive; NCCL tag-less point-to-point FIFO.
                grad=torch.empty_like(latent.detach(),memory_format=torch.contiguous_format)
                received=dist.irecv(grad,src=consumer)
                outbound=latent.detach().contiguous()
                sent=dist.isend(outbound,dst=consumer)
                inflight.append((i,latent,grad,received,sent,outbound,offloader))
                self._event('generator_forward_send',i)
                peak=max(peak,len(inflight))
                transport_peak=max(transport_peak,storage_bytes(
                    [(x[1],x[2],x[5]) for x in inflight]))
                # Drop producer-loop aliases; the in-flight entry owns storage.
                del latent,grad,received,sent,outbound,offloader
            while inflight:flush()
            # Sharded FSDP2 gradients have ALREADY been reduce-scattered.
            # A second allreduce would corrupt the Adam update.
            if self.profile_components:dist.barrier(group=gen_group)
            with components.span('generator_commit'):
                if layout.generator_parallelism=='replicated':
                    _sync_gradients(module,gen_group,group_size)
                optimizer.step();optimizer.zero_grad(set_to_none=True)
            self._event('generator_commit',None)
        else:
            for i in range(task.microbatches):
                producer,consumer=layout.owners(i)
                if consumer!=self.rank:continue
                latent=torch.empty(task.output_shape,device=self.device,dtype=getattr(torch,task.dtype))
                dist.recv(latent,src=producer)
                _validate_tensor(latent,task,'received latent')
                with components.span('score',i):
                    source=latent.detach()
                    if self.bindings.score_input_gradient is not None:
                        with torch.no_grad():
                            gradient=self.bindings.score_input_gradient(module,source,self._sample(i,'score'))
                        _validate_tensor(gradient,task,'constructed latent gradient')
                        require(not gradient.requires_grad and gradient.grad_fn is None,
                                'constructed gradient must be detached first-order data',path='role.runtime')
                    else:
                        with torch.enable_grad():
                            source.requires_grad_(True)
                            loss=self.bindings.score_loss(module,source,self._sample(i,'score'))
                            require(isinstance(loss,torch.Tensor) and loss.ndim==0 and
                                    bool(torch.isfinite(loss)) and loss.requires_grad,
                                    'score must produce a finite differentiable scalar loss',path='role.runtime')
                            gradient=torch.autograd.grad(loss,source,create_graph=False)[0]
                    gradient=(gradient/task.microbatches).detach().contiguous()
                    _validate_tensor(gradient,task,'score input gradient')
                send_work.append((dist.isend(gradient,dst=producer),gradient))
                if task.score_updates:local_score_data.append((latent.detach(),i))
                self._event('score_receive_vjp_send',i)
                transport_peak=max(transport_peak,storage_bytes(
                    (latent,source,gradient,[x[1] for x in send_work],local_score_data)))
            for work,_ in send_work:work.wait()
            send_work.clear()
            if 'gradient' in locals():del gradient,source,latent
            if 'loss' in locals():del loss
        # Score parameter/version update MUST follow all generator VJP use.
        dist.barrier()
        if self.role=='score' and task.score_updates:
            from .role_score import execute_updates
            with components.span('score_updates'):
                transport_peak=max(transport_peak,execute_updates(self,local_score_data))
        dist.barrier()
        if self.device.type=='cuda':torch.cuda.synchronize(self.device)
        compute_transport_wall_ns=time.perf_counter_ns()-started
        # All-gather small receipts only (no weights or model tensors transmitted).
        cuda_peak=(torch.cuda.max_memory_allocated(self.device)
                   if self.device.type=='cuda' else None)
        if self.profile_components and self.device.type=='cuda':
            cuda_peak=max([cuda_peak]+[r['peak_allocated_bytes'] for r in components.records])
        if (self.role=='generator' and layout.generator_parallelism=='fsdp2_keep_forward') or (
             self.role=='score' and layout.score_parallelism in ('fsdp2_frozen','fsdp2_train')):
            from .role_sharding import sharded_model_identity
            model_digest=sharded_model_identity(module,gen_group if self.role=='generator' else score_group,group_size)
        else:
            model_digest=_model_identity(module)
        local={'rank':self.rank,'role':self.role,'events':self.events,
               'components':components.result(),
               'compute_transport_wall_ns':compute_transport_wall_ns,
               'activation_policy':layout.activation_policy,
               'offload':{'packed_tensors':sum(x['packed_tensors'] for x in offload_records),
                          'host_bytes_packed':sum(x['packed_host_bytes'] for x in offload_records),
                          'prefetch_tensors':sum(x['prefetched_tensors'] for x in offload_records),
                          'unpacked_tensors':sum(x['unpacked_tensors'] for x in offload_records),
                          'peak_host_bytes':host_budget.peak,
                          'true_cuda_async_copy':self.device.type=='cuda' and bool(offload_records)},
               'transport_peak_bytes':transport_peak,
               'transport_bound_bytes':transport_bytes_by_rank(task,layout)[self.rank],
               'cuda_peak_allocated_bytes':cuda_peak,
               'capacity_scope':'known transport bound plus measured CUDA allocator peak; not unseen-shape guarantee',
               'initial_storage_bytes':initial_bytes,
               'final_storage_bytes':storage_bytes((module,optimizer)),
               'max_inflight':peak,'elapsed_ns':time.perf_counter_ns()-started,
               'model_sha256':model_digest,'sharding':self._sharding,
               'generator_version':graph.metadata['final_versions']['generator'],
               'score_version':graph.metadata['final_versions']['score'],
               'cycle':self._cycle,
               'gradient_source':'constructed_no_grad' if self.bindings.score_input_gradient is not None else 'scalar_loss_vjp',
               'input_gradients':sum(x['action']=='score_receive_vjp_send' for x in self.events),
               'gradient_transport':'explicit_tensor_vjp_p2p_no_remote_autograd',
               'host_leases_closed':host_budget.used==0}
        rows=[None]*self.hardware.world
        dist.all_gather_object(rows,local)
        require({x['rank'] for x in rows}==set(range(self.hardware.world)),
                'missing rank receipt',path='role.runtime')
        require(all(r['transport_peak_bytes']<=r['transport_bound_bytes'] for r in rows),
                'observed transport exceeds declared bound',path='role.capacity')
        require(all(r['cuda_peak_allocated_bytes'] is None or
                    r['cuda_peak_allocated_bytes']<=self.hardware.capacities[r['rank']] for r in rows),
                'observed CUDA peak exceeds allocated capacity',path='role.capacity')
        for role,ranks in (('generator',layout.generator),('score',layout.score)):
            digests={rows[r]['model_sha256'] for r in ranks}
            require(len(digests)==1,f'{role} replicas diverged',path='role.runtime')
        require(sum(x['input_gradients'] for x in rows)==task.microbatches,
                'missing transported input VJPs',path='role.runtime')
        return {'schema_version':'distillschedule.role_pool_execution.v1',
                'status':'PASS_ROLE_POOL_EXECUTION','task_sha256':task.digest,
                'layout':layout.to_dict(),'ranks':rows,
                'cycle':self._cycle,'initial_versions':dict(self._versions),
                'final_versions':dict(graph.metadata['final_versions']),
                'wall_ns':max(x['elapsed_ns'] for x in rows),
                'compute_transport_wall_ns':max(x['compute_transport_wall_ns'] for x in rows),
                'gpu_execution':'NATIVE_CUDA_RUN_NOT_VALIDATED' if self.device.type=='cuda' else 'NOT_RUN',
                'quality_validation':'NOT_RUN','claimable_speedup':False},module,optimizer
