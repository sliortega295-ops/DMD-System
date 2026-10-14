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
from .role_program import RoleTask, PoolLayout, AllocatedWorld
from .role_offload import HostBudget, ActivationOffloader


@dataclass(frozen=True)
class RoleFunctions:
    """User-supplied math and inputs; never placement, timing or schedule hints."""
    make_model: Callable[[str, torch.device], nn.Module]
    make_optimizer: Callable[[str, nn.Module], torch.optim.Optimizer]
    sample: Callable[[int, str, torch.device], Any]
    generator_forward: Callable[[nn.Module, Any], torch.Tensor]
    score_loss: Callable[[nn.Module, torch.Tensor, Any], torch.Tensor]
    score_train_loss: Callable[[nn.Module, torch.Tensor, Any], torch.Tensor] | None = None
    data_identity: str = ''  # source/cohort SHA, NOT a preselected layout

    def validate(self, task):
        require(all(callable(x) for x in (self.make_model, self.make_optimizer,
                    self.sample, self.generator_forward, self.score_loss)),
                'role workload must supply callable math and data', path='role.bindings')
        require(task.score_updates in (0,1), 'repeated score updates need an explicit fresh-input semantic adapter',
                path='role.bindings')
        require((self.score_train_loss is None) == (task.score_updates==0),
                'score update semantics and function mismatch', path='role.bindings')


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
                 bindings:RoleFunctions, *, device=None):
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
        self.role='generator' if self.rank in layout.generator else 'score'
        self.group=None
        self.events=[]

    def _event(self, action, i=None):
        self.events.append({'action':action,'microbatch':i,'rank':self.rank})

    def run(self):
        task,layout=self.task,self.layout
        # Collective group creation order MUST be identical on every rank.
        gen_group=dist.new_group(ranks=list(layout.generator))
        score_group=dist.new_group(ranks=list(layout.score))
        group=gen_group if self.role=='generator' else score_group
        group_size=len(layout.generator if self.role=='generator' else layout.score)
        module=self.bindings.make_model(self.role,self.device)
        require(isinstance(module,nn.Module),'model factory must return a torch Module',path='role.runtime')
        module=module.to(self.device)
        optimizer=self.bindings.make_optimizer(self.role,module)
        require(isinstance(optimizer,torch.optim.Optimizer),
                'optimizer factory must return a torch Optimizer',path='role.runtime')
        params=_parameters(module)
        require(params,'role has no trainable parameters',path='role.runtime')
        # No parameter/model sharing across roles is assumed. Within each role,
        # fail closed if replicas did not start from identical weights.
        identities=[None for _ in range(group_size)]
        dist.all_gather_object(identities,_model_identity(module),group=group)
        require(len(set(identities))==1,'role replica model initializations differ',path='role.runtime')
        from .torch_backend import storage_bytes
        initial_bytes=storage_bytes((module,optimizer))
        optimizer.zero_grad(set_to_none=True)
        dist.barrier()
        started=time.perf_counter_ns()
        send_work=[]
        host_budget=HostBudget(self.hardware.host_capacities[self.rank])
        offload_records=[]
        local_score_data=[]
        peak=0
        if self.role=='generator':
            inflight=deque()
            def flush():
                i, latent, grad, recv_work, sent, source, offloader = inflight.popleft()
                if offloader is not None:offloader.prefetch()
                recv_work.wait();sent.wait()
                _validate_tensor(grad,task,'returned latent gradient')
                with torch.enable_grad():latent.backward(grad)
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
                with torch.enable_grad():
                    with offloader.hooks() if offloader is not None else nullcontext():
                        latent=self.bindings.generator_forward(module,self.bindings.sample(i,'generator',self.device))
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
            while inflight:flush()
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
                with torch.enable_grad():
                    source=latent.detach().requires_grad_(True)
                    loss=self.bindings.score_loss(module,source,self.bindings.sample(i,'score',self.device))
                    require(isinstance(loss,torch.Tensor) and loss.ndim==0 and
                            bool(torch.isfinite(loss)) and loss.requires_grad,
                            'score must produce a finite differentiable scalar loss',path='role.runtime')
                    gradient=torch.autograd.grad(loss,source,create_graph=False)[0]
                    gradient=(gradient/task.microbatches).detach().contiguous()
                _validate_tensor(gradient,task,'score input gradient')
                send_work.append((dist.isend(gradient,dst=producer),gradient))
                if task.score_updates:local_score_data.append((latent.detach(),i))
                self._event('score_receive_vjp_send',i)
            for work,_ in send_work:work.wait()
        # Score parameter/version update MUST follow all generator VJP use.
        dist.barrier()
        if self.role=='score' and task.score_updates:
            for source,i in local_score_data:
                with torch.enable_grad():
                    loss=self.bindings.score_train_loss(module,source,self.bindings.sample(i,'score',self.device))
                    require(isinstance(loss,torch.Tensor) and loss.ndim==0 and bool(torch.isfinite(loss)),
                            'score train objective must be scalar/finite',path='role.runtime')
                    (loss/task.microbatches).backward()
                self._event('score_train_backward',i)
            _sync_gradients(module,score_group,group_size)
            optimizer.step();optimizer.zero_grad(set_to_none=True)
            self._event('score_commit',None)
        dist.barrier()
        if self.device.type=='cuda':torch.cuda.synchronize(self.device)
        # All-gather small receipts only (no weights or model tensors transmitted).
        local={'rank':self.rank,'role':self.role,'events':self.events,
               'activation_policy':layout.activation_policy,
               'offload':{'packed_tensors':sum(x['packed_tensors'] for x in offload_records),
                          'host_bytes_packed':sum(x['packed_host_bytes'] for x in offload_records),
                          'prefetch_tensors':sum(x['prefetched_tensors'] for x in offload_records),
                          'unpacked_tensors':sum(x['unpacked_tensors'] for x in offload_records),
                          'peak_host_bytes':host_budget.peak,
                          'true_cuda_async_copy':self.device.type=='cuda' and bool(offload_records)},
               'initial_storage_bytes':initial_bytes,
               'final_storage_bytes':storage_bytes((module,optimizer)),
               'max_inflight':peak,'elapsed_ns':time.perf_counter_ns()-started,
               'model_sha256':_model_identity(module),'generator_version':1,
               'score_version':task.score_updates,
               'input_gradients':sum(x['action']=='score_receive_vjp_send' for x in self.events),
               'gradient_transport':'explicit_tensor_vjp_p2p_no_remote_autograd',
               'host_leases_closed':host_budget.used==0}
        rows=[None]*self.hardware.world
        dist.all_gather_object(rows,local)
        require({x['rank'] for x in rows}==set(range(self.hardware.world)),
                'missing rank receipt',path='role.runtime')
        for role,ranks in (('generator',layout.generator),('score',layout.score)):
            digests={rows[r]['model_sha256'] for r in ranks}
            require(len(digests)==1,f'{role} replicas diverged',path='role.runtime')
        require(sum(x['input_gradients'] for x in rows)==task.microbatches,
                'missing transported input VJPs',path='role.runtime')
        return {'schema_version':'distillschedule.role_pool_execution.v1',
                'status':'PASS_ROLE_POOL_EXECUTION','task_sha256':task.digest,
                'layout':layout.to_dict(),'ranks':rows,
                'wall_ns':max(x['elapsed_ns'] for x in rows),
                'gpu_execution':'NATIVE_CUDA_RUN_NOT_VALIDATED' if self.device.type=='cuda' else 'NOT_RUN',
                'quality_validation':'NOT_RUN','claimable_speedup':False},module,optimizer
