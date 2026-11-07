"""Original Score update semantics, independent of placement and pipeline order.

Repeated updates require an explicit workload input binding. This module does
not invent fresh Generator rollouts or change the loss to fit the scheduler.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import torch
from distillschedule.core.contracts import require


@dataclass(frozen=True)
class ScoreUpdateContext:
    cycle: int
    update: int
    microbatch: int
    retained_latent: torch.Tensor
    conditions: Any
    device: torch.device


def execute_updates(executor, retained):
    """Apply all declared Score commits; release temporary inputs every batch.

    The legacy single-update route receives identical data and normalization.
    A repeated-update provider explicitly chooses its own source and conditions;
    returning the retained cohort is allowed only as that provider's semantics.
    """
    from .role_transport import _validate_tensor
    checked=executor._checked
    from .torch_backend import storage_bytes
    task, layout = executor.task, executor.layout
    module, optimizer = executor.module, executor.optimizer
    active=executor.role=='score'
    checked(lambda:require(optimizer is not None,
                'Score update needs its original optimizer',path='role.score_update') if active else None,
            'Score optimizer admission')
    peak = storage_bytes(retained)
    for update in range(task.score_updates):
        checked(lambda:require(all(p.grad is None for p in module.parameters()),
                    'Score update begins with pending gradients',path='role.score_update') if active else None,
                'Score gradient boundary')
        # Replicated pools may have uneven work. Metadata rendezvous uses local
        # waves; idle ranks participate without adding model calls or samples.
        waves=(task.microbatches+len(layout.score)-1)//len(layout.score)
        for slot in range(waves):
            def prepare_input():
                if not active or slot>=len(retained):return None
                original,index=retained[slot]
                conditions=executor._sample(index,'score')
                source=original
                provider=executor.bindings.score_update_input
                if provider is not None:
                    context=ScoreUpdateContext(executor._cycle,update,index,
                                               original,conditions,executor.device)
                    version=original._version
                    with torch.no_grad():provided=provider(context)
                    require(isinstance(provided,tuple) and len(provided)==2,
                            'Score input provider must return (latent, conditions)',path='role.score_update')
                    require(original._version==version,
                            'Score input provider mutated the retained cohort',path='role.score_update')
                    source,conditions=provided
                _validate_tensor(source,task,'Score training source')
                require(source.device==executor.device and not source.requires_grad
                        and source.grad_fn is None,
                        'Score training source must be detached on its assigned device',path='role.score_update')
                return source,conditions,index
            prepared=checked(prepare_input,'Score training input admission')
            if prepared is not None:
                peak=max(peak,storage_bytes((retained,prepared[0])))
            def objective():
                if prepared is None:return None
                source,conditions,_=prepared
                with torch.enable_grad():
                    loss=executor.bindings.score_train_loss(module,source,conditions)
                    require(isinstance(loss,torch.Tensor) and loss.ndim==0
                            and bool(torch.isfinite(loss)) and loss.requires_grad,
                            'Score train objective must be finite scalar with gradients',path='role.score_update')
                return loss
            loss=checked(objective,'Score training objective admission')
            def backward():
                if loss is None:return
                with torch.enable_grad():
                    factor=len(layout.score) if layout.score_parallelism=='fsdp2_train' else 1
                    (loss*(factor/task.microbatches)).backward()
                executor._event('score_train_backward',prepared[2])
                executor.events[-1]['score_update']=update
            checked(backward,'Score training backward')
            del prepared,loss
        executor._commit_update('score')
        if active:executor.events[-1]['score_update']=update
    return peak
