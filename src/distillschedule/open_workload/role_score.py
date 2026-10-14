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
    from .role_transport import _validate_tensor, _sync_gradients
    from .torch_backend import storage_bytes
    task, layout = executor.task, executor.layout
    module, optimizer = executor.module, executor.optimizer
    require(optimizer is not None, 'Score update needs its original optimizer',
            path='role.score_update')
    peak = storage_bytes(retained)
    for update in range(task.score_updates):
        require(all(p.grad is None for p in module.parameters()),
                'Score update begins with pending gradients',path='role.score_update')
        for original, index in retained:
            conditions = executor._sample(index, 'score')
            source = original
            provider = executor.bindings.score_update_input
            if provider is not None:
                context = ScoreUpdateContext(executor._cycle, update, index,
                                             original, conditions, executor.device)
                version = original._version
                with torch.no_grad():
                    provided = provider(context)
                require(isinstance(provided, tuple) and len(provided)==2,
                        'Score input provider must return (latent, conditions)',
                        path='role.score_update')
                require(original._version==version,
                        'Score input provider mutated the retained cohort',
                        path='role.score_update')
                source, conditions = provided
            _validate_tensor(source, task, 'Score training source')
            require(source.device==executor.device and not source.requires_grad
                    and source.grad_fn is None,
                    'Score training source must be detached on its assigned device',
                    path='role.score_update')
            peak = max(peak, storage_bytes((retained, source)))
            with torch.enable_grad():
                loss = executor.bindings.score_train_loss(module, source, conditions)
                require(isinstance(loss, torch.Tensor) and loss.ndim==0
                        and bool(torch.isfinite(loss)) and loss.requires_grad,
                        'Score train objective must be finite scalar with gradients',
                        path='role.score_update')
                # Native FSDP2 MEAN-reduces parameter gradients. Cancel that
                # mean once because the objective is already GLOBAL normalized.
                factor = len(layout.score) if layout.score_parallelism=='fsdp2_train' else 1
                (loss*(factor/task.microbatches)).backward()
            executor._event('score_train_backward', index)
            executor.events[-1]['score_update'] = update
            # Never keep one update's autograd graph or transformed input in
            # the next update's live set; retained itself is a detached cohort.
            del loss, source, conditions
            if provider is not None:
                del provided, context
        if layout.score_parallelism=='replicated':
            _sync_gradients(module, executor.score_group, len(layout.score))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        executor._event('score_commit', None)
        executor.events[-1]['score_update'] = update
    return peak
