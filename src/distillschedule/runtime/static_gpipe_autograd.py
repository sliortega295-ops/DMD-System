"""External-loss GPipe boundary preserving native pyramid input gradients.

Each inner forward is B1. The outer native scheduler/loss sees a cohort of
eight independent samples. Its batch-mean gradient is not divided twice.
Inputs detach only at this transport boundary, then receive their actual
first-stage gradients on *both* stages, retaining earlier pyramid graphs.
"""
from __future__ import annotations

import torch
import torch.distributed as dist


class _NativeBatchBoundary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, owner, trigger, *inputs):
        if any(not isinstance(t, torch.Tensor) or t.ndim == 0 or t.shape[0] != owner.schedule.native_microbatches
               for t in inputs):
            raise ValueError('explicit admitted sample tensors required at GPipe boundary')
        ctx.owner = owner
        ctx.needs_native_grad = tuple(t.requires_grad for t in inputs)
        ctx.input_metadata = tuple((tuple(t.shape), t.dtype, t.device) for t in inputs)
        detached = [t.detach().requires_grad_(t.requires_grad) for t in inputs]
        with torch.enable_grad():
            output = owner.schedule.forward_batch(*detached)
        owner.schedule.bind_external_graph(ctx)
        return output.detach()

    @staticmethod
    def backward(ctx, output_gradient):
        owner = ctx.owner
        owner.schedule.backward_output_gradient(output_gradient)
        count = len(ctx.needs_native_grad)
        gradients = (owner.schedule.first_stage_input_gradients(count)
                     if owner.stage_index == 0 else [None] * count)
        for i, needed in enumerate(ctx.needs_native_grad):
            if not needed:
                gradients[i] = None
                continue
            shape, dtype, device = ctx.input_metadata[i]
            if owner.stage_index == 0:
                value = gradients[i]
                if value is None or tuple(value.shape) != shape or value.dtype != dtype:
                    raise ValueError('native upstream gradient missing or changed')
            else:
                gradients[i] = torch.empty(shape, dtype=dtype, device=device)
            dist.broadcast(gradients[i], src=dist.get_global_rank(owner.group, 0),
                           group=owner.group)
        return (None, None, *gradients)


class NativeBatchGPipeAutograd:
    def __init__(self, schedule, *, stage_index, group, device):
        if not dist.is_initialized() or dist.get_world_size(group) != 2:
            raise ValueError('requires explicit PP2')
        if stage_index != dist.get_rank(group):
            raise ValueError('PP coordinate mismatch')
        self.schedule = schedule
        self.stage_index, self.group = stage_index, group
        self.device = torch.device(device)

    def __call__(self, *inputs):
        if not torch.is_grad_enabled():
            raise ValueError('this boundary is only for actual grad-enabled native calls')
        # Parameter-only forwards also require an outer autograd node. This
        # private trigger never changes model inputs, losses, or training RNG.
        trigger = torch.zeros((), device=self.device, requires_grad=True)
        return _NativeBatchBoundary.apply(self, trigger, *inputs)
