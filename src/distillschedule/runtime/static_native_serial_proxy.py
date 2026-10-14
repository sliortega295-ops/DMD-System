"""Source-bound serial B1 PP2 capacity fallback, not GPipe1 or a DMD optimization.

The Torch2.10 GPipe minimum is two microbatches. This separate synchronous
transport keeps one B1 graph per role invocation and accumulates eight samples
before the existing DP2 reduction/clip/optimizer. Prefix, blocks and suffix are
the same admitted source functions. Both PP stages see the original output;
only stage1's loss gradient is authoritative, avoiding duplicate gradients.
"""
from __future__ import annotations

import torch
import torch.distributed as dist
from torch import nn

from .static_native_pp_stage import NativeSourcePPStage
from .static_native_proxy import NativeStaticPPProxy


def _description(values):
    if not values or any(not isinstance(v, torch.Tensor) for v in values):
        raise ValueError('serial PP transport requires real tensors')
    return [(tuple(v.shape), str(v.dtype), v.requires_grad) for v in values]


class _SerialBoundary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, owner, trigger, *inputs):
        ctx.owner = owner
        ctx.needs = tuple(v.requires_grad for v in inputs)
        ctx.inputs = tuple(v.detach().requires_grad_(need)
                           for v, need in zip(inputs, ctx.needs, strict=True))
        with torch.enable_grad():
            values, ctx.local_outputs, ctx.boundary_inputs = owner._execute(ctx.inputs, grad=True)
        owner.pending += 1
        return tuple(v.detach() for v in values)

    @staticmethod
    def backward(ctx, *output_gradients):
        owner = ctx.owner
        # Stage0 mirrors the native loss only to preserve its upstream graph.
        # Its output gradient MUST NOT be applied a second time on stage1.
        if owner.stage.stage_index == 1:
            differentiable = [(v, torch.zeros_like(v) if g is None else g.detach())
                              for v, g in zip(ctx.local_outputs, output_gradients, strict=True)
                              if v.requires_grad]
            if not differentiable:
                raise ValueError('native serial grad call has no differentiable output')
            torch.autograd.backward(tuple(v for v, _ in differentiable),
                                    tuple(g for _, g in differentiable))
            for value in ctx.boundary_inputs:
                if value.requires_grad:
                    owner._send(torch.zeros_like(value) if value.grad is None else value.grad)
        else:
            differentiable = [(v, owner._receive(tuple(v.shape), v.dtype))
                              for v in ctx.local_outputs if v.requires_grad]
            if not differentiable:
                raise ValueError('native prefix has no differentiable boundary')
            torch.autograd.backward(tuple(v for v, _ in differentiable),
                                    tuple(g for _, g in differentiable))
        upstream = []
        for value, need in zip(ctx.inputs, ctx.needs, strict=True):
            if not need:
                upstream.append(None)
                continue
            gradient = (torch.zeros_like(value) if value.grad is None else value.grad)
            if owner.stage.stage_index == 1:
                gradient = torch.empty_like(value)
            dist.broadcast(gradient, src=owner.first_rank, group=owner.pp_group)
            upstream.append(gradient)
        owner.pending -= 1
        owner.completed_backwards += 1
        return None, None, *upstream


class NativeSerialPPProxy(NativeStaticPPProxy):
    """One physical B1 per invocation, original eight-sample optimizer boundary.

    A deliberately synchronous fallback. No default GPipe behavior is changed,
    no parameter sharing/merge/cache/fusion is added, and no native result is
    implied by this component's CPU tests.
    """
    def __init__(self, stage, *, pp_group, device, role):
        nn.Module.__init__(self)
        if not isinstance(stage, NativeSourcePPStage) or role not in ('generator', 'critic'):
            raise ValueError('requires admitted native role/stage')
        if dist.get_world_size(pp_group) != 2 or dist.get_rank(pp_group) != stage.stage_index:
            raise ValueError('serial PP2 group/coordinate mismatch')
        self.stage = stage
        self.pp_group, self.device, self.role = pp_group, torch.device(device), role
        self.first_rank = dist.get_global_rank(pp_group, 0)
        self.last_rank = dist.get_global_rank(pp_group, 1)
        self.peer = self.last_rank if stage.stage_index == 0 else self.first_rank
        self.static_cohort_batch_size = 1
        self.application_calls = self.grad_application_calls = self.local_stage_calls = 0
        self.pending = self.completed_backwards = 0
        self.peak_allocated_observed = self.peak_reserved_observed = 0
        self._distillschedule_has_adapter = True
        self._first_step = False

    def _send(self, tensor):
        dist.send(tensor.contiguous(), dst=self.peer, group=self.pp_group)

    def _receive(self, shape, dtype):
        tensor = torch.empty(shape, dtype=dtype, device=self.device)
        dist.recv(tensor, src=self.peer, group=self.pp_group)
        return tensor

    def _packet(self, value, source):
        values = [value]
        dist.broadcast_object_list(values, src=source, group=self.pp_group, device=self.device)
        return values[0]

    def _execute(self, inputs, *, grad):
        fields = self.stage.TRANSPORT_FIELDS
        if self.stage.stage_index == 0:
            context = self.stage.split.prefix(self.native_model, *inputs,
                return_dict=False, is_first_denoising_step=self._first_step)
            if context.get('gan_mode') or context.get('enable_navit'):
                raise ValueError('serial plain transport cannot silently accept GAN/NaViT')
            context = self.stage.split.blocks(self.native_model, context, self.stage.start, self.stage.stop)
            local = tuple(context[name].contiguous() for name in fields)
            metadata = {k: v for k, v in context.items() if k not in fields}
            if any(isinstance(v, torch.Tensor) for v in metadata.values()):
                raise ValueError('unregistered tensor in native prefix metadata')
            packet = dict(fields=_description(local), context=metadata)
        else:
            packet = None
        packet = self._packet(packet, self.first_rank)
        boundary = ()
        if self.stage.stage_index == 0:
            for value in local:
                self._send(value.detach())
        else:
            dtypes = {str(t): t for t in (torch.bfloat16, torch.float32, torch.float64)}
            boundary = tuple(self._receive(shape, dtypes[dtype]).requires_grad_(grad and needed)
                             for shape, dtype, needed in packet['fields'])
            context = dict(packet['context'])
            context.update(zip(fields, boundary, strict=True))
            context = self.stage.split.blocks(self.native_model, context, self.stage.start, self.stage.stop)
            output = self.stage.split.suffix(self.native_model, context)
            if not isinstance(output, tuple) or not output or not isinstance(output[0], torch.Tensor):
                raise ValueError('native suffix tuple signature changed')
            local = (output[0],)
        signature = self._packet(_description(local) if self.stage.stage_index == 1 else None,
                                 self.last_rank)
        if len(signature) != 1:
            raise ValueError('serial plain model output must be one latent tensor')
        if self.stage.stage_index == 1:
            for value in local:
                self._send(value.detach())
            result = local
        else:
            dtypes = {str(t): t for t in (torch.bfloat16, torch.float32, torch.float64)}
            result = tuple(self._receive(shape, dtypes[dtype]) for shape, dtype, _ in signature)
        self.local_stage_calls += 1
        return result, local, boundary

    def forward(self, *args, **kwargs):
        permitted = set(self.INPUT_NAMES) | {'return_dict', 'is_first_denoising_step', 'gan_mode', 'attention_kwargs'}
        if args or set(kwargs) - permitted or kwargs.get('gan_mode') or kwargs.get('attention_kwargs'):
            raise ValueError('unsupported serial native model call')
        if kwargs.get('return_dict', True):
            raise ValueError('serial native call requires tuple output')
        inputs = self._cohort_inputs(kwargs)
        self._first_step = bool(kwargs.get('is_first_denoising_step', False))
        grad = torch.is_grad_enabled()
        if grad:
            trigger = torch.zeros((), device=self.device, requires_grad=True)
            result = _SerialBoundary.apply(self, trigger, *inputs)
        else:
            result, _, _ = self._execute(inputs, grad=False)
        if result[0].shape != inputs[0].shape or result[0].dtype != torch.bfloat16:
            raise ValueError('serial native latent signature changed')
        self.application_calls += 1
        self.grad_application_calls += int(grad)
        self.note_memory_peak()
        return result

    def receipt(self):
        if self.application_calls != self.local_stage_calls or self.application_calls % 8 or self.grad_application_calls % 8:
            raise ValueError('serial native receipt requires all eight independent B1 samples')
        return dict(role=self.role, transport='SERIAL_B1_PP2_NOT_GPIPE1',
            application_forward_per_logical_sample=self.application_calls // 8,
            grad_forward_per_logical_sample=self.grad_application_calls // 8,
            actual_stage_B1_forward_count=self.local_stage_calls,
            actual_stage_B1_grad_forward_count=self.grad_application_calls,
            pending_backward_schedules=self.pending,
            pipeline_microbatches_per_invocation=1, logical_microbatches_per_DP=8,
            gradient_transport=[dict(completed_forward_backward_batches=self.completed_backwards,
                schedule='SERIAL_B1_PP2', gradient_scaling='OUTER_MEAN1_DIVIDE8_DP_SUM_DIVIDE2')],
            merged_lora=False, DMD_optimizations=False,
            peak_allocated_observed=self.peak_allocated_observed,
            peak_reserved_observed=self.peak_reserved_observed,
            GPU='NOT_RUN_COMPONENT_ONLY')
