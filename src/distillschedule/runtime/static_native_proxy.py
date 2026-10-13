"""Native Wan PP2/TP4 model proxy; eight actual B1 forwards per DP cohort.

No adapter merge, kernel replacement, compute cache, or schedule reordering.
The source-derived prefix/postfix retain their original math. Shapes and scalar
context are admitted once per call signature, never guessed from GPU timing.
"""
from __future__ import annotations

import json

import torch
import torch.distributed as dist
from torch import nn

from .static_gpipe_autograd import NativeBatchGPipeAutograd
from .static_gpipe_external_loss import ExternalLossGPipe, build_external_loss_gpipe
from .static_native_pp_stage import NativeSourcePPStage
from .static_pipeline import build_static_gpipe


class _StageCall(nn.Module):
    def __init__(self, stage, metadata, first_step, boundary_gradient_required=None):
        super().__init__()
        # Each live checkpoint graph retains its native block arguments. The
        # stage itself is shared only within this one physical role, not G/C.
        self.stage = stage
        self.metadata, self.first_step = metadata, first_step
        self.boundary_gradient_required = boundary_gradient_required
        self.calls = self.grad_calls = 0

    def forward(self, *args):
        if not args or args[0].shape[0] != 1:
            raise ValueError('actual PP stage must execute B1, not a batched model')
        self.calls += 1
        self.grad_calls += int(torch.is_grad_enabled())
        if self.stage.stage_index == 0:
            fields = self.stage(*args, is_first_denoising_step=self.first_step)
            if self.boundary_gradient_required is not None:
                for name, tensor, required in zip(self.stage.TRANSPORT_FIELDS, fields,
                                                  self.boundary_gradient_required, strict=True):
                    if tensor.requires_grad != required:
                        raise ValueError('native PP boundary gradient owner changed: '+name)
            return fields
        context = dict(self.metadata)
        context.update(zip(self.stage.TRANSPORT_FIELDS, args, strict=True))
        context = self.stage.split.blocks(self.stage.model, context,
                                         self.stage.start, self.stage.stop)
        output = self.stage.split.suffix(self.stage.model, context)
        if not isinstance(output, tuple) or not isinstance(output[0], torch.Tensor):
            raise ValueError('native last stage no longer returns a tensor tuple')
        return output[0]


class NativeStaticPPProxy(nn.Module):
    INPUT_NAMES = ('hidden_states', 'timestep', 'encoder_hidden_states',
        'indices_hidden_states', 'indices_latents_history_short',
        'indices_latents_history_mid', 'indices_latents_history_long',
        'latents_history_short', 'latents_history_mid', 'latents_history_long')

    def __init__(self, stage, *, pp_group, device, role):
        super().__init__()
        if not isinstance(stage, NativeSourcePPStage) or role not in ('generator', 'critic'):
            raise ValueError('requires a source-admitted Wan role stage')
        if dist.get_world_size(pp_group) != 2 or dist.get_rank(pp_group) != stage.stage_index:
            raise ValueError('PP coordinates/group mismatch')
        self.stage = stage
        self.pp_group, self.device, self.role = pp_group, torch.device(device), role
        self._schedules = {}
        self.application_calls = self.grad_application_calls = 0
        self.peak_allocated_observed = self.peak_reserved_observed = 0
        self._distillschedule_has_adapter = True
        self.static_cohort_batch_size = 8

    @property
    def native_model(self):
        return self.stage.model

    def enable_adapters(self):
        return self.native_model.enable_adapters()

    def disable_adapters(self):
        return self.native_model.disable_adapters()

    def _cohort_inputs(self, values):
        result = []
        for name in self.INPUT_NAMES:
            value = values.get(name)
            if value is None:
                raise ValueError('native cohort requires explicit history and index tensors: '+name)
            if not isinstance(value, torch.Tensor):
                raise ValueError('only actual tensors cross native stage0 inputs')
            if value.ndim == 0:
                value = value.reshape(1)
            if name.startswith('indices_') and value.ndim == 1:
                value = value.unsqueeze(0)
            if value.shape[0] == 1:
                value = value.expand(8, *value.shape[1:])
            if value.shape[0] != 8:
                raise ValueError('native cohort lost eight independent sample rows: '+name)
            # PP validates strides as well as shape/dtype. A shared B1
            # condition expands with stride0, whereas per-sample conditions
            # arrive contiguous; canonicalize their transport, preserving
            # values and autograd, before reusing a static schedule.
            result.append(value.contiguous())
        return tuple(result)

    def _prepare(self, inputs, first_step, grad):
        key = json.dumps(dict(args=[(list(t.shape[1:]), str(t.dtype), t.requires_grad) for t in inputs],
                              first_step=first_step, grad=grad), sort_keys=True)
        if key in self._schedules:
            return self._schedules[key]
        micro = tuple(t[:1].detach().requires_grad_(grad and t.requires_grad) for t in inputs)
        packet = [None]
        boundary = None
        if self.stage.stage_index == 0:
            with torch.set_grad_enabled(grad):
                context = self.stage.split.prefix(self.native_model, *micro,
                    return_dict=False, is_first_denoising_step=first_step)
            metadata = {k: v for k, v in context.items() if k not in self.stage.TRANSPORT_FIELDS}
            if any(isinstance(v, torch.Tensor) for v in metadata.values()):
                raise ValueError('unregistered native prefix tensor in stage metadata')
            required = [bool(context[n].requires_grad) for n in self.stage.TRANSPORT_FIELDS]
            # Hidden-state ownership also includes the trainable native blocks
            # before this boundary. Each actual stage0 call audits the result.
            required[0] |= grad and any(p.requires_grad for p in
                self.native_model.blocks[self.stage.start:self.stage.stop].parameters())
            packet[0] = dict(metadata=metadata, fields=[(list(context[n].shape), str(context[n].dtype))
                                for n in self.stage.TRANSPORT_FIELDS], boundary_gradient_required=required)
            boundary = tuple(context[n].detach().contiguous() for n in self.stage.TRANSPORT_FIELDS)
        dist.broadcast_object_list(packet, src=dist.get_global_rank(self.pp_group, 0),
                                   group=self.pp_group, device=self.device)
        if self.stage.stage_index == 1:
            dtypes = {str(t): t for t in (torch.bfloat16, torch.float32, torch.float64)}
            boundary = tuple(torch.empty(shape, dtype=dtypes[dtype], device=self.device)
                             for shape, dtype in packet[0]['fields'])
        output_example = inputs[0][:1].detach()
        required = tuple(packet[0]['boundary_gradient_required']) if grad else None
        module = _StageCall(self.stage, packet[0]['metadata'], first_step, required)
        options = dict(stage_index=self.stage.stage_index, group=self.pp_group,
            device=self.device, input_args=micro if self.stage.stage_index == 0 else boundary,
            output_args=boundary if self.stage.stage_index == 0 else (output_example,))
        if grad:
            schedule = build_external_loss_gpipe(module, **options, final_output_args=(output_example,),
                constant_boundary_inputs=tuple(i for i, needed in enumerate(required) if not needed),
                boundary_input_names=self.stage.TRANSPORT_FIELDS)
            run = NativeBatchGPipeAutograd(schedule, stage_index=self.stage.stage_index,
                                          group=self.pp_group, device=self.device)
        else:
            _stage, schedule = build_static_gpipe(module, **options)
            run = None
        self._schedules[key] = (schedule, run)
        return schedule, run

    def forward(self, *args, **kwargs):
        if args:
            raise ValueError('native model calls must use the explicit keyword contract')
        permitted = set(self.INPUT_NAMES) | {'return_dict', 'is_first_denoising_step', 'gan_mode', 'attention_kwargs'}
        if set(kwargs) - permitted or kwargs.get('gan_mode') or kwargs.get('attention_kwargs'):
            raise ValueError('unsupported native 3D model call')
        if kwargs.get('return_dict', True):
            raise ValueError('native 3D baseline requires tuple output')
        inputs = self._cohort_inputs(kwargs)
        grad = torch.is_grad_enabled()
        schedule, run = self._prepare(inputs, bool(kwargs.get('is_first_denoising_step', False)), grad)
        if grad:
            output = run(*inputs)
        else:
            output = schedule.step(*inputs)
            transported = (output.detach().contiguous() if self.stage.stage_index == 1
                           else torch.empty_like(inputs[0]))
            dist.broadcast(transported, src=dist.get_global_rank(self.pp_group, 1), group=self.pp_group)
            output = transported
        if output.shape != inputs[0].shape or output.dtype != torch.bfloat16:
            raise ValueError('native latent output signature changed')
        self.application_calls += 1
        self.grad_application_calls += int(grad)
        if self.device.type == 'cuda':
            # Read allocator counters while the B1 samplers are blocked; their
            # native phase-local telemetry may later reset CUDA peak counters.
            self.note_memory_peak()
        return (output,)

    def note_memory_peak(self):
        if self.device.type == 'cuda':
            self.peak_allocated_observed = max(self.peak_allocated_observed,
                torch.cuda.max_memory_allocated(self.device))
            self.peak_reserved_observed = max(self.peak_reserved_observed,
                torch.cuda.max_memory_reserved(self.device))

    def receipt(self):
        pending = sum(bool(getattr(s, 'pending', False)) for s, _ in self._schedules.values())
        calls = sum(s._stage.submod.calls for s, _ in self._schedules.values())
        grad_calls = sum(s._stage.submod.grad_calls for s, _ in self._schedules.values())
        if calls != 8 * self.application_calls or grad_calls != 8 * self.grad_application_calls:
            raise ValueError('actual B1 stage forward census differs from native cohort calls')
        return dict(role=self.role, application_forward_per_logical_sample=self.application_calls,
            grad_forward_per_logical_sample=self.grad_application_calls,
            actual_stage_B1_forward_count=calls, actual_stage_B1_grad_forward_count=grad_calls,
            microbatch_size=1, microbatches=8, pending_backward_schedules=pending,
            peak_allocated_observed_bytes=self.peak_allocated_observed,
            peak_reserved_observed_bytes=self.peak_reserved_observed,
            gradient_transport=[s.receipt() for s, _ in self._schedules.values()
                                if isinstance(s, ExternalLossGPipe)],
            stage=self.stage.receipt(), merged_lora=False, DMD_optimizations=False)
