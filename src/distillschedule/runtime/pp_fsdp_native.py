"""Opt-in PP2 stages with FSDP2, independent B1 data lanes, and mean16 updates.

Unlike the static TP reference, no projection is tensor-parallelized. Native
prefix/block/postfix math and the existing external-loss GPipe bridge are
reused. GPU and workload admission are owned by the experiment launcher.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import (
    FSDPModule, MixedPrecisionPolicy, fully_shard, register_fsdp_forward_method,
)

from .pp_fsdp_layout import PPFSDPLayout
from .static_gpipe_autograd import NativeBatchGPipeAutograd
from .static_gpipe_external_loss import ExternalLossGPipe, build_external_loss_gpipe
from .static_native_pp_stage import NativeSourcePPStage
from .static_native_proxy import NativeStaticPPProxy
from .static_pipeline import build_static_gpipe


class NativeFSDPSourcePPStage(NativeSourcePPStage):
    def prefix_context(self, *args, **kwargs):
        if self.stage_index != 0:
            raise ValueError('only stage0 owns native prefix parameters')
        return self.split.prefix(self.model, *args, **kwargs)

    def forward(self, *args, **kwargs):
        if self.stage_index == 0:
            return super().forward(*args, **kwargs)
        metadata = kwargs.pop('pp_context_metadata', None)
        if kwargs or len(args) != len(self.TRANSPORT_FIELDS) or not isinstance(metadata, dict):
            raise ValueError('stage1 needs its exact per-call native context')
        if any(isinstance(v, torch.Tensor) for v in metadata.values()):
            raise ValueError('unregistered PP conditioning tensor')
        context = dict(metadata)
        context.update(zip(self.TRANSPORT_FIELDS, args, strict=True))
        context = self.split.blocks(self.model, context, self.start, self.stop)
        output = self.split.suffix(self.model, context)
        if not isinstance(output, tuple) or not isinstance(output[0], torch.Tensor):
            raise ValueError('native postfix output contract changed')
        return output[0]


def shard_native_pp_stage(stage, *, mesh, layout, param_dtype=None):
    if not isinstance(stage, NativeFSDPSourcePPStage) or not isinstance(layout, PPFSDPLayout):
        raise ValueError('an admitted source-derived stage and PP/FSDP layout are required')
    if tuple(mesh.shape) != (layout.replicate, layout.shard):
        raise ValueError('FSDP mesh must exclude the PP dimension')
    if param_dtype not in (None, torch.bfloat16):
        raise ValueError('no precision changes beyond the source policy')
    policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=torch.float32,
                                  output_dtype=None, cast_forward_inputs=False)
    for block in stage.model.blocks[stage.start:stage.stop]:
        fully_shard(block, mesh=mesh, reshard_after_forward=True, mp_policy=policy)
    fully_shard(stage, mesh=mesh, reshard_after_forward=True, mp_policy=policy)
    register_fsdp_forward_method(stage, 'prefix_context')
    count = 0
    for module in stage.modules():
        if isinstance(module, FSDPModule):
            module.set_force_sum_reduction_for_comms(True)
            module.set_gradient_divide_factor(8.0)
            count += 1
    stage._pp_fsdp_policy = dict(TP=1, fsdp_units=count, shard=layout.shard,
        replicate=layout.replicate, param_dtype=str(param_dtype), reduce_dtype='torch.float32',
        cast_forward_inputs=False, reshard_after_forward=True,
        reduction='SUM_THEN_DIVIDE8_DATA_LANES; LOSS_ALREADY_MEAN2_WAVES',
        prefix_prepare='REGISTERED_FSDP_FORWARD_METHOD_NOT_BYPASSED')
    return stage._pp_fsdp_policy


class _StageCall(nn.Module):
    def __init__(self, stage, metadata, first_step, required):
        super().__init__()
        self.stage, self.metadata, self.first_step, self.required = stage, metadata, first_step, required
        self.calls = self.grad_calls = 0

    def forward(self, *args):
        if not args or args[0].shape[0] != 1:
            raise ValueError('PP/FSDP executes actual B1, never a B2 model forward')
        self.calls += 1
        self.grad_calls += int(torch.is_grad_enabled())
        if self.stage.stage_index == 1:
            return self.stage(*args, pp_context_metadata=self.metadata)
        fields = self.stage(*args, is_first_denoising_step=self.first_step)
        if self.required is not None:
            for field, tensor, required in zip(self.stage.TRANSPORT_FIELDS, fields, self.required, strict=True):
                if tensor.requires_grad != required:
                    raise ValueError('native PP boundary gradient owner changed: ' + field)
        return fields


class NativePPFSDPProxy(NativeStaticPPProxy):
    def __init__(self, stage, *, pp_group, device, role):
        if not hasattr(stage, '_pp_fsdp_policy'):
            raise ValueError('the stage must already be genuinely fully_shard-ed')
        super().__init__(stage, pp_group=pp_group, device=device, role=role, pipeline_microbatches=2)

    def _prepare(self, inputs, first_step, grad):
        key = json.dumps(dict(args=[(list(t.shape[1:]), str(t.dtype), t.requires_grad) for t in inputs],
                              first_step=first_step, grad=grad), sort_keys=True)
        if key in self._schedules:
            return self._schedules[key]
        micro = tuple(t[:1].detach().requires_grad_(grad and t.requires_grad) for t in inputs)
        packet, boundary = [None], None
        if self.stage.stage_index == 0:
            with torch.set_grad_enabled(grad):
                context = self.stage.prefix_context(*micro, return_dict=False,
                                                    is_first_denoising_step=first_step)
            metadata = {k: v for k, v in context.items() if k not in self.stage.TRANSPORT_FIELDS}
            if any(isinstance(v, torch.Tensor) for v in metadata.values()):
                raise ValueError('unregistered prefix context tensor')
            required = [bool(context[n].requires_grad) for n in self.stage.TRANSPORT_FIELDS]
            required[0] |= grad and any(p.requires_grad for p in
                self.native_model.blocks[self.stage.start:self.stage.stop].parameters())
            packet[0] = dict(metadata=metadata,
                fields=[(list(context[n].shape), str(context[n].dtype)) for n in self.stage.TRANSPORT_FIELDS],
                boundary_gradient_required=required)
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
        options = dict(stage_index=self.stage.stage_index, group=self.pp_group, device=self.device,
            input_args=micro if self.stage.stage_index == 0 else boundary,
            output_args=boundary if self.stage.stage_index == 0 else (output_example,))
        if grad:
            schedule = build_external_loss_gpipe(module, **options, final_output_args=(output_example,),
                constant_boundary_inputs=tuple(i for i, needed in enumerate(required) if not needed),
                boundary_input_names=self.stage.TRANSPORT_FIELDS, pipeline_microbatches=2)
            run = NativeBatchGPipeAutograd(schedule, stage_index=self.stage.stage_index,
                                          group=self.pp_group, device=self.device)
        else:
            _stage, schedule = build_static_gpipe(module, **options, pipeline_microbatches=2)
            run = None
        self._schedules[key] = (schedule, run)
        return schedule, run

    def receipt(self):
        pending = sum(int(s.pending) for s, _ in self._schedules.values() if isinstance(s, ExternalLossGPipe))
        calls = sum(s._stage.submod.calls for s, _ in self._schedules.values())
        grad_calls = sum(s._stage.submod.grad_calls for s, _ in self._schedules.values())
        if calls % 2 or grad_calls % 2:
            raise ValueError('both independent B1 waves must have completed')
        return dict(role=self.role, application_forward_per_logical_sample=calls//2,
            grad_forward_per_logical_sample=grad_calls//2, actual_stage_B1_forward_count=calls,
            actual_stage_B1_grad_forward_count=grad_calls, microbatch_size=1, microbatches=2,
            pending_backward_schedules=pending, pipeline_microbatches_per_invocation=2,
            logical_microbatches_per_data_lane=2, TP=1, fsdp=self.stage._pp_fsdp_policy,
            stage=self.stage.receipt(), merged_lora=False, DMD_optimizations=False,
            peak_allocated_observed_bytes=self.peak_allocated_observed,
            peak_reserved_observed_bytes=self.peak_reserved_observed)


class PPFSDPTrainableState:
    """FSDP performs mean8; clip includes both PP halves but no DP duplicates."""
    def __init__(self, native, *, mesh, pp_group, max_norm=1.0):
        self.placements = tuple(SimpleNamespace(name=n, parameter=p)
            for n,p in sorted(native.named_parameters()) if p.requires_grad)
        if not self.placements or any(not hasattr(x.parameter, 'to_local') for x in self.placements):
            raise ValueError('all trainable parameters must be real FSDP2 DTensors')
        self.mesh, self.pp_group, self.max_norm = mesh, pp_group, max_norm
        self.accumulations = self.commits = 0

    def mean_microbatch_group_loss(self, loss, *, group_size):
        if group_size != 2 or self.accumulations or loss.ndim != 0:
            raise ValueError('one mean2 B1-wave loss per update, not two commits')
        self.accumulations = 2
        return loss

    @torch.no_grad()
    def synchronize_and_clip(self):
        if self.accumulations != 2:
            raise ValueError('both B1 waves must contribute before commit')
        parameters = [x.parameter for x in self.placements]
        if any(p.grad is None or p.grad.dtype != torch.float32 for p in parameters):
            raise ValueError('source FP32 trainable gradient is missing')
        local = torch.zeros((), device=parameters[0].to_local().device, dtype=torch.float32)
        for p in parameters:
            local.add_(p.grad.to_local().square().sum())
        dist.all_reduce(local, group=self.mesh['fsdp'].get_group())
        dist.all_reduce(local, group=self.pp_group)
        norm = local.sqrt()
        factor = (self.max_norm/(norm+1e-6)).clamp(max=1)
        for p in parameters:
            p.grad.to_local().mul_(factor)
        self.accumulations = 0
        self.commits += 1
        return norm

    def receipt(self):
        return dict(TP=1, global_batch=16, microbatch_size=1, microbatches_per_data_lane=2,
            reduction='FSDP_MEAN8_AFTER_MICROBATCH_MEAN2',
            grad_norm='SUM_FSDP_SHARDS_AND_PP_STAGES_ONCE; NO_REPLICA_DOUBLE_COUNT',
            commits=self.commits, pending_microbatches=self.accumulations)
