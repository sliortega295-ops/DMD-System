"""Helios logical-state and precision binding for the static 3D reference.

Unlike Wan's systems-only bootstrap, the 14B checkpoint has real Adam and EMA
state. Preserve it, including dtype and physical step; never replace it with
fresh zeros. This module does not change canonical workload defaults.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .static_3d_state import shard_bootstrap_tensor
from .fsdp2_static import _logical_tensor, ema_update_reference


class MasterComputeProjection(nn.Module):
    """Original Linear/Conv3d equation, FP32 master and BF16 compute view.

    Helios FSDP2's param_dtype=BF16 does this conversion at unshard. Static
    placement needs an explicit equivalent instead of accidentally inheriting
    Wan's FP32 LoRA compute. PEFT still owns adapter enable/disable and scaling.
    """
    def __init__(self, original):
        super().__init__()
        if type(original) not in (nn.Linear, nn.Conv3d):
            raise ValueError('only original Linear/Conv3d LoRA projections admitted')
        if original.weight.dtype != torch.float32 or not original.weight.requires_grad:
            raise ValueError('LoRA master must remain trainable FP32')
        self.weight = original.weight
        self.bias = original.bias
        self._conv_forward = original._conv_forward if isinstance(original, nn.Conv3d) else None
        self.forward_count = 0

    def forward(self, value):
        weight = self.weight.to(torch.bfloat16)
        bias = self.bias.to(torch.bfloat16) if self.bias is not None else None
        self.forward_count += 1
        value = value.to(torch.bfloat16)
        if self._conv_forward is None:
            return F.linear(value, weight, bias)
        return self._conv_forward(value, weight, bias)


@torch.no_grad()
def mirror_fsdp_frozen_compute_dtype(model):
    """Frozen casts are constant; match the reference FSDP2 BF16 views."""
    converted = 0
    for parameter in model.parameters():
        if not parameter.requires_grad and parameter.is_floating_point() and parameter.dtype != torch.bfloat16:
            parameter.data = parameter.data.to(torch.bfloat16)
            converted += 1
    return dict(frozen_compute_dtype='BF16', converted_constant_tensors=converted,
                rule='MATCH_REFERENCE_FSDP2_PARAM_DTYPE_NOT_AN_OPTIMIZATION')


def bind_helios_bf16_lora_compute(model):
    from .static_tp_lora import _PlacedLinear

    count = 0
    for module in list(model.modules()):
        if not hasattr(module, 'lora_A') or not hasattr(module, 'lora_B'):
            continue
        if getattr(module, 'merged', False):
            raise ValueError('merged LoRA is excluded from the static reference')
        for dictionary in (module.lora_A, module.lora_B):
            for key, projection in dictionary.items():
                if isinstance(projection, _PlacedLinear):
                    if projection.weight.dtype != torch.float32:
                        raise ValueError('TP LoRA master changed dtype')
                    projection.compute_dtype = torch.bfloat16
                else:
                    dictionary[key] = MasterComputeProjection(projection)
                count += 1
    if not count:
        raise ValueError('missing original LoRA projections')
    return dict(master_dtype='FP32', forward_compute_dtype='BF16', projections=count,
                precision_source='REFERENCE_FSDP2_MIXED_PRECISION_POLICY_PARAM_DTYPE_BF16',
                merged_count=0)


def restore_logical_static_optimizer(*, optimizer, scheduler, placements,
                                     package, component, tp_rank):
    spec = package.component(component)
    if spec['optimizer']['algorithm'] != 'AdamW':
        raise ValueError('logical optimizer algorithm changed')
    source_groups = spec['optimizer']['param_groups']
    if len(source_groups) != 1 or len(optimizer.param_groups) != 1:
        raise ValueError('only the source single optimizer group is admitted')
    for key in ('lr', 'betas', 'eps', 'weight_decay'):
        wanted, actual = source_groups[0][key], optimizer.param_groups[0][key]
        if (tuple(wanted) if key == 'betas' else wanted) != actual:
            raise ValueError('logical optimizer hyperparameter changed: '+key)
    known = set(spec['parameter_names'])
    if not placements or len({p.name for p in placements}) != len(placements):
        raise ValueError('missing/duplicate owned placements')
    if any(p.name not in known for p in placements):
        raise ValueError('owned trainable absent from logical state')
    optimizer.state.clear()
    restored = 0
    for item in placements:
        state = optimizer.state[item.parameter]
        state['step'] = torch.tensor(float(spec['optimizer_step']), dtype=torch.float32)
        for key, payload in (('exp_avg', 'first_moment'), ('exp_avg_sq', 'second_moment')):
            full = _logical_tensor(package, component, payload, item.name)
            local = shard_bootstrap_tensor(full, item, tp_rank)
            state[key] = local.to(device=item.parameter.device, dtype=torch.float32, copy=True)
            restored += local.numel()
    scheduler.load_state_dict(dict(spec['scheduler_state']))
    return dict(optimizer_step=int(spec['optimizer_step']),
                scheduler_step=int(scheduler.last_epoch), owned_tensors=len(placements),
                restored_local_moment_numel=restored,
                moments='EXACT_LOGICAL_FP32_PAYLOAD_TO_DECLARED_TP_PLACEMENT')


class StaticLogicalEMA:
    """Unchanged sparse EMA recurrence over the uniquely owned TP/PP subset."""
    def __init__(self, *, placements, package, tp_rank, device):
        root = package.root / 'ema'
        metadata = json.loads((root/'sparse_ema_manifest.json').read_text())
        if metadata['schema_version'] != 'distillschedule.sparse_ema.v1':
            raise ValueError('unadmitted EMA checkpoint')
        normalize = lambda n: n.removeprefix('module.')
        all_names = {normalize(n) for n in metadata['tracked_names']}
        if all_names != set(package.component('generator')['parameter_names']):
            raise ValueError('EMA/logical Generator trainable inventory mismatch')
        owned = {p.name: p for p in placements}
        if not owned or not set(owned) <= all_names:
            raise ValueError('owned EMA placement is absent from source')
        values = {}
        # Shared read-only mmap checkpoint pages, not eight private full EMA
        # CPU copies. Only owned slices become GPU shadows, before the timer.
        for shard in metadata['shards']:
            tensors = torch.load(root/shard['file'], map_location='cpu', weights_only=True, mmap=True)
            for name, full in tensors.items():
                key = normalize(name)
                if key in owned:
                    if key in values:
                        raise ValueError('duplicate logical EMA tensor')
                    dtype = full.dtype
                    values[key] = shard_bootstrap_tensor(full.float(), owned[key], tp_rank).to(
                        device=device, dtype=dtype, copy=True)
            del tensors
        if set(values) != set(owned):
            raise ValueError('logical EMA restore did not cover owned state')
        self._parameters = [p.parameter for p in placements]
        self._shadows = [values[p.name] for p in placements]
        self.decay = float(metadata['decay'])
        self.update_after_step = int(metadata['update_after_step'])
        self.optimization_step = int(metadata['optimization_step'])
        self.initial_step = self.optimization_step
        self.cur_decay_value = metadata.get('cur_decay_value')
        if self.optimization_step != int(package.manifest['ema']['optimization_step']):
            raise ValueError('logical/sparse EMA version mismatch')

    def get_decay(self, optimization_step):
        step = max(0, optimization_step-self.update_after_step-1)
        return 0. if step <= 0 else max(0., min((1+step)/(10+step), self.decay))

    @torch.no_grad()
    def step(self):
        self.optimization_step += 1
        decay = self.get_decay(self.optimization_step)
        self.cur_decay_value = decay
        for shadow, parameter in zip(self._shadows, self._parameters, strict=True):
            ema_update_reference(shadow, parameter.detach().to(shadow.dtype), decay)
        return decay

    def receipt(self):
        return dict(initial_optimization_step=self.initial_step,
                    optimization_step=self.optimization_step, tracked_owned_tensors=len(self._shadows),
                    shadow_dtypes=sorted({str(t.dtype) for t in self._shadows}),
                    local_bytes=sum(t.numel()*t.element_size() for t in self._shadows),
                    recurrence='ORIGINAL_EMA_REFERENCE', source='SPARSE_LOGICAL_CHECKPOINT')
