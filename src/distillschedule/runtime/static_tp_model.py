"""Explicit TP4 binding for the pinned Wan/Helios transformer structure.

This is NOT a PP runner, optimizer, or timing baseline. It keeps the original
block/attention/PEFT forwards, and partitions only their projection children.
The caller must bind a checked source SHA and supply an initialized TP4 group.
No automatic installation or canonical default changes.
"""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn

from .static_tp_lora import StaticTPRMSNorm, _PlacedLinear, bind_peft_linear_tp


def bind_wan_helios_tp4(model, *, group, expected_transformer_source_sha256):
    from peft.tuners.lora.layer import Linear

    if not dist.is_initialized() or dist.get_world_size(group) != 4:
        raise ValueError('native static layout requires initialized TP4')
    if model.__class__.__name__ != 'HeliosTransformer3DModel':
        raise ValueError('no implicit binding of a different transformer architecture')
    source = Path(inspect.getsourcefile(type(model)))
    observed = hashlib.sha256(source.read_bytes()).hexdigest()
    if observed != expected_transformer_source_sha256:
        raise ValueError('transformer source does not match the admitted checkpoint/runtime')
    blocks = getattr(model, 'blocks', None)
    if blocks is None:
        blocks = getattr(model, 'transformer_blocks', None)
    if blocks is None or len(blocks) not in (30, 40):
        raise ValueError('only the checked Wan30/Helios40 block contract is bound')
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and not any(k in name for k in ('.lora_A.', '.lora_B.')):
            raise ValueError('non-LoRA trainable parameter requires its own TP/DP placement')
    if getattr(model, '_static_tp4_bound', False):
        raise ValueError('model has already been partitioned')
    projections = []

    def validate_projection(child, style, path):
        """Fail before mutating any child of the admitted transformer."""
        if isinstance(child, Linear):
            if child.merged or getattr(child, '_static_tp_bound', False):
                raise ValueError('merged/already-partitioned adapter: '+path)
            if any(getattr(child, 'use_dora', {}).values()):
                raise ValueError('ordinary LoRA required: '+path)
            if any(getattr(d, 'p', 0) != 0 for d in child.lora_dropout.values()):
                raise ValueError('adapter dropout needs an explicit RNG contract: '+path)
            base=child.base_layer
            for adapter, a in child.lora_A.items():
                b=child.lora_B[adapter]
                if a.weight.dtype != torch.float32 or b.weight.dtype != torch.float32:
                    raise ValueError('LoRA compute must remain FP32: '+path)
                if (a.weight.shape[1] != base.weight.shape[1]
                        or b.weight.shape[0] != base.weight.shape[0]
                        or a.weight.shape[0] != b.weight.shape[1]
                        or a.bias is not None or b.bias is not None):
                    raise ValueError('unsupported LoRA shapes/bias: '+path)
        elif isinstance(child, nn.Linear):
            base=child
        else:
            raise ValueError('unsupported projection child: '+path)
        if any(p.requires_grad for p in base.parameters()) or base.weight.dtype != torch.bfloat16:
            raise ValueError('frozen projection must remain frozen BF16: '+path)
        dim=0 if style=='column' else 1
        if base.weight.shape[dim] % 4:
            raise ValueError('TP4 projection dimensions must divide: '+path)

    def linear(parent, name, style, path):
        child = parent[int(name)] if isinstance(parent, nn.ModuleList) else getattr(parent, name)
        if isinstance(child, Linear):
            placed = bind_peft_linear_tp(child, group=group, style=style)
            kind = 'PEFT_SHELL_PRESERVED'
        elif isinstance(child, nn.Linear):
            if any(p.requires_grad for p in child.parameters()) or child.weight.dtype != torch.bfloat16:
                raise ValueError('frozen projection must remain frozen BF16')
            placed = _PlacedLinear(child, group=group,
                placement='output' if style == 'column' else 'input',
                exchange='copy_input_backward_sum' if style == 'column' else 'row_output_sum', frozen=True)
            kind = 'FROZEN_BASE_ONLY'
        else:
            raise ValueError('unsupported projection child: '+path)
        if isinstance(parent, nn.ModuleList):
            parent[int(name)] = placed
        else:
            setattr(parent, name, placed)
        projections.append(dict(path=path, style=style, kind=kind))

    # Validate the full graph before mutating any module.
    for index, block in enumerate(blocks):
        for name in ('attn1', 'attn2'):
            attention = getattr(block, name)
            if attention.__class__.__name__ != 'HeliosAttention' or attention.heads % 4:
                raise ValueError('TP4 must partition complete attention heads')
            if (getattr(attention, 'fused_projections', False)
                    or getattr(attention, 'restrict_lora', False)
                    or attention.added_kv_proj_dim is not None
                    or getattr(attention, 'is_amplify_history', False)):
                raise ValueError('unsupported fused/restricted/image/history-scale attention path')
            if getattr(attention.to_out[1], 'p', 0) != 0:
                raise ValueError('attention dropout needs an explicit replicated RNG contract')
            if not all(isinstance(getattr(attention, n), nn.RMSNorm) for n in ('norm_q', 'norm_k')):
                raise ValueError('requires original across-head RMSNorm')
            for projection in ('to_q','to_k','to_v'):
                validate_projection(getattr(attention,projection),'column',f'block{index}.{name}.{projection}')
            validate_projection(attention.to_out[0],'row',f'block{index}.{name}.to_out.0')
            for norm_name in ('norm_q','norm_k'):
                norm=getattr(attention,norm_name)
                if norm.weight.requires_grad or norm.weight.ndim != 1 or norm.weight.numel() % 4:
                    raise ValueError('unsupported global RMS weight: '+name+'.'+norm_name)
        ffn = block.ffn
        if ffn.net[0].__class__.__name__ not in ('GELU', 'ApproximateGELU'):
            raise ValueError('gated FFN requires paired-column sharding; never slice halves blindly')
        if getattr(ffn.net[1], 'p', 0) != 0:
            raise ValueError('FFN dropout requires an explicit replicated RNG contract')
        validate_projection(ffn.net[0].proj,'column',f'block{index}.ffn.net.0.proj')
        validate_projection(ffn.net[2],'row',f'block{index}.ffn.net.2')

    for index, block in enumerate(blocks):
        for name in ('attn1', 'attn2'):
            attention = getattr(block, name)
            for projection in ('to_q', 'to_k', 'to_v'):
                linear(attention, projection, 'column', f'block{index}.{name}.{projection}')
            linear(attention.to_out, '0', 'row', f'block{index}.{name}.to_out.0')
            for norm_name in ('norm_q', 'norm_k'):
                original = getattr(attention, norm_name)
                setattr(attention, norm_name, StaticTPRMSNorm(original.weight.detach(),
                    group=group, eps=original.eps))
            attention.heads //= 4
            attention.inner_dim //= 4
            attention.kv_inner_dim //= 4
        linear(block.ffn.net[0], 'proj', 'column', f'block{index}.ffn.net.0.proj')
        linear(block.ffn.net, '2', 'row', f'block{index}.ffn.net.2')
    model._static_tp4_bound = True
    return dict(status='BOUND_TP4_ONLY_PP_DP_OPTIMIZER_NOT_IMPLEMENTED',
                source_sha256=observed, blocks=len(blocks), projections=projections,
                original_forward_preserved=True, original_peft_adapter_control_preserved=True,
                global_qk_rms_norm=True, merged_lora=False, dmd_optimizations=False,
                timing='NOT_RUN_NOT_A_COMPLETE_STATIC_3D_BASELINE')
