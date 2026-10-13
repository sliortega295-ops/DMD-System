"""LoRA-aware TP linear primitives for the static 3D baseline prototype.

These primitives do not install themselves, change PEFT, create process groups,
or imply a native DMD/PP executor. DP synchronization and global clipping still
need the model binding to distinguish replicated from partitioned parameters.
"""
from __future__ import annotations
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F


class _CopyToTP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, group):
        ctx.group=group
        return value

    @staticmethod
    def backward(ctx, gradient):
        value=gradient.contiguous().clone()
        dist.all_reduce(value,op=dist.ReduceOp.SUM,group=ctx.group)
        return value,None


class _ReduceFromTP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, group):
        output=value.contiguous().clone()
        dist.all_reduce(output,op=dist.ReduceOp.SUM,group=group)
        return output

    @staticmethod
    def backward(ctx, gradient):
        return gradient,None


class StaticTPLoRALinear(nn.Module):
    """Column: replicated A/partitioned B; row: partitioned A/replicated B.

    Input/output activation partitioning is explicit: column takes replicated
    input and returns local output; row takes local input and returns replicated
    output. The frozen base and unmerged LoRA branches retain their dtypes.
    """
    def __init__(self, base_weight, lora_a, lora_b, *, group, style, scale,
                 base_bias=None, cpu_reference_dtype=False):
        super().__init__()
        if not dist.is_initialized() or dist.get_world_size(group)!=4:
            raise ValueError('static baseline requires an initialized TP4 group')
        if style not in ('column','row'):
            raise ValueError('unknown TP linear placement')
        if cpu_reference_dtype:
            if base_weight.device.type!='cpu' or base_weight.dtype!=torch.float64:
                raise ValueError('FP64 reference override is CPU-test-only')
        elif base_weight.dtype!=torch.bfloat16:
            raise ValueError('real frozen base compute must remain BF16')
        if lora_a.dtype!=torch.float32 or lora_b.dtype!=torch.float32:
            raise ValueError('unmerged LoRA compute must remain FP32')
        out_features,in_features=base_weight.shape
        if (lora_a.shape[1]!=in_features or lora_b.shape[0]!=out_features
                or lora_a.shape[0]!=lora_b.shape[1]):
            raise ValueError('incompatible base/A/B shapes')
        if in_features%4 or out_features%4:
            raise ValueError('static TP4 dimensions must divide evenly')
        self.group,self.style,self.scale=group,style,float(scale)
        self.tp_rank=dist.get_rank(group)
        dim=0 if style=='column' else 1
        self.base_weight=nn.Parameter(base_weight.chunk(4,dim=dim)[self.tp_rank].clone(),requires_grad=False)
        self.lora_A=nn.Parameter(lora_a.clone() if style=='column' else lora_a.chunk(4,dim=1)[self.tp_rank].clone())
        self.lora_B=nn.Parameter(lora_b.chunk(4,dim=0)[self.tp_rank].clone() if style=='column' else lora_b.clone())
        if base_bias is None:
            self.register_parameter('base_bias',None)
        else:
            # Row bias is added once AFTER the SUM, not once per TP rank.
            bias=base_bias.chunk(4)[self.tp_rank] if style=='column' else base_bias
            self.base_bias=nn.Parameter(bias.clone(),requires_grad=False)

    def forward(self,x):
        if self.style=='column':
            base=F.linear(_CopyToTP.apply(x,self.group),self.base_weight,self.base_bias)
            small=F.linear(x.to(self.lora_A.dtype),self.lora_A)
            delta=F.linear(_CopyToTP.apply(small,self.group),self.lora_B)*self.scale
        else:
            base=_ReduceFromTP.apply(F.linear(x,self.base_weight),self.group)
            if self.base_bias is not None:base=base+self.base_bias
            small=_ReduceFromTP.apply(F.linear(x.to(self.lora_A.dtype),self.lora_A),self.group)
            delta=F.linear(small,self.lora_B)*self.scale
        # PEFT adds the FP32 update before restoring the base result dtype.
        # Rounding the update to BF16 first would introduce a second rounding.
        return (base+delta).to(base.dtype)

    def parameter_layout(self):
        return dict(base_weight='output_shard' if self.style=='column' else 'input_shard',
                    lora_A='replicated' if self.style=='column' else 'input_shard',
                    lora_B='output_shard' if self.style=='column' else 'replicated',
                    frozen_base=True,merged_execution=False)


class _SumBothWaysTP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, group):
        ctx.group = group
        output = value.contiguous().clone()
        dist.all_reduce(output, op=dist.ReduceOp.SUM, group=group)
        return output

    @staticmethod
    def backward(ctx, gradient):
        output = gradient.contiguous().clone()
        dist.all_reduce(output, op=dist.ReduceOp.SUM, group=ctx.group)
        return output, None


class StaticTPRMSNorm(nn.Module):
    """Across-head RMSNorm: do not normalize each TP head shard separately.

    Each rank contributes its local sum of squares to the original global
    hidden-width denominator. Backward also sums the scalar-statistic gradient.
    This is a placement requirement, not a DMD kernel optimization.
    """
    def __init__(self, weight, *, group, eps, cpu_reference_dtype=False):
        super().__init__()
        if not dist.is_initialized() or dist.get_world_size(group) != 4:
            raise ValueError('requires initialized TP4')
        if weight.ndim != 1 or weight.numel() % 4:
            raise ValueError('requires a divisible one-dimensional RMS weight')
        if cpu_reference_dtype and (weight.device.type != 'cpu' or weight.dtype != torch.float64):
            raise ValueError('FP64 reference is CPU-only')
        self.group = group
        self.eps = float(eps)
        self.global_width = weight.numel()
        self.reference = cpu_reference_dtype
        self.weight = nn.Parameter(weight.chunk(4)[dist.get_rank(group)].clone(), requires_grad=False)

    def forward(self, x):
        if x.shape[-1] != self.weight.numel():
            raise ValueError('RMS input is not the declared local head shard')
        dtype = x.dtype
        values = x if self.reference else x.float()
        sum_squares = _SumBothWaysTP.apply(values.square().sum(-1, keepdim=True), self.group)
        normalized = values * torch.rsqrt(sum_squares / self.global_width + self.eps)
        return (normalized * self.weight.to(normalized.dtype)).to(dtype)


class _PlacedLinear(nn.Module):
    """A PEFT child projection, preserving the original named adapter shell."""
    def __init__(self, original, *, group, placement, exchange, frozen):
        super().__init__()
        if not dist.is_initialized() or dist.get_world_size(group) != 4:
            raise ValueError('requires initialized TP4')
        rank = dist.get_rank(group)
        weight = original.weight.detach()
        if placement == 'output':
            if weight.shape[0] % 4:
                raise ValueError('output width does not divide by TP4')
            weight = weight.chunk(4, dim=0)[rank]
        elif placement == 'input':
            if weight.shape[1] % 4:
                raise ValueError('input width does not divide by TP4')
            weight = weight.chunk(4, dim=1)[rank]
        elif placement != 'replicated':
            raise ValueError('unknown TP placement')
        self.weight = nn.Parameter(weight.clone(), requires_grad=not frozen)
        bias = getattr(original, 'bias', None)
        if bias is None:
            self.register_parameter('bias', None)
        else:
            if placement == 'input' and not frozen:
                raise ValueError('row LoRA projection must be bias-free')
            value = bias.detach().chunk(4)[rank] if placement == 'output' else bias.detach()
            self.bias = nn.Parameter(value.clone(), requires_grad=not frozen)
        self.group, self.exchange = group, exchange
        self.static_tp_placement = placement
        self.static_tp_partition_dim = {'output': 0, 'input': 1}.get(placement)
        self.in_features, self.out_features = self.weight.shape[1], self.weight.shape[0]

    def forward(self, x):
        if self.exchange == 'copy_input_backward_sum':
            x = _CopyToTP.apply(x, self.group)
        # A row base bias is replicated, and must be added AFTER the SUM.
        bias = None if self.exchange == 'row_output_sum' else self.bias
        result = F.linear(x, self.weight, bias)
        if self.exchange in ('row_output_sum', 'lora_a_output_sum'):
            result = _ReduceFromTP.apply(result, self.group)
        if self.exchange == 'row_output_sum' and self.bias is not None:
            result = result + self.bias
        return result


def bind_peft_linear_tp(module, *, group, style, cpu_reference_dtype=False):
    """Partition children without replacing PEFT enable/disable/adapter logic.

    Retaining the PEFT shell is essential for Teacher adapter-disable and
    checkpoint replay. The native PEFT forward still controls update casting.
    This does not install a transformer/PP executor, or change defaults.
    """
    from peft.tuners.lora.layer import Linear
    if not isinstance(module, Linear) or style not in ('column', 'row'):
        raise ValueError('requires an ordinary PEFT Linear and explicit placement')
    if module.merged or getattr(module, '_static_tp_bound', False):
        raise ValueError('merged or already-partitioned adapter is unsupported')
    if any(getattr(module, 'use_dora', {}).values()):
        raise ValueError('static reference only accepts ordinary LoRA')
    if any(getattr(dropout, 'p', 0) != 0 for dropout in module.lora_dropout.values()):
        raise ValueError('LoRA dropout requires a replicated RNG binding first')
    if any(parameter.requires_grad for parameter in module.base_layer.parameters()):
        raise ValueError('frozen backbone must not acquire a trainable gradient')
    base = module.base_layer
    if cpu_reference_dtype:
        if base.weight.device.type != 'cpu' or base.weight.dtype != torch.float64:
            raise ValueError('FP64 reference override is CPU-only')
    elif base.weight.dtype != torch.bfloat16:
        raise ValueError('frozen base must remain BF16')
    for name in module.lora_A:
        if module.lora_A[name].weight.dtype != torch.float32 or module.lora_B[name].weight.dtype != torch.float32:
            raise ValueError('LoRA compute must remain FP32')
    module.base_layer = _PlacedLinear(base, group=group,
        placement='output' if style == 'column' else 'input',
        exchange='copy_input_backward_sum' if style == 'column' else 'row_output_sum', frozen=True)
    for name in module.lora_A:
        module.lora_A[name] = _PlacedLinear(module.lora_A[name], group=group,
            placement='replicated' if style == 'column' else 'input',
            exchange='none' if style == 'column' else 'lora_a_output_sum', frozen=False)
        module.lora_B[name] = _PlacedLinear(module.lora_B[name], group=group,
            placement='output' if style == 'column' else 'replicated',
            exchange='copy_input_backward_sum' if style == 'column' else 'none', frozen=False)
    module._static_tp_bound = True
    module._static_tp_layout = {'style': style, 'teacher_adapter_disable': 'ORIGINAL_PEFT_SHELL',
                               'merged': False, 'base_compute': 'BF16', 'lora_compute': 'FP32'}
    return module

