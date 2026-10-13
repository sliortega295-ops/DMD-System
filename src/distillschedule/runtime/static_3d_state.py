"""Explicit DP mean and TP/PP global clipping for the static baseline.

No automatic installation, new optimizer math, or native timing implied.
PP owns each parameter on one stage. TP-replicated gradients have already been
summed by the TP projection backward; they must count only once in grad norm.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
import torch.distributed as dist


@dataclass(frozen=True)
class StaticParameterPlacement:
    name: str
    parameter: torch.nn.Parameter
    tp_placement: str
    partition_dim: int | None

    def __post_init__(self):
        if self.tp_placement not in ('replicated', 'output', 'input'):
            raise ValueError('every trainable requires an explicit TP placement')
        if self.partition_dim != {'replicated': None, 'output': 0, 'input': 1}[self.tp_placement]:
            raise ValueError('TP partition dimension does not match its placement')
        if not self.parameter.requires_grad or self.parameter.dtype != torch.float32:
            raise ValueError('the unmerged baseline LoRA trainables must be FP32')


def placed_trainables(model):
    """Only declared PEFT child placements are accepted; no guessed fallback."""
    from .static_tp_lora import _PlacedLinear
    parameter_names = {id(p): n for n, p in model.named_parameters() if p.requires_grad}
    result = []
    for _, module in model.named_modules():
        if isinstance(module, _PlacedLinear):
            for _, p in module.named_parameters(recurse=False):
                if p.requires_grad:
                    result.append(StaticParameterPlacement(parameter_names[id(p)],p,
                        module.static_tp_placement,module.static_tp_partition_dim))
    if len({id(x.parameter) for x in result}) != len(parameter_names):
        raise ValueError('trainable outside the declared TP/PP placement: explicit binding required')
    return sorted(result,key=lambda x:x.name)


def shard_bootstrap_tensor(full, placement, tp_rank):
    """Restore the same full parameter/Adam tensor in its declared TP shard."""
    if full.dtype != torch.float32 or not 0 <= tp_rank < 4:
        raise ValueError('baseline bootstrap is FP32 with TP4 coordinates')
    if placement.partition_dim is None:
        local = full
    else:
        if full.shape[placement.partition_dim] % 4:
            raise ValueError('bootstrap dimension does not divide by TP4')
        local = full.chunk(4,dim=placement.partition_dim)[tp_rank]
    if local.shape != placement.parameter.shape:
        raise ValueError('bootstrap does not match the actual local parameter shard')
    return local


class StaticDataParallelState:
    """Eight independent B1 losses/DP replica, DP2 mean, one optimizer commit.

    model_group is the TP4 x PP2 group for ONE DP replica. It must not include
    both DP copies or count each replicated LoRA A/B four times.
    """
    def __init__(self, placements, *, dp_group, model_group, max_norm):
        if (not dist.is_initialized() or dist.get_world_size(dp_group)!=2
                or dist.get_world_size(model_group)!=8):
            raise ValueError('requires explicit DP2 and one TP4xPP2 model group')
        if not math.isfinite(max_norm) or max_norm <= 0 or not placements:
            raise ValueError('requires trainables and the original positive clip norm')
        self.placements=tuple(placements)
        if len({id(x.parameter) for x in self.placements}) != len(self.placements):
            raise ValueError('duplicate trainable placement')
        self.dp_group,self.model_group,self.max_norm=dp_group,model_group,float(max_norm)
        self.accumulations=0
        self.commits=0

    def mean_microbatch_loss(self,loss):
        if loss.ndim != 0 or self.accumulations >= 8:
            raise ValueError('eight scalar B1 losses, not eight optimizer updates')
        self.accumulations += 1
        return loss/8

    def note_gpipe_mean_backward(self):
        """GPipe scale_grads=True already divided by eight: do not do it twice."""
        if self.accumulations != 0:
            raise ValueError('cannot mix manual and GPipe gradient scaling')
        self.accumulations=8

    @torch.no_grad()
    def synchronize_and_clip(self):
        if self.accumulations != 8:
            raise ValueError('cannot commit before all eight independent B1 samples')
        parameters=[x.parameter for x in self.placements]
        if any(p.grad is None or p.grad.dtype != torch.float32 for p in parameters):
            raise ValueError('all declared FP32 LoRA gradients must be present')
        for parameter in parameters:
            dist.all_reduce(parameter.grad,op=dist.ReduceOp.SUM,group=self.dp_group)
            parameter.grad.div_(2)
        local=torch.zeros((),dtype=torch.float32,device=parameters[0].device)
        for item in self.placements:
            term=item.parameter.grad.square().sum()
            if item.tp_placement=='replicated':
                term=term/4
            local.add_(term)
        dist.all_reduce(local,op=dist.ReduceOp.SUM,group=self.model_group)
        norm=local.sqrt()
        # Same clip equation as clip_grad_norm_, with a model-global norm.
        coefficient=(self.max_norm/(norm+1e-6)).clamp(max=1)
        for parameter in parameters:
            parameter.grad.mul_(coefficient)
        self.accumulations=0
        self.commits+=1
        return norm

    def receipt(self):
        return dict(microbatch_size=1,microbatches_per_DP=8,DP=2,global_batch=16,
            reduction='DP_SUM_POST_DIVIDE2_AFTER_MICROBATCH_MEAN8',
            grad_norm='TP4_PP2_SUM_WITH_REPLICATED_TP_DEDUPLICATION',
            commits=self.commits,pending_microbatches=self.accumulations,
            timing='NOT_RUN_COMPONENT_ONLY')
