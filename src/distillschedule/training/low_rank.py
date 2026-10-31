"""Model-neutral, unmerged native Linear LoRA; no inference weight merging."""
import math

import torch
from torch import nn


def _require(value, message):
    if not value:
        raise ValueError('native trainable scope: ' + message)


class TrainableLowRankLinear(nn.Module):
    """Frozen W*x + (alpha/rank)*B(A*x), with separately owned trainable A/B."""
    def __init__(self, base, *, rank, alpha):
        super().__init__()
        _require(type(base) is nn.Linear and type(rank) is int and 0 < rank <= min(base.in_features, base.out_features)
                 and type(alpha) in (int, float) and math.isfinite(alpha) and alpha > 0,
                 'invalid native Linear LoRA scope/rank/alpha')
        self.base = base.requires_grad_(False)
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=base.weight.dtype))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=base.weight.dtype))
        nn.init.kaiming_uniform_(self.lora_a, a=5 ** .5)
        self.scale = float(alpha) / rank

    def forward(self, value):
        from torch.nn import functional as F
        return self.base(value) + self.scale * F.linear(F.linear(value, self.lora_a), self.lora_b)


def trainable_scope(native, config):
    _require(isinstance(config, dict) and config.get('mode') in ('full', 'lora'), 'explicit full or lora trainable scope required')
    if config['mode'] == 'full':
        _require(set(config) == {'mode'}, 'full mode accepts no ignored LoRA settings')
        native.requires_grad_(True)
        return
    _require(set(config) == {'mode', 'rank', 'alpha', 'targets'} and isinstance(config['targets'], list)
             and config['targets'] and all(isinstance(name, str) and name for name in config['targets'])
             and len(config['targets']) == len(set(config['targets'])), 'exact nonempty native Linear LoRA targets required')
    targets = {}
    for name in config['targets']:
        try:
            targets[name] = native.get_submodule(name)
        except AttributeError as exc:
            raise ValueError('missing actual native LoRA target: ' + name) from exc
        _require(type(targets[name]) is nn.Linear, 'LoRA target is not an actual native Linear: ' + name)
    # Ancestor/descendant targets cannot both be Linear, so replacement is unambiguous.
    native.requires_grad_(False)
    for name, module in targets.items():
        native.set_submodule(name, TrainableLowRankLinear(module, rank=config['rank'], alpha=config['alpha']))
