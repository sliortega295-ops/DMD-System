"""Minimal LoRA used by the DMD critic to share frozen teacher weights."""
from contextlib import contextmanager

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank=32, alpha=None):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.rank = rank
        self.scale = (alpha or rank) / rank
        self.a = nn.Parameter(torch.zeros(rank, base.in_features))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.normal_(self.a, std=1.0 / rank)
        self.enabled = True

    def forward(self, x):
        y = self.base(x)
        if not self.enabled:
            return y
        h = nn.functional.linear(x.to(self.a.dtype), self.a)
        return y + nn.functional.linear(h, self.b).to(y.dtype) * self.scale


TARGETS = ('self_attn.q', 'self_attn.k', 'self_attn.v', 'self_attn.o',
           'ffn.0', 'ffn.2')


def inject_lora(model, rank=32, alpha=None, targets=TARGETS):
    """Wrap target linear layers and return the trainable parameters."""
    for p in model.parameters():
        p.requires_grad_(False)
    n = 0
    for blk in model.blocks:
        for name in targets:
            parent, _, leaf = name.rpartition('.')
            mod = blk.get_submodule(parent) if parent else blk
            lin = getattr(mod, leaf) if not leaf.isdigit() else mod[int(leaf)]
            if isinstance(lin, LoRALinear):
                continue
            wrapped = LoRALinear(lin, rank, alpha).to(lin.weight.device)
            wrapped.a.data = wrapped.a.data.float()
            wrapped.b.data = wrapped.b.data.float()
            if leaf.isdigit():
                mod[int(leaf)] = wrapped
            else:
                setattr(mod, leaf, wrapped)
            n += 1
    params = [p for p in model.parameters() if p.requires_grad]
    total = sum(p.numel() for p in params)
    return params, n, total


def set_lora(model, on):
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.enabled = on


@contextmanager
def lora_enabled(model, on):
    prev = [m.enabled for m in model.modules() if isinstance(m, LoRALinear)]
    set_lora(model, on)
    try:
        yield
    finally:
        for m, p in zip((m for m in model.modules()
                         if isinstance(m, LoRALinear)), prev):
            m.enabled = p


def lora_state_dict(model):
    return {k: v for k, v in model.state_dict().items()
            if k.endswith('.a') or k.endswith('.b')}
