"""Actual torch non-reentrant replay proof for source-preserving PP extraction."""
import copy
import hashlib
import inspect
from pathlib import Path

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from distillschedule.runtime.static_native_forward import create_admitted_forward_split


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Parameter(torch.randn(5, 5, dtype=torch.float64), requires_grad=False)
        self.lora_A = nn.Parameter(torch.randn(2, 5, dtype=torch.float64) * .02)
        self.lora_B = nn.Parameter(torch.randn(5, 2, dtype=torch.float64) * .02)

    def forward(self, hidden_states, condition):
        return torch.tanh(torch.nn.functional.linear(hidden_states, self.base)
            + torch.nn.functional.linear(torch.nn.functional.linear(hidden_states, self.lora_A), self.lora_B)
            + condition)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.condition = nn.Linear(5, 5, dtype=torch.float64)
        self.blocks = nn.ModuleList([_Block() for _ in range(4)])
        self.output = nn.Linear(5, 3, dtype=torch.float64)
        self.gradient_checkpointing = True
        self.original_checkpoint_calls = 0

    def _gradient_checkpointing_func(self, function, *args):
        self.original_checkpoint_calls += 1
        return checkpoint(function, *args, use_reentrant=False)

    def forward(self, hidden_states, encoder_hidden_states, *, return_dict=True):
        encoder_hidden_states = self.condition(encoder_hidden_states)
        hidden_states = hidden_states + encoder_hidden_states
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for iidx, block in enumerate(self.blocks):
                hidden_states = self._gradient_checkpointing_func(block, hidden_states, encoder_hidden_states)
        else:
            for iidx, block in enumerate(self.blocks):
                hidden_states = block(hidden_states, encoder_hidden_states)
        output = self.output(hidden_states)
        if not return_dict:
            return (output,)
        return {"sample": output}


def test_actual_checkpoint_backward_prefix_block_postfix_gradients_bitwise():
    torch.manual_seed(91)
    reference = _Model()
    candidate = copy.deepcopy(reference)
    x = torch.randn(2, 5, dtype=torch.float64, requires_grad=True)
    condition = torch.randn(2, 5, dtype=torch.float64, requires_grad=True)
    xc = x.detach().clone().requires_grad_(True)
    cc = condition.detach().clone().requires_grad_(True)
    expected = reference(x, condition)["sample"]
    split = create_admitted_forward_split(candidate,
        expected_source_sha256=hashlib.sha256(Path(inspect.getsourcefile(_Model)).read_bytes()).hexdigest(),
        expected_blocks=4)
    context = split.prefix(candidate, xc, cc)
    context = split.blocks(candidate, context, 0, 2)
    context = split.blocks(candidate, context, 2, 4)
    actual = split.suffix(candidate, context)["sample"]
    # An external score is evaluated after the Generator forward and before
    # its checkpoint replay, as required by the multi-role DMD boundary.
    score = expected.detach().square() * .3
    candidate_score = actual.detach().square() * .3
    (expected * score).sum().backward()
    (actual * candidate_score).sum().backward()
    assert torch.equal(expected, actual)
    assert torch.equal(x.grad, xc.grad)
    assert torch.equal(condition.grad, cc.grad)
    for (name, p), (other_name, q) in zip(reference.named_parameters(), candidate.named_parameters(), strict=True):
        assert name == other_name
        if p.requires_grad:
            assert torch.equal(p.grad, q.grad), name
        else:
            assert p.grad is q.grad is None
    assert reference.original_checkpoint_calls == candidate.original_checkpoint_calls == 4
    assert not torch.cuda.is_initialized()
