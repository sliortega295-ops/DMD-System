"""Real CPU autograd/checkpoint tests; not a native-model or GPU result."""
import copy
import hashlib
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from distillschedule.runtime.static_native_forward import create_admitted_forward_split
from distillschedule.runtime.static_native_gan_pp_stage import NativeSourceGANPPStage
from distillschedule.runtime.static_native_pp_stage import NativeSourcePPStage


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, hidden_states, encoder_hidden_states, timestep_proj, temb, rotary_emb):
        return (hidden_states + self.linear(hidden_states).tanh()
                + .01 * encoder_hidden_states + .02 * timestep_proj
                + .03 * temb + .04 * rotary_emb)


class _NativeFixture(nn.Module):
    """Same prefix / two checkpoint branches / tap / suffix protocol."""
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(_Block() for _ in range(4))
        self.input_projection = nn.Linear(4, 4)
        self.out = nn.Linear(4, 4)
        self.gan_hooks = (0, 2)
        self.gan_heads = nn.ModuleDict({"0": nn.Linear(4, 1), "2": nn.Linear(4, 1)})
        self.gan_final_head = nn.Linear(4, 1)
        self.gradient_checkpointing = True
        self.is_use_gan = True
        self.is_use_gan_hooks = True
        self.is_use_gan_final = True

    def _gradient_checkpointing_func(self, fn, *args):
        return checkpoint(fn, *args, use_reentrant=False)

    def forward(self, hidden_states, encoder_hidden_states, timestep_proj, temb,
                rotary_emb, gan_mode=False, return_dict=False):
        hidden_states = self.input_projection(hidden_states)
        logits_hidden = []
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for iidx, block in enumerate(self.blocks):
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, temb, rotary_emb)
                if gan_mode and iidx in self.gan_hooks:
                    logits_hidden.append(hidden_states[:, -3:, :])
        else:
            for iidx, block in enumerate(self.blocks):
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, temb, rotary_emb)
                if gan_mode and iidx in self.gan_hooks:
                    logits_hidden.append(hidden_states[:, -3:, :])
        output = self.out(hidden_states)
        logits = []
        if gan_mode:
            logits.append(self.gan_final_head(output.mean(dim=1)))
            for idx, (_, head) in enumerate(self.gan_heads.items()):
                logits.append(head(logits_hidden[idx].mean(dim=1)))
            logits = torch.cat(logits, dim=1)
        if return_dict:
            return {"sample": output, "logits": logits}
        return output, logits


def _stages(model, inputs):
    digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    split = create_admitted_forward_split(model, expected_source_sha256=digest, expected_blocks=4)
    template = split.prefix(model, *inputs, gan_mode=False, return_dict=False)
    left, right = copy.deepcopy(model), copy.deepcopy(model)
    a = NativeSourcePPStage(left, split=split, stage_index=0, context_template=template,
                           remove_root_modules=("out", "gan_heads", "gan_final_head"))
    b = NativeSourcePPStage(right, split=split, stage_index=1, context_template=template,
                           remove_root_modules=("input_projection",))
    return a, b, NativeSourceGANPPStage(a), NativeSourceGANPPStage(b)


@pytest.mark.parametrize("grad_enabled", [False, True])
def test_exact_native_sample_logits_and_gradients(grad_enabled):
    torch.manual_seed(5)
    model = _NativeFixture()
    inputs = [torch.randn(2, 3, 4) for _ in range(5)]
    a, b, ga, gb = _stages(model, inputs)
    with torch.set_grad_enabled(grad_enabled):
        x_ref = inputs[0].clone().requires_grad_(grad_enabled)
        x_pp = inputs[0].clone().requires_grad_(grad_enabled)
        reference = model(x_ref, *inputs[1:], gan_mode=True)
        boundary = ga(x_pp, *inputs[1:], gan_mode=True)
        assert len(boundary) == 6  # five original fields, one stage0 tap
        candidate = gb(*boundary)
        for x, y in zip(reference, candidate, strict=True):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
        if grad_enabled:
            sum(v.square().mean() for v in reference).backward()
            sum(v.square().mean() for v in candidate).backward()
            torch.testing.assert_close(x_ref.grad, x_pp.grad, rtol=1e-6, atol=1e-7)
            expected = dict(model.named_parameters())
            observed = dict(a.model.named_parameters()) | dict(b.model.named_parameters())
            assert observed.keys() == expected.keys()
            for name, value in expected.items():
                torch.testing.assert_close(value.grad, observed[name].grad, rtol=1e-6, atol=1e-7)
    assert {id(p) for p in ga.parameters()} == {id(p) for p in a.parameters()}
    assert {id(p) for p in gb.parameters()} == {id(p) for p in b.parameters()}
    assert ga.receipt()["target_GPU"] == "NOT_RUN_COMPONENT_ONLY"
    assert gb.receipt()["complete_GAN_worker"] == "NOT_BOUND"


def test_missing_tap_and_implicit_gan_mode_fail_closed():
    torch.manual_seed(5)
    model = _NativeFixture()
    inputs = [torch.randn(2, 3, 4) for _ in range(5)]
    _, _, ga, gb = _stages(model, inputs)
    with pytest.raises(ValueError, match="gan_mode=True"):
        ga(*inputs)
    boundary = ga(*inputs, gan_mode=True)
    with pytest.raises(ValueError, match="every ordered native field"):
        gb(*boundary[:-1])
