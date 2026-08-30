from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import torch

from distillschedule.runtime.helios_history_current_fusion import (
    _rewrite_forward_source,
    install_exact_history_current_fusion,
)


class _Block:
    def forward(
        self,
        hidden_states,
        attn_output,
        history_seq_len,
        enable_navit=False,
    ):
        if enable_navit:
            hidden_states = hidden_states + attn_output
        else:
            history_hidden_states, hidden_states = (
                hidden_states[:, :history_seq_len],
                hidden_states[:, history_seq_len:],
            )
            hidden_states = hidden_states + attn_output
            hidden_states = torch.cat([history_hidden_states, hidden_states], dim=1)
        return hidden_states


def _module() -> SimpleNamespace:
    return SimpleNamespace(
        __file__=__file__,
        HeliosTransformerBlock=type("FreshBlock", (_Block,), {}),
    )


def _fused(hidden_states, attention_output, history_tokens):
    return torch.cat(
        [
            hidden_states[:, :history_tokens],
            hidden_states[:, history_tokens:] + attention_output,
        ],
        dim=1,
    )


def test_rewrite_is_exact_and_unique() -> None:
    rewritten = _rewrite_forward_source(
        """def forward(self, hidden_states, attn_output, history_seq_len):
    if True:
        pass
    else:
        history_hidden_states, hidden_states = (
            hidden_states[:, :history_seq_len],
            hidden_states[:, history_seq_len:],
        )
        hidden_states = hidden_states + attn_output
        hidden_states = torch.cat([history_hidden_states, hidden_states], dim=1)
    return hidden_states
"""
    )
    assert "distillschedule_full_hidden_states = hidden_states" in rewritten
    assert "_distillschedule_exact_history_current_add(" in rewritten
    assert "torch.cat([history_hidden_states, hidden_states]" not in rewritten


def test_default_off_does_not_patch() -> None:
    module = _module()
    original = module.HeliosTransformerBlock.forward
    receipt = install_exact_history_current_fusion(
        enabled=False,
        role="generator",
        transformer_module=module,
        fused_function=_fused,
    )
    assert receipt["enabled"] is False
    assert receipt["disabled_reason"] == "NOT_REQUESTED"
    assert module.HeliosTransformerBlock.forward is original


def test_installed_forward_preserves_output_and_gradients() -> None:
    torch.manual_seed(91)
    reference_module = _module()
    candidate_module = _module()
    receipt = install_exact_history_current_fusion(
        enabled=True,
        role="generator",
        transformer_module=candidate_module,
        fused_function=_fused,
    )
    hidden_reference = torch.randn(3, 9, 8, requires_grad=True)
    hidden_candidate = hidden_reference.detach().clone().requires_grad_(True)
    attention_reference = torch.randn(3, 5, 8, requires_grad=True)
    attention_candidate = attention_reference.detach().clone().requires_grad_(True)
    upstream = torch.randn(3, 9, 8)
    expected = reference_module.HeliosTransformerBlock().forward(
        hidden_reference, attention_reference, 4
    )
    actual = candidate_module.HeliosTransformerBlock().forward(
        hidden_candidate, attention_candidate, 4
    )
    expected.backward(upstream)
    actual.backward(upstream)
    assert torch.equal(expected, actual)
    assert torch.equal(hidden_reference.grad, hidden_candidate.grad)
    assert torch.equal(attention_reference.grad, attention_candidate.grad)
    assert receipt["enabled"] is True
    assert receipt["call_count"] == 1
    assert receipt["batch_counts"] == {"3": 1}
    assert receipt["reference_fallback_count"] == 0
    assert candidate_module.HeliosTransformerBlock().forward(
        hidden_candidate.detach(), torch.zeros_like(hidden_candidate), 4, True
    ).shape == (3, 9, 8)
    assert receipt["call_count"] == 1


def test_repository_plumbing_mentions_default_off_flag() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = (
        root / "src/distillschedule/runtime/fsdp2_p0.py",
        root / "scripts/distillschedule/probe_static_fsdp2_p0_world16.py",
        root / "scripts/distillschedule/run_static_fsdp2_p0_world16.py",
        root / "scripts/distillschedule/launch_static_fsdp2_p0_world16_node.sh",
        root / "scripts/distillschedule/summarize_static_fsdp2_p0_world16.py",
    )
    for path in paths:
        assert "exact-history-current-fusion" in path.read_text().replace("_", "-")
