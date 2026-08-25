from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from distillschedule.runtime.helios_cross_kv_cache import (
    generator_nograd_cross_kv_cache_scope,
    install_generator_nograd_cross_kv_cache,
)


class _Projection:
    def __init__(self, offset: float) -> None:
        self.offset = float(offset)
        self.calls = 0

    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return value + self.offset


class _Attention:
    def __init__(self, *, cross: bool) -> None:
        self.is_cross_attention = bool(cross)
        self.fused_projections = False
        self.to_q = _Projection(1.0)
        self.to_k = _Projection(2.0)
        self.to_v = _Projection(3.0)


def _fake_module(tmp_path: Path):
    source = tmp_path / "transformer_helios.py"
    source.write_text("# pinned fake Helios source\n")

    def get_qkv(attn, hidden_states, encoder_hidden_states):
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        return (
            attn.to_q(hidden_states),
            attn.to_k(encoder_hidden_states),
            attn.to_v(encoder_hidden_states),
        )

    module = SimpleNamespace(
        __file__=str(source),
        __name__="fake_transformer_helios",
        _get_qkv_projections=get_qkv,
    )
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    return module, digest


def _model():
    return SimpleNamespace(
        blocks=[
            SimpleNamespace(attn1=_Attention(cross=False), attn2=_Attention(cross=True))
            for _ in range(40)
        ]
    )


def _execute_calls(module, model, count: int, *, encoder_width: int = 4):
    outputs = []
    for call in range(count):
        hidden = torch.full((1, 2, encoder_width), float(call))
        # Deliberately allocate a fresh tensor: Helios' text projection may return
        # a new allocation on each DiT call even though its exact value is stable.
        encoder = torch.full((1, 3, encoder_width), 7.0).clone()
        per_call = []
        for block in model.blocks:
            module._get_qkv_projections(block.attn1, hidden, None)
            per_call.append(
                module._get_qkv_projections(block.attn2, hidden, encoder)
            )
        outputs.append(per_call)
    return outputs


def test_default_off_does_not_patch_projection_helper(tmp_path: Path) -> None:
    module, digest = _fake_module(tmp_path)
    original = module._get_qkv_projections
    receipt = install_generator_nograd_cross_kv_cache(
        enabled=False,
        role="generator",
        transformer_module=module,
        expected_source_sha256=digest,
    )
    assert receipt["enabled"] is False
    assert receipt["disabled_reason"] == "NOT_REQUESTED"
    assert module._get_qkv_projections is original


def test_one_rollout_fills_reuses_and_clears_all_40_cross_attentions(
    tmp_path: Path,
) -> None:
    module, digest = _fake_module(tmp_path)
    install = install_generator_nograd_cross_kv_cache(
        enabled=True,
        role="generator",
        transformer_module=module,
        expected_source_sha256=digest,
    )
    model = _model()
    with torch.no_grad():
        with generator_nograd_cross_kv_cache_scope(
            model, enabled=True, expected_model_calls=3
        ) as receipt:
            outputs = _execute_calls(module, model, 3)

    assert receipt["status"] == "PASS"
    assert receipt["cross_projection_call_count"] == 120
    assert receipt["self_projection_call_count"] == 120
    assert receipt["query_projection_count"] == 120
    assert receipt["cache_fill_count"] == 40
    assert receipt["cache_reuse_count"] == 80
    assert receipt["cache_miss_count"] == 0
    assert receipt["cached_attention_count_before_clear"] == 40
    assert receipt["cached_attention_count_after_clear"] == 0
    assert install["rollout_scope_count"] == 1
    assert install["cache_fill_count"] == 40
    assert install["cache_reuse_count"] == 80
    assert sum(block.attn2.to_q.calls for block in model.blocks) == 120
    assert sum(block.attn2.to_k.calls for block in model.blocks) == 40
    assert sum(block.attn2.to_v.calls for block in model.blocks) == 40
    for call in range(1, 3):
        for block in range(40):
            torch.testing.assert_close(outputs[call][block][1], outputs[0][block][1])
            torch.testing.assert_close(outputs[call][block][2], outputs[0][block][2])


def test_encoder_signature_change_fails_closed_and_cleans(tmp_path: Path) -> None:
    module, digest = _fake_module(tmp_path)
    install = install_generator_nograd_cross_kv_cache(
        enabled=True,
        role="generator",
        transformer_module=module,
        expected_source_sha256=digest,
    )
    model = _model()
    with pytest.raises(RuntimeError, match="encoder signature changed"):
        with torch.no_grad():
            with generator_nograd_cross_kv_cache_scope(
                model, enabled=True, expected_model_calls=2
            ):
                _execute_calls(module, model, 1, encoder_width=4)
                hidden = torch.zeros((1, 2, 5))
                encoder = torch.zeros((1, 3, 5))
                module._get_qkv_projections(model.blocks[0].attn2, hidden, encoder)
    assert install["encoder_signature_mismatch_count"] == 1


def test_missing_expected_projection_calls_fails_closed(tmp_path: Path) -> None:
    module, digest = _fake_module(tmp_path)
    install_generator_nograd_cross_kv_cache(
        enabled=True,
        role="generator",
        transformer_module=module,
        expected_source_sha256=digest,
    )
    model = _model()
    with pytest.raises(RuntimeError, match="lifecycle mismatch"):
        with torch.no_grad():
            with generator_nograd_cross_kv_cache_scope(
                model, enabled=True, expected_model_calls=2
            ):
                _execute_calls(module, model, 1)


def test_grad_enabled_scope_fails_closed(tmp_path: Path) -> None:
    module, digest = _fake_module(tmp_path)
    install_generator_nograd_cross_kv_cache(
        enabled=True,
        role="generator",
        transformer_module=module,
        expected_source_sha256=digest,
    )
    with pytest.raises(RuntimeError, match="requires no-grad"):
        with generator_nograd_cross_kv_cache_scope(
            _model(), enabled=True, expected_model_calls=2
        ):
            pass


def test_source_identity_mismatch_fails_closed(tmp_path: Path) -> None:
    module, _digest = _fake_module(tmp_path)
    with pytest.raises(RuntimeError, match="source identity changed"):
        install_generator_nograd_cross_kv_cache(
            enabled=True,
            role="generator",
            transformer_module=module,
            expected_source_sha256="0" * 64,
        )


def test_launcher_controller_worker_and_summarizer_are_wired() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = (
        root / "scripts/distillschedule/probe_static_fsdp2_p0_world16.py",
        root / "scripts/distillschedule/run_static_fsdp2_p0_world16.py",
        root / "scripts/distillschedule/launch_static_fsdp2_p0_world16_node.sh",
        root / "scripts/distillschedule/summarize_static_fsdp2_p0_world16.py",
    )
    for path in paths:
        assert "generator_nograd_cross_kv_cache" in path.read_text()
    launcher = paths[2].read_text()
    assert "DISTILLSCHEDULE_HELIOS_GENERATOR_NOGRAD_CROSS_KV_CACHE" in launcher
    summarizer = paths[3].read_text()
    assert "rank_generator_nograd_cross_kv_cache" in summarizer
    assert "cache_reuse_count" in summarizer
