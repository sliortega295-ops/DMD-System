from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from distillschedule.runtime.helios_history_kv_cache import (
    generator_nograd_history_kv_cache_scope,
    install_generator_nograd_history_kv_cache,
)


def _fake_module(
    tmp_path: Path, *, populate: bool = True, mutate_on_reuse: bool = False
):
    source = tmp_path / "transformer_helios.py"
    source.write_text("# pinned fake Helios source\n")

    class Processor:
        def __init__(self):
            self.cache_enabled = False
            self.kv_cache = None

        def enable_cache(self):
            self.cache_enabled = True
            self.kv_cache = None

        def disable_cache(self):
            self.cache_enabled = False
            self.kv_cache = None

        def __call__(
            self,
            attn,
            hidden_states=None,
            encoder_hidden_states=None,
            attention_mask=None,
            rotary_emb=None,
            original_context_length=None,
            original_context_length_list=None,
            enable_navit=False,
            is_first_denoising_step=False,
        ):
            del (
                encoder_hidden_states,
                attention_mask,
                rotary_emb,
                original_context_length,
                original_context_length_list,
            )
            assert attn.restrict_self_attn
            assert not enable_navit
            if self.cache_enabled and is_first_denoising_step and populate:
                self.kv_cache = {
                    "key_history": torch.ones(2, dtype=torch.bfloat16),
                    "value_history": torch.full(
                        (2,), 2, dtype=torch.bfloat16
                    ),
                    "history_hidden_states": torch.full(
                        (2,), 3, dtype=torch.bfloat16
                    ),
                }
            elif self.cache_enabled and mutate_on_reuse and self.kv_cache is not None:
                self.kv_cache["key_history"].add_(1)
            return hidden_states

    module = SimpleNamespace(__file__=str(source), HeliosAttnProcessor=Processor)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    return module, digest


def _model(processor_class):
    blocks = []
    for _ in range(40):
        attention = SimpleNamespace(
            processor=processor_class(), restrict_self_attn=True
        )
        blocks.append(SimpleNamespace(attn1=attention))
    return SimpleNamespace(blocks=blocks)


def _execute_calls(model, count: int) -> None:
    for call in range(count):
        for block in model.blocks:
            block.attn1.processor(
                block.attn1,
                hidden_states=torch.tensor([call], dtype=torch.bfloat16),
                enable_navit=False,
                is_first_denoising_step=call == 0,
            )


def test_default_off_does_not_patch_processor(tmp_path: Path) -> None:
    module, digest = _fake_module(tmp_path)
    original = module.HeliosAttnProcessor.__call__
    receipt = install_generator_nograd_history_kv_cache(
        enabled=False,
        role="generator",
        transformer_module=module,
        expected_source_sha256=digest,
    )
    assert receipt["enabled"] is False
    assert receipt["disabled_reason"] == "NOT_REQUESTED"
    assert module.HeliosAttnProcessor.__call__ is original


def test_one_rollout_fills_reuses_and_clears_all_40_processors(
    tmp_path: Path,
) -> None:
    module, digest = _fake_module(tmp_path)
    install = install_generator_nograd_history_kv_cache(
        enabled=True,
        role="generator",
        transformer_module=module,
        expected_source_sha256=digest,
    )
    model = _model(module.HeliosAttnProcessor)
    with torch.no_grad():
        with generator_nograd_history_kv_cache_scope(
            model, enabled=True, expected_model_calls=3
        ) as receipt:
            _execute_calls(model, 3)

    assert receipt["status"] == "PASS"
    assert receipt["enabled_processor_calls"] == 120
    assert receipt["first_step_calls"] == 40
    assert receipt["cache_fill_count"] == 40
    assert receipt["cache_reuse_count"] == 80
    assert receipt["cache_miss_count"] == 0
    assert receipt["grad_enabled_call_count"] == 0
    assert receipt["cached_tensor_mutation_count"] == 0
    assert receipt["cache_resident_bytes_before_clear"] > 0
    assert receipt["populated_before_clear"] == 40
    assert receipt["cache_enabled_after_scope"] == 0
    assert receipt["populated_after_scope"] == 0
    assert install["rollout_scope_count"] == 1
    assert install["cache_fill_count"] == 40
    assert install["cache_reuse_count"] == 80


def test_missing_first_step_population_fails_closed_and_cleans(
    tmp_path: Path,
) -> None:
    module, digest = _fake_module(tmp_path, populate=False)
    install_generator_nograd_history_kv_cache(
        enabled=True,
        role="generator",
        transformer_module=module,
        expected_source_sha256=digest,
    )
    model = _model(module.HeliosAttnProcessor)
    with pytest.raises(RuntimeError, match="did not populate"):
        with torch.no_grad():
            with generator_nograd_history_kv_cache_scope(
                model, enabled=True, expected_model_calls=2
            ):
                _execute_calls(model, 2)
    assert all(not block.attn1.processor.cache_enabled for block in model.blocks)
    assert all(block.attn1.processor.kv_cache is None for block in model.blocks)


def test_source_identity_mismatch_fails_closed(tmp_path: Path) -> None:
    module, _digest = _fake_module(tmp_path)
    with pytest.raises(RuntimeError, match="source identity changed"):
        install_generator_nograd_history_kv_cache(
            enabled=True,
            role="generator",
            transformer_module=module,
            expected_source_sha256="0" * 64,
        )


def test_cached_tensor_mutation_fails_closed_and_cleans(tmp_path: Path) -> None:
    module, digest = _fake_module(tmp_path, mutate_on_reuse=True)
    install_generator_nograd_history_kv_cache(
        enabled=True,
        role="generator",
        transformer_module=module,
        expected_source_sha256=digest,
    )
    model = _model(module.HeliosAttnProcessor)
    with pytest.raises(RuntimeError, match="was mutated"):
        with torch.no_grad():
            with generator_nograd_history_kv_cache_scope(
                model, enabled=True, expected_model_calls=2
            ):
                _execute_calls(model, 2)
    assert all(not block.attn1.processor.cache_enabled for block in model.blocks)
    assert all(block.attn1.processor.kv_cache is None for block in model.blocks)


def test_launcher_controller_worker_and_summarizer_are_wired() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = (
        root / "scripts/distillschedule/probe_static_fsdp2_p0_world16.py",
        root / "scripts/distillschedule/run_static_fsdp2_p0_world16.py",
        root / "scripts/distillschedule/launch_static_fsdp2_p0_world16_node.sh",
        root / "scripts/distillschedule/summarize_static_fsdp2_p0_world16.py",
    )
    for path in paths:
        assert "generator_nograd_history_kv_cache" in path.read_text()
    launcher = paths[2].read_text()
    assert "DISTILLSCHEDULE_HELIOS_GENERATOR_NOGRAD_HISTORY_KV_CACHE" in launcher
    summarizer = paths[3].read_text()
    assert "rank_generator_nograd_history_kv_cache" in summarizer
    assert "cache_reuse_count" in summarizer
