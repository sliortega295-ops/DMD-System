from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from distillschedule.runtime import helios_generator_free_memory as target


def _fake_distribution_matching_loss() -> None:
    return None


class _FakeCachedGrad:
    pass


_executions = 0


def _fake_free_memory() -> None:
    global _executions
    _executions += 1


def _module(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    module = SimpleNamespace(
        compute_distribution_matching_loss=_fake_distribution_matching_loss,
        Gan_D_Loss_With_Cached_Grad=_FakeCachedGrad,
        free_memory=_fake_free_memory,
    )
    monkeypatch.setattr(
        target,
        "HELIOS_DISTRIBUTION_MATCHING_SHA256",
        target.source_sha256(_fake_distribution_matching_loss),
    )
    monkeypatch.setattr(
        target,
        "HELIOS_GAN_CACHED_GRAD_SHA256",
        target.source_sha256(_FakeCachedGrad),
    )
    monkeypatch.setattr(
        target,
        "DIFFUSERS_FREE_MEMORY_SHA256",
        target.source_sha256(_fake_free_memory),
    )
    return module


def test_disabled_receipt_does_not_import_or_patch() -> None:
    receipt = target.prepare_generator_free_memory_receipt(enabled=False)
    assert receipt["enabled"] is False
    assert receipt["disabled_reason"] == "NOT_REQUESTED"


def test_correctness_executes_four_calls_and_restores(monkeypatch) -> None:
    global _executions
    _executions = 0
    module = _module(monkeypatch)
    receipt = target.prepare_generator_free_memory_receipt(
        enabled=True, module=module
    )
    original = module.free_memory
    with target.generator_free_memory_scope(receipt, elide=False, module=module):
        for _ in range(4):
            module.free_memory()
    assert module.free_memory is original
    assert _executions == 4
    assert receipt["scope_entry_count"] == 1
    assert receipt["call_count"] == 4
    assert receipt["executed_count"] == 4
    assert receipt["elided_count"] == 0
    assert receipt["restored_count"] == 1


def test_timing_elides_exactly_four_calls_and_restores(monkeypatch) -> None:
    global _executions
    _executions = 0
    module = _module(monkeypatch)
    receipt = target.prepare_generator_free_memory_receipt(
        enabled=True, module=module
    )
    original = module.free_memory
    with target.generator_free_memory_scope(receipt, elide=True, module=module):
        for _ in range(4):
            module.free_memory()
    assert module.free_memory is original
    assert _executions == 0
    assert receipt["call_count"] == 4
    assert receipt["executed_count"] == 0
    assert receipt["elided_count"] == 4
    assert receipt["restored_count"] == 1


def test_unexpected_call_count_fails_closed_after_restore(monkeypatch) -> None:
    module = _module(monkeypatch)
    receipt = target.prepare_generator_free_memory_receipt(
        enabled=True, module=module
    )
    original = module.free_memory
    with pytest.raises(RuntimeError, match="call count changed"):
        with target.generator_free_memory_scope(
            receipt, elide=True, module=module
        ):
            module.free_memory()
    assert module.free_memory is original
    assert receipt["restored_count"] == 1


def test_body_error_restores_without_masking_original_error(monkeypatch) -> None:
    module = _module(monkeypatch)
    receipt = target.prepare_generator_free_memory_receipt(
        enabled=True, module=module
    )
    original = module.free_memory
    with pytest.raises(ValueError, match="body failed"):
        with target.generator_free_memory_scope(
            receipt, elide=True, module=module
        ):
            raise ValueError("body failed")
    assert module.free_memory is original
    assert receipt["restored_count"] == 1


def test_generator_free_memory_policy_is_propagated_end_to_end() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = (
        "scripts/distillschedule/run_static_fsdp2_p0_world16.py",
        "scripts/distillschedule/launch_static_fsdp2_p0_world16_node.sh",
        "scripts/distillschedule/probe_static_fsdp2_p0_world16.py",
        "scripts/distillschedule/summarize_static_fsdp2_p0_world16.py",
    )
    for path in paths:
        assert "elide-official-generator-free-memory" in (root / path).read_text(), path
    baseline = (
        root / "scripts/distillschedule/probe_static_fsdp2_full5_world16.py"
    ).read_text()
    assert "generator_free_memory_scope" in baseline
    assert "elide_official_generator_free_memory" in baseline
