from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from distillschedule.runtime.wan13b_norm_policy import ENV, install, requested, validate


def test_default_off_requires_no_model_or_device(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert requested() is False
    assert install(role="generator", placement="resident", device=None)["enabled"] is False


def test_invalid_flag_and_placement_fail_closed(monkeypatch):
    monkeypatch.setenv(ENV, "true")
    with pytest.raises(ValueError):
        requested()
    monkeypatch.setenv(ENV, "1")
    with pytest.raises(ValueError):
        install(role="generator", placement="world16-fsdp", device=None)


def test_enabled_requires_actual_calls_and_zero_fallback():
    receipt = dict(requested=True, enabled=True, registered_width=1536,
                   fused_call_count=1, layernorm_fused_call_count=1,
                   backward_kernel_call_count=1, reference_fallback_count=0)
    build = {role: {"metadata": {"modulated_norm_fusion": dict(receipt)}} for role in ("generator", "critic")}
    assert validate(build, enabled=True)["enabled"] is True
    build["critic"]["metadata"]["modulated_norm_fusion"]["backward_kernel_call_count"] = 0
    with pytest.raises(AssertionError):
        validate(build, enabled=True)


def test_runtime_receipt_cannot_disguise_requested_or_backward_fallback():
    receipt = dict(requested=True, enabled=True, registered_width=1536,
                   fused_call_count=1, layernorm_fused_call_count=1,
                   backward_kernel_call_count=1, reference_fallback_count=0,
                   backward_kernel_reference_fallback_count=0)
    build = {role: {"metadata": {"modulated_norm_fusion": dict(receipt)}} for role in ("generator", "critic")}
    with pytest.raises(AssertionError):
        validate(build, enabled=False)
    build["generator"]["metadata"]["modulated_norm_fusion"]["backward_kernel_reference_fallback_count"] = 1
    with pytest.raises(AssertionError, match="backward fallback"):
        validate(build, enabled=True)


