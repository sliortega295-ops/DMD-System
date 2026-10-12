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


def test_controller_defaults_off_and_rejects_fsdp():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts/distillschedule/run_wan13b_fair_world16.py"
    spec = importlib.util.spec_from_file_location("wan_norm_controller_test", path)
    assert spec and spec.loader
    import sys
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    args = module.parser().parse_args([
        "--run-id", "cpu", "--mode", "validate", "--placement", "world16-fsdp",
        "--expected-head", "a" * 40, "--model-root", "/model", "--bootstrap", "/bootstrap",
        "--bootstrap-hash", "b" * 64,
    ])
    assert args.modulated_norm_fusion is False
    args.modulated_norm_fusion = True
    with pytest.raises(ValueError, match="norm fusion"):
        module._command(args, 0, "127.0.0.1")
