from __future__ import annotations

import pytest
import json
import os
from pathlib import Path
import subprocess
import sys

from distillschedule.runtime.wan13b_lora_grad_epilogue import (
    GRAD_EPILOGUE_ENV,
    grad_epilogue_requested,
    install_grad_lora_fused_epilogue,
    make_epilogue_autograd_function,
    runtime_precision_contract,
    validate_grad_epilogue_receipts,
)


def test_default_off_without_model_or_torch_import(monkeypatch):
    monkeypatch.delenv(GRAD_EPILOGUE_ENV, raising=False)
    class Model:
        pass
    model = Model()
    receipt = install_grad_lora_fused_epilogue(model, role="generator")
    assert receipt["requested"] is False
    assert receipt["enabled"] is False
    assert receipt["disabled_reason"] == "NOT_REQUESTED"


def test_switch_is_explicit_and_fail_closed(monkeypatch):
    for value, expected in (("0", False), ("1", True)):
        monkeypatch.setenv(GRAD_EPILOGUE_ENV, value)
        assert grad_epilogue_requested() is expected
    monkeypatch.setenv(GRAD_EPILOGUE_ENV, "typo")
    with pytest.raises(ValueError):
        grad_epilogue_requested()


@pytest.mark.parametrize("scale", [0.0, 0.0625, 0.3, 1.0, 2.0])
@pytest.mark.parametrize("base_grad", [False, True])
def test_real_cpu_autograd_matches_native_output_and_both_gradients(scale, base_grad):
    torch = pytest.importorskip("torch")
    torch.manual_seed(20260930)
    base = torch.randn(7, 13, dtype=torch.bfloat16, requires_grad=base_grad)
    update = torch.randn(7, 13, dtype=torch.float32, requires_grad=True)
    b = base.detach().clone().requires_grad_(base_grad)
    u = update.detach().clone().requires_grad_(True)
    gradient = torch.randn(7, 13, dtype=torch.bfloat16)
    native = (base + update * scale).to(base.dtype)
    function = make_epilogue_autograd_function(
        lambda lhs, rhs, s: (lhs + rhs * s).to(lhs.dtype)
    )
    candidate = function.apply(b, u, scale)
    native.backward(gradient)
    candidate.backward(gradient)
    torch.testing.assert_close(candidate, native, rtol=0, atol=0)
    torch.testing.assert_close(u.grad, update.grad, rtol=0, atol=0)
    if base_grad:
        torch.testing.assert_close(b.grad, base.grad, rtol=0, atol=0)


def test_mismatched_dtype_rejected():
    torch = pytest.importorskip("torch")
    function = make_epilogue_autograd_function(lambda a, b, s: a+b*s)
    with pytest.raises(ValueError, match="BF16"):
        function.apply(torch.ones(1), torch.ones(1), 1.0)


def test_precision_receipt_does_not_mutate_runtime(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "1")
    before = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
              torch.backends.cudnn.allow_tf32)
    value = runtime_precision_contract()
    assert value["tf32_cublas_override"] == "1"
    assert all(isinstance(x, str) for x in value.values())
    assert before == (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
                      torch.backends.cudnn.allow_tf32)


def test_runtime_receipts_require_actual_execution():
    build = {role: {"metadata": {"grad_lora_fused_epilogue": {
        "requested": True, "enabled": True, "fused_invocation_count": 0
    }}} for role in ("generator", "critic")}
    with pytest.raises(AssertionError, match="never executed"):
        validate_grad_epilogue_receipts(build, requested=True)
    for role in build:
        build[role]["metadata"]["grad_lora_fused_epilogue"]["fused_invocation_count"] = 1
    assert set(validate_grad_epilogue_receipts(build, requested=True)) == set(build)


@pytest.mark.parametrize("enabled", [False, True])
def test_controller_pins_explicit_flags_and_shared_precision(enabled):
    root = Path(__file__).resolve().parents[1]
    command = [sys.executable, str(root / "scripts/distillschedule/run_wan13b_fair_world16.py"),
        "--run-id", "grad-epilogue-plan", "--mode", "trajectory",
        "--placement", "resident-async128-tree-chunk2048", "--expected-head", "a"*40,
        "--model-root", "/model", "--bootstrap", "/bootstrap", "--bootstrap-hash", "b"*64,
        "--master-addr", "127.0.0.1", "--trajectory-cycles", "2", "--trajectory-arm-id", "resident",
        "--generator-nograd-lora-fused-epilogue", "--tf32-cublas-override", "1", "--dry-run"]
    if enabled:
        command.append("--grad-lora-fused-epilogue")
    env = dict(os.environ, PYTHONPATH=str(root / "src"))
    plan = json.loads(subprocess.check_output(command, env=env, text=True))
    for row in plan["commands"]:
        rendered = " ".join(row)
        assert f"{GRAD_EPILOGUE_ENV}={int(enabled)}" in rendered
        assert "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1" in rendered
