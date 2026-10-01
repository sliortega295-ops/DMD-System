"""Default-off C40 contract; GPU parity lives in its bounded canary."""
from pathlib import Path
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from distillschedule.runtime.livewan_lora_shared_input import (
    PLAN_KERNEL_CONTRACT, eligible, runtime_receipt_errors,
)

ROOT = Path(__file__).resolve().parents[1]


def test_import_is_lazy_and_default_worker_is_unchanged():
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-c",
        "import sys; from distillschedule.runtime import livewan_lora_shared_input; "
        "assert 'torch' not in sys.modules; assert 'triton' not in sys.modules"],
        env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    worker = (ROOT / "src/distillschedule/runtime/livewan_crossscale_worker.py").read_text()
    assert 'elif name == "scale_one_autocast_shared_bf16_input_v1"' in worker
    kernel = (ROOT / "src/distillschedule/runtime/livewan_lora_kernel.py").read_text()
    assert "shared_autocast_input: bool = False" in kernel
    assert "if shared_autocast_input:" in kernel


def test_two_outputs_share_payload_but_not_bf16_gradient_accumulation():
    source = (ROOT / "src/distillschedule/runtime/livewan_lora_shared_input.py").read_text()
    assert source.count("x.to(torch.bfloat16)") == 1
    assert "return y.view_as(y), y.view_as(y)" in source
    assert 'gradient_accumulation_dtype="float32"' in source
    assert "add.rn.f32" in source
    assert "ctx.set_materialize_grads(False)" in source
    assert ".cpu(" not in source
    assert "synchronize(" not in source


def test_eligibility_preserves_autocast_leaf_accumulation_and_unsupported_shapes():
    torch = SimpleNamespace(float32="fp32", is_grad_enabled=lambda: True)
    values = dict(is_cuda=True, dtype="fp32", ndim=3, shape=(2, 19320, 1536),
        is_contiguous=lambda: True, requires_grad=True, is_leaf=False)
    assert eligible(torch, SimpleNamespace(**values))
    for override in (dict(is_leaf=True), dict(shape=(3, 19320, 1536)),
            dict(dtype="bf16"), dict(is_contiguous=lambda: False), dict(is_cuda=False)):
        assert not eligible(torch, SimpleNamespace(**dict(values, **override)))
    assert eligible(torch, SimpleNamespace(**dict(values, is_leaf=True, requires_grad=False)))


def test_plan_is_one_factor_and_rejects_changed_gradient_precision(monkeypatch, tmp_path):
    from distillschedule.core.config import load_plan
    from distillschedule.runtime.livewan_crossscale_worker import _critic_lora_scale_one_enabled
    for key in ("LIVEWAN_STUDENT_CHECKPOINT", "LIVEWAN_CRITIC_CHECKPOINT",
            "LIVEWAN_TEACHER_CHECKPOINT", "LIVEWAN_OPERATION_TRACE", "LIVEWAN_SYSTEMS_BOOTSTRAP",
            "DISTILLSCHEDULE_OUTPUT_DIR", "DMD_SYSTEM_ROOT", "DISTILLSCHEDULE_PYTHON"):
        monkeypatch.setenv(key, str(ROOT) if key == "DMD_SYSTEM_ROOT" else str(tmp_path / key))
    plans = ROOT / "configs/distillschedule/plans"
    ref = load_plan(plans / "livewan_wan13b_14b_sf_dmd_c38_continuous_b16_score_fa3.json").plan.to_dict()
    cand = load_plan(plans / "livewan_wan13b_14b_sf_dmd_c40_continuous_b16_score_shared_input.json").plan.to_dict()
    assert cand["metadata"]["critic_lora_kernel"] == PLAN_KERNEL_CONTRACT
    assert _critic_lora_scale_one_enabled(cand, mode="pipeline")
    cand["id"] = ref["id"]
    cand["metadata"]["critic_lora_kernel"] = ref["metadata"]["critic_lora_kernel"]
    assert cand == ref
    changed = dict(PLAN_KERNEL_CONTRACT, gradient_accumulation_dtype="bfloat16")
    cand["metadata"]["critic_lora_kernel"] = changed
    with pytest.raises(Exception, match="contract changed"):
        _critic_lora_scale_one_enabled(cand, mode="pipeline")


def test_runtime_gate_requires_observed_forward_and_backward_not_config_only():
    row = dict(name=PLAN_KERNEL_CONTRACT["name"], shared_autocast_input=True,
        role="score", module_count=180,
        autocast_input_dtype_histogram={"torch.float32": 300},
        shared_input_kernel=dict(production_installed=True, contract_failures=0,
            gradient_accumulation_dtype="float32", forward_calls=300, backward_calls=100))
    assert not runtime_receipt_errors(row, rank=8)
    row["shared_input_kernel"]["backward_calls"] = 0
    assert runtime_receipt_errors(row, rank=8)
    row["role"] = "generator_shadow"
    assert not runtime_receipt_errors(row, rank=0)
    row["shared_input_kernel"]["forward_calls"] = 0
    assert runtime_receipt_errors(row, rank=0)
    row["shared_input_kernel"]["gradient_accumulation_dtype"] = "bfloat16"
    assert runtime_receipt_errors(row, rank=0)


@pytest.mark.parametrize("rank", range(16))
def test_split_dmd_frozen_shadow_has_forward_but_no_backward(rank):
    # Reproduce the preserved world16 r1 ledger, not inferred config counts.
    forward, backward = (480, 0) if rank < 8 else (2400, 1170)
    row = dict(name=PLAN_KERNEL_CONTRACT["name"], shared_autocast_input=True,
        role="generator_shadow" if rank < 8 else "score", module_count=180,
        autocast_input_dtype_histogram={"torch.float32": forward},
        shared_input_kernel=dict(production_installed=True, contract_failures=0,
            gradient_accumulation_dtype="float32", forward_calls=forward,
            backward_calls=backward))
    assert not runtime_receipt_errors(row, rank=rank)
    row["role"] = "score" if rank < 8 else "generator_shadow"
    assert runtime_receipt_errors(row, rank=rank)
    row["role"] = "generator_shadow" if rank < 8 else "score"
    row["shared_input_kernel"]["backward_calls"] = 1 if rank < 8 else 0
    assert runtime_receipt_errors(row, rank=rank)
    row["shared_input_kernel"]["backward_calls"] = backward
    row["autocast_input_dtype_histogram"]["torch.float32"] += 1
    assert runtime_receipt_errors(row, rank=rank)
