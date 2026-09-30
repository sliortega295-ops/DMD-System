"""Metadata/plan scope; real numerical and performance gates run on CUDA."""
import inspect
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from distillschedule.runtime.livewan_score_layernorm import CANARY_CONTRACT, build, validate


def inputs(b=1, d=1536):
    s = 19320
    return SimpleNamespace(bfloat16="bf16"), SimpleNamespace(
        is_cuda=True, dtype="bf16", ndim=3, shape=(b, s, d),
        stride=lambda: (s*d, d, 1)), 1e-6


@pytest.mark.parametrize("b,d", [(b, d) for b in (1, 2) for d in (1536, 5120)])
def test_registered_row_major_frozen_shapes(b, d):
    validate(*inputs(b, d))


@pytest.mark.parametrize("attr,value", [
    ("dtype", "fp32"), ("is_cuda", False), ("ndim", 2),
    ("shape", (3, 19320, 1536)), ("shape", (1, 19320, 1280)),
    ("stride", lambda: (19320*1536, 1, 19320)),
])
def test_metadata_fail_closed(attr, value):
    t, x, eps = inputs(); setattr(x, attr, value)
    with pytest.raises(RuntimeError, match="contract changed"):
        validate(t, x, eps)


@pytest.mark.parametrize("eps", [0, 1e-5, float("nan"), float("inf"), "1e-6"])
def test_epsilon_fail_closed(eps):
    t, x, _ = inputs()
    with pytest.raises(RuntimeError, match="contract changed"):
        validate(t, x, eps)


def test_no_implicit_install_sync_copy_or_precision_change():
    source = inspect.getsource(build)
    assert CANARY_CONTRACT["production_installed"] is False
    assert CANARY_CONTRACT["numerical_gate"] == "UNCHANGED_DEFAULT_P0_PARITY"
    assert "enable_fp_fusion=False" in source
    assert "CANARY_ONLY_NOT_PRODUCTION_INSTALLED" in source
    assert "ctx.save_for_backward(x, mean, rstd)" in source
    for token in (".cpu(", ".item(", ".tolist(", "synchronize(", "all_reduce", "WanLayerNorm.forward ="):
        assert token not in source


def test_plan_only_has_no_torch_gpu_or_model_dependency(tmp_path):
    root = Path(__file__).resolve().parents[1]
    path = tmp_path / "never_created.json"
    r = subprocess.run([sys.executable, str(root / "scripts/distillschedule/canary_livewan_score_layernorm.py"),
        "--output", str(path), "--plan-only"], text=True, capture_output=True, check=True, timeout=15)
    x = json.loads(r.stdout)
    assert x["status"] == x["world16"] == "NOT_RUN"
    assert x["production_installed"] is False and not path.exists()


def test_real_canary_checks_source_grad_replay_and_operator_parent():
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts/distillschedule/canary_livewan_score_layernorm.py").read_text()
    assert "require_tensor_parity" in source and "tensor_parity_metrics" not in source
    assert "checkpoint_execution_counts" in source and "counts == [2, 2]" in source
    assert "event.cpu_parent" in source and "native_layer_norm_backward" in source
    assert "assert source_sha == SOURCE_SHA256" in source
    assert "assert WanLayerNorm.forward is native_forward" in source
