from types import SimpleNamespace
import inspect
import pytest

from distillschedule.runtime.livewan_score_rmsnorm import (
    validate_frozen_rms_inputs,
    build_frozen_score_rmsnorm,
)


def tensors():
    torch = SimpleNamespace(bfloat16="bf16")
    x = SimpleNamespace(is_cuda=True, device="cuda:0", dtype="bf16", ndim=3,
                        shape=(2, 19320, 1536), is_contiguous=lambda: True)
    w = SimpleNamespace(is_cuda=True, device="cuda:0", dtype="bf16", shape=(1536,),
                        is_contiguous=lambda: True, requires_grad=False)
    return torch, x, w


def test_registered_metadata():
    validate_frozen_rms_inputs(*tensors(), 1e-6)


@pytest.mark.parametrize("attribute,value", [
    ("dtype", "fp32"), ("is_cuda", False), ("requires_grad", True),
    ("device", "cuda:1"), ("shape", (5120,)),
])
def test_affine_contract_fail_closed(attribute, value):
    torch, x, w = tensors()
    setattr(w, attribute, value)
    with pytest.raises(RuntimeError, match="contract changed"):
        validate_frozen_rms_inputs(torch, x, w, 1e-6)


def test_epsilon_fail_closed():
    with pytest.raises(RuntimeError):
        validate_frozen_rms_inputs(*tensors(), 1e-5)


def test_not_installed_by_default():
    source = inspect.getsource(build_frozen_score_rmsnorm)
    assert "enable_fp_fusion=False" in source
    assert "CANARY_ONLY_NOT_PRODUCTION_INSTALLED" in source
    assert "torch.cuda.synchronize" not in source
    assert ".cpu()" not in source


def test_c25_has_exactly_one_plan_difference():
    import json
    from pathlib import Path
    from distillschedule.runtime.livewan_score_rmsnorm import PLAN_CONTRACT

    root = Path(__file__).resolve().parents[1] / "configs/distillschedule/plans"
    parent = json.loads((root / "livewan_wan13b_14b_sf_dmd_c24_continuous_b16_score_modulation.json").read_text())
    candidate = json.loads((root / "livewan_wan13b_14b_sf_dmd_c25_continuous_b16_score_rmsnorm.json").read_text())
    assert candidate["metadata"].pop("score_rmsnorm_kernel") == PLAN_CONTRACT
    assert candidate == parent


def test_plan_contract_cannot_change_epsilon_or_precision(monkeypatch):
    from distillschedule.runtime.livewan_score_rmsnorm import PLAN_CONTRACT
    from distillschedule.runtime.livewan_crossscale_worker import _score_rmsnorm_kernel_enabled
    import copy
    import json
    from pathlib import Path
    from distillschedule.core.config import load_plan
    for key in ("LIVEWAN_STUDENT_CHECKPOINT", "LIVEWAN_CRITIC_CHECKPOINT", "LIVEWAN_TEACHER_CHECKPOINT",
                "LIVEWAN_OPERATION_TRACE", "LIVEWAN_SYSTEMS_BOOTSTRAP", "DISTILLSCHEDULE_OUTPUT_DIR"):
        monkeypatch.setenv(key, f"/tmp/{key.lower()}")
    path = Path(__file__).resolve().parents[1] / "configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_c25_continuous_b16_score_rmsnorm.json"
    plan = load_plan(path).plan.to_dict()
    assert _score_rmsnorm_kernel_enabled(plan, mode="pipeline")
    with pytest.raises(RuntimeError):
        _score_rmsnorm_kernel_enabled(plan, mode="baseline")
    for key, value in [("epsilon", 1e-5), ("input_output_affine_dtype", "float32"), ("enabled", False)]:
        changed = copy.deepcopy(plan)
        changed["metadata"]["score_rmsnorm_kernel"][key] = value
        with pytest.raises(RuntimeError):
            _score_rmsnorm_kernel_enabled(changed, mode="pipeline")


def test_runtime_receipt_fail_closed():
    from distillschedule.runtime.livewan_score_rmsnorm import runtime_receipt_errors, PLAN_CONTRACT, REFERENCE_CLASS_SHA256
    good = {"name": PLAN_CONTRACT["name"], "source_class_sha256": REFERENCE_CLASS_SHA256,
            "install_status": "PASS", "contract_failures": 0, "forward_calls": 100,
            "backward_calls": 50, "shape_histogram": {"2x19320x1536": 80, "2x19320x5120": 20}}
    assert runtime_receipt_errors(good, rank=8) == []
    assert runtime_receipt_errors({}, rank=8)
    assert runtime_receipt_errors(dict(good, contract_failures=1), rank=8)
    assert runtime_receipt_errors(dict(good, backward_calls=0), rank=8)
    assert runtime_receipt_errors({"execution_status": "NOT_APPLICABLE", "forward_calls": 0, "backward_calls": 0}, rank=0) == []
    assert runtime_receipt_errors(good, rank=0)


@pytest.mark.parametrize("kind", ["trajectory", "two-cycle-timing"])
def test_c25_controller_dry_run_reaches_exact_launcher(tmp_path, kind):
    import json
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    args = [sys.executable, str(root / "scripts/distillschedule/run_livewan_crossscale_world16.py"),
        "--hosts", "node0,node1", "--run-id", "test-c25", "--run-kind", kind,
        "--execution-mode", "pipeline", "--pipeline-variant", "c25-continuous-b16-score-rmsnorm",
        "--expected-head", "a" * 40, "--master-addr", "127.0.0.1", "--dry-run",
        "--local-evidence-root", str(tmp_path)]
    for key in ["remote-root", "remote-output-root", "student-checkpoint", "critic-checkpoint",
                "teacher-checkpoint", "trace", "bootstrap", "source-root", "source-receipt", "python-overlay"]:
        args += ["--" + key, str(tmp_path / key)]
    if kind == "trajectory":
        args += ["--trajectory-cycles", "2", "--trajectory-arm-id", "c25", "--trajectory-seed", "17"]
    else:
        args += ["--two-cycle-arm-id", "c25", "--two-cycle-seed", "17",
                 "--correctness-receipt", str(tmp_path / "gate.json"), "--correctness-sha256", "b" * 64]
    p = subprocess.run(args, check=True, text=True, capture_output=True)
    value = json.loads(p.stdout)
    assert len(value["commands"]) == 2
    assert all("c25-continuous-b16-score-rmsnorm" in " ".join(c) for c in value["commands"])
