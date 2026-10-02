import copy
import inspect
import json
from pathlib import Path

import pytest

from distillschedule.runtime.livewan_generator_fused_rms import (
    CONTRACT, FusedGeneratorRMS, build, enabled, runtime_receipt_errors,
)
from distillschedule.runtime.livewan_generator_rms_epilogue import (
    GeneratorRMSEpilogue, runtime_receipt_errors as general_errors,
)


def test_only_kernel_replaced_and_default_import_does_not_build():
    assert FusedGeneratorRMS.scope is GeneratorRMSEpilogue.scope
    assert FusedGeneratorRMS.receipt is GeneratorRMSEpilogue.receipt
    assert not enabled({}, mode="pipeline")
    source = inspect.getsource(build)
    assert "tl.sum(x * x, 0) / D" in source
    assert ".to(tl.bfloat16).to(tl.float32)" in source
    assert "enable_fp_fusion=False" in source
    assert "dtype=torch.float32" in source
    for token in ("synchronize(", ".cpu(", ".item(", ".tolist(", ".step("):
        assert token not in source
    assert CONTRACT["reduction_tree_changed"] is True
    assert CONTRACT["semantic_math_changed"] is False
    assert CONTRACT["native_mean_preserved"] is False


def test_new_plan_changes_only_norm_and_registers_exact_parent(monkeypatch):
    from distillschedule.core.config import load_plan
    from distillschedule.runtime.livewan_crossscale_worker import (
        _generator_rms_epilogue_enabled, _generator_rms_cast_square_enabled,
        _validate_plan_identity,
    )
    root = Path(__file__).resolve().parents[1] / "configs/distillschedule/plans"
    for k in ("LIVEWAN_STUDENT_CHECKPOINT", "LIVEWAN_CRITIC_CHECKPOINT",
              "LIVEWAN_TEACHER_CHECKPOINT", "LIVEWAN_OPERATION_TRACE",
              "LIVEWAN_SYSTEMS_BOOTSTRAP", "DISTILLSCHEDULE_OUTPUT_DIR"):
        monkeypatch.setenv(k, "/unread/" + k)
    parent = load_plan(root / "livewan_wan13b_14b_sf_dmd_c49_generator_readonly_splitkv.json").plan.to_dict()
    plan = load_plan(root / "livewan_wan13b_14b_sf_dmd_c51_generator_fused_rms.json").plan.to_dict()
    a, b = copy.deepcopy(parent), copy.deepcopy(plan)
    a.pop("id"); b.pop("id")
    a["metadata"].pop("generator_rms_epilogue")
    b["metadata"].pop("generator_rms_epilogue")
    assert a == b
    assert enabled(plan, mode="pipeline")
    assert _generator_rms_epilogue_enabled(plan, mode="pipeline")
    assert not _generator_rms_cast_square_enabled(plan, mode="pipeline")
    _validate_plan_identity(plan, mode="pipeline")
    with pytest.raises(RuntimeError):
        enabled(plan, mode="baseline")
    for key, value in (("num_warps", 8), ("output_dtype", "bfloat16"),
                       ("native_mean_preserved", True), ("grad_path_changed", True)):
        bad = copy.deepcopy(plan)
        bad["metadata"]["generator_rms_epilogue"][key] = value
        with pytest.raises(RuntimeError):
            enabled(bad, mode="pipeline")
    for key in ("critic_ffn_down_checkpoint", "critic_step1_overlap",
                "step0_wave1_early_delivery"):
        bad = copy.deepcopy(plan); bad["metadata"][key] = {"enabled": True}
        with pytest.raises(RuntimeError):
            enabled(bad, mode="pipeline")


def test_runtime_receipt_is_not_configuration_only():
    r = dict(CONTRACT, execution_status="PASS", scope_entries=20,scope_exits=20,
             active=False,scope_failures=0,contract_failures=0,
             scope_forward_calls=[1920]*20,calls=38400,
             shape_histogram={"1x2760x1536":38400})
    assert not runtime_receipt_errors(r,rank=0,cycles=2)
    assert not general_errors(r,rank=0,cycles=2)
    assert runtime_receipt_errors(r,rank=8,cycles=2)
    assert not runtime_receipt_errors({"execution_status":"NOT_APPLICABLE"},rank=8,cycles=2)
    for key,value in (("calls",0),("active",True),("scope_entries",19),
                      ("scope_failures",1),("output_dtype","bfloat16"),
                      ("native_mean_preserved",True),("shape_histogram",{}),
                      ("contract_failures",1)):
        assert runtime_receipt_errors(dict(r,**{key:value}),rank=0,cycles=2)


def test_grad_inputs_rejected_before_new_kernel_work():
    class Torch:
        @staticmethod
        def is_grad_enabled(): return True
    from distillschedule.runtime.livewan_generator_rms_epilogue import validate_inputs
    with pytest.raises(RuntimeError,match="requires no_grad"):
        validate_inputs(Torch,None,None,1e-6)
