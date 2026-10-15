"""CPU admission for a default-off port, not proof of H100 numerical parity."""
import ast
import hashlib
import os
from pathlib import Path

import pytest

from distillschedule.runtime.wan13b_norm_policy import ENV, install, requested, validate

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "scripts/distillschedule/probe_wan13b_two_cycle_timing_world16.py"


def function(name):
    return next(n for n in ast.parse(WORKER.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == name)


def test_default_off_needs_no_model_or_cuda(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert requested() is False
    assert install(role="generator", placement="resident", device=None)["enabled"] is False


@pytest.mark.parametrize("bad", ["true", "on", "2", ""])
def test_nonliteral_flags_fail_closed(monkeypatch, bad):
    monkeypatch.setenv(ENV, bad)
    with pytest.raises(ValueError):
        requested()


def test_no_normalization_on_other_placements(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    with pytest.raises(ValueError):
        install(role="generator", placement="world16-fsdp", device=None)


def test_existing_h100_cycle_graph_is_unchanged():
    # Exact 7a668bc native two-B1/one-commit implementation, no reduced calls,
    # new scheduling, RNG or timer operations hidden in this port.
    # Hash literal source, not ast.dump: Python3.12 adds an empty type_params
    # field that the server's Python3.10 AST does not have.
    observed = hashlib.sha256(ast.get_source_segment(
        WORKER.read_text(), function("_run_workload_cycle")).encode()).hexdigest()
    assert observed == "4749e78fa9c26fe37c2c1e48e7b6829e398ce221fdc037f9f482f89c6b52083c"


def test_arm_fingerprint_binds_independent_flag(monkeypatch):
    ns = dict(Any=object, os=os, RESIDENT_WINNER="resident-async128-tree-chunk2048",
              FSDP_BASELINE="world16-fsdp", RESIDENT_REFERENCE="resident",
              _expected_arm_contract=lambda p: p)
    exec(compile(ast.Module(body=[function("_validate_arm_environment")],
                            type_ignores=[]), str(WORKER), "exec"), ns)
    monkeypatch.setenv("DISTILLSCHEDULE_WAN13B_GENERATOR_NOGRAD_LORA_FUSED_EPILOGUE", "1")
    monkeypatch.setenv("DISTILLSCHEDULE_WAN13B_GENERATOR_NOGRAD_LORA_CAST_CACHE", "0")
    monkeypatch.setenv(ENV, "0")
    control = ns["_validate_arm_environment"](ns["RESIDENT_WINNER"])
    monkeypatch.setenv(ENV, "1")
    candidate = ns["_validate_arm_environment"](ns["RESIDENT_WINNER"])
    assert candidate == dict(control, modulated_norm_fusion=True)
    with pytest.raises(ValueError, match="norm fusion"):
        ns["_validate_arm_environment"](ns["RESIDENT_REFERENCE"])


def test_requested_flag_is_not_runtime_evidence():
    row = dict(requested=True, enabled=True, registered_width=1536,
               fused_call_count=1, layernorm_fused_call_count=1,
               backward_kernel_call_count=1, reference_fallback_count=0,
               backward_kernel_reference_fallback_count=0)
    build = {r: {"metadata": {"modulated_norm_fusion": dict(row)}}
             for r in ("generator", "critic")}
    assert validate(build, enabled=True)["counter_scope"].startswith("PROCESS_SHARED")
    for field in ("fused_call_count", "layernorm_fused_call_count", "backward_kernel_call_count"):
        bad = {r: {"metadata": {"modulated_norm_fusion": dict(row)}}
               for r in ("generator", "critic")}
        bad["critic"]["metadata"]["modulated_norm_fusion"][field] = 0
        with pytest.raises(AssertionError):
            validate(bad, enabled=True)
    with pytest.raises(AssertionError):
        validate(build, enabled=False)


def test_no_graph_or_shadow_enabled_by_this_port():
    source = WORKER.read_text()
    assert "rollout_cuda_graph" not in source
    assert "rollout_merged_shadow" not in source
