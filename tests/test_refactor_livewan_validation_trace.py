"""Real config/worker trace admission only: no model construction or CUDA work."""
import json
from pathlib import Path

import pytest

from distillschedule.core.config import load_plan
from distillschedule.runtime import livewan_crossscale_worker as worker


@pytest.mark.parametrize("value, expected", [(None, 0), ("0", 0), ("1", 1),
                                             ("2", 2), ("5", 5), ("50", 50)])
def test_validation_cycle_override_is_explicit(value, expected):
    env = {} if value is None else {"LIVEWAN_TRAJECTORY_CYCLES": value}
    assert worker._validation_trajectory_cycles("validate", env) == expected


@pytest.mark.parametrize("value", ["-1", "3", "8", "", "2.0", "invalid"])
def test_validation_cycle_override_fails_closed(value):
    with pytest.raises(ValueError):
        worker._validation_trajectory_cycles("validate", {"LIVEWAN_TRAJECTORY_CYCLES": value})


@pytest.mark.parametrize("kind", ["capacity", "correctness", "trajectory",
                                "two-cycle-timing", "timing", "profile"])
def test_nonvalidation_run_ignores_validation_parser(kind):
    assert worker._validation_trajectory_cycles(kind, {"LIVEWAN_TRAJECTORY_CYCLES": "invalid"}) == 0


@pytest.mark.parametrize("mode, cycles, expected", [
    ("baseline", None, 1), ("pipeline", None, 3),
    ("baseline", 2, 2), ("pipeline", 2, 2),
    ("pipeline", 1, 1), ("pipeline", 5, 5), ("pipeline", 50, 50),
])
def test_real_worker_passes_declared_cycles_to_trace_loader(monkeypatch, tmp_path, mode, cycles, expected):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("DISTILLSCHEDULE_HARDWARE_PROFILE", "h100-8-b16")
    monkeypatch.setenv("LIVEWAN_BASELINE_PLACEMENT", "legacy-hybrid")
    monkeypatch.setenv("LIVEWAN_PIPELINE_TEACHER_PLACEMENT", "legacy-sharded")
    monkeypatch.setenv("LIVEWAN_TEACHER_PARAMETER_OFFLOAD", "off")
    monkeypatch.setenv("LIVEWAN_PROFILE_TWO_CYCLES", "0")
    for key in ("LIVEWAN_STUDENT_CHECKPOINT", "LIVEWAN_CRITIC_CHECKPOINT",
                "LIVEWAN_TEACHER_CHECKPOINT", "LIVEWAN_OPERATION_TRACE",
                "LIVEWAN_SYSTEMS_BOOTSTRAP", "DISTILLSCHEDULE_OUTPUT_DIR"):
        monkeypatch.setenv(key, "/not-loaded/" + key)
    if cycles is None:
        monkeypatch.delenv("LIVEWAN_TRAJECTORY_CYCLES", raising=False)
    else:
        monkeypatch.setenv("LIVEWAN_TRAJECTORY_CYCLES", str(cycles))
        monkeypatch.setenv("LIVEWAN_TRAJECTORY_SEED", "20260830")
        monkeypatch.setenv("LIVEWAN_TRAJECTORY_ARM_ID", "validation-test")
    filename = ("livewan_wan13b_14b_sf_dmd_g1c5_world16_fsdp2.json" if mode == "baseline"
                else "livewan_wan13b_14b_sf_dmd_c34_continuous_b16_residual_layout.json")
    value = load_plan(root / "configs/distillschedule/plans" / filename).plan.to_dict()
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(value))
    actual = []

    class TraceLoaderReached(Exception):
        pass

    def trace_load(cls, path, *, expected_cycles, expected_global_batch):
        actual.append((expected_cycles, expected_global_batch))
        raise TraceLoaderReached

    monkeypatch.setattr(worker.FormalTrace, "load", classmethod(trace_load))
    with pytest.raises(TraceLoaderReached):
        worker.run_worker(plan_path=path, mode=mode, run_kind="validate", repo_root=root)
    assert actual == [(expected, 16)]
