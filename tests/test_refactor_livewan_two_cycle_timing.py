from __future__ import annotations

import inspect
import json
from pathlib import Path
import subprocess
import sys

import pytest

from distillschedule.core.config import load_plan
from distillschedule.runtime.livewan_crossscale_worker import (
    TWO_CYCLE_TIMING_CONTINUOUS_NAME,
    _apply_two_cycle_timing_contract,
    _record_two_cycle_endpoint,
    _run_baseline_two_cycle_timing,
    _run_pipeline_8x8_continuous,
)


ROOT = Path(__file__).resolve().parents[1]
HEAD = "a" * 40
IDENTITY = {
    "trace_sha256": "1" * 64,
    "bootstrap_sha256": "2" * 64,
    "source_receipt_sha256": "3" * 64,
}


@pytest.fixture(autouse=True)
def plan_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for key in (
        "LIVEWAN_STUDENT_CHECKPOINT",
        "LIVEWAN_CRITIC_CHECKPOINT",
        "LIVEWAN_TEACHER_CHECKPOINT",
        "LIVEWAN_OPERATION_TRACE",
        "LIVEWAN_SYSTEMS_BOOTSTRAP",
        "DISTILLSCHEDULE_OUTPUT_DIR",
    ):
        monkeypatch.setenv(key, str(tmp_path / key.lower()))


def _trajectory(path: Path, *, mode: str, variant: str, run_id: str) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "distillschedule.livewan_trajectory_summary.v1",
                "status": "PASS",
                "expected_head": HEAD,
                "head": HEAD,
                "heads": {mode: HEAD},
                "execution_mode": mode,
                "pipeline_variant": variant,
                "cycles": 2,
                "rank_cycle_receipts": 32,
                "expected_rank_cycle_receipts": 32,
                "final_versions": {"generator": 2, "critic": 10, "ema": 0},
                "formal_timing_eligible": False,
                "run_id": run_id,
                "arm_id": f"{mode}-arm",
                "seed": 17,
                "identity": IDENTITY,
            }
        )
    )


def _comparison(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "distillschedule.trajectory_coarse_comparison.v1",
                "comparison_kind": "semantic",
                "expected_cycles": 2,
                "expected_world_size": 16,
                "complete": True,
                "status": "TRAJECTORY_REVIEW_50C_COMPLETED",
                "contract": {"status": "PASS", "mismatches": []},
                "summary": {"state_comparison_count": 8},
                "state_comparisons": [
                    {
                        "cycle": cycle,
                        "component": component,
                        "finite": True,
                    }
                    for cycle in (1, 2)
                    for component in (
                        "generator_trainable",
                        "generator_adam",
                        "critic_trainable",
                        "critic_adam",
                    )
                ],
                "reference": {
                    "run_ids": ["baseline-run"],
                    "variants": ["baseline-arm"],
                    "seeds": [17],
                },
                "candidate": {
                    "run_ids": ["pipeline-run"],
                    "variants": ["pipeline-arm"],
                    "seeds": [17],
                },
                "source_binding": {
                    "expected_head": HEAD,
                    "baseline": {
                        "run_id": "baseline-run",
                        "arm_id": "baseline-arm",
                    },
                    "pipeline": {
                        "run_id": "pipeline-run",
                        "arm_id": "pipeline-arm",
                    },
                },
            }
        )
    )


def test_two_cycle_contract_is_exact_and_does_not_add_midpoint_sync() -> None:
    baseline = load_plan(
        ROOT
        / "configs/distillschedule/plans/"
        "livewan_wan13b_14b_sf_dmd_g1c5_world16_fsdp2.json"
    ).plan.to_dict()
    pipeline = load_plan(
        ROOT
        / "configs/distillschedule/plans/"
        "livewan_wan13b_14b_sf_dmd_c24_continuous_b16_score_modulation.json"
    ).plan.to_dict()
    _apply_two_cycle_timing_contract(baseline, mode="baseline")
    _apply_two_cycle_timing_contract(pipeline, mode="pipeline")
    assert baseline["evidence"]["steps"] == pipeline["evidence"]["steps"] == 10
    assert baseline["metadata"]["two_cycle_timing"]["cycles"] == 2
    assert pipeline["metadata"]["continuous_pipeline"]["name"] == (
        TWO_CYCLE_TIMING_CONTINUOUS_NAME
    )
    assert pipeline["metadata"]["continuous_pipeline"][
        "critic_shadow_sync_versions"
    ] == [5]

    baseline_source = inspect.getsource(_run_baseline_two_cycle_timing)
    endpoint_source = inspect.getsource(_record_two_cycle_endpoint)
    continuous_source = inspect.getsource(_run_pipeline_8x8_continuous)
    assert "cuda.synchronize" not in baseline_source
    assert "dist.barrier" not in baseline_source
    assert "cuda.Event(enable_timing=True)" in endpoint_source
    assert "event.record" in endpoint_source
    endpoint = continuous_source.index("_record_two_cycle_endpoint")
    lookahead = continuous_source.index("if cycle + 1 < trace.cycles")
    assert endpoint < lookahead


def test_gate_binds_exact_two_cycle_trajectories_and_cross_arm_sha(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline.json"
    pipeline = tmp_path / "pipeline.json"
    comparison = tmp_path / "comparison.json"
    output = tmp_path / "gate.json"
    _trajectory(baseline, mode="baseline", variant="b1", run_id="baseline-run")
    _trajectory(
        pipeline,
        mode="pipeline",
        variant="c24-continuous-b16-score-modulation",
        run_id="pipeline-run",
    )
    _comparison(comparison)
    command = [
        sys.executable,
        str(ROOT / "scripts/distillschedule/build_livewan_two_cycle_timing_gate.py"),
        "--baseline-trajectory-summary",
        str(baseline),
        "--pipeline-trajectory-summary",
        str(pipeline),
        "--trajectory-comparison",
        str(comparison),
        "--expected-head",
        HEAD,
        "--output",
        str(output),
    ]
    subprocess.run(command, check=True)
    gate = json.loads(output.read_text())
    assert gate["status"] == "PASS"
    assert gate["claimable_for_two_cycle_timing"] is True
    assert len(gate["trajectory_comparison"]["sha256"]) == 64

    broken = json.loads(comparison.read_text())
    broken["contract"]["status"] = "FAIL"
    comparison.write_text(json.dumps(broken))
    failed = subprocess.run(command, capture_output=True, text=True)
    assert failed.returncode != 0
    assert "not complete/PASS" in failed.stderr


def _timing_summary(path: Path, *, mode: str, seconds: float) -> None:
    variant = "b1" if mode == "baseline" else "c24-continuous-b16-score-modulation"
    path.write_text(
        json.dumps(
            {
                "schema_version": "distillschedule.livewan_two_cycle_timing_summary.v1",
                "status": "PASS",
                "execution_mode": mode,
                "pipeline_variant": variant,
                "run_kind": "two-cycle-timing",
                "native_unprofiled": True,
                "profiled": False,
                "profiler_status": "NOT_RUN",
                "formal_timing_eligible": True,
                "cycles_in_same_process": 2,
                "final_versions": {"generator": 2, "critic": 10, "ema": 0},
                "head": HEAD,
                "run_id": f"{mode}-timing",
                "matched_workload_fingerprint": "4" * 64,
                "identity": IDENTITY,
                "trace_sha256": IDENTITY["trace_sha256"],
                "logical_call_inventory": {"same": 1},
                "correctness_binding": {
                    "gate_sha256": "5" * 64,
                    "trajectory_comparison_sha256": "6" * 64,
                },
                "implementation": {"fingerprint": mode},
                "cycle_1": {
                    "claimable": False,
                    "world_completion_endpoint_seconds": 100.0,
                },
                "cycle_2": {"claimable": True, "seconds": seconds},
            }
        )
    )


def test_comparator_uses_cycle2_only_and_rejects_profiled_summary(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    pipeline = tmp_path / "pipeline.json"
    output = tmp_path / "pair.json"
    _timing_summary(baseline, mode="baseline", seconds=80.0)
    _timing_summary(pipeline, mode="pipeline", seconds=50.0)
    command = [
        sys.executable,
        str(ROOT / "scripts/distillschedule/compare_livewan_two_cycle_timing.py"),
        "--baseline-summary",
        str(baseline),
        "--pipeline-summary",
        str(pipeline),
        "--expected-head",
        HEAD,
        "--output",
        str(output),
    ]
    subprocess.run(command, check=True)
    result = json.loads(output.read_text())
    assert result["cycle_2_speedup"] == pytest.approx(1.6)
    assert result["timing_contract"]["average_cycles"] is False

    broken = json.loads(pipeline.read_text())
    broken["profiled"] = True
    pipeline.write_text(json.dumps(broken))
    failed = subprocess.run(command, capture_output=True, text=True)
    assert failed.returncode != 0
    assert "profiled" in failed.stderr


def test_controller_and_launcher_register_fail_closed_native_path() -> None:
    controller = (
        ROOT / "scripts/distillschedule/run_livewan_crossscale_world16.py"
    ).read_text()
    launcher = (
        ROOT / "scripts/distillschedule/launch_livewan_crossscale_world16_node.sh"
    ).read_text()
    summarizer = (
        ROOT / "scripts/distillschedule/summarize_livewan_two_cycle_timing_world16.py"
    ).read_text()
    for source in (controller, launcher):
        assert "two-cycle-timing" in source
        assert "c24-continuous-b16-score-modulation" in source
    assert 'export LIVEWAN_EXPECTED_HEAD="${EXPECTED_HEAD}"' in launcher
    assert "MAX_SCORE_POOL_COMPLETION_2_MINUS_MAX_SCORE_POOL_COMPLETION_1" in summarizer
    assert "statistics.mean(values)" in summarizer  # distribution only, not denominator
    assert '"seconds": c2_seconds' in summarizer
