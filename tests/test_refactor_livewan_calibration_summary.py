"""Real CPU summary CLI over synthetic trace/launcher/rank receipts; no GPU.

Nothing is mocked except argv. The fake prompts file is only hash-read, never
loaded as model data. Calibration raw-cost admission is covered separately in
test_refactor_native_macro_calibration.py; this tests summary eligibility.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

from distillschedule.workloads.best_known.registry import file_sha256
from test_refactor_livewan_trajectory import _source_trace


ROOT = Path(__file__).resolve().parents[1]
HEAD = "a" * 40


def _module(name):
    path = ROOT / "scripts/distillschedule" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"cpu_test_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _inputs(tmp_path, monkeypatch, mode):
    module = _module("summarize_livewan_two_cycle_timing_world16")
    source = tmp_path / "source-trace.json"
    _source_trace(source)
    prompts = tmp_path / "cpu-prompts.json"
    prompts.write_text('{"fixture_scope": "CPU_ONLY_NOT_MODEL_DATA"}')
    payload = json.loads(source.read_text())
    payload["prompts"] = dict(path=str(prompts), sha256=file_sha256(prompts))
    _json(source, payload)
    trace_path = tmp_path / "two-cycle-trace.json"
    with monkeypatch.context() as cli:
        cli.setattr(sys, "argv", ["trace-builder", "--source-trace", str(source),
            "--output", str(trace_path), "--cycles", "2"])
        assert _module("build_livewan_continuous_trace").main() == 0
    trace = module.FormalTrace.load(trace_path, expected_cycles=2, expected_global_batch=16)
    logical, physical = module._expected_inventory(trace, mode=mode)
    run = tmp_path / "cpu-calibration"
    variant = "b1" if mode == "baseline" else "c24-continuous-b16-score-modulation"
    effective = dict(evidence=dict(steps=10), metadata=dict(execution_mode=mode))
    effective_sha = module._sha256_json(effective)
    for node in (0, 1):
        launch = run / "launcher" / f"node_{node}"
        launch.mkdir(parents=True)
        manifest = dict(node_rank=node, head=HEAD, execution_mode=mode,
            pipeline_variant=variant, run_kind="two-cycle-timing", run_id=run.name,
            two_cycle_count=2, profiled="false", profiler_status="NOT_RUN",
            claim_boundary="STEADY_STATE_CYCLE2_SCORE_COMPLETION_ENDPOINT_DELTA",
            two_cycle_seed=17, two_cycle_arm_id="cpu-arm", two_cycle_correctness_sha256="b" * 64)
        (launch / "launch_manifest.txt").write_text("".join(f"{k}={v}\n" for k, v in manifest.items()))
        (launch / "teardown_status.txt").write_text(
            f"status=PASS\ngpu_process_count=0\nowner_run_id={run.name}\n")
        for boundary in ("before", "after"):
            (launch / f"cgroup_memory_events_{boundary}.txt").write_text("oom 0\noom_kill 0\n")
        _json(run / f"effective_plan.node{node}.json", dict(effective, effective_plan_sha256=effective_sha))

    for rank in range(16):
        generator = rank < 8
        def local_counts(counts):
            if mode == "baseline":
                return {key: count // 16 for key, count in counts.items()}
            return {key: count // 8 if key.startswith("student/") == generator else 0
                    for key, count in counts.items()}
        local_logical, local_physical = local_counts(logical), local_counts(physical)
        commits = (2, 10) if mode == "baseline" else ((2, 0) if generator else (0, 10))
        contracts = [dict(cycle=cycle, local_versions=dict(
            generator=cycle if commits[0] else 0, critic=cycle * 5 if commits[1] else 0, ema=0),
            logical_call_inventory_delta={k: v // 2 for k, v in local_logical.items()},
            physical_call_inventory_delta={k: v // 2 for k, v in local_physical.items()}) for cycle in (1, 2)]
        timing = dict(schema_version="distillschedule.livewan_two_cycle_timing_rank.v1",
            cycles_in_same_process=2, profiled=False, profiler_status="NOT_RUN",
            state_anchors="NOT_COLLECTED_IN_TIMING_MODE", evidence_extraction="AFTER_BOTH_TIMED_CYCLE_ENDPOINTS",
            cycle_1=dict(role="WARMUP_DIAGNOSTIC", claimable=False, completion_offset_seconds=100 + rank),
            cycle_2=dict(role="STEADY_STATE_CLAIMABLE", claimable=True,
                completion_offset_seconds=150 + rank, local_endpoint_delta_seconds=50),
            correctness_binding=dict(sha256="b" * 64, trajectory_comparison_sha256="c" * 64,
                arm_id="cpu-arm", pipeline_variant=variant, status="PASS"), per_cycle_contract=contracts)
        raw = dict(status="PASS", rank=rank, run_kind="two-cycle-timing", mode=mode,
            profiled=False, trace_sha256=trace.sha256, bootstrap_sha256="d" * 64,
            source_receipt_sha256="e" * 64, logical_generator_updates=2, logical_critic_updates=10,
            physical_generator_commits=commits[0], physical_critic_commits=commits[1],
            final_versions=dict(generator=2, critic=10, ema=0), effective_plan_sha256=effective_sha,
            logical_call_inventory=local_logical, physical_call_inventory=local_physical,
            two_cycle_timing=timing, operation_calibration=dict(enabled=True, status="PASS",
                calibration_only=True, formal_timing_eligible=False,
                path=str(run / "operation_calibration" / f"rank_{rank:03d}.json")))
        if mode == "pipeline":
            raw.update(continuous_pipeline=dict(name=module.TWO_CYCLE_TIMING_CONTINUOUS_NAME,
                cycles=2, execution_status="PASS", critic_physical_batch_size=2,
                critic_physical_backward_calls_per_step=1, critic_step1_single_commit=True,
                source_reuse_across_cycles=True), generator_runahead_transport=dict(
                execution_status="PASS", role="generator" if generator else "score",
                cycles=2, cross_cycle_lookahead=True), critic_shadow_hot_sync=dict(
                execution_status="PASS", role="generator_receiver" if generator else "score_source",
                events=[dict(version=5)]))
        _json(run / "receipts" / f"rank_{rank:03d}.json", raw)
    output = run / "summary.json"
    monkeypatch.setattr(sys, "argv", ["summary", "--run-root", str(run), "--execution-mode", mode,
        "--pipeline-variant", variant, "--trace", str(trace_path), "--expected-head", HEAD,
        "--output", str(output)])
    return module, run, output


@pytest.mark.parametrize("mode", ["baseline", "pipeline"])
@pytest.mark.parametrize("enabled", [True, False], ids=["calibration", "timing-control"])
def test_enabled_calibration_is_never_claimable_and_timing_control_is_unchanged(tmp_path, monkeypatch, mode, enabled):
    module, run, output = _inputs(tmp_path, monkeypatch, mode)
    if not enabled:
        for path in (run / "receipts").glob("rank_*.json"):
            raw = json.loads(path.read_text())
            raw["operation_calibration"] = {"enabled": False}
            _json(path, raw)
    assert module.main() == 0
    result = json.loads(output.read_text())
    assert result["status"] == "PASS" and result["errors"] == []
    assert result["rank_count"] == 16 and result["calibration_only"] is enabled
    assert result["formal_timing_eligible"] is not enabled
    assert result["cycle_2"]["claimable"] is not enabled
    assert result["cycle_2"]["seconds"] == 50
    assert result["claim_boundary"] == (
        "OPERATION_CALIBRATION_DIAGNOSTIC_ONLY" if enabled else "CYCLE2_ONLY_NATIVE_UNPROFILED")


@pytest.mark.parametrize("mode", ["baseline", "pipeline"])
@pytest.mark.parametrize("mutation,error", [
    ("mixed", "operation calibration policy differs across ranks"),
    ("failed", "operation calibration rank receipt failed"),
    ("versions", "logical updates or final versions changed"),
])
def test_calibration_still_fails_closed_on_mixed_policy_or_native_failure(tmp_path, monkeypatch, mode, mutation, error):
    module, run, output = _inputs(tmp_path, monkeypatch, mode)
    path = run / "receipts/rank_015.json"
    raw = json.loads(path.read_text())
    if mutation == "mixed": raw["operation_calibration"]["enabled"] = False
    if mutation == "failed": raw["operation_calibration"]["status"] = "FAIL"
    if mutation == "versions": raw["final_versions"]["critic"] = 9
    _json(path, raw)
    assert module.main() == 1
    result = json.loads(output.read_text())
    assert result["status"] == "FAIL" and any(error in item for item in result["errors"])
    assert result["calibration_only"] is True
    assert result["formal_timing_eligible"] is False and result["cycle_2"]["claimable"] is False
