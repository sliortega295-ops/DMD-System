"""Synthetic CPU-only bridge tests; not GPU calibration evidence."""
import json
import subprocess
import sys

import pytest

from distillschedule.search.execution import resolve_candidate
from distillschedule.search.wan_implementation import WAN, WanImplementation, PRECISION
from distillschedule.search.wan_model_adapter import ADAPTER, native_graph, operation_contract, cost_key
from distillschedule.workloads.best_known.registry import fingerprint, load_profile
from test_refactor_wan13b_two_cycle_timing import _synthetic_run, HEAD, SUMMARIZER


def stage_receipt(rank):
    impl = WanImplementation("fsdp", 512, False)
    memory = dict(allocated_bytes=100, reserved_bytes=140, driver_free_bytes=800,
                  driver_total_bytes=1000)
    stages = []
    for index, op in enumerate(native_graph().operations):
        contract = operation_contract(op, impl)
        stages.append(dict(status="PASS", operation_id=op.operation_id, key=cost_key(op, impl),
            operation_contract=contract, operation_contract_sha256=fingerprint(contract),
            start_ns=1000*index, stop_ns=1000*index+100, duration_ns=100,
            memory_before=memory, memory_after={**memory, "peak_allocated_bytes": 200,
                "peak_reserved_bytes": 250}, workspace_bytes_per_rank=100,
            errors=[], calibration_only=True, formal_timing_eligible=False, profiled=False,
            evidence_scope="MEASURED", kind="native_operation"))
    return dict(schema_version="distillschedule.wan_stage_rank_calibration.v1", status="PASS",
        adapter=ADAPTER, workload=WAN, source_head=HEAD, rank=rank,
        implementation_id=impl.id, implementation=impl.assignment, precision_fingerprint=PRECISION.digest,
        native_graph_sha256=fingerprint(native_graph().to_dict()),
        post_setup_static_bytes=100, post_setup_memory=memory, peak_allocated_bytes=200,
        peak_reserved_bytes=250, minimum_observed_driver_free_bytes=800,
        completed_operation_ids=[s["operation_id"] for s in stages], missing_operation_ids=[],
        errors=[], stages=stages, calibration_only=True, formal_timing_eligible=False,
        profiled=False, evidence_scope="MEASURED", kind="native_operation")


def calibration_run(tmp_path):
    run, output = _synthetic_run(tmp_path)
    for node in (0, 1):
        p = run / "launcher" / f"node_{node}" / "launch_manifest.txt"
        p.write_text(p.read_text().replace("mode=two-cycle-timing", "mode=two-cycle-calibration")
            .replace("two_cycle_claimable_cycle=2", "two_cycle_claimable_cycle=NONE")
            .replace("STEADY_STATE_CYCLE2_UNPROFILED", "NATIVE_STAGE_CALIBRATION_ONLY"))
    for rank in range(16):
        p = run / "execution" / "ranks" / f"rank_{rank:02d}.json"
        row = json.loads(p.read_text())
        row.update(schema_version="distillschedule.wan13b_two_cycle_calibration_rank.v1",
                   calibration_only=True, formal_timing_eligible=False, claimable_cycle=None,
                   stage_calibration=stage_receipt(rank))
        for cycle in row["cycles"]:
            cycle["role"] = "STAGE_CALIBRATION_DIAGNOSTIC"
        p.write_text(json.dumps(row))
    return run, output


def summarize(run, output, *flags):
    return subprocess.run([sys.executable, str(SUMMARIZER), "--run-root", str(run),
        "--placement", "world16-fsdp", "--expected-head", HEAD, "--output", str(output), *flags],
        capture_output=True, text=True)


def test_calibration_is_explicit_generated_only_and_canonical_is_unchanged():
    before = load_profile(WAN)
    plan = resolve_candidate(WAN, "joint-wan1-resident-c1024-e1", mode="calibration")
    assert plan["native_mode"] == "two-cycle-calibration"
    assert load_profile(WAN) == before
    assert "calibration" not in before["allowed_run_modes"]
    with pytest.raises(ValueError):
        resolve_candidate(WAN, "fsdp-reference", mode="calibration")


def test_timing_rejects_calibration_even_if_old_schema_is_forged(tmp_path):
    run, output = _synthetic_run(tmp_path)
    p = run / "execution/ranks/rank_00.json"
    row = json.loads(p.read_text())
    row["calibration_only"] = True
    row["stage_calibration"] = stage_receipt(0)
    p.write_text(json.dumps(row))
    assert summarize(run, output).returncode == 1
    assert json.loads(output.read_text())["formal_eligibility"] is False


def test_calibration_keeps_native_gates_but_cannot_claim_timing(tmp_path):
    run, output = calibration_run(tmp_path)
    completed = summarize(run, output, "--calibrate-stages")
    assert completed.returncode == 0, completed.stdout + completed.stderr
    value = json.loads(output.read_text())
    assert value["schema_version"] == "distillschedule.wan13b_two_cycle_calibration_summary.v1"
    assert value["status"] == "PASS" and value["rank_count"] == 16
    assert value["formal_eligibility"] is False
    assert value["cycle_2"]["claimable"] is False
    assert value["speed_claim_denominator"] is None
    assert value["stage_calibration"]["operation_count_per_rank"] == 14


@pytest.mark.parametrize("mutation", ["missing_stage", "wrong_cost", "nan_duration", "bad_census", "fixture"])
def test_invalid_calibration_or_semantics_cannot_be_used(tmp_path, mutation):
    run, output = calibration_run(tmp_path)
    p = run / "execution/ranks/rank_10.json"
    row = json.loads(p.read_text())
    if mutation == "missing_stage":
        row["stage_calibration"]["stages"].pop()
    elif mutation == "wrong_cost":
        row["stage_calibration"]["stages"][0]["key"] = "bad"
    elif mutation == "nan_duration":
        row["stage_calibration"]["stages"][0]["duration_ns"] = float("nan")
    elif mutation == "fixture":
        row["stage_calibration"]["evidence_scope"] = "CPU_FIXTURE_ONLY"
    else:
        row["updates"]["generator"] = 0
    p.write_text(json.dumps(row))
    completed = summarize(run, output, "--calibrate-stages")
    assert completed.returncode == 1, completed.stdout + completed.stderr
    value = json.loads(output.read_text())
    assert value["status"] == "FAIL" and value["formal_eligibility"] is False


def test_real_worker_call_sites_default_and_calibration_have_identical_model_calls():
    """Execute the actual function AST with metadata-only stand-ins, not a copy."""
    import ast
    import __future__
    from contextlib import contextmanager, nullcontext
    from types import SimpleNamespace
    from test_refactor_wan13b_two_cycle_timing import WORKER

    node = next(n for n in ast.parse(WORKER.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == "_run_workload_cycle")
    calls, stage_calls = [], []
    def g(**kw):
        calls.append(("G", kw))
        return "G-record"
    def rollout(**kw):
        calls.append(("rollout", {k: v for k, v in kw.items() if k != "entry_outputs"}))
        kw["entry_outputs"].update({i: f"entry-{i}" for i in range(10, 15)})
        return {i: f"x-{i}" for i in range(10, 15)}, "rollout-records", None
    def c(**kw):
        calls.append(("C", kw))
        return f"C-record-{kw['step']}"
    ns = dict(p0=SimpleNamespace(_run_nogan_generator_update_p0=g,
        _precompute_critic_rollouts=rollout, _run_nogan_critic_precomputed=c),
        SOURCE_STEPS=tuple(range(10, 15)), nullcontext=nullcontext)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(WORKER), "exec",
                 flags=__future__.annotations.compiler_flag), ns)
    class Observer:
        @contextmanager
        def stage(self, cycle, name):
            stage_calls.append((cycle, name))
            yield
    kwargs = dict(rank=0, device="metadata", generator_engine="G", critic_engine="C",
        ema=SimpleNamespace(step=lambda: calls.append(("EMA", {}))), trace="trace",
        negative="negative", diagnostics="diagnostics", policy="policy", clean_runtime=True,
        runtime_cache={}, deferred_finite_checks=[])
    reference = ns["_run_workload_cycle"](**kwargs)
    reference_calls = list(calls)
    calls.clear()
    observed = ns["_run_workload_cycle"](**kwargs, calibrator=Observer(), cycle=2)
    assert observed == reference and calls == reference_calls
    assert [x[0] for x in calls] == ["G", "rollout"] + ["C", "EMA"]*5
    assert stage_calls == [(2, name) for name in ["generator_update", "critic_rollouts",
                          *[f"critic{i}_update_ema" for i in range(5)]]]


def test_native_assembly_exports_reconcilable_fragment_not_full_search_data(tmp_path, monkeypatch):
    from distillschedule.search import native
    from distillschedule.search.wan_calibration_evidence import assemble_native
    from distillschedule.search.runtime_identity import observed_binding
    from distillschedule.search.cost_model import CostModel
    from test_refactor_model_validation import fake_observations

    run, summary_path = calibration_run(tmp_path)
    for rank in range(16):
        p = run / "execution/ranks" / f"rank_{rank:02d}.json"
        row = json.loads(p.read_text())
        row["runtime_contract"]["planner_execution_precision"] = {"status": "PASS", "precision_fingerprint": PRECISION.digest}
        value = row["stage_calibration"]
        for m in [value["post_setup_memory"], *[s[k] for s in value["stages"] for k in ("memory_before", "memory_after")]]:
            m["driver_total_bytes"] = 40 * 2**30
            m["driver_free_bytes"] = 35 * 2**30
        value["minimum_observed_driver_free_bytes"] = 35 * 2**30
        p.write_text(json.dumps(row))
    completed = summarize(run, summary_path, "--calibrate-stages")
    assert completed.returncode == 0, completed.stdout + completed.stderr
    summary = json.loads(summary_path.read_text())
    (run / "bound-summary.json").write_text(json.dumps(summary))
    monkeypatch.setattr(native, "checked_summary", lambda receipt: (run, summary))
    comparison = run / "comparison.json"
    comparison.write_text(json.dumps(dict(schema_version="distillschedule.trajectory_coarse_comparison.v1",
        comparison_kind="semantic", expected_cycles=2, expected_world_size=16, complete=True,
        contract={"status": "PASS", "mismatches": []}, status="TRAJECTORY_REVIEW_50C_COMPLETED")))
    nodes = fake_observations()
    for n in nodes:
        n["source_head"] = HEAD
    identity = dict(source_head=HEAD, asset_fingerprint="a"*64, semantic_fingerprint="b"*64,
        precision_fingerprint=PRECISION.digest, **observed_binding(nodes, source_head=HEAD))
    impl = WanImplementation("fsdp", 512, False)
    receipt = dict(native_mode="two-cycle-calibration", mode="calibration", integration_head=HEAD,
        asset_fingerprint="a"*64, workload_semantics_hash="b"*64, search_candidate={"id": impl.id}, run_id=summary["run_id"])
    preparation = dict(source_head=HEAD, candidate_id=impl.id,
        comparison=native.binding(comparison), correctness=native.binding(summary_path), strict_numerical_gate="NOT_ESTABLISHED")
    for i in (0, 1):
        (run / f"search-resources.node{i}.json").write_text(json.dumps(dict(status="PASS", node_rank=i,
            run_id=summary["run_id"], samples=[dict(cgroup_events=dict(oom=0, oom_kill=0))]*2)))
    result = assemble_native(receipt, preparation, identity, {"before": nodes, "after": nodes})
    assert result["status"] == "PASS_NATIVE_STAGE_CALIBRATION"
    assert result["speedup"] is None and result["complete_planning_layout"] is False
    costs = CostModel.from_dict(json.loads((run / "stage-costs/operation-costs.json").read_text()))
    costs.verify_receipts()
    assert len(costs.records) == 14 and all(c.samples_ns == (100,) for c in costs.records)
    assert all(c.workspace_bytes_per_rank == 100 for c in costs.records)
    with pytest.raises(FileExistsError):
        assemble_native(receipt, preparation, identity, {"before": nodes, "after": nodes})


def test_actual_launcher_heredoc_accepts_two_cycle_calibration_gate(tmp_path):
    import re
    from test_refactor_wan13b_two_cycle_timing import LAUNCHER
    code = next(s for s in re.findall(r"<<'PY'\n(.*?)\nPY", LAUNCHER.read_text(), re.S)
                if "two_cycle_seed, two_cycle_arm_id" in s)
    gate = dict(schema_version="distillschedule.wan13b_trajectory_summary.v1", status="PASS",
        placement="world16-fsdp", cycles=2, rank_count=16, expected_head=HEAD,
        diagnostic_only=True, formal_timing_eligible=False,
        final_versions=dict(generator=4, critic=20, ema=20),
        matched_contract=dict(bootstrap_hash="b"*64, trace_bundle_hash="c"*64,
            trace_replay_policy="FIXED_FIVE_STEP_COHORT_REPLAY", seed=7,
            global_batch=16, updates="G1_C5_EMA5", rollout_batch_size=1, score_packing=False),
        implementation_contract={"trajectory_arm_id": "test"},
        runtime_receipts=[dict(helios_transformer_sha256="d"*64)]*16)
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(gate))
    completed = subprocess.run([sys.executable, "-", str(path), "world16-fsdp", "two-cycle-calibration",
        "d"*64, HEAD, "b"*64, "c"*64, "7", "test"], input=code, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_collector_preserves_fsdp_inner_rollout_peak_resets():
    from test_refactor_wan_stage_calibration import FakeBackend, DEVICE, SCOPE
    from distillschedule.search.wan_stage_calibration import StageCalibrator
    backend = FakeBackend()
    original = backend.reset_peak_memory_stats
    c = StageCalibrator(torch=backend, distributed=backend, device=DEVICE, rank=0,
        implementation=WanImplementation("fsdp", 512, False), source_head=HEAD,
        clock=lambda: backend.now, evidence_scope=SCOPE)
    with c.stage(1, "generator_update"):
        backend.peak_allocated = 1200
        backend.peak_reserved = 1400
        backend.reset_peak_memory_stats(DEVICE)
        backend.peak_allocated = 300
        backend.peak_reserved = 400
    assert backend.reset_peak_memory_stats == original
    row = c.snapshot()["stages"][0]
    assert row["memory_after"]["peak_allocated_bytes"] == 1200
    assert row["workspace_bytes_per_rank"] == 1100
    assert row["nested_peak_reset_count"] == 1
