"""CPU fixtures exercise the native bridge; they are not H3 GPU evidence."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from distillschedule.search.contracts import semantic_contract
from distillschedule.search.h3_implementation import H3, H3Implementation, current_implementation
from distillschedule.search.h3_native import compare_native_two_cycle, summarize_native_two_cycle
from distillschedule.workloads.best_known.registry import load_profile

HEAD = "a" * 40


def _state(rank):
    row = dict(name="weight", local_shape=[4], stored_shape=[64], dtype="torch.bfloat16",
               local_flat_indices=[0, 1, 2, 3], values_fp32=[1., 2., 3., 4.],
               representation="DTENSOR_LOCAL_SHARD")
    rows = [row]
    component = dict(inventory_tensor_count=1, sampled_tensor_count=1, rows=rows,
                     inventory_sha256=hashlib.sha256(json.dumps(["weight"]).encode()).hexdigest(),
                     sample_sha256=hashlib.sha256(json.dumps(rows, sort_keys=True, allow_nan=False).encode()).hexdigest())
    return dict(schema_version="distillschedule.h3_local_shard_fingerprint.v1", rank=rank,
                status="CAPTURED_FINITE_SAMPLES", scope="SAME_RANK_LOCAL_LAYOUT_SAMPLED_COORDINATES_ONLY",
                max_tensors_per_component=32, samples_per_tensor=16,
                full_state_finite_check="NOT_RUN", numerical_cross_reference="NOT_RUN",
                components={name: copy.deepcopy(component) for name in
                            ("student_parameters", "student_optimizer", "fake_parameters", "fake_optimizer")})


def _receipts(impl=None):
    impl = impl or H3Implementation()
    p = impl.lower_profile()
    binding = impl.receipt_binding(p["runtime"]["model_config"], p["runtime"]["environment"])
    rows = []
    for rank in range(16):
        cycles = []
        for cycle in (1, 2):
            census = dict(fixed_end_step_idx=3, forward_counts=dict(student=24, fake=6, teacher=1),
                          grad_forward_counts=dict(student=1, fake=5, teacher=0),
                          backward_counts=dict(student=1, fake=5),
                          samples=[dict(stage=stage, identity=f"{cycle}:{rank}:{stage}") for stage in
                                   ("student", "fake_0", "fake_1", "fake_2", "fake_3", "fake_4")])
            replay = dict(checkpoint_wrap_count=300, scoped_execution_count=600,
                          cache_hit_count=600, cache_miss_count=0, missing_key_count=0)
            cycles.append(dict(cycle=cycle, status="PASS_NATIVE_RUNTIME_CONTRACTS", census=census,
                               native_wall_seconds=100 + rank - cycle, gpu_completion_elapsed_s=90 + rank - cycle,
                               updates=dict(student=1, fake=5, ema=0),
                               cumulative_updates=dict(student=cycle, fake=5 * cycle, ema=0),
                               grid_replay_delta=replay, grid_replay={k: v * cycle for k, v in replay.items()},
                               finite_losses=True, losses=dict(student=[.25], fake=[.5] * 5)))
        rows.append(dict(schema_version="distillschedule.h3_native_two_cycle.v1",
                         status="PASS_NATIVE_RUNTIME_CONTRACTS", rank=rank, integration_head=HEAD,
                         semantics="approximate_grid1000", cycles=cycles,
                         updates=dict(student=2, fake=10, ema=0),
                         final_versions=dict(generator=2, critic=10, ema=0),
                         native_wall_clock="HOST_MONOTONIC_CUDA_FENCED", native_boundary_fences=4,
                         formal_timing_eligible=False, calibration_only=False,
                         source_environment=p["runtime"]["environment"],
                         state_fingerprints=_state(rank), **binding))
    return rows


@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("block_n", [64, 128])
def test_exact_generated_roundtrip_and_lowering_without_canonical_mutation(depth, block_n):
    canonical = load_profile(H3)
    before = copy.deepcopy(canonical)
    impl = H3Implementation(depth, block_n)
    assert H3Implementation.parse(impl.id) == impl
    assert H3Implementation.from_assignment(impl.assignment) == impl
    assert current_implementation({"DISTILLSCHEDULE_SEARCH_CANDIDATE_ID": impl.id}) == impl
    p = impl.lower_profile()
    assert load_profile(H3) == before
    assert semantic_contract(p) == semantic_contract(canonical)
    assert p["memory"]["pinned_staging_depth"] == depth
    assert p["runtime"]["model_config"]["training"]["max_train_iters"] == 2
    assert p["runtime"]["model_config"]["training"]["a100"]["fa3_replay_cache"]["max_d2h_inflight"] == depth
    assert p["runtime"]["environment"]["H3_LORA_EPILOGUE_BLOCK_N"] == str(block_n)
    assert p["runtime"]["environment"]["H3_PLANNER_CALIBRATION"] == "0"
    assert not p["runtime"]["search_two_cycle_ready"]
    assert p["runtime"]["environment"]["H3_PLANNER_CLOCK"] == "host_fenced"


@pytest.mark.parametrize("key", ["joint-h3v1-d0-n128", "joint-h3v1-d2-n32", "joint-h3v1-d2-n128-extra",
                                 "joint-h3v1-d02-n128", "joint-h3v1-d2-n128\n", "joint-h3v2-d2-n128"])
def test_generated_ids_fail_closed(key):
    with pytest.raises(ValueError):
        H3Implementation.parse(key)


def test_typed_decisions_reject_bool_and_extra_fields():
    with pytest.raises(ValueError):
        H3Implementation(True, 128)
    with pytest.raises(ValueError):
        H3Implementation.from_assignment({**H3Implementation().assignment, "attention": "fa2"})
    assert current_implementation({}) is None


def test_launch_binding_validates_mode_and_all_frozen_config_not_only_two_decisions():
    impl = H3Implementation(1, 64)
    p = impl.lower_profile()["runtime"]
    first = impl.receipt_binding(p["model_config"], p["environment"])
    other = copy.deepcopy(p["model_config"])
    other["training"]["output_dir"] = "/different/run/h3"
    assert impl.receipt_binding(other, p["environment"]) == first
    other["training"]["teacher"]["guidance_scale"] = 2.
    with pytest.raises(ValueError, match="model config"):
        impl.receipt_binding(other, p["environment"])
    with pytest.raises(ValueError, match="H3_FA3_GRAD_NUM_SPLITS"):
        impl.receipt_binding(p["model_config"], {**p["environment"], "H3_FA3_GRAD_NUM_SPLITS": "1"})
    with pytest.raises(ValueError, match="calibration"):
        impl.receipt_binding(p["model_config"], p["environment"], mode="calibration")
    binding = impl.receipt_binding(p["model_config"], {**p["environment"], "H3_PLANNER_CALIBRATION": "1"}, mode="calibration")
    assert binding["candidate_id"] == impl.id


def test_binding_accepts_exact_pinned_lightx2v_loader_representation(tmp_path):
    # d034a6b runtime/config.py uses PyYAML -> OmegaConf and appends
    # config_path. YAML 1.1 reads JSON 1e-08 / 2e-05 as strings; the native
    # optimizer builder explicitly converts those same values with float().
    import yaml
    impl = H3Implementation(1, 128)
    runtime = impl.lower_profile()["runtime"]
    path = tmp_path / "h3-config-node0.json"
    path.write_text(json.dumps(runtime["model_config"]))
    loaded = yaml.safe_load(path.read_text())
    loaded["config_path"] = str(path.resolve())
    env = {**runtime["environment"], "H3_CONFIG": str(path)}
    assert isinstance(loaded["training"]["fake"]["optimizer"]["learning_rate"], str)
    assert impl.receipt_binding(loaded, env) == impl.receipt_binding(runtime["model_config"], env)
    assert loaded["config_path"] == str(path.resolve())  # no mutation
    assert isinstance(loaded["training"]["fake"]["optimizer"]["learning_rate"], str)


@pytest.mark.parametrize("mutation", ["config_path", "missing_path_env", "learning_rate", "epsilon", "unknown_key", "nonfinite"])
def test_loader_metadata_does_not_relax_native_config_contract(tmp_path, mutation):
    import yaml
    impl = H3Implementation(1, 128)
    runtime = impl.lower_profile()["runtime"]
    path = tmp_path / "h3-config.json"
    loaded = yaml.safe_load(json.dumps(runtime["model_config"]))
    loaded["config_path"] = str(path.resolve())
    env = {**runtime["environment"], "H3_CONFIG": str(path)}
    if mutation == "config_path":
        loaded["config_path"] = str(tmp_path / "unbound.json")
    elif mutation == "missing_path_env":
        env.pop("H3_CONFIG")
    elif mutation == "learning_rate":
        loaded["training"]["fake"]["optimizer"]["learning_rate"] = "2e-04"
    elif mutation == "epsilon":
        loaded["training"]["student"]["optimizer"]["adam_epsilon"] = "1e-06"
    elif mutation == "nonfinite":
        loaded["training"]["fake"]["optimizer"]["learning_rate"] = "nan"
    else:
        loaded["unknown_loader_key"] = True
    with pytest.raises(ValueError):
        impl.receipt_binding(loaded, env)


def test_summary_uses_rank_max_host_wall_not_cuda_or_cross_node_endpoints():
    rows = _receipts()
    result = summarize_native_two_cycle(rows, expected_head=HEAD, candidate_id=H3Implementation().id)
    assert result["status"] == "PASS"
    assert result["cycle_wall_seconds"] == {"1": 114., "2": 113.}
    assert result["cycle_cuda_elapsed_seconds"] == {"1": 104., "2": 103.}
    assert result["timer_scope"] == "RANK_MAX_NATIVE_HOST_FENCED_CYCLE_WALL"
    assert result["cross_rank_clock_alignment"] == "NOT_ESTABLISHED"
    assert not result["formal_timing_eligible"] and not result["search_ready"]


@pytest.mark.parametrize("mutation", ["duplicate_rank", "missing_cycle", "wrong_head", "wrong_candidate",
                                     "config_hash", "forward", "grad", "backward", "replay", "versions",
                                     "nonfinite_wall", "cuda_only", "sample_reuse", "missing_binding"])
def test_summary_rejects_partial_or_changed_native_receipts(mutation):
    rows = _receipts()
    row, cycle = rows[15], rows[15]["cycles"][1]
    if mutation == "duplicate_rank": row["rank"] = 14
    if mutation == "missing_cycle": row["cycles"].pop()
    if mutation == "wrong_head": row["integration_head"] = "b" * 40
    if mutation == "wrong_candidate": row["candidate_id"] = H3Implementation(1, 64).id
    if mutation == "config_hash": row["effective_configuration_sha256"] = "0" * 64
    if mutation == "forward": cycle["census"]["forward_counts"]["student"] = 23
    if mutation == "grad": cycle["census"]["grad_forward_counts"]["student"] = 2
    if mutation == "backward": cycle["census"]["backward_counts"]["fake"] = 4
    if mutation == "replay": cycle["grid_replay_delta"]["cache_miss_count"] = 1
    if mutation == "versions": row["final_versions"]["critic"] = 9
    if mutation == "nonfinite_wall": cycle["native_wall_seconds"] = float("nan")
    if mutation == "cuda_only": cycle.pop("native_wall_seconds")
    if mutation == "sample_reuse": cycle["census"]["samples"] = copy.deepcopy(rows[0]["cycles"][1]["census"]["samples"])
    if mutation == "missing_binding": row.pop("effective_configuration_sha256")
    with pytest.raises(ValueError):
        summarize_native_two_cycle(rows, expected_head=HEAD, candidate_id=H3Implementation().id)


def test_pair_metrics_are_same_rank_sampled_only_and_missing_rng_stays_not_run():
    ref, cand = _receipts(), _receipts(H3Implementation(1, 64))
    result = compare_native_two_cycle(ref, cand, expected_head=HEAD)
    assert result["sampled_state_status"] == "PASS"
    assert result["loss_status"] == result["sample_status"] == "PASS"
    assert result["rng_status"] == "NOT_RUN"
    assert result["strict_numerical_gate"] == "NOT_RUN"
    assert not result["formal_timing_eligible"] and not result["search_ready"]
    assert result["status"] == "NOT_RUN"
    metric = result["ranks"][0]["sampled_state"]["components"]["student_parameters"][0]["metrics"]
    assert {"mean_relative", "normalized_l2", "cosine", "max_abs"} <= set(metric)


def test_pair_missing_state_is_not_run_and_sparse_error_is_fail():
    ref, cand = _receipts(), _receipts(H3Implementation(1, 64))
    cand[0].pop("state_fingerprints")
    assert compare_native_two_cycle(ref, cand, expected_head=HEAD)["sampled_state_status"] == "NOT_RUN"
    cand = _receipts(H3Implementation(1, 64))
    state = cand[0]["state_fingerprints"]["components"]["student_parameters"]
    state["rows"][0]["values_fp32"][0] = 10.
    state["sample_sha256"] = hashlib.sha256(json.dumps(state["rows"], sort_keys=True, allow_nan=False).encode()).hexdigest()
    result = compare_native_two_cycle(ref, cand, expected_head=HEAD)
    assert result["status"] == result["sampled_state_status"] == "FAIL"


def test_launcher_syntax_and_opt_in_guard_precede_gpu_launch():
    path = ROOT / "scripts/distillschedule/launch_minimax_h3_best_node.sh"
    subprocess.run(["bash", "-n", str(path)], check=True)
    text = path.read_text()
    assert "--porcelain" in text
    assert "H3_PLANNER_CALIBRATION" in text
    assert "teardown_status.txt" in text
    assert text.index("flock -n 9") < text.index("--query-compute-apps")


def _with_rng(rows):
    for row in rows:
        states = [{k: hashlib.sha256(f"{row['rank']}:{i}:{k}".encode()).hexdigest()
                   for k in ("torch_cpu", "torch_cuda")} for i in range(3)]
        for i, cycle in enumerate(row["cycles"]):
            cycle.update(rng_before=states[i], rng_after=states[i + 1])
        digest = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
        row["rng_source_binding"] = dict(integration_head=HEAD, rank=row["rank"],
            initial_sha256=digest(states[0]), final_sha256=digest(states[2]),
            scope="NATIVE_RNG_STATE_BEFORE_AND_AFTER_TWO_CYCLES")
    return rows


@pytest.mark.parametrize("mutation", ["missing_boundary", "changed_middle", "changed_final", "wrong_hash"])
def test_rng_binding_requires_actual_all_cycle_boundaries(mutation):
    ref, cand = _with_rng(_receipts()), _with_rng(_receipts(H3Implementation(1, 64)))
    row = cand[0]
    if mutation == "missing_boundary": row["cycles"][0].pop("rng_before")
    if mutation == "changed_middle": row["cycles"][0]["rng_after"]["torch_cuda"] = "f" * 64
    if mutation == "changed_final": row["cycles"][1]["rng_after"]["torch_cpu"] = "f" * 64
    if mutation == "wrong_hash": row["rng_source_binding"]["final_sha256"] = "f" * 64
    result = compare_native_two_cycle(ref, cand, expected_head=HEAD)
    assert result["rng_status"] == ("NOT_RUN" if mutation == "missing_boundary" else "FAIL")
    assert result["status"] == ("NOT_RUN" if mutation == "missing_boundary" else "FAIL")


def test_complete_sampled_pair_is_review_never_strict_pass():
    from dataclasses import asdict
    from distillschedule.runtime.p0_parity import DEFAULT_P0_PARITY
    from distillschedule.search.h3_native_bridge import validate_comparison
    result = compare_native_two_cycle(_with_rng(_receipts()),
        _with_rng(_receipts(H3Implementation(1, 64))), expected_head=HEAD)
    validate_comparison(result)
    assert result["status"] == "REVIEW_SAMPLED_PARITY_ONLY"
    assert result["thresholds"] == asdict(DEFAULT_P0_PARITY)
    assert result["strict_numerical_gate"] == result["full_state_comparison"] == "NOT_RUN"


def _bridge_fixture(monkeypatch, tmp_path):
    from distillschedule.search import native
    from distillschedule.search.h3_native import read_native_receipts
    impl = H3Implementation(1, 64)
    rows = _with_rng(_receipts(impl))
    run = tmp_path / "timing"
    ranks = run / "execution/ranks"
    ranks.mkdir(parents=True)
    for row in rows:
        for cycle in row["cycles"]:
            for boundary in ("memory_before", "memory_after"):
                cycle[boundary] = {"hbm": dict(peak_allocated_bytes=10, peak_reserved_bytes=20)}
        (ranks / f"rank_{row['rank']:03d}.json").write_text(json.dumps(row))
    summary = summarize_native_two_cycle(rows, expected_head=HEAD, candidate_id=impl.id)
    summary["native_evidence"] = read_native_receipts(run)[1]
    (run / "bound-summary.json").write_text(json.dumps(summary))
    comparison = compare_native_two_cycle(_with_rng(_receipts()), rows, expected_head=HEAD)
    path = tmp_path / "comparison.json"
    path.write_text(json.dumps(comparison))
    preparation = dict(source_head=HEAD, candidate_id=impl.id, comparison=native.binding(path),
                       correctness=native.binding(run / "bound-summary.json"))
    receipt = dict(profile_id=H3, integration_head=HEAD, asset_fingerprint="assets",
                   workload_semantics_hash="semantics", mode="diagnostic-timing", run_id="timing",
                   search_candidate={"id": impl.id}, nodes=[],
                   asset_config={"master_addr": "node0", "master_port": 1234})
    for node in (0, 1):
        launch = run / "launcher" / f"node_{node}"
        launch.mkdir(parents=True)
        env = {**impl.lower_profile()["runtime"]["environment"], "H3_CONFIG": f"/config{node}.json",
               "DISTILLSCHEDULE_RUN_ID": "timing", "DISTILLSCHEDULE_RUN_ROOT": str(run),
               "DISTILLSCHEDULE_RUN_KIND": "diagnostic-timing", "EXPECTED_INTEGRATION_HEAD": HEAD}
        receipt["nodes"].append(dict(node_rank=node, python="/native/bin/python", environment=env))
        command = ["/native/bin/python", "-m", "torch.distributed.run", "--nnodes=2", "--nproc-per-node=8",
                   f"--node-rank={node}", "--master-addr=node0", "--master-port=1234", "--module",
                   "distillschedule.backends.training.minimax_h3.integrated_entrypoint", "--config", env["H3_CONFIG"]]
        manifest = dict(status="PASS_SOURCE_PREFLIGHT", integration_head=HEAD, source_clean=True,
            python_executable="/native/bin/python", mode="diagnostic-timing", command=command,
            planner_binding=impl.receipt_binding(impl.lower_profile()["runtime"]["model_config"], env),
            environment={k: v for k, v in env.items() if k != "EXPECTED_INTEGRATION_HEAD"})
        (launch / "launch_manifest.json").write_text(json.dumps(manifest))
        (launch / "teardown_status.txt").write_text("status=PASS\ngpu_process_count=0\nowner_run_id=timing\n")
        sample = dict(cgroup_events=dict(oom=0, oom_kill=0), cgroup_current_bytes=100,
                      gpus=[dict(device=i, free_bytes=20) for i in range(8)])
        (run / f"search-resources.node{node}.json").write_text(json.dumps(dict(
            status="PASS", node_rank=node, run_id="timing", samples=[sample, sample])))
    monkeypatch.setattr(native, "checked_summary", lambda receipt: (run, json.loads((run / "bound-summary.json").read_text())))
    identity = dict(source_head=HEAD, semantic_fingerprint="semantics", asset_fingerprint="assets", hardware_fingerprint="hardware")
    return run, receipt, preparation, identity


@pytest.mark.parametrize("mutation", ["missing_manifest", "profile_wrapper", "calibration", "raw_census", "raw_missing_rng"])
def test_bridge_rejects_missing_positive_profiler_proof_and_stale_summary(monkeypatch, tmp_path, mutation):
    from distillschedule.search.h3_native_bridge import normalize_native
    run, receipt, preparation, identity = _bridge_fixture(monkeypatch, tmp_path)
    path = run / "launcher/node_1/launch_manifest.json"
    manifest = json.loads(path.read_text())
    if mutation == "missing_manifest": path.unlink()
    if mutation == "profile_wrapper":
        manifest["command"][:0] = ["nsys", "profile"]
        path.write_text(json.dumps(manifest))
    if mutation == "calibration":
        manifest["environment"]["H3_PLANNER_CALIBRATION"] = "1"
        path.write_text(json.dumps(manifest))
    if mutation.startswith("raw_"):
        path = run / "execution/ranks/rank_015.json"
        row = json.loads(path.read_text())
        if mutation == "raw_census": row["cycles"][1]["census"]["forward_counts"]["student"] = 23
        else: row.pop("rng_source_binding")
        path.write_text(json.dumps(row))
    with pytest.raises((ValueError, FileNotFoundError)):
        normalize_native(receipt, preparation, identity)


def test_bridge_complete_native_diagnostic_stays_review(monkeypatch, tmp_path):
    from distillschedule.search.h3_native_bridge import normalize_native
    run, receipt, preparation, identity = _bridge_fixture(monkeypatch, tmp_path)
    result = normalize_native(receipt, preparation, identity)
    assert result["numerical_status"] == "REVIEW"
    assert result["profiler_status"] == "DISABLED"
    assert result["formal_timing_eligible"] is False
    assert any(e["path"].endswith("launch_manifest.json") for e in result["native_evidence"])
