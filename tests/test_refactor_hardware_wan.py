"""Synthetic CPU receipts only: no historical time becomes native calibration."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import shlex

import pytest

from distillschedule.search import hardware_wan as wan
from distillschedule.runtime.p0_parity import DEFAULT_P0_PARITY
from distillschedule.search.allocation import ExecutionAllocation, PhysicalDevice
from distillschedule.search.cost_model import CostModel
from distillschedule.search.model_search import search_plans
from distillschedule.workloads.best_known.registry import fingerprint, file_sha256


def fixture_deployment(tmp_path):
    a = ExecutionAllocation(tuple(PhysicalDevice(r, 0, r, f"GPU-fixture-{r}", "sm90", 80*1024**3)
                                  for r in range(8)), 16,
        {role: tuple((r, r+8) for r in range(8)) for role in ("generator", "critic", "teacher")})
    d = {"schema_version": "distillschedule.hardware_wan_deployment.v1",
        "native_source": {"head": wan.NATIVE_HEAD, "repo_root": str(tmp_path/"native"), **wan.NATIVE_FILES},
        "allocation": a.to_dict(), "environment": {"python": "3.10.20", "torch": "2.10.0+cu128",
            "ld_library_path": "", "driver_version": "535.183.06"},
        "assets": {k: wan.SEMANTICS[k] for k in ("bootstrap_hash", "trace_bundle_hash", "model_identity_sha256",
                                                "negative_prompt_manifest_hash")},
        "semantics": deepcopy(wan.SEMANTICS), "precision": deepcopy(wan.PRECISION), "evidence_scope": "CPU_FIXTURE_ONLY",
        "site": {"project_root": str(tmp_path), "python": str(tmp_path/"bin/python"),
            "helios_root": str(tmp_path/"frozen-helios"), "asset_root": str(tmp_path/"assets"),
            "run_root": str(tmp_path/"research_artifacts/five_workload_h100_20260930")},
        "transport": {"host": "fixture.invalid", "ssh_port": 22, "address_family": "ipv4"}}
    d["binding"] = {"source_head": "a"*40, "hardware_fingerprint": a.digest,
        **{f"{prefix}_fingerprint": fingerprint(d[field]) for prefix, field in
           (("environment", "environment"), ("asset", "assets"), ("semantic", "semantics"), ("precision", "precision"))}}
    return d


def _trace():
    result = []
    for i, counts in enumerate(((2, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 1), (2, 1, 1), (2, 2, 2))):
        stages = {str(j): c for j, c in enumerate(counts)}
        inventory = {"call_count": sum(counts), "dummy_call_count": 0, "logical_call_count": sum(counts),
                     "logical_stage_call_count": stages, "stage_call_count": stages}
        operation = "step/10/generator/rollout" if i == 0 else f"step/{i+9}/critic/rollout"
        result.append({"inventory": inventory, "operation": operation})
    assert fingerprint(result) == wan.SEMANTICS["trace_rollout_contract_sha256"]
    return result


def fixture_summary(placement, run_id, *, mode="timing", seconds=(12., 10.), correctness=None, comparison=None):
    arm = wan._arm(placement)
    impl = {**arm, "hardware_profile": "h100-8-b16", "microbatch_count": 2, "trajectory_arm_id": f"h100-{placement}"}
    health = {"status": "PASS", "exit_code": 0, "expected_head": wan.NATIVE_HEAD,
        "cgroup_oom_delta": {"oom": 0, "oom_kill": 0}, "cgroup_before": {"oom": 0, "oom_kill": 0},
        "cgroup_after": {"oom": 0, "oom_kill": 0}, "gpu_processes_after": []}
    rows = []
    for rank in range(8):
        fused = {"generator": {"requested": False, "enabled": False, "fused_invocation_count": 0},
                 "critic": {"enabled": False}}
        if placement == wan.PLACEMENTS[2]:
            fused = {"generator": {"requested": True, "enabled": True, "model_call_count": 112,
                "fused_invocation_count": 34160, "reference_fallback_count": 448,
                "reference_fallback_reasons": {"BASE_DTYPE_torch.float32": 448}},
                "critic": {"enabled": False, "disabled_reason": "GENERATOR_ONLY"}}
        cycles, census = [], []
        start = 1_000_000_000
        for cycle, duration in enumerate(seconds, 1):
            cycles.append({"cycle": cycle, "role": wan._CYCLE_CONTRACT[f"cycle_{cycle}_role"],
                "local_elapsed_seconds": duration, "start_unix_ns": start,
                "stop_unix_ns": start + round(duration*1e9)})
            start += round(duration*1e9) + 1000
            samples = []
            for slot in (rank, rank+8):
                ids = [fingerprint(["SYNTHETIC_CPU_FIXTURE", slot, i]) for i in range(6)]
                samples.append({"generator_updates": 1, "critic_updates": 5, "ema_updates": 5,
                    "generator_score_forwards": 3, "critic_score_forwards": 5, "physical_rollout_batches": 5,
                    "rollout_batch_size": 1, "generator_trace_payload_identity": ids[0],
                    "critic_trace_payload_identities": ids[1:], "sample_payload_identities": ids,
                    "sample_occurrence_identities": [f"cycle={cycle}/slot={i}/payload={p}" for i, p in enumerate(ids)],
                    "rollout_operation_keys": [x["operation"] for x in _trace()],
                    "rollout_call_inventory": [x["inventory"] for x in _trace()]})
            census.append({"generator_updates": 1, "critic_updates": 5, "ema_updates": 5,
                "physical_rollout_batches": 10, "rollout_batch_size": 1, "logical_slots": [rank, rank+8],
                "logical_sample_census": samples})
        row = {"schema_version": "distillschedule.wan13b_two_cycle_timing_rank.v1", "status": "PASS",
            "rank": rank, "local_rank": rank, "world_size": 8, "logical_world_size": 16,
            "hardware_profile": "h100-8-b16", "logical_slots": [rank, rank+8], "microbatch_count": 2,
            "run_id": run_id, "mode": mode, "placement": placement, "expected_head": wan.NATIVE_HEAD,
            "profiled": False, "profiler_status": "NOT_RUN", "finite_status": [True, True],
            "cycle_contract": deepcopy(wan._CYCLE_CONTRACT), "initial_versions": {"generator": 2, "critic": 10, "ema": 10},
            "cycle_versions": [{"generator": 3, "critic": 15, "ema": 15}, {"generator": 4, "critic": 20, "ema": 20}],
            "final_versions": {"generator": 4, "critic": 20, "ema": 20}, "updates": {"generator": 2, "critic": 10, "ema": 10},
            "matched_contract": deepcopy(wan.SEMANTICS), "workload_fingerprint": fingerprint(wan.SEMANTICS),
            "arm_contract": arm, "implementation_fingerprint": fingerprint(arm), "runtime_contract": wan._runtime(placement),
            "trace_rollout_contract_sha256": wan.SEMANTICS["trace_rollout_contract_sha256"],
            "trace_rollout_contract": _trace(), "cycle_census": census, "cycles": cycles,
            "formal_timing_eligible": mode == "timing", "timing_seed": wan.SEMANTICS["seed"],
            "rank_runtime_seed": wan.SEMANTICS["seed"]+rank, "timing_arm_id": f"h100-{placement}",
            "hostname": "fixture.invalid", "correctness_binding": correctness or {"status": "NOT_RUN"},
            "trajectory_comparison_binding": comparison or {"status": "NOT_RUN"},
            "configuration_fingerprint": fingerprint({"workload_fingerprint": fingerprint(wan.SEMANTICS),
                                                       "implementation_fingerprint": fingerprint(arm)}),
            "model_relocation": {"content_validation": "SHA256_PASS" if rank == 0 else "SIZE_PASS_HASH_VERIFIED_BY_RANK0",
                                 "model_identity_sha256": wan.SEMANTICS["model_identity_sha256"]},
            "memory": {"max_allocated_bytes": 8*1024**3, "max_reserved_bytes": 10*1024**3, "driver_total_bytes": 80*1024**3},
            "fused_epilogue_receipt": fused}
        rows.append(row)
    return {"schema_version": "distillschedule.wan13b_trajectory_summary.v1", "status": "PASS", "errors": [],
        "run_id": run_id, "mode": mode, "placement": placement, "expected_head": wan.NATIVE_HEAD,
        "cycles": 2, "rank_count": 8, "logical_world_size": 16, "hardware_profile": "h100-8-b16",
        "final_versions": {"generator": 4, "critic": 20, "ema": 20}, "matched_contract": deepcopy(wan.SEMANTICS),
        "workload_fingerprint": fingerprint(wan.SEMANTICS), "implementation_contract": impl,
        "implementation_fingerprint": fingerprint(impl), "diagnostic_only": mode != "timing",
        "formal_timing_eligible": mode == "timing", "launcher_health": health, "logical_slots": list(range(16)),
        "ranks": rows, "cycle_seconds_max_rank": list(seconds), "evidence_scope": "CPU_FIXTURE_ONLY"}


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value if isinstance(value, str) else json.dumps(value, sort_keys=True))
    return {"path": str(path), "sha256": file_sha256(path)}


def fixture_comparison(*, review=True):
    metric = {"finite": True, "same_shape": True, "same_dtype": True, "status": "PASS",
        "reference_shape": [1], "candidate_shape": [1], "reference_dtype": "torch.float32", "candidate_dtype": "torch.float32",
        "max_abs": 0.0, "max_abs_limit": 0.06, "mean_abs": 0.0, "mean_relative": 0.0, "normalized_l2": 0.0,
        "cosine": 1.0, "cosine_policy": "standard", "reference_peak_abs": 1.0, "thresholds": asdict(DEFAULT_P0_PARITY)}
    rows = [{"kind": "rollout", "cycle": c, "logical_slot": s, "step": step, **deepcopy(metric)}
            for c in (1, 2) for s in range(16) for step in range(10, 15)]
    rows += [{"kind": "sampled_state", "component": name, **deepcopy(metric)} for name in sorted(wan._STATE_COMPONENTS)]
    if review:
        rows[0].update(status="FAIL", normalized_l2=0.02)
    return {"schema_version": "distillschedule.trajectory_coarse_comparison.v1",
        "status": "HARDWARE_TWO_CYCLE_REVIEW" if review else "HARDWARE_TWO_CYCLE_PASS",
        "numerical_status": "REVIEW" if review else "PASS_OBSERVED_ROLLOUTS_AND_SAMPLED_STATE",
        "comparison_kind": "semantic", "expected_cycles": 2, "expected_world_size": 8, "hardware_profile": "h100-8-b16",
        "contract": {"status": "PASS", "mismatches": []}, "expected_head": wan.NATIVE_HEAD, "complete": True,
        "diagnostic_timing_eligible": True, "strict_full_state_parity": "NOT_RUN", "strict_claimable_for_paired_timing": False,
        "numeric_metrics": rows, "evidence_scope": "CPU_FIXTURE_ONLY"}


def fixture_correctness(d, placement, run_id):
    root = Path(d["site"]["run_root"])/run_id
    summary = fixture_summary(placement, run_id, mode="correctness")
    sketch = {"inventory_sha256": fingerprint(["CPU_FIXTURE_TENSOR"]), "inventory_tensor_count": 1,
        "tensor_count": 1, "coordinate_count": 1,
        "coordinates": [{"tensor": "CPU_FIXTURE_TENSOR", "flat_index": 0, "value_fp32": 0.0}]}
    anchor = {"schema_version": "distillschedule.wan13b_trajectory_state_anchor.v2", "cycle": 2,
        "writer_rank": 0, "placement": placement, "evidence_scope": "CPU_FIXTURE_ONLY",
        "components": {name: {"numeric_sketch": deepcopy(sketch)} for name in wan._STATE_COMPONENTS}}
    anchor["anchor_sha256"] = fingerprint(anchor)
    path = root/"execution/trajectory/anchors/cycle_02_global.json"
    _write(path, anchor)
    anchor_ref = {"path": str(path), "sha256": anchor["anchor_sha256"]}
    for row in summary["ranks"]:
        row["rollout_evidence"] = [{"cycle": c, "logical_slot": slot,
            **_write(root/f"execution/rollouts/cycle{c}_slot{slot:02d}.pt", f"CPU_FIXTURE_ONLY:{c}:{slot}")}
            for c in (1, 2) for slot in row["logical_slots"]]
        row["state_anchors"] = anchor_ref
        row["build"] = {role: {"lora_compute_dtype_contract": {"records": [{"base_weight_dtype": "torch.bfloat16",
            "lora_a_weight_dtype": "torch.float32", "lora_b_weight_dtype": "torch.float32"}]}}
            for role in ("generator", "critic")}
    return summary, _write(root/"summary.json", summary)


def fixture_observation(d, placement, run_id, *, seconds=(12., 10.)):
    root = Path(d["site"]["run_root"])/run_id
    c, cref = fixture_correctness(d, placement, run_id+"-correctness")
    other = "resident" if placement == wan.PLACEMENTS[0] else wan.PLACEMENTS[0]
    reference, rref = fixture_correctness(d, other, run_id+"-reference")
    cb = {**cref, "status": "PASS", "schema_version": c["schema_version"], "expected_head": wan.NATIVE_HEAD,
          "run_id": c["run_id"], "matched_contract": c["matched_contract"], "implementation_contract": c["implementation_contract"]}
    comp = {**fixture_comparison(),
        "candidate": {"run_ids": [c["run_id"]], "variants": [f"h100-{placement}"]},
        "reference": {"run_ids": [reference["run_id"]], "variants": [f"h100-{other}"]},
        "summary_sha256": {"candidate": cref["sha256"], "reference": rref["sha256"]}}
    pref = _write(root.parent/(run_id+"-comparison")/"comparison.json", comp)
    pb = {**pref, "contract_status": "PASS", "complete": True, "status": comp["status"],
          "matched_correctness_run_id": c["run_id"], "matched_side": "candidate"}
    s = fixture_summary(placement, run_id, seconds=seconds, correctness=cb, comparison=pb)
    obs = {"placement": placement, "run_root": str(root), "summary": _write(root/"summary.json", s),
        "rank_receipts": [_write(root/f"execution/ranks/rank_{r['rank']:02d}.json", r) for r in s["ranks"]],
        "launcher_health": _write(root/"run_status.json", s["launcher_health"]), "correctness": cref, "comparison": pref}
    argv, env = wan._worker_argv(d, root, run_id, placement, cref, pref), wan._environment(d, placement)
    obs["launcher_argv"] = _write(root/"manifests/command.sh", shlex.join(argv)+"\n")
    obs["launcher_environment"] = _write(root/"manifests/environment.txt", "".join(f"{k}={v}\n" for k, v in sorted(env.items())))
    launch = {"schema_version": "distillschedule.hardware_wan_launch.v1", "deployment_sha256": fingerprint(d),
        "integration_head": d["binding"]["source_head"], "native_head": wan.NATIVE_HEAD,
        "integration_clean": True, "native_clean": True, "allocation_sha256": fingerprint(d["allocation"]),
        "placement": placement, "run_id": run_id, "argv_sha256": obs["launcher_argv"]["sha256"],
        "environment_sha256": obs["launcher_environment"]["sha256"], "native_launcher_sha256": wan.NATIVE_FILES["launcher_sha256"],
        "native_worker_sha256": wan.NATIVE_FILES["worker_sha256"], "observed_devices": d["allocation"]["devices"],
        "started_unix_ns": 1, "completed_unix_ns": 1+max(r["cycles"][-1]["stop_unix_ns"] for r in s["ranks"]),
        "observed_hostname": "fixture.invalid", "observed_environment": d["environment"],
        "native_worker_argv": argv, "native_worker_environment": env}
    obs["launch_binding"] = _write(root/"manifests/hardware_wan_launch.json", launch)
    return obs


@pytest.fixture
def setup(tmp_path):
    d = fixture_deployment(tmp_path)
    obs = [fixture_observation(d, p, f"fixture-{i}", seconds=(14.+i, 12.-i)) for i, p in enumerate(wan.PLACEMENTS)]
    return d, obs


def test_common_planner_roundtrip_and_coupled_costs(setup, tmp_path):
    d, obs = setup
    problem, costs = wan.assemble_problem_and_costs(d, obs, output_dir=tmp_path/"costs")
    assert len(problem.graph.operations) == 2 and len(costs.records) == 6
    assert not problem.states and not problem.lifetimes
    result = search_plans(problem, costs, algorithm="hierarchical")
    assert result["predicted_best"]["assignment"] == {"placement": wan.PLACEMENTS[2]}
    assert result["predicted_best"]["predicted_cycle2_ns"] == 10_000_000_000
    plan = wan.lower_selection(result, problem, costs, d, run_id="fresh-selected")
    assert wan.validate_lowered_plan(json.loads(json.dumps(plan))) == plan
    assert plan["executable"] is False and plan["claimable_speedup"] is False
    assert plan["numerical_status"] == "REVIEW" and plan["strict_full_state_parity"] == "NOT_RUN"


def test_uncalibrated_choice_has_no_winner_fallback(setup, tmp_path):
    d, obs = setup
    problem, costs = wan.assemble_problem_and_costs(d, obs[:2], output_dir=tmp_path/"costs")
    result = search_plans(problem, costs, exhaustive=True)
    assert result["predicted_best"]["assignment"] == {"placement": "resident"}
    assert any(r["status"] == "UNCALIBRATED" for r in result["rejections"])


@pytest.mark.parametrize("mutation", [
    lambda s: s["ranks"].pop(),
    lambda s: s["ranks"][1].update(rank=0),
    lambda s: s["ranks"][1].update(logical_slots=[0, 8]),
    lambda s: s["ranks"][0].update(logical_slots=[False, 8]),
    lambda s: s.update(logical_slots=[float(i) for i in range(16)]),
    lambda s: s["ranks"][0].update(finite_status=[True, False]),
    lambda s: s["ranks"][0]["cycle_versions"][1].update(ema=19),
    lambda s: s["ranks"][0]["cycle_census"][0].update(critic_updates=4),
    lambda s: s["launcher_health"].update(gpu_processes_after=["pid 42"]),
    lambda s: s["launcher_health"]["cgroup_after"].update(oom=1),
    lambda s: s["ranks"][0]["fused_epilogue_receipt"]["generator"].update(fused_invocation_count=0),
    lambda s: s["ranks"][0]["runtime_contract"].update(nccl_algo="AUTO"),
    lambda s: s.update(cycle_seconds_max_rank=[1., 1.]),
    lambda s: s["ranks"][0]["cycles"][1].update(local_elapsed_seconds=float("nan")),
])
def test_native_rank_fail_closed(setup, mutation):
    d, obs = setup
    s = json.loads(Path(obs[2]["summary"]["path"]).read_text())
    mutation(s)
    with pytest.raises(ValueError):
        wan.validate_native_summary(s, expected_native_head=wan.NATIVE_HEAD,
                                    placement=wan.PLACEMENTS[2], allocation=d["allocation"])


def test_duplicated_sample_content_rejected(setup):
    d, obs = setup
    s = json.loads(Path(obs[0]["summary"]["path"]).read_text())
    for c in (0, 1):
        s["ranks"][1]["cycle_census"][c]["logical_sample_census"] = deepcopy(s["ranks"][0]["cycle_census"][c]["logical_sample_census"])
    with pytest.raises(ValueError, match="duplicated"):
        wan.validate_native_summary(s, expected_native_head=wan.NATIVE_HEAD,
                                    placement=wan.PLACEMENTS[0], allocation=d["allocation"])


@pytest.mark.parametrize("key", ["head", "launcher_sha256", "worker_sha256"])
def test_native_source_registry_not_user_asserted(setup, key):
    d, _ = setup
    d["native_source"][key] = "f" * len(d["native_source"][key])
    with pytest.raises(ValueError, match="registered"):
        wan.validate_deployment(d)


@pytest.mark.parametrize("field", ["hardware_fingerprint", "environment_fingerprint", "asset_fingerprint", "semantic_fingerprint", "precision_fingerprint"])
def test_deployment_identity_binding(setup, field):
    d, _ = setup
    d["binding"][field] = "e"*64
    with pytest.raises(ValueError, match="identity mismatch"):
        wan.validate_deployment(d)


def test_allocation_and_semantics_not_generalized(setup):
    d, _ = setup
    altered = deepcopy(d)
    altered["allocation"]["logical_global_batch"] = 8
    with pytest.raises(ValueError):
        wan.validate_deployment(altered)
    altered = deepcopy(d)
    altered["semantics"]["updates"] = "G1_C1_EMA1"
    with pytest.raises(ValueError, match="matched workload"):
        wan.validate_deployment(altered)
    d["workload"] = "minimax_h3_grid1000"
    with pytest.raises(ValueError, match="NOT_ADMITTED"):
        wan.validate_deployment(d)


def test_old_head_is_not_new_calibration(setup, tmp_path):
    d, obs = setup
    s = json.loads(Path(obs[0]["summary"]["path"]).read_text())
    s["expected_head"] = "1"*40
    obs[0]["summary"] = _write(Path(obs[0]["summary"]["path"]), s)
    with pytest.raises(ValueError, match="summary identity"):
        wan.assemble_problem_and_costs(d, obs, output_dir=tmp_path/"costs")


@pytest.mark.parametrize("kind", ["argv", "environment", "source", "interval", "rank"])
def test_raw_evidence_revalidated_not_just_wrapper(setup, tmp_path, kind):
    d, obs = setup
    _, costs = wan.assemble_problem_and_costs(d, obs, output_dir=tmp_path/"costs")
    raw = json.loads(Path(costs.records[0].evidence["source_receipt"]).read_text())
    o = raw["observations"][0]
    if kind == "argv":
        p = Path(o["launcher_argv"]["path"])
        o["launcher_argv"] = _write(p, p.read_text().replace("--nproc-per-node=8", "--nproc-per-node=16"))
    elif kind == "environment":
        p = Path(o["launcher_environment"]["path"])
        o["launcher_environment"] = _write(p, p.read_text()+"NCCL_ALGO=Tree\n")
    elif kind in ("source", "interval"):
        p = Path(o["launch_binding"]["path"])
        launch = json.loads(p.read_text())
        launch.update(integration_head="f"*40) if kind == "source" else launch.update(started_unix_ns=2_000_000_000)
        o["launch_binding"] = _write(p, launch)
    else:
        p = Path(o["rank_receipts"][0]["path"])
        rank = json.loads(p.read_text()); rank["finite_status"] = [False, False]
        o["rank_receipts"][0] = _write(p, rank)
    with pytest.raises(ValueError):
        wan.validate_component_receipt(raw)


def test_component_cannot_change_samples_or_memory(setup, tmp_path):
    d, obs = setup
    _, costs = wan.assemble_problem_and_costs(d, obs, output_dir=tmp_path/"costs")
    raw = json.loads(Path(costs.records[0].evidence["source_receipt"]).read_text())
    wan.validate_component_receipt(raw)
    for key, value in (("samples_ns", [1]), ("workspace_bytes_per_rank", 1)):
        corrupt = {**raw, key: value}
        with pytest.raises(ValueError, match="samples/memory"):
            wan.validate_component_receipt(corrupt)


def test_cpu_fixture_relabel_to_measured_rejected(setup, tmp_path):
    d, obs = setup
    measured = deepcopy(d); measured["evidence_scope"] = "MEASURED"
    # Rehashing all wrapper identities still cannot upgrade explicitly marked raw fixtures.
    for o in obs:
        launch = json.loads(Path(o["launch_binding"]["path"]).read_text())
        launch["deployment_sha256"] = fingerprint(measured)
        o["launch_binding"] = _write(Path(o["launch_binding"]["path"]), launch)
    with pytest.raises(ValueError, match="CPU fixture"):
        wan.assemble_problem_and_costs(measured, obs, output_dir=tmp_path/"costs")


def test_cost_model_scope_upgrade_rejected(setup, tmp_path):
    d, obs = setup
    _, costs = wan.assemble_problem_and_costs(d, obs, output_dir=tmp_path/"costs")
    raw = costs.to_dict(); raw["scope"] = "MEASURED"
    with pytest.raises(ValueError):
        CostModel.from_dict(raw).verify_receipts()


def test_lowered_plan_tamper_even_after_rehash(setup, tmp_path):
    d, obs = setup
    problem, costs = wan.assemble_problem_and_costs(d, obs, output_dir=tmp_path/"costs")
    result = search_plans(problem, costs, exhaustive=True)
    plan = wan.lower_selection(result, problem, costs, d, run_id="fresh-selected")
    plan["selected_placement"] = wan.PLACEMENTS[0]
    plan["plan_sha256"] = fingerprint({k: v for k, v in plan.items() if k != "plan_sha256"})
    with pytest.raises(ValueError, match="saved placement"):
        wan.validate_lowered_plan(plan)


def test_dispatch_search_uses_real_common_bridge(setup, tmp_path):
    from distillschedule.search.hardware_wan_dispatch import search, run_selected
    d, obs = setup
    result = search(d, obs, output_dir=tmp_path/"dispatch", run_id="dispatch-selected")
    assert result["selected_placement"] == wan.PLACEMENTS[2] and result["gpu_execution"] == "NOT_RUN"
    plan = json.loads(Path(result["plan"]).read_text())
    with pytest.raises(ValueError, match="CPU fixtures"):
        run_selected(plan, output_dir=tmp_path/"must-not-launch")
    assert not (tmp_path/"must-not-launch").exists()


@pytest.mark.parametrize("field", ["observed_hostname", "observed_environment", "native_worker_argv", "native_worker_environment"])
def test_measured_sidecar_cannot_drop_execution_proof(setup, field):
    d, obs = setup
    d["evidence_scope"] = "MEASURED"
    o = obs[0]
    launch = json.loads(Path(o["launch_binding"]["path"]).read_text())
    launch["deployment_sha256"] = fingerprint(d)
    del launch[field]
    o["launch_binding"] = _write(Path(o["launch_binding"]["path"]), launch)
    # Validation fails on the dropped proof before the independent fixture gate.
    with pytest.raises(ValueError, match="missing actual host"):
        wan._validate_observation(d, o)


def test_comparison_must_hash_the_actual_correctness_summary(setup):
    d, obs = setup
    o = obs[0]
    comp = json.loads(Path(o["comparison"]["path"]).read_text())
    comp["summary_sha256"]["candidate"] = "f"*64
    ref = _write(Path(o["comparison"]["path"]), comp)
    summary = json.loads(Path(o["summary"]["path"]).read_text())
    for i, rank in enumerate(summary["ranks"]):
        rank["trajectory_comparison_binding"].update(ref)
        o["rank_receipts"][i] = _write(Path(o["rank_receipts"][i]["path"]), rank)
    o["summary"] = _write(Path(o["summary"]["path"]), summary)
    o["comparison"] = ref
    with pytest.raises(ValueError, match="correctness summary"):
        wan._validate_observation(d, o)


def test_flat_pool_lowering_and_bottom_source_tamper(setup, tmp_path, monkeypatch):
    """Exercise the real flat-pool validator with a fixture-only source reader.

    Public pooling intentionally requires MEASURED. This narrow reader shim
    keeps every raw/source hash check but never relabels synthetic CPU costs.
    """
    from distillschedule.search import component_calibration as pooling
    from distillschedule.search.cost_model import OperationCost
    d, obs = setup
    one = obs[2]
    two = fixture_observation(d, wan.PLACEMENTS[2], "fixture-repeat", seconds=(17., 8.))
    problem, m1 = wan.assemble_problem_and_costs(d, [one], output_dir=tmp_path/"one")
    _, m2 = wan.assemble_problem_and_costs(d, [two], output_dir=tmp_path/"two")

    def fixture_source(path, sha, binding):
        raw = wan._read_ref({"path": str(path), "sha256": sha})
        wan.validate_component_receipt(raw)
        assert raw["binding"] == binding and raw["evidence_scope"] == "CPU_FIXTURE_ONLY"
        return OperationCost(raw["key"], tuple(raw["samples_ns"]), raw["workspace_bytes_per_rank"],
                             {"component_contract": raw["component_contract"]})

    monkeypatch.setattr(pooling, "_flat_source", fixture_source)
    records = []
    for r1, r2 in zip(m1.records, m2.records):
        sources = [{"path": r.evidence["source_receipt"], "sha256": r.evidence["source_receipt_sha256"]} for r in (r1, r2)]
        raw = {"schema_version": "distillschedule.operation_calibration.v1", "binding": d["binding"],
            "key": r1.key, "status": "PASS", "profiled": False, "kind": "native_operation",
            "operation_contract_sha256": r1.key, "component_contract": r1.evidence["component_contract"],
            "samples_ns": [*r1.samples_ns, *r2.samples_ns], "workspace_bytes_per_rank": r1.workspace_bytes_per_rank,
            "calibration_only": True, "formal_timing_eligible": False,
            "aggregation": "POOLED_NATIVE_COMPONENT_SAMPLES", "component_sources": sources}
        ref = _write(tmp_path/"pooled"/f"{r1.key}.json", raw)
        records.append(OperationCost(r1.key, tuple(raw["samples_ns"]), raw["workspace_bytes_per_rank"],
            {**r1.evidence, "source_receipt": ref["path"], "source_receipt_sha256": ref["sha256"]}))
    pooled = CostModel(d["binding"], tuple(records), "CPU_FIXTURE_ONLY")
    result = search_plans(problem, pooled, exhaustive=True)
    plan = wan.lower_selection(result, problem, pooled, d, run_id="pooled-selected")
    assert len(plan["selected_observations"]) == 2 and plan["executable"] is False
    assert wan.validate_lowered_plan(plan) == plan
    Path(one["rank_receipts"][0]["path"]).write_text("{}")
    with pytest.raises(ValueError, match="raw evidence hash changed"):
        wan.validate_lowered_plan(plan)


@pytest.mark.parametrize("review", [False, True])
def test_complete_comparison_preserves_original_finite_review_semantics(review):
    value = fixture_comparison(review=review)
    # Observed latent FP32 is legal; do not invent a BF16-only rollout gate.
    wan.validate_native_comparison(value)
    assert value["strict_full_state_parity"] == "NOT_RUN"


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "missing_state", "finite", "same_shape", "same_dtype",
    "nan", "numerical_status", "top_status", "metric_status", "threshold", "shape_values", "dtype_values", "max_abs_limit"])
def test_native_comparison_incomplete_or_contradictory_metrics_fail(mutation):
    value = fixture_comparison()
    rows = value["numeric_metrics"]
    if mutation == "missing": rows.pop(0)
    elif mutation == "duplicate": rows[1] = deepcopy(rows[0])
    elif mutation == "missing_state": rows.pop()
    elif mutation in ("finite", "same_shape", "same_dtype"): rows[0][mutation] = False
    elif mutation == "nan": rows[0]["normalized_l2"] = float("nan")
    elif mutation == "numerical_status": value["numerical_status"] = "FAIL"
    elif mutation == "top_status": value["status"] = "HARDWARE_TWO_CYCLE_PASS"
    elif mutation == "metric_status": rows[0]["status"] = "PASS"
    elif mutation == "threshold": rows[0]["thresholds"]["normalized_l2"] = 1.0
    elif mutation == "shape_values": rows[0]["candidate_shape"] = [2]
    elif mutation == "dtype_values": rows[0]["candidate_dtype"] = "torch.bfloat16"
    else: rows[0]["max_abs_limit"] = 1.0
    with pytest.raises(ValueError):
        wan.validate_native_comparison(value)


@pytest.mark.parametrize("scope_path", ["top", "nested_metric"])
def test_measured_comparison_rejects_recursive_fixture_scope(scope_path):
    value = fixture_comparison()
    if scope_path == "nested_metric":
        value.pop("evidence_scope")
        value["numeric_metrics"][0]["fixture_scope"] = "CPU_FIXTURE_ONLY"
    with pytest.raises(ValueError, match="CPU fixture"):
        wan.validate_native_comparison(value, evidence_scope="MEASURED")


@pytest.mark.parametrize("side", ["reference", "candidate"])
@pytest.mark.parametrize("damage", ["rollout_delete", "rollout_corrupt", "anchor_delete", "anchor_corrupt"])
def test_saved_plan_reopens_both_correctness_artifact_chains(setup, tmp_path, side, damage):
    d, observations = setup
    observation = observations[2]
    problem, costs = wan.assemble_problem_and_costs(d, [observation], output_dir=tmp_path/"costs")
    result = search_plans(problem, costs, exhaustive=True)
    plan = wan.lower_selection(result, problem, costs, d, run_id="fresh-selected")
    comparison = wan._read_ref(observation["comparison"])
    run_id = comparison[side]["run_ids"][0]
    summary = json.loads((Path(d["site"]["run_root"])/run_id/"summary.json").read_text())
    rank = summary["ranks"][0]
    spec = rank["rollout_evidence"][0] if damage.startswith("rollout") else rank["state_anchors"]
    path = Path(spec["path"])
    if damage.endswith("delete"):
        path.unlink()
    elif damage == "rollout_corrupt":
        path.write_bytes(b"CORRUPTED_CPU_FIXTURE")
    else:
        anchor = json.loads(path.read_text())
        anchor["cycle"] = 1
        path.write_text(json.dumps(anchor))
    with pytest.raises(ValueError, match="artifact|anchor"):
        wan.validate_lowered_plan(plan)


def test_correctness_anchors_use_canonical_digest_and_rollouts_streaming_hash(setup, monkeypatch):
    d, observations = setup
    observation = observations[0]
    summary = wan._read_ref(observation["correctness"])
    spec = summary["ranks"][0]["state_anchors"]
    path = Path(spec["path"])
    assert file_sha256(path) != spec["sha256"]
    path.write_text(json.dumps(json.loads(path.read_text()), indent=4)+"\n\n")
    read_bytes = Path.read_bytes
    def no_tensor_read_bytes(path):
        assert path.suffix != ".pt", "rollout tensors must be streamed, never read into one planner byte string"
        return read_bytes(path)
    monkeypatch.setattr(Path, "read_bytes", no_tensor_read_bytes)
    wan.validate_native_comparison_inputs(d, wan._read_ref(observation["comparison"]))


@pytest.mark.parametrize("mutation", ["duplicate_rollout", "missing_dtype", "different_global_anchor"])
def test_correctness_proof_inventory_is_not_only_summary_pass(setup, mutation):
    d, observations = setup
    reference = observations[0]["correctness"]
    summary = wan._read_ref(reference)
    row = summary["ranks"][0]
    if mutation == "duplicate_rollout": row["rollout_evidence"][1] = row["rollout_evidence"][0]
    elif mutation == "missing_dtype": row["build"]["generator"]["lora_compute_dtype_contract"]["records"] = []
    else: row["state_anchors"] = {**row["state_anchors"], "sha256": "f"*64}
    new_ref = _write(Path(reference["path"]), summary)
    with pytest.raises(ValueError):
        wan.validate_native_correctness(d, new_ref)


def test_measured_deployment_requires_observed_driver_version(tmp_path):
    d = fixture_deployment(tmp_path)
    d["evidence_scope"] = "MEASURED"
    d["environment"].pop("driver_version")
    d["binding"]["environment_fingerprint"] = fingerprint(d["environment"])
    with pytest.raises(ValueError, match="driver_version"):
        wan.validate_deployment(d)
