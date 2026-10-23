"""Bind the opt-in native H3 observer to existing search gates/resources.

No training loop here. Sampled parity is diagnostic REVIEW, never full-state
equivalence. Missing RNG/samples, non-finite values and changed contracts fail.
"""
from pathlib import Path
import copy

from .contracts import MEASUREMENT_SCHEMA, require, validate_measurement
from .h3_implementation import H3, H3Implementation
from .h3_native import compare_native_two_cycle, read_native_receipts, summarize_native_two_cycle, _rng
from distillschedule.workloads.best_known.registry import fingerprint, file_sha256, write_new_json


def validate_comparison(value):
    from .h3_noise_admission import SCHEMA, validate_admission
    if value.get("schema_version") == SCHEMA:
        validate_admission(value)
        return
    require(value.get("schema_version") == "distillschedule.h3_native_two_cycle_comparison.v1"
            and value.get("status") == "REVIEW_SAMPLED_PARITY_ONLY", "H3 sampled numerical gate failed/incomplete")
    require(all(value.get(k) == "PASS" for k in ("sampled_state_status", "loss_status", "sample_status", "rng_status")),
            "H3 sampled state/loss/sample/RNG gate failed")


def _checked_native(receipt):
    from .native import checked_summary
    out, summary = checked_summary(receipt)
    if "execution_transport" in receipt:
        from .r9_native import verify_execution
        verify_execution(receipt)
    rows, bindings = read_native_receipts(out)
    current = summarize_native_two_cycle(rows, expected_head=receipt["integration_head"],
                                         candidate_id=receipt["search_candidate"]["id"])
    require(all(summary.get(k) == v for k, v in current.items()), "H3 bound summary differs from native receipts")
    require(summary.get("native_evidence") == bindings, "H3 native evidence changed after summary")
    return out, summary, rows, bindings


def _unprofiled_launch_evidence(receipt, out, summary, *, mode="diagnostic-timing"):
    """Positive source-bound direct-launch proof, not an absent-file inference.

    The clean source launcher runs this exact command in clean_environment;
    calibration and a profiler wrapper cannot enter the admitted command.
    This does not claim system-wide profiler detection outside that launcher.
    """
    from .native import read, binding
    require(mode in ("diagnostic-timing", "calibration") and receipt["mode"] == mode,
            "H3 direct launch mode mismatch")
    if "execution_transport" in receipt:
        from .r9_native import verify_execution
        return verify_execution(receipt)
    evidence = []
    for node_rank, node in enumerate(receipt["nodes"]):
        env = node["environment"]
        path = out / "launcher" / f"node_{node_rank}" / "launch_manifest.json"
        manifest = read(path)
        expected = [node["python"], "-m", "torch.distributed.run", "--nnodes=2", "--nproc-per-node=8",
            f"--node-rank={node_rank}", f"--master-addr={receipt['asset_config']['master_addr']}",
            f"--master-port={receipt['asset_config']['master_port']}", "--module",
            "distillschedule.backends.training.minimax_h3.integrated_entrypoint", "--config", env["H3_CONFIG"]]
        require(manifest.get("status") == "PASS_SOURCE_PREFLIGHT" and manifest.get("source_clean") is True
                and manifest.get("integration_head") == receipt["integration_head"]
                and manifest.get("mode") == mode and manifest.get("command") == expected
                and manifest.get("python_executable") == node["python"], "H3 direct unprofiled launch proof failed")
        observed = manifest.get("environment", {})
        wanted = {k: v for k, v in env.items() if k.startswith("H3_")}
        wanted.update(DISTILLSCHEDULE_RUN_ID=receipt["run_id"], DISTILLSCHEDULE_RUN_ROOT=str(out),
                      DISTILLSCHEDULE_RUN_KIND=mode, H3_PLANNER_CALIBRATION=str(int(mode == "calibration")))
        require(all(observed.get(k) == v for k, v in wanted.items()), "H3 launch environment/calibration proof failed")
        require(manifest.get("planner_binding") == {k: summary[k] for k in
            ("candidate_id", "planner_implementation", "effective_configuration_sha256")}, "H3 launch binding differs")
        evidence.append(binding(path))
    require(len(evidence) == 2, "H3 both node launch proofs required")
    return evidence


def prepare_pair(reference, candidate, *, repeat=None, noise_policy=None):
    from .native import binding
    for name in ("profile_id", "integration_head", "asset_fingerprint", "workload_semantics_hash"):
        require(reference[name] == candidate[name], f"H3 pair {name} differs")
    require(reference["profile_id"] == H3 and reference["mode"] == candidate["mode"] == "correctness",
            "H3 pair must be fresh correctness runs")
    require((repeat is None) == (noise_policy is None), "H3 diagnostic policy requires a control repeat")
    left, _, rows0, sources0 = _checked_native(reference)
    right, _, rows1, sources1 = _checked_native(candidate)
    comparison = compare_native_two_cycle(rows0, rows1, expected_head=reference["integration_head"])
    out = right / "search-gate"
    out.mkdir(exist_ok=False)
    path = out / "comparison.json"
    write_new_json(path, comparison)  # Failures are preserved before raising.
    strict_path = path
    if noise_policy is not None:
        from .h3_noise_admission import qualify
        for key in ("profile_id", "integration_head", "asset_fingerprint", "workload_semantics_hash", "mode"):
            require(repeat.get(key) == reference[key], "H3 control repeat source/assets/semantics changed")
        require(repeat["run_id"] not in (reference["run_id"], candidate["run_id"])
                and repeat["search_candidate"] == reference["search_candidate"],
                "H3 diagnostic repeat must be an independent run of the control")
        _, _, repeated, sources2 = _checked_native(repeat)
        comparison = qualify(rows0, repeated, rows1, policy=noise_policy,
                             expected_head=reference["integration_head"])
        comparison["native_evidence"] = dict(reference=sources0, repeat=sources2, candidate=sources1)
        comparison["execution_plans"] = {}
        for name, plan in (("reference", reference), ("repeat", repeat), ("candidate", candidate)):
            saved_plan = out / (name + "-execution-plan.json")
            write_new_json(saved_plan, plan)
            comparison["execution_plans"][name] = binding(saved_plan)
        path = out / "noise-diagnostic-admission.json"
        write_new_json(path, comparison)
    validate_comparison(comparison)
    result = {}
    for plan, run in ((reference, left), (candidate, right)):
        cid = plan["search_candidate"]["id"]
        H3Implementation.parse(cid)
        gate = dict(status="PASS", numerical_status="REVIEW", comparison=binding(path),
                    integration_binding={k: plan[k] for k in ("integration_head", "profile_sha256", "asset_fingerprint")})
        gate_path = out / (cid + "-bound-gate.json")
        write_new_json(gate_path, gate)
        assets = copy.deepcopy(plan["asset_config"])
        assets["correctness"] = binding(gate_path)
        result[cid] = dict(source_head=plan["integration_head"], candidate_id=cid, semantics_status="PASS",
            numerical_status="REVIEW", strict_numerical_gate="NOT_ESTABLISHED_BY_LOCAL_SHARD_SAMPLES",
            comparison=binding(path), correctness=binding(run / "bound-summary.json"), assets=assets)
        if noise_policy is not None:
            result[cid].update(diagnostic_admission="NOISE_QUALIFIED_CACHE_ONLY",
                               diagnostic_noise_policy=copy.deepcopy(noise_policy),
                               original_strict_comparison=binding(strict_path), formal_timing_eligible=False)
        write_new_json(out / (cid + "-preparation.json"), result[cid])
    return result


def validate_preparation(preparation, *, expected_head, candidate_id, noise_policy=None):
    from .native import read
    from .h3_noise_admission import SCHEMA, validate_policy
    require(preparation.get("source_head") == expected_head and preparation.get("candidate_id") == candidate_id,
            "H3 cached preparation source/candidate changed")
    require(preparation.get("diagnostic_noise_policy") == noise_policy,
            "H3 cached preparation admission policy differs from explicit opt-in")
    for key in ("comparison", "correctness"):
        ref = preparation[key]
        require(file_sha256(Path(ref["path"])) == ref["sha256"], "H3 cached preparation witness changed")
    comparison = read(Path(preparation["comparison"]["path"]))
    if noise_policy is not None:
        validate_policy(noise_policy, expected_head=expected_head)
        require(comparison.get("schema_version") == SCHEMA and comparison.get("policy") == noise_policy,
                "H3 cached diagnostic comparison does not bind current policy")
    else:
        require(comparison.get("schema_version") != SCHEMA, "H3 diagnostic cache requires current explicit opt-in")
    validate_comparison(comparison)
    require(comparison["integration_head"] == expected_head
            and candidate_id in (comparison["reference_candidate_id"], comparison["candidate_id"]),
            "H3 cached comparison source/candidate changed")
    return dict(status="VALIDATED_PREPARATION", source_head=expected_head, candidate_id=candidate_id,
                numerical_status="REVIEW", formal_timing_eligible=False)


def normalize_native(receipt, preparation, identity):
    from .native import read, binding
    from .registry import get_candidate
    out, summary, raw_rows, bindings = _checked_native(receipt)
    c = get_candidate(H3, receipt["search_candidate"]["id"])
    require(receipt["mode"] == "diagnostic-timing" and not summary["calibration_only"], "H3 calibration/probe is not timing")
    for field, name in (("source_head", "integration_head"), ("asset_fingerprint", "asset_fingerprint"),
                        ("semantic_fingerprint", "workload_semantics_hash")):
        require(identity[field] == receipt[name], f"H3 timing {field} differs")
    require(preparation["source_head"] == identity["source_head"] and preparation["candidate_id"] == c.id,
            "H3 preparation changed")
    for k in ("comparison", "correctness"):
        require(file_sha256(Path(preparation[k]["path"])) == preparation[k]["sha256"], "H3 gate changed")
    comparison = read(Path(preparation["comparison"]["path"]))
    validate_comparison(comparison)
    require(comparison["integration_head"] == identity["source_head"]
            and c.id in (comparison["reference_candidate_id"], comparison["candidate_id"]), "H3 comparison source/candidate differs")
    require(not list(out.rglob("*.nsys-rep")) and not list(out.rglob("*.sqlite")), "H3 profiled artifacts in timing")
    evidence = [binding(out / "bound-summary.json"), preparation["comparison"], preparation["correctness"]]
    evidence += _unprofiled_launch_evidence(receipt, out, summary)
    resources = []
    for node in (0, 1):
        p = out / f"search-resources.node{node}.json"
        resource = read(p)
        require(resource["status"] == "PASS" and resource["node_rank"] == node and resource["run_id"] == receipt["run_id"]
                and len(resource["samples"]) >= 2, "H3 resource observer missing")
        for event in ("oom", "oom_kill"):
            counters = [s["cgroup_events"][event] for s in resource["samples"]]
            require(max(counters) == min(counters), "H3 host OOM")
        evidence.append(binding(p))
        if "execution_transport" not in receipt:
            path = out / "launcher" / f"node_{node}" / "teardown_status.txt"
            flags = dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)
            require(flags == dict(status="PASS", gpu_process_count="0", owner_run_id=receipt["run_id"]), "H3 teardown missing")
            evidence.append(binding(path))
        resources.append(resource)
    evidence.extend(bindings)
    ranks = []
    for raw in raw_rows:
        rank = raw["rank"]
        require(_rng(raw, raw, rank, identity["source_head"]) == "PASS", "H3 timing RNG boundary proof missing/failed")
        memories = [c[k]["hbm"] for c in raw["cycles"] for k in ("memory_before", "memory_after")]
        samples = resources[rank // 8]["samples"]
        gpu = [next(g for g in s["gpus"] if g["device"] == rank % 8) for s in samples]
        ranks.append(dict(rank=rank, status="PASS", **{k: "PASS" for k in
            ("census", "versions", "sample_rng", "finite", "dtype", "memory")},
            cuda_oom_delta=0, cgroup_oom_delta=0,
            peak_allocated_bytes=max(m["peak_allocated_bytes"] for m in memories),
            peak_reserved_bytes=max(m["peak_reserved_bytes"] for m in memories),
            min_driver_free_bytes=min(g["free_bytes"] for g in gpu),
            host_peak_bytes=max(s["cgroup_current_bytes"] for s in samples),
            gate_authority="NATIVE_CENSUS_REPLAY_LOSS_AND_PREPARATION_SAMPLED_RNG_PARITY",
            cuda_oom_evidence="COMPLETED_NATIVE_RUN_NOT_A_DRIVER_COUNTER"))
    value = dict(schema_version=MEASUREMENT_SCHEMA, status="PASS", **identity,
        candidate_id=c.id, execution_fingerprint=fingerprint(c.to_dict()), profiled=False,
        same_process=True, initializations=1, semantics_status="PASS", numerical_status="REVIEW",
        numerical_scope="LOCAL_SHARD_SAMPLES_NOT_FULL_STATE_EQUIVALENCE", profiler_status="DISABLED",
        profiler_evidence_scope="SOURCE_BOUND_DIRECT_TORCHRUN_COMMAND_AND_CALIBRATION_DISABLED",
        rng_scope="TORCH_DEFAULT_CPU_AND_RANK_CUDA_BOUNDARIES_ONLY; OTHER_GENERATORS_NOT_OBSERVED",
        formal_timing_eligible=False, search_ready=False, strict_numerical_gate="NOT_RUN",
        resource_status="EXCLUSIVE_LOCKED", cleanup_status="PASS", timer_scope="NATIVE_CYCLE_OUTSIDE_PROBES",
        native_timer_denominator=summary["timer_scope"], cycle_wall_seconds=summary["cycle_wall_seconds"],
        ranks=ranks, native_evidence=evidence)
    validate_measurement(value, identity, c)
    return value
