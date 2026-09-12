"""Native evidence -> optimizer observations, never exit-code -> PASS.

The workload-owned summary is rerun by dispatch after teardown, including its
rank/trace/call/version/kernel gates. This adapter additionally binds source,
assets, native gate, two-cycle timing and measured resource evidence.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from distillschedule.workloads.best_known.registry import fingerprint, file_sha256
from .contracts import MEASUREMENT_SCHEMA, require, validate_measurement
from .native import read, binding, checked_summary, validate_native_comparison
from .registry import WAN, get_candidate


def normalize_native(receipt: dict[str, Any], preparation: dict[str, Any],
                     identity: dict[str, str]) -> dict[str, Any]:
    out, summary = checked_summary(receipt)
    c = get_candidate(receipt["profile_id"], receipt["search_candidate"]["id"])
    require(receipt["native_mode"] == "two-cycle-timing", "correctness/profile times are not measurements")
    require(identity["source_head"] == receipt["integration_head"] == preparation["source_head"], "source differs")
    require(identity["asset_fingerprint"] == receipt["asset_fingerprint"], "assets differ")
    require(identity["semantic_fingerprint"] == receipt["workload_semantics_hash"], "semantics differ")
    require(preparation["candidate_id"] == c.id, "gate candidate differs")
    for name in ("comparison", "correctness"):
        require(file_sha256(Path(preparation[name]["path"])) == preparation[name]["sha256"], f"{name} changed")
    comparison = read(Path(preparation["comparison"]["path"]))
    validate_native_comparison(comparison)
    require(summary.get("profiled") is False and summary.get("profiler_status") == "NOT_RUN", "profiled receipt")
    require(not summary.get("calibration_only"), "operation calibration is not native timing")
    require(summary.get("cycles_in_same_process") == 2 and summary["cycle_2"]["claimable"] is True,
            "native two-cycle timer missing")
    require(not list(out.rglob("*.nsys-rep")) and not list(out.rglob("*.sqlite")), "profile artifacts in native run")
    rows, evidence, resources = [], [binding(out / "bound-summary.json"), preparation["comparison"], preparation["correctness"]], []
    for node in (0, 1):
        path = out / f"search-resources.node{node}.json"
        resource = read(path)
        require(resource["status"] == "PASS" and resource["run_id"] == receipt["run_id"]
                and resource["node_rank"] == node and len(resource["samples"]) >= 2,
                "resource observer missing/failed")
        samples = resource["samples"]
        for event in ("oom", "oom_kill"):
            require(all(r["cgroup_events"][event] == samples[0]["cgroup_events"][event] for r in samples),
                    "cgroup OOM counter increased")
        evidence.append(binding(path))
        resources.append(resource)
        # The native launchers acquire the shared flock before GPU audit, hold
        # it through teardown, and bind the clean teardown to this run ID.
        launch = out / "launcher" / f"node_{node}"
        values = dict(line.split("=", 1) for line in (launch / "teardown_status.txt").read_text().splitlines() if "=" in line)
        require(values.get("status") == "PASS" and values.get("gpu_process_count") == "0"
                and values.get("owner_run_id") == receipt["run_id"], "owned teardown not proven")
        evidence += [binding(launch / "launch_manifest.txt"), binding(launch / "teardown_status.txt")]
    expected_versions = {"generator": 4, "critic": 20, "ema": 20} if c.workload == WAN else {"generator": 2, "critic": 10, "ema": 0}
    require(summary.get("final_versions") == expected_versions, "final versions differ")
    for rank in range(16):
        path = out / (f"execution/ranks/rank_{rank:02d}.json" if c.workload == WAN else f"receipts/rank_{rank:03d}.json")
        raw = read(path)
        require(raw["rank"] == rank and raw["status"] == "PASS" and raw["final_versions"] == expected_versions,
                "rank status/versions differ")
        if c.workload == WAN:
            require(raw["finite_status"] == [True, True] and len(raw["cycle_census"]) == 2,
                    "native finite/census gate incomplete")
            memory = raw["memory"]
        else:
            require(len(raw["two_cycle_timing"]["per_cycle_contract"]) == 2, "LiveWan cycle contract missing")
            memory = raw["peak_memory"]
        samples = resources[rank // 8]["samples"]
        gpu = [next(g for g in sample["gpus"] if g["device"] == rank % 8) for sample in samples]
        rows.append({"rank": rank, "status": "PASS", **{gate: "PASS" for gate in
            ("census", "versions", "sample_rng", "finite", "dtype", "memory")},
            "gate_authority": "SOURCE_BOUND_WORKLOAD_OWNED_SUMMARIZER_AND_TRAJECTORY_COMPARATOR",
            "native_receipt": binding(path), "cuda_oom_delta": 0, "cgroup_oom_delta": 0,
            "cuda_oom_evidence": "NATIVE_COMPLETION_AND_NO_WORKER_OOM; NOT_A_DRIVER_COUNTER",
            "peak_allocated_bytes": memory["max_allocated_bytes"],
            "peak_reserved_bytes": memory["max_reserved_bytes"],
            "min_driver_free_bytes": min(g["free_bytes"] for g in gpu),
            "host_peak_bytes": max(s["cgroup_current_bytes"] for s in samples),
            "telemetry_scope": "NATIVE_ALLOCATOR_PEAK; SAMPLED_DRIVER_MIN; NODE_CGROUP_PEAK_SHARED_BY_8_RANKS"})
        evidence.append(binding(path))
    cycle1 = summary["cycle_1"]["world_max_seconds" if c.workload == WAN else "world_completion_endpoint_seconds"]
    cycle2 = summary["cycle_2"]["world_max_seconds" if c.workload == WAN else "seconds"]
    value = {"schema_version": MEASUREMENT_SCHEMA, "status": "PASS", **identity,
        "candidate_id": c.id, "execution_fingerprint": fingerprint(c.to_dict()),
        "profiled": False, "same_process": True, "initializations": 1,
        "semantics_status": "PASS", "numerical_status": "REVIEW",
        "native_numerical_status": comparison["status"],
        "numerical_scope": "COARSE_TRAJECTORY_IS_NOT_STRICT_P0_PARITY; STRICT_NOT_ESTABLISHED",
        "profiler_status": "DISABLED", "resource_status": "EXCLUSIVE_LOCKED", "cleanup_status": "PASS",
        "timer_scope": "NATIVE_CYCLE_OUTSIDE_PROBES",
        "native_timer_denominator": summary.get("speed_claim_denominator", summary["cycle_2"].get("denominator")),
        "cycle_wall_seconds": {"1": cycle1, "2": cycle2}, "ranks": rows,
        "native_evidence": evidence, "runtime_validation": summary.get("implementation_contract", summary.get("implementation")),
        "construction": summary.get("construction", "SEE_NATIVE_SETUP_RECEIPTS")}
    validate_measurement(value, identity, c)
    return value
