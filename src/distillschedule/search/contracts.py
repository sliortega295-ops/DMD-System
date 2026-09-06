"""Search evidence contracts; numerical REVIEW is never strict equivalence."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

from distillschedule.workloads.best_known.registry import fingerprint

SCHEMA = "distillschedule.search.v1"
PROFILE_SCHEMA = "distillschedule.search.profile.v1"
MEASUREMENT_SCHEMA = "distillschedule.search.measurement.v1"
PILOTS = ("wan13b_nogan_resident_best", "livewan_crossscale_pipeline_best")
CAPACITY_FAILURES = frozenset({"CUDA_OOM", "HOST_CGROUP_OOM"})
FAILURES = CAPACITY_FAILURES | frozenset({
    "NCCL_ERROR", "SOURCE_ERROR", "ASSET_ERROR", "CORRECTNESS_ERROR",
    "LAUNCHER_ERROR", "TIMEOUT", "OTHER_RUNTIME_ERROR", "INCOMPLETE_RECEIPT",
})


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def finite_number(value: Any, name: str, *, positive: bool = False) -> float:
    require(type(value) in (int, float), f"{name} must be a number")
    number = float(value)
    require(math.isfinite(number) and (number > 0 if positive else number >= 0),
            f"invalid {name}: {value!r}")
    return number


@dataclass(frozen=True)
class Candidate:
    id: str
    workload: str
    placement: str
    kernels: tuple[str, ...] = ()
    targets: tuple[str, ...] = ()
    parent: str | None = None
    mechanism: str = "reference"
    variant: str | None = None
    executable: bool = True
    gpu_validation: str = "NOT_RUN_ON_SEARCH_BRANCH"

    def to_dict(self) -> dict[str, Any]:
        # JSON round trips must preserve search/resume identity (tuples become
        # lists otherwise, making an unchanged journal impossible to resume).
        value = asdict(self)
        value.update(kernels=list(self.kernels), targets=list(self.targets))
        return value


def semantic_contract(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Remove execution labels, not math, from the immutable canonical contract.

    Two-cycle execution advances existing parameters/state. It never resets a
    model between cycles or averages cycle1 into the steady-state denominator.
    """
    excluded = {"winner", "variant", "micro_waves", "continuous_cycles", "final_versions"}
    contract = {k: v for k, v in profile["contract"].items() if k not in excluded}
    return {
        "workload": profile["id"], "semantics": profile["semantics"],
        "contract": contract, "cycles": 2, "measured_cycle": 2,
        "same_process": True, "initializations": 1,
        "runtime_seed": 20260903 if profile["id"] == PILOTS[0] else profile["contract"].get("seed"),
        "physical_inventory": "VALIDATE_NATIVE_PLAN_SPECIFIC_INVENTORY",
        "sigma_distribution_search": False, "global_batch_search": False,
        "numerical_thresholds": "UNCHANGED_WORKLOAD_OWNED",
    }


def validate_measurement(value: Mapping[str, Any], binding: Mapping[str, Any],
                         candidate: Candidate) -> dict[str, Any]:
    require(value.get("schema_version") == MEASUREMENT_SCHEMA, "measurement schema mismatch")
    for key in ("source_head", "semantic_fingerprint", "asset_fingerprint", "hardware_fingerprint"):
        require(value.get(key) == binding.get(key) and bool(value.get(key)), f"measurement {key} mismatch")
    require(value.get("candidate_id") == candidate.id, "candidate identity mismatch")
    require(value.get("execution_fingerprint") == fingerprint(candidate.to_dict()), "execution fingerprint mismatch")
    require(value.get("profiled") is False, "profiled time cannot rank a candidate")
    require(value.get("same_process") is True and value.get("initializations") == 1,
            "two cycles require one process lifetime/initialization")
    rows = value.get("ranks", [])
    require(len(rows) == 16 and {r.get("rank") for r in rows} == set(range(16)), "16/16 unique rank receipts required")
    require(value.get("semantics_status") == "PASS", "semantic contract failed")
    require(value.get("numerical_status") in {"PASS", "REVIEW"}, "numerical gate is neither PASS nor REVIEW")
    for row in rows:
        require(row.get("status") == "PASS", f"rank {row.get('rank')} failed")
        for gate in ("census", "versions", "sample_rng", "finite", "dtype", "memory"):
            require(row.get(gate) == "PASS", f"rank {row['rank']} {gate} failed/missing")
        require(row.get("cuda_oom_delta") == 0 and row.get("cgroup_oom_delta") == 0,
                "OOM counter changed")
        for field in ("peak_allocated_bytes", "peak_reserved_bytes", "min_driver_free_bytes", "host_peak_bytes"):
            finite_number(row.get(field), field)
        require(row["peak_reserved_bytes"] >= row["peak_allocated_bytes"], "invalid HBM telemetry")
    require(value.get("profiler_status") == "DISABLED", "profiler state not proven disabled")
    require(value.get("resource_status") == "EXCLUSIVE_LOCKED", "GPU ownership/lock not proven")
    require(value.get("cleanup_status") == "PASS", "owned process cleanup incomplete")
    times = value.get("cycle_wall_seconds", {})
    cycle1 = finite_number(times.get("1"), "cycle1", positive=True)
    cycle2 = finite_number(times.get("2"), "cycle2", positive=True)
    require(value.get("timer_scope") == "NATIVE_CYCLE_OUTSIDE_PROBES", "unknown timing denominator")
    require(bool(value.get("native_evidence")), "native evidence references are required")
    return {
        "candidate_id": candidate.id, "cycle1_seconds": cycle1, "cycle2_seconds": cycle2,
        "numerical_status": value["numerical_status"], "semantics_status": "PASS",
        "strict_eligible": value["numerical_status"] == "PASS",
        "measurement_sha256": fingerprint(value), "native_evidence": value["native_evidence"],
        "peak_allocated_bytes": max(r["peak_allocated_bytes"] for r in rows),
        "min_driver_free_bytes": min(r["min_driver_free_bytes"] for r in rows),
        "observations": value.get("observations", {}),
    }
