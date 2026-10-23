#!/usr/bin/env python3
"""Compare matched Wan1.3B two-cycle native runs using cycle 2 only."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


SCHEMA = "distillschedule.wan13b_two_cycle_timing_summary.v1"
BASELINE = "world16-fsdp"
OPTIMIZED = "resident-async128-tree-chunk2048"


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path, *, expected_placement: str) -> dict[str, Any]:
    value = json.loads(path.read_text())
    required = {
        "schema_version": SCHEMA,
        "status": "PASS",
        "placement": expected_placement,
        "rank_count": 16,
        "cycles_in_same_process": 2,
        "speed_claim_denominator": "CYCLE2_WORLD_MAX_ONLY",
        "cycle_1_excluded_from_speed_claim": True,
        "profiled": False,
        "profiler_status": "NOT_RUN",
        "formal_eligibility": True,
        "initial_versions": {"generator": 2, "critic": 10, "ema": 10},
        "final_versions": {"generator": 4, "critic": 20, "ema": 20},
        "updates": {"generator": 2, "critic": 10, "ema": 10},
    }
    for key, expected in required.items():
        if value.get(key) != expected:
            raise ValueError(f"{path}: timing summary changed {key}")
    if value.get("cycle_1", {}).get("claimable") is not False:
        raise ValueError(f"{path}: cycle1 was incorrectly made claimable")
    if value.get("cycle_2", {}).get("claimable") is not True:
        raise ValueError(f"{path}: cycle2 is not claimable")
    for key in ("cycle_1", "cycle_2"):
        seconds = float(value.get(key, {}).get("world_max_seconds", float("nan")))
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError(f"{path}: invalid {key} world-max time")
    if len(str(value.get("workload_fingerprint", ""))) != 64:
        raise ValueError(f"{path}: workload fingerprint missing")
    if len(str(value.get("implementation_fingerprint", ""))) != 64:
        raise ValueError(f"{path}: implementation fingerprint missing")
    if len(str(value.get("trajectory_comparison_sha256", ""))) != 64:
        raise ValueError(f"{path}: cross-arm trajectory comparison binding missing")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--optimized-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = _load(args.baseline_summary, expected_placement=BASELINE)
    optimized = _load(args.optimized_summary, expected_placement=OPTIMIZED)
    if baseline["expected_head"] != optimized["expected_head"]:
        raise ValueError("paired arms use different source HEADs")
    if baseline["workload_fingerprint"] != optimized["workload_fingerprint"]:
        raise ValueError("paired arms do not share one workload fingerprint")
    if baseline["matched_contract"] != optimized["matched_contract"]:
        raise ValueError("paired arms do not share one matched workload contract")
    if baseline["trajectory_comparison_sha256"] != optimized["trajectory_comparison_sha256"]:
        raise ValueError("paired arms are not bound to the same trajectory comparison")
    if baseline["correctness_receipt_sha256"] == optimized["correctness_receipt_sha256"]:
        # Each arm has its own placement-qualified two-cycle correctness run.
        raise ValueError("paired arms unexpectedly share one correctness receipt")
    if baseline["implementation_fingerprint"] == optimized["implementation_fingerprint"]:
        raise ValueError("paired arms do not identify distinct implementations")

    baseline_cycle2 = float(baseline["cycle_2"]["world_max_seconds"])
    optimized_cycle2 = float(optimized["cycle_2"]["world_max_seconds"])
    speedup = baseline_cycle2 / optimized_cycle2
    result = {
        "schema_version": "distillschedule.wan13b_two_cycle_timing_pair.v1",
        "status": "PASS",
        "formal_eligibility": True,
        "expected_head": baseline["expected_head"],
        "workload_fingerprint": baseline["workload_fingerprint"],
        "matched_contract": baseline["matched_contract"],
        "trajectory_comparison_sha256": baseline["trajectory_comparison_sha256"],
        "baseline": {
            "placement": BASELINE,
            "summary": str(args.baseline_summary.resolve()),
            "summary_sha256": _sha256(args.baseline_summary),
            "cycle_1_seconds": baseline["cycle_1"]["world_max_seconds"],
            "cycle_2_seconds": baseline_cycle2,
            "implementation_fingerprint": baseline["implementation_fingerprint"],
        },
        "optimized": {
            "placement": OPTIMIZED,
            "summary": str(args.optimized_summary.resolve()),
            "summary_sha256": _sha256(args.optimized_summary),
            "cycle_1_seconds": optimized["cycle_1"]["world_max_seconds"],
            "cycle_2_seconds": optimized_cycle2,
            "implementation_fingerprint": optimized["implementation_fingerprint"],
        },
        "cycle_2_speedup": speedup,
        "cycle_2_wall_reduction_seconds": baseline_cycle2 - optimized_cycle2,
        "cycle_2_wall_reduction_fraction": 1.0 - optimized_cycle2 / baseline_cycle2,
        "claim_boundary": "MATCHED_UNPROFILED_CYCLE2_WORLD_MAX_ONLY",
        "cycle_1_use": "WARMUP_DIAGNOSTIC_EXCLUDED",
        "profiled_timing_used": False,
        "averaged_cycle_timing_used": False,
    }
    _atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
