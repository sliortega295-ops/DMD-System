#!/usr/bin/env python3
"""Build per-rank transfer attribution from profile-side receipts.

Application counters are logical bytes at an explicit boundary; CUDA activity
bytes are observed bytes.  The tool never assigns an observed copy to FSDP or
an application label by wall-clock overlap.  Missing correlation-id joins are
reported as ``UNATTRIBUTED`` instead of being guessed.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


APPLICATION_KEYS = {
    "INPUT": ("INPUT_H2D/",),
    "EMA": ("EMA_CPU_D2H/",),
    "DIAGNOSTIC": (
        "DIAG_OUTPUT_SHA_D2H/",
        "DIAG_LOSS_D2H/",
        "DIAG_FINGERPRINT_D2H/",
    ),
    "FSDP_NCCL_FRAMEWORK": (
        "FSDP_PRE_FORWARD_ALL_GATHER/",
        "FSDP_POST_FORWARD_RESHARD/",
        "FSDP_PRE_BACKWARD_ALL_GATHER/",
        "FSDP_REDUCE_SCATTER/",
        "OPTIMIZER_STEP/",
    ),
}


def _bucket_for(direction: str, key: str) -> str:
    """Classify an application counter without pretending it is CUDA activity.

    The profile-side counter schema stores direction and logical key as
    separate fields (for example ``H2D/generator_entry`` and
    ``D2H/ema_shadow``).  Earlier code matched the bucket prefixes against only
    the logical key, which silently put every normal input/EMA counter in
    ``UNATTRIBUTED``.  Keep this mapping explicit and fail closed for keys that
    are not part of the registered contract.
    """
    token = f"{direction}/{key}"
    if direction == "H2D" and (key.startswith("generator_") or key.startswith("critic_")):
        return "INPUT"
    if key == "ema_shadow":
        return "EMA"
    for bucket, prefixes in APPLICATION_KEYS.items():
        if token.startswith(prefixes):
            return bucket
    return "UNATTRIBUTED"


def _diagnostic_rows(run_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(run_root.glob("execution/diagnostics/rank_*.json")):
        value = json.loads(path.read_text())
        rank = int(value["rank"])
        buckets: dict[tuple[str, str], dict[str, int]] = {
            (direction, name): {"bytes": 0, "events": 0}
            for direction in ("H2D", "D2H", "D2D", "GPU", "OTHER")
            for name in (*APPLICATION_KEYS, "UNATTRIBUTED")
        }
        for key, counter in value.get("counters", {}).items():
            direction, logical_key = key.split("/", 1)
            direction = direction if direction in {"H2D", "D2H", "D2D", "GPU"} else "OTHER"
            bucket = _bucket_for(direction, logical_key)
            buckets[(direction, bucket)]["bytes"] += int(counter.get("bytes", 0))
            buckets[(direction, bucket)]["events"] += int(counter.get("events", 0))
        for (direction, bucket), counters in buckets.items():
            rows.append(
                {
                    "scope": "rank",
                    "rank": rank,
                    "rank_mapping": "application_receipt",
                    "direction": direction,
                    "category": bucket,
                    "logical_bytes": counters["bytes"],
                    "logical_events": counters["events"],
                    "observed_bytes": None,
                    "observed_events": None,
                    "observed_sum_duration_ns": None,
                    "observed_union_ns": None,
                    "attribution_method": "application_boundary_counter_only",
                    "claim": "logical_only_unless_runtime_activity_correlation_join_is_present",
                }
            )
    return rows


def _observed_rows(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    """Expose observed activity without assigning it to an app boundary.

    The current node export has no reliable correlationId -> application NVTX
    join.  We therefore retain per-process and node observed H2D/D2H/D2D
    metrics under ``UNATTRIBUTED`` instead of guessing by wall-clock overlap.
    ``rank`` is explicitly marked as inferred by the bounded analyzer.
    """
    rows: list[dict[str, Any]] = []
    metric_directions = {"h2d": "H2D", "d2h": "D2H", "d2d": "D2D", "other_copy": "OTHER"}
    for result in bundle.get("results", []):
        for process in result.get("per_process", []):
            metrics = process.get("metrics", {})
            for metric, direction in metric_directions.items():
                value = metrics.get(metric, {})
                rows.append({
                    "scope": "rank",
                    "rank": process.get("rank"),
                    "rank_mapping": process.get("rank_mapping"),
                    "direction": direction,
                    "category": "UNATTRIBUTED",
                    "logical_bytes": 0,
                    "logical_events": 0,
                    "observed_bytes": value.get("bytes", 0),
                    "observed_events": value.get("events", 0),
                    "observed_sum_duration_ns": value.get("sum_duration_ns", 0),
                    "observed_union_ns": value.get("union_ns", 0),
                    "attribution_method": "bounded_nsys_activity_without_correlation_join",
                    "claim": "observed_copy_unattributed_no_wall_overlap_guess",
                })
        node = result.get("node_metrics", {})
        for metric, direction in metric_directions.items():
            value = node.get(metric, {})
            rows.append({
                "scope": "node",
                "rank": -1,
                "rank_mapping": "node_aggregate",
                "direction": direction,
                "category": "UNATTRIBUTED",
                "logical_bytes": 0,
                "logical_events": 0,
                "observed_bytes": value.get("bytes", 0),
                "observed_events": value.get("events", 0),
                "observed_sum_duration_ns": value.get("sum_duration_ns", 0),
                "observed_union_ns": value.get("union_ns", 0),
                "attribution_method": "bounded_nsys_activity_without_correlation_join",
                "claim": "observed_copy_unattributed_no_wall_overlap_guess",
            })
    return rows


def build(run_root: Path, nsys_bundle: Path | None = None) -> dict[str, Any]:
    rows = _diagnostic_rows(run_root)
    if nsys_bundle is not None:
        rows.extend(_observed_rows(json.loads(nsys_bundle.read_text())))
    return {
        "schema_version": "distillschedule.static_transfer_attribution.v2",
        "run_root": str(run_root),
        "rows": rows,
        "observed_copy_assignment": "UNATTRIBUTED_NO_CORRELATION_JOIN" if nsys_bundle else "NOT_RUN",
        "method": {
            "runtime_activity_join": "required_correlationId_pid_thread_rank_join; not available in current export",
            "wall_clock_overlap_guessing": False,
            "unmatched_observed_bytes": "UNATTRIBUTED",
        },
        "claim_boundary": {
            "logical_application_bytes_are_not_observed_cuda_memcpy_bytes": True,
            "formal_timing_unchanged": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--nsys-bundle", type=Path)
    args = parser.parse_args()
    result = build(args.run_root, args.nsys_bundle)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["rows"][0]) if result["rows"] else ["rank", "category"])
        writer.writeheader()
        writer.writerows(result["rows"])
    print(json.dumps({"rows": len(result["rows"]), "observed_copy_assignment": result["observed_copy_assignment"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
