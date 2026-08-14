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
    if direction == "H2D" and key in {"generator_entry", "critic_entry/step10", "critic_entry/step11", "critic_entry/step12", "critic_entry/step13", "critic_entry/step14"}:
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
                    "rank": rank,
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


def build(run_root: Path) -> dict[str, Any]:
    rows = _diagnostic_rows(run_root)
    return {
        "schema_version": "distillschedule.static_transfer_attribution.v1",
        "run_root": str(run_root),
        "rows": rows,
        "observed_copy_assignment": "NOT_RUN",
        "method": {
            "runtime_activity_join": "required_correlationId_pid_thread_rank_join",
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
    args = parser.parse_args()
    result = build(args.run_root)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["rows"][0]) if result["rows"] else ["rank", "category"])
        writer.writeheader()
        writer.writerows(result["rows"])
    print(json.dumps({"rows": len(result["rows"]), "observed_copy_assignment": "NOT_RUN"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
