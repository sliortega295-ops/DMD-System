#!/usr/bin/env python3
"""Report pooled world16 GPU activity for a static FSDP2 Full5 profile."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from distillschedule.profiling.nsight_activity import (
    COMPUTE,
    NCCL,
    ALL_COPY,
    NO_TRACED_ACTIVITY,
    aggregate_rank_reports,
    analyze_sqlite,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=("no-gan", "gan"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pattern = {"full5": rf"STATIC_FSDP2_FULL5/{args.variant}"}
    reports = []
    for rank in range(16):
        path = args.profile_dir / f"rank_{rank:02d}.sqlite"
        report = analyze_sqlite(
            path,
            rank=rank,
            phase_patterns=pattern,
            timeline_bin_ns=500_000_000,
        )
        if "full5" not in report["phases"]:
            raise RuntimeError(f"rank {rank} has no exact static FSDP2 Full5 NVTX window")
        reports.append(report)
    aggregate = aggregate_rank_reports(reports)
    metric = aggregate["phases"]["full5"]
    categories = metric["categories"]
    summary = {
        "schema_version": "distillschedule.static_fsdp2_nsys.v1",
        "variant": args.variant,
        "rank_count": 16,
        "denominator": "sum(rank category interval union) / sum(rank exact Full5 NVTX wall)",
        "gpu_wall_seconds": metric["gpu_wall_seconds"],
        "compute_kernel_gpu_seconds": categories[COMPUTE]["gpu_seconds"],
        "compute_kernel_percent": categories[COMPUTE]["pooled_percent"],
        "nccl_gpu_seconds": categories[NCCL]["gpu_seconds"],
        "nccl_percent": categories[NCCL]["pooled_percent"],
        "all_copy_gpu_seconds": categories[ALL_COPY]["gpu_seconds"],
        "all_copy_percent": categories[ALL_COPY]["pooled_percent"],
        "no_traced_gpu_activity_seconds": categories[NO_TRACED_ACTIVITY]["gpu_seconds"],
        "no_traced_gpu_activity_percent": categories[NO_TRACED_ACTIVITY]["pooled_percent"],
        "claim_boundary": {
            "profiled_latency_is_diagnostic_only": True,
            "compute_nccl_copy_may_overlap_and_are_not_additive": True,
            "no_traced_gpu_activity_is_not_proof_of_cpu_idle": True,
            "formal_full5_time_comes_from_unprofiled_runs": True,
        },
        "aggregate": aggregate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "variant": args.variant,
        "compute_kernel_percent": summary["compute_kernel_percent"],
        "nccl_percent": summary["nccl_percent"],
        "all_copy_percent": summary["all_copy_percent"],
        "no_traced_gpu_activity_percent": summary["no_traced_gpu_activity_percent"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
