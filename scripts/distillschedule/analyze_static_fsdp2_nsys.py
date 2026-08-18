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
    OTHER_COPY,
    aggregate_rank_reports,
    analyze_sqlite,
)
from distillschedule.profiling.static_nsys_bounded import analyze_bounded_sqlite


def _aggregate_bounded_nodes(reports: list[dict], *, variant: str) -> dict:
    """Aggregate node reports while preserving per-process/rank detail."""
    if len(reports) != 2:
        raise ValueError(f"static world16 node analysis requires two nodes, got {len(reports)}")
    phase_wall_ns = sum(int(report["phase_window"]["wall_ns"]) for report in reports)
    categories = (COMPUTE, NCCL, H2D, D2H, D2D, OTHER_COPY, ALL_COPY, ACTIVE, NO_TRACED_ACTIVITY)
    metrics: dict[str, dict] = {}
    for category in categories:
        values = [report["node_metrics"][category] for report in reports]
        union_ns = sum(int(value["union_ns"]) for value in values)
        metrics[category] = {
            "union_ns": union_ns,
            "union_seconds": union_ns / 1e9,
            "percent_of_sum_node_phase_wall": 100.0 * union_ns / phase_wall_ns if phase_wall_ns else 0.0,
            "events": sum(int(value.get("events", 0)) for value in values),
            "bytes": sum(int(value.get("bytes", 0)) for value in values),
            "sum_duration_ns": sum(int(value.get("sum_duration_ns", 0)) for value in values),
            "sum_duration_seconds": sum(int(value.get("sum_duration_ns", 0)) for value in values) / 1e9,
            "sum_duration_percent_of_sum_node_phase_wall": 100.0 * sum(int(value.get("sum_duration_ns", 0)) for value in values) / phase_wall_ns if phase_wall_ns else 0.0,
        }
    per_rank = [item for report in reports for item in report["per_process"]]
    per_rank.sort(key=lambda item: (item["rank"] is None, item["rank"] if item["rank"] is not None else item["pid"]))
    reconciliation = {
        "raw_memcpy_rows_in_window": {key: sum(int(report["memcpy_totals_reconciliation"]["raw_memcpy_rows_in_window"][key]) for report in reports) for key in ("events", "bytes", "sum_duration_ns")},
        "classified_memcpy_rows_in_window": {key: sum(int(report["memcpy_totals_reconciliation"]["classified_memcpy_rows_in_window"][key]) for report in reports) for key in ("events", "bytes", "sum_duration_ns")},
    }
    reconciliation["residual"] = {key: reconciliation["raw_memcpy_rows_in_window"][key] - reconciliation["classified_memcpy_rows_in_window"][key] for key in reconciliation["raw_memcpy_rows_in_window"]}
    reconciliation["status"] = "PASS" if all(value == 0 for value in reconciliation["residual"].values()) else "FAIL"
    return {
        "schema_version": "distillschedule.static_nsys_bounded.aggregate.v1",
        "variant": variant,
        "scope": "node",
        "scope_count": 2,
        "rank_count": len(per_rank),
        "ranks": [item["rank"] for item in per_rank],
        "phase_wall_seconds_sum_nodes": phase_wall_ns / 1e9,
        "metrics": metrics,
        "per_rank": per_rank,
        "memcpy_totals_reconciliation": reconciliation,
        "metric_definitions": {
            "copy_engine_union_percent": "union of CUDA memcpy intervals across both node reports divided by the sum of the two exact NVTX phase walls; overlaps counted once",
            "copy_sum_duration_percent": "sum of memcpy event durations across both node reports divided by the same phase-wall denominator; overlaps counted repeatedly",
            "d2h_percent": "D2H interval union percent, not all-copy and not PCIe wall time",
            "formal_timing": "not used; profiled phase is diagnostic-only",
            "old_20_09_percent": "the preserved 15-rank report's 20.090% was D2H interval-union percent over the 15 valid-rank exact phase-wall denominator, not a 16-rank copy-engine union or formal Full5 percentage",
        },
        "claim_boundary": {
            "diagnostic_only": True,
            "rank_mapping_in_v3_is_inferred_from_worker_pid_order": True,
            "formal_timing_unchanged": True,
            "categories_overlap_and_are_not_additive": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--profile-dir", type=Path)
    scope.add_argument(
        "--node-profile-dir",
        type=Path,
        help="node-level profile root emitted by the one-session-per-node launcher",
    )
    parser.add_argument("--variant", choices=("no-gan", "gan"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    profile_root = args.node_profile_dir or args.profile_dir
    pattern = {"full5": rf"STATIC_FSDP2_FULL5/{args.variant}"}
    reports = []
    if args.node_profile_dir is not None:
        scopes = [(node, profile_root / f"node_{node}" / f"static_fsdp2_full5_node{node}.sqlite") for node in range(2)]
        scope_name = "node"
        phase_label = f"STATIC_FSDP2_FULL5/{args.variant}"
        for node, path in scopes:
            reports.append(
                analyze_bounded_sqlite(
                    path,
                    phase_label=phase_label,
                    node_rank_base=node * 8,
                )
            )
        summary = _aggregate_bounded_nodes(reports, variant=args.variant)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps({
            "variant": args.variant,
            "scope": "node",
            "rank_count": summary["rank_count"],
            "compute_union_percent": summary["metrics"][COMPUTE]["percent_of_sum_node_phase_wall"],
            "nccl_union_percent": summary["metrics"][NCCL]["percent_of_sum_node_phase_wall"],
            "all_copy_union_percent": summary["metrics"][ALL_COPY]["percent_of_sum_node_phase_wall"],
            "d2h_union_percent": summary["metrics"][D2H]["percent_of_sum_node_phase_wall"],
            "copy_sum_duration_percent": summary["metrics"][ALL_COPY]["sum_duration_percent_of_sum_node_phase_wall"],
            "memcpy_reconciliation": summary["memcpy_totals_reconciliation"],
        }, indent=2, sort_keys=True))
        return 0
    else:
        scopes = [(rank, profile_root / f"rank_{rank:02d}.sqlite") for rank in range(16)]
        scope_name = "rank"
    for scope, path in scopes:
        report = analyze_sqlite(
            path,
            rank=scope,
            phase_patterns=pattern,
            timeline_bin_ns=500_000_000,
        )
        if "full5" not in report["phases"]:
            raise RuntimeError(f"{scope_name} {scope} has no exact static FSDP2 Full5 NVTX window")
        reports.append(report)
    aggregate = aggregate_rank_reports(reports)
    metric = aggregate["phases"]["full5"]
    categories = metric["categories"]
    summary = {
        "schema_version": "distillschedule.static_fsdp2_nsys.v1",
        "variant": args.variant,
        "rank_count": 16 if args.node_profile_dir is None else None,
        "scope_count": len(reports),
        "scope": scope_name,
        "denominator": (
            "sum(rank category interval union) / sum(rank exact Full5 NVTX wall)"
            if args.node_profile_dir is None
            else "sum(node activity interval union) / sum(node exact Full5 NVTX wall)"
        ),
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
