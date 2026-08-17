#!/usr/bin/env python3
"""Report pooled GPU activity and NCCL operation attribution for one P0 profile."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from distillschedule.profiling.nsight_activity import (
    ACTIVE,
    ALL_COPY,
    COMPUTE,
    D2H,
    H2D,
    NCCL,
    NO_TRACED_ACTIVITY,
    aggregate_rank_reports,
    analyze_sqlite,
)
from distillschedule.profiling.p0_nccl_attribution import analyze_p0_nccl_sqlite


def main() -> int:
    parser = argparse.ArgumentParser()
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--profile-dir", type=Path)
    scope.add_argument("--node-profile-dir", type=Path)
    parser.add_argument("--variant", choices=("no-gan", "gan"), required=True)
    parser.add_argument("--mesh-policy", choices=("world16-fsdp", "node-hsdp"), required=True)
    parser.add_argument("--rollout-batch-size", type=int, choices=(1, 2, 5), required=True)
    parser.add_argument("--pack-real-score-pair", choices=("on", "off"), default="off")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    policy_id = (
        f"{args.mesh_policy}-rollout-b{args.rollout_batch_size}-"
        f"realpair-{int(args.pack_real_score_pair == 'on')}"
    )
    pattern = {"full5": rf"STATIC_FSDP2_P0_FULL5/{args.variant}/{policy_id}"}
    profile_root = args.node_profile_dir or args.profile_dir
    if args.node_profile_dir is not None:
        scopes = [
            (node, profile_root / f"node_{node}" / f"static_fsdp2_p0_node{node}.sqlite")
            for node in range(2)
        ]
        scope_name = "node"
    else:
        scopes = [(rank, profile_root / f"rank_{rank:02d}.sqlite") for rank in range(16)]
        scope_name = "rank"

    reports = []
    nccl_attribution = []
    for scope_id, path in scopes:
        report = analyze_sqlite(
            path,
            rank=scope_id,
            phase_patterns=pattern,
            timeline_bin_ns=500_000_000,
        )
        if "full5" not in report["phases"]:
            raise RuntimeError(f"{scope_name} {scope_id} has no exact P0 Full5 NVTX window")
        reports.append(report)
        nccl_attribution.append(
            {
                "scope_id": scope_id,
                **analyze_p0_nccl_sqlite(
                    path,
                    variant=args.variant,
                    policy_id=policy_id,
                ),
            }
        )
    aggregate = aggregate_rank_reports(reports)
    metric = aggregate["phases"]["full5"]
    categories = metric["categories"]

    def summary(category: str) -> dict[str, float | int]:
        value = categories[category]
        return {
            "gpu_seconds": value["gpu_seconds"],
            "pooled_percent": value["pooled_percent"],
            "event_count": value.get("event_count", 0),
            "bytes": value.get("bytes", 0),
        }

    observed_rollout_all_gather_events = 0
    observed_rollout_reduce_scatter_events = 0
    for scope in nccl_attribution:
        for window in scope.get("rollout_windows", []):
            by_operation = window.get("nccl", {}).get("by_operation", {})
            observed_rollout_all_gather_events += int(
                by_operation.get("all_gather", {}).get("kernel_events", 0)
            )
            observed_rollout_reduce_scatter_events += int(
                by_operation.get("reduce_scatter", {}).get("kernel_events", 0)
            )

    output = {
        "schema_version": "distillschedule.static_fsdp2_p0_nsys.v2",
        "variant": args.variant,
        "policy": {
            "id": policy_id,
            "mesh_policy": args.mesh_policy,
            "rollout_batch_size": args.rollout_batch_size,
            "pack_real_score_pair": args.pack_real_score_pair == "on",
        },
        "scope": scope_name,
        "scope_count": len(reports),
        "gpu_wall_seconds": metric["gpu_wall_seconds"],
        "categories": {
            "compute": summary(COMPUTE),
            "nccl": summary(NCCL),
            "h2d": summary(H2D),
            "d2h": summary(D2H),
            "all_copy": summary(ALL_COPY),
            "active_union": summary(ACTIVE),
            "no_traced_activity": summary(NO_TRACED_ACTIVITY),
        },
        "nccl_operation_attribution": nccl_attribution,
        "rollout_nccl_observed": {
            "all_gather_kernel_events": observed_rollout_all_gather_events,
            "reduce_scatter_kernel_events": observed_rollout_reduce_scatter_events,
            "interpretation": (
                "Compare matched B1/B2/B5 captures. Kernel-event ratios are direct "
                "Nsight evidence of communication reduction, but one NCCL kernel event "
                "must not be labelled one FSDP materialization group."
            ),
        },
        "claim_boundary": {
            "profiled_latency_is_diagnostic_only": True,
            "category_unions_overlap_and_are_not_additive": True,
            "no_traced_activity_is_not_proof_of_cpu_idle": True,
            "formal_full5_time_comes_from_unprofiled_runs": True,
            "node_scope_is_not_rank_pooled": scope_name == "node",
            "nccl_kernel_event_is_not_collective_group": True,
            "portable_nccl_payload_bytes_unavailable": True,
        },
        "aggregate": aggregate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "variant": args.variant,
                "policy_id": policy_id,
                "compute_percent": output["categories"]["compute"]["pooled_percent"],
                "nccl_percent": output["categories"]["nccl"]["pooled_percent"],
                "h2d_percent": output["categories"]["h2d"]["pooled_percent"],
                "d2h_percent": output["categories"]["d2h"]["pooled_percent"],
                "active_percent": output["categories"]["active_union"]["pooled_percent"],
                "rollout_all_gather_kernel_events": observed_rollout_all_gather_events,
                "rollout_reduce_scatter_kernel_events": observed_rollout_reduce_scatter_events,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
