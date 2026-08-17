"""Operation-level NCCL attribution for static-FSDP2 P0 Nsight captures.

Nsight's CUDA kernel trace exposes NCCL kernel names and exact time intervals but
not a portable collective payload-byte field on every version.  We therefore
report observed kernel events/unions separately from analytical FSDP byte
estimates and never equate one kernel with one materialization group.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Sequence

from .nsight_activity import (
    NCCL,
    ActivityInterval,
    Interval,
    PhaseWindow,
    interval_union_ns,
    merge_intervals,
    read_cuda_activities,
    read_nvtx_phase_windows,
)


NCCL_OPERATION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("all_gather", re.compile(r"all[_ ]?gather|allgather", re.I)),
    ("reduce_scatter", re.compile(r"reduce[_ ]?scatter|reducescatter", re.I)),
    ("all_reduce", re.compile(r"all[_ ]?reduce|allreduce", re.I)),
    ("broadcast", re.compile(r"broadcast|bcast", re.I)),
    ("send_recv", re.compile(r"send|recv|p2p", re.I)),
)


def classify_nccl_kernel(name: str) -> str:
    for label, pattern in NCCL_OPERATION_PATTERNS:
        if pattern.search(name):
            return label
    return "other_nccl"


def _clips(activity: ActivityInterval, window: Interval) -> bool:
    return activity.interval.clip(window) is not None


def _summarize_activities(
    activities: Sequence[ActivityInterval], windows: Sequence[Interval]
) -> dict[str, Any]:
    windows = merge_intervals(windows)
    rows = [
        activity
        for activity in activities
        if activity.category == NCCL and any(_clips(activity, window) for window in windows)
    ]
    by_kind: dict[str, list[ActivityInterval]] = {}
    for row in rows:
        by_kind.setdefault(classify_nccl_kernel(row.name), []).append(row)

    def summarize(selected: Iterable[ActivityInterval]) -> dict[str, Any]:
        selected = list(selected)
        clipped: list[Interval] = []
        contributing_events = 0
        for activity in selected:
            contributed = False
            for window in windows:
                part = activity.interval.clip(window)
                if part is not None:
                    clipped.append(part)
                    contributed = True
            if contributed:
                contributing_events += 1
        return {
            "kernel_events": contributing_events,
            "summed_kernel_seconds": sum(value.duration_ns for value in clipped) / 1e9,
            "interval_union_seconds": interval_union_ns(clipped) / 1e9,
            "payload_bytes": None,
            "payload_bytes_status": "UNAVAILABLE_IN_PORTABLE_CUDA_KERNEL_TRACE",
        }

    return {
        "all_nccl": summarize(rows),
        "by_operation": {
            kind: summarize(selected) for kind, selected in sorted(by_kind.items())
        },
        "unique_kernel_names": sorted({row.name for row in rows}),
    }


def analyze_p0_nccl_sqlite(
    path: str | Path,
    *,
    variant: str,
    policy_id: str,
) -> dict[str, Any]:
    path = Path(path)
    with sqlite3.connect(str(path)) as connection:
        activities = read_cuda_activities(connection)
        phases = read_nvtx_phase_windows(
            connection,
            {
                "full5": rf"STATIC_FSDP2_P0_FULL5/{re.escape(variant)}/{re.escape(policy_id)}",
                "rollout_batch": r"FSDP_BATCHED_GENERATOR_ROLLOUT/batch[0-9]+/B[125]",
                "critic": r"FSDP_(PRE_FORWARD_ALL_GATHER|PRE_BACKWARD_ALL_GATHER)/critic|OPTIMIZER_STEP/critic",
                "generator_train": r"FSDP_(PRE_FORWARD_ALL_GATHER|PRE_BACKWARD_ALL_GATHER)/generator|OPTIMIZER_STEP/generator",
                "score": r"P0_SCORE/",
            },
        )

    by_name: dict[str, list[PhaseWindow]] = {}
    for phase in phases:
        by_name.setdefault(phase.name, []).append(phase)
    if not by_name.get("full5"):
        raise RuntimeError(f"{path} has no exact P0 Full5 NVTX window")

    result: dict[str, Any] = {
        "schema_version": "distillschedule.p0_nccl_attribution.v1",
        "sqlite": str(path),
        "variant": variant,
        "policy_id": policy_id,
        "claim_boundary": {
            "kernel_event_is_not_collective_group": True,
            "kernel_trace_has_no_portable_payload_bytes": True,
            "compare_B1_B2_B5_event_counts_and_unions_under_same_capture_policy": True,
        },
        "phases": {},
        "rollout_windows": [],
    }
    for name in ("full5", "critic", "generator_train", "score"):
        windows = by_name.get(name, [])
        result["phases"][name] = (
            _summarize_activities(activities, [value.interval for value in windows])
            if windows
            else {"status": "NO_NAMED_NVTX_WINDOW"}
        )

    for phase in by_name.get("rollout_batch", []):
        label = phase.source.removeprefix("nvtx:")
        match = re.search(r"batch([0-9]+)/B([125])", label)
        batch_index = int(match.group(1)) if match else None
        physical_batch = int(match.group(2)) if match else None
        result["rollout_windows"].append(
            {
                "label": label,
                "batch_index": batch_index,
                "physical_batch": physical_batch,
                "wall_seconds": phase.interval.duration_ns / 1e9,
                "nccl": _summarize_activities(activities, [phase.interval]),
            }
        )
    result["rollout_windows"].sort(
        key=lambda value: (-1 if value["batch_index"] is None else value["batch_index"])
    )
    return result


__all__ = ["analyze_p0_nccl_sqlite", "classify_nccl_kernel"]
