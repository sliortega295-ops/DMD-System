#!/usr/bin/env python3
"""Representative GPU0 timeline decomposition for Iteration 336."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def union_ns(intervals: list[tuple[int, int]]) -> int:
    intervals = sorted((start, end) for start, end in intervals if end > start)
    if not intervals:
        return 0
    total = 0
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return total + current_end - current_start


def clipped(intervals: list[tuple[int, int]], start: int, end: int) -> list[tuple[int, int]]:
    return [
        (max(left, start), min(right, end))
        for left, right in intervals
        if right > start and left < end
    ]


def intersection_union_ns(
    left: list[tuple[int, int]], right: list[tuple[int, int]]
) -> int:
    left = sorted(left)
    right = sorted(right)
    intersections: list[tuple[int, int]] = []
    i = j = 0
    while i < len(left) and j < len(right):
        start = max(left[i][0], right[j][0])
        end = min(left[i][1], right[j][1])
        if end > start:
            intersections.append((start, end))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return union_ns(intersections)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    connection = sqlite3.connect(f"file:{args.sqlite}?mode=ro", uri=True)
    connection.execute("PRAGMA query_only=ON")
    full = connection.execute(
        "SELECT start, end FROM NVTX_EVENTS WHERE text=? AND end IS NOT NULL ORDER BY start LIMIT 1",
        ("WAN13B_FULL5/resident-clean",),
    ).fetchone()
    if full is None:
        raise RuntimeError("missing WAN13B_FULL5/resident-clean")
    full_start, full_end = map(int, full)
    phases = [
        (str(text), int(start), int(end))
        for start, end, text in connection.execute(
            "SELECT start, end, text FROM NVTX_EVENTS "
            "WHERE end IS NOT NULL AND (text='WAN13B/G_UPDATE' "
            "OR text='WAN13B/G_ROLLOUTS' OR text LIKE 'WAN13B/C_UPDATE/%' "
            "OR text LIKE 'WAN13B/EMA/%') ORDER BY start"
        )
        if int(end) > full_start and int(start) < full_end
    ]
    compute: list[tuple[int, int]] = []
    nccl: list[tuple[int, int]] = []
    nccl_names: dict[str, dict[str, int]] = {}
    for start, end, name in connection.execute(
        "SELECT k.start, k.end, COALESCE(s.value,'') "
        "FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        "LEFT JOIN StringIds s ON k.shortName=s.id "
        "WHERE k.deviceId=0 AND k.end>? AND k.start<?",
        (full_start, full_end),
    ):
        start, end, name = int(start), int(end), str(name)
        interval = (max(start, full_start), min(end, full_end))
        if "nccl" in name.lower():
            nccl.append(interval)
            values = nccl_names.setdefault(name, {"event_count": 0, "sum_duration_ns": 0})
            values["event_count"] += 1
            values["sum_duration_ns"] += interval[1] - interval[0]
        else:
            compute.append(interval)
    copies: dict[int, list[tuple[int, int]]] = {1: [], 2: [], 8: []}
    copy_stats: dict[int, dict[str, int]] = {
        kind: {"event_count": 0, "bytes": 0, "sum_duration_ns": 0}
        for kind in copies
    }
    for start, end, kind, byte_count in connection.execute(
        "SELECT start, end, copyKind, bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY "
        "WHERE deviceId=0 AND end>? AND start<?",
        (full_start, full_end),
    ):
        kind = int(kind)
        if kind not in copies:
            continue
        interval = (max(int(start), full_start), min(int(end), full_end))
        copies[kind].append(interval)
        copy_stats[kind]["event_count"] += 1
        copy_stats[kind]["bytes"] += int(byte_count)
        copy_stats[kind]["sum_duration_ns"] += interval[1] - interval[0]

    def range_stats(name: str, start: int, end: int) -> dict:
        all_activity = compute + nccl + copies[1] + copies[2] + copies[8]
        return {
            "phase": name,
            "wall_ns": end - start,
            "compute_union_ns": union_ns(clipped(compute, start, end)),
            "nccl_union_ns": union_ns(clipped(nccl, start, end)),
            "h2d_union_ns": union_ns(clipped(copies[1], start, end)),
            "d2h_union_ns": union_ns(clipped(copies[2], start, end)),
            "d2d_union_ns": union_ns(clipped(copies[8], start, end)),
            "active_union_ns": union_ns(clipped(all_activity, start, end)),
            "no_traced_gpu_activity_ns": (end - start)
            - union_ns(clipped(all_activity, start, end)),
        }

    result = {
        "schema_version": 1,
        "run_id": "iter336_profile_resident_gpu_residual_20260823",
        "scope": "node0 representative rank/device0; exact Full5 NVTX range",
        "sqlite": str(args.sqlite),
        "full5": range_stats("FULL5", full_start, full_end),
        "phases": [range_stats(name, start, end) for name, start, end in phases],
        "copy_stats": {
            {1: "H2D", 2: "D2H", 8: "D2D"}[kind]: {
                **values,
                "union_duration_ns": union_ns(copies[kind]),
            }
            for kind, values in copy_stats.items()
        },
        "nccl": {
            "event_count": len(nccl),
            "union_duration_ns": union_ns(nccl),
            "compute_overlap_ns": intersection_union_ns(nccl, compute),
            "kernels": nccl_names,
        },
    }
    connection.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
