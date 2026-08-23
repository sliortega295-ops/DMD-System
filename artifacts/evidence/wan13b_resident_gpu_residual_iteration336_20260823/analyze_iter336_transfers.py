#!/usr/bin/env python3
"""Exact memcpy attribution for Iteration 336's profiled Full5 range."""

from __future__ import annotations

import argparse
import collections
import json
import sqlite3
import statistics
from pathlib import Path


COPY_KIND_NAMES = {1: "H2D", 2: "D2H", 8: "D2D"}
OLD_RESIDUAL_SIZES = {33_914_880, 14_008_320, 9_031_680}


def interval_union_ns(intervals: list[tuple[int, int]]) -> int:
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


def analyze(sqlite_path: Path, node: str) -> dict:
    connection = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    connection.execute("PRAGMA query_only=ON")
    full_ranges = connection.execute(
        "SELECT start, end FROM NVTX_EVENTS "
        "WHERE text=? AND end IS NOT NULL ORDER BY start",
        ("WAN13B_FULL5/resident-clean",),
    ).fetchall()
    if not full_ranges:
        raise RuntimeError(f"missing exact Full5 NVTX range in {sqlite_path}")
    range_start = min(int(row[0]) for row in full_ranges)
    range_end = max(int(row[1]) for row in full_ranges)

    events: dict[int, dict[int, list[tuple[int, int, int]]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    crossing_events = 0
    query = (
        "SELECT deviceId, start, end, copyKind, bytes "
        "FROM CUPTI_ACTIVITY_KIND_MEMCPY "
        "WHERE end>? AND start<?"
    )
    for device, start, end, kind, byte_count in connection.execute(
        query, (range_start, range_end)
    ):
        start, end = int(start), int(end)
        if start < range_start or end > range_end:
            crossing_events += 1
        events[int(device)][int(kind)].append(
            (max(start, range_start), min(end, range_end), int(byte_count))
        )

    per_device = []
    for device in sorted(events):
        by_kind = {}
        for kind in sorted(events[device]):
            rows = events[device][kind]
            histogram = collections.Counter(byte_count for _, _, byte_count in rows)
            by_kind[str(kind)] = {
                "copy_kind": kind,
                "direction": COPY_KIND_NAMES.get(kind, f"COPY_KIND_{kind}"),
                "event_count": len(rows),
                "bytes": sum(row[2] for row in rows),
                "sum_duration_ns": sum(row[1] - row[0] for row in rows),
                "union_duration_ns": interval_union_ns(
                    [(row[0], row[1]) for row in rows]
                ),
                "size_histogram": [
                    {"bytes_per_event": size, "event_count": count}
                    for size, count in sorted(
                        histogram.items(), key=lambda item: (-item[0] * item[1], -item[0])
                    )
                ],
                "old_residual_signature_event_count": sum(
                    count for size, count in histogram.items() if size in OLD_RESIDUAL_SIZES
                ),
                "old_residual_signature_bytes": sum(
                    size * count
                    for size, count in histogram.items()
                    if size in OLD_RESIDUAL_SIZES
                ),
            }
        per_device.append({"node": node, "device_id": device, "by_kind": by_kind})

    connection.close()
    return {
        "node": node,
        "sqlite": str(sqlite_path),
        "full5_range": {
            "start_ns": range_start,
            "end_ns": range_end,
            "wall_ns": range_end - range_start,
            "nvtx_range_count": len(full_ranges),
        },
        "crossing_range_event_count": crossing_events,
        "device_count": len(per_device),
        "per_device": per_device,
    }


def aggregate(nodes: list[dict]) -> dict:
    rows = [row for node in nodes for row in node["per_device"]]
    directions = {}
    for kind, direction in COPY_KIND_NAMES.items():
        values = []
        for row in rows:
            value = row["by_kind"].get(
                str(kind),
                {
                    "event_count": 0,
                    "bytes": 0,
                    "sum_duration_ns": 0,
                    "union_duration_ns": 0,
                    "old_residual_signature_event_count": 0,
                    "old_residual_signature_bytes": 0,
                },
            )
            values.append(value)
        directions[direction] = {
            "world_event_count": sum(value["event_count"] for value in values),
            "world_bytes": sum(value["bytes"] for value in values),
            "world_sum_duration_ns": sum(value["sum_duration_ns"] for value in values),
            "per_device_union_ns": [value["union_duration_ns"] for value in values],
            "per_device_union_median_ns": int(
                statistics.median(value["union_duration_ns"] for value in values)
            ),
            "per_device_union_min_ns": min(value["union_duration_ns"] for value in values),
            "per_device_union_max_ns": max(value["union_duration_ns"] for value in values),
            "old_residual_signature_event_count": sum(
                value["old_residual_signature_event_count"] for value in values
            ),
            "old_residual_signature_bytes": sum(
                value["old_residual_signature_bytes"] for value in values
            ),
        }
    return {
        "device_count": len(rows),
        "directions": directions,
        "reconciliation": {
            "classified_event_count": sum(
                value["event_count"]
                for row in rows
                for value in row["by_kind"].values()
            ),
            "classified_bytes": sum(
                value["bytes"]
                for row in rows
                for value in row["by_kind"].values()
            ),
            "crossing_range_event_count": sum(
                node["crossing_range_event_count"] for node in nodes
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--node0-sqlite", type=Path, required=True)
    parser.add_argument("--node1-sqlite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "schema_version": 1,
        "run_id": "iter336_profile_resident_gpu_residual_20260823",
        "scope": "exact Full5 NVTX range, all 16 devices, memcpy activity only",
        "copy_kind_mapping": COPY_KIND_NAMES,
        "nodes": [
            analyze(args.node0_sqlite, "node0"),
            analyze(args.node1_sqlite, "node1"),
        ],
    }
    result["world"] = aggregate(result["nodes"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
