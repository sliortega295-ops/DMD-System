#!/usr/bin/env python3
"""Summarize per-GPU busy, compute, NCCL, and idle time in W16 pool Nsight DBs."""
from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
from pathlib import Path
from typing import Iterable


def merge_interval_nanoseconds(intervals: Iterable[tuple[int, int]]) -> int:
    total = 0
    current_start: int | None = None
    current_end: int | None = None
    for start, end in intervals:
        if end <= start:
            continue
        if current_start is None:
            current_start, current_end = start, end
        elif start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    if current_start is not None:
        assert current_end is not None
        total += current_end - current_start
    return total


def classify_kernel(name: str) -> str:
    return "nccl" if "nccl" in name.lower() else "compute"


def _profile_window(connection: sqlite3.Connection) -> tuple[int, int]:
    row = connection.execute(
        """
        SELECT n.start, n.end
        FROM NVTX_EVENTS AS n
        LEFT JOIN StringIds AS s ON s.id = n.textId
        WHERE COALESCE(n.text, s.value) = 'distillschedule_profile'
          AND n.end IS NOT NULL
        ORDER BY n.end - n.start DESC
        LIMIT 1
        """
    ).fetchone()
    if row is None:
        raise ValueError("Nsight database has no closed distillschedule_profile range")
    return int(row[0]), int(row[1])


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "min": min(values),
        "max": max(values),
    }


def analyze_report(path: Path) -> dict[str, object]:
    connection = sqlite3.connect(str(path))
    try:
        window_start, window_end = _profile_window(connection)
        duration_ns = window_end - window_start
        device_ids = [
            int(row[0])
            for row in connection.execute(
                "SELECT DISTINCT deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY deviceId"
            )
        ]
        devices: list[dict[str, object]] = []
        for device_id in device_ids:
            all_intervals: list[tuple[int, int]] = []
            class_intervals: dict[str, list[tuple[int, int]]] = {
                "compute": [],
                "nccl": [],
            }
            class_sum_ns = {"compute": 0, "nccl": 0}
            class_count = {"compute": 0, "nccl": 0}
            query = """
                SELECT MAX(k.start, ?), MIN(k.end, ?), strings.value
                FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
                JOIN StringIds AS strings ON strings.id = k.shortName
                WHERE k.deviceId = ? AND k.end > ? AND k.start < ?
                ORDER BY k.start
            """
            for start, end, kernel_name in connection.execute(
                query,
                (window_start, window_end, device_id, window_start, window_end),
            ):
                start, end = int(start), int(end)
                if end <= start:
                    continue
                interval = (start, end)
                kind = classify_kernel(str(kernel_name))
                all_intervals.append(interval)
                class_intervals[kind].append(interval)
                class_sum_ns[kind] += end - start
                class_count[kind] += 1
            kernel_busy_ns = merge_interval_nanoseconds(all_intervals)
            compute_busy_ns = merge_interval_nanoseconds(class_intervals["compute"])
            nccl_busy_ns = merge_interval_nanoseconds(class_intervals["nccl"])
            devices.append(
                {
                    "device_id": device_id,
                    "kernel_busy_seconds": kernel_busy_ns / 1e9,
                    "kernel_busy_fraction": kernel_busy_ns / duration_ns,
                    "idle_fraction": 1.0 - kernel_busy_ns / duration_ns,
                    "compute_union_seconds": compute_busy_ns / 1e9,
                    "compute_union_fraction": compute_busy_ns / duration_ns,
                    "nccl_union_seconds": nccl_busy_ns / 1e9,
                    "nccl_union_fraction": nccl_busy_ns / duration_ns,
                    "compute_kernel_sum_seconds": class_sum_ns["compute"] / 1e9,
                    "nccl_kernel_sum_seconds": class_sum_ns["nccl"] / 1e9,
                    "compute_kernel_count": class_count["compute"],
                    "nccl_kernel_count": class_count["nccl"],
                }
            )
        if not devices:
            raise ValueError("Nsight database has no CUDA kernel rows")
        top_kernels = [
            {
                "name": str(name),
                "count": int(count),
                "sum_seconds": float(sum_seconds),
                "kind": classify_kernel(str(name)),
            }
            for name, count, sum_seconds in connection.execute(
                """
                SELECT strings.value, COUNT(*), SUM(k.end-k.start)/1e9
                FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
                JOIN StringIds AS strings ON strings.id = k.shortName
                WHERE k.end > ? AND k.start < ?
                GROUP BY strings.value
                ORDER BY SUM(k.end-k.start) DESC
                LIMIT 20
                """,
                (window_start, window_end),
            )
        ]
        result: dict[str, object] = {
            "sqlite_path": str(path.resolve()),
            "sqlite_bytes": path.stat().st_size,
            "capture_seconds": duration_ns / 1e9,
            "devices": devices,
            "top_kernels": top_kernels,
        }
        for field in (
            "kernel_busy_fraction",
            "idle_fraction",
            "compute_union_fraction",
            "nccl_union_fraction",
        ):
            result[f"{field}_summary"] = _summary(
                [float(device[field]) for device in devices]
            )
        return result
    finally:
        connection.close()


def _parse_report(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("report must be LABEL=/path/to/report.sqlite")
    return label, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        action="append",
        type=_parse_report,
        required=True,
        help="repeatable LABEL=/path/to/report.sqlite",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    labels = [label for label, _ in args.report]
    if len(labels) != len(set(labels)):
        parser.error("report labels must be unique")
    value = {
        "schema_version": "distillschedule.w16_pool_nsys_analysis.v1",
        "evidence_label": "Measured diagnostic profiler data",
        "reports": {
            label: analyze_report(path) for label, path in args.report
        },
        "claim_boundary": (
            "Profiler time is diagnostic only; formal speed claims use the separate "
            "unprofiled validation manifests. Busy-union lower bounds ignore CPU, "
            "dependency, and synchronization constraints."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "output": str(args.output), "reports": labels}))


if __name__ == "__main__":
    main()
