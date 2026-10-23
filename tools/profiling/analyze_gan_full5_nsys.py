#!/usr/bin/env python3
"""Summarize exact-range GAN Full5 Nsight SQLite reports.

The reports are diagnostic only.  Wall time comes from the validated per-rank
range receipts rather than from a profiled process launch or the first/last GPU
event, so idle time is not silently discarded.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Iterator


def merge_interval_nanoseconds(intervals: Iterable[tuple[int, int]]) -> int:
    total = 0
    current_start: int | None = None
    current_end: int | None = None
    for start, end in intervals:
        start, end = int(start), int(end)
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


def _receipt_wall_seconds(root: Path, node_rank: int) -> tuple[float, list[int]]:
    expected = list(range(node_rank * 8, node_rank * 8 + 8))
    starts: list[int] = []
    stops: list[int] = []
    observed: list[int] = []
    for rank in expected:
        path = root / f"rank_{rank:03d}.json"
        value = json.loads(path.read_text())
        if value.get("status") != "PASS" or value.get("logical_window") != [1000, 1004]:
            raise ValueError(f"invalid exact-range receipt: {path}")
        observed.append(int(value["rank"]))
        starts.append(int(value["start_unix_ns"]))
        stops.append(int(value["stop_unix_ns"]))
    if observed != expected:
        raise ValueError(f"node {node_rank} receipt ranks drift: {observed}")
    # This is the same global-range definition as formal timing, applied to
    # the eight ranks captured by this node-local report.  max(elapsed) is not
    # equivalent when rank start/stop timestamps are skewed.
    return (max(stops) - min(starts)) / 1e9, observed


def _interval_query(
    connection: sqlite3.Connection,
    query: str,
    parameters: tuple[object, ...],
) -> Iterator[tuple[int, int]]:
    for start, end in connection.execute(query, parameters):
        yield int(start), int(end)


def _union_seconds(
    connection: sqlite3.Connection,
    query: str,
    parameters: tuple[object, ...],
) -> float:
    return merge_interval_nanoseconds(
        _interval_query(connection, query, parameters)
    ) / 1e9


def analyze_report(
    sqlite_path: Path,
    *,
    receipt_root: Path,
    node_rank: int,
) -> dict[str, object]:
    wall_seconds, receipt_ranks = _receipt_wall_seconds(receipt_root, node_rank)
    connection = sqlite3.connect(str(sqlite_path))
    try:
        devices = [
            int(row[0])
            for row in connection.execute(
                """
                SELECT deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL
                UNION SELECT deviceId FROM CUPTI_ACTIVITY_KIND_MEMCPY
                ORDER BY deviceId
                """
            )
        ]
        per_device: list[dict[str, object]] = []
        for device in devices:
            kernel_query = """
                SELECT start,end FROM CUPTI_ACTIVITY_KIND_KERNEL
                WHERE deviceId=? ORDER BY start,end
            """
            compute_query = """
                SELECT k.start,k.end FROM CUPTI_ACTIVITY_KIND_KERNEL k
                JOIN StringIds s ON s.id=k.shortName
                WHERE k.deviceId=? AND lower(s.value) NOT LIKE '%nccl%'
                ORDER BY k.start,k.end
            """
            nccl_query = """
                SELECT k.start,k.end FROM CUPTI_ACTIVITY_KIND_KERNEL k
                JOIN StringIds s ON s.id=k.shortName
                WHERE k.deviceId=? AND lower(s.value) LIKE '%nccl%'
                ORDER BY k.start,k.end
            """
            copy_query = """
                SELECT start,end FROM CUPTI_ACTIVITY_KIND_MEMCPY
                WHERE deviceId=? ORDER BY start,end
            """
            busy_query = """
                SELECT start,end FROM (
                  SELECT start,end FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE deviceId=?
                  UNION ALL
                  SELECT start,end FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE deviceId=?
                ) ORDER BY start,end
            """
            compute_copy_query = """
                SELECT start,end FROM (
                  SELECT k.start,k.end FROM CUPTI_ACTIVITY_KIND_KERNEL k
                  JOIN StringIds s ON s.id=k.shortName
                  WHERE k.deviceId=? AND lower(s.value) NOT LIKE '%nccl%'
                  UNION ALL
                  SELECT start,end FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE deviceId=?
                ) ORDER BY start,end
            """
            kernel = _union_seconds(connection, kernel_query, (device,))
            compute = _union_seconds(connection, compute_query, (device,))
            nccl = _union_seconds(connection, nccl_query, (device,))
            copy = _union_seconds(connection, copy_query, (device,))
            busy = _union_seconds(connection, busy_query, (device, device))
            compute_or_copy = _union_seconds(
                connection, compute_copy_query, (device, device)
            )
            copy_rows = [
                {
                    "copy_kind": str(label),
                    "count": int(count),
                    "bytes": int(total_bytes or 0),
                    "sum_seconds": float(total_ns or 0) / 1e9,
                }
                for label, count, total_bytes, total_ns in connection.execute(
                    """
                    SELECT e.label,COUNT(*),SUM(m.bytes),SUM(m.end-m.start)
                    FROM CUPTI_ACTIVITY_KIND_MEMCPY m
                    JOIN ENUM_CUDA_MEMCPY_OPER e ON e.id=m.copyKind
                    WHERE m.deviceId=? GROUP BY e.label ORDER BY SUM(m.end-m.start) DESC
                    """,
                    (device,),
                )
            ]
            per_device.append(
                {
                    "device_id": device,
                    "wall_seconds": wall_seconds,
                    "kernel_union_seconds": kernel,
                    "compute_union_seconds": compute,
                    "nccl_union_seconds": nccl,
                    "copy_union_seconds": copy,
                    "gpu_busy_union_seconds": busy,
                    "gpu_busy_fraction": min(1.0, busy / wall_seconds),
                    "gpu_idle_fraction": max(0.0, 1.0 - busy / wall_seconds),
                    "compute_copy_overlap_seconds": max(
                        0.0, compute + copy - compute_or_copy
                    ),
                    "copy_by_kind": copy_rows,
                }
            )
        top_kernels = [
            {
                "name": str(name),
                "count": int(count),
                "sum_seconds": float(total_ns) / 1e9,
                "kind": "nccl" if "nccl" in str(name).lower() else "compute",
            }
            for name, count, total_ns in connection.execute(
                """
                SELECT s.value,COUNT(*),SUM(k.end-k.start)
                FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.shortName
                GROUP BY s.value ORDER BY SUM(k.end-k.start) DESC LIMIT 20
                """
            )
        ]
        top_cuda_api = [
            {
                "name": str(name),
                "count": int(count),
                "sum_seconds": float(total_ns) / 1e9,
            }
            for name, count, total_ns in connection.execute(
                """
                SELECT s.value,COUNT(*),SUM(r.end-r.start)
                FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId
                GROUP BY s.value ORDER BY SUM(r.end-r.start) DESC LIMIT 20
                """
            )
        ]
        top_nvtx = [
            {
                "name": str(name),
                "count": int(count),
                "sum_seconds": float(total_ns) / 1e9,
            }
            for name, count, total_ns in connection.execute(
                """
                SELECT COALESCE(n.text,s.value),COUNT(*),SUM(n.end-n.start)
                FROM NVTX_EVENTS n LEFT JOIN StringIds s ON s.id=n.textId
                WHERE n.end IS NOT NULL
                GROUP BY COALESCE(n.text,s.value)
                ORDER BY SUM(n.end-n.start) DESC LIMIT 40
                """
            )
        ]
        return {
            "sqlite_path": str(sqlite_path),
            "sqlite_bytes": sqlite_path.stat().st_size,
            "node_rank": node_rank,
            "receipt_ranks": receipt_ranks,
            "wall_seconds_from_max_exact_receipt": wall_seconds,
            "devices": per_device,
            "top_kernels": top_kernels,
            "top_cuda_api": top_cuda_api,
            "top_nvtx": top_nvtx,
        }
    finally:
        connection.close()


def _report(value: str) -> tuple[str, int, Path, Path]:
    fields = value.split("=", 1)
    if len(fields) != 2:
        raise argparse.ArgumentTypeError(
            "report must be LABEL=NODE_RANK,SQLITE,RECEIPT_DIR"
        )
    label, payload = fields
    values = payload.split(",", 2)
    if not label or len(values) != 3:
        raise argparse.ArgumentTypeError(
            "report must be LABEL=NODE_RANK,SQLITE,RECEIPT_DIR"
        )
    return label, int(values[0]), Path(values[1]), Path(values[2])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="append", type=_report, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    labels = [value[0] for value in args.report]
    if len(labels) != len(set(labels)):
        parser.error("report labels must be unique")
    output = {
        "schema_version": "distillschedule.gan_full5_nsys_analysis.v1",
        "status": "PASS",
        "evidence_label": "Measured diagnostic profiler data",
        "formal_speedup_eligible": False,
        "reports": {
            label: analyze_report(sqlite, receipt_root=receipts, node_rank=node)
            for label, node, sqlite, receipts in args.report
        },
        "claim_boundary": (
            "Nsight is diagnostic only. Wall time uses exact range receipts; CUDA API "
            "sums overlap across eight processes and are not additive wall time."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "output": str(args.output)}))


if __name__ == "__main__":
    main()
