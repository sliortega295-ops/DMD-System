#!/usr/bin/env python3
"""Overlap-aware attribution for the LiveWan 8G+8Score diagnostic profile.

Nsight launches one session per node and traces eight worker processes.  NVTX
``globalTid`` values and CUDA ``globalPid`` values share the process-qualified
high bits, so this analyzer first binds every rank range to its issuing process
and only then attributes that process's kernels and copies.  It never unions
activity from different GPUs into a wall-clock utilization number.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sqlite3
import statistics
from typing import Iterable


COPY_KIND = {1: "h2d", 2: "d2h", 3: "d2d", 8: "d2d", 10: "d2d"}
COLLECTIVES = ("all_gather", "reduce_scatter", "all_reduce", "p2p", "broadcast", "other_nccl")


def _tables(db: sqlite3.Connection) -> set[str]:
    return {str(row[0]) for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _columns(db: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in db.execute(f'PRAGMA table_info("{table}")')}


def _strings(db: sqlite3.Connection) -> dict[int, str]:
    return {int(key): str(value) for key, value in db.execute("SELECT id,value FROM StringIds")}


def _string(value: object, strings: dict[int, str]) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return strings.get(int(value), f"UNRESOLVED_STRING_ID_{int(value)}")
    return str(value)


def _process_id(global_tid: int) -> int:
    return (int(global_tid) >> 24) << 24


def _clip(interval: tuple[int, int], window: tuple[int, int]) -> tuple[int, int] | None:
    start, end = max(interval[0], window[0]), min(interval[1], window[1])
    return (start, end) if end > start else None


def _union(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    result: list[list[int]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return [(start, end) for start, end in result]


def _duration(intervals: Iterable[tuple[int, int]]) -> int:
    return sum(end - start for start, end in _union(intervals))


def _intersection(left: Iterable[tuple[int, int]], right: Iterable[tuple[int, int]]) -> int:
    a, b = _union(left), _union(right)
    i = j = total = 0
    while i < len(a) and j < len(b):
        total += max(0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return total


def _collective(name: str) -> str:
    lowered = name.lower()
    if "allgather" in lowered or "all_gather" in lowered:
        return "all_gather"
    if "reducescatter" in lowered or "reduce_scatter" in lowered:
        return "reduce_scatter"
    if "allreduce" in lowered or "all_reduce" in lowered:
        return "all_reduce"
    if any(token in lowered for token in ("sendrecv", "send", "recv")):
        return "p2p"
    if "broadcast" in lowered:
        return "broadcast"
    return "other_nccl"


def _metric(window: tuple[int, int], intervals: dict[str, list[tuple[int, int]]]) -> dict:
    clipped = {
        category: [part for value in values if (part := _clip(value, window))]
        for category, values in intervals.items()
    }
    compute = _duration(clipped["compute"])
    nccl = _duration(clipped["nccl"])
    copy = _duration(clipped["copy"])
    active = _duration(clipped["compute"] + clipped["nccl"] + clipped["copy"])
    wall = window[1] - window[0]
    return {
        "wall_ns": wall,
        "compute_union_ns": compute,
        "nccl_union_ns": nccl,
        "nccl_hidden_by_compute_ns": _intersection(clipped["nccl"], clipped["compute"]),
        "nccl_exposed_vs_compute_ns": nccl - _intersection(clipped["nccl"], clipped["compute"]),
        "copy_union_ns": copy,
        "copy_hidden_by_compute_ns": _intersection(clipped["copy"], clipped["compute"]),
        "copy_exposed_vs_compute_ns": copy - _intersection(clipped["copy"], clipped["compute"]),
        "active_union_ns": active,
        "no_traced_gpu_activity_ns": wall - active,
    }


def _validate_schema(db: sqlite3.Connection, node: int) -> None:
    required = {"CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_MEMCPY", "NVTX_EVENTS", "StringIds"}
    missing = sorted(required - _tables(db))
    if missing:
        raise RuntimeError(f"node{node}: missing tables {missing}")
    for table, columns in {
        "CUPTI_ACTIVITY_KIND_KERNEL": {"start", "end", "deviceId", "globalPid"},
        "CUPTI_ACTIVITY_KIND_MEMCPY": {"start", "end", "deviceId", "globalPid", "bytes", "copyKind"},
        "NVTX_EVENTS": {"text", "start", "end", "globalTid"},
    }.items():
        absent = sorted(columns - _columns(db, table))
        if absent:
            raise RuntimeError(f"node{node}: {table} missing columns {absent}")


def analyze_node(path: Path, node: int) -> dict:
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    db.execute("PRAGMA query_only=ON")
    try:
        _validate_schema(db, node)
        strings = _strings(db)
        kernel_columns = _columns(db, "CUPTI_ACTIVITY_KIND_KERNEL")
        name_column = "demangledName" if "demangledName" in kernel_columns else "shortName"
        nvtx_by_pid: dict[int, list[tuple[str, int, int]]] = defaultdict(list)
        outer_by_rank: dict[int, tuple[int, int, int]] = {}
        for text, start, end, global_tid in db.execute(
            "SELECT text,start,end,globalTid FROM NVTX_EVENTS "
            "WHERE text IS NOT NULL AND start IS NOT NULL AND end IS NOT NULL AND end>start"
        ):
            label = str(text)
            pid = _process_id(int(global_tid))
            row = (label, int(start), int(end))
            nvtx_by_pid[pid].append(row)
            prefix = "livewan/full5/pipeline/rank_"
            if label.startswith(prefix):
                rank = int(label[len(prefix):])
                if rank in outer_by_rank:
                    raise RuntimeError(f"node{node}: duplicate outer range for rank {rank}")
                outer_by_rank[rank] = (int(start), int(end), pid)

        expected_ranks = list(range(node * 8, node * 8 + 8))
        if sorted(outer_by_rank) != expected_ranks:
            raise RuntimeError(
                f"node{node}: outer rank coverage {sorted(outer_by_rank)}, expected {expected_ranks}"
            )

        ranks = []
        for rank in expected_ranks:
            outer_start, outer_end, pid = outer_by_rank[rank]
            outer = (outer_start, outer_end)
            local_device = rank % 8
            observed_devices = {
                int(row[0])
                for row in db.execute(
                    "SELECT DISTINCT deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE globalPid=?",
                    (pid,),
                )
            }
            if observed_devices != {local_device}:
                raise RuntimeError(
                    f"node{node}/rank{rank}: CUDA devices {sorted(observed_devices)}, expected [{local_device}]"
                )

            intervals = {category: [] for category in ("compute", "nccl", "copy")}
            collective_intervals = {kind: [] for kind in COLLECTIVES}
            collective_events = Counter()
            unresolved_ns = kernel_sum_ns = 0
            for start, end, raw_name in db.execute(
                f"SELECT start,end,{name_column} FROM CUPTI_ACTIVITY_KIND_KERNEL "
                "WHERE globalPid=? AND deviceId=? AND end>? AND start<? ORDER BY start",
                (pid, local_device, outer_start, outer_end),
            ):
                part = _clip((int(start), int(end)), outer)
                if not part:
                    continue
                name = _string(raw_name, strings)
                category = "nccl" if any(token in name.lower() for token in ("nccl", "nvls", "collnet")) else "compute"
                intervals[category].append(part)
                duration = part[1] - part[0]
                kernel_sum_ns += duration
                if not name or name.startswith("UNRESOLVED_STRING_ID_"):
                    unresolved_ns += duration
                if category == "nccl":
                    kind = _collective(name)
                    collective_intervals[kind].append(part)
                    collective_events[kind] += 1

            copy_rows = {direction: {"events": 0, "bytes": 0, "sum_duration_ns": 0, "intervals": []}
                         for direction in ("h2d", "d2h", "d2d", "other")}
            for start, end, num_bytes, copy_kind in db.execute(
                "SELECT start,end,bytes,copyKind FROM CUPTI_ACTIVITY_KIND_MEMCPY "
                "WHERE globalPid=? AND deviceId=? AND end>? AND start<? ORDER BY start",
                (pid, local_device, outer_start, outer_end),
            ):
                part = _clip((int(start), int(end)), outer)
                if not part:
                    continue
                direction = COPY_KIND.get(int(copy_kind), "other")
                intervals["copy"].append(part)
                copy_rows[direction]["events"] += 1
                copy_rows[direction]["bytes"] += int(num_bytes or 0)
                copy_rows[direction]["sum_duration_ns"] += part[1] - part[0]
                copy_rows[direction]["intervals"].append(part)

            intervals = {name: _union(values) for name, values in intervals.items()}
            phases = []
            role_prefix = "livewan/pipeline/g/" if rank < 8 else "livewan/pipeline/score/"
            for label, start, end in nvtx_by_pid[pid]:
                if label.startswith(role_prefix) and start >= outer_start and end <= outer_end:
                    phases.append({"label": label, **_metric((start, end), intervals)})
            phases.sort(key=lambda row: (int(row["label"].split("step_")[1].split("/")[0]), row["label"]))

            collectives = {}
            for kind in COLLECTIVES:
                union_ns = _duration(collective_intervals[kind])
                hidden_ns = _intersection(collective_intervals[kind], intervals["compute"])
                collectives[kind] = {
                    "events": collective_events[kind],
                    "union_ns": union_ns,
                    "hidden_by_compute_ns": hidden_ns,
                    "exposed_vs_compute_ns": union_ns - hidden_ns,
                }
            copies = {}
            for direction, row in copy_rows.items():
                copies[direction] = {
                    "events": row["events"],
                    "bytes": row["bytes"],
                    "sum_duration_ns": row["sum_duration_ns"],
                    "union_ns": _duration(row["intervals"]),
                }
            ranks.append({
                "node": node,
                "rank": rank,
                "device": local_device,
                "role": "generator" if rank < 8 else "score",
                "global_pid": pid,
                **_metric(outer, intervals),
                "unresolved_kernel_duration_ratio": unresolved_ns / max(kernel_sum_ns, 1),
                "collectives": collectives,
                "copies": copies,
                "phases": phases,
            })
        return {"node": node, "sqlite": str(path), "ranks": ranks}
    finally:
        db.close()


def _seconds(value: int) -> float:
    return value / 1e9


def _summaries(rows: list[dict], keys: tuple[str, ...]) -> dict:
    result = {}
    for key in keys:
        values = [int(row[key]) for row in rows]
        result[key] = {
            "sum_ns": sum(values),
            "min_ns": min(values),
            "median_ns": int(statistics.median(values)),
            "max_ns": max(values),
        }
    return result


def aggregate(nodes: list[dict]) -> dict:
    ranks = [row for node in nodes for row in node["ranks"]]
    keys = (
        "wall_ns", "compute_union_ns", "nccl_union_ns", "nccl_hidden_by_compute_ns",
        "nccl_exposed_vs_compute_ns", "copy_union_ns", "copy_exposed_vs_compute_ns",
        "no_traced_gpu_activity_ns",
    )
    by_role = {
        role: _summaries([row for row in ranks if row["role"] == role], keys)
        for role in ("generator", "score")
    }
    phases: dict[str, list[dict]] = defaultdict(list)
    for rank in ranks:
        for phase in rank["phases"]:
            phases[phase["label"]].append(phase)
    phase_summary = []
    for label, rows in sorted(phases.items()):
        if len(rows) != 8:
            raise RuntimeError(f"phase {label}: expected 8 process-qualified ranges, found {len(rows)}")
        phase_summary.append({"label": label, "rank_count": len(rows), **{
            key: {
                "sum_ns": sum(int(row[key]) for row in rows),
                "median_ns": int(statistics.median(int(row[key]) for row in rows)),
                "max_ns": max(int(row[key]) for row in rows),
            }
            for key in keys
        }})
    collective_summary = {}
    for kind in COLLECTIVES:
        collective_summary[kind] = {
            field: sum(int(row["collectives"][kind][field]) for row in ranks)
            for field in ("events", "union_ns", "hidden_by_compute_ns", "exposed_vs_compute_ns")
        }
    copies = {}
    for direction in ("h2d", "d2h", "d2d", "other"):
        copies[direction] = {
            field: sum(int(row["copies"][direction][field]) for row in ranks)
            for field in ("events", "bytes", "sum_duration_ns", "union_ns")
        }
    return {
        "rank_count": len(ranks),
        "per_rank": ranks,
        "by_role": by_role,
        "phases": phase_summary,
        "collectives": collective_summary,
        "copies": copies,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--node0-sqlite", required=True, type=Path)
    parser.add_argument("--node1-sqlite", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    nodes = [analyze_node(args.node0_sqlite, 0), analyze_node(args.node1_sqlite, 1)]
    result = aggregate(nodes)
    valid = (
        result["rank_count"] == 16
        and max(row["unresolved_kernel_duration_ratio"] for row in result["per_rank"]) < 0.01
    )
    document = {
        "schema_version": "distillschedule.livewan_crossscale_nsys.v1",
        "status": "PASS" if valid else "FAIL_PROFILE_VALIDITY",
        "diagnostic_only": True,
        "formal_speedup_eligible": False,
        "definitions": {
            "denominator": "sum of process-qualified per-rank GPU phase walls; GPU-seconds, not global wall-clock",
            "nccl_exposed_vs_compute": "NCCL interval union minus overlap with non-NCCL CUDA kernels on the same rank/device",
            "no_traced_gpu_activity": "phase wall not covered by CUDA kernel or memcpy activity; not proof of CPU idle",
        },
        "nodes": nodes,
        "aggregate": result,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "livewan_pipeline_nsys_attribution.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rank_rows = []
    for row in result["per_rank"]:
        rank_rows.append({
            "rank": row["rank"], "role": row["role"],
            "wall_s": _seconds(row["wall_ns"]),
            "compute_s": _seconds(row["compute_union_ns"]),
            "nccl_s": _seconds(row["nccl_union_ns"]),
            "nccl_exposed_s": _seconds(row["nccl_exposed_vs_compute_ns"]),
            "copy_s": _seconds(row["copy_union_ns"]),
            "no_traced_s": _seconds(row["no_traced_gpu_activity_ns"]),
            "unresolved_kernel_ratio": row["unresolved_kernel_duration_ratio"],
        })
    _write_csv(args.output_dir / "per_rank_attribution.csv", rank_rows)
    phase_rows = []
    for phase in result["phases"]:
        phase_rows.append({
            "phase": phase["label"],
            "median_wall_s": _seconds(phase["wall_ns"]["median_ns"]),
            "median_compute_s": _seconds(phase["compute_union_ns"]["median_ns"]),
            "median_nccl_s": _seconds(phase["nccl_union_ns"]["median_ns"]),
            "median_nccl_exposed_s": _seconds(phase["nccl_exposed_vs_compute_ns"]["median_ns"]),
            "median_copy_s": _seconds(phase["copy_union_ns"]["median_ns"]),
            "median_no_traced_s": _seconds(phase["no_traced_gpu_activity_ns"]["median_ns"]),
        })
    _write_csv(args.output_dir / "phase_attribution.csv", phase_rows)
    generator_wall = _seconds(result["by_role"]["generator"]["wall_ns"]["median_ns"])
    score_wall = _seconds(result["by_role"]["score"]["wall_ns"]["median_ns"])
    lines = [
        "# LiveWan cross-scale 8G+8Score Nsight attribution",
        "", f"Status: **{document['status']}** (diagnostic only).", "",
        f"Median Generator-rank outer wall: {generator_wall:.6f} s.",
        f"Median Score-rank outer wall: {score_wall:.6f} s.", "",
        "Profiled wall is not a formal speed denominator. Exposed NCCL is an overlap-aware upper bound, not automatically removable critical-path time.",
        "", "| Phase | Median wall (s) | Compute (s) | NCCL (s) | NCCL exposed (s) | No traced GPU activity (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in phase_rows:
        lines.append(
            f"| `{row['phase']}` | {row['median_wall_s']:.6f} | {row['median_compute_s']:.6f} | "
            f"{row['median_nccl_s']:.6f} | {row['median_nccl_exposed_s']:.6f} | {row['median_no_traced_s']:.6f} |"
        )
    (args.output_dir / "livewan_pipeline_nsys_attribution.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": document["status"],
        "generator_median_wall_s": generator_wall,
        "score_median_wall_s": score_wall,
        "max_unresolved_kernel_ratio": max(
            row["unresolved_kernel_duration_ratio"] for row in result["per_rank"]
        ),
    }, indent=2))
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
