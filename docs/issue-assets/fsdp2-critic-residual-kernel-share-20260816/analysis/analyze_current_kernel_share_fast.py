#!/usr/bin/env python3
"""Exact per-rank compute/NCCL union for the current Iteration-304 trace.

This is deliberately narrower than the older full wall-attribution analyzer:
it stages only kernel start/end/name-class rows, then computes interval unions
inside the exact Full5 and phase NVTX windows.  Category unions overlap and are
therefore not additive.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sqlite3
import statistics
import time
from collections import defaultdict
from pathlib import Path


def load_marker_module(path: Path):
    spec = importlib.util.spec_from_file_location("iter304_markers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import marker parser: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stats(values):
    xs = list(values)
    return {
        "n": len(xs),
        "min": min(xs),
        "median": statistics.median(xs),
        "mean": statistics.mean(xs),
        "max": max(xs),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--node", type=int, choices=(0, 1), required=True)
    ap.add_argument("--sqlite", type=Path, required=True)
    ap.add_argument("--marker-parser", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    t0 = time.time()
    markers = load_marker_module(args.marker_parser)
    c = sqlite3.connect(f"file:{args.sqlite}?mode=ro", uri=True)
    c.execute("pragma temp_store=memory")
    c.execute("pragma cache_size=-8000000")
    gps = [
        (int(gp), int(device))
        for gp, device in c.execute(
            "select globalPid,deviceId from CUPTI_ACTIVITY_KIND_KERNEL "
            "group by globalPid,deviceId order by deviceId"
        )
    ]
    if len(gps) != 8:
        raise RuntimeError(f"expected eight GPU processes, got {gps}")

    c.execute(
        "create temp table windows("
        "gp integer, global_rank integer, phase text, window_index integer, "
        "start integer, end integer, wall_ns integer)"
    )
    rows = []
    marker_counts = {}
    for gp, device in gps:
        rank = args.node * 8 + device
        parsed = markers.load_markers(c, gp)
        lo, hi = parsed["full_range"]
        if lo is None or hi is None:
            raise RuntimeError(f"rank {rank} has no Full5 marker")
        rows.append((gp, rank, "FULL5", 0, lo, hi, hi - lo))
        for phase, values in parsed["windows"].items():
            for value in values:
                rows.append(
                    (
                        gp,
                        rank,
                        phase,
                        int(value["window_index"]),
                        int(value["start"]),
                        int(value["end"]),
                        int(value["end"]) - int(value["start"]),
                    )
                )
        marker_counts[str(rank)] = parsed["counts"]
    c.executemany("insert into windows values(?,?,?,?,?,?,?)", rows)
    c.execute("create index windows_gp_start on windows(gp,start,end)")

    # One source-table pass.  Keeping only four scalar columns makes the
    # interval-union query much cheaper than the broad gap/copy analyzer.
    c.execute(
        "create temp table tagged_kernel as "
        "select k.globalPid gp,k.start start,k.end end,"
        "case "
        " when lower(d.value) like '%nccl%allgather%' then 'all-gather'"
        " when lower(d.value) like '%nccl%reducescatter%' then 'reduce-scatter'"
        " when lower(d.value) like '%nccl%allreduce%' then 'all-reduce'"
        " when lower(d.value) like '%nccl%broadcast%' then 'broadcast'"
        " when lower(d.value) like '%nccl%send%' or lower(d.value) like '%nccl%recv%' then 'P2P-send-recv'"
        " when lower(d.value) like '%nccl%' or lower(d.value) like '%nvls%' or lower(d.value) like '%collnet%' then 'NCCL-other'"
        " else 'compute' end category "
        "from CUPTI_ACTIVITY_KIND_KERNEL k "
        "join StringIds d on d.id=k.demangledName"
    )
    c.execute("create index tagged_kernel_gp_start on tagged_kernel(gp,start,end)")
    print(f"node {args.node}: staged tagged kernels in {time.time()-t0:.1f}s", flush=True)

    query = """
    with clipped as (
      select w.global_rank,w.phase,w.window_index,w.wall_ns,
             t.category,max(t.start,w.start) start,min(t.end,w.end) end
      from windows w indexed by windows_gp_start
      join tagged_kernel t indexed by tagged_kernel_gp_start
        on t.gp=w.gp and t.start<w.end and t.end>w.start
    ), expanded as (
      select * from clipped where end>start
      union all
      select global_rank,phase,window_index,wall_ns,'NCCL-total',start,end
      from clipped where category!='compute' and end>start
    ), ordered as (
      select *,max(end) over(
        partition by global_rank,phase,window_index,category
        order by start,end rows between unbounded preceding and 1 preceding
      ) previous_max_end
      from expanded
    ), grouped as (
      select *,sum(case when previous_max_end is null or start>previous_max_end then 1 else 0 end)
        over(partition by global_rank,phase,window_index,category order by start,end) interval_group
      from ordered
    ), merged as (
      select global_rank,phase,window_index,wall_ns,category,interval_group,
             min(start) start,max(end) end
      from grouped
      group by global_rank,phase,window_index,category,interval_group
    )
    select global_rank,phase,window_index,wall_ns,category,sum(end-start) union_ns
    from merged
    group by global_rank,phase,window_index,wall_ns,category
    order by global_rank,phase,window_index,category
    """
    union_rows = [
        {
            "global_rank": int(rank),
            "phase": str(phase),
            "window_index": int(index),
            "wall_ns": int(wall),
            "category": str(category),
            "union_ns": int(union_ns),
        }
        for rank, phase, index, wall, category, union_ns in c.execute(query)
    ]
    c.close()
    print(f"node {args.node}: union query complete in {time.time()-t0:.1f}s", flush=True)

    by_key = defaultdict(dict)
    for row in union_rows:
        key = (row["global_rank"], row["phase"], row["window_index"], row["wall_ns"])
        by_key[key][row["category"]] = row["union_ns"]
    windows = []
    for (rank, phase, index, wall), values in sorted(by_key.items()):
        windows.append(
            {
                "global_rank": rank,
                "phase": phase,
                "window_index": index,
                "wall_ns": wall,
                "categories": values,
                "compute_percent": 100.0 * values.get("compute", 0) / wall,
                "nccl_percent": 100.0 * values.get("NCCL-total", 0) / wall,
            }
        )

    full = [row for row in windows if row["phase"] == "FULL5"]
    total_wall = sum(row["wall_ns"] for row in full)
    total_compute = sum(row["categories"].get("compute", 0) for row in full)
    total_nccl = sum(row["categories"].get("NCCL-total", 0) for row in full)
    payload = {
        "schema_version": "iter304.current_kernel_share.v1",
        "status": "PASS_EXACT_KERNEL_UNION",
        "node": args.node,
        "sqlite": str(args.sqlite),
        "diagnostic_only": True,
        "category_unions_overlap_and_are_not_additive": True,
        "denominator": "sum of exact per-rank NVTX window walls on this node",
        "full5": {
            "rank_count": len(full),
            "gpu_wall_seconds": total_wall / 1e9,
            "compute_gpu_seconds": total_compute / 1e9,
            "compute_pooled_percent": 100.0 * total_compute / total_wall,
            "compute_rank_percent_stats": stats(row["compute_percent"] for row in full),
            "nccl_gpu_seconds": total_nccl / 1e9,
            "nccl_pooled_percent": 100.0 * total_nccl / total_wall,
            "nccl_rank_percent_stats": stats(row["nccl_percent"] for row in full),
        },
        "marker_counts": marker_counts,
        "windows": windows,
        "elapsed_analysis_seconds": time.time() - t0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["full5"], indent=2), flush=True)


if __name__ == "__main__":
    main()
