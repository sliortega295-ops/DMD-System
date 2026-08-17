#!/usr/bin/env python3
"""Exact kernel temporal coverage plus Nsight Systems GPU hardware metrics.

The old report used the label ``compute`` for every non-NCCL CUDA kernel. That
quantity is only a temporal coverage metric: it answers whether at least one
non-NCCL CUDA kernel is running, not how fully the SMs or Tensor Cores are used.

This analyzer therefore reports two separate layers:

1. exact interval unions for non-NCCL CUDA kernels and NCCL kernels inside the
   existing rank-qualified NVTX windows;
2. periodic Nsight Systems GPU Metrics (SM Active, SM Issue, Tensor Active and
   available VRAM/DRAM bandwidth counters) aligned to the same windows.

GPU Metrics are device-level, not process-level. The world16 contract used by
this evidence bundle has one exclusive rank per CUDA device, which is why the
per-rank window/device alignment is valid for this experiment.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sqlite3
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from distillschedule.profiling.nsight_gpu_metrics import (  # noqa: E402
    aggregate_window_metrics,
    load_gpu_metric_series,
    summarize_device_window,
)


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
    ap.add_argument(
        "--require-gpu-metrics",
        action="store_true",
        help=(
            "fail if the SQLite export does not contain Nsight Systems GPU_METRICS "
            "with SM Active/SM Issue/Tensor Active; old captures remain analyzable "
            "without this flag"
        ),
    )
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
    if [device for _gp, device in gps] != list(range(8)):
        raise RuntimeError(f"expected CUDA device ids 0..7, got {gps}")

    c.execute(
        "create temp table windows("
        "gp integer, device integer, global_rank integer, phase text, window_index integer, "
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
        rows.append((gp, device, rank, "FULL5", 0, lo, hi, hi - lo))
        for phase, values in parsed["windows"].items():
            for value in values:
                rows.append(
                    (
                        gp,
                        device,
                        rank,
                        phase,
                        int(value["window_index"]),
                        int(value["start"]),
                        int(value["end"]),
                        int(value["end"]) - int(value["start"]),
                    )
                )
        marker_counts[str(rank)] = parsed["counts"]
    c.executemany("insert into windows values(?,?,?,?,?,?,?,?)", rows)
    c.execute("create index windows_gp_start on windows(gp,start,end)")

    # One kernel-table pass. The category name deliberately avoids "compute":
    # non-NCCL kernels being present on a timeline does not imply high SM/MFU.
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
        " else 'non-NCCL-kernel' end category "
        "from CUPTI_ACTIVITY_KIND_KERNEL k "
        "join StringIds d on d.id=k.demangledName"
    )
    c.execute("create index tagged_kernel_gp_start on tagged_kernel(gp,start,end)")
    print(f"node {args.node}: staged tagged kernels in {time.time()-t0:.1f}s", flush=True)

    query = """
    with clipped as (
      select w.global_rank,w.device,w.phase,w.window_index,w.start window_start,
             w.end window_end,w.wall_ns,
             t.category,max(t.start,w.start) start,min(t.end,w.end) end
      from windows w indexed by windows_gp_start
      join tagged_kernel t indexed by tagged_kernel_gp_start
        on t.gp=w.gp and t.start<w.end and t.end>w.start
    ), expanded as (
      select * from clipped where end>start
      union all
      select global_rank,device,phase,window_index,window_start,window_end,wall_ns,
             'NCCL-total',start,end
      from clipped where category!='non-NCCL-kernel' and end>start
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
      select global_rank,device,phase,window_index,window_start,window_end,wall_ns,
             category,interval_group,min(start) start,max(end) end
      from grouped
      group by global_rank,device,phase,window_index,window_start,window_end,
               wall_ns,category,interval_group
    )
    select global_rank,device,phase,window_index,window_start,window_end,wall_ns,
           category,sum(end-start) union_ns
    from merged
    group by global_rank,device,phase,window_index,window_start,window_end,
             wall_ns,category
    order by global_rank,phase,window_index,category
    """
    union_rows = [
        {
            "global_rank": int(rank),
            "device": int(device),
            "phase": str(phase),
            "window_index": int(index),
            "window_start_ns": int(window_start),
            "window_end_ns": int(window_end),
            "wall_ns": int(wall),
            "category": str(category),
            "union_ns": int(union_ns),
        }
        for (
            rank,
            device,
            phase,
            index,
            window_start,
            window_end,
            wall,
            category,
            union_ns,
        ) in c.execute(query)
    ]
    print(f"node {args.node}: kernel union query complete in {time.time()-t0:.1f}s", flush=True)

    by_key = defaultdict(dict)
    for row in union_rows:
        key = (
            row["global_rank"],
            row["device"],
            row["phase"],
            row["window_index"],
            row["window_start_ns"],
            row["window_end_ns"],
            row["wall_ns"],
        )
        by_key[key][row["category"]] = row["union_ns"]

    gpu_metric_series = load_gpu_metric_series(c)
    if args.require_gpu_metrics and gpu_metric_series.get("status") != "PASS":
        raise RuntimeError(
            "required Nsight GPU Metrics are unavailable or incomplete: "
            + json.dumps(
                {
                    "status": gpu_metric_series.get("status"),
                    "reason": gpu_metric_series.get("reason"),
                    "missing_core_metrics": gpu_metric_series.get("missing_core_metrics"),
                    "available_metric_names": gpu_metric_series.get("available_metric_names"),
                },
                sort_keys=True,
            )
        )

    windows = []
    for key, values in sorted(by_key.items()):
        rank, device, phase, index, window_start, window_end, wall = key
        hardware = summarize_device_window(
            gpu_metric_series,
            gpu_id=device,
            start_ns=window_start,
            end_ns=window_end,
        )
        windows.append(
            {
                "global_rank": rank,
                "device": device,
                "phase": phase,
                "window_index": index,
                "window_start_ns": window_start,
                "window_end_ns": window_end,
                "wall_ns": wall,
                "categories": values,
                "non_nccl_kernel_coverage_percent": (
                    100.0 * values.get("non-NCCL-kernel", 0) / wall
                ),
                "nccl_kernel_coverage_percent": (
                    100.0 * values.get("NCCL-total", 0) / wall
                ),
                "gpu_metrics": hardware,
            }
        )

    c.close()
    full = [row for row in windows if row["phase"] == "FULL5"]
    total_wall = sum(row["wall_ns"] for row in full)
    total_non_nccl_kernel = sum(
        row["categories"].get("non-NCCL-kernel", 0) for row in full
    )
    total_nccl = sum(row["categories"].get("NCCL-total", 0) for row in full)
    full_gpu_metrics = aggregate_window_metrics(full)
    status = (
        "PASS_EXACT_KERNEL_UNION_AND_GPU_METRICS"
        if full_gpu_metrics["status"] == "PASS"
        else "PASS_EXACT_KERNEL_UNION_GPU_METRICS_NOT_COMPLETE"
    )
    payload = {
        "schema_version": "iter304.current_kernel_share.v2",
        "status": status,
        "node": args.node,
        "sqlite": str(args.sqlite),
        "diagnostic_only": True,
        "category_unions_overlap_and_are_not_additive": True,
        "kernel_coverage_is_not_hardware_utilization": True,
        "gpu_metrics_are_device_level_not_process_level": True,
        "denominator": "sum of exact per-rank NVTX window walls on this node",
        "full5": {
            "rank_count": len(full),
            "gpu_wall_seconds": total_wall / 1e9,
            "non_nccl_kernel_gpu_seconds": total_non_nccl_kernel / 1e9,
            "non_nccl_kernel_pooled_percent": (
                100.0 * total_non_nccl_kernel / total_wall
            ),
            "non_nccl_kernel_rank_percent_stats": stats(
                row["non_nccl_kernel_coverage_percent"] for row in full
            ),
            "nccl_gpu_seconds": total_nccl / 1e9,
            "nccl_pooled_percent": 100.0 * total_nccl / total_wall,
            "nccl_rank_percent_stats": stats(
                row["nccl_kernel_coverage_percent"] for row in full
            ),
            "gpu_metrics": full_gpu_metrics,
        },
        "gpu_metrics_capture": {
            "status": gpu_metric_series.get("status"),
            "reason": gpu_metric_series.get("reason"),
            "missing_core_metrics": gpu_metric_series.get("missing_core_metrics", []),
            "available_metric_names": gpu_metric_series.get("available_metric_names", []),
            "selected_source_names": gpu_metric_series.get("selected_source_names", {}),
            "required_by_cli": bool(args.require_gpu_metrics),
        },
        "semantics": {
            "non_nccl_kernel_coverage": (
                "exact temporal union of non-NCCL CUDA kernels; this is not SM occupancy, "
                "Tensor Core utilization, MFU, or achieved FLOP/s"
            ),
            "nccl_kernel_coverage": "exact temporal union of NCCL/NVLS/CollNet kernels",
            "gpu_metrics": (
                "time-weighted periodic device-level Nsight Systems counters aligned to "
                "rank-qualified NVTX windows; valid here because one exclusive rank owns "
                "each CUDA device"
            ),
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
