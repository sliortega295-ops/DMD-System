#!/usr/bin/env python3
"""Combine the two node-level current-path kernel/GPU-metric reports."""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
OUT_JSON = DATA / "current_kernel_share_summary.json"
OUT_MD = BASE / "reports" / "current_kernel_share.md"
OUT_PNG = BASE / "figures" / "current_kernel_partition.png"
NATIVE_SECONDS = 141.411127367
PROFILE_SECONDS = 339.151646643


def stats(xs):
    values = list(xs)
    return {
        "n": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "max": max(values),
    }


def _non_nccl_ns(row: dict[str, Any]) -> int:
    categories = row.get("categories", {})
    # Backward compatibility for the already committed Iteration-304 JSON. The
    # old key was named "compute" but semantically meant "non-NCCL kernel".
    return int(categories.get("non-NCCL-kernel", categories.get("compute", 0)))


def _aggregate_gpu_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    totals: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"weighted": 0.0, "covered_ns": 0, "means": [], "source": None}
    )
    for row in rows:
        for canonical, metric in (row.get("gpu_metrics") or {}).get("metrics", {}).items():
            mean = metric.get("time_weighted_mean_pct")
            covered = int(metric.get("covered_ns") or 0)
            if mean is None or covered <= 0:
                continue
            totals[canonical]["weighted"] += float(mean) * covered
            totals[canonical]["covered_ns"] += covered
            totals[canonical]["means"].append(float(mean))
            totals[canonical]["source"] = metric.get("source_metric_name")
    metrics = {}
    for canonical, value in sorted(totals.items()):
        covered = int(value["covered_ns"])
        means = list(value["means"])
        metrics[canonical] = {
            "source_metric_name": value["source"],
            "time_weighted_mean_pct": value["weighted"] / covered,
            "covered_gpu_seconds": covered / 1e9,
            "window_mean_stats": stats(means),
        }
    return {"status": "PASS" if metrics else "NOT_CAPTURED", "metrics": metrics}


def summarize(rows):
    wall = sum(row["wall_ns"] for row in rows)
    non_nccl = sum(_non_nccl_ns(row) for row in rows)
    nccl = sum(row["categories"].get("NCCL-total", 0) for row in rows)
    return {
        "rank_count": len(rows),
        "gpu_wall_seconds": wall / 1e9,
        "non_nccl_kernel_gpu_seconds": non_nccl / 1e9,
        "nccl_gpu_seconds": nccl / 1e9,
        "non_nccl_kernel_pooled_percent": 100 * non_nccl / wall,
        "nccl_pooled_percent": 100 * nccl / wall,
        "wall_rank_seconds_stats": stats(row["wall_ns"] / 1e9 for row in rows),
        "non_nccl_kernel_rank_percent_stats": stats(
            100 * _non_nccl_ns(row) / row["wall_ns"] for row in rows
        ),
        "nccl_rank_percent_stats": stats(
            100 * row["categories"].get("NCCL-total", 0) / row["wall_ns"] for row in rows
        ),
        "gpu_metrics": _aggregate_gpu_metrics(rows),
    }


def _fmt_metric(summary: dict[str, Any], canonical: str) -> str:
    metric = summary.get("gpu_metrics", {}).get("metrics", {}).get(canonical)
    if not metric:
        return "n/a"
    return f"{metric['time_weighted_mean_pct']:.2f}%"


def main() -> None:
    node_reports = [
        json.loads((DATA / "nsight" / f"kernel_share_node{node}.json").read_text())
        for node in (0, 1)
    ]
    windows = [row for report in node_reports for row in report["windows"]]
    full_rows = [row for row in windows if row["phase"] == "FULL5"]

    by_phase_rank: dict[tuple[str, int], dict[str, Any]] = defaultdict(
        lambda: {"wall_ns": 0, "categories": defaultdict(int), "gpu_metric_rows": []}
    )
    for row in windows:
        if row["phase"] == "FULL5":
            continue
        key = (row["phase"], row["global_rank"])
        by_phase_rank[key]["wall_ns"] += row["wall_ns"]
        for category, value in row["categories"].items():
            by_phase_rank[key]["categories"][category] += value
        by_phase_rank[key]["gpu_metric_rows"].append(row)

    phases = {}
    for phase in sorted({key[0] for key in by_phase_rank}):
        rows = []
        for (name, rank), value in sorted(by_phase_rank.items()):
            if name != phase:
                continue
            hardware = _aggregate_gpu_metrics(value["gpu_metric_rows"])
            row = {
                "global_rank": rank,
                "wall_ns": value["wall_ns"],
                "categories": dict(value["categories"]),
                "gpu_metrics": {"status": hardware["status"], "metrics": {}},
            }
            for canonical, metric in hardware["metrics"].items():
                row["gpu_metrics"]["metrics"][canonical] = {
                    "source_metric_name": metric.get("source_metric_name"),
                    "covered_ns": int(metric["covered_gpu_seconds"] * 1e9),
                    "time_weighted_mean_pct": metric["time_weighted_mean_pct"],
                }
            rows.append(row)
        phases[phase] = summarize(rows)

    telemetry = json.loads((DATA / "native_telemetry_summary.json").read_text())
    payload = {
        "schema_version": "iter304.current_kernel_report.v2",
        "status": "PASS_CURRENT_PATH_PROFILE_AND_NATIVE_TELEMETRY",
        "run_ids": {
            "native": "iter304_nogan_gpu_residual_full5_native_telemetry_r2_20260816",
            "nsight": "iter304_nogan_gpu_residual_full5_nsys_20260816",
        },
        "policy": {
            "mesh_policy": "world16-fsdp",
            "rollout_batch_size": 1,
            "pack_real_score_pair": False,
            "critic_residual_storage": "cuda_out_of_place",
            "parameter_offload": False,
            "parameter_streaming": False,
        },
        "native": {
            "formal_full5_seconds": NATIVE_SECONDS,
            "claimable_timing": True,
            "nvml_gpu_utilization": telemetry["combined_pooled"],
        },
        "nsight": {
            "profile_receipt_full5_seconds": PROFILE_SECONDS,
            "diagnostic_only": True,
            "profile_receipt_vs_native_fraction": PROFILE_SECONDS / NATIVE_SECONDS - 1,
            "profile_receipt_factor": PROFILE_SECONDS / NATIVE_SECONDS,
            "full5": summarize(full_rows),
            "phases": phases,
            "semantics": {
                "non_nccl_kernel_coverage": (
                    "exact temporal union of non-NCCL CUDA kernels; not SM occupancy, "
                    "Tensor Core utilization, MFU, or achieved FLOP/s"
                ),
                "nccl": "exact temporal union of NCCL/NVLS/CollNet kernels",
                "gpu_metrics": (
                    "time-weighted Nsight Systems periodic device counters when captured; "
                    "device-level rather than process-level"
                ),
                "non_additive": (
                    "kernel and NCCL temporal unions may overlap across CUDA streams; "
                    "neither NCCL-only exposure nor GPU idle is inferred by subtraction"
                ),
            },
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")

    f = payload["nsight"]["full5"]
    md = [
        "# Iteration 304 current FSDP2 path: kernel coverage and GPU hardware metrics",
        "",
        "## Matched runs",
        "",
        f"- Native Full5: **{NATIVE_SECONDS:.3f} s**, claimable timing.",
        f"- Nsight profile receipt timer: **{PROFILE_SECONDS:.3f} s**, diagnostic only; "
        f"{PROFILE_SECONDS / NATIVE_SECONDS:.2f}x the native timer. It includes profiler stop/finalization and is not a training-slowdown denominator.",
        f"- Native NVML GPU utilization: **{telemetry['combined_pooled']['gpu_util_mean_pct']:.2f}% mean**. This is a coarse device-busy sample, not SM Active.",
        "",
        "## Exact Full5 CUDA-kernel coverage",
        "",
        f"- Exact per-rank `STATIC_FSDP2_P0_FULL5` NVTX wall median: **{f['wall_rank_seconds_stats']['median']:.3f} s**.",
        f"- Non-NCCL CUDA-kernel temporal coverage: **{f['non_nccl_kernel_pooled_percent']:.2f}%** "
        f"({f['non_nccl_kernel_gpu_seconds']:.2f}/{f['gpu_wall_seconds']:.2f} GPU-s); "
        f"rank median **{f['non_nccl_kernel_rank_percent_stats']['median']:.2f}%**.",
        f"- NCCL-kernel temporal coverage: **{f['nccl_pooled_percent']:.2f}%**.",
        "- The old label `compute union` referred to the first quantity above. It must not be read as GPU compute utilization.",
        "",
        "## Nsight Systems hardware metrics",
        "",
    ]
    if f["gpu_metrics"]["status"] == "NOT_CAPTURED":
        md += [
            "This historical Iteration-304 capture did **not** collect `GPU_METRICS`; therefore SM Active, SM Issue, Tensor Active, and VRAM/DRAM bandwidth cannot be reconstructed from it. Re-profile with GPU Metrics enabled and rerun the v2 analyzer.",
            "",
        ]
    else:
        md += [
            f"- SM Active: **{_fmt_metric(f, 'sm_active_percent')}**",
            f"- SM Issue: **{_fmt_metric(f, 'sm_issue_percent')}**",
            f"- Tensor Active: **{_fmt_metric(f, 'tensor_active_percent')}**",
            f"- VRAM bandwidth: **{_fmt_metric(f, 'vram_bandwidth_percent')}**",
            f"- DRAM read: **{_fmt_metric(f, 'dram_read_percent')}**",
            f"- DRAM write: **{_fmt_metric(f, 'dram_write_percent')}**",
            "",
        ]

    md += [
        "## Phase view",
        "",
        "|Phase|rank wall median s|non-NCCL kernel coverage|SM Active|SM Issue|Tensor Active|VRAM BW|NCCL coverage|",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for phase, value in sorted(phases.items()):
        md.append(
            f"|{phase}|{value['wall_rank_seconds_stats']['median']:.3f}|"
            f"{value['non_nccl_kernel_pooled_percent']:.2f}%|"
            f"{_fmt_metric(value, 'sm_active_percent')}|"
            f"{_fmt_metric(value, 'sm_issue_percent')}|"
            f"{_fmt_metric(value, 'tensor_active_percent')}|"
            f"{_fmt_metric(value, 'vram_bandwidth_percent')}|"
            f"{value['nccl_pooled_percent']:.2f}%|"
        )
    md += [
        "",
        "## Interpretation boundary",
        "",
        "Kernel coverage answers whether a CUDA kernel is present on the timeline. Hardware GPU Metrics answer how active the SM/Tensor/memory subsystems are while sampled. Neither turns NCCL activity directly into native wall-time savings; exposed critical-path communication still requires interval attribution and any optimization still requires a matched no-Nsight A/B.",
    ]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(md) + "\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        labels = ["Full5"] + list(sorted(phases))
        values = [f] + [phases[p] for p in sorted(phases)]
        fields = [
            ("non_nccl_kernel_pooled_percent", "non-NCCL kernel coverage"),
            ("nccl_pooled_percent", "NCCL kernel coverage"),
        ]
        fig, ax = plt.subplots(figsize=(14, max(6, 0.55 * len(labels))), constrained_layout=True)
        y = np.arange(len(labels))
        height = 0.36
        for offset, (field, legend) in zip((-height / 2, height / 2), fields):
            x = np.array([row[field] for row in values])
            ax.barh(y + offset, x, height=height, label=legend)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel("Percent of exact per-rank NVTX window wall (16-rank pooled)")
        ax.set_title("Current fixed Critic-residual FSDP2 path: CUDA kernel temporal coverage")
        ax.legend(ncol=2, loc="lower right")
        ax.grid(axis="x", alpha=0.2)
        fig.savefig(OUT_PNG, dpi=170)
        plt.close(fig)
    except Exception as exc:
        payload["plot_error"] = repr(exc)
        OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
