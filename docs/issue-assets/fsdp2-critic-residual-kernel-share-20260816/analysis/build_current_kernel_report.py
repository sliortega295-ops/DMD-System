#!/usr/bin/env python3
"""Combine the two node-level current-path kernel-union reports."""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path


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


def summarize(rows):
    wall = sum(row["wall_ns"] for row in rows)
    compute = sum(row["categories"].get("compute", 0) for row in rows)
    nccl = sum(row["categories"].get("NCCL-total", 0) for row in rows)
    return {
        "rank_count": len(rows),
        "gpu_wall_seconds": wall / 1e9,
        "compute_gpu_seconds": compute / 1e9,
        "nccl_gpu_seconds": nccl / 1e9,
        "compute_pooled_percent": 100 * compute / wall,
        "nccl_pooled_percent": 100 * nccl / wall,
        "wall_rank_seconds_stats": stats(row["wall_ns"] / 1e9 for row in rows),
        "compute_rank_percent_stats": stats(
            100 * row["categories"].get("compute", 0) / row["wall_ns"] for row in rows
        ),
        "nccl_rank_percent_stats": stats(
            100 * row["categories"].get("NCCL-total", 0) / row["wall_ns"] for row in rows
        ),
    }


def main() -> None:
    node_reports = [
        json.loads((DATA / "nsight" / f"kernel_share_node{node}.json").read_text())
        for node in (0, 1)
    ]
    windows = [row for report in node_reports for row in report["windows"]]
    full_rows = [row for row in windows if row["phase"] == "FULL5"]

    by_phase_rank = defaultdict(lambda: defaultdict(int))
    for row in windows:
        if row["phase"] == "FULL5":
            continue
        key = (row["phase"], row["global_rank"])
        by_phase_rank[key]["wall_ns"] += row["wall_ns"]
        for category, value in row["categories"].items():
            by_phase_rank[key][category] += value
    phases = {}
    for phase in sorted({key[0] for key in by_phase_rank}):
        rows = []
        for (name, rank), value in sorted(by_phase_rank.items()):
            if name != phase:
                continue
            rows.append(
                {
                    "global_rank": rank,
                    "wall_ns": value["wall_ns"],
                    "categories": {k: v for k, v in value.items() if k != "wall_ns"},
                }
            )
        phases[phase] = summarize(rows)

    telemetry = json.loads((DATA / "native_telemetry_summary.json").read_text())
    payload = {
        "schema_version": "iter304.current_kernel_report.v1",
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
                "compute": "exact temporal union of non-NCCL CUDA kernels inside each per-rank NVTX window",
                "nccl": "exact temporal union of NCCL/NVLS/CollNet kernels",
                "non_additive": "compute and NCCL may overlap across CUDA streams; neither NCCL-only exposure nor no-kernel complement is inferred by subtraction",
            },
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")

    f = payload["nsight"]["full5"]
    md = [
        "# Iteration 304 current FSDP2 path: kernel share",
        "",
        "## Matched runs",
        "",
        f"- Native Full5: **{NATIVE_SECONDS:.3f} s**, claimable timing.",
        f"- Nsight profile receipt timer: **{PROFILE_SECONDS:.3f} s**, diagnostic only; "
        f"{PROFILE_SECONDS / NATIVE_SECONDS:.2f}x the native timer. It includes profiler stop/finalization and is not a training-slowdown denominator.",
        f"- Native NVML GPU utilization: **{telemetry['combined_pooled']['gpu_util_mean_pct']:.2f}% mean**. This is not a compute-kernel metric.",
        "",
        "## Exact Full5 CUDA-kernel union (16-rank pooled denominator)",
        "",
        f"- Exact per-rank `STATIC_FSDP2_P0_FULL5` NVTX wall median: **{f['wall_rank_seconds_stats']['median']:.3f} s**; this, not the 339.152 s receipt timer, is the kernel-share denominator.",
        f"- Compute-kernel union: **{f['compute_pooled_percent']:.2f}%** "
        f"({f['compute_gpu_seconds']:.2f}/{f['gpu_wall_seconds']:.2f} GPU-s); "
        f"rank median **{f['compute_rank_percent_stats']['median']:.2f}%**.",
        f"- NCCL-kernel union: **{f['nccl_pooled_percent']:.2f}%**.",
        "- Compute and NCCL unions can overlap across streams and must not be added. This narrow pass does not infer NCCL-only critical-path exposure or GPU idle.",
        "",
        "## Phase view",
        "",
        "|Phase|rank wall median s|compute union|NCCL union|",
        "|---|---:|---:|---:|",
    ]
    for phase, value in sorted(phases.items()):
        md.append(
            f"|{phase}|{value['wall_rank_seconds_stats']['median']:.3f}|"
            f"{value['compute_pooled_percent']:.2f}%|{value['nccl_pooled_percent']:.2f}%|"
        )
    md += [
        "",
        "## Interpretation boundary",
        "",
        "The current trace is sufficient to record exact compute- and NCCL-kernel temporal shares. It is not sufficient to turn the NCCL union into native wall-time savings: overlap and critical-path exposure need a dedicated interval-intersection pass, and any code change still needs a matched no-Nsight A/B.",
    ]
    OUT_MD.write_text("\n".join(md) + "\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        labels = ["Full5"] + list(sorted(phases))
        values = [f] + [phases[p] for p in sorted(phases)]
        fields = [
            ("compute_pooled_percent", "compute-kernel union", "#0f9d76"),
            ("nccl_pooled_percent", "NCCL-kernel union", "#f59e0b"),
        ]
        fig, ax = plt.subplots(figsize=(14, max(6, 0.55 * len(labels))), constrained_layout=True)
        y = np.arange(len(labels))
        height = 0.36
        for offset, (field, legend, color) in zip((-height / 2, height / 2), fields):
            x = np.array([row[field] for row in values])
            ax.barh(y + offset, x, height=height, label=legend, color=color)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel("Percent of exact per-rank NVTX window wall (16-rank pooled)")
        ax.set_title("Current fixed Critic-residual FSDP2 path: CUDA kernel temporal unions")
        ax.legend(ncol=2, loc="lower right")
        ax.grid(axis="x", alpha=0.2)
        fig.savefig(OUT_PNG, dpi=170)
        plt.close(fig)
    except Exception as exc:
        payload["plot_error"] = repr(exc)
        OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
