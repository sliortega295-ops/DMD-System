#!/usr/bin/env python3
"""Summarize world16 MiniMax-H3 Nsight movement attribution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from distillschedule.profiling.minimax_nsys_attribution import (
    COMPUTE,
    D2H,
    H2D,
    NCCL,
    NCCL_ALL_GATHER,
    NCCL_ALL_REDUCE,
    NCCL_OTHER,
    NCCL_REDUCE_SCATTER,
    OFFLOAD,
    analyze_minimax_world16,
)


def _seconds(metric: dict[str, Any]) -> float:
    return float(metric.get("rank_max_seconds", metric.get("gpu_seconds", 0.0)))


def _fmt_seconds(value: float) -> str:
    return f"{value:.3f} s"


def _fmt_percent(value: float) -> str:
    return f"{value:.2f}%"


def _fmt_gib(value: float) -> str:
    return f"{value:.2f} GiB"


def _render_markdown(report: dict[str, Any], formal_reference_seconds: float | None) -> str:
    cycle = report["cycle"]
    categories = cycle["categories"]
    exposure = cycle["exposure"]
    lines = [
        "# MiniMax-H3 Native Nsight Attribution",
        "",
        "> Diagnostic profile only. Nsight-profiled wall time is not a formal timing denominator. ",
        "> `exposed` means the activity interval is not overlapped by a non-NCCL CUDA compute kernel on the same rank; it is not a directly recoverable speedup prediction.",
        "",
        "## Cycle summary",
        "",
        f"- Profiled rank-max cycle wall: **{cycle['rank_max_wall_seconds']:.3f} s**",
        f"- Profiled rank-median cycle wall: **{cycle['rank_median_wall_seconds']:.3f} s**",
    ]
    if formal_reference_seconds is not None:
        lines.append(
            f"- Unprofiled reference wall supplied by caller: **{formal_reference_seconds:.3f} s** (context only; never mixed into interval percentages)"
        )
    lines.extend(
        [
            f"- Compute union: **{categories[COMPUTE]['pooled_percent']:.2f}%** pooled GPU-seconds",
            f"- NCCL union: **{categories[NCCL]['pooled_percent']:.2f}%**, rank-max {_seconds(categories[NCCL]):.3f} s",
            f"- NCCL exposed vs compute: **{exposure['nccl_exposed_vs_compute']['pooled_percent']:.2f}%**, rank-max {_seconds(exposure['nccl_exposed_vs_compute']):.3f} s",
            f"- Offload-candidate union: **{categories[OFFLOAD]['pooled_percent']:.2f}%**, rank-max {_seconds(categories[OFFLOAD]):.3f} s",
            f"- Offload exposed vs compute: **{exposure['offload_exposed_vs_compute']['pooled_percent']:.2f}%**, rank-max {_seconds(exposure['offload_exposed_vs_compute']):.3f} s",
            f"- NCCL ∪ offload exposed vs compute: **{exposure['movement_exposed_vs_compute']['pooled_percent']:.2f}%**, rank-max {_seconds(exposure['movement_exposed_vs_compute']):.3f} s",
            "",
            "## Transfer volume",
            "",
            f"- H2D: **{categories[H2D]['gib']:.2f} GiB** across 16 ranks",
            f"- D2H: **{categories[D2H]['gib']:.2f} GiB** across 16 ranks",
            f"- Large-copy offload candidates: **{categories[OFFLOAD]['gib']:.2f} GiB** across 16 ranks",
            f"- Offload heuristic threshold: **{report['offload_min_bytes'] / (1024**2):.1f} MiB/event**",
            "",
            "The offload bucket is deliberately conservative: it contains only large H2D/D2H copies. The JSON keeps per-rank copy-size histograms and CUDA-runtime correlations so the dominant copy sizes can be audited before calling them activation offload.",
            "",
            "## NCCL kernel breakdown",
            "",
            "| class | pooled union | rank-max union |",
            "|---|---:|---:|",
        ]
    )
    for key, label in (
        (NCCL_ALL_GATHER, "all-gather"),
        (NCCL_REDUCE_SCATTER, "reduce-scatter"),
        (NCCL_ALL_REDUCE, "all-reduce"),
        (NCCL_OTHER, "other/generic NCCL"),
    ):
        metric = categories[key]
        lines.append(
            f"| {label} | {metric['pooled_percent']:.2f}% | {_seconds(metric):.3f} s |"
        )

    lines.extend(
        [
            "",
            "## Stage decomposition",
            "",
            "| stage | rank-max wall | compute | NCCL | exposed NCCL | offload | exposed offload | exposed movement |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for key in ["student/0"] + [f"fake/{index}" for index in range(5)]:
        value = report["stages"][key]
        stage_categories = value["categories"]
        stage_exposure = value["exposure"]
        lines.append(
            "| {key} | {wall:.3f}s | {compute:.1f}% | {nccl:.1f}% | {enccl:.3f}s | "
            "{offload:.1f}% | {eoff:.3f}s | {emove:.3f}s |".format(
                key=key,
                wall=value["rank_max_wall_seconds"],
                compute=stage_categories[COMPUTE]["pooled_percent"],
                nccl=stage_categories[NCCL]["pooled_percent"],
                enccl=_seconds(stage_exposure["nccl_exposed_vs_compute"]),
                offload=stage_categories[OFFLOAD]["pooled_percent"],
                eoff=_seconds(stage_exposure["offload_exposed_vs_compute"]),
                emove=_seconds(stage_exposure["movement_exposed_vs_compute"]),
            )
        )

    lines.extend(
        [
            "",
            "## How to read the result",
            "",
            "Use `rank_max_seconds` to find the worst-rank exposure and pooled GPU-seconds to describe overall overlap. Do **not** add compute, NCCL and copy percentages: these categories overlap by construction.",
            "",
            "The most useful decision metrics are:",
            "1. `nccl_exposed_vs_compute`: communication that is not hidden by local compute;",
            "2. `offload_exposed_vs_compute`: large host/device copies not hidden by local compute;",
            "3. `movement_exposed_vs_compute`: union of those two, so overlap between NCCL and offload is counted only once;",
            "4. `nccl_unique_exposed` / `offload_unique_exposed` / `nccl_offload_shared_exposed`: separates who owns exposed movement when NCCL and offload overlap each other.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--node0-sqlite", type=Path, required=True)
    parser.add_argument("--node1-sqlite", type=Path, required=True)
    parser.add_argument("--mode", choices=("native", "distillschedule"), default="native")
    parser.add_argument("--offload-min-mib", type=float, default=64.0)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument(
        "--formal-reference-seconds",
        type=float,
        help="optional unprofiled wall time printed as context only; never used as the Nsight denominator",
    )
    args = parser.parse_args()
    if args.offload_min_mib <= 0:
        parser.error("--offload-min-mib must be positive")
    if args.formal_reference_seconds is not None and args.formal_reference_seconds <= 0:
        parser.error("--formal-reference-seconds must be positive")

    report = analyze_minimax_world16(
        (args.node0_sqlite, args.node1_sqlite),
        mode=args.mode,
        offload_min_bytes=int(args.offload_min_mib * 1024 * 1024),
    )
    report["formal_reference_seconds"] = args.formal_reference_seconds
    report["formal_reference_usage"] = (
        "context_only_not_an_interval_denominator"
        if args.formal_reference_seconds is not None
        else None
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(
            _render_markdown(report, args.formal_reference_seconds) + "\n"
        )

    cycle = report["cycle"]
    print(
        json.dumps(
            {
                "mode": args.mode,
                "rank_count": report["rank_count"],
                "profiled_rank_max_wall_seconds": cycle["rank_max_wall_seconds"],
                "compute_pooled_percent": cycle["categories"][COMPUTE]["pooled_percent"],
                "nccl_pooled_percent": cycle["categories"][NCCL]["pooled_percent"],
                "nccl_exposed_rank_max_seconds": cycle["exposure"]["nccl_exposed_vs_compute"]["rank_max_seconds"],
                "offload_candidate_gib": cycle["categories"][OFFLOAD]["gib"],
                "offload_exposed_rank_max_seconds": cycle["exposure"]["offload_exposed_vs_compute"]["rank_max_seconds"],
                "movement_exposed_rank_max_seconds": cycle["exposure"]["movement_exposed_vs_compute"]["rank_max_seconds"],
                "diagnostic_only": True,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
