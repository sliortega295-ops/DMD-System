#!/usr/bin/env python3
"""Build exact semantic + CUDA-activity overlays from Nsight SQLite exports.

Every input is one exact rank/scope. A node-wide SQLite may be used only after
it has been filtered to a rank/process scope; the script never guesses rank
ownership from process IDs.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping

from distillschedule.profiling.nsight_activity import (
    ACTIVE,
    ALL_COPY,
    COMPUTE,
    D2H,
    H2D,
    Interval,
    NCCL,
    NO_TRACED_ACTIVITY,
    aggregate_rank_reports,
    analyze_sqlite,
    write_json,
)

DEFAULT_PHASE_PATTERNS = {
    "phase_a": r"(?i)(phase[_ -]?a|generator[_ -]?update)",
    "handoff_to_b": r"(?i)(handoff.*to[_ -]?b|a.?to.?b|transition.*phase[_ -]?b)",
    "phase_b": r"(?i)(phase[_ -]?b|critic[_ -]?only)",
    "handoff_to_a": r"(?i)(handoff.*to[_ -]?a|b.?to.?a|transition.*phase[_ -]?a)",
}


def _rank_path(value: str) -> tuple[int, Path]:
    try:
        rank_text, path_text = value.split("=", 1)
        rank = int(rank_text)
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError("expected RANK=/path/to/report.sqlite") from error
    if rank < 0:
        raise argparse.ArgumentTypeError("rank must be non-negative")
    return rank, Path(path_text)


def _rank_range(value: str) -> tuple[int, Interval]:
    try:
        rank_text, range_text = value.split("=", 1)
        start_text, end_text = range_text.split(":", 1)
        return int(rank_text), Interval(int(start_text), int(end_text))
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError("expected RANK=START_NS:END_NS") from error


def _name_pattern(value: str) -> tuple[str, str]:
    try:
        name, pattern = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected NAME=REGEX") from error
    if not name or not pattern:
        raise argparse.ArgumentTypeError("phase name and regex must be non-empty")
    return name, pattern


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--input", type=_rank_path, action="append", required=True)
    value.add_argument("--capture-range", type=_rank_range, action="append", default=[])
    value.add_argument("--phase-pattern", type=_name_pattern, action="append", default=[])
    value.add_argument("--output-json", type=Path, required=True)
    value.add_argument("--output-csv", type=Path)
    value.add_argument("--plot-dir", type=Path)
    value.add_argument("--bin-ms", type=float, default=500.0)
    value.add_argument(
        "--coverage",
        choices=("exact", "inferred", "receipt_reconstructed"),
        default="exact",
    )
    return value


def _write_csv(path: Path, reports: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "rank",
        "phase",
        "wall_seconds",
        "compute_percent",
        "nccl_percent",
        "h2d_percent",
        "d2h_percent",
        "all_copy_percent",
        "active_percent",
        "no_traced_gpu_activity_percent",
        "h2d_bytes",
        "d2h_bytes",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for report in reports:
            scopes = {"capture": report["capture_metrics"], **report.get("phases", {})}
            for phase, metric in scopes.items():
                categories = metric["categories"]
                writer.writerow(
                    {
                        "rank": report["rank"],
                        "phase": phase,
                        "wall_seconds": metric["wall_seconds"],
                        "compute_percent": categories[COMPUTE]["percent_of_wall"],
                        "nccl_percent": categories[NCCL]["percent_of_wall"],
                        "h2d_percent": categories[H2D]["percent_of_wall"],
                        "d2h_percent": categories[D2H]["percent_of_wall"],
                        "all_copy_percent": categories[ALL_COPY]["percent_of_wall"],
                        "active_percent": categories[ACTIVE]["percent_of_wall"],
                        "no_traced_gpu_activity_percent": categories[NO_TRACED_ACTIVITY]["percent_of_wall"],
                        "h2d_bytes": categories[H2D]["bytes"],
                        "d2h_bytes": categories[D2H]["bytes"],
                    }
                )


def _plot_report(report: Mapping[str, Any], output: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as error:
        raise RuntimeError("--plot-dir requires matplotlib and numpy") from error

    timeline = report["timeline"]
    starts = timeline["bin_start_seconds"]
    durations = timeline["bin_duration_seconds"]
    if not starts:
        raise ValueError("timeline has no display bins")
    right = starts[-1] + durations[-1]
    categories = [COMPUTE, NCCL, H2D, D2H, NO_TRACED_ACTIVITY]
    matrix = np.asarray(
        [timeline["activity_fraction"][category] for category in categories],
        dtype=float,
    )

    figure = plt.figure(figsize=(max(12, right / 45.0), 5.8))
    grid = figure.add_gridspec(2, 1, height_ratios=[1.0, 2.7], hspace=0.08)
    semantic_axis = figure.add_subplot(grid[0])
    activity_axis = figure.add_subplot(grid[1], sharex=semantic_axis)

    hatch = {"exact": "", "inferred": "//", "receipt_reconstructed": "xx"}
    for phase in timeline["semantic_windows"]:
        semantic_axis.broken_barh(
            [(phase["start_seconds"], phase["duration_seconds"])],
            (0.1, 0.8),
            hatch=hatch.get(phase["coverage"], ".."),
            alpha=0.75,
        )
        semantic_axis.text(
            phase["start_seconds"] + phase["duration_seconds"] / 2.0,
            0.5,
            phase["name"],
            ha="center",
            va="center",
            fontsize=8,
        )
    semantic_axis.set_ylim(0, 1)
    semantic_axis.set_yticks([0.5], ["semantic envelope"])
    semantic_axis.tick_params(axis="x", labelbottom=False)
    semantic_axis.set_title(
        f"Rank {report['rank']}: semantic envelope above exact CUDA activity occupancy"
    )

    activity_axis.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        extent=(0.0, right, len(categories), 0.0),
        vmin=0.0,
        vmax=1.0,
    )
    activity_axis.set_yticks(
        [index + 0.5 for index in range(len(categories))],
        categories,
    )
    activity_axis.set_xlabel("seconds from closed profiler-range start")
    activity_axis.set_ylabel("exact union occupancy per display bin")
    activity_axis.grid(False)
    figure.text(
        0.01,
        0.01,
        "No traced GPU activity is an activity complement, not proof of CPU/scheduler idle. "
        "Compute, NCCL and copy unions may overlap.",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> int:
    args = parser().parse_args()
    if args.bin_ms <= 0:
        raise ValueError("--bin-ms must be positive")
    inputs = dict(args.input)
    if len(inputs) != len(args.input):
        raise ValueError("duplicate rank in --input")
    ranges = dict(args.capture_range)
    patterns = dict(DEFAULT_PHASE_PATTERNS)
    patterns.update(dict(args.phase_pattern))
    reports = [
        analyze_sqlite(
            path,
            rank=rank,
            phase_patterns=patterns,
            capture_window=ranges.get(rank),
            capture_coverage=args.coverage if rank in ranges else "inferred",
            timeline_bin_ns=max(1, int(args.bin_ms * 1_000_000)),
        )
        for rank, path in sorted(inputs.items())
    ]
    result = {
        "schema_version": "distillschedule.nsight_activity_bundle.v2",
        "reports": reports,
        "aggregate": aggregate_rank_reports(reports),
        "claim_boundary": {
            "diagnostic_only": True,
            "native_speedup_claimed": False,
            "profiled_time_not_used_as_formal_latency": True,
        },
    }
    write_json(args.output_json, result)
    if args.output_csv is not None:
        _write_csv(args.output_csv, reports)
    if args.plot_dir is not None:
        for report in reports:
            _plot_report(
                report,
                args.plot_dir / f"rank_{int(report['rank']):03d}_activity_overlay.png",
            )
    print(json.dumps(result["aggregate"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
