#!/usr/bin/env python3
"""Summarize unprofiled GAN Role-Phase producer/consumer service rates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from distillschedule.runtime.gan_rolephase_optimization import summarize_service_records


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--rank-dir", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--markdown", type=Path)
    return value


def _markdown(value: dict) -> str:
    rows = [
        "# GAN Role-Phase service-rate diagnosis",
        "",
        "> Diagnostic only. Layouts with different batch contracts are not native speedups.",
        "",
        "| Metric | Count | Mean (s) | Median (s) | Max (s) |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, key in (
        ("G produce/sample", "producer_sample_seconds"),
        ("G send wait", "producer_send_wait_seconds"),
        ("G commit wait", "producer_commit_wait_seconds"),
        ("C compute+commit", "critic_compute_commit_seconds"),
        ("C receive wait", "critic_receive_wait_seconds"),
        ("Phase-B wall", "phase_b_wall_seconds"),
    ):
        item = value[key]
        def show(number):
            return "—" if number is None else f"{number:.6f}"
        rows.append(
            f"| {label} | {item['count']} | {show(item['mean'])} | "
            f"{show(item['median'])} | {show(item['max'])} |"
        )
    rows += ["", "## Bounded layout estimates", ""]
    if not value["layout_estimates"]:
        rows.append("Insufficient producer/Critic service samples.")
    else:
        rows += [
            "| Layout | Batch contract | Predicted samples/s | Bottleneck | Matched to current B12 |",
            "|---|---|---:|---|---|",
        ]
        for estimate in value["layout_estimates"]:
            layout = estimate["layout"]
            rows.append(
                f"| {layout['name']} | {layout['contract_id']} | "
                f"{estimate['predicted_samples_per_second']:.6f} | "
                f"{estimate['bottleneck']} | {estimate['matched_to_current_b12']} |"
            )
    rows += [
        "",
        "Only the B12 contract is directly matched to the current Phase-B workload. "
        "Other rows are preregistration guidance for new contracts, not measured speedups.",
        "",
    ]
    return "\n".join(rows)


def main() -> int:
    args = parser().parse_args()
    files = sorted(args.rank_dir.glob("rank_*.json"))
    if not files:
        raise FileNotFoundError(f"no rank_*.json under {args.rank_dir}")
    results = [json.loads(path.read_text()) for path in files]
    value = summarize_service_records(results)
    value["rank_files"] = [str(path) for path in files]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(_markdown(value))
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
