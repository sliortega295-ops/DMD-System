#!/usr/bin/env python3
"""Compare repeated MiniMax cycles and report Helios PR16 only as cross-workload context."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--minimax-summary", type=Path, action="append", required=True)
    p.add_argument(
        "--helios-pr16-result",
        type=Path,
        default=Path("docs/issue-assets/fsdp2-critic-residual-kernel-share-20260816/data/iteration_303_matched_ab.json"),
    )
    p.add_argument("--output", type=Path, required=True)
    return p


def _mad(values: list[float]) -> float:
    median = statistics.median(values)
    return statistics.median(abs(value - median) for value in values)


def main() -> int:
    args = parser().parse_args()
    summaries = [json.loads(path.read_text()) for path in args.minimax_summary]
    if len(summaries) < 3:
        raise ValueError("formal MiniMax timing requires at least three unprofiled runs")
    if any(value.get("status") != "PASS" for value in summaries):
        raise ValueError("all MiniMax summaries must PASS")
    identities = {
        (
            value.get("contract_name"),
            value.get("upstream_commit"),
            value.get("benchmark_patch_sha256"),
            value.get("benchmark_end_step_idx"),
            value.get("global_batch"),
            value.get("student_updates"),
            value.get("fake_updates")
        )
        for value in summaries
    }
    if len(identities) != 1:
        raise ValueError("MiniMax timing summaries do not share one benchmark contract")
    seconds = [float(value["cycle_seconds"]) for value in summaries]
    median = statistics.median(seconds)
    mean = statistics.mean(seconds)
    stdev = statistics.stdev(seconds) if len(seconds) > 1 else 0.0
    helios = json.loads(args.helios_pr16_result.read_text())
    helios_seconds = [float(x) for x in helios["aggregate"]["experiment_seconds"]]

    payload = {
        "schema_version": "distillschedule.lightx2v_minimax_vs_helios_context.v1",
        "status": "PASS",
        "minimax_upstream": {
            "runs": seconds,
            "median_seconds": median,
            "mean_seconds": mean,
            "stdev_seconds": stdev,
            "cv_percent": 100.0 * stdev / mean if mean else 0.0,
            "mad_seconds": _mad(seconds),
            "contract_identity": list(next(iter(identities)))
        },
        "helios_pr16_cross_workload_context": {
            "runs": helios_seconds,
            "median_seconds": statistics.median(helios_seconds),
            "scope": helios.get("scientific_scope")
        },
        "interpretation": {
            "faster_absolute_wall": "minimax_upstream" if median < statistics.median(helios_seconds) else "helios_pr16",
            "speedup_claim": "PROHIBITED_CROSS_WORKLOAD",
            "reason": (
                "MiniMax-H3 and Helios use different model architectures, resolutions, frame/audio workloads, "
                "and update semantics (MiniMax G1/F5/no EMA vs Helios G1/C5/EMA5). "
                "A claimable systems speedup requires running a DistillSchedule MiniMax variant under the exact same MiniMax contract."
            )
        }
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
