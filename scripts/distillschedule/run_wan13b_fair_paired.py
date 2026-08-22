#!/usr/bin/env python3
"""Run three alternating formal pairs for two fair Wan1.3B endpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

from distillschedule.runtime.wan13b_fair import FAIR_PLACEMENTS


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--run-prefix", required=True)
    value.add_argument("--reference-placement", choices=FAIR_PLACEMENTS, required=True)
    value.add_argument("--candidate-placement", choices=FAIR_PLACEMENTS, required=True)
    value.add_argument("--expected-head", required=True)
    value.add_argument("--model-root", type=Path, required=True)
    value.add_argument("--bootstrap", type=Path, required=True)
    value.add_argument("--bootstrap-hash", required=True)
    value.add_argument("--correctness-receipt", type=Path, required=True)
    value.add_argument("--remote-root", type=Path, required=True)
    value.add_argument("--master-addr", required=True)
    value.add_argument("--master-port", type=int, default=29860)
    value.add_argument(
        "--local-evidence-root",
        type=Path,
        default=Path(
            "/home/lyy/Helios/research_artifacts/DMD-System/"
            "wan13b_fair_resident_fsdp2_20260822"
        ),
    )
    value.add_argument("--output", type=Path, required=True)
    return value


def _run(args: argparse.Namespace, placement: str, suffix: str, port: int) -> Path:
    run_id = f"{args.run_prefix}_{suffix}_{placement.replace('-', '_')}"
    controller = Path(__file__).with_name("run_wan13b_fair_world16.py")
    command = [
        sys.executable,
        str(controller),
        "--run-id",
        run_id,
        "--mode",
        "timing",
        "--placement",
        placement,
        "--expected-head",
        args.expected_head,
        "--model-root",
        str(args.model_root),
        "--bootstrap",
        str(args.bootstrap),
        "--bootstrap-hash",
        args.bootstrap_hash,
        "--remote-root",
        str(args.remote_root),
        "--master-addr",
        args.master_addr,
        "--master-port",
        str(port),
        "--correctness-receipt",
        str(args.correctness_receipt),
        "--correctness-sha256",
        _sha256(args.correctness_receipt),
        "--local-evidence-root",
        str(args.local_evidence_root),
    ]
    subprocess.run(command, check=True)
    return args.local_evidence_root / run_id / "summary.json"


def _cv(values: list[float]) -> float:
    return statistics.stdev(values) / statistics.mean(values) if len(values) > 1 else 0.0


def main() -> int:
    args = parser().parse_args()
    if args.reference_placement == args.candidate_placement:
        raise ValueError("formal pair requires two different placements")
    correctness = json.loads(args.correctness_receipt.read_text())
    if correctness.get("status") != "PASS" or not correctness.get(
        "claimable_for_paired_timing"
    ):
        raise ValueError("paired timing requires a PASS fair correctness receipt")
    covered = {
        correctness.get("reference", {}).get("placement"),
        correctness.get("candidate", {}).get("placement"),
    }
    requested = {args.reference_placement, args.candidate_placement}
    if covered != requested:
        raise ValueError(
            f"correctness receipt covers {covered}, requested formal pair is {requested}"
        )
    if args.output.exists():
        raise FileExistsError(args.output)

    reference = args.reference_placement
    candidate = args.candidate_placement
    order = ((reference, candidate), (candidate, reference), (reference, candidate))
    warmups = {
        placement: str(
            _run(args, placement, f"warmup_{index}", args.master_port + index)
        )
        for index, placement in enumerate((reference, candidate))
    }
    trials: dict[str, list[float]] = {reference: [], candidate: []}
    pairs: list[dict[str, Any]] = []
    for pair_index, pair_order in enumerate(order):
        summaries: dict[str, Path] = {}
        for order_index, placement in enumerate(pair_order):
            summaries[placement] = _run(
                args,
                placement,
                f"pair{pair_index}_order{order_index}",
                args.master_port + 10 + pair_index * 2 + order_index,
            )
        values = {
            placement: float(json.loads(path.read_text())["full5_seconds_world_max"])
            for placement, path in summaries.items()
        }
        for placement, seconds in values.items():
            trials[placement].append(seconds)
        ratio = values[reference] / values[candidate]
        pairs.append(
            {
                "pair": pair_index,
                "order": list(pair_order),
                "reference_seconds": values[reference],
                "candidate_seconds": values[candidate],
                "candidate_speedup": ratio,
                "candidate_wall_reduction_fraction": 1.0 - values[candidate] / values[reference],
                "summaries": {key: str(value) for key, value in summaries.items()},
            }
        )

    reference_median = statistics.median(trials[reference])
    candidate_median = statistics.median(trials[candidate])
    ratios = [float(value["candidate_speedup"]) for value in pairs]
    result = {
        "schema_version": "distillschedule.wan13b_fair_paired.v1",
        "status": "PASS",
        "diagnostic_only": False,
        "reference_placement": reference,
        "candidate_placement": candidate,
        "correctness_receipt": str(args.correctness_receipt),
        "correctness_sha256": _sha256(args.correctness_receipt),
        "warmups_excluded": warmups,
        "pairs": pairs,
        "reference": {
            "raw_seconds": trials[reference],
            "median_seconds": reference_median,
            "mean_seconds": statistics.mean(trials[reference]),
            "cv": _cv(trials[reference]),
        },
        "candidate": {
            "raw_seconds": trials[candidate],
            "median_seconds": candidate_median,
            "mean_seconds": statistics.mean(trials[candidate]),
            "cv": _cv(trials[candidate]),
        },
        "comparison": {
            "median_candidate_speedup": reference_median / candidate_median,
            "median_candidate_wall_reduction_fraction": 1.0 - candidate_median / reference_median,
            "geometric_mean_candidate_speedup": math.exp(
                statistics.mean(math.log(value) for value in ratios)
            ),
            "claim_boundary": (
                "real Wan1.3B frozen workload; no Helios14B linear extrapolation"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
