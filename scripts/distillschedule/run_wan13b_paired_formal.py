#!/usr/bin/env python3
"""Run warmups plus three crossover pairs for the Wan1.3B placement study."""
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


ORDER = (
    ("resident", "world16-fsdp"),
    ("world16-fsdp", "resident"),
    ("resident", "world16-fsdp"),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--run-prefix", required=True)
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
            "wan13b_fsdp_upper_bound_20260820"
        ),
    )
    value.add_argument("--output", type=Path, required=True)
    return value


def _run(args: argparse.Namespace, placement: str, suffix: str, port: int) -> Path:
    run_id = f"{args.run_prefix}_{suffix}_{placement.replace('-', '_')}"
    controller = Path(__file__).with_name("run_wan13b_upper_bound_world16.py")
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
    correctness = json.loads(args.correctness_receipt.read_text())
    if correctness.get("status") != "PASS" or not correctness.get(
        "claimable_for_paired_timing"
    ):
        raise ValueError("paired timing requires a PASS correctness receipt")
    if args.output.exists():
        raise FileExistsError(args.output)

    # Warmups are full fresh processes and are excluded from every formal row.
    warmups = {
        placement: str(
            _run(args, placement, f"warmup_{index}", args.master_port + index)
        )
        for index, placement in enumerate(("resident", "world16-fsdp"))
    }
    pairs: list[dict[str, Any]] = []
    trials: dict[str, list[float]] = {"resident": [], "world16-fsdp": []}
    for pair_index, order in enumerate(ORDER):
        summaries: dict[str, Path] = {}
        for order_index, placement in enumerate(order):
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
        for placement, value in values.items():
            trials[placement].append(value)
        pairs.append(
            {
                "pair": pair_index,
                "order": list(order),
                "resident_seconds": values["resident"],
                "fsdp_seconds": values["world16-fsdp"],
                "fsdp_overhead_seconds": values["world16-fsdp"] - values["resident"],
                "fsdp_overhead_fraction": values["world16-fsdp"] / values["resident"] - 1.0,
                "efficiency_vs_resident": values["resident"] / values["world16-fsdp"],
                "summaries": {key: str(value) for key, value in summaries.items()},
            }
        )

    resident_median = statistics.median(trials["resident"])
    fsdp_median = statistics.median(trials["world16-fsdp"])
    ratios = [float(value["efficiency_vs_resident"]) for value in pairs]
    result = {
        "schema_version": "distillschedule.wan13b_paired_formal.v1",
        "status": "PASS",
        "diagnostic_only": False,
        "correctness_receipt": str(args.correctness_receipt),
        "correctness_sha256": _sha256(args.correctness_receipt),
        "warmups_excluded": warmups,
        "pairs": pairs,
        "resident": {
            "raw_seconds": trials["resident"],
            "median_seconds": resident_median,
            "mean_seconds": statistics.mean(trials["resident"]),
            "cv": _cv(trials["resident"]),
        },
        "world16_fsdp": {
            "raw_seconds": trials["world16-fsdp"],
            "median_seconds": fsdp_median,
            "mean_seconds": statistics.mean(trials["world16-fsdp"]),
            "cv": _cv(trials["world16-fsdp"]),
        },
        "comparison": {
            "median_fsdp_overhead_seconds": fsdp_median - resident_median,
            "median_fsdp_overhead_fraction": fsdp_median / resident_median - 1.0,
            "median_efficiency_vs_resident": resident_median / fsdp_median,
            "geometric_mean_efficiency_vs_resident": math.exp(
                statistics.mean(math.log(value) for value in ratios)
            ),
            "claim_boundary": (
                "real Wan1.3B practical resident upper-bound proxy; not a linear Helios14B extrapolation"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
