#!/usr/bin/env python3
"""Compare the baseline and persistent-pool correctness runs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from safetensors.torch import load_file

from distillschedule.runtime.p0_parity import DEFAULT_P0_PARITY, tensor_parity_metrics
from distillschedule.runtime.livewan_crossscale_worker import atomic_json


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--baseline-run-root", required=True, type=Path)
    value.add_argument("--pipeline-run-root", required=True, type=Path)
    value.add_argument("--baseline-summary", required=True, type=Path)
    value.add_argument("--pipeline-summary", required=True, type=Path)
    value.add_argument("--output", required=True, type=Path)
    return value


def _summary(path: Path, mode: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "PASS" or value.get("run_kind") != "correctness":
        raise ValueError(f"{mode} correctness summary is not PASS")
    if value.get("execution_mode") != mode or value.get("rank_count") != 16:
        raise ValueError(f"{mode} correctness identity mismatch")
    return value


def _rollouts(root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for rank in range(16):
        receipt = json.loads((root / "receipts" / f"rank_{rank:03d}.json").read_text())
        artifact = receipt.get("output_tensor_artifact")
        if artifact is None:
            continue
        for key, tensor in load_file(artifact["path"], device="cpu").items():
            if key in result:
                raise ValueError(f"duplicate rollout operation key: {key}")
            result[key] = tensor
    return result


def _metrics(summary_root: Path, name: str) -> dict[str, float]:
    rows = []
    for rank in range(16):
        receipt = json.loads((summary_root / "receipts" / f"rank_{rank:03d}.json").read_text())
        value = receipt.get(name)
        if value is not None:
            rows.append(value)
    if not rows:
        raise ValueError(f"missing {name}")
    return rows[0]


def _relative(reference: float, candidate: float) -> float:
    return abs(reference - candidate) / max(abs(reference), 1e-12)


def main() -> int:
    args = parser().parse_args()
    baseline = _summary(args.baseline_summary, "baseline")
    pipeline = _summary(args.pipeline_summary, "pipeline")
    errors: list[str] = []
    if baseline["identity"] != pipeline["identity"]:
        errors.append("source/bootstrap/trace identity differs")
    if baseline["workload"] != pipeline["workload"]:
        errors.append("logical workload contract differs")
    if baseline["call_inventory"] != pipeline["call_inventory"]:
        errors.append("physical application call inventory differs")

    reference = _rollouts(args.baseline_run_root)
    candidate = _rollouts(args.pipeline_run_root)
    if set(reference) != set(candidate) or len(reference) != 80:
        errors.append("rollout operation-key inventory differs")
    parity_rows = []
    for key in sorted(set(reference) & set(candidate)):
        metrics = tensor_parity_metrics(reference[key], candidate[key], thresholds=DEFAULT_P0_PARITY)
        parity_rows.append({"operation_key": key, **metrics})
        if metrics["status"] != "PASS":
            errors.append(f"rollout parity failed: {key}")

    state_rows = []
    for name in ("generator_state_metrics", "critic_state_metrics"):
        ref = _metrics(args.baseline_run_root, name)
        cand = _metrics(args.pipeline_run_root, name)
        keys = ("sum", "sum_sq", "max_abs", "weighted_sum", "weighted_sum_sq")
        row = {
            "state": name,
            "numel_equal": int(ref["numel"]) == int(cand["numel"]),
            "relative": {key: _relative(float(ref[key]), float(cand[key])) for key in keys},
        }
        row["status"] = "PASS" if row["numel_equal"] and max(row["relative"].values()) <= 5e-3 else "FAIL"
        state_rows.append(row)
        if row["status"] != "PASS":
            errors.append(f"{name} aggregate fingerprint failed")

    result = {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "reference": {"mode": "baseline", "summary": str(args.baseline_summary)},
        "candidate": {"mode": "pipeline", "summary": str(args.pipeline_summary)},
        "identity": baseline.get("identity"),
        "heads": {"baseline": baseline.get("head"), "pipeline": pipeline.get("head")},
        "thresholds": {
            "mean_relative": DEFAULT_P0_PARITY.mean_relative,
            "normalized_l2": DEFAULT_P0_PARITY.normalized_l2,
            "min_cosine": DEFAULT_P0_PARITY.min_cosine,
            "max_abs_atol": DEFAULT_P0_PARITY.max_abs_atol,
            "max_abs_rtol": DEFAULT_P0_PARITY.max_abs_rtol,
        },
        "rollout": {
            "count": len(parity_rows),
            "failure_count": sum(row["status"] != "PASS" for row in parity_rows),
            "worst_normalized_l2": max((row["normalized_l2"] for row in parity_rows), default=None),
            "minimum_cosine": min((row["cosine"] for row in parity_rows), default=None),
            "rows": parity_rows,
        },
        "trainable_state": state_rows,
        "claimable_for_timing": not errors,
    }
    atomic_json(args.output, result)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
