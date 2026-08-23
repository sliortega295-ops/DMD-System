#!/usr/bin/env python3
"""Compare any two fair Wan1.3B placement correctness artifacts."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from safetensors.torch import load_file

from distillschedule.runtime.p0_parity import DEFAULT_P0_PARITY, tensor_parity_metrics
from distillschedule.runtime.wan13b_fair import FAIR_PLACEMENTS


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _load_summary(path: Path, placement: str) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if value.get("status") != "PASS" or value.get("mode") != "correctness":
        raise ValueError(f"{placement} correctness summary is not PASS")
    if value.get("placement") != placement or value.get("rank_count") != 16:
        raise ValueError(f"{placement} summary identity changed")
    return value


def _compare_states(reference_path: Path, candidate_path: Path) -> dict[str, Any]:
    reference = load_file(str(reference_path), device="cpu")
    candidate = load_file(str(candidate_path), device="cpu")
    if set(reference) != set(candidate):
        raise ValueError("state tensor inventory differs by placement")
    rows = []
    failures = []
    for key in sorted(reference):
        metrics = tensor_parity_metrics(
            reference[key], candidate[key], thresholds=DEFAULT_P0_PARITY
        )
        row = {"tensor": key, **metrics}
        rows.append(row)
        if metrics.get("status") != "PASS":
            failures.append(row)
    return {
        "status": "PASS" if not failures else "FAIL",
        "tensor_count": len(rows),
        "failure_count": len(failures),
        "failures": failures[:100],
        "worst": {
            "max_abs": max((float(row["max_abs"]) for row in rows), default=0.0),
            "mean_relative": max(
                (float(row["mean_relative"]) for row in rows), default=0.0
            ),
            "normalized_l2": max(
                (float(row["normalized_l2"]) for row in rows), default=0.0
            ),
            "min_cosine": min(
                (float(row["cosine"]) for row in rows), default=1.0
            ),
        },
    }


def _dtype_rows(summary: dict[str, Any]) -> set[tuple[Any, ...]]:
    evidence = summary.get("correctness_evidence", {})
    rows: set[tuple[Any, ...]] = set()
    for component in ("generator", "critic"):
        contract = evidence.get("lora_compute_dtype", {}).get(component, {})
        for row in contract.get("records", []):
            rows.add(
                (
                    component,
                    row.get("module"),
                    bool(row.get("grad_enabled")),
                    row.get("input_dtype"),
                    row.get("base_weight_dtype"),
                    row.get("lora_a_weight_dtype"),
                    row.get("lora_b_weight_dtype"),
                )
            )
    return rows


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--reference-summary", type=Path, required=True)
    value.add_argument("--candidate-summary", type=Path, required=True)
    value.add_argument("--reference-placement", choices=FAIR_PLACEMENTS, required=True)
    value.add_argument("--candidate-placement", choices=FAIR_PLACEMENTS, required=True)
    value.add_argument("--output", type=Path, required=True)
    return value


def main() -> int:
    args = parser().parse_args()
    reference = _load_summary(args.reference_summary, args.reference_placement)
    candidate = _load_summary(args.candidate_summary, args.candidate_placement)
    errors: list[str] = []
    if reference["identities"] != candidate["identities"]:
        errors.append("bootstrap/model/trace identity differs")
    if reference["workload"] != candidate["workload"]:
        errors.append("fair workload or dtype contract differs")

    reference_dtypes = _dtype_rows(reference)
    candidate_dtypes = _dtype_rows(candidate)
    if not reference_dtypes or not candidate_dtypes:
        errors.append("runtime LoRA compute dtype evidence is missing")
    if reference_dtypes != candidate_dtypes:
        errors.append("runtime LoRA/base compute dtypes differ")

    reference_evidence = reference["correctness_evidence"]
    candidate_evidence = candidate["correctness_evidence"]
    rollout_rows: list[dict[str, Any]] = []
    for rank in range(16):
        reference_state = load_file(
            reference_evidence["rollouts"][rank]["path"], device="cpu"
        )
        candidate_state = load_file(
            candidate_evidence["rollouts"][rank]["path"], device="cpu"
        )
        if set(reference_state) != set(candidate_state):
            errors.append(f"rank{rank}: rollout step inventory differs")
            continue
        for key in sorted(reference_state):
            metrics = tensor_parity_metrics(
                reference_state[key],
                candidate_state[key],
                thresholds=DEFAULT_P0_PARITY,
            )
            row = {"rank": rank, "key": key, **metrics}
            rollout_rows.append(row)
            if metrics.get("status") != "PASS":
                errors.append(f"rank{rank}/{key}: rollout parity failed")

    state_results: dict[str, Any] = {}
    for component in (
        "generator",
        "critic",
        "generator_optimizer",
        "critic_optimizer",
    ):
        reference_path = Path(
            reference_evidence["final_trainables"][component]["path"]
        )
        candidate_path = Path(
            candidate_evidence["final_trainables"][component]["path"]
        )
        state_results[component] = _compare_states(reference_path, candidate_path)
        if state_results[component]["status"] != "PASS":
            errors.append(f"{component} final-state parity failed")

    output = {
        "schema_version": "distillschedule.wan13b_fair_correctness.v1",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "reference": {
            "placement": args.reference_placement,
            "summary": str(args.reference_summary),
        },
        "candidate": {
            "placement": args.candidate_placement,
            "summary": str(args.candidate_summary),
        },
        "identities": reference["identities"],
        "workload": reference["workload"],
        "runtime_dtype_contract": {
            "status": "PASS" if reference_dtypes == candidate_dtypes else "FAIL",
            "rows": [list(value) for value in sorted(reference_dtypes)],
        },
        "thresholds": {
            "mean_relative": DEFAULT_P0_PARITY.mean_relative,
            "normalized_l2": DEFAULT_P0_PARITY.normalized_l2,
            "min_cosine": DEFAULT_P0_PARITY.min_cosine,
            "max_abs_atol": DEFAULT_P0_PARITY.max_abs_atol,
            "max_abs_rtol": DEFAULT_P0_PARITY.max_abs_rtol,
        },
        "rollout": {
            "status": (
                "PASS"
                if all(row.get("status") == "PASS" for row in rollout_rows)
                and len(rollout_rows) == 80
                else "FAIL"
            ),
            "row_count": len(rollout_rows),
            "worst_mean_relative": max(
                (float(row["mean_relative"]) for row in rollout_rows), default=None
            ),
            "worst_normalized_l2": max(
                (float(row["normalized_l2"]) for row in rollout_rows), default=None
            ),
            "min_cosine": min(
                (float(row["cosine"]) for row in rollout_rows), default=None
            ),
            "failures": [
                row for row in rollout_rows if row.get("status") != "PASS"
            ],
        },
        "final_state": state_results,
        "claimable_for_paired_timing": not errors,
    }
    _atomic_json(args.output, output)
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
