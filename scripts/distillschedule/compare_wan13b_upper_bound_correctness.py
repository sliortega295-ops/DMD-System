#!/usr/bin/env python3
"""Compare resident and FSDP2 Wan1.3B correctness artifacts."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from safetensors.torch import load_file

from distillschedule.runtime.p0_parity import DEFAULT_P0_PARITY, tensor_parity_metrics


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


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--resident-summary", type=Path, required=True)
    value.add_argument("--fsdp-summary", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    return value


def main() -> int:
    args = parser().parse_args()
    resident = _load_summary(args.resident_summary, "resident")
    fsdp = _load_summary(args.fsdp_summary, "world16-fsdp")
    errors: list[str] = []
    if resident["identities"] != fsdp["identities"]:
        errors.append("bootstrap/model/trace identity differs by placement")
    if resident["workload"] != fsdp["workload"]:
        errors.append("world16 workload receipt differs by placement")

    resident_evidence = resident["correctness_evidence"]
    fsdp_evidence = fsdp["correctness_evidence"]
    rollout_rows: list[dict[str, Any]] = []
    for rank in range(16):
        resident_state = load_file(
            resident_evidence["rollouts"][rank]["path"], device="cpu"
        )
        fsdp_state = load_file(fsdp_evidence["rollouts"][rank]["path"], device="cpu")
        if set(resident_state) != set(fsdp_state):
            errors.append(f"rank{rank}: rollout step inventory differs")
            continue
        for key in sorted(resident_state):
            metrics = tensor_parity_metrics(
                resident_state[key], fsdp_state[key], thresholds=DEFAULT_P0_PARITY
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
            resident_evidence["final_trainables"][component]["path"]
        )
        candidate_path = Path(fsdp_evidence["final_trainables"][component]["path"])
        state_results[component] = _compare_states(reference_path, candidate_path)
        if state_results[component]["status"] != "PASS":
            errors.append(f"{component} final-state parity failed")

    output = {
        "schema_version": "distillschedule.wan13b_upper_bound_correctness.v1",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "resident_summary": str(args.resident_summary),
        "fsdp_summary": str(args.fsdp_summary),
        "identities": resident["identities"],
        "workload": resident["workload"],
        "thresholds": {
            "mean_relative": DEFAULT_P0_PARITY.mean_relative,
            "normalized_l2": DEFAULT_P0_PARITY.normalized_l2,
            "min_cosine": DEFAULT_P0_PARITY.min_cosine,
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
            "failures": [row for row in rollout_rows if row.get("status") != "PASS"],
        },
        "final_state": state_results,
        "claimable_for_paired_timing": not errors,
    }
    _atomic_json(args.output, output)
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
