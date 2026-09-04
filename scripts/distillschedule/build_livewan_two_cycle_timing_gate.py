#!/usr/bin/env python3
"""Build the exact-head evidence gate for LiveWan two-cycle native timing."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


BASELINE_VARIANT = "b1"
PIPELINE_VARIANT = "c24-continuous-b16-score-modulation"
FINAL_VERSIONS = {"generator": 2, "critic": 10, "ema": 0}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _single(values: Any, *, label: str) -> Any:
    if not isinstance(values, list) or len(values) != 1:
        raise ValueError(f"{label} must contain exactly one value")
    return values[0]


def _trajectory(
    path: Path,
    *,
    mode: str,
    variant: str,
    expected_head: str,
) -> dict[str, Any]:
    value = _load(path)
    required = {
        "schema_version": "distillschedule.livewan_trajectory_summary.v1",
        "status": "PASS",
        "expected_head": expected_head,
        "head": expected_head,
        "execution_mode": mode,
        "pipeline_variant": variant,
        "cycles": 2,
        "rank_cycle_receipts": 32,
        "expected_rank_cycle_receipts": 32,
        "final_versions": FINAL_VERSIONS,
        "formal_timing_eligible": False,
    }
    for key, expected in required.items():
        if value.get(key) != expected:
            raise ValueError(f"{mode} trajectory summary changed {key}")
    if value.get("heads") != {mode: expected_head}:
        raise ValueError(f"{mode} trajectory source HEAD is not exact")
    if not value.get("run_id") or not value.get("arm_id"):
        raise ValueError(f"{mode} trajectory run/arm identity is missing")
    if not isinstance(value.get("seed"), int) or int(value["seed"]) < 0:
        raise ValueError(f"{mode} trajectory seed is missing")
    identity = value.get("identity")
    if not isinstance(identity, dict) or set(identity) != {
        "trace_sha256",
        "bootstrap_sha256",
        "source_receipt_sha256",
    }:
        raise ValueError(f"{mode} trajectory asset identity is incomplete")
    if any(not isinstance(item, str) or len(item) != 64 for item in identity.values()):
        raise ValueError(f"{mode} trajectory asset SHA is invalid")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-trajectory-summary", type=Path, required=True)
    parser.add_argument("--pipeline-trajectory-summary", type=Path, required=True)
    parser.add_argument("--trajectory-comparison", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.expected_head) != 40:
        raise ValueError("--expected-head must be a full Git SHA")

    baseline = _trajectory(
        args.baseline_trajectory_summary,
        mode="baseline",
        variant=BASELINE_VARIANT,
        expected_head=args.expected_head,
    )
    pipeline = _trajectory(
        args.pipeline_trajectory_summary,
        mode="pipeline",
        variant=PIPELINE_VARIANT,
        expected_head=args.expected_head,
    )
    if baseline["seed"] != pipeline["seed"]:
        raise ValueError("trajectory seeds differ")
    if baseline["identity"] != pipeline["identity"]:
        raise ValueError("trajectory trace/bootstrap/source identity differs")

    comparison = _load(args.trajectory_comparison)
    state_rows = list(comparison.get("state_comparisons", ()))
    expected_state_rows = {
        (cycle, component)
        for cycle in (1, 2)
        for component in (
            "generator_trainable",
            "generator_adam",
            "critic_trainable",
            "critic_adam",
        )
    }
    observed_state_rows = {
        (int(row.get("cycle", -1)), str(row.get("component", "")))
        for row in state_rows
    }
    if (
        comparison.get("schema_version")
        != "distillschedule.trajectory_coarse_comparison.v1"
        or comparison.get("comparison_kind") != "semantic"
        or comparison.get("expected_cycles") != 2
        or comparison.get("expected_world_size") != 16
        or comparison.get("complete") is not True
        or dict(comparison.get("contract", {})).get("status") != "PASS"
        or dict(comparison.get("contract", {})).get("mismatches") != []
        or int(dict(comparison.get("summary", {})).get("state_comparison_count", -1))
        != 8
        or observed_state_rows != expected_state_rows
        or any(row.get("finite") is not True for row in state_rows)
    ):
        raise ValueError(
            "cross-arm two-cycle trajectory contract/state comparison is not complete/PASS"
        )
    reference = dict(comparison.get("reference", {}))
    candidate = dict(comparison.get("candidate", {}))
    source_binding = dict(comparison.get("source_binding", {}))
    source_baseline = dict(source_binding.get("baseline", {}))
    source_pipeline = dict(source_binding.get("pipeline", {}))
    if (
        _single(reference.get("run_ids"), label="reference run_ids")
        != baseline["run_id"]
        or _single(candidate.get("run_ids"), label="candidate run_ids")
        != pipeline["run_id"]
        or _single(reference.get("variants"), label="reference variants")
        != baseline["arm_id"]
        or _single(candidate.get("variants"), label="candidate variants")
        != pipeline["arm_id"]
        or _single(reference.get("seeds"), label="reference seeds")
        != baseline["seed"]
        or _single(candidate.get("seeds"), label="candidate seeds")
        != pipeline["seed"]
    ):
        raise ValueError("trajectory comparison is not bound to these exact arms")
    if (
        source_binding.get("expected_head") != args.expected_head
        or source_baseline.get("run_id") != baseline["run_id"]
        or source_baseline.get("arm_id") != baseline["arm_id"]
        or source_pipeline.get("run_id") != pipeline["run_id"]
        or source_pipeline.get("arm_id") != pipeline["arm_id"]
    ):
        raise ValueError("trajectory comparison source binding is not exact")

    result = {
        "schema_version": "distillschedule.livewan_two_cycle_timing_gate.v1",
        "status": "PASS",
        "claimable_for_two_cycle_timing": True,
        "claim_boundary": (
            "EXACT_HEAD_TWO_CYCLE_TRAJECTORY_AND_NUMERIC_STATE_COMPARISON"
        ),
        "expected_head": args.expected_head,
        "heads": {"baseline": args.expected_head, "pipeline": args.expected_head},
        "cycles": 2,
        "final_versions": FINAL_VERSIONS,
        "seed": baseline["seed"],
        "identity": baseline["identity"],
        "trajectory_comparison": {
            "path": str(args.trajectory_comparison.resolve()),
            "sha256": _sha256(args.trajectory_comparison),
            "schema_version": comparison["schema_version"],
            "status": comparison.get("status"),
            "contract_status": "PASS",
            "complete": True,
        },
        "arms": {
            "baseline": {
                "run_id": baseline["run_id"],
                "arm_id": baseline["arm_id"],
                "pipeline_variant": BASELINE_VARIANT,
                "trajectory_summary": str(
                    args.baseline_trajectory_summary.resolve()
                ),
                "trajectory_summary_sha256": _sha256(
                    args.baseline_trajectory_summary
                ),
            },
            "pipeline": {
                "run_id": pipeline["run_id"],
                "arm_id": pipeline["arm_id"],
                "pipeline_variant": PIPELINE_VARIANT,
                "trajectory_summary": str(
                    args.pipeline_trajectory_summary.resolve()
                ),
                "trajectory_summary_sha256": _sha256(
                    args.pipeline_trajectory_summary
                ),
            },
        },
    }
    _atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
