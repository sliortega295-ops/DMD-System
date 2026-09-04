#!/usr/bin/env python3
"""Compare the exact two-cycle LiveWan trajectory arms before timing.

This is deliberately a contract/evidence command, not a timing command.  It
loads all 32 append-only rank-cycle receipts from each arm and preserves the
generic comparator's post-run numerical review without using that review as an
early-stop or timing measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from distillschedule.evidence.trajectory_coarse import (
    TrajectoryStateComparison,
    compare_trajectories,
    load_trajectory_jsonl,
)
from distillschedule.evidence.trajectory_state_sketch import (
    SCHEMA_VERSION as STATE_SKETCH_SCHEMA_VERSION,
)


BASELINE_VARIANT = "b1"
PIPELINE_VARIANT = "c24-continuous-b16-score-modulation"
STATE_COMPONENTS = (
    "generator_trainable",
    "generator_adam",
    "critic_trainable",
    "critic_adam",
)


def _launch_manifest(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            rows[key] = value
    return rows


def _validate_arm(
    root: Path,
    *,
    mode: str,
    variant: str,
    expected_head: str,
) -> dict[str, Any]:
    manifests = [
        _launch_manifest(root / "launcher" / f"node_{node}" / "launch_manifest.txt")
        for node in range(2)
    ]
    expected = {
        "head": expected_head,
        "run_kind": "trajectory",
        "trajectory_cycles": "2",
        "execution_mode": mode,
        "pipeline_variant": variant,
        "claim_boundary": "DIAGNOSTIC_ONLY",
    }
    for node, manifest in enumerate(manifests):
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ValueError(f"{mode} node{node} launch manifest changed {key}")
    for key in ("run_id", "trajectory_arm_id", "trajectory_seed"):
        values = {manifest.get(key) for manifest in manifests}
        if len(values) != 1 or None in values or "" in values:
            raise ValueError(f"{mode} two-node launch manifest disagrees on {key}")

    rows = []
    for rank in range(16):
        path = root / "trajectory" / f"rank_{rank:03d}.jsonl"
        rank_rows = load_trajectory_jsonl(path)
        if len(rank_rows) != 2 or [row.cycle for row in rank_rows] != [1, 2]:
            raise ValueError(f"{mode} rank{rank} is not a complete two-cycle trajectory")
        rows.extend(rank_rows)
    return {
        "rows": rows,
        "run_id": manifests[0]["run_id"],
        "arm_id": manifests[0]["trajectory_arm_id"],
        "seed": int(manifests[0]["trajectory_seed"]),
        "run_root": str(root.resolve()),
    }


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_anchor(root: Path, cycle: int) -> dict[str, Any]:
    path = root / "trajectory" / "anchors" / f"cycle_{cycle:02d}_global.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    recorded = str(value.get("anchor_sha256", ""))
    canonical = dict(value)
    canonical.pop("anchor_sha256", None)
    actual = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if (
        value.get("schema_version")
        != "distillschedule.livewan_trajectory_state_anchor.v1"
        or int(value.get("cycle", -1)) != cycle
        or recorded != actual
        or set(dict(value.get("components", {}))) != set(STATE_COMPONENTS)
    ):
        raise ValueError(f"invalid LiveWan state anchor: {path}")
    return value


def _vector_metrics(reference: list[float], candidate: list[float]) -> tuple[float, float, float]:
    if len(reference) != len(candidate) or not reference:
        raise ValueError("numeric sketch vectors must be equally sized and non-empty")
    dot = sum(left * right for left, right in zip(reference, candidate, strict=True))
    reference_norm = math.sqrt(sum(value * value for value in reference))
    candidate_norm = math.sqrt(sum(value * value for value in candidate))
    delta_norm = math.sqrt(
        sum(
            (left - right) * (left - right)
            for left, right in zip(reference, candidate, strict=True)
        )
    )
    if reference_norm == 0.0 and candidate_norm == 0.0:
        return 1.0, 0.0, 1.0
    cosine = 0.0 if reference_norm == 0.0 or candidate_norm == 0.0 else dot / (
        reference_norm * candidate_norm
    )
    denominator = max(reference_norm, 1e-30)
    return cosine, delta_norm / denominator, candidate_norm / denominator


def _compare_state(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    cycle: int,
    component: str,
) -> TrajectoryStateComparison:
    for label, value in (("reference", reference), ("candidate", candidate)):
        if (
            value.get("schema_version") != STATE_SKETCH_SCHEMA_VERSION
            or value.get("coordinate_schema")
            != "canonical_tensor_name/global_flat_index/value_fp32"
        ):
            raise ValueError(f"{label} {component} numeric-sketch schema changed")
    for field in (
        "inventory_tensor_count",
        "inventory_sha256",
        "tensor_count",
        "coordinate_count",
        "samples_per_tensor",
        "max_tensors",
    ):
        if reference.get(field) != candidate.get(field):
            raise ValueError(f"{component} numeric sketch {field} mismatch")

    def indexed(value: dict[str, Any]) -> dict[tuple[str, int], float]:
        result: dict[tuple[str, int], float] = {}
        for row in value.get("coordinates", ()):
            key = (str(row["tensor"]), int(row["flat_index"]))
            number = float(row["value_fp32"])
            if key in result or not math.isfinite(number):
                raise ValueError(f"invalid {component} numeric coordinate {key}")
            result[key] = number
        return result

    left = indexed(reference)
    right = indexed(candidate)
    if left.keys() != right.keys() or not left:
        raise ValueError(f"{component} numeric coordinate inventory mismatch")
    keys = sorted(left)
    aggregate = _vector_metrics([left[key] for key in keys], [right[key] for key in keys])
    named_cosines = []
    for name in sorted({name for name, _index in keys}):
        named = [key for key in keys if key[0] == name]
        named_cosines.append(
            _vector_metrics([left[key] for key in named], [right[key] for key in named])[0]
        )
    return TrajectoryStateComparison(
        cycle=cycle,
        component=component,
        state_kind="numeric_state_sketch",
        cosine=aggregate[0],
        normalized_l2=aggregate[1],
        norm_ratio=aggregate[2],
        min_named_tensor_cosine=min(named_cosines),
        reference_digest=hashlib.sha256(
            json.dumps(reference, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        candidate_digest=hashlib.sha256(
            json.dumps(candidate, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run-root", type=Path, required=True)
    parser.add_argument("--pipeline-run-root", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.expected_head) != 40:
        raise ValueError("--expected-head must be a full Git SHA")

    baseline = _validate_arm(
        args.baseline_run_root,
        mode="baseline",
        variant=BASELINE_VARIANT,
        expected_head=args.expected_head,
    )
    pipeline = _validate_arm(
        args.pipeline_run_root,
        mode="pipeline",
        variant=PIPELINE_VARIANT,
        expected_head=args.expected_head,
    )
    if baseline["seed"] != pipeline["seed"]:
        raise ValueError("trajectory arm seeds differ")

    state_comparisons = []
    for cycle in (1, 2):
        baseline_anchor = _load_anchor(args.baseline_run_root, cycle)
        pipeline_anchor = _load_anchor(args.pipeline_run_root, cycle)
        for component in STATE_COMPONENTS:
            state_comparisons.append(
                _compare_state(
                    baseline_anchor["components"][component],
                    pipeline_anchor["components"][component],
                    cycle=cycle,
                    component=component,
                )
            )

    result = compare_trajectories(
        baseline["rows"],
        pipeline["rows"],
        state_comparisons=state_comparisons,
        expected_cycles=2,
        expected_world_size=16,
        comparison_kind="semantic",
        required_state_cycles=(1, 2),
    )
    result["source_binding"] = {
        "expected_head": args.expected_head,
        "baseline": {
            "run_id": baseline["run_id"],
            "arm_id": baseline["arm_id"],
            "run_root": baseline["run_root"],
        },
        "pipeline": {
            "run_id": pipeline["run_id"],
            "arm_id": pipeline["arm_id"],
            "run_root": pipeline["run_root"],
        },
    }
    _atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("complete") and result.get("contract", {}).get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
