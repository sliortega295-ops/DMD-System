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
EXTREME_NUMERIC_THRESHOLDS = {
    "minimum_cosine": 0.9,
    "maximum_normalized_l2": 0.5,
    "minimum_norm_ratio": 0.5,
    "maximum_norm_ratio": 2.0,
}


def _launch_manifest(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            rows[key] = value
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_summary(
    path: Path,
    *,
    root: Path,
    mode: str,
    variant: str,
    expected_head: str,
) -> dict[str, Any]:
    resolved_path = path.resolve(strict=True)
    root = root.resolve(strict=True)
    value = json.loads(resolved_path.read_text(encoding="utf-8"))
    required = {
        "schema_version": "distillschedule.livewan_trajectory_summary.v1",
        "status": "PASS",
        "expected_head": expected_head,
        "head": expected_head,
        "heads": {mode: expected_head},
        "execution_mode": mode,
        "pipeline_variant": variant,
        "cycles": 2,
        "rank_cycle_receipts": 32,
        "expected_rank_cycle_receipts": 32,
        "final_versions": {"generator": 2, "critic": 10, "ema": 0},
        "formal_timing_eligible": False,
    }
    for key, expected in required.items():
        if value.get(key) != expected:
            raise ValueError(f"{mode} trajectory summary changed {key}")
    if Path(str(value.get("run_root", ""))).resolve(strict=True) != root:
        raise ValueError(f"{mode} trajectory summary run_root is stale")
    if resolved_path.parent != root:
        raise ValueError(f"{mode} trajectory summary is outside its canonical run_root")
    return {
        "value": value,
        "path": str(resolved_path),
        "sha256": _sha256(resolved_path),
    }


def _validate_arm(
    root: Path,
    *,
    summary_path: Path,
    mode: str,
    variant: str,
    expected_head: str,
) -> dict[str, Any]:
    root = root.resolve(strict=True)
    summary = _load_summary(
        summary_path,
        root=root,
        mode=mode,
        variant=variant,
        expected_head=expected_head,
    )
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
    summary_value = summary["value"]
    if (
        summary_value.get("run_id") != manifests[0]["run_id"]
        or summary_value.get("arm_id") != manifests[0]["trajectory_arm_id"]
        or int(summary_value.get("seed", -1))
        != int(manifests[0]["trajectory_seed"])
    ):
        raise ValueError(f"{mode} trajectory summary identity differs from launch")

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
        "run_root": str(root),
        "summary": summary,
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
    return {
        "value": value,
        "binding": {
            "cycle": cycle,
            "path": str(path.resolve()),
            "payload_sha256": recorded,
            "file_sha256": _sha256(path.resolve()),
        },
    }


def _validate_summary_anchor_binding(
    summary: dict[str, Any], anchor: dict[str, Any], *, cycle: int, mode: str
) -> None:
    rows = list(summary.get("numeric_state_anchors", ()))
    matching = [row for row in rows if int(row.get("cycle", -1)) == cycle]
    if len(matching) != 1:
        raise ValueError(f"{mode} summary does not bind exactly one cycle{cycle} anchor")
    row = matching[0]
    binding = anchor["binding"]
    if (
        Path(str(row.get("path", ""))).resolve(strict=True)
        != Path(binding["path"])
        or row.get("sha256") != binding["payload_sha256"]
    ):
        raise ValueError(f"{mode} summary cycle{cycle} anchor binding is stale")


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
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--pipeline-summary", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--pipeline-variant", choices=(PIPELINE_VARIANT, "c25-continuous-b16-score-rmsnorm", "c26-continuous-b16-critic-attention-checkpoint", "c27-continuous-b16-pinned-prompt-staging"), default=PIPELINE_VARIANT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.expected_head) != 40:
        raise ValueError("--expected-head must be a full Git SHA")

    baseline = _validate_arm(
        args.baseline_run_root,
        summary_path=args.baseline_summary,
        mode="baseline",
        variant=BASELINE_VARIANT,
        expected_head=args.expected_head,
    )
    pipeline = _validate_arm(
        args.pipeline_run_root,
        summary_path=args.pipeline_summary,
        mode="pipeline",
        variant=args.pipeline_variant,
        expected_head=args.expected_head,
    )
    if baseline["seed"] != pipeline["seed"]:
        raise ValueError("trajectory arm seeds differ")
    if (
        baseline["summary"]["value"].get("identity")
        != pipeline["summary"]["value"].get("identity")
    ):
        raise ValueError("trajectory trace/bootstrap/source identity differs")

    state_comparisons = []
    anchor_bindings: dict[str, list[dict[str, Any]]] = {
        "baseline": [],
        "pipeline": [],
    }
    for cycle in (1, 2):
        baseline_anchor = _load_anchor(args.baseline_run_root, cycle)
        pipeline_anchor = _load_anchor(args.pipeline_run_root, cycle)
        _validate_summary_anchor_binding(
            baseline["summary"]["value"],
            baseline_anchor,
            cycle=cycle,
            mode="baseline",
        )
        _validate_summary_anchor_binding(
            pipeline["summary"]["value"],
            pipeline_anchor,
            cycle=cycle,
            mode="pipeline",
        )
        anchor_bindings["baseline"].append(baseline_anchor["binding"])
        anchor_bindings["pipeline"].append(pipeline_anchor["binding"])
        for component in STATE_COMPONENTS:
            state_comparisons.append(
                _compare_state(
                    baseline_anchor["value"]["components"][component],
                    pipeline_anchor["value"]["components"][component],
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
    violations: list[dict[str, Any]] = []
    for row in result.get("state_comparisons", ()):
        checks = (
            ("cosine", float(row.get("cosine", float("nan"))), ">=", 0.9),
            (
                "normalized_l2",
                float(row.get("normalized_l2", float("nan"))),
                "<=",
                0.5,
            ),
            ("norm_ratio", float(row.get("norm_ratio", float("nan"))), ">=", 0.5),
            ("norm_ratio", float(row.get("norm_ratio", float("nan"))), "<=", 2.0),
        )
        for metric, value, operator, threshold in checks:
            failed = (
                not math.isfinite(value)
                or (operator == ">=" and value < threshold)
                or (operator == "<=" and value > threshold)
            )
            if failed:
                violations.append(
                    {
                        "cycle": int(row.get("cycle", -1)),
                        "component": str(row.get("component", "")),
                        "metric": metric,
                        "value": value if math.isfinite(value) else None,
                        "required": f"{operator}{threshold}",
                    }
                )
    result["numeric_eligibility"] = {
        "schema_version": "distillschedule.livewan_two_cycle_numeric_eligibility.v1",
        "status": "PASS" if not violations else "FAIL_EXTREME_DRIFT",
        "policy": "REJECT_EXTREME_ONLY_RECORD_ORDINARY_REVIEW",
        "thresholds": EXTREME_NUMERIC_THRESHOLDS,
        "violations": violations,
    }
    result["source_binding"] = {
        "expected_head": args.expected_head,
        "baseline": {
            "run_id": baseline["run_id"],
            "arm_id": baseline["arm_id"],
            "run_root": baseline["run_root"],
            "seed": baseline["seed"],
            "identity": baseline["summary"]["value"]["identity"],
            "summary": {
                "path": baseline["summary"]["path"],
                "sha256": baseline["summary"]["sha256"],
            },
            "anchors": anchor_bindings["baseline"],
        },
        "pipeline": {
            "run_id": pipeline["run_id"],
            "arm_id": pipeline["arm_id"],
            "run_root": pipeline["run_root"],
            "seed": pipeline["seed"],
            "identity": pipeline["summary"]["value"]["identity"],
            "summary": {
                "path": pipeline["summary"]["path"],
                "sha256": pipeline["summary"]["sha256"],
            },
            "anchors": anchor_bindings["pipeline"],
        },
    }
    _atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return (
        0
        if result.get("complete")
        and result.get("contract", {}).get("status") == "PASS"
        and result["numeric_eligibility"]["status"] == "PASS"
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
