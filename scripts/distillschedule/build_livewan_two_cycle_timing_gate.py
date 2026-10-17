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
    resolved_path = path.resolve(strict=True)
    value = _load(resolved_path)
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
    run_root = Path(str(value.get("run_root", ""))).resolve(strict=True)
    if resolved_path.parent != run_root:
        raise ValueError(f"{mode} trajectory summary is outside its canonical run_root")
    anchor_rows = list(value.get("numeric_state_anchors", ()))
    anchor_bindings = []
    for cycle in (1, 2):
        matches = [row for row in anchor_rows if int(row.get("cycle", -1)) == cycle]
        if len(matches) != 1:
            raise ValueError(f"{mode} summary does not bind exactly one cycle{cycle} anchor")
        expected_path = (
            run_root / "trajectory" / "anchors" / f"cycle_{cycle:02d}_global.json"
        ).resolve(strict=True)
        if Path(str(matches[0].get("path", ""))).resolve(strict=True) != expected_path:
            raise ValueError(f"{mode} summary cycle{cycle} anchor path is stale")
        anchor = _load(expected_path)
        canonical = dict(anchor)
        recorded = str(canonical.pop("anchor_sha256", ""))
        actual = hashlib.sha256(
            json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if (
            anchor.get("schema_version")
            != "distillschedule.livewan_trajectory_state_anchor.v1"
            or int(anchor.get("cycle", -1)) != cycle
            or recorded != actual
            or matches[0].get("sha256") != actual
        ):
            raise ValueError(f"{mode} cycle{cycle} anchor digest is stale")
        anchor_bindings.append(
            {
                "cycle": cycle,
                "path": str(expected_path),
                "payload_sha256": actual,
                "file_sha256": _sha256(expected_path),
            }
        )
    value["_artifact_binding"] = {
        "run_root": str(run_root),
        "summary": {"path": str(resolved_path), "sha256": _sha256(resolved_path)},
        "anchors": anchor_bindings,
    }
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-trajectory-summary", type=Path, required=True)
    parser.add_argument("--pipeline-trajectory-summary", type=Path, required=True)
    parser.add_argument("--trajectory-comparison", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--pipeline-variant", choices=("c24-continuous-b16-score-modulation", "c25-continuous-b16-score-rmsnorm", "c26-continuous-b16-critic-attention-checkpoint", "c27-continuous-b16-pinned-prompt-staging", "c28-continuous-b16-rollout-cast-cache", "c29-continuous-b16-generator-rms-epilogue", "c30-continuous-b16-generator-rms-critic-checkpoint", "c31-continuous-b16-generator-rms-cast-square", "c32-continuous-b16-generator-modulation", "c33-continuous-b16-generator-gate", "c34-continuous-b16-residual-layout", "c35-continuous-b16-score-gate", "c36-continuous-b16-score-layernorm", "c37-continuous-b16-step1-overlap", 'c38-continuous-b16-score-fa3', 'c49-continuous-b16-generator-readonly-splitkv', 'c51-continuous-b16-generator-fused-rms', 'c51-critic-fa3-recompute-v1', 'c52-c51-critic-ffn-down-retention-v1', 'c39-continuous-b16-step0-delivery', 'c40-continuous-b16-score-shared-input', 'c44-continuous-b16-critic-ffn-down-cache'), default="c24-continuous-b16-score-modulation")
    parser.add_argument("--reference-variant", choices=(BASELINE_VARIANT, "c51-continuous-b16-generator-fused-rms"), default=BASELINE_VARIANT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.expected_head) != 40:
        raise ValueError("--expected-head must be a full Git SHA")

    conditional = args.reference_variant != BASELINE_VARIANT
    if conditional != (args.pipeline_variant in ("c51-critic-fa3-recompute-v1", "c52-c51-critic-ffn-down-retention-v1")):
        raise ValueError("C51 timing reference is only registered for its replay ablation")
    reference_mode = "pipeline" if conditional else "baseline"
    baseline = _trajectory(
        args.baseline_trajectory_summary,
        mode=reference_mode,
        variant=args.reference_variant,
        expected_head=args.expected_head,
    )
    pipeline = _trajectory(
        args.pipeline_trajectory_summary,
        mode="pipeline",
        variant=args.pipeline_variant,
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
        for component in STATE_COMPONENTS
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
        or len(state_rows) != 8
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
    numeric_eligibility = dict(comparison.get("numeric_eligibility", {}))
    extreme_violations = []
    for row in state_rows:
        cycle = int(row.get("cycle", -1))
        component = str(row.get("component", ""))
        for metric, value, operator, threshold in (
            ("cosine", float(row.get("cosine", float("nan"))), ">=", 0.9),
            (
                "normalized_l2",
                float(row.get("normalized_l2", float("nan"))),
                "<=",
                0.5,
            ),
            ("norm_ratio", float(row.get("norm_ratio", float("nan"))), ">=", 0.5),
            ("norm_ratio", float(row.get("norm_ratio", float("nan"))), "<=", 2.0),
        ):
            failed = (
                not isinstance(value, float)
                or value != value
                or value in (float("inf"), float("-inf"))
                or (operator == ">=" and value < threshold)
                or (operator == "<=" and value > threshold)
            )
            if failed:
                extreme_violations.append(
                    {
                        "cycle": cycle,
                        "component": component,
                        "metric": metric,
                        "value": value,
                    }
                )
    if (
        numeric_eligibility.get("schema_version")
        != "distillschedule.livewan_two_cycle_numeric_eligibility.v1"
        or numeric_eligibility.get("status") != "PASS"
        or numeric_eligibility.get("policy")
        != "REJECT_EXTREME_ONLY_RECORD_ORDINARY_REVIEW"
        or numeric_eligibility.get("thresholds") != EXTREME_NUMERIC_THRESHOLDS
        or numeric_eligibility.get("violations") != []
        or extreme_violations
    ):
        raise ValueError("cross-arm trajectory has extreme numerical drift")

    def expected_binding(value: dict[str, Any]) -> dict[str, Any]:
        artifact = dict(value["_artifact_binding"])
        return {
            "run_id": value["run_id"],
            "arm_id": value["arm_id"],
            "run_root": artifact["run_root"],
            "seed": value["seed"],
            "identity": value["identity"],
            "summary": artifact["summary"],
            "anchors": artifact["anchors"],
        }

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
    if source_binding.get("expected_head") != args.expected_head or (
        source_baseline != expected_binding(baseline)
        or source_pipeline != expected_binding(pipeline)
    ):
        raise ValueError("trajectory comparison source binding is not exact")

    numerical_review = {
        "policy": numeric_eligibility["policy"],
        "extreme_thresholds": numeric_eligibility["thresholds"],
        "extreme_failure_count": 0,
        "state_comparisons": [
            row for row in state_rows if row.get("status") != "PASS"
        ],
        "loss_curves": {
            key: row
            for key, row in dict(comparison.get("curves", {})).items()
            if row.get("status") != "PASS"
        },
        "gradient_norm_curves": {
            key: row
            for key, row in dict(comparison.get("gradient_norm_curves", {})).items()
            if row.get("status") != "PASS"
        },
    }

    result = {
        "schema_version": "distillschedule.livewan_two_cycle_timing_gate.v2",
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
        "numerical_review": numerical_review,
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
                "artifact_binding": baseline["_artifact_binding"],
            },
            "pipeline": {
                "run_id": pipeline["run_id"],
                "arm_id": pipeline["arm_id"],
                "pipeline_variant": args.pipeline_variant,
                "trajectory_summary": str(
                    args.pipeline_trajectory_summary.resolve()
                ),
                "trajectory_summary_sha256": _sha256(
                    args.pipeline_trajectory_summary
                ),
                "artifact_binding": pipeline["_artifact_binding"],
            },
        },
    }
    if conditional:
        if (source_binding.get("reference_execution_mode") != "pipeline"
            or source_binding.get("reference_variant") != args.reference_variant
            or source_binding.get("candidate_variant") != args.pipeline_variant):
            raise ValueError("conditional replay comparison binding changed")
        reference_arm = result["arms"]["baseline"]
        reference_arm["pipeline_variant"] = args.reference_variant
        result.update(
            schema_version="distillschedule.livewan_conditional_replay_timing_gate.v1",
            claim_boundary=("SAME_HEAD_FULL_C51_VS_FROZEN_FFN_REUSE_DIAGNOSTIC_ONLY" if args.pipeline_variant == "c52-c51-critic-ffn-down-retention-v1" else "SAME_HEAD_FULL_C51_VS_REPLAY_RECOMPUTE_DIAGNOSTIC_ONLY"),
            heads={"reference": args.expected_head, "candidate": args.expected_head},
            arms={"reference": reference_arm, "candidate": result["arms"]["pipeline"]})
    _atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
