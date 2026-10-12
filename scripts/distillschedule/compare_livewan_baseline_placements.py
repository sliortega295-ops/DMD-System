#!/usr/bin/env python3
"""Bind a new placement's two-cycle correctness evidence before native timing.

Reuses the existing sample/RNG/census/state comparator and extreme-drift policy.
Reference may be historical on a different explicitly recorded source HEAD;
this gate is NOT evidence of a matched performance comparison across revisions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from compare_livewan_two_cycle_trajectories import (
    STATE_COMPONENTS, EXTREME_NUMERIC_THRESHOLDS, _validate_arm, _load_anchor,
    _validate_summary_anchor_binding, _compare_state, _atomic_json, _sha256,
)
from distillschedule.evidence.trajectory_coarse import compare_trajectories
from distillschedule.runtime.livewan_baseline_placement import BaselinePlacement


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-root", type=Path, required=True)
    p.add_argument("--reference-head", required=True)
    p.add_argument("--candidate-root", type=Path, required=True)
    p.add_argument("--candidate-head", required=True)
    p.add_argument("--placement", required=True)
    p.add_argument("--teacher-offload", default="off")
    p.add_argument("--optimizer-device", default="cuda", choices=("cuda", "cpu"))
    p.add_argument("--student-rollout-padding", default="off", choices=("off", "max-noop"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--timing-gate", type=Path, required=True)
    args = p.parse_args()
    spec = BaselinePlacement(args.placement, args.teacher_offload,
                             args.optimizer_device, args.student_rollout_padding)
    arms = []
    for root, head in ((args.reference_root, args.reference_head),
                       (args.candidate_root, args.candidate_head)):
        arms.append(_validate_arm(root, summary_path=root / "summary.json",
                                  mode="baseline", variant="b1", expected_head=head))
    left, right = arms
    if (left["seed"] != right["seed"] or
        left["summary"]["value"]["identity"] != right["summary"]["value"]["identity"]):
        raise ValueError("placement comparison changed assets or RNG seed")
    comparisons = []
    for cycle in (1, 2):
        anchors = [_load_anchor(root, cycle) for root in (args.reference_root, args.candidate_root)]
        for arm, anchor in zip(arms, anchors):
            _validate_summary_anchor_binding(arm["summary"]["value"], anchor,
                                           cycle=cycle, mode="baseline")
        for component in STATE_COMPONENTS:
            comparisons.append(_compare_state(anchors[0]["value"]["components"][component],
                anchors[1]["value"]["components"][component], cycle=cycle, component=component))
    result = compare_trajectories(left["rows"], right["rows"], state_comparisons=comparisons,
        expected_cycles=2, expected_world_size=16, comparison_kind="semantic", required_state_cycles=(1, 2))
    violations = []
    for row in result.get("state_comparisons", ()):
        for metric, op, threshold in (("cosine", ">=", .9), ("normalized_l2", "<=", .5),
                                      ("norm_ratio", ">=", .5), ("norm_ratio", "<=", 2.)):
            value = float(row.get(metric, float("nan")))
            if not math.isfinite(value) or (value < threshold if op == ">=" else value > threshold):
                violations.append({"cycle": row["cycle"], "component": row["component"],
                                   "metric": metric, "value": value if math.isfinite(value) else None,
                                   "required": f"{op}{threshold}"})
    result["numeric_eligibility"] = {"status": "PASS" if not violations else "FAIL_EXTREME_DRIFT",
        "policy": "REJECT_EXTREME_ONLY_RECORD_ORDINARY_REVIEW",
        "thresholds": EXTREME_NUMERIC_THRESHOLDS, "violations": violations}
    result["placement"] = spec.receipt()
    result["source_binding"] = {"reference_head": args.reference_head, "candidate_head": args.candidate_head,
        "reference": {key:left[key] for key in ("run_id", "arm_id", "run_root", "seed")},
        "candidate": {key:right[key] for key in ("run_id", "arm_id", "run_root", "seed")}}
    result["matched_performance_comparison"] = "NOT_RUN"
    _atomic_json(args.output, result)
    passed = (result.get("complete") is True and result.get("contract", {}).get("status") == "PASS"
              and not violations)
    if not passed:
        print(json.dumps({"status": "FAIL", "comparison": str(args.output)}, indent=2))
        return 1
    for node in (0, 1):
        manifest = dict(line.split("=", 1) for line in (args.candidate_root / "launcher" / f"node_{node}" /
            "launch_manifest.txt").read_text().splitlines() if "=" in line)
        if (manifest.get("baseline_placement") != spec.name or
            manifest.get("teacher_parameter_offload") != spec.teacher_offload or
            manifest.get("optimizer_device", "cuda") != spec.optimizer_device or
            manifest.get("student_rollout_padding", "off") != spec.student_padding):
            raise ValueError("candidate placement differs from its launch manifest")
    gate = {"schema_version": "distillschedule.livewan_baseline_placement_timing_gate.v1",
        "status": "PASS", "claimable_for_two_cycle_timing": True, "cycles": 2,
        "final_versions": {"generator": 2, "critic": 10, "ema": 0},
        "expected_head": args.candidate_head, "baseline_placement": spec.receipt(),
        "identity": right["summary"]["value"]["identity"], "seed": right["seed"], "arm_id": right["arm_id"],
        "numerical_policy": "REJECT_EXTREME_ONLY_RECORD_ORDINARY_REVIEW",
        "correctness_summary": {"path": right["summary"]["path"], "sha256": right["summary"]["sha256"]},
        "comparison": {"path": str(args.output.resolve()), "sha256": _sha256(args.output)}}
    _atomic_json(args.timing_gate, gate)
    print(json.dumps({"status": "PASS", "timing_gate": str(args.timing_gate),
                      "sha256": _sha256(args.timing_gate)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
