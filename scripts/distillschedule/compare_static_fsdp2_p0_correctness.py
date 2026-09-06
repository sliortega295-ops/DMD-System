#!/usr/bin/env python3
"""Compare a P0 candidate correctness summary against the B1/off control.

Exact local-shard SHA equality is the only condition labelled exact training
state equivalence.  Per-step gradient checksums and loss differences make the
source of any divergence visible before performance runs.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if value.get("status") != "PASS" or not value.get("claimable_correctness"):
        raise ValueError(f"correctness summary is not claimable PASS: {path}")
    return value


def _rows(group: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["name"]): row for row in group.get("tensors", [])}


def _relative(left: float, right: float, floor: float = 1e-12) -> float:
    return abs(left - right) / max(abs(left), floor)


def _compare_group(
    left_group: dict[str, Any],
    right_group: dict[str, Any],
    *,
    label: str,
) -> tuple[bool, dict[str, Any]]:
    left_rows = _rows(left_group)
    right_rows = _rows(right_group)
    if set(left_rows) != set(right_rows):
        raise ValueError(f"{label} tensor inventory mismatch")
    group_exact = left_group.get("sha256") == right_group.get("sha256")
    tensor_rows = []
    for name in sorted(left_rows):
        a, b = left_rows[name], right_rows[name]
        if int(a["numel"]) != int(b["numel"]) or a["shape"] != b["shape"]:
            raise ValueError(f"{label}/{name} shape mismatch")
        tensor_rows.append(
            {
                "name": name,
                "exact_sha256_match": a.get("sha256") == b.get("sha256"),
                "l2_stat_relative_difference": _relative(float(a["l2"]), float(b["l2"])),
                "max_abs_stat_relative_difference": _relative(
                    float(a["max_abs"]), float(b["max_abs"]), 1e-6
                ),
                "sum_stat_relative_difference": _relative(
                    float(a["sum"]), float(b["sum"]), 1e-6
                ),
            }
        )
    return group_exact, {
        "exact_sha256_match": group_exact,
        "tensors": tensor_rows,
    }


def _numeric_leaves(value: Any, prefix: str = "") -> dict[str, float]:
    result: dict[str, float] = {}
    if isinstance(value, bool):
        return result
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        result[prefix or "value"] = float(value)
    elif isinstance(value, dict):
        for key, child in value.items():
            result.update(_numeric_leaves(child, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            result.update(_numeric_leaves(child, f"{prefix}[{index}]"))
    return result


def _compare_losses(left: Any, right: Any) -> dict[str, Any]:
    left_values = _numeric_leaves(left)
    right_values = _numeric_leaves(right)
    shared = sorted(set(left_values) & set(right_values))
    return {
        "shared_numeric_fields": len(shared),
        "missing_from_candidate": sorted(set(left_values) - set(right_values)),
        "extra_in_candidate": sorted(set(right_values) - set(left_values)),
        "differences": [
            {
                "field": field,
                "control": left_values[field],
                "candidate": right_values[field],
                "absolute": abs(left_values[field] - right_values[field]),
                "relative": _relative(left_values[field], right_values[field], 1e-8),
            }
            for field in shared
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    control = _load(args.control)
    candidate = _load(args.candidate)
    expected_control = {
        "mesh_policy": "world16-fsdp",
        "rollout_batch_size": 1,
        "pack_real_score_pair": False,
        "rollout_backend": "native-fsdp2-batch",
    }
    if control.get("policy") != expected_control:
        raise ValueError(f"control is not the required B1/off policy: {control.get('policy')}")
    if candidate.get("variant") != control.get("variant"):
        raise ValueError("control/candidate variant mismatch")

    control_audits = control.get("rank_training_state_audits")
    candidate_audits = candidate.get("rank_training_state_audits")
    if not isinstance(control_audits, list) or not isinstance(candidate_audits, list):
        raise ValueError("both summaries must contain rank_training_state_audits")
    if len(control_audits) != 16 or len(candidate_audits) != 16:
        raise ValueError("training-state comparison requires 16 rank audits")

    comparisons = []
    exact = True
    for rank, (left, right) in enumerate(zip(control_audits, candidate_audits, strict=True)):
        if not isinstance(left, dict) or not isinstance(right, dict):
            raise ValueError(f"rank{rank} missing training state audit")
        for component in ("generator", "critic"):
            left_component = left[component]
            right_component = right[component]
            if int(left_component["physical_step"]) != int(right_component["physical_step"]):
                raise ValueError(f"rank{rank}/{component} physical step mismatch")
            for group_name in ("parameters", "optimizer"):
                group_exact, detail = _compare_group(
                    left_component[group_name],
                    right_component[group_name],
                    label=f"rank{rank}/{component}/{group_name}",
                )
                exact = exact and group_exact
                comparisons.append(
                    {
                        "rank": rank,
                        "component": component,
                        "group": group_name,
                        **detail,
                    }
                )

            left_gradients = left_component.get("gradient_steps", [])
            right_gradients = right_component.get("gradient_steps", [])
            if len(left_gradients) != len(right_gradients):
                raise ValueError(f"rank{rank}/{component} gradient-step count mismatch")
            for index, (left_step, right_step) in enumerate(
                zip(left_gradients, right_gradients, strict=True)
            ):
                if int(left_step["physical_step_before"]) != int(right_step["physical_step_before"]):
                    raise ValueError(
                        f"rank{rank}/{component}/gradient{index} physical-step mismatch"
                    )
                gradient_exact, detail = _compare_group(
                    left_step["gradient"],
                    right_step["gradient"],
                    label=f"rank{rank}/{component}/gradient{index}",
                )
                exact = exact and gradient_exact
                comparisons.append(
                    {
                        "rank": rank,
                        "component": component,
                        "group": f"gradient_step_{index}",
                        "physical_step_before": int(left_step["physical_step_before"]),
                        **detail,
                    }
                )

    control_losses = control.get("rank_loss_audits")
    candidate_losses = candidate.get("rank_loss_audits")
    loss_comparisons = []
    if isinstance(control_losses, list) and isinstance(candidate_losses, list):
        if len(control_losses) != len(candidate_losses):
            raise ValueError("control/candidate rank_loss_audits length mismatch")
        for rank, (left, right) in enumerate(zip(control_losses, candidate_losses, strict=True)):
            loss_comparisons.append({"rank": rank, **_compare_losses(left, right)})

    result = {
        "schema_version": "distillschedule.static_fsdp2_p0_state_compare.v2",
        "status": "PASS" if exact else "DIFF",
        "variant": control["variant"],
        "control_policy": control["policy"],
        "candidate_policy": candidate["policy"],
        "exact_training_state_equivalence": exact,
        "comparisons": comparisons,
        "loss_comparisons": loss_comparisons,
        "claim_boundary": {
            "exact_sha_match_proves_local_shard_bitwise_equality": True,
            "gradient_sha_match_proves_local_pre_step_gradient_bitwise_equality": True,
            "stat_differences_are_diagnostic_not_elementwise_error_bounds": True,
            "loss_differences_are_diagnostic_and_do_not_replace_tensor_state_checks": True,
            "candidate_with_DIFF_requires_explicit_quality/numerical_tolerance_study": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "variant": result["variant"],
                "exact_training_state_equivalence": result["exact_training_state_equivalence"],
            },
            indent=2,
        )
    )
    return 0 if exact else 2


if __name__ == "__main__":
    raise SystemExit(main())
