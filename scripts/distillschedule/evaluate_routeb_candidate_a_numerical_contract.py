#!/usr/bin/env python3
"""Fail-closed evaluator for the frozen Candidate-A same-layout contract."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def evaluate(
    contract: dict[str, Any],
    semantic_pair: dict[str, Any],
    state_pair: dict[str, Any],
) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    if contract.get("status") != "ACTIVE" or not contract.get("frozen_before_pipeline"):
        errors.append({"field": "contract", "reason": "contract is not active and frozen"})
    if semantic_pair.get("status") != "PASS":
        errors.append({"field": "exact_contract", "errors": semantic_pair.get("exact_contract_errors")})

    limits = contract["frozen_limits"]
    max_loss = max(
        (float(value.get("max_loss_abs", float("inf"))) for value in semantic_pair.get("numerical_diagnostics", [])),
        default=float("inf"),
    )
    if max_loss > float(limits["loss_max_abs"]):
        errors.append({"field": "loss_max_abs", "observed": max_loss, "limit": limits["loss_max_abs"]})

    for state_name in ("master", "exp_avg", "exp_avg_sq"):
        observed = state_pair["critic"][state_name]
        limit = limits["critic"][state_name]
        for metric in ("relative_l2", "max_abs"):
            if float(observed[metric]) > float(limit[metric]):
                errors.append(
                    {
                        "field": f"critic.{state_name}.{metric}",
                        "observed": observed[metric],
                        "limit": limit[metric],
                    }
                )
    if limits.get("generator_all_state_exact"):
        for state_name in ("master", "exp_avg", "exp_avg_sq"):
            observed = state_pair["generator"][state_name]
            if float(observed["relative_l2"]) != 0.0 or float(observed["exact_fraction"]) != 1.0:
                errors.append({"field": f"generator.{state_name}", "observed": observed})

    return {
        "schema_version": "distillschedule.routeb_candidate_a_numerical_evaluation.v1",
        "status": "PASS" if not errors else "FAIL",
        "contract_transition_manifest_sha256": contract["transition_manifest_sha256"],
        "reference_run": semantic_pair.get("reference_run"),
        "candidate_run": semantic_pair.get("candidate_run"),
        "observed_loss_max_abs": max_loss,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--semantic-pair", type=Path, required=True)
    parser.add_argument("--state-pair", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(
        json.loads(args.contract.read_text()),
        json.loads(args.semantic_pair.read_text()),
        json.loads(args.state_pair.read_text()),
    )
    _atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
