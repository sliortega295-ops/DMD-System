#!/usr/bin/env python3
"""Evaluate a Candidate-B pair against a frozen same-layout noise contract."""
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


def evaluate(contract: dict[str, Any], semantic: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    if contract.get("status") != "ACTIVE" or not contract.get("frozen_before_pipeline"):
        errors.append({"field": "contract", "reason": "contract is not active/frozen"})
    if semantic.get("status") != "PASS":
        errors.append({"field": "exact_contract", "errors": semantic.get("exact_contract_errors")})
    if state.get("status") != "PASS" or not state.get("critic", {}).get("initial", {}).get("input_sha_exact"):
        errors.append({"field": "initial_state", "reason": "DP12 initial state is not exact"})

    limits = contract["frozen_limits"]
    observed_loss = max(
        (float(value.get("max_loss_abs", float("inf"))) for value in semantic.get("numerical_diagnostics", [])),
        default=float("inf"),
    )
    if observed_loss > float(limits["loss_max_abs"]):
        errors.append({"field": "loss_max_abs", "observed": observed_loss, "limit": limits["loss_max_abs"]})
    final = state["critic"]["final"]
    for state_name in ("master", "exp_avg", "exp_avg_sq"):
        for metric in ("relative_l2", "max_abs"):
            observed = float(final[state_name][metric])
            limit = float(limits["critic"][state_name][metric])
            if observed > limit:
                errors.append({"field": f"critic.{state_name}.{metric}", "observed": observed, "limit": limit})

    result = {
        "schema_version": "distillschedule.routeb_candidate_b_numerical_evaluation.v1",
        "status": "PASS" if not errors else "FAIL",
        "reference_run": semantic.get("reference_run"),
        "candidate_run": semantic.get("candidate_run"),
        "transition_package_hash": contract.get("transition_package_hash"),
        "observed_loss_max_abs": observed_loss,
        "errors": errors,
    }
    return result


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
