#!/usr/bin/env python3
"""Evaluate continuous Full5 same-layout correctness against a frozen contract."""
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
    if contract.get("status") != "ACTIVE" or contract.get("frozen_before_first_gpu_run") is not True:
        errors.append({"field": "contract", "reason": "contract is not active and pre-run frozen"})
    if semantic.get("status") != "PASS":
        errors.append({"field": "exact_contract", "errors": semantic.get("exact_contract_errors")})
    initial = state.get("critic", {}).get("initial", {})
    if state.get("status") != "PASS" or initial.get("input_sha_exact") is not True:
        errors.append({"field": "initial_state", "reason": "DP12 initial partitions are not exact"})

    limits = contract["numerical_limits"]
    observed_loss = max(
        (float(value.get("max_loss_abs", float("inf"))) for value in semantic.get("numerical_diagnostics", [])),
        default=float("inf"),
    )
    if observed_loss > float(limits["loss_max_abs"]):
        errors.append({"field": "loss_max_abs", "observed": observed_loss, "limit": limits["loss_max_abs"]})
    final = state.get("critic", {}).get("final", {})
    for state_name in ("master", "exp_avg", "exp_avg_sq"):
        for metric in ("relative_l2", "max_abs"):
            observed = float(final.get(state_name, {}).get(metric, float("inf")))
            limit = float(limits["critic"][state_name][metric])
            if observed > limit:
                errors.append({"field": f"critic.{state_name}.{metric}", "observed": observed, "limit": limit})

    result = {
        "schema_version": "distillschedule.rb16_full5_2g6c_continuous_evaluation.v1",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "reference_run": semantic.get("reference_run"),
        "candidate_run": semantic.get("candidate_run"),
        "claim_scope": contract.get("claim_boundary"),
        "observed": {
            "loss_max_abs": observed_loss,
            "critic_final": final,
        },
        "contract_schema": contract.get("schema_version"),
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
