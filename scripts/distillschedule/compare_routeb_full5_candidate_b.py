#!/usr/bin/env python3
"""Compare two same-layout Candidate-B runs without timing fields."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from runtime.routeb_full5 import RouteBCandidateBTopology, RouteBFull5Contract


CONTRACT = RouteBFull5Contract.candidate_b()
TOPOLOGY = RouteBCandidateBTopology(CONTRACT)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _load(root: Path) -> dict[int, dict[str, Any]]:
    paths = sorted((root / "execution" / "ranks").glob("rank_*.json"))
    result = {int((value := json.loads(path.read_text()))["rank"]): value for path in paths}
    if set(result) != set(range(16)):
        raise ValueError(f"{root}: rank inventory is not exactly 0..15")
    return result


def _without(value: dict[str, Any], *keys: str) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key not in keys}


def _g_wave(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "update": value.get("update"),
        "logical_step": value.get("logical_step"),
        "production_round": value.get("production_round"),
        "destination_global_rank": value.get("destination_global_rank"),
        "trace_key": value.get("trace_key"),
        "trace_identity_sha256": value.get("trace_identity_sha256"),
        "source_trace_rank": value.get("source_trace_rank"),
        "x0_identity": value.get("x0_identity"),
        "x0_stride": value.get("x0_stride"),
        "wire": value.get("wire"),
        "ack": _without(value.get("ack", {}), "run_nonce"),
    }


def _c_wave(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "update": value.get("update"),
        "logical_step": value.get("logical_step"),
        "production_round": value.get("production_round"),
        "producer_global_rank": value.get("producer_global_rank"),
        "trace_key": value.get("trace_key"),
        "trace_identity_sha256": value.get("trace_identity_sha256"),
        "source_trace_rank": value.get("source_trace_rank"),
        "x0_identity": value.get("x0_identity"),
        "critic_noise_sha256": value.get("critic_noise_sha256"),
        "critic_timestep": value.get("critic_timestep"),
        "gradient_accumulation_boundary": value.get("gradient_accumulation_boundary"),
        "gradient_clipping_config": value.get("gradient_clipping", {}).get("configured_max_norm"),
        "critic_version_before": value.get("critic_version_before"),
        "critic_version_after": value.get("critic_version_after"),
        "scheduler_before": value.get("scheduler_before"),
        "scheduler_after": value.get("scheduler_after"),
    }


def _state_sha(value: dict[str, Any], phase: str) -> dict[str, str]:
    record = value.get(f"{phase}_state_export", {})
    return {
        name: str(item.get("sha256"))
        for name, item in record.get("records", {}).items()
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference, candidate = _load(args.reference_root), _load(args.candidate_root)
    exact_errors: list[dict[str, Any]] = []
    numerical: list[dict[str, Any]] = []
    ranks: list[dict[str, Any]] = []

    for rank in range(16):
        left, right = reference[rank], candidate[rank]
        role = TOPOLOGY.role(rank)
        exact: dict[str, bool] = {
            "status": left.get("status") == right.get("status") == "PASS",
            "scope": left.get("execution_scope") == right.get("execution_scope") == "critic_phase_from_post_step10a_target_layout",
            "claim_scope": left.get("claim_scope") == right.get("claim_scope"),
            "identities": left.get("identities") == right.get("identities"),
            "contract": left.get("contract") == right.get("contract"),
            "role": left.get("role") == right.get("role") == role,
        }
        lcycle, rcycle = left.get("cycle", {}), right.get("cycle", {})
        if role == "generator":
            exact.update(
                initialization=left.get("generator_initialization") == right.get("generator_initialization"),
                generator_trainable=(
                    lcycle.get("generator_trainable_sha256_before")
                    == rcycle.get("generator_trainable_sha256_before")
                    == lcycle.get("generator_trainable_sha256_after")
                    == rcycle.get("generator_trainable_sha256_after")
                ),
                generator_gradients=lcycle.get("generator_gradients") == rcycle.get("generator_gradients"),
                waves=[_g_wave(value) for value in lcycle.get("waves", [])]
                == [_g_wave(value) for value in rcycle.get("waves", [])],
                ema=[_without(value, "seconds") for value in lcycle.get("ema", [])]
                == [_without(value, "seconds") for value in rcycle.get("ema", [])],
                queue_flush=(
                    lcycle.get("queue", {}).get("produced")
                    == lcycle.get("queue", {}).get("sent")
                    == lcycle.get("queue", {}).get("acknowledged")
                    == rcycle.get("queue", {}).get("produced")
                    == rcycle.get("queue", {}).get("sent")
                    == rcycle.get("queue", {}).get("acknowledged")
                    == 15
                    and lcycle.get("queue", {}).get("producer_slot")
                    == rcycle.get("queue", {}).get("producer_slot")
                    == "EMPTY"
                    and lcycle.get("queue", {}).get("outstanding_transport")
                    == rcycle.get("queue", {}).get("outstanding_transport")
                    == 0
                ),
            )
        else:
            lwaves, rwaves = lcycle.get("waves", []), rcycle.get("waves", [])
            exact.update(
                student_resolved=left.get("student_initialization", {}).get("resolved") == right.get("student_initialization", {}).get("resolved"),
                initial_state_sha=_state_sha(left, "initial") == _state_sha(right, "initial"),
                waves=[_c_wave(value) for value in lwaves]
                == [_c_wave(value) for value in rwaves],
                queue=lcycle.get("queue") == rcycle.get("queue"),
                version_trajectory=(
                    lcycle.get("critic_version_before"),
                    lcycle.get("critic_version_after"),
                    lcycle.get("optimizer_step_before"),
                    lcycle.get("optimizer_step_after"),
                    lcycle.get("scheduler_step_before"),
                    lcycle.get("scheduler_step_after"),
                ) == (10, 15, 10, 15, 10, 15) == (
                    rcycle.get("critic_version_before"),
                    rcycle.get("critic_version_after"),
                    rcycle.get("optimizer_step_before"),
                    rcycle.get("optimizer_step_after"),
                    rcycle.get("scheduler_step_before"),
                    rcycle.get("scheduler_step_after"),
                ),
            )
            numerical.append({
                "rank": rank,
                "prediction_exact": [value.get("prediction_sha256") for value in lwaves]
                == [value.get("prediction_sha256") for value in rwaves],
                "loss_exact": [value.get("loss") for value in lwaves]
                == [value.get("loss") for value in rwaves],
                "max_loss_abs": max(
                    (abs(float(a.get("loss")) - float(b.get("loss"))) for a, b in zip(lwaves, rwaves, strict=True)),
                    default=0.0,
                ),
                "final_trainable_summary_exact": lcycle.get("final_trainable") == rcycle.get("final_trainable"),
                "final_optimizer_summary_exact": lcycle.get("final_optimizer") == rcycle.get("final_optimizer"),
            })
        failed = sorted(key for key, passed in exact.items() if not passed)
        if failed:
            exact_errors.append({"rank": rank, "fields": failed})
        ranks.append({"rank": rank, "role": role, "exact": exact})

    result = {
        "schema_version": "distillschedule.routeb_candidate_b_same_layout_pair.v1",
        "status": "PASS" if not exact_errors else "FAIL",
        "reference_run": args.reference_root.name,
        "candidate_run": args.candidate_root.name,
        "exact_contract_errors": exact_errors,
        "numerical_diagnostics": numerical,
        "ranks": ranks,
    }
    _atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not exact_errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
