#!/usr/bin/env python3
"""Compare two Candidate-A critic-phase runs without timing contamination."""
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


def _load_ranks(root: Path) -> dict[int, dict[str, Any]]:
    paths = sorted((root / "execution" / "ranks").glob("rank_*.json"))
    values = {int((value := json.loads(path.read_text()))["rank"]): value for path in paths}
    if set(values) != set(range(16)):
        raise ValueError(f"{root}: rank inventory is not exactly 0..15")
    return values


def _ema_projection(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "seconds"}


def _ack_projection(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "run_nonce"}


def _g_wave(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "micro_wave": value.get("micro_wave"),
        "logical_step": value.get("logical_step"),
        "production_round": value.get("production_round"),
        "trace_key": value.get("trace_key"),
        "trace_identity_sha256": value.get("trace_identity_sha256"),
        "source_trace_rank": value.get("source_trace_rank"),
        "wire": value.get("wire"),
        "ack": _ack_projection(value.get("ack", {})),
    }


def _c_wave_exact(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "micro_wave": value.get("micro_wave"),
        "logical_step": value.get("logical_step"),
        "production_round": value.get("production_round"),
        "trace_key": value.get("trace_key"),
        "trace_identity_sha256": value.get("trace_identity_sha256"),
        "source_trace_rank": value.get("source_trace_rank"),
        "x0_identity": value.get("x0_identity"),
        "critic_noise_sha256": value.get("critic_noise_sha256"),
        "critic_timestep": value.get("critic_timestep"),
        "gradient_accumulation_boundary": value.get("gradient_accumulation_boundary"),
        "critic_version_before": value.get("critic_version_before"),
        "critic_version_after": value.get("critic_version_after"),
        "scheduler_before": value.get("scheduler_before"),
        "scheduler_after": value.get("scheduler_after"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = _load_ranks(args.reference_root)
    candidate = _load_ranks(args.candidate_root)
    exact_errors: list[dict[str, Any]] = []
    numerical: list[dict[str, Any]] = []
    rank_results: list[dict[str, Any]] = []

    for rank in range(16):
        left, right = reference[rank], candidate[rank]
        exact: dict[str, bool] = {
            "status": left.get("status") == right.get("status") == "PASS",
            "execution_scope": left.get("execution_scope") == right.get("execution_scope") == "critic_phase_from_common_transition_checkpoint",
            "identities": left.get("identities") == right.get("identities"),
            "contract": left.get("contract") == right.get("contract"),
            "transition_resume": left.get("transition_resume") == right.get("transition_resume"),
        }
        lphase, rphase = left.get("critic_phase", {}), right.get("critic_phase", {})
        if rank < 8:
            exact.update(
                generator_trainable=(
                    lphase.get("generator_trainable_sha256_before")
                    == rphase.get("generator_trainable_sha256_before")
                    == lphase.get("generator_trainable_sha256_after")
                    == rphase.get("generator_trainable_sha256_after")
                ),
                generator_gradients=lphase.get("generator_gradients") == rphase.get("generator_gradients"),
                queue=lphase.get("queue") == rphase.get("queue"),
                waves=[_g_wave(v) for v in lphase.get("micro_waves", [])]
                == [_g_wave(v) for v in rphase.get("micro_waves", [])],
                ema=[_ema_projection(v) for v in lphase.get("ema", [])]
                == [_ema_projection(v) for v in rphase.get("ema", [])],
            )
        else:
            lwaves, rwaves = lphase.get("micro_waves", []), rphase.get("micro_waves", [])
            exact.update(
                queue=lphase.get("queue") == rphase.get("queue"),
                waves=[_c_wave_exact(v) for v in lwaves]
                == [_c_wave_exact(v) for v in rwaves],
                initial_trainable=lphase.get("initial_trainable") == rphase.get("initial_trainable"),
                initial_optimizer=lphase.get("initial_optimizer") == rphase.get("initial_optimizer"),
            )
            numerical.append(
                {
                    "rank": rank,
                    "prediction_exact": [v.get("prediction_sha256") for v in lwaves]
                    == [v.get("prediction_sha256") for v in rwaves],
                    "loss_exact": [v.get("loss") for v in lwaves]
                    == [v.get("loss") for v in rwaves],
                    "max_loss_abs": max(
                        (abs(float(a.get("loss")) - float(b.get("loss"))) for a, b in zip(lwaves, rwaves, strict=True)),
                        default=0.0,
                    ),
                    "final_trainable_summary_exact": lphase.get("final_trainable") == rphase.get("final_trainable"),
                    "final_optimizer_summary_exact": lphase.get("final_optimizer") == rphase.get("final_optimizer"),
                }
            )
        failed = sorted(key for key, passed in exact.items() if not passed)
        if failed:
            exact_errors.append({"rank": rank, "fields": failed})
        rank_results.append({"rank": rank, "role": left.get("role"), "exact": exact})

    result = {
        "schema_version": "distillschedule.routeb_candidate_a_common_transition_pair.v1",
        "status": "PASS" if not exact_errors else "FAIL",
        "reference_run": args.reference_root.name,
        "candidate_run": args.candidate_root.name,
        "exact_contract_errors": exact_errors,
        "numerical_diagnostics": numerical,
        "ranks": rank_results,
    }
    _atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not exact_errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
