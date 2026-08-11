#!/usr/bin/env python3
"""Compare continuous Full5 Sequential/Pipeline runs without timing fields."""
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


def _step10_sample(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.get(key)
        for key in (
            "source_id", "production_round", "trace_identity_sha256",
            "x0_sha256", "fake_sha256", "gradient_sha256",
            "gradient_accumulation_boundary", "schedule_event",
        )
    }


def _g_wave(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.get(key)
        for key in (
            "update", "logical_step", "production_round",
            "destination_global_rank", "trace_key",
            "trace_identity_sha256", "source_trace_rank",
            "x0_identity", "x0_stride", "wire",
        )
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
        "gradient_clipping_max_norm": value.get("gradient_clipping", {}).get("configured_max_norm"),
        "critic_version_before": value.get("critic_version_before"),
        "critic_version_after": value.get("critic_version_after"),
        "scheduler_before": value.get("scheduler_before"),
        "scheduler_after": value.get("scheduler_after"),
    }


def _state_sha(value: dict[str, Any], phase: str) -> dict[str, str]:
    return {
        name: str(item.get("sha256"))
        for name, item in value.get(f"{phase}_state_export", {}).get("records", {}).items()
    }


def _queue_flush(role: str, value: dict[str, Any]) -> dict[str, Any]:
    if role == "generator":
        return {
            key: value.get(key)
            for key in (
                "produced_global_batches", "committed_global_batches",
                "produced_payloads", "sent_payloads", "per_route_commit_ack",
                "producer_slot", "outstanding_transport",
            )
        }
    return value


def _ema(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "seconds"}


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
            "scope": left.get("execution_scope") == right.get("execution_scope") == "continuous_full5_step10_to_step14",
            "claim_scope": left.get("claim_scope") == right.get("claim_scope"),
            "identities": left.get("identities") == right.get("identities"),
            "step10a_contract": left.get("step10a_contract") == right.get("step10a_contract"),
            "critic_contract": left.get("critic_contract") == right.get("critic_contract"),
            "source_role": left.get("source_role") == right.get("source_role"),
            "target_role": left.get("target_role") == right.get("target_role") == role,
            "final_versions": left.get("final_versions") == right.get("final_versions") == {
                "generator": 3, "critic": 15, "ema": 15,
                "next_boundary": "Step15-A_not_executed",
            },
            "step10a_samples": [_step10_sample(v) for v in left.get("step10a", {}).get("samples", [])]
            == [_step10_sample(v) for v in right.get("step10a", {}).get("samples", [])],
        }
        lstep, rstep = left.get("step10a", {}), right.get("step10a", {})
        if rank < 8:
            exact.update(
                step10a_update_trajectory=(
                    lstep.get("generator_before"), lstep.get("generator_after"),
                    lstep.get("global_steps_before"), lstep.get("global_steps_after"),
                    lstep.get("accumulation_boundaries"),
                ) == (2, 3, 2, 3, 1) == (
                    rstep.get("generator_before"), rstep.get("generator_after"),
                    rstep.get("global_steps_before"), rstep.get("global_steps_after"),
                    rstep.get("accumulation_boundaries"),
                ),
                step10a_final_trainable=lstep.get("trainable_after") == rstep.get("trainable_after"),
            )

        ltransition, rtransition = left.get("transition", {}), right.get("transition", {})
        exact["transition_semantics"] = {
            key: ltransition.get(key)
            for key in (
                "source_role", "target_role", "model_reused",
                "target_model_source", "critic_version",
                "optimizer_step", "scheduler_step", "adapter_reactivation",
            )
        } == {
            key: rtransition.get(key)
            for key in (
                "source_role", "target_role", "model_reused",
                "target_model_source", "critic_version",
                "optimizer_step", "scheduler_step", "adapter_reactivation",
            )
        }

        lphase, rphase = left.get("critic_phase", {}), right.get("critic_phase", {})
        if role == "generator":
            lg3 = ltransition.get("generator_g3_redistribution", {})
            rg3 = rtransition.get("generator_g3_redistribution", {})
            exact.update(
                generator_g3=(
                    lg3.get("parameter_version") == rg3.get("parameter_version") == 3
                    and lg3.get("trainable_sha256_after")
                    == rg3.get("trainable_sha256_after")
                    == lphase.get("generator_trainable_sha256_before")
                    == lphase.get("generator_trainable_sha256_after")
                    == rphase.get("generator_trainable_sha256_before")
                    == rphase.get("generator_trainable_sha256_after")
                ),
                generator_gradients=lphase.get("generator_gradients") == rphase.get("generator_gradients"),
                critic_waves=[_g_wave(v) for v in lphase.get("waves", [])]
                == [_g_wave(v) for v in rphase.get("waves", [])],
                ema=[_ema(v) for v in lphase.get("ema", [])]
                == [_ema(v) for v in rphase.get("ema", [])],
                queue_flush=_queue_flush(role, lphase.get("queue", {}))
                == _queue_flush(role, rphase.get("queue", {})),
            )
        else:
            lwaves, rwaves = lphase.get("waves", []), rphase.get("waves", [])
            exact.update(
                student_resolved=left.get("student_initialization", {}).get("resolved")
                == right.get("student_initialization", {}).get("resolved"),
                initial_state_sha=_state_sha(left, "initial") == _state_sha(right, "initial"),
                critic_waves=[_c_wave(v) for v in lwaves] == [_c_wave(v) for v in rwaves],
                queue_flush=_queue_flush(role, lphase.get("queue", {}))
                == _queue_flush(role, rphase.get("queue", {})),
                version_trajectory=(
                    lphase.get("critic_version_before"), lphase.get("critic_version_after"),
                    lphase.get("optimizer_step_before"), lphase.get("optimizer_step_after"),
                    lphase.get("scheduler_step_before"), lphase.get("scheduler_step_after"),
                ) == (10, 15, 10, 15, 10, 15) == (
                    rphase.get("critic_version_before"), rphase.get("critic_version_after"),
                    rphase.get("optimizer_step_before"), rphase.get("optimizer_step_after"),
                    rphase.get("scheduler_step_before"), rphase.get("scheduler_step_after"),
                ),
            )
            numerical.append({
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
            })
        failed = sorted(key for key, passed in exact.items() if not passed)
        if failed:
            exact_errors.append({"rank": rank, "fields": failed})
        ranks.append({"rank": rank, "role": role, "exact": exact})

    result = {
        "schema_version": "distillschedule.rb16_full5_2g6c_continuous_pair.v1",
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
