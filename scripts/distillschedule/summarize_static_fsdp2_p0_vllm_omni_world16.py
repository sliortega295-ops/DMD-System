#!/usr/bin/env python3
"""Fail-closed summary for a vLLM-Omni latent Full5 rollout run."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from distillschedule.runtime.vllm_omni_rollout import (
    VLLM_OMNI_BACKEND,
    expected_generator_version,
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--variant", choices=("no-gan", "gan"), required=True)
    parser.add_argument(
        "--mode", choices=("validate", "correctness", "timing"), required=True
    )
    parser.add_argument(
        "--mesh-policy", choices=("world16-fsdp", "node-hsdp"), required=True
    )
    parser.add_argument(
        "--rollout-batch-size", type=int, choices=(1, 2, 5), required=True
    )
    parser.add_argument(
        "--pack-real-score-pair", choices=("on", "off"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = sorted((args.run_root / "execution" / "ranks").glob("rank_*.json"))
    errors: list[str] = []
    if len(paths) != 16:
        errors.append(f"rank_receipts={len(paths)}, expected=16")
    ranks = [_load(path) for path in paths]
    expected_policy = {
        "mesh_policy": args.mesh_policy,
        "rollout_batch_size": args.rollout_batch_size,
        "pack_real_score_pair": args.pack_real_score_pair == "on",
        "rollout_backend": VLLM_OMNI_BACKEND,
    }
    expected_final = (
        {"generator_logical_version": 3, "critic_logical_version": 15, "ema_step": 15}
        if args.variant == "no-gan"
        else {
            "generator_logical_version": 201,
            "critic_logical_version": 1005,
            "ema_step": 801,
        }
    )
    expected_ema = 5 if args.variant == "no-gan" else 1
    expected_version = expected_generator_version(args.variant)
    state_audits: list[dict[str, Any] | None] = []
    external_receipts: list[dict[str, Any] | None] = []

    for rank, value in enumerate(ranks):
        if value.get("rank") != rank:
            errors.append(f"rank receipt ordering drift at index {rank}")
        if value.get("status") != "PASS":
            errors.append(f"rank{rank} status={value.get('status')}")
        if value.get("variant") != args.variant:
            errors.append(f"rank{rank} variant mismatch")
        policy = value.get("p0_policy", {})
        for key, expected in expected_policy.items():
            if policy.get(key) != expected:
                errors.append(
                    f"rank{rank} policy {key}={policy.get(key)!r}, expected={expected!r}"
                )

        if args.mode == "validate":
            external_receipts.append(None)
            state_audits.append(None)
            continue

        if value.get("updates") != {
            "generator": 1,
            "critic": 5,
            "ema": expected_ema,
        }:
            errors.append(f"rank{rank} update inventory changed: {value.get('updates')}")
        final = value.get("final", {})
        for key, expected in expected_final.items():
            if final.get(key) != expected:
                errors.append(f"rank{rank} {key}={final.get(key)}, expected={expected}")
        if not value.get("external_rollout_engine_used"):
            errors.append(f"rank{rank} did not use the external rollout engine")
        estimate = value.get("rollout_collective_estimate", {})
        if estimate.get("status") != "NOT_APPLICABLE_TO_EXTERNAL_VLLM_OMNI_ENGINE":
            errors.append(f"rank{rank} retained a native FSDP rollout estimate")

        receipt = value.get("vllm_omni_rollout")
        external_receipts.append(receipt if isinstance(receipt, dict) else None)
        if not isinstance(receipt, dict):
            errors.append(f"rank{rank} lacks vLLM-Omni rollout receipt")
        else:
            if (
                receipt.get("status") != "PASS"
                or receipt.get("backend") != VLLM_OMNI_BACKEND
            ):
                errors.append(f"rank{rank} external receipt failed: {receipt}")
            if int(receipt.get("generator_version", -1)) != expected_version:
                errors.append(f"rank{rank} external Generator version changed")
            capabilities = receipt.get("capabilities", {})
            if capabilities.get("status") != "PASS":
                errors.append(f"rank{rank} capability gate failed")
            sync = receipt.get("sync", {})
            if sync.get("status") != "PASS" or not sync.get("hot_reload_complete"):
                errors.append(f"rank{rank} Generator hot-sync failed: {sync}")
            source_checksum = str(sync.get("source_trainable_sha256", ""))
            loaded_checksum = str(sync.get("loaded_trainable_sha256", ""))
            if (
                not sync.get("checksum_match")
                or not source_checksum
                or source_checksum != loaded_checksum
            ):
                errors.append(f"rank{rank} Generator LoRA checksum mismatch: {sync}")
            batches = receipt.get("batches", [])
            expected_batches = (5 + args.rollout_batch_size - 1) // args.rollout_batch_size
            if len(batches) != expected_batches:
                errors.append(
                    f"rank{rank} external batch count={len(batches)}, expected={expected_batches}"
                )
            for batch in batches:
                if batch.get("output_type") != "latent" or int(
                    batch.get("vae_decode_calls", -1)
                ) != 0:
                    errors.append(f"rank{rank} batch used VAE/non-latent output")
                if not batch.get("exact_operation_trace_replay"):
                    errors.append(f"rank{rank} batch lacks exact trace replay")
                if not batch.get("trace_schedule_match"):
                    errors.append(
                        f"rank{rank} batch call/stage receipt did not match trace"
                    )
                parity = batch.get("batch_parity", {})
                if args.mode == "correctness":
                    if parity.get("status") != "PASS" or not parity.get("enabled"):
                        errors.append(f"rank{rank} external/native parity={parity}")
                    for sample in parity.get("samples", []):
                        if sample.get("status") != "PASS":
                            errors.append(f"rank{rank} rollout sample parity={sample}")
                        for field in (
                            "finite",
                            "max_abs",
                            "normalized_l2",
                            "cosine",
                        ):
                            if field not in sample:
                                errors.append(
                                    f"rank{rank} rollout sample lacks {field}"
                                )

        audit = final.get("training_state_audit") if args.mode == "correctness" else None
        state_audits.append(audit)
        if args.mode == "correctness":
            if not isinstance(audit, dict):
                errors.append(f"rank{rank} missing final training_state_audit")
            else:
                for component, expected_gradients in (("generator", 1), ("critic", 5)):
                    component_audit = audit.get(component, {})
                    gradients = component_audit.get("gradient_steps", [])
                    if len(gradients) != expected_gradients:
                        errors.append(
                            f"rank{rank} {component} gradient steps={len(gradients)}, "
                            f"expected={expected_gradients}"
                        )
                    for group in ("parameters", "optimizer"):
                        record = component_audit.get(group, {})
                        if not record.get("finite") or not record.get("sha256"):
                            errors.append(
                                f"rank{rank} {component}/{group} audit is incomplete"
                            )

    starts = [
        int(value["timing_start_unix_ns"])
        for value in ranks
        if "timing_start_unix_ns" in value
    ]
    stops = [
        int(value["timing_stop_unix_ns"])
        for value in ranks
        if "timing_stop_unix_ns" in value
    ]
    global_seconds = (
        (max(stops) - min(starts)) / 1e9
        if len(starts) == len(stops) == 16
        else None
    )
    rank_seconds = [
        float(value["full5_seconds"])
        for value in ranks
        if "full5_seconds" in value
    ]
    summary = {
        "schema_version": "distillschedule.static_fsdp2_p0_vllm_omni_summary.v1",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "variant": args.variant,
        "mode": args.mode,
        "rank_count": len(ranks),
        "policy": expected_policy,
        "global_full5_seconds": global_seconds,
        "rank_full5_median_seconds": (
            statistics.median(rank_seconds) if rank_seconds else None
        ),
        "rank_full5_min_seconds": min(rank_seconds) if rank_seconds else None,
        "rank_full5_max_seconds": max(rank_seconds) if rank_seconds else None,
        "claimable_correctness": not errors and args.mode == "correctness",
        "claimable_timing": not errors and args.mode == "timing",
        "rank_training_state_audits": (
            state_audits if args.mode == "correctness" else None
        ),
        "rank_external_rollout_receipts": external_receipts,
        "profile_boundary": {
            "training_node_profile": "client wait and latent return only",
            "rollout_engine_profile": "separate vLLM-Omni service-side capture required",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
