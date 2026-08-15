#!/usr/bin/env python3
"""Fail-closed summary for one static FSDP2 P0 world16 run."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _loss_audit(value: dict[str, Any]) -> dict[str, Any]:
    generator = value.get("generator", {})
    return {
        "generator": generator.get("surrogate_loss", generator.get("losses")),
        "critic": [
            {"logical_step": record.get("logical_step"), "loss": record.get("loss")}
            for record in value.get("critic", [])
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--variant", choices=("no-gan", "gan"), required=True)
    parser.add_argument("--mode", choices=("validate", "correctness", "timing", "profile"), required=True)
    parser.add_argument("--mesh-policy", choices=("world16-fsdp", "node-hsdp"), required=True)
    parser.add_argument("--rollout-batch-size", type=int, choices=(1, 2, 5), required=True)
    parser.add_argument("--pack-real-score-pair", choices=("on", "off"), required=True)
    parser.add_argument(
        "--diagnostic-timing",
        action="store_true",
        help="Mark timing as diagnostic-only when numerical parity was explicitly waived.",
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
        "rollout_backend": "native-fsdp2-batch",
    }
    expected_final = (
        {"generator_logical_version": 3, "critic_logical_version": 15, "ema_step": 15}
        if args.variant == "no-gan"
        else {"generator_logical_version": 201, "critic_logical_version": 1005, "ema_step": 801}
    )
    expected_ema = 5 if args.variant == "no-gan" else 1
    expected_gradient_steps = {"generator": 1, "critic": 5}
    state_audits: list[dict[str, Any] | None] = []
    loss_audits: list[dict[str, Any] | None] = []
    input_reuse: list[dict[str, Any] | None] = []

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
                errors.append(f"rank{rank} policy {key}={policy.get(key)!r}, expected={expected!r}")

        hardening = value.get("p0_hardening", {})
        input_reuse.append(hardening.get("input_reuse"))
        if args.mode != "validate" and hardening.get("tensor_parity_policy") not in {
            "multi_metric",
            "multi_metric_pre_update",
        }:
            errors.append(f"rank{rank} missing hardened multi-metric parity receipt")

        if args.mode != "validate":
            updates = value.get("updates", {})
            if updates != {"generator": 1, "critic": 5, "ema": expected_ema}:
                errors.append(f"rank{rank} update inventory changed: {updates}")
            final = value.get("final", {})
            for key, expected in expected_final.items():
                if final.get(key) != expected:
                    errors.append(f"rank{rank} {key}={final.get(key)}, expected={expected}")
            if value.get("transition_count") != 0 or value.get("model_swap_count") != 0:
                errors.append(f"rank{rank} introduced a transition/model swap")
            if value.get("parameter_streaming_count") != 0:
                errors.append(f"rank{rank} introduced parameter streaming")
            generator = value.get("generator", {})
            if expected_policy["pack_real_score_pair"]:
                expected_physical = 2 if args.variant == "no-gan" else 3
                if generator.get("physical_score_forwards") != expected_physical:
                    errors.append(
                        f"rank{rank} physical Score forwards="
                        f"{generator.get('physical_score_forwards')}, expected={expected_physical}"
                    )
                if not generator.get("real_score_pair_packed"):
                    errors.append(f"rank{rank} did not use the packed Real-Score pair")
                if args.mode == "correctness":
                    parity = generator.get("score_pack_parity", {})
                    if parity.get("status") != "PASS":
                        errors.append(f"rank{rank} Score packing parity={parity}")
            batches = value.get("rollout_batches", [])
            expected_batches = (5 + args.rollout_batch_size - 1) // args.rollout_batch_size
            if len(batches) != expected_batches:
                errors.append(
                    f"rank{rank} rollout batch count={len(batches)}, expected={expected_batches}"
                )
            if args.mode == "correctness" and args.rollout_batch_size > 1:
                for batch in batches:
                    parity = batch.get("batch_parity", {})
                    if parity.get("status") != "PASS":
                        errors.append(f"rank{rank} rollout batch parity={parity}")
                    for sample in parity.get("samples", []):
                        if sample.get("status") != "PASS":
                            errors.append(f"rank{rank} rollout sample parity={sample}")
                        for field in ("max_abs", "normalized_l2", "cosine", "finite"):
                            if field not in sample:
                                errors.append(f"rank{rank} rollout parity lacks {field}")

            audit = final.get("training_state_audit") if args.mode == "correctness" else None
            state_audits.append(audit)
            loss_audits.append(_loss_audit(value) if args.mode == "correctness" else None)
            if args.mode == "correctness":
                if not isinstance(audit, dict):
                    errors.append(f"rank{rank} missing final training_state_audit")
                else:
                    for component in ("generator", "critic"):
                        component_audit = audit.get(component, {})
                        for group in ("parameters", "optimizer"):
                            record = component_audit.get(group, {})
                            if not record.get("finite"):
                                errors.append(f"rank{rank} {component}/{group} audit is non-finite")
                            if not record.get("sha256"):
                                errors.append(f"rank{rank} {component}/{group} lacks checksum")
                        gradient_steps = component_audit.get("gradient_steps", [])
                        if len(gradient_steps) != expected_gradient_steps[component]:
                            errors.append(
                                f"rank{rank} {component} gradient steps={len(gradient_steps)}, "
                                f"expected={expected_gradient_steps[component]}"
                            )
                        for index, gradient_step in enumerate(gradient_steps):
                            gradient = gradient_step.get("gradient", {})
                            if not gradient.get("finite"):
                                errors.append(
                                    f"rank{rank} {component} gradient step{index} is non-finite"
                                )
                            if not gradient.get("sha256"):
                                errors.append(
                                    f"rank{rank} {component} gradient step{index} lacks checksum"
                                )
        else:
            state_audits.append(None)
            loss_audits.append(None)

    starts = [int(value["timing_start_unix_ns"]) for value in ranks if "timing_start_unix_ns" in value]
    stops = [int(value["timing_stop_unix_ns"]) for value in ranks if "timing_stop_unix_ns" in value]
    global_seconds = (max(stops) - min(starts)) / 1e9 if len(starts) == len(stops) == 16 else None
    rank_seconds = [float(value["full5_seconds"]) for value in ranks if "full5_seconds" in value]
    summary = {
        "schema_version": "distillschedule.static_fsdp2_p0_summary.v3",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "variant": args.variant,
        "mode": args.mode,
        "rank_count": len(ranks),
        "policy": expected_policy,
        "global_full5_seconds": global_seconds,
        "rank_full5_median_seconds": statistics.median(rank_seconds) if rank_seconds else None,
        "rank_full5_min_seconds": min(rank_seconds) if rank_seconds else None,
        "rank_full5_max_seconds": max(rank_seconds) if rank_seconds else None,
        "claimable_correctness": not errors and args.mode == "correctness",
        "claimable_timing": (
            not errors and args.mode == "timing" and not args.diagnostic_timing
        ),
        "diagnostic_timing": bool(args.diagnostic_timing),
        "profiled_latency_diagnostic_only": args.mode == "profile",
        "rollout_collective_estimate": ranks[0].get("rollout_collective_estimate") if ranks else None,
        "gan_score_fusion": ranks[0].get("gan_score_fusion") if ranks else None,
        "generator_score_inventory": (
            {
                "logical_score_requests": ranks[0].get("generator", {}).get("logical_score_requests"),
                "physical_score_forwards": ranks[0].get("generator", {}).get("physical_score_forwards"),
                "real_score_pair_packed": ranks[0].get("generator", {}).get("real_score_pair_packed"),
            }
            if ranks and args.mode != "validate"
            else None
        ),
        "rank_training_state_audits": state_audits if args.mode == "correctness" else None,
        "rank_loss_audits": loss_audits if args.mode == "correctness" else None,
        "rank_input_reuse": input_reuse,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
