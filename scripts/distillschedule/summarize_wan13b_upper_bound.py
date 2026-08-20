#!/usr/bin/env python3
"""Fail-closed 16-rank summarizer for one Wan1.3B placement run."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


WORLD_SIZE = 16


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--run-root", type=Path, required=True)
    value.add_argument("--placement", choices=("resident", "world16-fsdp"), required=True)
    value.add_argument("--mode", choices=("validate", "correctness", "timing"), required=True)
    value.add_argument("--output", type=Path, required=True)
    return value


def main() -> int:
    args = parser().parse_args()
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for rank in range(WORLD_SIZE):
        path = args.run_root / "execution/ranks" / f"rank_{rank:02d}.json"
        if not path.is_file():
            errors.append(f"rank{rank}: missing receipt")
            continue
        try:
            row = json.loads(path.read_text())
        except Exception as error:
            errors.append(f"rank{rank}: invalid receipt: {error}")
            continue
        rows.append(row)
        if row.get("status") != "PASS":
            errors.append(f"rank{rank}: status={row.get('status')}")
        if int(row.get("rank", -1)) != rank:
            errors.append(f"rank{rank}: rank identity mismatch")
        if row.get("placement") != args.placement:
            errors.append(f"rank{rank}: placement mismatch")
    if len(rows) != WORLD_SIZE:
        errors.append(f"rank receipt count={len(rows)}, expected=16")

    identities = {
        (
            row.get("identities", {}).get("bootstrap_hash"),
            row.get("identities", {}).get("model_identity_sha256"),
            row.get("identities", {}).get("trace_bundle_hash"),
            row.get("identities", {}).get("trace_binding_hash"),
        )
        for row in rows
    }
    if len(identities) != 1:
        errors.append("source/bootstrap/trace identities differ by rank")

    if args.mode == "validate":
        if any(not row.get("validate_only") for row in rows):
            errors.append("validate run contains a non-validate receipt")
    else:
        for row in rows:
            rank = int(row["rank"])
            if row.get("updates") != {"generator": 1, "critic": 5, "ema": 5}:
                errors.append(f"rank{rank}: update inventory changed")
            final = row.get("final", {})
            expected = {
                "generator_physical_step": 3,
                "critic_physical_step": 15,
                "generator_logical_version": 3,
                "critic_logical_version": 15,
                "ema_step": 15,
            }
            for key, value in expected.items():
                if final.get(key) != value:
                    errors.append(f"rank{rank}: {key}={final.get(key)}, expected={value}")
            rollouts = row.get("rollout_batches", [])
            if len(rollouts) != 5:
                errors.append(f"rank{rank}: rollout batch count changed")
            for index, rollout in enumerate(rollouts):
                if rollout.get("batch_size") != 1:
                    errors.append(f"rank{rank}: rollout{index} is not physical B1")
                if rollout.get("call_count") != rollout.get("logical_call_count", rollout.get("call_count")):
                    errors.append(f"rank{rank}: rollout{index} physical/logical calls differ")
            if row.get("generator", {}).get("physical_score_forwards") != 3:
                errors.append(f"rank{rank}: Score packing or Score-call drift")
            if args.placement == "resident":
                sync = row.get("gradient_sync", {})
                if args.mode == "correctness" and (
                    len(sync.get("generator", [])) != 1
                    or len(sync.get("critic", [])) != 5
                ):
                    errors.append(f"rank{rank}: resident LoRA all-reduce receipt count changed")
                for receipt in sync.get("generator", []) + sync.get("critic", []):
                    if receipt.get("world_size") != WORLD_SIZE:
                        errors.append(f"rank{rank}: resident gradient sync is not world16")
                    if receipt.get("reduction") != "SUM_THEN_DIVIDE_WORLD_SIZE":
                        errors.append(f"rank{rank}: resident gradient reduction changed")
            build = row.get("build", {})
            for component in ("generator", "critic"):
                placement = build.get(component, {}).get("placement", {})
                if placement.get("block_count") != 30:
                    errors.append(f"rank{rank}: {component} is not 30-block Wan1.3B")
                if placement.get("cpu_parameter_offload") is not False:
                    errors.append(f"rank{rank}: {component} parameter offload enabled")
            residency = build.get("static_residency", {})
            forbidden = (
                "parameter_offload",
                "activation_offload",
                "optimizer_offload",
                "frozen_parameter_streaming",
                "role_transition",
                "model_swap",
            )
            if any(residency.get(key) is not False for key in forbidden):
                errors.append(f"rank{rank}: common residency contract changed")

    rollout_identities: list[str] = []
    if args.mode != "validate":
        for row in rows:
            for rollout in row.get("rollout_batches", []):
                rollout_identities.extend(rollout.get("member_trace_identities", []))
        if len(rollout_identities) != 80 or len(set(rollout_identities)) != 80:
            errors.append(
                "five world16 rollouts do not contain 80 unique rank-step identities"
            )

    full5_values = [float(row["full5_seconds"]) for row in rows if "full5_seconds" in row]
    memory_rows = [
        row.get("final", {}).get("memory", {})
        for row in rows
        if row.get("final", {}).get("memory")
    ]
    summary: dict[str, Any] = {
        "schema_version": "distillschedule.wan13b_upper_bound_summary.v1",
        "status": "PASS" if not errors else "FAIL",
        "run_id": rows[0].get("run_id") if rows else args.run_root.name,
        "placement": args.placement,
        "mode": args.mode,
        "rank_count": len(rows),
        "errors": errors,
        "identities": (
            {
                "bootstrap_hash": next(iter(identities))[0],
                "model_identity_sha256": next(iter(identities))[1],
                "trace_bundle_hash": next(iter(identities))[2],
                "trace_binding_hash": next(iter(identities))[3],
            }
            if len(identities) == 1
            else None
        ),
        "workload": {
            "world_size": 16,
            "global_batch": 16,
            "rollout_batch_size": 1,
            "score_packing": False,
            "updates": {"generator": 1, "critic": 5, "ema": 5},
            "final_versions": {"generator": 3, "critic": 15, "ema": 15},
            "unique_rank_step_rollout_identities": len(set(rollout_identities)),
        },
        "full5_seconds_world_max": max(full5_values) if full5_values else None,
        "rank_full5_seconds": full5_values,
        "memory": {
            "max_allocated_bytes": max(
                (int(value.get("max_allocated_bytes", 0)) for value in memory_rows),
                default=None,
            ),
            "max_reserved_bytes": max(
                (int(value.get("max_reserved_bytes", 0)) for value in memory_rows),
                default=None,
            ),
            "min_driver_free_bytes": min(
                (int(value.get("driver_free_bytes", 0)) for value in memory_rows),
                default=None,
            ),
        },
        "rank_receipts": [
            str(args.run_root / "execution/ranks" / f"rank_{rank:02d}.json")
            for rank in range(WORLD_SIZE)
        ],
        "correctness_evidence": (
            {
                "rollouts": [row.get("rollout_evidence") for row in rows],
                "final_trainables": rows[0].get("final", {}).get("trainable_evidence"),
                "generator_metrics": rows[0].get("final", {}).get("generator_trainable_metrics"),
                "critic_metrics": rows[0].get("final", {}).get("critic_trainable_metrics"),
                "ema_checksum_rank0": rows[0].get("final", {}).get("ema_checksum"),
            }
            if args.mode == "correctness" and rows
            else "NOT_RUN"
        ),
        "formal_eligibility": args.mode == "timing" and not errors,
    }
    _atomic_json(args.output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
