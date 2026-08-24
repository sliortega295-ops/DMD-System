#!/usr/bin/env python3
"""Fail-closed 16-rank summarizer for the fair Wan1.3B placement ladder."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from distillschedule.runtime.wan13b_fair import (
    ASYNC_128_RESIDENT,
    ASYNC_128_TREE_RESIDENT,
    ASYNC_128_TREE_CHUNK2048_RESIDENT,
    ASYNC_128_TREE_CHUNK2048_COMPILE_RESIDENT,
    ASYNC_256_RESIDENT,
    CLEAN_RESIDENT,
    FAIR_PLACEMENTS,
    FSDP_PLACEMENT,
    FLAT_256_RESIDENT,
    FLAT_512_RESIDENT,
    LORA_ZERO2_RESIDENT,
    ROLLOUT_FUSED_RESIDENT,
    LOCAL_FAST_RESIDENT,
    RESIDENT_PLACEMENTS,
)
from distillschedule.runtime.wan13b_upper_bound import WORLD_SIZE


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--run-root", type=Path, required=True)
    value.add_argument("--placement", choices=FAIR_PLACEMENTS, required=True)
    value.add_argument(
        "--mode",
        choices=("validate", "correctness", "timing", "diagnostic-timing", "profile"),
        required=True,
    )
    value.add_argument("--output", type=Path, required=True)
    return value


def _validate_dtype_contract(
    row: dict[str, Any], *, rank: int, correctness_mode: bool, errors: list[str]
) -> None:
    build = row.get("build", {})
    for component in ("generator", "critic"):
        component_build = build.get(component, {})
        placement = component_build.get("placement", {})
        if placement.get("param_compute_dtype_policy") != "PRESERVE_ORIGINAL_PER_PARAMETER":
            errors.append(f"rank{rank}/{component}: compute dtype policy is not matched")
        if placement.get("trainable_parameter_dtypes") != ["torch.float32"]:
            errors.append(f"rank{rank}/{component}: LoRA storage dtype is not FP32")
        if "torch.bfloat16" not in placement.get("frozen_parameter_dtypes", []):
            errors.append(f"rank{rank}/{component}: frozen base lacks BF16")
        if placement.get("gradient_reduction") != "SUM_THEN_DIVIDE_WORLD_SIZE":
            errors.append(f"rank{rank}/{component}: gradient mean convention changed")
        contract = component_build.get("lora_compute_dtype_contract", {})
        if contract.get("trainable_lora") != "torch.float32":
            errors.append(f"rank{rank}/{component}: LoRA compute contract changed")
        if not correctness_mode:
            continue
        records = contract.get("records")
        if not isinstance(records, list) or not records:
            errors.append(f"rank{rank}/{component}: missing runtime LoRA dtype audit")
            continue
        for record in records:
            if record.get("lora_a_weight_dtype") != "torch.float32":
                errors.append(
                    f"rank{rank}/{component}/{record.get('module')}: lora_A not FP32"
                )
            if record.get("lora_b_weight_dtype") != "torch.float32":
                errors.append(
                    f"rank{rank}/{component}/{record.get('module')}: lora_B not FP32"
                )
            if record.get("base_weight_dtype") != "torch.bfloat16":
                errors.append(
                    f"rank{rank}/{component}/{record.get('module')}: base weight not BF16"
                )


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
        errors.append(f"rank receipt count={len(rows)}, expected={WORLD_SIZE}")

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

    correctness_mode = args.mode == "correctness"
    residual_runtime_hashes: set[str] = set()
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
                if rollout.get("call_count") != rollout.get(
                    "logical_call_count", rollout.get("call_count")
                ):
                    errors.append(f"rank{rank}: rollout{index} physical/logical calls differ")
            if row.get("generator", {}).get("physical_score_forwards") != 3:
                errors.append(f"rank{rank}: Score packing or Score-call drift")

            sync = row.get("gradient_sync", {})
            if args.placement in RESIDENT_PLACEMENTS and correctness_mode:
                if (
                    len(sync.get("generator", [])) != 1
                    or len(sync.get("critic", [])) != 5
                ):
                    errors.append(f"rank{rank}: resident sync receipt count changed")
                expected_backend = (
                    "LORA_ZERO2_REDUCE_SCATTER_ALL_GATHER"
                    if args.placement in {LORA_ZERO2_RESIDENT, ROLLOUT_FUSED_RESIDENT}
                    else (
                        "GRAD_READY_ASYNC_FLAT_ALL_REDUCE"
                        if args.placement
                        in {
                            ASYNC_128_RESIDENT,
                            ASYNC_128_TREE_RESIDENT,
                            ASYNC_128_TREE_CHUNK2048_RESIDENT,
                            ASYNC_128_TREE_CHUNK2048_COMPILE_RESIDENT,
                            ASYNC_256_RESIDENT,
                        }
                        else (
                            "PERSISTENT_FLAT_ALL_REDUCE"
                            if args.placement in {FLAT_256_RESIDENT, FLAT_512_RESIDENT}
                            else (
                                "MANUAL_ALL_REDUCE_COALESCED"
                                if args.placement
                                in {"resident", CLEAN_RESIDENT, LOCAL_FAST_RESIDENT}
                                else "DDP_GRADIENT_BUCKET_VIEW"
                            )
                        )
                    )
                )
                for receipt in sync.get("generator", []) + sync.get("critic", []):
                    if receipt.get("world_size") != WORLD_SIZE:
                        errors.append(f"rank{rank}: resident sync is not world16")
                    if receipt.get("reduction") != "SUM_THEN_DIVIDE_WORLD_SIZE":
                        errors.append(f"rank{rank}: resident reduction changed")
                    if receipt.get("backend") != expected_backend:
                        errors.append(f"rank{rank}: resident sync backend changed")
                    if args.placement in {
                        ASYNC_128_RESIDENT,
                        ASYNC_128_TREE_RESIDENT,
                        ASYNC_128_TREE_CHUNK2048_RESIDENT,
                        ASYNC_128_TREE_CHUNK2048_COMPILE_RESIDENT,
                        ASYNC_256_RESIDENT,
                    }:
                        if receipt.get("overlap_with_backward") is not True:
                            errors.append(f"rank{rank}: async reducer overlap disabled")
                        if receipt.get("hook_count_min") != 1 or receipt.get(
                            "hook_count_max"
                        ) != 1:
                            errors.append(f"rank{rank}: async reducer hook census changed")
                        expected_order = list(
                            range(int(receipt.get("coalesced_bucket_count", -1)))
                        )
                        if receipt.get("bucket_issue_order") != expected_order:
                            errors.append(f"rank{rank}: async reducer issue order changed")

            _validate_dtype_contract(
                row,
                rank=rank,
                correctness_mode=correctness_mode,
                errors=errors,
            )
            build = row.get("build", {})
            for component in ("generator", "critic"):
                component_build = build.get(component, {})
                metadata = component_build.get("metadata", {})
                residual_runtime = metadata.get("chunked_residual_runtime", {})
                modulation_chunk = metadata.get("modulation_chunk_runtime", {})
                if metadata.get("chunked_residual_storage") != "cuda_out_of_place":
                    errors.append(f"rank{rank}/{component}: residual storage is not CUDA")
                if residual_runtime.get("requested_storage") != "cuda_out_of_place":
                    errors.append(
                        f"rank{rank}/{component}: residual runtime request changed"
                    )
                if residual_runtime.get("storage_capable") is not True:
                    errors.append(
                        f"rank{rank}/{component}: runtime cannot honor CUDA residuals"
                    )
                expected_chunk = {
                    ASYNC_128_TREE_RESIDENT: 512,
                    ASYNC_128_TREE_CHUNK2048_RESIDENT: 2048,
                    ASYNC_128_TREE_CHUNK2048_COMPILE_RESIDENT: 2048,
                }.get(args.placement)
                if expected_chunk is not None:
                    if modulation_chunk.get("effective_tokens") != [expected_chunk]:
                        errors.append(
                            f"rank{rank}/{component}: modulation chunk tokens="
                            f"{modulation_chunk.get('effective_tokens')}, "
                            f"expected={[expected_chunk]}"
                        )
                    if modulation_chunk.get("block_count") != 30:
                        errors.append(
                            f"rank{rank}/{component}: modulation chunk block census changed"
                        )
                    if modulation_chunk.get("all_blocks_match") is not True:
                        errors.append(
                            f"rank{rank}/{component}: modulation chunks are not uniform"
                        )
                compile_runtime = metadata.get("nograd_compile_runtime", {})
                if args.placement == ASYNC_128_TREE_CHUNK2048_COMPILE_RESIDENT:
                    expected_enabled = component == "generator"
                    if compile_runtime.get("enabled") is not expected_enabled:
                        errors.append(
                            f"rank{rank}/{component}: compile enabled="
                            f"{compile_runtime.get('enabled')}, expected={expected_enabled}"
                        )
                    if compile_runtime.get("scope") != (
                        "GENERATOR_NOGRAD_TRANSFORMER_BLOCKS_ONLY"
                    ):
                        errors.append(f"rank{rank}/{component}: compile scope drift")
                    if compile_runtime.get("block_count") != 30:
                        errors.append(f"rank{rank}/{component}: compile block census drift")
                    if component == "generator":
                        if compile_runtime.get("all_blocks_wrapped") is not True:
                            errors.append(f"rank{rank}/generator: not all blocks compiled")
                        if int(compile_runtime.get("compiled_dispatch_count", 0)) <= 0:
                            errors.append(f"rank{rank}/generator: compile path was not exercised")
                        if int(compile_runtime.get("eager_dispatch_count", 0)) <= 0:
                            errors.append(f"rank{rank}/generator: grad path did not stay eager")
                elif compile_runtime.get("enabled") is True:
                    errors.append(f"rank{rank}/{component}: unexpected compile enablement")
                runtime_sha = residual_runtime.get("transformer_source_sha256")
                if not isinstance(runtime_sha, str) or len(runtime_sha) != 64:
                    errors.append(
                        f"rank{rank}/{component}: missing residual runtime source identity"
                    )
                else:
                    residual_runtime_hashes.add(runtime_sha)
                placement = component_build.get("placement", {})
                engine = component_build.get("engine", {})
                expected_optimizer_impl = (
                    "PYTORCH_FUSED_LOCAL_SHARD"
                    if args.placement in {LORA_ZERO2_RESIDENT, ROLLOUT_FUSED_RESIDENT}
                    else (
                        "PYTORCH_FUSED"
                        if args.placement
                        in {
                            "resident-fast",
                            LOCAL_FAST_RESIDENT,
                            FLAT_256_RESIDENT,
                            FLAT_512_RESIDENT,
                        }
                        else "PYTORCH_DEFAULT"
                    )
                )
                if engine.get("optimizer_implementation") != expected_optimizer_impl:
                    errors.append(
                        f"rank{rank}/{component}: optimizer implementation="
                        f"{engine.get('optimizer_implementation')}, "
                        f"expected={expected_optimizer_impl}"
                    )
                if (
                    args.placement
                    in {LOCAL_FAST_RESIDENT, FLAT_256_RESIDENT, FLAT_512_RESIDENT}
                    and engine.get("gradient_clip_implementation")
                    != "PYTORCH_FOREACH_TRUE"
                ):
                    errors.append(
                        f"rank{rank}/{component}: gradient clip implementation="
                        f"{engine.get('gradient_clip_implementation')}, "
                        "expected=PYTORCH_FOREACH_TRUE"
                    )
                if (
                    args.placement in {LORA_ZERO2_RESIDENT, ROLLOUT_FUSED_RESIDENT}
                    and engine.get("gradient_clip_implementation")
                    != "GLOBAL_L2_FROM_LOCAL_SHARDS"
                ):
                    errors.append(
                        f"rank{rank}/{component}: LoRA ZeRO gradient clip changed"
                    )
                if placement.get("block_count") != 30:
                    errors.append(f"rank{rank}: {component} is not 30-block Wan1.3B")
                if placement.get("cpu_parameter_offload") is not False:
                    errors.append(f"rank{rank}: {component} parameter offload enabled")
                if args.placement == FSDP_PLACEMENT:
                    if placement.get("force_sum_reduction_for_comms") is not True:
                        errors.append(f"rank{rank}: FSDP sum reduction not forced")
                    if placement.get("fsdp_group_count") != 31:
                        errors.append(
                            f"rank{rank}: FSDP group count="
                            f"{placement.get('fsdp_group_count')}, expected=31"
                        )
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

            clean = row.get("resident_clean_runtime", {})
            if args.placement in {
                CLEAN_RESIDENT,
                ASYNC_128_RESIDENT,
                ASYNC_128_TREE_RESIDENT,
                ASYNC_128_TREE_CHUNK2048_RESIDENT,
                ASYNC_128_TREE_CHUNK2048_COMPILE_RESIDENT,
                ASYNC_256_RESIDENT,
                LOCAL_FAST_RESIDENT,
                FLAT_256_RESIDENT,
                FLAT_512_RESIDENT,
                LORA_ZERO2_RESIDENT,
                ROLLOUT_FUSED_RESIDENT,
            }:
                if clean.get("enabled") is not True:
                    errors.append(f"rank{rank}: resident-clean runtime not enabled")
                if clean.get("internal_cuda_housekeeping_skipped") is not True:
                    errors.append(f"rank{rank}: resident-clean housekeeping still enabled")
                if clean.get("trace_entries_reused") != 5:
                    errors.append(f"rank{rank}: resident-clean did not reuse five entries")
                generator_state = clean.get("generator_model_state") or {}
                critic_state = clean.get("critic_model_state") or {}
                if generator_state.get("train_transitions") != 0:
                    errors.append(f"rank{rank}: Generator train mode changed")
                if generator_state.get("adapter_enable_transitions") != 0:
                    errors.append(f"rank{rank}: Generator adapter state changed")
                if generator_state.get("adapter_disable_transitions") != 0:
                    errors.append(f"rank{rank}: Generator adapter was disabled")
                if critic_state.get("train_transitions") != 0:
                    errors.append(f"rank{rank}: Critic train mode changed")
                if critic_state.get("adapter_enable_transitions") != 1:
                    errors.append(f"rank{rank}: Critic adapter-on transition count changed")
                if critic_state.get("adapter_disable_transitions") != 1:
                    errors.append(f"rank{rank}: Critic adapter-off transition count changed")
                if args.mode == "diagnostic-timing":
                    if clean.get("deferred_finite_status") is not True:
                        errors.append(f"rank{rank}: deferred finite gate did not pass")
                    if int(clean.get("deferred_finite_check_count", 0)) <= 0:
                        errors.append(f"rank{rank}: deferred finite gate has no inputs")

            shadow = row.get("rollout_shadow", {})
            if args.placement == ROLLOUT_FUSED_RESIDENT:
                if shadow.get("enabled") is not True:
                    errors.append(f"rank{rank}: rollout shadow not enabled")
                if shadow.get("refresh_count") != 1:
                    errors.append(f"rank{rank}: rollout shadow refresh count changed")
                if int(shadow.get("routed_call_count", 0)) <= 0:
                    errors.append(f"rank{rank}: rollout shadow handled no calls")
                if shadow.get("fused") is not True:
                    errors.append(f"rank{rank}: rollout shadow was not fused")

        if len(residual_runtime_hashes) != 1:
            errors.append(
                "Helios residual runtime source differs across ranks or components"
            )

    rollout_identities: list[str] = []
    if args.mode != "validate":
        for row in rows:
            for rollout in row.get("rollout_batches", []):
                rollout_identities.extend(rollout.get("member_trace_identities", []))
        if len(rollout_identities) != 80 or len(set(rollout_identities)) != 80:
            errors.append("five world16 rollouts do not contain 80 unique identities")

    full5_values = [
        float(row["full5_seconds"]) for row in rows if "full5_seconds" in row
    ]
    memory_rows = [
        row.get("final", {}).get("memory", {})
        for row in rows
        if row.get("final", {}).get("memory")
    ]
    summary: dict[str, Any] = {
        "schema_version": "distillschedule.wan13b_fair_summary.v1",
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
                "helios_transformer_sha256": (
                    next(iter(residual_runtime_hashes))
                    if len(residual_runtime_hashes) == 1
                    else None
                ),
            }
            if len(identities) == 1
            else None
        ),
        "workload": {
            "world_size": WORLD_SIZE,
            "global_batch": 16,
            "rollout_batch_size": 1,
            "score_packing": False,
            "updates": {"generator": 1, "critic": 5, "ema": 5},
            "final_versions": {"generator": 3, "critic": 15, "ema": 15},
            "unique_rank_step_rollout_identities": len(set(rollout_identities)),
            "lora_compute_dtype": "torch.float32",
            "frozen_base_compute_dtype": "torch.bfloat16",
            "gradient_mean_convention": "SUM_THEN_DIVIDE_WORLD_SIZE",
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
                "final_trainables": rows[0].get("final", {}).get(
                    "trainable_evidence"
                ),
                "generator_metrics": rows[0].get("final", {}).get(
                    "generator_trainable_metrics"
                ),
                "critic_metrics": rows[0].get("final", {}).get(
                    "critic_trainable_metrics"
                ),
                "ema_checksum_rank0": rows[0].get("final", {}).get("ema_checksum"),
                "lora_compute_dtype": {
                    component: rows[0]
                    .get("build", {})
                    .get(component, {})
                    .get("lora_compute_dtype_contract")
                    for component in ("generator", "critic")
                },
            }
            if correctness_mode and rows
            else "NOT_RUN"
        ),
        "formal_eligibility": args.mode == "timing" and not errors,
        "diagnostic_only": args.mode in {"profile", "diagnostic-timing"},
    }
    _atomic_json(args.output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
