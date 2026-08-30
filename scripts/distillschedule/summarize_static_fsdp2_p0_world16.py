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
        "--attention-backend",
        choices=("sdpa", "local-fa2"),
        required=True,
    )
    parser.add_argument(
        "--modulation-chunk-tokens",
        type=int,
        choices=(512, 1024, 2048),
        default=512,
    )
    parser.add_argument(
        "--selective-flash-checkpoint-stride",
        type=int,
        choices=(0, 2, 4, 10),
        default=0,
    )
    parser.add_argument(
        "--generator-nograd-cross-kv-cache",
        choices=("on", "off"),
        default="off",
    )
    parser.add_argument(
        "--generator-nograd-static-context-cache",
        choices=("on", "off"),
        default="off",
    )
    parser.add_argument(
        "--generator-nograd-persistent-unshard",
        choices=("on", "off"),
        default="off",
    )
    parser.add_argument(
        "--generator-residual-storage",
        choices=("cpu_inplace", "cuda_out_of_place"),
        required=True,
    )
    parser.add_argument(
        "--critic-residual-storage",
        choices=("cpu_inplace", "cuda_out_of_place"),
        required=True,
    )
    parser.add_argument(
        "--generator-nograd-lora-fused-epilogue",
        choices=("on", "off"),
        required=True,
    )
    parser.add_argument(
        "--all-role-lora-scale1-add",
        choices=("on", "off"),
        required=True,
    )
    parser.add_argument(
        "--optimizer-dispatch-warmup",
        choices=("on", "off"),
        required=True,
    )
    parser.add_argument(
        "--generator-nograd-residual-fusion",
        choices=("on", "off"),
        required=True,
    )
    parser.add_argument(
        "--exact-grad-residual-fusion",
        choices=("on", "off"),
        required=True,
    )
    parser.add_argument(
        "--exact-modulated-norm-epilogue",
        choices=("on", "off"),
        required=True,
    )
    parser.add_argument(
        "--exact-modulated-norm-materialization",
        choices=("on", "off"),
        required=True,
    )
    parser.add_argument(
        "--exact-modulated-norm-backward",
        choices=("on", "off"),
        required=True,
    )
    parser.add_argument(
        "--exact-modulated-norm-layernorm",
        choices=("on", "off"),
        required=True,
    )
    parser.add_argument(
        "--exact-modulated-norm-fused-backward",
        choices=("on", "off"),
        required=True,
    )
    parser.add_argument(
        "--elide-explicit-rollout-sync",
        choices=("on", "off"),
        default="off",
    )
    parser.add_argument(
        "--elide-critic-scalar-logs",
        choices=("on", "off"),
        default="off",
    )
    parser.add_argument(
        "--elide-official-generator-free-memory",
        choices=("on", "off"),
        default="off",
    )
    parser.add_argument("--critic-only-step", type=int, choices=(10,))
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
        "attention_backend": args.attention_backend,
        "modulation_chunk_tokens": args.modulation_chunk_tokens,
        "selective_flash_checkpoint_stride": (
            args.selective_flash_checkpoint_stride
        ),
        "generator_nograd_cross_kv_cache": (
            args.generator_nograd_cross_kv_cache == "on"
        ),
        "generator_nograd_static_context_cache": (
            args.generator_nograd_static_context_cache == "on"
        ),
        "generator_nograd_persistent_unshard": (
            args.generator_nograd_persistent_unshard == "on"
        ),
        "generator_residual_storage": args.generator_residual_storage,
        "critic_residual_storage": args.critic_residual_storage,
        "generator_nograd_lora_fused_epilogue": (
            args.generator_nograd_lora_fused_epilogue == "on"
        ),
        "all_role_lora_scale1_add": args.all_role_lora_scale1_add == "on",
        "optimizer_dispatch_warmup": args.optimizer_dispatch_warmup == "on",
        "generator_nograd_residual_fusion": (
            args.generator_nograd_residual_fusion == "on"
        ),
        "exact_grad_residual_fusion": args.exact_grad_residual_fusion == "on",
        "exact_modulated_norm_epilogue": (
            args.exact_modulated_norm_epilogue == "on"
        ),
        "exact_modulated_norm_materialization": (
            args.exact_modulated_norm_materialization == "on"
        ),
        "exact_modulated_norm_backward": (
            args.exact_modulated_norm_backward == "on"
        ),
        "exact_modulated_norm_layernorm": (
            args.exact_modulated_norm_layernorm == "on"
        ),
        "exact_modulated_norm_fused_backward": (
            args.exact_modulated_norm_fused_backward == "on"
        ),
        "elide_explicit_rollout_sync": (
            args.elide_explicit_rollout_sync == "on"
        ),
        "elide_critic_scalar_logs": args.elide_critic_scalar_logs == "on",
        "elide_official_generator_free_memory": (
            args.elide_official_generator_free_memory == "on"
        ),
    }
    if args.critic_only_step is not None:
        if args.variant != "no-gan" or args.mode != "profile":
            raise ValueError("--critic-only-step requires No-GAN profile mode")
        expected_final = {
            "generator_logical_version": 3,
            "critic_logical_version": 11,
            "ema_step": 10,
        }
        expected_ema = 0
        expected_critic_updates = 1
        expected_logical_rollouts = 1
    else:
        expected_final = (
            {"generator_logical_version": 3, "critic_logical_version": 15, "ema_step": 15}
            if args.variant == "no-gan"
            else {"generator_logical_version": 201, "critic_logical_version": 1005, "ema_step": 801}
        )
        expected_ema = 5 if args.variant == "no-gan" else 1
        expected_critic_updates = 5
        expected_logical_rollouts = 5
    expected_gradient_steps = {"generator": 1, "critic": expected_critic_updates}
    state_audits: list[dict[str, Any] | None] = []
    loss_audits: list[dict[str, Any] | None] = []
    input_reuse: list[dict[str, Any] | None] = []
    optimizer_warmups: list[dict[str, Any] | None] = []
    residual_fusions: list[dict[str, Any] | None] = []
    modulated_norm_fusions: list[dict[str, Any] | None] = []
    attention_backends: list[dict[str, Any] | None] = []
    modulation_chunks: list[dict[str, Any] | None] = []
    checkpoint_policies: list[dict[str, Any] | None] = []
    cross_kv_caches: list[dict[str, Any] | None] = []
    static_context_caches: list[dict[str, Any] | None] = []
    persistent_unshard_receipts: list[dict[str, Any] | None] = []
    generator_free_memory_receipts: list[dict[str, Any] | None] = []

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

        attention = value.get("attention_backend", {})
        attention_backends.append(attention)
        prefix = f"rank{rank} attention backend"
        if attention.get("status") != "PASS":
            errors.append(f"{prefix} receipt did not pass")
        if attention.get("requested") != args.attention_backend:
            errors.append(f"{prefix} request drifted")
        if attention.get("runtime") != args.attention_backend:
            errors.append(f"{prefix} runtime={attention.get('runtime')!r}")
        if args.attention_backend == "sdpa":
            if attention.get("local_kernels"):
                errors.append(f"{prefix} inherited LOCAL_KERNELS")
            if attention.get("flash_attn_func_available") is not False:
                errors.append(f"{prefix} unexpectedly loaded FlashAttention")
        else:
            if attention.get("flash_attn_func_available") is not True:
                errors.append(f"{prefix} did not load FlashAttention")
            if "flash_attn2" not in str(attention.get("flash_attn_func_module")):
                errors.append(f"{prefix} callable is not FlashAttention-2")
            if attention.get("extension_sha256") != (
                "c2acd487c5d1d77b7f6721eabc0fee4f8c5c8f9e682aadf43bfa99b394eecba7"
            ):
                errors.append(f"{prefix} extension identity changed")

        free_memory = value.get("generator_free_memory", {})
        generator_free_memory_receipts.append(free_memory)
        free_memory_prefix = f"rank{rank} official Generator free_memory"
        if expected_policy["elide_official_generator_free_memory"]:
            if args.variant != "gan":
                errors.append(f"{free_memory_prefix} requested outside GAN")
            if not free_memory.get("enabled"):
                errors.append(f"{free_memory_prefix} scope was not enabled")
            if free_memory.get("scope") != (
                "OFFICIAL_GAN_GENERATOR_SCORE_DMD_GAN_G_ONLY"
            ):
                errors.append(f"{free_memory_prefix} scope changed")
            if free_memory.get("shared_source_modified") is not False:
                errors.append(f"{free_memory_prefix} modified shared source")
            expected_sources = {
                "compute_distribution_matching_loss": (
                    "c479e4f3eec2730d1e991ecb19e06588ec30510dabe48e2daf3846d9f7960db1"
                ),
                "Gan_D_Loss_With_Cached_Grad": (
                    "8455f5d265b189208491681e6372790f7b5809773953a6c855a668f10b632643"
                ),
                "free_memory": (
                    "d16707d1b9935038a99f9a927043844d7eea8d22a9b6e008fbfc014c49885df3"
                ),
            }
            if free_memory.get("source_sha256") != expected_sources:
                errors.append(f"{free_memory_prefix} source identity changed")
            if args.mode == "validate":
                expected_scopes = expected_calls = expected_executed = expected_elided = 0
            elif args.mode == "correctness":
                expected_scopes, expected_calls = 1, 4
                expected_executed, expected_elided = 4, 0
            else:
                expected_scopes, expected_calls = 1, 4
                expected_executed, expected_elided = 0, 4
            observed = {
                "scope_entry_count": expected_scopes,
                "call_count": expected_calls,
                "executed_count": expected_executed,
                "elided_count": expected_elided,
                "restored_count": expected_scopes,
            }
            for key, expected in observed.items():
                if int(free_memory.get(key, -1)) != expected:
                    errors.append(
                        f"{free_memory_prefix} {key}={free_memory.get(key)!r}, "
                        f"expected={expected}"
                    )
        elif free_memory.get("enabled"):
            errors.append(f"{free_memory_prefix} enabled outside requested policy")

        if args.mode != "validate":
            expected_sync = not expected_policy["elide_explicit_rollout_sync"]
            generator_rollout = value.get("generator", {}).get("rollout", {})
            if generator_rollout.get("empty_cache_before_rollout") is not expected_sync:
                errors.append(f"rank{rank} Generator rollout empty_cache policy drifted")
            if generator_rollout.get("synchronize_after_rollout") is not expected_sync:
                errors.append(f"rank{rank} Generator rollout sync policy drifted")
            rollout_batches = value.get("rollout_batches", [])
            if any(
                batch.get("synchronize_after_rollout") is not expected_sync
                for batch in rollout_batches
            ):
                errors.append(f"rank{rank} Critic rollout sync policy drifted")
            fence = value.get("rollout_fence_policy", {})
            expected_batch_count = len(rollout_batches)
            if fence.get("elide_explicit_rollout_sync") is not (
                expected_policy["elide_explicit_rollout_sync"]
            ):
                errors.append(f"rank{rank} rollout fence request drifted")
            if int(fence.get("generator_update_empty_cache_count", -1)) != int(
                expected_sync
            ):
                errors.append(f"rank{rank} rollout empty_cache census drifted")
            if int(fence.get("generator_update_sync_count", -1)) != int(expected_sync):
                errors.append(f"rank{rank} Generator rollout sync census drifted")
            if int(fence.get("critic_rollout_sync_count", -1)) != (
                expected_batch_count if expected_sync else 0
            ):
                errors.append(f"rank{rank} Critic rollout sync census drifted")
            if fence.get("outer_full5_boundary_sync_preserved") is not True:
                errors.append(f"rank{rank} outer Full5 boundary sync was not preserved")

            persistence = value.get("generator_nograd_persistent_unshard", {})
            persistent_unshard_receipts.append(persistence)
            persistence_prefix = f"rank{rank} Generator persistent unshard"
            if expected_policy["generator_nograd_persistent_unshard"]:
                expected_persistence = {
                    "enabled": True,
                    "scope": "GENERATOR_NOGRAD_ROLLOUTS_ONLY",
                    "expected_fsdp_module_count": 41,
                    "fsdp_module_count": 41,
                    "parameter_group_count": 41,
                    "rollout_batch_count": expected_logical_rollouts,
                    "pre_empty_cache_count": 1,
                    "post_empty_cache_count": 1,
                    "set_no_reshard_count": 41,
                    "restore_reshard_policy_count": 41,
                    "explicit_reshard_count": 41,
                    "fully_unsharded_before_exit_count": 41,
                    "unsharded_after_exit_count": 0,
                    "restored": True,
                }
                for key, expected in expected_persistence.items():
                    if persistence.get(key) != expected:
                        errors.append(
                            f"{persistence_prefix} {key}="
                            f"{persistence.get(key)!r}, expected={expected!r}"
                        )
                rollout_memory = persistence.get("rollout_memory", [])
                if len(rollout_memory) != expected_logical_rollouts:
                    errors.append(
                        f"{persistence_prefix} memory samples="
                        f"{len(rollout_memory)}, expected={expected_logical_rollouts}"
                    )
            elif persistence.get("enabled"):
                errors.append(
                    f"{persistence_prefix} enabled outside requested policy"
                )

            scalar_logs = value.get("critic_scalar_logs", {})
            scalar_prefix = f"rank{rank} official Critic scalar logs"
            if expected_policy["elide_critic_scalar_logs"]:
                if args.variant != "gan":
                    errors.append(f"{scalar_prefix} requested outside GAN")
                if not scalar_logs.get("enabled"):
                    errors.append(f"{scalar_prefix} patch was not enabled")
                if scalar_logs.get("scope") != "OFFICIAL_GAN_CRITIC_SCALAR_LOGS_ONLY":
                    errors.append(f"{scalar_prefix} scope changed")
                if scalar_logs.get("shared_source_modified") is not False:
                    errors.append(f"{scalar_prefix} modified shared source")
                expected_calls = expected_critic_updates
                if int(scalar_logs.get("call_count", -1)) != expected_calls:
                    errors.append(
                        f"{scalar_prefix} call_count={scalar_logs.get('call_count')}"
                    )
                if args.mode == "correctness":
                    expected_collected, expected_elided = 4 * expected_calls, 0
                else:
                    expected_collected, expected_elided = 0, 4 * expected_calls
                if int(scalar_logs.get("collected_scalar_count", -1)) != expected_collected:
                    errors.append(
                        f"{scalar_prefix} collected="
                        f"{scalar_logs.get('collected_scalar_count')}"
                    )
                if int(scalar_logs.get("elided_scalar_count", -1)) != expected_elided:
                    errors.append(
                        f"{scalar_prefix} elided="
                        f"{scalar_logs.get('elided_scalar_count')}"
                    )
            elif scalar_logs.get("enabled"):
                errors.append(f"{scalar_prefix} enabled outside requested policy")

            role_chunks = value.get("build", {})
            rank_chunks: dict[str, Any] = {}
            for role in ("generator", "critic"):
                receipt = role_chunks.get(role, {}).get("modulation_chunks", {})
                rank_chunks[role] = receipt
                chunk_prefix = f"rank{rank} {role} modulation chunks"
                if receipt.get("status") != "PASS":
                    errors.append(f"{chunk_prefix} receipt did not pass")
                if receipt.get("role") != role:
                    errors.append(f"{chunk_prefix} role drifted")
                if int(receipt.get("requested_chunk_tokens", -1)) != int(
                    args.modulation_chunk_tokens
                ):
                    errors.append(f"{chunk_prefix} request drifted")
                if receipt.get("block_count") != 40:
                    errors.append(f"{chunk_prefix} block inventory drifted")
                if receipt.get("observed_chunk_tokens") != [
                    int(args.modulation_chunk_tokens)
                ]:
                    errors.append(f"{chunk_prefix} assignment drifted")
            modulation_chunks.append(rank_chunks)

            role_checkpoint_policies: dict[str, Any] = {}
            for role in ("generator", "critic"):
                checkpoint_policy = (
                    value.get("build", {})
                    .get(role, {})
                    .get("checkpoint_policy", {})
                )
                role_checkpoint_policies[role] = checkpoint_policy
                prefix = f"rank{rank} {role} checkpoint policy"
                expected_stride = (
                    expected_policy["selective_flash_checkpoint_stride"]
                    if role == "critic"
                    else 0
                )
                # Historical synthetic summarizer fixtures predate this
                # opt-in receipt.  Missing data is admissible only when the
                # selective candidate is disabled; an enabled run fails closed.
                if not checkpoint_policy:
                    if expected_stride:
                        errors.append(f"{prefix} receipt is missing")
                    continue
                if checkpoint_policy.get("block_count") != 40:
                    errors.append(f"{prefix} block inventory drifted")
                if checkpoint_policy.get("adds_synchronization") is not False:
                    errors.append(f"{prefix} introduced synchronization")
                if checkpoint_policy.get("changes_module_math") is not False:
                    errors.append(f"{prefix} changed module math")
                if (
                    checkpoint_policy.get("changes_physical_model_call_inventory")
                    is not False
                ):
                    errors.append(f"{prefix} changed physical model calls")

                expected_indices = (
                    list(range(0, 40, expected_stride))
                    if expected_stride
                    else []
                )
                if checkpoint_policy.get("selective_flash_enabled") is not bool(
                    expected_stride
                ):
                    errors.append(f"{prefix} selective Flash enabled state changed")
                if checkpoint_policy.get("selective_flash_stride") != expected_stride:
                    errors.append(f"{prefix} selective Flash stride changed")
                if (
                    checkpoint_policy.get("selective_flash_block_indices")
                    != expected_indices
                ):
                    errors.append(f"{prefix} selective Flash block selection changed")

                counter_keys = (
                    "selective_flash_context_count",
                    "selective_flash_original_save_count",
                    "selective_flash_recompute_reuse_count",
                )
                if not expected_stride:
                    if any(
                        int(checkpoint_policy.get(key, 0)) != 0
                        for key in counter_keys
                    ):
                        errors.append(f"{prefix} selective Flash ran outside policy")
                    continue

                if checkpoint_policy.get("selective_flash_op") != (
                    "_flash_attn2_cuda_a511a4c."
                    "_flash_attn_forward.default"
                ):
                    errors.append(f"{prefix} selective Flash op identity changed")
                counts = checkpoint_policy.get("block_call_counts", {})
                if set(counts) != {str(index) for index in range(40)}:
                    errors.append(f"{prefix} lacks per-block checkpoint census")
                    continue
                per_block = [int(counts[str(index)]) for index in range(40)]
                if min(per_block, default=0) <= 0:
                    errors.append(f"{prefix} selective Flash was not exercised")
                if len(set(per_block)) != 1:
                    errors.append(f"{prefix} per-block calls drifted")
                invocations = per_block[0] if per_block else 0
                expected_contexts = invocations * len(expected_indices)
                contexts = int(
                    checkpoint_policy.get("selective_flash_context_count", -1)
                )
                original_saves = int(
                    checkpoint_policy.get("selective_flash_original_save_count", -1)
                )
                replay_reuses = int(
                    checkpoint_policy.get(
                        "selective_flash_recompute_reuse_count", -1
                    )
                )
                if contexts != expected_contexts:
                    errors.append(
                        f"{prefix} selective contexts={contexts}, "
                        f"expected={expected_contexts}"
                    )
                # Each pinned Helios block invokes the same fixed-length FA2
                # dispatcher once for self-attention and once for cross-attention.
                if original_saves != expected_contexts * 2:
                    errors.append(
                        f"{prefix} original saves={original_saves}, "
                        f"expected={expected_contexts * 2}"
                    )
                if replay_reuses != original_saves:
                    errors.append(
                        f"{prefix} replay reuses={replay_reuses}, "
                        f"expected={original_saves}"
                    )
                if int(checkpoint_policy.get("checkpoint_forward_count", -1)) != (
                    invocations * 40
                ):
                    errors.append(f"{prefix} checkpoint call census changed")
            checkpoint_policies.append(role_checkpoint_policies)
        else:
            modulation_chunks.append(None)
            checkpoint_policies.append(None)

        hardening = value.get("p0_hardening", {})
        input_reuse.append(hardening.get("input_reuse"))
        if args.mode != "validate" and hardening.get("tensor_parity_policy") not in {
            "multi_metric",
            "multi_metric_pre_update",
        }:
            errors.append(f"rank{rank} missing hardened multi-metric parity receipt")

        if args.mode != "validate":
            role_warmups: dict[str, Any] = {}
            for role in ("generator", "critic"):
                warmup = (
                    value.get("build", {})
                    .get(role, {})
                    .get("optimizer_restore", {})
                    .get("optimizer_dispatch_warmup", {})
                )
                role_warmups[role] = warmup
                prefix = f"rank{rank} {role} optimizer warmup"
                if expected_policy["optimizer_dispatch_warmup"]:
                    if not warmup.get("enabled"):
                        errors.append(f"{prefix} was not enabled")
                    if warmup.get("scope") != "SETUP_ONLY_BEFORE_FORMAL_TIMER":
                        errors.append(f"{prefix} scope changed")
                    if warmup.get("formal_timer_includes_warmup") is not False:
                        errors.append(f"{prefix} contaminated formal timing")
                    if warmup.get("parameter_changed_elements_before_restore") != 0:
                        errors.append(f"{prefix} changed parameters")
                    if warmup.get("state_changed_elements_after_restore") != 0:
                        errors.append(f"{prefix} did not restore Adam state")
                    if warmup.get("step_state_restored") is not True:
                        errors.append(f"{prefix} did not restore step state")
                    if warmup.get("gradients_cleared") is not True:
                        errors.append(f"{prefix} left gradients attached")
                elif warmup.get("enabled"):
                    errors.append(f"{prefix} enabled outside requested policy")
            optimizer_warmups.append(role_warmups)
            residual_fusion = (
                value.get("build", {})
                .get("generator", {})
                .get("generator_nograd_residual_fusion", {})
            )
            residual_fusions.append(residual_fusion)
            if (
                expected_policy["generator_nograd_residual_fusion"]
                or expected_policy["exact_grad_residual_fusion"]
            ):
                prefix = f"rank{rank} residual fusion"
                if not residual_fusion.get("enabled"):
                    errors.append(f"{prefix} was not enabled")
                expected_scope = (
                    "GENERATOR_NOGRAD_AND_TRAINABLE_TEMB_EXACT_GRAD_GATED_RESIDUAL"
                    if expected_policy["generator_nograd_residual_fusion"]
                    and expected_policy["exact_grad_residual_fusion"]
                    else (
                        "TRAINABLE_TEMB_EXACT_GRAD_GATED_RESIDUAL_ONLY"
                        if expected_policy["exact_grad_residual_fusion"]
                        else "GENERATOR_NOGRAD_GATED_RESIDUAL_ONLY"
                    )
                )
                if residual_fusion.get("scope") != expected_scope:
                    errors.append(f"{prefix} scope changed")
                warmup = residual_fusion.get("warmup", {})
                if warmup.get("status") != "PASS":
                    errors.append(f"{prefix} warmup did not pass")
                if warmup.get("formal_timer_includes_warmup") is not False:
                    errors.append(f"{prefix} warmup contaminated formal timing")
                if expected_policy["generator_nograd_residual_fusion"]:
                    if int(residual_fusion.get("fused_call_count", 0)) <= 0:
                        errors.append(f"{prefix} fused no no-grad helper calls")
                    if int(residual_fusion.get("fused_chunk_count", 0)) <= 0:
                        errors.append(f"{prefix} fused no no-grad chunks")
                    if int(residual_fusion.get("reference_fallback_count", -1)) != 0:
                        errors.append(
                            f"{prefix} no-grad fallback="
                            f"{residual_fusion.get('reference_fallback_count')}"
                        )
                if expected_policy["exact_grad_residual_fusion"]:
                    if residual_fusion.get("exact_grad_requested") is not True:
                        errors.append(f"{prefix} exact-grad request was lost")
                    if warmup.get("backward_kernel") is not True:
                        errors.append(f"{prefix} backward kernel was not warmed")
                    for key in (
                        "grad_scope_active_call_count",
                        "grad_fused_call_count",
                        "grad_fused_chunk_count",
                        "grad_backward_call_count",
                        "grad_backward_chunk_count",
                    ):
                        if int(residual_fusion.get(key, 0)) <= 0:
                            errors.append(f"{prefix} {key}=0")
                    if int(residual_fusion.get("grad_reference_fallback_count", -1)) != 0:
                        errors.append(
                            f"{prefix} exact-grad fallback="
                            f"{residual_fusion.get('grad_reference_fallback_count')}"
                        )
                    fused_calls = int(
                        residual_fusion.get("grad_fused_call_count", 0)
                    )
                    saved_residuals = int(
                        residual_fusion.get("grad_saved_residual_tensor_count", -1)
                    )
                    if saved_residuals != fused_calls:
                        errors.append(
                            f"{prefix} saved residual count={saved_residuals}, "
                            f"expected={fused_calls}"
                        )
                    if int(
                        residual_fusion.get("grad_trainable_temb_call_count", 0)
                    ) <= 0:
                        errors.append(f"{prefix} saw no trainable timestep inputs")
                    if int(
                        residual_fusion.get("grad_temb_gradient_tensor_count", 0)
                    ) <= 0:
                        errors.append(f"{prefix} produced no timestep gradients")
                    descriptor_counts = residual_fusion.get(
                        "grad_temb_descriptor_counts", {}
                    )
                    if not descriptor_counts or not set(descriptor_counts).issubset(
                        {"tensor", "phase_local"}
                    ):
                        errors.append(
                            f"{prefix} timestep descriptor counts changed: "
                            f"{descriptor_counts!r}"
                        )
                    if set(descriptor_counts) != {"phase_local"}:
                        errors.append(
                            f"{prefix} live timestep descriptors are not exclusively "
                            f"phase_local: {descriptor_counts!r}"
                        )
                    dtype_counts = residual_fusion.get(
                        "grad_temb_dtype_counts", {}
                    )
                    if not dtype_counts or not set(dtype_counts).issubset(
                        {"torch.bfloat16", "torch.float32"}
                    ):
                        errors.append(
                            f"{prefix} timestep dtype counts changed: "
                            f"{dtype_counts!r}"
                        )
                    if set(dtype_counts) != {"torch.bfloat16"}:
                        errors.append(
                            f"{prefix} live timestep dtypes are not exclusively BF16: "
                            f"{dtype_counts!r}"
                        )
                    backward_chunks = int(
                        residual_fusion.get("grad_backward_chunk_count", 0)
                    )
                    fused_staging_chunks = int(
                        residual_fusion.get(
                            "grad_phase_local_fused_staging_chunk_count", 0
                        )
                    )
                    if fused_staging_chunks != backward_chunks:
                        errors.append(
                            f"{prefix} fused timestep staging chunks="
                            f"{fused_staging_chunks}, expected={backward_chunks}"
                        )
            elif residual_fusion.get("enabled"):
                errors.append(f"rank{rank} residual fusion enabled outside requested policy")
            modulated_norm_fusion = (
                value.get("build", {})
                .get("generator", {})
                .get("exact_modulated_norm_epilogue", {})
            )
            modulated_norm_fusions.append(modulated_norm_fusion)
            prefix = f"rank{rank} exact modulated-norm epilogue"
            if expected_policy["exact_modulated_norm_epilogue"]:
                if not modulated_norm_fusion.get("enabled"):
                    errors.append(f"{prefix} was not enabled")
                if modulated_norm_fusion.get("scope") != (
                    "ALL_ROLE_REGISTERED_B1_B3_MODULATED_NORM_FORWARD_EPILOGUE"
                ):
                    errors.append(f"{prefix} scope changed")
                expected_backward = expected_policy[
                    "exact_modulated_norm_backward"
                ]
                expected_layernorm = expected_policy[
                    "exact_modulated_norm_layernorm"
                ]
                expected_fused_backward = expected_policy[
                    "exact_modulated_norm_fused_backward"
                ]
                expected_semantic_contract = (
                    (
                        "FUSED_BF16_INPUT_FP32_AFFINE_FREE_LAYERNORM_WIDTH5120;"
                        if expected_layernorm
                        else "PYTORCH_FP32_LAYERNORM_UNCHANGED;"
                    )
                    + "ADD_RN_F32_ONE_PLUS_SCALE;"
                    "MUL_RN_F32;ADD_RN_F32_SHIFT;BF16_RN_STORE;"
                    + (
                        (
                            "FUSED_TRITON_LAYERNORM_BACKWARD_AND_DETERMINISTIC_"
                            "COMPONENT_GRADIENTS"
                            if expected_fused_backward
                            else "EXPLICIT_NATIVE_LAYERNORM_BACKWARD_AND_COMPONENT_GRADIENTS"
                        )
                        if expected_backward
                        else "SOURCE_BACKWARD_UNCHANGED"
                    )
                )
                if (
                    modulated_norm_fusion.get("semantic_contract")
                    != expected_semantic_contract
                ):
                    errors.append(f"{prefix} semantic contract changed")
                warmup = modulated_norm_fusion.get("warmup", {})
                if warmup.get("status") != "PASS":
                    errors.append(f"{prefix} warmup did not pass")
                if warmup.get("scope") != "SETUP_ONLY_BEFORE_FORMAL_TIMER":
                    errors.append(f"{prefix} warmup scope changed")
                if warmup.get("formal_timer_includes_warmup") is not False:
                    errors.append(f"{prefix} warmup contaminated formal timing")
                if int(modulated_norm_fusion.get("fused_call_count", 0)) <= 0:
                    errors.append(f"{prefix} fused no calls")
                if int(modulated_norm_fusion.get("fused_chunk_count", 0)) <= 0:
                    errors.append(f"{prefix} fused no chunks")
                if int(
                    modulated_norm_fusion.get("reference_fallback_count", -1)
                ) != 0:
                    errors.append(
                        f"{prefix} fallback="
                        f"{modulated_norm_fusion.get('reference_fallback_count')}"
                    )
                expected_materialization = expected_policy[
                    "exact_modulated_norm_materialization"
                ]
                if (
                    modulated_norm_fusion.get("materialization_requested")
                    is not expected_materialization
                ):
                    errors.append(f"{prefix} materialization request drifted")
                if expected_materialization:
                    if modulated_norm_fusion.get("materialization_contract") != (
                        "SOURCE_TABLE_PLUS_COMPONENT_ADD_RN_F32_FUSED_BEFORE_"
                        "EXISTING_EPILOGUE"
                    ):
                        errors.append(f"{prefix} materialization contract changed")
                    if int(
                        modulated_norm_fusion.get(
                            "materialization_fused_call_count", 0
                        )
                    ) <= 0:
                        errors.append(f"{prefix} fused no materialization calls")
                    if int(
                        modulated_norm_fusion.get(
                            "materialization_fused_chunk_count", 0
                        )
                    ) <= 0:
                        errors.append(f"{prefix} fused no materialization chunks")
                    if warmup.get("fuse_materialization") is not True:
                        errors.append(f"{prefix} did not warm materialization kernel")
                if (
                    modulated_norm_fusion.get("layernorm_requested")
                    is not expected_layernorm
                ):
                    errors.append(f"{prefix} LayerNorm request drifted")
                if expected_layernorm:
                    if modulated_norm_fusion.get("layernorm_contract") != (
                        "BF16_INPUT;FP32_AFFINE_FREE_LAYERNORM_WIDTH5120;"
                        "SOURCE_TABLE_COMPONENT_FP32_ADDS;"
                        "SOURCE_MODULATION_FP32_FORMULA;BF16_OUTPUT;"
                        + (
                            "TRITON_FUSED_LAYERNORM_BACKWARD"
                            if expected_fused_backward
                            else "EXPLICIT_NATIVE_LAYERNORM_BACKWARD_UNCHANGED"
                        )
                    ):
                        errors.append(f"{prefix} LayerNorm contract changed")
                    if int(
                        modulated_norm_fusion.get("layernorm_fused_call_count", 0)
                    ) <= 0:
                        errors.append(f"{prefix} fused no LayerNorm calls")
                    if int(
                        modulated_norm_fusion.get("layernorm_fused_chunk_count", 0)
                    ) <= 0:
                        errors.append(f"{prefix} fused no LayerNorm chunks")
                    if warmup.get("fuse_layernorm") is not True:
                        errors.append(f"{prefix} did not warm fused LayerNorm kernel")
                else:
                    if int(
                        modulated_norm_fusion.get("layernorm_fused_call_count", 0)
                    ) != 0:
                        errors.append(f"{prefix} fused LayerNorm ran while disabled")
                    if warmup.get("fuse_layernorm") is not False:
                        errors.append(f"{prefix} unexpectedly warmed fused LayerNorm")
                if (
                    modulated_norm_fusion.get("backward_requested")
                    is not expected_backward
                ):
                    errors.append(f"{prefix} backward request drifted")
                if expected_backward:
                    expected_backward_contract = (
                        "SOURCE_CHUNK_ORDER;TRITON_FP32_LAYERNORM_BACKWARD;"
                        "DETERMINISTIC_BF16_COMPONENT_GRADIENT_ACCUMULATION"
                        if expected_fused_backward
                        else (
                            "SOURCE_CHUNK_ORDER;FP32_NATIVE_LAYERNORM_BACKWARD;"
                            "BF16_COMPONENT_GRADIENT_ACCUMULATION"
                        )
                    )
                    if (
                        modulated_norm_fusion.get("backward_contract")
                        != expected_backward_contract
                    ):
                        errors.append(f"{prefix} backward contract changed")
                    if args.mode != "validate":
                        backward_calls = int(
                            modulated_norm_fusion.get("backward_call_count", 0)
                        )
                        backward_chunks = int(
                            modulated_norm_fusion.get("backward_chunk_count", 0)
                        )
                        if backward_calls <= 0 or backward_chunks <= 0:
                            errors.append(f"{prefix} explicit backward executed no work")
                        if not modulated_norm_fusion.get("backward_layout_counts"):
                            errors.append(f"{prefix} explicit backward recorded no layout")
                if (
                    modulated_norm_fusion.get("backward_kernel_requested")
                    is not expected_fused_backward
                ):
                    errors.append(f"{prefix} fused backward request drifted")
                if expected_fused_backward:
                    if modulated_norm_fusion.get("backward_kernel_contract") != (
                        "SOURCE_CHUNK_ORDER;BF16_INPUT_AND_UPSTREAM;FP32_AFFINE_"
                        "FREE_LAYERNORM_GRAD_INPUT;DETERMINISTIC_PHASE_LOCAL_OR_"
                        "TENSOR_COMPONENT_REDUCTIONS;BF16_GRADIENT_OUTPUT"
                    ):
                        errors.append(f"{prefix} fused backward contract changed")
                    if int(
                        modulated_norm_fusion.get(
                            "backward_kernel_reference_fallback_count", -1
                        )
                    ) != 0:
                        errors.append(f"{prefix} fused backward used a fallback")
                    backward_warmup = warmup.get("fused_backward", {})
                    if backward_warmup.get("status") != "PASS":
                        errors.append(f"{prefix} fused backward warmup did not pass")
                    if backward_warmup.get("scope") != (
                        "SETUP_ONLY_BEFORE_FORMAL_TIMER"
                    ):
                        errors.append(f"{prefix} fused backward warmup scope changed")
                    if args.mode != "validate":
                        if int(
                            modulated_norm_fusion.get(
                                "backward_kernel_call_count", 0
                            )
                        ) <= 0:
                            errors.append(f"{prefix} fused backward executed no calls")
                        if int(
                            modulated_norm_fusion.get(
                                "backward_hidden_kernel_launch_count", 0
                            )
                        ) <= 0:
                            errors.append(f"{prefix} fused hidden-gradient kernel did not run")
                        if int(
                            modulated_norm_fusion.get(
                                "backward_component_kernel_launch_count", 0
                            )
                        ) <= 0:
                            errors.append(f"{prefix} fused component kernel did not run")
                else:
                    if int(
                        modulated_norm_fusion.get("backward_kernel_call_count", 0)
                    ) != 0:
                        errors.append(f"{prefix} fused backward ran while disabled")
            elif modulated_norm_fusion.get("enabled"):
                errors.append(f"{prefix} enabled outside requested policy")
            cross_kv_cache = (
                value.get("build", {})
                .get("generator", {})
                .get("generator_nograd_cross_kv_cache", {})
            )
            cross_kv_caches.append(cross_kv_cache)
            prefix = f"rank{rank} Generator no-grad cross-attention KV cache"
            if expected_policy["generator_nograd_cross_kv_cache"]:
                install_prefix = (
                    f"rank{rank} Generator no-grad cross-attention KV cache"
                )
                if not cross_kv_cache.get("enabled"):
                    errors.append(f"{prefix} was not enabled")
                if cross_kv_cache.get("scope") != (
                    "GENERATOR_NOGRAD_ONE_ROLLOUT_ONLY"
                ):
                    errors.append(f"{prefix} scope changed")
                if cross_kv_cache.get("changes_model_call_count") is not False:
                    errors.append(f"{prefix} changed model-call semantics")
                if cross_kv_cache.get("changes_attention_math") is not False:
                    errors.append(f"{prefix} changed attention math")
                if cross_kv_cache.get("changes_fsdp_collectives") is not False:
                    errors.append(f"{prefix} changed FSDP collectives")
                if cross_kv_cache.get("cross_rollout_reuse") is not False:
                    errors.append(f"{prefix} crossed a rollout boundary")
                for key in (
                    "cache_miss_count",
                    "encoder_signature_mismatch_count",
                    "cached_tensor_mutation_count",
                    "grad_enabled_call_count",
                    "unexpected_cross_attention_count",
                    "source_encoder_identity_mismatch_count",
                    "missing_source_encoder_count",
                ):
                    if int(cross_kv_cache.get(key, -1)) != 0:
                        errors.append(
                            f"{prefix} {key}={cross_kv_cache.get(key)}"
                        )
            elif cross_kv_cache.get("enabled"):
                errors.append(f"{prefix} enabled outside requested policy")
            static_context_cache = (
                value.get("build", {})
                .get("generator", {})
                .get("generator_nograd_static_context_cache", {})
            )
            static_context_caches.append(static_context_cache)
            context_prefix = f"rank{rank} Generator no-grad static-context cache"
            if expected_policy["generator_nograd_static_context_cache"]:
                if not static_context_cache.get("enabled"):
                    errors.append(f"{context_prefix} was not enabled")
                if static_context_cache.get("scope") != (
                    "GENERATOR_NOGRAD_ONE_ROLLOUT_ONLY"
                ):
                    errors.append(f"{context_prefix} scope changed")
                for key in (
                    "changes_model_call_count",
                    "changes_dit_attention_ffn_calls",
                    "changes_fsdp_collectives",
                    "cross_rollout_reuse",
                ):
                    if static_context_cache.get(key) is not False:
                        errors.append(f"{context_prefix} {key} changed")
                if static_context_cache.get("current_latent_patch_recomputed") is not True:
                    errors.append(f"{context_prefix} cached current-latent work")
                for key in (
                    "grad_enabled_call_count",
                    "source_encoder_identity_mismatch_count",
                    "static_input_identity_mismatch_count",
                    "cached_tensor_mutation_count",
                    "unsupported_layout_count",
                ):
                    if int(static_context_cache.get(key, -1)) != 0:
                        errors.append(
                            f"{context_prefix} {key}="
                            f"{static_context_cache.get(key)}"
                        )
            elif static_context_cache.get("enabled"):
                errors.append(f"{context_prefix} enabled outside requested policy")
            if expected_policy["generator_nograd_lora_fused_epilogue"]:
                epilogue = value.get("build", {}).get("generator", {}).get(
                    "generator_nograd_lora_fused_epilogue", {}
                )
                if not epilogue.get("enabled"):
                    errors.append(f"rank{rank} fused LoRA epilogue was not enabled")
                if int(epilogue.get("model_call_count", 0)) <= 0:
                    errors.append(f"rank{rank} fused LoRA epilogue saw no model calls")
                if int(epilogue.get("fused_invocation_count", 0)) <= 0:
                    errors.append(f"rank{rank} fused LoRA epilogue fused no calls")
                if int(epilogue.get("reference_fallback_count", -1)) != 0:
                    errors.append(
                        f"rank{rank} fused LoRA epilogue fallback="
                        f"{epilogue.get('reference_fallback_count')}"
                    )
            if expected_policy["all_role_lora_scale1_add"]:
                for role in ("generator", "critic"):
                    epilogue = value.get("build", {}).get(role, {}).get(
                        "all_role_lora_scale1_add", {}
                    )
                    prefix = f"rank{rank} {role} all-role LoRA add"
                    if not epilogue.get("enabled"):
                        errors.append(f"{prefix} was not enabled")
                    if epilogue.get("scope") != "ALL_ROLE_SCALE1_MODEL_FORWARD":
                        errors.append(f"{prefix} scope changed")
                    if int(epilogue.get("model_call_count", 0)) <= 0:
                        errors.append(f"{prefix} saw no model calls")
                    if int(epilogue.get("fused_invocation_count", 0)) <= 0:
                        errors.append(f"{prefix} fused no calls")
                    if int(epilogue.get("fused_grad_invocation_count", 0)) <= 0:
                        errors.append(f"{prefix} fused no grad calls")
                    if int(epilogue.get("reference_fallback_count", -1)) != 0:
                        errors.append(
                            f"{prefix} fallback="
                            f"{epilogue.get('reference_fallback_count')}"
                        )
                    if epilogue.get("fused_update_dtype_counts", {}) != {
                        "torch.bfloat16": epilogue.get("fused_invocation_count")
                    }:
                        errors.append(f"{prefix} dtype inventory changed")
                    if epilogue.get("fused_scale_counts", {}) != {
                        "1.0": epilogue.get("fused_invocation_count")
                    }:
                        errors.append(f"{prefix} scale inventory changed")
            updates = value.get("updates", {})
            if updates != {
                "generator": 1,
                "critic": expected_critic_updates,
                "ema": expected_ema,
            }:
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
            expected_batches = (
                expected_logical_rollouts + args.rollout_batch_size - 1
            ) // args.rollout_batch_size
            if len(batches) != expected_batches:
                errors.append(
                    f"rank{rank} rollout batch count={len(batches)}, expected={expected_batches}"
                )
            if expected_policy["generator_nograd_cross_kv_cache"]:
                expected_total_calls = 0
                expected_total_reuses = 0
                for batch_index, batch in enumerate(batches):
                    cache = batch.get("cross_kv_cache", {})
                    call_count = int(batch.get("call_count", 0))
                    expected_total_calls += call_count * 40
                    expected_total_reuses += max(0, call_count - 1) * 40
                    cache_prefix = (
                        f"rank{rank} rollout batch{batch_index} "
                        "cross-attention KV"
                    )
                    if cache.get("status") != "PASS":
                        errors.append(
                            f"{cache_prefix} status={cache.get('status')}"
                        )
                    if cache.get("source_encoder_identity_contract") != (
                        "EXACT_STORAGE_POINTER_OFFSET_VERSION_DTYPE_SHAPE_"
                        "STRIDE_DEVICE"
                    ):
                        errors.append(f"{cache_prefix} source identity drifted")
                    expected_cache = {
                        "attention_count": 40,
                        "expected_model_calls": call_count,
                        "cross_projection_call_count": call_count * 40,
                        "self_projection_call_count": call_count * 40,
                        "query_projection_count": call_count * 40,
                        "cache_fill_count": 40,
                        "cache_reuse_count": max(0, call_count - 1) * 40,
                        "cache_miss_count": 0,
                        "encoder_signature_mismatch_count": 0,
                        "cached_tensor_mutation_count": 0,
                        "grad_enabled_call_count": 0,
                        "unexpected_cross_attention_count": 0,
                        "model_call_count": call_count,
                        "source_encoder_identity_mismatch_count": 0,
                        "missing_source_encoder_count": 0,
                        "cached_attention_count_before_clear": 40,
                        "cached_attention_count_after_clear": 0,
                    }
                    for key, expected in expected_cache.items():
                        if cache.get(key) != expected:
                            errors.append(
                                f"{cache_prefix} {key}={cache.get(key)!r}, "
                                f"expected={expected!r}"
                            )
                    if int(cache.get("cache_resident_bytes_before_clear", 0)) <= 0:
                        errors.append(f"{cache_prefix} cached no encoder K/V bytes")
                expected_install = {
                    "rollout_scope_count": expected_batches,
                    "cross_projection_call_count": expected_total_calls,
                    "self_projection_call_count": expected_total_calls,
                    "query_projection_count": expected_total_calls,
                    "cache_fill_count": expected_batches * 40,
                    "cache_reuse_count": expected_total_reuses,
                    "cache_miss_count": 0,
                    "encoder_signature_mismatch_count": 0,
                    "cached_tensor_mutation_count": 0,
                    "grad_enabled_call_count": 0,
                    "unexpected_cross_attention_count": 0,
                    "model_call_count": expected_total_calls // 40,
                    "source_encoder_identity_mismatch_count": 0,
                    "missing_source_encoder_count": 0,
                }
                for key, expected in expected_install.items():
                    if cross_kv_cache.get(key) != expected:
                        errors.append(
                            f"{install_prefix} {key}="
                            f"{cross_kv_cache.get(key)!r}, "
                            f"expected={expected!r}"
                        )
                if int(cross_kv_cache.get("cache_resident_bytes_sum", 0)) <= 0:
                    errors.append(
                        f"{install_prefix} cached no encoder K/V bytes"
                    )
            if expected_policy["generator_nograd_static_context_cache"]:
                expected_calls = 0
                expected_reuses = 0
                for batch_index, batch in enumerate(batches):
                    cache = batch.get("static_context_cache", {})
                    call_count = int(batch.get("call_count", 0))
                    current_layouts = sum(
                        int(count) > 0
                        for count in batch.get("stage_call_count", {}).values()
                    )
                    expected_calls += call_count
                    expected_reuses += max(0, call_count - 1)
                    cache_prefix = (
                        f"rank{rank} rollout batch{batch_index} static context"
                    )
                    if cache.get("status") != "PASS":
                        errors.append(
                            f"{cache_prefix} status={cache.get('status')}"
                        )
                    if cache.get("source_identity_contract") != (
                        "EXACT_STORAGE_POINTER_OFFSET_VERSION_DTYPE_SHAPE_"
                        "STRIDE_DEVICE"
                    ):
                        errors.append(f"{cache_prefix} source identity drifted")
                    if cache.get("current_structure_contract") != (
                        "POST_PATCH_B_T_H_W_TOKENS_DTYPE_DEVICE_LAYOUT"
                    ):
                        errors.append(
                            f"{cache_prefix} current structure contract drifted"
                        )
                    expected_cache = {
                        "expected_model_calls": call_count,
                        "expected_current_layouts": current_layouts,
                        "model_call_count": call_count,
                        "process_call_count": call_count,
                        "process_fill_count": 1,
                        "process_reuse_count": max(0, call_count - 1),
                        "text_projection_call_count": call_count,
                        "text_fill_count": 1,
                        "text_reuse_count": max(0, call_count - 1),
                        "current_layout_count": current_layouts,
                        "current_rope_fill_count": current_layouts,
                        "current_rope_reuse_count": max(
                            0, call_count - current_layouts
                        ),
                        "grad_enabled_call_count": 0,
                        "source_encoder_identity_mismatch_count": 0,
                        "static_input_identity_mismatch_count": 0,
                        "cached_tensor_mutation_count": 0,
                        "unsupported_layout_count": 0,
                        "cache_entries_before_clear": 3 + current_layouts,
                        "cache_entries_after_clear": 0,
                    }
                    for key, expected in expected_cache.items():
                        if cache.get(key) != expected:
                            errors.append(
                                f"{cache_prefix} {key}={cache.get(key)!r}, "
                                f"expected={expected!r}"
                            )
                    if int(cache.get("cache_resident_bytes_before_clear", 0)) <= 0:
                        errors.append(f"{cache_prefix} cached no static bytes")
                expected_install = {
                    "rollout_scope_count": expected_batches,
                    "model_call_count": expected_calls,
                    "process_call_count": expected_calls,
                    "process_fill_count": expected_batches,
                    "process_reuse_count": expected_reuses,
                    "text_projection_call_count": expected_calls,
                    "text_fill_count": expected_batches,
                    "text_reuse_count": expected_reuses,
                    "current_rope_fill_count": sum(
                        sum(
                            int(count) > 0
                            for count in batch.get("stage_call_count", {}).values()
                        )
                        for batch in batches
                    ),
                    "current_rope_reuse_count": sum(
                        max(
                            0,
                            int(batch.get("call_count", 0))
                            - sum(
                                int(count) > 0
                                for count in batch.get(
                                    "stage_call_count", {}
                                ).values()
                            ),
                        )
                        for batch in batches
                    ),
                    "grad_enabled_call_count": 0,
                    "source_encoder_identity_mismatch_count": 0,
                    "static_input_identity_mismatch_count": 0,
                    "cached_tensor_mutation_count": 0,
                    "unsupported_layout_count": 0,
                }
                for key, expected in expected_install.items():
                    if static_context_cache.get(key) != expected:
                        errors.append(
                            f"{context_prefix} {key}="
                            f"{static_context_cache.get(key)!r}, "
                            f"expected={expected!r}"
                        )
                if int(static_context_cache.get("cache_resident_bytes_sum", 0)) <= 0:
                    errors.append(f"{context_prefix} cached no static bytes")
            if args.mode == "correctness" and (
                args.rollout_batch_size > 1
                or expected_policy["generator_nograd_cross_kv_cache"]
                or expected_policy["generator_nograd_static_context_cache"]
            ):
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
            optimizer_warmups.append(None)
            residual_fusions.append(None)
            modulated_norm_fusions.append(None)
            cross_kv_caches.append(None)
            persistent_unshard_receipts.append(None)

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
        "critic_only_step": args.critic_only_step,
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
        "rank_optimizer_dispatch_warmup": optimizer_warmups,
        "rank_generator_nograd_residual_fusion": residual_fusions,
        "rank_exact_modulated_norm_epilogue": modulated_norm_fusions,
        "rank_generator_nograd_cross_kv_cache": cross_kv_caches,
        "rank_generator_nograd_static_context_cache": static_context_caches,
        "rank_generator_nograd_persistent_unshard": (
            persistent_unshard_receipts
        ),
        "rank_generator_free_memory": generator_free_memory_receipts,
        "rank_attention_backends": attention_backends,
        "rank_modulation_chunks": modulation_chunks,
        "rank_checkpoint_policies": checkpoint_policies,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
