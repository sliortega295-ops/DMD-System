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
        "--checkpoint-bypass-stride",
        type=int,
        choices=(0, 2, 4, 10),
        required=True,
    )
    parser.add_argument(
        "--checkpoint-bypass-role",
        choices=("all", "generator", "critic"),
        required=True,
    )
    parser.add_argument(
        "--selective-flash-checkpoint-stride",
        type=int,
        choices=(0, 2, 4, 10),
        required=True,
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
        "checkpoint_bypass_stride": args.checkpoint_bypass_stride,
        "checkpoint_bypass_role": args.checkpoint_bypass_role,
        "selective_flash_checkpoint_stride": (
            args.selective_flash_checkpoint_stride
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
    checkpoint_policies: list[dict[str, Any] | None] = []

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
            role_checkpoint_policies: dict[str, Any] = {}
            for role in ("generator", "critic"):
                checkpoint_policy = (
                    value.get("build", {}).get(role, {}).get("checkpoint_policy", {})
                )
                role_checkpoint_policies[role] = checkpoint_policy
                prefix = f"rank{rank} {role} checkpoint policy"
                requested_stride = expected_policy["checkpoint_bypass_stride"]
                requested_role = expected_policy["checkpoint_bypass_role"]
                expected_stride = (
                    requested_stride if requested_role in {"all", role} else 0
                )
                expected_indices = (
                    list(range(0, 40, expected_stride)) if expected_stride else []
                )
                if checkpoint_policy.get("enabled") is not bool(expected_stride):
                    errors.append(f"{prefix} enabled state changed")
                if checkpoint_policy.get("block_count") != 40:
                    errors.append(f"{prefix} block count changed")
                if checkpoint_policy.get("bypass_stride") != expected_stride:
                    errors.append(f"{prefix} stride changed")
                if checkpoint_policy.get("bypass_block_indices") != expected_indices:
                    errors.append(f"{prefix} block selection changed")
                if expected_stride:
                    counts = checkpoint_policy.get("block_call_counts", {})
                    if set(counts) != {str(index) for index in range(40)}:
                        errors.append(f"{prefix} lacks the 40-block runtime census")
                    else:
                        per_block = [int(counts[str(index)]) for index in range(40)]
                        if not per_block or min(per_block) <= 0:
                            errors.append(f"{prefix} was not exercised")
                        if len(set(per_block)) != 1:
                            errors.append(f"{prefix} per-block call counts drifted")
                        invocations = per_block[0] if per_block else 0
                        direct = int(checkpoint_policy.get("direct_forward_count", -1))
                        checkpointed = int(
                            checkpoint_policy.get("checkpoint_forward_count", -1)
                        )
                        if direct != invocations * len(expected_indices):
                            errors.append(f"{prefix} direct-forward census changed")
                        if checkpointed != invocations * (40 - len(expected_indices)):
                            errors.append(f"{prefix} checkpoint census changed")
                    auxiliary = int(
                        checkpoint_policy.get("auxiliary_checkpoint_forward_count", -1)
                    )
                    auxiliary_counts = checkpoint_policy.get(
                        "auxiliary_checkpoint_module_counts", {}
                    )
                    if auxiliary < 0 or auxiliary != sum(
                        int(count) for count in auxiliary_counts.values()
                    ):
                        errors.append(f"{prefix} auxiliary checkpoint census changed")
                elif int(checkpoint_policy.get("direct_forward_count", -1)) != 0:
                    errors.append(f"{prefix} bypassed blocks outside requested policy")
                expected_flash_stride = (
                    expected_policy["selective_flash_checkpoint_stride"]
                    if role == "critic"
                    else 0
                )
                expected_flash_indices = (
                    list(range(0, 40, expected_flash_stride))
                    if expected_flash_stride
                    else []
                )
                if checkpoint_policy.get("selective_flash_enabled") is not bool(
                    expected_flash_stride
                ):
                    errors.append(f"{prefix} selective Flash enabled state changed")
                if checkpoint_policy.get("selective_flash_stride") != expected_flash_stride:
                    errors.append(f"{prefix} selective Flash stride changed")
                if (
                    checkpoint_policy.get("selective_flash_block_indices")
                    != expected_flash_indices
                ):
                    errors.append(f"{prefix} selective Flash block selection changed")
                if expected_flash_stride:
                    if checkpoint_policy.get("selective_flash_op") != (
                        "_flash_attn2_cuda_a511a4c."
                        "_flash_attn_varlen_forward.default"
                    ):
                        errors.append(f"{prefix} selective Flash op identity changed")
                    counts = checkpoint_policy.get("block_call_counts", {})
                    if set(counts) != {str(index) for index in range(40)}:
                        errors.append(f"{prefix} lacks selective Flash block census")
                    else:
                        per_block = [int(counts[str(index)]) for index in range(40)]
                        if not per_block or min(per_block) <= 0:
                            errors.append(f"{prefix} selective Flash was not exercised")
                        if len(set(per_block)) != 1:
                            errors.append(
                                f"{prefix} selective Flash per-block calls drifted"
                            )
                        invocations = per_block[0] if per_block else 0
                        expected_contexts = invocations * len(expected_flash_indices)
                        contexts = int(
                            checkpoint_policy.get(
                                "selective_flash_context_count", -1
                            )
                        )
                        original_saves = int(
                            checkpoint_policy.get(
                                "selective_flash_original_save_count", -1
                            )
                        )
                        recompute_reuses = int(
                            checkpoint_policy.get(
                                "selective_flash_recompute_reuse_count", -1
                            )
                        )
                        if contexts != expected_contexts:
                            errors.append(
                                f"{prefix} selective Flash context census changed"
                            )
                        # The pinned Critic block performs exactly self- and
                        # cross-attention through this varlen FA2 op.
                        if original_saves != expected_contexts * 2:
                            errors.append(
                                f"{prefix} selective Flash original-save census changed"
                            )
                        if recompute_reuses != original_saves:
                            errors.append(
                                f"{prefix} selective Flash replay census changed"
                            )
                        if int(
                            checkpoint_policy.get("direct_forward_count", -1)
                        ) != 0:
                            errors.append(
                                f"{prefix} selective Flash unexpectedly bypassed blocks"
                            )
                        if int(
                            checkpoint_policy.get("checkpoint_forward_count", -1)
                        ) != invocations * 40:
                            errors.append(
                                f"{prefix} selective Flash changed checkpoint calls"
                            )
                elif any(
                    int(checkpoint_policy.get(key, 0)) != 0
                    for key in (
                        "selective_flash_context_count",
                        "selective_flash_original_save_count",
                        "selective_flash_recompute_reuse_count",
                    )
                ):
                    errors.append(f"{prefix} selective Flash ran outside policy")
            checkpoint_policies.append(role_checkpoint_policies)
            prefix = f"rank{rank} exact modulated-norm epilogue"
            if expected_policy["exact_modulated_norm_epilogue"]:
                if not modulated_norm_fusion.get("enabled"):
                    errors.append(f"{prefix} was not enabled")
                if modulated_norm_fusion.get("scope") != (
                    "ALL_ROLE_B1_MODULATED_NORM_FORWARD_EPILOGUE"
                ):
                    errors.append(f"{prefix} scope changed")
                if modulated_norm_fusion.get("semantic_contract") != (
                    "PYTORCH_FP32_LAYERNORM_UNCHANGED;ADD_RN_F32_ONE_PLUS_SCALE;"
                    "MUL_RN_F32;ADD_RN_F32_SHIFT;BF16_RN_STORE;"
                    "SOURCE_BACKWARD_UNCHANGED"
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
            elif modulated_norm_fusion.get("enabled"):
                errors.append(f"{prefix} enabled outside requested policy")
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
            optimizer_warmups.append(None)
            residual_fusions.append(None)
            modulated_norm_fusions.append(None)
            checkpoint_policies.append(None)

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
        "rank_checkpoint_policies": checkpoint_policies,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
