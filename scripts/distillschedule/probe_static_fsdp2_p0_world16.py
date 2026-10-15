#!/usr/bin/env python3
"""P0 static-FSDP2/HSDP Full5 execution for Helios.

This worker is stacked on the validated PR #13 baseline. It preserves the same
logical updates while changing only physical execution:

1. after the Generator commit, produce all five Critic inputs before C updates;
2. batch those no-grad rollouts in B1/B2/B5 groups;
3. optionally pack Real-Score cond/uncond into one B2 call (No-GAN);
4. optionally replace world16 FSDP with node-local FSDP8 x replica2 HSDP.
"""
from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import os
import socket
import sys
import time
import traceback
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

PINNED_FLASH_ATTN2_EXTENSION_SHA256 = (
    "c2acd487c5d1d77b7f6721eabc0fee4f8c5c8f9e682aadf43bfa99b394eecba7"
)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_BASE_PATH = ROOT / "scripts/distillschedule/probe_static_fsdp2_full5_world16.py"
_SPEC = importlib.util.spec_from_file_location("distillschedule_static_fsdp2_baseline", _BASE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load baseline worker from {_BASE_PATH}")
base = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = base
_SPEC.loader.exec_module(base)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _attention_backend_receipt(expected: str) -> dict[str, Any]:
    """Fail closed on the attention implementation imported by this worker."""

    import_binding = None
    if os.environ.get("DISTILLSCHEDULE_HARDWARE_PROFILE") == "h100-8-b16":
        from distillschedule.runtime.helios_attention_import import import_registered_attention_backend

        attention_dispatch, import_binding = import_registered_attention_backend(expected)
    else:
        from helios.modules.helios_kernels import attention_dispatch

    local_kernels = os.environ.get("LOCAL_KERNELS", "")
    function = attention_dispatch.flash_attn_func
    function_module = getattr(function, "__module__", None)
    receipt: dict[str, Any] = {
        "status": "PASS",
        "requested": expected,
        "local_kernels": local_kernels,
        "flash_attn_func_available": callable(function),
        "flash_attn_func_module": function_module,
        "extension_path": None,
        "extension_sha256": None,
    }
    if import_binding is not None:
        receipt["registered_import"] = import_binding
    if expected == "sdpa":
        if local_kernels:
            raise AssertionError("SDPA control inherited LOCAL_KERNELS")
        if function is not None:
            raise AssertionError(
                "SDPA control unexpectedly imported an external FlashAttention function"
            )
        receipt["runtime"] = "sdpa"
        return receipt

    if expected != "local-fa2":
        raise AssertionError(f"unsupported attention backend: {expected}")
    prefix = "kernels-community/flash-attn2="
    if not local_kernels.startswith(prefix):
        raise AssertionError("local-fa2 requires the pinned LOCAL_KERNELS mapping")
    kernel_root = Path(local_kernels[len(prefix) :])
    extension = kernel_root / "_flash_attn2_cuda_a511a4c.abi3.so"
    if not extension.is_file():
        raise AssertionError(f"local FlashAttention extension is missing: {extension}")
    extension_sha256 = _file_sha256(extension)
    if extension_sha256 != PINNED_FLASH_ATTN2_EXTENSION_SHA256:
        raise AssertionError(
            "local FlashAttention extension identity changed: "
            f"{extension_sha256}"
        )
    if not callable(function) or "flash_attn2" not in str(function_module):
        raise AssertionError(
            "local-fa2 did not import the expected FlashAttention-2 callable"
        )
    receipt.update(
        runtime="local-fa2",
        extension_path=str(extension),
        extension_sha256=extension_sha256,
    )
    return receipt

from distillschedule.profiling.static_profile_diagnostics import StaticProfileDiagnostics
from distillschedule.profiling.fsdp2_operation_source import (
    DEFAULT_TARGET_BYTES,
    OperationSourceCapture,
)
from distillschedule.runtime.fsdp2_p0 import (
    StaticFSDP2P0Policy,
    apply_blockwise_fsdp2_policy,
    chunked,
    estimate_rollout_collective_groups,
    score_forward_real_pair,
)
from distillschedule.runtime.hardware_profile import (
    HardwareProfile,
    MicrobatchCommitEngine,
    hardware_profile,
    runtime_physical_world_size,
)
from distillschedule.runtime.fsdp2_nograd_persistence import (
    generator_nograd_persistent_unshard_scope,
)
from distillschedule.runtime.fsdp2_static import FSDP2ShardEMA, build_fsdp2_engine, configure_fsdp2_checkpointing, contract_for_variant
from distillschedule.runtime.helios_resident_zero1 import (
    ResidentZero1ShardEMA, build_resident_zero1_engine,
)
from distillschedule.runtime.helios_official_role_swap import OfficialRoleSwap
from distillschedule.runtime.helios_cross_kv_cache import (
    install_generator_nograd_cross_kv_cache,
)
from distillschedule.runtime.helios_critic_scalar_logs import (
    install_critic_scalar_log_elision,
)
from distillschedule.runtime.helios_generator_free_memory import (
    prepare_generator_free_memory_receipt,
)
from distillschedule.runtime.helios_history_current_fusion import (
    install_exact_history_current_fusion,
)
from distillschedule.runtime.helios_rollout_context_cache import (
    install_generator_nograd_static_context_cache,
)
from distillschedule.runtime.helios_lora_epilogue import (
    install_generator_nograd_lora_fused_epilogue,
)
from distillschedule.runtime.helios_modulation_chunks import (
    configure_helios_modulation_chunk_tokens,
)
from distillschedule.runtime.helios_modulated_norm_fusion import (
    install_exact_modulated_norm_epilogue,
    warmup_exact_modulated_norm_epilogue,
)
from distillschedule.runtime.helios_residual_fusion import (
    generator_nograd_residual_fusion_scope,
    install_generator_nograd_residual_fusion,
    warmup_generator_nograd_residual_fusion,
)
from distillschedule.runtime.nsys_range import StaticFSDP2Full5NsightRange
from distillschedule.runtime.p0_parity import require_tensor_parity
from runtime.critic_pool_checkpoint import LogicalCheckpointPackage, load_logical_master_into_trainables
from runtime.critic_pool_helios import (
    build_helios_pool_model,
    prepare_student_batch,
    run_generator_trace_rollout,
    student_forward_loss,
)
from runtime.dmd_boundary import detached_dmd_surrogate_loss
from runtime.gan_rolephase_official import critic_official_kwargs, load_official_config
from runtime.gan_score_packing import (
    PACKED_SCORE_MEAN_REL_TOLERANCE,
    packed_gan_real_score_pair,
)
from runtime.gst_pool_contract import NegativePromptAsset
from runtime.gst_pool_helios import construct_dmd_gradient, prepare_dmd_score_batch, run_generator_update_trace_rollout, score_forward
from runtime.official_baseline import suspend_transformers_zero3_init, trim_host_allocator
from runtime.static_fsdp2_p0_helios import run_generator_trace_rollout_batch
from runtime.static_world16_trace import StaticWorld16Trace


WORLD_SIZE = runtime_physical_world_size()
TIMING_OMITTED = base.TIMING_OMITTED


def _build_static_model(
    *,
    args: Any,
    role: str,
    package: LogicalCheckpointPackage,
    device: torch.device,
    policy: StaticFSDP2P0Policy,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    if role not in {"generator", "critic"}:
        raise ValueError(role)
    root = args.generator_root if role == "generator" else args.score_root
    precast = args.precast_generator_root if role == "generator" else args.precast_score_root
    subfolder = "transformer_ode" if role == "generator" else "transformer"
    with suspend_transformers_zero3_init():
        model, metadata = build_helios_pool_model(
            role=role,
            model_root=root,
            subfolder=subfolder,
            precast_model_root=precast,
            enable_gan=(args.variant == "gan" and role == "critic"),
            chunked_residual_storage=(
                args.critic_residual_storage
                if role == "critic"
                else args.generator_residual_storage
            ),
        )
        logical_restore = load_logical_master_into_trainables(
            model,
            package,
            component=role,
            verify_payload_sha256=(dist.get_rank() == 0),
        )
    if args.generator_nograd_lora_fused_epilogue and args.all_role_lora_scale1_add:
        raise ValueError("LoRA epilogue candidates are mutually exclusive")
    modulation_chunks = configure_helios_modulation_chunk_tokens(
        model,
        chunk_tokens=args.modulation_chunk_tokens,
        role=role,
    )
    lora_fused_epilogue = install_generator_nograd_lora_fused_epilogue(
        model,
        enabled=bool(
            args.generator_nograd_lora_fused_epilogue
            or args.all_role_lora_scale1_add
        ),
        role=role,
        all_role_scale1=bool(args.all_role_lora_scale1_add),
    )
    residual_fusion = (
        install_generator_nograd_residual_fusion(
            enabled=bool(args.generator_nograd_residual_fusion),
            exact_grad_enabled=bool(args.exact_grad_residual_fusion),
            role=role,
        )
        if role == "generator"
        else {
            "requested": bool(args.generator_nograd_residual_fusion),
            "exact_grad_requested": bool(args.exact_grad_residual_fusion),
            "enabled": False,
            "role": role,
            "disabled_reason": "GENERATOR_INSTALL_ONLY",
        }
    )
    modulated_norm_fusion = (
        install_exact_modulated_norm_epilogue(
            enabled=bool(args.exact_modulated_norm_epilogue),
            role=role,
            fuse_materialization=bool(
                args.exact_modulated_norm_materialization
            ),
            fuse_backward=bool(args.exact_modulated_norm_backward),
            fuse_layernorm=bool(args.exact_modulated_norm_layernorm),
            fuse_backward_kernel=bool(
                args.exact_modulated_norm_fused_backward
            ),
        )
        if role == "generator"
        else {
            "requested": bool(args.exact_modulated_norm_epilogue),
            "enabled": False,
            "role": role,
            "disabled_reason": "INSTALL_ONCE_FROM_GENERATOR_BUILD",
        }
    )
    history_current_fusion = (
        install_exact_history_current_fusion(
            enabled=bool(args.exact_history_current_fusion),
            role=role,
        )
        if role == "generator"
        else {
            "requested": bool(args.exact_history_current_fusion),
            "enabled": False,
            "role": role,
            "disabled_reason": "INSTALL_ONCE_FROM_GENERATOR_BUILD",
        }
    )
    cross_kv_cache = install_generator_nograd_cross_kv_cache(
        enabled=bool(args.generator_nograd_cross_kv_cache),
        role=role,
    )
    static_context_cache = install_generator_nograd_static_context_cache(
        model,
        enabled=bool(args.generator_nograd_static_context_cache),
        role=role,
    )
    checkpoint_policy = configure_fsdp2_checkpointing(
        model,
        operation_source=bool(args.operation_source),
        # Saving attention results is intentionally Critic-only.  Generator
        # DMD holds several differentiable graphs concurrently and would
        # multiply the activation-residency budget.
        selective_flash_checkpoint_stride=(
            int(args.selective_flash_checkpoint_stride)
            if role == "critic"
            else 0
        ),
    )
    # A node-HSDP shard owns twice as many parameters as a world16 shard.  If
    # the already-sharded Generator remains resident while the Critic is first
    # materialized in full on one device, the cold construction peak exceeds a
    # 40-GB A100 before FSDP has a chance to shard the Critic.  Composable
    # FSDP2 moves each managed group to the mesh device inside ``fully_shard``;
    # applying the existing bottom-up block policy directly to the CPU model
    # therefore materializes at most one block before immediately sharding it.
    # This is construction-only and does not add parameter offload or change
    # the persistent HSDP execution graph.
    fsdp_first_construction = policy.mesh_policy == "node-hsdp"
    if not fsdp_first_construction:
        model.to(device)
    resident = args.parameter_placement in ("gpu-resident-zero1", "official-role-swap-zero1")
    sharding = (
        {"enabled": False, "reason": "FULL_PARAMETER_GPU_RESIDENT_REFERENCE"}
        if resident else apply_blockwise_fsdp2_policy(
            model, mesh_policy=policy.mesh_policy, world_size=WORLD_SIZE,
        )
    )
    if role == "generator":
        warmup_generator_nograd_residual_fusion(
            residual_fusion,
            device=device,
            chunk_tokens=args.modulation_chunk_tokens,
        )
        warmup_exact_modulated_norm_epilogue(
            modulated_norm_fusion,
            device=device,
            chunk_tokens=args.modulation_chunk_tokens,
        )
    torch.cuda.synchronize(device)
    gc.collect()
    host_trimmed = trim_host_allocator()
    torch.cuda.empty_cache()
    metadata_offload = args.parameter_placement == "official-role-swap-zero1"
    return model, {
        "metadata": metadata,
        "logical_restore": logical_restore,
        "generator_nograd_lora_fused_epilogue": (
            lora_fused_epilogue
            if args.generator_nograd_lora_fused_epilogue
            else {"enabled": False, "disabled_reason": "NOT_REQUESTED"}
        ),
        "all_role_lora_scale1_add": (
            lora_fused_epilogue
            if args.all_role_lora_scale1_add
            else {"enabled": False, "disabled_reason": "NOT_REQUESTED"}
        ),
        "generator_nograd_residual_fusion": residual_fusion,
        "exact_modulated_norm_epilogue": modulated_norm_fusion,
        "exact_history_current_fusion": history_current_fusion,
        "generator_nograd_cross_kv_cache": cross_kv_cache,
        "generator_nograd_static_context_cache": static_context_cache,
        "modulation_chunks": modulation_chunks,
        "checkpoint_policy": checkpoint_policy,
        "fsdp2": sharding,
        **({"resident": {
            "backend": "FULL_PARAMETER_GPU_RESIDENT_ZERO1", "world_size": 8,
            "logical_global_batch": 16, "physical_b1_rows_per_rank": 2,
            "full_parameters_on_gpu": not metadata_offload, "parameter_offload": metadata_offload,
            "activation_offload": False, "optimizer_cpu_offload": False,
            "parameter_streaming": False, "model_swap": metadata_offload,
            "model_compute_dtype": "torch.bfloat16", "gradient_communication_dtype": "torch.float32",
        }} if resident else {}),
        "construction_order": (
            "CPU_LOGICAL_RESTORE_THEN_FULL_GPU_MODEL" if resident else
            "CPU_RESTORE_THEN_BLOCKWISE_FSDP_DEVICE_SHARD"
            if fsdp_first_construction
            else "FULL_DEVICE_MATERIALIZE_THEN_BLOCKWISE_FSDP_SHARD"
        ),
        "construction_only_compatibility": fsdp_first_construction,
        "host_allocator_trimmed_after_shard": host_trimmed,
        "memory_after_shard": base._memory(device),
    }


def _build_engines(
    *,
    args: Any,
    package: LogicalCheckpointPackage,
    device: torch.device,
    policy: StaticFSDP2P0Policy,
) -> tuple[Any, Any, dict[str, Any]]:
    if args.parameter_placement == "official-role-swap-zero1":
        swap = OfficialRoleSwap()
        engines, records = {}, {}
        for role, lr in (("generator", 2e-6), ("critic", 4e-7)):
            model, build = _build_static_model(args=args, role=role, package=package,
                                               device=device, policy=policy)
            engine, restore = build_resident_zero1_engine(module=model, package=package,
                component=role, lr=lr, betas=(0.0, 0.999), eps=1e-8, weight_decay=1e-3)
            swap.register(engine, role=role)
            swap.move_to_cpu(engine)
            engines[role] = engine
            records[role] = {**build, "optimizer_restore": restore,
                             "construction_memory_after_role_offload": base._memory(device)}
            dist.barrier()
        args._official_role_swap = swap
        return engines["generator"], engines["critic"], {
            **records, "static_residency": {
                "both_backbones_sharded_for_full5": False,
                "both_backbones_full_parameter_gpu_resident": False,
                "optimizer_state_placement": "GPU_ZERO1_PARTITIONS",
                "mesh_policy": policy.mesh_policy, "parameter_offload": True,
                "frozen_parameter_streaming": False, "role_transition": True,
                "process_group_rebuild_in_full5": False,
                "construction_order": "G_MODEL_ENGINE_RESTORE_OFFLOAD_THEN_SCORE_MODEL_ENGINE_RESTORE_OFFLOAD"},
            "official_role_swap": swap.receipt()}
    generator, generator_build = _build_static_model(
        args=args,
        role="generator",
        package=package,
        device=device,
        policy=policy,
    )
    dist.barrier()
    critic, critic_build = _build_static_model(
        args=args,
        role="critic",
        package=package,
        device=device,
        policy=policy,
    )
    dist.barrier()
    builder = (build_resident_zero1_engine
               if args.parameter_placement == "gpu-resident-zero1" else build_fsdp2_engine)
    generator_engine, generator_restore = builder(
        module=generator,
        package=package,
        component="generator",
        lr=2e-6,
        betas=(0.0, 0.999),
        eps=1e-8,
        weight_decay=1e-3,
        optimizer_dispatch_warmup=policy.optimizer_dispatch_warmup,
    )
    critic_engine, critic_restore = builder(
        module=critic,
        package=package,
        component="critic",
        lr=4e-7,
        betas=(0.0, 0.999),
        eps=1e-8,
        weight_decay=1e-3,
        optimizer_dispatch_warmup=policy.optimizer_dispatch_warmup,
    )
    return generator_engine, critic_engine, {
        "generator": {**generator_build, "optimizer_restore": generator_restore},
        "critic": {**critic_build, "optimizer_restore": critic_restore},
        "static_residency": {
            "both_backbones_sharded_for_full5": args.parameter_placement == "fsdp2",
            **({"both_backbones_full_parameter_gpu_resident": True,
                "optimizer_state_placement": "GPU_ZERO1_PARTITIONS"}
               if args.parameter_placement == "gpu-resident-zero1" else {}),
            "mesh_policy": policy.mesh_policy,
            "parameter_offload": False,
            "frozen_parameter_streaming": False,
            "role_transition": False,
            "process_group_rebuild_in_full5": False,
        },
    }


def _score_pack_error(reference: torch.Tensor, packed: torch.Tensor) -> dict[str, float]:
    difference = (reference.float() - packed.float()).abs()
    denominator = reference.float().abs().mean().clamp_min(1e-6)
    return {
        "max_abs": float(difference.max().detach().cpu()),
        "mean_abs": float(difference.mean().detach().cpu()),
        "mean_relative": float((difference.mean() / denominator).detach().cpu()),
    }


def _run_nogan_generator_update_p0(
    *,
    rank: int,
    device: torch.device,
    generator_engine: Any,
    critic_engine: Any,
    trace: StaticWorld16Trace,
    negative: torch.Tensor,
    collect_diagnostics: bool,
    diagnostics: StaticProfileDiagnostics,
    policy: StaticFSDP2P0Policy,
    score_pack_parity_check: bool,
    clean_runtime: bool = False,
    runtime_cache: dict[str, Any] | None = None,
    deferred_finite_checks: list[torch.Tensor] | None = None,
    collect_scalars: bool = False,
    expect_optimizer_commit: bool = True,
    role_swap: Any = None,
) -> dict[str, Any]:
    if role_swap is not None:
        role_swap.activate("generator", device)
    with diagnostics.scope("INPUT_H2D/generator_entry"):
        entry = trace.load_generator_entry(rank=rank, device=device)
    diagnostics.add_tensor_tree("generator_entry", entry, direction="H2D")
    before_step = int(generator_engine.global_steps)
    before = (
        TIMING_OMITTED
        if not collect_diagnostics
        else base._profile_fingerprint(
            generator_engine.module, diagnostics, "generator_before"
        )
    )
    with diagnostics.scope("FSDP_PRE_FORWARD_ALL_GATHER/generator"):
        x0, gradient_mask, rollout = run_generator_update_trace_rollout(
            engine=generator_engine,
            entry=entry,
            device=device,
            detailed_profile=False,
            collect_diagnostics=collect_diagnostics,
            empty_cache_before_rollout=not policy.elide_explicit_rollout_sync,
            synchronize_after_rollout=not policy.elide_explicit_rollout_sync,
            clean_runtime=clean_runtime,
            runtime_cache=runtime_cache,
        )
    batch = prepare_dmd_score_batch(
        generated_x0=x0,
        entry=entry,
        device=device,
        runtime_cache=runtime_cache,
    )
    if role_swap is not None:
        role_swap.activate("critic", device)
    fake = score_forward(
        critic_engine.module,
        batch,
        condition=batch["encoder_hidden_states"],
        adapter_enabled=True,
        validate_finite=deferred_finite_checks is None,
    )
    parity: dict[str, Any] = {"enabled": False}
    if policy.pack_real_score_pair:
        real_cond, real_uncond = score_forward_real_pair(
            critic_engine.module,
            batch,
            conditional=batch["encoder_hidden_states"],
            unconditional=negative,
        )
        if score_pack_parity_check:
            reference_cond = score_forward(
                critic_engine.module,
                batch,
                condition=batch["encoder_hidden_states"],
                adapter_enabled=False,
                validate_finite=deferred_finite_checks is None,
            )
            reference_uncond = score_forward(
                critic_engine.module,
                batch,
                condition=negative,
                adapter_enabled=False,
                validate_finite=deferred_finite_checks is None,
            )
            cond_error = _score_pack_error(reference_cond, real_cond)
            uncond_error = _score_pack_error(reference_uncond, real_uncond)
            reference_gradient, _reference_auxiliary = construct_dmd_gradient(
                batch=batch,
                fake_flow_cond=fake,
                real_flow_cond=reference_cond,
                real_flow_uncond=reference_uncond,
            )
            packed_gradient, _packed_auxiliary = construct_dmd_gradient(
                batch=batch,
                fake_flow_cond=fake,
                real_flow_cond=real_cond,
                real_flow_uncond=real_uncond,
            )
            gradient_error = _score_pack_error(reference_gradient, packed_gradient)
            worst = max(
                cond_error["mean_relative"],
                uncond_error["mean_relative"],
                gradient_error["mean_relative"],
            )
            if worst > PACKED_SCORE_MEAN_REL_TOLERANCE:
                raise AssertionError(
                    "packed Real-Score pair exceeds mean-relative tolerance: "
                    f"{worst:.6g} > {PACKED_SCORE_MEAN_REL_TOLERANCE:.6g}"
                )
            parity = {
                "enabled": True,
                "status": "PASS",
                "conditional": cond_error,
                "unconditional": uncond_error,
                "dmd_gradient": gradient_error,
                "threshold_mean_relative": PACKED_SCORE_MEAN_REL_TOLERANCE,
            }
            del reference_gradient, _reference_auxiliary, packed_gradient, _packed_auxiliary
        physical_score_forwards = 2
    else:
        real_cond = score_forward(
            critic_engine.module,
            batch,
            condition=batch["encoder_hidden_states"],
            adapter_enabled=False,
            validate_finite=deferred_finite_checks is None,
        )
        real_uncond = score_forward(
            critic_engine.module,
            batch,
            condition=negative,
            adapter_enabled=False,
            validate_finite=deferred_finite_checks is None,
        )
        physical_score_forwards = 3
    gradient, auxiliary = construct_dmd_gradient(
        batch=batch,
        fake_flow_cond=fake,
        real_flow_cond=real_cond,
        real_flow_uncond=real_uncond,
        validate_finite=deferred_finite_checks is None,
    )
    loss = detached_dmd_surrogate_loss(x0, gradient, gradient_mask)
    if role_swap is not None:
        role_swap.activate("generator", device)
    if deferred_finite_checks is not None:
        deferred_finite_checks.append(torch.isfinite(loss.detach()))
    with diagnostics.scope("FSDP_PRE_BACKWARD_ALL_GATHER/generator"):
        generator_engine.backward(loss)
    with diagnostics.scope("OPTIMIZER_STEP/generator"):
        generator_engine.step()
    if int(generator_engine.global_steps) != before_step + int(expect_optimizer_commit):
        raise AssertionError("No-GAN P0 Generator microbatch commit drift")
    after = (
        TIMING_OMITTED
        if not collect_diagnostics
        else base._profile_fingerprint(
            generator_engine.module, diagnostics, "generator_after"
        )
    )
    if collect_diagnostics:
        if expect_optimizer_commit and before == after:
            raise AssertionError("No-GAN P0 Generator trainables did not update")
        if not expect_optimizer_commit and before != after:
            raise AssertionError("No-GAN P0 Generator changed before the logical commit")
    del entry, x0, gradient_mask, batch, fake, real_cond, real_uncond, gradient, auxiliary
    return {
        "logical_step": 10,
        "source_rank": rank,
        "physical_generator_step_after": int(generator_engine.global_steps),
        "optimizer_committed": bool(expect_optimizer_commit),
        "local_trainable_before": before,
        "local_trainable_after": after,
        "rollout": rollout,
        "surrogate_loss": (
            float(loss.detach().float().cpu())
            if collect_scalars
            else (
                diagnostics.diagnostic_scalar("generator_surrogate", loss)
                if collect_diagnostics
                else TIMING_OMITTED
            )
        ),
        "logical_score_requests": 3,
        "physical_score_forwards": physical_score_forwards,
        "real_score_pair_packed": bool(policy.pack_real_score_pair),
        "score_pack_parity": parity,
    }


def _rollout_pack_error(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    difference = (reference.float() - candidate.float()).abs()
    denominator = reference.float().abs().mean().clamp_min(1e-6)
    cosine = torch.nn.functional.cosine_similarity(
        reference.float().flatten().unsqueeze(0),
        candidate.float().flatten().unsqueeze(0),
        dim=1,
        eps=1e-8,
    )[0]
    return {
        "max_abs": float(difference.max().detach().cpu()),
        "mean_abs": float(difference.mean().detach().cpu()),
        "mean_relative": float((difference.mean() / denominator).detach().cpu()),
        "cosine": float(cosine.detach().cpu()),
    }


def _precompute_critic_rollouts(
    *,
    rank: int,
    steps: tuple[int, ...],
    device: torch.device,
    generator_engine: Any,
    trace: StaticWorld16Trace,
    policy: StaticFSDP2P0Policy,
    collect_diagnostics: bool,
    diagnostics: StaticProfileDiagnostics,
    rollout_parity_check: bool,
    clean_runtime: bool = False,
    runtime_cache: dict[str, Any] | None = None,
    entry_outputs: dict[int, Any] | None = None,
) -> tuple[dict[int, torch.Tensor], list[dict[str, Any]], dict[str, Any]]:
    outputs: dict[int, torch.Tensor] = {}
    records: list[dict[str, Any]] = []
    with generator_nograd_persistent_unshard_scope(
        generator_engine.module,
        enabled=policy.generator_nograd_persistent_unshard,
        flush_allocator_cache=(
            policy.generator_nograd_persistent_unshard_flush_allocator_cache
        ),
    ) as persistence:
        persistence["rollout_memory"] = []
        for batch_index, step_group in enumerate(
            chunked(steps, policy.rollout_batch_size)
        ):
            entries = []
            for step in step_group:
                with diagnostics.scope(f"INPUT_H2D/critic_entry_rollout/step{step}"):
                    entry = trace.load_critic_entry(rank=rank, step=step, device=device)
                diagnostics.add_tensor_tree(
                    f"critic_entry_rollout/step{step}", entry, direction="H2D"
                )
                entries.append(entry)
                if entry_outputs is not None:
                    entry_outputs[int(step)] = entry
            with diagnostics.scope(
                f"FSDP_BATCHED_GENERATOR_ROLLOUT/batch{batch_index}/B{len(entries)}"
            ):
                generated, record = run_generator_trace_rollout_batch(
                    model=generator_engine.module,
                    loaded_entries=entries,
                    device=device,
                    detailed_profile=False,
                    record_output_sha256=collect_diagnostics,
                    collect_diagnostics=collect_diagnostics,
                    diagnostics_scope=diagnostics.scope,
                    use_cross_kv_cache=policy.generator_nograd_cross_kv_cache,
                    use_static_context_cache=(
                        policy.generator_nograd_static_context_cache
                    ),
                    synchronize_after_rollout=not policy.elide_explicit_rollout_sync,
                    clean_runtime=clean_runtime,
                    runtime_cache=runtime_cache,
                )
            persistence["rollout_batch_count"] += 1
            persistence["rollout_memory"].append(
                {
                    "batch_index": batch_index,
                    "allocated_bytes": int(torch.cuda.memory_allocated(device)),
                    "reserved_bytes": int(torch.cuda.memory_reserved(device)),
                    "max_allocated_bytes": int(
                        torch.cuda.max_memory_allocated(device)
                    ),
                    "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                }
            )
            parity_rows: list[dict[str, Any]] = []
            needs_reference = rollout_parity_check and (
                len(entries) > 1
                or policy.generator_nograd_cross_kv_cache
                or policy.generator_nograd_static_context_cache
            )
            if needs_reference:
                for entry, candidate in zip(entries, generated, strict=True):
                    # Keep every already-validated C5 no-grad fusion active and
                    # disable only the new K/V cache in the reference arm.
                    with generator_nograd_residual_fusion_scope():
                        reference, _reference_record = run_generator_trace_rollout(
                            model=generator_engine.module,
                            loaded=entry,
                            device=device,
                            detailed_profile=False,
                            record_output_sha256=False,
                            empty_cache_before_rollout=False,
                            collect_diagnostics=False,
                        )
                    if (
                        policy.generator_nograd_cross_kv_cache
                        or policy.generator_nograd_static_context_cache
                    ):
                        error = require_tensor_parity(
                            reference,
                            candidate,
                            label=(
                                "Generator no-grad context-cache B1 step"
                                f"{entry.pool_entry.logical_step}"
                            ),
                        )
                    else:
                        error = _rollout_pack_error(reference, candidate)
                        if error["mean_relative"] > PACKED_SCORE_MEAN_REL_TOLERANCE:
                            raise AssertionError(
                                "batched Generator rollout exceeds mean-relative "
                                "tolerance: "
                                f"step={entry.pool_entry.logical_step} "
                                f"{error['mean_relative']:.6g} > "
                                f"{PACKED_SCORE_MEAN_REL_TOLERANCE:.6g}"
                            )
                    parity_rows.append(
                        {
                            "logical_step": int(entry.pool_entry.logical_step),
                            **error,
                        }
                    )
                    del reference, _reference_record
            for step, value in zip(step_group, generated, strict=True):
                outputs[int(step)] = value
            record["batch_index"] = batch_index
            record["batch_parity"] = {
                "enabled": bool(needs_reference),
                "status": "PASS" if parity_rows else "NOT_RUN",
                "reason": (
                    "CONTEXT_CACHE_B1_REFERENCE"
                    if (
                        policy.generator_nograd_cross_kv_cache
                        or policy.generator_nograd_static_context_cache
                    )
                    and len(entries) == 1
                    else "BATCHED_VS_B1_REFERENCE"
                ),
                "tolerance_mean_relative": PACKED_SCORE_MEAN_REL_TOLERANCE,
                "samples": parity_rows,
            }
            records.append(record)
            del entries, generated
    if set(outputs) != set(steps):
        raise AssertionError("P0 rollout precompute missed one or more logical steps")
    return outputs, records, persistence


def _run_nogan_critic_precomputed(
    *,
    rank: int,
    step: int,
    x0: torch.Tensor,
    device: torch.device,
    critic_engine: Any,
    trace: StaticWorld16Trace,
    collect_diagnostics: bool,
    diagnostics: StaticProfileDiagnostics,
    entry: Any | None = None,
    runtime_cache: dict[str, Any] | None = None,
    deferred_finite_checks: list[torch.Tensor] | None = None,
    collect_scalars: bool = False,
) -> dict[str, Any]:
    if entry is None:
        with diagnostics.scope(f"INPUT_H2D/critic_entry_update/step{step}"):
            entry = trace.load_critic_entry(rank=rank, step=step, device=device)
    diagnostics.add_tensor_tree(f"critic_entry_update/step{step}", entry, direction="H2D")
    batch = prepare_student_batch(
        generated_x0=x0,
        loaded=entry,
        device=device,
        runtime_cache=runtime_cache,
    )
    with diagnostics.scope("FSDP_PRE_FORWARD_ALL_GATHER/critic"):
        _prediction, loss = student_forward_loss(
            critic_engine,
            batch,
            validate_finite=deferred_finite_checks is None,
            deferred_finite_checks=deferred_finite_checks,
        )
    with diagnostics.scope("FSDP_PRE_BACKWARD_ALL_GATHER/critic"):
        critic_engine.backward(loss)
    with diagnostics.scope("OPTIMIZER_STEP/critic"):
        critic_engine.step()
    return {
        "logical_step": step,
        "source_rank": rank,
        "physical_critic_step_after": int(critic_engine.global_steps),
        "loss": (
            float(loss.detach().float().cpu())
            if collect_scalars
            else (
                diagnostics.diagnostic_scalar(f"critic_step{step}", loss)
                if collect_diagnostics
                else TIMING_OMITTED
            )
        ),
        "score_forwards": 1,
        "generator_rollout_precomputed": True,
    }


def _run_gan_critic_precomputed(
    *,
    rank: int,
    step: int,
    x0: torch.Tensor,
    device: torch.device,
    critic_engine: Any,
    trace: StaticWorld16Trace,
    cfg: Any,
    collect_diagnostics: bool,
    collect_official_scalar_logs: bool,
    diagnostics: StaticProfileDiagnostics,
    expect_optimizer_commit: bool = True,
) -> dict[str, Any]:
    from helios.utils.utils_helios_post import _critic_loss

    with diagnostics.scope(f"INPUT_H2D/critic_entry_update/step{step}"):
        entry = trace.load_critic_entry(rank=rank, step=step, device=device)
    diagnostics.add_tensor_tree(f"critic_entry_update/step{step}", entry, direction="H2D")
    source = trace.source(rank=rank, device=device)
    diagnostics.add_tensor_tree(f"critic_source/step{step}", source, direction="H2D")
    kwargs = critic_official_kwargs(
        cfg,
        entry.operation_trace,
        source,
        step,
        x0,
        critic_engine,
    )
    # The unmodified pinned helper has no scalar-policy keyword and already
    # collects the official logs. Only the opt-in elision hook accepts False.
    if not collect_official_scalar_logs:
        kwargs["collect_scalar_logs"] = False
    before = int(critic_engine.global_steps)
    with diagnostics.scope("FSDP_PRE_FORWARD_ALL_GATHER/critic"):
        loss, logs = _critic_loss(fake_score_model=critic_engine.module, **kwargs)
    if not bool(logs.get("critic_optimizer_committed")):
        raise AssertionError("GAN P0 Critic did not commit")
    if int(critic_engine.global_steps) != before + int(expect_optimizer_commit):
        raise AssertionError("GAN P0 Critic optimizer step drift")
    return {
        "logical_step": step,
        "source_rank": rank,
        "physical_critic_step_after": int(critic_engine.global_steps),
        "loss": (
            diagnostics.diagnostic_scalar(f"gan_critic_step{step}", loss)
            if collect_diagnostics
            else TIMING_OMITTED
        ),
        "score_forwards": 2,
        "critic_optimizer_committed": bool(expect_optimizer_commit),
        "critic_logical_microbatch_boundary_reached": True,
        "generator_rollout_precomputed": True,
    }


def _aggregate_logical_records(
    records: list[dict[str, Any]], *, rank: int
) -> dict[str, Any]:
    """Keep legacy B1 receipts unchanged and expose both H100 logical rows."""
    if not records:
        raise AssertionError("a logical update has no source rows")
    if len(records) == 1:
        return records[0]
    result = dict(records[-1])
    result.update(
        source_rank=rank,
        source_ranks=[int(row["source_rank"]) for row in records],
        logical_b1_microbatch_count=len(records),
        microbatches=records,
    )
    for field in ("logical_score_requests", "physical_score_forwards", "score_forwards"):
        if all(field in row for row in records):
            result[field] = sum(int(row[field]) for row in records)
    if "local_trainable_before" in records[0]:
        result["local_trainable_before"] = records[0]["local_trainable_before"]
    if all("rollout" in row for row in records):
        rollouts = [row["rollout"] for row in records]
        stages = {
            stage: sum(int(row["stage_call_count"].get(stage, 0)) for row in rollouts)
            for stage in range(3)
        }
        total = sum(int(row["call_count"]) for row in rollouts)
        result["rollout"] = {
            "call_count": total,
            "logical_call_count": total,
            "stage_call_count": stages,
            "logical_stage_call_count": dict(stages),
            "dummy_call_count": 0,
            "microbatches": rollouts,
            "logical_b1_microbatch_count": len(records),
        }
    if all("score_pack_parity" in row for row in records):
        parity_rows = [row["score_pack_parity"] for row in records]
        enabled = any(bool(row.get("enabled")) for row in parity_rows)
        result["score_pack_parity"] = {
            "enabled": enabled,
            "status": (
                "PASS" if enabled and all(row.get("status") == "PASS" for row in parity_rows)
                else "NOT_RUN"
            ),
            "microbatches": parity_rows,
        }
    return result


def _run_logical_generator_update_p0(
    *,
    args: Any,
    rank: int,
    hardware: HardwareProfile,
    device: torch.device,
    generator_engine: Any,
    critic_engine: Any,
    trace: StaticWorld16Trace,
    cfg: Any,
    negative: torch.Tensor,
    collect_diagnostics: bool,
    diagnostics: StaticProfileDiagnostics,
    policy: StaticFSDP2P0Policy,
    score_pack_parity_check: bool,
    generator_free_memory: dict[str, Any] | None,
) -> dict[str, Any]:
    """Accumulate original rank-qualified B1 rows into one Generator commit."""
    count = hardware.microbatch_count
    engine = (
        MicrobatchCommitEngine(generator_engine, microbatch_count=count)
        if count > 1 else generator_engine
    )
    initial_step = int(generator_engine.global_steps)
    records: list[dict[str, Any]] = []
    for index, slot in enumerate(hardware.logical_slots(rank)):
        expected_commit = index == count - 1
        with engine.microbatch(index) if count > 1 else nullcontext():
            commit_kwargs = {"expect_optimizer_commit": expected_commit} if count > 1 else {}
            if args.variant == "gan":
                kwargs = dict(
                    args=args, rank=slot, device=device, generator_engine=engine,
                    critic_engine=critic_engine, trace=trace, cfg=cfg, negative=negative,
                    collect_diagnostics=collect_diagnostics, diagnostics=diagnostics,
                    generator_free_memory_receipt=generator_free_memory,
                    elide_official_generator_free_memory=bool(
                        policy.elide_official_generator_free_memory
                        and (args.timing_mode or args.profile_mode)
                    ),
                    **commit_kwargs,
                )
                if policy.pack_real_score_pair:
                    with packed_gan_real_score_pair(parity_check=score_pack_parity_check) as packing:
                        record = base._run_gan_generator_update(**kwargs)
                    record.update(
                        logical_score_requests=4,
                        physical_score_forwards=packing.physical_dmd_score_forwards + 1,
                        real_score_pair_packed=True,
                        score_pack_parity=(
                            packing.parity if packing.parity is not None
                            else {"enabled": False, "status": "NOT_RUN"}
                        ),
                    )
                else:
                    record = base._run_gan_generator_update(**kwargs)
                    record.update(
                        logical_score_requests=4, physical_score_forwards=4,
                        real_score_pair_packed=False,
                        score_pack_parity={"enabled": False, "status": "DISABLED"},
                    )
            else:
                record = _run_nogan_generator_update_p0(
                    rank=slot, device=device, generator_engine=engine,
                    critic_engine=critic_engine, trace=trace, negative=negative,
                    collect_diagnostics=collect_diagnostics, diagnostics=diagnostics,
                    policy=policy, score_pack_parity_check=score_pack_parity_check,
                    role_swap=getattr(args, "_official_role_swap", None),
                    **commit_kwargs,
                )
        expected_step = initial_step + int(expected_commit)
        if int(generator_engine.global_steps) != expected_step:
            raise AssertionError("Generator changed version before/after its outer boundary")
        if int(record["source_rank"]) != slot:
            raise AssertionError("Generator receipt changed the original logical slot")
        if count > 1:
            record.update(logical_slot=slot, physical_rank=rank,
                          optimizer_committed=expected_commit)
        records.append(record)
    return _aggregate_logical_records(records, rank=rank)


def _precompute_logical_critic_rollouts(
    *,
    rank: int,
    hardware: HardwareProfile,
    steps: tuple[int, ...],
    device: torch.device,
    generator_engine: Any,
    trace: StaticWorld16Trace,
    policy: StaticFSDP2P0Policy,
    collect_diagnostics: bool,
    diagnostics: StaticProfileDiagnostics,
    rollout_parity_check: bool,
) -> tuple[dict[int, dict[int, torch.Tensor]], list[dict[str, Any]], dict[str, Any]]:
    outputs: dict[int, dict[int, torch.Tensor]] = {}
    records: list[dict[str, Any]] = []
    persistence_records: list[dict[str, Any]] = []
    for slot in hardware.logical_slots(rank):
        values, rows, persistence = _precompute_critic_rollouts(
            rank=slot, steps=steps, device=device, generator_engine=generator_engine,
            trace=trace, policy=policy, collect_diagnostics=collect_diagnostics,
            diagnostics=diagnostics, rollout_parity_check=rollout_parity_check,
        )
        outputs[slot] = values
        if hardware.microbatch_count > 1:
            for row in rows:
                row.update(logical_slot=slot, physical_rank=rank)
        records.extend(rows)
        persistence_records.append(persistence)
    if len(persistence_records) == 1:
        persistence = persistence_records[0]
    else:
        persistence = {
            "enabled": all(bool(row.get("enabled")) for row in persistence_records),
            "logical_b1_microbatch_count": len(persistence_records),
            "rollout_batch_count": sum(int(row["rollout_batch_count"]) for row in persistence_records),
            "microbatches": [
                {"logical_slot": slot, "persistence": row}
                for slot, row in zip(hardware.logical_slots(rank), persistence_records, strict=True)
            ],
        }
    return outputs, records, persistence


def _run_logical_critic_update_p0(
    *,
    args: Any,
    rank: int,
    hardware: HardwareProfile,
    step: int,
    rollout_outputs: dict[int, dict[int, torch.Tensor]],
    device: torch.device,
    critic_engine: Any,
    trace: StaticWorld16Trace,
    cfg: Any,
    collect_diagnostics: bool,
    diagnostics: StaticProfileDiagnostics,
    policy: StaticFSDP2P0Policy,
) -> dict[str, Any]:
    """Keep the two GAN backwards per B1 row and commit after the outer GAS."""
    count = hardware.microbatch_count
    engine = (
        MicrobatchCommitEngine(critic_engine, microbatch_count=count)
        if count > 1 else critic_engine
    )
    initial_step = int(critic_engine.global_steps)
    records: list[dict[str, Any]] = []
    for index, slot in enumerate(hardware.logical_slots(rank)):
        expected_commit = index == count - 1
        with engine.microbatch(index) if count > 1 else nullcontext():
            kwargs = dict(
                rank=slot, step=step, x0=rollout_outputs[slot][step], device=device,
                critic_engine=engine, trace=trace,
                collect_diagnostics=collect_diagnostics, diagnostics=diagnostics,
            )
            if args.variant == "gan":
                record = _run_gan_critic_precomputed(
                    cfg=cfg,
                    collect_official_scalar_logs=not (
                        policy.elide_critic_scalar_logs and (args.timing_mode or args.profile_mode)
                    ),
                    **({"expect_optimizer_commit": expected_commit} if count > 1 else {}),
                    **kwargs,
                )
            else:
                record = _run_nogan_critic_precomputed(**kwargs)
        if int(critic_engine.global_steps) != initial_step + int(expected_commit):
            raise AssertionError("Critic changed version before/after its outer boundary")
        if int(record["source_rank"]) != slot or int(record["logical_step"]) != step:
            raise AssertionError("Critic receipt changed the original logical slot/step")
        if count > 1:
            record.update(logical_slot=slot, physical_rank=rank,
                          optimizer_committed=expected_commit)
        records.append(record)
    return _aggregate_logical_records(records, rank=rank)


def _parser() -> Any:
    import argparse

    value = base.parser()
    value.description = __doc__
    value.add_argument("--parameter-placement", choices=("fsdp2", "gpu-resident-zero1", "official-role-swap-zero1"), default="fsdp2")
    value.add_argument(
        "--mesh-policy",
        choices=("world16-fsdp", "node-hsdp"),
        default="world16-fsdp",
    )
    value.add_argument(
        "--rollout-batch-size",
        type=int,
        choices=(1, 2, 5),
        default=5,
    )
    value.add_argument(
        "--pack-real-score-pair",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    value.add_argument(
        "--rollout-backend",
        choices=("native-fsdp2-batch",),
        default="native-fsdp2-batch",
    )
    value.add_argument(
        "--attention-backend",
        choices=("sdpa", "local-fa2"),
        default="sdpa",
        help="Pinned physical attention implementation; training semantics are unchanged",
    )
    value.add_argument(
        "--modulation-chunk-tokens",
        type=int,
        choices=(512, 1024, 2048),
        default=512,
        help=(
            "Pinned token chunk used only by Helios modulation/residual/cross-norm "
            "helpers; 512 reproduces the source default"
        ),
    )
    value.add_argument(
        "--generator-residual-storage",
        choices=("cpu_inplace", "cuda_out_of_place"),
        default="cuda_out_of_place",
        help=(
            "Generator storage policy for the two frozen gated residuals per "
            "Helios block"
        ),
    )
    value.add_argument(
        "--critic-residual-storage",
        choices=("cpu_inplace", "cuda_out_of_place"),
        default="cuda_out_of_place",
        help=(
            "Critic storage policy for the two frozen gated residuals per "
            "Helios block"
        ),
    )
    value.add_argument(
        "--generator-nograd-lora-fused-epilogue",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fuse only the Generator no-grad PEFT LoRA scale/add/BF16 epilogue",
    )
    value.add_argument(
        "--all-role-lora-scale1-add",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Replace exact BF16 update*1+base with one add for enabled LoRA "
            "calls in both model roles, including grad-enabled calls"
        ),
    )
    value.add_argument(
        "--optimizer-dispatch-warmup",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Setup-only zero-gradient AdamW/DTensor dispatch warmup with exact "
            "parameter and optimizer-state restoration; node-HSDP only"
        ),
    )
    value.add_argument(
        "--generator-nograd-residual-fusion",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Fuse only B1 Generator no-grad FP32 gated residual tiles; "
            "grad and Critic paths remain source-pinned"
        ),
    )
    value.add_argument(
        "--exact-grad-residual-fusion",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Fuse B1 grad-enabled gated residual forward/backward only when "
            "the AdaLN table is frozen, including exact gradients for the "
            "observed tensor or phase-local timestep inputs"
        ),
    )
    value.add_argument(
        "--exact-modulated-norm-epilogue",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Fuse only the B1 FP32 LayerNorm scale/shift forward epilogue; "
            "PyTorch LayerNorm and the source backward remain unchanged"
        ),
    )
    value.add_argument(
        "--exact-modulated-norm-materialization",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Also fuse the two exact FP32 table-plus-timestep component "
            "materializations into the B1 modulated-norm epilogue"
        ),
    )
    value.add_argument(
        "--exact-modulated-norm-backward",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Replace per-chunk autograd graph reconstruction with the exact "
            "native LayerNorm backward and explicit timestep-component gradients"
        ),
    )
    value.add_argument(
        "--exact-modulated-norm-layernorm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Fuse the BF16 input cast, affine-free FP32 width-5120 LayerNorm, "
            "table/component adds, and modulation into one exact C14 kernel"
        ),
    )
    value.add_argument(
        "--exact-modulated-norm-fused-backward",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Fuse the registered C11 LayerNorm grad-input and deterministic "
            "timestep-component reductions on top of the complete C14 path"
        ),
    )
    value.add_argument(
        "--exact-history-current-fusion",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Fuse only the pinned non-NaViT guidance-cross-attention current "
            "BF16 add and history concatenation with exact gradients"
        ),
    )
    value.add_argument(
        "--selective-flash-checkpoint-stride",
        type=int,
        choices=(0, 2, 4, 10),
        default=0,
        help=(
            "Critic-only systems candidate: retain the exact pinned FA2 result "
            "for every Nth checkpointed block; 0 preserves all-recompute"
        ),
    )
    value.add_argument(
        "--generator-nograd-cross-kv-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Reuse exact cross-attention encoder K/V projections only within "
            "each B1 Generator no-grad rollout"
        ),
    )
    value.add_argument(
        "--generator-nograd-static-context-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Reuse exact history patch/RoPE/text context only within each "
            "B1 Generator no-grad rollout"
        ),
    )
    value.add_argument(
        "--generator-nograd-persistent-unshard",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Retain all 41 Generator FSDP2 units only across the five B1 "
            "no-grad rollouts, then explicitly restore the sharded policy"
        ),
    )
    value.add_argument(
        "--generator-nograd-persistent-unshard-flush-allocator-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Flush the CUDA allocator cache immediately before and after the "
            "bounded Generator persistent-unshard scope"
        ),
    )
    value.add_argument(
        "--cuda-allocator-expandable-segments",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use process-start PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
            "for the bounded persistent-unshard no-flush capacity candidate"
        ),
    )
    value.add_argument(
        "--elide-explicit-rollout-sync",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Omit only the application-owned rollout cuda synchronizations and "
            "Generator-update empty_cache; outer Full5 synchronization is unchanged"
        ),
    )
    value.add_argument(
        "--elide-critic-scalar-logs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Keep official GAN Critic scalar .item() logs in correctness, but "
            "return omission sentinels in low-overhead timing/profile runs"
        ),
    )
    value.add_argument(
        "--elide-official-generator-free-memory",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Execute the four source-pinned official GAN Generator free_memory "
            "calls in correctness, but elide them in low-overhead timing/profile"
        ),
    )
    value.add_argument(
        "--critic-only-step",
        type=int,
        choices=(10,),
        help="diagnostic-only: capture one No-GAN Critic update after its native rollout",
    )
    value.add_argument(
        "--operation-source",
        action="store_true",
        help="diagnostic-only: trace the known Critic D2H copy source on one rank",
    )
    value.add_argument("--operation-source-rank", type=int, default=0)
    value.add_argument(
        "--operation-source-target-bytes",
        type=int,
        default=DEFAULT_TARGET_BYTES,
    )
    value.add_argument("--operation-source-max-events", type=int, default=256)
    return value


def main() -> int:
    args = _parser().parse_args()
    hardware = hardware_profile()
    if args.parameter_placement in ("gpu-resident-zero1", "official-role-swap-zero1"):
        if hardware.name != "h100-8-b16" or args.mesh_policy != "world16-fsdp":
            raise ValueError("registered full-resident reference requires h100-8-b16")
        if args.profile_mode or args.ema_device != "cuda":
            raise ValueError("resident reference requires unprofiled CUDA EMA")
    if hardware.microbatch_count > 1 and args.rollout_batch_size != 1:
        raise ValueError("H100 logical-batch16 migration requires two independent B1 rows")
    if args.timing_mode and args.profile_mode:
        raise ValueError("profiled runs are diagnostic and cannot be formal timing runs")
    diagnostic_single_critic = args.critic_only_step is not None
    if diagnostic_single_critic and (
        args.variant != "no-gan" or not args.profile_mode
    ):
        raise ValueError("--critic-only-step requires No-GAN --profile-mode")
    if args.operation_source and not diagnostic_single_critic:
        raise ValueError("--operation-source requires --critic-only-step")
    if not 0 <= args.operation_source_rank < hardware.physical_world_size:
        raise ValueError("--operation-source-rank exceeds the selected physical world")
    if args.operation_source_target_bytes <= 0 or args.operation_source_max_events <= 0:
        raise ValueError("operation-source byte/event limits must be positive")
    if args.generator_nograd_residual_fusion and (
        args.rollout_batch_size != 1
        or args.generator_residual_storage != "cuda_out_of_place"
    ):
        raise ValueError(
            "Generator no-grad residual fusion requires B1 and cuda_out_of_place"
        )
    if args.exact_grad_residual_fusion and (
        args.rollout_batch_size != 1
        or args.generator_residual_storage != "cuda_out_of_place"
        or args.critic_residual_storage != "cuda_out_of_place"
    ):
        raise ValueError(
            "Exact grad residual fusion requires B1 and cuda_out_of_place "
            "for both Generator and Critic"
        )
    if args.exact_modulated_norm_epilogue and args.rollout_batch_size != 1:
        raise ValueError("Exact modulated-norm epilogue requires B1")
    if (
        args.exact_modulated_norm_materialization
        and not args.exact_modulated_norm_epilogue
    ):
        raise ValueError(
            "Exact modulation materialization requires the exact epilogue"
        )
    if args.exact_modulated_norm_backward and not args.exact_modulated_norm_epilogue:
        raise ValueError(
            "Exact modulated-norm backward requires the exact epilogue"
        )
    if args.exact_modulated_norm_layernorm and not (
        args.exact_modulated_norm_epilogue
        and args.exact_modulated_norm_materialization
        and args.exact_modulated_norm_backward
    ):
        raise ValueError(
            "Fused modulated LayerNorm requires epilogue, materialization, "
            "and explicit backward"
        )
    if args.exact_modulated_norm_fused_backward and not (
        args.exact_modulated_norm_epilogue
        and args.exact_modulated_norm_materialization
        and args.exact_modulated_norm_backward
        and args.exact_modulated_norm_layernorm
    ):
        raise ValueError(
            "Fused modulated LayerNorm backward requires the complete C14 substrate"
        )
    if args.exact_history_current_fusion and args.rollout_batch_size != 1:
        raise ValueError("Exact history/current fusion requires B1")
    if (
        args.selective_flash_checkpoint_stride
        and args.attention_backend != "local-fa2"
    ):
        raise ValueError(
            "selective Flash checkpoint save requires the pinned local-fa2 backend"
        )
    if args.generator_nograd_cross_kv_cache and args.rollout_batch_size != 1:
        raise ValueError(
            "Generator no-grad cross-attention KV reuse requires B1"
        )
    if (
        args.generator_nograd_static_context_cache
        and args.rollout_batch_size != 1
    ):
        raise ValueError("Generator no-grad static-context reuse requires B1")
    if args.generator_nograd_persistent_unshard and (
        args.variant != "gan"
        or args.mesh_policy != "world16-fsdp"
        or args.rollout_batch_size != 1
    ):
        from summarize_helios_hardware_p0 import validate_registered_h100_retention_request
        validate_registered_h100_retention_request(
            args.variant, args.mesh_policy, args.rollout_batch_size,
        )
    if args.operation_source and args.selective_flash_checkpoint_stride:
        raise ValueError(
            "operation-source diagnostics cannot be combined with selective Flash save"
        )
    policy = StaticFSDP2P0Policy(
        mesh_policy=args.mesh_policy,
        rollout_batch_size=args.rollout_batch_size,
        pack_real_score_pair=bool(args.pack_real_score_pair),
        rollout_backend=args.rollout_backend,
        optimizer_dispatch_warmup=bool(args.optimizer_dispatch_warmup),
        generator_nograd_residual_fusion=bool(
            args.generator_nograd_residual_fusion
        ),
        exact_grad_residual_fusion=bool(args.exact_grad_residual_fusion),
        exact_modulated_norm_epilogue=bool(args.exact_modulated_norm_epilogue),
        exact_modulated_norm_materialization=bool(
            args.exact_modulated_norm_materialization
        ),
        exact_modulated_norm_backward=bool(args.exact_modulated_norm_backward),
        exact_modulated_norm_layernorm=bool(args.exact_modulated_norm_layernorm),
        exact_modulated_norm_fused_backward=bool(
            args.exact_modulated_norm_fused_backward
        ),
        exact_history_current_fusion=bool(args.exact_history_current_fusion),
        modulation_chunk_tokens=int(args.modulation_chunk_tokens),
        selective_flash_checkpoint_stride=int(
            args.selective_flash_checkpoint_stride
        ),
        generator_nograd_cross_kv_cache=bool(
            args.generator_nograd_cross_kv_cache
        ),
        generator_nograd_static_context_cache=bool(
            args.generator_nograd_static_context_cache
        ),
        generator_nograd_persistent_unshard=bool(
            args.generator_nograd_persistent_unshard
        ),
        generator_nograd_persistent_unshard_flush_allocator_cache=bool(
            args.generator_nograd_persistent_unshard_flush_allocator_cache
        ),
        cuda_allocator_expandable_segments=bool(
            args.cuda_allocator_expandable_segments
        ),
        elide_explicit_rollout_sync=bool(args.elide_explicit_rollout_sync),
        elide_critic_scalar_logs=bool(args.elide_critic_scalar_logs),
        elide_official_generator_free_memory=bool(
            args.elide_official_generator_free_memory
        ),
    )
    if args.parameter_placement in ("gpu-resident-zero1", "official-role-swap-zero1") and (
        policy != StaticFSDP2P0Policy()
        or args.attention_backend != "sdpa"
        or args.generator_nograd_lora_fused_epilogue
        or args.all_role_lora_scale1_add
        or args.generator_residual_storage != "cuda_out_of_place"
        or args.critic_residual_storage != "cuda_out_of_place"
    ):
        raise ValueError("full-resident reference forbids unrelated performance/offload options")
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    if world != hardware.physical_world_size:
        raise AssertionError(
            f"P0 hardware profile {hardware.name} requires physical world "
            f"{hardware.physical_world_size}, got {world}"
        )
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    expected_allocator_conf = (
        "expandable_segments:True" if policy.cuda_allocator_expandable_segments else ""
    )
    actual_allocator_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if actual_allocator_conf != expected_allocator_conf:
        raise AssertionError(
            "CUDA allocator policy drifted: "
            f"{actual_allocator_conf!r} != {expected_allocator_conf!r}"
        )
    output = args.output_dir / "ranks" / f"rank_{rank:02d}.json"
    result: dict[str, Any] = {
        "schema_version": "distillschedule.static_fsdp2_p0_rank.v1",
        "status": "RUNNING",
        "run_id": args.run_id,
        "variant": args.variant,
        "rank": rank,
        "local_rank": local_rank,
        "hostname": socket.gethostname(),
        "world_size": world,
        "hardware_profile": {
            "name": hardware.name,
            "physical_world_size": world,
            "logical_global_batch": hardware.global_batch,
            "logical_b1_microbatches_per_rank": hardware.microbatch_count,
            "logical_slots": list(hardware.logical_slots(rank)),
            "loss_scale_per_logical_b1": 1.0 / hardware.microbatch_count,
            "rank_mapping": "slot = physical_rank + microbatch * physical_world_size",
        },
        "timing_mode": bool(args.timing_mode),
        "profile_mode": bool(args.profile_mode),
        "p0_policy": {
            "id": policy.id,
            "mesh_policy": policy.mesh_policy,
            "rollout_batch_size": policy.rollout_batch_size,
            "pack_real_score_pair": policy.pack_real_score_pair,
            "rollout_backend": policy.rollout_backend,
            "attention_backend": args.attention_backend,
            "modulation_chunk_tokens": policy.modulation_chunk_tokens,
            "selective_flash_checkpoint_stride": (
                policy.selective_flash_checkpoint_stride
            ),
            "generator_nograd_cross_kv_cache": (
                policy.generator_nograd_cross_kv_cache
            ),
            "generator_nograd_static_context_cache": (
                policy.generator_nograd_static_context_cache
            ),
            "generator_nograd_persistent_unshard": (
                policy.generator_nograd_persistent_unshard
            ),
            "generator_nograd_persistent_unshard_flush_allocator_cache": (
                policy.generator_nograd_persistent_unshard_flush_allocator_cache
            ),
            "cuda_allocator_expandable_segments": (
                policy.cuda_allocator_expandable_segments
            ),
            "generator_residual_storage": args.generator_residual_storage,
            "critic_residual_storage": args.critic_residual_storage,
            "generator_nograd_lora_fused_epilogue": bool(
                args.generator_nograd_lora_fused_epilogue
            ),
            "all_role_lora_scale1_add": bool(args.all_role_lora_scale1_add),
            "optimizer_dispatch_warmup": bool(
                args.optimizer_dispatch_warmup
            ),
            "generator_nograd_residual_fusion": (
                policy.generator_nograd_residual_fusion
            ),
            "exact_grad_residual_fusion": policy.exact_grad_residual_fusion,
            "exact_modulated_norm_epilogue": (
                policy.exact_modulated_norm_epilogue
            ),
            "exact_modulated_norm_materialization": (
                policy.exact_modulated_norm_materialization
            ),
            "exact_modulated_norm_backward": policy.exact_modulated_norm_backward,
            "exact_modulated_norm_layernorm": policy.exact_modulated_norm_layernorm,
            "exact_modulated_norm_fused_backward": (
                policy.exact_modulated_norm_fused_backward
            ),
            "exact_history_current_fusion": policy.exact_history_current_fusion,
            "elide_explicit_rollout_sync": policy.elide_explicit_rollout_sync,
            "elide_critic_scalar_logs": policy.elide_critic_scalar_logs,
            "elide_official_generator_free_memory": (
                policy.elide_official_generator_free_memory
            ),
        },
    }
    if args.parameter_placement in ("gpu-resident-zero1", "official-role-swap-zero1"):
        result["parameter_placement"] = args.parameter_placement
        result["p0_policy"]["parameter_placement"] = args.parameter_placement
        result["p0_policy"]["id"] += "/" + args.parameter_placement
    try:
        # GAN diagnostics also import Helios. Resolve the registered backend
        # before any hook can trigger the frozen Hopper auto-detection.
        result["attention_backend"] = _attention_backend_receipt(
            args.attention_backend
        )
        critic_scalar_logs = install_critic_scalar_log_elision(
            enabled=bool(policy.elide_critic_scalar_logs and args.variant == "gan")
        )
        result["critic_scalar_logs"] = critic_scalar_logs
        generator_free_memory = prepare_generator_free_memory_receipt(
            enabled=bool(
                policy.elide_official_generator_free_memory
                and args.variant == "gan"
            )
        )
        result["generator_free_memory"] = generator_free_memory
        contract = contract_for_variant(args.variant)
        nsys_range = StaticFSDP2Full5NsightRange.from_env(
            variant=args.variant,
            logical_window=(args.critic_only_step, args.critic_only_step)
            if diagnostic_single_critic
            else None,
            start_point=(
                f"before_step{args.critic_only_step}_critic_compute"
                if diagnostic_single_critic
                else "before_first_model_compute"
            ),
            stop_point=(
                f"after_step{args.critic_only_step}_critic_commit"
                if diagnostic_single_critic
                else None
            ),
        )
        if args.profile_mode and not nsys_range.enabled:
            raise ValueError("profile mode requires the exact Nsight API range")
        logical = (
            LogicalCheckpointPackage.load_systems_only_gan(args.logical_package)
            if args.variant == "gan"
            else LogicalCheckpointPackage.load(args.logical_package)
        )
        if args.expected_logical_hash and logical.manifest_hash != args.expected_logical_hash:
            raise ValueError("logical checkpoint identity changed")
        trace = StaticWorld16Trace(args.trace_bundle, variant=args.variant)
        if args.expected_trace_hash and trace.bundle_hash != args.expected_trace_hash:
            raise ValueError("trace bundle identity changed")
        if int(trace.rank_control_validation["rank_count"]) != hardware.global_batch:
            raise ValueError("trace must retain all sixteen original logical sample slots")
        from distillschedule.runtime.fsdp2_p0 import validate_rollout_entries

        for slot in hardware.logical_slots(rank):
            control_entries = [
                trace.load_critic_entry(rank=slot, step=step, device="cpu")
                for step in contract.logical_steps
            ]
            for group in chunked(control_entries, policy.rollout_batch_size):
                validate_rollout_entries(group)
            del control_entries
        result["contract"] = base.dataclasses_asdict(contract)
        result["identities"] = {
            "logical_manifest_hash": logical.manifest_hash,
            "trace_bundle_hash": trace.bundle_hash,
            "trace_binding_hash": trace.binding_hash,
            "rank_control_validation": trace.rank_control_validation,
            "cross_step_rollout_batchability": "PASS",
        }
        if args.validate_only:
            result.update(status="PASS", validate_only=True)
            base._atomic_json(output, result)
            dist.barrier()
            return 0
        if args.variant == "gan" and args.gan_config is None:
            raise ValueError("GAN variant requires --gan-config")
        negative = NegativePromptAsset.load(args.negative_prompt).embedding.to(
            device=device, dtype=torch.bfloat16
        )
        cfg = load_official_config(args.gan_config) if args.variant == "gan" else None
        generator_engine, critic_engine, build = _build_engines(
            args=args,
            package=logical,
            device=device,
            policy=policy,
        )
        ema_type = (ResidentZero1ShardEMA
                    if args.parameter_placement in ("gpu-resident-zero1", "official-role-swap-zero1") else FSDP2ShardEMA)
        ema = ema_type.load_from_logical(
            generator_engine.module,
            logical.root,
            device=args.ema_device,
            nvtx_enabled=args.profile_mode,
        )
        diagnostics = StaticProfileDiagnostics(
            enabled=args.profile_mode,
            rank=rank,
            output_path=(
                args.output_dir / "diagnostics" / f"rank_{rank:02d}.json"
                if args.profile_mode
                else None
            ),
            collect_cpu_diagnostics=(not args.profile_mode) or args.profile_diagnostics,
        )
        collect_diagnostics = (not args.timing_mode) and args.profile_diagnostics
        score_pack_parity_check = (not args.timing_mode) and (not args.profile_mode)
        result["build"] = build
        result["profile_diagnostics"] = bool(args.profile_diagnostics)
        result["ema_device"] = args.ema_device
        result["diagnostic_single_critic"] = bool(diagnostic_single_critic)
        result["operation_source"] = {
            "enabled": bool(args.operation_source),
            "record_rank": int(args.operation_source_rank),
            "target_bytes": int(args.operation_source_target_bytes),
            "max_events": int(args.operation_source_max_events),
        }
        initial_ema_step = int(ema.optimization_step)
        result["initial"] = {
            "generator_physical_step": int(generator_engine.global_steps),
            "critic_physical_step": int(critic_engine.global_steps),
            "ema_step": int(ema.optimization_step),
            "memory": base._memory(device),
        }
        dist.barrier()
        torch.cuda.synchronize(device)
        started_perf = time.perf_counter()
        result["timing_start_unix_ns"] = time.time_ns()
        if args.profile_mode and not diagnostic_single_critic:
            dist.barrier()
            nsys_range.before_model_compute(contract.logical_steps[0])
            torch.cuda.nvtx.range_push(
                f"STATIC_FSDP2_P0_FULL5/{args.variant}/{policy.id}"
            )
            dist.barrier()

        generator_record = _run_logical_generator_update_p0(
            args=args,
            rank=rank,
            hardware=hardware,
            device=device,
            generator_engine=generator_engine,
            critic_engine=critic_engine,
            trace=trace,
            cfg=cfg,
            negative=negative,
            collect_diagnostics=collect_diagnostics,
            diagnostics=diagnostics,
            policy=policy,
            score_pack_parity_check=score_pack_parity_check,
            generator_free_memory=generator_free_memory,
        )

        execution_steps = (
            (int(args.critic_only_step),)
            if diagnostic_single_critic
            else contract.logical_steps
        )
        role_swap = getattr(args, "_official_role_swap", None)
        if role_swap is not None:
            rollout_outputs, rollout_records = {}, []
            generator_nograd_persistent_unshard = {
                "enabled": False, "rollout_batch_count": 0,
                "reason": "OFFICIAL_PER_STEP_G_SCORE_SWITCH"}
        else:
            (
                rollout_outputs,
                rollout_records,
                generator_nograd_persistent_unshard,
            ) = _precompute_logical_critic_rollouts(
                rank=rank,
                hardware=hardware,
                steps=execution_steps,
                device=device,
                generator_engine=generator_engine,
                trace=trace,
                policy=policy,
                collect_diagnostics=collect_diagnostics,
                diagnostics=diagnostics,
                rollout_parity_check=(
                    (not args.timing_mode) and (not args.profile_mode)
                ),
            )
        if args.profile_mode and diagnostic_single_critic:
            torch.cuda.synchronize(device)
            dist.barrier()
            nsys_range.before_model_compute(execution_steps[0])
            torch.cuda.nvtx.range_push(
                f"STATIC_FSDP2_P0_CRITIC_CANARY/no-gan/step{execution_steps[0]}"
            )
            dist.barrier()
        critic_records: list[dict[str, Any]] = []
        ema_records: list[dict[str, Any]] = []
        operation_source_records: list[dict[str, Any]] = []
        for step in execution_steps:
            if role_swap is not None:
                role_swap.activate("generator", device)
                rollout_outputs, rows, _persistence = _precompute_logical_critic_rollouts(
                    rank=rank, hardware=hardware, steps=(step,), device=device,
                    generator_engine=generator_engine, trace=trace, policy=policy,
                    collect_diagnostics=collect_diagnostics, diagnostics=diagnostics,
                    rollout_parity_check=False)
                rollout_records.extend(rows)
                generator_nograd_persistent_unshard["rollout_batch_count"] += len(rows)
                role_swap.activate("critic", device)
            record_this_rank = bool(
                args.operation_source and rank == args.operation_source_rank
            )
            with OperationSourceCapture(
                enabled=record_this_rank,
                rank=rank,
                target_bytes=args.operation_source_target_bytes,
                max_events=args.operation_source_max_events,
            ) as operation_source:
                critic_record = _run_logical_critic_update_p0(
                    args=args,
                    rank=rank,
                    hardware=hardware,
                    step=step,
                    rollout_outputs=rollout_outputs,
                    device=device,
                    critic_engine=critic_engine,
                    trace=trace,
                    cfg=cfg,
                    collect_diagnostics=collect_diagnostics,
                    diagnostics=diagnostics,
                    policy=policy,
                )
            if record_this_rank:
                source_path = (
                    args.output_dir
                    / "operation_source"
                    / f"rank_{rank:02d}_step_{int(step)}.json"
                )
                operation_source.trace.dump(
                    source_path,
                    metadata={
                        "logical_step": int(step),
                        "variant": args.variant,
                        "critic_only": bool(diagnostic_single_critic),
                        "saved_tensors_hooks_installed_by_tracer": False,
                    },
                )
                operation_source_records.append(
                    {
                        "logical_step": int(step),
                        "matched_events": operation_source.trace.matched_events,
                        "recorded_events": len(operation_source.trace.events),
                        "dropped_events": operation_source.trace.dropped_events,
                        "path": str(source_path),
                    }
                )
            critic_records.append(critic_record)
            if not diagnostic_single_critic and step in contract.ema_update_steps:
                decay = ema.step()
                diagnostics.add_bytes(
                    "ema_shadow",
                    int(getattr(ema, "last_transfer_bytes", 0)),
                    direction="D2H" if args.ema_device == "cpu" else "GPU",
                    source="fsdp2_ema_local_shard",
                )
                ema_records.append(
                    {
                        "logical_step": step,
                        "decay": decay,
                        "optimization_step": int(ema.optimization_step),
                    }
                )

        if args.profile_mode:
            torch.cuda.synchronize(device)
            dist.barrier()
            torch.cuda.nvtx.range_pop()
            nsys_range.after_critic_commit(execution_steps[-1])
            dist.barrier()
            nsys_range.assert_complete()
            diagnostics.dump_after_capture()
        torch.cuda.synchronize(device)
        result["timing_stop_unix_ns"] = time.time_ns()
        result["full5_seconds"] = time.perf_counter() - started_perf

        expected_g = int(logical.component("generator")["optimizer_step"]) + 1
        expected_c = int(logical.component("critic")["optimizer_step"]) + len(
            execution_steps
        )
        if int(generator_engine.global_steps) != expected_g:
            raise AssertionError("P0 Generator update count changed")
        if int(critic_engine.global_steps) != expected_c:
            raise AssertionError("P0 Critic update count changed")
        expected_ema = (
            initial_ema_step
            if diagnostic_single_critic
            else int(contract.final_versions["ema"])
        )
        if int(ema.optimization_step) != expected_ema:
            raise AssertionError("P0 EMA update count changed")
        physical_calls = int(rollout_records[0]["call_count"])
        rollout_estimate = estimate_rollout_collective_groups(
            physical_calls_per_rollout=physical_calls,
            logical_rollouts=len(execution_steps) * hardware.microbatch_count,
            batch_size=policy.rollout_batch_size,
        )
        result.update(
            status="PASS",
            generator=generator_record,
            rollout_batches=rollout_records,
            generator_nograd_persistent_unshard=(
                generator_nograd_persistent_unshard
            ),
            rollout_fence_policy={
                "elide_explicit_rollout_sync": policy.elide_explicit_rollout_sync,
                "generator_update_empty_cache_count": int(
                    not policy.elide_explicit_rollout_sync
                ) * hardware.microbatch_count,
                "generator_update_sync_count": int(
                    not policy.elide_explicit_rollout_sync
                ) * hardware.microbatch_count,
                "critic_rollout_sync_count": sum(
                    int(bool(record.get("synchronize_after_rollout")))
                    for record in rollout_records
                ),
                "outer_full5_boundary_sync_preserved": True,
            },
            critic_scalar_logs=critic_scalar_logs,
            generator_free_memory=generator_free_memory,
            rollout_collective_estimate=rollout_estimate,
            critic=critic_records,
            ema=ema_records,
            updates={
                "generator": 1,
                "critic": len(execution_steps),
                "ema": len(ema_records),
            },
            operation_source_records=operation_source_records,
            final={
                "generator_physical_step": int(generator_engine.global_steps),
                "critic_physical_step": int(critic_engine.global_steps),
                "generator_logical_version": int(contract.final_versions["generator"]),
                "critic_logical_version": (
                    int(contract.final_versions["critic"])
                    if not diagnostic_single_critic
                    else int(logical.component("critic")["optimizer_step"]) + 1
                ),
                "ema_step": int(ema.optimization_step),
                "ema_local_checksum": (
                    TIMING_OMITTED if args.timing_mode else ema.checksum()
                ),
                "memory": base._memory(device),
            },
            formal_timer_scope=(
                "diagnostic prelude plus one profiled Critic commit; not comparable to Full5"
                if diagnostic_single_critic
                else "Generator update; all five batched G201 critic rollouts; "
                "five ordered Critic commits and required EMA"
            ),
            transition_count=0,
            model_swap_count=(sum(v["events"] for v in role_swap.transfers.values())
                              if role_swap is not None else 0),
            parameter_streaming_count=0,
            external_rollout_engine_used=False,
            gan_score_fusion=(
                "real_cond_uncond_b2"
                if policy.pack_real_score_pair
                else "disabled"
            ),
        )
        if role_swap is not None:
            result["official_role_swap"] = role_swap.receipt()
            result["build"]["official_role_swap"] = role_swap.receipt()
        base._atomic_json(output, result)
        dist.barrier()
        return 0
    except BaseException as error:
        result.update(
            status=(
                "OOM"
                if isinstance(error, torch.cuda.OutOfMemoryError)
                or "out of memory" in str(error).lower()
                else "FAIL"
            ),
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
        )
        try:
            result["memory_failure"] = base._memory(device)
        except BaseException:
            pass
        base._atomic_json(output, result)
        raise
    finally:
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except BaseException:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
