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
import importlib.util
import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

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
from distillschedule.runtime.fsdp2_static import FSDP2ShardEMA, build_fsdp2_engine, configure_fsdp2_checkpointing, contract_for_variant
from distillschedule.runtime.nsys_range import StaticFSDP2Full5NsightRange
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


WORLD_SIZE = 16
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
        )
        logical_restore = load_logical_master_into_trainables(
            model,
            package,
            component=role,
            verify_payload_sha256=(dist.get_rank() == 0),
        )
    configure_fsdp2_checkpointing(
        model,
        operation_source=bool(args.operation_source),
    )
    model.to(device)
    torch.cuda.synchronize(device)
    sharding = apply_blockwise_fsdp2_policy(
        model,
        mesh_policy=policy.mesh_policy,
        world_size=WORLD_SIZE,
    )
    gc.collect()
    host_trimmed = trim_host_allocator()
    torch.cuda.empty_cache()
    return model, {
        "metadata": metadata,
        "logical_restore": logical_restore,
        "fsdp2": sharding,
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
    generator_engine, generator_restore = build_fsdp2_engine(
        module=generator,
        package=package,
        component="generator",
        lr=2e-6,
        betas=(0.0, 0.999),
        eps=1e-8,
        weight_decay=1e-3,
    )
    critic_engine, critic_restore = build_fsdp2_engine(
        module=critic,
        package=package,
        component="critic",
        lr=4e-7,
        betas=(0.0, 0.999),
        eps=1e-8,
        weight_decay=1e-3,
    )
    return generator_engine, critic_engine, {
        "generator": {**generator_build, "optimizer_restore": generator_restore},
        "critic": {**critic_build, "optimizer_restore": critic_restore},
        "static_residency": {
            "both_backbones_sharded_for_full5": True,
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
) -> dict[str, Any]:
    with diagnostics.scope("INPUT_H2D/generator_entry"):
        entry = trace.load_generator_entry(rank=rank, device=device)
    diagnostics.add_tensor_tree("generator_entry", entry, direction="H2D")
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
        )
    batch = prepare_dmd_score_batch(generated_x0=x0, entry=entry, device=device)
    fake = score_forward(
        critic_engine.module,
        batch,
        condition=batch["encoder_hidden_states"],
        adapter_enabled=True,
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
            )
            reference_uncond = score_forward(
                critic_engine.module,
                batch,
                condition=negative,
                adapter_enabled=False,
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
        )
        real_uncond = score_forward(
            critic_engine.module,
            batch,
            condition=negative,
            adapter_enabled=False,
        )
        physical_score_forwards = 3
    gradient, auxiliary = construct_dmd_gradient(
        batch=batch,
        fake_flow_cond=fake,
        real_flow_cond=real_cond,
        real_flow_uncond=real_uncond,
    )
    loss = detached_dmd_surrogate_loss(x0, gradient, gradient_mask)
    with diagnostics.scope("FSDP_PRE_BACKWARD_ALL_GATHER/generator"):
        generator_engine.backward(loss)
    with diagnostics.scope("OPTIMIZER_STEP/generator"):
        generator_engine.step()
    after = (
        TIMING_OMITTED
        if not collect_diagnostics
        else base._profile_fingerprint(
            generator_engine.module, diagnostics, "generator_after"
        )
    )
    if collect_diagnostics and before == after:
        raise AssertionError("No-GAN P0 Generator trainables did not update")
    del entry, x0, gradient_mask, batch, fake, real_cond, real_uncond, gradient, auxiliary
    return {
        "logical_step": 10,
        "source_rank": rank,
        "physical_generator_step_after": int(generator_engine.global_steps),
        "local_trainable_before": before,
        "local_trainable_after": after,
        "rollout": rollout,
        "surrogate_loss": (
            diagnostics.diagnostic_scalar("generator_surrogate", loss)
            if collect_diagnostics
            else TIMING_OMITTED
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
) -> tuple[dict[int, torch.Tensor], list[dict[str, Any]]]:
    outputs: dict[int, torch.Tensor] = {}
    records: list[dict[str, Any]] = []
    for batch_index, step_group in enumerate(chunked(steps, policy.rollout_batch_size)):
        entries = []
        for step in step_group:
            with diagnostics.scope(f"INPUT_H2D/critic_entry_rollout/step{step}"):
                entry = trace.load_critic_entry(rank=rank, step=step, device=device)
            diagnostics.add_tensor_tree(
                f"critic_entry_rollout/step{step}", entry, direction="H2D"
            )
            entries.append(entry)
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
            )
        parity_rows: list[dict[str, Any]] = []
        if rollout_parity_check and len(entries) > 1:
            for entry, candidate in zip(entries, generated, strict=True):
                reference, _reference_record = run_generator_trace_rollout(
                    model=generator_engine.module,
                    loaded=entry,
                    device=device,
                    detailed_profile=False,
                    record_output_sha256=False,
                    empty_cache_before_rollout=False,
                    collect_diagnostics=False,
                )
                error = _rollout_pack_error(reference, candidate)
                parity_rows.append(
                    {
                        "logical_step": int(entry.pool_entry.logical_step),
                        **error,
                    }
                )
                if error["mean_relative"] > PACKED_SCORE_MEAN_REL_TOLERANCE:
                    raise AssertionError(
                        "batched Generator rollout exceeds mean-relative tolerance: "
                        f"step={entry.pool_entry.logical_step} "
                        f"{error['mean_relative']:.6g} > "
                        f"{PACKED_SCORE_MEAN_REL_TOLERANCE:.6g}"
                    )
                del reference, _reference_record
        for step, value in zip(step_group, generated, strict=True):
            outputs[int(step)] = value
        record["batch_index"] = batch_index
        record["batch_parity"] = {
            "enabled": bool(rollout_parity_check and len(entries) > 1),
            "status": "PASS" if parity_rows else "NOT_RUN",
            "tolerance_mean_relative": PACKED_SCORE_MEAN_REL_TOLERANCE,
            "samples": parity_rows,
        }
        records.append(record)
        del entries, generated
    if set(outputs) != set(steps):
        raise AssertionError("P0 rollout precompute missed one or more logical steps")
    return outputs, records


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
) -> dict[str, Any]:
    with diagnostics.scope(f"INPUT_H2D/critic_entry_update/step{step}"):
        entry = trace.load_critic_entry(rank=rank, step=step, device=device)
    diagnostics.add_tensor_tree(f"critic_entry_update/step{step}", entry, direction="H2D")
    batch = prepare_student_batch(generated_x0=x0, loaded=entry, device=device)
    with diagnostics.scope("FSDP_PRE_FORWARD_ALL_GATHER/critic"):
        _prediction, loss = student_forward_loss(critic_engine, batch)
    with diagnostics.scope("FSDP_PRE_BACKWARD_ALL_GATHER/critic"):
        critic_engine.backward(loss)
    with diagnostics.scope("OPTIMIZER_STEP/critic"):
        critic_engine.step()
    return {
        "logical_step": step,
        "source_rank": rank,
        "physical_critic_step_after": int(critic_engine.global_steps),
        "loss": (
            diagnostics.diagnostic_scalar(f"critic_step{step}", loss)
            if collect_diagnostics
            else TIMING_OMITTED
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
    diagnostics: StaticProfileDiagnostics,
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
    before = int(critic_engine.global_steps)
    with diagnostics.scope("FSDP_PRE_FORWARD_ALL_GATHER/critic"):
        loss, logs = _critic_loss(fake_score_model=critic_engine.module, **kwargs)
    if not bool(logs.get("critic_optimizer_committed")):
        raise AssertionError("GAN P0 Critic did not commit")
    if int(critic_engine.global_steps) != before + 1:
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
        "critic_optimizer_committed": True,
        "generator_rollout_precomputed": True,
    }


def _parser() -> Any:
    import argparse

    value = base.parser()
    value.description = __doc__
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
    if args.timing_mode and args.profile_mode:
        raise ValueError("profiled runs are diagnostic and cannot be formal timing runs")
    diagnostic_single_critic = args.critic_only_step is not None
    if diagnostic_single_critic and (
        args.variant != "no-gan" or not args.profile_mode
    ):
        raise ValueError("--critic-only-step requires No-GAN --profile-mode")
    if args.operation_source and not diagnostic_single_critic:
        raise ValueError("--operation-source requires --critic-only-step")
    if not 0 <= args.operation_source_rank < WORLD_SIZE:
        raise ValueError("--operation-source-rank must be in [0, 15]")
    if args.operation_source_target_bytes <= 0 or args.operation_source_max_events <= 0:
        raise ValueError("operation-source byte/event limits must be positive")
    policy = StaticFSDP2P0Policy(
        mesh_policy=args.mesh_policy,
        rollout_batch_size=args.rollout_batch_size,
        pack_real_score_pair=bool(args.pack_real_score_pair),
        rollout_backend=args.rollout_backend,
    )
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    if world != WORLD_SIZE:
        raise AssertionError(f"P0 static FSDP2 requires world16, got {world}")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
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
        "timing_mode": bool(args.timing_mode),
        "profile_mode": bool(args.profile_mode),
        "p0_policy": {
            "id": policy.id,
            "mesh_policy": policy.mesh_policy,
            "rollout_batch_size": policy.rollout_batch_size,
            "pack_real_score_pair": policy.pack_real_score_pair,
            "rollout_backend": policy.rollout_backend,
        },
    }
    try:
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
        control_entries = [
            trace.load_critic_entry(rank=rank, step=step, device="cpu")
            for step in contract.logical_steps
        ]
        from distillschedule.runtime.fsdp2_p0 import validate_rollout_entries

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
        ema = FSDP2ShardEMA.load_from_logical(
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

        if args.variant == "gan":
            if policy.pack_real_score_pair:
                with packed_gan_real_score_pair(
                    parity_check=score_pack_parity_check
                ) as gan_pack_state:
                    generator_record = base._run_gan_generator_update(
                        args=args,
                        rank=rank,
                        device=device,
                        generator_engine=generator_engine,
                        critic_engine=critic_engine,
                        trace=trace,
                        cfg=cfg,
                        negative=negative,
                        collect_diagnostics=collect_diagnostics,
                        diagnostics=diagnostics,
                    )
                generator_record.update(
                    logical_score_requests=4,
                    physical_score_forwards=(
                        gan_pack_state.physical_dmd_score_forwards + 1
                    ),
                    real_score_pair_packed=True,
                    score_pack_parity=(
                        gan_pack_state.parity
                        if gan_pack_state.parity is not None
                        else {"enabled": False, "status": "NOT_RUN"}
                    ),
                )
            else:
                generator_record = base._run_gan_generator_update(
                    args=args,
                    rank=rank,
                    device=device,
                    generator_engine=generator_engine,
                    critic_engine=critic_engine,
                    trace=trace,
                    cfg=cfg,
                    negative=negative,
                    collect_diagnostics=collect_diagnostics,
                    diagnostics=diagnostics,
                )
                generator_record.update(
                    logical_score_requests=4,
                    physical_score_forwards=4,
                    real_score_pair_packed=False,
                    score_pack_parity={"enabled": False, "status": "DISABLED"},
                )
        else:
            generator_record = _run_nogan_generator_update_p0(
                rank=rank,
                device=device,
                generator_engine=generator_engine,
                critic_engine=critic_engine,
                trace=trace,
                negative=negative,
                collect_diagnostics=collect_diagnostics,
                diagnostics=diagnostics,
                policy=policy,
                score_pack_parity_check=score_pack_parity_check,
            )

        execution_steps = (
            (int(args.critic_only_step),)
            if diagnostic_single_critic
            else contract.logical_steps
        )
        rollout_outputs, rollout_records = _precompute_critic_rollouts(
            rank=rank,
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
            record_this_rank = bool(
                args.operation_source and rank == args.operation_source_rank
            )
            with OperationSourceCapture(
                enabled=record_this_rank,
                rank=rank,
                target_bytes=args.operation_source_target_bytes,
                max_events=args.operation_source_max_events,
            ) as operation_source:
                critic_record = (
                    _run_gan_critic_precomputed(
                        rank=rank,
                        step=step,
                        x0=rollout_outputs[step],
                        device=device,
                        critic_engine=critic_engine,
                        trace=trace,
                        cfg=cfg,
                        collect_diagnostics=collect_diagnostics,
                        diagnostics=diagnostics,
                    )
                    if args.variant == "gan"
                    else _run_nogan_critic_precomputed(
                        rank=rank,
                        step=step,
                        x0=rollout_outputs[step],
                        device=device,
                        critic_engine=critic_engine,
                        trace=trace,
                        collect_diagnostics=collect_diagnostics,
                        diagnostics=diagnostics,
                    )
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
            logical_rollouts=len(execution_steps),
            batch_size=policy.rollout_batch_size,
        )
        result.update(
            status="PASS",
            generator=generator_record,
            rollout_batches=rollout_records,
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
            model_swap_count=0,
            parameter_streaming_count=0,
            external_rollout_engine_used=False,
            gan_score_fusion=(
                "real_cond_uncond_b2"
                if policy.pack_real_score_pair
                else "disabled"
            ),
        )
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
