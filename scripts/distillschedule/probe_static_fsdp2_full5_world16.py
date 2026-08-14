#!/usr/bin/env python3
"""Static world16 FSDP2 Full5 baseline for No-GAN and GAN-on Helios.

Both 14B physical backbones are constructed once, block-wise FSDP2 sharded on
the same default world16, and kept in that representation for the entire
measured Full5.  There is no role transition, process-group rebuild, CPU frozen
parameter streaming, or parameter offload in the timed window.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
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

from distillschedule.runtime.fsdp2_static import (
    FSDP2ShardEMA,
    apply_blockwise_fsdp2,
    build_fsdp2_engine,
    configure_fsdp2_checkpointing,
    contract_for_variant,
)
from runtime.critic_pool_checkpoint import (
    LogicalCheckpointPackage,
    load_logical_master_into_trainables,
)
from runtime.critic_pool_helios import (
    build_helios_pool_model,
    prepare_student_batch,
    run_generator_trace_rollout,
    student_forward_loss,
)
from runtime.dmd_boundary import detached_dmd_surrogate_loss
from runtime.gan_rolephase_official import (
    critic_official_kwargs,
    load_official_config,
    score_official_kwargs,
)
from runtime.gst_pool_contract import NegativePromptAsset
from runtime.gst_pool_helios import (
    construct_dmd_gradient,
    prepare_dmd_score_batch,
    run_generator_update_trace_rollout,
    score_forward,
)
from runtime.official_baseline import suspend_transformers_zero3_init, trim_host_allocator
from runtime.static_world16_trace import StaticWorld16Trace


WORLD_SIZE = 16
TIMING_OMITTED = "omitted_in_low_overhead_timing_mode"


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _memory(device: torch.device) -> dict[str, float]:
    free, total = torch.cuda.mem_get_info(device)
    gib = float(2**30)
    return {
        "allocated_gib": torch.cuda.memory_allocated(device) / gib,
        "reserved_gib": torch.cuda.memory_reserved(device) / gib,
        "max_allocated_gib": torch.cuda.max_memory_allocated(device) / gib,
        "max_reserved_gib": torch.cuda.max_memory_reserved(device) / gib,
        "driver_free_gib": free / gib,
        "driver_total_gib": total / gib,
    }


def _local_trainable_fingerprint(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        local = parameter.detach().to_local() if hasattr(parameter, "to_local") else parameter.detach()
        value = local.float()
        stats = torch.stack(
            (
                value.sum(dtype=torch.float64),
                value.square().sum(dtype=torch.float64),
                value.abs().max().double(),
                torch.tensor(float(value.numel()), device=value.device, dtype=torch.float64),
            )
        ).cpu().numpy().tobytes()
        digest.update(name.encode())
        digest.update(stats)
    return digest.hexdigest()


def _build_static_model(
    *,
    args: argparse.Namespace,
    role: str,
    package: LogicalCheckpointPackage,
    device: torch.device,
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
    # FSDP2's mixed-precision materializer can expose different metadata for
    # the original and replayed checkpoint tensors.  Keep Helios' enabled
    # non-reentrant checkpointing and RNG semantics; disable only PyTorch's
    # metadata validator through the FSDP2-specific compatibility adapter.
    configure_fsdp2_checkpointing(model)
    # Only one full 14B model is materialized on a GPU at a time.  It is
    # immediately FSDP2-sharded before construction of the second backbone.
    model.to(device)
    torch.cuda.synchronize(device)
    fsdp = apply_blockwise_fsdp2(model, world_size=WORLD_SIZE)
    gc.collect()
    host_trimmed = trim_host_allocator()
    torch.cuda.empty_cache()
    return model, {
        "metadata": metadata,
        "logical_restore": logical_restore,
        "fsdp2": fsdp,
        "host_allocator_trimmed_after_shard": host_trimmed,
        "memory_after_shard": _memory(device),
    }


def _build_engines(
    *,
    args: argparse.Namespace,
    package: LogicalCheckpointPackage,
    device: torch.device,
) -> tuple[Any, Any, dict[str, Any]]:
    generator, generator_build = _build_static_model(
        args=args, role="generator", package=package, device=device
    )
    dist.barrier()
    critic, critic_build = _build_static_model(
        args=args, role="critic", package=package, device=device
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
            "both_backbones_fsdp2_sharded_on_every_rank": True,
            "parameter_offload": False,
            "frozen_parameter_streaming": False,
            "role_transition": False,
            "process_group_rebuild_in_full5": False,
        },
    }


def _run_nogan_generator_update(
    *,
    rank: int,
    device: torch.device,
    generator_engine: Any,
    critic_engine: Any,
    trace: StaticWorld16Trace,
    negative: torch.Tensor,
    timing_mode: bool,
) -> dict[str, Any]:
    entry = trace.load_generator_entry(rank=rank, device=device)
    before = TIMING_OMITTED if timing_mode else _local_trainable_fingerprint(generator_engine.module)
    x0, gradient_mask, rollout = run_generator_update_trace_rollout(
        engine=generator_engine,
        entry=entry,
        device=device,
        detailed_profile=False,
    )
    batch = prepare_dmd_score_batch(generated_x0=x0, entry=entry, device=device)
    fake = score_forward(
        critic_engine.module,
        batch,
        condition=batch["encoder_hidden_states"],
        adapter_enabled=True,
    )
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
    gradient, auxiliary = construct_dmd_gradient(
        batch=batch,
        fake_flow_cond=fake,
        real_flow_cond=real_cond,
        real_flow_uncond=real_uncond,
    )
    loss = detached_dmd_surrogate_loss(x0, gradient, gradient_mask)
    generator_engine.backward(loss)
    generator_engine.step()
    after = TIMING_OMITTED if timing_mode else _local_trainable_fingerprint(generator_engine.module)
    if not timing_mode and before == after:
        raise AssertionError("No-GAN FSDP2 Generator trainables did not update")
    del entry, x0, gradient_mask, batch, fake, real_cond, real_uncond, gradient, auxiliary
    return {
        "logical_step": 10,
        "source_rank": rank,
        "physical_generator_step_after": int(generator_engine.global_steps),
        "local_trainable_before": before,
        "local_trainable_after": after,
        "rollout": rollout,
        "surrogate_loss": TIMING_OMITTED if timing_mode else float(loss.detach().cpu()),
        "score_forwards": 3,
    }


def _gan_score_boundary_gradient(
    *,
    score_model: torch.nn.Module,
    generated_latent: torch.Tensor,
    gan_g_weight: float,
    official_kwargs: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute d(DMD + w*GAN-G)/dx without mutating FSDP2 requires_grad state."""

    from helios.utils.utils_helios_post import compute_distribution_matching_loss

    preexisting = {
        name: id(parameter.grad)
        for name, parameter in score_model.named_parameters()
        if parameter.grad is not None
    }
    leaf = generated_latent.detach().requires_grad_(True)
    outputs = compute_distribution_matching_loss(
        real_fake_score_model=score_model,
        image_or_video=leaf,
        **official_kwargs,
    )
    if len(outputs) != 5:
        raise AssertionError("official GAN score return signature changed")
    dmd_loss, _ca, _dm, gan_g_loss, _logs = outputs
    total = dmd_loss + gan_g_loss * float(gan_g_weight)
    gradient = torch.autograd.grad(
        total,
        leaf,
        retain_graph=False,
        create_graph=False,
        only_inputs=True,
    )[0].detach()
    for name, parameter in score_model.named_parameters():
        if parameter.grad is None:
            continue
        if name not in preexisting or id(parameter.grad) != preexisting[name]:
            raise AssertionError("GAN-G/DMD read-only FSDP2 Score produced parameter gradients")
    return gradient.to(dtype=generated_latent.dtype), {
        "total": float(total.detach().float().cpu()),
        "dmd": float(dmd_loss.detach().float().cpu()),
        "gan_g_raw": float(gan_g_loss.detach().float().cpu()),
    }


def _run_gan_generator_update(
    *,
    args: argparse.Namespace,
    rank: int,
    device: torch.device,
    generator_engine: Any,
    critic_engine: Any,
    trace: StaticWorld16Trace,
    cfg: Any,
    negative: torch.Tensor,
    timing_mode: bool,
) -> dict[str, Any]:
    entry = trace.load_generator_entry(rank=rank, device=device)
    source = trace.source(rank=rank, device=device)
    before = TIMING_OMITTED if timing_mode else _local_trainable_fingerprint(generator_engine.module)
    x0, gradient_mask, rollout = run_generator_update_trace_rollout(
        engine=generator_engine,
        entry=entry,
        device=device,
        detailed_profile=False,
    )
    gradient, losses = _gan_score_boundary_gradient(
        score_model=critic_engine.module,
        generated_latent=x0,
        gan_g_weight=float(cfg.training_config.gan_g_weight),
        official_kwargs=score_official_kwargs(
            cfg,
            entry.operation_trace,
            source,
            1000,
            negative,
            device,
        ),
    )
    # The Score helper already returned the normalized latent derivative.  Use
    # a SUM chain-rule surrogate, matching the validated GAN pipeline seam.
    surrogate = (x0.float() * gradient.float()).sum()
    generator_engine.backward(surrogate)
    generator_engine.step()
    after = TIMING_OMITTED if timing_mode else _local_trainable_fingerprint(generator_engine.module)
    if not timing_mode and before == after:
        raise AssertionError("GAN FSDP2 Generator trainables did not update")
    return {
        "logical_step": 1000,
        "source_rank": rank,
        "physical_generator_step_after": int(generator_engine.global_steps),
        "local_trainable_before": before,
        "local_trainable_after": after,
        "rollout": rollout,
        "score_forwards": 4,
        "losses": TIMING_OMITTED if timing_mode else losses,
    }


def _run_nogan_critic(
    *,
    rank: int,
    step: int,
    device: torch.device,
    generator_engine: Any,
    critic_engine: Any,
    trace: StaticWorld16Trace,
    timing_mode: bool,
) -> dict[str, Any]:
    entry = trace.load_critic_entry(rank=rank, step=step, device=device)
    x0, rollout = run_generator_trace_rollout(
        model=generator_engine.module,
        loaded=entry,
        device=device,
        detailed_profile=False,
        record_output_sha256=not timing_mode,
        empty_cache_before_rollout=False,
    )
    batch = prepare_student_batch(generated_x0=x0, loaded=entry, device=device)
    _prediction, loss = student_forward_loss(critic_engine, batch)
    critic_engine.backward(loss)
    critic_engine.step()
    return {
        "logical_step": step,
        "source_rank": rank,
        "physical_critic_step_after": int(critic_engine.global_steps),
        "loss": TIMING_OMITTED if timing_mode else float(loss.detach().cpu()),
        "rollout": rollout,
        "score_forwards": 1,
    }


def _run_gan_critic(
    *,
    rank: int,
    step: int,
    device: torch.device,
    generator_engine: Any,
    critic_engine: Any,
    trace: StaticWorld16Trace,
    cfg: Any,
    timing_mode: bool,
) -> dict[str, Any]:
    from helios.utils.utils_helios_post import _critic_loss

    entry = trace.load_critic_entry(rank=rank, step=step, device=device)
    x0, rollout = run_generator_trace_rollout(
        model=generator_engine.module,
        loaded=entry,
        device=device,
        detailed_profile=False,
        record_output_sha256=not timing_mode,
        empty_cache_before_rollout=False,
    )
    source = trace.source(rank=rank, device=device)
    kwargs = critic_official_kwargs(
        cfg,
        entry.operation_trace,
        source,
        step,
        x0,
        critic_engine,
    )
    before = int(critic_engine.global_steps)
    loss, logs = _critic_loss(fake_score_model=critic_engine.module, **kwargs)
    if not bool(logs.get("critic_optimizer_committed")):
        raise AssertionError("GAN static FSDP2 Critic did not commit")
    if int(critic_engine.global_steps) != before + 1:
        raise AssertionError("GAN static FSDP2 Critic optimizer step drift")
    return {
        "logical_step": step,
        "source_rank": rank,
        "physical_critic_step_after": int(critic_engine.global_steps),
        "loss": TIMING_OMITTED if timing_mode else float(loss.detach().cpu()),
        "rollout": rollout,
        "score_forwards": 2,
        "critic_optimizer_committed": True,
    }


def _nvtx_push(args: argparse.Namespace) -> None:
    if args.profile_mode:
        torch.cuda.nvtx.range_push(f"STATIC_FSDP2_FULL5/{args.variant}")


def _nvtx_pop(args: argparse.Namespace) -> None:
    if args.profile_mode:
        torch.cuda.nvtx.range_pop()


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--variant", choices=("no-gan", "gan"), required=True)
    value.add_argument("--run-id", required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--logical-package", type=Path, required=True)
    value.add_argument("--expected-logical-hash")
    value.add_argument("--trace-bundle", type=Path, required=True)
    value.add_argument("--expected-trace-hash")
    value.add_argument("--generator-root", type=Path, required=True)
    value.add_argument("--precast-generator-root", type=Path, required=True)
    value.add_argument("--score-root", type=Path, required=True)
    value.add_argument("--precast-score-root", type=Path, required=True)
    value.add_argument("--negative-prompt", type=Path, required=True)
    value.add_argument("--gan-config", type=Path)
    value.add_argument("--timing-mode", action="store_true")
    value.add_argument("--profile-mode", action="store_true")
    value.add_argument("--validate-only", action="store_true")
    return value


def main() -> int:
    args = parser().parse_args()
    if args.timing_mode and args.profile_mode:
        raise ValueError("profiled runs are diagnostic and cannot be formal timing runs")
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    if world != WORLD_SIZE:
        raise AssertionError(f"static FSDP2 baseline requires world16, got {world}")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    output = args.output_dir / "ranks" / f"rank_{rank:02d}.json"
    result: dict[str, Any] = {
        "schema_version": "distillschedule.static_fsdp2_full5_rank.v1",
        "status": "RUNNING",
        "run_id": args.run_id,
        "variant": args.variant,
        "rank": rank,
        "local_rank": local_rank,
        "hostname": socket.gethostname(),
        "world_size": world,
        "timing_mode": bool(args.timing_mode),
        "profile_mode": bool(args.profile_mode),
    }
    try:
        contract = contract_for_variant(args.variant)
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
        result["contract"] = dataclasses_asdict(contract)
        result["identities"] = {
            "logical_manifest_hash": logical.manifest_hash,
            "trace_bundle_hash": trace.bundle_hash,
            "trace_binding_hash": trace.binding_hash,
            "rank_control_validation": trace.rank_control_validation,
        }
        if args.validate_only:
            result.update(status="PASS", validate_only=True)
            _atomic_json(output, result)
            dist.barrier()
            return 0
        if args.variant == "gan" and args.gan_config is None:
            raise ValueError("GAN variant requires --gan-config")
        negative = NegativePromptAsset.load(args.negative_prompt).embedding.to(
            device=device, dtype=torch.bfloat16
        )
        cfg = load_official_config(args.gan_config) if args.variant == "gan" else None
        generator_engine, critic_engine, build = _build_engines(
            args=args, package=logical, device=device
        )
        ema = FSDP2ShardEMA.load_from_logical(generator_engine.module, logical.root)
        result["build"] = build
        result["initial"] = {
            "generator_physical_step": int(generator_engine.global_steps),
            "critic_physical_step": int(critic_engine.global_steps),
            "ema_step": int(ema.optimization_step),
            "memory": _memory(device),
        }
        dist.barrier()
        torch.cuda.synchronize(device)
        started_perf = time.perf_counter()
        result["timing_start_unix_ns"] = time.time_ns()
        _nvtx_push(args)
        generator_record = None
        critic_records: list[dict[str, Any]] = []
        ema_records: list[dict[str, Any]] = []
        for index, step in enumerate(contract.logical_steps):
            if index == 0:
                generator_record = (
                    _run_gan_generator_update(
                        args=args,
                        rank=rank,
                        device=device,
                        generator_engine=generator_engine,
                        critic_engine=critic_engine,
                        trace=trace,
                        cfg=cfg,
                        negative=negative,
                        timing_mode=args.timing_mode,
                    )
                    if args.variant == "gan"
                    else _run_nogan_generator_update(
                        rank=rank,
                        device=device,
                        generator_engine=generator_engine,
                        critic_engine=critic_engine,
                        trace=trace,
                        negative=negative,
                        timing_mode=args.timing_mode,
                    )
                )
            critic_record = (
                _run_gan_critic(
                    rank=rank,
                    step=step,
                    device=device,
                    generator_engine=generator_engine,
                    critic_engine=critic_engine,
                    trace=trace,
                    cfg=cfg,
                    timing_mode=args.timing_mode,
                )
                if args.variant == "gan"
                else _run_nogan_critic(
                    rank=rank,
                    step=step,
                    device=device,
                    generator_engine=generator_engine,
                    critic_engine=critic_engine,
                    trace=trace,
                    timing_mode=args.timing_mode,
                )
            )
            critic_records.append(critic_record)
            if step in contract.ema_update_steps:
                ema_records.append(
                    {
                        "logical_step": step,
                        "decay": ema.step(),
                        "optimization_step": int(ema.optimization_step),
                    }
                )
        _nvtx_pop(args)
        torch.cuda.synchronize(device)
        result["timing_stop_unix_ns"] = time.time_ns()
        result["full5_seconds"] = time.perf_counter() - started_perf
        expected_g = int(logical.component("generator")["optimizer_step"]) + 1
        expected_c = int(logical.component("critic")["optimizer_step"]) + 5
        if int(generator_engine.global_steps) != expected_g:
            raise AssertionError("static FSDP2 Generator update count changed")
        if int(critic_engine.global_steps) != expected_c:
            raise AssertionError("static FSDP2 Critic update count changed")
        if int(ema.optimization_step) != int(contract.final_versions["ema"]):
            raise AssertionError("static FSDP2 EMA update count changed")
        result.update(
            status="PASS",
            generator=generator_record,
            critic=critic_records,
            ema=ema_records,
            updates={"generator": 1, "critic": 5, "ema": len(ema_records)},
            final={
                "generator_physical_step": int(generator_engine.global_steps),
                "critic_physical_step": int(critic_engine.global_steps),
                "generator_logical_version": int(contract.final_versions["generator"]),
                "critic_logical_version": int(contract.final_versions["critic"]),
                "ema_step": int(ema.optimization_step),
                "ema_local_checksum": (
                    TIMING_OMITTED if args.timing_mode else ema.checksum()
                ),
                "memory": _memory(device),
            },
            formal_timer_scope=(
                "immediately_before_step0_model_compute_through_final_critic_commit_and_required_ema"
            ),
            transition_count=0,
            model_swap_count=0,
            parameter_streaming_count=0,
        )
        _atomic_json(output, result)
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
            result["memory_failure"] = _memory(device)
        except BaseException:
            pass
        _atomic_json(output, result)
        raise
    finally:
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except BaseException:
                pass


def dataclasses_asdict(value: Any) -> dict[str, Any]:
    import dataclasses

    return dataclasses.asdict(value)


if __name__ == "__main__":
    raise SystemExit(main())
