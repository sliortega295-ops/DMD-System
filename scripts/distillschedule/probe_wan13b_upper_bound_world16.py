#!/usr/bin/env python3
"""Matched real-Wan1.3B resident-vs-world16-FSDP2 No-GAN Full5 worker."""
from __future__ import annotations

import argparse
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

_P0_PATH = ROOT / "scripts/distillschedule/probe_static_fsdp2_p0_world16.py"
_SPEC = importlib.util.spec_from_file_location("wan13b_pinned_p0_worker", _P0_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot import {_P0_PATH}")
p0 = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = p0
_SPEC.loader.exec_module(p0)

from distillschedule.profiling.static_profile_diagnostics import StaticProfileDiagnostics
from distillschedule.runtime.fsdp2_p0 import StaticFSDP2P0Policy, apply_blockwise_fsdp2_policy
from distillschedule.runtime.fsdp2_static import configure_fsdp2_checkpointing, contract_for_variant
from distillschedule.runtime.nsys_range import StaticFSDP2Full5NsightRange
from distillschedule.runtime.wan13b_upper_bound import (
    FreshTrainableEMA,
    PLACEMENTS,
    WAN13B_BLOCK_COUNT,
    WAN13B_PYRAMID_PATCH_SEED,
    WORLD_SIZE,
    Wan13BBootstrap,
    build_fresh_engine,
    file_sha256,
    model_trainable_metrics,
)
from runtime.critic_pool_helios import build_helios_pool_model
from runtime.gst_pool_contract import NegativePromptAsset
from runtime.official_baseline import suspend_transformers_zero3_init, trim_host_allocator
from runtime.static_world16_trace import StaticWorld16Trace


TIMING_OMITTED = "omitted_in_timing_mode"


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _memory(device: torch.device) -> dict[str, int]:
    free, total = torch.cuda.mem_get_info(device)
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "driver_free_bytes": int(free),
        "driver_total_bytes": int(total),
    }


def _tensor_metrics(value: torch.Tensor) -> dict[str, Any]:
    tensor = value.detach().float()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "finite": bool(torch.isfinite(tensor).all().cpu()),
        "sum": float(tensor.double().sum().cpu()),
        "l1": float(tensor.double().abs().sum().cpu()),
        "l2": float(tensor.double().square().sum().sqrt().cpu()),
        "max_abs": float(tensor.abs().max().cpu()),
        "mean": float(tensor.mean().cpu()),
    }


def _save_rollout_evidence(
    *, output_dir: Path, rank: int, values: dict[int, torch.Tensor]
) -> dict[str, Any]:
    from safetensors.torch import save_file

    path = output_dir / "correctness" / f"rank_{rank:02d}_critic_rollouts.safetensors"
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        f"step_{step}": value.detach().cpu().contiguous()
        for step, value in sorted(values.items())
    }
    save_file(
        state,
        str(path),
        metadata={"rank": str(rank), "purpose": "cross-placement-parity"},
    )
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
        "tensor_keys": list(state),
    }


def _save_trainable_evidence(
    *, output_dir: Path, rank: int, component: str, module: torch.nn.Module
) -> dict[str, Any]:
    """Gather one trainable tensor at a time and persist only on rank0."""

    from safetensors.torch import save_file

    state: dict[str, torch.Tensor] = {}
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        if hasattr(parameter, "full_tensor"):
            full = parameter.detach().full_tensor()
        else:
            full = parameter.detach()
        if rank == 0:
            state[name] = full.cpu().float().contiguous()
        del full
    if rank != 0:
        return {"writer_rank": 0, "status": "PARTICIPATED_IN_GATHER"}
    path = output_dir / "correctness" / f"final_{component}_trainables.safetensors"
    save_file(
        state,
        str(path),
        metadata={"component": component, "purpose": "cross-placement-final-state-parity"},
    )
    return {
        "writer_rank": 0,
        "status": "PASS",
        "path": str(path),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
        "tensor_count": len(state),
    }


def _save_optimizer_evidence(
    *, output_dir: Path, rank: int, component: str, engine: Any
) -> dict[str, Any]:
    from safetensors.torch import save_file

    trainables = {
        name: parameter
        for name, parameter in engine.module.named_parameters()
        if parameter.requires_grad
    }
    state: dict[str, torch.Tensor] = {}
    sharded_evidence = getattr(engine, "full_optimizer_state_evidence", None)
    if callable(sharded_evidence):
        values = sharded_evidence()
        for name in trainables:
            for key in ("exp_avg", "exp_avg_sq"):
                if rank == 0:
                    state[f"{key}/{name}"] = (
                        values[name][key].detach().cpu().float().contiguous()
                    )
    else:
        for name, parameter in trainables.items():
            optimizer_state = engine.optimizer.state[parameter]
            for key in ("exp_avg", "exp_avg_sq"):
                value = optimizer_state[key]
                if hasattr(value, "full_tensor"):
                    full = value.detach().full_tensor()
                else:
                    full = value.detach()
                if rank == 0:
                    state[f"{key}/{name}"] = full.cpu().float().contiguous()
                del full
    if rank != 0:
        return {"writer_rank": 0, "status": "PARTICIPATED_IN_GATHER"}
    path = output_dir / "correctness" / f"final_{component}_optimizer.safetensors"
    save_file(
        state,
        str(path),
        metadata={"component": component, "purpose": "cross-placement-adam-state-parity"},
    )
    return {
        "writer_rank": 0,
        "status": "PASS",
        "path": str(path),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
        "tensor_count": len(state),
        "optimizer_step": int(engine.global_steps),
        "scheduler_step": int(engine.lr_scheduler.last_epoch),
    }


def _build_model(
    *,
    role: str,
    args: argparse.Namespace,
    bootstrap: Wan13BBootstrap,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    seed = int(bootstrap.component(role)["seed"])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        with suspend_transformers_zero3_init():
            model, metadata = build_helios_pool_model(
                role=role,
                model_root=args.model_root,
                subfolder="transformer",
                precast_model_root=None,
                lora_rank=256,
                enable_gan=False,
                chunked_residual_storage="cuda_out_of_place",
                missing_pyramid_patch_seed=WAN13B_PYRAMID_PATCH_SEED,
            )
    if int(metadata["block_count"]) != WAN13B_BLOCK_COUNT:
        raise AssertionError("Wan1.3B worker requires exactly 30 Transformer blocks")
    restore = bootstrap.restore_trainables(model, component=role)
    if args.placement == "world16-fsdp":
        configure_fsdp2_checkpointing(model)
    model.to(device)
    torch.cuda.synchronize(device)
    if args.placement == "world16-fsdp":
        placement = apply_blockwise_fsdp2_policy(
            model,
            mesh_policy="world16-fsdp",
            world_size=WORLD_SIZE,
            expected_block_count=WAN13B_BLOCK_COUNT,
        )
    else:
        parameters = list(model.parameters())
        if any(hasattr(value, "device_mesh") for value in parameters):
            raise AssertionError("resident Wan1.3B control unexpectedly contains DTensors")
        if any(value.device != device for value in parameters):
            raise AssertionError("resident Wan1.3B has non-resident parameters")
        placement = {
            "backend": "FULL_REPLICA_GPU_RESIDENT",
            "world_size": WORLD_SIZE,
            "block_count": WAN13B_BLOCK_COUNT,
            "wrap_policy": "NONE",
            "reshard_after_forward": False,
            "cpu_parameter_offload": False,
            "parameter_streaming": False,
            "transition": False,
        }
    gc.collect()
    host_trimmed = trim_host_allocator()
    torch.cuda.empty_cache()
    return model, {
        "metadata": metadata,
        "bootstrap_restore": restore,
        "placement": placement,
        "host_allocator_trimmed": host_trimmed,
        "memory_after_placement": _memory(device),
    }


def _build_engines(
    *, args: argparse.Namespace, bootstrap: Wan13BBootstrap, device: torch.device
) -> tuple[Any, Any, FreshTrainableEMA, dict[str, Any]]:
    generator, generator_build = _build_model(
        role="generator", args=args, bootstrap=bootstrap, device=device
    )
    dist.barrier()
    critic, critic_build = _build_model(
        role="critic", args=args, bootstrap=bootstrap, device=device
    )
    dist.barrier()
    generator_engine, generator_engine_build = build_fresh_engine(
        module=generator,
        placement=args.placement,
        physical_step=int(bootstrap.component("generator")["optimizer_step"]),
        lr=2e-6,
        betas=(0.0, 0.999),
        eps=1e-8,
        weight_decay=1e-3,
        audit_gradients=not (args.timing_mode or args.profile_mode),
    )
    critic_engine, critic_engine_build = build_fresh_engine(
        module=critic,
        placement=args.placement,
        physical_step=int(bootstrap.component("critic")["optimizer_step"]),
        lr=4e-7,
        betas=(0.0, 0.999),
        eps=1e-8,
        weight_decay=1e-3,
        audit_gradients=not (args.timing_mode or args.profile_mode),
    )
    ema = FreshTrainableEMA(
        generator,
        optimization_step=int(bootstrap.manifest["ema"]["optimization_step"]),
        decay=float(bootstrap.manifest["ema"]["decay"]),
        update_after_step=int(bootstrap.manifest["ema"]["update_after_step"]),
    )
    return generator_engine, critic_engine, ema, {
        "generator": {**generator_build, "engine": generator_engine_build},
        "critic": {**critic_build, "engine": critic_engine_build},
        "static_residency": {
            "placement": args.placement,
            "both_backbones_present_for_full5": True,
            "parameter_offload": False,
            "activation_offload": False,
            "optimizer_offload": False,
            "frozen_parameter_streaming": False,
            "role_transition": False,
            "model_swap": False,
            "residual_storage": "cuda_out_of_place",
            "ema_device": "cuda",
        },
        "memory_after_optimizer_and_ema": _memory(device),
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--run-id", required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--placement", choices=PLACEMENTS, required=True)
    value.add_argument("--bootstrap", type=Path, required=True)
    value.add_argument("--expected-bootstrap-hash", required=True)
    value.add_argument("--model-root", type=Path, required=True)
    value.add_argument("--trace-bundle", type=Path, required=True)
    value.add_argument("--expected-trace-hash", required=True)
    value.add_argument("--negative-prompt", type=Path, required=True)
    value.add_argument("--validate-only", action="store_true")
    value.add_argument("--timing-mode", action="store_true")
    value.add_argument("--profile-mode", action="store_true")
    return value


def main() -> int:
    args = parser().parse_args()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    if world != WORLD_SIZE:
        raise AssertionError(f"Wan1.3B upper-bound experiment requires world16, got {world}")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    output = args.output_dir / "ranks" / f"rank_{rank:02d}.json"
    result: dict[str, Any] = {
        "schema_version": "distillschedule.wan13b_upper_bound_rank.v1",
        "status": "RUNNING",
        "run_id": args.run_id,
        "placement": args.placement,
        "rank": rank,
        "local_rank": local_rank,
        "hostname": socket.gethostname(),
        "world_size": world,
        "timing_mode": bool(args.timing_mode),
        "profile_mode": bool(args.profile_mode),
    }
    try:
        if args.timing_mode and args.profile_mode:
            raise ValueError("timing mode and profile mode are mutually exclusive")
        low_overhead = bool(args.timing_mode or args.profile_mode)
        clean_runtime = args.placement in {
            "resident-clean",
            "resident-async128",
            "resident-async128-tree",
            "resident-async128-tree-chunk2048",
            "resident-async128-tree-chunk2048-compile",
            "resident-async256",
            "resident-local-fast",
            "resident-flat256",
            "resident-flat512",
            "resident-lora-zero2",
            "resident-rollout-fused",
        }
        runtime_cache: dict[str, Any] | None = {} if clean_runtime else None
        critic_entries: dict[int, Any] | None = {} if clean_runtime else None
        deferred_finite_checks: list[torch.Tensor] | None = (
            [] if clean_runtime and low_overhead else None
        )
        contract = contract_for_variant("no-gan")
        bootstrap = Wan13BBootstrap.load(
            args.bootstrap,
            expected_hash=args.expected_bootstrap_hash,
            verify_payloads=rank == 0,
        )
        trace = StaticWorld16Trace(args.trace_bundle, variant="no-gan")
        if trace.bundle_hash != args.expected_trace_hash:
            raise ValueError("trace bundle identity changed")
        if Path(bootstrap.manifest["model_root"]).resolve() != args.model_root.resolve():
            raise ValueError("bootstrap/model-root path identity changed")
        result["contract"] = {
            "logical_steps": list(contract.logical_steps),
            "global_batch": contract.global_batch,
            "generator_updates": contract.generator_updates,
            "critic_updates": contract.critic_updates,
            "ema_update_steps": list(contract.ema_update_steps),
            "initial_versions": dict(contract.initial_versions),
            "final_versions": dict(contract.final_versions),
            "rollout_batch_size": 1,
            "score_packing": False,
        }
        result["identities"] = {
            "bootstrap_hash": bootstrap.manifest_hash,
            "model_identity_sha256": bootstrap.manifest["model_identity_sha256"],
            "trace_bundle_hash": trace.bundle_hash,
            "trace_binding_hash": trace.binding_hash,
            "rank_control_validation": trace.rank_control_validation,
        }
        if args.validate_only:
            result.update(status="PASS", validate_only=True)
            _atomic_json(output, result)
            dist.barrier()
            return 0

        negative = NegativePromptAsset.load(args.negative_prompt).embedding.to(
            device=device, dtype=torch.bfloat16
        )
        generator_engine, critic_engine, ema, build = _build_engines(
            args=args, bootstrap=bootstrap, device=device
        )
        policy = StaticFSDP2P0Policy(
            mesh_policy="world16-fsdp",
            rollout_batch_size=1,
            pack_real_score_pair=False,
            rollout_backend="native-fsdp2-batch",
        )
        diagnostics = StaticProfileDiagnostics(
            enabled=not low_overhead,
            rank=rank,
            output_path=None,
            collect_cpu_diagnostics=not low_overhead,
        )
        nsys_range = StaticFSDP2Full5NsightRange.from_env(
            variant="no-gan",
            stop_point="after_step14_ema_update",
        )
        if args.profile_mode and not nsys_range.enabled:
            raise ValueError("profile mode requires the exact Nsight range contract")
        result["build"] = build
        result["initial"] = {
            "generator_physical_step": int(generator_engine.global_steps),
            "critic_physical_step": int(critic_engine.global_steps),
            "ema_step": int(ema.optimization_step),
            "generator_trainable_metrics": (
                model_trainable_metrics(generator_engine.module, placement=args.placement)
                if not low_overhead
                else TIMING_OMITTED
            ),
            "critic_trainable_metrics": (
                model_trainable_metrics(critic_engine.module, placement=args.placement)
                if not low_overhead
                else TIMING_OMITTED
            ),
            "memory": _memory(device),
        }
        dist.barrier()
        torch.cuda.synchronize(device)
        if clean_runtime:
            # One cycle-level peak boundary replaces per-call resets.
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        started_ns = time.time_ns()
        if args.profile_mode:
            dist.barrier()
            nsys_range.before_model_compute(contract.logical_steps[0])
            if local_rank == 0:
                torch.cuda.nvtx.range_push(f"WAN13B_FULL5/{args.placement}")
            dist.barrier()

        if args.profile_mode and local_rank == 0:
            torch.cuda.nvtx.range_push("WAN13B/G_UPDATE")
        generator_record = p0._run_nogan_generator_update_p0(
            rank=rank,
            device=device,
            generator_engine=generator_engine,
            critic_engine=critic_engine,
            trace=trace,
            negative=negative,
            collect_diagnostics=not low_overhead,
            diagnostics=diagnostics,
            policy=policy,
            score_pack_parity_check=False,
            clean_runtime=clean_runtime,
            runtime_cache=runtime_cache,
            deferred_finite_checks=deferred_finite_checks,
        )
        if args.profile_mode and local_rank == 0:
            torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_push("WAN13B/G_ROLLOUTS")
        rollout_outputs, rollout_records = p0._precompute_critic_rollouts(
            rank=rank,
            steps=contract.logical_steps,
            device=device,
            generator_engine=generator_engine,
            trace=trace,
            policy=policy,
            collect_diagnostics=not low_overhead,
            diagnostics=diagnostics,
            rollout_parity_check=False,
            clean_runtime=clean_runtime,
            runtime_cache=runtime_cache,
            entry_outputs=critic_entries,
        )
        if args.profile_mode and local_rank == 0:
            torch.cuda.nvtx.range_pop()
        rollout_metrics = (
            {
                str(step): _tensor_metrics(rollout_outputs[step])
                for step in contract.logical_steps
            }
            if not low_overhead
            else TIMING_OMITTED
        )
        rollout_evidence = (
            _save_rollout_evidence(
                output_dir=args.output_dir, rank=rank, values=rollout_outputs
            )
            if not low_overhead
            else TIMING_OMITTED
        )
        critic_records: list[dict[str, Any]] = []
        ema_records: list[dict[str, Any]] = []
        for step in contract.logical_steps:
            if args.profile_mode and local_rank == 0:
                torch.cuda.nvtx.range_push(f"WAN13B/C_UPDATE/step_{step}")
            critic_records.append(
                p0._run_nogan_critic_precomputed(
                    rank=rank,
                    step=step,
                    x0=rollout_outputs[step],
                    device=device,
                    critic_engine=critic_engine,
                    trace=trace,
                    collect_diagnostics=not low_overhead,
                    diagnostics=diagnostics,
                    entry=(
                        critic_entries.get(step)
                        if critic_entries is not None
                        else None
                    ),
                    runtime_cache=runtime_cache,
                    deferred_finite_checks=deferred_finite_checks,
                )
            )
            if args.profile_mode and local_rank == 0:
                torch.cuda.nvtx.range_pop()
                torch.cuda.nvtx.range_push(f"WAN13B/EMA/step_{step}")
            decay = ema.step()
            ema_records.append(
                {
                    "logical_step": step,
                    "decay": decay,
                    "optimization_step": int(ema.optimization_step),
                }
            )
            del rollout_outputs[step]
            if args.profile_mode and local_rank == 0:
                torch.cuda.nvtx.range_pop()

        torch.cuda.synchronize(device)
        deferred_finite_status: bool | str = "NOT_USED"
        if deferred_finite_checks is not None:
            deferred_finite_status = bool(
                torch.stack(deferred_finite_checks).all().detach().cpu()
            )
            if not deferred_finite_status:
                raise AssertionError("resident-clean deferred cycle finite gate failed")
        dist.barrier()
        if args.profile_mode:
            if local_rank == 0:
                torch.cuda.nvtx.range_pop()
            nsys_range.after_critic_commit(contract.logical_steps[-1])
            dist.barrier()
            nsys_range.assert_complete()
        stopped_ns = time.time_ns()
        elapsed = time.perf_counter() - started
        expected_g = int(contract.final_versions["generator"])
        expected_c = int(contract.final_versions["critic"])
        expected_ema = int(contract.final_versions["ema"])
        if int(generator_engine.global_steps) != expected_g:
            raise AssertionError("Wan1.3B Generator update inventory changed")
        if int(critic_engine.global_steps) != expected_c:
            raise AssertionError("Wan1.3B Critic update inventory changed")
        if int(ema.optimization_step) != expected_ema:
            raise AssertionError("Wan1.3B EMA update inventory changed")
        if len(rollout_records) != 5 or any(
            int(value.get("batch_size", 1)) != 1 for value in rollout_records
        ):
            raise AssertionError("Wan1.3B rollout B1/physical inventory changed")

        final_generator_metrics = (
            model_trainable_metrics(generator_engine.module, placement=args.placement)
            if not low_overhead
            else TIMING_OMITTED
        )
        final_critic_metrics = (
            model_trainable_metrics(critic_engine.module, placement=args.placement)
            if not low_overhead
            else TIMING_OMITTED
        )
        final_trainable_evidence = (
            {
                "generator": _save_trainable_evidence(
                    output_dir=args.output_dir,
                    rank=rank,
                    component="generator",
                    module=generator_engine.module,
                ),
                "critic": _save_trainable_evidence(
                    output_dir=args.output_dir,
                    rank=rank,
                    component="critic",
                    module=critic_engine.module,
                ),
                "generator_optimizer": _save_optimizer_evidence(
                    output_dir=args.output_dir,
                    rank=rank,
                    component="generator",
                    engine=generator_engine,
                ),
                "critic_optimizer": _save_optimizer_evidence(
                    output_dir=args.output_dir,
                    rank=rank,
                    component="critic",
                    engine=critic_engine,
                ),
            }
            if not low_overhead
            else TIMING_OMITTED
        )

        result.update(
            status="PASS",
            timing_start_unix_ns=started_ns,
            timing_stop_unix_ns=stopped_ns,
            full5_seconds=elapsed,
            generator=generator_record,
            rollout_batches=rollout_records,
            rollout_metrics=rollout_metrics,
            rollout_evidence=rollout_evidence,
            critic=critic_records,
            ema=ema_records,
            updates={"generator": 1, "critic": 5, "ema": 5},
            final={
                "generator_physical_step": int(generator_engine.global_steps),
                "critic_physical_step": int(critic_engine.global_steps),
                "generator_logical_version": expected_g,
                "critic_logical_version": expected_c,
                "ema_step": int(ema.optimization_step),
                "generator_trainable_metrics": final_generator_metrics,
                "critic_trainable_metrics": final_critic_metrics,
                "trainable_evidence": final_trainable_evidence,
                "ema_checksum": ema.checksum() if not low_overhead else TIMING_OMITTED,
                "memory": _memory(device),
            },
            gradient_sync={
                "generator": getattr(generator_engine, "gradient_sync_receipts", []),
                "critic": getattr(critic_engine, "gradient_sync_receipts", []),
            },
            rollout_shadow={
                "enabled": bool(hasattr(generator_engine, "shadow_refresh_count")),
                "refresh_count": int(
                    getattr(generator_engine, "shadow_refresh_count", 0)
                ),
                "routed_call_count": int(
                    getattr(generator_engine, "shadow_routed_call_count", 0)
                ),
                "fused": bool(getattr(generator_engine, "_shadow_fused", False)),
            },
            formal_timer_scope=(
                "step10 Generator update; five B1 no-grad Generator rollouts; "
                "five Critic commits; five GPU EMA updates"
            ),
            parameter_offload_count=0,
            activation_offload_count=0,
            model_swap_count=0,
            parameter_streaming_count=0,
            resident_clean_runtime={
                "enabled": clean_runtime,
                "internal_cuda_housekeeping_skipped": clean_runtime,
                "trace_entries_reused": (
                    len(critic_entries) if critic_entries is not None else 0
                ),
                "runtime_cache_keys": (
                    sorted(runtime_cache) if runtime_cache is not None else []
                ),
                "deferred_finite_check_count": (
                    len(deferred_finite_checks)
                    if deferred_finite_checks is not None
                    else 0
                ),
                "deferred_finite_status": deferred_finite_status,
                "generator_model_state": (
                    generator_engine.module._distillschedule_resident_clean_state.receipt()
                    if clean_runtime
                    else None
                ),
                "critic_model_state": (
                    critic_engine.module._distillschedule_resident_clean_state.receipt()
                    if clean_runtime
                    else None
                ),
            },
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


if __name__ == "__main__":
    raise SystemExit(main())
