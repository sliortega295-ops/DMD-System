#!/usr/bin/env python3
"""Eight-rank real-14B Generator half of the N2 G/S/T capacity gate."""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from runtime.activation_offload import install_activation_offload_hooks
from runtime.critic_pool_checkpoint import PoolTargetPackage
from runtime.deepspeed_backend import (
    build_client_optimizer_offload_overrides,
    build_zero_optimization_config,
)
from runtime.dmd_boundary import detached_dmd_surrogate_loss
from runtime.frozen_block_streaming import install_frozen_block_streamer
from runtime.gst_pool_contract import GeneratorOffloadCandidate
from runtime.gst_pool_exchange import (
    atomic_torch_save,
    exchange_paths,
    make_input_payload,
    validate_gradient_payload,
    wait_for_file,
)
from runtime.gst_pool_helios import run_generator_update_trace_rollout
from runtime.gst_pool_trace import load_gst_trace_entry
from runtime.official_baseline import (
    optimizer_state_checksum,
    parameter_gradient_diagnostics,
    trainable_checksum,
)
from runtime.zero3_checkpoint import configure_block_gradient_checkpointing
from runtime.critic_pool_helios import build_helios_pool_model


def _parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--run-id", required=True)
    result.add_argument("--candidate-manifest", type=Path, required=True)
    result.add_argument("--contract-digest", required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--exchange-dir", type=Path, required=True)
    result.add_argument("--rendezvous-dir", type=Path, required=True)
    result.add_argument("--trace-bundle", type=Path, required=True)
    result.add_argument("--generator-root", type=Path, required=True)
    result.add_argument("--precast-generator-root", type=Path, required=True)
    result.add_argument("--target-package", type=Path, required=True)
    result.add_argument("--target-native-root", type=Path, required=True)
    result.add_argument("--mmap-cache-root", type=Path, required=True)
    result.add_argument("--timeout-seconds", type=float, default=7200.0)
    result.add_argument("--construction-wave", type=int, default=4)
    result.add_argument("--activation-min-bytes", type=int, default=32 * 2**20)
    result.add_argument("--detailed-profile", action="store_true")
    return result


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _snapshot(device: torch.device, stage: str) -> dict[str, Any]:
    free, total = torch.cuda.mem_get_info(device)
    return {
        "stage": stage,
        "time_unix": time.time(),
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "driver_free_bytes": int(free),
        "driver_total_bytes": int(total),
    }


def _resolved_int(engine: Any, name: str) -> int:
    value = getattr(engine, name)
    return int(value() if callable(value) else value)


def _optimizer_config() -> dict[str, Any]:
    zero = build_zero_optimization_config(
        stage=1,
        reduce_bucket_mib=64,
        allgather_bucket_mib=64,
        overlap_comm=False,
        contiguous_gradients=False,
        stage3_param_persistence_threshold=0,
        optimizer_offload_device="cpu",
        optimizer_offload_pin_memory=False,
    )
    value = {
        "train_batch_size": 16,
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 2,
        "fp16": {"enabled": False},
        "bf16": {"enabled": False},
        "communication_data_type": "fp32",
        "gradient_clipping": 1.0,
        "zero_allow_untested_optimizer": True,
        "zero_optimization": zero,
    }
    value.update(build_client_optimizer_offload_overrides("cpu"))
    return value


def _block_profile(event_path: Path, block_count: int) -> list[dict[str, Any]]:
    materialized: dict[int, list[dict[str, Any]]] = {
        block: [] for block in range(block_count)
    }
    released: dict[int, list[dict[str, Any]]] = {
        block: [] for block in range(block_count)
    }
    for line in event_path.read_text().splitlines():
        event = json.loads(line)
        if event.get("event") == "materialize_block":
            materialized[int(event["block_index"])].append(event)
        elif event.get("event") == "release_block":
            released[int(event["block_index"])].append(event)
    profile: list[dict[str, Any]] = []
    for block in range(block_count):
        mats = materialized[block]
        rels = released[block]
        if not mats:
            continue
        lifetime = 0.0
        for materialize, release in zip(mats, rels):
            lifetime += max(
                0.0,
                float(release["time_unix"]) - float(materialize["time_unix"]),
            )
        bytes_moved = max(int(item["bytes"]) for item in mats)
        pressure = max(
            (
                max(
                    0,
                    int(item.get("cuda_before_materialize_driver_free_bytes", 0))
                    - int(item.get("cuda_after_materialize_driver_free_bytes", 0)),
                )
                for item in mats
            ),
            default=0,
        )
        profile.append(
            {
                "block": block,
                "memory_relief_bytes": bytes_moved,
                "allocator_pressure_bytes": pressure,
                "lifetime_seconds": lifetime,
                "materialize_count": len(mats),
                "materialized_bytes_total": sum(int(item["bytes"]) for item in mats),
                "h2d_wall_seconds": sum(
                    float(item["elapsed_seconds"]) for item in mats
                ),
            }
        )
    return profile


def _ready(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(f"pid={os.getpid()}\ntime={time.time()}\n")
    os.replace(temporary, path)


def _construct_model(args: argparse.Namespace, rank: int):
    model = None
    if args.construction_wave <= 0 or 8 % args.construction_wave:
        raise ValueError("construction wave must divide eight ranks")
    for begin in range(0, 8, args.construction_wave):
        if begin <= rank < begin + args.construction_wave:
            model, metadata = build_helios_pool_model(
                role="generator",
                model_root=args.generator_root,
                subfolder="transformer_ode",
                precast_model_root=args.precast_generator_root,
            )
        dist.barrier()
    if model is None:
        raise AssertionError("Generator construction wave skipped a rank")
    return model, metadata


def main() -> int:
    args = _parser().parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    import deepspeed

    deepspeed.init_distributed(dist_backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    if world != 8 or rank != local_rank:
        raise RuntimeError("N2 Generator executor requires one local world8")
    candidate = GeneratorOffloadCandidate.load(
        args.candidate_manifest,
        expected_contract_digest=args.contract_digest,
    )
    if candidate.run_id != args.run_id:
        raise ValueError("Generator run ID differs from the immutable candidate")
    output = args.output_dir / "generator" / f"rank_{rank:02d}.json"
    result: dict[str, Any] = {
        "schema_version": "distillschedule.gst_pool_generator_rank.v1",
        "status": "RUNNING",
        "run_id": args.run_id,
        "rank": rank,
        "world_size": world,
        "candidate_manifest_sha256": candidate.manifest_sha256,
        "contract_digest": candidate.contract_digest,
        "streamed_blocks": list(candidate.streamed_blocks),
        "execution_graph": "target_no_gan_gst_pool",
        "memory": [],
        "waves": [],
    }
    streamer = None
    activation = None
    try:
        package = PoolTargetPackage.load(args.target_package)
        model, construction = _construct_model(args, rank)
        configure_block_gradient_checkpointing(
            model, zero_stage=1, policy="strict"
        )
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable,
            lr=2e-6,
            betas=(0.0, 0.999),
            eps=1e-8,
            weight_decay=1e-3,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        engine, _, _, _ = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            config=_optimizer_config(),
            dist_init_required=False,
        )
        resolved = {
            "dp_world_size": _resolved_int(engine, "dp_world_size"),
            "train_micro_batch_size_per_gpu": _resolved_int(
                engine, "train_micro_batch_size_per_gpu"
            ),
            "gradient_accumulation_steps": _resolved_int(
                engine, "gradient_accumulation_steps"
            ),
            "train_batch_size": _resolved_int(engine, "train_batch_size"),
        }
        if resolved != {
            "dp_world_size": 8,
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 2,
            "train_batch_size": 16,
        }:
            raise AssertionError(f"resolved Generator batch contract changed: {resolved}")
        checkpoint_root = args.target_native_root / "checkpoint-11"
        load_path, client_state = engine.load_checkpoint(
            str(checkpoint_root),
            tag="pytorch_model",
            load_module_strict=True,
            load_optimizer_states=True,
            load_lr_scheduler_states=False,
        )
        if load_path is None:
            raise RuntimeError("native world8 G3 restore failed")
        engine.lr_scheduler.load_state_dict(
            dict(package.component("generator")["scheduler_state"])
        )
        optimizer_before = optimizer_state_checksum(engine.optimizer)
        if int(engine.global_steps) != 3 or optimizer_before.get("step_min") != 3:
            raise AssertionError(
                f"Generator native state is not G3: steps={engine.global_steps}/{optimizer_before}"
            )
        result["initialization"] = {
            "construction": construction,
            "deepspeed_version": deepspeed.__version__,
            "resolved": resolved,
            "load_path": str(load_path),
            "client_state_keys": sorted((client_state or {}).keys()),
            "generator_version": 3,
            "optimizer": optimizer_before,
            "scheduler_last_epoch": int(engine.lr_scheduler.last_epoch),
        }
        result["memory"].append(_snapshot(device, "native_g3_loaded"))

        event_root = args.output_dir / "events"
        event_root.mkdir(parents=True, exist_ok=True)
        stream_event = event_root / f"generator_rank_{rank:02d}_blocks.jsonl"
        if candidate.streamed_blocks:
            streamer = install_frozen_block_streamer(
                engine.module,
                logical_component="generator",
                device=device,
                window_size=1,
                streamed_block_count=len(candidate.streamed_blocks),
                streamed_block_indices=candidate.streamed_blocks,
                host_policy="DROP_RELOAD_FROM_MMAP",
                optimizer_owned_parameter_ids={
                    id(parameter)
                    for parameter in engine.module.parameters()
                    if parameter.requires_grad
                },
                checkpoint_identity=(
                    "helios-distilled-frozen-backbone:"
                    + str(construction["precast_manifest_sha256"])
                ),
                event_path=stream_event,
                mmap_cache_root=args.mmap_cache_root,
                detailed_events=True,
                memory_snapshot_events=True,
            )
        providers = (
            ()
            if streamer is None
            else (streamer.dynamic_excluded_storage_owner,)
        )
        activation = install_activation_offload_hooks(
            engine.module,
            logical_component="generator_selected",
            event_path=event_root
            / f"generator_rank_{rank:02d}_activation.jsonl",
            pin_memory=True,
            min_offload_bytes=args.activation_min_bytes,
            detailed_events=False,
            dynamic_excluded_storage_providers=providers,
        )
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        before = trainable_checksum(engine.module)
        result["memory"].append(_snapshot(device, "streaming_runtime_ready"))
        dist.barrier()
        if rank == 0:
            _ready(args.rendezvous_dir / "G_READY")
        wait_for_file(args.rendezvous_dir / "ST_READY", args.timeout_seconds)
        dist.barrier()

        states: list[dict[str, Any]] = []
        for wave in range(2):
            source_id = wave * 8 + rank
            entry = load_gst_trace_entry(
                args.trace_bundle, source_id=source_id, device=device
            )
            started = time.perf_counter()
            x0, gradient_mask, details = run_generator_update_trace_rollout(
                engine=engine,
                entry=entry,
                device=device,
                detailed_profile=args.detailed_profile,
            )
            payload = make_input_payload(
                run_id=args.run_id,
                source_id=source_id,
                generator_version=3,
                trace_identity_sha256=entry.identity_sha256,
                x0=x0,
            )
            paths = exchange_paths(args.exchange_dir, source_id)
            atomic_torch_save(paths["input"], payload)
            record = {
                "source_id": source_id,
                "wave": wave,
                "trace_identity_sha256": entry.identity_sha256,
                "operation_trace_hash": entry.operation_trace.trace_hash,
                "x0_sha256": payload["x0_sha256"],
                "rollout_seconds": time.perf_counter() - started,
                "rollout": details,
                "input_path": str(paths["input"]),
                "input_file_bytes": paths["input"].stat().st_size,
            }
            result["waves"].append(record)
            states.append(
                {
                    "source_id": source_id,
                    "x0": x0,
                    "gradient_mask": gradient_mask,
                    "trace_identity_sha256": entry.identity_sha256,
                    "x0_sha256": payload["x0_sha256"],
                    "gradient_path": paths["gradient"],
                    "record": record,
                }
            )
            del entry, payload

        global_before = int(engine.global_steps)
        boundaries = 0
        for state in states:
            source_id = int(state["source_id"])
            waited = wait_for_file(
                state["gradient_path"], args.timeout_seconds
            )
            response = torch.load(
                state["gradient_path"], map_location="cpu", weights_only=False
            )
            gradient_cpu = validate_gradient_payload(
                response,
                run_id=args.run_id,
                source_id=source_id,
                trace_identity_sha256=state["trace_identity_sha256"],
                x0_sha256=state["x0_sha256"],
            )
            gradient = gradient_cpu.to(device=device, dtype=torch.bfloat16)
            loss = detached_dmd_surrogate_loss(
                state["x0"], gradient, state["gradient_mask"]
            )
            state["record"]["dmd"] = {
                "gradient_sha256": response["input_gradient_sha256"],
                "gradient_wait_seconds": waited,
                "loss": float(loss.detach().cpu()),
            }
            dist.barrier()
            started = time.perf_counter()
            engine.backward(loss)
            boundary = bool(engine.is_gradient_accumulation_boundary())
            boundaries += int(boundary)
            engine.step()
            torch.cuda.synchronize(device)
            state["record"]["backward_step_seconds"] = time.perf_counter() - started
            state["record"]["gradient_accumulation_boundary"] = boundary
            state["x0"] = None
            state["gradient_mask"] = None
            del response, gradient_cpu, gradient, loss

        if int(engine.global_steps) != global_before + 1 or boundaries != 1:
            raise AssertionError(
                "Step15-A must perform exactly one Generator optimizer update"
            )
        after = trainable_checksum(engine.module)
        if after["sum"] == before["sum"] and after["norm"] == before["norm"]:
            raise AssertionError("Generator G3 did not advance to G4")
        gradients = parameter_gradient_diagnostics(engine.module)
        if gradients["frozen_grad_present"] != 0:
            raise AssertionError("frozen Generator backbone acquired gradients")
        if streamer is not None:
            streamer.assert_idle()
        activation.offloader.assert_scope_closed()
        result["final"] = {
            "generator_version_before": 3,
            "generator_version_after": 4,
            "global_steps_before": global_before,
            "global_steps_after": int(engine.global_steps),
            "accumulation_boundaries": boundaries,
            "trainable_before": before,
            "trainable_after": after,
            "gradient_diagnostics": gradients,
            "optimizer": optimizer_state_checksum(engine.optimizer),
            "scheduler_last_epoch": int(engine.lr_scheduler.last_epoch),
            "activation_stats": dict(activation.offloader.stats),
            "streamer_stats": None if streamer is None else dict(streamer.stats),
        }
        if streamer is not None:
            result["block_profile"] = _block_profile(
                stream_event, len(engine.module.blocks)
            )
        else:
            result["block_profile"] = []
        result["memory"].append(_snapshot(device, "complete"))
        result["status"] = "PASS"
        _atomic_json(output, result)
        dist.barrier()
        return 0
    except BaseException as exc:
        is_oom = isinstance(exc, torch.cuda.OutOfMemoryError) or (
            "out of memory" in str(exc).lower()
        )
        result.update(
            {
                "status": "OOM" if is_oom else "FAIL",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        with contextlib.suppress(BaseException):
            result["memory"].append(_snapshot(device, "failure"))
        _atomic_json(output, result)
        return 42 if is_oom else 1
    finally:
        with contextlib.suppress(BaseException):
            if activation is not None:
                activation.remove()
        with contextlib.suppress(BaseException):
            if streamer is not None:
                streamer.remove()
        with contextlib.suppress(BaseException):
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
