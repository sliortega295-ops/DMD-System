#!/usr/bin/env python3
"""Continuous real-14B Route-B Candidate-A Full5 executor.

The process graph remains alive across Step10-A, the 8G+4S+4T -> 8G+8C
transition, Step10-B, and Critic-only steps 11--14.  Sequential and Pipeline
share every model, trace, transport, loss, optimizer and EMA primitive; only
the legal production/consumption order differs.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import resource
import socket
import sys
import time
import traceback
from datetime import timedelta
from pathlib import Path
from typing import Any

import psutil
import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.activation_offload import install_activation_offload_hooks
from runtime.critic_pool import (
    ChecksumKind,
    CriticPoolTopology,
    CriticWaveHeader,
    StudentDPGroupAdapter,
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
from runtime.critic_pool_transport import (
    CriticPoolP2PTransport,
    prepare_transport_payload,
    tensor_identity,
)
from runtime.deepspeed_backend import (
    build_client_optimizer_offload_overrides,
    build_zero_optimization_config,
)
from runtime.dmd_boundary import detached_dmd_surrogate_loss
from runtime.frozen_block_streaming import install_frozen_block_streamer
from runtime.gst_pool_contract import NegativePromptAsset
from runtime.gst_pool_helios import (
    construct_dmd_gradient,
    prepare_dmd_score_batch,
    run_generator_update_trace_rollout,
    score_forward,
)
from runtime.official_baseline import (
    TrainableGradientObserver,
    optimizer_state_checksum,
    parameter_gradient_diagnostics,
    trainable_checksum,
    trainable_sha256,
)
from runtime.operation_trace import tensor_sha256
from runtime.routeb_full5 import RouteBFull5Contract
from runtime.routeb_full5_trace import RouteBFull5Trace
from runtime.routeb_full5_transport import (
    RouteBWaveAck,
    assert_ack_matches_header,
    receive_ack,
    send_ack,
)
from runtime.sparse_ema import SparseShardEMA
from runtime.zero3_checkpoint import configure_block_gradient_checkpointing


G_RANKS = tuple(range(8))
C_RANKS = tuple(range(8, 16))
STUDENT_STEP10A_RANKS = tuple(range(8, 12))
TEACHER_STEP10A_RANKS = tuple(range(12, 16))
G4_BLOCKS = (0, 24, 1, 5)


def _step10a_model_roles(rank: int) -> tuple[str, str]:
    """Return physical construction role and logical Step10-A execution role.

    Real Score, Fake Score and Critic share one physical Score backbone in
    Helios.  Teacher execution therefore uses a Critic-capable Score object
    with adapters explicitly disabled, so the same object can become the
    resident Critic after Step10-A without constructing a second 14B model.
    """

    if rank in G_RANKS:
        return "generator", "generator"
    if rank in STUDENT_STEP10A_RANKS:
        return "critic", "student"
    if rank in TEACHER_STEP10A_RANKS:
        return "critic", "teacher"
    raise ValueError(f"rank {rank} is outside Candidate-A world16")


def _reactivate_critic_adapter(model: torch.nn.Module) -> dict[str, int]:
    """Restore Critic trainability after adapter-disabled Teacher inference."""

    if not bool(getattr(model, "_distillschedule_has_adapter", False)):
        raise AssertionError("physical Score is not Critic-capable")
    model.enable_adapters()
    trainables = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainables:
        raise AssertionError("Critic adapter reactivation produced no trainable tensors")
    return {
        "trainable_tensor_count": len(trainables),
        "trainable_numel": sum(parameter.numel() for parameter in trainables),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", choices=("sequential", "pipeline"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-nonce", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trace-bundle", type=Path, required=True)
    parser.add_argument("--logical-package", type=Path, required=True)
    parser.add_argument("--native-world8-root", type=Path, required=True)
    parser.add_argument("--generator-root", type=Path, required=True)
    parser.add_argument("--precast-generator-root", type=Path, required=True)
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--precast-score-root", type=Path, required=True)
    parser.add_argument("--negative-prompt-asset", type=Path, required=True)
    parser.add_argument("--mmap-cache-root", type=Path, required=True)
    parser.add_argument("--construction-wave", type=int, default=4)
    parser.add_argument("--activation-min-bytes", type=int, default=32 * 2**20)
    parser.add_argument("--checksum", choices=("sha256", "gpu"), default="sha256")
    parser.add_argument("--timing-mode", action="store_true")
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--save-transition-checkpoint", action="store_true")
    parser.add_argument("--resume-transition-checkpoint", type=Path)
    parser.add_argument("--expected-transition-manifest-sha256")
    parser.add_argument("--timing-correctness-evaluation", type=Path)
    parser.add_argument("--expected-timing-correctness-sha256")
    parser.add_argument("--stop-after-transition", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=7200.0)
    return parser


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _atomic_torch(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


TIMING_OMITTED = "omitted_in_low_overhead_timing_mode"


def _timing_correctness_binding(args: argparse.Namespace) -> dict[str, Any] | None:
    """Bind low-overhead critic timing to the already-passed correctness run."""

    critic_timing = bool(args.timing_mode and args.resume_transition_checkpoint)
    supplied = (
        args.timing_correctness_evaluation is not None
        or args.expected_timing_correctness_sha256 is not None
    )
    if not critic_timing:
        if supplied:
            raise ValueError(
                "timing correctness binding is only legal for transition-resumed timing"
            )
        return None
    path = args.timing_correctness_evaluation
    expected_sha = args.expected_timing_correctness_sha256
    if path is None or not expected_sha:
        raise ValueError("critic timing requires a frozen correctness evaluation")
    if not path.is_file():
        raise FileNotFoundError(path)
    observed_sha = _sha256_file(path)
    if observed_sha != expected_sha:
        raise ValueError("timing correctness evaluation SHA-256 changed")
    evaluation = json.loads(path.read_text())
    if evaluation.get("status") != "PASS" or evaluation.get("errors") != []:
        raise ValueError("timing correctness evaluation is not PASS")
    if (
        evaluation.get("contract_transition_manifest_sha256")
        != args.expected_transition_manifest_sha256
    ):
        raise ValueError("timing correctness transition identity changed")
    return {
        "schema_version": "distillschedule.routeb_timing_correctness_binding.v1",
        "path": str(path.resolve()),
        "sha256": observed_sha,
        "status": "PASS",
        "reference_run": evaluation.get("reference_run"),
        "candidate_run": evaluation.get("candidate_run"),
        "transition_manifest_sha256": evaluation.get(
            "contract_transition_manifest_sha256"
        ),
    }


def _optional_trainable_sha(model: torch.nn.Module, *, timing_mode: bool) -> str:
    return TIMING_OMITTED if timing_mode else trainable_sha256(model)


def _optional_trainable_checksum(
    model: torch.nn.Module, *, timing_mode: bool
) -> dict[str, Any] | str:
    return TIMING_OMITTED if timing_mode else trainable_checksum(model)


def _optional_optimizer_checksum(
    optimizer: Any, *, timing_mode: bool
) -> dict[str, Any] | str:
    return TIMING_OMITTED if timing_mode else optimizer_state_checksum(optimizer)


def _wait(path: Path, timeout_seconds: float) -> float:
    started = time.monotonic()
    while not path.is_file():
        if time.monotonic() - started >= timeout_seconds:
            raise TimeoutError(path)
        time.sleep(0.05)
    return time.monotonic() - started


def _gib(value: int) -> float:
    return float(value) / float(2**30)


def _snapshot(device: torch.device, stage: str) -> dict[str, Any]:
    process = psutil.Process()
    usage = resource.getrusage(resource.RUSAGE_SELF)
    free, total = torch.cuda.mem_get_info(device)
    return {
        "stage": stage,
        "time_monotonic": time.perf_counter(),
        "rss_gib": _gib(process.memory_info().rss),
        "peak_rss_gib": _gib(int(usage.ru_maxrss) * 1024),
        "cuda": {
            "allocated_gib": _gib(torch.cuda.memory_allocated(device)),
            "reserved_gib": _gib(torch.cuda.memory_reserved(device)),
            "max_allocated_gib": _gib(torch.cuda.max_memory_allocated(device)),
            "max_reserved_gib": _gib(torch.cuda.max_memory_reserved(device)),
            "driver_free_gib": _gib(free),
            "driver_total_gib": _gib(total),
        },
    }


def _create_groups(topology: CriticPoolTopology):
    # The global control plane fences role-asymmetric host work (exact SHA,
    # EMA, checkpoint I/O), not CUDA payloads.  A NCCL barrier here can arm
    # the ten-minute watchdog while another role is still doing legitimate
    # host-side correctness work.  Keep all model/data collectives on NCCL,
    # but make these long-lived lifecycle fences explicit Gloo operations.
    control = dist.new_group(
        ranks=list(range(16)), backend="gloo", timeout=timedelta(hours=2)
    )
    generators = dist.new_group(ranks=list(G_RANKS), backend="nccl")
    critics = dist.new_group(ranks=list(C_RANKS), backend="nccl")
    step10a_students = dist.new_group(
        ranks=list(STUDENT_STEP10A_RANKS), backend="nccl"
    )
    step10a_teachers = dist.new_group(
        ranks=list(TEACHER_STEP10A_RANKS), backend="nccl"
    )
    pair_groups: dict[int, Any] = {}
    ack_groups: dict[int, Any] = {}
    for generator_rank, critic_rank in zip(G_RANKS, C_RANKS, strict=True):
        pair = dist.new_group(ranks=[generator_rank, critic_rank], backend="nccl")
        ack = dist.new_group(ranks=[generator_rank, critic_rank], backend="gloo")
        pair_groups[generator_rank] = pair_groups[critic_rank] = pair
        ack_groups[generator_rank] = ack_groups[critic_rank] = ack
    singleton = {
        rank: dist.new_group(ranks=[rank], backend="nccl") for rank in range(16)
    }
    return (
        control,
        generators,
        critics,
        step10a_students,
        step10a_teachers,
        pair_groups,
        ack_groups,
        singleton,
    )


def _control_barrier(control_group: Any) -> None:
    """Fence role transitions without occupying a CUDA/NCCL work slot."""

    dist.barrier(group=control_group)


def _zero1_config(*, dp_world_size: int, gas: int, lr: float) -> dict[str, Any]:
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
    config = {
        "train_batch_size": dp_world_size * gas,
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": gas,
        "fp16": {"enabled": False},
        "bf16": {"enabled": True},
        "communication_data_type": "fp32",
        "gradient_clipping": 1.0,
        "zero_allow_untested_optimizer": True,
        "zero_optimization": zero,
    }
    config.update(build_client_optimizer_offload_overrides("cpu"))
    return config


def _resolved_int(engine: Any, name: str) -> int:
    value = getattr(engine, name)
    return int(value() if callable(value) else value)


def _assert_engine_contract(
    engine: Any,
    *,
    dp: int,
    gas: int,
    batch: int,
    expected_dp_group: Any,
) -> dict[str, int]:
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
    expected = {
        "dp_world_size": dp,
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": gas,
        "train_batch_size": batch,
    }
    if resolved != expected:
        raise AssertionError(f"DeepSpeed batch contract changed: {resolved} != {expected}")
    if int(engine.zero_optimization_stage()) != 1:
        raise AssertionError("Route-B requires ZeRO-1")
    if bool(getattr(engine.optimizer, "partition_gradients", False)):
        raise AssertionError("Route-B accidentally enabled ZeRO-2 gradient partitioning")
    if getattr(engine.optimizer, "dp_process_group", None) is not expected_dp_group:
        raise AssertionError("ZeRO-1 optimizer escaped its role-local DP group")
    return resolved


def _optimizer_step(optimizer: Any) -> int:
    current = optimizer
    seen: set[int] = set()
    while not hasattr(current, "state") and hasattr(current, "optimizer"):
        if id(current) in seen:
            raise RuntimeError("optimizer wrapper cycle")
        seen.add(id(current))
        current = current.optimizer
    steps: set[int] = set()
    for value in current.state.values():
        if not isinstance(value, dict) or "step" not in value:
            continue
        step = value["step"]
        steps.add(int(step.item()) if isinstance(step, torch.Tensor) else int(step))
    if len(steps) != 1:
        raise AssertionError(f"expected one optimizer step, got {steps}")
    return next(iter(steps))


def _scheduler_step(engine: Any) -> int:
    return int(engine.lr_scheduler.state_dict()["last_epoch"])


def _checkpoint_parent(root: Path, component: str) -> Path:
    suffixes = (
        ("checkpoint-10", "critic") if component == "critic" else ("checkpoint-10",),
        ("critic",) if component == "critic" else (),
    )
    for suffix in suffixes:
        candidate = root.joinpath(*suffix)
        if (candidate / "pytorch_model").is_dir():
            return candidate
    raise FileNotFoundError(f"native world8 {component} checkpoint under {root}")


def _construct_models(
    args: argparse.Namespace,
    *,
    rank: int,
    local_rank: int,
    package: LogicalCheckpointPackage,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    if args.construction_wave <= 0 or 8 % args.construction_wave:
        raise ValueError("construction wave must divide eight local ranks")
    model = metadata = None
    for begin in range(0, 8, args.construction_wave):
        if begin <= local_rank < begin + args.construction_wave:
            role, execution_role = _step10a_model_roles(rank)
            if role == "generator":
                root = args.generator_root
                subfolder = "transformer_ode"
                precast = args.precast_generator_root
            else:
                root = args.score_root
                subfolder = "transformer"
                precast = args.precast_score_root
            model, metadata = build_helios_pool_model(
                role=role,
                model_root=root,
                subfolder=subfolder,
                precast_model_root=precast,
            )
            if role == "critic":
                metadata["logical_restore"] = load_logical_master_into_trainables(
                    model,
                    package,
                    component="critic",
                    verify_payload_sha256=(rank == 8),
                )
                metadata["execution_role"] = execution_role
                metadata["physical_score_reused_for_critic"] = True
            if role == "critic":
                model.to(device)
        dist.barrier()
    if model is None or metadata is None:
        raise AssertionError("construction wave skipped this rank")
    return model, metadata


def _initialize_generator_engine(
    args: argparse.Namespace,
    *,
    model: torch.nn.Module,
    package: LogicalCheckpointPackage,
    rank: int,
    group: Any,
    singleton: Any,
) -> tuple[Any, dict[str, Any]]:
    import deepspeed

    configure_block_gradient_checkpointing(model, zero_stage=1, policy="strict")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=2e-6, betas=(0.0, 0.999), eps=1e-8, weight_decay=1e-3
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    mpu = StudentDPGroupAdapter(
        data_parallel_group=group,
        data_parallel_ranks=G_RANKS,
        global_rank=rank,
        singleton_model_parallel_group=singleton,
    )
    engine, _, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        config=_zero1_config(dp_world_size=8, gas=2, lr=2e-6),
        mpu=mpu,
        dist_init_required=False,
    )
    resolved = _assert_engine_contract(
        engine,
        dp=8,
        gas=2,
        batch=16,
        expected_dp_group=group,
    )
    parent = _checkpoint_parent(args.native_world8_root, "generator")
    load_path, client_state = engine.load_checkpoint(
        str(parent),
        tag="pytorch_model",
        load_module_strict=True,
        load_optimizer_states=True,
        load_lr_scheduler_states=False,
    )
    if load_path is None:
        raise RuntimeError("native world8 G2 restore failed")
    engine.lr_scheduler.load_state_dict(
        dict(package.component("generator")["scheduler_state"])
    )
    if (
        int(engine.global_steps),
        _optimizer_step(engine.optimizer),
        _scheduler_step(engine),
    ) != (2, 2, 2):
        raise AssertionError("Generator did not restore G2")
    return engine, {
        "resolved": resolved,
        "load_path": str(load_path),
        "client_state_keys": sorted((client_state or {}).keys()),
        "optimizer": optimizer_state_checksum(engine.optimizer),
    }


def _input_path(root: Path, source_id: int) -> Path:
    return root / "step10a" / f"input_{source_id:02d}.pt"


def _fake_path(root: Path, source_id: int) -> Path:
    return root / "step10a" / f"fake_{source_id:02d}.pt"


def _gradient_path(root: Path, source_id: int) -> Path:
    return root / "step10a" / f"gradient_{source_id:02d}.pt"


def _validate_step10a_input(
    value: dict[str, Any], *, source_id: int, identity: str
) -> torch.Tensor:
    expected = {
        "schema_version": "distillschedule.routeb_step10a_input.v1",
        "logical_step": 10,
        "source_id": source_id,
        "generator_version": 2,
        "trace_identity_sha256": identity,
    }
    mismatch = {key: (value.get(key), item) for key, item in expected.items() if value.get(key) != item}
    if mismatch:
        raise ValueError(f"Step10-A input mismatch: {mismatch}")
    x0 = value.get("x0")
    if not isinstance(x0, torch.Tensor) or list(x0.shape) != [1, 16, 9, 48, 80]:
        raise ValueError("Step10-A x0 shape changed")
    if x0.dtype != torch.bfloat16 or tensor_sha256(x0) != value.get("x0_sha256"):
        raise ValueError("Step10-A x0 dtype/checksum changed")
    return x0


def _run_step10a_generator(
    args: argparse.Namespace,
    *,
    rank: int,
    device: torch.device,
    engine: Any,
    trace: RouteBFull5Trace,
    exchange: Path,
    result: dict[str, Any],
) -> None:
    records: list[dict[str, Any]] = []
    before = trainable_checksum(engine.module)
    global_before = int(engine.global_steps)
    boundaries = 0
    for production_round in range(2):
        source_id = production_round * 8 + rank
        entry = trace.load_step10a_entry(source_id=source_id, device=device)
        started = time.perf_counter()
        x0, gradient_mask, details = run_generator_update_trace_rollout(
            engine=engine,
            entry=entry,
            device=device,
            detailed_profile=not args.timing_mode,
        )
        x0_cpu = x0.detach().to("cpu").contiguous()
        payload = {
            "schema_version": "distillschedule.routeb_step10a_input.v1",
            "logical_step": 10,
            "source_id": source_id,
            "generator_version": 2,
            "trace_identity_sha256": entry.identity_sha256,
            "x0": x0_cpu,
            "x0_sha256": tensor_sha256(x0_cpu),
        }
        _atomic_torch(_input_path(exchange, source_id), payload)
        record = {
            "source_id": source_id,
            "trace_identity_sha256": entry.identity_sha256,
            "x0_sha256": payload["x0_sha256"],
            "rollout_seconds": time.perf_counter() - started,
            "rollout": details,
        }
        # Consume this detached gradient before creating the next selected
        # graph.  GAS2 is an optimizer/update contract, not permission to keep
        # two 14B activation graphs live at once.  This preserves the two
        # micro-batch accumulation semantics while bounding GPU memory.
        waited = _wait(_gradient_path(exchange, source_id), args.timeout_seconds)
        gradient_payload = torch.load(
            _gradient_path(exchange, source_id),
            map_location="cpu",
            weights_only=False,
        )
        expected = {
            "schema_version": "distillschedule.routeb_step10a_gradient.v1",
            "logical_step": 10,
            "source_id": source_id,
            "generator_version": 2,
            "critic_version": 10,
            "trace_identity_sha256": entry.identity_sha256,
            "x0_sha256": payload["x0_sha256"],
        }
        mismatch = {
            key: (gradient_payload.get(key), item)
            for key, item in expected.items()
            if gradient_payload.get(key) != item
        }
        if mismatch:
            raise ValueError(f"Step10-A gradient identity mismatch: {mismatch}")
        gradient_cpu = gradient_payload.get("gradient")
        if (
            not isinstance(gradient_cpu, torch.Tensor)
            or tensor_sha256(gradient_cpu)
            != gradient_payload.get("gradient_sha256")
        ):
            raise ValueError("Step10-A detached gradient checksum changed")
        gradient = gradient_cpu.to(device=device, dtype=torch.bfloat16)
        loss = detached_dmd_surrogate_loss(x0, gradient, gradient_mask)
        started = time.perf_counter()
        engine.backward(loss)
        boundary = bool(engine.is_gradient_accumulation_boundary())
        boundaries += int(boundary)
        engine.step()
        torch.cuda.synchronize(device)
        record.update(
            {
                "gradient_wait_seconds": waited,
                "gradient_sha256": gradient_payload["gradient_sha256"],
                "surrogate_loss": float(loss.detach().cpu()),
                "backward_step_seconds": time.perf_counter() - started,
                "gradient_accumulation_boundary": boundary,
            }
        )
        records.append(record)
        del entry, x0_cpu, payload, x0, gradient_mask
        del gradient_payload, gradient_cpu, gradient, loss
    if int(engine.global_steps) != global_before + 1 or boundaries != 1:
        raise AssertionError("Step10-A must perform exactly one G2 -> G3 update")
    after = trainable_checksum(engine.module)
    if before == after:
        raise AssertionError("Step10-A did not change Generator trainables")
    result["step10a"] = {
        "role": "generator",
        "generator_before": 2,
        "generator_after": 3,
        "global_steps_before": global_before,
        "global_steps_after": int(engine.global_steps),
        "accumulation_boundaries": boundaries,
        "samples": records,
        "trainable_before": before,
        "trainable_after": after,
        "optimizer": optimizer_state_checksum(engine.optimizer),
    }


def _run_step10a_score(
    args: argparse.Namespace,
    *,
    rank: int,
    device: torch.device,
    model: torch.nn.Module,
    trace: RouteBFull5Trace,
    exchange: Path,
    negative_embedding: torch.Tensor | None,
    result: dict[str, Any],
) -> None:
    student = rank in STUDENT_STEP10A_RANKS
    lane = rank - (8 if student else 12)
    records = []
    for source_id in range(lane, 16, 4):
        entry = trace.load_step10a_entry(source_id=source_id, device=device)
        input_wait = _wait(_input_path(exchange, source_id), args.timeout_seconds)
        payload = torch.load(
            _input_path(exchange, source_id), map_location="cpu", weights_only=False
        )
        x0_cpu = _validate_step10a_input(
            payload, source_id=source_id, identity=entry.identity_sha256
        )
        x0 = x0_cpu.to(device=device, dtype=torch.bfloat16)
        batch = prepare_dmd_score_batch(generated_x0=x0, entry=entry, device=device)
        started = time.perf_counter()
        if student:
            fake = score_forward(
                model,
                batch,
                condition=batch["encoder_hidden_states"],
                adapter_enabled=True,
            )
            fake_cpu = fake.detach().cpu().contiguous()
            fake_payload = {
                "schema_version": "distillschedule.routeb_step10a_fake.v1",
                "logical_step": 10,
                "source_id": source_id,
                "generator_version": 2,
                "critic_version": 10,
                "trace_identity_sha256": entry.identity_sha256,
                "x0_sha256": payload["x0_sha256"],
                "fake": fake_cpu,
                "fake_sha256": tensor_sha256(fake_cpu),
            }
            _atomic_torch(_fake_path(exchange, source_id), fake_payload)
            record = {
                "source_id": source_id,
                "x0_sha256": payload["x0_sha256"],
                "fake_sha256": fake_payload["fake_sha256"],
                "input_wait_seconds": input_wait,
                "compute_seconds": time.perf_counter() - started,
            }
            del fake, fake_cpu, fake_payload
        else:
            if negative_embedding is None:
                raise AssertionError("Teacher rank lacks negative prompt")
            real_cond = score_forward(
                model,
                batch,
                condition=batch["encoder_hidden_states"],
                adapter_enabled=False,
            )
            real_uncond = score_forward(
                model,
                batch,
                condition=negative_embedding,
                adapter_enabled=False,
            )
            fake_wait = _wait(_fake_path(exchange, source_id), args.timeout_seconds)
            fake_payload = torch.load(
                _fake_path(exchange, source_id), map_location="cpu", weights_only=False
            )
            expected = {
                "logical_step": 10,
                "source_id": source_id,
                "generator_version": 2,
                "critic_version": 10,
                "trace_identity_sha256": entry.identity_sha256,
                "x0_sha256": payload["x0_sha256"],
            }
            mismatch = {key: (fake_payload.get(key), item) for key, item in expected.items() if fake_payload.get(key) != item}
            if mismatch:
                raise ValueError(f"Step10-A fake identity mismatch: {mismatch}")
            fake_cpu = fake_payload["fake"]
            if tensor_sha256(fake_cpu) != fake_payload["fake_sha256"]:
                raise ValueError("Step10-A fake checksum changed")
            fake = fake_cpu.to(device=device, dtype=torch.bfloat16)
            gradient, auxiliary = construct_dmd_gradient(
                batch=batch,
                fake_flow_cond=fake,
                real_flow_cond=real_cond,
                real_flow_uncond=real_uncond,
            )
            gradient_cpu = gradient.detach().cpu().contiguous()
            gradient_payload = {
                "schema_version": "distillschedule.routeb_step10a_gradient.v1",
                "logical_step": 10,
                "source_id": source_id,
                "generator_version": 2,
                "critic_version": 10,
                "teacher_version": "official_frozen_base",
                "trace_identity_sha256": entry.identity_sha256,
                "x0_sha256": payload["x0_sha256"],
                "gradient": gradient_cpu,
                "gradient_sha256": tensor_sha256(gradient_cpu),
            }
            _atomic_torch(_gradient_path(exchange, source_id), gradient_payload)
            record = {
                "source_id": source_id,
                "x0_sha256": payload["x0_sha256"],
                "fake_sha256": fake_payload["fake_sha256"],
                "gradient_sha256": gradient_payload["gradient_sha256"],
                "input_wait_seconds": input_wait,
                "fake_wait_seconds": fake_wait,
                "compute_seconds": time.perf_counter() - started,
            }
            del real_cond, real_uncond, fake_payload, fake_cpu, fake, gradient, gradient_cpu, gradient_payload, auxiliary
        records.append(record)
        del entry, payload, x0_cpu, x0, batch
        torch.cuda.empty_cache()
    result["step10a"] = {
        "role": "student" if student else "teacher",
        "parameter_version": 10 if student else "official_frozen_base",
        "samples": records,
    }


class _ResidentGuard:
    def __init__(self, model: torch.nn.Module, device: torch.device) -> None:
        self.device = device
        self.inventory = {
            name: (
                int(parameter.untyped_storage().data_ptr()),
                int(parameter.untyped_storage().nbytes()),
                str(parameter.dtype),
                bool(parameter.requires_grad),
            )
            for name, parameter in model.named_parameters()
        }
        if not self.inventory or any(p.device != device for p in model.parameters()):
            raise AssertionError("resident Critic parameter ownership failed")

    def assert_unchanged(self, model: torch.nn.Module) -> None:
        current = {
            name: (
                int(parameter.untyped_storage().data_ptr()),
                int(parameter.untyped_storage().nbytes()),
                str(parameter.dtype),
                bool(parameter.requires_grad),
            )
            for name, parameter in model.named_parameters()
        }
        if current != self.inventory or any(p.device != self.device for p in model.parameters()):
            raise AssertionError("Critic parameter residency changed")


def _transition_critic(
    args: argparse.Namespace,
    *,
    rank: int,
    local_rank: int,
    device: torch.device,
    model: torch.nn.Module,
    package: LogicalCheckpointPackage,
    critic_group: Any,
    singleton: Any,
    result: dict[str, Any],
) -> tuple[Any, _ResidentGuard, Any]:
    import deepspeed

    if rank not in C_RANKS:
        raise AssertionError("only Score-pool ranks may enter the Critic transition")
    if not bool(getattr(model, "_distillschedule_has_adapter", False)):
        raise AssertionError(
            "Step10-A Score object is not Critic-capable; refusing a second 14B construction"
        )
    source_role = "student" if rank in STUDENT_STEP10A_RANKS else "teacher"
    metadata = {
        "transition": "reuse_step10a_physical_score_model",
        "source_execution_role": source_role,
        "physical_role": "critic",
        "adapter_reactivation": _reactivate_critic_adapter(model),
    }
    restore = load_logical_master_into_trainables(
        model,
        package,
        component="critic",
        verify_payload_sha256=(rank == 8),
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=4e-7, betas=(0.0, 0.999), eps=1e-8, weight_decay=1e-3
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    mpu = StudentDPGroupAdapter(
        data_parallel_group=critic_group,
        data_parallel_ranks=C_RANKS,
        global_rank=rank,
        singleton_model_parallel_group=singleton,
    )
    engine, _, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        config=_zero1_config(dp_world_size=8, gas=2, lr=4e-7),
        mpu=mpu,
        dist_init_required=False,
    )
    resolved = _assert_engine_contract(
        engine,
        dp=8,
        gas=2,
        batch=16,
        expected_dp_group=critic_group,
    )
    parent = _checkpoint_parent(args.native_world8_root, "critic")
    load_path, client_state = engine.load_checkpoint(
        str(parent),
        tag="pytorch_model",
        load_module_strict=True,
        load_optimizer_states=True,
        load_lr_scheduler_states=False,
    )
    if load_path is None:
        raise RuntimeError("native world8 C10 restore failed")
    engine.lr_scheduler.load_state_dict(dict(package.component("critic")["scheduler_state"]))
    if (
        int(engine.global_steps),
        _optimizer_step(engine.optimizer),
        _scheduler_step(engine),
    ) != (10, 10, 10):
        raise AssertionError("Critic did not restore C10")
    event_root = args.output_dir / "events"
    event_root.mkdir(parents=True, exist_ok=True)
    activation = install_activation_offload_hooks(
        engine.module,
        logical_component="critic_train",
        event_path=event_root / f"rank_{rank:02d}_critic_activation.jsonl",
        pin_memory=True,
        min_offload_bytes=args.activation_min_bytes,
        detailed_events=False,
        dynamic_excluded_storage_providers=(),
    )
    guard = _ResidentGuard(engine.module, device)
    result["transition"] = {
        "source_role": source_role,
        "target_role": "critic",
        "model_reused": True,
        "metadata": metadata,
        "logical_restore": restore,
        "resolved": resolved,
        "load_path": str(load_path),
        "client_state_keys": sorted((client_state or {}).keys()),
        "critic_version": 10,
        "optimizer": optimizer_state_checksum(engine.optimizer),
        "parameter_policy": "RESIDENT",
    }
    return engine, guard, activation


def _checksum_kind(args: argparse.Namespace) -> ChecksumKind:
    return ChecksumKind.SHA256 if args.checksum == "sha256" else ChecksumKind.GPU_FINGERPRINT_V1


def _make_header(
    args: argparse.Namespace,
    *,
    topology: CriticPoolTopology,
    rank: int,
    pool_rank: int,
    micro_wave: int,
    entry: Any,
    x0: torch.Tensor,
) -> CriticWaveHeader:
    update = micro_wave // 2
    kind = _checksum_kind(args)
    return CriticWaveHeader(
        run_nonce=args.run_nonce,
        cycle=0,
        logical_step=10 + update,
        wave=micro_wave,
        producer_global_rank=rank,
        producer_pool_rank=pool_rank,
        student_global_rank=topology.paired_student(rank),
        student_pool_rank=pool_rank,
        generator_version=3,
        expected_critic_before=10 + update,
        expected_critic_after=11 + update,
        trace_sha256=bytes.fromhex(entry.identity_sha256),
        payload_checksum=tensor_identity(x0, kind),
        checksum_kind=kind,
    )


def _produce_critic_wave(
    args: argparse.Namespace,
    *,
    engine: Any,
    trace: RouteBFull5Trace,
    pool_rank: int,
    micro_wave: int,
    device: torch.device,
) -> tuple[torch.Tensor, Any, dict[str, Any]]:
    update, production_round = divmod(micro_wave, 2)
    loaded = trace.load_critic_entry(
        update=update,
        production_round=production_round,
        producer_pool_rank=pool_rank,
        device=device,
    )
    started = time.perf_counter()
    x0, details = run_generator_trace_rollout(
        model=engine.module,
        loaded=loaded,
        device=device,
        detailed_profile=not args.timing_mode,
        record_output_sha256=args.checksum == "sha256",
    )
    x0 = prepare_transport_payload(x0)
    return x0, loaded.pool_entry, {
        "micro_wave": micro_wave,
        "logical_step": 10 + update,
        "production_round": production_round,
        "trace_key": loaded.pool_entry.trace_key,
        "trace_identity_sha256": loaded.pool_entry.identity_sha256,
        "source_trace_rank": loaded.pool_entry.source_trace_rank,
        "produce_seconds": time.perf_counter() - started,
        "rollout": details,
    }


def _run_generator_critic_phase(
    args: argparse.Namespace,
    *,
    rank: int,
    pool_rank: int,
    device: torch.device,
    engine: Any,
    ema: SparseShardEMA | None,
    trace: RouteBFull5Trace,
    topology: CriticPoolTopology,
    transport: CriticPoolP2PTransport,
    ack_group: Any,
    control_group: Any,
    result: dict[str, Any],
) -> None:
    if not args.timing_mode:
        print(f"[rank{rank}] Generator initial exact checksum begin", flush=True)
    initial_sha = _optional_trainable_sha(
        engine.module, timing_mode=args.timing_mode
    )
    if not args.timing_mode:
        print(f"[rank{rank}] Generator initial exact checksum complete", flush=True)
    wave_records: list[dict[str, Any]] = []
    ema_records: list[dict[str, Any]] = []
    cycle_started = time.perf_counter()

    def send_payload(micro_wave: int, payload: torch.Tensor, entry: Any, record: dict[str, Any]):
        header = _make_header(
            args,
            topology=topology,
            rank=rank,
            pool_rank=pool_rank,
            micro_wave=micro_wave,
            entry=entry,
            x0=payload,
        )
        started = time.perf_counter()
        pending = transport.send_wave(
            header,
            payload,
            destination_global_rank=topology.paired_student(rank),
        )
        pending.wait()
        transport.reap_completed()
        record["send_seconds"] = time.perf_counter() - started
        record["wire"] = {
            "logical_step": header.logical_step,
            "micro_wave": header.wave,
            "generator_version": header.generator_version,
            "critic_before": header.expected_critic_before,
            "critic_after": header.expected_critic_after,
            "trace_identity_sha256": header.trace_sha256.hex(),
            "x0_checksum_kind": header.checksum_kind.name,
            "x0_checksum": header.payload_checksum.hex(),
        }
        return header

    def finish_current(header: CriticWaveHeader, record: dict[str, Any]) -> None:
        started = time.perf_counter()
        ack = receive_ack(
            group=ack_group,
            source_global_rank=topology.paired_student(rank),
        )
        record["ack_wait_seconds"] = time.perf_counter() - started
        expected_commit = header.wave % 2 == 1
        assert_ack_matches_header(ack, header, expected_commit=expected_commit)
        record["ack"] = ack.__dict__
        if expected_commit:
            update = header.wave // 2
            if rank == 0:
                if ema is None:
                    raise AssertionError("EMA owner is missing")
                before = ema.optimization_step
                started_ema = time.perf_counter()
                decay = ema.step(engine.module.parameters())
                expected = (10 + update, 11 + update, 0.0)
                if (before, ema.optimization_step, decay) != expected:
                    raise AssertionError("official early-step EMA recurrence changed")
                comparison = (
                    {"matches_exact": TIMING_OMITTED}
                    if args.timing_mode
                    else ema.compare_to_parameters(engine.module.parameters())
                )
                if (
                    not args.timing_mode
                    and comparison["matches_exact"] is not True
                ):
                    raise AssertionError("official early-step EMA value changed")
                ema_records.append(
                    {
                        "logical_step": 10 + update,
                        "before": before,
                        "after": ema.optimization_step,
                        "decay": decay,
                        "matches_generator_exact": comparison["matches_exact"],
                        "seconds": time.perf_counter() - started_ema,
                        "checksum": (
                            TIMING_OMITTED
                            if args.timing_mode
                            else ema.checksum()
                        ),
                    }
                )
            _control_barrier(control_group)

    if args.schedule == "sequential":
        for micro_wave in range(10):
            payload, entry, record = _produce_critic_wave(
                args,
                engine=engine,
                trace=trace,
                pool_rank=pool_rank,
                micro_wave=micro_wave,
                device=device,
            )
            header = send_payload(micro_wave, payload, entry, record)
            finish_current(header, record)
            wave_records.append(record)
            del payload, entry
    else:
        payload, entry, record = _produce_critic_wave(
            args,
            engine=engine,
            trace=trace,
            pool_rank=pool_rank,
            micro_wave=0,
            device=device,
        )
        header = send_payload(0, payload, entry, record)
        del payload, entry
        for current in range(10):
            next_state = None
            if current + 1 < 10:
                next_state = _produce_critic_wave(
                    args,
                    engine=engine,
                    trace=trace,
                    pool_rank=pool_rank,
                    micro_wave=current + 1,
                    device=device,
                )
            finish_current(header, record)
            wave_records.append(record)
            if next_state is not None:
                payload, entry, record = next_state
                header = send_payload(current + 1, payload, entry, record)
                del payload, entry

    transport.flush()
    transport.assert_flushed()
    if not args.timing_mode:
        print(f"[rank{rank}] Generator final exact checksum begin", flush=True)
    final_sha = _optional_trainable_sha(
        engine.module, timing_mode=args.timing_mode
    )
    if not args.timing_mode:
        print(f"[rank{rank}] Generator final exact checksum complete", flush=True)
    if not args.timing_mode and final_sha != initial_sha:
        raise AssertionError("Generator changed during Critic phase")
    if args.timing_mode:
        gradients: dict[str, Any] = {
            "measurement": TIMING_OMITTED,
            "trainable_grad_present": TIMING_OMITTED,
            "frozen_grad_present": TIMING_OMITTED,
        }
        versions = (
            int(engine.global_steps),
            _optimizer_step(engine.optimizer),
            _scheduler_step(engine),
        )
        if versions != (3, 3, 3):
            raise AssertionError(f"Generator version changed during timing: {versions}")
    else:
        gradients = parameter_gradient_diagnostics(engine.module)
        if gradients["trainable_grad_present"] or gradients["frozen_grad_present"]:
            raise AssertionError("Generator retained Critic-phase gradients")
    if rank == 0 and len(ema_records) != 5:
        raise AssertionError("Full5 must execute five EMA updates")
    result["critic_phase"] = {
        "schedule": args.schedule,
        "seconds": time.perf_counter() - cycle_started,
        "micro_waves": wave_records,
        "generator_version_before": 3,
        "generator_version_after": 3,
        "generator_trainable_sha256_before": initial_sha,
        "generator_trainable_sha256_after": final_sha,
        "generator_gradients": gradients,
        "instrumentation_profile": (
            "low_overhead_timing_v1" if args.timing_mode else "correctness_v1"
        ),
        "ema": ema_records,
        "queue": {
            "produced": 10,
            "sent": 10,
            "acknowledged": 10,
            "producer_slot": "EMPTY",
            "outstanding_transport": 0,
        },
    }


def _validate_header(
    header: CriticWaveHeader,
    *,
    topology: CriticPoolTopology,
    rank: int,
    pool_rank: int,
    entry: Any,
    micro_wave: int,
) -> None:
    update = micro_wave // 2
    expected = {
        "logical_step": (header.logical_step, 10 + update),
        "wave": (header.wave, micro_wave),
        "producer_global_rank": (
            header.producer_global_rank,
            topology.paired_generator(rank),
        ),
        "producer_pool_rank": (header.producer_pool_rank, pool_rank),
        "student_global_rank": (header.student_global_rank, rank),
        "student_pool_rank": (header.student_pool_rank, pool_rank),
        "generator_version": (header.generator_version, 3),
        "critic_before": (header.expected_critic_before, 10 + update),
        "critic_after": (header.expected_critic_after, 11 + update),
        "trace_identity": (header.trace_sha256.hex(), entry.identity_sha256),
    }
    mismatch = {key: value for key, value in expected.items() if value[0] != value[1]}
    if mismatch:
        raise ValueError(f"received Route-B wave identity mismatch: {mismatch}")


def _run_critic_phase(
    args: argparse.Namespace,
    *,
    rank: int,
    pool_rank: int,
    device: torch.device,
    engine: Any,
    guard: _ResidentGuard,
    activation: Any,
    trace: RouteBFull5Trace,
    topology: CriticPoolTopology,
    transport: CriticPoolP2PTransport,
    ack_group: Any,
    control_group: Any,
    result: dict[str, Any],
) -> None:
    observer = None if args.timing_mode else TrainableGradientObserver(engine.module)
    initial_trainable = _optional_trainable_checksum(
        engine.module, timing_mode=args.timing_mode
    )
    initial_optimizer = _optional_optimizer_checksum(
        engine.optimizer, timing_mode=args.timing_mode
    )
    wave_records = []
    cycle_started = time.perf_counter()
    for micro_wave in range(10):
        update, production_round = divmod(micro_wave, 2)
        entry = trace.entry(update, production_round, pool_rank)
        receive_started = time.perf_counter()
        header, x0 = transport.receive_wave(
            source_global_rank=topology.paired_generator(rank)
        )
        receive_seconds = time.perf_counter() - receive_started
        _validate_header(
            header,
            topology=topology,
            rank=rank,
            pool_rank=pool_rank,
            entry=entry,
            micro_wave=micro_wave,
        )
        loaded = trace.load_critic_entry(
            update=update,
            production_round=production_round,
            producer_pool_rank=pool_rank,
            device=device,
        )
        batch = prepare_student_batch(generated_x0=x0, loaded=loaded, device=device)
        optimizer_before = _optimizer_step(engine.optimizer)
        scheduler_before = _scheduler_step(engine)
        if observer is not None:
            observer.begin()
        forward_started = time.perf_counter()
        prediction, loss = student_forward_loss(engine, batch)
        torch.cuda.synchronize(device)
        forward_seconds = time.perf_counter() - forward_started
        prediction_sha = tensor_sha256(prediction) if args.checksum == "sha256" else None
        noise_sha = tensor_sha256(batch["critic_noise"]) if args.checksum == "sha256" else None
        loss_value = float(loss.detach().cpu())
        backward_started = time.perf_counter()
        engine.backward(loss)
        torch.cuda.synchronize(device)
        backward_seconds = time.perf_counter() - backward_started
        delivered = observer.end() if observer is not None else {"timing_mode": True}
        if observer is not None and delivered["parameters_with_gradient"] == 0:
            raise AssertionError("Critic micro-wave delivered no LoRA gradient")
        boundary = bool(engine.is_gradient_accumulation_boundary())
        expected_boundary = production_round == 1
        if boundary != expected_boundary:
            raise AssertionError(
                f"GAS2 boundary changed at micro-wave {micro_wave}: {boundary}"
            )
        step_started = time.perf_counter()
        engine.step()
        torch.cuda.synchronize(device)
        step_seconds = time.perf_counter() - step_started
        optimizer_after = _optimizer_step(engine.optimizer)
        scheduler_after = _scheduler_step(engine)
        expected_delta = int(expected_boundary)
        if (
            optimizer_after,
            scheduler_after,
            int(engine.global_steps),
        ) != (
            optimizer_before + expected_delta,
            scheduler_before + expected_delta,
            optimizer_before + expected_delta,
        ):
            raise AssertionError("Critic optimizer/scheduler/version order changed")
        activation.offloader.assert_scope_closed()
        ack = RouteBWaveAck(
            run_nonce=header.run_nonce,
            cycle=header.cycle,
            logical_step=header.logical_step,
            micro_wave=micro_wave,
            producer_global_rank=header.producer_global_rank,
            critic_global_rank=rank,
            observed_generator_version=3,
            critic_before=optimizer_before,
            critic_after=optimizer_after,
            optimizer_step_before=optimizer_before,
            optimizer_step_after=optimizer_after,
            scheduler_step_before=scheduler_before,
            scheduler_step_after=scheduler_after,
            optimizer_committed=expected_boundary,
        )
        ack_started = time.perf_counter()
        send_ack(
            ack,
            group=ack_group,
            destination_global_rank=topology.paired_generator(rank),
        )
        ack_seconds = time.perf_counter() - ack_started
        record = {
            "micro_wave": micro_wave,
            "logical_step": 10 + update,
            "production_round": production_round,
            "trace_key": entry.trace_key,
            "trace_identity_sha256": entry.identity_sha256,
            "source_trace_rank": entry.source_trace_rank,
            "x0_identity": {
                "kind": header.checksum_kind.name,
                "value": header.payload_checksum.hex(),
            },
            "prediction_sha256": prediction_sha,
            "critic_noise_sha256": noise_sha,
            "critic_timestep": [int(v) for v in batch["timestep"].detach().cpu().tolist()],
            "loss": loss_value,
            "receive_seconds": receive_seconds,
            "forward_seconds": forward_seconds,
            "backward_seconds": backward_seconds,
            "optimizer_seconds": step_seconds,
            "ack_seconds": ack_seconds,
            "gradient_delivery": delivered,
            "gradient_accumulation_boundary": boundary,
            "critic_version_before": optimizer_before,
            "critic_version_after": optimizer_after,
            "scheduler_before": scheduler_before,
            "scheduler_after": scheduler_after,
        }
        wave_records.append(record)
        del x0, loaded, batch, prediction, loss
        if expected_boundary:
            _control_barrier(control_group)
    transport.flush()
    transport.assert_flushed()
    if not args.timing_mode:
        print(f"[rank{rank}] Critic exact epilogue begin", flush=True)
    if observer is not None:
        observer.close()
    guard.assert_unchanged(engine.module)
    if (
        int(engine.global_steps),
        _optimizer_step(engine.optimizer),
        _scheduler_step(engine),
    ) != (15, 15, 15):
        raise AssertionError("Critic did not finish at C15")
    result["critic_phase"] = {
        "schedule": args.schedule,
        "seconds": time.perf_counter() - cycle_started,
        "micro_waves": wave_records,
        "critic_version_before": 10,
        "critic_version_after": 15,
        "optimizer_step_before": 10,
        "optimizer_step_after": 15,
        "scheduler_step_before": 10,
        "scheduler_step_after": 15,
        "initial_trainable": initial_trainable,
        "final_trainable": _optional_trainable_checksum(
            engine.module, timing_mode=args.timing_mode
        ),
        "initial_optimizer": initial_optimizer,
        "final_optimizer": _optional_optimizer_checksum(
            engine.optimizer, timing_mode=args.timing_mode
        ),
        "instrumentation_profile": (
            "low_overhead_timing_v1" if args.timing_mode else "correctness_v1"
        ),
        "queue": {
            "received": 10,
            "consumed": 10,
            "optimizer_commits": 5,
            "consumer_slot": "EMPTY",
            "outstanding_transport": 0,
        },
    }
    if not args.timing_mode:
        print(f"[rank{rank}] Critic exact epilogue complete", flush=True)


def _save_rng(path: Path, *, rank: int) -> None:
    value = {
        "rank": rank,
        "cpu_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(),
    }
    _atomic_torch(path / f"random_states_{rank}.pkl", value)


def _load_rng(path: Path, *, rank: int) -> None:
    value = torch.load(
        path / f"random_states_{rank}.pkl",
        map_location="cpu",
        weights_only=False,
    )
    if int(value.get("rank", -1)) != rank:
        raise AssertionError("transition RNG rank identity changed")
    torch.set_rng_state(value["cpu_rng_state"])
    torch.cuda.set_rng_state(value["cuda_rng_state"])


def _checkpoint_inventory(root: Path) -> dict[str, Any]:
    files = [path for path in root.rglob("*") if path.is_file()]
    expected_shards = {
        f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt"
        for rank in range(8)
    }
    for component, component_root in (
        ("generator", root / "pytorch_model"),
        ("critic", root / "critic" / "pytorch_model"),
    ):
        observed = {path.name for path in component_root.glob("*optim_states.pt")}
        if observed != expected_shards:
            raise ValueError(
                f"{component} optimizer shard inventory changed: {sorted(observed)}"
            )
        if not (component_root / "mp_rank_00_model_states.pt").is_file():
            raise FileNotFoundError(f"{component} model state missing")
    rng = sorted(root.glob("random_states_*.pkl"))
    if len(rng) != 16:
        raise ValueError("world16 RNG inventory changed")
    if not (root / "model_ema" / "sparse_ema_manifest.json").is_file():
        raise FileNotFoundError("sparse EMA state missing")
    return {
        "file_count_before_manifest": len(files),
        "bytes_before_manifest": sum(path.stat().st_size for path in files),
        "generator_optimizer_shards": 8,
        "critic_optimizer_shards": 8,
        "rng_files": 16,
        "sparse_ema": True,
    }


def _transition_manifest(
    root: Path,
    *,
    identities: dict[str, Any],
) -> dict[str, Any]:
    manifest_path = root / "transition_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    value = json.loads(manifest_path.read_text())
    expected = {
        "schema_version": "distillschedule.routeb_transition_checkpoint.v1",
        "logical_global_step": 10,
        "generator_version": 3,
        "critic_version": 10,
        "ema_optimization_step": 10,
    }
    mismatch = {
        key: (value.get(key), expected_value)
        for key, expected_value in expected.items()
        if value.get(key) != expected_value
    }
    if mismatch:
        raise ValueError(f"transition checkpoint contract changed: {mismatch}")
    for key in (
        "logical_manifest_hash",
        "execution_trace_bundle_hash",
        "routeb_trace_binding_hash",
    ):
        if value.get("identities", {}).get(key) != identities.get(key):
            raise ValueError(f"transition checkpoint identity changed: {key}")
    return {
        "path": str(root.resolve()),
        "manifest_sha256": _sha256_file(manifest_path),
        "manifest": value,
    }


def _save_checkpoint(
    args: argparse.Namespace,
    *,
    rank: int,
    generator_engine: Any | None,
    critic_engine: Any | None,
    ema: SparseShardEMA | None,
    control_group: Any,
    device: torch.device,
    root_name: str = "checkpoint-15",
    logical_global_step: int = 15,
    generator_version: int = 3,
    critic_version: int = 15,
    ema_optimization_step: int = 15,
    transition_identities: dict[str, Any] | None = None,
) -> None:
    del device
    root = args.output_dir / root_name
    if rank == 0:
        root.mkdir(parents=True, exist_ok=False)
        (root / "critic").mkdir()
    _control_barrier(control_group)

    # DeepSpeed 0.18.4 save_checkpoint contains default-world barriers and
    # tag-validation all-reduces (and explicitly requires every process to
    # call it).  Serially asking only G ranks and then only C ranks to save is
    # therefore a guaranteed world16 deadlock even though each engine's data
    # parallel group is only eight ranks.  Both roles call the same primitive
    # concurrently, with disjoint directories and identical tags; model and
    # optimizer tensors remain scoped to their existing DP groups.
    if rank in G_RANKS:
        if generator_engine is None:
            raise AssertionError("Generator engine missing during checkpoint")
        engine = generator_engine
        save_root = root
        client_state = {
            "logical_global_step": logical_global_step,
            "generator_version": generator_version,
        }
        component = "generator"
    else:
        if critic_engine is None:
            raise AssertionError("Critic engine missing during checkpoint")
        engine = critic_engine
        save_root = root / "critic"
        client_state = {
            "logical_global_step": logical_global_step,
            "critic_version": critic_version,
        }
        component = "critic"
    if not args.timing_mode:
        print(f"[rank{rank}] {component} DeepSpeed checkpoint begin", flush=True)
    engine.save_checkpoint(
        str(save_root),
        tag="pytorch_model",
        client_state=client_state,
    )
    if not args.timing_mode:
        print(f"[rank{rank}] {component} DeepSpeed checkpoint complete", flush=True)
    _control_barrier(control_group)
    if rank == 0:
        if ema is None or ema.optimization_step != ema_optimization_step:
            raise AssertionError(
                f"EMA{ema_optimization_step} missing during checkpoint"
            )
        ema.save_pretrained(root / "model_ema")
    _save_rng(root, rank=rank)
    _control_barrier(control_group)
    if rank == 0 and transition_identities is not None:
        manifest = {
            "schema_version": "distillschedule.routeb_transition_checkpoint.v1",
            "logical_global_step": logical_global_step,
            "generator_version": generator_version,
            "critic_version": critic_version,
            "ema_optimization_step": ema_optimization_step,
            "identities": dict(transition_identities),
            "inventory": _checkpoint_inventory(root),
        }
        _atomic_json(root / "transition_manifest.json", manifest)
    if transition_identities is not None:
        _control_barrier(control_group)


def _load_transition_checkpoint(
    args: argparse.Namespace,
    *,
    rank: int,
    generator_engine: Any | None,
    critic_engine: Any | None,
    ema: SparseShardEMA | None,
    control_group: Any,
    identities: dict[str, Any],
) -> tuple[SparseShardEMA | None, dict[str, Any]]:
    root = args.resume_transition_checkpoint
    if root is None:
        raise ValueError("transition checkpoint path is required")
    transition = _transition_manifest(root, identities=identities)
    expected_sha = args.expected_transition_manifest_sha256
    if not expected_sha:
        raise ValueError("expected transition manifest SHA-256 is required")
    if transition["manifest_sha256"] != expected_sha:
        raise ValueError("transition manifest SHA-256 changed")
    _control_barrier(control_group)
    if rank in G_RANKS:
        if generator_engine is None:
            raise AssertionError("Generator engine missing during transition restore")
        engine = generator_engine
        load_root = root
        expected_client = {
            "logical_global_step": 10,
            "generator_version": 3,
        }
        component = "generator"
    else:
        if critic_engine is None:
            raise AssertionError("Critic engine missing during transition restore")
        engine = critic_engine
        load_root = root / "critic"
        expected_client = {
            "logical_global_step": 10,
            "critic_version": 10,
        }
        component = "critic"
    load_path, client_state = engine.load_checkpoint(
        str(load_root),
        tag="pytorch_model",
        load_module_strict=True,
        load_optimizer_states=True,
        load_lr_scheduler_states=True,
    )
    if load_path is None:
        raise RuntimeError(f"{component} transition checkpoint restore failed")
    for key, expected in expected_client.items():
        if int((client_state or {}).get(key, -1)) != expected:
            raise AssertionError(f"{component} transition client state changed: {key}")
    _control_barrier(control_group)
    if rank == 0:
        ema = SparseShardEMA.load_pretrained(
            generator_engine.module,
            root / "model_ema",
        )
        if ema.optimization_step != 10:
            raise AssertionError("transition checkpoint did not restore EMA10")
    _load_rng(root, rank=rank)
    _control_barrier(control_group)
    if rank in G_RANKS:
        versions = (
            int(engine.global_steps),
            _optimizer_step(engine.optimizer),
            _scheduler_step(engine),
        )
        if versions != (3, 3, 3):
            raise AssertionError(f"transition restore did not produce G3: {versions}")
    else:
        versions = (
            int(engine.global_steps),
            _optimizer_step(engine.optimizer),
            _scheduler_step(engine),
        )
        if versions != (10, 10, 10):
            raise AssertionError(f"transition restore did not produce C10: {versions}")
    restored = {
        "component": component,
        "load_path": str(load_path),
        "client_state": {key: int(value) for key, value in expected_client.items()},
        "versions": list(versions),
        "trainable_sha256": trainable_sha256(engine.module),
        "optimizer": optimizer_state_checksum(engine.optimizer),
        "transition_manifest_sha256": transition["manifest_sha256"],
    }
    if rank == 0:
        restored["ema_checksum"] = ema.checksum()
        restored["ema_optimization_step"] = ema.optimization_step
    return ema, restored


def main() -> int:
    args = _parser().parse_args()
    if args.timing_mode and args.checksum != "gpu":
        raise ValueError("timing mode requires low-overhead GPU checksum")
    if args.save_transition_checkpoint and not args.stop_after_transition:
        raise ValueError("transition checkpoint save requires stop-after-transition")
    if args.resume_transition_checkpoint is not None and args.stop_after_transition:
        raise ValueError("transition resume cannot execute Step10-A")
    if (
        args.resume_transition_checkpoint is None
        and args.expected_transition_manifest_sha256 is not None
    ):
        raise ValueError("transition manifest hash requires transition resume")
    timing_correctness = _timing_correctness_binding(args)
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if world != 16:
        raise RuntimeError("Candidate A requires world16")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    import deepspeed

    deepspeed.init_distributed(dist_backend="nccl")
    if rank == 0:
        if args.output_dir.exists():
            raise FileExistsError(args.output_dir)
        args.output_dir.mkdir(parents=True)
        (args.output_dir / "exchange").mkdir()
    dist.barrier()
    exchange = args.output_dir / "exchange"
    if not exchange.is_dir():
        raise FileNotFoundError("rank0 did not publish the execution directory")
    topology = CriticPoolTopology(
        world_size=16, generator_ranks=G_RANKS, student_ranks=C_RANKS
    )
    (
        control_group,
        generator_group,
        critic_group,
        _student_step10a_group,
        _teacher_step10a_group,
        pair_groups,
        ack_groups,
        singleton_groups,
    ) = _create_groups(topology)
    role = "generator" if rank in G_RANKS else "score"
    pool_rank = topology.pool_rank(rank)
    output = args.output_dir / "ranks" / f"rank_{rank:02d}.json"
    critic_phase_only = args.resume_transition_checkpoint is not None
    result: dict[str, Any] = {
        "schema_version": "distillschedule.routeb_full5_candidate_a_rank.v1",
        "status": "RUNNING",
        "run_id": args.run_id,
        "schedule": args.schedule,
        "contract": RouteBFull5Contract.candidate_a().__dict__,
        "execution_scope": (
            "critic_phase_from_common_transition_checkpoint"
            if critic_phase_only
            else "continuous_full5_step10_to_step14"
        ),
        "hostname": socket.gethostname(),
        "rank": rank,
        "local_rank": local_rank,
        "role": role,
        "pool_rank": pool_rank,
        "stages": [_snapshot(device, "process_start")],
    }
    if timing_correctness is not None:
        result["timing_correctness_binding"] = timing_correctness
    _atomic_json(output, result)
    generator_engine = critic_engine = None
    activation = streamer = ema = None
    transport = None
    started = time.perf_counter()
    try:
        package = LogicalCheckpointPackage.load(args.logical_package)
        trace = RouteBFull5Trace(
            args.trace_bundle,
            RouteBFull5Contract.candidate_a(),
        )
        # The package trace hash records how the frozen G2/C10/EMA10 state was
        # produced.  The rank16 bundle records the inputs replayed by this new
        # execution.  They are intentionally distinct identities (the same
        # distinction already used by the validated Critic-pool target
        # package); conflating them would reject a valid immutable initial
        # state or, worse, hide which workload was actually executed.
        state_provenance_trace_hash = str(package.manifest["input_trace_hash"])
        result["identities"] = {
            "logical_manifest_hash": package.manifest_hash,
            "state_provenance_trace_hash": state_provenance_trace_hash,
            "execution_trace_bundle_hash": trace.bundle_hash,
            "routeb_trace_binding_hash": trace.binding_hash,
        }
        if critic_phase_only:
            transition_identity = _transition_manifest(
                args.resume_transition_checkpoint,
                identities=result["identities"],
            )
            expected_sha = args.expected_transition_manifest_sha256
            if (
                not expected_sha
                or transition_identity["manifest_sha256"] != expected_sha
            ):
                raise ValueError("transition manifest identity is not frozen")
            result["identities"]["transition_manifest_sha256"] = expected_sha
            result["identities"]["transition_checkpoint_path"] = str(
                args.resume_transition_checkpoint.resolve()
            )
        model, metadata = _construct_models(
            args,
            rank=rank,
            local_rank=local_rank,
            package=package,
            device=device,
        )
        result["construction"] = metadata
        if rank in G_RANKS:
            generator_engine, initialization = _initialize_generator_engine(
                args,
                model=model,
                package=package,
                rank=rank,
                group=generator_group,
                singleton=singleton_groups[rank],
            )
            result["generator_initialization"] = initialization
            if not critic_phase_only:
                event_root = args.output_dir / "events"
                event_root.mkdir(parents=True, exist_ok=True)
                streamer = install_frozen_block_streamer(
                    generator_engine.module,
                    logical_component="generator_step10a",
                    device=device,
                    window_size=1,
                    streamed_block_count=4,
                    streamed_block_indices=G4_BLOCKS,
                    host_policy="DROP_RELOAD_FROM_MMAP",
                    optimizer_owned_parameter_ids={
                        id(parameter)
                        for parameter in generator_engine.module.parameters()
                        if parameter.requires_grad
                    },
                    checkpoint_identity=(
                        "helios-distilled-frozen-backbone:"
                        + str(metadata["precast_manifest_sha256"])
                    ),
                    event_path=event_root / f"rank_{rank:02d}_g4_blocks.jsonl",
                    mmap_cache_root=args.mmap_cache_root,
                    detailed_events=not args.timing_mode,
                    memory_snapshot_events=False,
                )
                activation = install_activation_offload_hooks(
                    generator_engine.module,
                    logical_component="generator_selected_step10a",
                    event_path=event_root / f"rank_{rank:02d}_generator_activation.jsonl",
                    pin_memory=True,
                    min_offload_bytes=args.activation_min_bytes,
                    detailed_events=False,
                    dynamic_excluded_storage_providers=(
                        streamer.dynamic_excluded_storage_owner,
                    ),
                )
            if rank == 0:
                ema = SparseShardEMA.load_pretrained(
                    generator_engine.module, args.logical_package / "ema"
                )
                if ema.optimization_step != 10:
                    raise AssertionError("logical package did not restore EMA10")
        else:
            negative_embedding = None
            if rank in TEACHER_STEP10A_RANKS and not critic_phase_only:
                negative = NegativePromptAsset.load(
                    args.negative_prompt_asset,
                    verify_payload_sha256=(rank == 12),
                )
                negative_embedding = negative.embedding.to(
                    device=device, dtype=torch.bfloat16
                )
            else:
                negative_embedding = None
        _control_barrier(control_group)
        result["stages"].append(
            _snapshot(
                device,
                "critic_resume_construction_ready"
                if critic_phase_only
                else "step10a_ready",
            )
        )
        torch.cuda.synchronize(device)
        full5_started = None if critic_phase_only else time.perf_counter()
        if not critic_phase_only:
            if rank in G_RANKS:
                _run_step10a_generator(
                    args,
                    rank=rank,
                    device=device,
                    engine=generator_engine,
                    trace=trace,
                    exchange=exchange,
                    result=result,
                )
            else:
                _run_step10a_score(
                    args,
                    rank=rank,
                    device=device,
                    model=model,
                    trace=trace,
                    exchange=exchange,
                    negative_embedding=negative_embedding,
                    result=result,
                )
            _control_barrier(control_group)
            result["stages"].append(_snapshot(device, "step10a_complete"))

        if rank in G_RANKS:
            if critic_phase_only:
                result["transition"] = {
                    "source_role": "common_transition_checkpoint",
                    "target_role": "generator_critic_phase_resident",
                    "parameter_policy": "RESIDENT",
                    "step10a_execution": "Not Run",
                }
            else:
                activation.offloader.assert_scope_closed()
                activation.remove()
                activation = None
                streamer.assert_idle()
                result["transition"] = {
                    "source_role": "generator_step10a_g4",
                    "target_role": "generator_critic_phase_resident",
                    "promotion": streamer.promote_to_resident(),
                    "streamer_stats": dict(streamer.stats),
                }
                torch.cuda.empty_cache()
        else:
            if rank in TEACHER_STEP10A_RANKS and not critic_phase_only:
                del negative_embedding
            critic_engine, guard, activation = _transition_critic(
                args,
                rank=rank,
                local_rank=local_rank,
                device=device,
                model=model,
                package=package,
                critic_group=critic_group,
                singleton=singleton_groups[rank],
                result=result,
            )
        if critic_phase_only:
            ema, restored = _load_transition_checkpoint(
                args,
                rank=rank,
                generator_engine=generator_engine,
                critic_engine=critic_engine,
                ema=ema,
                control_group=control_group,
                identities=result["identities"],
            )
            result["transition_resume"] = restored
        _control_barrier(control_group)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        result["stages"].append(_snapshot(device, "critic_phase_ready"))
        critic_subgraph_started = time.perf_counter()
        if args.stop_after_transition:
            if rank in G_RANKS:
                if int(generator_engine.global_steps) != 3:
                    raise AssertionError("transition gate did not preserve G3")
                if rank == 0 and (ema is None or ema.optimization_step != 10):
                    raise AssertionError("transition gate did not preserve EMA10")
            else:
                if (
                    int(critic_engine.global_steps),
                    _optimizer_step(critic_engine.optimizer),
                    _scheduler_step(critic_engine),
                ) != (10, 10, 10):
                    raise AssertionError("transition gate did not preserve C10")
                guard.assert_unchanged(critic_engine.module)
            result["transition_only"] = {
                "status": "PASS",
                "generator_version": 3,
                "critic_version": 10,
                "ema_optimization_step": 10,
                "critic_updates": 0,
                "ema_updates": 0,
            }
            if args.save_transition_checkpoint:
                _save_checkpoint(
                    args,
                    rank=rank,
                    generator_engine=generator_engine,
                    critic_engine=critic_engine,
                    ema=ema,
                    control_group=control_group,
                    device=device,
                    root_name="checkpoint-transition-g3-c10-ema10",
                    logical_global_step=10,
                    generator_version=3,
                    critic_version=10,
                    ema_optimization_step=10,
                    transition_identities=result["identities"],
                )
                transition_root = (
                    args.output_dir / "checkpoint-transition-g3-c10-ema10"
                )
                transition = _transition_manifest(
                    transition_root,
                    identities=result["identities"],
                )
                result["transition_only"]["checkpoint"] = {
                    "path": transition["path"],
                    "manifest_sha256": transition["manifest_sha256"],
                    "inventory": transition["manifest"]["inventory"],
                }
            if full5_started is None:
                raise AssertionError("transition-only run lost its Full5 start")
            result["full5_execution_seconds"] = time.perf_counter() - full5_started
            result["status"] = "PASS"
            result["duration_seconds"] = time.perf_counter() - started
            result["stages"].append(_snapshot(device, "transition_gate_complete"))
            _atomic_json(output, result)
            return 0
        transport = CriticPoolP2PTransport(
            group=pair_groups[rank], device=device
        )
        cycle_started = time.perf_counter()
        if rank in G_RANKS:
            _run_generator_critic_phase(
                args,
                rank=rank,
                pool_rank=pool_rank,
                device=device,
                engine=generator_engine,
                ema=ema,
                trace=trace,
                topology=topology,
                transport=transport,
                ack_group=ack_groups[rank],
                control_group=control_group,
                result=result,
            )
        else:
            _run_critic_phase(
                args,
                rank=rank,
                pool_rank=pool_rank,
                device=device,
                engine=critic_engine,
                guard=guard,
                activation=activation,
                trace=trace,
                topology=topology,
                transport=transport,
                ack_group=ack_groups[rank],
                control_group=control_group,
                result=result,
            )
        result["critic_phase_wall_seconds"] = time.perf_counter() - cycle_started
        _control_barrier(control_group)
        torch.cuda.synchronize(device)
        if critic_phase_only:
            result["critic_subgraph_execution_seconds"] = (
                time.perf_counter() - critic_subgraph_started
            )
        else:
            if full5_started is None:
                raise AssertionError("continuous Full5 run lost its start time")
            result["full5_execution_seconds"] = time.perf_counter() - full5_started
        if args.save_checkpoint:
            _save_checkpoint(
                args,
                rank=rank,
                generator_engine=generator_engine,
                critic_engine=critic_engine,
                ema=ema,
                control_group=control_group,
                device=device,
            )
        result["status"] = "PASS"
        result["duration_seconds"] = time.perf_counter() - started
        result["stages"].append(_snapshot(device, "complete"))
        _atomic_json(output, result)
        return 0
    except BaseException as error:
        is_oom = isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()
        result.update(
            {
                "status": "OOM" if is_oom else "FAIL",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "duration_seconds": time.perf_counter() - started,
            }
        )
        with contextlib.suppress(BaseException):
            result["stages"].append(_snapshot(device, "failure"))
        _atomic_json(output, result)
        return 42 if is_oom else 1
    finally:
        with contextlib.suppress(BaseException):
            if transport is not None:
                transport.flush()
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
