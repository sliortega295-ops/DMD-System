#!/usr/bin/env python3
"""Real-14B GAN Full5 role-phase executor.

This is a thin GAN binding over the validated PR6 lifecycle substrate:

* Phase A: independent node-local worlds, node0 DP8 ZeRO-3 Generator and
  node1 resident read-only Score, B16 as two B8 waves;
* timed A->B handoff: destroy the obsolete ZeRO-3 representation, form one
  global world16, and bind 4 resident Generators plus DP12 ZeRO-1 Critics;
* Phase B: ACK-free one-batch-ahead 4G->12C B12 for steps 1000..1004;
* timed B->A handoff: restore a node-local DP8 ZeRO-3 Generator carrying G201
  optimizer state and leave node1's C1005 Score ready for the next Phase A.

No new Helios model math lives here.  Generator rollout, official Score-side
DMD/GAN-G, official Critic denoise/GAN-D/R1, optimizer and EMA primitives are
imported from the already audited runtime helpers.
"""
from __future__ import annotations

import argparse
import gc
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

from runtime.activation_offload import install_activation_offload_hooks
from runtime.critic_pool import ChecksumKind, StudentDPGroupAdapter
from runtime.critic_pool_checkpoint import (
    LogicalCheckpointPackage,
    restore_zero1_optimizer_partition_from_logical,
    restore_zero3_optimizer_partition_from_logical,
)
from runtime.critic_pool_helios import (
    build_helios_pool_model,
    run_generator_trace_rollout,
)
from runtime.critic_pool_transport import tensor_identity
from runtime.frozen_block_streaming import (
    HostStatePolicy,
    install_frozen_block_streamer,
)
from runtime.gan_rolephase_node1 import (
    assert_phase_b_zero1_contract,
    phase_b_zero1_config,
)
from runtime.gan_rolephase_official import (
    critic_official_kwargs,
    load_logical_trainables,
    load_official_config,
    score_official_kwargs,
)
from runtime.gan_rolephase_trace import GanRolePhaseTrace
from runtime.gst_pool_contract import NegativePromptAsset
from runtime.gst_pool_helios import run_generator_update_trace_rollout
from runtime.official_baseline import (
    optimizer_state_checksum,
    parameter_gradient_diagnostics,
    suspend_transformers_zero3_init,
    trainable_checksum,
    trainable_sha256,
)
from runtime.operation_trace import tensor_sha256
from runtime.sparse_ema import SparseShardEMA
from runtime.zero3_checkpoint import configure_block_gradient_checkpointing
from scripts.distillschedule import probe_routeb_full5_2g6c_continuous as pr6
from distillschedule.runtime.gan_pipeline_helios import (
    compute_official_score_boundary_grad,
)
from distillschedule.runtime.gan_rolephase import GanRolePhaseTopology
from distillschedule.runtime.gan_rolephase_groups import (
    create_phase_b_execution_groups,
)
from distillschedule.runtime.gan_rolephase_transport import (
    RolePhaseCriticP2PTransport,
    make_phase_b_header,
    make_phase_b_route,
)


TOPOLOGY = GanRolePhaseTopology()
PHASE_A_WORLD_SIZE = 8
PHASE_B_WORLD_SIZE = 16
PHASE_A_G_RANKS = tuple(range(8))
PHASE_A_SCORE_RANKS = tuple(range(8, 16))
PHASE_B_G_RANKS = tuple(range(4))
PHASE_B_C_RANKS = tuple(range(4, 16))
FLEX_RANKS = tuple(range(4, 8))
TIMING_OMITTED = "omitted_in_low_overhead_timing_mode"
SOURCE_GENERATOR_OPTIMIZER_STEP = 2
SOURCE_CRITIC_OPTIMIZER_STEP = 10
TARGET_GENERATOR_OPTIMIZER_STEP = 3
TARGET_CRITIC_OPTIMIZER_STEP = 15


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _atomic_torch(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _wait(path: Path, timeout_seconds: float) -> float:
    started = time.monotonic()
    while not path.is_file():
        if time.monotonic() - started > timeout_seconds:
            raise TimeoutError(path)
        time.sleep(0.05)
    return time.monotonic() - started


def _fingerprint(tensor: torch.Tensor, *, timing_mode: bool) -> tuple[str, str]:
    if timing_mode:
        kind = ChecksumKind.GPU_FINGERPRINT_V1
        return kind.name, tensor_identity(tensor, kind).hex()
    cpu = tensor.detach().cpu().contiguous()
    return "SHA256_LOGICAL_V1", tensor_sha256(cpu)


def _validate_fingerprint(
    tensor: torch.Tensor, *, kind: str, value: str, timing_mode: bool
) -> None:
    expected_kind, observed = _fingerprint(tensor, timing_mode=timing_mode)
    if kind != expected_kind or observed != value:
        raise ValueError(
            f"payload fingerprint mismatch: {(kind, value)} != {(expected_kind, observed)}"
        )


def _remote_gradient_sum_surrogate(
    generated: torch.Tensor,
    latent_gradient: torch.Tensor,
) -> torch.Tensor:
    """Inject an already-normalized Score-side dL/dx without rescaling it."""

    if generated.shape != latent_gradient.shape:
        raise ValueError("remote latent gradient shape differs from Generator output")
    loss = (generated.float() * latent_gradient.detach().float()).sum()
    if not loss.requires_grad or loss.grad_fn is None:
        raise AssertionError("remote Score surrogate lost Generator autograd")
    return loss


def _phase_a_input_path(exchange: Path, source_id: int) -> Path:
    return exchange / "phase_a" / f"x0_{source_id:02d}.pt"


def _phase_a_gradient_path(exchange: Path, source_id: int) -> Path:
    return exchange / "phase_a" / f"gradient_{source_id:02d}.pt"


def _assert_versions(logical: LogicalCheckpointPackage) -> None:
    expected = {
        "generator": SOURCE_GENERATOR_OPTIMIZER_STEP,
        "critic": SOURCE_CRITIC_OPTIMIZER_STEP,
    }
    observed = {
        name: int(logical.component(name)["parameter_version"])
        for name in expected
    }
    if observed != expected:
        raise ValueError(f"systems-only logical versions changed: {observed}")
    systems_control = logical.manifest.get("systems_only_control", {})
    if systems_control.get("logical_schedule_versions") != {
        "generator": 200,
        "critic": 1000,
        "ema": 800,
    }:
        raise ValueError("systems-only logical schedule version mapping changed")
    ema = json.loads(
        (logical.root / "ema" / "sparse_ema_manifest.json").read_text()
    )
    if int(ema["optimization_step"]) != 800:
        raise ValueError("systems-only logical package did not restore EMA800")


def _capture_local_phase_a_rendezvous(*, local_rank: int) -> dict[str, str]:
    keys = ("RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")
    value = {key: os.environ[key] for key in keys}
    if int(value["RANK"]) != local_rank or int(value["WORLD_SIZE"]) != 8:
        raise ValueError("launcher did not provide one node-local default world8")
    return value


def _restore_local_phase_a_rendezvous(
    value: dict[str, str],
    *,
    master_port: int,
) -> None:
    expected = {"RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"}
    if set(value) != expected or int(value["WORLD_SIZE"]) != 8:
        raise ValueError("invalid saved Phase-A rendezvous environment")
    if not 1 <= int(master_port) <= 65535:
        raise ValueError("invalid reverse Phase-A rendezvous port")
    restored = dict(value)
    # torchrun --standalone keeps its elastic-agent TCPStore listener alive
    # even after the worker destroys the first local process group. Reusing
    # that port deterministically fails rank 0 with EADDRINUSE, so the reverse
    # world uses a preregistered node-local port.
    restored["MASTER_PORT"] = str(master_port)
    os.environ.update(restored)


def _validated_group_ranks(
    group: Any,
    *,
    expected: tuple[int, ...],
    global_rank: int,
) -> tuple[int, ...]:
    """Validate one subgroup without introspecting NON_GROUP_MEMBER.

    PyTorch represents every non-member subgroup handle with the same sentinel.
    Passing that sentinel to ``get_process_group_ranks`` can therefore report
    the ranks of an unrelated, most recently created group.  Only members may
    introspect the handle; non-members instead prove they received the sentinel
    while the world-ordered construction contract supplies the expected ranks.
    """

    if global_rank in expected:
        observed = tuple(dist.get_process_group_ranks(group))
        if observed != expected:
            raise AssertionError(
                f"member subgroup mismatch: {observed} != {expected}"
            )
        return observed
    if group != dist.GroupMember.NON_GROUP_MEMBER:
        raise AssertionError("non-member rank unexpectedly owns subgroup handle")
    return expected


def _construct_phase_a_source(
    *,
    args: argparse.Namespace,
    global_rank: int,
    device: torch.device,
    logical: LogicalCheckpointPackage,
    control_group: Any,
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    if global_rank in PHASE_A_G_RANKS:
        model, metadata = build_helios_pool_model(
            role="generator",
            model_root=args.generator_root,
            subfolder="transformer_ode",
            precast_model_root=args.precast_generator_root,
        )
        restore = load_logical_trainables(model, logical.root, "generator")
        role = "generator"
    else:
        model, metadata = build_helios_pool_model(
            role="critic",
            model_root=args.score_root,
            subfolder="transformer",
            precast_model_root=args.precast_score_root,
            enable_gan=True,
        )
        restore = load_logical_trainables(model, logical.root, "critic")
        model.to(device)
        torch.cuda.synchronize(device)
        role = "score_gradient_service"
    metadata["execution_role"] = role
    pr6._control_barrier(control_group)
    return model, metadata, restore


def _initialize_phase_a_generator(
    *,
    args: argparse.Namespace,
    model: torch.nn.Module,
    logical: LogicalCheckpointPackage,
    local_rank: int,
) -> tuple[Any, dict[str, Any]]:
    import deepspeed
    from deepspeed.utils import get_z3_leaf_modules, set_z3_leaf_modules

    configure_block_gradient_checkpointing(
        model, zero_stage=3, policy="zero3_metadata_tolerant"
    )
    leaves = set_z3_leaf_modules(model, ["HeliosTransformerBlock"])
    if len(leaves) != 40:
        raise AssertionError(f"Phase-A block leaf count changed: {len(leaves)}")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=2e-6,
        betas=(0.0, 0.999),
        eps=1e-8,
        weight_decay=1e-3,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    config = pr6._step10a_generator_config(
        param_persistence_threshold=args.phase_a_param_persistence_threshold,
        module_granularity_threshold=0,
        prefetch_bucket_elements=args.phase_a_prefetch_bucket_elements,
        max_live_parameters=args.phase_a_max_live_parameters,
        max_reuse_distance=args.phase_a_max_reuse_distance,
        overlap_comm=False,
    )
    engine, _, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        config=config,
        dist_init_required=False,
    )
    resolved = {
        "dp_world_size": pr6._resolved_int(engine, "dp_world_size"),
        "train_micro_batch_size_per_gpu": pr6._resolved_int(
            engine, "train_micro_batch_size_per_gpu"
        ),
        "gradient_accumulation_steps": pr6._resolved_int(
            engine, "gradient_accumulation_steps"
        ),
        "train_batch_size": pr6._resolved_int(engine, "train_batch_size"),
    }
    expected = {
        "dp_world_size": 8,
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 2,
        "train_batch_size": 16,
    }
    if resolved != expected or int(engine.zero_optimization_stage()) != 3:
        raise AssertionError(f"Phase-A ZeRO-3 contract drift: {resolved}")
    pr6._require_process_group_ranks(
        engine.data_parallel_group,
        expected=tuple(range(8)),
        label="Phase-A Generator DP group",
    )
    restore = restore_zero3_optimizer_partition_from_logical(
        engine,
        logical,
        component="generator",
        partition_rank=local_rank,
        dp_world_size=8,
    )
    versions = (
        int(engine.global_steps),
        pr6._optimizer_step(engine.optimizer),
        pr6._scheduler_step(engine),
    )
    if versions != (2, 2, 2):
        raise AssertionError(
            "Phase-A Generator did not restore source optimizer step 2: "
            f"{versions}"
        )
    leaf_inventory = list(get_z3_leaf_modules(engine.module))
    if len(leaf_inventory) != 40:
        raise AssertionError("Phase-A ZeRO-3 block leaves changed after initialize")
    return engine, {
        "backend": "NODE_LOCAL_DEFAULT_WORLD8_ZERO3",
        "resolved": resolved,
        "config": config,
        "logical_restore": restore,
        "parameter_version": 200,
        "explicit_block_leaf_count": len(leaf_inventory),
    }


def _build_cpu_model(
    *,
    args: argparse.Namespace,
    role: str,
    logical: LogicalCheckpointPackage,
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    if role == "generator":
        model, metadata = build_helios_pool_model(
            role="generator",
            model_root=args.generator_root,
            subfolder="transformer_ode",
            precast_model_root=args.precast_generator_root,
        )
        restore = load_logical_trainables(model, logical.root, "generator")
        metadata["resident_trainable_cast"] = (
            pr6._cast_resident_generator_trainables_to_bf16(model)
        )
    elif role == "critic":
        model, metadata = build_helios_pool_model(
            role="critic",
            model_root=args.score_root,
            subfolder="transformer",
            precast_model_root=args.precast_score_root,
            enable_gan=True,
        )
        restore = load_logical_trainables(model, logical.root, "critic")
    else:
        raise ValueError(role)
    if any(parameter.is_cuda for parameter in model.parameters()):
        raise AssertionError("preconstructed transition target entered CUDA")
    return model, metadata, restore


def _preconstruct_targets(
    *,
    args: argparse.Namespace,
    global_rank: int,
    logical: LogicalCheckpointPackage,
    control_group: Any,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "forward_model": None,
        "forward_metadata": None,
        "forward_restore": None,
        "forward_pin": None,
        "reverse_model": None,
        "reverse_metadata": None,
        "reverse_restore": None,
        "reverse_pin": None,
        "reverse_policy": "LAZY_AFTER_PHASE_B_CRITIC_RELEASE",
    }
    if global_rank in PHASE_B_G_RANKS:
        model, metadata, restore = _build_cpu_model(
            args=args, role="generator", logical=logical
        )
        value.update(
            forward_model=model,
            forward_metadata=metadata,
            forward_restore=restore,
            forward_pin=pr6.pin_module_cpu_registered_state(model),
        )
    elif global_rank in FLEX_RANKS:
        model, metadata, restore = _build_cpu_model(
            args=args, role="critic", logical=logical
        )
        value.update(
            forward_model=model,
            forward_metadata=metadata,
            forward_restore=restore,
            forward_pin=pr6.pin_module_cpu_registered_state(model),
        )
    pr6._control_barrier(control_group)
    return value


def _phase_a_payload(
    *,
    tensor: torch.Tensor,
    source_id: int,
    trace_hash: str,
    trace_identity: str,
    timing_mode: bool,
    kind: str,
) -> dict[str, Any]:
    checksum_kind, checksum = _fingerprint(tensor, timing_mode=timing_mode)
    return {
        "schema_version": f"distillschedule.gan_rolephase_phase_a_{kind}.v1",
        "logical_step": 1000,
        "source_id": source_id,
        "wave": source_id // 8,
        "lane": source_id % 8,
        "generator_version": 200,
        "critic_version": 1000,
        "trace_hash": trace_hash,
        "trace_identity_sha256": trace_identity,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "checksum_kind": checksum_kind,
        "checksum": checksum,
        "tensor": tensor.detach().cpu().contiguous(),
    }


def _validate_phase_a_payload(
    payload: dict[str, Any],
    *,
    source_id: int,
    trace_hash: str,
    trace_identity: str,
    timing_mode: bool,
    kind: str,
    device: torch.device,
) -> torch.Tensor:
    expected = {
        "schema_version": f"distillschedule.gan_rolephase_phase_a_{kind}.v1",
        "logical_step": 1000,
        "source_id": source_id,
        "wave": source_id // 8,
        "lane": source_id % 8,
        "generator_version": 200,
        "critic_version": 1000,
        "trace_hash": trace_hash,
        "trace_identity_sha256": trace_identity,
        "shape": [1, 16, 9, 48, 80],
        "dtype": "torch.bfloat16",
    }
    mismatch = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatch:
        raise ValueError(f"Phase-A {kind} identity mismatch: {mismatch}")
    cpu = payload.get("tensor")
    if not isinstance(cpu, torch.Tensor):
        raise TypeError(f"Phase-A {kind} payload has no tensor")
    value = cpu.to(device=device, dtype=torch.bfloat16)
    _validate_fingerprint(
        value,
        kind=str(payload["checksum_kind"]),
        value=str(payload["checksum"]),
        timing_mode=timing_mode,
    )
    return value


def _run_phase_a_generator(
    *,
    args: argparse.Namespace,
    local_rank: int,
    device: torch.device,
    engine: Any,
    trace: GanRolePhaseTrace,
    exchange: Path,
    result: dict[str, Any],
) -> None:
    records: list[dict[str, Any]] = []
    states: list[dict[str, Any]] = []
    before = TIMING_OMITTED if args.timing_mode else trainable_checksum(engine.module)
    global_before = int(engine.global_steps)
    for wave in range(2):
        source_id = wave * 8 + local_rank
        entry = trace.load_generator_entry(source_id=source_id, device=device)
        started = time.perf_counter()
        x0, gradient_mask, details = run_generator_update_trace_rollout(
            engine=engine,
            entry=entry,
            device=device,
            detailed_profile=not args.timing_mode,
        )
        payload = _phase_a_payload(
            tensor=x0,
            source_id=source_id,
            trace_hash=entry.operation_trace.trace_hash,
            trace_identity=entry.identity_sha256,
            timing_mode=args.timing_mode,
            kind="x0",
        )
        _atomic_torch(_phase_a_input_path(exchange, source_id), payload)
        record = {
            "logical_step": 1000,
            "wave": wave,
            "source_id": source_id,
            "trace_hash": entry.operation_trace.trace_hash,
            "trace_identity_sha256": entry.identity_sha256,
            "x0_checksum_kind": payload["checksum_kind"],
            "x0_checksum": payload["checksum"],
            "rollout_seconds": time.perf_counter() - started,
            "rollout": details,
            "schedule_event": "selected_graph_published_before_gradient_wait",
        }
        records.append(record)
        states.append(
            {
                "entry": entry,
                "x0": x0,
                "gradient_mask": gradient_mask,
                "record": record,
            }
        )
        del payload

    boundaries = 0
    for state in states:
        entry = state["entry"]
        source_id = entry.source_id
        path = _phase_a_gradient_path(exchange, source_id)
        wait_seconds = _wait(path, args.timeout_seconds)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        gradient = _validate_phase_a_payload(
            payload,
            source_id=source_id,
            trace_hash=entry.operation_trace.trace_hash,
            trace_identity=entry.identity_sha256,
            timing_mode=args.timing_mode,
            kind="gradient",
            device=device,
        )
        # ``compute_official_score_boundary_grad`` returns the *already
        # normalized* derivative of DMD + gan_g_weight * GAN-G with respect to
        # x0.  The validated PR7 Helios seam injects it with a SUM surrogate.
        # Reusing the ordinary DMD 0.5*MSE surrogate here would divide the
        # derivative by the latent element count a second time and silently
        # change the Phase-A update.
        loss = _remote_gradient_sum_surrogate(state["x0"], gradient)
        started = time.perf_counter()
        engine.backward(loss)
        boundary = bool(engine.is_gradient_accumulation_boundary())
        boundaries += int(boundary)
        engine.step()
        torch.cuda.synchronize(device)
        state["record"].update(
            gradient_wait_seconds=wait_seconds,
            gradient_checksum_kind=payload["checksum_kind"],
            gradient_checksum=payload["checksum"],
            backward_step_seconds=time.perf_counter() - started,
            gradient_accumulation_boundary=boundary,
            surrogate_loss=(
                TIMING_OMITTED if args.timing_mode else float(loss.detach().cpu())
            ),
        )
        state["x0"] = None
        state["gradient_mask"] = None
        del payload, gradient, loss, entry
    if (
        int(engine.global_steps) != TARGET_GENERATOR_OPTIMIZER_STEP
        or global_before != SOURCE_GENERATOR_OPTIMIZER_STEP
        or boundaries != 1
    ):
        raise AssertionError("Phase A must perform exactly one G200->G201 commit")
    after = TIMING_OMITTED if args.timing_mode else trainable_checksum(engine.module)
    if not args.timing_mode and before == after:
        raise AssertionError("Phase-A Generator trainables did not change")
    result["phase_a"] = {
        "role": "generator",
        "schedule": "two_selected_graphs_first_then_two_backwards",
        "generator_before": 200,
        "generator_after": 201,
        "accumulation_boundaries": boundaries,
        "samples": records,
        "trainable_before": before,
        "trainable_after": after,
        "optimizer": (
            TIMING_OMITTED
            if args.timing_mode
            else optimizer_state_checksum(engine.optimizer)
        ),
    }


def _run_phase_a_score(
    *,
    args: argparse.Namespace,
    local_rank: int,
    device: torch.device,
    model: torch.nn.Module,
    cfg: Any,
    negative: torch.Tensor,
    trace: GanRolePhaseTrace,
    exchange: Path,
    result: dict[str, Any],
) -> None:
    records: list[dict[str, Any]] = []
    for wave in range(2):
        source_id = wave * 8 + local_rank
        entry = trace.load_generator_entry(source_id=source_id, device=device)
        source = trace.source(source_rank=source_id, device=device)
        wait_seconds = _wait(
            _phase_a_input_path(exchange, source_id), args.timeout_seconds
        )
        payload = torch.load(
            _phase_a_input_path(exchange, source_id),
            map_location="cpu",
            weights_only=False,
        )
        x0 = _validate_phase_a_payload(
            payload,
            source_id=source_id,
            trace_hash=entry.operation_trace.trace_hash,
            trace_identity=entry.identity_sha256,
            timing_mode=args.timing_mode,
            kind="x0",
            device=device,
        )
        before_calls = int(result.get("score_forward_calls", 0))
        started = time.perf_counter()
        boundary = compute_official_score_boundary_grad(
            score_model=model,
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
            collect_diagnostics=not args.timing_mode,
        )
        gradient_payload = _phase_a_payload(
            tensor=boundary.gradient,
            source_id=source_id,
            trace_hash=entry.operation_trace.trace_hash,
            trace_identity=entry.identity_sha256,
            timing_mode=args.timing_mode,
            kind="gradient",
        )
        _atomic_torch(_phase_a_gradient_path(exchange, source_id), gradient_payload)
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise AssertionError("read-only Phase-A Score created parameter gradients")
        records.append(
            {
                "logical_step": 1000,
                "wave": wave,
                "source_id": source_id,
                "trace_hash": entry.operation_trace.trace_hash,
                "trace_identity_sha256": entry.identity_sha256,
                "input_wait_seconds": wait_seconds,
                "compute_seconds": time.perf_counter() - started,
                "total_loss": (
                    TIMING_OMITTED if args.timing_mode else boundary.total_loss
                ),
                "dmd_loss": (
                    TIMING_OMITTED if args.timing_mode else boundary.dmd_loss
                ),
                "gan_g_loss_raw": (
                    TIMING_OMITTED if args.timing_mode else boundary.gan_g_loss_raw
                ),
                "gradient_checksum_kind": gradient_payload["checksum_kind"],
                "gradient_checksum": gradient_payload["checksum"],
                "x0_checksum_kind": payload["checksum_kind"],
                "x0_checksum": payload["checksum"],
                "expected_score_forwards": 4,
                "score_forward_calls_before": before_calls,
            }
        )
        del entry, source, payload, x0, boundary, gradient_payload
    result["phase_a"] = {
        "role": "score_gradient_service",
        "critic_version": 1000,
        "optimizer_commits": 0,
        "samples": records,
    }


@torch.no_grad()
def _capture_zero3_optimizer(engine: Any) -> dict[str, Any]:
    zero = engine.optimizer
    names: list[str] = []
    state: dict[str, dict[str, torch.Tensor]] = {}
    started = time.perf_counter()
    for name, parameter in engine.module.named_parameters():
        if not parameter.requires_grad:
            continue
        names.append(name)
        values: dict[str, torch.Tensor] = {}
        for target, key in (
            ("master", None),
            ("exp_avg", "exp_avg"),
            ("exp_avg_sq", "exp_avg_sq"),
        ):
            value = zero.get_local_fp32_param(parameter, key)
            values[target] = value.detach().cpu().contiguous()
            del value
        state[name] = values
    scheduler = dict(engine.lr_scheduler.state_dict())
    return {
        "parameter_names": names,
        "state": state,
        "logical_parameter_version": 201,
        "source_optimizer_step": int(engine.global_steps),
        "optimizer_step": pr6._optimizer_step(zero),
        "scheduler_state": scheduler,
        "capture_seconds": time.perf_counter() - started,
    }


@torch.no_grad()
def _restore_captured_zero3(engine: Any, captured: dict[str, Any]) -> dict[str, Any]:
    if (
        int(captured["logical_parameter_version"]),
        int(captured["source_optimizer_step"]),
        int(captured["optimizer_step"]),
    ) != (
        201,
        TARGET_GENERATOR_OPTIMIZER_STEP,
        TARGET_GENERATOR_OPTIMIZER_STEP,
    ):
        raise AssertionError("captured Generator optimizer is not G201")
    trainable = {
        name: parameter
        for name, parameter in engine.module.named_parameters()
        if parameter.requires_grad
    }
    if list(trainable) != list(captured["parameter_names"]):
        raise AssertionError("reverse ZeRO-3 trainable order changed")
    started = time.perf_counter()
    for name, parameter in trainable.items():
        values = captured["state"][name]
        for target, key in (
            ("master", None),
            ("exp_avg", "exp_avg"),
            ("exp_avg_sq", "exp_avg_sq"),
        ):
            engine.optimizer.set_local_hp_param(values[target], parameter, key)
        shard = parameter.ds_tensor
        shard.copy_(values["master"].to(device=shard.device, dtype=shard.dtype))
    engine.global_steps = TARGET_GENERATOR_OPTIMIZER_STEP
    engine.lr_scheduler.load_state_dict(dict(captured["scheduler_state"]))
    base_optimizer = pr6._unwrap_optimizer(engine.optimizer)
    for value in base_optimizer.state.values():
        if isinstance(value, dict) and "step" in value:
            value["step"] = torch.tensor(
                float(TARGET_GENERATOR_OPTIMIZER_STEP),
                dtype=torch.float32,
                device=value["exp_avg"].device,
            )
    return {
        "parameter_version": 201,
        "optimizer_step": pr6._optimizer_step(engine.optimizer),
        "scheduler_step": pr6._scheduler_step(engine),
        "tensor_count": len(trainable),
        "restore_seconds": time.perf_counter() - started,
    }


def _initialize_phase_b_critic(
    *,
    args: argparse.Namespace,
    model: torch.nn.Module,
    logical: LogicalCheckpointPackage,
    global_rank: int,
    critic_group: Any,
    singleton_group: Any,
    device: torch.device,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    import deepspeed

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=4e-7,
        betas=(0.0, 0.999),
        eps=1e-8,
        weight_decay=1e-3,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    adapter = StudentDPGroupAdapter(
        data_parallel_group=critic_group,
        data_parallel_ranks=PHASE_B_C_RANKS,
        global_rank=global_rank,
        singleton_model_parallel_group=singleton_group,
    )
    config = phase_b_zero1_config()
    started = time.perf_counter()
    engine, _, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        config=config,
        mpu=adapter,
        dist_init_required=False,
    )
    init_seconds = time.perf_counter() - started
    resolved = assert_phase_b_zero1_contract(
        engine, expected_dp_group=critic_group
    )
    restore = restore_zero1_optimizer_partition_from_logical(
        engine,
        logical,
        component="critic",
        partition_rank=global_rank - 4,
        dp_world_size=12,
    )
    versions = (
        int(engine.global_steps),
        pr6._optimizer_step(engine.optimizer),
        pr6._scheduler_step(engine),
    )
    if versions != (10, 10, 10):
        raise AssertionError(
            "Phase-B Critic did not restore source optimizer step 10: "
            f"{versions}"
        )

    if args.phase_b_streamed_blocks == 0:
        residency = pr6._ResidentStudentParameterGuard(engine.module, device)
        policy = "FULL_RESIDENT_SCORE"
    else:
        if args.phase_b_streamed_blocks != 4:
            raise ValueError("only preregistered Phase-B candidates k=0 or tail-4 are legal")
        residency = install_frozen_block_streamer(
            engine.module,
            device=device,
            window_size=1,
            streamed_block_count=4,
            streamed_block_indices=(36, 37, 38, 39),
            host_policy=HostStatePolicy.HOST_SHARED_IMMUTABLE,
            optimizer_owned_parameter_ids={
                id(parameter)
                for parameter in engine.module.parameters()
                if parameter.requires_grad
            },
            checkpoint_identity=f"{args.score_root}::gan_bf16",
            event_path=args.output_dir / "events" / f"rank_{global_rank:02d}_phase_b_stream.jsonl",
            mmap_cache_root=args.mmap_cache_root / "score",
            detailed_events=not args.timing_mode,
        )
        policy = "TAIL4_FROZEN_BLOCK_STREAMING_CAPACITY_FALLBACK"
    activation = install_activation_offload_hooks(
        engine.module,
        logical_component="gan_rolephase_phase_b_critic",
        event_path=args.output_dir / "events" / f"rank_{global_rank:02d}_phase_b_activation.jsonl",
        pin_memory=True,
        min_offload_bytes=128 * 2**20,
        detailed_events=False,
        dynamic_excluded_storage_providers=(
            (residency.dynamic_excluded_storage_owner,)
            if hasattr(residency, "dynamic_excluded_storage_owner")
            else ()
        ),
    )
    return engine, residency, activation, {
        "backend": "DP12_ZERO1_CPU_OPTIMIZER",
        "config": config,
        "resolved": resolved,
        "logical_restore": restore,
        "initialize_seconds": init_seconds,
        "parameter_policy": policy,
        "activation_policy": "PINNED_SAVED_TENSOR_OFFLOAD_A128",
    }


def _phase_b_route(entry: Any, *, step: int, slot: int):
    mapping = TOPOLOGY.phase_b_slot(slot)
    return make_phase_b_route(
        logical_step=step,
        logical_slot=slot,
        generator_rank=mapping.generator_rank,
        critic_rank=mapping.critic_rank,
        generator_version=201,
        critic_before=1000 + (step - 1000),
        trace_sha256=bytes.fromhex(entry.pool_entry.identity_sha256),
    )


def _produce_phase_b_batch(
    *,
    args: argparse.Namespace,
    model: torch.nn.Module,
    trace: GanRolePhaseTrace,
    global_rank: int,
    device: torch.device,
    update: int,
) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for production_round in range(3):
        entry = trace.load_critic_entry(
            update=update,
            production_round=production_round,
            producer_pool_rank=global_rank,
            device=device,
        )
        started = time.perf_counter()
        x0, details = run_generator_trace_rollout(
            model=model,
            loaded=entry,
            device=device,
            detailed_profile=not args.timing_mode,
            record_output_sha256=not args.timing_mode,
            empty_cache_before_rollout=False,
        )
        slot = production_round * 4 + global_rank
        route = _phase_b_route(
            entry, step=1000 + update, slot=slot
        )
        checksum = tensor_identity(x0, ChecksumKind.GPU_FINGERPRINT_V1)
        header = make_phase_b_header(
            run_nonce=args.run_nonce,
            route=route,
            payload_checksum=checksum,
            checksum_kind=ChecksumKind.GPU_FINGERPRINT_V1,
        )
        values.append(
            {
                "entry": entry,
                "x0": x0,
                "route": route,
                "header": header,
                "record": {
                    "logical_step": 1000 + update,
                    "logical_slot": slot,
                    "production_round": production_round,
                    "trace_hash": entry.operation_trace.trace_hash,
                    "trace_identity_sha256": entry.pool_entry.identity_sha256,
                    "destination_global_rank": route.destination_global_rank,
                    "x0_checksum": checksum.hex(),
                    "produce_seconds": time.perf_counter() - started,
                    "rollout": details,
                },
            }
        )
    return values


def _run_phase_b_generator(
    *,
    args: argparse.Namespace,
    global_rank: int,
    device: torch.device,
    model: torch.nn.Module,
    trace: GanRolePhaseTrace,
    control_group: Any,
    ema: SparseShardEMA | None,
    result: dict[str, Any],
) -> None:
    transport = RolePhaseCriticP2PTransport(
        group=dist.group.WORLD,
        device=device,
        global_rank=global_rank,
    )
    initial = TIMING_OMITTED if args.timing_mode else trainable_sha256(model)
    records: list[dict[str, Any]] = []
    ema_records: list[dict[str, Any]] = []
    produced = committed = sent = 0
    max_uncommitted = 0

    def produce(update: int):
        nonlocal produced, max_uncommitted
        batch = _produce_phase_b_batch(
            args=args,
            model=model,
            trace=trace,
            global_rank=global_rank,
            device=device,
            update=update,
        )
        produced += 1
        max_uncommitted = max(max_uncommitted, produced - committed)
        if produced - committed > 2:
            raise AssertionError("Phase-B producer exceeded Q2")
        return batch

    def post(batch: list[dict[str, Any]]):
        nonlocal sent
        for item in batch:
            item["pending"] = transport.send(
                item["header"], item["x0"], route=item["route"]
            )
            sent += 1
        return batch

    current = post(produce(0))
    for update in range(5):
        wait_started = time.perf_counter()
        for item in current:
            transport.reap_send_completion(item["pending"])
        send_wait = time.perf_counter() - wait_started
        for item in current:
            item["record"]["send_completion_wait_seconds"] = send_wait
            records.append(item["record"])
            item["x0"] = None
            item["entry"] = None
        future = produce(update + 1) if update < 4 else None
        pr6._control_barrier(control_group)
        committed += 1
        if update == 0 and global_rank == 0:
            if ema is None:
                raise AssertionError("rank0 lacks sparse EMA800")
            before = ema.optimization_step
            started = time.perf_counter()
            decay = ema.step(model.parameters())
            if (before, ema.optimization_step) != (800, 801):
                raise AssertionError("official GAN EMA800->EMA801 recurrence changed")
            ema_records.append(
                {
                    "logical_step": 1000,
                    "before": before,
                    "after": ema.optimization_step,
                    "decay": decay,
                    "seconds": time.perf_counter() - started,
                    "checksum": TIMING_OMITTED if args.timing_mode else ema.checksum(),
                }
            )
        pr6._control_barrier(control_group)
        if future is not None:
            current = post(future)

    transport.flush()
    transport.assert_flushed()
    final = TIMING_OMITTED if args.timing_mode else trainable_sha256(model)
    if not args.timing_mode and final != initial:
        raise AssertionError("Generator changed during Phase B")
    gradients = (
        {"measurement": TIMING_OMITTED}
        if args.timing_mode
        else parameter_gradient_diagnostics(model)
    )
    if not args.timing_mode and (
        gradients["trainable_grad_present"] or gradients["frozen_grad_present"]
    ):
        raise AssertionError("Generator retained Phase-B gradients")
    if (produced, committed, sent) != (5, 5, 15):
        raise AssertionError("Phase-B Generator queue accounting changed")
    if global_rank == 0 and len(ema_records) != 1:
        raise AssertionError("GAN Full5 requires exactly one EMA update")
    result["phase_b"] = {
        "role": "generator",
        "generator_version_before": 201,
        "generator_version_after": 201,
        "samples": records,
        "ema": ema_records,
        "queue": {
            "produced_global_batches": produced,
            "committed_global_batches": committed,
            "sent_payloads": sent,
            "produced_minus_committed_max": max_uncommitted,
            "per_route_optimizer_ack": False,
            "payload_credit": "NCCL_SEND_COMPLETION",
            "producer_slot": "EMPTY",
            "outstanding_transport": 0,
        },
        "trainable_sha256_before": initial,
        "trainable_sha256_after": final,
        "gradients": gradients,
    }


def _run_phase_b_critic(
    *,
    args: argparse.Namespace,
    cfg: Any,
    global_rank: int,
    device: torch.device,
    engine: Any,
    trace: GanRolePhaseTrace,
    control_group: Any,
    result: dict[str, Any],
) -> None:
    from helios.utils.utils_helios_post import _critic_loss

    slot = global_rank - 4
    production_round, producer = divmod(slot, 4)
    transport = RolePhaseCriticP2PTransport(
        group=dist.group.WORLD,
        device=device,
        global_rank=global_rank,
    )
    records: list[dict[str, Any]] = []
    for update in range(5):
        step = 1000 + update
        entry = trace.load_critic_entry(
            update=update,
            production_round=production_round,
            producer_pool_rank=producer,
            device=device,
        )
        route = _phase_b_route(entry, step=step, slot=slot)
        pending = transport.post_receive(route=route)
        wait_started = time.perf_counter()
        header, x0 = transport.finish_receive(pending)
        receive_wait = time.perf_counter() - wait_started
        if header.run_nonce != args.run_nonce:
            raise AssertionError("Phase-B run nonce changed")
        source = trace.source(source_rank=slot, device=device)
        kwargs = critic_official_kwargs(
            cfg,
            entry.operation_trace,
            source,
            step,
            x0,
            engine,
        )
        started = time.perf_counter()
        loss, logs = _critic_loss(fake_score_model=engine.module, **kwargs)
        compute_seconds = time.perf_counter() - started
        expected_optimizer_step = SOURCE_CRITIC_OPTIMIZER_STEP + update + 1
        expected_logical_version = 1001 + update
        if not bool(logs.get("critic_optimizer_committed")):
            raise AssertionError("DP12/B1 Critic call did not commit")
        if int(engine.global_steps) != expected_optimizer_step:
            raise AssertionError(
                "Critic optimizer-step drift after "
                f"Step{step}: {engine.global_steps} != {expected_optimizer_step}"
            )
        records.append(
            {
                "logical_step": step,
                "logical_slot": slot,
                "source_generator_rank": producer,
                "trace_hash": entry.operation_trace.trace_hash,
                "trace_identity_sha256": entry.pool_entry.identity_sha256,
                "x0_checksum_kind": header.checksum_kind.name,
                "x0_checksum": header.payload_checksum.hex(),
                "receive_wait_seconds": receive_wait,
                "critic_compute_commit_seconds": compute_seconds,
                "loss": TIMING_OMITTED if args.timing_mode else float(loss.detach()),
                "critic_before": 1000 + update,
                "critic_after": expected_logical_version,
                "optimizer_step_before": expected_optimizer_step - 1,
                "optimizer_step_after": expected_optimizer_step,
                "optimizer_committed": True,
                "physical_critic_loss_calls": 1,
                "expected_score_forwards": 2,
                "transport_micro_wave": 0,
                "helios_pipeline_micro_wave": None,
            }
        )
        pr6._control_barrier(control_group)
        pr6._control_barrier(control_group)
        del entry, source, kwargs, loss, logs, x0
    transport.flush()
    transport.assert_flushed()
    if int(engine.global_steps) != TARGET_CRITIC_OPTIMIZER_STEP:
        raise AssertionError("Phase B did not reach C1005")
    result["phase_b"] = {
        "role": "critic",
        "critic_before": 1000,
        "critic_after": 1005,
        "critic_updates": 5,
        "samples": records,
        "queue": {"consumer_slot": "EMPTY", "outstanding_transport": 0},
        "optimizer": (
            TIMING_OMITTED
            if args.timing_mode
            else optimizer_state_checksum(engine.optimizer)
        ),
    }


def _release_obsolete_zero1_engine(engine: Any, *, device: torch.device) -> dict[str, Any]:
    before = pr6._snapshot(device, "before_obsolete_zero1_release")
    module = engine.module
    optimizer = engine.optimizer
    for parameter in module.parameters():
        parameter.grad = None
    release = pr6._destroy_obsolete_zero1_gpu_flat_buffers(optimizer)
    try:
        engine.destroy()
    except BaseException:
        pass
    modules = engine.__dict__.get("_modules")
    if isinstance(modules, dict):
        modules["module"] = None
    engine.__dict__["module"] = None
    engine.optimizer = None
    engine.lr_scheduler = None
    del module, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    return {
        "policy": "DISCARD_OBSOLETE_PHASE_B_CRITIC_ON_FLEX_RANK",
        "before": before,
        "after": pr6._snapshot(device, "after_obsolete_zero1_release"),
        "zero1": release,
    }


def _reverse_to_phase_a(
    *,
    args: argparse.Namespace,
    global_rank: int,
    local_rank: int,
    device: torch.device,
    logical: LogicalCheckpointPackage,
    captured_g201: dict[str, Any] | None,
    generator_model: torch.nn.Module | None,
    critic_engine: Any | None,
    reverse_model: torch.nn.Module | None,
    control_group: Any,
    exchange: Path,
    local_rendezvous: dict[str, str],
) -> tuple[dict[str, Any], Any | None, torch.nn.Module | None]:
    record: dict[str, Any] = {}
    pr6._control_barrier(control_group)
    if global_rank in FLEX_RANKS:
        if critic_engine is None:
            raise AssertionError("flex rank lacks obsolete Phase-B Critic")
        record["flex_critic_release"] = _release_obsolete_zero1_engine(
            critic_engine, device=device
        )
        critic_engine = None
        if reverse_model is not None:
            raise AssertionError(
                "flex reverse Generator must not outlive Phase-B Critic"
            )
        reverse_model, reverse_metadata, reverse_restore = _build_cpu_model(
            args=args, role="generator", logical=logical
        )
        reverse_host_preparation = pr6.pin_module_cpu_registered_state(
            reverse_model
        )
        reverse_host_preparation["timing_scope"] = "included_reverse_handoff"
        record["reverse_target_construction"] = {
            "policy": "LAZY_AFTER_PHASE_B_CRITIC_RELEASE",
            "metadata": reverse_metadata,
            "logical_restore": reverse_restore,
            "host_preparation": reverse_host_preparation,
        }
    pr6._control_barrier(control_group)
    dist.destroy_process_group()
    if dist.is_initialized():
        raise AssertionError("global Phase-B world survived reverse handoff")
    _restore_local_phase_a_rendezvous(
        local_rendezvous,
        master_port=args.phase_a_reverse_master_port,
    )
    pr6._initialize_phase_a_distributed()
    import deepspeed

    deepspeed.init_distributed(dist_backend="nccl", dist_init_required=False)
    if int(deepspeed.comm.get_world_size()) != PHASE_A_WORLD_SIZE:
        raise AssertionError("DeepSpeed did not rebind to reverse Phase-A world8")
    local_group = pr6._create_phase_a_groups(node_rank=args.node_rank)

    reverse_engine = None
    if args.node_rank == 0:
        if global_rank in PHASE_B_G_RANKS:
            model = generator_model
            if model is None:
                raise AssertionError("fixed Generator rank lost G201 model")
            materialization = {"policy": "REUSE_RESIDENT_G201"}
        else:
            model = reverse_model
            if model is None:
                raise AssertionError("flex rank lacks reverse Generator target")
            materialization = pr6.materialize_module_from_persistent_pinned_cpu(
                model,
                device=device,
                cleanup_policy="post_full5",
            )
        redistribution = pr6._broadcast_generator_g3(
            model=model,
            logical=logical,
            rank=global_rank,
            group=local_group,
            timing_mode=args.timing_mode,
            bucket_mib=args.generator_broadcast_bucket_mib,
        )
        redistribution["parameter_version"] = 201
        reverse_engine, initialization = _initialize_phase_a_generator(
            args=args,
            model=model,
            logical=logical,
            local_rank=local_rank,
        )
        if captured_g201 is None:
            raise AssertionError("rank lacks captured post-update ZeRO-3 optimizer state")
        restore = _restore_captured_zero3(reverse_engine, captured_g201)
        if (
            int(reverse_engine.global_steps),
            pr6._optimizer_step(reverse_engine.optimizer),
            pr6._scheduler_step(reverse_engine),
        ) != (3, 3, 3):
            raise AssertionError("reverse Phase-A Generator is not G201-ready")
        record.update(
            role="phase_a_generator",
            materialization=materialization,
            redistribution=redistribution,
            zero3_initialization=initialization,
            captured_optimizer_restore=restore,
        )
    else:
        if (
            critic_engine is None
            or int(critic_engine.global_steps) != TARGET_CRITIC_OPTIMIZER_STEP
        ):
            raise AssertionError("fixed Score rank did not retain C1005")
        model = critic_engine.module
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise AssertionError("C1005 Score retained gradients at reverse boundary")
        record.update(
            role="phase_a_score_gradient_service",
            parameter_version=1005,
            physical_model_reused=True,
            repeated_backbone_materialization=False,
        )

    pr6._phase_a_file_barrier(
        exchange=exchange,
        name="reverse_phase_a_ready",
        node_rank=args.node_rank,
        control_group=local_group,
        timeout_seconds=args.timeout_seconds,
    )
    return record, reverse_engine, model


def _parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--node-rank", type=int, choices=(0, 1), required=True)
    value.add_argument("--phase-b-master-addr", required=True)
    value.add_argument("--phase-b-master-port", type=int, required=True)
    value.add_argument("--phase-a-reverse-master-port", type=int, required=True)
    value.add_argument("--run-id", required=True)
    value.add_argument("--run-nonce", type=int, required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--config", type=Path, required=True)
    value.add_argument("--logical-package", type=Path, required=True)
    value.add_argument("--expected-logical-manifest-hash", required=True)
    value.add_argument("--trace-bundle", type=Path, required=True)
    value.add_argument("--generator-root", type=Path, required=True)
    value.add_argument("--precast-generator-root", type=Path, required=True)
    value.add_argument("--score-root", type=Path, required=True)
    value.add_argument("--precast-score-root", type=Path, required=True)
    value.add_argument("--negative-prompt", type=Path, required=True)
    value.add_argument("--mmap-cache-root", type=Path, required=True)
    value.add_argument("--timeout-seconds", type=float, default=7200.0)
    value.add_argument("--timing-mode", action="store_true")
    value.add_argument("--phase-b-streamed-blocks", type=int, choices=(0, 4), default=0)
    value.add_argument("--phase-a-param-persistence-threshold", type=int, default=2_000_000)
    value.add_argument("--phase-a-prefetch-bucket-elements", type=int, default=402_653_184)
    value.add_argument("--phase-a-max-live-parameters", type=int, default=805_306_368)
    value.add_argument("--phase-a-max-reuse-distance", type=int, default=268_435_456)
    value.add_argument("--generator-broadcast-bucket-mib", type=int, default=256)
    value.add_argument("--validate-only", action="store_true")
    return value


def main() -> int:
    args = _parser().parse_args()
    phase_a_rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) != 8 or phase_a_rank != local_rank:
        raise ValueError("GAN role-phase must launch one independent local world8 per node")
    global_rank = args.node_rank * 8 + local_rank
    local_rendezvous = _capture_local_phase_a_rendezvous(local_rank=local_rank)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    import deepspeed

    pr6._initialize_phase_a_distributed()
    phase_a_group = pr6._create_phase_a_groups(node_rank=args.node_rank)
    if global_rank == 0:
        if args.output_dir.exists():
            raise FileExistsError(args.output_dir)
        args.output_dir.mkdir(parents=True)
        (args.output_dir / "exchange").mkdir()
    exchange = args.output_dir / "exchange"
    pr6._control_barrier(phase_a_group)
    if phase_a_rank == 0:
        pr6._wait_for_directory(exchange, args.timeout_seconds)
    pr6._control_barrier(phase_a_group)
    pr6._phase_a_file_barrier(
        exchange=exchange,
        name="output_ready",
        node_rank=args.node_rank,
        control_group=phase_a_group,
        timeout_seconds=args.timeout_seconds,
    )

    output = args.output_dir / "ranks" / f"rank_{global_rank:02d}.json"
    result: dict[str, Any] = {
        "schema_version": "distillschedule.gan_rolephase_world16_rank.v1",
        "status": "RUNNING",
        "run_id": args.run_id,
        "rank": global_rank,
        "local_rank": local_rank,
        "node_rank": args.node_rank,
        "hostname": socket.gethostname(),
        "timing_mode": bool(args.timing_mode),
        "phase_a": None,
        "phase_b": None,
        "updates": {"generator": 0, "critic": 0, "ema": 0},
        "phase_b_streamed_blocks": args.phase_b_streamed_blocks,
        "stages": [],
    }
    started = time.perf_counter()
    phase_a_engine = None
    phase_a_activation = None
    phase_b_activation = None
    phase_b_residency = None
    critic_engine = None
    generator_model = None
    reverse_engine = None
    score_forward_hook = None
    score_forward_calls = {"count": 0}
    phase_b_score_hook = None
    phase_b_score_forward_calls = {"count": 0}
    try:
        logical = LogicalCheckpointPackage.load_systems_only_gan(
            args.logical_package
        )
        if logical.manifest_hash != args.expected_logical_manifest_hash:
            raise ValueError("systems-only logical manifest hash changed")
        _assert_versions(logical)
        trace = GanRolePhaseTrace(args.trace_bundle)
        if logical.manifest.get("input_trace_hash") != trace.bundle_hash:
            raise ValueError(
                "systems checkpoint and GAN trace output bundle disagree"
            )
        cfg = load_official_config(args.config)
        result["identities"] = {
            "logical_manifest_hash": logical.manifest_hash,
            "trace_bundle_hash": trace.bundle_hash,
            "trace_source_bundle_hash": trace.source_bundle_hash,
            "trace_binding_hash": trace.binding_hash,
            "rank_control_validation": trace.rank_control_validation,
            "local_phase_a_rendezvous": local_rendezvous,
        }

        if args.validate_only:
            pr6._phase_a_file_barrier(
                exchange=exchange,
                name="validate_phase_a_ready",
                node_rank=args.node_rank,
                control_group=phase_a_group,
                timeout_seconds=args.timeout_seconds,
            )
            phase_a_group = None
            dist.destroy_process_group()
            world = pr6._initialize_phase_b_distributed(
                deepspeed,
                global_rank=global_rank,
                master_addr=args.phase_b_master_addr,
                master_port=args.phase_b_master_port,
            )
            groups = create_phase_b_execution_groups()
            control_group = groups.control
            generator_group = groups.generator
            critic_group = groups.critic
            topology = {
                "world": tuple(dist.get_process_group_ranks(dist.group.WORLD)),
                "generator": _validated_group_ranks(
                    generator_group,
                    expected=PHASE_B_G_RANKS,
                    global_rank=global_rank,
                ),
                "critic": _validated_group_ranks(
                    critic_group,
                    expected=PHASE_B_C_RANKS,
                    global_rank=global_rank,
                ),
                "singleton": tuple(dist.get_process_group_ranks(
                    groups.singleton_for_rank(global_rank)
                )),
            }
            expected_topology = {
                "world": tuple(range(16)),
                "generator": PHASE_B_G_RANKS,
                "critic": PHASE_B_C_RANKS,
                "singleton": (global_rank,),
            }
            if topology != expected_topology:
                raise AssertionError(
                    f"Phase-B topology mismatch: {topology} != {expected_topology}"
                )
            pr6._control_barrier(control_group)
            dist.destroy_process_group()
            _restore_local_phase_a_rendezvous(
                local_rendezvous,
                master_port=args.phase_a_reverse_master_port,
            )
            pr6._initialize_phase_a_distributed()
            deepspeed.init_distributed(
                dist_backend="nccl", dist_init_required=False
            )
            phase_a_group = pr6._create_phase_a_groups(node_rank=args.node_rank)
            pr6._phase_a_file_barrier(
                exchange=exchange,
                name="validate_reverse_phase_a_ready",
                node_rank=args.node_rank,
                control_group=phase_a_group,
                timeout_seconds=args.timeout_seconds,
            )
            result.update(
                status="PASS",
                validation_only=True,
                phase_b_world=world,
                phase_b_topology={
                    key: list(value) for key, value in topology.items()
                },
                reverse_phase_a={
                    "world_size": dist.get_world_size(),
                    "rank": dist.get_rank(),
                    "expected_local_rank": local_rank,
                    "master_port": args.phase_a_reverse_master_port,
                },
                duration_seconds=time.perf_counter() - started,
            )
            _atomic_json(output, result)
            return 0

        source_model, source_metadata, source_restore = _construct_phase_a_source(
            args=args,
            global_rank=global_rank,
            device=device,
            logical=logical,
            control_group=phase_a_group,
        )
        result["source_construction"] = {
            "metadata": source_metadata,
            "logical_restore": source_restore,
        }
        if global_rank in PHASE_A_G_RANKS:
            phase_a_engine, initialization = _initialize_phase_a_generator(
                args=args,
                model=source_model,
                logical=logical,
                local_rank=local_rank,
            )
            result["phase_a_initialization"] = initialization
            phase_a_activation = install_activation_offload_hooks(
                phase_a_engine.module,
                logical_component="gan_rolephase_phase_a_generator_selected",
                event_path=args.output_dir / "events" / f"rank_{global_rank:02d}_phase_a_activation.jsonl",
                pin_memory=True,
                min_offload_bytes=0,
                detailed_events=False,
                allowed_offload_signatures=pr6.STEP10A_BOUNDARY_OFFLOAD_SIGNATURES,
            )
        else:
            def count_forward(_module: Any, _inputs: Any) -> None:
                score_forward_calls["count"] += 1
            score_forward_hook = source_model.register_forward_pre_hook(count_forward)
            phase_a_activation = install_activation_offload_hooks(
                source_model,
                logical_component="gan_rolephase_phase_a_score_gradient_service",
                event_path=(
                    args.output_dir
                    / "events"
                    / f"rank_{global_rank:02d}_phase_a_score_activation.jsonl"
                ),
                pin_memory=True,
                min_offload_bytes=128 * 2**20,
                detailed_events=False,
                dynamic_excluded_storage_providers=(),
            )

        targets = _preconstruct_targets(
            args=args,
            global_rank=global_rank,
            logical=logical,
            control_group=phase_a_group,
        )
        ema = None
        if global_rank == 0:
            ema = SparseShardEMA.load_pretrained(
                targets["forward_model"], logical.root / "ema"
            )
            if ema.optimization_step != 800:
                raise AssertionError("EMA owner did not restore EMA800")
        negative = None
        if global_rank in PHASE_A_SCORE_RANKS:
            negative = NegativePromptAsset.load(args.negative_prompt).embedding.to(
                device=device, dtype=torch.bfloat16
            )

        pr6._phase_a_file_barrier(
            exchange=exchange,
            name="models_ready",
            node_rank=args.node_rank,
            control_group=phase_a_group,
            timeout_seconds=args.timeout_seconds,
        )
        result["stages"].append(pr6._snapshot(device, "models_and_targets_ready"))
        _atomic_json(output, result)
        torch.cuda.synchronize(device)
        pr6._phase_a_file_barrier(
            exchange=exchange,
            name="formal_start",
            node_rank=args.node_rank,
            control_group=phase_a_group,
            timeout_seconds=args.timeout_seconds,
        )
        full5_started = time.perf_counter()
        result["timing_start_unix_ns"] = time.time_ns()

        if global_rank in PHASE_A_G_RANKS:
            _run_phase_a_generator(
                args=args,
                local_rank=local_rank,
                device=device,
                engine=phase_a_engine,
                trace=trace,
                exchange=exchange,
                result=result,
            )
            result["updates"]["generator"] = 1
        else:
            _run_phase_a_score(
                args=args,
                local_rank=local_rank,
                device=device,
                model=source_model,
                cfg=cfg,
                negative=negative,
                trace=trace,
                exchange=exchange,
                result=result,
            )
            result["score_forward_calls_phase_a"] = score_forward_calls["count"]
            if score_forward_calls["count"] != 8:
                raise AssertionError(
                    f"Phase-A Score call inventory changed: {score_forward_calls['count']} != 8"
                )
        if phase_a_activation is not None:
            stats = dict(phase_a_activation.offloader.stats)
            phase_a_activation.remove()
            phase_a_activation = None
            if int(stats.get("live_cpu_bytes", 0)) != 0:
                raise AssertionError("Phase-A activation handles leaked")
            result["phase_a_activation_offload"] = stats

        pr6._phase_a_file_barrier(
            exchange=exchange,
            name="phase_a_complete",
            node_rank=args.node_rank,
            control_group=phase_a_group,
            timeout_seconds=args.timeout_seconds,
        )
        result["phase_a_wall_seconds"] = time.perf_counter() - full5_started

        transition_started = time.perf_counter()
        captured_g201 = None
        extraction = None
        if global_rank in PHASE_A_G_RANKS:
            captured_g201 = _capture_zero3_optimizer(phase_a_engine)
            extraction = pr6._materialize_zero3_g3_into_target(
                engine=phase_a_engine,
                target_model=(targets["forward_model"] if global_rank == 0 else None),
                logical=logical,
                rank=global_rank,
                timing_mode=args.timing_mode,
            )
            extraction["parameter_version"] = 201
        pr6._phase_a_file_barrier(
            exchange=exchange,
            name="g201_captured",
            node_rank=args.node_rank,
            control_group=phase_a_group,
            timeout_seconds=args.timeout_seconds,
        )

        source_release = None
        if global_rank in PHASE_A_G_RANKS:
            source_model = None
            source_release = pr6._destroy_obsolete_zero3_engine(
                phase_a_engine, device=device
            )
            phase_a_engine = None
            source_release = pr6._verify_obsolete_zero3_release(
                source_release, device=device
            )
        pr6._phase_a_file_barrier(
            exchange=exchange,
            name="phase_a_source_release",
            node_rank=args.node_rank,
            control_group=phase_a_group,
            timeout_seconds=args.timeout_seconds,
        )
        phase_a_group = None
        dist.destroy_process_group()
        world = pr6._initialize_phase_b_distributed(
            deepspeed,
            global_rank=global_rank,
            master_addr=args.phase_b_master_addr,
            master_port=args.phase_b_master_port,
        )
        groups = create_phase_b_execution_groups()
        control_group = groups.control
        generator_group = groups.generator
        critic_group = groups.critic

        if global_rank in PHASE_B_G_RANKS:
            model = targets["forward_model"]
            materialization = pr6.materialize_module_from_persistent_pinned_cpu(
                model, device=device, cleanup_policy="post_full5"
            )
            redistribution = pr6._broadcast_generator_g3(
                model=model,
                logical=logical,
                rank=global_rank,
                group=generator_group,
                timing_mode=args.timing_mode,
                bucket_mib=args.generator_broadcast_bucket_mib,
            )
            redistribution["parameter_version"] = 201
            generator_model = model
            generator_model.enable_adapters()
            generator_model.train()
            transition = {
                "source_role": "phase_a_generator",
                "target_role": "phase_b_generator",
                "materialization": materialization,
                "redistribution": redistribution,
                "g201_extraction": extraction,
                "source_release": source_release,
            }
        else:
            if global_rank in FLEX_RANKS:
                model = targets["forward_model"]
                materialization = pr6.materialize_module_from_persistent_pinned_cpu(
                    model, device=device, cleanup_policy="post_full5"
                )
                source = "preconstructed_pinned_gan_critic"
            else:
                model = source_model
                materialization = {"policy": "REUSE_RESIDENT_SCORE"}
                source = "reuse_phase_a_score"
            pr6._reactivate_critic_adapter(model)
            critic_engine, phase_b_residency, phase_b_activation, initialization = (
                _initialize_phase_b_critic(
                    args=args,
                    model=model,
                    logical=logical,
                    global_rank=global_rank,
                    critic_group=critic_group,
                    singleton_group=groups.singleton_for_rank(global_rank),
                    device=device,
                )
            )
            def count_phase_b_forward(_module: Any, _inputs: Any) -> None:
                phase_b_score_forward_calls["count"] += 1

            phase_b_score_hook = critic_engine.module.register_forward_pre_hook(
                count_phase_b_forward
            )
            transition = {
                "source_role": "phase_a_generator" if global_rank in FLEX_RANKS else "phase_a_score",
                "target_role": "phase_b_critic",
                "model_source": source,
                "materialization": materialization,
                "initialization": initialization,
            }
        pr6._control_barrier(control_group)
        result["handoff_to_b"] = {
            **transition,
            "phase_world": world,
            "wall_seconds": time.perf_counter() - transition_started,
        }
        result["stages"].append(pr6._snapshot(device, "phase_b_ready"))
        if not args.timing_mode:
            _atomic_json(output, result)

        phase_b_started = time.perf_counter()
        if global_rank in PHASE_B_G_RANKS:
            _run_phase_b_generator(
                args=args,
                global_rank=global_rank,
                device=device,
                model=generator_model,
                trace=trace,
                control_group=control_group,
                ema=ema,
                result=result,
            )
            if global_rank == 0:
                result["updates"]["ema"] = 1
        else:
            _run_phase_b_critic(
                args=args,
                cfg=cfg,
                global_rank=global_rank,
                device=device,
                engine=critic_engine,
                trace=trace,
                control_group=control_group,
                result=result,
            )
            result["score_forward_calls_phase_b"] = phase_b_score_forward_calls[
                "count"
            ]
            if phase_b_score_forward_calls["count"] != 10:
                raise AssertionError(
                    "Phase-B Score call inventory changed: "
                    f"{phase_b_score_forward_calls['count']} != 10"
                )
            result["updates"]["critic"] = 5
        result["phase_b_wall_seconds"] = time.perf_counter() - phase_b_started
        if phase_b_activation is not None:
            stats = dict(phase_b_activation.offloader.stats)
            phase_b_activation.remove()
            phase_b_activation = None
            if int(stats.get("live_cpu_bytes", 0)) != 0:
                raise AssertionError("Phase-B activation handles leaked")
            result["phase_b_activation_offload"] = stats

        reverse_started = time.perf_counter()
        reverse, reverse_engine, ready_model = _reverse_to_phase_a(
            args=args,
            global_rank=global_rank,
            local_rank=local_rank,
            device=device,
            logical=logical,
            captured_g201=captured_g201,
            generator_model=generator_model,
            critic_engine=critic_engine,
            reverse_model=targets["reverse_model"],
            control_group=control_group,
            exchange=exchange,
            local_rendezvous=local_rendezvous,
        )
        result["handoff_to_a"] = {
            **reverse,
            "wall_seconds": time.perf_counter() - reverse_started,
        }
        torch.cuda.synchronize(device)
        result["timing_stop_unix_ns"] = time.time_ns()
        result["full5_seconds"] = time.perf_counter() - full5_started
        result["formal_timer_scope"] = (
            "before_phase_a_first_model_compute_through_reverse_phase_a_ready"
        )
        result["final_versions"] = {
            "generator": 201,
            "critic": 1005,
            "ema": 801,
            "next_boundary": "phase_a_ready_step1005_not_executed",
        }
        result["status"] = "PASS"
        result["duration_seconds"] = time.perf_counter() - started
        result["memory"] = pr6._snapshot(device, "complete")
        _atomic_json(output, result)
        del ready_model
        return 0
    except BaseException as error:
        message = str(error).lower()
        is_oom = isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in message
        result.update(
            status="OOM" if is_oom else "FAIL",
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
            duration_seconds=time.perf_counter() - started,
        )
        try:
            result["memory"] = pr6._snapshot(device, "failure")
        except BaseException:
            pass
        _atomic_json(output, result)
        raise
    finally:
        for activation in (phase_a_activation, phase_b_activation):
            if activation is not None:
                try:
                    activation.remove()
                except BaseException:
                    pass
        if score_forward_hook is not None:
            try:
                score_forward_hook.remove()
            except BaseException:
                pass
        if phase_b_score_hook is not None:
            try:
                phase_b_score_hook.remove()
            except BaseException:
                pass
        if phase_b_residency is not None and hasattr(phase_b_residency, "remove"):
            try:
                phase_b_residency.remove()
            except BaseException:
                pass
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except BaseException:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
