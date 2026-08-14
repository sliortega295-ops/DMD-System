#!/usr/bin/env python3
"""Continuous real-14B Full5 executor: 8G4S4T -> per-node 2G6C.

The processes remain alive from the immutable G2/C10/EMA10 boundary through
Step10-A, an in-memory role/state transition, Step10-B, and steps 11--14.
Step10-A gives node 0's eight Generator ranks their own *default* world8 so
the validated ZeRO-3 DP8/MB1/GAS2/B16 backend does not run as an unsafe
subgroup of a larger default world.  After G2->G3 is gathered, both node-local
worlds are destroyed and the same sixteen processes form the Critic phase's
global 4G+12C world with DP12/MB1/GAS1/B12.

No whole-model transition checkpoint or per-route optimizer ACK appears in
the measured path.  NCCL send completion provides bounded payload
backpressure, while two world-control rendezvous per logical update order the
Critic commit and the official early-step EMA recurrence.  Sequential and
Pipeline differ only in whether the next global fake batch is produced while
the current Critic batch executes.
"""
from __future__ import annotations

import argparse
import datetime
import gc
import hashlib
import json
import os
import resource
import socket
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import psutil
import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.activation_offload import install_activation_offload_hooks
from runtime.bounded_materialization import (
    drain_deferred_host_allocator_cleanups,
    materialize_module_from_cpu_with_bounded_pinned_staging,
    materialize_module_from_persistent_pinned_cpu,
    pin_module_cpu_registered_state,
    start_deferred_host_allocator_cleanup,
)
from runtime.critic_pool import (
    ChecksumKind,
    CriticWaveHeader,
    StudentDPGroupAdapter,
)
from runtime.critic_pool_checkpoint import (
    LogicalCheckpointPackage,
    load_logical_master_into_trainables,
    restore_zero1_optimizer_partition_from_logical,
    restore_zero3_optimizer_partition_from_logical,
)
from runtime.critic_pool_helios import (
    build_helios_pool_model,
    prepare_student_batch,
    run_generator_trace_rollout,
    student_forward_loss,
)
from runtime.critic_pool_transport import (
    CriticPoolP2PTransport,
    assert_ack_matches_header,
    payload_checksum,
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
    suspend_transformers_zero3_init,
    trainable_checksum,
    trainable_sha256,
    trim_host_allocator,
)
from runtime.operation_trace import tensor_sha256
from runtime.routeb_full5 import (
    RouteBCandidateBTopology,
    RouteBFull5Contract,
)
from runtime.routeb_full5_trace import RouteBFull5Trace
from runtime.saved_tensor_census import install_saved_tensor_census_hooks
from runtime.sparse_ema import SparseShardEMA
from runtime.zero3_checkpoint import configure_block_gradient_checkpointing


CONTRACT = RouteBFull5Contract.candidate_b()
TOPOLOGY = RouteBCandidateBTopology(CONTRACT)
PHASE_A_WORLD_SIZE = 8
PHASE_B_WORLD_SIZE = 16
STEP10A_G_RANKS = tuple(range(8))
STEP10A_STUDENT_RANKS = tuple(range(8, 12))
STEP10A_TEACHER_RANKS = tuple(range(12, 16))
SWITCH_TO_CRITIC_RANKS = tuple(rank for rank in CONTRACT.critic_ranks if rank in STEP10A_G_RANKS)
SWITCH_TO_GENERATOR_RANKS = tuple(
    rank for rank in CONTRACT.generator_ranks if rank not in STEP10A_G_RANKS
)
PRECONSTRUCT_TARGET_RANKS = tuple(
    sorted((*CONTRACT.generator_ranks, *SWITCH_TO_CRITIC_RANKS))
)
STEP10A_BOUNDARY_OFFLOAD_SIGNATURES = frozenset(
    {
        ((1, 11040, 5120), "torch.bfloat16", 113_049_600),
        ((1, 4560, 5120), "torch.bfloat16", 46_694_400),
        ((1, 2940, 5120), "torch.bfloat16", 30_105_600),
    }
)


@contextmanager
def _nvtx_range(label: str):
    """Emit profiling-only NVTX ranges without perturbing formal timing runs."""

    if os.environ.get("DISTILLSCHEDULE_NSYS_ANNOTATE", "0") != "1":
        yield
        return
    torch.cuda.nvtx.range_push(label)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _persist_diagnostic_stage(
    *,
    args: argparse.Namespace,
    result: dict[str, Any],
    output: Path,
    device: torch.device,
    stage: str,
) -> None:
    if args.timing_mode:
        return
    result["diagnostic_stage"] = stage
    result["stages"].append(_snapshot(device, stage))
    _atomic_json(output, result)


def _initialize_phase_a_distributed() -> None:
    """Create one node-local default world8.

    Node 0 uses this default world for the Step10-A Generator ZeRO-3 engine;
    node 1 uses an independent world8 for Student/Teacher lifecycle fences.
    Keeping ZeRO-3 on the default world is the validated recovery from the
    bounded larger-world subgroup-forward stall.
    """

    if dist.is_initialized():
        raise RuntimeError("Phase-A distributed runtime was initialized twice")
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=2))
    if dist.get_world_size() != PHASE_A_WORLD_SIZE:
        raise AssertionError("Phase-A must be a node-local default world8")


def _phase_a_global_rank(*, node_rank: int, local_rank: int) -> int:
    if node_rank not in (0, 1) or not 0 <= local_rank < PHASE_A_WORLD_SIZE:
        raise ValueError("invalid Phase-A node/local rank")
    return node_rank * PHASE_A_WORLD_SIZE + local_rank


def _create_phase_a_groups(*, node_rank: int):
    # node_rank is checked here so a malformed launcher cannot silently create
    # two valid-looking but misidentified node-local worlds. Phase A already
    # owns one validated default NCCL world8; an extra full-membership Gloo
    # group previously stalled before model construction and adds no semantics.
    _phase_a_global_rank(node_rank=node_rank, local_rank=0)
    dist.barrier(group=dist.group.WORLD)
    return dist.group.WORLD


def _initialize_phase_b_distributed(
    deepspeed_module: Any,
    *,
    global_rank: int,
    master_addr: str,
    master_port: int,
) -> dict[str, Any]:
    if dist.is_initialized():
        raise RuntimeError("Phase-A process groups survived before Phase-B init")
    os.environ["RANK"] = str(global_rank)
    os.environ["WORLD_SIZE"] = str(PHASE_B_WORLD_SIZE)
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    elastic_agent_store = os.environ.pop("TORCHELASTIC_USE_AGENT_STORE", None)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        rank=global_rank,
        world_size=PHASE_B_WORLD_SIZE,
        timeout=datetime.timedelta(hours=2),
    )
    deepspeed_module.init_distributed(
        dist_backend="nccl", dist_init_required=False
    )
    if (
        dist.get_rank() != global_rank
        or dist.get_world_size() != PHASE_B_WORLD_SIZE
        or int(deepspeed_module.comm.get_world_size()) != PHASE_B_WORLD_SIZE
    ):
        raise AssertionError("Phase-B global world16 mapping changed")
    return {
        "global_rank": global_rank,
        "world_size": PHASE_B_WORLD_SIZE,
        "master_addr": master_addr,
        "master_port": master_port,
        "torchelastic_use_agent_store_before": elastic_agent_store,
        "torchelastic_use_agent_store_after": os.environ.get(
            "TORCHELASTIC_USE_AGENT_STORE"
        ),
    }


def _create_phase_b_groups(topology: RouteBCandidateBTopology):
    # Gloo owns long, role-asymmetric lifecycle fences so a fast role cannot
    # arm NCCL's watchdog while another role is hashing or updating CPU EMA.
    control = dist.new_group(
        ranks=list(range(16)), backend="gloo", timeout=datetime.timedelta(hours=2)
    )
    # ``new_group`` is a world-ordered operation even for ranks that are not
    # members of the subgroup.  Non-members can return before members finish
    # connecting, so creating the twelve overlapping Candidate-B P2P groups
    # back-to-back lets different ranks race ahead to different store keys.
    # Candidate A did not expose this at eight edges, while the first real
    # Candidate-B launch stalled before model construction.  Serialize this
    # cold-start-only work on the host control plane; it cannot affect the
    # measured model schedule or payload transport.
    dist.barrier(group=control)

    def ordered_group(*, ranks: list[int], backend: str):
        group = dist.new_group(ranks=ranks, backend=backend)
        dist.barrier(group=control)
        return group

    generators = ordered_group(
        ranks=list(topology.contract.generator_ranks), backend="nccl"
    )
    critics = ordered_group(
        ranks=list(topology.contract.critic_ranks), backend="nccl"
    )
    pair_groups: dict[tuple[int, int], Any] = {}
    for generator_rank, critic_rank, _round in topology.edges:
        key = (generator_rank, critic_rank)
        pair_groups[key] = ordered_group(
            ranks=[generator_rank, critic_rank], backend="nccl"
        )
    singleton = {
        rank: ordered_group(ranks=[rank], backend="nccl")
        for rank in range(16)
    }
    return (
        control,
        generators,
        critics,
        pair_groups,
        singleton,
    )


def _control_barrier(group: Any) -> None:
    dist.barrier(group=group)


@contextmanager
def _transition_phase(
    timeline: dict[str, float],
    label: str,
):
    """Record one low-overhead host-wall transition interval.

    The transition already contains the synchronization required by its state
    machine.  This helper deliberately adds no CUDA synchronize or collective;
    a recorded interval therefore means host-observed wall time at the existing
    legal boundary, not pure device execution time.
    """

    if label in timeline:
        raise AssertionError(f"transition subphase was recorded twice: {label}")
    started = time.perf_counter()
    try:
        yield
    finally:
        timeline[label] = time.perf_counter() - started


def _require_process_group_ranks(
    group: Any, *, expected: tuple[int, ...], label: str
) -> tuple[int, ...]:
    """Validate communication-domain membership, not ProcessGroup identity."""

    ranks = tuple(int(rank) for rank in dist.get_process_group_ranks(group))
    if ranks != expected:
        raise AssertionError(f"{label} ranks changed: {ranks} != {expected}")
    return ranks


def _phase_a_file_barrier(
    *,
    exchange: Path,
    name: str,
    node_rank: int,
    control_group: Any,
    timeout_seconds: float,
) -> None:
    """Cross-node fence while Phase A consists of two independent worlds."""

    _control_barrier(control_group)
    if dist.get_rank() == 0:
        marker_root = exchange / "phase_a_barriers"
        marker_root.mkdir(parents=True, exist_ok=True)
        _atomic_json(
            marker_root / f"{name}.node{node_rank}.json",
            {"name": name, "node_rank": node_rank, "status": "READY"},
        )
        _wait_for_path(
            marker_root / f"{name}.node{1 - node_rank}.json",
            timeout_seconds,
        )
    _control_barrier(control_group)


def _gib(value: int) -> float:
    return float(value) / float(2**30)


def _snapshot(device: torch.device, stage: str) -> dict[str, Any]:
    process = psutil.Process()
    usage = resource.getrusage(resource.RUSAGE_SELF)
    free, total = torch.cuda.mem_get_info(device)
    return {
        "stage": stage,
        "elapsed_monotonic_s": time.monotonic(),
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


def _destroy_obsolete_zero1_gpu_flat_buffers(optimizer: Any) -> dict[str, Any]:
    """Release ZeRO-1 GPU owners after an engine's final legal use.

    ``module.to(cpu)`` moves the Parameter views, but DeepSpeed ZeRO-1 keeps
    the original BF16 flattened owner in ``bit16_groups_flat`` and keeps
    partition views in ``parallel_partitioned_bit16_groups``.  Those aliases
    accounted for the measured 2.3635-GiB transition residual in Attempt004.
    This helper is deliberately valid only for an obsolete engine: it removes
    ZeRO hooks first, clears non-owning views, and finally drops the owners.
    """

    flat_buffers = list(getattr(optimizer, "bit16_groups_flat", ()))
    flat_cuda_bytes = sum(
        int(tensor.numel()) * int(tensor.element_size())
        for tensor in flat_buffers
        if isinstance(tensor, torch.Tensor) and tensor.is_cuda
    )
    partition_views = list(
        getattr(optimizer, "parallel_partitioned_bit16_groups", ())
    )
    partition_view_count = sum(
        len(group) if isinstance(group, (tuple, list)) else 0
        for group in partition_views
    )

    destroy = getattr(optimizer, "destroy", None)
    if not callable(destroy):
        raise AssertionError("obsolete optimizer lacks DeepSpeed destroy()")
    destroy()

    # Clear aliases/views before the owning flat tensors.  Grad collections
    # are step-local and must also be empty once the source role is obsolete.
    optimizer.averaged_gradients.clear()
    optimizer.all_grad_tensors.clear()
    optimizer.parallel_partitioned_bit16_groups.clear()
    optimizer.bit16_groups_flat.clear()
    if optimizer.parallel_partitioned_bit16_groups:
        raise AssertionError("obsolete ZeRO-1 partition views survived cleanup")
    if optimizer.bit16_groups_flat:
        raise AssertionError("obsolete ZeRO-1 flat buffers survived cleanup")
    return {
        "flat_cuda_bytes": flat_cuda_bytes,
        "flat_cuda_gib": _gib(flat_cuda_bytes),
        "flat_buffer_count": len(flat_buffers),
        "partition_view_count": partition_view_count,
    }


@torch.no_grad()
def _evict_relinquished_engine_to_cpu(
    engine: Any, *, device: torch.device
) -> dict[str, Any]:
    """Evict a role's obsolete GPU storages with the validated Helios primitive.

    Ranks 2--7 never use their Step10-A Generator again after G2->G3 has
    been retained by the target Generator pool.  DeepSpeed retains parameter
    aliases beyond local ``del`` statements, while PyTorch correctly updates
    those aliases during the standard module CPU conversion.  This one-time
    D2H is part of the measured live transition; it is not a checkpoint round
    trip and it cannot recur during the following five Critic/EMA updates.
    """

    module = engine.module
    before = {
        "allocated_gib": _gib(torch.cuda.memory_allocated(device)),
        "reserved_gib": _gib(torch.cuda.memory_reserved(device)),
    }
    for parameter in module.parameters():
        parameter.grad = None
    started = time.perf_counter()
    module.to(device=torch.device("cpu"), non_blocking=False)
    copy_wall_seconds = time.perf_counter() - started
    if any(parameter.is_cuda for parameter in module.parameters()):
        raise AssertionError("relinquished module retained a CUDA parameter")
    if any(buffer.is_cuda for buffer in module.buffers()):
        raise AssertionError("relinquished module retained a CUDA buffer")
    optimizer_release = _destroy_obsolete_zero1_gpu_flat_buffers(
        engine.optimizer
    )
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    after = {
        "allocated_gib": _gib(torch.cuda.memory_allocated(device)),
        "reserved_gib": _gib(torch.cuda.memory_reserved(device)),
    }
    if optimizer_release["flat_cuda_gib"] < 2.0:
        raise AssertionError(
            "obsolete Generator ZeRO-1 flat-buffer inventory was unexpectedly small"
        )
    if after["allocated_gib"] >= before["allocated_gib"] - 25.0:
        raise AssertionError(
            "relinquished Generator did not release its expected CUDA storage"
        )
    return {
        "policy": "EVICT_OBSOLETE_ZERO1_ROLE_TO_CPU_THEN_DESTROY_FLAT_BUFFERS",
        "d2h_copy_wall_seconds": copy_wall_seconds,
        "release_wall_seconds": time.perf_counter() - started,
        "zero1_optimizer_release": optimizer_release,
        "before": before,
        "after": after,
        "released_allocated_gib": before["allocated_gib"]
        - after["allocated_gib"],
    }


class _ResidentStudentParameterGuard:
    """Fail closed if the dedicated Student GPU stops owning the full model.

    CPU optimizer master/moment partitions are intentionally outside this
    guard.  It covers the BF16 frozen Score backbone and the live Critic LoRA
    module parameters whose storage must stay on the dedicated S GPU for the
    whole five-update experiment.
    """

    def __init__(self, model: torch.nn.Module, device: torch.device) -> None:
        self.device = torch.device(device)
        self._inventory: dict[str, tuple[int, int, str, bool]] = {}
        unique: dict[tuple[int, int], tuple[int, bool]] = {}
        for name, parameter in model.named_parameters():
            if parameter.device != self.device:
                raise AssertionError(
                    f"Student parameter {name} is not resident on {self.device}"
                )
            storage = parameter.untyped_storage()
            data_ptr = int(storage.data_ptr())
            storage_bytes = int(storage.nbytes())
            self._inventory[name] = (
                data_ptr,
                storage_bytes,
                str(parameter.dtype),
                bool(parameter.requires_grad),
            )
            unique.setdefault(
                (data_ptr, storage_bytes),
                (storage_bytes, bool(parameter.requires_grad)),
            )
        if not self._inventory:
            raise AssertionError("Student resident parameter inventory is empty")
        self.parameter_tensor_count = len(self._inventory)
        self.unique_parameter_storage_bytes = sum(
            item[0] for item in unique.values()
        )
        self.trainable_storage_bytes = sum(
            item[0] for item in unique.values() if item[1]
        )
        self.frozen_storage_bytes = sum(
            item[0] for item in unique.values() if not item[1]
        )

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
        if current != self._inventory:
            raise AssertionError(
                "Student parameter storage/device ownership changed during cycle"
            )
        if any(parameter.device != self.device for parameter in model.parameters()):
            raise AssertionError("Student parameter left its dedicated S GPU")

    def record(self) -> dict[str, Any]:
        return {
            "parameter_policy": "RESIDENT",
            "device": str(self.device),
            "parameter_tensor_count": self.parameter_tensor_count,
            "unique_parameter_storage_bytes": self.unique_parameter_storage_bytes,
            "unique_parameter_storage_gib": _gib(
                self.unique_parameter_storage_bytes
            ),
            "frozen_storage_bytes": self.frozen_storage_bytes,
            "frozen_storage_gib": _gib(self.frozen_storage_bytes),
            "trainable_storage_bytes": self.trainable_storage_bytes,
            "trainable_storage_gib": _gib(self.trainable_storage_bytes),
            "cpu_optimizer_state_excluded": True,
        }


def _unwrap_optimizer(optimizer: Any) -> Any:
    current = optimizer
    seen: set[int] = set()
    while not hasattr(current, "state") and hasattr(current, "optimizer"):
        if id(current) in seen:
            raise RuntimeError("optimizer wrapper cycle")
        seen.add(id(current))
        current = current.optimizer
    return current


def _optimizer_step(optimizer: Any) -> int:
    state = getattr(_unwrap_optimizer(optimizer), "state", {})
    steps: set[int] = set()
    for value in state.values():
        if not isinstance(value, dict) or "step" not in value:
            continue
        step = value["step"]
        if isinstance(step, torch.Tensor):
            step = int(step.item())
        steps.add(int(step))
    if len(steps) != 1:
        raise AssertionError(f"expected one optimizer step counter, got {steps}")
    return next(iter(steps))


def _scheduler_step(engine: Any) -> int:
    if engine.lr_scheduler is None:
        raise AssertionError("Student scheduler is missing")
    return int(engine.lr_scheduler.state_dict()["last_epoch"])


def _tensor_digest(tensor: torch.Tensor) -> str:
    return tensor_sha256(tensor.detach())


def _checksum_kind(args: argparse.Namespace) -> ChecksumKind:
    return (
        ChecksumKind.SHA256
        if args.checksum == "sha256"
        else ChecksumKind.GPU_FINGERPRINT_V1
    )


def _identity_record(
    tensor: torch.Tensor,
    kind: ChecksumKind,
) -> dict[str, str]:
    return {
        "kind": kind.name,
        "value": tensor_identity(tensor, kind).hex(),
    }


class _LocalGradientFingerprintObserver:
    """Hash autograd-delivered local gradients before ZeRO-1 DP reduction.

    The observer is correctness-only: each selected gradient is copied to the
    host and SHA-256 hashed inside its parameter hook.  It never retains the
    gradient tensor or changes the value returned to autograd.
    """

    def __init__(self, module: torch.nn.Module):
        self._active = False
        self._records: dict[str, dict[str, Any]] = {}
        self._handles = []
        self._expected_names: set[str] = set()
        for name, parameter in module.named_parameters():
            if not parameter.requires_grad:
                continue
            self._expected_names.add(name)

            def observe(
                gradient: torch.Tensor, *, parameter_name: str = name
            ) -> torch.Tensor:
                if not self._active:
                    return gradient
                if parameter_name in self._records:
                    raise AssertionError(
                        "local gradient delivered twice for "
                        f"{parameter_name}"
                    )
                self._records[parameter_name] = {
                    "shape": list(gradient.shape),
                    "dtype": str(gradient.dtype),
                    "numel": int(gradient.numel()),
                    "sha256": _tensor_digest(gradient),
                }
                return gradient

            self._handles.append(parameter.register_hook(observe))
        if not self._handles:
            raise ValueError("local gradient fingerprint found no trainables")

    def begin(self) -> None:
        if self._active:
            raise RuntimeError("local gradient fingerprint is already active")
        self._records.clear()
        self._active = True

    def end(
        self,
        *,
        root: Path,
        wave: int,
        rank: int,
    ) -> dict[str, Any]:
        if not self._active:
            raise RuntimeError("local gradient fingerprint is not active")
        self._active = False
        observed = set(self._records)
        if observed != self._expected_names:
            raise AssertionError(
                "local gradient fingerprint inventory changed: "
                f"missing={sorted(self._expected_names - observed)[:8]} "
                f"extra={sorted(observed - self._expected_names)[:8]}"
            )
        records = {
            name: self._records[name] for name in sorted(self._records)
        }
        inventory_payload = json.dumps(
            records, sort_keys=True, separators=(",", ":")
        ).encode()
        manifest = {
            "schema_version": (
                "distillschedule.critic_pool_local_gradient_fingerprint.v1"
            ),
            "semantic_point": (
                "autograd_parameter_hook_pre_zero1_backward_epilogue_dp_reduce"
            ),
            "wave": wave,
            "rank": rank,
            "parameter_count": len(records),
            "gradient_numel": sum(
                int(record["numel"]) for record in records.values()
            ),
            "inventory_sha256": hashlib.sha256(
                inventory_payload
            ).hexdigest(),
            "records": records,
        }
        destination = (
            root
            / "local_gradient_fingerprint"
            / f"wave_{wave:02d}"
            / f"rank_{rank:02d}"
        )
        destination.mkdir(parents=True, exist_ok=False)
        _atomic_json(destination / "manifest.json", manifest)
        self._records.clear()
        return manifest

    def close(self) -> None:
        self._active = False
        self._records.clear()
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _optimizer_partition_tensors(engine: Any) -> dict[str, torch.Tensor]:
    """Return the three rank-local FP32 ZeRO-1 state vectors.

    These vectors are exported only in correctness runs, after/before the
    measured cycle.  They make update-delta and Adam m/v comparisons
    well-conditioned instead of relying on scalar checksums.
    """

    optimizer = _unwrap_optimizer(engine.optimizer)
    masters = getattr(engine.optimizer, "single_partition_of_fp32_groups", None)
    if not isinstance(masters, (tuple, list)) or len(masters) != 1:
        raise AssertionError("expected one ZeRO-1 FP32 master partition")
    states = [value for value in optimizer.state.values() if isinstance(value, dict)]
    if len(states) != 1:
        raise AssertionError("expected one rank-local Adam state partition")
    result = {
        "master": masters[0],
        "exp_avg": states[0].get("exp_avg"),
        "exp_avg_sq": states[0].get("exp_avg_sq"),
    }
    expected_numel = int(masters[0].numel())
    for name, tensor in result.items():
        if not isinstance(tensor, torch.Tensor):
            raise AssertionError(f"missing optimizer partition tensor {name}")
        if tensor.dtype != torch.float32 or int(tensor.numel()) != expected_numel:
            raise AssertionError(
                f"{name} partition changed: dtype={tensor.dtype} "
                f"numel={tensor.numel()} expected={expected_numel}"
            )
    return result


def _export_optimizer_partition(
    *,
    engine: Any,
    root: Path,
    phase: str,
    rank: int,
) -> dict[str, Any]:
    """Write one rank's logical correctness state as raw FP32 vectors."""

    destination = root / phase / f"rank_{rank:02d}"
    destination.mkdir(parents=True, exist_ok=False)
    tensors = _optimizer_partition_tensors(engine)
    records: dict[str, Any] = {}
    for name, tensor in tensors.items():
        path = destination / f"{name}.f32"
        temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
        value = tensor.detach().cpu().contiguous()
        value.numpy().tofile(temporary)
        os.replace(temporary, path)
        records[name] = {
            "path": str(path),
            "dtype": "float32",
            "shape": list(value.shape),
            "numel": int(value.numel()),
            "bytes": int(path.stat().st_size),
            "sha256": _sha256_file(path),
        }
    manifest = {
        "schema_version": "distillschedule.critic_pool_rank_state.v1",
        "phase": phase,
        "rank": rank,
        "optimizer_step": _optimizer_step(engine.optimizer),
        "scheduler_step": _scheduler_step(engine),
        "records": records,
    }
    _atomic_json(destination / "manifest.json", manifest)
    return manifest


def _export_pre_step_gradient_partition(
    *,
    engine: Any,
    root: Path,
    wave: int,
    rank: int,
) -> dict[str, Any]:
    """Export the ZeRO-1 post-reduce, pre-optimizer FP32 gradient partition."""

    masters = getattr(engine.optimizer, "single_partition_of_fp32_groups", None)
    if not isinstance(masters, (tuple, list)) or len(masters) != 1:
        raise AssertionError("expected one ZeRO-1 FP32 master partition")
    gradient = masters[0].grad
    if not isinstance(gradient, torch.Tensor):
        raise AssertionError("ZeRO-1 pre-step gradient partition is missing")
    if gradient.dtype != torch.float32 or int(gradient.numel()) != int(
        masters[0].numel()
    ):
        raise AssertionError(
            "ZeRO-1 pre-step gradient partition dtype/size changed"
        )

    destination = (
        root / "pre_step_gradient" / f"wave_{wave:02d}" / f"rank_{rank:02d}"
    )
    destination.mkdir(parents=True, exist_ok=False)
    path = destination / "gradient.f32"
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    value = gradient.detach().cpu().contiguous()
    value.numpy().tofile(temporary)
    os.replace(temporary, path)
    manifest = {
        "schema_version": "distillschedule.critic_pool_gradient_partition.v1",
        "semantic_point": "post_dp_reduce_pre_unscale_clip_pre_optimizer",
        "wave": wave,
        "rank": rank,
        "optimizer_step_before": _optimizer_step(engine.optimizer),
        "scheduler_step_before": _scheduler_step(engine),
        "record": {
            "path": str(path),
            "dtype": "float32",
            "shape": list(value.shape),
            "numel": int(value.numel()),
            "bytes": int(path.stat().st_size),
            "sha256": _sha256_file(path),
        },
    }
    _atomic_json(destination / "manifest.json", manifest)
    return manifest


def _optimizer_partition_inventory(engine: Any) -> dict[str, Any]:
    """Hash rank-local master/moment partitions for same-layout comparison."""

    optimizer = _unwrap_optimizer(engine.optimizer)
    state_records: list[dict[str, Any]] = []
    for index, value in enumerate(optimizer.state.values()):
        if not isinstance(value, dict):
            continue
        record: dict[str, Any] = {"state_index": index}
        for key in ("exp_avg", "exp_avg_sq"):
            tensor = value.get(key)
            if isinstance(tensor, torch.Tensor):
                record[key] = {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "sha256": _tensor_digest(tensor),
                }
        step = value.get("step")
        record["step"] = int(step.item()) if isinstance(step, torch.Tensor) else int(step)
        state_records.append(record)

    masters: list[dict[str, Any]] = []
    seen: set[int] = set()
    for attribute in (
        "single_partition_of_fp32_groups",
        "fp32_groups_flat",
        "fp32_groups",
    ):
        values = getattr(engine.optimizer, attribute, None)
        if values is None:
            continue
        for index, value in enumerate(values):
            tensors = value if isinstance(value, (tuple, list)) else (value,)
            for inner, tensor in enumerate(tensors):
                if not isinstance(tensor, torch.Tensor) or id(tensor) in seen:
                    continue
                seen.add(id(tensor))
                masters.append(
                    {
                        "attribute": attribute,
                        "index": index,
                        "inner": inner,
                        "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype),
                        "sha256": _tensor_digest(tensor),
                    }
                )
    return {"state": state_records, "master_partitions": masters}


def _student_config(student_dp_world_size: int) -> dict[str, Any]:
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
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1,
        "train_batch_size": student_dp_world_size,
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


def _assert_student_contract(engine: Any, *, group: Any) -> dict[str, int]:
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
        "dp_world_size": 12,
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1,
        "train_batch_size": 12,
    }
    if resolved != expected:
        raise AssertionError(f"Candidate-B batch contract changed: {resolved}")
    if int(engine.zero_optimization_stage()) != 1:
        raise AssertionError("Candidate B requires ZeRO-1")
    if bool(getattr(engine.optimizer, "partition_gradients", False)):
        raise AssertionError("Candidate B accidentally enabled ZeRO-2")
    if getattr(engine.optimizer, "dp_process_group", None) is not group:
        raise AssertionError("Candidate-B optimizer escaped the DP12 group")
    if float(getattr(engine.optimizer, "clip_grad", -1.0)) != 1.0:
        raise AssertionError("official gradient clipping changed")
    return resolved


def _step10a_role(rank: int) -> tuple[str, str]:
    if rank in STEP10A_G_RANKS:
        return "generator", "generator"
    if rank in STEP10A_STUDENT_RANKS:
        return "critic", "student"
    if rank in STEP10A_TEACHER_RANKS:
        return "critic", "teacher"
    raise ValueError(f"rank {rank} is outside world16")


def _reactivate_critic_adapter(model: torch.nn.Module) -> dict[str, int]:
    if not bool(getattr(model, "_distillschedule_has_adapter", False)):
        raise AssertionError("physical Score is not Critic-capable")
    model.enable_adapters()
    trainables = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainables:
        raise AssertionError("Critic adapter reactivation produced no trainables")
    return {
        "trainable_tensor_count": len(trainables),
        "trainable_numel": sum(parameter.numel() for parameter in trainables),
    }


@torch.no_grad()
def _cast_resident_generator_trainables_to_bf16(
    model: torch.nn.Module,
) -> dict[str, Any]:
    """Match the resident Generator representation used after DeepSpeed.

    LoRA parameters are created in FP32 by PEFT.  The prior validated resident
    path reached BF16 through DeepSpeed engine construction, while the new
    cold-preconstructed transition target intentionally bypasses DeepSpeed.
    Cast only trainable Parameters; frozen FP32 normalization state remains in
    the checkpoint's mixed-precision representation.
    """

    trainable_count = 0
    trainable_numel = 0
    before: set[str] = set()
    after: set[str] = set()
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        if parameter.device.type != "cpu":
            raise AssertionError("resident Generator trainable cast must occur on CPU")
        trainable_count += 1
        trainable_numel += parameter.numel()
        before.add(str(parameter.dtype))
        parameter.data = parameter.detach().to(dtype=torch.bfloat16).contiguous()
        after.add(str(parameter.dtype))
    if trainable_count == 0 or after != {"torch.bfloat16"}:
        raise AssertionError(
            "resident Generator trainables did not become BF16: "
            f"count={trainable_count}, after={sorted(after)}"
        )
    return {
        "policy": "TRAINABLE_LORA_ONLY_BF16",
        "trainable_tensor_count": trainable_count,
        "trainable_numel": trainable_numel,
        "dtype_before": sorted(before),
        "dtype_after": sorted(after),
        "frozen_tensors_untouched": True,
    }


def _materialize_transition_target(
    *,
    args: argparse.Namespace,
    model: torch.nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    if args.transition_h2d_policy == "module_to":
        started = time.perf_counter()
        model.to(device)
        torch.cuda.synchronize(device)
        return {
            "policy": "PYTORCH_MODULE_TO_PAGEABLE_SOURCE",
            "wall_seconds": time.perf_counter() - started,
            "ready_on_return": True,
        }
    if args.transition_h2d_policy == "bounded_pinned":
        return materialize_module_from_cpu_with_bounded_pinned_staging(
            model,
            device=device,
            chunk_mib=args.transition_h2d_chunk_mib,
            slot_count=args.transition_h2d_slot_count,
        )
    if args.transition_h2d_policy == "persistent_pinned":
        return materialize_module_from_persistent_pinned_cpu(
            model,
            device=device,
            cleanup_policy=args.transition_pinned_cleanup_policy,
        )
    raise AssertionError(f"unknown transition H2D policy {args.transition_h2d_policy}")


def _step10a_generator_config(
    *,
    param_persistence_threshold: int = 0,
    module_granularity_threshold: int = 0,
    prefetch_bucket_elements: int = 16_777_216,
    max_live_parameters: int = 268_435_456,
    max_reuse_distance: int = 268_435_456,
    overlap_comm: bool = False,
) -> dict[str, Any]:
    zero = build_zero_optimization_config(
        stage=3,
        reduce_bucket_mib=64,
        allgather_bucket_mib=64,
        overlap_comm=overlap_comm,
        contiguous_gradients=False,
        stage3_param_persistence_threshold=param_persistence_threshold,
        stage3_module_granularity_threshold=module_granularity_threshold,
        stage3_prefetch_bucket_elements=prefetch_bucket_elements,
        stage3_max_live_parameters=max_live_parameters,
        stage3_max_reuse_distance=max_reuse_distance,
        optimizer_offload_device="cpu",
        optimizer_offload_pin_memory=False,
    )
    config = {
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 2,
        "train_batch_size": 16,
        "fp16": {"enabled": False},
        "bf16": {"enabled": True},
        "communication_data_type": "fp32",
        "gradient_clipping": 1.0,
        "zero_allow_untested_optimizer": True,
        "zero_optimization": zero,
    }
    config.update(build_client_optimizer_offload_overrides("cpu"))
    return config


def _checkpoint_parent(root: Path, component: str) -> Path:
    suffixes = (
        (("checkpoint-10", component) if component == "critic" else ("checkpoint-10",)),
        ((component,) if component == "critic" else ()),
    )
    for suffix in suffixes:
        candidate = root.joinpath(*suffix)
        if (candidate / "pytorch_model").is_dir():
            return candidate
    raise FileNotFoundError(f"native world8 {component} checkpoint under {root}")


def _construct_step10a_source(
    *,
    args: argparse.Namespace,
    rank: int,
    device: torch.device,
    logical: LogicalCheckpointPackage,
    control_group: Any,
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    # Mixed-BF16 precast loading is the validated construction substrate.
    # Every local rank constructs concurrently; no historical admission wave
    # remains in this active executor.
    physical_role, execution_role = _step10a_role(rank)
    if physical_role == "generator":
        model, metadata = build_helios_pool_model(
            role="generator",
            model_root=args.generator_root,
            subfolder="transformer_ode",
            precast_model_root=args.precast_generator_root,
        )
        restore = load_logical_master_into_trainables(
            model,
            logical,
            component="generator",
            verify_payload_sha256=(rank == STEP10A_G_RANKS[0]),
        )
        restore["restore_source"] = "logical_G2_before_zero3_partitioning"
    else:
        model, metadata = build_helios_pool_model(
            role="critic",
            model_root=args.score_root,
            subfolder="transformer",
            precast_model_root=args.precast_score_root,
        )
        restore = load_logical_master_into_trainables(
            model,
            logical,
            component="critic",
            verify_payload_sha256=(rank == STEP10A_STUDENT_RANKS[0]),
        )
        metadata["execution_role"] = execution_role
        metadata["physical_score_reused_for_critic"] = True
        model.to(device)
        torch.cuda.synchronize(device)
    _control_barrier(control_group)
    return model, metadata, restore


def _initialize_step10a_generator(
    *,
    args: argparse.Namespace,
    model: torch.nn.Module,
    logical: LogicalCheckpointPackage,
    rank: int,
    group: Any,
) -> tuple[Any, dict[str, Any]]:
    import deepspeed
    from deepspeed.utils import get_z3_leaf_modules, set_z3_leaf_modules

    configure_block_gradient_checkpointing(
        model, zero_stage=3, policy="zero3_metadata_tolerant"
    )
    configured_leaf_layers: list[torch.nn.Module] = []
    if args.step10a_block_leaf_fetch:
        configured_leaf_layers = set_z3_leaf_modules(
            model, ["HeliosTransformerBlock"]
        )
        if len(configured_leaf_layers) != 40:
            raise AssertionError(
                "Step10-A explicit block leaf count changed: "
                f"{len(configured_leaf_layers)}"
            )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=2e-6,
        betas=(0.0, 0.999),
        eps=1e-8,
        weight_decay=1e-3,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    if group is not dist.group.WORLD:
        raise AssertionError("Step10-A ZeRO-3 must own the default world8")
    config = _step10a_generator_config(
        param_persistence_threshold=args.step10a_param_persistence_threshold,
        module_granularity_threshold=args.step10a_module_granularity_threshold,
        prefetch_bucket_elements=args.step10a_prefetch_bucket_elements,
        max_live_parameters=args.step10a_max_live_parameters,
        max_reuse_distance=args.step10a_max_reuse_distance,
        overlap_comm=args.step10a_overlap_comm,
    )
    engine, _, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        config=config,
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
    expected = {
        "dp_world_size": 8,
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 2,
        "train_batch_size": 16,
    }
    if resolved != expected:
        raise AssertionError(f"Step10-A DeepSpeed contract changed: {resolved}")
    if int(engine.zero_optimization_stage()) != 3:
        raise AssertionError("Step10-A requires the validated ZeRO-3 backend")
    expected_dp_ranks = tuple(range(PHASE_A_WORLD_SIZE))
    engine_dp_ranks = _require_process_group_ranks(
        engine.data_parallel_group,
        expected=expected_dp_ranks,
        label="Step10-A engine data-parallel group",
    )
    optimizer_dp_ranks = _require_process_group_ranks(
        engine.optimizer.dp_process_group,
        expected=expected_dp_ranks,
        label="Step10-A optimizer data-parallel group",
    )
    logical_restore = restore_zero3_optimizer_partition_from_logical(
        engine,
        logical,
        component="generator",
        partition_rank=rank,
        dp_world_size=len(STEP10A_G_RANKS),
    )
    leaf_layers = list(get_z3_leaf_modules(engine.module))
    module_names = {
        id(module): name or "<root>"
        for name, module in engine.module.named_modules()
    }
    leaf_inventory = [
        {
            "name": module_names.get(id(module), "<unknown>"),
            "module_type": module.__class__.__name__,
            "logical_parameter_numel": sum(
                int(getattr(parameter, "ds_numel", parameter.numel()))
                for parameter in module.parameters()
            ),
        }
        for module in leaf_layers
    ]
    if args.step10a_module_granularity_threshold == 8_000_000:
        block_leaves = [
            record
            for record in leaf_inventory
            if record["module_type"] == "HeliosTransformerBlock"
        ]
        if len(leaf_inventory) != 47 or len(block_leaves) != 40:
            raise AssertionError(
                "Step10-A module-leaf inventory changed: "
                f"all={len(leaf_inventory)}, blocks={len(block_leaves)}"
            )
        if max(record["logical_parameter_numel"] for record in leaf_inventory) != 382_065_152:
            raise AssertionError("Step10-A maximum leaf size changed")
    if args.step10a_block_leaf_fetch:
        if len(leaf_inventory) != 40 or any(
            record["module_type"] != "HeliosTransformerBlock"
            for record in leaf_inventory
        ):
            raise AssertionError(
                f"Step10-A explicit leaf inventory changed: {leaf_inventory}"
            )
        if max(record["logical_parameter_numel"] for record in leaf_inventory) != 382_065_152:
            raise AssertionError("Step10-A explicit maximum leaf size changed")
    versions = (
        int(engine.global_steps),
        _optimizer_step(engine.optimizer),
        _scheduler_step(engine),
    )
    if versions != (2, 2, 2):
        raise AssertionError(f"Step10-A did not restore G2: {versions}")
    return engine, {
        "config": config,
        "resolved": resolved,
        "logical_restore": logical_restore,
        "process_world_policy": "NODE0_DEFAULT_WORLD8_NO_SUBGROUP",
        "engine_dp_ranks": list(engine_dp_ranks),
        "optimizer_dp_ranks": list(optimizer_dp_ranks),
        "process_group_validation": "rank_membership_not_python_identity",
        "checkpoint_policy": "zero3_metadata_tolerant_block_local_non_reentrant",
        "module_leaf_inventory": leaf_inventory,
        "parameter_version": 2,
    }


def _preconstruct_switch_target(
    *,
    args: argparse.Namespace,
    rank: int,
    local_rank: int,
    logical: LogicalCheckpointPackage,
    control_group: Any,
) -> tuple[torch.nn.Module | None, dict[str, Any] | None, dict[str, Any] | None]:
    """Build only switching ranks' future model on CPU before Full5 timing."""

    target_model = target_metadata = target_restore = None
    if rank in PRECONSTRUCT_TARGET_RANKS:
        # A future resident target is an ordinary full CPU module, not a
        # child of the currently active Step10-A ZeRO-3 context. Mixed-BF16
        # precast loading removes the historical FP32 construction transient,
        # so every admitted local rank constructs concurrently.
        with suspend_transformers_zero3_init():
            if rank in SWITCH_TO_CRITIC_RANKS:
                target_model, target_metadata = build_helios_pool_model(
                    role="critic",
                    model_root=args.score_root,
                    subfolder="transformer",
                    precast_model_root=args.precast_score_root,
                )
                target_restore = load_logical_master_into_trainables(
                    target_model,
                    logical,
                    component="critic",
                    verify_payload_sha256=(rank == SWITCH_TO_CRITIC_RANKS[0]),
                )
            else:
                target_model, target_metadata = build_helios_pool_model(
                    role="generator",
                    model_root=args.generator_root,
                    subfolder="transformer_ode",
                    precast_model_root=args.precast_generator_root,
                )
                target_metadata["resident_trainable_cast"] = (
                    _cast_resident_generator_trainables_to_bf16(target_model)
                )
                target_restore = {
                    "component": "generator",
                    "parameter_version": "G3 supplied after Step10-A",
                    "optimizer_policy": "BACKING_ONLY",
                }
    _control_barrier(control_group)
    if rank in PRECONSTRUCT_TARGET_RANKS:
        if target_model is None or target_metadata is None or target_restore is None:
            raise AssertionError("switching rank lacks a preconstructed target model")
        if any(parameter.is_cuda for parameter in target_model.parameters()):
            raise AssertionError("future target model entered GPU before timed transition")
    return target_model, target_metadata, target_restore


def _atomic_torch(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _wait_for_path(path: Path, timeout_seconds: float) -> float:
    started = time.monotonic()
    while not path.is_file():
        if time.monotonic() - started >= timeout_seconds:
            raise TimeoutError(path)
        time.sleep(0.05)
    return time.monotonic() - started


def _wait_for_directory(path: Path, timeout_seconds: float) -> float:
    """Wait for a directory without weakening file-only payload fences."""

    started = time.monotonic()
    while not path.is_dir():
        if time.monotonic() - started >= timeout_seconds:
            raise TimeoutError(path)
        time.sleep(0.05)
    return time.monotonic() - started


def _step10a_input_path(root: Path, source_id: int) -> Path:
    return root / "step10a" / f"input_{source_id:02d}.pt"


def _step10a_fake_path(root: Path, source_id: int) -> Path:
    return root / "step10a" / f"fake_{source_id:02d}.pt"


def _step10a_gradient_path(root: Path, source_id: int) -> Path:
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
    mismatch = {
        key: (value.get(key), expected_value)
        for key, expected_value in expected.items()
        if value.get(key) != expected_value
    }
    if mismatch:
        raise ValueError(f"Step10-A input mismatch: {mismatch}")
    x0 = value.get("x0")
    if not isinstance(x0, torch.Tensor) or list(x0.shape) != [1, 16, 9, 48, 80]:
        raise ValueError("Step10-A x0 shape changed")
    if x0.dtype != torch.bfloat16 or tensor_sha256(x0) != value.get("x0_sha256"):
        raise ValueError("Step10-A x0 dtype/checksum changed")
    return x0


def _run_step10a_generator(
    *,
    args: argparse.Namespace,
    rank: int,
    device: torch.device,
    engine: Any,
    trace: RouteBFull5Trace,
    exchange: Path,
    output: Path,
    result: dict[str, Any],
) -> None:
    """Publish both selected graphs before consuming either detached gradient.

    This is the pre-regression Step10-A schedule: rollout wave 1 legally
    overlaps Student/Teacher work for wave 0.  GAS2 still produces exactly one
    G2->G3 optimizer commit after the two backwards.
    """

    records: list[dict[str, Any]] = []
    states: list[dict[str, Any]] = []
    _persist_diagnostic_stage(
        args=args,
        result=result,
        output=output,
        device=device,
        stage="step10a_generator_checksum_begin",
    )
    before = TIMING_OMITTED if args.timing_mode else trainable_checksum(engine.module)
    _persist_diagnostic_stage(
        args=args,
        result=result,
        output=output,
        device=device,
        stage="step10a_generator_checksum_complete",
    )
    global_before = int(engine.global_steps)
    for production_round in range(2):
        collective_plan = trace.step10a_collective_plan(
            production_round=production_round
        )
        source_id = production_round * 8 + rank
        _persist_diagnostic_stage(
            args=args,
            result=result,
            output=output,
            device=device,
            stage=f"step10a_wave{production_round}_trace_load_begin",
        )
        entry = trace.load_step10a_entry(source_id=source_id, device=device)
        _persist_diagnostic_stage(
            args=args,
            result=result,
            output=output,
            device=device,
            stage=f"step10a_wave{production_round}_rollout_begin",
        )
        started = time.perf_counter()
        with _nvtx_range(
            f"step10a/rank={rank}/wave={production_round}/phase=rollout"
        ):
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
        _atomic_torch(_step10a_input_path(exchange, source_id), payload)
        record = {
            "source_id": source_id,
            "production_round": production_round,
            "trace_identity_sha256": entry.identity_sha256,
            "x0_sha256": payload["x0_sha256"],
            "rollout_seconds": time.perf_counter() - started,
            "rollout": details,
            "schedule_event": "selected_graph_published_before_gradient_wait",
            "collective_plan": {
                "source_ids": list(collective_plan.source_ids),
                "logical_call_counts": list(collective_plan.logical_call_counts),
                "synchronized_stage_exits": list(
                    collective_plan.synchronized_stage_exits
                ),
                "physical_stage_call_counts": list(
                    collective_plan.physical_stage_call_counts
                ),
                "physical_call_count": collective_plan.physical_call_count,
                "dummy_call_counts": list(collective_plan.dummy_call_counts),
            },
        }
        records.append(record)
        states.append(
            {
                "source_id": source_id,
                "trace_identity_sha256": entry.identity_sha256,
                "x0_sha256": payload["x0_sha256"],
                "x0": x0,
                "gradient_mask": gradient_mask,
                "gradient_path": _step10a_gradient_path(exchange, source_id),
                "record": record,
            }
        )
        del entry, x0_cpu, payload

    boundaries = 0
    for state in states:
        waited = _wait_for_path(state["gradient_path"], args.timeout_seconds)
        gradient_payload = torch.load(
            state["gradient_path"], map_location="cpu", weights_only=False
        )
        expected = {
            "schema_version": "distillschedule.routeb_step10a_gradient.v1",
            "logical_step": 10,
            "source_id": state["source_id"],
            "generator_version": 2,
            "critic_version": 10,
            "trace_identity_sha256": state["trace_identity_sha256"],
            "x0_sha256": state["x0_sha256"],
        }
        mismatch = {
            key: (gradient_payload.get(key), expected_value)
            for key, expected_value in expected.items()
            if gradient_payload.get(key) != expected_value
        }
        if mismatch:
            raise ValueError(f"Step10-A gradient identity mismatch: {mismatch}")
        gradient_cpu = gradient_payload.get("gradient")
        if (
            not isinstance(gradient_cpu, torch.Tensor)
            or tensor_sha256(gradient_cpu) != gradient_payload.get("gradient_sha256")
        ):
            raise ValueError("Step10-A detached gradient checksum changed")
        gradient = gradient_cpu.to(device=device, dtype=torch.bfloat16)
        loss = detached_dmd_surrogate_loss(
            state["x0"], gradient, state["gradient_mask"]
        )
        started = time.perf_counter()
        with _nvtx_range(
            f"step10a/rank={rank}/wave={state['record']['production_round']}/phase=backward"
        ):
            engine.backward(loss)
        split_diagnostic = bool(args.stop_after_step10a and args.timing_mode)
        backward_seconds = None
        if split_diagnostic:
            # This synchronization is diagnostic-only.  It is forbidden in the
            # continuous Full5 path because it would change the measured
            # backward/optimizer overlap boundary.
            torch.cuda.synchronize(device)
            backward_seconds = time.perf_counter() - started
        boundary = bool(engine.is_gradient_accumulation_boundary())
        boundaries += int(boundary)
        step_started = time.perf_counter()
        with _nvtx_range(
            f"step10a/rank={rank}/wave={state['record']['production_round']}/phase=engine_step"
        ):
            engine.step()
        torch.cuda.synchronize(device)
        completed = time.perf_counter()
        state["record"].update(
            {
                "gradient_wait_seconds": waited,
                "gradient_sha256": gradient_payload["gradient_sha256"],
                "surrogate_loss": float(loss.detach().cpu()),
                "backward_step_seconds": completed - started,
                "backward_seconds": backward_seconds,
                "engine_step_seconds": (
                    completed - step_started if split_diagnostic else None
                ),
                "diagnostic_split_synchronize": split_diagnostic,
                "gradient_accumulation_boundary": boundary,
            }
        )
        state["x0"] = None
        state["gradient_mask"] = None
        del gradient_payload, gradient_cpu, gradient, loss
    if int(engine.global_steps) != global_before + 1 or boundaries != 1:
        raise AssertionError("Step10-A must perform exactly one G2->G3 update")
    after = TIMING_OMITTED if args.timing_mode else trainable_checksum(engine.module)
    if not args.timing_mode and before == after:
        raise AssertionError("Step10-A did not change Generator trainables")
    result["step10a"] = {
        "role": "generator",
        "schedule": "two_selected_graphs_first_then_two_backwards",
        "generator_before": 2,
        "generator_after": 3,
        "global_steps_before": global_before,
        "global_steps_after": int(engine.global_steps),
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


def _run_step10a_score(
    *,
    args: argparse.Namespace,
    rank: int,
    device: torch.device,
    model: torch.nn.Module,
    trace: RouteBFull5Trace,
    exchange: Path,
    negative_embedding: torch.Tensor | None,
    result: dict[str, Any],
) -> None:
    student = rank in STEP10A_STUDENT_RANKS
    lane = rank - (8 if student else 12)
    records: list[dict[str, Any]] = []
    for source_id in range(lane, 16, 4):
        entry = trace.load_step10a_entry(source_id=source_id, device=device)
        input_wait = _wait_for_path(
            _step10a_input_path(exchange, source_id), args.timeout_seconds
        )
        payload = torch.load(
            _step10a_input_path(exchange, source_id),
            map_location="cpu",
            weights_only=False,
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
            _atomic_torch(_step10a_fake_path(exchange, source_id), fake_payload)
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
                raise AssertionError("Teacher rank lacks the official negative prompt")
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
            fake_wait = _wait_for_path(
                _step10a_fake_path(exchange, source_id), args.timeout_seconds
            )
            fake_payload = torch.load(
                _step10a_fake_path(exchange, source_id),
                map_location="cpu",
                weights_only=False,
            )
            expected = {
                "logical_step": 10,
                "source_id": source_id,
                "generator_version": 2,
                "critic_version": 10,
                "trace_identity_sha256": entry.identity_sha256,
                "x0_sha256": payload["x0_sha256"],
            }
            mismatch = {
                key: (fake_payload.get(key), expected_value)
                for key, expected_value in expected.items()
                if fake_payload.get(key) != expected_value
            }
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
            _atomic_torch(
                _step10a_gradient_path(exchange, source_id), gradient_payload
            )
            record = {
                "source_id": source_id,
                "x0_sha256": payload["x0_sha256"],
                "fake_sha256": fake_payload["fake_sha256"],
                "gradient_sha256": gradient_payload["gradient_sha256"],
                "input_wait_seconds": input_wait,
                "fake_wait_seconds": fake_wait,
                "compute_seconds": time.perf_counter() - started,
            }
            del real_cond, real_uncond, fake_payload, fake_cpu, fake
            del gradient, gradient_cpu, gradient_payload, auxiliary
        records.append(record)
        del entry, payload, x0_cpu, x0, batch
        torch.cuda.empty_cache()
    result["step10a"] = {
        "role": "student" if student else "teacher",
        "parameter_version": 10 if student else "official_frozen_base",
        "samples": records,
    }


@torch.no_grad()
def _materialize_zero3_g3_into_target(
    *,
    engine: Any,
    target_model: torch.nn.Module | None,
    logical: LogicalCheckpointPackage,
    rank: int,
    timing_mode: bool,
) -> dict[str, Any]:
    """Gather updated G3 trainables before destroying the Step10-A engine.

    All eight ZeRO-3 ranks must execute every gather in identical order.  Only
    rank 0 writes the gathered FP32 masters into its preconstructed resident
    Generator, rounded once to the target BF16 execution representation.  The
    other target Generator ranks receive that exact representation through the
    post-transition DP4 broadcast.
    """

    expected_names = [
        str(name) for name in logical.component("generator")["parameter_names"]
    ]
    source = {
        name: parameter
        for name, parameter in engine.module.named_parameters()
        if parameter.requires_grad
    }
    if list(source) != expected_names:
        raise AssertionError("ZeRO-3 G3 source trainable order changed")
    target = None
    if rank == 0:
        if target_model is None:
            raise AssertionError("rank0 lacks the preconstructed G3 target model")
        target = {
            name: parameter
            for name, parameter in target_model.named_parameters()
            if parameter.requires_grad
        }
        if list(target) != expected_names:
            raise AssertionError("resident G3 target trainable order changed")

    started = time.perf_counter()
    copied = 0
    for name in expected_names:
        full_master = engine.optimizer.get_full_hp_param(source[name])
        if tuple(full_master.shape) != tuple(
            int(value)
            for value in logical.component("generator")["records"][name]["shape"]
        ):
            raise AssertionError(f"gathered G3 shape changed for {name}")
        if rank == 0:
            destination = target[name]
            if destination.dtype != torch.bfloat16:
                destination.data = destination.data.to(dtype=torch.bfloat16)
            destination.copy_(
                full_master.detach().to(device="cpu", dtype=torch.bfloat16)
            )
            copied += int(destination.numel())
        del full_master
    torch.cuda.synchronize()
    return {
        "source_backend": "ZeRO-3_DP8_CPU_optimizer",
        "source_parameter_version": 3,
        "gather_participant": True,
        "writer_global_rank": 0,
        "writer": rank == 0,
        "copied_logical_numel": copied,
        "target_dtype": "torch.bfloat16",
        "target_trainable_sha256": (
            TIMING_OMITTED
            if timing_mode or rank != 0
            else trainable_sha256(target_model)
        ),
        "seconds": time.perf_counter() - started,
    }


def _destroy_obsolete_zero3_engine(
    engine: Any, *, device: torch.device
) -> dict[str, Any]:
    """Detach the no-longer-legal DP8 ZeRO-3 representation.

    The caller still owns the final ``step10a_engine`` alias while this helper
    executes.  Consequently, checking CUDA storage here is necessarily too
    early: the wrapper (and any references reachable only through it) cannot be
    collected until this function returns and the caller clears that alias.
    Return the pre-release snapshot and let the caller validate the release
    only after it has dropped its owner.
    """

    before = _snapshot(device, "before_zero3_source_destroy")
    module = engine.module
    optimizer = engine.optimizer
    # DeepSpeed 0.18 installs Engine forward hooks whose closures retain the
    # Engine, and ``_set_client_model`` stores the client module twice: once in
    # ``_modules`` and once directly in ``__dict__``.  ``engine.module = None``
    # clears only the registered-module slot, leaving the direct owner alive.
    # Remove the hook closures and both owners explicitly before dropping the
    # wrapper; otherwise every ZeRO-3 parameter shard survives GC.
    for hook_name in ("module_forward_pre_hook", "module_forward_post_hook"):
        hook = getattr(engine, hook_name, None)
        if hook is not None:
            hook.remove()
            setattr(engine, hook_name, None)
    engine.destroy()
    drained_partition_bytes = 0
    drained_partition_tensors = 0
    for parameter in module.parameters():
        if hasattr(parameter, "_z3_optimizer"):
            delattr(parameter, "_z3_optimizer")
        parameter.grad = None
        # DeepSpeed's ZeRO-3 parameter callables and Init metadata form
        # long-lived Python cycles even after Engine/optimizer destruction.
        # The obsolete low-precision shard is the only large CUDA owner still
        # required by those objects.  Replace it explicitly after G3 has been
        # gathered; the source representation has no legal future use.
        for shard_name in ("ds_tensor", "ds_secondary_tensor"):
            shard = getattr(parameter, shard_name, None)
            if not isinstance(shard, torch.Tensor) or not shard.is_cuda:
                continue
            drained_partition_bytes += int(shard.numel() * shard.element_size())
            drained_partition_tensors += 1
            setattr(
                parameter,
                shard_name,
                torch.empty(0, dtype=shard.dtype, device="cpu"),
            )
    # Break the large optimizer<->module cycle.  The caller drops its separate
    # module/engine references immediately after this helper returns.
    if hasattr(optimizer, "module"):
        optimizer.module = None
    parameter_offload = getattr(optimizer, "parameter_offload", None)
    if parameter_offload is not None and hasattr(parameter_offload, "module"):
        parameter_offload.module = None
    compression = getattr(engine, "compression_scheduler", None)
    if compression is not None and hasattr(compression, "model"):
        compression.model = None
    modules = engine.__dict__.get("_modules")
    if isinstance(modules, dict):
        modules["module"] = None
    engine.__dict__["module"] = None
    engine.optimizer = None
    engine.lr_scheduler = None
    del module, optimizer
    return {
        "policy": "DESTROY_OBSOLETE_DP8_ZERO3_BEFORE_TARGET_MATERIALIZATION",
        "before": before,
        "drained_partition_tensors": drained_partition_tensors,
        "drained_partition_bytes": drained_partition_bytes,
    }


def _verify_obsolete_zero3_release(
    release: dict[str, Any], *, device: torch.device
) -> dict[str, Any]:
    """Validate release after the caller has removed its last engine alias."""

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    after = _snapshot(device, "after_zero3_source_destroy")
    before = release["before"]
    if after["cuda"]["allocated_gib"] >= before["cuda"]["allocated_gib"]:
        raise AssertionError("obsolete ZeRO-3 source storage did not decrease")
    release.update(
        {
            "after": after,
            "released_allocated_gib": (
                before["cuda"]["allocated_gib"]
                - after["cuda"]["allocated_gib"]
            ),
        }
    )
    return release


@torch.no_grad()
def _broadcast_generator_g3(
    *,
    model: torch.nn.Module,
    logical: LogicalCheckpointPackage,
    rank: int,
    group: Any,
    timing_mode: bool,
    bucket_mib: int = 0,
) -> dict[str, Any]:
    expected_names = [
        str(name) for name in logical.component("generator")["parameter_names"]
    ]
    trainables = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if list(trainables) != expected_names:
        raise AssertionError("target Generator trainable order/inventory changed")
    if bucket_mib < 0:
        raise ValueError("G3 broadcast bucket must be non-negative")
    for parameter in trainables.values():
        if parameter.dtype != torch.bfloat16:
            parameter.data = parameter.data.to(dtype=torch.bfloat16)
        if not parameter.data.is_contiguous():
            raise AssertionError("target Generator trainable storage is not contiguous")

    ordered = list(trainables.items())
    if bucket_mib == 0:
        max_bucket_bytes = max(
            parameter.numel() * parameter.element_size()
            for _name, parameter in ordered
        )
        buckets = [[item] for item in ordered]
        policy = "LEGACY_ONE_TENSOR_PER_BROADCAST"
    else:
        max_bucket_bytes = bucket_mib * 2**20
        buckets = []
        current: list[tuple[str, torch.nn.Parameter]] = []
        current_bytes = 0
        for name, parameter in ordered:
            parameter_bytes = parameter.numel() * parameter.element_size()
            if parameter_bytes > max_bucket_bytes:
                raise AssertionError(
                    f"Generator trainable {name} exceeds the bounded G3 bucket"
                )
            if current and current_bytes + parameter_bytes > max_bucket_bytes:
                buckets.append(current)
                current = []
                current_bytes = 0
            current.append((name, parameter))
            current_bytes += parameter_bytes
        if current:
            buckets.append(current)
        policy = "FIXED_ORDER_BOUNDED_BUCKET_BROADCAST"
    if not buckets:
        raise AssertionError("target Generator has no trainables to broadcast")

    before = TIMING_OMITTED if timing_mode else trainable_sha256(model)
    started = time.perf_counter()
    observed_names: list[str] = []
    bucket_bytes: list[int] = []
    for bucket in buckets:
        bucket_numel = sum(parameter.numel() for _name, parameter in bucket)
        flat = torch.empty(
            bucket_numel,
            dtype=torch.bfloat16,
            device=next(iter(trainables.values())).device,
        )
        offset = 0
        if rank == 0:
            for _name, parameter in bucket:
                next_offset = offset + parameter.numel()
                flat[offset:next_offset].copy_(parameter.data.reshape(-1))
                offset = next_offset
        dist.broadcast(flat, src=0, group=group)
        offset = 0
        for name, parameter in bucket:
            next_offset = offset + parameter.numel()
            parameter.data.copy_(flat[offset:next_offset].view_as(parameter.data))
            observed_names.append(name)
            offset = next_offset
        if offset != bucket_numel:
            raise AssertionError("G3 bucket scatter did not consume all elements")
        bucket_bytes.append(flat.numel() * flat.element_size())
        del flat
    torch.cuda.synchronize(parameter.device)
    if observed_names != expected_names:
        raise AssertionError("bucketed G3 broadcast changed trainable order")
    after = TIMING_OMITTED if timing_mode else trainable_sha256(model)
    return {
        "source_global_rank": 0,
        "target_generator_ranks": list(CONTRACT.generator_ranks),
        "tensor_count": len(trainables),
        "logical_numel": sum(parameter.numel() for parameter in trainables.values()),
        "dtype": "torch.bfloat16",
        "policy": policy,
        "bucket_mib": bucket_mib,
        "collective_count": len(buckets),
        "bucket_bytes": bucket_bytes,
        "max_bucket_bytes": max(bucket_bytes),
        "seconds": time.perf_counter() - started,
        "trainable_sha256_before": before,
        "trainable_sha256_after": after,
        "parameter_version": 3,
    }


def _initialize_student(
    *,
    args: argparse.Namespace,
    model: torch.nn.Module,
    metadata: dict[str, Any],
    construction_restore: dict[str, Any],
    package: LogicalCheckpointPackage,
    rank: int,
    pool_rank: int,
    student_group: Any,
    singleton_group: Any,
    device: torch.device,
    result: dict[str, Any],
) -> tuple[Any, _ResidentStudentParameterGuard, Any]:
    import deepspeed

    subphases: dict[str, float] = {}
    preparation_started = time.perf_counter()
    expected = list(package.component("critic")["parameter_names"])
    observed = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if observed != expected:
        raise AssertionError("Critic target order/inventory changed")
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
        data_parallel_group=student_group,
        data_parallel_ranks=CONTRACT.critic_ranks,
        global_rank=rank,
        singleton_model_parallel_group=singleton_group,
    )
    config = _student_config(CONTRACT.critic_dp_world_size)
    subphases["optimizer_scheduler_adapter_prepare_host_wall_seconds"] = (
        time.perf_counter() - preparation_started
    )
    initialize_started = time.perf_counter()
    engine, _, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        config=config,
        mpu=adapter,
        dist_init_required=False,
    )
    subphases["deepspeed_initialize_host_wall_seconds"] = (
        time.perf_counter() - initialize_started
    )
    contract_started = time.perf_counter()
    resolved = _assert_student_contract(engine, group=student_group)
    subphases["contract_assert_host_wall_seconds"] = (
        time.perf_counter() - contract_started
    )
    restore_started = time.perf_counter()
    logical_restore = restore_zero1_optimizer_partition_from_logical(
        engine,
        package,
        component="critic",
        partition_rank=pool_rank,
        dp_world_size=CONTRACT.critic_dp_world_size,
    )
    if (
        _optimizer_step(engine.optimizer),
        _scheduler_step(engine),
        int(engine.global_steps),
    ) != (10, 10, 10):
        raise AssertionError("Student target package did not restore C10")
    subphases["logical_optimizer_restore_host_wall_seconds"] = (
        time.perf_counter() - restore_started
    )

    runtime_started = time.perf_counter()
    event_root = args.output_dir / "events"
    event_root.mkdir(parents=True, exist_ok=True)
    # The S GPUs are dedicated to Score/Critic.  Keep the complete live module
    # resident and use CPU only for the ZeRO-1 optimizer partition.  In
    # particular, do not reinterpret ``streamed_block_count=0`` as "none":
    # the historical streamer API uses zero to mean all frozen blocks.
    residency = _ResidentStudentParameterGuard(engine.module, device)
    activation = install_activation_offload_hooks(
        engine.module,
        logical_component="critic_train",
        event_path=event_root / f"rank_{rank:02d}_activation_offload.jsonl",
        pin_memory=True,
        min_offload_bytes=args.activation_min_bytes,
        detailed_events=False,
        dynamic_excluded_storage_providers=(),
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    subphases["runtime_hook_install_host_wall_seconds"] = (
        time.perf_counter() - runtime_started
    )
    result["student_initialization"] = {
        "construction": metadata,
        "construction_restore": construction_restore,
        "logical_partition_restore": logical_restore,
        "deepspeed_version": deepspeed.__version__,
        "config": config,
        "resolved": resolved,
        "scheduler_restore_source": "logical_world16_step10_v3/logical_state",
        "critic_version": 10,
        "optimizer_step": 10,
        "scheduler_step": 10,
        "subphase_seconds": subphases,
        "timing_semantics": (
            "host_wall_at_existing_boundaries_no_added_cuda_synchronize"
        ),
        "runtime": {
            "parameter_policy": "RESIDENT",
            "frozen_block_streaming": False,
            "host_policy": "NONE_FOR_MODEL_PARAMETERS",
            "activation_offload_min_bytes": args.activation_min_bytes,
            "parameter_residency": residency.record(),
        },
    }
    result["stages"].append(_snapshot(device, "student_c10_dp12_runtime_ready"))
    return engine, residency, activation


TIMING_OMITTED = "omitted_in_low_overhead_timing_mode"


def _header(
    *,
    args: argparse.Namespace,
    topology: RouteBCandidateBTopology,
    rank: int,
    producer_pool_rank: int,
    update: int,
    production_round: int,
    destination: int,
    trace_identity: str,
    payload_digest: bytes,
) -> CriticWaveHeader:
    return CriticWaveHeader(
        run_nonce=args.run_nonce,
        cycle=0,
        logical_step=10 + update,
        wave=update,
        producer_global_rank=rank,
        producer_pool_rank=producer_pool_rank,
        student_global_rank=destination,
        student_pool_rank=topology.critic_pool_rank(destination),
        generator_version=3,
        expected_critic_before=10 + update,
        expected_critic_after=11 + update,
        trace_sha256=bytes.fromhex(trace_identity),
        payload_checksum=payload_digest,
        checksum_kind=_checksum_kind(args),
    )


def _produce(
    *,
    args: argparse.Namespace,
    model: torch.nn.Module,
    trace: RouteBFull5Trace,
    update: int,
    production_round: int,
    producer_pool_rank: int,
    device: torch.device,
) -> tuple[torch.Tensor, Any, dict[str, Any]]:
    loaded = trace.load_critic_entry(
        update=update,
        production_round=production_round,
        producer_pool_rank=producer_pool_rank,
        device=device,
    )
    started = time.perf_counter()
    x0, details = run_generator_trace_rollout(
        model=model,
        loaded=loaded,
        device=device,
        detailed_profile=not args.timing_mode,
        record_output_sha256=args.checksum == "sha256",
    )
    x0 = prepare_transport_payload(x0)
    kind = _checksum_kind(args)
    digest = tensor_identity(x0, kind)
    record = {
        "update": update,
        "logical_step": 10 + update,
        "production_round": production_round,
        "trace_key": loaded.pool_entry.trace_key,
        "trace_identity_sha256": loaded.pool_entry.identity_sha256,
        "source_trace_rank": loaded.pool_entry.source_trace_rank,
        "produce_seconds": time.perf_counter() - started,
        "rollout": details,
        "x0_identity": {"kind": kind.name, "value": digest.hex()},
        "x0_stride": list(x0.stride()),
    }
    return x0, loaded.pool_entry, record


def _validate_received_header(
    *,
    header: CriticWaveHeader,
    topology: RouteBCandidateBTopology,
    rank: int,
    entry: Any,
    update: int,
    producer_rank: int,
    producer_pool_rank: int,
) -> None:
    expected = {
        "wave": (header.wave, update),
        "logical_step": (header.logical_step, 10 + update),
        "producer_global_rank": (header.producer_global_rank, producer_rank),
        "producer_pool_rank": (header.producer_pool_rank, producer_pool_rank),
        "student_global_rank": (header.student_global_rank, rank),
        "student_pool_rank": (
            header.student_pool_rank,
            topology.critic_pool_rank(rank),
        ),
        "generator_version": (header.generator_version, 3),
        "critic_before": (header.expected_critic_before, 10 + update),
        "critic_after": (header.expected_critic_after, 11 + update),
        "trace_identity": (header.trace_sha256.hex(), entry.identity_sha256),
    }
    mismatch = {key: value for key, value in expected.items() if value[0] != value[1]}
    if mismatch:
        raise ValueError(f"Candidate-B received wave identity mismatch: {mismatch}")


def _run_generator_cycle(
    *,
    args: argparse.Namespace,
    topology: RouteBCandidateBTopology,
    rank: int,
    pool_rank: int,
    device: torch.device,
    model: torch.nn.Module,
    initial_sha: str,
    ema: SparseShardEMA | None,
    trace: RouteBFull5Trace,
    transports: dict[int, CriticPoolP2PTransport],
    control_group: Any,
    result: dict[str, Any],
) -> None:
    """Execute Sequential or ACK-free one-batch-ahead production.

    Send completion is the payload credit.  A pair of world-control barriers
    per logical update orders C_t->C_{t+1} and the official EMA recurrence;
    there is no per-route optimizer acknowledgement.
    """

    wave_records: list[dict[str, Any]] = []
    ema_records: list[dict[str, Any]] = []
    produced_payloads = sent_payloads = 0
    produced_batches = committed_batches = 0
    max_produced_minus_committed = 0
    max_ready_not_sent = 0
    cycle_started = time.perf_counter()

    def produce_batch(update: int):
        nonlocal produced_payloads, produced_batches
        nonlocal max_produced_minus_committed, max_ready_not_sent
        batch = []
        for production_round in range(CONTRACT.generator_production_rounds_per_update):
            destination = topology.critic_for(rank, production_round)
            x0, entry, record = _produce(
                args=args,
                model=model,
                trace=trace,
                update=update,
                production_round=production_round,
                producer_pool_rank=pool_rank,
                device=device,
            )
            header = _header(
                args=args,
                topology=topology,
                rank=rank,
                producer_pool_rank=pool_rank,
                update=update,
                production_round=production_round,
                destination=destination,
                trace_identity=entry.identity_sha256,
                payload_digest=bytes.fromhex(record["x0_identity"]["value"]),
            )
            record["destination_global_rank"] = destination
            batch.append((destination, x0, entry, header, record))
            produced_payloads += 1
        produced_batches += 1
        max_ready_not_sent = max(max_ready_not_sent, 1)
        max_produced_minus_committed = max(
            max_produced_minus_committed,
            produced_batches - committed_batches,
        )
        if produced_batches - committed_batches > 2:
            raise AssertionError("ACK-free producer exceeded the double-buffer bound")
        return batch

    def post_batch(batch):
        nonlocal sent_payloads
        current = []
        for destination, x0, entry, header, record in batch:
            started = time.perf_counter()
            pending = transports[destination].send_wave(
                header,
                x0,
                destination_global_rank=destination,
            )
            record["send_post_seconds"] = time.perf_counter() - started
            record["wire"] = {
                "logical_step": header.logical_step,
                "update": header.wave,
                "producer_global_rank": header.producer_global_rank,
                "student_global_rank": header.student_global_rank,
                "generator_version": header.generator_version,
                "critic_before": header.expected_critic_before,
                "critic_after": header.expected_critic_after,
                "trace_identity_sha256": header.trace_sha256.hex(),
                "x0_checksum_kind": header.checksum_kind.name,
                "x0_checksum": header.payload_checksum.hex(),
            }
            current.append((destination, pending, record))
            sent_payloads += 1
            del x0, entry
        return current

    def release_payload_credit(update: int, current):
        started = time.perf_counter()
        for destination, pending, _record in current:
            pending.wait()
            transports[destination].reap_completed()
        elapsed = time.perf_counter() - started
        for _destination, _pending, record in current:
            record["queue_backpressure_seconds"] = elapsed
            record["payload_credit"] = "NCCL_send_completion_no_commit_ack"
        return elapsed

    def order_commit_and_ema(update: int) -> None:
        nonlocal committed_batches
        # Critics enter only after the common DP12 optimizer commit.  Generators
        # enter after preparing the optional future batch.
        _control_barrier(control_group)
        committed_batches += 1
        if rank == 0:
            if ema is None:
                raise AssertionError("continuous Full5 EMA owner is missing")
            before = ema.optimization_step
            started = time.perf_counter()
            decay = ema.step(model.parameters())
            expected = (10 + update, 11 + update, 0.0)
            if (before, ema.optimization_step, decay) != expected:
                raise AssertionError("official early-step EMA recurrence changed")
            comparison = (
                {"matches_exact": TIMING_OMITTED}
                if args.timing_mode
                else ema.compare_to_parameters(model.parameters())
            )
            if not args.timing_mode and comparison["matches_exact"] is not True:
                raise AssertionError("EMA no longer equals the current G3 parameters")
            ema_records.append(
                {
                    "logical_step": 10 + update,
                    "before": before,
                    "after": ema.optimization_step,
                    "decay": decay,
                    "matches_generator_exact": comparison["matches_exact"],
                    "seconds": time.perf_counter() - started,
                    "checksum": TIMING_OMITTED if args.timing_mode else ema.checksum(),
                }
            )
        # No Critic posts update t+1's receive until rank0 completes EMA_t.
        _control_barrier(control_group)

    if args.schedule == "sequential":
        for update in range(CONTRACT.logical_critic_updates):
            current = post_batch(produce_batch(update))
            release_payload_credit(update, current)
            order_commit_and_ema(update)
            wave_records.extend(record for _destination, _pending, record in current)
    else:
        current = post_batch(produce_batch(0))
        for update in range(CONTRACT.logical_critic_updates):
            release_payload_credit(update, current)
            future = (
                produce_batch(update + 1)
                if update + 1 < CONTRACT.logical_critic_updates
                else None
            )
            order_commit_and_ema(update)
            wave_records.extend(record for _destination, _pending, record in current)
            if future is not None:
                current = post_batch(future)

    for transport in transports.values():
        transport.flush()
        transport.assert_flushed()
    execution_seconds = time.perf_counter() - cycle_started
    if args.timing_mode:
        final_sha = TIMING_OMITTED
        gradients = {
            "measurement": TIMING_OMITTED,
            "trainable_grad_present": TIMING_OMITTED,
            "frozen_grad_present": TIMING_OMITTED,
        }
    else:
        final_sha = trainable_sha256(model)
        if final_sha != initial_sha:
            raise AssertionError("Generator changed during the Critic phase")
        gradients = parameter_gradient_diagnostics(model)
        if gradients["trainable_grad_present"] or gradients["frozen_grad_present"]:
            raise AssertionError("Generator retained Critic-phase gradients")
    if rank == 0 and len(ema_records) != 5:
        raise AssertionError("continuous Full5 requires five EMA updates")
    if (produced_batches, committed_batches) != (5, 5):
        raise AssertionError("ACK-free queue did not commit all five global batches")
    result["critic_phase"] = {
        "schedule": args.schedule,
        "transport_policy": "ACK_FREE_NCCL_SEND_COMPLETION_BACKPRESSURE",
        "seconds": execution_seconds,
        "waves": wave_records,
        "generator_version_before": 3,
        "generator_version_after": 3,
        "generator_trainable_sha256_before": (
            initial_sha if not args.timing_mode else TIMING_OMITTED
        ),
        "generator_trainable_sha256_after": final_sha,
        "generator_gradients": gradients,
        "ema": ema_records,
        "queue": {
            "produced_global_batches": produced_batches,
            "committed_global_batches": committed_batches,
            "produced_payloads": produced_payloads,
            "sent_payloads": sent_payloads,
            "per_route_commit_ack": False,
            "producer_ready_not_sent_max": max_ready_not_sent,
            "produced_minus_committed_max": max_produced_minus_committed,
            "producer_slot": "EMPTY",
            "outstanding_transport": 0,
        },
    }


def _run_student_cycle(
    *,
    args: argparse.Namespace,
    topology: RouteBCandidateBTopology,
    rank: int,
    pool_rank: int,
    device: torch.device,
    engine: Any,
    residency: _ResidentStudentParameterGuard,
    activation: Any,
    trace: RouteBFull5Trace,
    transport: CriticPoolP2PTransport,
    control_group: Any,
    result: dict[str, Any],
) -> None:
    observer = None if args.timing_mode else TrainableGradientObserver(engine.module)
    initial_optimizer = (
        TIMING_OMITTED if args.timing_mode else optimizer_state_checksum(engine.optimizer)
    )
    initial_trainable = (
        TIMING_OMITTED if args.timing_mode else trainable_checksum(engine.module)
    )
    wave_records: list[dict[str, Any]] = []
    producer_rank, production_round = topology.producer_for_critic(rank)
    producer_pool_rank = topology.generator_pool_rank(producer_rank)
    received = consumed = committed = 0
    cycle_started = time.perf_counter()

    for update in range(CONTRACT.logical_critic_updates):
        entry = trace.entry(update, production_round, producer_pool_rank)
        receive_started = time.perf_counter()
        header, x0 = transport.receive_wave(source_global_rank=producer_rank)
        receive_seconds = time.perf_counter() - receive_started
        received += 1
        _validate_received_header(
            header=header,
            topology=topology,
            rank=rank,
            entry=entry,
            update=update,
            producer_rank=producer_rank,
            producer_pool_rank=producer_pool_rank,
        )
        loaded = trace.load_critic_entry(
            update=update,
            production_round=production_round,
            producer_pool_rank=producer_pool_rank,
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
        kind = _checksum_kind(args)
        prediction_identity = _identity_record(prediction, kind)
        noise_identity = _identity_record(batch["critic_noise"], kind)
        loss_value = float(loss.detach().cpu())

        backward_started = time.perf_counter()
        engine.backward(loss)
        torch.cuda.synchronize(device)
        backward_seconds = time.perf_counter() - backward_started
        delivered = observer.end() if observer is not None else {"timing_mode": True}
        if observer is not None and delivered["parameters_with_gradient"] == 0:
            raise AssertionError("Candidate-B Critic delivered no LoRA gradient")
        diagnostics = (
            parameter_gradient_diagnostics(engine.module)
            if not args.timing_mode
            else {"measurement": TIMING_OMITTED}
        )
        if not args.timing_mode and diagnostics["frozen_grad_present"]:
            raise AssertionError("frozen Score backbone received a gradient")
        if not bool(engine.is_gradient_accumulation_boundary()):
            raise AssertionError("Candidate-B GAS1 boundary changed")
        consumed += 1

        step_started = time.perf_counter()
        engine.step()
        torch.cuda.synchronize(device)
        step_seconds = time.perf_counter() - step_started
        preclip_norm = float(engine.get_global_grad_norm())
        clip_scale = min(1.0, 1.0 / (preclip_norm + 1e-6))
        optimizer_after = _optimizer_step(engine.optimizer)
        scheduler_after = _scheduler_step(engine)
        expected = (10 + update, 11 + update, 10 + update, 11 + update, 11 + update)
        observed = (
            optimizer_before,
            optimizer_after,
            scheduler_before,
            scheduler_after,
            int(engine.global_steps),
        )
        if observed != expected:
            raise AssertionError(f"Candidate-B C version order changed: {observed}")
        committed += 1
        activation.offloader.assert_scope_closed()

        commit_rendezvous_started = time.perf_counter()
        # All twelve Critic ranks arrive only after the common ZeRO-1 commit.
        # Generator ranks arrive after optionally preparing the future batch.
        _control_barrier(control_group)
        commit_rendezvous_seconds = time.perf_counter() - commit_rendezvous_started
        ema_order_started = time.perf_counter()
        # Rank0 performs EMA between the two rendezvous.  This second fence
        # prevents any Critic from posting update t+1's receive too early.
        _control_barrier(control_group)
        ema_order_seconds = time.perf_counter() - ema_order_started
        record = {
            "update": update,
            "logical_step": 10 + update,
            "production_round": production_round,
            "producer_global_rank": producer_rank,
            "trace_key": entry.trace_key,
            "trace_identity_sha256": entry.identity_sha256,
            "source_trace_rank": entry.source_trace_rank,
            "x0_identity": {
                "kind": header.checksum_kind.name,
                "value": header.payload_checksum.hex(),
            },
            "prediction_identity": prediction_identity,
            "critic_noise_identity": noise_identity,
            "critic_timestep": [
                int(value) for value in batch["timestep"].detach().cpu().tolist()
            ],
            "loss": loss_value,
            "receive_seconds": receive_seconds,
            "forward_seconds": forward_seconds,
            "backward_seconds": backward_seconds,
            "optimizer_seconds": step_seconds,
            "per_route_commit_ack": False,
            "commit_rendezvous_seconds": commit_rendezvous_seconds,
            "ema_order_seconds": ema_order_seconds,
            "gradient_delivery": delivered,
            "gradient_diagnostics_before_step": diagnostics,
            "gradient_accumulation_boundary": True,
            "gradient_clipping": {
                "configured_max_norm": 1.0,
                "preclip_norm": preclip_norm,
                "predicted_scale": clip_scale,
            },
            "critic_version_before": optimizer_before,
            "critic_version_after": optimizer_after,
            "scheduler_before": scheduler_before,
            "scheduler_after": scheduler_after,
        }
        if kind is ChecksumKind.SHA256:
            record.update(
                {
                    "prediction_sha256": prediction_identity["value"],
                    "critic_noise_sha256": noise_identity["value"],
                }
            )
        wave_records.append(record)
        del x0, loaded, batch, prediction, loss

    transport.flush()
    transport.assert_flushed()
    if observer is not None:
        observer.close()
    execution_seconds = time.perf_counter() - cycle_started
    residency.assert_unchanged(engine.module)
    if args.export_state:
        result["final_state_export"] = _export_optimizer_partition(
            engine=engine,
            root=args.output_dir / "state",
            phase="final",
            rank=rank,
        )
    if (
        int(engine.global_steps),
        _optimizer_step(engine.optimizer),
        _scheduler_step(engine),
    ) != (15, 15, 15):
        raise AssertionError("Candidate-B Critic did not finish at C15")
    result["critic_phase"] = {
        "schedule": args.schedule,
        "seconds": execution_seconds,
        "waves": wave_records,
        "critic_version_before": 10,
        "critic_version_after": 15,
        "optimizer_step_before": 10,
        "optimizer_step_after": 15,
        "scheduler_step_before": 10,
        "scheduler_step_after": 15,
        "initial_optimizer": initial_optimizer,
        "final_optimizer": (
            TIMING_OMITTED
            if args.timing_mode
            else optimizer_state_checksum(engine.optimizer)
        ),
        "initial_trainable": initial_trainable,
        "final_trainable": (
            TIMING_OMITTED if args.timing_mode else trainable_checksum(engine.module)
        ),
        "final_module_sha256_pool_rank0": (
            trainable_sha256(engine.module)
            if pool_rank == 0 and not args.timing_mode
            else TIMING_OMITTED
        ),
        "optimizer_partition_inventory": (
            TIMING_OMITTED
            if args.timing_mode
            else _optimizer_partition_inventory(engine)
        ),
        "parameter_residency": residency.record(),
        "queue": {
            "received": received,
            "consumed": consumed,
            "committed": committed,
            "per_route_commit_ack": False,
            "consumer_slot": "EMPTY",
            "outstanding_transport": 0,
        },
    }

def _timing_correctness_binding(args: argparse.Namespace) -> dict[str, Any] | None:
    supplied = args.correctness_evaluation is not None or args.expected_correctness_sha256
    if not args.timing_mode:
        if supplied:
            raise ValueError("correctness binding is only legal in timing mode")
        return None
    if args.correctness_evaluation is None or not args.expected_correctness_sha256:
        raise ValueError("Candidate-B timing requires a frozen correctness evaluation")
    observed = _sha256_file(args.correctness_evaluation)
    if observed != args.expected_correctness_sha256:
        raise ValueError("Candidate-B correctness evaluation SHA-256 changed")
    value = json.loads(args.correctness_evaluation.read_text())
    if value.get("status") != "PASS" or value.get("errors") != []:
        raise ValueError("Candidate-B correctness evaluation is not PASS")
    return {
        "path": str(args.correctness_evaluation.resolve()),
        "sha256": observed,
        "status": "PASS",
        "reference_run": value.get("reference_run"),
        "candidate_run": value.get("candidate_run"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--node-rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--phase-b-master-addr", required=True)
    parser.add_argument("--phase-b-master-port", type=int, required=True)
    parser.add_argument("--schedule", choices=("sequential", "pipeline"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-nonce", type=int, required=True)
    parser.add_argument("--trace-bundle", type=Path, required=True)
    parser.add_argument("--logical-package", type=Path, required=True)
    parser.add_argument("--expected-logical-manifest-hash", required=True)
    parser.add_argument("--native-world8-root", type=Path, required=True)
    parser.add_argument("--generator-root", type=Path, required=True)
    parser.add_argument("--precast-generator-root", type=Path, required=True)
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--precast-score-root", type=Path, required=True)
    parser.add_argument("--negative-prompt-asset", type=Path, required=True)
    parser.add_argument("--mmap-cache-root", type=Path, required=True)
    parser.add_argument("--activation-min-bytes", type=int, default=32 * 2**20)
    parser.add_argument(
        "--step10a-param-persistence-threshold",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--step10a-module-granularity-threshold", type=int, default=0
    )
    parser.add_argument("--step10a-block-leaf-fetch", action="store_true")
    parser.add_argument(
        "--step10a-prefetch-bucket-elements", type=int, default=16_777_216
    )
    parser.add_argument(
        "--step10a-max-live-parameters", type=int, default=268_435_456
    )
    parser.add_argument(
        "--step10a-max-reuse-distance", type=int, default=268_435_456
    )
    parser.add_argument("--step10a-overlap-comm", action="store_true")
    parser.add_argument("--step10a-saved-tensor-census", action="store_true")
    parser.add_argument("--step10a-boundary-pinned-sync", action="store_true")
    parser.add_argument(
        "--transition-h2d-policy",
        choices=("module_to", "bounded_pinned", "persistent_pinned"),
        default="module_to",
    )
    parser.add_argument("--transition-h2d-chunk-mib", type=int, default=64)
    parser.add_argument("--transition-h2d-slot-count", type=int, choices=(1, 2), default=2)
    parser.add_argument(
        "--transition-pinned-cleanup-policy",
        choices=("immediate", "deferred", "late_deferred", "post_full5"),
        default="immediate",
    )
    parser.add_argument("--g3-broadcast-bucket-mib", type=int, default=0)
    parser.add_argument("--checksum", choices=("sha256", "gpu"), default="sha256")
    parser.add_argument("--timing-mode", action="store_true")
    parser.add_argument("--stop-after-step10a", action="store_true")
    parser.add_argument("--export-state", action="store_true")
    parser.add_argument("--correctness-evaluation", type=Path)
    parser.add_argument("--expected-correctness-sha256")
    parser.add_argument("--timeout-seconds", type=float, default=7200.0)
    args = parser.parse_args()
    if args.timing_mode and args.checksum != "gpu":
        parser.error("--timing-mode requires --checksum gpu")
    if args.timing_mode and args.export_state:
        parser.error("--timing-mode forbids state export")
    if args.stop_after_step10a and not args.timing_mode:
        parser.error("--stop-after-step10a is a timing diagnostic only")
    if args.step10a_saved_tensor_census and not (
        args.stop_after_step10a and args.timing_mode
    ):
        parser.error(
            "--step10a-saved-tensor-census requires Step10-only timing diagnostic"
        )
    if args.step10a_saved_tensor_census and args.step10a_boundary_pinned_sync:
        parser.error(
            "saved-tensor census and boundary pinned policy are mutually exclusive"
        )
    if args.step10a_param_persistence_threshold < 0:
        parser.error("--step10a-param-persistence-threshold must be non-negative")
    if args.step10a_module_granularity_threshold < 0:
        parser.error("--step10a-module-granularity-threshold must be non-negative")
    if (
        args.step10a_block_leaf_fetch
        and args.step10a_module_granularity_threshold != 0
    ):
        parser.error(
            "explicit block leaves and module-granularity threshold are mutually exclusive"
        )
    if args.step10a_prefetch_bucket_elements <= 0:
        parser.error("--step10a-prefetch-bucket-elements must be positive")
    if args.step10a_max_live_parameters <= 0:
        parser.error("--step10a-max-live-parameters must be positive")
    if args.step10a_max_reuse_distance <= 0:
        parser.error("--step10a-max-reuse-distance must be positive")
    if args.transition_h2d_chunk_mib <= 0:
        parser.error("--transition-h2d-chunk-mib must be positive")
    if args.g3_broadcast_bucket_mib < 0:
        parser.error("--g3-broadcast-bucket-mib must be non-negative")
    binding = _timing_correctness_binding(args)

    phase_a_rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    phase_a_world_size = int(os.environ["WORLD_SIZE"])
    if phase_a_world_size != PHASE_A_WORLD_SIZE or phase_a_rank != local_rank:
        raise ValueError(
            "continuous Full5 Phase A requires an independent node-local world8"
        )
    rank = _phase_a_global_rank(node_rank=args.node_rank, local_rank=local_rank)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    import deepspeed

    _initialize_phase_a_distributed()
    phase_a_control_group = _create_phase_a_groups(
        node_rank=args.node_rank
    )
    if rank == 0:
        if args.output_dir.exists():
            raise FileExistsError(args.output_dir)
        args.output_dir.mkdir(parents=True)
        (args.output_dir / "exchange").mkdir()
    exchange = args.output_dir / "exchange"
    _control_barrier(phase_a_control_group)
    if phase_a_rank == 0:
        _wait_for_directory(exchange, args.timeout_seconds)
    _control_barrier(phase_a_control_group)
    if not exchange.is_dir():
        raise FileNotFoundError("rank0 did not publish the exchange directory")
    _phase_a_file_barrier(
        exchange=exchange,
        name="output_ready",
        node_rank=args.node_rank,
        control_group=phase_a_control_group,
        timeout_seconds=args.timeout_seconds,
    )

    source_physical_role, source_execution_role = _step10a_role(rank)
    target_role = TOPOLOGY.role(rank)
    target_pool_rank = (
        TOPOLOGY.generator_pool_rank(rank)
        if target_role == "generator"
        else TOPOLOGY.critic_pool_rank(rank)
    )
    output = args.output_dir / "ranks" / f"rank_{rank:02d}.json"
    result: dict[str, Any] = {
        "schema_version": "distillschedule.rb16_full5_2g6c_continuous_rank.v1",
        "status": "RUNNING",
        "experiment_id": "RB16-F5-CONT-8G4S4T-TO-2G6C",
        "run_id": args.run_id,
        "schedule": args.schedule,
        "execution_scope": (
            "step10a_only_timing_diagnostic"
            if args.stop_after_step10a
            else "continuous_full5_step10_to_step14"
        ),
        "claim_scope": "batch12_resource_throughput_pareto_not_matched_batch16_speedup",
        "step10a_param_persistence_threshold": (
            args.step10a_param_persistence_threshold
        ),
        "step10a_module_granularity_threshold": (
            args.step10a_module_granularity_threshold
        ),
        "step10a_block_leaf_fetch": args.step10a_block_leaf_fetch,
        "step10a_prefetch_bucket_elements": args.step10a_prefetch_bucket_elements,
        "step10a_max_live_parameters": args.step10a_max_live_parameters,
        "step10a_max_reuse_distance": args.step10a_max_reuse_distance,
        "step10a_overlap_comm": args.step10a_overlap_comm,
        "step10a_saved_tensor_census": args.step10a_saved_tensor_census,
        "step10a_boundary_pinned_sync": args.step10a_boundary_pinned_sync,
        "transition_h2d_policy": args.transition_h2d_policy,
        "transition_h2d_chunk_mib": args.transition_h2d_chunk_mib,
        "transition_h2d_slot_count": args.transition_h2d_slot_count,
        "transition_pinned_cleanup_policy": (
            args.transition_pinned_cleanup_policy
        ),
        "g3_broadcast_bucket_mib": args.g3_broadcast_bucket_mib,
        "hostname": socket.gethostname(),
        "rank": rank,
        "local_rank": local_rank,
        "world_size": PHASE_B_WORLD_SIZE,
        "phase_a": {
            "node_rank": args.node_rank,
            "default_rank": phase_a_rank,
            "world_size": phase_a_world_size,
            "process_world_policy": "TWO_NODE_LOCAL_DEFAULT_WORLD8S",
        },
        "source_role": source_execution_role,
        "target_role": target_role,
        "target_pool_rank": target_pool_rank,
        "step10a_contract": {
            "generator_ranks": list(STEP10A_G_RANKS),
            "student_ranks": list(STEP10A_STUDENT_RANKS),
            "teacher_ranks": list(STEP10A_TEACHER_RANKS),
            "generator_backend": "ZeRO-3",
            "generator_parameter_offload": False,
            "generator_optimizer_offload": "cpu",
            "generator_checkpoint_policy": (
                "zero3_metadata_tolerant_block_local_non_reentrant"
            ),
            "dp": 8,
            "micro_batch": 1,
            "gradient_accumulation_steps": 2,
            "global_batch": 16,
        },
        "critic_contract": CONTRACT.__dict__,
        "transport_policy": "ACK_FREE_NCCL_SEND_COMPLETION_BACKPRESSURE",
        "timing_mode": bool(args.timing_mode),
        "checksum": args.checksum,
        "stages": [_snapshot(device, "process_start")],
    }
    if binding is not None:
        result["timing_correctness_binding"] = binding
    _atomic_json(output, result)

    source_model = target_model = generator_model = None
    step10a_engine = critic_engine = None
    streamer = step10a_activation = critic_activation = step10a_census = None
    residency = ema = negative_embedding = None
    transports: dict[int, CriticPoolP2PTransport] = {}
    started = time.perf_counter()
    try:
        logical = LogicalCheckpointPackage.load(args.logical_package)
        if logical.manifest_hash != args.expected_logical_manifest_hash:
            raise ValueError("logical G2/C10/EMA10 manifest hash changed")
        trace = RouteBFull5Trace(args.trace_bundle, CONTRACT)
        result["identities"] = {
            "execution_trace_bundle_hash": trace.bundle_hash,
            "routeb_trace_binding_hash": trace.binding_hash,
            "logical_manifest_hash": logical.manifest_hash,
            "state_provenance_trace_hash": str(logical.manifest["input_trace_hash"]),
        }

        source_model, source_metadata, source_restore = _construct_step10a_source(
            args=args,
            rank=rank,
            device=device,
            logical=logical,
            control_group=phase_a_control_group,
        )
        result["source_construction"] = {
            "physical_role": source_physical_role,
            "execution_role": source_execution_role,
            "metadata": source_metadata,
            "logical_restore": source_restore,
        }

        if rank in STEP10A_G_RANKS:
            step10a_engine, step10a_initialization = _initialize_step10a_generator(
                args=args,
                model=source_model,
                logical=logical,
                rank=rank,
                group=dist.group.WORLD,
            )
            result["step10a_generator_initialization"] = step10a_initialization
            result["step10a_generator_runtime"] = {
                "backend": "ZeRO-3",
                "parameter_offload": False,
                "optimizer_offload": "cpu",
                "frozen_block_streaming": False,
                "activation_offload": bool(args.step10a_boundary_pinned_sync),
                "activation_offload_policy": (
                    "SELECTIVE_EXACT_SIGNATURE_PINNED_SYNC"
                    if args.step10a_boundary_pinned_sync
                    else "NONE"
                ),
                "boundary_pinned_sync": bool(args.step10a_boundary_pinned_sync),
                "gradient_checkpointing": (
                    "zero3_metadata_tolerant_block_local_non_reentrant"
                ),
                "reference": "validated_8G4S4T_Step10A_backend",
            }
        elif rank in STEP10A_TEACHER_RANKS:
            negative = NegativePromptAsset.load(
                args.negative_prompt_asset,
                verify_payload_sha256=(rank == STEP10A_TEACHER_RANKS[0]),
            )
            negative_embedding = negative.embedding.to(
                device=device, dtype=torch.bfloat16
            )

        _control_barrier(phase_a_control_group)
        target_model, target_metadata, target_restore = _preconstruct_switch_target(
            args=args,
            rank=rank,
            local_rank=local_rank,
            logical=logical,
            control_group=phase_a_control_group,
        )
        target_host_preparation = None
        if (
            target_model is not None
            and args.transition_h2d_policy == "persistent_pinned"
        ):
            target_host_preparation = pin_module_cpu_registered_state(target_model)
        result["target_preconstruction"] = {
            "target_preconstructed": rank in PRECONSTRUCT_TARGET_RANKS,
            "target_role": target_role,
            "device": "cpu" if target_model is not None else "retained_source_model",
            "metadata": target_metadata,
            "restore": target_restore,
            "host_preparation": target_host_preparation,
            "timing_scope": "excluded_cold_preparation",
        }
        if rank == 0:
            if target_model is None or target_role != "generator":
                raise AssertionError("EMA owner lacks preconstructed target Generator")
            ema = SparseShardEMA.load_pretrained(
                target_model, args.logical_package / "ema"
            )
            if ema.optimization_step != 10:
                raise AssertionError("logical package did not restore EMA10")
        if args.step10a_saved_tensor_census and rank in STEP10A_G_RANKS:
            event_root = args.output_dir / "events"
            event_root.mkdir(parents=True, exist_ok=True)
            step10a_census = install_saved_tensor_census_hooks(
                step10a_engine.module,
                event_path=(
                    event_root / f"rank_{rank:02d}_step10a_saved_tensor_census.jsonl"
                ),
                logical_component="step10a_generator_selected",
            )
        if args.step10a_boundary_pinned_sync and rank in STEP10A_G_RANKS:
            event_root = args.output_dir / "events"
            event_root.mkdir(parents=True, exist_ok=True)
            step10a_activation = install_activation_offload_hooks(
                step10a_engine.module,
                logical_component="step10a_generator_selected_boundary_pinned_sync",
                event_path=(
                    event_root
                    / f"rank_{rank:02d}_step10a_boundary_pinned_sync.jsonl"
                ),
                pin_memory=True,
                min_offload_bytes=0,
                detailed_events=False,
                allowed_offload_signatures=STEP10A_BOUNDARY_OFFLOAD_SIGNATURES,
            )
        _phase_a_file_barrier(
            exchange=exchange,
            name="models_ready",
            node_rank=args.node_rank,
            control_group=phase_a_control_group,
            timeout_seconds=args.timeout_seconds,
        )
        result["stages"].append(_snapshot(device, "step10a_and_future_cpu_models_ready"))
        _atomic_json(output, result)

        torch.cuda.synchronize(device)
        _phase_a_file_barrier(
            exchange=exchange,
            name="full5_start",
            node_rank=args.node_rank,
            control_group=phase_a_control_group,
            timeout_seconds=args.timeout_seconds,
        )
        profile_step10a = os.environ.get(
            "DISTILLSCHEDULE_STEP10A_CUDA_PROFILER", "0"
        ) == "1"
        if profile_step10a:
            if not args.stop_after_step10a or not args.timing_mode:
                raise ValueError("Step10-A CUDA profiler requires bounded timing mode")
            if args.node_rank != 0:
                raise ValueError("Step10-A CUDA profiler is restricted to node0 G8")
            torch.cuda.profiler.start()
        full5_started = time.perf_counter()
        with _nvtx_range(f"step10a/rank={rank}/role={source_execution_role}"):
            if rank in STEP10A_G_RANKS:
                _run_step10a_generator(
                    args=args,
                    rank=rank,
                    device=device,
                    engine=step10a_engine,
                    trace=trace,
                    exchange=exchange,
                    output=output,
                    result=result,
                )
            else:
                _run_step10a_score(
                    args=args,
                    rank=rank,
                    device=device,
                    model=source_model,
                    trace=trace,
                    exchange=exchange,
                    negative_embedding=negative_embedding,
                    result=result,
                )
        if step10a_census is not None:
            result["step10a_saved_tensor_census"] = step10a_census.remove()
            step10a_census = None
        if step10a_activation is not None:
            step10a_activation.remove()
            stats = dict(step10a_activation.offloader.stats)
            if stats["live_cpu_bytes"] != 0:
                raise RuntimeError(
                    "Step10-A boundary pinned handles remained live after backward"
                )
            result["step10a_boundary_pinned_sync"] = {
                "policy": "SELECTIVE_EXACT_SIGNATURE_PINNED_SYNC",
                "allowed_signatures": [
                    {
                        "shape": list(shape),
                        "dtype": dtype,
                        "logical_bytes": logical_bytes,
                    }
                    for shape, dtype, logical_bytes in sorted(
                        STEP10A_BOUNDARY_OFFLOAD_SIGNATURES
                    )
                ],
                "stats": stats,
            }
            step10a_activation = None
        _phase_a_file_barrier(
            exchange=exchange,
            name="step10a_complete",
            node_rank=args.node_rank,
            control_group=phase_a_control_group,
            timeout_seconds=args.timeout_seconds,
        )
        result["step10a_wall_seconds"] = time.perf_counter() - full5_started
        result["stages"].append(_snapshot(device, "step10a_complete"))
        if profile_step10a:
            torch.cuda.profiler.stop()

        if args.stop_after_step10a:
            # The diagnostic deliberately ends before G3 redistribution or any
            # Critic/EMA phase.  It reuses the passed Full5 correctness binding
            # but makes no claim about the unexecuted transition or steps 10-B
            # through 14.
            result["diagnostic_scope"] = {
                "included": "Step10-A G2_to_G3 plus frozen S/T DMD service",
                "excluded": [
                    "G3 representation transition",
                    "Critic forward_backward_optimizer",
                    "EMA update",
                    "continuous Full5 timing",
                ],
                "split_synchronize": "diagnostic_only_between_backward_and_engine_step",
            }
            result["final_versions"] = {
                "generator": 3,
                "critic": 10,
                "ema": 10,
                "next_boundary": "Step10-B_not_executed",
            }
            result["status"] = "PASS"
            result["duration_seconds"] = time.perf_counter() - started
            result["stages"].append(_snapshot(device, "step10a_diagnostic_complete"))
            _atomic_json(output, result)
            return

        transition_started = time.perf_counter()
        transition_subphases: dict[str, float] = {}
        if negative_embedding is not None:
            del negative_embedding
            negative_embedding = None

        # The validated Step10-A backend is DP8 ZeRO-3, while the Critic phase
        # requires full resident no-grad Generator replicas on ranks 0/1/8/9.
        # Export the updated logical G3 trainables while all eight source ranks
        # still participate in the same collective order.  Never reinterpret a
        # sharded engine in place as a resident module.
        g3_extraction = None
        with _transition_phase(transition_subphases, "g3_extract_local_host_wall_seconds"):
            if rank in STEP10A_G_RANKS:
                g3_extraction = _materialize_zero3_g3_into_target(
                    engine=step10a_engine,
                    target_model=(target_model if rank == 0 else None),
                    logical=logical,
                    rank=rank,
                    timing_mode=args.timing_mode,
                )
        with _transition_phase(transition_subphases, "g3_extract_fence_wall_seconds"):
            _phase_a_file_barrier(
                exchange=exchange,
                name="g3_gather_complete",
                node_rank=args.node_rank,
                control_group=phase_a_control_group,
                timeout_seconds=args.timeout_seconds,
            )

        source_release = None
        with _transition_phase(transition_subphases, "source_release_local_host_wall_seconds"):
            if rank in STEP10A_G_RANKS:
                # Drop the caller's second alias before the destroy helper measures
                # released storage; the engine remains the sole module owner.
                source_model = None
                source_release = _destroy_obsolete_zero3_engine(
                    step10a_engine, device=device
                )
                step10a_engine = None
                source_release = _verify_obsolete_zero3_release(
                    source_release, device=device
                )
            elif rank in SWITCH_TO_GENERATOR_RANKS:
                # Ranks 8/9 relinquish the Step10-A physical Score before their
                # preconstructed resident Generator enters GPU memory.
                source_model = None
                gc.collect()
                torch.cuda.empty_cache()
        with _transition_phase(transition_subphases, "source_release_fence_wall_seconds"):
            _phase_a_file_barrier(
                exchange=exchange,
                name="source_release_complete",
                node_rank=args.node_rank,
                control_group=phase_a_control_group,
                timeout_seconds=args.timeout_seconds,
            )

        # The node-local Phase-A worlds are now obsolete.  Destroy every old
        # group before creating the global target world; no ZeRO-3 shard or
        # NCCL work is reinterpreted in place.  The updated logical G3 tensor
        # already resides in rank 0's separately preconstructed CPU target.
        # phase_a_control_group aliases the default world; destroy it exactly
        # once before forming the explicit global Phase-B world16.
        phase_a_control_group = None
        with _transition_phase(transition_subphases, "phase_a_world_destroy_wall_seconds"):
            dist.destroy_process_group()
        if dist.is_initialized():
            raise AssertionError("Phase-A process groups survived destroy")
        with _transition_phase(transition_subphases, "phase_b_world_init_wall_seconds"):
            phase_b_process_world = _initialize_phase_b_distributed(
                deepspeed,
                global_rank=rank,
                master_addr=args.phase_b_master_addr,
                master_port=args.phase_b_master_port,
            )
        with _transition_phase(transition_subphases, "phase_b_group_create_wall_seconds"):
            (
                control_group,
                target_generator_group,
                target_critic_group,
                pair_groups,
                singleton_groups,
            ) = _create_phase_b_groups(TOPOLOGY)
        result["phase_transition"] = {
            "source": "node_local_default_world8",
            "target": "global_world16",
            "gather_before_destroy": True,
            "in_place_zero_reinterpretation": False,
            **phase_b_process_world,
        }
        _atomic_json(output, result)

        if target_role == "generator":
            if target_model is None or rank not in CONTRACT.generator_ranks:
                raise AssertionError("target Generator rank lacks its CPU model")
            with _transition_phase(transition_subphases, "target_model_h2d_wall_seconds"):
                target_materialization = _materialize_transition_target(
                    args=args,
                    model=target_model,
                    device=device,
                )
            generator_model = target_model
            target_model = None
            result["transition"] = {
                "source_role": source_execution_role,
                "target_role": "critic_phase_generator",
                "model_reused": False,
                "target_model_source": "cold_preconstructed_cpu_model",
                "step10a_backend": "ZeRO-3_DP8",
                "g3_extraction": g3_extraction,
                "source_release": source_release,
                "target_model_materialization": target_materialization,
            }
            generator_model.enable_adapters()
            generator_model.train()
            with _transition_phase(transition_subphases, "g3_broadcast_wall_seconds"):
                redistribution = _broadcast_generator_g3(
                    model=generator_model,
                    logical=logical,
                    rank=rank,
                    group=target_generator_group,
                    timing_mode=args.timing_mode,
                    bucket_mib=args.g3_broadcast_bucket_mib,
                )
            result["transition"]["generator_g3_redistribution"] = redistribution
            generator_initial_sha = (
                TIMING_OMITTED
                if args.timing_mode
                else trainable_sha256(generator_model)
            )
        else:
            if rank in SWITCH_TO_CRITIC_RANKS:
                if target_model is None:
                    raise AssertionError("switch-to-Critic rank lacks its CPU model")
                with _transition_phase(transition_subphases, "target_model_h2d_wall_seconds"):
                    target_materialization = _materialize_transition_target(
                        args=args,
                        model=target_model,
                        device=device,
                    )
                critic_model = target_model
                target_model = None
                construction_restore = target_restore
                critic_metadata = target_metadata
                transition_source = "cold_preconstructed_cpu_score_after_generator_release"
            else:
                if rank not in (10, 11, 12, 13, 14, 15):
                    raise AssertionError("unexpected retained Critic rank")
                critic_model = source_model
                construction_restore = source_restore
                critic_metadata = source_metadata
                transition_source = "reuse_step10a_physical_score_model"
            adapter_reactivation = _reactivate_critic_adapter(critic_model)
            with _transition_phase(transition_subphases, "student_initialize_total_host_wall_seconds"):
                critic_engine, residency, critic_activation = _initialize_student(
                    args=args,
                    model=critic_model,
                    metadata=critic_metadata,
                    construction_restore=construction_restore,
                    package=logical,
                    rank=rank,
                    pool_rank=target_pool_rank,
                    student_group=target_critic_group,
                    singleton_group=singleton_groups[rank],
                    device=device,
                    result=result,
                )
            result["transition"] = {
                "source_role": source_execution_role,
                "target_role": "critic",
                "model_reused": rank not in SWITCH_TO_CRITIC_RANKS,
                "target_model_source": transition_source,
                "adapter_reactivation": adapter_reactivation,
                "source_release": source_release,
                "target_model_materialization": (
                    target_materialization
                    if rank in SWITCH_TO_CRITIC_RANKS
                    else {"policy": "REUSE_RESIDENT_SCORE", "ready_on_return": True}
                ),
                "step10a_backend": (
                    "ZeRO-3_DP8" if rank in STEP10A_G_RANKS else "resident_Score"
                ),
                "g3_extraction": g3_extraction,
                "critic_version": 10,
                "optimizer_step": 10,
                "scheduler_step": 10,
            }

        with _transition_phase(transition_subphases, "target_ready_fence_wall_seconds"):
            _control_barrier(control_group)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        result["transition"]["rank_transition_seconds"] = (
            time.perf_counter() - transition_started
        )
        result["transition_wall_seconds"] = result["transition"][
            "rank_transition_seconds"
        ]
        result["transition"]["subphase_seconds"] = transition_subphases
        result["transition"]["timing_semantics"] = (
            "host_wall_at_existing_boundaries_no_added_cuda_synchronize"
        )
        result["stages"].append(_snapshot(device, "target_4g12c_layout_ready"))
        _atomic_json(output, result)

        if target_role == "generator":
            for destination in TOPOLOGY.critics_for_generator(rank):
                key = (rank, destination)
                transports[destination] = CriticPoolP2PTransport(
                    group=pair_groups[key], device=device
                )
        else:
            source_rank, _production_round = TOPOLOGY.producer_for_critic(rank)
            key = (source_rank, rank)
            transports[source_rank] = CriticPoolP2PTransport(
                group=pair_groups[key], device=device
            )
            if args.export_state:
                result["initial_state_export"] = _export_optimizer_partition(
                    engine=critic_engine,
                    root=args.output_dir / "state",
                    phase="initial",
                    rank=rank,
                )

        _control_barrier(control_group)
        critic_phase_started = time.perf_counter()
        if args.transition_pinned_cleanup_policy == "late_deferred":
            materialization = result["transition"]["target_model_materialization"]
            if materialization.get("policy") == "PERSISTENT_PINNED_DIRECT_H2D":
                cleanup_token = start_deferred_host_allocator_cleanup()
                materialization["deferred_host_cleanup_token"] = cleanup_token
                materialization["late_cleanup_start"] = (
                    "critic_phase_entry_after_global_target_ready"
                )
        if target_role == "generator":
            _run_generator_cycle(
                args=args,
                topology=TOPOLOGY,
                rank=rank,
                pool_rank=target_pool_rank,
                device=device,
                model=generator_model,
                initial_sha=generator_initial_sha,
                ema=ema,
                trace=trace,
                transports=transports,
                control_group=control_group,
                result=result,
            )
        else:
            source_rank, _production_round = TOPOLOGY.producer_for_critic(rank)
            _run_student_cycle(
                args=args,
                topology=TOPOLOGY,
                rank=rank,
                pool_rank=target_pool_rank,
                device=device,
                engine=critic_engine,
                residency=residency,
                activation=critic_activation,
                trace=trace,
                transport=transports[source_rank],
                control_group=control_group,
                result=result,
            )
        if args.transition_pinned_cleanup_policy == "post_full5":
            # Reach the semantic Step15-A boundary before touching the pinned
            # allocator cache.  The cache is a residency choice, not work in
            # the four Critic-only updates.  Cleanup remains mandatory and a
            # cleanup failure still invalidates the run.
            _control_barrier(control_group)
            torch.cuda.synchronize(device)
            result["critic_phase_wall_seconds"] = (
                time.perf_counter() - critic_phase_started
            )
            result["full5_execution_seconds"] = time.perf_counter() - full5_started
            materialization = result["transition"]["target_model_materialization"]
            result["deferred_host_cleanup"] = []
            cleanup_started = time.perf_counter()
            cleanup_trimmed = False
            if materialization.get("policy") == "PERSISTENT_PINNED_DIRECT_H2D":
                # This boundary provides no useful overlap: the old code
                # spawned a Python thread and immediately joined it.  Keep
                # torch's pinned-cache/GC cleanup on the CUDA-owning main
                # thread after all device work is synchronized.
                cleanup_trimmed = trim_host_allocator()
                materialization["deferred_host_cleanup_token"] = None
                materialization["late_cleanup_start"] = (
                    "after_step15a_boundary_outside_full5_timing_main_thread"
                )
            result["post_full5_host_cleanup"] = [
                {
                    "status": "PASS",
                    "execution_context": "main_thread",
                    "host_allocator_trimmed": cleanup_trimmed,
                    "wall_seconds": time.perf_counter() - cleanup_started,
                    "timing_scope": "outside_full5_execution_seconds",
                }
            ]
            _control_barrier(control_group)
        else:
            result["deferred_host_cleanup"] = (
                drain_deferred_host_allocator_cleanups()
            )
            _control_barrier(control_group)
            torch.cuda.synchronize(device)
            result["critic_phase_wall_seconds"] = (
                time.perf_counter() - critic_phase_started
            )
            result["full5_execution_seconds"] = time.perf_counter() - full5_started
        result["final_versions"] = {
            "generator": 3,
            "critic": 15,
            "ema": 15,
            "next_boundary": "Step15-A_not_executed",
        }
        result["status"] = "PASS"
        result["duration_seconds"] = time.perf_counter() - started
        result["stages"].append(_snapshot(device, "complete"))
        _atomic_json(output, result)
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
        try:
            result["stages"].append(_snapshot(device, "failure"))
        except BaseException:
            pass
        _atomic_json(output, result)
        raise
    finally:
        try:
            drain_deferred_host_allocator_cleanups()
        except BaseException:
            pass
        for transport in transports.values():
            try:
                transport.flush()
            except BaseException:
                pass
        for activation in (step10a_activation, critic_activation):
            if activation is not None:
                try:
                    activation.remove()
                except BaseException:
                    pass
        if step10a_census is not None:
            try:
                step10a_census.remove()
            except BaseException:
                pass
        if streamer is not None:
            try:
                streamer.remove()
            except BaseException:
                pass
        try:
            dist.destroy_process_group()
        except BaseException:
            pass


if __name__ == "__main__":
    main()
