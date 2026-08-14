#!/usr/bin/env python3
"""Real-14B Route-B Candidate-B executor on a node-local 2G+6C layout.

The two policies share model construction, target C11 restore, trace replay,
NCCL wire format, Student update, and flush checks.  Their only difference is
the Generator-side order:

* Sequential waits for wave ``i``'s commit before producing ``i+1``.
* Pipeline-Q1 produces ``i+1`` while Student consumes ``i``, then waits for the
  commit before sending the ready wave.

The immutable starting boundary is the already validated post-Step10-A
G3/C10/EMA10 state.  Five C updates execute with DP12/MB1/GAS1/global-batch12;
the only Sequential/Pipeline difference is legal G(next) || C(current)
overlap.  This is a batch12 resource-throughput Pareto point and is never used
as a direct speedup denominator for Candidate A's matched batch16 result.
"""
from __future__ import annotations

import argparse
import datetime
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
from runtime.critic_pool import (
    ChecksumKind,
    CriticCommitAck,
    CriticWaveHeader,
    StudentDPGroupAdapter,
)
from runtime.critic_pool_checkpoint import (
    CandidateBTransitionPackage,
    LogicalCheckpointPackage,
    load_candidate_b_generator_module_into_trainables,
    load_logical_master_into_trainables,
    restore_zero1_optimizer_partition_from_logical,
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
from runtime.official_baseline import (
    TrainableGradientObserver,
    optimizer_state_checksum,
    parameter_gradient_diagnostics,
    trainable_checksum,
    trainable_sha256,
)
from runtime.operation_trace import tensor_sha256
from runtime.routeb_full5 import (
    RouteBCandidateBTopology,
    RouteBFull5Contract,
)
from runtime.routeb_full5_trace import RouteBFull5Trace
from runtime.sparse_ema import SparseShardEMA


CONTRACT = RouteBFull5Contract.candidate_b()
TOPOLOGY = RouteBCandidateBTopology(CONTRACT)


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


def _initialize_distributed_runtime(deepspeed_module: Any) -> None:
    """Initialize torch.distributed before attaching DeepSpeed's wrapper.

    In this torch/DeepSpeed build, letting ``deepspeed.init_distributed`` own
    default-group creation makes Candidate B's twelve overlapping subgroups
    stall.  A bounded world16 probe reproduced the stall, while native torch
    initialization followed by ``dist_init_required=False`` completed the
    identical group graph.  This is an initialization-only substrate choice;
    DeepSpeed still owns the ZeRO-1 Student engine.
    """

    if dist.is_initialized():
        raise RuntimeError("Candidate-B distributed runtime was initialized twice")
    dist.init_process_group(
        backend="nccl", timeout=datetime.timedelta(hours=2)
    )
    deepspeed_module.init_distributed(
        dist_backend="nccl", dist_init_required=False
    )


def _create_groups(topology: RouteBCandidateBTopology):
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
    ack_groups: dict[tuple[int, int], Any] = {}
    for generator_rank, critic_rank, _round in topology.edges:
        key = (generator_rank, critic_rank)
        pair_groups[key] = ordered_group(
            ranks=[generator_rank, critic_rank], backend="nccl"
        )
        # Keep Candidate A's validated pair-then-ACK construction order.
        ack_groups[key] = ordered_group(
            ranks=[generator_rank, critic_rank], backend="gloo"
        )
    singleton = {
        rank: ordered_group(ranks=[rank], backend="nccl")
        for rank in topology.contract.critic_ranks
    }
    return control, pair_groups, ack_groups, generators, critics, singleton


def _control_barrier(group: Any) -> None:
    dist.barrier(group=group)


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


def _load_rng(package: CandidateBTransitionPackage, *, rank: int) -> None:
    value = torch.load(
        package.rng_path(rank), map_location="cpu", weights_only=False
    )
    source_rank = int(value.get("rank", rank))
    if source_rank != rank:
        raise AssertionError("Candidate-B transition RNG rank changed")
    torch.set_rng_state(value["cpu_rng_state"])
    torch.cuda.set_rng_state(value["cuda_rng_state"])


def _construct_model(
    *,
    args: argparse.Namespace,
    rank: int,
    local_rank: int,
    device: torch.device,
    transition: CandidateBTransitionPackage,
    logical: LogicalCheckpointPackage,
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    if args.construction_wave <= 0 or 8 % args.construction_wave:
        raise ValueError("construction wave must divide eight local ranks")
    model = metadata = restore = None
    for begin in range(0, 8, args.construction_wave):
        if begin <= local_rank < begin + args.construction_wave:
            if rank in CONTRACT.generator_ranks:
                model, metadata = build_helios_pool_model(
                    role="generator",
                    model_root=args.generator_root,
                    subfolder="transformer_ode",
                    precast_model_root=args.precast_generator_root,
                )
                restore = load_candidate_b_generator_module_into_trainables(
                    model, transition
                )
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
                    verify_payload_sha256=(rank == CONTRACT.critic_ranks[0]),
                )
            model.to(device)
            torch.cuda.synchronize(device)
        dist.barrier()
    if model is None or metadata is None or restore is None:
        raise AssertionError("construction wave skipped this rank")
    return model, metadata, restore


def _initialize_generator(
    *,
    model: torch.nn.Module,
    metadata: dict[str, Any],
    restore: dict[str, Any],
    pool_rank: int,
    device: torch.device,
    result: dict[str, Any],
) -> tuple[torch.nn.Module, str]:
    before = trainable_sha256(model)
    result["generator_initialization"] = {
        "construction": metadata,
        "restore": restore,
        "trainable_sha256": before,
        "optimizer_materialized": False,
        "scheduler_materialized": False,
        "parameter_version": 3,
    }
    result["stages"].append(_snapshot(device, "generator_resident"))
    return model, before


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
    engine, _, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        config=config,
        mpu=adapter,
        dist_init_required=False,
    )
    resolved = _assert_student_contract(engine, group=student_group)
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
    wave_records: list[dict[str, Any]] = []
    ema_records: list[dict[str, Any]] = []
    produced = sent = acknowledged = 0
    max_ready_not_sent = 0
    cycle_started = time.perf_counter()

    def produce_batch(update: int):
        nonlocal produced, max_ready_not_sent
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
            digest = bytes.fromhex(record["x0_identity"]["value"])
            header = _header(
                args=args,
                topology=topology,
                rank=rank,
                producer_pool_rank=pool_rank,
                update=update,
                production_round=production_round,
                destination=destination,
                trace_identity=entry.identity_sha256,
                payload_digest=digest,
            )
            record["destination_global_rank"] = destination
            batch.append((destination, x0, entry, header, record))
            produced += 1
        max_ready_not_sent = max(max_ready_not_sent, len(batch))
        return batch

    def send_batch(batch):
        nonlocal sent
        headers = []
        for destination, x0, entry, header, record in batch:
            started = time.perf_counter()
            pending = transports[destination].send_wave(
                header, x0, destination_global_rank=destination
            )
            pending.wait()
            transports[destination].reap_completed()
            record["send_seconds"] = time.perf_counter() - started
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
            headers.append((destination, header, record))
            sent += 1
            del x0, entry
        return headers

    def finish_update(update: int, headers):
        nonlocal acknowledged
        for destination, header, record in headers:
            started = time.perf_counter()
            ack = transports[destination].receive_ack(
                source_global_rank=destination
            )
            record["ack_wait_seconds"] = time.perf_counter() - started
            assert_ack_matches_header(ack, header)
            record["ack"] = ack.__dict__
            acknowledged += 1
            wave_records.append(record)
        if rank == 0:
            if ema is None:
                raise AssertionError("Candidate-B EMA owner is missing")
            before = ema.optimization_step
            started = time.perf_counter()
            decay = ema.step(model.parameters())
            if (before, ema.optimization_step, decay) != (
                10 + update,
                11 + update,
                0.0,
            ):
                raise AssertionError("official early-step EMA recurrence changed")
            comparison = (
                {"matches_exact": TIMING_OMITTED}
                if args.timing_mode
                else ema.compare_to_parameters(model.parameters())
            )
            if not args.timing_mode and comparison["matches_exact"] is not True:
                raise AssertionError("Candidate-B EMA no longer equals G3")
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
        _control_barrier(control_group)

    if args.schedule == "sequential":
        for update in range(CONTRACT.logical_critic_updates):
            batch = produce_batch(update)
            headers = send_batch(batch)
            # No Critic may begin until all twelve rank-qualified samples are
            # received; this is the true non-overlapped baseline.
            _control_barrier(control_group)
            finish_update(update, headers)
    else:
        current = send_batch(produce_batch(0))
        for update in range(CONTRACT.logical_critic_updates):
            next_batch = (
                produce_batch(update + 1)
                if update + 1 < CONTRACT.logical_critic_updates
                else None
            )
            finish_update(update, current)
            if next_batch is not None:
                current = send_batch(next_batch)

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
            raise AssertionError("Generator changed during Candidate-B Critic phase")
        gradients = parameter_gradient_diagnostics(model)
        if gradients["trainable_grad_present"] or gradients["frozen_grad_present"]:
            raise AssertionError("Generator retained Candidate-B gradients")
    if rank == 0 and len(ema_records) != 5:
        raise AssertionError("Candidate B must execute five official EMA updates")
    result["cycle"] = {
        "schedule": args.schedule,
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
            "produced": produced,
            "sent": sent,
            "acknowledged": acknowledged,
            "producer_ready_not_sent_max": max_ready_not_sent,
            "producer_global_batches_ahead_max": 1 if args.schedule == "pipeline" else 0,
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
        if args.schedule == "sequential":
            _control_barrier(control_group)
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

        ack = CriticCommitAck(
            run_nonce=header.run_nonce,
            cycle=header.cycle,
            logical_step=header.logical_step,
            wave=update,
            producer_global_rank=producer_rank,
            student_global_rank=rank,
            observed_generator_version=3,
            critic_before=optimizer_before,
            critic_after=optimizer_after,
            optimizer_step_before=optimizer_before,
            optimizer_step_after=optimizer_after,
            scheduler_step_before=scheduler_before,
            scheduler_step_after=scheduler_after,
        )
        ack_started = time.perf_counter()
        pending = transport.send_ack(ack, destination_global_rank=producer_rank)
        pending.wait()
        transport.reap_completed()
        ack_seconds = time.perf_counter() - ack_started
        # The end-of-update fence also orders official EMA and prevents C_{t+1}
        # from beginning before every rank has committed C_t.
        _control_barrier(control_group)
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
            "ack_seconds": ack_seconds,
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
    result["cycle"] = {
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
    parser.add_argument("--schedule", choices=("sequential", "pipeline"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-nonce", type=int, required=True)
    parser.add_argument("--trace-bundle", type=Path, required=True)
    parser.add_argument("--logical-package", type=Path, required=True)
    parser.add_argument("--expected-logical-manifest-hash", required=True)
    parser.add_argument("--transition-package", type=Path, required=True)
    parser.add_argument("--expected-transition-package-hash", required=True)
    parser.add_argument("--expected-source-transition-manifest-sha256", required=True)
    parser.add_argument("--generator-root", type=Path, required=True)
    parser.add_argument("--precast-generator-root", type=Path, required=True)
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--precast-score-root", type=Path, required=True)
    parser.add_argument("--construction-wave", type=int, default=4)
    parser.add_argument("--activation-min-bytes", type=int, default=32 * 2**20)
    parser.add_argument("--checksum", choices=("sha256", "gpu"), default="sha256")
    parser.add_argument("--timing-mode", action="store_true")
    parser.add_argument("--export-state", action="store_true")
    parser.add_argument("--correctness-evaluation", type=Path)
    parser.add_argument("--expected-correctness-sha256")
    args = parser.parse_args()
    if args.timing_mode and args.checksum != "gpu":
        parser.error("--timing-mode requires --checksum gpu")
    if args.timing_mode and args.export_state:
        parser.error("--timing-mode forbids state export")
    binding = _timing_correctness_binding(args)

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 16:
        raise ValueError(f"Candidate B requires world16, got {world_size}")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    import deepspeed

    _initialize_distributed_runtime(deepspeed)
    if rank == 0:
        if args.output_dir.exists():
            raise FileExistsError(args.output_dir)
        args.output_dir.mkdir(parents=True)
    dist.barrier()
    (
        control_group,
        pair_groups,
        ack_groups,
        _generator_group,
        critic_group,
        singleton_groups,
    ) = _create_groups(TOPOLOGY)
    role = TOPOLOGY.role(rank)
    pool_rank = (
        TOPOLOGY.generator_pool_rank(rank)
        if role == "generator"
        else TOPOLOGY.critic_pool_rank(rank)
    )
    transports: dict[int, CriticPoolP2PTransport] = {}
    if role == "generator":
        for destination in TOPOLOGY.critics_for_generator(rank):
            key = (rank, destination)
            transports[destination] = CriticPoolP2PTransport(
                group=pair_groups[key],
                device=device,
                ack_group=ack_groups[key],
                ack_device="cpu",
            )
    else:
        source, _round = TOPOLOGY.producer_for_critic(rank)
        key = (source, rank)
        transports[source] = CriticPoolP2PTransport(
            group=pair_groups[key],
            device=device,
            ack_group=ack_groups[key],
            ack_device="cpu",
        )

    output = args.output_dir / "ranks" / f"rank_{rank:02d}.json"
    result: dict[str, Any] = {
        "schema_version": "distillschedule.routeb_full5_candidate_b_rank.v1",
        "status": "RUNNING",
        "run_id": args.run_id,
        "schedule": args.schedule,
        "contract": CONTRACT.__dict__,
        "execution_scope": "critic_phase_from_post_step10a_target_layout",
        "claim_scope": "batch12_resource_throughput_pareto_not_candidate_a_speedup",
        "hostname": socket.gethostname(),
        "rank": rank,
        "local_rank": local_rank,
        "role": role,
        "pool_rank": pool_rank,
        "timing_mode": bool(args.timing_mode),
        "checksum": args.checksum,
        "stages": [_snapshot(device, "process_start")],
    }
    if binding is not None:
        result["timing_correctness_binding"] = binding
    _atomic_json(output, result)
    started = time.perf_counter()
    model = engine = residency = activation = ema = None
    try:
        logical = LogicalCheckpointPackage.load(args.logical_package)
        if logical.manifest_hash != args.expected_logical_manifest_hash:
            raise ValueError("Candidate-B logical manifest hash changed")
        transition = CandidateBTransitionPackage.load(
            args.transition_package,
            expected_transition_manifest_sha256=(
                args.expected_source_transition_manifest_sha256
            ),
            expected_critic_logical_manifest_hash=logical.manifest_hash,
            verify_generator_payload_sha256=(rank == 0),
        )
        if transition.manifest_hash != args.expected_transition_package_hash:
            raise ValueError("Candidate-B transition package hash changed")
        trace = RouteBFull5Trace(args.trace_bundle, CONTRACT)
        result["identities"] = {
            "execution_trace_bundle_hash": trace.bundle_hash,
            "routeb_trace_binding_hash": trace.binding_hash,
            "logical_manifest_hash": logical.manifest_hash,
            "transition_package_hash": transition.manifest_hash,
            "source_transition_manifest_sha256": (
                args.expected_source_transition_manifest_sha256
            ),
        }
        model, metadata, construction_restore = _construct_model(
            args=args,
            rank=rank,
            local_rank=local_rank,
            device=device,
            transition=transition,
            logical=logical,
        )
        if role == "generator":
            model, initial_sha = _initialize_generator(
                model=model,
                metadata=metadata,
                restore=construction_restore,
                pool_rank=pool_rank,
                device=device,
                result=result,
            )
            if rank == 0:
                ema = SparseShardEMA.load_pretrained(model, transition.ema_root)
                if ema.optimization_step != 10:
                    raise AssertionError("Candidate-B transition did not restore EMA10")
        else:
            engine, residency, activation = _initialize_student(
                args=args,
                model=model,
                metadata=metadata,
                construction_restore=construction_restore,
                package=logical,
                rank=rank,
                pool_rank=pool_rank,
                student_group=critic_group,
                singleton_group=singleton_groups[rank],
                device=device,
                result=result,
            )
        _load_rng(transition, rank=rank)
        _control_barrier(control_group)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        result["stages"].append(_snapshot(device, "candidate_b_target_layout_ready"))
        if args.export_state and role == "critic":
            result["initial_state_export"] = _export_optimizer_partition(
                engine=engine,
                root=args.output_dir / "state",
                phase="initial",
                rank=rank,
            )
        _atomic_json(output, result)

        _control_barrier(control_group)
        cycle_started = time.perf_counter()
        if role == "generator":
            _run_generator_cycle(
                args=args,
                topology=TOPOLOGY,
                rank=rank,
                pool_rank=pool_rank,
                device=device,
                model=model,
                initial_sha=initial_sha,
                ema=ema,
                trace=trace,
                transports=transports,
                control_group=control_group,
                result=result,
            )
        else:
            source, _round = TOPOLOGY.producer_for_critic(rank)
            _run_student_cycle(
                args=args,
                topology=TOPOLOGY,
                rank=rank,
                pool_rank=pool_rank,
                device=device,
                engine=engine,
                residency=residency,
                activation=activation,
                trace=trace,
                transport=transports[source],
                control_group=control_group,
                result=result,
            )
        _control_barrier(control_group)
        torch.cuda.synchronize(device)
        result["cycle_wall_seconds"] = time.perf_counter() - cycle_started
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
        for transport in transports.values():
            try:
                transport.flush()
            except BaseException:
                pass
        if activation is not None:
            try:
                activation.remove()
            except BaseException:
                pass
        try:
            dist.destroy_process_group()
        except BaseException:
            pass


if __name__ == "__main__":
    main()
