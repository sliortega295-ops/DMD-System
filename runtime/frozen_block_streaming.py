"""Immutable frozen-block parameter streaming for the bounded M0 B3 baseline.

This is deliberately a hand-written feasibility primitive, not the DistillSchedule
planner. Only parameters that are both inside ``model.blocks`` and immutable
(``requires_grad=False`` and not optimizer-owned) are eligible. Trainable LoRA,
optimizer state, non-block parameters, and all buffers retain the existing whole-model
placement policy.
"""
from __future__ import annotations

import contextvars
import fcntl
import hashlib
import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch
from torch.utils.checkpoint import checkpoint


EAGER_FORWARD = "EAGER_FORWARD"
ORIGINAL_FORWARD = "ORIGINAL_FORWARD"
CHECKPOINT_RECOMPUTE = "CHECKPOINT_RECOMPUTE"


class HostStatePolicy(str, Enum):
    HOST_PRIVATE = "HOST_PRIVATE"
    HOST_SHARED_IMMUTABLE = "HOST_SHARED_IMMUTABLE"
    DROP_RELOAD_FROM_MMAP = "DROP_RELOAD_FROM_MMAP"


def _storage_id(tensor: torch.Tensor) -> tuple[str, int, int]:
    storage = tensor.untyped_storage()
    return (str(tensor.device), int(storage.data_ptr()), int(storage.nbytes()))


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _resolve_streamed_block_indices(
    block_count: int,
    *,
    streamed_block_count: int,
    streamed_block_indices: Sequence[int] = (),
) -> frozenset[int]:
    """Resolve the legacy tail-count or an explicit immutable block set.

    ``streamed_block_count == 0`` retains the legacy full-B3 meaning when no
    explicit set is supplied.  An explicit set is useful for the hybrid policy:
    it lets a measured pressure set stream while every other frozen block stays
    phase-resident.  Requiring a matching non-zero count at the environment
    contract layer keeps manifests unambiguous; the lower-level primitive also
    accepts count zero so direct callers can use the explicit set alone.
    """

    block_count = int(block_count)
    streamed_block_count = int(streamed_block_count)
    if block_count <= 0:
        raise ValueError("model block count must be positive")
    if streamed_block_count < 0 or streamed_block_count > block_count:
        raise ValueError(
            "streamed_block_count must be between 0 and the model block count"
        )
    explicit = tuple(int(index) for index in streamed_block_indices)
    if len(set(explicit)) != len(explicit):
        raise ValueError("streamed block indices must be unique")
    if any(index < 0 or index >= block_count for index in explicit):
        raise ValueError("streamed block index is outside the model block range")
    if explicit:
        if streamed_block_count not in (0, len(explicit)):
            raise ValueError(
                "streamed_block_count must match the explicit streamed block indices"
            )
        return frozenset(explicit)
    selected_count = streamed_block_count or block_count
    return frozenset(range(block_count - selected_count, block_count))


def selected_frozen_block_parameter_ids(
    model: torch.nn.Module,
    *,
    streamed_block_count: int,
    streamed_block_indices: Sequence[int] = (),
    optimizer_owned_parameter_ids: set[int] | None = None,
) -> frozenset[int]:
    """Return the exact frozen parameter ids owned by one block streamer.

    Hybrid residency uses this before either residency owner is installed so
    the phase-resident mmap backing can exclude precisely the parameters that
    the fine-grained streamer will own.  The selection intentionally mirrors
    :class:`ImmutableFrozenBlockStreamer`: an explicit set wins; otherwise ``0``
    means all blocks for the legacy B3 contract and a positive count selects the
    tail blocks.
    """

    blocks = getattr(model, "blocks", None)
    if blocks is None or len(blocks) == 0:
        raise ValueError("model has no transformer blocks to stream")
    selected_indices = _resolve_streamed_block_indices(
        len(blocks),
        streamed_block_count=streamed_block_count,
        streamed_block_indices=streamed_block_indices,
    )
    optimizer_owned_ids = optimizer_owned_parameter_ids or set()
    result: set[int] = set()
    globally_seen: set[int] = set()
    for block_index, block in enumerate(blocks):
        for local_name, parameter in block.named_parameters(
            recurse=True, remove_duplicate=True
        ):
            parameter_id = id(parameter)
            if parameter_id in globally_seen:
                raise RuntimeError(
                    "cross-block parameter alias is not safe for B3 streaming: "
                    f"{local_name}"
                )
            globally_seen.add(parameter_id)
            if block_index not in selected_indices or parameter.requires_grad:
                continue
            if parameter_id in optimizer_owned_ids:
                raise RuntimeError(
                    f"optimizer-owned parameter cannot be streamed: {local_name}"
                )
            result.add(parameter_id)
    if not result:
        raise RuntimeError("no immutable frozen block parameters were found")
    return frozenset(result)


@dataclass
class StreamedParameterRecord:
    block_index: int
    name: str
    parameter: torch.nn.Parameter
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    nbytes: int
    checkpoint_identity: str
    host_policy: HostStatePolicy
    host_tensor: torch.Tensor | None
    backing_path: Path | None = None
    resident_storage_id: tuple[str, int, int] | None = None

    @property
    def resident(self) -> bool:
        return self.parameter.device.type == "cuda" and self.parameter.numel() != 0


class ImmutableFrozenBlockStreamer:
    """Synchronously streams immutable block weights with a fixed look-ahead window."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        logical_component: str,
        device: torch.device | str,
        window_size: int,
        streamed_block_count: int = 0,
        streamed_block_indices: Sequence[int] = (),
        host_policy: HostStatePolicy | str = HostStatePolicy.HOST_PRIVATE,
        shared_immutable_block_indices: Sequence[int] = (),
        optimizer_owned_parameter_ids: set[int] | None = None,
        checkpoint_identity: str,
        event_path: Path,
        mmap_cache_root: Path | None = None,
        pin_host_state: bool = False,
        detailed_events: bool = True,
        memory_snapshot_events: bool = False,
    ) -> None:
        if int(window_size) not in (1, 2, 4):
            raise ValueError("B3 streaming window must be one of 1, 2, or 4")
        self.model = model
        self.logical_component = logical_component
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("B3 block streamer requires a CUDA destination")
        self.window_size = int(window_size)
        self.host_policy = HostStatePolicy(host_policy)
        self.pin_host_state = bool(pin_host_state)
        if self.pin_host_state and self.host_policy is not HostStatePolicy.HOST_PRIVATE:
            raise ValueError("pinned frozen host state requires HOST_PRIVATE ownership")
        self.optimizer_owned_parameter_ids = optimizer_owned_parameter_ids or set()
        self.checkpoint_identity = str(checkpoint_identity)
        if not self.checkpoint_identity:
            raise ValueError("immutable checkpoint identity must be non-empty")
        self.event_path = Path(event_path)
        self.event_path.parent.mkdir(parents=True, exist_ok=True)
        if self.event_path.exists():
            raise FileExistsError(f"refusing to overwrite block-stream trace: {self.event_path}")
        self.mmap_cache_root = Path(mmap_cache_root) if mmap_cache_root else None
        if self.host_policy is not HostStatePolicy.HOST_PRIVATE and self.mmap_cache_root is None:
            raise ValueError(f"{self.host_policy.value} requires mmap_cache_root")

        blocks = getattr(model, "blocks", None)
        if blocks is None or len(blocks) == 0:
            raise ValueError("model has no transformer blocks to stream")
        self.blocks: Sequence[torch.nn.Module] = blocks
        raw_shared_indices = tuple(int(index) for index in shared_immutable_block_indices)
        if len(set(raw_shared_indices)) != len(raw_shared_indices):
            raise ValueError("shared immutable block indices must be unique")
        if any(index < 0 or index >= len(self.blocks) for index in raw_shared_indices):
            raise ValueError("shared immutable block index is outside the model block range")
        if raw_shared_indices and self.host_policy is not HostStatePolicy.HOST_PRIVATE:
            raise ValueError(
                "mixed shared immutable blocks require HOST_PRIVATE as the base policy"
            )
        if raw_shared_indices and self.mmap_cache_root is None:
            raise ValueError("mixed shared immutable blocks require mmap_cache_root")
        self.shared_immutable_block_indices = frozenset(raw_shared_indices)
        self.streamed_block_indices = _resolve_streamed_block_indices(
            len(self.blocks),
            streamed_block_count=streamed_block_count,
            streamed_block_indices=streamed_block_indices,
        )
        if not self.shared_immutable_block_indices.issubset(self.streamed_block_indices):
            raise ValueError("shared immutable blocks must also be streamed blocks")
        self.detailed_events = bool(detailed_events)
        self.memory_snapshot_events = bool(memory_snapshot_events)
        self.checkpoint_boundary_offload_min_bytes = int(
            os.environ.get(
                "DISTILLSCHEDULE_CHECKPOINT_BOUNDARY_OFFLOAD_MIN_BYTES", "0"
            )
        )
        if self.checkpoint_boundary_offload_min_bytes < 0:
            raise ValueError("checkpoint boundary offload threshold must be non-negative")
        self._generation = contextvars.ContextVar(
            f"distillschedule_stream_generation_{id(self)}", default=EAGER_FORWARD
        )
        self._checkpoint_boundary_offload_active = contextvars.ContextVar(
            f"distillschedule_checkpoint_boundary_offload_{id(self)}", default=False
        )
        self._records_by_block: list[list[StreamedParameterRecord]] = [
            [] for _ in range(len(self.blocks))
        ]
        self._record_by_parameter_id: dict[int, StreamedParameterRecord] = {}
        self._resident_storage_owners: dict[tuple[str, int, int], dict[str, Any]] = {}
        self._hook_handles: list[Any] = []
        # A recompute forward can materialize a frozen block long before the
        # autograd engine has finished that block's backward.  Count recompute
        # invocations rather than releasing at recompute-context exit: the
        # latter is only the end of the forward replay and was the source of
        # the world16 resident-byte accumulation.
        self._pending_recompute_backward_releases: dict[int, int] = {}
        self._release_on_recompute_exit: set[int] = set()
        self._event_order = 0
        self._original_checkpoint_function = getattr(model, "_gradient_checkpointing_func", None)
        self.stats: dict[str, int | float] = {
            "captured_parameter_count": 0,
            "captured_bytes": 0,
            "private_host_bytes": 0,
            "shared_immutable_host_bytes": 0,
            "backing_only_bytes": 0,
            "streamed_block_count": len(self.streamed_block_indices),
            "shared_immutable_block_count": len(self.shared_immutable_block_indices),
            "materialize_count": 0,
            "materialized_bytes": 0,
            "release_count": 0,
            "released_bytes": 0,
            "peak_resident_bytes": 0,
            "resident_bytes": 0,
            "pinned_host_bytes": 0,
            "h2d_wall_seconds": 0.0,
            "checkpoint_boundary_d2h_count": 0,
            "checkpoint_boundary_d2h_bytes": 0,
            "checkpoint_boundary_d2h_wall_seconds": 0.0,
            "checkpoint_boundary_h2d_count": 0,
            "checkpoint_boundary_h2d_bytes": 0,
            "checkpoint_boundary_h2d_wall_seconds": 0.0,
        }
        self._capture_records()
        self._install_hooks()
        self._install_checkpoint_context()
        self._write_event(
            "streamer_installed",
            block_count=len(self.blocks),
            streamed_block_count=len(self.streamed_block_indices),
            streamed_block_indices=sorted(self.streamed_block_indices),
            window_size=self.window_size,
            host_policy=self.host_policy.value,
            shared_immutable_block_indices=sorted(self.shared_immutable_block_indices),
            pin_host_state=self.pin_host_state,
            checkpoint_identity=self.checkpoint_identity,
            checkpoint_mode="nonreentrant_tensor_flattened",
            checkpoint_boundary_offload_min_bytes=(
                self.checkpoint_boundary_offload_min_bytes
            ),
            memory_snapshot_events=self.memory_snapshot_events,
            captured_parameter_count=self.stats["captured_parameter_count"],
            captured_bytes=self.stats["captured_bytes"],
        )

    @property
    def managed_parameter_ids(self) -> frozenset[int]:
        """Parameter ids exclusively owned by this streamer."""

        return frozenset(self._record_by_parameter_id)

    def _write_event(self, event: str, **fields: Any) -> None:
        if not self.detailed_events and event not in {
            "streamer_installed",
            "streamer_removed",
        }:
            return
        self._event_order += 1
        record = {
            "event_order": self._event_order,
            "time_unix": time.time(),
            "event": event,
            "logical_component": self.logical_component,
            **fields,
        }
        with self.event_path.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def _cuda_memory_fields(self, prefix: str) -> dict[str, int]:
        """Capture scalar allocator/driver state without synchronizing CUDA.

        This is intentionally opt-in diagnostic instrumentation.  It does not
        enter the final timing contract, and avoiding a synchronize keeps it
        from becoming an accidental scheduling barrier.
        """

        if not self.memory_snapshot_events:
            return {}
        with torch.cuda.device(self.device):
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            return {
                f"{prefix}_allocated_bytes": int(
                    torch.cuda.memory_allocated(self.device)
                ),
                f"{prefix}_reserved_bytes": int(
                    torch.cuda.memory_reserved(self.device)
                ),
                f"{prefix}_max_allocated_bytes": int(
                    torch.cuda.max_memory_allocated(self.device)
                ),
                f"{prefix}_max_reserved_bytes": int(
                    torch.cuda.max_memory_reserved(self.device)
                ),
                f"{prefix}_driver_free_bytes": int(free_bytes),
                f"{prefix}_driver_total_bytes": int(total_bytes),
            }

    def _copy_checkpoint_boundary(
        self,
        tensor: torch.Tensor,
        target: torch.device | str,
        *,
        direction: str,
        generation: str,
    ) -> torch.Tensor:
        target = torch.device(target)
        if tensor.device == target:
            return tensor
        nbytes = _tensor_bytes(tensor)
        nvtx_enabled = (
            os.environ.get("DISTILLSCHEDULE_NSYS_ANNOTATE", "0") == "1"
            and torch.cuda.is_available()
        )
        if nvtx_enabled:
            rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
            torch.cuda.nvtx.range_push(
                "helios_transfer"
                f"|rank={rank}"
                f"|component={self.logical_component}"
                "|kind=checkpoint_boundary"
                f"|direction={direction}"
                f"|generation={generation}"
                f"|bytes={nbytes}"
            )
        started = time.perf_counter()
        try:
            copied = tensor.to(target, non_blocking=False)
        finally:
            elapsed = time.perf_counter() - started
            if nvtx_enabled:
                torch.cuda.nvtx.range_pop()
        prefix = f"checkpoint_boundary_{direction}"
        self.stats[f"{prefix}_count"] += 1
        self.stats[f"{prefix}_bytes"] += nbytes
        self.stats[f"{prefix}_wall_seconds"] += elapsed
        self._write_event(
            "checkpoint_boundary_copy",
            direction=direction,
            generation=generation,
            bytes=nbytes,
            elapsed_seconds=elapsed,
            source_device=str(tensor.device),
            target_device=str(target),
        )
        return copied

    def _cache_paths(self, name: str) -> tuple[Path, Path, Path]:
        assert self.mmap_cache_root is not None
        identity_hash = hashlib.sha256(self.checkpoint_identity.encode()).hexdigest()[:16]
        name_hash = hashlib.sha256(name.encode()).hexdigest()[:20]
        root = self.mmap_cache_root / identity_hash / self.logical_component
        root.mkdir(parents=True, exist_ok=True)
        return (
            root / f"{name_hash}.bin",
            root / f"{name_hash}.json",
            root / f"{name_hash}.lock",
        )

    def _create_or_validate_mmap(
        self,
        *,
        name: str,
        source: torch.Tensor,
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> Path:
        if not source.is_contiguous():
            raise ValueError(f"mmap host policy requires contiguous parameter: {name}")
        data_path, metadata_path, lock_path = self._cache_paths(name)
        metadata = {
            "schema_version": 1,
            "checkpoint_identity": self.checkpoint_identity,
            "logical_component": self.logical_component,
            "parameter_name": name,
            "shape": list(shape),
            "dtype": _dtype_name(dtype),
            "numel": int(source.numel()),
            "bytes": _tensor_bytes(source),
        }

        # B3 is installed by every rank on a node. Hold a per-parameter
        # filesystem lock across validation and the two-file publication so a
        # reader cannot observe the .bin after it is published but before the
        # matching .json exists. A unique run cache remains fail-closed if a
        # prior attempt left a partial pair behind.
        with lock_path.open("a+b") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            try:
                if data_path.exists() or metadata_path.exists():
                    if not data_path.exists() or not metadata_path.exists():
                        raise RuntimeError(f"partial immutable mmap cache for {name}")
                    observed = json.loads(metadata_path.read_text())
                    if observed != metadata or data_path.stat().st_size != metadata["bytes"]:
                        raise RuntimeError(f"immutable mmap cache identity mismatch for {name}")
                    return data_path

                # Atomic per-parameter publication. The lock serializes local
                # ranks while temporary files keep readers from seeing partial
                # contents.
                token = f"{os.getpid()}-{time.time_ns()}"
                temp_data = data_path.with_suffix(f".bin.{token}.tmp")
                temp_meta = metadata_path.with_suffix(f".json.{token}.tmp")
                mapped = torch.from_file(
                    str(temp_data), shared=True, size=int(source.numel()), dtype=dtype
                ).view(shape)
                mapped.copy_(source)
                del mapped
                with temp_data.open("rb") as handle:
                    os.fsync(handle.fileno())
                temp_meta.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
                with temp_meta.open("rb") as handle:
                    os.fsync(handle.fileno())
                os.replace(temp_data, data_path)
                os.replace(temp_meta, metadata_path)
                return data_path
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    def _mapped_tensor(self, record: StreamedParameterRecord) -> torch.Tensor:
        if record.backing_path is None:
            raise RuntimeError(f"record has no mmap backing: {record.name}")
        return torch.from_file(
            str(record.backing_path),
            shared=False,
            size=int(torch.tensor(record.shape).prod().item()),
            dtype=record.dtype,
        ).view(record.shape)

    def _capture_records(self) -> None:
        globally_seen: set[int] = set()
        for block_index, block in enumerate(self.blocks):
            for local_name, parameter in block.named_parameters(recurse=True, remove_duplicate=True):
                parameter_id = id(parameter)
                if parameter_id in globally_seen:
                    raise RuntimeError(
                        f"cross-block parameter alias is not safe for B3 streaming: {local_name}"
                    )
                globally_seen.add(parameter_id)
                if block_index not in self.streamed_block_indices:
                    continue
                if parameter.requires_grad:
                    continue
                if parameter_id in self.optimizer_owned_parameter_ids:
                    raise RuntimeError(f"optimizer-owned parameter cannot be streamed: {local_name}")
                if parameter.layout != torch.strided:
                    raise ValueError(f"non-strided frozen parameter cannot be streamed: {local_name}")
                shape = tuple(int(x) for x in parameter.shape)
                stride = tuple(int(x) for x in parameter.stride())
                source = parameter.detach()
                if source.device.type != "cpu":
                    source = source.to("cpu", non_blocking=False)
                full_name = f"blocks.{block_index}.{local_name}"
                record_host_policy = (
                    HostStatePolicy.HOST_SHARED_IMMUTABLE
                    if block_index in self.shared_immutable_block_indices
                    else self.host_policy
                )
                backing_path: Path | None = None
                if record_host_policy is HostStatePolicy.HOST_PRIVATE:
                    if self.pin_host_state:
                        # Allocate the immutable owner directly in page-locked
                        # storage. ``empty_strided`` preserves parameter metadata
                        # rather than silently making a non-contiguous parameter
                        # contiguous. This is a synchronous M1 primitive; async
                        # streams/events remain a separate policy.
                        host_tensor = torch.empty_strided(
                            shape,
                            stride,
                            dtype=parameter.dtype,
                            device="cpu",
                            pin_memory=True,
                        )
                        host_tensor.copy_(source)
                    else:
                        host_tensor = source
                else:
                    backing_path = self._create_or_validate_mmap(
                        name=full_name,
                        source=source,
                        shape=shape,
                        dtype=parameter.dtype,
                    )
                    host_tensor = self._mapped_tensor(
                        StreamedParameterRecord(
                            block_index,
                            full_name,
                            parameter,
                            shape,
                            stride,
                            parameter.dtype,
                            _tensor_bytes(parameter),
                            self.checkpoint_identity,
                            record_host_policy,
                            None,
                            backing_path,
                        )
                    )
                record = StreamedParameterRecord(
                    block_index=block_index,
                    name=full_name,
                    parameter=parameter,
                    shape=shape,
                    stride=stride,
                    dtype=parameter.dtype,
                    nbytes=_tensor_bytes(parameter),
                    checkpoint_identity=self.checkpoint_identity,
                    host_policy=record_host_policy,
                    host_tensor=host_tensor,
                    backing_path=backing_path,
                )
                self._records_by_block[block_index].append(record)
                self._record_by_parameter_id[parameter_id] = record
                self.stats["captured_parameter_count"] += 1
                self.stats["captured_bytes"] += record.nbytes
                if record_host_policy is HostStatePolicy.HOST_PRIVATE:
                    self.stats["private_host_bytes"] += record.nbytes
                elif record_host_policy is HostStatePolicy.HOST_SHARED_IMMUTABLE:
                    self.stats["shared_immutable_host_bytes"] += record.nbytes
                else:
                    self.stats["backing_only_bytes"] += record.nbytes
                if host_tensor is not None and host_tensor.is_pinned():
                    self.stats["pinned_host_bytes"] += record.nbytes
                if record_host_policy is HostStatePolicy.DROP_RELOAD_FROM_MMAP:
                    parameter.data = torch.empty(0, device="cpu", dtype=record.dtype)
                    record.host_tensor = None
                else:
                    assert host_tensor is not None
                    parameter.data = host_tensor
        if not self._record_by_parameter_id:
            raise RuntimeError("no immutable frozen block parameters were found")

    def _host_source(self, record: StreamedParameterRecord) -> torch.Tensor:
        if record.host_policy is HostStatePolicy.DROP_RELOAD_FROM_MMAP:
            return self._mapped_tensor(record)
        if record.host_tensor is None:
            raise RuntimeError(f"missing host tensor for {record.name}")
        return record.host_tensor

    def dynamic_excluded_storage_owner(
        self, storage_id: tuple[str, int, int]
    ) -> Mapping[str, str] | None:
        return self._resident_storage_owners.get(storage_id)

    @property
    def resident_bytes(self) -> int:
        return int(self.stats["resident_bytes"])

    def _materialize_block(self, block_index: int, generation: str) -> None:
        records = self._records_by_block[block_index]
        if not records or all(record.resident for record in records):
            return
        bytes_moved = 0
        parameter_count = 0
        memory_before = self._cuda_memory_fields("cuda_before_materialize")
        started = time.perf_counter()
        for record in records:
            if record.resident:
                continue
            source = self._host_source(record)
            materialized = source.to(self.device, non_blocking=False)
            if tuple(materialized.shape) != record.shape or materialized.dtype != record.dtype:
                raise RuntimeError(f"materialized immutable parameter changed metadata: {record.name}")
            record.parameter.data = materialized
            sid = _storage_id(record.parameter)
            record.resident_storage_id = sid
            self._resident_storage_owners[sid] = {
                "kind": "streamed_parameter",
                "name": record.name,
            }
            bytes_moved += record.nbytes
            parameter_count += 1
        elapsed = time.perf_counter() - started
        if bytes_moved:
            memory_after = self._cuda_memory_fields("cuda_after_materialize")
            self.stats["materialize_count"] += 1
            self.stats["materialized_bytes"] += bytes_moved
            self.stats["h2d_wall_seconds"] += elapsed
            self.stats["resident_bytes"] += bytes_moved
            self.stats["peak_resident_bytes"] = max(
                int(self.stats["peak_resident_bytes"]), int(self.stats["resident_bytes"])
            )
            self._write_event(
                "materialize_block",
                block_index=block_index,
                generation=generation,
                parameter_count=parameter_count,
                bytes=bytes_moved,
                elapsed_seconds=elapsed,
                resident_bytes=self.stats["resident_bytes"],
                **memory_before,
                **memory_after,
            )

    def _release_block(self, block_index: int, generation: str, reason: str) -> None:
        records = self._records_by_block[block_index]
        if not records or not any(record.resident for record in records):
            return
        bytes_released = 0
        parameter_count = 0
        memory_before = self._cuda_memory_fields("cuda_before_release")
        for record in records:
            if not record.resident:
                continue
            old_sid = record.resident_storage_id
            if old_sid is not None:
                self._resident_storage_owners.pop(old_sid, None)
            if record.host_policy is HostStatePolicy.DROP_RELOAD_FROM_MMAP:
                record.parameter.data = torch.empty(0, device="cpu", dtype=record.dtype)
            else:
                if record.host_tensor is None:
                    raise RuntimeError(f"missing host tensor for {record.name}")
                record.parameter.data = record.host_tensor
            record.resident_storage_id = None
            bytes_released += record.nbytes
            parameter_count += 1
        if bytes_released:
            memory_after = self._cuda_memory_fields("cuda_after_release")
            self.stats["release_count"] += 1
            self.stats["released_bytes"] += bytes_released
            self.stats["resident_bytes"] -= bytes_released
            if int(self.stats["resident_bytes"]) < 0:
                raise RuntimeError("negative streamed resident byte count")
            self._write_event(
                "release_block",
                block_index=block_index,
                generation=generation,
                reason=reason,
                parameter_count=parameter_count,
                bytes=bytes_released,
                resident_bytes=self.stats["resident_bytes"],
                **memory_before,
                **memory_after,
            )

    def _window_indices(self, block_index: int, generation: str) -> range:
        if generation == CHECKPOINT_RECOMPUTE:
            return range(block_index, max(-1, block_index - self.window_size), -1)
        return range(block_index, min(len(self.blocks), block_index + self.window_size))

    def _before_block(self, block_index: int, inputs: tuple[Any, ...]) -> None:
        generation = self._generation.get()
        for candidate in self._window_indices(block_index, generation):
            self._materialize_block(candidate, generation)
        if generation == CHECKPOINT_RECOMPUTE and self._records_by_block[block_index]:
            # The full-backward hook below fires after this block's input and
            # parameter gradients are complete.  It is intentionally attached
            # to the module rather than to a guessed subset of checkpoint
            # boundary tensors: Helios blocks have several optional inputs,
            # some of which can require grad but be unused on a particular
            # route, so an ``all`` boundary hook can wait forever.
            self._pending_recompute_backward_releases[block_index] = (
                self._pending_recompute_backward_releases.get(block_index, 0) + 1
            )
            if not self._contains_differentiable_input(inputs):
                # If no checkpoint boundary can receive an upstream gradient,
                # this block's frozen weights are not needed by backward.  The
                # full-backward hook is still the normal path; this fallback
                # covers the first block of a chain where PyTorch may fire the
                # module hook for the original graph before replaying it.
                self._release_on_recompute_exit.add(block_index)

    @staticmethod
    def _contains_differentiable_input(value: Any) -> bool:
        if isinstance(value, torch.Tensor):
            return bool(value.requires_grad)
        if isinstance(value, Mapping):
            return any(
                ImmutableFrozenBlockStreamer._contains_differentiable_input(child)
                for child in value.values()
            )
        if isinstance(value, (tuple, list)):
            return any(
                ImmutableFrozenBlockStreamer._contains_differentiable_input(child)
                for child in value
            )
        segments = getattr(value, "segments", None)
        if isinstance(segments, (tuple, list)):
            return ImmutableFrozenBlockStreamer._contains_differentiable_input(segments)
        return False

    def _consume_pending_recompute_release(self, block_index: int, reason: str) -> None:
        pending = self._pending_recompute_backward_releases.get(block_index, 0)
        if pending <= 0:
            return
        if pending == 1:
            del self._pending_recompute_backward_releases[block_index]
            self._release_block(block_index, CHECKPOINT_RECOMPUTE, reason)
        else:
            self._pending_recompute_backward_releases[block_index] = pending - 1

    def _after_block(self, block_index: int) -> None:
        generation = self._generation.get()
        if generation != CHECKPOINT_RECOMPUTE:
            self._release_block(block_index, generation, "forward_compute_enqueued")

    def _after_block_backward(self, block_index: int) -> None:
        pending = self._pending_recompute_backward_releases.get(block_index, 0)
        if pending <= 0:
            return
        self._release_on_recompute_exit.discard(block_index)
        self._consume_pending_recompute_release(
            block_index, "checkpoint_block_backward_complete"
        )

    def _install_hooks(self) -> None:
        for block_index, block in enumerate(self.blocks):
            pre = block.register_forward_pre_hook(
                lambda _module, inputs, i=block_index: self._before_block(i, inputs)
            )
            post = block.register_forward_hook(
                lambda _module, _inputs, _output, i=block_index: self._after_block(i),
                always_call=True,
            )
            backward = block.register_full_backward_hook(
                lambda _module, _grad_input, _grad_output, i=block_index: (
                    self._after_block_backward(i)
                )
            )
            self._hook_handles.extend((pre, post, backward))

        def root_cleanup(_module, _inputs, _output):
            # Handles branch exits and exceptions; normal paths should already be empty.
            self.release_all(reason="root_forward_exit")

        self._hook_handles.append(self.model.register_forward_hook(root_cleanup, always_call=True))

    @contextmanager
    def _generation_context(
        self, generation: str, *, boundary_offload_active: bool = False
    ) -> Iterator[None]:
        token = self._generation.set(generation)
        offload_token = self._checkpoint_boundary_offload_active.set(
            boundary_offload_active
        )
        try:
            yield
        finally:
            if generation == CHECKPOINT_RECOMPUTE and self._release_on_recompute_exit:
                release = tuple(self._release_on_recompute_exit)
                self._release_on_recompute_exit.clear()
                for block_index in release:
                    self._consume_pending_recompute_release(
                        block_index, "recompute_has_no_differentiable_boundary"
                    )
            self._checkpoint_boundary_offload_active.reset(offload_token)
            self._generation.reset(token)

    def _install_checkpoint_context(self) -> None:
        def encode_checkpoint_value(
            value: Any, flat_tensors: list[torch.Tensor]
        ) -> tuple[Any, ...]:
            if isinstance(value, torch.Tensor):
                index = len(flat_tensors)
                flat_tensors.append(value)
                return ("tensor", index)
            segments = getattr(value, "segments", None)
            if isinstance(segments, (tuple, list)):
                encoded_segments = tuple(
                    (encode_checkpoint_value(tensor, flat_tensors), int(length))
                    for tensor, length in segments
                )
                return ("segment_carrier", type(value), encoded_segments)
            if isinstance(value, tuple):
                return (
                    "tuple",
                    tuple(encode_checkpoint_value(child, flat_tensors) for child in value),
                )
            if isinstance(value, list):
                return (
                    "list",
                    tuple(encode_checkpoint_value(child, flat_tensors) for child in value),
                )
            if isinstance(value, Mapping):
                return (
                    "mapping",
                    tuple(
                        (key, encode_checkpoint_value(child, flat_tensors))
                        for key, child in value.items()
                    ),
                )
            return ("constant", value)

        def decode_checkpoint_value(
            specification: tuple[Any, ...],
            flat_tensors: Sequence[torch.Tensor],
        ) -> Any:
            kind = specification[0]
            if kind == "tensor":
                return flat_tensors[specification[1]]
            if kind == "segment_carrier":
                carrier_type, segments = specification[1], specification[2]
                return carrier_type(
                    [
                        (decode_checkpoint_value(tensor_spec, flat_tensors), length)
                        for tensor_spec, length in segments
                    ]
                )
            if kind == "tuple":
                return tuple(
                    decode_checkpoint_value(child, flat_tensors)
                    for child in specification[1]
                )
            if kind == "list":
                return [
                    decode_checkpoint_value(child, flat_tensors)
                    for child in specification[1]
                ]
            if kind == "mapping":
                return {
                    key: decode_checkpoint_value(child, flat_tensors)
                    for key, child in specification[1]
                }
            if kind == "constant":
                return specification[1]
            raise RuntimeError(f"unknown checkpoint argument specification: {kind}")

        def checkpoint_with_streaming(module: torch.nn.Module, *args: Any):
            flat_tensors: list[torch.Tensor] = []
            encoded_args = tuple(
                encode_checkpoint_value(argument, flat_tensors) for argument in args
            )
            device_tensor = next(
                (tensor for tensor in flat_tensors if tensor.device.type == "cuda"),
                None,
            )
            if device_tensor is None:
                raise RuntimeError("streamed checkpoint has no CUDA tensor input")
            original_devices = tuple(tensor.device for tensor in flat_tensors)
            checkpoint_tensors = tuple(
                self._copy_checkpoint_boundary(
                    tensor,
                    "cpu",
                    direction="d2h",
                    generation=ORIGINAL_FORWARD,
                )
                if (
                    self.checkpoint_boundary_offload_min_bytes > 0
                    and tensor.device.type == "cuda"
                    and _tensor_bytes(tensor)
                    >= self.checkpoint_boundary_offload_min_bytes
                )
                else tensor
                for tensor in flat_tensors
            )
            boundary_offload_active = any(
                tensor.device != original_device
                for tensor, original_device in zip(
                    checkpoint_tensors, original_devices, strict=True
                )
            )

            def run_nonreentrant(*checkpoint_tensors: torch.Tensor) -> torch.Tensor:
                restored_tensors = tuple(
                    self._copy_checkpoint_boundary(
                        tensor,
                        original_device,
                        direction="h2d",
                        generation=self._generation.get(),
                    )
                    if tensor.device != original_device
                    else tensor
                    for tensor, original_device in zip(
                        checkpoint_tensors, original_devices, strict=True
                    )
                )
                reconstructed_args = tuple(
                    decode_checkpoint_value(specification, restored_tensors)
                    for specification in encoded_args
                )
                output = module(*reconstructed_args)
                if not isinstance(output, torch.Tensor):
                    raise RuntimeError(
                        "B3 non-reentrant streamed checkpoint requires a tensor output"
                    )
                return output

            def checkpoint_contexts():
                return (
                    self._generation_context(
                        ORIGINAL_FORWARD,
                        boundary_offload_active=boundary_offload_active,
                    ),
                    self._generation_context(
                        CHECKPOINT_RECOMPUTE,
                        boundary_offload_active=boundary_offload_active,
                    ),
                )

            # DeepSpeed ZeRO-1/2 installs one reduction hook per parameter and
            # cannot accept the nested backward invocations created when the same
            # checkpointed backbone is reused several times in one rollout.  The
            # official Diffusers checkpoint path is non-reentrant; keep the
            # tensor-flattening adapter for Helios carriers, but preserve that
            # single autograd graph so every parameter is accumulated/reduced once.
            return checkpoint(
                run_nonreentrant,
                *checkpoint_tensors,
                use_reentrant=False,
                context_fn=checkpoint_contexts,
                preserve_rng_state=True,
            )

        self.model._gradient_checkpointing_func = checkpoint_with_streaming

    def release_all(self, *, reason: str = "explicit_release") -> None:
        for block_index in range(len(self.blocks)):
            self._release_block(block_index, self._generation.get(), reason)
        self._pending_recompute_backward_releases.clear()
        self._release_on_recompute_exit.clear()
        if self._resident_storage_owners:
            raise RuntimeError("resident streamed storage registry leaked after release_all")

    def move_non_streamed_state(
        self,
        target_device: torch.device | str,
        *,
        non_blocking: bool,
    ) -> None:
        target = torch.device(target_device)
        if target.type == "cpu":
            self.release_all(reason="whole_model_move_to_cpu")
        for parameter in self.model.parameters():
            if id(parameter) in self._record_by_parameter_id:
                continue
            if parameter.device != target:
                parameter.data = parameter.data.to(target, non_blocking=non_blocking)
            if parameter.grad is not None and parameter.grad.device != target:
                parameter.grad.data = parameter.grad.data.to(target, non_blocking=non_blocking)
        for buffer in self.model.buffers():
            if buffer.device != target:
                buffer.data = buffer.data.to(target, non_blocking=non_blocking)
        self._write_event(
            "move_non_streamed_state",
            target_device=str(target),
            non_blocking=bool(non_blocking),
        )

    def assert_idle(self) -> None:
        if int(self.stats["resident_bytes"]) != 0 or self._resident_storage_owners:
            raise RuntimeError("streamer is not idle")
        if self._pending_recompute_backward_releases:
            raise RuntimeError(
                "streamer retained pending checkpoint block backward releases: "
                f"{self._pending_recompute_backward_releases}"
            )
        for record in self._record_by_parameter_id.values():
            if record.resident:
                raise RuntimeError(f"streamed parameter remained resident: {record.name}")
            if record.parameter.requires_grad:
                raise RuntimeError(f"streamed parameter became trainable: {record.name}")
            if id(record.parameter) in self.optimizer_owned_parameter_ids:
                raise RuntimeError(f"streamed parameter became optimizer-owned: {record.name}")

    def remove(self) -> None:
        self.release_all(reason="remove")
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        if self._original_checkpoint_function is not None:
            self.model._gradient_checkpointing_func = self._original_checkpoint_function
        self.assert_idle()
        self._write_event("streamer_removed", stats=self.stats)


def install_frozen_block_streamer(
    model: torch.nn.Module,
    **kwargs: Any,
) -> ImmutableFrozenBlockStreamer:
    return ImmutableFrozenBlockStreamer(model, **kwargs)
