"""Node-shared immutable backing for phase-level whole-model swapping.

This module deliberately does *not* install forward hooks or stream blocks. It
keeps the existing phase-level CPU/GPU schedule while replacing rank-private
CPU copies of immutable parameters with one file-backed copy per node.

Only parameters that are frozen and not optimizer-owned are eligible. Trainable
LoRA/adapters, their gradients and optimizer state, plus every buffer, retain the
existing rank-private placement policy.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch


CACHE_SCHEMA = "distillschedule.node_shared_immutable_model.v1"


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _safe_node_identity(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-") or "unknown"
    digest = hashlib.sha256(value.encode()).hexdigest()[:8]
    return f"{cleaned[:64]}-{digest}"


def resolve_node_cache_root(
    cache_root: str | Path,
    *,
    node_identity: str | None = None,
) -> Path:
    """Return a node-qualified directory even when cache_root is shared."""

    identity = node_identity or socket.gethostname()
    return Path(cache_root) / f"node-{_safe_node_identity(identity)}"


@dataclass(frozen=True)
class ImmutableParameterSpec:
    name: str
    parameter: torch.nn.Parameter
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    numel: int
    nbytes: int


@dataclass
class ImmutableParameterRecord:
    spec: ImmutableParameterSpec
    backing_path: Path
    host_tensor: torch.Tensor


def _eligible_parameters(
    model: torch.nn.Module,
    optimizer_owned_parameter_ids: set[int],
    externally_managed_parameter_ids: set[int] | None = None,
) -> tuple[ImmutableParameterSpec, ...]:
    externally_managed_ids = externally_managed_parameter_ids or set()
    result: list[ImmutableParameterSpec] = []
    storage_owners: dict[tuple[str, int, int], str] = {}
    for name, parameter in model.named_parameters(recurse=True, remove_duplicate=True):
        if (
            parameter.requires_grad
            or id(parameter) in optimizer_owned_parameter_ids
            or id(parameter) in externally_managed_ids
        ):
            continue
        if parameter.grad is not None:
            raise RuntimeError(f"frozen parameter unexpectedly has a gradient: {name}")
        if parameter.layout != torch.strided:
            raise ValueError(f"non-strided immutable parameter is unsupported: {name}")
        if not parameter.is_contiguous():
            raise ValueError(f"immutable mmap backing requires contiguous parameter: {name}")
        storage = parameter.untyped_storage()
        storage_id = (str(parameter.device), int(storage.data_ptr()), int(storage.nbytes()))
        prior = storage_owners.get(storage_id)
        if prior is not None:
            raise RuntimeError(
                "distinct immutable parameters share one storage; "
                f"cannot safely rebind {prior!r} and {name!r}"
            )
        storage_owners[storage_id] = name
        result.append(
            ImmutableParameterSpec(
                name=name,
                parameter=parameter,
                shape=tuple(int(value) for value in parameter.shape),
                stride=tuple(int(value) for value in parameter.stride()),
                dtype=parameter.dtype,
                numel=int(parameter.numel()),
                nbytes=_tensor_bytes(parameter),
            )
        )
    if not result:
        raise RuntimeError("no immutable non-optimizer parameters were found")
    return tuple(result)


def _identity_root(
    cache_root: Path,
    *,
    checkpoint_identity: str,
    logical_component: str,
) -> Path:
    if not checkpoint_identity:
        raise ValueError("immutable checkpoint identity must be non-empty")
    if not logical_component:
        raise ValueError("logical component must be non-empty")
    identity_hash = hashlib.sha256(checkpoint_identity.encode()).hexdigest()[:20]
    return cache_root / identity_hash / logical_component


def _record_metadata(
    spec: ImmutableParameterSpec,
    *,
    checkpoint_identity: str,
    logical_component: str,
) -> dict[str, Any]:
    return {
        "schema_version": CACHE_SCHEMA,
        "checkpoint_identity": checkpoint_identity,
        "logical_component": logical_component,
        "parameter_name": spec.name,
        "shape": list(spec.shape),
        "stride": list(spec.stride),
        "dtype": _dtype_name(spec.dtype),
        "numel": spec.numel,
        "bytes": spec.nbytes,
    }


def _record_paths(root: Path, name: str) -> tuple[Path, Path, Path]:
    name_hash = hashlib.sha256(name.encode()).hexdigest()[:24]
    return (
        root / f"{name_hash}.bin",
        root / f"{name_hash}.json",
        root / f"{name_hash}.lock",
    )


def _validate_record_files(
    data_path: Path,
    metadata_path: Path,
    expected: dict[str, Any],
) -> None:
    if not data_path.exists() or not metadata_path.exists():
        raise RuntimeError(f"partial immutable mmap cache for {expected['parameter_name']}")
    observed = json.loads(metadata_path.read_text())
    if observed != expected:
        raise RuntimeError(
            f"immutable mmap metadata mismatch for {expected['parameter_name']}"
        )
    if data_path.stat().st_size != int(expected["bytes"]):
        raise RuntimeError(
            f"immutable mmap byte-size mismatch for {expected['parameter_name']}"
        )


def _publish_record(
    spec: ImmutableParameterSpec,
    *,
    root: Path,
    checkpoint_identity: str,
    logical_component: str,
) -> bool:
    """Publish one parameter atomically; return True when reusing a cache file."""

    expected = _record_metadata(
        spec,
        checkpoint_identity=checkpoint_identity,
        logical_component=logical_component,
    )
    data_path, metadata_path, lock_path = _record_paths(root, spec.name)
    root.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            if data_path.exists() or metadata_path.exists():
                _validate_record_files(data_path, metadata_path, expected)
                return True

            token = f"{os.getpid()}-{time.time_ns()}"
            temporary_data = data_path.with_name(f"{data_path.name}.{token}.tmp")
            temporary_metadata = metadata_path.with_name(
                f"{metadata_path.name}.{token}.tmp"
            )
            mapped = torch.from_file(
                str(temporary_data),
                shared=True,
                size=spec.numel,
                dtype=spec.dtype,
            ).view(spec.shape)
            # Copy directly into the file mapping. In particular, do not first
            # create a rank-private CPU tensor for a CUDA source.
            mapped.copy_(spec.parameter.detach(), non_blocking=False)
            if spec.parameter.device.type == "cuda":
                torch.cuda.synchronize(spec.parameter.device)
            del mapped
            with temporary_data.open("rb") as handle:
                os.fsync(handle.fileno())
            with temporary_metadata.open("w") as handle:
                json.dump(expected, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_data, data_path)
            os.replace(temporary_metadata, metadata_path)
            return False
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def prepare_node_shared_immutable_cache(
    model: torch.nn.Module,
    *,
    logical_component: str,
    checkpoint_identity: str,
    cache_root: str | Path,
    optimizer_owned_parameter_ids: Iterable[int] = (),
    externally_managed_parameter_ids: Iterable[int] = (),
    node_identity: str | None = None,
) -> dict[str, Any]:
    """Create or validate a cache on one local rank before a global barrier."""

    optimizer_owned_ids = {int(value) for value in optimizer_owned_parameter_ids}
    externally_managed_ids = {
        int(value) for value in externally_managed_parameter_ids
    }
    specs = _eligible_parameters(model, optimizer_owned_ids, externally_managed_ids)
    node_root = resolve_node_cache_root(cache_root, node_identity=node_identity)
    root = _identity_root(
        node_root,
        checkpoint_identity=checkpoint_identity,
        logical_component=logical_component,
    )
    root.mkdir(parents=True, exist_ok=True)
    reused = 0
    manifest_records: list[dict[str, Any]] = []
    for spec in specs:
        reused += int(
            _publish_record(
                spec,
                root=root,
                checkpoint_identity=checkpoint_identity,
                logical_component=logical_component,
            )
        )
        data_path, metadata_path, _ = _record_paths(root, spec.name)
        manifest_records.append(
            {
                **_record_metadata(
                    spec,
                    checkpoint_identity=checkpoint_identity,
                    logical_component=logical_component,
                ),
                "data_file": data_path.name,
                "metadata_file": metadata_path.name,
            }
        )

    manifest = {
        "schema_version": CACHE_SCHEMA,
        "checkpoint_identity": checkpoint_identity,
        "logical_component": logical_component,
        "node_identity": node_identity or socket.gethostname(),
        "parameter_count": len(specs),
        "immutable_bytes": sum(spec.nbytes for spec in specs),
        "records": manifest_records,
    }
    manifest_path = root / "manifest.json"
    lock_path = root / "manifest.lock"
    with lock_path.open("a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            if manifest_path.exists():
                observed = json.loads(manifest_path.read_text())
                if observed != manifest:
                    raise RuntimeError("immutable model cache manifest mismatch")
            else:
                temporary = manifest_path.with_name(
                    f"{manifest_path.name}.{os.getpid()}-{time.time_ns()}.tmp"
                )
                with temporary.open("w") as handle:
                    json.dump(manifest, handle, indent=2, sort_keys=True)
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, manifest_path)
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    return {
        "cache_root": str(root),
        "parameter_count": len(specs),
        "immutable_bytes": int(manifest["immutable_bytes"]),
        "created_parameter_count": len(specs) - reused,
        "reused_parameter_count": reused,
    }


class NodeSharedImmutableModelBacking:
    """Own shared host backing while preserving whole-model phase transfers."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        logical_component: str,
        checkpoint_identity: str,
        cache_root: str | Path,
        optimizer_owned_parameter_ids: Iterable[int] = (),
        externally_managed_parameter_ids: Iterable[int] = (),
        event_path: str | Path,
        node_identity: str | None = None,
        detailed_events: bool = True,
    ) -> None:
        self.model = model
        self.logical_component = str(logical_component)
        self.checkpoint_identity = str(checkpoint_identity)
        self.optimizer_owned_parameter_ids = {
            int(value) for value in optimizer_owned_parameter_ids
        }
        self._externally_managed_parameter_ids = frozenset(
            int(value) for value in externally_managed_parameter_ids
        )
        self.node_root = resolve_node_cache_root(
            cache_root, node_identity=node_identity
        )
        self.root = _identity_root(
            self.node_root,
            checkpoint_identity=self.checkpoint_identity,
            logical_component=self.logical_component,
        )
        self.event_path = Path(event_path)
        self.event_path.parent.mkdir(parents=True, exist_ok=True)
        if self.event_path.exists():
            raise FileExistsError(
                f"refusing to overwrite immutable-backing trace: {self.event_path}"
            )
        self.detailed_events = bool(detailed_events)
        self._event_order = 0
        self._records: tuple[ImmutableParameterRecord, ...] = ()
        self._record_by_parameter_id: dict[int, ImmutableParameterRecord] = {}
        self.release_observations: list[dict[str, Any]] = []
        self.stats: dict[str, int | float] = {
            "backed_parameter_count": 0,
            "shared_immutable_bytes": 0,
            "private_parameter_count": 0,
            "private_parameter_bytes": 0,
            "externally_managed_parameter_count": 0,
            "externally_managed_parameter_bytes": 0,
            "buffer_count": sum(1 for _ in model.buffers()),
            "materialize_count": 0,
            "materialized_bytes": 0,
            "materialize_wall_seconds": 0.0,
            "release_count": 0,
            # A rebind proves that no frozen-weight D2H copy was requested. It
            # does not, by itself, prove that CUDA storage became reclaimable:
            # autograd may still hold an alias.  The allocator-drop counters
            # below are the physical observation and are expected to be large
            # for checkpointed Generator blocks.
            "rebound_without_d2h_bytes": 0,
            "released_without_d2h_bytes": 0,
            "cuda_rebound_without_d2h_bytes": 0,
            "cuda_allocator_observation_count": 0,
            "cuda_allocator_drop_bytes": 0,
        }
        self._attach()
        self._write_event(
            "immutable_model_backing_installed",
            checkpoint_identity=self.checkpoint_identity,
            cache_root=str(self.root),
            stats=dict(self.stats),
        )

    def _write_event(self, event: str, **fields: Any) -> None:
        if not self.detailed_events and event not in {
            "immutable_model_backing_installed",
            "immutable_model_backing_removed",
        }:
            return
        self._event_order += 1
        payload = {
            "event_order": self._event_order,
            "time_unix": time.time(),
            "event": event,
            "logical_component": self.logical_component,
            **fields,
        }
        with self.event_path.open("a") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def _attach(self) -> None:
        manifest_path = self.root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"immutable model cache is not prepared: {manifest_path}"
            )
        manifest = json.loads(manifest_path.read_text())
        model_parameters = tuple(self.model.parameters())
        model_parameter_ids = {id(parameter) for parameter in model_parameters}
        unknown_external_ids = (
            set(self._externally_managed_parameter_ids) - model_parameter_ids
        )
        if unknown_external_ids:
            raise ValueError(
                "externally managed parameters must belong to the backed model"
            )
        externally_managed_parameters = [
            parameter
            for parameter in model_parameters
            if id(parameter) in self._externally_managed_parameter_ids
        ]
        for parameter in externally_managed_parameters:
            if parameter.requires_grad:
                raise ValueError(
                    "externally managed immutable parameter became trainable"
                )
            if id(parameter) in self.optimizer_owned_parameter_ids:
                raise ValueError(
                    "externally managed immutable parameter is optimizer-owned"
                )
        specs = _eligible_parameters(
            self.model,
            self.optimizer_owned_parameter_ids,
            set(self._externally_managed_parameter_ids),
        )
        expected_names = [spec.name for spec in specs]
        observed_names = [str(record["parameter_name"]) for record in manifest["records"]]
        if (
            manifest.get("schema_version") != CACHE_SCHEMA
            or manifest.get("checkpoint_identity") != self.checkpoint_identity
            or manifest.get("logical_component") != self.logical_component
            or observed_names != expected_names
        ):
            raise RuntimeError("immutable model cache inventory mismatch")

        records: list[ImmutableParameterRecord] = []
        manifest_by_name = {
            str(record["parameter_name"]): record for record in manifest["records"]
        }
        for spec in specs:
            expected = _record_metadata(
                spec,
                checkpoint_identity=self.checkpoint_identity,
                logical_component=self.logical_component,
            )
            observed = manifest_by_name[spec.name]
            for key, value in expected.items():
                if observed.get(key) != value:
                    raise RuntimeError(
                        f"immutable model cache metadata mismatch for {spec.name}"
                    )
            data_path = self.root / str(observed["data_file"])
            metadata_path = self.root / str(observed["metadata_file"])
            _validate_record_files(data_path, metadata_path, expected)
            host_tensor = torch.from_file(
                str(data_path),
                shared=False,
                size=spec.numel,
                dtype=spec.dtype,
            ).view(spec.shape)
            record = ImmutableParameterRecord(
                spec=spec,
                backing_path=data_path,
                host_tensor=host_tensor,
            )
            records.append(record)
            self._record_by_parameter_id[id(spec.parameter)] = record

        self._records = tuple(records)
        self.stats["backed_parameter_count"] = len(records)
        self.stats["shared_immutable_bytes"] = sum(
            record.spec.nbytes for record in records
        )
        private_parameters = [
            parameter
            for parameter in model_parameters
            if id(parameter) not in self._record_by_parameter_id
            and id(parameter) not in self._externally_managed_parameter_ids
        ]
        self.stats["private_parameter_count"] = len(private_parameters)
        self.stats["private_parameter_bytes"] = sum(
            _tensor_bytes(parameter) for parameter in private_parameters
        )
        self.stats["externally_managed_parameter_count"] = len(
            externally_managed_parameters
        )
        self.stats["externally_managed_parameter_bytes"] = sum(
            _tensor_bytes(parameter) for parameter in externally_managed_parameters
        )
        self.release_to_backing(reason="install")

    @property
    def shared_immutable_bytes(self) -> int:
        return int(self.stats["shared_immutable_bytes"])

    @property
    def backed_parameter_ids(self) -> frozenset[int]:
        return frozenset(self._record_by_parameter_id)

    @property
    def externally_managed_parameter_ids(self) -> frozenset[int]:
        return self._externally_managed_parameter_ids

    def _is_host_backed(self, record: ImmutableParameterRecord) -> bool:
        parameter = record.spec.parameter
        if parameter.device.type != "cpu" or parameter.numel() != record.spec.numel:
            return False
        return (
            parameter.untyped_storage().data_ptr()
            == record.host_tensor.untyped_storage().data_ptr()
        )

    def release_to_backing(self, *, reason: str) -> None:
        bytes_rebound = 0
        cuda_bytes_rebound = 0
        cuda_devices = {
            record.spec.parameter.device
            for record in self._records
            if record.spec.parameter.device.type == "cuda"
            and not self._is_host_backed(record)
        }
        allocated_before = {
            str(device): int(torch.cuda.memory_allocated(device))
            for device in cuda_devices
        }
        for record in self._records:
            parameter = record.spec.parameter
            if parameter.requires_grad:
                raise RuntimeError(
                    f"backed parameter became trainable: {record.spec.name}"
                )
            if id(parameter) in self.optimizer_owned_parameter_ids:
                raise RuntimeError(
                    f"backed parameter became optimizer-owned: {record.spec.name}"
                )
            if parameter.grad is not None:
                raise RuntimeError(
                    f"backed parameter gained a gradient: {record.spec.name}"
                )
            if self._is_host_backed(record):
                continue
            if parameter.device.type == "cuda":
                cuda_bytes_rebound += record.spec.nbytes
            parameter.data = record.host_tensor
            bytes_rebound += record.spec.nbytes
        allocated_after = {
            str(device): int(torch.cuda.memory_allocated(device))
            for device in cuda_devices
        }
        allocator_drop_bytes = sum(
            max(0, allocated_before[name] - allocated_after[name])
            for name in allocated_before
        )
        if bytes_rebound:
            self.stats["release_count"] += 1
            self.stats["rebound_without_d2h_bytes"] += bytes_rebound
            # Backward-compatible alias.  New evidence should use
            # rebound_without_d2h_bytes plus cuda_allocator_drop_bytes.
            self.stats["released_without_d2h_bytes"] += bytes_rebound
            self.stats["cuda_rebound_without_d2h_bytes"] += cuda_bytes_rebound
        if allocated_before:
            self.stats["cuda_allocator_observation_count"] += 1
            self.stats["cuda_allocator_drop_bytes"] += allocator_drop_bytes
        observation = {
            "reason": reason,
            "rebound_without_d2h_bytes": bytes_rebound,
            "cuda_rebound_without_d2h_bytes": cuda_bytes_rebound,
            "cuda_allocated_before": allocated_before,
            "cuda_allocated_after": allocated_after,
            "cuda_allocator_drop_bytes": allocator_drop_bytes,
        }
        self.release_observations.append(observation)
        self._write_event(
            "release_to_backing",
            **observation,
            physical_release_requires_no_live_aliases=True,
        )

    def materialize(
        self,
        target_device: torch.device | str,
        *,
        non_blocking: bool,
    ) -> None:
        target = torch.device(target_device)
        if target.type == "cpu":
            self.release_to_backing(reason="materialize_cpu")
            return
        started = time.perf_counter()
        bytes_moved = 0
        for record in self._records:
            parameter = record.spec.parameter
            if parameter.device == target and parameter.numel() == record.spec.numel:
                continue
            materialized = record.host_tensor.to(target, non_blocking=non_blocking)
            if (
                tuple(materialized.shape) != record.spec.shape
                or materialized.dtype != record.spec.dtype
            ):
                raise RuntimeError(
                    f"immutable materialization changed metadata: {record.spec.name}"
                )
            parameter.data = materialized
            bytes_moved += record.spec.nbytes
        if non_blocking and target.type == "cuda":
            torch.cuda.synchronize(target)
        elapsed = time.perf_counter() - started
        if bytes_moved:
            self.stats["materialize_count"] += 1
            self.stats["materialized_bytes"] += bytes_moved
            self.stats["materialize_wall_seconds"] += elapsed
        self._write_event(
            "materialize_from_backing",
            target_device=str(target),
            bytes=bytes_moved,
            elapsed_seconds=elapsed,
        )

    def move_non_backed_state(
        self,
        target_device: torch.device | str,
        *,
        non_blocking: bool,
    ) -> None:
        target = torch.device(target_device)
        for parameter in self.model.parameters():
            if (
                id(parameter) in self._record_by_parameter_id
                or id(parameter) in self._externally_managed_parameter_ids
            ):
                continue
            if parameter.device != target:
                parameter.data = parameter.data.to(
                    target, non_blocking=non_blocking
                )
            if parameter.grad is not None and parameter.grad.device != target:
                parameter.grad.data = parameter.grad.data.to(
                    target, non_blocking=non_blocking
                )
        for buffer in self.model.buffers():
            if buffer.device != target:
                buffer.data = buffer.data.to(target, non_blocking=non_blocking)
        self._write_event(
            "move_private_state",
            target_device=str(target),
            non_blocking=bool(non_blocking),
        )

    def move_to_cpu(self, *, non_blocking: bool) -> None:
        # Immutable parameters are rebound; no GPU-to-CPU copy is performed.
        self.release_to_backing(reason="whole_model_move_to_cpu")
        self.move_non_backed_state("cpu", non_blocking=non_blocking)

    def move_to_device(
        self,
        target_device: torch.device | str,
        *,
        non_blocking: bool,
    ) -> None:
        # Preserve the whole-model phase contract: all immutable parameters are
        # materialized at the model boundary, not lazily by forward hooks.
        self.materialize(target_device, non_blocking=non_blocking)
        self.move_non_backed_state(target_device, non_blocking=non_blocking)

    def assert_integrity(self) -> None:
        for record in self._records:
            if not record.backing_path.exists():
                raise RuntimeError(
                    f"immutable backing disappeared: {record.backing_path}"
                )
            if record.spec.parameter.requires_grad:
                raise RuntimeError(
                    f"backed parameter became trainable: {record.spec.name}"
                )
            if id(record.spec.parameter) in self.optimizer_owned_parameter_ids:
                raise RuntimeError(
                    f"backed parameter became optimizer-owned: {record.spec.name}"
                )
        for parameter in self.model.parameters():
            if id(parameter) not in self._externally_managed_parameter_ids:
                continue
            if parameter.requires_grad:
                raise RuntimeError(
                    "externally managed immutable parameter became trainable"
                )
            if id(parameter) in self.optimizer_owned_parameter_ids:
                raise RuntimeError(
                    "externally managed immutable parameter became optimizer-owned"
                )

    def evidence_summary(self) -> dict[str, Any]:
        """Return rank-local proof that all mappings target one node backing."""

        manifest_path = self.root / "manifest.json"
        manifest_bytes = manifest_path.read_bytes()
        inventory = []
        for record in self._records:
            stat = record.backing_path.stat()
            inventory.append(
                {
                    "name": record.spec.name,
                    "path": str(record.backing_path),
                    "device": int(stat.st_dev),
                    "inode": int(stat.st_ino),
                    "bytes": int(stat.st_size),
                }
            )
        inventory_digest = hashlib.sha256(
            json.dumps(inventory, sort_keys=True).encode()
        ).hexdigest()
        return {
            "logical_component": self.logical_component,
            "checkpoint_identity": self.checkpoint_identity,
            "node_cache_root": str(self.node_root),
            "component_cache_root": str(self.root),
            "manifest_path": str(manifest_path),
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "backing_inventory_sha256": inventory_digest,
            "backing_parameter_count": len(inventory),
            "stats": dict(self.stats),
            "release_observations": list(self.release_observations),
        }

    def remove(self) -> None:
        self.move_to_cpu(non_blocking=False)
        self.assert_integrity()
        self._write_event("immutable_model_backing_removed", stats=dict(self.stats))


def install_node_shared_immutable_model_backing(
    model: torch.nn.Module,
    **kwargs: Any,
) -> NodeSharedImmutableModelBacking:
    return NodeSharedImmutableModelBacking(model, **kwargs)
