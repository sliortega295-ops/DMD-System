"""Low-overhead, planner-oriented primitive profiling receipts.

The recorder is intentionally orthogonal to formal Full5 timing. It captures
small calibration canaries and correctness/profile runs, including resource
snapshots and exact operation identities. Importing this module does not
require PyTorch; CUDA and saved-tensor features activate only when available.
"""
from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import resource
import time
from pathlib import Path
from typing import Any, Iterator, Mapping

EVENT_SCHEMA = "distillschedule.planner_profile_event.v0"
MODEL_SCHEMA = "distillschedule.planner_model_inventory.v0"
HARDWARE_SCHEMA = "distillschedule.planner_hardware_calibration.v0"


def _atomic_jsonl_append(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(dict(value), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_int(path: Path) -> int | None:
    try:
        text = path.read_text().strip()
    except OSError:
        return None
    if text == "max":
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _read_events(path: Path) -> dict[str, int]:
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return {}
    result: dict[str, int] = {}
    for line in lines:
        parts = line.split()
        if len(parts) == 2:
            try:
                result[parts[0]] = int(parts[1])
            except ValueError:
                continue
    return result


def detect_cgroup_v2_root() -> Path | None:
    override = os.environ.get("DISTILLSCHEDULE_CGROUP_ROOT")
    if override:
        root = Path(override)
        return root if root.exists() else None
    root = Path("/sys/fs/cgroup")
    return root if (root / "cgroup.controllers").exists() else None


def cgroup_snapshot(root: Path | None = None) -> dict[str, Any]:
    root = detect_cgroup_v2_root() if root is None else root
    if root is None:
        return {"available": False}
    return {
        "available": True,
        "root": str(root),
        "memory_current_bytes": _read_int(root / "memory.current"),
        "memory_peak_bytes": _read_int(root / "memory.peak"),
        "memory_max_bytes": _read_int(root / "memory.max"),
        "memory_events": _read_events(root / "memory.events"),
    }


def process_snapshot() -> dict[str, Any]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss = None
    try:
        resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
        rss = resident_pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        pass
    return {
        "pid": os.getpid(),
        "rss_bytes": rss,
        "peak_rss_bytes": int(usage.ru_maxrss) * 1024,
    }


def cuda_snapshot(device: Any = None) -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"available": False}
    if not torch.cuda.is_available():
        return {"available": False}
    value = torch.device(device) if device is not None else torch.device("cuda", torch.cuda.current_device())
    free, total = torch.cuda.mem_get_info(value)
    return {
        "available": True,
        "device": str(value),
        "allocated_bytes": int(torch.cuda.memory_allocated(value)),
        "reserved_bytes": int(torch.cuda.memory_reserved(value)),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(value)),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved(value)),
        "driver_free_bytes": int(free),
        "driver_total_bytes": int(total),
    }


@dataclasses.dataclass(frozen=True)
class ProfileIdentity:
    run_id: str
    workload: str
    rank: int
    source_sha: str
    trace_sha: str | None = None
    checkpoint_sha: str | None = None

    def __post_init__(self) -> None:
        if not self.run_id or not self.workload or not self.source_sha:
            raise ValueError("profile identity fields cannot be empty")
        if self.rank < 0:
            raise ValueError("rank cannot be negative")


class PlannerProfileRecorder:
    """Append-only operation recorder consumed by build_planner_profile.py."""

    def __init__(
        self,
        path: str | Path,
        *,
        identity: ProfileIdentity,
        device: Any = None,
        synchronize: bool = False,
    ) -> None:
        self.path = Path(path)
        self.identity = identity
        self.device = device
        self.synchronize = bool(synchronize)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise FileExistsError(f"refusing to overwrite planner profile: {self.path}")
        self._order = 0

    def _sync(self) -> None:
        if not self.synchronize:
            return
        try:
            import torch
        except ImportError:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    @contextlib.contextmanager
    def span(
        self,
        *,
        operation: str,
        role: str,
        backend: str,
        mesh_size: int,
        local_batch: int,
        stage: int | None = None,
        grad_mode: str = "no_grad",
        residency: str = "resident",
        activation_policy: str = "none",
        offload_blocks: tuple[int, ...] = (),
        logical_step: int | None = None,
        call_index: int | None = None,
        model_version: int | str | None = None,
        h2d_bytes: int = 0,
        d2h_bytes: int = 0,
        collective_bytes: int = 0,
        materialization_count: int = 0,
        metadata: Mapping[str, Any] | None = None,
    ) -> Iterator[dict[str, Any]]:
        if mesh_size <= 0 or local_batch <= 0:
            raise ValueError("mesh and local batch must be positive")
        if min(h2d_bytes, d2h_bytes, collective_bytes, materialization_count) < 0:
            raise ValueError("profile counters cannot be negative")
        before = {
            "process": process_snapshot(),
            "cgroup": cgroup_snapshot(),
            "cuda": cuda_snapshot(self.device),
        }
        cuda_begin = cuda_end = None
        try:
            import torch
            if torch.cuda.is_available():
                cuda_begin = torch.cuda.Event(enable_timing=True)
                cuda_end = torch.cuda.Event(enable_timing=True)
        except ImportError:
            pass
        self._sync()
        if cuda_begin is not None:
            cuda_begin.record()
        started_perf = time.perf_counter()
        started_ns = time.time_ns()
        mutable: dict[str, Any] = {}
        error: BaseException | None = None
        try:
            yield mutable
        except BaseException as caught:
            error = caught
            raise
        finally:
            if cuda_end is not None:
                cuda_end.record()
            self._sync()
            wall = time.perf_counter() - started_perf
            cuda_seconds = None
            if cuda_begin is not None and cuda_end is not None and self.synchronize:
                cuda_seconds = float(cuda_begin.elapsed_time(cuda_end)) / 1000.0
            after = {
                "process": process_snapshot(),
                "cgroup": cgroup_snapshot(),
                "cuda": cuda_snapshot(self.device),
            }
            self._order += 1
            payload: dict[str, Any] = {
                "schema_version": EVENT_SCHEMA,
                "event_order": self._order,
                "run_id": self.identity.run_id,
                "workload": self.identity.workload,
                "rank": self.identity.rank,
                "source_sha": self.identity.source_sha,
                "trace_sha": self.identity.trace_sha,
                "checkpoint_sha": self.identity.checkpoint_sha,
                "operation": operation,
                "role": role,
                "backend": backend,
                "mesh_size": mesh_size,
                "local_batch": local_batch,
                "stage": stage,
                "grad_mode": grad_mode,
                "residency": residency,
                "activation_policy": activation_policy,
                "offload_blocks": list(offload_blocks),
                "logical_step": logical_step,
                "call_index": call_index,
                "model_version": model_version,
                "start_unix_ns": started_ns,
                "wall_seconds": wall,
                "cuda_device_seconds": cuda_seconds,
                "h2d_bytes": h2d_bytes + int(mutable.get("h2d_bytes", 0)),
                "d2h_bytes": d2h_bytes + int(mutable.get("d2h_bytes", 0)),
                "collective_bytes": collective_bytes + int(mutable.get("collective_bytes", 0)),
                "parameter_materialization_count": materialization_count
                + int(mutable.get("parameter_materialization_count", 0)),
                "before": before,
                "after": after,
                "metadata": {**dict(metadata or {}), **dict(mutable.get("metadata", {}))},
                "status": "FAIL" if error is not None else "PASS",
            }
            if error is not None:
                payload["error_type"] = type(error).__name__
                payload["error"] = str(error)
            for key in ("compute_seconds", "nccl_seconds", "h2d_seconds", "d2h_seconds", "activation_peak_bytes"):
                if key in mutable:
                    payload[key] = mutable[key]
            _atomic_jsonl_append(self.path, payload)

    def record_model_inventory(self, model: Any, *, role: str) -> Mapping[str, Any]:
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("model inventory requires PyTorch") from error
        parameters = list(model.named_parameters(remove_duplicate=True))
        buffers = list(model.named_buffers(remove_duplicate=True))
        parameter_bytes = sum(p.numel() * p.element_size() for _, p in parameters)
        trainable_bytes = sum(p.numel() * p.element_size() for _, p in parameters if p.requires_grad)
        block_bytes = tuple(
            sum(p.numel() * p.element_size() for p in block.parameters(recurse=True))
            for block in getattr(model, "blocks", ())
        )
        payload = {
            "schema_version": MODEL_SCHEMA,
            "run_id": self.identity.run_id,
            "workload": self.identity.workload,
            "rank": self.identity.rank,
            "role": role,
            "parameter_bytes": int(parameter_bytes),
            "frozen_parameter_bytes": int(parameter_bytes - trainable_bytes),
            "trainable_parameter_bytes": int(trainable_bytes),
            "trainable_numel": int(sum(p.numel() for _, p in parameters if p.requires_grad)),
            "buffer_bytes": int(sum(b.numel() * b.element_size() for _, b in buffers)),
            "block_parameter_bytes": [int(value) for value in block_bytes],
            "dtype_inventory": sorted({str(p.dtype) for _, p in parameters}),
            "parameter_tensor_count": len(parameters),
            "buffer_tensor_count": len(buffers),
            "source_sha": self.identity.source_sha,
        }
        _atomic_jsonl_append(self.path, payload)
        return payload

    def record_hardware_calibration(self, **values: Any) -> Mapping[str, Any]:
        payload = {
            "schema_version": HARDWARE_SCHEMA,
            "run_id": self.identity.run_id,
            "rank": self.identity.rank,
            "source_sha": self.identity.source_sha,
            **values,
        }
        _atomic_jsonl_append(self.path, payload)
        return payload


@dataclasses.dataclass
class SavedTensorCensus:
    live_bytes: int = 0
    peak_bytes: int = 0
    saved_bytes: int = 0
    saved_tensors: int = 0
    unpacked_tensors: int = 0

    def as_dict(self) -> dict[str, int]:
        return dataclasses.asdict(self)


@contextlib.contextmanager
def saved_tensor_census() -> Iterator[SavedTensorCensus]:
    """Measure saved-tensor liveness without changing tensor placement.

    This hook is for a dedicated calibration run. It returns tensors unchanged
    and therefore does not implement activation offload or alter backward math.
    """

    try:
        import torch
    except ImportError as error:
        raise RuntimeError("saved tensor census requires PyTorch") from error
    census = SavedTensorCensus()
    sizes: dict[int, int] = {}

    def pack(tensor: Any) -> Any:
        size = int(tensor.numel() * tensor.element_size())
        key = id(tensor)
        sizes[key] = sizes.get(key, 0) + size
        census.live_bytes += size
        census.saved_bytes += size
        census.saved_tensors += 1
        census.peak_bytes = max(census.peak_bytes, census.live_bytes)
        return tensor

    def unpack(tensor: Any) -> Any:
        size = sizes.pop(id(tensor), 0)
        census.live_bytes = max(0, census.live_bytes - size)
        census.unpacked_tensors += 1
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        yield census


def load_profile_events(paths: list[str | Path]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for raw in paths:
        path = Path(raw)
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid profile JSONL {path}:{line_number}") from error
    return result


__all__ = [
    "EVENT_SCHEMA",
    "HARDWARE_SCHEMA",
    "MODEL_SCHEMA",
    "PlannerProfileRecorder",
    "ProfileIdentity",
    "SavedTensorCensus",
    "cgroup_snapshot",
    "cuda_snapshot",
    "detect_cgroup_v2_root",
    "load_profile_events",
    "process_snapshot",
    "saved_tensor_census",
]
