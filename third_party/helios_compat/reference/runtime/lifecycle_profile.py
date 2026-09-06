"""Low-overhead causal lifecycle profiler for fixed-placement Helios.

This module is instrumentation only.  It monkey-patches the already frozen M0
primitives at process startup; it does not change keep/offload/recompute policy,
copy direction, blocking mode, loss, optimizer, trace, or checkpoint state.

Two modes are deliberately separate:

``counters``
    Cheap issued-byte, unique-storage, host-call, wait, and resource counters.
    This is the only mode eligible for the paired <=3% overhead check.

``events``
    Adds CUDA events around existing runtime-owned copies.  Events are resolved
    only at shutdown, so profiling does not insert per-copy synchronization.
    These intervals are copy-associated stream intervals, not CUPTI/Nsight
    memcpy activity.  Observed memcpy bytes therefore remain ``null`` until an
    external CUDA-activity trace is reconciled.

No event stores a Tensor reference.  Storage/alias identity is converted to
plain integers before the wrapped operation returns.
"""
from __future__ import annotations

import argparse
import atexit
import contextvars
import functools
import json
import os
import resource
import runpy
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Hashable, Mapping, MutableMapping, TypeVar

import torch


SCHEMA_VERSION = "distillschedule.lifecycle_profile.v1"
T = TypeVar("T")
_ACTIVE: "LifecycleProfiler | None" = None
_PATCHED = False
_CONTEXT = contextvars.ContextVar(
    "distillschedule_lifecycle_context",
    default={"owner": "runtime", "operation": "unscoped"},
)


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def _storage_identity(tensor: torch.Tensor) -> tuple[str, int, int]:
    storage = tensor.untyped_storage()
    return (str(tensor.device), int(storage.data_ptr()), int(storage.nbytes()))


def _alias_identity(tensor: torch.Tensor) -> tuple[int, int, tuple[int, ...], tuple[int, ...], str]:
    storage = tensor.untyped_storage()
    return (
        int(storage.data_ptr()),
        int(tensor.storage_offset()),
        tuple(int(item) for item in tensor.shape),
        tuple(int(item) for item in tensor.stride()),
        str(tensor.dtype),
    )


def _read_int(path: str) -> int | None:
    try:
        return int(Path(path).read_text().strip())
    except (OSError, ValueError):
        return None


def _memory_stat() -> dict[str, int]:
    wanted = {
        "anon",
        "file",
        "shmem",
        "file_mapped",
        "inactive_anon",
        "active_anon",
        "inactive_file",
        "active_file",
    }
    result: dict[str, int] = {}
    try:
        for line in Path("/sys/fs/cgroup/memory.stat").read_text().splitlines():
            key, value = line.split()
            if key in wanted:
                result[key] = int(value)
    except (OSError, ValueError):
        pass
    return result


def _smaps_rollup() -> dict[str, int]:
    result: dict[str, int] = {}
    wanted = {
        "Rss": "rss_bytes",
        "Pss": "pss_bytes",
        "Private_Clean": "private_clean_bytes",
        "Private_Dirty": "private_dirty_bytes",
        "Shared_Clean": "shared_clean_bytes",
        "Shared_Dirty": "shared_dirty_bytes",
        "Locked": "locked_bytes",
    }
    try:
        for line in Path("/proc/self/smaps_rollup").read_text().splitlines():
            key = line.split(":", 1)[0]
            if key in wanted:
                result[wanted[key]] = int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return result


def _resource_snapshot(*, expensive: bool) -> dict[str, Any]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    result: dict[str, Any] = {
        "time_monotonic": time.perf_counter(),
        "major_faults": int(usage.ru_majflt),
        "minor_faults": int(usage.ru_minflt),
        "voluntary_context_switches": int(usage.ru_nvcsw),
        "involuntary_context_switches": int(usage.ru_nivcsw),
        "rss_peak_bytes": int(usage.ru_maxrss) * 1024,
        "cgroup_memory_current_bytes": _read_int("/sys/fs/cgroup/memory.current"),
        "cgroup_memory_peak_bytes": _read_int("/sys/fs/cgroup/memory.peak"),
    }
    if expensive:
        result["cgroup_memory_stat"] = _memory_stat()
        result["smaps_rollup"] = _smaps_rollup()
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        device = torch.cuda.current_device()
        stats = torch.cuda.memory_stats(device)
        free, total = torch.cuda.mem_get_info(device)
        result["cuda"] = {
            "device": int(device),
            "allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "active_bytes": int(stats.get("active_bytes.all.current", 0)),
            "inactive_split_bytes": int(
                stats.get("inactive_split_bytes.all.current", 0)
            ),
            "driver_free_bytes": int(free),
            "driver_total_bytes": int(total),
        }
    return result


@dataclass
class _CudaInterval:
    event_id: int
    direction: str
    owner: str
    operation: str
    global_step: int | None
    start: torch.cuda.Event
    end: torch.cuda.Event
    issued_bytes: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


def _interval_union(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    merged = 0.0
    current_start, current_end = sorted(intervals)[0]
    for start, end in sorted(intervals)[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            merged += current_end - current_start
            current_start, current_end = start, end
    return merged + current_end - current_start


class LifecycleProfiler:
    """Aggregate runtime-owned lifecycle events without retaining payload tensors."""

    def __init__(
        self,
        root: Path,
        *,
        mode: str,
        detailed_events: bool | None = None,
    ) -> None:
        if mode not in {"counters", "events"}:
            raise ValueError("lifecycle profile mode must be counters or events")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.rank = _rank()
        self.local_rank = _local_rank()
        self.detailed_events = mode == "events" if detailed_events is None else bool(
            detailed_events
        )
        self.event_path = self.root / f"rank_{self.rank:03d}.jsonl"
        self.summary_path = self.root / f"rank_{self.rank:03d}_summary.json"
        if self.event_path.exists() or self.summary_path.exists():
            raise FileExistsError(
                f"refusing to overwrite lifecycle profile rank {self.rank} under {self.root}"
            )
        self._lock = threading.Lock()
        self._closed = False
        self._closing = False
        self._event_order = 0
        self._step: int | None = None
        self._step_start_resources: dict[str, Any] | None = None
        self._unique_storage_keys: set[str] = set()
        self._unique_storage_keys_by_step: dict[str, set[str]] = defaultdict(set)
        self._unique_storage_bytes = 0
        self._issued_bytes = 0
        self._issued_count = 0
        self._host_call_seconds = 0.0
        self._host_call_residual_seconds = 0.0
        self._wait_seconds = 0.0
        self._wait_count = 0
        self._measurement_wait_seconds = 0.0
        self._measurement_wait_count = 0
        self._barrier_seconds = 0.0
        self._barrier_count = 0
        self._by_key: dict[str, dict[str, float | int]] = defaultdict(
            lambda: {
                "issued_bytes": 0,
                "unique_storage_bytes": 0,
                "call_count": 0,
                "host_call_seconds": 0.0,
                "wait_seconds": 0.0,
            }
        )
        self._by_step: dict[str, dict[str, float | int]] = defaultdict(
            lambda: {
                "issued_bytes": 0,
                "unique_storage_bytes": 0,
                "transfer_count": 0,
                "host_call_seconds": 0.0,
                "exposed_wait_seconds": 0.0,
                "exposed_wait_count": 0,
                "measurement_wait_seconds": 0.0,
                "measurement_wait_count": 0,
                "barrier_seconds": 0.0,
                "barrier_count": 0,
            }
        )
        self._cuda_intervals: list[_CudaInterval] = []
        self._anchor: torch.cuda.Event | None = None
        self._next_interval_id = 0
        self._original_synchronize: Callable[..., Any] | None = None
        self._started_unix = time.time()
        self._started_monotonic = time.perf_counter()
        self._write(
            {
                "event": "profiler_start",
                "schema_version": SCHEMA_VERSION,
                "mode": self.mode,
                "observed_memcpy_bytes": None,
                "observed_memcpy_source": "not_available_without_cuda_activity",
                "resource": _resource_snapshot(expensive=self.mode == "events"),
            },
            force=True,
        )

    def _write(self, record: Mapping[str, Any], *, force: bool = False) -> None:
        if not self.detailed_events and not force:
            return
        with self._lock:
            self._event_order += 1
            payload = {
                "schema_version": SCHEMA_VERSION,
                "event_order": self._event_order,
                "time_unix": time.time(),
                "time_monotonic": time.perf_counter(),
                "pid": os.getpid(),
                "rank": self.rank,
                "local_rank": self.local_rank,
                "global_step": self._step,
                **dict(record),
            }
            with self.event_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")

    def _ensure_anchor(self) -> torch.cuda.Event | None:
        if self.mode != "events" or not torch.cuda.is_available():
            return None
        if self._anchor is None:
            self._anchor = torch.cuda.Event(enable_timing=True)
            self._anchor.record(torch.cuda.current_stream())
        return self._anchor

    @staticmethod
    def _key(direction: str, owner: str, operation: str) -> str:
        return f"{direction}:{owner}:{operation}"

    def _step_key(self) -> str:
        return "construction" if self._step is None else str(self._step)

    def measure_transfer(
        self,
        *,
        direction: str,
        owner: str,
        operation: str,
        unique_storage_key: Hashable | None,
        logical_bytes: int,
        issued_bytes_before: int,
        issued_bytes_after: Callable[[], int],
        call: Callable[[], T],
        metadata: Mapping[str, Any] | None = None,
    ) -> T:
        if direction not in {"h2d", "d2h"}:
            raise ValueError(f"unsupported transfer direction {direction}")
        start_event = end_event = None
        if self._ensure_anchor() is not None:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record(torch.cuda.current_stream())
        started = time.perf_counter()
        outcome = "success"
        try:
            result = call()
            return result
        except BaseException:
            outcome = "error"
            raise
        finally:
            host_seconds = time.perf_counter() - started
            if end_event is not None:
                end_event.record(torch.cuda.current_stream())
            after = int(issued_bytes_after())
            issued = max(0, after - int(issued_bytes_before))
            unique = 0
            key_string = None
            if unique_storage_key is not None and issued > 0:
                key_string = repr(unique_storage_key)
                if key_string not in self._unique_storage_keys:
                    self._unique_storage_keys.add(key_string)
                    unique = int(logical_bytes)
                    self._unique_storage_bytes += unique
            step_key = self._step_key()
            step_unique = 0
            if (
                key_string is not None
                and issued > 0
                and key_string not in self._unique_storage_keys_by_step[step_key]
            ):
                self._unique_storage_keys_by_step[step_key].add(key_string)
                step_unique = int(logical_bytes)
            aggregate_key = self._key(direction, owner, operation)
            bucket = self._by_key[aggregate_key]
            bucket["issued_bytes"] += issued
            bucket["unique_storage_bytes"] += unique
            bucket["call_count"] += 1
            bucket["host_call_seconds"] += host_seconds
            self._issued_bytes += issued
            self._issued_count += 1
            self._host_call_seconds += host_seconds
            step_bucket = self._by_step[step_key]
            step_bucket["issued_bytes"] += issued
            step_bucket["unique_storage_bytes"] += step_unique
            step_bucket["transfer_count"] += 1
            step_bucket["host_call_seconds"] += host_seconds
            metadata_value = dict(metadata or {})
            if start_event is not None and end_event is not None:
                self._next_interval_id += 1
                self._cuda_intervals.append(
                    _CudaInterval(
                        event_id=self._next_interval_id,
                        direction=direction,
                        owner=owner,
                        operation=operation,
                        global_step=self._step,
                        start=start_event,
                        end=end_event,
                        issued_bytes=issued,
                        metadata=metadata_value,
                    )
                )
            self._write(
                {
                    "event": "runtime_owned_transfer",
                    "direction": direction,
                    "owner": owner,
                    "operation": operation,
                    "outcome": outcome,
                    "logical_unique_bytes_added": unique,
                    "issued_transfer_bytes": issued,
                    "host_call_seconds": host_seconds,
                    "cuda_interval_pending": start_event is not None,
                    "observed_memcpy_bytes": None,
                    "storage_key": key_string,
                    **metadata_value,
                }
            )

    def measure_wait(
        self,
        *,
        owner: str,
        operation: str,
        call: Callable[[], T],
        wait_kind: str,
    ) -> T:
        started = time.perf_counter()
        outcome = "success"
        try:
            return call()
        except BaseException:
            outcome = "error"
            raise
        finally:
            elapsed = time.perf_counter() - started
            key = self._key("wait", owner, operation)
            self._by_key[key]["wait_seconds"] += elapsed
            self._by_key[key]["call_count"] += 1
            if wait_kind == "distributed_barrier":
                self._barrier_seconds += elapsed
                self._barrier_count += 1
                self._by_step[self._step_key()]["barrier_seconds"] += elapsed
                self._by_step[self._step_key()]["barrier_count"] += 1
            elif wait_kind == "measurement_boundary_synchronize":
                self._measurement_wait_seconds += elapsed
                self._measurement_wait_count += 1
                self._by_step[self._step_key()][
                    "measurement_wait_seconds"
                ] += elapsed
                self._by_step[self._step_key()][
                    "measurement_wait_count"
                ] += 1
            else:
                self._wait_seconds += elapsed
                self._wait_count += 1
                self._by_step[self._step_key()]["exposed_wait_seconds"] += elapsed
                self._by_step[self._step_key()]["exposed_wait_count"] += 1
            self._write(
                {
                    "event": "runtime_wait",
                    "owner": owner,
                    "operation": operation,
                    "wait_kind": wait_kind,
                    "elapsed_seconds": elapsed,
                    "outcome": outcome,
                }
            )

    def step_start(self, global_step: int) -> None:
        self._step = int(global_step)
        self._step_start_resources = _resource_snapshot(
            expensive=self.mode == "events"
        )
        self._write(
            {
                "event": "lifecycle_step_start",
                "resource": self._step_start_resources,
            }
        )

    def step_end(self, global_step: int, *, train_generator: bool) -> None:
        end = _resource_snapshot(expensive=self.mode == "events")
        start = self._step_start_resources or {}
        delta: dict[str, Any] = {}
        for key in (
            "major_faults",
            "minor_faults",
            "voluntary_context_switches",
            "involuntary_context_switches",
        ):
            if key in start and key in end:
                delta[key] = int(end[key]) - int(start[key])
        self._write(
            {
                "event": "lifecycle_step_end",
                "train_generator": bool(train_generator),
                "resource": end,
                "resource_delta": delta,
            }
        )
        self._step_start_resources = None
        self._step = int(global_step)

    def _resolve_cuda_intervals(self) -> dict[str, Any]:
        if not self._cuda_intervals or self._anchor is None:
            return {
                "interval_count": 0,
                "copy_associated_interval_union_seconds": 0.0,
                "sum_interval_seconds": 0.0,
                "by_key": {},
                "observed_memcpy_bytes": None,
            }
        # One shutdown synchronization resolves all already-recorded events.  It
        # is excluded from exposed wait because _closing suppresses the global
        # synchronize wrapper.
        if self._original_synchronize is not None:
            self._original_synchronize()
        else:
            torch.cuda.synchronize()
        intervals: list[tuple[float, float]] = []
        by_key: dict[str, list[tuple[float, float]]] = defaultdict(list)
        by_step: dict[str, list[tuple[float, float]]] = defaultdict(list)
        records = []
        for interval in self._cuda_intervals:
            start_ms = float(self._anchor.elapsed_time(interval.start))
            end_ms = float(self._anchor.elapsed_time(interval.end))
            start_seconds = start_ms / 1000.0
            end_seconds = max(start_seconds, end_ms / 1000.0)
            intervals.append((start_seconds, end_seconds))
            key = self._key(
                interval.direction, interval.owner, interval.operation
            )
            by_key[key].append((start_seconds, end_seconds))
            step_key = (
                "construction"
                if interval.global_step is None
                else str(interval.global_step)
            )
            by_step[step_key].append((start_seconds, end_seconds))
            records.append(
                {
                    "event_id": interval.event_id,
                    "direction": interval.direction,
                    "owner": interval.owner,
                    "operation": interval.operation,
                    "global_step": interval.global_step,
                    "start_seconds": start_seconds,
                    "end_seconds": end_seconds,
                    "duration_seconds": end_seconds - start_seconds,
                    "issued_bytes": interval.issued_bytes,
                    "metadata": interval.metadata,
                }
            )
        return {
            "interval_count": len(intervals),
            "copy_associated_interval_union_seconds": _interval_union(intervals),
            "sum_interval_seconds": sum(end - start for start, end in intervals),
            "by_key": {
                key: {
                    "interval_count": len(value),
                    "interval_union_seconds": _interval_union(value),
                    "sum_interval_seconds": sum(end - start for start, end in value),
                }
                for key, value in sorted(by_key.items())
            },
            "by_step": {
                key: {
                    "interval_count": len(value),
                    "interval_union_seconds": _interval_union(value),
                    "sum_interval_seconds": sum(end - start for start, end in value),
                }
                for key, value in sorted(by_step.items())
            },
            "records": records,
            "observed_memcpy_bytes": None,
            "observed_memcpy_source": "requires_external_cuda_activity_reconciliation",
        }

    def close(self) -> dict[str, Any] | None:
        if self._closed or self._closing:
            return None
        self._closing = True
        try:
            cuda = self._resolve_cuda_intervals()
            cuda_seconds = float(cuda["copy_associated_interval_union_seconds"])
            host_residual = max(0.0, self._host_call_seconds - cuda_seconds)
            self._host_call_residual_seconds = host_residual
            summary = {
                "schema_version": SCHEMA_VERSION,
                "status": "PASS",
                "mode": self.mode,
                "pid": os.getpid(),
                "rank": self.rank,
                "local_rank": self.local_rank,
                "elapsed_seconds": time.perf_counter() - self._started_monotonic,
                "logical_unique_storage_bytes": self._unique_storage_bytes,
                "issued_transfer_bytes": self._issued_bytes,
                "issued_transfer_count": self._issued_count,
                "runtime_owned_copy_reconciliation": {
                    "issued_bytes_from_wrapped_sites": self._issued_bytes,
                    "observed_memcpy_bytes": None,
                    "observed_memcpy_bytes_status": "Not Run",
                    "residual_bytes": self._issued_bytes,
                },
                "host_call_seconds": self._host_call_seconds,
                "host_call_minus_copy_associated_interval_union_seconds": host_residual,
                "exposed_dependency_wait_seconds": self._wait_seconds,
                "exposed_dependency_wait_count": self._wait_count,
                "measurement_boundary_wait_seconds": self._measurement_wait_seconds,
                "measurement_boundary_wait_count": self._measurement_wait_count,
                "distributed_barrier_seconds": self._barrier_seconds,
                "distributed_barrier_count": self._barrier_count,
                "by_key": {
                    key: dict(value) for key, value in sorted(self._by_key.items())
                },
                "by_step": {
                    key: dict(value) for key, value in sorted(self._by_step.items())
                },
                "cuda_intervals": cuda,
                "final_resource": _resource_snapshot(
                    expensive=self.mode == "events"
                ),
                "limitations": [
                    "CUDA event intervals are copy-associated stream intervals, not CUDA activity memcpy records.",
                    "Observed memcpy bytes remain null until Nsight/CUPTI reconciliation.",
                    "Synchronous call wall time can include allocation, page fault, enqueue, Python, and wait.",
                    "Only explicit existing synchronize/barrier sites are classified as exposed dependency wait.",
                ],
            }
            temporary = self.summary_path.with_name(
                self.summary_path.name + f".tmp-{os.getpid()}"
            )
            temporary.write_text(
                json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n"
            )
            os.replace(temporary, self.summary_path)
            self._write(
                {
                    "event": "profiler_end",
                    "summary_path": str(self.summary_path),
                    "issued_transfer_bytes": self._issued_bytes,
                    "logical_unique_storage_bytes": self._unique_storage_bytes,
                },
                force=True,
            )
            self._closed = True
            return summary
        finally:
            self._closing = False


def _profile() -> LifecycleProfiler | None:
    return _ACTIVE


def _context(owner: str, operation: str):
    token = _CONTEXT.set({"owner": owner, "operation": operation})

    class _Reset:
        def __enter__(self):
            return None

        def __exit__(self, _type, _value, _traceback):
            _CONTEXT.reset(token)

    return _Reset()


def _wrap_transfer_method(
    cls: type[Any],
    name: str,
    builder: Callable[[Any, tuple[Any, ...], dict[str, Any]], Mapping[str, Any]],
) -> None:
    original = getattr(cls, name)
    if getattr(original, "_distillschedule_lifecycle_wrapped", False):
        return

    @functools.wraps(original)
    def wrapped(self: Any, *args: Any, **kwargs: Any):
        profiler = _profile()
        if profiler is None:
            return original(self, *args, **kwargs)
        spec = dict(builder(self, args, kwargs))
        issued_before = int(spec.pop("issued_before", 0))
        issued_after = spec.pop("issued_after")
        call = lambda: original(self, *args, **kwargs)
        with _context(str(spec["owner"]), str(spec["operation"])):
            return profiler.measure_transfer(
                direction=str(spec["direction"]),
                owner=str(spec["owner"]),
                operation=str(spec["operation"]),
                unique_storage_key=spec.get("unique_storage_key"),
                logical_bytes=int(spec.get("logical_bytes", 0)),
                issued_bytes_before=issued_before,
                issued_bytes_after=issued_after,
                call=call,
                metadata=spec.get("metadata"),
            )

    wrapped._distillschedule_lifecycle_wrapped = True  # type: ignore[attr-defined]
    setattr(cls, name, wrapped)


def _activation_pack_spec(self: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Mapping[str, Any]:
    tensor = args[0]
    sid = _storage_identity(tensor)
    alias = _alias_identity(tensor)
    before = int(self.stats["offloaded_storage_bytes"])
    return {
        "direction": "d2h",
        "owner": self.logical_component,
        "operation": "saved_tensor_pack",
        # Unique logical storage is storage-deduplicated.  Alias identity is
        # still emitted as metadata because the frozen offloader can issue
        # multiple legal copies for distinct strided aliases of one storage.
        "unique_storage_key": ("activation", self.logical_component, sid),
        "logical_bytes": int(tensor.untyped_storage().nbytes()),
        "issued_before": before,
        "issued_after": lambda: int(self.stats["offloaded_storage_bytes"]),
        "metadata": {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "storage_bytes": int(tensor.untyped_storage().nbytes()),
            "alias_id": list(alias),
        },
    }


def _activation_unpack_spec(self: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Mapping[str, Any]:
    packed = args[0]
    cpu_tensor = getattr(packed, "cpu_tensor", None)
    bytes_value = (
        int(cpu_tensor.untyped_storage().nbytes())
        if isinstance(cpu_tensor, torch.Tensor)
        else 0
    )
    storage_key = (
        ("activation_unpack", self.logical_component, _storage_identity(cpu_tensor))
        if isinstance(cpu_tensor, torch.Tensor)
        else None
    )
    counter = {"value": 0}

    def after() -> int:
        if bytes_value:
            counter["value"] = bytes_value
        return counter["value"]

    return {
        "direction": "h2d",
        "owner": self.logical_component,
        "operation": "saved_tensor_unpack",
        "unique_storage_key": storage_key,
        "logical_bytes": bytes_value,
        "issued_before": 0,
        "issued_after": after,
        "metadata": {"storage_bytes": bytes_value},
    }


def _stream_block_spec(self: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Mapping[str, Any]:
    block_index = int(args[0])
    generation = str(args[1])
    before = int(self.stats["materialized_bytes"])
    logical = sum(
        int(record.nbytes)
        for record in self._records_by_block[block_index]
        if not record.resident
    )
    return {
        "direction": "h2d",
        "owner": self.logical_component,
        "operation": "frozen_block_materialize",
        "unique_storage_key": (
            "frozen_block",
            self.logical_component,
            block_index,
        ),
        "logical_bytes": logical,
        "issued_before": before,
        "issued_after": lambda: int(self.stats["materialized_bytes"]),
        "metadata": {
            "block_index": block_index,
            "generation": generation,
            "host_policy": self.host_policy.value,
            "window_size": int(self.window_size),
        },
    }


def _nonstreamed_spec(self: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Mapping[str, Any]:
    target = torch.device(args[0] if args else kwargs["target_device"])
    source_direction = "h2d" if target.type == "cuda" else "d2h"
    bytes_value = 0
    parameter_ids = set(self._record_by_parameter_id)
    for parameter in self.model.parameters():
        if id(parameter) in parameter_ids:
            continue
        if parameter.device != target:
            bytes_value += int(parameter.numel() * parameter.element_size())
        if parameter.grad is not None and parameter.grad.device != target:
            bytes_value += int(parameter.grad.numel() * parameter.grad.element_size())
    for buffer in self.model.buffers():
        if buffer.device != target:
            bytes_value += int(buffer.numel() * buffer.element_size())
    counter = {"value": 0}

    def after() -> int:
        counter["value"] = bytes_value
        return counter["value"]

    return {
        "direction": source_direction,
        "owner": self.logical_component,
        "operation": "nonstreamed_state_move",
        "unique_storage_key": None,
        "logical_bytes": bytes_value,
        "issued_before": 0,
        "issued_after": after,
        "metadata": {
            "target_device": str(target),
            "non_blocking": bool(kwargs.get("non_blocking", False)),
        },
    }


def _install_patches(profiler: LifecycleProfiler) -> None:
    global _PATCHED
    if _PATCHED:
        return
    from helios.utils.utils_helios_post import OptimizedLowVRAMManager
    from runtime.activation_offload import AuditedSavedTensorOffloader
    from runtime.frozen_block_streaming import ImmutableFrozenBlockStreamer
    from runtime.official_baseline import PhaseTracer

    _wrap_transfer_method(
        AuditedSavedTensorOffloader, "_pack", _activation_pack_spec
    )
    _wrap_transfer_method(
        AuditedSavedTensorOffloader, "_unpack", _activation_unpack_spec
    )
    _wrap_transfer_method(
        ImmutableFrozenBlockStreamer, "_materialize_block", _stream_block_spec
    )
    _wrap_transfer_method(
        ImmutableFrozenBlockStreamer, "move_non_streamed_state", _nonstreamed_spec
    )

    original_finish_move = OptimizedLowVRAMManager._finish_move

    @functools.wraps(original_finish_move)
    def finish_move(self: Any, *args: Any, **kwargs: Any):
        owner = self._label(kwargs["model"], kwargs["model_to_move"])
        with _context(owner, f"model_move_{kwargs['direction']}_dependency"):
            return original_finish_move(self, *args, **kwargs)

    OptimizedLowVRAMManager._finish_move = finish_move

    original_record = PhaseTracer.record

    @functools.wraps(original_record)
    def phase_record(self: Any, phase: str, **extra: Any):
        active = _profile()
        if active is not None:
            if phase == "training_step_start":
                active.step_start(int(extra["global_step"]))
            elif phase == "training_step_end":
                active.step_end(
                    int(extra["global_step"]),
                    train_generator=bool(extra["train_generator"]),
                )
        result = original_record(self, phase, **extra)
        if active is not None and phase == "training_complete_skip_final_save":
            active.close()
        return result

    PhaseTracer.record = phase_record

    original_sync = torch.cuda.synchronize
    profiler._original_synchronize = original_sync

    @functools.wraps(original_sync)
    def synchronize(*args: Any, **kwargs: Any):
        active = _profile()
        if active is None or active._closing:
            return original_sync(*args, **kwargs)
        context = _CONTEXT.get()
        wait_kind = (
            "measurement_boundary_synchronize"
            if str(context["operation"]) == "unscoped"
            else "cuda_synchronize"
        )
        return active.measure_wait(
            owner=str(context["owner"]),
            operation=str(context["operation"]),
            call=lambda: original_sync(*args, **kwargs),
            wait_kind=wait_kind,
        )

    torch.cuda.synchronize = synchronize

    if hasattr(torch.distributed, "barrier"):
        original_barrier = torch.distributed.barrier

        @functools.wraps(original_barrier)
        def barrier(*args: Any, **kwargs: Any):
            active = _profile()
            if active is None:
                return original_barrier(*args, **kwargs)
            context = _CONTEXT.get()
            return active.measure_wait(
                owner=str(context["owner"]),
                operation=str(context["operation"]),
                call=lambda: original_barrier(*args, **kwargs),
                wait_kind="distributed_barrier",
            )

        torch.distributed.barrier = barrier
    _PATCHED = True


def install_from_environment() -> LifecycleProfiler | None:
    """Install once when ``DISTILLSCHEDULE_LIFECYCLE_PROFILE_ROOT`` is set."""

    global _ACTIVE
    root = os.environ.get("DISTILLSCHEDULE_LIFECYCLE_PROFILE_ROOT")
    if not root:
        return None
    if _ACTIVE is not None:
        return _ACTIVE
    mode = os.environ.get("DISTILLSCHEDULE_LIFECYCLE_PROFILE_MODE", "counters")
    _ACTIVE = LifecycleProfiler(Path(root), mode=mode)
    _install_patches(_ACTIVE)
    atexit.register(_ACTIVE.close)
    return _ACTIVE


def _run_training_script(arguments: list[str]) -> None:
    if not arguments:
        raise SystemExit(
            "usage: python runtime/lifecycle_profile.py -- TRAINING_SCRIPT [ARGS...]"
        )
    script = arguments[0]
    sys.argv = arguments
    install_from_environment()
    runpy.run_path(script, run_name="__main__")


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    parsed = parser.parse_args()
    arguments = parsed.arguments
    if arguments and arguments[0] == "--":
        arguments = arguments[1:]
    _run_training_script(arguments)


if __name__ == "__main__":
    main()
