"""Bounded CPU->GPU module materialization with reusable pinned staging.

The helper changes only physical transfer/residency.  It preserves every
registered parameter/buffer value, shape, dtype, stride and Parameter object,
and fails closed on storage aliases or non-contiguous official state rather
than silently changing layout semantics.
"""
from __future__ import annotations

import time
import threading
from dataclasses import dataclass
from typing import Any, Iterable

import torch


@dataclass
class _PinnedSlot:
    storage: torch.Tensor
    ready: torch.cuda.Event
    pending: bool = False


@dataclass
class _DeferredHostCleanup:
    token: int
    thread: threading.Thread
    result: dict[str, Any]


_DEFERRED_HOST_CLEANUPS: dict[int, _DeferredHostCleanup] = {}
_DEFERRED_HOST_CLEANUPS_LOCK = threading.Lock()
_DEFERRED_HOST_CLEANUP_TOKEN = 0


def start_deferred_host_allocator_cleanup() -> int:
    from runtime.official_baseline import trim_host_allocator

    global _DEFERRED_HOST_CLEANUP_TOKEN
    with _DEFERRED_HOST_CLEANUPS_LOCK:
        _DEFERRED_HOST_CLEANUP_TOKEN += 1
        token = _DEFERRED_HOST_CLEANUP_TOKEN
    result: dict[str, Any] = {
        "token": token,
        "status": "RUNNING",
        "started_monotonic_seconds": time.monotonic(),
    }

    def cleanup() -> None:
        started = time.perf_counter()
        try:
            result["host_allocator_trimmed"] = trim_host_allocator()
            result["status"] = "PASS"
        except BaseException as error:  # propagated by the mandatory drain
            result["status"] = "FAIL"
            result["error_type"] = type(error).__name__
            result["error"] = str(error)
        finally:
            result["wall_seconds"] = time.perf_counter() - started
            result["finished_monotonic_seconds"] = time.monotonic()

    thread = threading.Thread(
        target=cleanup,
        name=f"distillschedule-host-cleanup-{token}",
        daemon=False,
    )
    handle = _DeferredHostCleanup(token=token, thread=thread, result=result)
    with _DEFERRED_HOST_CLEANUPS_LOCK:
        _DEFERRED_HOST_CLEANUPS[token] = handle
    thread.start()
    return token


def drain_deferred_host_allocator_cleanups() -> list[dict[str, Any]]:
    """Join and validate every process-local deferred pinned-cache cleanup."""

    with _DEFERRED_HOST_CLEANUPS_LOCK:
        handles = list(_DEFERRED_HOST_CLEANUPS.values())
    results: list[dict[str, Any]] = []
    for handle in handles:
        wait_started = time.perf_counter()
        handle.thread.join()
        observed = dict(handle.result)
        observed["join_wait_host_wall_seconds"] = time.perf_counter() - wait_started
        results.append(observed)
        with _DEFERRED_HOST_CLEANUPS_LOCK:
            _DEFERRED_HOST_CLEANUPS.pop(handle.token, None)
        if observed.get("status") != "PASS":
            raise RuntimeError(
                f"deferred host cleanup {handle.token} failed: {observed}"
            )
    return results


def _unique_registered_tensors(
    module: torch.nn.Module,
) -> list[tuple[str, str, torch.Tensor]]:
    values: list[tuple[str, str, torch.Tensor]] = []
    seen_objects: set[int] = set()
    for kind, iterator in (
        ("parameter", module.named_parameters(remove_duplicate=False)),
        ("buffer", module.named_buffers(remove_duplicate=False)),
    ):
        for name, tensor in iterator:
            if id(tensor) in seen_objects:
                continue
            seen_objects.add(id(tensor))
            values.append((kind, name, tensor))
    return values


def _validate_cpu_state(
    tensors: Iterable[tuple[str, str, torch.Tensor]],
) -> dict[str, Any]:
    storages: dict[tuple[int, int], tuple[str, str]] = {}
    tensor_count = 0
    logical_bytes = 0
    dtype_bytes: dict[str, int] = {}
    for kind, name, tensor in tensors:
        tensor_count += 1
        if tensor.device.type != "cpu":
            raise ValueError(f"{kind} {name} is not on CPU before materialization")
        if isinstance(tensor, torch.nn.Parameter) and tensor.grad is not None:
            raise ValueError(f"parameter {name} has a live gradient during materialization")
        if tensor.layout != torch.strided:
            raise ValueError(f"{kind} {name} has unsupported layout {tensor.layout}")
        if tensor.numel() and not tensor.is_contiguous():
            raise ValueError(f"{kind} {name} is not contiguous")
        if tensor.numel():
            storage = tensor.untyped_storage()
            key = (int(storage.data_ptr()), int(storage.nbytes()))
            prior = storages.get(key)
            if prior is not None:
                raise ValueError(
                    f"distinct registered tensors alias one CPU storage: {prior} and {(kind, name)}"
                )
            storages[key] = (kind, name)
        size = tensor.numel() * tensor.element_size()
        logical_bytes += size
        dtype_bytes[str(tensor.dtype)] = dtype_bytes.get(str(tensor.dtype), 0) + size
    if tensor_count == 0:
        raise ValueError("module has no registered tensors to materialize")
    return {
        "tensor_count": tensor_count,
        "logical_bytes": logical_bytes,
        "dtype_bytes": dtype_bytes,
        "unique_storage_count": len(storages),
    }


def _registered_metadata(
    tensors: Iterable[tuple[str, str, torch.Tensor]],
) -> dict[int, dict[str, Any]]:
    return {
        id(tensor): {
            "object_id": id(tensor),
            "shape": tuple(tensor.shape),
            "stride": tuple(tensor.stride()),
            "dtype": tensor.dtype,
            "requires_grad": bool(getattr(tensor, "requires_grad", False)),
        }
        for _kind, _name, tensor in tensors
    }


def _assert_registered_metadata(
    tensors: Iterable[tuple[str, str, torch.Tensor]],
    *,
    before: dict[int, dict[str, Any]],
    device: torch.device,
) -> None:
    for kind, name, tensor in tensors:
        expected = before.get(id(tensor))
        if expected is None or expected["object_id"] != id(tensor):
            raise AssertionError(f"{kind} {name} object identity changed")
        if (
            tuple(tensor.shape) != expected["shape"]
            or tuple(tensor.stride()) != expected["stride"]
            or tensor.dtype != expected["dtype"]
            or tensor.device != device
            or bool(getattr(tensor, "requires_grad", False))
            != expected["requires_grad"]
        ):
            raise AssertionError(f"{kind} {name} metadata changed")


@torch.no_grad()
def pin_module_cpu_registered_state(module: torch.nn.Module) -> dict[str, Any]:
    """Replace pageable registered storage with private persistent pinned storage.

    This operation is intended for excluded cold preparation.  Parameter and
    buffer objects remain identical; only their physical CPU backing changes.
    """

    from runtime.official_baseline import trim_host_allocator

    tensors = _unique_registered_tensors(module)
    inventory = _validate_cpu_state(tensors)
    before = _registered_metadata(tensors)
    started = time.perf_counter()
    allocation_seconds = 0.0
    copy_seconds = 0.0
    for kind, name, source in tensors:
        allocation_started = time.perf_counter()
        target = torch.empty_strided(
            tuple(source.shape),
            tuple(source.stride()),
            dtype=source.dtype,
            device="cpu",
            pin_memory=True,
        )
        allocation_seconds += time.perf_counter() - allocation_started
        copy_started = time.perf_counter()
        target.copy_(source)
        copy_seconds += time.perf_counter() - copy_started
        source.data = target
        if not source.is_pinned():
            raise AssertionError(f"{kind} {name} did not acquire pinned backing")

    _assert_registered_metadata(tensors, before=before, device=torch.device("cpu"))
    trim_started = time.perf_counter()
    allocator_trimmed = trim_host_allocator()
    trim_seconds = time.perf_counter() - trim_started
    if any(not tensor.is_pinned() for _kind, _name, tensor in tensors):
        raise AssertionError("registered state lost pinned backing during host trim")
    return {
        "policy": "PER_RANK_PRIVATE_PERSISTENT_PINNED_REGISTERED_STATE",
        **inventory,
        "runtime_owned_active_pinned_bytes": inventory["logical_bytes"],
        "allocation_host_wall_seconds": allocation_seconds,
        "pageable_to_pinned_copy_host_wall_seconds": copy_seconds,
        "allocator_trim_host_wall_seconds": trim_seconds,
        "host_allocator_trimmed": allocator_trimmed,
        "wall_seconds": time.perf_counter() - started,
        "all_registered_state_pinned": True,
        "timing_scope": "excluded_cold_preparation",
    }


@torch.no_grad()
def materialize_module_from_persistent_pinned_cpu(
    module: torch.nn.Module,
    *,
    device: torch.device | str,
    cleanup_policy: str = "immediate",
) -> dict[str, Any]:
    """Directly materialize a fully pinned CPU module on a dedicated H2D stream."""

    from runtime.official_baseline import trim_host_allocator

    target_device = torch.device(device)
    if target_device.type != "cuda":
        raise ValueError("persistent pinned materialization requires a CUDA target")
    if cleanup_policy not in (
        "immediate",
        "deferred",
        "late_deferred",
        "post_full5",
    ):
        raise ValueError(f"unsupported persistent pinned cleanup policy {cleanup_policy}")
    tensors = _unique_registered_tensors(module)
    inventory = _validate_cpu_state(tensors)
    if any(not tensor.is_pinned() for _kind, _name, tensor in tensors):
        raise ValueError("persistent pinned materialization received pageable state")
    before = _registered_metadata(tensors)
    stream = torch.cuda.Stream(device=target_device)
    copy_begin = torch.cuda.Event(enable_timing=True)
    copy_end = torch.cuda.Event(enable_timing=True)
    started = time.perf_counter()
    enqueue_started = time.perf_counter()
    pinned_source_handles: list[torch.Tensor] = []
    with torch.cuda.stream(stream):
        copy_begin.record(stream)
        for _kind, _name, source in tensors:
            # ``Module.to`` replaces registered buffer objects.  Keep the
            # exact Parameter/buffer objects and retain a view of each pinned
            # source until its nonblocking copy has completed.
            pinned_source = source.detach()
            pinned_source_handles.append(pinned_source)
            target = torch.empty_strided(
                tuple(source.shape),
                tuple(source.stride()),
                dtype=source.dtype,
                device=target_device,
            )
            target.copy_(pinned_source, non_blocking=True)
            source.data = target
        copy_end.record(stream)
    enqueue_seconds = time.perf_counter() - enqueue_started
    wait_started = time.perf_counter()
    stream.synchronize()
    wait_seconds = time.perf_counter() - wait_started
    pinned_source_handles.clear()
    if tensors:
        del pinned_source
    h2d_device_seconds = copy_begin.elapsed_time(copy_end) / 1000.0
    _assert_registered_metadata(tensors, before=before, device=target_device)
    if any(tensor.device.type == "cpu" for _kind, _name, tensor in tensors):
        raise AssertionError("module retained CPU registered state after materialization")
    cleanup_token = None
    trim_seconds = 0.0
    allocator_trimmed = False
    if cleanup_policy == "immediate":
        trim_started = time.perf_counter()
        allocator_trimmed = trim_host_allocator()
        trim_seconds = time.perf_counter() - trim_started
    elif cleanup_policy == "deferred":
        cleanup_token = start_deferred_host_allocator_cleanup()
    result = {
        "policy": "PERSISTENT_PINNED_DIRECT_H2D",
        "device": str(target_device),
        **inventory,
        "h2d_enqueue_host_wall_seconds": enqueue_seconds,
        "completion_wait_host_wall_seconds": wait_seconds,
        "h2d_copy_stream_device_seconds": h2d_device_seconds,
        "allocator_trim_host_wall_seconds": trim_seconds,
        "host_allocator_trimmed": allocator_trimmed,
        "host_allocator_cleanup_policy": cleanup_policy,
        "deferred_host_cleanup_token": cleanup_token,
        "released_runtime_owned_active_pinned_bytes": inventory["logical_bytes"],
        "runtime_owned_active_pinned_bytes_after_return": 0,
        "runtime_owned_cached_pinned_bytes_pending_cleanup": (
            inventory["logical_bytes"]
            if cleanup_policy in ("deferred", "late_deferred", "post_full5")
            else 0
        ),
        "wall_seconds": time.perf_counter() - started,
        "ready_on_return": True,
    }
    return result


@torch.no_grad()
def materialize_module_from_cpu_with_bounded_pinned_staging(
    module: torch.nn.Module,
    *,
    device: torch.device | str,
    chunk_mib: int = 64,
    slot_count: int = 2,
) -> dict[str, Any]:
    """Move registered state to one CUDA device using bounded ping-pong slots.

    CPU pageable->pinned staging for chunk ``i+1`` can overlap the nonblocking
    pinned->device copy of chunk ``i``.  The function returns only after the
    dedicated H2D stream completes, so callers observe the same readiness
    boundary as synchronous ``module.to(device)``.
    """

    target_device = torch.device(device)
    if target_device.type != "cuda":
        raise ValueError("bounded pinned materialization requires a CUDA target")
    if chunk_mib <= 0:
        raise ValueError("chunk_mib must be positive")
    if slot_count not in (1, 2):
        raise ValueError("slot_count must be one or two")

    tensors = _unique_registered_tensors(module)
    inventory = _validate_cpu_state(tensors)
    chunk_bytes = chunk_mib * 2**20
    stream = torch.cuda.Stream(device=target_device)
    slots_by_dtype: dict[torch.dtype, list[_PinnedSlot]] = {}
    slot_cursor: dict[torch.dtype, int] = {}
    pinned_arena_bytes = 0
    chunk_count = 0
    pageable_to_pinned_host_wall_seconds = 0.0
    h2d_enqueue_host_wall_seconds = 0.0
    allocation_host_wall_seconds = 0.0
    started = time.perf_counter()
    copy_begin = torch.cuda.Event(enable_timing=True)
    copy_end = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(stream):
        copy_begin.record(stream)

    def slots_for(dtype: torch.dtype) -> list[_PinnedSlot]:
        nonlocal pinned_arena_bytes, allocation_host_wall_seconds
        existing = slots_by_dtype.get(dtype)
        if existing is not None:
            return existing
        element_size = torch.empty((), dtype=dtype).element_size()
        capacity = max(1, chunk_bytes // element_size)
        allocation_started = time.perf_counter()
        created = [
            _PinnedSlot(
                storage=torch.empty(capacity, dtype=dtype, pin_memory=True),
                ready=torch.cuda.Event(),
            )
            for _ in range(slot_count)
        ]
        allocation_host_wall_seconds += time.perf_counter() - allocation_started
        slots_by_dtype[dtype] = created
        slot_cursor[dtype] = 0
        pinned_arena_bytes += sum(
            slot.storage.numel() * slot.storage.element_size() for slot in created
        )
        return created

    for kind, name, source in tensors:
        before = {
            "shape": tuple(source.shape),
            "stride": tuple(source.stride()),
            "dtype": source.dtype,
            "requires_grad": bool(getattr(source, "requires_grad", False)),
        }
        if source.numel() == 0:
            target = torch.empty_strided(
                before["shape"],
                before["stride"],
                dtype=before["dtype"],
                device=target_device,
            )
        else:
            target = torch.empty_strided(
                before["shape"],
                before["stride"],
                dtype=before["dtype"],
                device=target_device,
            )
            source_flat = source.reshape(-1)
            target_flat = target.reshape(-1)
            slots = slots_for(source.dtype)
            capacity = slots[0].storage.numel()
            for offset in range(0, source.numel(), capacity):
                count = min(capacity, source.numel() - offset)
                index = slot_cursor[source.dtype]
                slot_cursor[source.dtype] = (index + 1) % len(slots)
                slot = slots[index]
                if slot.pending:
                    slot.ready.synchronize()
                    slot.pending = False
                cpu_started = time.perf_counter()
                slot.storage[:count].copy_(source_flat[offset : offset + count])
                pageable_to_pinned_host_wall_seconds += time.perf_counter() - cpu_started
                enqueue_started = time.perf_counter()
                with torch.cuda.stream(stream):
                    target_flat[offset : offset + count].copy_(
                        slot.storage[:count], non_blocking=True
                    )
                    slot.ready.record(stream)
                h2d_enqueue_host_wall_seconds += time.perf_counter() - enqueue_started
                slot.pending = True
                chunk_count += 1

        # Preserve the registered Tensor/Parameter object and its requires_grad
        # ownership; only the underlying physical storage changes.
        source.data = target
        if (
            tuple(source.shape) != before["shape"]
            or tuple(source.stride()) != before["stride"]
            or source.dtype != before["dtype"]
            or source.device != target_device
            or bool(getattr(source, "requires_grad", False)) != before["requires_grad"]
        ):
            raise AssertionError(f"{kind} {name} metadata changed during materialization")

    with torch.cuda.stream(stream):
        copy_end.record(stream)
    wait_started = time.perf_counter()
    stream.synchronize()
    completion_wait_host_wall_seconds = time.perf_counter() - wait_started
    h2d_device_seconds = copy_begin.elapsed_time(copy_end) / 1000.0

    if any(tensor.device != target_device for _kind, _name, tensor in tensors):
        raise AssertionError("module retained CPU registered state after materialization")
    return {
        "policy": "BOUNDED_PINNED_PING_PONG_H2D",
        "device": str(target_device),
        "chunk_mib": chunk_mib,
        "slot_count": slot_count,
        "pinned_arena_bytes": pinned_arena_bytes,
        "chunk_count": chunk_count,
        **inventory,
        "allocation_host_wall_seconds": allocation_host_wall_seconds,
        "pageable_to_pinned_host_wall_seconds": pageable_to_pinned_host_wall_seconds,
        "h2d_enqueue_host_wall_seconds": h2d_enqueue_host_wall_seconds,
        "completion_wait_host_wall_seconds": completion_wait_host_wall_seconds,
        "h2d_copy_stream_device_seconds": h2d_device_seconds,
        "wall_seconds": time.perf_counter() - started,
        "ready_on_return": True,
    }
