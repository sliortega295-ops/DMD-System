"""Exact compact FA3 replay cache for selected native checkpoint blocks.

The pinned split2 custom op returns large accumulation workspaces in addition
to the public output/LSE pair.  PyTorch selective checkpointing caches an
operation's entire return tree, so marking that raw op would retain about
2.54 GiB per block.  This module exposes an opaque custom op whose public
contract contains only the exact output and softmax LSE.  Its implementation
still calls the same pinned split2 forward, and its autograd function calls the
same pinned backward.

The feature is default-off and only active inside explicitly selected native
per-block checkpoint calls.  It does not change no-grad attention, checkpoint
boundaries, FSDP placement, or application-level call counts.
"""

from __future__ import annotations

import contextvars
import concurrent.futures
import dataclasses
import functools
import math
import threading
from collections import defaultdict, deque
from collections.abc import Callable, Iterable
from typing import Any

import torch
from loguru import logger
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.checkpoint import checkpoint, create_selective_checkpoint_contexts


EXPECTED_BLOCK_COUNT = 50
EXPECTED_GRAD_TRANSFORMER_FORWARDS = 6
EXPECTED_HEADS = 56
EXPECTED_HEAD_DIM = 128
EXPECTED_SEQUENCE = 37_760
COMPACT_FORWARD_SCHEMA = (
    "h3_a100::fa3_forward_compact(Tensor q, Tensor k, Tensor v, "
    "float softmax_scale) -> (Tensor, Tensor)"
)
SUPPORTED_STORAGE = ("cuda", "cpu", "cpu_staged")


@dataclasses.dataclass
class FA3ReplayCacheStats:
    grad_transformer_forwards: int = 0
    completed_grad_transformer_forwards: int = 0
    selected_checkpoint_wraps: int = 0
    selected_scoped_executions: int = 0
    compact_attention_entries: int = 0
    compact_forward_impl_calls: int = 0
    compact_backward_calls: int = 0
    cached_logical_bytes: int = 0
    cpu_d2h_entries: int = 0
    cpu_h2d_entries: int = 0
    cpu_d2h_tensors: int = 0
    cpu_h2d_tensors: int = 0
    cpu_d2h_bytes: int = 0
    cpu_h2d_bytes: int = 0
    cpu_pool_allocated_bytes: int = 0
    cpu_pool_reused_tensors: int = 0
    cpu_pool_busy_waits: int = 0
    cpu_pool_busy_misses: int = 0
    cpu_d2h_backpressure_waits: int = 0
    cpu_stage_allocated_bytes: int = 0
    cpu_forward_host_copy_entries: int = 0
    cpu_forward_host_copy_bytes: int = 0
    cpu_backward_host_copy_entries: int = 0
    cpu_backward_host_copy_bytes: int = 0
    cpu_backward_prefetches: int = 0
    cpu_backward_prefetch_misses: int = 0
    allocator_trim_calls: int = 0
    unexpected_cpu_storage_calls: int = 0
    unexpected_kernel_contract_calls: int = 0
    unexpected_checkpoint_contract_calls: int = 0


@dataclasses.dataclass
class _Runtime:
    raw_forward: Callable[..., Any]
    raw_backward: Callable[..., Any]
    stats: FA3ReplayCacheStats
    staged_pool: "_PageableStagingPool | None" = None


_RUNTIME: _Runtime | None = None
_CACHE_ACTIVE: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "h3_a100_fa3_replay_cache_active", default=False
)
_CACHE_BLOCK_INDEX: contextvars.ContextVar[int] = contextvars.ContextVar(
    "h3_a100_fa3_replay_cache_block_index", default=-1
)


@torch.library.custom_op("h3_a100::fa3_forward_compact", mutates_args=())
def _fa3_forward_compact(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    runtime = _RUNTIME
    if runtime is None:
        raise RuntimeError("H3 compact FA3 runtime was used before installation")
    out, softmax_lse, *_workspace = runtime.raw_forward(
        q,
        k,
        v,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        float(softmax_scale),
        causal=False,
        window_size_left=-1,
        window_size_right=-1,
        attention_chunk=0,
        softcap=0.0,
        num_splits=2,
        pack_gqa=None,
        sm_margin=0,
    )
    runtime.stats.compact_forward_impl_calls += 1
    runtime.stats.cached_logical_bytes += int(
        out.numel() * out.element_size()
        + softmax_lse.numel() * softmax_lse.element_size()
    )
    return out, softmax_lse


@_fa3_forward_compact.register_fake
def _(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    del k, v, softmax_scale
    return torch.empty_like(q), torch.empty(
        (q.shape[0], q.shape[2], q.shape[1]),
        dtype=torch.float32,
        device=q.device,
    )


class _CompactFA3Autograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float):
        out, softmax_lse = _fa3_forward_compact(q, k, v, float(scale))
        ctx.save_for_backward(q, k, v, out, softmax_lse)
        ctx.scale = float(scale)
        ctx.block_index = int(_CACHE_BLOCK_INDEX.get())
        return out

    @staticmethod
    def backward(ctx: Any, dout: torch.Tensor):
        runtime = _RUNTIME
        if runtime is None:
            raise RuntimeError("H3 compact FA3 backward ran before installation")
        q, k, v, out, softmax_lse = ctx.saved_tensors
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        runtime.raw_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            None,
            None,
            None,
            None,
            None,
            None,
            dq,
            dk,
            dv,
            ctx.scale,
            False,
            -1,
            -1,
            0.0,
            False,
            0,
        )
        runtime.stats.compact_backward_calls += 1
        if runtime.staged_pool is not None:
            runtime.staged_pool.prefetch_after_backward(int(ctx.block_index))
        return dq, dk, dv, None


@dataclasses.dataclass
class _PinnedBuffer:
    tensor: torch.Tensor
    reusable_after: torch.cuda.Event | None = None


@dataclasses.dataclass
class _CPUCacheEntry:
    buffers: tuple[_PinnedBuffer, ...]
    ready: torch.cuda.Event
    device: torch.device


class _PinnedPool:
    """Process-local pinned buffers reused across the six sequential grad graphs."""

    def __init__(self, stats: FA3ReplayCacheStats, max_d2h_inflight: int) -> None:
        self.stats = stats
        self.max_d2h_inflight = int(max_d2h_inflight)
        self._buffers: dict[tuple[Any, ...], list[_PinnedBuffer]] = defaultdict(list)
        self._copy_streams: dict[int, torch.cuda.Stream] = {}
        self._pending_d2h: dict[int, deque[torch.cuda.Event]] = defaultdict(deque)

    @staticmethod
    def _key(block_index: int, tensor_index: int, value: torch.Tensor) -> tuple[Any, ...]:
        return (
            int(block_index),
            int(tensor_index),
            tuple(value.shape),
            tuple(value.stride()),
            value.dtype,
        )

    def copy_stream(self, device: torch.device) -> torch.cuda.Stream:
        index = int(device.index if device.index is not None else torch.cuda.current_device())
        stream = self._copy_streams.get(index)
        if stream is None:
            stream = torch.cuda.Stream(device=index)
            self._copy_streams[index] = stream
        return stream

    def acquire(
        self, block_index: int, tensor_index: int, value: torch.Tensor
    ) -> _PinnedBuffer:
        key = self._key(block_index, tensor_index, value)
        candidates = self._buffers[key]
        if candidates:
            # There is exactly one logical cache entry per block in each grad
            # graph. Reuse its process-local host buffer across sequential
            # graphs. If the previous H2D has not completed, the D2H copy
            # stream waits on that event before overwriting the host buffer.
            candidate = candidates[0]
            if candidate.reusable_after is not None and not candidate.reusable_after.query():
                self.stats.cpu_pool_busy_waits += 1
            self.stats.cpu_pool_reused_tensors += 1
            return candidate
        tensor = torch.empty_like(
            value, device="cpu", pin_memory=True, memory_format=torch.preserve_format
        )
        candidate = _PinnedBuffer(tensor=tensor)
        candidates.append(candidate)
        self.stats.cpu_pool_allocated_bytes += int(
            tensor.numel() * tensor.element_size()
        )
        return candidate

    def throttle_before_forward(self, device: torch.device) -> None:
        index = int(device.index if device.index is not None else torch.cuda.current_device())
        pending = self._pending_d2h[index]
        while pending and pending[0].query():
            pending.popleft()
        if len(pending) < self.max_d2h_inflight:
            return
        # Add a device-side dependency rather than a host synchronize. The
        # next compact FA3 allocation cannot run until the oldest D2H has
        # released its source, bounding retained GPU outputs while allowing
        # intervening block compute to overlap with the copy stream.
        oldest = pending.popleft()
        torch.cuda.current_stream(device).wait_event(oldest)
        self.stats.cpu_d2h_backpressure_waits += 1

    def record_d2h(self, device: torch.device, ready: torch.cuda.Event) -> None:
        index = int(device.index if device.index is not None else torch.cuda.current_device())
        self._pending_d2h[index].append(ready)

    @staticmethod
    def release(buffer: _PinnedBuffer, reusable_after: torch.cuda.Event) -> None:
        buffer.reusable_after = reusable_after


class _HostCachingMode(TorchDispatchMode):
    def __init__(
        self,
        storage: deque[_CPUCacheEntry],
        pool: _PinnedPool,
        stats: FA3ReplayCacheStats,
        block_index: int,
    ) -> None:
        super().__init__()
        self.storage = storage
        self.pool = pool
        self.stats = stats
        self.block_index = int(block_index)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        del types
        kwargs = {} if kwargs is None else kwargs
        if func == _fa3_forward_compact._opoverload:
            self.pool.throttle_before_forward(args[0].device)
        result = func(*args, **kwargs)
        if func != _fa3_forward_compact._opoverload:
            return result
        values = tuple(result)
        if len(values) != 2 or any(value.device.type != "cuda" for value in values):
            self.stats.unexpected_cpu_storage_calls += 1
            raise RuntimeError("H3 CPU FA3 cache expected two CUDA output tensors")
        device = values[0].device
        if any(value.device != device for value in values):
            self.stats.unexpected_cpu_storage_calls += 1
            raise RuntimeError("H3 CPU FA3 cache outputs span multiple CUDA devices")

        buffers = tuple(
            self.pool.acquire(self.block_index, index, value)
            for index, value in enumerate(values)
        )
        producer = torch.cuda.Event()
        producer.record(torch.cuda.current_stream(device))
        copy_stream = self.pool.copy_stream(device)
        ready = torch.cuda.Event()
        with torch.cuda.stream(copy_stream):
            copy_stream.wait_event(producer)
            for buffer, value in zip(buffers, values):
                if buffer.reusable_after is not None:
                    copy_stream.wait_event(buffer.reusable_after)
                    buffer.reusable_after = None
                buffer.tensor.copy_(value, non_blocking=True)
                value.record_stream(copy_stream)
            ready.record(copy_stream)
        self.pool.record_d2h(device, ready)
        logical_bytes = sum(
            int(buffer.tensor.numel() * buffer.tensor.element_size())
            for buffer in buffers
        )
        self.storage.append(_CPUCacheEntry(buffers, ready, device))
        self.stats.cpu_d2h_entries += 1
        self.stats.cpu_d2h_tensors += len(buffers)
        self.stats.cpu_d2h_bytes += logical_bytes
        return result


class _HostCachedMode(TorchDispatchMode):
    def __init__(
        self,
        storage: deque[_CPUCacheEntry],
        pool: _PinnedPool,
        stats: FA3ReplayCacheStats,
    ) -> None:
        super().__init__()
        self.storage = storage
        self.pool = pool
        self.stats = stats

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        del types
        kwargs = {} if kwargs is None else kwargs
        if func != _fa3_forward_compact._opoverload:
            return func(*args, **kwargs)
        if not self.storage:
            self.stats.unexpected_cpu_storage_calls += 1
            raise RuntimeError("H3 CPU FA3 replay had no staged host entry")
        entry = self.storage.popleft()
        stream = torch.cuda.current_stream(entry.device)
        stream.wait_event(entry.ready)
        values = tuple(
            torch.empty_like(buffer.tensor, device=entry.device)
            for buffer in entry.buffers
        )
        for value, buffer in zip(values, entry.buffers):
            value.copy_(buffer.tensor, non_blocking=True)
        reusable_after = torch.cuda.Event()
        reusable_after.record(stream)
        logical_bytes = sum(
            int(buffer.tensor.numel() * buffer.tensor.element_size())
            for buffer in entry.buffers
        )
        for buffer in entry.buffers:
            self.pool.release(buffer, reusable_after)
        self.stats.cpu_h2d_entries += 1
        self.stats.cpu_h2d_tensors += len(entry.buffers)
        self.stats.cpu_h2d_bytes += logical_bytes
        return values


def _create_cpu_cache_contexts(
    registration: "FA3ReplayCacheRegistration", block_index: int
) -> tuple[TorchDispatchMode, TorchDispatchMode]:
    if registration.cpu_pool is None:
        registration.stats.unexpected_cpu_storage_calls += 1
        raise RuntimeError("H3 CPU FA3 cache has no pinned-buffer pool")
    storage: deque[_CPUCacheEntry] = deque()
    return (
        _HostCachingMode(storage, registration.cpu_pool, registration.stats, block_index),
        _HostCachedMode(storage, registration.cpu_pool, registration.stats),
    )


@dataclasses.dataclass
class _PageableBuffer:
    tensor: torch.Tensor
    reusable_after: torch.cuda.Event | None = None


@dataclasses.dataclass
class _StagePair:
    tensors: tuple[torch.Tensor, torch.Tensor]
    in_use: bool = False


@dataclasses.dataclass
class _PageableCacheEntry:
    block_index: int
    pages: tuple[_PageableBuffer, _PageableBuffer]
    shapes: tuple[tuple[int, ...], tuple[int, ...]]
    forward_future: concurrent.futures.Future[Any]
    device: torch.device
    backward_future: concurrent.futures.Future[Any] | None = None


class _PageableStagingPool:
    """Pageable cache reservoir with a bounded reusable pinned staging window.

    The 50-block raw pinned reservoir was rejected by a real cgroup OOM.  This
    pool keeps the long-lived exact out/LSE values in ordinary host memory and
    uses at most ``max_d2h_inflight`` pinned pairs for physical CUDA transfers.
    Host memcpy is issued by one process-local worker and is overlapped with
    the intervening block compute.  Before backward, the next block is copied
    into a pinned pair, but its GPU allocation/H2D remains demand-driven so the
    candidate does not add a second 0.51-GiB GPU cache entry.
    """

    _OUT_SHAPE = (1, EXPECTED_SEQUENCE, EXPECTED_HEADS, EXPECTED_HEAD_DIM)
    _LSE_SHAPE = (1, EXPECTED_HEADS, EXPECTED_SEQUENCE)

    def __init__(
        self,
        stats: FA3ReplayCacheStats,
        selected: tuple[int, ...],
        max_d2h_inflight: int,
    ) -> None:
        self.stats = stats
        self.selected = tuple(sorted(int(index) for index in selected))
        self.max_d2h_inflight = int(max_d2h_inflight)
        self._descending = tuple(reversed(self.selected))
        self._next_lower = {
            current: self._descending[position + 1]
            for position, current in enumerate(self._descending[:-1])
        }
        self._condition = threading.Condition()
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="h3-fa3-host-copy"
        )
        self._pending_forward: deque[concurrent.futures.Future[Any]] = deque()
        self._entries: dict[int, _PageableCacheEntry] = {}
        self._pages: dict[tuple[int, int], _PageableBuffer] = {}
        self._stages: list[_StagePair] = []
        self._copy_streams: dict[int, torch.cuda.Stream] = {}
        self._preallocate()

    @staticmethod
    def _nbytes(tensor: torch.Tensor) -> int:
        return int(tensor.numel() * tensor.element_size())

    @staticmethod
    def _view(tensor: torch.Tensor, tensor_index: int, shape: tuple[int, ...]) -> torch.Tensor:
        del tensor_index
        # Use a flat prefix rather than slicing the sequence dimension from a
        # max-shape tensor. In particular, slicing LSE's last dimension would
        # retain the max-sequence row stride and turn one logical transfer into
        # many strided copies. The prefix has the exact contiguous layout that
        # the production FA3 op returned for this rank-qualified sequence.
        return tensor.view(-1)[: math.prod(shape)].view(shape)

    def _preallocate(self) -> None:
        templates = (
            (self._OUT_SHAPE, torch.bfloat16),
            (self._LSE_SHAPE, torch.float32),
        )
        for block_index in self.selected:
            for tensor_index, (shape, dtype) in enumerate(templates):
                tensor = torch.empty(shape, dtype=dtype, device="cpu")
                self._pages[(block_index, tensor_index)] = _PageableBuffer(tensor)
                self.stats.cpu_pool_allocated_bytes += self._nbytes(tensor)
        for _ in range(self.max_d2h_inflight):
            tensors = tuple(
                torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
                for shape, dtype in templates
            )
            pair = _StagePair(tensors=tensors)  # type: ignore[arg-type]
            self._stages.append(pair)
            self.stats.cpu_stage_allocated_bytes += sum(
                self._nbytes(tensor) for tensor in tensors
            )

    def copy_stream(self, device: torch.device) -> torch.cuda.Stream:
        index = int(device.index if device.index is not None else torch.cuda.current_device())
        stream = self._copy_streams.get(index)
        if stream is None:
            stream = torch.cuda.Stream(device=index)
            self._copy_streams[index] = stream
        return stream

    def begin_graph(self) -> None:
        if self._entries:
            self.stats.unexpected_cpu_storage_calls += 1
            raise RuntimeError(
                "H3 staged FA3 cache began a grad graph before the prior graph drained"
            )

    def _acquire_stage(self) -> _StagePair:
        with self._condition:
            while True:
                for pair in self._stages:
                    if not pair.in_use:
                        pair.in_use = True
                        return pair
                self.stats.cpu_pool_busy_waits += 1
                self._condition.wait()

    def _release_stage(self, pair: _StagePair) -> None:
        with self._condition:
            pair.in_use = False
            self._condition.notify_all()

    def _throttle_forward(self) -> None:
        while self._pending_forward and self._pending_forward[0].done():
            self._pending_forward.popleft().result()
        if len(self._pending_forward) < self.max_d2h_inflight:
            return
        self.stats.cpu_d2h_backpressure_waits += 1
        self._pending_forward.popleft().result()

    def stage_forward(
        self, block_index: int, values: tuple[torch.Tensor, torch.Tensor]
    ) -> _PageableCacheEntry:
        if block_index in self._entries:
            self.stats.unexpected_cpu_storage_calls += 1
            raise RuntimeError(f"duplicate staged FA3 entry for block {block_index}")
        shapes = (tuple(values[0].shape), tuple(values[1].shape))
        sequence = int(shapes[0][1]) if len(shapes[0]) == 4 else -1
        valid_shapes = (
            shapes[0] == (1, sequence, EXPECTED_HEADS, EXPECTED_HEAD_DIM)
            and shapes[1] == (1, EXPECTED_HEADS, sequence)
            and 0 < sequence <= EXPECTED_SEQUENCE
        )
        if not valid_shapes:
            self.stats.unexpected_cpu_storage_calls += 1
            raise RuntimeError(
                "H3 staged FA3 cache observed an unexpected production shape: "
                f"{shapes[0]}/{shapes[1]}"
            )
        self._throttle_forward()
        pair = self._acquire_stage()
        pages = (
            self._pages[(block_index, 0)],
            self._pages[(block_index, 1)],
        )
        device = values[0].device
        producer = torch.cuda.Event()
        producer.record(torch.cuda.current_stream(device))
        ready = torch.cuda.Event()
        copy_stream = self.copy_stream(device)
        with torch.cuda.stream(copy_stream):
            copy_stream.wait_event(producer)
            for tensor_index, (stage, value, shape) in enumerate(
                zip(pair.tensors, values, shapes)
            ):
                self._view(stage, tensor_index, shape).copy_(value, non_blocking=True)
                value.record_stream(copy_stream)
            ready.record(copy_stream)

        def finish_forward_copy() -> None:
            ready.synchronize()
            for tensor_index, (page, stage, shape) in enumerate(
                zip(pages, pair.tensors, shapes)
            ):
                if page.reusable_after is not None:
                    page.reusable_after.synchronize()
                    page.reusable_after = None
                self._view(page.tensor, tensor_index, shape).copy_(
                    self._view(stage, tensor_index, shape)
                )
            logical_bytes = sum(self._nbytes(value) for value in values)
            self.stats.cpu_forward_host_copy_entries += 1
            self.stats.cpu_forward_host_copy_bytes += logical_bytes
            self._release_stage(pair)

        future = self._executor.submit(finish_forward_copy)
        self._pending_forward.append(future)
        entry = _PageableCacheEntry(block_index, pages, shapes, future, device)
        self._entries[block_index] = entry
        logical_bytes = sum(self._nbytes(value) for value in values)
        self.stats.cpu_d2h_entries += 1
        self.stats.cpu_d2h_tensors += len(pages)
        self.stats.cpu_d2h_bytes += logical_bytes
        return entry

    def _prefetch_block(self, block_index: int) -> None:
        entry = self._entries.get(int(block_index))
        if entry is None:
            self.stats.cpu_backward_prefetch_misses += 1
            raise RuntimeError(f"missing staged FA3 entry for backward block {block_index}")
        if entry.backward_future is not None:
            return

        def prepare_stage() -> _StagePair:
            entry.forward_future.result()
            pair = self._acquire_stage()
            for tensor_index, (stage, page, shape) in enumerate(
                zip(pair.tensors, entry.pages, entry.shapes)
            ):
                if page.reusable_after is not None:
                    page.reusable_after.synchronize()
                    page.reusable_after = None
                self._view(stage, tensor_index, shape).copy_(
                    self._view(page.tensor, tensor_index, shape)
                )
            logical_bytes = sum(
                self._nbytes(self._view(page.tensor, index, shape))
                for index, (page, shape) in enumerate(zip(entry.pages, entry.shapes))
            )
            self.stats.cpu_backward_host_copy_entries += 1
            self.stats.cpu_backward_host_copy_bytes += logical_bytes
            return pair

        entry.backward_future = self._executor.submit(prepare_stage)
        self.stats.cpu_backward_prefetches += 1

    def prefetch_first_before_backward(self) -> None:
        if not self._descending:
            return
        self._prefetch_block(self._descending[0])

    def prefetch_after_backward(self, block_index: int) -> None:
        next_block = self._next_lower.get(int(block_index))
        if next_block is not None:
            self._prefetch_block(next_block)

    def replay(self, entry: _PageableCacheEntry) -> tuple[torch.Tensor, torch.Tensor]:
        if entry.backward_future is None:
            self.stats.cpu_backward_prefetch_misses += 1
            self._prefetch_block(entry.block_index)
        assert entry.backward_future is not None
        pair = entry.backward_future.result()
        stream = torch.cuda.current_stream(entry.device)
        values = tuple(
            torch.empty(shape, dtype=page.tensor.dtype, device=entry.device)
            for page, shape in zip(entry.pages, entry.shapes)
        )
        for tensor_index, (value, stage, shape) in enumerate(
            zip(values, pair.tensors, entry.shapes)
        ):
            value.copy_(self._view(stage, tensor_index, shape), non_blocking=True)
        reusable_after = torch.cuda.Event()
        reusable_after.record(stream)
        for page in entry.pages:
            page.reusable_after = reusable_after

        def release_after_h2d() -> None:
            reusable_after.synchronize()
            self._release_stage(pair)

        self._executor.submit(release_after_h2d)
        # Queue the next lower block now. The worker first observes the H2D
        # completion above, then performs pageable->pinned staging while this
        # block is recomputed and differentiated. Waiting until FA backward
        # finishes would expose most of that host copy on the next block.
        next_block = self._next_lower.get(int(entry.block_index))
        if next_block is not None:
            self._prefetch_block(next_block)
        logical_bytes = sum(self._nbytes(value) for value in values)
        self.stats.cpu_h2d_entries += 1
        self.stats.cpu_h2d_tensors += len(entry.pages)
        self.stats.cpu_h2d_bytes += logical_bytes
        self._entries.pop(entry.block_index, None)
        return values  # type: ignore[return-value]


class _StagedCachingMode(TorchDispatchMode):
    def __init__(self, storage: deque[_PageableCacheEntry], pool: _PageableStagingPool, block_index: int):
        super().__init__()
        self.storage = storage
        self.pool = pool
        self.block_index = int(block_index)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        del types
        result = func(*args, **({} if kwargs is None else kwargs))
        if func != _fa3_forward_compact._opoverload:
            return result
        values = tuple(result)
        if len(values) != 2 or any(value.device.type != "cuda" for value in values):
            self.pool.stats.unexpected_cpu_storage_calls += 1
            raise RuntimeError("H3 staged FA3 cache expected two CUDA outputs")
        self.storage.append(self.pool.stage_forward(self.block_index, values))
        return result


class _StagedCachedMode(TorchDispatchMode):
    def __init__(self, storage: deque[_PageableCacheEntry], pool: _PageableStagingPool):
        super().__init__()
        self.storage = storage
        self.pool = pool

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        del types
        if func != _fa3_forward_compact._opoverload:
            return func(*args, **({} if kwargs is None else kwargs))
        if not self.storage:
            self.pool.stats.unexpected_cpu_storage_calls += 1
            raise RuntimeError("H3 staged FA3 replay had no host entry")
        return self.pool.replay(self.storage.popleft())


def _create_staged_cache_contexts(
    registration: "FA3ReplayCacheRegistration", block_index: int
) -> tuple[TorchDispatchMode, TorchDispatchMode]:
    if registration.staged_pool is None:
        registration.stats.unexpected_cpu_storage_calls += 1
        raise RuntimeError("H3 staged FA3 cache has no pageable/staging pool")
    storage: deque[_PageableCacheEntry] = deque()
    return (
        _StagedCachingMode(storage, registration.staged_pool, block_index),
        _StagedCachedMode(storage, registration.staged_pool),
    )


@dataclasses.dataclass
class FA3ReplayCacheRegistration:
    enabled: bool
    block_indices: tuple[int, ...]
    transformer: torch.nn.Module | None
    original_checkpoint: Callable[..., Any] | None
    original_kernel: Callable[..., Any] | None
    parent_split_registration: Any | None
    pre_hook: Any | None
    post_hook: Any | None
    stats: FA3ReplayCacheStats
    raw_forward_schema: str | None = None
    raw_backward_schema: str | None = None
    storage: str = "cuda"
    cpu_pool: _PinnedPool | None = None
    staged_pool: _PageableStagingPool | None = None
    max_d2h_inflight: int = 2
    trim_before_backward: bool = False

    def snapshot(self) -> dict[str, int]:
        return dataclasses.asdict(self.stats)

    def receipt(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "block_indices": list(self.block_indices),
            "compact_forward_schema": COMPACT_FORWARD_SCHEMA,
            "raw_forward_schema": self.raw_forward_schema,
            "raw_backward_schema": self.raw_backward_schema,
            "storage": self.storage,
            "max_d2h_inflight": self.max_d2h_inflight,
            "trim_before_backward": self.trim_before_backward,
            "stats": self.snapshot(),
        }


def parse_block_indices(value: str | Iterable[int] | None) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return ()
        parsed: list[int] = []
        for field in value.split(","):
            field = field.strip()
            if not field:
                continue
            if "-" in field:
                first, last = (int(part) for part in field.split("-", 1))
                if last < first:
                    raise ValueError(f"descending FA3 replay-cache range: {field}")
                parsed.extend(range(first, last + 1))
            else:
                parsed.append(int(field))
    else:
        parsed = [int(item) for item in value]
    result = tuple(sorted(set(parsed)))
    if len(result) != len(parsed):
        raise ValueError(f"duplicate FA3 replay-cache blocks: {parsed}")
    if any(index < 0 or index >= EXPECTED_BLOCK_COUNT for index in result):
        raise ValueError(f"FA3 replay-cache blocks must be in [0,49], got {result}")
    return result


def parse_storage(value: str | None) -> str:
    storage = "cuda" if value is None else str(value).strip().lower()
    if storage not in SUPPORTED_STORAGE:
        raise ValueError(
            f"FA3 replay-cache storage must be one of {SUPPORTED_STORAGE}, got {value!r}"
        )
    return storage


def parse_max_d2h_inflight(value: int | str | None) -> int:
    parsed = 2 if value is None else int(value)
    if parsed < 1:
        raise ValueError(f"FA3 replay-cache max D2H in-flight must be >=1, got {value!r}")
    return parsed


def trim_allocator_before_backward(
    registration: "FA3ReplayCacheRegistration | None",
) -> bool:
    """Release inactive allocator segments for the bounded CPU cache path."""
    if (
        registration is None
        or not registration.enabled
        or not registration.trim_before_backward
    ):
        return False
    if registration.storage not in {"cpu", "cpu_staged"}:
        raise RuntimeError(
            "FA3 replay-cache allocator trim is authorized only for CPU storage"
        )
    torch.cuda.empty_cache()
    registration.stats.allocator_trim_calls += 1
    return True


def prepare_staged_cache_before_backward(
    registration: "FA3ReplayCacheRegistration | None",
) -> bool:
    """Prime the first pageable cache entry before checkpoint replay.

    The operation is default-off and only applies to ``cpu_staged``. It does
    not allocate a GPU cache entry: one host worker copies the highest selected
    block from its pageable reservoir into the bounded pinned stage. Subsequent
    blocks are primed from replay while the current block is being recomputed.
    """
    if registration is None or not registration.enabled:
        return False
    if registration.storage != "cpu_staged":
        return False
    if registration.staged_pool is None:
        registration.stats.unexpected_cpu_storage_calls += 1
        raise RuntimeError("H3 staged FA3 cache registration has no staging pool")
    registration.staged_pool.prefetch_first_before_backward()
    return True


def _schema(value: Any) -> str | None:
    schema = getattr(value, "_schema", None)
    return None if schema is None else str(schema)


def install_fa3_replay_cache(
    transformer: torch.nn.Module,
    *,
    block_indices: str | Iterable[int] | None,
    parent_split_registration: Any,
    kernel_config: Any | None = None,
    raw_forward: Callable[..., Any] | None = None,
    raw_backward: Callable[..., Any] | None = None,
    storage: str = "cuda",
    max_d2h_inflight: int = 2,
    trim_before_backward: bool = False,
) -> FA3ReplayCacheRegistration:
    """Install the exact replay cache after Grid scope and split2 wrappers."""

    global _RUNTIME
    selected = parse_block_indices(block_indices)
    storage = parse_storage(storage)
    max_d2h_inflight = parse_max_d2h_inflight(max_d2h_inflight)
    stats = FA3ReplayCacheStats()
    if not selected:
        return FA3ReplayCacheRegistration(
            False, (), None, None, None, None, None, None, stats
        )
    if _RUNTIME is not None:
        raise RuntimeError("H3 FA3 replay cache was installed more than once")
    blocks = list(getattr(transformer, "transformer_blocks", ()))
    if len(blocks) != EXPECTED_BLOCK_COUNT:
        raise RuntimeError(
            f"H3 FA3 replay cache requires {EXPECTED_BLOCK_COUNT} blocks, got {len(blocks)}"
        )
    original_checkpoint = getattr(transformer, "_gradient_checkpointing_func", None)
    if not callable(original_checkpoint):
        raise RuntimeError("H3 FA3 replay cache requires callable checkpoint function")
    if parent_split_registration is None or not bool(
        getattr(parent_split_registration, "enabled", False)
    ):
        raise RuntimeError("H3 FA3 replay cache requires the active FA3 split registration")
    if int(getattr(parent_split_registration, "grad_num_splits", -1)) != 2:
        raise RuntimeError("H3 FA3 replay cache requires grad num_splits=2")

    if kernel_config is None:
        from diffusers.models.attention_dispatch import (
            AttentionBackendName,
            _HUB_KERNELS_REGISTRY,
        )

        kernel_config = _HUB_KERNELS_REGISTRY[AttentionBackendName._FLASH_3_HUB]
    original_kernel = getattr(kernel_config, "kernel_fn", None)
    if not callable(original_kernel):
        raise RuntimeError("H3 FA3 replay cache requires a bound kernel function")
    if getattr(original_kernel, "_h3_fa3_replay_cache_wrapper", False):
        raise RuntimeError("H3 FA3 replay cache kernel wrapper was installed twice")
    raw_forward = raw_forward or getattr(kernel_config, "wrapped_forward_fn", None)
    raw_backward = raw_backward or getattr(kernel_config, "wrapped_backward_fn", None)
    if not callable(raw_forward) or not callable(raw_backward):
        raise RuntimeError("H3 FA3 replay cache requires pinned raw forward/backward ops")

    staged_pool = (
        _PageableStagingPool(stats, selected, max_d2h_inflight)
        if storage == "cpu_staged"
        else None
    )
    registration = FA3ReplayCacheRegistration(
        enabled=True,
        block_indices=selected,
        transformer=transformer,
        original_checkpoint=original_checkpoint,
        original_kernel=original_kernel,
        parent_split_registration=parent_split_registration,
        pre_hook=None,
        post_hook=None,
        stats=stats,
        raw_forward_schema=_schema(raw_forward),
        raw_backward_schema=_schema(raw_backward),
        storage=storage,
        cpu_pool=(
            _PinnedPool(stats, max_d2h_inflight) if storage == "cpu" else None
        ),
        staged_pool=staged_pool,
        max_d2h_inflight=max_d2h_inflight,
        trim_before_backward=bool(trim_before_backward),
    )
    _RUNTIME = _Runtime(raw_forward, raw_backward, stats, staged_pool)

    @functools.wraps(original_kernel)
    def kernel_wrapper(
        q,
        k,
        v,
        softmax_scale=None,
        causal=False,
        qv=None,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        window_size=(-1, -1),
        attention_chunk=0,
        softcap=0.0,
        num_splits=1,
        pack_gqa=None,
        deterministic=False,
        sm_margin=0,
        return_attn_probs=False,
    ):
        if not _CACHE_ACTIVE.get():
            return original_kernel(
                q,
                k,
                v,
                softmax_scale=softmax_scale,
                causal=causal,
                qv=qv,
                q_descale=q_descale,
                k_descale=k_descale,
                v_descale=v_descale,
                window_size=window_size,
                attention_chunk=attention_chunk,
                softcap=softcap,
                num_splits=num_splits,
                pack_gqa=pack_gqa,
                deterministic=deterministic,
                sm_margin=sm_margin,
                return_attn_probs=return_attn_probs,
            )
        errors = []
        if not torch.is_grad_enabled():
            errors.append("grad_disabled")
        if qv is not None or q_descale is not None or k_descale is not None or v_descale is not None:
            errors.append("optional_qv_or_descale")
        if bool(causal) or tuple(window_size) != (-1, -1) or int(attention_chunk) != 0:
            errors.append("mask_or_chunk")
        if float(softcap) != 0.0 or int(num_splits) != 1:
            errors.append("softcap_or_input_splits")
        if pack_gqa is not None or bool(deterministic) or int(sm_margin) != 0:
            errors.append("pack_deterministic_or_margin")
        if bool(return_attn_probs):
            errors.append("return_attn_probs")
        if q.ndim != 4 or tuple(q.shape[:1] + q.shape[2:]) != (
            1,
            EXPECTED_HEADS,
            EXPECTED_HEAD_DIM,
        ):
            errors.append(f"q_shape={tuple(q.shape)}")
        if k.shape != q.shape or v.shape != q.shape:
            errors.append("kv_shape")
        if errors:
            stats.unexpected_kernel_contract_calls += 1
            raise RuntimeError("H3 FA3 replay cache kernel contract failed: " + ",".join(errors))

        # The parent split wrapper is intentionally bypassed for this exact
        # compact implementation, so retain its entry-level physical census.
        parent = parent_split_registration.stats
        parent.total_calls += 1
        parent.grad_calls += 1
        parent.rewritten_calls += 1
        parent.grad_rewritten_calls += 1
        stats.compact_attention_entries += 1
        scale = q.shape[-1] ** (-0.5) if softmax_scale is None else float(softmax_scale)
        return _CompactFA3Autograd.apply(q, k, v, float(scale))

    kernel_wrapper._h3_fa3_replay_cache_wrapper = True
    kernel_config.kernel_fn = kernel_wrapper

    state = {"grad": False, "call_index": 0, "selected": 0}
    cuda_context_fn = functools.partial(
        create_selective_checkpoint_contexts, [_fa3_forward_compact._opoverload]
    )

    def pre_hook(_module: Any, _inputs: tuple[Any, ...]) -> None:
        state["grad"] = bool(torch.is_grad_enabled())
        state["call_index"] = 0
        state["selected"] = 0
        if state["grad"]:
            if registration.staged_pool is not None:
                registration.staged_pool.begin_graph()
            stats.grad_transformer_forwards += 1

    def post_hook(_module: Any, _inputs: tuple[Any, ...], _output: Any) -> None:
        if not state["grad"]:
            return
        if state["call_index"] != EXPECTED_BLOCK_COUNT or state["selected"] != len(selected):
            stats.unexpected_checkpoint_contract_calls += 1
            raise RuntimeError(
                "H3 FA3 replay cache checkpoint contract failed: "
                f"calls={state['call_index']} selected={state['selected']} "
                f"expected={EXPECTED_BLOCK_COUNT}/{len(selected)}"
            )
        stats.completed_grad_transformer_forwards += 1

    def checkpoint_wrapper(function: Callable[..., Any], *args: Any, **kwargs: Any):
        if not torch.is_grad_enabled():
            return original_checkpoint(function, *args, **kwargs)
        index = int(state["call_index"])
        state["call_index"] = index + 1
        if index not in selected:
            return original_checkpoint(function, *args, **kwargs)
        if "context_fn" in kwargs or "use_reentrant" in kwargs:
            stats.unexpected_checkpoint_contract_calls += 1
            raise RuntimeError(
                "H3 FA3 replay cache observed an existing checkpoint policy override"
            )
        state["selected"] += 1
        stats.selected_checkpoint_wraps += 1
        if registration.storage == "cuda":
            context_fn = cuda_context_fn
        elif registration.storage == "cpu":
            context_fn = functools.partial(_create_cpu_cache_contexts, registration, index)
        else:
            context_fn = functools.partial(_create_staged_cache_contexts, registration, index)

        def scoped(*fn_args: Any, **fn_kwargs: Any):
            stats.selected_scoped_executions += 1
            active_token = _CACHE_ACTIVE.set(True)
            block_token = _CACHE_BLOCK_INDEX.set(index)
            try:
                return function(*fn_args, **fn_kwargs)
            finally:
                _CACHE_BLOCK_INDEX.reset(block_token)
                _CACHE_ACTIVE.reset(active_token)

        # Pinned Diffusers wraps torch.checkpoint in a closure that accepts
        # only ``(module, *args)`` and therefore cannot forward context_fn.
        # Use the exact same non-reentrant policy directly for selected calls.
        # The Grid key wrapper is installed outside this wrapper, so its
        # scoped function remains the callable checkpointed here.
        return checkpoint(
            scoped,
            *args,
            use_reentrant=False,
            context_fn=context_fn,
            **kwargs,
        )

    registration.pre_hook = transformer.register_forward_pre_hook(pre_hook)
    registration.post_hook = transformer.register_forward_hook(post_hook, always_call=True)
    transformer._gradient_checkpointing_func = checkpoint_wrapper
    logger.info(
        "[h3-a100][fa3-replay-cache] installed blocks={} storage={} "
        "max_d2h_inflight={} "
        "trim_before_backward={} cached_outputs=out+lse split=2 "
        "pageable_pool_bytes={} pinned_stage_bytes={}",
        list(selected),
        storage,
        max_d2h_inflight,
        bool(trim_before_backward),
        stats.cpu_pool_allocated_bytes,
        stats.cpu_stage_allocated_bytes,
    )
    return registration


def validate_cycle(
    registration: FA3ReplayCacheRegistration, start: dict[str, int]
) -> dict[str, int]:
    if not registration.enabled:
        return {}
    current = registration.snapshot()
    delta = {key: int(current[key]) - int(start.get(key, 0)) for key in current}
    selected_per_cycle = len(registration.block_indices) * EXPECTED_GRAD_TRANSFORMER_FORWARDS
    expected = {
        "grad_transformer_forwards": EXPECTED_GRAD_TRANSFORMER_FORWARDS,
        "completed_grad_transformer_forwards": EXPECTED_GRAD_TRANSFORMER_FORWARDS,
        "selected_checkpoint_wraps": selected_per_cycle,
        "selected_scoped_executions": 2 * selected_per_cycle,
        "compact_attention_entries": 2 * selected_per_cycle,
        "compact_forward_impl_calls": selected_per_cycle,
        "compact_backward_calls": selected_per_cycle,
        "unexpected_kernel_contract_calls": 0,
        "unexpected_checkpoint_contract_calls": 0,
        "allocator_trim_calls": (
            EXPECTED_GRAD_TRANSFORMER_FORWARDS
            if registration.trim_before_backward
            else 0
        ),
        "unexpected_cpu_storage_calls": 0,
    }
    if registration.storage in {"cpu", "cpu_staged"}:
        expected.update(
            {
                "cpu_d2h_entries": selected_per_cycle,
                "cpu_h2d_entries": selected_per_cycle,
                "cpu_d2h_tensors": 2 * selected_per_cycle,
                "cpu_h2d_tensors": 2 * selected_per_cycle,
            }
        )
        if registration.storage == "cpu_staged":
            expected.update(
                {
                    "cpu_forward_host_copy_entries": selected_per_cycle,
                    "cpu_backward_host_copy_entries": selected_per_cycle,
                    "cpu_backward_prefetches": selected_per_cycle,
                    "cpu_backward_prefetch_misses": 0,
                }
            )
    else:
        expected.update(
            {
                "cpu_d2h_entries": 0,
                "cpu_h2d_entries": 0,
                "cpu_d2h_tensors": 0,
                "cpu_h2d_tensors": 0,
                "cpu_d2h_bytes": 0,
                "cpu_h2d_bytes": 0,
                "cpu_pool_allocated_bytes": 0,
                "cpu_pool_reused_tensors": 0,
                "cpu_pool_busy_misses": 0,
            }
        )
    errors = [
        f"{key}={delta[key]} expected={value}"
        for key, value in expected.items()
        if delta[key] != value
    ]
    if delta["cached_logical_bytes"] <= 0:
        errors.append("cached_logical_bytes=0")
    if registration.storage in {"cpu", "cpu_staged"}:
        if delta["cpu_d2h_bytes"] != delta["cached_logical_bytes"]:
            errors.append(
                "cpu_d2h_bytes="
                f"{delta['cpu_d2h_bytes']} expected={delta['cached_logical_bytes']}"
            )
        if delta["cpu_h2d_bytes"] != delta["cached_logical_bytes"]:
            errors.append(
                "cpu_h2d_bytes="
                f"{delta['cpu_h2d_bytes']} expected={delta['cached_logical_bytes']}"
            )
        if registration.storage == "cpu_staged":
            if delta["cpu_forward_host_copy_bytes"] != delta["cached_logical_bytes"]:
                errors.append(
                    "cpu_forward_host_copy_bytes="
                    f"{delta['cpu_forward_host_copy_bytes']} "
                    f"expected={delta['cached_logical_bytes']}"
                )
            if delta["cpu_backward_host_copy_bytes"] != delta["cached_logical_bytes"]:
                errors.append(
                    "cpu_backward_host_copy_bytes="
                    f"{delta['cpu_backward_host_copy_bytes']} "
                    f"expected={delta['cached_logical_bytes']}"
                )
    if errors:
        raise RuntimeError("H3 FA3 replay cache cycle failed: " + "; ".join(errors))
    return delta
