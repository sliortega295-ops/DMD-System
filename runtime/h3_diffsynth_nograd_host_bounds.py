"""Default-off, call-local immutable packed bounds reuse; no tensor math changes.

The pinned A100 no-grad host-scope mechanism is adapted to FSDP hooks rather
than ZeRO leaf hooks. Grad/recompute keep the original metadata reads. Four
outstanding no-grad blocks bound host run-ahead; no parameter prefetch is added.
"""
from __future__ import annotations

from collections import Counter, deque
from contextvars import ContextVar
import functools
import hashlib
import inspect
from pathlib import Path
import textwrap

DIT_SHA = "bd617a4312c70405e916707e1908100b484b2259dfbeba4af7de5bb145206d80"


class BoundsFrame:
    """Hold source objects, not just reusable allocator pointers, for one call."""

    def __init__(self, counts):
        self.entries = {}
        self.counts = counts

    def read(self, tensor):
        key = id(tensor)
        if key in self.entries:
            source, version, values = self.entries[key]
            if source is not tensor or tensor._version != version:
                raise RuntimeError("IMMUTABLE_PACKED_BOUNDS_MUTATED")
            self.counts["cached_bounds_hits"] += 1
            return values
        version = tensor._version
        values = tensor.tolist()
        if tensor._version != version:
            raise RuntimeError("PACKED_BOUNDS_CHANGED_DURING_HOST_READ")
        self.entries[key] = (tensor, version, values)
        self.counts["actual_nograd_bounds_host_reads"] += 1
        return values

    def close(self):
        for tensor, version, _ in self.entries.values():
            if tensor._version != version:
                raise RuntimeError("IMMUTABLE_PACKED_BOUNDS_MUTATED_AT_EXIT")
        self.entries.clear()


class BlockQueue:
    def __init__(self, event_factory, counts, *, window=4):
        if window != 4:
            raise ValueError("ONLY_PREREGISTERED_FOUR_BLOCK_HOST_WINDOW")
        self.event_factory = event_factory
        self.counts = counts
        self.events = deque()

    def retire(self):
        while self.events and self.events[0].query():
            self.events.popleft()

    def before(self):
        self.retire()
        if len(self.events) >= 4:
            self.events.popleft().synchronize()
            self.counts["oldest_nograd_block_host_waits"] += 1

    def after(self):
        event = self.event_factory()
        event.record()
        self.events.append(event)
        self.counts["nograd_main_blocks"] += 1
        self.counts["peak_queued_blocks"] = max(self.counts["peak_queued_blocks"], len(self.events))
        if len(self.events) > 4:
            raise RuntimeError("NOGRAD_HOST_QUEUE_OVERFLOW")


class NativeNoGradHostBounds:
    def __init__(self, native, dit):
        if hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest() != DIT_SHA:
            raise ValueError("PINNED_NATIVE_DIT_SOURCE_MISMATCH")
        if len(dit.blocks) not in (2, 50):
            raise ValueError("PINNED_TINY2_OR_REAL50_BLOCKS_REQUIRED")
        original_attention = native._sdpa_varlen_attention
        original_forward = native.MiniMaxH3DiT.forward
        body = textwrap.dedent(inspect.getsource(original_attention))
        old = "bounds = cu_seqlens.tolist()"
        if body.count(old) != 1:
            raise ValueError("PINNED_NATIVE_BOUNDS_READ_SITE_MISMATCH")
        import torch

        self.counts = Counter()
        self.previous = Counter()
        self.active_calls = 0
        self.scope = ContextVar("h100_w6_nograd_packed_bounds", default=None)
        self.queue = BlockQueue(lambda: torch.cuda.Event(enable_timing=False), self.counts)

        def read_bounds(tensor):
            frame = self.scope.get()
            if frame is None or torch.is_grad_enabled():
                self.counts["original_grad_or_unscoped_bounds_reads"] += 1
                return tensor.tolist()
            return frame.read(tensor)

        env = dict(original_attention.__globals__, _h100_read_bounds=read_bounds)
        exec(compile(body.replace(old, "bounds = _h100_read_bounds(cu_seqlens)"),
                     "<native_nograd_bounds_read_only>", "exec"), env)
        native._sdpa_varlen_attention = env[original_attention.__name__]

        @functools.wraps(original_forward)
        def forward(model, *args, **kwargs):
            if torch.is_grad_enabled():
                return original_forward(model, *args, **kwargs)
            if self.scope.get() is not None:
                raise RuntimeError("NO_NESTED_NATIVE_DIT_BOUNDS_SCOPE")
            frame = BoundsFrame(self.counts)
            token = self.scope.set(frame)
            self.active_calls += 1
            self.counts["nograd_transformer_calls"] += 1
            try:
                return original_forward(model, *args, **kwargs)
            finally:
                try:
                    frame.close()
                finally:
                    self.active_calls -= 1
                    self.scope.reset(token)

        native.MiniMaxH3DiT.forward = forward

        def before(module, args):
            if self.scope.get() is not None and not torch.is_grad_enabled():
                self.queue.before()

        def after(module, args, output):
            if self.scope.get() is not None and not torch.is_grad_enabled():
                self.queue.after()

        self.hooks = []
        for block in dit.blocks:
            self.hooks.append(block.register_forward_pre_hook(before, prepend=True))
            self.hooks.append(block.register_forward_hook(after))

    def finish_cycle(self):
        # The existing cycle terminal synchronize has already completed here.
        self.queue.retire()
        if self.active_calls or self.scope.get() is not None or self.queue.events:
            raise RuntimeError("NATIVE_BOUNDS_SCOPE_OR_GPU_WORK_SURVIVED_CYCLE")
        current = self.counts.copy()
        delta = current - self.previous
        self.previous = current
        return {"enabled": True, "source_sha256": DIT_SHA,
                "policy": "per_call_immutable_nograd_bounds_four_block_host_window",
                "counts": dict(current), "cycle_delta": dict(delta),
                "active_calls": 0, "queued_blocks_after_terminal_sync": 0,
                "parameter_prefetch_added": False, "grad_and_recompute_bounds": "original"}


def install(native, dit, *, enabled=False):
    if not enabled:
        return None
    return NativeNoGradHostBounds(native, dit)
