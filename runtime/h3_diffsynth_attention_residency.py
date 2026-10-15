"""Bounded GPU storage for exact native attention cache entries.

Only storage changes: the pinned nine-result Flash ABI and native backward stay
in the existing source-validated cache. CPU entries retain its original FIFO and
one-block prefetch. A budget is shared by the two independent logical-slot pools.
"""
from __future__ import annotations

import collections


def storage_bytes(tensor):
    if tensor.numel() == 0:
        return 0
    if any(stride < 0 for stride in tensor.stride()):
        raise ValueError("negative cache stride is unsupported")
    extent = 1 + sum((size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride()))
    return extent * tensor.element_size()


class GPUCacheBudget:
    def __init__(self, capacity_bytes):
        if capacity_bytes <= 0:
            raise ValueError("positive GPU cache budget required")
        self.capacity_bytes = int(capacity_bytes)
        self.owned_bytes = 0
        self.peak_owned_bytes = 0

    def acquire(self, size):
        if size < 0 or self.owned_bytes + size > self.capacity_bytes:
            raise RuntimeError("GPU_CACHE_BUDGET_EXCEEDED_NO_SILENT_POLICY_FALLBACK")
        self.owned_bytes += size
        self.peak_owned_bytes = max(self.peak_owned_bytes, self.owned_bytes)

    def release(self, size):
        if not 0 <= size <= self.owned_bytes:
            raise RuntimeError("GPU cache ownership accounting mismatch")
        self.owned_bytes -= size


def resident_pool_type(prefetch_module, *, tail_blocks, budget):
    """Extend the verified pool without modifying any shared helper module."""
    if tail_blocks <= 0:
        raise ValueError("positive tail-block count required")
    import torch
    from torch.utils._pytree import tree_flatten
    from native_sdpa_cpu_replay_r1 import signature

    class ResidentTailPool(prefetch_module.PrefetchPool):
        def __init__(self, *, enabled=False):
            super().__init__(enabled=enabled)
            self.residency = collections.Counter()

        def _resident(self, namespace):
            if not 0 < tail_blocks <= len(self.blocks):
                raise RuntimeError("GPU cache policy exceeds native block count")
            role, index = namespace
            if role not in ("studentG0", "fakeG0") or not 0 <= index < len(self.blocks):
                raise RuntimeError("unexpected native cache namespace")
            return index >= len(self.blocks) - tail_blocks

        def store(self, namespace, ordinal, args, kwargs, result):
            if not self._resident(namespace):
                return super().store(namespace, ordinal, args, kwargs, result)
            leaves, spec = tree_flatten(result)
            tensors = [value for value in leaves if isinstance(value, torch.Tensor)]
            native = [value for value in tensors if value.is_cuda or self.cpu_test]
            if not native or any(value.device != native[0].device for value in native):
                raise RuntimeError("resident entry must have one native device")
            device = native[0].device
            charged = sum(storage_bytes(value) for value in native)
            logical = sum(value.numel() * value.element_size() for value in native)
            budget.acquire(charged)
            values = []
            try:
                for value in leaves:
                    if isinstance(value, torch.Tensor):
                        if value.device != device and value.numel() > 16:
                            raise RuntimeError("unexpected large host cache leaf")
                        # Independent storage and preserved strides prevent a
                        # native consumer from mutating the immutable source.
                        copy = torch.empty_strided(value.shape, value.stride(), dtype=value.dtype, device=value.device)
                        copy.copy_(value.detach())
                        values.append(copy)
                    else:
                        values.append(value)
                ready = None if self.cpu_test else torch.cuda.Event()
                if not self.cpu_test:
                    ready.record(torch.cuda.current_stream(device))
                    # Keep the existing source-group/FIFO accounting. This
                    # event is a GPU copy completion, not a D2H completion.
                    self.pending[device].append(ready)
            except BaseException:
                budget.release(charged)
                raise
            self.counters["cached_original_attention_calls"] += 1
            self.residency["source_entries"] += 1
            self.residency["source_logical_bytes"] += logical
            self.residency["D2D_logical_bytes"] += logical
            self.residency["avoided_D2H_logical_bytes"] += logical
            return {"buffers": [], "values": values, "spec": spec, "device": device,
                    "ready": ready, "signature": signature((args, kwargs)),
                    "GPU_cache_owned_bytes": charged, "GPU_cache_logical_bytes": logical}

        def restore(self, entry, args, kwargs):
            resident = "GPU_cache_owned_bytes" in entry
            if resident:
                # Fresh Tensor wrappers retain independent autograd metadata
                # while sharing immutable cached storage. No tensor copy here.
                entry["values"] = [value.detach() if isinstance(value, torch.Tensor) else value
                                   for value in entry["values"]]
                from selected_attention_pool_r1 import _PHASE
                key = (id(entry), entry["remaining"], _PHASE.get())
                if key in self.prefetched:
                    saved, values, ready, namespace = self.prefetched[key]
                    values = [value.detach() if isinstance(value, torch.Tensor) else value for value in values]
                    self.prefetched[key] = (saved, values, ready, namespace)
            result = super().restore(entry, args, kwargs)
            if resident:
                self.residency["consumer_entries"] += 1
                self.residency["avoided_H2D_logical_bytes"] += entry["GPU_cache_logical_bytes"]
                if entry["remaining"] == 0:
                    budget.release(entry["GPU_cache_owned_bytes"])
                    entry["values"] = [None] * len(entry["values"])
                    self.residency["released_entries"] += 1
            return result

        def finish_selected(self):
            result = super().finish_selected()
            expected_sources, expected_consumers = 4 * tail_blocks, 6 * tail_blocks
            if self.residency["source_entries"] != expected_sources or self.residency["consumer_entries"] != expected_consumers:
                raise RuntimeError("GPU cache actual native consumer census mismatch")
            if self.residency["released_entries"] != expected_sources:
                raise RuntimeError("GPU cache source ownership leaked")
            for entry in self.immutable_entries:
                if "GPU_cache_owned_bytes" in entry and any(isinstance(value, torch.Tensor) for value in entry["values"]):
                    raise RuntimeError("GPU cache entry still owns tensor storage")
            # Parent names this counter D2H_block_groups, although it actually
            # counts source groups. Do not publish GPU groups as D2H activity.
            counters = result["counters"]
            counters["source_block_groups_total"] = counters.pop("D2H_block_groups")
            counters["actual_D2H_source_block_groups"] = 2 * (len(self.blocks) - tail_blocks)
            result.update(storage="pinned_cpu_with_bounded_GPU_tail", GPU_cache_tail_blocks=tail_blocks,
                          GPU_cache=dict(self.residency), GPU_cache_budget_bytes=budget.capacity_bytes,
                          GPU_cache_peak_owned_bytes_across_logical_slots=budget.peak_owned_bytes,
                          GPU_cache_owned_bytes_at_epoch_finish=budget.owned_bytes,
                          GPU_cache_ownership_scope="cache-held storage; native autograd/allocator live bytes measured separately",
                          max_D2H_inflight="original one source group; resident groups issue D2D only")
            return result

        def reset_residency_epoch(self):
            self.residency.clear()

    return ResidentTailPool
