"""C27 immutable pinned CPU prompt staging. Default uninstalled.

This is not a GPU input cache: every call still performs the original FP32 H2D
on the caller's current stream. Keep the pinned source alive until teardown.
"""
from __future__ import annotations

import hashlib
import inspect
from typing import Any


NAME = "score_immutable_pinned_prompt_h2d_v1"
SHAPE = (512, 4096)
BYTES_PER_REQUEST = 512 * 4096 * 4
SOURCE_SHA256 = "13bf037518f10dbf1b020edd5fefebfdaed6938ed011125a88dd113a09365c02"
PLAN_CONTRACT = {"enabled": True, "name": NAME, "scope": "score_ranks_only",
                 "source_context_method_sha256": SOURCE_SHA256,
                 "cpu_cache_dtype": "float32", "gpu_input_cache": False,
                 "h2d_per_request": True, "pin_position": "first_consumption_inside_cycle"}


class PinnedScoreContextStager:
    def __init__(self, torch: Any, prompts: Any, *, allowed_indices: set[int]):
        if not allowed_indices or any(not isinstance(i, int) or i < 0 for i in allowed_indices):
            raise ValueError("C27 requires explicit nonempty trace dataset identities")
        self.torch = torch
        self.prompts = prompts
        self.allowed_indices = frozenset(allowed_indices)
        self.cache: dict[int, Any] = {}
        self.calls = 0
        self.hits = 0
        self.failures = 0

    def context_from_trace(self, row: Any, *, device: Any):
        try:
            index = row.dataset_index
            if index not in self.allowed_indices or self.torch.device(device).type != "cuda":
                raise RuntimeError("C27 unregistered sample/device")
            if index not in self.cache:
                original = self.prompts["pos"][index]
                if (original.device.type != "cpu" or original.requires_grad
                    or tuple(original.shape) != SHAPE):
                    raise RuntimeError("C27 immutable CPU prompt shape/grad/device changed")
                self.cache[index] = original.float().pin_memory()
            else:
                self.hits += 1
            context = self.cache[index]
            if not context.is_pinned() or context.dtype != self.torch.float32:
                raise RuntimeError("C27 pinned FP32 input storage changed")
            result = context.to(device, non_blocking=True).unsqueeze(0)
            self.calls += 1
            return result
        except Exception:
            self.failures += 1
            raise

    def receipt(self) -> dict[str, Any]:
        return {"name": NAME, "requests": self.calls, "cache_hits": self.hits,
                "cache_entries": len(self.cache), "contract_failures": self.failures,
                "pinned_cpu_bytes": len(self.cache) * BYTES_PER_REQUEST,
                "h2d_bytes": self.calls * BYTES_PER_REQUEST,
                "gpu_input_cache": False, "h2d_per_request": True,
                "pin_position": "FIRST_CONSUMPTION_INSIDE_CYCLE",
                "dtype": "torch.float32", "shape": [1, *SHAPE]}


def install(torch: Any, loader: Any, *, allowed_indices: set[int]):
    observed = hashlib.sha256(inspect.getsource(loader.context_from_trace).encode()).hexdigest()
    if observed != SOURCE_SHA256:
        raise RuntimeError(f"C27 original context source changed: {observed}")
    helper = PinnedScoreContextStager(torch, loader.prompts, allowed_indices=allowed_indices)
    loader.context_from_trace = helper.context_from_trace
    return helper


def runtime_receipt_errors(r: dict[str, Any], *, rank: int, cycles: int) -> list[str]:
    if rank < 8:
        return [] if r.get("execution_status") == "NOT_APPLICABLE" else ["C27 Generator input path changed"]
    count = cycles * 10
    if (r.get("name") != NAME or r.get("execution_status") != "PASS"
        or r.get("source_context_method_sha256") != SOURCE_SHA256
        or r.get("requests") != count or r.get("h2d_bytes") != count * BYTES_PER_REQUEST
        or r.get("cache_entries") != 10 or r.get("cache_hits") != count - 10
        or r.get("pinned_cpu_bytes") != 10 * BYTES_PER_REQUEST
        or r.get("contract_failures") != 0 or r.get("gpu_input_cache") is not False
        or r.get("h2d_per_request") is not True):
        return ["C27 input staging source/count/bytes contract failed"]
    return []
