"""C27 immutable pinned CPU prompt staging. Default uninstalled.

This is not a GPU input cache: every call still performs the original FP32 H2D
on the caller's current stream. Keep the pinned source alive until teardown.
"""
from __future__ import annotations

from typing import Any


NAME = "score_immutable_pinned_prompt_h2d_v1"
SHAPE = (512, 4096)
BYTES_PER_REQUEST = 512 * 4096 * 4


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
