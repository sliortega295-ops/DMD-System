"""Bounded saved-tensor CPU offload for the H3 A100 capacity candidate.

This module is intentionally small and opt-in.  It does not change the H3
math, FSDP2 placement, checkpoint boundaries, or application-level call
counts.  During the original grad-enabled forward, saved tensors whose
storage is at least ``min_offload_bytes`` are copied to pinned CPU memory.
Before ``backward`` the scope switches to replay mode: tensors produced by
checkpoint recomputation stay on their original device, while handles created
by the original forward are still unpacked from CPU.

Keeping replay tensors on GPU is important.  A saved-tensor hook that remains
fully active during checkpoint replay would recursively offload recomputed
intermediates and can turn one bounded memory mechanism into unbounded PCIe
traffic.  Parameters and buffers are always excluded.
"""

from __future__ import annotations

import contextlib
import gc
from dataclasses import dataclass
from typing import Iterator

import torch
from loguru import logger


def _storage_key(tensor: torch.Tensor) -> tuple[str, int, int]:
    try:
        storage = tensor.untyped_storage()
        return (str(tensor.device), int(storage.data_ptr()), int(storage.nbytes()))
    except Exception:
        return (str(tensor.device), 0, 0)


def _storage_bytes(tensor: torch.Tensor) -> int:
    try:
        return int(tensor.untyped_storage().nbytes())
    except Exception:
        return int(tensor.numel() * tensor.element_size())


@dataclass
class _KeepOnDevice:
    tensor: torch.Tensor


@dataclass
class _PackedOnCPU:
    cpu_tensor: torch.Tensor
    original_device: torch.device


class SelectiveSavedTensorOffload:
    """One forward/backward scope of thresholded saved-tensor offload."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        logical_component: str,
        min_offload_bytes: int,
        pin_memory: bool = True,
    ) -> None:
        self.model = model
        self.logical_component = str(logical_component)
        self.min_offload_bytes = int(min_offload_bytes)
        self.pin_memory = bool(pin_memory)
        if self.min_offload_bytes < 0:
            raise ValueError("min_offload_bytes must be non-negative")
        self._phase = "forward"
        self._excluded: set[tuple[str, int, int]] = set()
        self._handles: list[_PackedOnCPU] = []
        self.stats = {
            "pack_count": 0,
            "unpack_count": 0,
            "offload_count": 0,
            "offloaded_storage_bytes": 0,
            "offloaded_logical_bytes": 0,
            "keep_parameter_or_buffer_count": 0,
            "keep_non_cuda_count": 0,
            "keep_non_strided_count": 0,
            "keep_zero_stride_count": 0,
            "keep_below_threshold_count": 0,
            "replay_pack_count": 0,
            "replay_unpack_count": 0,
        }

    def _refresh_excluded(self) -> None:
        excluded: set[tuple[str, int, int]] = set()
        for tensor in list(self.model.parameters()) + list(self.model.buffers()):
            if tensor.device.type == "cuda":
                excluded.add(_storage_key(tensor))
        self._excluded = excluded

    def _pack(self, tensor: torch.Tensor):
        self.stats["pack_count"] += 1
        if self._phase != "forward":
            self.stats["replay_pack_count"] += 1
            return _KeepOnDevice(tensor)
        if tensor.device.type != "cuda":
            self.stats["keep_non_cuda_count"] += 1
            return _KeepOnDevice(tensor)
        if _storage_key(tensor) in self._excluded:
            self.stats["keep_parameter_or_buffer_count"] += 1
            return _KeepOnDevice(tensor)
        if tensor.layout != torch.strided:
            self.stats["keep_non_strided_count"] += 1
            return _KeepOnDevice(tensor)
        if any(size > 1 and stride == 0 for size, stride in zip(tensor.shape, tensor.stride())):
            self.stats["keep_zero_stride_count"] += 1
            return _KeepOnDevice(tensor)
        storage_bytes = _storage_bytes(tensor)
        if storage_bytes < self.min_offload_bytes:
            self.stats["keep_below_threshold_count"] += 1
            return _KeepOnDevice(tensor)

        detached = tensor.detach()
        # Preserve the original view shape/stride.  The saved-tensor API only
        # requires an unpacked tensor with the same logical metadata; copying
        # the view's storage also avoids aliasing the live GPU activation.
        cpu_tensor = torch.empty_strided(
            tuple(detached.shape),
            tuple(detached.stride()),
            dtype=detached.dtype,
            device="cpu",
            pin_memory=self.pin_memory,
        )
        cpu_tensor.copy_(detached, non_blocking=False)
        packed = _PackedOnCPU(cpu_tensor, tensor.device)
        self._handles.append(packed)
        self.stats["offload_count"] += 1
        self.stats["offloaded_storage_bytes"] += int(cpu_tensor.untyped_storage().nbytes())
        self.stats["offloaded_logical_bytes"] += int(tensor.numel() * tensor.element_size())
        return packed

    def _unpack(self, handle):
        self.stats["unpack_count"] += 1
        if isinstance(handle, _KeepOnDevice):
            if self._phase != "forward":
                self.stats["replay_unpack_count"] += 1
            return handle.tensor
        if not isinstance(handle, _PackedOnCPU):
            raise TypeError(f"unexpected saved-tensor handle: {type(handle)!r}")
        return handle.cpu_tensor.to(handle.original_device, non_blocking=False)

    def begin_backward(self) -> None:
        """Stop offloading checkpoint-replay intermediates."""

        if self._phase != "forward":
            raise RuntimeError(f"saved-tensor offload phase is already {self._phase!r}")
        self._phase = "replay"

    @contextlib.contextmanager
    def context(self) -> Iterator["SelectiveSavedTensorOffload"]:
        self._refresh_excluded()
        try:
            with torch.autograd.graph.saved_tensors_hooks(self._pack, self._unpack):
                yield self
        finally:
            logger.info(
                "[h3-a100][activation-offload] component={} phase={} stats={}",
                self.logical_component,
                self._phase,
                self.stats,
            )
            # Release CPU copies before returning to the next logical update.
            self._handles.clear()
            gc.collect()


@contextlib.contextmanager
def maybe_saved_tensor_offload(
    model: torch.nn.Module,
    *,
    enabled: bool,
    logical_component: str,
    min_offload_bytes: int,
    pin_memory: bool = True,
) -> Iterator[SelectiveSavedTensorOffload | None]:
    """Yield a scope when enabled, otherwise a no-op ``None`` scope."""

    if not enabled:
        yield None
        return
    scope = SelectiveSavedTensorOffload(
        model,
        logical_component=logical_component,
        min_offload_bytes=min_offload_bytes,
        pin_memory=pin_memory,
    )
    with scope.context():
        yield scope
