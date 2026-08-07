"""Reusable helpers from the measured official Helios A100 strong baseline.

This compatibility module deliberately keeps the portions that are consumed by
active DMD-System experiments: node construction/swap admission, stable block
noise, parameter/update fingerprints, and lifetime-neutral gradient observers.
The large historical option/launcher surface remains outside the stable API.
"""
from __future__ import annotations

import ctypes
import gc
import hashlib
import json
import math
from contextlib import contextmanager
from typing import Any, Callable, Iterable, Iterator, TypeVar

import torch

_T = TypeVar("_T")

try:
    _LIBC = ctypes.CDLL(None)
    _MALLOC_TRIM = _LIBC.malloc_trim
    _MALLOC_TRIM.argtypes = [ctypes.c_size_t]
    _MALLOC_TRIM.restype = ctypes.c_int
except (AttributeError, OSError):
    _MALLOC_TRIM = None


def trim_host_allocator() -> bool:
    """Return unused glibc arenas to the host cgroup."""
    gc.collect()
    host_empty_cache = getattr(torch._C, "_host_emptyCache", None)
    if host_empty_cache is not None:
        host_empty_cache()
    if _MALLOC_TRIM is None:
        return False
    return bool(_MALLOC_TRIM(0))


@contextmanager
def suspend_transformers_zero3_init() -> Iterator[None]:
    """Temporarily keep short-lived preprocessing models out of ZeRO-3 init."""
    from transformers.integrations import deepspeed as transformers_deepspeed

    config_ref = transformers_deepspeed._hf_deepspeed_config_weak_ref
    config = config_ref() if config_ref is not None else None
    if config is not None:
        transformers_deepspeed.unset_hf_deepspeed_config()
    try:
        yield
    finally:
        if config is not None:
            transformers_deepspeed.set_hf_deepspeed_config(config)


def swap_wave_ranges(local_world_size: int, wave_size: int) -> tuple[tuple[int, int], ...]:
    """Partition local ranks into bounded contiguous residency waves."""
    if local_world_size <= 0:
        raise ValueError("local_world_size must be positive")
    if wave_size <= 0:
        raise ValueError("wave_size must be positive")
    return tuple(
        (start, min(start + wave_size, local_world_size))
        for start in range(0, local_world_size, wave_size)
    )


def construct_in_local_rank_waves(
    factory: Callable[[], _T],
    *,
    local_rank: int,
    local_world_size: int,
    wave_size: int,
    barrier: Callable[[], None],
    on_wave_start: Callable[[int, int, int], None] | None = None,
    on_wave_end: Callable[[int, int, int], None] | None = None,
) -> _T:
    """Run one rank-local constructor while bounding node concurrency."""
    if local_world_size <= 0:
        raise ValueError("local_world_size must be positive")
    if not 0 <= local_rank < local_world_size:
        raise ValueError("local_rank must be in [0, local_world_size)")
    if wave_size <= 0:
        return factory()

    result: _T | None = None
    constructed = False
    for wave_index, (start, end) in enumerate(
        swap_wave_ranges(local_world_size, wave_size)
    ):
        barrier()
        active = start <= local_rank < end
        if active:
            if on_wave_start is not None:
                on_wave_start(wave_index, start, end)
            try:
                result = factory()
                constructed = True
            finally:
                if on_wave_end is not None:
                    on_wave_end(wave_index, start, end)
        barrier()
    if not constructed:
        raise RuntimeError("local rank was not admitted by any construction wave")
    return result  # type: ignore[return-value]


def plan_swap_wave_size(
    *,
    local_world_size: int,
    memory_limit_bytes: int,
    steady_state_bytes: int,
    transient_per_rank_bytes: int,
    reserve_bytes: int,
) -> int:
    """Choose the largest local rank wave admitted by a measured host budget."""
    values = {
        "local_world_size": local_world_size,
        "memory_limit_bytes": memory_limit_bytes,
        "steady_state_bytes": steady_state_bytes,
        "transient_per_rank_bytes": transient_per_rank_bytes,
        "reserve_bytes": reserve_bytes,
    }
    if local_world_size <= 0:
        raise ValueError("local_world_size must be positive")
    for name in ("memory_limit_bytes", "transient_per_rank_bytes"):
        if values[name] <= 0:
            raise ValueError(f"{name} must be positive")
    for name in ("steady_state_bytes", "reserve_bytes"):
        if values[name] < 0:
            raise ValueError(f"{name} must be non-negative")
    available = memory_limit_bytes - steady_state_bytes - reserve_bytes
    admitted = available // transient_per_rank_bytes
    if admitted < 1:
        raise MemoryError(
            "host budget cannot admit one model-swap rank: "
            f"available={available}, transient_per_rank={transient_per_rank_bytes}"
        )
    return min(local_world_size, int(admitted))


def stable_block_noise(
    block_number: int,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Sample the exact PSD block covariance without a fragile Cholesky."""
    samples = torch.randn(block_number, 4, device=device, dtype=dtype)
    samples = samples - samples.mean(dim=-1, keepdim=True)
    return samples * math.sqrt(4.0 / 3.0)


def parameter_summary(module: torch.nn.Module) -> dict[str, Any]:
    logical_numel = 0
    local_numel = 0
    trainable_logical_numel = 0
    logical_bytes_by_dtype: dict[str, int] = {}
    devices: dict[str, int] = {}
    for parameter in module.parameters():
        numel = int(getattr(parameter, "ds_numel", parameter.numel()))
        logical_numel += numel
        if parameter.requires_grad:
            trainable_logical_numel += numel
        dtype = str(parameter.dtype).replace("torch.", "")
        logical_bytes_by_dtype[dtype] = logical_bytes_by_dtype.get(dtype, 0) + numel * parameter.element_size()
        backing = getattr(parameter, "ds_tensor", parameter.data)
        local_numel += backing.numel()
        device = str(backing.device)
        devices[device] = devices.get(device, 0) + backing.numel() * backing.element_size()
    return {
        "logical_numel": logical_numel,
        "local_numel": local_numel,
        "trainable_logical_numel": trainable_logical_numel,
        "logical_bytes_by_dtype": logical_bytes_by_dtype,
        "local_bytes_by_device": devices,
    }


def _backing_tensor(parameter: torch.nn.Parameter) -> torch.Tensor:
    tensor = getattr(parameter, "ds_tensor", None)
    return tensor if tensor is not None else parameter.data


@torch.no_grad()
def trainable_checksum(module: torch.nn.Module) -> dict[str, Any]:
    total_sum = 0.0
    total_sq = 0.0
    numel = 0
    tensors = 0
    for parameter in module.parameters():
        if not parameter.requires_grad:
            continue
        value = _backing_tensor(parameter).detach().float()
        total_sum += value.sum().item()
        total_sq += value.square().sum().item()
        numel += value.numel()
        tensors += 1
    return {"num_tensors": tensors, "numel": numel, "sum": total_sum, "norm": math.sqrt(total_sq)}


@torch.no_grad()
def trainable_sha256(module: torch.nn.Module) -> str:
    """Hash trainable parameter names, metadata, and bytes one tensor at a time."""
    digest = hashlib.sha256()
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        value = _backing_tensor(parameter).detach().to(device="cpu").contiguous()
        metadata = {
            "name": name,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "stride": list(value.stride()),
        }
        digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode())
        digest.update(b"\0")
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def parameter_gradient_diagnostics(module: torch.nn.Module) -> dict[str, int]:
    """Count gradient ownership without retaining or materializing gradients."""
    result = {
        "trainable_parameters": 0,
        "trainable_grad_present": 0,
        "frozen_parameters": 0,
        "frozen_grad_present": 0,
    }
    for parameter in module.parameters():
        prefix = "trainable" if parameter.requires_grad else "frozen"
        result[f"{prefix}_parameters"] += 1
        if parameter.grad is not None:
            result[f"{prefix}_grad_present"] += 1
    return result


class TrainableGradientObserver:
    """Observe autograd delivery without retaining gradient tensors."""
    def __init__(self, module: torch.nn.Module):
        self._active = False
        self._hook_calls = 0
        self._gradient_numel = 0
        self._parameter_names: set[str] = set()
        self._handles = []
        self.trainable_parameter_count = 0
        for name, parameter in module.named_parameters():
            if not parameter.requires_grad:
                continue
            self.trainable_parameter_count += 1

            def observe(gradient: torch.Tensor, *, parameter_name: str = name) -> torch.Tensor:
                if self._active:
                    self._hook_calls += 1
                    self._gradient_numel += gradient.numel()
                    self._parameter_names.add(parameter_name)
                return gradient

            self._handles.append(parameter.register_hook(observe))
        if not self._handles:
            raise ValueError("gradient observer found no trainable parameters")

    def begin(self) -> None:
        if self._active:
            raise RuntimeError("gradient observer scope is already active")
        self._hook_calls = 0
        self._gradient_numel = 0
        self._parameter_names.clear()
        self._active = True

    def end(self) -> dict[str, int]:
        if not self._active:
            raise RuntimeError("gradient observer scope is not active")
        self._active = False
        return {
            "trainable_parameter_count": self.trainable_parameter_count,
            "parameters_with_gradient": len(self._parameter_names),
            "hook_calls": self._hook_calls,
            "gradient_numel": self._gradient_numel,
        }

    def close(self) -> None:
        self._active = False
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def should_restore_dataloader_state(checkpoint_step: int, updates_per_epoch: int) -> bool:
    """Restore a mid-epoch cursor, but reset a checkpoint taken at an epoch boundary."""
    if checkpoint_step < 0:
        raise ValueError("checkpoint_step must be non-negative")
    if updates_per_epoch <= 0:
        raise ValueError("updates_per_epoch must be positive")
    return checkpoint_step % updates_per_epoch != 0


def optimizer_state_checksum(optimizer: Any) -> dict[str, Any]:
    """Summarize Adam moments after a checkpoint load without retaining tensors."""
    current = optimizer
    seen: set[int] = set()
    while not hasattr(current, "state") and hasattr(current, "optimizer"):
        if id(current) in seen:
            raise ValueError("optimizer wrapper cycle")
        seen.add(id(current))
        current = current.optimizer
    state = getattr(current, "state", {})
    result: dict[str, Any] = {
        "optimizer_type": type(current).__name__,
        "state_count": len(state),
        "step_min": None,
        "step_max": None,
    }
    steps: list[int] = []
    for value in state.values():
        step = value.get("step") if isinstance(value, dict) else None
        if isinstance(step, torch.Tensor):
            step = int(step.item())
        if isinstance(step, (int, float)):
            steps.append(int(step))
    if steps:
        result["step_min"] = min(steps)
        result["step_max"] = max(steps)
    for key in ("exp_avg", "exp_avg_sq"):
        total_sum = 0.0
        total_sq = 0.0
        numel = 0
        tensors = 0
        for value in state.values():
            tensor = value.get(key) if isinstance(value, dict) else None
            if not isinstance(tensor, torch.Tensor):
                continue
            item = tensor.detach().float()
            total_sum += item.sum().item()
            total_sq += item.square().sum().item()
            numel += item.numel()
            tensors += 1
        result[key] = {
            "num_tensors": tensors,
            "numel": numel,
            "sum": total_sum,
            "norm": math.sqrt(total_sq),
        }
    return result
