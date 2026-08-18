"""Low-overhead, opt-in memory attribution for the MiniMax FSDP-first canary.

This module deliberately records allocator counters and tensor metadata only.  It
does not synchronize CUDA, copy tensors to the host, or retain tensor references.
It is enabled only when ``LIGHTX2V_MINIMAX_MEMORY_ATTRIBUTION=1`` and is not part
of the normal benchmark path.
"""
from __future__ import annotations

import contextlib
import json
import os
import time
from pathlib import Path
from typing import Any, Iterator


def _bytes(value: Any) -> int:
    try:
        return int(value.numel()) * int(value.element_size())
    except Exception:
        return 0


def _local(value: Any) -> Any:
    return value.to_local() if hasattr(value, "to_local") else value


def _tensor_bytes(value: Any) -> int:
    try:
        return _bytes(_local(value))
    except Exception:
        return 0


class MemoryAttribution:
    """Per-rank JSONL recorder for persistent and transient memory classes."""

    def __init__(self, torch_module: Any, *, rank: int, role_objects: dict[str, Any] | None = None) -> None:
        self.torch = torch_module
        self.enabled = os.environ.get("LIGHTX2V_MINIMAX_MEMORY_ATTRIBUTION", "0") == "1"
        self.rank = int(rank)
        self.role_objects = role_objects or {}
        self.pack_bytes = 0
        self.unpack_bytes = 0
        self.pack_count = 0
        self.unpack_count = 0
        self.saved_live_bytes = 0
        self.saved_live_peak_bytes = 0
        self._path: Path | None = None
        self._handle = None
        if self.enabled:
            root = Path(os.environ.get("LIGHTX2V_MINIMAX_MEMORY_ATTRIBUTION_DIR", ""))
            if not root:
                raise RuntimeError("memory attribution enabled without an output directory")
            root.mkdir(parents=True, exist_ok=True)
            self._path = root / f"rank_{self.rank:03d}.jsonl"
            self._handle = self._path.open("a", encoding="utf-8")
            self.record("attribution_init", phase="construction")

    def close(self) -> None:
        if self._handle is not None:
            self._handle.flush()
            self._handle.close()
            self._handle = None

    def _role_bytes(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for role, obj in self.role_objects.items():
            logical = 0
            trainable = 0
            frozen = 0
            grad = 0
            try:
                for parameter in obj.parameters():
                    size = _tensor_bytes(parameter)
                    logical += size
                    if bool(getattr(parameter, "requires_grad", False)):
                        trainable += size
                    else:
                        frozen += size
                    if getattr(parameter, "grad", None) is not None:
                        grad += _tensor_bytes(parameter.grad)
            except Exception:
                # Metadata collection must never change or abort the workload.
                pass
            result[f"{role}_parameter_bytes"] = logical
            result[f"{role}_trainable_parameter_bytes"] = trainable
            result[f"{role}_frozen_parameter_bytes"] = frozen
            result[f"{role}_gradient_bytes"] = grad
        return result

    def _optimizer_bytes(self, optimizers: dict[str, Any] | None) -> dict[str, int]:
        result: dict[str, int] = {}
        for name, optimizer in (optimizers or {}).items():
            device_bytes = 0
            cpu_bytes = 0
            try:
                for state in optimizer.state.values():
                    if not isinstance(state, dict):
                        continue
                    for value in state.values():
                        if not hasattr(value, "device"):
                            continue
                        size = _tensor_bytes(value)
                        if getattr(value.device, "type", None) == "cuda":
                            device_bytes += size
                        else:
                            cpu_bytes += size
            except Exception:
                pass
            result[f"{name}_optimizer_cuda_bytes"] = device_bytes
            result[f"{name}_optimizer_cpu_bytes"] = cpu_bytes
        return result

    def _allocator(self) -> dict[str, Any]:
        torch = self.torch
        device = torch.cuda.current_device()
        free, total = torch.cuda.mem_get_info(device)
        stats = torch.cuda.memory_stats(device)
        keys = (
            "active_bytes.all.current",
            "allocated_bytes.all.current",
            "inactive_split_bytes.all.current",
            "reserved_bytes.all.current",
            "segment.all.current",
        )
        return {
            "device": int(device),
            "driver_free_bytes": int(free),
            "driver_total_bytes": int(total),
            "torch_allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "torch_reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "torch_max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "torch_max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "allocator": {key.replace(".", "_"): int(stats.get(key, 0)) for key in keys},
        }

    def record(self, event: str, *, phase: str, role: str | None = None, call_index: int | None = None,
               optimizers: dict[str, Any] | None = None, extra: dict[str, Any] | None = None) -> None:
        if not self.enabled or self._handle is None:
            return
        value: dict[str, Any] = {
            "schema_version": "distillschedule.minimax_memory_attribution.v1",
            "timestamp": time.time(),
            "rank": self.rank,
            "event": event,
            "phase": phase,
            "role": role,
            "call_index": call_index,
            "saved_tensors": {
                "pack_count": self.pack_count,
                "unpack_count": self.unpack_count,
                "pack_bytes": self.pack_bytes,
                "unpack_bytes": self.unpack_bytes,
                "live_bytes_estimate": self.saved_live_bytes,
                "live_peak_bytes_estimate": self.saved_live_peak_bytes,
            },
            **self._allocator(),
            **self._role_bytes(),
            **self._optimizer_bytes(optimizers),
        }
        if extra:
            value["extra"] = extra
        self._handle.write(json.dumps(value, sort_keys=True) + "\n")
        self._handle.flush()

    @contextlib.contextmanager
    def saved_tensors(self) -> Iterator[None]:
        """Count saved tensors without copying or retaining them."""
        if not self.enabled:
            yield
            return

        torch = self.torch

        def pack(value: Any) -> Any:
            size = _tensor_bytes(value)
            self.pack_count += 1
            self.pack_bytes += size
            self.saved_live_bytes += size
            self.saved_live_peak_bytes = max(self.saved_live_peak_bytes, self.saved_live_bytes)
            return value

        def unpack(value: Any) -> Any:
            size = _tensor_bytes(value)
            self.unpack_count += 1
            self.unpack_bytes += size
            self.saved_live_bytes = max(0, self.saved_live_bytes - size)
            return value

        with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
            yield

    def exception(self, exc: BaseException, *, phase: str, role: str | None = None,
                  call_index: int | None = None, optimizers: dict[str, Any] | None = None) -> None:
        extra: dict[str, Any] = {
            "exception_type": type(exc).__name__,
            "exception": str(exc),
        }
        try:
            extra["memory_summary"] = self.torch.cuda.memory_summary()
        except Exception as summary_error:
            extra["memory_summary_error"] = repr(summary_error)
        try:
            extra["memory_stats"] = self.torch.cuda.memory_stats()
        except Exception as stats_error:
            extra["memory_stats_error"] = repr(stats_error)
        self.record("exception", phase=phase, role=role, call_index=call_index,
                    optimizers=optimizers, extra=extra)

