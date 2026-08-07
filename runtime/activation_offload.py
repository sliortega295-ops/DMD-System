"""Audited coarse saved-tensor CPU offload for the M0 B2 baseline.

The hook is deliberately attached only to grad-enabled top-level Generator and Score
calls. Checkpoint recomputation invokes blocks directly after the top-level context has
closed, so B2 does not recursively offload recomputed intermediates.
"""
from __future__ import annotations

import json
import threading
import time
import weakref
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

import torch


ORIGINAL_FORWARD = "ORIGINAL_FORWARD"
CHECKPOINT_RECOMPUTE = "CHECKPOINT_RECOMPUTE"
REPLAY_FORWARD = "REPLAY_FORWARD"


def _storage_id(tensor: torch.Tensor) -> tuple[str, int, int]:
    storage = tensor.untyped_storage()
    return (str(tensor.device), int(storage.data_ptr()), int(storage.nbytes()))


def _alias_id(tensor: torch.Tensor) -> tuple[int, int, tuple[int, ...], tuple[int, ...], str]:
    return (
        int(tensor._cdata),
        int(tensor.storage_offset()),
        tuple(int(x) for x in tensor.shape),
        tuple(int(x) for x in tensor.stride()),
        str(tensor.dtype),
    )


def _logical_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


@dataclass
class _KeepOnDevice:
    tensor: torch.Tensor


@dataclass
class _PackedOnCPU:
    cpu_tensor: torch.Tensor
    original_device: torch.device
    alias_key: tuple[int, int, tuple[int, ...], tuple[int, ...], str]
    remaining_unpacks: int = 1


class AuditedSavedTensorOffloader:
    """A save_on_cpu(pin_memory=True)-equivalent hook with fail-closed auditing."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        logical_component: str,
        event_path: Path,
        pin_memory: bool = True,
        min_offload_bytes: int = 0,
        detailed_events: bool = True,
        dynamic_excluded_storage_providers: tuple[
            Callable[[tuple[str, int, int]], Mapping[str, str] | None], ...
        ] = (),
    ) -> None:
        self.model = model
        self.logical_component = logical_component
        self.event_path = Path(event_path)
        self.event_path.parent.mkdir(parents=True, exist_ok=True)
        if self.event_path.exists():
            raise FileExistsError(f"refusing to overwrite activation census: {self.event_path}")
        self.pin_memory = bool(pin_memory)
        self.min_offload_bytes = int(min_offload_bytes)
        if self.min_offload_bytes < 0:
            raise ValueError("min_offload_bytes must be non-negative")
        self.detailed_events = bool(detailed_events)
        self.dynamic_excluded_storage_providers = tuple(dynamic_excluded_storage_providers)
        self._excluded: dict[tuple[str, int, int], dict[str, str]] = {}
        self._packed_by_alias: dict[
            tuple[int, int, tuple[int, ...], tuple[int, ...], str], _PackedOnCPU
        ] = {}
        self._active_scope = False
        self._event_order = 0
        self._scope_order = 0
        self._lock = threading.Lock()
        self.stats = {
            "scope_count": 0,
            "pack_count": 0,
            "unpack_count": 0,
            "cpu_copy_count": 0,
            "duplicate_pack_count": 0,
            "kept_parameter_or_buffer_count": 0,
            "kept_zero_stride_count": 0,
            "kept_non_strided_count": 0,
            "kept_below_threshold_count": 0,
            "kept_below_threshold_storage_bytes": 0,
            "offloaded_logical_bytes": 0,
            "offloaded_storage_bytes": 0,
            "peak_live_cpu_bytes": 0,
            "live_cpu_bytes": 0,
        }

    def _write_event(self, event: dict[str, Any]) -> None:
        if not self.detailed_events and event.get("event") not in {
            "scope_start",
            "scope_end",
        }:
            return
        with self._lock:
            self._event_order += 1
            record = {
                "event_order": self._event_order,
                "time_unix": time.time(),
                "logical_component": self.logical_component,
                **event,
            }
            with self.event_path.open("a") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")

    def _release_live_cpu_bytes(self, released_bytes: int) -> None:
        with self._lock:
            self.stats["live_cpu_bytes"] -= int(released_bytes)
            if self.stats["live_cpu_bytes"] < 0:
                raise RuntimeError("negative saved-tensor live CPU byte count")

    def _refresh_excluded_storages(self) -> None:
        excluded: dict[tuple[str, int, int], dict[str, str]] = {}
        for kind, iterator in (
            ("parameter", self.model.named_parameters()),
            ("buffer", self.model.named_buffers()),
        ):
            for name, tensor in iterator:
                if tensor.device.type != "cuda" and self.pin_memory:
                    continue
                sid = _storage_id(tensor)
                previous = excluded.get(sid)
                if previous is not None and previous != {"kind": kind, "name": name}:
                    continue
                excluded[sid] = {"kind": kind, "name": name}
        self._excluded = excluded

    def _excluded_owner(self, sid: tuple[str, int, int]) -> Mapping[str, str] | None:
        owner = self._excluded.get(sid)
        if owner is not None:
            return owner
        for provider in self.dynamic_excluded_storage_providers:
            owner = provider(sid)
            if owner is not None:
                return owner
        return None

    def _pack(self, tensor: torch.Tensor):
        self.stats["pack_count"] += 1
        sid = _storage_id(tensor)
        alias = _alias_id(tensor)
        base_event = {
            "event": "pack",
            "save_generation": ORIGINAL_FORWARD,
            "storage_id": list(sid),
            "alias_id": [alias[0], alias[1]],
            "shape": list(tensor.shape),
            "stride": list(tensor.stride()),
            "storage_offset": int(tensor.storage_offset()),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "logical_bytes": _logical_bytes(tensor),
            "storage_bytes": int(tensor.untyped_storage().nbytes()),
            "contiguous": bool(tensor.is_contiguous()),
            "zero_stride": any(size > 1 and stride == 0 for size, stride in zip(tensor.shape, tensor.stride())),
            "requires_grad": bool(tensor.requires_grad),
            "leaf": bool(tensor.is_leaf),
        }

        excluded = self._excluded_owner(sid)
        if excluded is not None:
            self.stats["kept_parameter_or_buffer_count"] += 1
            self._write_event({**base_event, "policy": "KEEP_EXCLUDED_STORAGE", "owner": excluded})
            return _KeepOnDevice(tensor)

        if tensor.layout != torch.strided:
            self.stats["kept_non_strided_count"] += 1
            self._write_event({**base_event, "policy": "KEEP_NON_STRIDED"})
            return _KeepOnDevice(tensor)

        if base_event["zero_stride"]:
            self.stats["kept_zero_stride_count"] += 1
            self._write_event({**base_event, "policy": "KEEP_ZERO_STRIDE_VIEW"})
            return _KeepOnDevice(tensor)

        if base_event["storage_bytes"] < self.min_offload_bytes:
            self.stats["kept_below_threshold_count"] += 1
            self.stats["kept_below_threshold_storage_bytes"] += base_event[
                "storage_bytes"
            ]
            self._write_event(
                {
                    **base_event,
                    "policy": "KEEP_BELOW_THRESHOLD",
                    "min_offload_bytes": self.min_offload_bytes,
                }
            )
            return _KeepOnDevice(tensor)

        existing = self._packed_by_alias.get(alias)
        if existing is not None:
            existing.remaining_unpacks += 1
            self.stats["duplicate_pack_count"] += 1
            self._write_event({**base_event, "policy": "OFFLOAD_CPU_DEDUP", "cpu_copy": False})
            return existing

        assert sid not in self._excluded, "parameter or buffer storage reached CPU pack"
        detached = tensor.detach()
        cpu_tensor = torch.empty_strided(
            tuple(detached.shape),
            tuple(detached.stride()),
            dtype=detached.dtype,
            device="cpu",
            pin_memory=self.pin_memory,
        )
        cpu_tensor.copy_(detached, non_blocking=False)
        copied_bytes = int(cpu_tensor.untyped_storage().nbytes())
        packed = _PackedOnCPU(cpu_tensor, tensor.device, alias)
        self._packed_by_alias[alias] = packed
        self.stats["cpu_copy_count"] += 1
        self.stats["offloaded_logical_bytes"] += _logical_bytes(tensor)
        self.stats["offloaded_storage_bytes"] += copied_bytes
        self.stats["live_cpu_bytes"] += copied_bytes
        self.stats["peak_live_cpu_bytes"] = max(
            self.stats["peak_live_cpu_bytes"], self.stats["live_cpu_bytes"]
        )
        weakref.finalize(packed, self._release_live_cpu_bytes, copied_bytes)
        self._write_event({**base_event, "policy": "OFFLOAD_CPU", "cpu_copy": True})
        return packed

    def _unpack(self, packed):
        self.stats["unpack_count"] += 1
        if isinstance(packed, _KeepOnDevice):
            tensor = packed.tensor
            self._write_event(
                {
                    "event": "unpack",
                    "save_generation": ORIGINAL_FORWARD,
                    "policy": "KEEP_GPU",
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "device": str(tensor.device),
                }
            )
            return tensor
        if not isinstance(packed, _PackedOnCPU):
            raise TypeError(f"unexpected saved tensor handle: {type(packed)!r}")
        result = packed.cpu_tensor.to(packed.original_device, non_blocking=False)
        packed.remaining_unpacks -= 1
        if packed.remaining_unpacks < 0:
            raise RuntimeError("saved tensor unpacked more times than it was packed")
        self._write_event(
            {
                "event": "unpack",
                "save_generation": ORIGINAL_FORWARD,
                "policy": "SYNC_H2D",
                "shape": list(result.shape),
                "stride": list(result.stride()),
                "dtype": str(result.dtype),
                "device": str(result.device),
                "remaining_unpacks": packed.remaining_unpacks,
            }
        )
        return result

    @contextmanager
    def context(self) -> Iterator[None]:
        if self._active_scope:
            raise RuntimeError("nested activation-offload scopes are forbidden")
        self._refresh_excluded_storages()
        self._active_scope = True
        self._scope_order += 1
        self.stats["scope_count"] += 1
        scope_id = self._scope_order
        self._write_event(
            {
                "event": "scope_start",
                "scope_id": scope_id,
                "save_generation": ORIGINAL_FORWARD,
                "excluded_storage_count": len(self._excluded),
            }
        )
        try:
            with torch.autograd.graph.saved_tensors_hooks(self._pack, self._unpack):
                yield
        finally:
            self._active_scope = False
            self._packed_by_alias.clear()
            self._write_event(
                {
                    "event": "scope_end",
                    "scope_id": scope_id,
                    "save_generation": ORIGINAL_FORWARD,
                    "stats": dict(self.stats),
                }
            )

    def assert_scope_closed(self) -> None:
        if self._active_scope:
            raise RuntimeError("activation-offload context leaked beyond model forward")
        if self._packed_by_alias:
            raise RuntimeError("activation-offload alias registry leaked beyond save scope")


@dataclass
class ActivationOffloadRegistration:
    offloader: AuditedSavedTensorOffloader
    pre_hook: Any
    post_hook: Any

    def remove(self) -> None:
        self.pre_hook.remove()
        self.post_hook.remove()
        self.offloader.assert_scope_closed()


def install_activation_offload_hooks(
    model: torch.nn.Module,
    *,
    logical_component: str,
    event_path: Path,
    pin_memory: bool = True,
    min_offload_bytes: int = 0,
    detailed_events: bool = True,
    dynamic_excluded_storage_providers: tuple[
        Callable[[tuple[str, int, int]], Mapping[str, str] | None], ...
    ] = (),
) -> ActivationOffloadRegistration:
    """Install selected-call-only hooks on a physical backbone."""

    offloader = AuditedSavedTensorOffloader(
        model=model,
        logical_component=logical_component,
        event_path=event_path,
        pin_memory=pin_memory,
        min_offload_bytes=min_offload_bytes,
        detailed_events=detailed_events,
        dynamic_excluded_storage_providers=dynamic_excluded_storage_providers,
    )
    context_stack: list[Any] = []

    def pre_hook(_module, _inputs):
        if not torch.is_grad_enabled():
            context_stack.append(None)
            return
        context = offloader.context()
        context.__enter__()
        context_stack.append(context)

    def post_hook(_module, _inputs, _output):
        if not context_stack:
            raise RuntimeError("activation-offload post-hook has no matching pre-hook")
        context = context_stack.pop()
        if context is not None:
            context.__exit__(None, None, None)
            offloader.assert_scope_closed()

    pre = model.register_forward_pre_hook(pre_hook)
    post = model.register_forward_hook(post_hook, always_call=True)
    return ActivationOffloadRegistration(offloader, pre, post)
