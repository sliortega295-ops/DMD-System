"""Metadata-only saved-tensor census for causal lifecycle attribution.

Pack and unpack are strict pass-through operations: no tensor is copied and no
Tensor reference is retained by the census.  Only scalar metadata and aggregate
signatures are persisted.
"""
from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch


def _signature(tensor: torch.Tensor) -> tuple[tuple[int, ...], str, int, int]:
    return (
        tuple(int(value) for value in tensor.shape),
        str(tensor.dtype),
        int(tensor.numel() * tensor.element_size()),
        int(tensor.untyped_storage().nbytes()),
    )


class PassThroughSavedTensorCensus:
    def __init__(self, *, event_path: Path, logical_component: str) -> None:
        self.event_path = Path(event_path)
        self.event_path.parent.mkdir(parents=True, exist_ok=True)
        if self.event_path.exists():
            raise FileExistsError(self.event_path)
        self.logical_component = logical_component
        self._active_scope = False
        self._scope_order = 0
        self._event_order = 0
        self._lock = threading.Lock()
        self._signature_counts: dict[
            tuple[tuple[int, ...], str, int, int], int
        ] = {}
        self.stats = {
            "scope_count": 0,
            "pack_count": 0,
            "unpack_count": 0,
            "pack_returns_original_tensor": True,
            "unpack_returns_same_tensor": True,
            "tensor_references_retained_by_census": 0,
        }

    def _write(self, event: dict[str, Any]) -> None:
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

    def _pack(self, tensor: torch.Tensor) -> torch.Tensor:
        self.stats["pack_count"] += 1
        signature = _signature(tensor)
        self._signature_counts[signature] = self._signature_counts.get(signature, 0) + 1
        storage = tensor.untyped_storage()
        self._write(
            {
                "event": "pack",
                "scope_id": self._scope_order,
                "object_id": id(tensor),
                "storage_data_ptr": int(storage.data_ptr()),
                "storage_bytes": signature[3],
                "logical_bytes": signature[2],
                "shape": list(signature[0]),
                "stride": [int(value) for value in tensor.stride()],
                "dtype": signature[1],
                "device": str(tensor.device),
                "storage_offset": int(tensor.storage_offset()),
                "requires_grad": bool(tensor.requires_grad),
                "leaf": bool(tensor.is_leaf),
                "policy": "PASS_THROUGH_ORIGINAL_TENSOR",
            }
        )
        return tensor

    def _unpack(self, tensor: torch.Tensor) -> torch.Tensor:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"saved-tensor census received {type(tensor)!r}")
        self.stats["unpack_count"] += 1
        self._write(
            {
                "event": "unpack",
                "scope_id": self._scope_order,
                "object_id": id(tensor),
                "shape": [int(value) for value in tensor.shape],
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
                "policy": "PASS_THROUGH_SAME_TENSOR",
            }
        )
        return tensor

    @contextmanager
    def context(self) -> Iterator[None]:
        if self._active_scope:
            raise RuntimeError("nested saved-tensor census scopes are forbidden")
        self._active_scope = True
        self._scope_order += 1
        self.stats["scope_count"] += 1
        scope_id = self._scope_order
        self._write({"event": "scope_start", "scope_id": scope_id})
        try:
            with torch.autograd.graph.saved_tensors_hooks(self._pack, self._unpack):
                yield
        finally:
            self._active_scope = False
            self._write({"event": "scope_end", "scope_id": scope_id})

    def summary(self) -> dict[str, Any]:
        if self._active_scope:
            raise RuntimeError("saved-tensor census scope is still active")
        signatures = [
            {
                "shape": list(shape),
                "dtype": dtype,
                "logical_bytes": logical_bytes,
                "storage_bytes": storage_bytes,
                "count": count,
            }
            for (shape, dtype, logical_bytes, storage_bytes), count in sorted(
                self._signature_counts.items(),
                key=lambda item: (-item[1], -item[0][2], item[0][0]),
            )
        ]
        return {
            "policy": "METADATA_ONLY_PASS_THROUGH_SAVED_TENSOR_CENSUS",
            "event_path": str(self.event_path),
            **self.stats,
            "signatures": signatures,
        }


@dataclass
class SavedTensorCensusRegistration:
    census: PassThroughSavedTensorCensus
    pre_hook: Any
    post_hook: Any
    context_stack: list[Any]
    removed: bool = False

    def remove(self) -> dict[str, Any]:
        if not self.removed:
            self.pre_hook.remove()
            self.post_hook.remove()
            self.removed = True
        if self.context_stack:
            raise RuntimeError("saved-tensor census forward context leaked")
        return self.census.summary()


def install_saved_tensor_census_hooks(
    model: torch.nn.Module,
    *,
    event_path: Path,
    logical_component: str,
) -> SavedTensorCensusRegistration:
    census = PassThroughSavedTensorCensus(
        event_path=event_path,
        logical_component=logical_component,
    )
    context_stack: list[Any] = []

    def pre_hook(_module, _inputs):
        if not torch.is_grad_enabled():
            context_stack.append(None)
            return
        context = census.context()
        context.__enter__()
        context_stack.append(context)

    def post_hook(_module, _inputs, _output):
        if not context_stack:
            raise RuntimeError("saved-tensor census post-hook has no matching pre-hook")
        context = context_stack.pop()
        if context is not None:
            context.__exit__(None, None, None)

    pre = model.register_forward_pre_hook(pre_hook)
    post = model.register_forward_hook(post_hook, always_call=True)
    return SavedTensorCensusRegistration(census, pre, post, context_stack)
