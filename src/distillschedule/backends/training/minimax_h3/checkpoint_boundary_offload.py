"""Pinned-CPU staging for native MiniMax-H3 checkpoint boundaries.

This is the shared-backbone adaptation of the world16 capacity path validated
in DMD-System.  It intentionally keeps native per-block checkpointing and
changes only where each checkpoint input is saved: the first tensor argument
(hidden_states) is copied to pinned CPU memory during the original forward and
restored when autograd later replays that block.

The saved-tensor hook exists only for the individual checkpoint call.  It is
closed before backward, so recomputed FFN/attention/LoRA intermediates are not
recursively offloaded.  Application-level DiT forward/backward counts are
unchanged.
"""

from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import torch

PRODUCTION_BLOCK_COUNT = 50
POLICY_NAME = "checkpoint_boundary_cpu"


def _logical_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _signature(tensor: torch.Tensor) -> tuple[tuple[int, ...], str, int]:
    return (
        tuple(int(value) for value in tensor.shape),
        str(tensor.dtype),
        _logical_bytes(tensor),
    )


def _alias(tensor: torch.Tensor) -> tuple[int, int, tuple[int, ...], tuple[int, ...], str]:
    return (
        int(tensor._cdata),
        int(tensor.storage_offset()),
        tuple(int(value) for value in tensor.shape),
        tuple(int(value) for value in tensor.stride()),
        str(tensor.dtype),
    )


@dataclass
class _KeepOnDevice:
    tensor: torch.Tensor


@dataclass
class _PackedOnCPU:
    cpu_tensor: torch.Tensor
    original_device: torch.device
    remaining_unpacks: int = 1


class _BoundarySaveScope:
    """Saved-tensor hook restricted to one checkpoint-input signature."""

    def __init__(
        self,
        *,
        target: torch.Tensor,
        pin_memory: bool,
        require_cuda: bool,
        stats: dict[str, int],
    ) -> None:
        if target.layout != torch.strided:
            raise RuntimeError("checkpoint boundary input must be strided")
        if any(size > 1 and stride == 0 for size, stride in zip(target.shape, target.stride())):
            raise RuntimeError("checkpoint boundary input cannot be a zero-stride view")
        if require_cuda and not target.is_cuda:
            raise RuntimeError(f"checkpoint boundary input must be CUDA, got {target.device}")
        self.target_signature = _signature(target)
        self.pin_memory = bool(pin_memory)
        self.stats = stats
        self._packed_by_alias: dict[
            tuple[int, int, tuple[int, ...], tuple[int, ...], str], _PackedOnCPU
        ] = {}
        self.scope_cpu_copies = 0

    def _pack(self, tensor: torch.Tensor):
        self.stats["pack_count"] += 1
        if _signature(tensor) != self.target_signature:
            self.stats["policy_filter_keep_count"] += 1
            return _KeepOnDevice(tensor)
        if tensor.layout != torch.strided:
            self.stats["non_strided_keep_count"] += 1
            return _KeepOnDevice(tensor)
        if any(size > 1 and stride == 0 for size, stride in zip(tensor.shape, tensor.stride())):
            self.stats["zero_stride_keep_count"] += 1
            return _KeepOnDevice(tensor)

        alias = _alias(tensor)
        existing = self._packed_by_alias.get(alias)
        if existing is not None:
            existing.remaining_unpacks += 1
            self.stats["duplicate_pack_count"] += 1
            return existing

        detached = tensor.detach()
        cpu_tensor = torch.empty_strided(
            tuple(detached.shape),
            tuple(detached.stride()),
            dtype=detached.dtype,
            device="cpu",
            pin_memory=self.pin_memory,
        )
        cpu_tensor.copy_(detached, non_blocking=False)
        packed = _PackedOnCPU(cpu_tensor=cpu_tensor, original_device=tensor.device)
        self._packed_by_alias[alias] = packed
        copied = int(cpu_tensor.untyped_storage().nbytes())
        self.scope_cpu_copies += 1
        self.stats["cpu_copy_count"] += 1
        self.stats["offloaded_logical_bytes"] += _logical_bytes(tensor)
        self.stats["offloaded_storage_bytes"] += copied
        return packed

    def _unpack(self, packed):
        self.stats["unpack_count"] += 1
        if isinstance(packed, _KeepOnDevice):
            return packed.tensor
        if not isinstance(packed, _PackedOnCPU):
            raise TypeError(f"unexpected checkpoint-boundary saved tensor: {type(packed)!r}")
        result = packed.cpu_tensor.to(packed.original_device, non_blocking=False)
        packed.remaining_unpacks -= 1
        if packed.remaining_unpacks < 0:
            raise RuntimeError("checkpoint-boundary tensor unpacked too many times")
        return result

    @contextmanager
    def context(self) -> Iterator[None]:
        with torch.autograd.graph.saved_tensors_hooks(self._pack, self._unpack):
            yield
        # PyTorch non-reentrant checkpoint saves one dummy tensor plus the
        # checkpoint inputs.  With the signature allow-list exactly the first
        # hidden-state boundary must have been copied once (duplicates are
        # deduplicated by alias above).
        if self.scope_cpu_copies != 1:
            raise RuntimeError(
                "checkpoint-boundary CPU staging expected exactly one boundary copy, "
                f"observed {self.scope_cpu_copies}"
            )
        self._packed_by_alias.clear()


@dataclass
class CheckpointBoundaryOffloadRegistration:
    transformer: torch.nn.Module
    original_checkpoint: Callable[..., Any]
    pre_hook: Any
    post_hook: Any
    role_getter: Callable[[], str]
    event_path: Path | None
    expected_block_count: int
    pin_memory: bool
    stats: dict[str, int]
    closed: bool = False

    def receipt(self) -> dict[str, Any]:
        return {
            "policy": POLICY_NAME,
            "expected_block_count": self.expected_block_count,
            "pin_memory": self.pin_memory,
            "event_path": None if self.event_path is None else str(self.event_path),
            "stats": dict(self.stats),
            "closed": self.closed,
        }

    def remove(self) -> None:
        if self.closed:
            return
        self.pre_hook.remove()
        self.post_hook.remove()
        self.transformer._gradient_checkpointing_func = self.original_checkpoint
        self.closed = True


def install_checkpoint_boundary_cpu_offload(
    transformer: torch.nn.Module,
    *,
    role_getter: Callable[[], str],
    event_path: Path | None = None,
    pin_memory: bool = True,
    expected_block_count: int = PRODUCTION_BLOCK_COUNT,
    require_cuda: bool = True,
) -> CheckpointBoundaryOffloadRegistration:
    """Stage every native MiniMax checkpoint input on CPU.

    Production uses 50 native per-block checkpoint calls.  ``expected_block_count``
    and ``require_cuda`` are configurable only so CPU unit tests can exercise the
    mechanism with a tiny toy transformer.
    """

    blocks = getattr(transformer, "transformer_blocks", None)
    if blocks is None or len(blocks) != int(expected_block_count):
        raise RuntimeError(
            "checkpoint-boundary CPU staging requires native per-block layout: "
            f"expected {expected_block_count} transformer blocks, got "
            f"{0 if blocks is None else len(blocks)}"
        )
    if not bool(getattr(transformer, "gradient_checkpointing", False)):
        raise RuntimeError("checkpoint-boundary CPU staging requires gradient checkpointing")
    original = getattr(transformer, "_gradient_checkpointing_func", None)
    if not callable(original):
        raise RuntimeError("MiniMax transformer has no callable gradient checkpoint function")

    path = None if event_path is None else Path(event_path)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f"refusing to overwrite checkpoint-boundary events: {path}")

    stats = {
        "transformer_forward_count": 0,
        "grad_transformer_forward_count": 0,
        "checkpoint_call_count": 0,
        "grad_checkpoint_call_count": 0,
        "cpu_copy_count": 0,
        "pack_count": 0,
        "unpack_count": 0,
        "duplicate_pack_count": 0,
        "policy_filter_keep_count": 0,
        "non_strided_keep_count": 0,
        "zero_stride_keep_count": 0,
        "offloaded_logical_bytes": 0,
        "offloaded_storage_bytes": 0,
        "student_grad_forward_count": 0,
        "fake_grad_forward_count": 0,
        "other_grad_forward_count": 0,
    }
    state: dict[str, Any] = {
        "call_count": 0,
        "grad": False,
        "role": "unknown",
    }
    lock = threading.Lock()

    def write_event(payload: dict[str, Any]) -> None:
        if path is None:
            return
        record = {"time_unix": time.time(), **payload}
        with lock, path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def wrapped(function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        index = int(state["call_count"])
        state["call_count"] = index + 1
        stats["checkpoint_call_count"] += 1
        if not torch.is_grad_enabled():
            return original(function, *args, **kwargs)
        if index >= int(expected_block_count):
            raise RuntimeError(
                f"checkpoint call count exceeded expected {expected_block_count}: index={index}"
            )
        if not args or not torch.is_tensor(args[0]):
            raise RuntimeError(f"checkpoint block {index} has no tensor boundary input")

        stats["grad_checkpoint_call_count"] += 1
        boundary = args[0]
        scope = _BoundarySaveScope(
            target=boundary,
            pin_memory=pin_memory,
            require_cuda=require_cuda,
            stats=stats,
        )
        before_copies = int(stats["cpu_copy_count"])
        with scope.context():
            result = original(function, *args, **kwargs)
        write_event(
            {
                "event": "checkpoint_boundary_saved",
                "block_index": index,
                "role": state["role"],
                "shape": list(boundary.shape),
                "dtype": str(boundary.dtype),
                "logical_bytes": _logical_bytes(boundary),
                "new_cpu_copies": int(stats["cpu_copy_count"]) - before_copies,
            }
        )
        return result

    def pre_hook(_module: Any, _inputs: tuple[Any, ...]) -> None:
        state["call_count"] = 0
        state["grad"] = bool(torch.is_grad_enabled())
        state["role"] = str(role_getter())
        stats["transformer_forward_count"] += 1

    def post_hook(_module: Any, _inputs: tuple[Any, ...], _output: Any) -> None:
        if not state["grad"]:
            return
        calls = int(state["call_count"])
        if calls != int(expected_block_count):
            raise RuntimeError(
                "native MiniMax checkpoint call count mismatch: "
                f"{calls} != {expected_block_count}"
            )
        stats["grad_transformer_forward_count"] += 1
        role = str(state["role"])
        if role == "student":
            stats["student_grad_forward_count"] += 1
        elif role == "fake":
            stats["fake_grad_forward_count"] += 1
        else:
            stats["other_grad_forward_count"] += 1
        write_event(
            {
                "event": "grad_transformer_forward_complete",
                "role": role,
                "checkpoint_calls": calls,
                "cpu_copies_total": int(stats["cpu_copy_count"]),
            }
        )

    pre = transformer.register_forward_pre_hook(pre_hook)
    post = transformer.register_forward_hook(post_hook, always_call=True)
    transformer._gradient_checkpointing_func = wrapped
    return CheckpointBoundaryOffloadRegistration(
        transformer=transformer,
        original_checkpoint=original,
        pre_hook=pre,
        post_hook=post,
        role_getter=role_getter,
        event_path=path,
        expected_block_count=int(expected_block_count),
        pin_memory=bool(pin_memory),
        stats=stats,
    )
