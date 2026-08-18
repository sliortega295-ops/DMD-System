"""Bounded CPU offload for selected native MiniMax checkpoint boundaries.

This module is intentionally narrower than ``save_on_cpu`` around a training
stage.  The pinned MiniMax forward invokes one checkpoint per transformer block.
The wrapper below scopes saved-tensor hooks to that *single checkpoint call*,
and only allows the checkpoint input (the first tensor argument) to be moved to
CPU.  The hook is closed before autograd later replays the checkpoint, so
recomputed intermediates are never recursively offloaded.

The production contract is fail-closed: native per-block checkpointing
(``segment=0``), exactly 50 calls per grad-enabled transformer forward, and a
manifest with explicit block indices/boundary metadata are required.
"""
from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import torch

from runtime.activation_offload import AuditedSavedTensorOffloader


SCHEMA_VERSION = "distillschedule.minimax_checkpoint_boundary_offload.v1"
PRODUCTION_BLOCK_COUNT = 50
SUPPORTED_POLICIES = ("none", "checkpoint_boundary_cpu")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _storage_bytes(value: torch.Tensor) -> int:
    return int(value.untyped_storage().nbytes())


def _tensor_metadata(value: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": [int(item) for item in value.shape],
        "stride": [int(item) for item in value.stride()],
        "storage_offset": int(value.storage_offset()),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "storage_bytes": _storage_bytes(value),
        "logical_bytes": int(value.numel() * value.element_size()),
    }


@dataclass(frozen=True)
class BoundaryManifest:
    path: Path
    payload_sha256: str
    checkpoint_segment: int
    block_count: int
    selected_block_indices: tuple[int, ...]
    expected_boundaries: Mapping[int, Mapping[str, Any]]
    source: Mapping[str, Any]

    @classmethod
    def load(cls, path: Path) -> "BoundaryManifest":
        path = Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"boundary offload manifest missing: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise RuntimeError(
                f"unsupported boundary manifest schema: {payload.get('schema_version')!r}"
            )
        segment = int(payload.get("checkpoint_segment", -1))
        block_count = int(payload.get("block_count", -1))
        selected = tuple(int(item) for item in payload.get("selected_block_indices", ()))
        expected_raw = payload.get("expected_boundaries", {})
        if not isinstance(expected_raw, dict):
            raise RuntimeError("expected_boundaries must be an object")
        expected: dict[int, Mapping[str, Any]] = {}
        for key, value in expected_raw.items():
            index = int(key)
            if not isinstance(value, dict):
                raise RuntimeError(f"expected boundary {key!r} must be an object")
            expected[index] = value
        if segment != 0:
            raise RuntimeError("checkpoint boundary offload requires native segment=0")
        if block_count != PRODUCTION_BLOCK_COUNT:
            raise RuntimeError(
                f"checkpoint boundary offload requires {PRODUCTION_BLOCK_COUNT} blocks, got {block_count}"
            )
        if not selected:
            raise RuntimeError("boundary offload manifest selects no blocks")
        if len(set(selected)) != len(selected):
            raise RuntimeError("selected boundary block indices must be unique")
        if any(index < 0 or index >= block_count for index in selected):
            raise RuntimeError("selected boundary block index is out of range")
        if tuple(sorted(selected)) != selected:
            raise RuntimeError("selected boundary block indices must be sorted")
        missing = [index for index in selected if index not in expected]
        if missing:
            raise RuntimeError(f"manifest lacks expected metadata for blocks {missing}")
        return cls(
            path=path,
            payload_sha256=sha256_file(path),
            checkpoint_segment=segment,
            block_count=block_count,
            selected_block_indices=selected,
            expected_boundaries=expected,
            source=payload.get("source", {}),
        )

    def receipt(self) -> dict[str, Any]:
        return {
            "policy": "checkpoint_boundary_cpu",
            "manifest": str(self.path),
            "manifest_sha256": self.payload_sha256,
            "checkpoint_segment": self.checkpoint_segment,
            "block_count": self.block_count,
            "selected_block_indices": list(self.selected_block_indices),
            "source": dict(self.source),
        }


@dataclass
class BoundaryOffloadRegistration:
    transformer: Any
    original_checkpoint: Callable[..., Any]
    wrapper: Callable[..., Any]
    pre_hook: Any
    post_hook: Any
    offloader: AuditedSavedTensorOffloader
    manifest: BoundaryManifest
    role: str
    event_path: Path
    expected_calls: int = PRODUCTION_BLOCK_COUNT
    call_count: int = 0
    grad_forward_count: int = 0
    selected_call_count: int = 0
    recompute_pack_count: int = 0
    last_forward_selected: list[int] | None = None
    closed: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def receipt(self) -> dict[str, Any]:
        stats = dict(self.offloader.stats)
        return {
            **self.manifest.receipt(),
            "role": self.role,
            "event_path": str(self.event_path),
            "expected_checkpoint_calls_per_grad_forward": self.expected_calls,
            "observed_checkpoint_calls": self.call_count,
            "grad_forward_count": self.grad_forward_count,
            "selected_checkpoint_calls": self.selected_call_count,
            "recompute_pack_count": self.recompute_pack_count,
            "offloader_stats": stats,
            "closed": self.closed,
        }

    def remove(self) -> None:
        if self.closed:
            return
        self.pre_hook.remove()
        self.post_hook.remove()
        self.transformer._gradient_checkpointing_func = self.original_checkpoint
        self.offloader.assert_scope_closed()
        self.closed = True


def _validate_transformer(transformer: Any, manifest: BoundaryManifest) -> list[Any]:
    blocks = getattr(transformer, "transformer_blocks", None)
    if blocks is None or len(blocks) != manifest.block_count:
        raise RuntimeError(
            "MiniMax checkpoint boundary offload block-count mismatch: "
            f"expected {manifest.block_count}, got {0 if blocks is None else len(blocks)}"
        )
    if getattr(transformer, "_distillschedule_segmented_checkpoint", 0):
        raise RuntimeError("boundary offload is mutually exclusive with segmented checkpointing")
    if not bool(getattr(transformer, "gradient_checkpointing", False)):
        raise RuntimeError("boundary offload requires native gradient checkpointing")
    return list(blocks)


def _check_boundary(
    value: Any,
    manifest: BoundaryManifest,
    block_index: int,
    *,
    require_cuda: bool = True,
) -> dict[str, Any]:
    if not torch.is_tensor(value):
        raise RuntimeError(
            f"checkpoint boundary block {block_index} first argument is not a tensor: {type(value)!r}"
        )
    if require_cuda and not value.is_cuda:
        raise RuntimeError(f"checkpoint boundary block {block_index} is not on CUDA: {value.device}")
    actual = _tensor_metadata(value)
    expected = manifest.expected_boundaries[block_index]
    for key in ("shape", "stride", "dtype", "storage_bytes", "logical_bytes"):
        if key in expected and actual[key] != expected[key]:
            raise RuntimeError(
                f"boundary metadata mismatch for block {block_index} field {key}: "
                f"{actual[key]!r} != {expected[key]!r}"
            )
    return actual


def install_checkpoint_boundary_offload(
    transformer: Any,
    manifest: BoundaryManifest,
    *,
    role: str,
    event_path: Path,
    pin_memory: bool = True,
    expected_block_count: int | None = None,
    require_cuda: bool = True,
) -> BoundaryOffloadRegistration:
    """Install selected native checkpoint-boundary offload on one transformer.

    ``expected_block_count`` exists only for CPU unit tests.  Production callers
    must use the manifest's 50-block contract.
    """
    blocks = _validate_transformer(transformer, manifest)
    expected_calls = int(expected_block_count or manifest.block_count)
    if expected_calls != manifest.block_count:
        raise RuntimeError("test-only expected call count differs from manifest block count")
    original = getattr(transformer, "_gradient_checkpointing_func", None)
    if not callable(original):
        raise RuntimeError("MiniMax transformer has no callable checkpoint function")
    event_path = Path(event_path)
    if event_path.exists():
        raise FileExistsError(f"refusing to overwrite boundary offload events: {event_path}")
    event_path.parent.mkdir(parents=True, exist_ok=True)
    offloader = AuditedSavedTensorOffloader(
        model=transformer,
        logical_component=f"{role}/checkpoint_boundary",
        event_path=event_path,
        pin_memory=pin_memory,
        detailed_events=True,
    )
    state: dict[str, Any] = {"call_count": 0, "grad": False, "selected": []}
    registration: BoundaryOffloadRegistration | None = None

    def wrapped(function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        index = int(state["call_count"])
        state["call_count"] = index + 1
        if index >= manifest.block_count:
            raise RuntimeError(
                f"native MiniMax checkpoint call count exceeded {manifest.block_count}"
            )
        if not torch.is_grad_enabled():
            return original(function, *args, **kwargs)
        state["grad"] = True
        if index not in manifest.selected_block_indices:
            return original(function, *args, **kwargs)
        if not args:
            raise RuntimeError(f"checkpoint boundary block {index} has no input tensor")
        _check_boundary(args[0], manifest, index, require_cuda=require_cuda)
        state["selected"].append(index)
        # Only the checkpoint input is an explicit argument in the pinned
        # MiniMax forward; temb/rotary/index tensors are closure values.  The
        # signature allow-list prevents unrelated saved tensors from being
        # copied if the checkpoint implementation changes.
        value = args[0]
        signature = (
            tuple(int(item) for item in value.shape),
            str(value.dtype),
            int(value.numel() * value.element_size()),
        )
        offloader.allowed_offload_signatures = frozenset((signature,))
        with offloader.context():
            result = original(function, *args, **kwargs)
        offloader.assert_scope_closed()
        return result

    def pre_hook(_module: Any, _inputs: tuple[Any, ...]) -> None:
        state["call_count"] = 0
        state["grad"] = bool(torch.is_grad_enabled())
        state["selected"] = []

    def post_hook(_module: Any, _inputs: tuple[Any, ...], _output: Any) -> None:
        if state["grad"]:
            if state["call_count"] != manifest.block_count:
                raise RuntimeError(
                    "native MiniMax checkpoint call count mismatch: "
                    f"{state['call_count']} != {manifest.block_count}"
                )
            if tuple(state["selected"]) != tuple(manifest.selected_block_indices):
                raise RuntimeError(
                    "selected checkpoint boundary sequence mismatch: "
                    f"{state['selected']!r} != {list(manifest.selected_block_indices)!r}"
                )
            assert registration is not None
            registration.grad_forward_count += 1
        assert registration is not None
        registration.call_count += int(state["call_count"])
        registration.selected_call_count += len(state["selected"])

    transformer._gradient_checkpointing_func = wrapped
    pre = transformer.register_forward_pre_hook(pre_hook)
    post = transformer.register_forward_hook(post_hook, always_call=True)
    registration = BoundaryOffloadRegistration(
        transformer=transformer,
        original_checkpoint=original,
        wrapper=wrapped,
        pre_hook=pre,
        post_hook=post,
        offloader=offloader,
        manifest=manifest,
        role=role,
        event_path=event_path,
        expected_calls=expected_calls,
    )
    return registration


def write_boundary_manifest(
    path: Path,
    *,
    boundaries: list[Mapping[str, Any]],
    selected_block_indices: list[int],
    source: Mapping[str, Any],
) -> Path:
    """Write a deterministic manifest from a no-copy boundary census."""
    payload = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_segment": 0,
        "block_count": len(boundaries),
        "selected_block_indices": [int(item) for item in selected_block_indices],
        "expected_boundaries": {
            str(int(row["block_index"])): dict(row) for row in boundaries
        },
        "source": dict(source),
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


@dataclass
class BoundaryCensusRegistration:
    """No-copy block-input census used to choose a bounded prefix."""

    transformer: Any
    output_path: Path
    pre_hook: Any
    post_hook: Any
    block_hooks: list[Any]
    rows: list[dict[str, Any]]
    closed: bool = False

    def remove(self) -> None:
        if self.closed:
            return
        self.pre_hook.remove()
        self.post_hook.remove()
        for hook in self.block_hooks:
            hook.remove()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in self.rows),
            encoding="utf-8",
        )
        self.closed = True


def install_boundary_census(
    transformer: Any,
    *,
    output_path: Path,
    expected_block_count: int = PRODUCTION_BLOCK_COUNT,
) -> BoundaryCensusRegistration:
    """Record block input metadata without retaining tensors or copying data."""
    blocks = list(getattr(transformer, "transformer_blocks", ()))
    if len(blocks) != expected_block_count:
        raise RuntimeError(
            f"boundary census requires {expected_block_count} transformer blocks, got {len(blocks)}"
        )
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite boundary census: {output_path}")
    state = {"active": False, "seen": set()}
    rows: list[dict[str, Any]] = []

    def top_pre(_module: Any, _inputs: tuple[Any, ...]) -> None:
        state["active"] = not torch.is_grad_enabled()
        state["seen"] = set()

    def top_post(_module: Any, _inputs: tuple[Any, ...], _output: Any) -> None:
        state["active"] = False

    def block_pre(index: int, _module: Any, inputs: tuple[Any, ...]) -> None:
        if not state["active"] or index in state["seen"]:
            return
        if not inputs or not torch.is_tensor(inputs[0]):
            raise RuntimeError(f"boundary census block {index} has no tensor input")
        value = inputs[0]
        state["seen"].add(index)
        rows.append({"block_index": index, **_tensor_metadata(value)})

    pre = transformer.register_forward_pre_hook(top_pre)
    post = transformer.register_forward_hook(top_post, always_call=True)
    block_hooks = [block.register_forward_pre_hook(lambda m, i, index=index: block_pre(index, m, i)) for index, block in enumerate(blocks)]
    registration = BoundaryCensusRegistration(
        transformer=transformer,
        output_path=output_path,
        pre_hook=pre,
        post_hook=post,
        block_hooks=block_hooks,
        rows=rows,
    )
    return registration
