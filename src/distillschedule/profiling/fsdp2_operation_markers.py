"""Opt-in, no-sync NVTX markers for the FSDP2 overlap audit.

This module is diagnostic-only. It never calls ``unshard``/``reshard`` and
does not add a stream, event, copy, or collective. The FSDP2 internals are
wrapped only when the worker requests profile mode.
"""
from __future__ import annotations

import contextlib
import functools
from collections.abc import Iterator
from typing import Any, Callable


_ACTIVE: "OperationMarkerSession | None" = None


def _push(label: str) -> None:
    session = _ACTIVE
    if session is None or not session.enabled:
        return
    import torch

    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(str(label))


def _pop() -> None:
    session = _ACTIVE
    if session is None or not session.enabled:
        return
    import torch

    if torch.cuda.is_available():
        torch.cuda.nvtx.range_pop()


@contextlib.contextmanager
def operation_marker(label: str) -> Iterator[None]:
    _push(label)
    try:
        yield
    finally:
        _pop()


class OperationMarkerSession:
    """Reversible process-local wrappers for composable FSDP2."""

    def __init__(self, *, enabled: bool = False) -> None:
        self.enabled = bool(enabled)
        self._patches: list[tuple[Any, str, Any]] = []
        self._handles: list[Any] = []
        self._installed = False
        self._previous: OperationMarkerSession | None = None
        self.block_markers: list[dict[str, Any]] = []

    def __enter__(self) -> "OperationMarkerSession":
        self.install()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.uninstall()

    def _wrap(self, owner: Any, name: str, label_fn: Callable[..., str]) -> None:
        original = getattr(owner, name, None)
        if original is None or getattr(original, "__distillschedule_marker__", False):
            return

        @functools.wraps(original)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            with operation_marker(label_fn(*args, **kwargs)):
                return original(*args, **kwargs)

        wrapped.__distillschedule_marker__ = True
        setattr(owner, name, wrapped)
        self._patches.append((owner, name, original))

    @staticmethod
    def _fqn(group: Any) -> str:
        return str(getattr(group, "_module_fqn", None) or "unknown").replace(" ", "_")

    def install(self) -> None:
        global _ACTIVE
        if self._installed:
            return
        self._installed = True
        self._previous = _ACTIVE
        _ACTIVE = self
        if not self.enabled:
            return
        try:
            from torch.distributed.fsdp._fully_shard import _fsdp_param_group
            from torch.distributed.fsdp._fully_shard import _fsdp_state
        except ImportError as error:  # pragma: no cover
            self.uninstall()
            raise RuntimeError("composable FSDP2 is required for markers") from error

        group = _fsdp_param_group.FSDPParamGroup
        state = _fsdp_state.FSDPState
        self._wrap(group, "pre_forward", lambda obj, *a, **k: f"FSDP_UNSHARD_PRE_FORWARD/{self._fqn(obj)}")
        self._wrap(group, "post_forward", lambda obj, *a, **k: f"FSDP_RESHARD_POST_FORWARD/{self._fqn(obj)}")
        self._wrap(group, "pre_backward", lambda obj, *a, **k: f"FSDP_UNSHARD_PRE_BACKWARD/{self._fqn(obj)}")
        self._wrap(group, "post_backward", lambda obj, *a, **k: f"FSDP_REDUCE_SCATTER/{self._fqn(obj)}")

        def reshard_label(obj: Any, *a: Any, **k: Any) -> str:
            state_name = getattr(getattr(obj, "_training_state", None), "name", "")
            suffix = "FORWARD" if state_name == "FORWARD" else "BACKWARD"
            return f"FSDP_RESHARD_POST_{suffix}/{self._fqn(obj)}"

        self._wrap(group, "reshard", reshard_label)
        self._wrap(
            group,
            "unshard",
            lambda obj, *a, **k: f"FSDP_UNSHARD_PRE_{'FORWARD' if getattr(getattr(obj, '_training_state', None), 'name', '') == 'FORWARD' else 'BACKWARD'}/{self._fqn(obj)}",
        )
        # These functions are imported into _fsdp_param_group; patch the bound
        # references used by the actual FSDP2 implementation.
        self._wrap(_fsdp_param_group, "foreach_all_gather", lambda *a, **k: "FSDP_PARAMETER_ALL_GATHER")
        self._wrap(_fsdp_param_group, "foreach_reduce", lambda *a, **k: "FSDP_REDUCE_SCATTER")
        self._wrap(state, "_root_post_backward_final_callback", lambda *a, **k: "UNKNOWN_FRAMEWORK_STAGE/root_post_backward")

    def install_module_markers(self, module: Any, *, role: str) -> dict[str, Any]:
        """Mark block compute; the wrappers do not touch tensors or streams."""
        if not self.enabled:
            return {"status": "DISABLED", "role": role, "block_count": 0}
        blocks = list(getattr(module, "blocks", ()))
        if len(blocks) != 40:
            raise AssertionError(f"expected 40 blocks for {role}, got {len(blocks)}")
        for index, block in enumerate(blocks):
            original = getattr(block, "forward")
            if getattr(original, "__distillschedule_block_marker__", False):
                continue

            @functools.wraps(original)
            def marked_forward(*args: Any, __original=original, __index=index, **kwargs: Any) -> Any:
                with operation_marker(f"FSDP_BLOCK_FORWARD/{role}/block_{__index:02d}"):
                    return __original(*args, **kwargs)

            marked_forward.__distillschedule_block_marker__ = True
            setattr(block, "forward", marked_forward)
            self._patches.append((block, "forward", original))

            def backward_pre(mod: Any, grad_outputs: Any, __index=index) -> None:
                _push(f"FSDP_BLOCK_BACKWARD/{role}/block_{__index:02d}")

            def backward_post(mod: Any, grad_inputs: Any, grad_outputs: Any, __index=index) -> None:
                _pop()

            self._handles.append(block.register_full_backward_pre_hook(backward_pre, prepend=True))
            self._handles.append(block.register_full_backward_hook(backward_post, prepend=False))
            self.block_markers.append({"role": role, "block_index": index, "forward": True, "backward": True})
        return {"status": "ENABLED", "role": role, "block_count": len(blocks), "markers": list(self.block_markers)}

    def uninstall(self) -> None:
        global _ACTIVE
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()
        for owner, name, original in reversed(self._patches):
            current = getattr(owner, name, None)
            if getattr(current, "__distillschedule_marker__", False) or getattr(current, "__distillschedule_block_marker__", False):
                setattr(owner, name, original)
        self._patches.clear()
        if _ACTIVE is self:
            _ACTIVE = self._previous
        self._previous = None
        self._installed = False

    @property
    def marker_status(self) -> dict[str, Any]:
        if not self.enabled:
            return {"status": "DISABLED", "binding": "NOT_INSTRUMENTED"}
        return {
            "status": "ENABLED",
            "FSDP_UNSHARD_PRE_FORWARD": "BOUND:FSDPParamGroup.pre_forward/unshard",
            "FSDP_RESHARD_POST_FORWARD": "BOUND:FSDPParamGroup.post_forward/reshard",
            "FSDP_UNSHARD_PRE_BACKWARD": "BOUND:FSDPParamGroup.pre_backward/unshard",
            "FSDP_REDUCE_SCATTER": "BOUND:FSDPParamGroup.post_backward/foreach_reduce",
            "FSDP_BLOCK_FORWARD": "BOUND:module.forward wrapper",
            "FSDP_BLOCK_BACKWARD": "BOUND:full_backward_pre/post_hook",
        }


def install_fsdp2_operation_markers(*, enabled: bool = False) -> OperationMarkerSession:
    return OperationMarkerSession(enabled=enabled)


__all__ = ["OperationMarkerSession", "install_fsdp2_operation_markers", "operation_marker"]
