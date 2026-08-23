"""Opt-in host-synchronization cleanup for formal timing paths.

The helpers here do not remove numerical validation. They enqueue the same
finite reductions on the current CUDA stream and defer the Python-visible
readback until after the caller's formal timing boundary.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class DeferredFiniteChecks:
    """Collect device-side finite flags and validate them after the timer."""

    _checks: list[tuple[str, torch.Tensor]] = field(default_factory=list)

    def record(self, name: str, value: torch.Tensor) -> None:
        if not torch.is_tensor(value):
            raise TypeError(f"finite check {name!r} is not a tensor")
        self._checks.append((str(name), torch.isfinite(value).all()))

    def finalize(self) -> dict[str, object]:
        names = [name for name, _flag in self._checks]
        if not self._checks:
            return {"status": "PASS", "count": 0, "failed": []}
        flags = torch.stack([flag for _name, flag in self._checks]).detach().cpu()
        failed = [name for name, ok in zip(names, flags.tolist(), strict=True) if not ok]
        if failed:
            raise AssertionError(f"deferred finite checks failed: {failed}")
        return {"status": "PASS", "count": len(names), "failed": []}
