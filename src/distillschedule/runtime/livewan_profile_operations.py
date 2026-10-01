"""Default-off issuing-thread autograd NVTX, diagnostic profiles only.

Torch emit_nvtx synchronizes on context entry/exit. Both calls are explicitly
outside the Full5 NVTX/timer boundary; no model-call fences or tensor reads are
added. This module never imports torch and cannot authorize native timing.
"""
from __future__ import annotations

from typing import Any, Mapping

FLAG = "LIVEWAN_PROFILE_OPERATION_MARKERS"
RANKS = "LIVEWAN_PROFILE_OPERATION_RANKS"


class OperationMarkers:
    def __init__(self, *, run_kind: str, rank: int, environ: Mapping[str, str]):
        flag = environ.get(FLAG, "0")
        if flag not in {"0", "1"}:
            raise RuntimeError("invalid profile operation marker flag")
        if flag == "1" and run_kind != "profile":
            raise RuntimeError("operation markers are diagnostic-only")
        raw = environ.get(RANKS, "0,8")
        try:
            ranks = [int(v) for v in raw.split(",")]
        except ValueError as exc:
            raise RuntimeError("invalid operation marker ranks") from exc
        if not ranks or len(set(ranks)) != len(ranks) or any(not 0 <= v < 16 for v in ranks):
            raise RuntimeError("invalid operation marker ranks")
        self.requested = flag == "1"
        self.selected = self.requested and rank in ranks
        self.ranks = ranks
        self.rank = rank
        self.context: Any = None
        self.entries = self.exits = 0
        self.active = False

    def start(self, torch: Any) -> None:
        if not self.selected:
            return
        if self.active or self.entries:
            raise RuntimeError("operation markers cannot re-enter")
        if str(torch.__version__) != "2.10.0+cu128":
            raise RuntimeError("operation markers require audited Torch 2.10 substrate")
        self.context = torch.autograd.profiler.emit_nvtx(record_shapes=True)
        self.context.__enter__()
        self.entries += 1
        self.active = True

    def stop(self) -> None:
        if not self.selected:
            return
        if not self.active or self.context is None:
            raise RuntimeError("operation marker lifecycle incomplete")
        self.context.__exit__(None, None, None)
        self.exits += 1
        self.active = False

    def receipt(self) -> dict[str, Any]:
        if self.selected and (self.active or (self.entries, self.exits) != (1, 1)):
            raise RuntimeError("operation markers not closed")
        return dict(
            enabled=self.requested, rank=self.rank, selected_ranks=self.ranks,
            execution_status="PASS" if self.selected else ("NOT_APPLICABLE" if self.requested else "DISABLED"),
            entries=self.entries, exits=self.exits, active=self.active,
            mechanism="torch.autograd.profiler.emit_nvtx",
            record_shapes=True, tensor_values_read=False, model_call_sync_added=False,
            boundary_sync="TORCH_EMIT_NVTX_ENTRY_EXIT_OUTSIDE_FULL5_ONLY",
            claim_boundary="DIAGNOSTIC_ONLY_NOT_NATIVE_TIMING",
        )
