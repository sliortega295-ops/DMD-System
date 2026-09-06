"""Low-overhead, profile-only phase and logical-byte diagnostics.

The recorder never changes a tensor's device or lifetime.  It emits NVTX
envelopes and counts logical tensor bytes at the application boundary; CUDA
activity bytes remain the authoritative observed-copy source in the SQLite
analyzer.  Records are kept in memory and written after ``cudaProfilerStop``
so output serialization cannot extend the captured range.
"""
from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


@dataclasses.dataclass
class ByteCounter:
    bytes: int = 0
    events: int = 0


class StaticProfileDiagnostics:
    def __init__(
        self,
        *,
        enabled: bool,
        rank: int,
        output_path: str | Path | None = None,
        collect_cpu_diagnostics: bool = True,
    ) -> None:
        self.enabled = bool(enabled)
        self.collect_cpu_diagnostics = bool(collect_cpu_diagnostics)
        self.rank = int(rank)
        self.output_path = Path(output_path) if output_path is not None else None
        self.counters: dict[str, ByteCounter] = {}
        self.events: list[dict[str, Any]] = []

    @contextlib.contextmanager
    def scope(self, label: str) -> Iterator[None]:
        if self.enabled:
            import torch

            torch.cuda.nvtx.range_push(str(label))
        try:
            yield
        finally:
            if self.enabled:
                import torch

                torch.cuda.nvtx.range_pop()

    def add_bytes(self, key: str, value: int, *, direction: str, source: str = "application") -> None:
        if not self.enabled:
            return
        counter = self.counters.setdefault(f"{direction}/{key}", ByteCounter())
        counter.bytes += int(value)
        counter.events += 1
        self.events.append({"kind": "logical_bytes", "key": key, "direction": direction, "bytes": int(value), "source": source})

    def add_tensor_tree(self, key: str, value: Any, *, direction: str) -> int:
        """Count each distinct tensor in a nested trace entry once."""
        if not self.enabled:
            return 0
        import torch

        seen: set[int] = set()

        def visit(item: Any) -> int:
            if isinstance(item, torch.Tensor):
                marker = id(item)
                if marker in seen:
                    return 0
                seen.add(marker)
                size = int(item.numel() * item.element_size())
                self.add_bytes(key, size, direction=direction, source="trace_tensor_tree")
                return size
            if dataclasses.is_dataclass(item):
                return sum(visit(getattr(item, field.name)) for field in dataclasses.fields(item))
            if isinstance(item, dict):
                return sum(visit(child) for child in item.values())
            if isinstance(item, (tuple, list)):
                return sum(visit(child) for child in item)
            return 0

        return visit(value)

    def diagnostic_scalar(self, key: str, value: Any) -> Any:
        """Materialize a diagnostic scalar only when diagnostics are enabled."""
        if not self.enabled or not self.collect_cpu_diagnostics:
            return None
        with self.scope(f"DIAG_LOSS_D2H/{key}"):
            return float(value.detach().float().cpu())

    def diagnostic_sha256(self, key: str, value: Any) -> str | None:
        if not self.enabled or not self.collect_cpu_diagnostics:
            return None
        import torch

        with self.scope(f"DIAG_OUTPUT_SHA_D2H/{key}"):
            raw = value.detach().contiguous().cpu().numpy().tobytes()
        digest = hashlib.sha256(raw).hexdigest()
        self.add_bytes(key, len(raw), direction="D2H", source="diagnostic_sha256")
        return digest

    def fingerprint_d2h(self, key: str, value: Any) -> bytes:
        if not self.enabled or not self.collect_cpu_diagnostics:
            return b""
        with self.scope(f"DIAG_FINGERPRINT_D2H/{key}"):
            return value.detach().float().cpu().numpy().tobytes()

    def dump_after_capture(self) -> None:
        if self.output_path is None:
            return
        payload = {
            "schema_version": "distillschedule.static_profile_diagnostics.v1",
            "rank": self.rank,
            "enabled": self.enabled,
            "collect_cpu_diagnostics": self.collect_cpu_diagnostics,
            "counters": {
                key: {"bytes": value.bytes, "events": value.events}
                for key, value in sorted(self.counters.items())
            },
            "events": self.events,
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output_path.with_suffix(self.output_path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        temporary.replace(self.output_path)


__all__ = ["ByteCounter", "StaticProfileDiagnostics"]
