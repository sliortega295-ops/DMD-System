"""Low-intrusion JSONL phase tracing for DistillSchedule experiments.

The tracer is deliberately independent of Helios internals. Callers annotate the
semantic phase and the tracer records wall-clock, CUDA-event, stream, and allocator
state without changing the model or optimizer. CUDA and NVTX are optional: the
same API remains usable in CPU-only tests and environments.
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional, Union

try:  # Keep trace parsing/tests usable in a CPU-only Python environment.
    import torch
except ImportError:  # pragma: no cover - only in torch-free environments.
    torch = None  # type: ignore[assignment]


TRACE_SCHEMA_VERSION = "distillschedule.phase_trace.v1"
PathLike = Union[str, os.PathLike[str]]


def _environment_rank() -> Dict[str, Optional[int]]:
    def read_int(name: str) -> Optional[int]:
        value = os.environ.get(name)
        if value is None:
            return None
        try:
            return int(value)
        except ValueError:
            return None

    return {
        "rank": read_int("RANK"),
        "local_rank": read_int("LOCAL_RANK"),
        "world_size": read_int("WORLD_SIZE"),
    }


def _json_default(value: Any) -> str:
    """Make diagnostic metadata serializable without failing the training step."""

    return str(value)


class PhaseTracer:
    """Append one JSON object per annotated phase.

    Args:
        output_path: Destination JSONL file.
        cuda: ``None`` auto-detects CUDA, ``False`` forces the CPU-only path,
            and ``True`` requests CUDA but still degrades if it is unavailable.
        nvtx: Emit an NVTX range when CUDA/NVTX are available.
        synchronize_cuda_events: Synchronize the phase end event to report a
            reliable GPU elapsed time. This creates a scheduling boundary, so
            disable it for low-perturbation traces.
        base_fields: Experiment-wide metadata included in every record.
    """

    def __init__(
        self,
        output_path: PathLike,
        *,
        cuda: Optional[bool] = None,
        nvtx: bool = True,
        synchronize_cuda_events: bool = True,
        base_fields: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.cuda_requested = cuda
        self.nvtx_requested = bool(nvtx)
        self.synchronize_cuda_events = bool(synchronize_cuda_events)
        self.base_fields = dict(base_fields or {})
        self._write_lock = threading.Lock()

    def _cuda_is_enabled(self) -> bool:
        if self.cuda_requested is False or torch is None:
            return False
        try:
            available = bool(torch.cuda.is_available())
        except Exception:
            return False
        return available and self.cuda_requested in (None, True)

    @staticmethod
    def _memory_snapshot(device: Any) -> Dict[str, int]:
        assert torch is not None
        return {
            "allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        }

    @staticmethod
    def _record_error(errors: list[str], where: str, error: BaseException) -> None:
        errors.append(f"{where}: {type(error).__name__}: {error}")

    def _append(self, record: Mapping[str, Any]) -> None:
        # O_APPEND plus one logical write avoids partially interleaved lines in
        # the common local/shared-filesystem case.
        payload = (json.dumps(record, sort_keys=True, default=_json_default) + "\n").encode(
            "utf-8"
        )
        with self._write_lock:
            descriptor = os.open(
                self.output_path,
                os.O_APPEND | os.O_CREAT | os.O_WRONLY,
                0o644,
            )
            try:
                view = memoryview(payload)
                while view:
                    written = os.write(descriptor, view)
                    view = view[written:]
            finally:
                os.close(descriptor)

    @contextmanager
    def phase(
        self,
        phase: str,
        *,
        op: Optional[str] = None,
        iteration: Optional[int] = None,
        section: Optional[int] = None,
        denoise: Optional[int] = None,
        grad_enabled: Optional[bool] = None,
        adapter: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Iterator[None]:
        """Trace one semantic operation while preserving exceptions."""

        errors: list[str] = []
        cuda_enabled = self._cuda_is_enabled()
        cuda_device: Any = None
        stream_id: Optional[int] = None
        start_event: Any = None
        end_event: Any = None
        start_memory: Dict[str, int] = {}
        nvtx_pushed = False

        if grad_enabled is None:
            grad_enabled = bool(torch.is_grad_enabled()) if torch is not None else None

        nvtx_label_parts = [f"phase={phase}"]
        if op is not None:
            nvtx_label_parts.append(f"op={op}")
        if iteration is not None:
            nvtx_label_parts.append(f"iter={iteration}")
        if section is not None:
            nvtx_label_parts.append(f"section={section}")
        if denoise is not None:
            nvtx_label_parts.append(f"denoise={denoise}")
        if adapter is not None:
            nvtx_label_parts.append(f"adapter={adapter}")
        nvtx_label = "|".join(nvtx_label_parts)

        if cuda_enabled:
            assert torch is not None
            try:
                cuda_device = torch.cuda.current_device()
                stream = torch.cuda.current_stream(cuda_device)
                raw_stream_id = getattr(stream, "cuda_stream", None)
                stream_id = int(raw_stream_id) if raw_stream_id is not None else None
                start_memory = self._memory_snapshot(cuda_device)
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record(stream)
            except Exception as error:
                self._record_error(errors, "cuda_start", error)
                cuda_enabled = False
                cuda_device = None
                stream_id = None
                start_event = None
                end_event = None
                start_memory = {}

        if cuda_enabled and self.nvtx_requested:
            assert torch is not None
            try:
                torch.cuda.nvtx.range_push(nvtx_label)
                nvtx_pushed = True
            except Exception as error:
                self._record_error(errors, "nvtx_push", error)

        start_unix_ns = time.time_ns()
        start_perf_ns = time.perf_counter_ns()
        caught_error: Optional[BaseException] = None

        try:
            yield
        except BaseException as error:
            caught_error = error
            raise
        finally:
            end_perf_ns = time.perf_counter_ns()
            end_unix_ns = time.time_ns()

            if nvtx_pushed:
                assert torch is not None
                try:
                    torch.cuda.nvtx.range_pop()
                except Exception as error:
                    self._record_error(errors, "nvtx_pop", error)

            cuda_event_time_ms: Optional[float] = None
            end_memory: Dict[str, int] = {}
            if cuda_enabled:
                assert torch is not None
                try:
                    end_event.record(torch.cuda.current_stream(cuda_device))
                    if self.synchronize_cuda_events:
                        end_event.synchronize()
                        cuda_event_time_ms = float(start_event.elapsed_time(end_event))
                    elif bool(end_event.query()):
                        cuda_event_time_ms = float(start_event.elapsed_time(end_event))
                except Exception as error:
                    self._record_error(errors, "cuda_event_end", error)
                try:
                    end_memory = self._memory_snapshot(cuda_device)
                except Exception as error:
                    self._record_error(errors, "cuda_memory_end", error)

            record: Dict[str, Any] = {
                "schema_version": TRACE_SCHEMA_VERSION,
                "phase": phase,
                "op": op,
                "iteration": iteration,
                "section": section,
                "denoise": denoise,
                "grad_enabled": grad_enabled,
                "adapter": adapter,
                "status": "error" if caught_error is not None else "ok",
                "wall_start_unix_ns": start_unix_ns,
                "wall_end_unix_ns": end_unix_ns,
                "wall_time_ms": (end_perf_ns - start_perf_ns) / 1_000_000.0,
                "cuda_requested": self.cuda_requested,
                "cuda_enabled": cuda_enabled,
                "cuda_event_synchronized": bool(
                    cuda_enabled and self.synchronize_cuda_events
                ),
                "cuda_event_time_ms": cuda_event_time_ms,
                "stream_id": stream_id,
                "memory_allocated_start_bytes": start_memory.get("allocated_bytes"),
                "memory_reserved_start_bytes": start_memory.get("reserved_bytes"),
                "max_memory_allocated_start_bytes": start_memory.get(
                    "max_allocated_bytes"
                ),
                "max_memory_reserved_start_bytes": start_memory.get(
                    "max_reserved_bytes"
                ),
                "memory_allocated_bytes": end_memory.get("allocated_bytes"),
                "memory_reserved_bytes": end_memory.get("reserved_bytes"),
                "max_memory_allocated_bytes": end_memory.get("max_allocated_bytes"),
                "max_memory_reserved_bytes": end_memory.get("max_reserved_bytes"),
                "nvtx_requested": self.nvtx_requested,
                "nvtx_emitted": nvtx_pushed,
                "process_id": os.getpid(),
                "thread_id": threading.get_ident(),
                "hostname": socket.gethostname(),
                "base_fields": self.base_fields,
                "metadata": dict(metadata or {}),
                "instrumentation_errors": errors,
            }
            record.update(_environment_rank())

            if cuda_device is not None and torch is not None:
                record["cuda_device"] = int(cuda_device)
                try:
                    record["cuda_device_name"] = torch.cuda.get_device_name(cuda_device)
                except Exception as error:
                    self._record_error(errors, "cuda_device_name", error)
                    record["cuda_device_name"] = None
            else:
                record["cuda_device"] = None
                record["cuda_device_name"] = None

            if caught_error is not None:
                record["error_type"] = type(caught_error).__name__
                record["error_message"] = str(caught_error)

            self._append(record)

    __call__ = phase


__all__ = ["PhaseTracer", "TRACE_SCHEMA_VERSION"]
