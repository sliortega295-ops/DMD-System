"""Opt-in native range observations without synchronizing asymmetric GPU pools.

CUDA events are issued on the existing current stream. They measure contextual
stream envelopes, NOT isolated service time or compute/NCCL utilization. A
consumer must preserve dependencies/contention or mark a cost uncalibrated.
There is no tensor read, stream change, collective, or synchronization here.
The worker must finish its ordinary terminal synchronization before finish().
"""
from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter_ns
import threading
from typing import Any

from .contracts import require


class AsyncRangeCollector:
    def __init__(self, *, torch: Any, device: Any, rank: int, source_head: str,
                 implementation: str, clock=perf_counter_ns, runtime_metadata=None):
        self.torch, self.device, self.clock = torch, device, clock
        self.thread = threading.get_ident()
        self.rows, self.stack = [], []
        self.boundaries = []
        self.cycle, self.finished = 0, False
        self.identity = dict(rank=rank, source_head=source_head, implementation=implementation)
        self.runtime_metadata = runtime_metadata
        self.origin = self._event()
        self.static_bytes = int(torch.cuda.memory_allocated(device))

    def boundary(self, *, cycle: int, step: int):
        """Native commit boundary, not a fabricated zero-duration GPU operation.

        Consecutive boundaries delimit an indivisible coupled G/Score envelope.
        Its service includes native waits/lookahead. It must NEVER be added to
        independent G/Score costs or used to simulate a different pool DAG.
        """
        require(not self.finished and threading.get_ident() == self.thread, "invalid boundary thread/lifecycle")
        self.boundaries.append(dict(cycle=cycle, step=step, event=self._event(),
            host_ns=self.clock(), peak_allocated_bytes=int(self.torch.cuda.max_memory_allocated(self.device)),
            peak_reserved_bytes=int(self.torch.cuda.max_memory_reserved(self.device))))

    def _event(self):
        event = self.torch.cuda.Event(enable_timing=True)
        event.record(self.torch.cuda.current_stream(self.device))
        return event

    @contextmanager
    def range(self, label: str, *, metadata=None):
        require(not self.finished and threading.get_ident() == self.thread,
                "range collector is single issuing-thread and cannot resume")
        index = len(self.rows)
        row = {"index": index, "parent": self.stack[-1] if self.stack else None,
               "cycle": self.cycle, "label": label, "start_event": self._event(),
               "host_start_ns": self.clock(), "status": "RUNNING"}
        if metadata is not None:
            row['metadata'] = dict(metadata)
        if self.runtime_metadata is not None:
            row['runtime_before'] = dict(self.runtime_metadata())
        self.rows.append(row)
        self.stack.append(index)
        try:
            yield
            row["status"] = "PASS"
        except BaseException as error:
            row.update(status="FAIL", error_type=type(error).__name__, error=str(error))
            raise
        finally:
            row["host_stop_ns"] = self.clock()
            row["stop_event"] = self._event()
            row["allocated_at_return_bytes"] = int(self.torch.cuda.memory_allocated(self.device))
            if self.runtime_metadata is not None:
                row['runtime_after'] = dict(self.runtime_metadata())
            require(self.stack.pop() == index, "range nesting changed")

    def finish(self) -> dict:
        require(not self.finished and not self.stack and self.rows,
                "unfinished/empty/already finalized range collector")
        self.finished = True
        rows = []
        for raw in self.rows:
            require(raw["start_event"].query() and raw["stop_event"].query(),
                    "native terminal synchronization has not completed")
            start = round(self.origin.elapsed_time(raw["start_event"]) * 1_000_000)
            stop = round(self.origin.elapsed_time(raw["stop_event"]) * 1_000_000)
            require(0 <= start <= stop, "invalid CUDA range clock")
            rows.append({k: v for k, v in raw.items() if not k.endswith("_event")} |
                        {"start_ns": start, "stop_ns": stop, "duration_ns": stop - start})
        boundaries = []
        for raw in self.boundaries:
            require(raw["event"].query(), "commit boundary incomplete")
            boundaries.append({k: v for k, v in raw.items() if k != "event"} |
                              {"time_ns": round(self.origin.elapsed_time(raw["event"]) * 1_000_000)})
        return {"schema_version": "distillschedule.async_range_calibration.v1", **self.identity,
                "status": "PASS" if all(r["status"] == "PASS" for r in rows) else "FAIL",
                "calibration_only": True, "formal_timing_eligible": False,
                "profiled": False, "post_setup_static_bytes": self.static_bytes,
                "clock": "CUDA_CURRENT_STREAM_RELATIVE_TO_RANK_LOCAL_ORIGIN",
                "cost_scope": "CONTEXTUAL_STREAM_ENVELOPES_INCLUDING_DEPENDENCY_AND_HOST_GAPS",
                "cross_rank_clock_alignment": "NOT_ESTABLISHED_BY_CUDA_EVENTS",
                "extra_synchronizations": 0, "extra_collectives": 0,
                "tensor_value_reads": 0, "rows": rows, "commit_boundaries": boundaries}
