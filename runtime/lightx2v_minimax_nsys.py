"""Profiling-only NVTX and CUDA-profiler controls for MiniMax-H3 DMD.

This module is intentionally opt-in.  It does not change model math, FSDP2,
activation offload, checkpointing, optimizer order, or formal timing.  It only
adds rank-qualified NVTX ranges and bounds a node-level Nsight Systems capture
with cudaProfilerStart/cudaProfilerStop.

Only local rank zero on each node controls the CUDA profiler API.  Every rank
writes a receipt.  This mirrors the validated static-FSDP2 profiling pattern
and prevents the fastest of eight workers from terminating the node capture.
"""
from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path
from typing import Any


def _enabled() -> bool:
    return os.environ.get("LIGHTX2V_MINIMAX_NSYS_PROFILE", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite MiniMax Nsight receipt: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


class MiniMaxNsightRange:
    """One profiling capture per node, with receipts from every global rank."""

    def __init__(self, receipt_dir: str | Path) -> None:
        from distillschedule.runtime.nsys_range import TorchCudaProfilerApi

        self.rank = int(os.environ.get("RANK", "-1"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        if self.rank < 0 or self.local_rank < 0:
            raise RuntimeError("RANK and LOCAL_RANK are required for MiniMax Nsight profiling")
        self.controls_profiler_api = self.local_rank == 0
        self.receipt_path = Path(receipt_dir) / f"rank_{self.rank:03d}.json"
        self.api = TorchCudaProfilerApi()
        self.started = False
        self.stopped = False
        self.start_unix_ns: int | None = None
        self.start_perf_ns: int | None = None

    def start(self) -> None:
        if self.started:
            return
        self.start_unix_ns = time.time_ns()
        self.start_perf_ns = time.perf_counter_ns()
        if self.controls_profiler_api:
            self.api.start()
        self.started = True

    def stop(self, *, status: str = "PASS", error: BaseException | None = None) -> None:
        if not self.started or self.stopped:
            return
        if self.controls_profiler_api:
            self.api.stop()
        stop_perf_ns = time.perf_counter_ns()
        stop_unix_ns = time.time_ns()
        self.stopped = True
        assert self.start_perf_ns is not None and self.start_unix_ns is not None
        _atomic_json(
            self.receipt_path,
            {
                "schema_version": "distillschedule.lightx2v_minimax_nsys_range.v1",
                "status": status,
                "diagnostic_only": True,
                "formal_speedup_eligible": False,
                "rank": self.rank,
                "local_rank": self.local_rank,
                "controls_profiler_api": self.controls_profiler_api,
                "start_unix_ns": self.start_unix_ns,
                "stop_unix_ns": stop_unix_ns,
                "elapsed_ms": (stop_perf_ns - self.start_perf_ns) / 1_000_000.0,
                "host": socket.gethostname(),
                "error_type": type(error).__name__ if error is not None else None,
                "error": str(error) if error is not None else None,
            },
        )


def install_minimax_nsys_markers(
    trainer_class: Any,
    *,
    receipt_dir: str | Path,
) -> Any:
    """Install profiling-only wrappers on the dynamic matched trainer class."""
    if not _enabled():
        return trainer_class
    if getattr(trainer_class, "_distillschedule_minimax_nsys_installed", False):
        return trainer_class

    import torch
    import torch.distributed as dist

    original_train_one_stage = trainer_class._train_one_stage
    original_predict_velocity = trainer_class._predict_velocity
    original_candidate_fake_update = getattr(trainer_class, "_candidate_fake_update", None)

    def get_range(self: Any) -> MiniMaxNsightRange:
        value = getattr(self, "_distillschedule_minimax_nsys_range", None)
        if value is None:
            root = os.environ.get("LIGHTX2V_MINIMAX_NSYS_RECEIPT_DIR", "").strip()
            if not root:
                root = str(Path(receipt_dir) / "nsys_range_receipts")
            value = MiniMaxNsightRange(root)
            self._distillschedule_minimax_nsys_range = value
        return value

    def ensure_profile_started(profiler: MiniMaxNsightRange) -> None:
        if profiler.started:
            return
        # Profiling-only barriers sit before the benchmark's own cycle timer.
        # They guarantee that local-rank zero starts the node capture before
        # any peer emits the first rank-qualified NVTX stage marker.
        if dist.is_initialized():
            dist.barrier()
        profiler.start()
        if dist.is_initialized():
            dist.barrier()

    def next_stage_index(self: Any, stage: str) -> int:
        counters = getattr(self, "_distillschedule_minimax_nsys_stage_counts", None)
        if counters is None:
            counters = {"student": 0, "fake": 0}
            self._distillschedule_minimax_nsys_stage_counts = counters
        index = int(counters.get(stage, 0))
        counters[stage] = index + 1
        return index

    def wrapped_train_one_stage(self: Any, samples: Any, stage: str, grad_accum_iters: int):
        profiler = get_range(self)
        ensure_profile_started(profiler)
        index = next_stage_index(self, stage)
        rank = int(os.environ["RANK"])
        label = f"MINIMAX_NSYS_STAGE/{self.benchmark_mode}/rank={rank}/{stage}/{index}"
        torch.cuda.nvtx.range_push(label)
        failed: BaseException | None = None
        try:
            return original_train_one_stage(self, samples, stage, grad_accum_iters)
        except BaseException as exc:
            failed = exc
            raise
        finally:
            torch.cuda.nvtx.range_pop()
            if failed is not None:
                profiler.stop(status="FAIL", error=failed)
            elif stage == "fake":
                completed = int(getattr(self, "_benchmark_backward_counts", {}).get("fake", 0))
                expected = int(getattr(self, "fake_update_ratio", 5))
                if completed >= expected:
                    profiler.stop()

    def wrapped_predict_velocity(self: Any, model: Any, *args: Any, **kwargs: Any):
        if model is self.model:
            role = "student"
        elif model is self.fake_model:
            role = "fake"
        elif model is self.teacher_model:
            role = "teacher"
        else:
            role = "unknown"
        call_index = int(getattr(self, "_distillschedule_minimax_nsys_dit_index", 0)) + 1
        self._distillschedule_minimax_nsys_dit_index = call_index
        grad = "grad" if torch.is_grad_enabled() else "nograd"
        rank = int(os.environ["RANK"])
        label = (
            f"MINIMAX_NSYS_DIT/{self.benchmark_mode}/rank={rank}/"
            f"{role}/{grad}/{call_index}"
        )
        torch.cuda.nvtx.range_push(label)
        try:
            return original_predict_velocity(self, model, *args, **kwargs)
        finally:
            torch.cuda.nvtx.range_pop()

    trainer_class._train_one_stage = wrapped_train_one_stage
    trainer_class._predict_velocity = wrapped_predict_velocity

    if original_candidate_fake_update is not None:
        def wrapped_candidate_fake_update(self: Any, plan: dict[str, Any]):
            profiler = get_range(self)
            ensure_profile_started(profiler)
            index = next_stage_index(self, "fake")
            rank = int(os.environ["RANK"])
            label = (
                f"MINIMAX_NSYS_STAGE/{self.benchmark_mode}/rank={rank}/fake/{index}"
            )
            torch.cuda.nvtx.range_push(label)
            failed: BaseException | None = None
            try:
                return original_candidate_fake_update(self, plan)
            except BaseException as exc:
                failed = exc
                raise
            finally:
                torch.cuda.nvtx.range_pop()
                if failed is not None:
                    profiler.stop(status="FAIL", error=failed)
                else:
                    completed = int(
                        getattr(self, "_benchmark_backward_counts", {}).get("fake", 0)
                    )
                    expected = int(getattr(self, "fake_update_ratio", 5))
                    if completed >= expected:
                        profiler.stop()

        trainer_class._candidate_fake_update = wrapped_candidate_fake_update

    trainer_class._distillschedule_minimax_nsys_installed = True
    return trainer_class


__all__ = ["MiniMaxNsightRange", "install_minimax_nsys_markers"]
