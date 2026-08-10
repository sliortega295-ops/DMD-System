"""Exact Nsight capture-range controller for the GAN Full5 model window.

Formal speedup is always measured by unprofiled wall clock.  This helper exists
only for diagnostic Nsight Systems attribution.  When profiling is requested,
each training rank starts CUDA profiler collection immediately before the first
model compute of logical step 1000 and stops it after the Critic optimizer
commit of logical step 1004.

The outer nsys command uses ``--capture-range=cudaProfilerApi``.  A receipt is
written only after a balanced cudaProfilerStart/cudaProfilerStop pair, so a
report without a matching receipt cannot be described as the exact five-step
profile.
"""
from __future__ import annotations

import json
import os
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


SCHEMA_VERSION = "distillschedule.gan_nsys_range.v1"
START_STEP = 1000
STOP_STEP = 1004


class ProfilerApi(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...


class TorchCudaProfilerApi:
    """Thin checked wrapper around CUDA Runtime profiler start/stop APIs."""

    @staticmethod
    def _check(result: Any, operation: str) -> None:
        # torch.cuda.cudart() bindings have returned either an integer-like CUDA
        # status or a tuple whose first element is the status across PyTorch
        # versions.  cudaSuccess is zero.
        status = result[0] if isinstance(result, tuple) and result else result
        if status is None:
            return
        try:
            code = int(status)
        except (TypeError, ValueError):
            code = int(getattr(status, "value", 0))
        if code != 0:
            raise RuntimeError(f"{operation} failed with CUDA runtime status {code}")

    def start(self) -> None:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable for Nsight capture")
        cudart = torch.cuda.cudart()
        self._check(cudart.cudaProfilerStart(), "cudaProfilerStart")

    def stop(self) -> None:
        import torch

        cudart = torch.cuda.cudart()
        self._check(cudart.cudaProfilerStop(), "cudaProfilerStop")


@dataclass(frozen=True)
class NsightRangeReceipt:
    rank: int
    start_step: int
    stop_step: int
    start_point: str
    stop_point: str
    start_unix_ns: int
    stop_unix_ns: int
    elapsed_ms: float
    host: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS",
            "profiled": True,
            "diagnostic_only": True,
            "rank": self.rank,
            "logical_window": [self.start_step, self.stop_step],
            "start_point": self.start_point,
            "stop_point": self.stop_point,
            "start_unix_ns": self.start_unix_ns,
            "stop_unix_ns": self.stop_unix_ns,
            "elapsed_ms": self.elapsed_ms,
            "host": self.host,
            "formal_speedup_eligible": False,
        }


class GanFull5NsightRange:
    """One-shot CUDA profiler range tied to exact logical training boundaries."""

    def __init__(
        self,
        *,
        api: ProfilerApi | None = None,
        receipt_path: str | Path | None = None,
        rank: int | None = None,
        enabled: bool = True,
    ) -> None:
        self.api = api or TorchCudaProfilerApi()
        self.receipt_path = Path(receipt_path) if receipt_path is not None else None
        self.rank = int(os.environ.get("RANK", "-1")) if rank is None else int(rank)
        self.enabled = bool(enabled)
        self._started = False
        self._stopped = False
        self._start_unix_ns: int | None = None
        self._start_perf_ns: int | None = None

    @classmethod
    def from_env(cls) -> "GanFull5NsightRange":
        raw = os.environ.get("DISTILLSCHEDULE_GAN_NSYS_EXACT_RANGE", "0")
        enabled = raw.strip().lower() in {"1", "true", "yes", "on"}
        receipt_root = os.environ.get("DISTILLSCHEDULE_GAN_NSYS_RECEIPT_DIR", "").strip()
        path = None
        if enabled:
            if not receipt_root:
                raise ValueError(
                    "DISTILLSCHEDULE_GAN_NSYS_RECEIPT_DIR is required for exact Nsight range"
                )
            rank = int(os.environ.get("RANK", "-1"))
            if rank < 0:
                raise ValueError("RANK is required for exact Nsight range receipt")
            path = Path(receipt_root) / f"rank_{rank:03d}.json"
        return cls(receipt_path=path, enabled=enabled)

    @property
    def active(self) -> bool:
        return self._started and not self._stopped

    def before_model_compute(self, logical_step: int) -> None:
        """Call at the model-compute boundary of every logical step."""

        if not self.enabled or logical_step != START_STEP:
            return
        if self._started:
            raise RuntimeError("GAN Full5 Nsight range started more than once")
        self._start_unix_ns = time.time_ns()
        self._start_perf_ns = time.perf_counter_ns()
        self.api.start()
        self._started = True

    def after_critic_commit(self, logical_step: int) -> NsightRangeReceipt | None:
        """Call immediately after the Critic optimizer/scheduler commit."""

        if not self.enabled or logical_step != STOP_STEP:
            return None
        if not self._started:
            raise RuntimeError("GAN Full5 Nsight range stopped before it started")
        if self._stopped:
            raise RuntimeError("GAN Full5 Nsight range stopped more than once")
        self.api.stop()
        stop_perf_ns = time.perf_counter_ns()
        stop_unix_ns = time.time_ns()
        self._stopped = True
        assert self._start_perf_ns is not None and self._start_unix_ns is not None
        receipt = NsightRangeReceipt(
            rank=self.rank,
            start_step=START_STEP,
            stop_step=STOP_STEP,
            start_point="before_first_model_compute",
            stop_point="after_step1004_critic_commit",
            start_unix_ns=self._start_unix_ns,
            stop_unix_ns=stop_unix_ns,
            elapsed_ms=(stop_perf_ns - self._start_perf_ns) / 1_000_000.0,
            host=socket.gethostname(),
        )
        if self.receipt_path is not None:
            self._write_receipt(receipt)
        return receipt

    def assert_complete(self) -> None:
        if self.enabled and not (self._started and self._stopped):
            raise AssertionError(
                "exact GAN Nsight capture did not observe both Full5 boundaries"
            )

    def _write_receipt(self, receipt: NsightRangeReceipt) -> None:
        assert self.receipt_path is not None
        path = self.receipt_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f"refusing to overwrite Nsight range receipt: {path}")
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(receipt.to_dict(), indent=2, sort_keys=True) + "\n")
        os.replace(tmp, path)


def validate_nsys_receipts(root: str | Path, *, expected_world_size: int = 16) -> dict[str, Any]:
    """Require one exact balanced receipt from every training rank."""

    root = Path(root)
    files = sorted(root.glob("rank_*.json"))
    if len(files) != expected_world_size:
        raise ValueError(
            f"Nsight receipt count={len(files)}, expected={expected_world_size}"
        )
    values: list[dict[str, Any]] = []
    for expected_rank, path in enumerate(files):
        value = json.loads(path.read_text())
        if value.get("schema_version") != SCHEMA_VERSION or value.get("status") != "PASS":
            raise ValueError(f"invalid Nsight range receipt: {path}")
        if int(value.get("rank", -1)) != expected_rank:
            raise ValueError(f"Nsight range receipt rank drift: {path}")
        if value.get("logical_window") != [START_STEP, STOP_STEP]:
            raise ValueError(f"Nsight range window drift: {path}")
        if value.get("start_point") != "before_first_model_compute":
            raise ValueError(f"Nsight range start boundary drift: {path}")
        if value.get("stop_point") != "after_step1004_critic_commit":
            raise ValueError(f"Nsight range stop boundary drift: {path}")
        if value.get("diagnostic_only") is not True or value.get("formal_speedup_eligible") is not False:
            raise ValueError(f"profiled run incorrectly marked formal: {path}")
        values.append(value)
    return {
        "schema_version": "distillschedule.gan_nsys_receipt_bundle.v1",
        "status": "PASS",
        "world_size": expected_world_size,
        "logical_window": [START_STEP, STOP_STEP],
        "profiled": True,
        "diagnostic_only": True,
        "formal_speedup_eligible": False,
        "receipts": [str(path) for path in files],
    }


__all__ = [
    "GanFull5NsightRange",
    "NsightRangeReceipt",
    "ProfilerApi",
    "SCHEMA_VERSION",
    "START_STEP",
    "STOP_STEP",
    "TorchCudaProfilerApi",
    "validate_nsys_receipts",
]
