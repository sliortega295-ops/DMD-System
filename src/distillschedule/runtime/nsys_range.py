"""Exact Nsight capture-range controllers for diagnostic Full5 windows.

Formal speedup is always measured by unprofiled wall clock.  This helper exists
only for diagnostic Nsight Systems attribution.  Every rank records the same
logical five-step boundary, while only local-rank zero on each node drives the
CUDA Runtime profiler start/stop API.  This avoids the fastest of eight local
ranks terminating a node-level Nsight capture early.

The node-local outer nsys command uses ``--capture-range=cudaProfilerApi``.  A
report is exact only when all 16 ranks emit balanced boundary receipts and the
two expected controller ranks (global 0 and 8 for 2x8) report API ownership.
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
STATIC_SCHEMA_VERSION = "distillschedule.static_fsdp2_nsys_range.v1"
START_STEP = 1000
STOP_STEP = 1004


class ProfilerApi(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...


class TorchCudaProfilerApi:
    """Thin checked wrapper around CUDA Runtime profiler start/stop APIs."""

    @staticmethod
    def _check(result: Any, operation: str) -> None:
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
    local_rank: int
    controls_profiler_api: bool
    start_step: int
    stop_step: int
    start_point: str
    stop_point: str
    start_unix_ns: int
    stop_unix_ns: int
    elapsed_ms: float
    host: str
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": "PASS",
            "profiled": True,
            "diagnostic_only": True,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "controls_profiler_api": self.controls_profiler_api,
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
        local_rank: int | None = None,
        enabled: bool = True,
        controls_profiler_api: bool | None = None,
    ) -> None:
        self.api = api or TorchCudaProfilerApi()
        self.receipt_path = Path(receipt_path) if receipt_path is not None else None
        self.rank = int(os.environ.get("RANK", "-1")) if rank is None else int(rank)
        self.local_rank = (
            int(os.environ.get("LOCAL_RANK", "-1"))
            if local_rank is None
            else int(local_rank)
        )
        self.enabled = bool(enabled)
        self.controls_profiler_api = (
            self.local_rank == 0
            if controls_profiler_api is None
            else bool(controls_profiler_api)
        )
        self._started = False
        self._stopped = False
        self._start_unix_ns: int | None = None
        self._start_perf_ns: int | None = None

    @classmethod
    def from_env(cls) -> "GanFull5NsightRange":
        raw = os.environ.get("DISTILLSCHEDULE_GAN_NSYS_EXACT_RANGE", "0")
        enabled = raw.strip().lower() in {"1", "true", "yes", "on"}
        receipt_root = os.environ.get("DISTILLSCHEDULE_GAN_NSYS_RECEIPT_DIR", "").strip()
        rank = int(os.environ.get("RANK", "-1"))
        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        path = None
        if enabled:
            if not receipt_root:
                raise ValueError(
                    "DISTILLSCHEDULE_GAN_NSYS_RECEIPT_DIR is required for exact Nsight range"
                )
            if rank < 0 or local_rank < 0:
                raise ValueError(
                    "RANK and LOCAL_RANK are required for exact Nsight range receipt"
                )
            path = Path(receipt_root) / f"rank_{rank:03d}.json"
        return cls(
            receipt_path=path,
            rank=rank,
            local_rank=local_rank,
            enabled=enabled,
            controls_profiler_api=(local_rank == 0),
        )

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
        if self.controls_profiler_api:
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
        if self.controls_profiler_api:
            self.api.stop()
        stop_perf_ns = time.perf_counter_ns()
        stop_unix_ns = time.time_ns()
        self._stopped = True
        assert self._start_perf_ns is not None and self._start_unix_ns is not None
        receipt = NsightRangeReceipt(
            rank=self.rank,
            local_rank=self.local_rank,
            controls_profiler_api=self.controls_profiler_api,
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


class StaticFSDP2Full5NsightRange:
    """Exact-range controller for the static world16 FSDP2 baseline.

    The static launcher uses one outer Nsight session per node.  Only local
    rank zero calls the CUDA profiler API; every rank writes a boundary receipt
    so that a node-level report cannot be mistaken for complete world16
    coverage.  The model path is unchanged and the controller is enabled only
    in explicit profile mode.
    """

    _WINDOWS = {
        "no-gan": (10, 14, "after_step14_critic_commit"),
        "gan": (1000, 1004, "after_step1004_critic_commit"),
    }

    def __init__(
        self,
        *,
        variant: str,
        api: ProfilerApi | None = None,
        receipt_path: str | Path | None = None,
        rank: int | None = None,
        local_rank: int | None = None,
        enabled: bool = True,
        controls_profiler_api: bool | None = None,
        logical_window: tuple[int, int] | None = None,
        start_point: str = "before_first_model_compute",
        stop_point: str | None = None,
    ) -> None:
        if variant not in self._WINDOWS:
            raise ValueError(f"unsupported static FSDP2 variant: {variant}")
        self.variant = variant
        default_start, default_stop, default_stop_point = self._WINDOWS[variant]
        self.start_step, self.stop_step = logical_window or (default_start, default_stop)
        self.start_point = str(start_point)
        self.stop_point = str(stop_point or default_stop_point)
        self.api = api or TorchCudaProfilerApi()
        self.receipt_path = Path(receipt_path) if receipt_path is not None else None
        self.rank = int(os.environ.get("RANK", "-1")) if rank is None else int(rank)
        self.local_rank = (
            int(os.environ.get("LOCAL_RANK", "-1"))
            if local_rank is None
            else int(local_rank)
        )
        self.enabled = bool(enabled)
        self.controls_profiler_api = (
            self.local_rank == 0
            if controls_profiler_api is None
            else bool(controls_profiler_api)
        )
        self._started = False
        self._stopped = False
        self._start_unix_ns: int | None = None
        self._start_perf_ns: int | None = None

    @classmethod
    def from_env(
        cls,
        *,
        variant: str,
        logical_window: tuple[int, int] | None = None,
        start_point: str = "before_first_model_compute",
        stop_point: str | None = None,
    ) -> "StaticFSDP2Full5NsightRange":
        raw = os.environ.get("DISTILLSCHEDULE_STATIC_FSDP2_NSYS_EXACT_RANGE", "0")
        enabled = raw.strip().lower() in {"1", "true", "yes", "on"}
        receipt_root = os.environ.get(
            "DISTILLSCHEDULE_STATIC_FSDP2_NSYS_RECEIPT_DIR", ""
        ).strip()
        rank = int(os.environ.get("RANK", "-1"))
        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        path = None
        if enabled:
            if not receipt_root:
                raise ValueError(
                    "DISTILLSCHEDULE_STATIC_FSDP2_NSYS_RECEIPT_DIR is required"
                )
            if rank < 0 or local_rank < 0:
                raise ValueError(
                    "RANK and LOCAL_RANK are required for static Nsight receipt"
                )
            path = Path(receipt_root) / f"rank_{rank:03d}.json"
        return cls(
            variant=variant,
            receipt_path=path,
            rank=rank,
            local_rank=local_rank,
            enabled=enabled,
            controls_profiler_api=(local_rank == 0),
            logical_window=logical_window,
            start_point=start_point,
            stop_point=stop_point,
        )

    @property
    def active(self) -> bool:
        return self._started and not self._stopped

    def before_model_compute(self, logical_step: int) -> None:
        if not self.enabled or logical_step != self.start_step:
            return
        if self._started:
            raise RuntimeError("static FSDP2 Nsight range started more than once")
        self._start_unix_ns = time.time_ns()
        self._start_perf_ns = time.perf_counter_ns()
        if self.controls_profiler_api:
            self.api.start()
        self._started = True

    def after_critic_commit(self, logical_step: int) -> NsightRangeReceipt | None:
        if not self.enabled or logical_step != self.stop_step:
            return None
        if not self._started:
            raise RuntimeError("static FSDP2 Nsight range stopped before it started")
        if self._stopped:
            raise RuntimeError("static FSDP2 Nsight range stopped more than once")
        if self.controls_profiler_api:
            self.api.stop()
        stop_perf_ns = time.perf_counter_ns()
        stop_unix_ns = time.time_ns()
        self._stopped = True
        assert self._start_perf_ns is not None and self._start_unix_ns is not None
        receipt = NsightRangeReceipt(
            rank=self.rank,
            local_rank=self.local_rank,
            controls_profiler_api=self.controls_profiler_api,
            start_step=self.start_step,
            stop_step=self.stop_step,
            start_point=self.start_point,
            stop_point=self.stop_point,
            start_unix_ns=self._start_unix_ns,
            stop_unix_ns=stop_unix_ns,
            elapsed_ms=(stop_perf_ns - self._start_perf_ns) / 1_000_000.0,
            host=socket.gethostname(),
            schema_version=STATIC_SCHEMA_VERSION,
        )
        if self.receipt_path is not None:
            self._write_receipt(receipt)
        return receipt

    def assert_complete(self) -> None:
        if self.enabled and not (self._started and self._stopped):
            raise AssertionError(
                "exact static FSDP2 Nsight capture did not observe both boundaries"
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


def validate_static_nsys_receipts(
    root: str | Path,
    *,
    variant: str,
    expected_world_size: int = 16,
    expected_local_world_size: int = 8,
    expected_window: tuple[int, int] | None = None,
    expected_start_point: str = "before_first_model_compute",
    expected_stop_point: str | None = None,
) -> dict[str, Any]:
    """Require exact static Full5 receipts and one controller per node."""

    if variant not in StaticFSDP2Full5NsightRange._WINDOWS:
        raise ValueError(f"unsupported static FSDP2 variant: {variant}")
    if expected_world_size <= 0 or expected_local_world_size <= 0:
        raise ValueError("expected world sizes must be positive")
    if expected_world_size % expected_local_world_size != 0:
        raise ValueError("world size must be divisible by local world size")
    root = Path(root)
    files = sorted(root.glob("rank_*.json"))
    if len(files) != expected_world_size:
        raise ValueError(
            f"static Nsight receipt count={len(files)}, expected={expected_world_size}"
        )
    default_start, default_stop, default_stop_point = (
        StaticFSDP2Full5NsightRange._WINDOWS[variant]
    )
    start_step, stop_step = expected_window or (default_start, default_stop)
    stop_point = expected_stop_point or default_stop_point
    controllers: list[int] = []
    for expected_rank, path in enumerate(files):
        value = json.loads(path.read_text())
        if value.get("schema_version") != STATIC_SCHEMA_VERSION or value.get("status") != "PASS":
            raise ValueError(f"invalid static Nsight range receipt: {path}")
        rank = int(value.get("rank", -1))
        local_rank = int(value.get("local_rank", -1))
        if rank != expected_rank:
            raise ValueError(f"static Nsight rank drift: {path}")
        if local_rank != rank % expected_local_world_size:
            raise ValueError(f"static Nsight local-rank drift: {path}")
        expected_controller = local_rank == 0
        if bool(value.get("controls_profiler_api")) != expected_controller:
            raise ValueError(f"static Nsight profiler-controller drift: {path}")
        if expected_controller:
            controllers.append(rank)
        if value.get("logical_window") != [start_step, stop_step]:
            raise ValueError(f"static Nsight range window drift: {path}")
        if value.get("start_point") != expected_start_point:
            raise ValueError(f"static Nsight range start boundary drift: {path}")
        if value.get("stop_point") != stop_point:
            raise ValueError(f"static Nsight range stop boundary drift: {path}")
        if value.get("diagnostic_only") is not True or value.get("formal_speedup_eligible") is not False:
            raise ValueError(f"static profiled run incorrectly marked formal: {path}")
    expected_controllers = list(range(0, expected_world_size, expected_local_world_size))
    if controllers != expected_controllers:
        raise ValueError(
            f"static Nsight controller ranks={controllers}, expected={expected_controllers}"
        )
    return {
        "schema_version": "distillschedule.static_fsdp2_nsys_receipt_bundle.v1",
        "status": "PASS",
        "variant": variant,
        "world_size": expected_world_size,
        "local_world_size": expected_local_world_size,
        "controller_ranks": controllers,
        "logical_window": [start_step, stop_step],
        "profiled": True,
        "diagnostic_only": True,
        "formal_speedup_eligible": False,
        "receipts": [str(path) for path in files],
    }


def validate_nsys_receipts(
    root: str | Path,
    *,
    expected_world_size: int = 16,
    expected_local_world_size: int = 8,
) -> dict[str, Any]:
    """Require exact balanced receipts and one profiler controller per node."""

    if expected_world_size <= 0 or expected_local_world_size <= 0:
        raise ValueError("expected world sizes must be positive")
    if expected_world_size % expected_local_world_size != 0:
        raise ValueError("world size must be divisible by local world size")
    root = Path(root)
    files = sorted(root.glob("rank_*.json"))
    if len(files) != expected_world_size:
        raise ValueError(
            f"Nsight receipt count={len(files)}, expected={expected_world_size}"
        )
    values: list[dict[str, Any]] = []
    controllers: list[int] = []
    for expected_rank, path in enumerate(files):
        value = json.loads(path.read_text())
        if value.get("schema_version") != SCHEMA_VERSION or value.get("status") != "PASS":
            raise ValueError(f"invalid Nsight range receipt: {path}")
        rank = int(value.get("rank", -1))
        local_rank = int(value.get("local_rank", -1))
        if rank != expected_rank:
            raise ValueError(f"Nsight range receipt rank drift: {path}")
        if local_rank != rank % expected_local_world_size:
            raise ValueError(f"Nsight range receipt local-rank drift: {path}")
        expected_controller = local_rank == 0
        if bool(value.get("controls_profiler_api")) != expected_controller:
            raise ValueError(f"Nsight profiler-controller drift: {path}")
        if expected_controller:
            controllers.append(rank)
        if value.get("logical_window") != [START_STEP, STOP_STEP]:
            raise ValueError(f"Nsight range window drift: {path}")
        if value.get("start_point") != "before_first_model_compute":
            raise ValueError(f"Nsight range start boundary drift: {path}")
        if value.get("stop_point") != "after_step1004_critic_commit":
            raise ValueError(f"Nsight range stop boundary drift: {path}")
        if value.get("diagnostic_only") is not True or value.get("formal_speedup_eligible") is not False:
            raise ValueError(f"profiled run incorrectly marked formal: {path}")
        values.append(value)
    expected_controllers = list(range(0, expected_world_size, expected_local_world_size))
    if controllers != expected_controllers:
        raise ValueError(
            f"Nsight controller ranks={controllers}, expected={expected_controllers}"
        )
    return {
        "schema_version": "distillschedule.gan_nsys_receipt_bundle.v1",
        "status": "PASS",
        "world_size": expected_world_size,
        "local_world_size": expected_local_world_size,
        "controller_ranks": controllers,
        "logical_window": [START_STEP, STOP_STEP],
        "profiled": True,
        "diagnostic_only": True,
        "formal_speedup_eligible": False,
        "receipts": [str(path) for path in files],
    }


__all__ = [
    "GanFull5NsightRange",
    "StaticFSDP2Full5NsightRange",
    "NsightRangeReceipt",
    "ProfilerApi",
    "SCHEMA_VERSION",
    "START_STEP",
    "STOP_STEP",
    "TorchCudaProfilerApi",
    "validate_nsys_receipts",
    "validate_static_nsys_receipts",
]
