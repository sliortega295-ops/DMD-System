"""Diagnostic, synchronized native Wan macro-stage envelopes; no torch import.

Construct after model/optimizer setup, then wrap only the fourteen operations
from ``native_graph``. This observer does not execute or inspect model tensors.
Barriers and memory queries are outside each duration; the trailing CUDA sync
is inside it. These perturbed macro-envelope durations are calibration only,
never a formal cycle timer or a sum-equivalent replacement for one.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from time import perf_counter_ns
from typing import Any, Callable, Iterator

from distillschedule.workloads.best_known.registry import fingerprint
from .contracts import require
from .plan_ir import nonnegative
from .wan_implementation import PRECISION, WAN, WanImplementation
from .wan_model_adapter import ADAPTER, RANKS, native_graph, operation_contract


class StageCalibrator:
    """Observe an injected backend without importing torch or transferring tensors.

    ``clock`` returns monotonic integer nanoseconds. Fake backends must explicitly
    select ``evidence_scope="CPU_FIXTURE_ONLY"``. Setup and each stage entry have
    a distributed barrier followed by device synchronization. Stage exit has no
    barrier, including on failure, so collecting evidence never adds a failure-
    path collective. Backend/model failures propagate, preserving earlier rows
    and available observations in ``snapshot()``; a failed collector cannot resume.

    Receipt fields used by the later controller/assembler:

    * ``post_setup_static_bytes`` and ``post_setup_memory`` record the allocated
      baseline, reserved bytes and driver free/total capacity after setup.
    * ``stages`` contains ordered operation IDs, ``key`` (the cost key), contract
      payload/hash, ``duration_ns``, ``memory_before`` and ``memory_after``.
      Memory dictionaries use ``allocated_bytes``, ``reserved_bytes``,
      ``driver_free_bytes`` and ``driver_total_bytes``; the after dictionary also
      has ``peak_allocated_bytes`` and ``peak_reserved_bytes``.
    * ``workspace_bytes_per_rank`` in each row is max(0, stage peak allocated -
      post-setup static), NOT the delta from entry. Retained outputs count here.
    * Top-level ``peak_allocated_bytes``, ``peak_reserved_bytes`` and
      ``minimum_observed_driver_free_bytes`` retain extrema across peak resets.
      Driver free is sampled, not a claim of the minimum inside a stage.

    ``finish()`` returns a detached PASS rank receipt only after all fourteen
    stages succeeded. PASS means observation completion, not numerical validation.
    Source HEAD and declared precision are recorded, but environment/assets/
    hardware binding and rank aggregation belong to the controller, not here.
    """

    SCHEMA = "distillschedule.wan_stage_rank_calibration.v1"
    ADAPTER = ADAPTER
    IMPLEMENTATION_TYPE = WanImplementation
    WORKLOAD = WAN
    PRECISION_FINGERPRINT = PRECISION.digest

    def operation_graph(self):
        return native_graph()

    def contract(self, operation):
        return operation_contract(operation, self._implementation)

    def __init__(self, *, torch: Any, distributed: Any, device: Any, rank: int,
                 implementation: WanImplementation, source_head: str,
                 clock: Callable[[], int] | None = None,
                 evidence_scope: str = "MEASURED") -> None:
        require(type(rank) is int and rank in RANKS, "native Wan rank must be in world16")
        require(isinstance(implementation, self.IMPLEMENTATION_TYPE),
                f"{self.IMPLEMENTATION_TYPE.__name__} required")
        require(isinstance(source_head, str) and len(source_head) == 40
                and all(c in "0123456789abcdef" for c in source_head), "invalid source_head")
        require(evidence_scope in {"MEASURED", "CPU_FIXTURE_ONLY"}, "unknown calibration evidence scope")
        self._clock = perf_counter_ns if clock is None else clock
        require(callable(self._clock), "clock must return monotonic integer nanoseconds")
        self._cuda, self._distributed, self._device = torch.cuda, distributed, device
        self._rank, self._implementation, self._source_head = rank, implementation, source_head
        self._labels = {"evidence_scope": evidence_scope, "calibration_only": True,
                        "formal_timing_eligible": False, "profiled": False, "kind": "native_operation"}
        self._graph = self.operation_graph()
        self._stages: list[dict[str, Any]] = []
        self._errors: list[dict[str, str]] = []
        self._active: dict[str, Any] | None = None
        self._completed = 0
        self._finished = False
        self._peak_allocated = self._peak_reserved = 0
        self._minimum_free: int | None = None
        self._post_setup: dict[str, int] = {}
        self._distributed.barrier()
        self._cuda.synchronize(self._device)
        self._observe_memory(self._post_setup)
        self._static_bytes = self._post_setup["allocated_bytes"]
        require(self._static_bytes > 0, "post-setup native model/state allocation must be positive")

    def _fail(self, row: dict[str, Any] | None, phase: str, error: BaseException) -> None:
        evidence = {"phase": phase, "error_type": type(error).__name__, "error": str(error)}
        if row is not None:
            evidence["operation_id"] = row["operation_id"]
            row["status"] = "FAIL"
            row["errors"].append(evidence)
        self._errors.append(evidence)

    def _reject(self, message: str) -> None:
        error = ValueError(message)
        self._fail(self._active, "protocol", error)
        raise error

    def _remember_memory(self, memory: dict[str, int]) -> None:
        self._peak_allocated = max(self._peak_allocated, memory.get("allocated_bytes", 0),
                                   memory.get("peak_allocated_bytes", 0))
        self._peak_reserved = max(self._peak_reserved, memory.get("reserved_bytes", 0),
                                  memory.get("peak_reserved_bytes", 0))
        if "driver_free_bytes" in memory:
            free = memory["driver_free_bytes"]
            self._minimum_free = free if self._minimum_free is None else min(self._minimum_free, free)

    def _observe_memory(self, target: dict[str, int], *,
                        row: dict[str, Any] | None = None) -> BaseException | None:
        """On exit collect independent fields even if another observation fails."""
        first_error = None
        fields = [("allocated_bytes", "memory_allocated"), ("reserved_bytes", "memory_reserved")]
        if row is not None:
            fields += [("peak_allocated_bytes", "max_memory_allocated"),
                       ("peak_reserved_bytes", "max_memory_reserved")]
        for key, api in [*fields, ("driver", "mem_get_info")]:
            try:
                value = getattr(self._cuda, api)(self._device)
                if key == "driver":
                    free, total = value
                    target["driver_free_bytes"] = nonnegative(free, "driver free bytes")
                    target["driver_total_bytes"] = nonnegative(total, "driver total bytes")
                else:
                    # CUDA memory APIs return Python ints; never coerce tensors.
                    target[key] = nonnegative(value, key)
            except BaseException as error:
                if row is None:
                    raise
                self._fail(row, f"memory_after.{api}", error)
                if first_error is None:
                    first_error = error
            finally:
                self._remember_memory(target)
        return first_error

    def _observe_after(self, row: dict[str, Any]) -> BaseException | None:
        first_error = None
        try:
            self._cuda.synchronize(self._device)
        except BaseException as error:
            self._fail(row, "after_synchronize", error)
            first_error = error
        else:
            try:
                row["stop_ns"] = nonnegative(self._clock(), "clock nanoseconds")
                row["duration_ns"] = nonnegative(row["stop_ns"] - row["start_ns"], "duration_ns")
            except BaseException as error:
                self._fail(row, "after_clock", error)
                first_error = error
        memory_error = self._observe_memory(row["memory_after"], row=row)
        for key in ("peak_allocated_bytes", "peak_reserved_bytes"):
            if key in row["memory_after"]:
                row["memory_after"][key] = max(row["memory_after"][key], row["inner_reset_peaks"][key])
        self._remember_memory(row["memory_after"])
        if "peak_allocated_bytes" in row["memory_after"]:
            row["workspace_bytes_per_rank"] = max(
                0, row["memory_after"]["peak_allocated_bytes"] - self._static_bytes)
        return first_error if first_error is not None else memory_error

    @contextmanager
    def _preserve_inner_peaks(self, row: dict[str, Any]) -> Iterator[None]:
        """Observe pre-reset peaks without suppressing existing runtime resets.

        FSDP's native rollout helpers reset statistics several times within the
        five-rollout macro-stage. This process-local wrapper exists only during
        explicit calibration, calls the original reset, adds no CUDA sync/copy,
        and is restored even when model execution fails. No site-package edits.
        """
        original = self._cuda.reset_peak_memory_stats

        def reset(*args: Any, **kwargs: Any) -> Any:
            for key, api in (("peak_allocated_bytes", "max_memory_allocated"),
                             ("peak_reserved_bytes", "max_memory_reserved")):
                peak = nonnegative(getattr(self._cuda, api)(self._device), key)
                row["inner_reset_peaks"][key] = max(row["inner_reset_peaks"][key], peak)
            self._remember_memory(row["inner_reset_peaks"])
            row["nested_peak_reset_count"] += 1
            return original(*args, **kwargs)

        self._cuda.reset_peak_memory_stats = reset
        try:
            yield
        finally:
            self._cuda.reset_peak_memory_stats = original

    @contextmanager
    def stage(self, cycle: int, stage_name: str) -> Iterator[None]:
        """Wrap one exact native stage; any invalid order permanently fails closed."""
        if self._active is not None:
            self._reject("nested calibration stages are forbidden")
        require(not self._errors, "calibration has failed; preserve snapshot and do not retry")
        if type(cycle) is not int or not isinstance(stage_name, str):
            self._reject("stage requires an integer cycle and exact native stage name")
        operation_id = f"cycle{cycle}.{stage_name}"
        if any(row["operation_id"] == operation_id for row in self._stages):
            self._reject(f"duplicate calibration stage: {operation_id}")
        if self._finished or self._completed == len(self._graph.operations):
            self._reject("extra calibration stage after all native operations")
        operation = self._graph.operations[self._completed]
        if operation_id != operation.operation_id:
            self._reject(f"reordered/unknown stage: expected {operation.operation_id}, got {operation_id}")
        contract = self.contract(operation)
        row = {**self._labels, "operation_id": operation_id, "cycle": cycle, "stage": stage_name,
               "key": fingerprint(contract), "operation_contract": contract,
               "operation_contract_sha256": fingerprint(contract),
               "memory_definition": contract.get("memory_definition", getattr(self, "MEMORY_DEFINITION", None)),
               "timing_definition": contract.get("timing_definition", getattr(self, "TIMING_DEFINITION", None)),
               "status": "INCOMPLETE", "start_ns": None, "stop_ns": None, "duration_ns": None,
               "workspace_bytes_per_rank": None, "memory_before": {}, "memory_after": {}, "errors": [],
               "nested_peak_reset_count": 0,
               "inner_reset_peaks": {"peak_allocated_bytes": 0, "peak_reserved_bytes": 0}}
        self._stages.append(row)
        self._active = row
        try:
            try:
                self._distributed.barrier()
                self._cuda.synchronize(self._device)
                self._observe_memory(row["memory_before"])
                self._cuda.reset_peak_memory_stats(self._device)
                row["start_ns"] = nonnegative(self._clock(), "clock nanoseconds")
            except BaseException as error:
                self._fail(row, "before", error)
                raise
            try:
                with self._preserve_inner_peaks(row):
                    yield
            except BaseException as error:
                self._fail(row, "body", error)
                self._observe_after(row)  # Never mask the original body exception.
                raise
            error = self._observe_after(row)
            if error is not None:
                raise error
            # A body that swallowed a nested-stage/protocol error still failed.
            require(not self._errors, "calibration failed during the active stage")
            row["status"] = "PASS"
            self._completed += 1
        finally:
            self._active = None

    def snapshot(self) -> dict[str, Any]:
        """Detached, JSON-ready evidence; does not query the backend or tensors."""
        completed = [row["operation_id"] for row in self._stages if row["status"] == "PASS"]
        return deepcopy({
            "schema_version": self.SCHEMA, **self._labels,
            "status": "FAIL" if self._errors else ("PASS" if self._finished else "INCOMPLETE"),
            "adapter": self.ADAPTER, "workload": self.WORKLOAD, "rank": self._rank, "device": str(self._device),
            "source_head": self._source_head, "implementation_id": self._implementation.id,
            "implementation": self._implementation.assignment, "precision_fingerprint": self.PRECISION_FINGERPRINT,
            "native_graph_sha256": fingerprint(self._graph.to_dict()),
            "post_setup_static_bytes": self._static_bytes, "post_setup_memory": self._post_setup,
            "peak_allocated_bytes": self._peak_allocated, "peak_reserved_bytes": self._peak_reserved,
            "minimum_observed_driver_free_bytes": self._minimum_free,
            "completed_operation_ids": completed,
            "missing_operation_ids": [op.operation_id for op in self._graph.operations
                                      if op.operation_id not in completed],
            "stages": self._stages, "errors": self._errors,
        })

    def finish(self) -> dict[str, Any]:
        """Seal only a complete successful rank; repeated finish is read-only."""
        if self._active is not None:
            self._reject("cannot finish with an active calibration stage")
        require(not self._errors, "calibration has failed; only partial snapshot is available")
        if self._completed != len(self._graph.operations):
            missing = [op.operation_id for op in self._graph.operations[self._completed:]]
            self._reject(f"missing successful calibration stages: {', '.join(missing)}")
        self._finished = True
        return self.snapshot()
