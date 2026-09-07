"""CPU_FIXTURE_ONLY observations; no CUDA execution or GPU performance claims."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
import sys

import pytest

from distillschedule.search.wan_implementation import PRECISION, WanImplementation
from distillschedule.search.wan_model_adapter import cost_key, native_graph, operation_contract
from distillschedule.search.wan_stage_calibration import StageCalibrator
from distillschedule.workloads.best_known.registry import fingerprint


SCOPE = "CPU_FIXTURE_ONLY"
SOURCE = "a" * 40
DEVICE = "cpu-fixture-cuda-device"
IMPLEMENTATION = WanImplementation("resident", 1024, True)


class FakeBackend:
    """Only host-side CUDA memory observations and synchronization are exposed."""

    def __init__(self):
        self.cuda = self
        self.events = []
        self.now = 0
        self.allocated = 100
        self.reserved = 180
        self.peak_allocated = 9000  # Setup peaks must not become stage peaks.
        self.peak_reserved = 10000
        self.free = 8000
        self.total = 8192
        self.failures = {}

    def __getattr__(self, name):
        raise AssertionError(f"{SCOPE}: forbidden non-observation API: {name}")

    def _event(self, name, device=None):
        if device is not None:
            assert device == DEVICE
        self.events.append(name)
        self.now += {"barrier": 500, "synchronize": 7, "clock": 0}.get(name, 30)
        if name in self.failures:
            raise self.failures.pop(name)

    def barrier(self):
        self._event("barrier")

    def synchronize(self, device):
        self._event("synchronize", device)

    def memory_allocated(self, device):
        self._event("memory_allocated", device)
        return self.allocated

    def memory_reserved(self, device):
        self._event("memory_reserved", device)
        return self.reserved

    def max_memory_allocated(self, device):
        self._event("max_memory_allocated", device)
        return self.peak_allocated

    def max_memory_reserved(self, device):
        self._event("max_memory_reserved", device)
        return self.peak_reserved

    def reset_peak_memory_stats(self, device):
        self._event("reset_peak_memory_stats", device)
        self.peak_allocated = self.allocated
        self.peak_reserved = self.reserved

    def mem_get_info(self, device):
        self._event("mem_get_info", device)
        return self.free, self.total

    def clock(self):
        self._event("clock")
        return self.now

    def body(self, *, allocated=120, reserved=220, peak_allocated=175,
             peak_reserved=300, driver_free=7800, duration_ns=400):
        self.events.append("body")
        self.now += duration_ns
        self.allocated = allocated
        self.reserved = reserved
        self.peak_allocated = max(self.peak_allocated, peak_allocated, allocated)
        self.peak_reserved = max(self.peak_reserved, peak_reserved, reserved)
        self.free = driver_free


def fixture(**overrides):
    backend = FakeBackend()
    collector = StageCalibrator(**{
        "torch": backend, "distributed": backend, "device": DEVICE, "rank": 3,
        "implementation": IMPLEMENTATION, "source_head": SOURCE,
        "clock": backend.clock, "evidence_scope": SCOPE, **overrides,
    })
    return collector, backend


def run_stage(collector, backend, operation, **body):
    with collector.stage(operation.metadata["cycle"], operation.metadata["stage"]):
        backend.body(**body)


def test_setup_and_stage_observation_order_excludes_barrier_and_memory_overhead():
    collector, backend = fixture()
    assert backend.events == ["barrier", "synchronize", "memory_allocated",
                              "memory_reserved", "mem_get_info"]
    initial = collector.snapshot()
    assert initial["status"] == "INCOMPLETE"
    assert initial["post_setup_static_bytes"] == 100
    assert initial["post_setup_memory"] == {
        "allocated_bytes": 100, "reserved_bytes": 180,
        "driver_free_bytes": 8000, "driver_total_bytes": 8192,
    }
    backend.events.clear()
    run_stage(collector, backend, native_graph().operations[0])
    assert backend.events == [
        "barrier", "synchronize", "memory_allocated", "memory_reserved", "mem_get_info",
        "reset_peak_memory_stats", "clock", "body", "synchronize", "clock",
        "memory_allocated", "memory_reserved", "max_memory_allocated",
        "max_memory_reserved", "mem_get_info",
    ]
    row = collector.snapshot()["stages"][0]
    assert row["status"] == "PASS"
    assert row["duration_ns"] == 407  # Body + completion sync, no observation/barrier time.
    assert row["memory_before"]["allocated_bytes"] == 100
    assert row["memory_after"] == {
        "allocated_bytes": 120, "reserved_bytes": 220,
        "peak_allocated_bytes": 175, "peak_reserved_bytes": 300,
        "driver_free_bytes": 7800, "driver_total_bytes": 8192,
    }
    assert row["workspace_bytes_per_rank"] == 75


def test_exact_fourteen_native_stages_contracts_and_detached_rank_receipt():
    collector, backend = fixture()
    graph = native_graph()
    assert [op.operation_id for op in graph.operations] == [
        f"cycle{cycle}.{stage}" for cycle in (1, 2)
        for stage in ("generator_update", "critic_rollouts", *[
            f"critic{i}_update_ema" for i in range(5)])
    ]
    for op in graph.operations:
        run_stage(collector, backend, op)
    assert collector.snapshot()["status"] == "INCOMPLETE"  # Explicit finish gate.
    before_finish = list(backend.events)
    receipt = collector.finish()
    assert backend.events == before_finish
    assert receipt["schema_version"] == "distillschedule.wan_stage_rank_calibration.v1"
    assert receipt["status"] == "PASS" and len(receipt["stages"]) == 14
    assert receipt["rank"] == 3 and receipt["source_head"] == SOURCE
    assert receipt["implementation_id"] == IMPLEMENTATION.id
    assert receipt["implementation"] == IMPLEMENTATION.assignment
    assert receipt["precision_fingerprint"] == PRECISION.digest
    assert receipt["native_graph_sha256"] == fingerprint(graph.to_dict())
    assert receipt["missing_operation_ids"] == []
    assert receipt["completed_operation_ids"] == [op.operation_id for op in graph.operations]
    assert "binding" not in receipt  # Controller must bind environment/assets/hardware later.
    for row, op in zip(receipt["stages"], graph.operations):
        contract = operation_contract(op, IMPLEMENTATION)
        assert row["operation_id"] == op.operation_id
        assert row["key"] == cost_key(op, IMPLEMENTATION)
        assert row["operation_contract"] == contract
        assert row["operation_contract_sha256"] == fingerprint(contract)
        assert row["memory_definition"] == contract["memory_definition"]
        assert row["timing_definition"] == contract["timing_definition"]
        assert row["duration_ns"] == 407 and row["status"] == "PASS"
    for record in (receipt, *receipt["stages"]):
        assert record["evidence_scope"] == SCOPE
        assert record["calibration_only"] is True
        assert record["formal_timing_eligible"] is False
        assert record["profiled"] is False
        assert record["kind"] == "native_operation"
    assert backend.events.count("reset_peak_memory_stats") == 14
    assert backend.events.count("body") == 14
    assert backend.events.count("barrier") == 15
    assert backend.events.count("synchronize") == 29
    assert json.loads(json.dumps(receipt)) == receipt
    receipt["stages"][0]["memory_before"]["allocated_bytes"] = -1
    assert collector.finish()["stages"][0]["memory_before"]["allocated_bytes"] == 100


def test_reset_peaks_aggregate_across_stages_and_include_retained_outputs():
    collector, backend = fixture()
    ops = native_graph().operations
    run_stage(collector, backend, ops[0], allocated=150, reserved=250,
              peak_allocated=1000, peak_reserved=2000, driver_free=6000)
    run_stage(collector, backend, ops[1], allocated=140, reserved=240,
              peak_allocated=800, peak_reserved=900, driver_free=6200)
    for op in ops[2:]:
        run_stage(collector, backend, op, peak_allocated=200, peak_reserved=300)
    receipt = collector.finish()
    assert receipt["peak_allocated_bytes"] == 1000
    assert receipt["peak_reserved_bytes"] == 2000
    assert receipt["minimum_observed_driver_free_bytes"] == 6000
    assert receipt["stages"][-1]["memory_after"]["peak_allocated_bytes"] == 200
    assert receipt["stages"][1]["memory_before"]["allocated_bytes"] == 150
    assert receipt["stages"][1]["workspace_bytes_per_rank"] == 700  # Not 800 - 150.


def test_body_failure_keeps_prior_pass_and_partial_memory_but_cannot_finish_or_retry():
    collector, backend = fixture()
    ops = native_graph().operations
    run_stage(collector, backend, ops[0])
    original = RuntimeError(f"{SCOPE} body failure")
    with pytest.raises(RuntimeError) as caught:
        with collector.stage(1, "critic_rollouts"):
            backend.body(peak_allocated=400)
            raise original
    assert caught.value is original
    receipt = collector.snapshot()
    assert receipt["status"] == "FAIL"
    assert [row["status"] for row in receipt["stages"]] == ["PASS", "FAIL"]
    assert receipt["stages"][1]["duration_ns"] == 407
    assert receipt["stages"][1]["memory_after"]["peak_allocated_bytes"] == 400
    assert receipt["stages"][1]["errors"][0]["phase"] == "body"
    assert receipt["completed_operation_ids"] == [ops[0].operation_id]
    assert ops[1].operation_id in receipt["missing_operation_ids"]
    events = list(backend.events)
    with pytest.raises(ValueError, match="failed"):
        run_stage(collector, backend, ops[1])
    with pytest.raises(ValueError, match="failed"):
        collector.finish()
    assert backend.events == events


@pytest.mark.parametrize("phase", ["barrier", "reset_peak_memory_stats"])
def test_pre_stage_observation_failure_never_runs_body(phase):
    collector, backend = fixture()
    backend.failures[phase] = RuntimeError(f"{SCOPE} {phase} failed")
    with pytest.raises(RuntimeError, match=phase):
        run_stage(collector, backend, native_graph().operations[0])
    receipt = collector.snapshot()
    assert receipt["status"] == "FAIL" and receipt["post_setup_static_bytes"] == 100
    row = receipt["stages"][0]
    assert row["status"] == "FAIL" and row["duration_ns"] is None
    assert "body" not in backend.events
    if phase == "reset_peak_memory_stats":
        assert row["memory_before"]["allocated_bytes"] == 100


@pytest.mark.parametrize("failing_api", ["synchronize", "max_memory_reserved", "mem_get_info"])
def test_post_stage_observation_failure_preserves_available_evidence(failing_api):
    collector, backend = fixture()
    original = RuntimeError(f"{SCOPE} {failing_api} failed")
    with pytest.raises(RuntimeError) as caught:
        with collector.stage(1, "generator_update"):
            backend.body(peak_allocated=450)
            backend.failures[failing_api] = original
    assert caught.value is original
    receipt = collector.snapshot()
    assert receipt["status"] == "FAIL"
    row = receipt["stages"][0]
    assert row["status"] == "FAIL" and receipt["completed_operation_ids"] == []
    assert row["memory_after"]["peak_allocated_bytes"] == 450
    assert receipt["peak_allocated_bytes"] == 450
    assert (row["duration_ns"] is None) == (failing_api == "synchronize")


def test_body_exception_is_not_masked_by_failed_completion_sync():
    collector, backend = fixture()
    original = FloatingPointError(f"{SCOPE} primary body error")
    with pytest.raises(FloatingPointError) as caught:
        with collector.stage(1, "generator_update"):
            backend.body(peak_allocated=500)
            backend.failures["synchronize"] = RuntimeError(f"{SCOPE} secondary sync error")
            raise original
    assert caught.value is original
    row = collector.snapshot()["stages"][0]
    assert row["status"] == "FAIL" and row["duration_ns"] is None
    assert [error["error_type"] for error in row["errors"]] == ["FloatingPointError", "RuntimeError"]
    assert row["memory_after"]["peak_allocated_bytes"] == 500


@pytest.mark.parametrize("cycle, stage", [(1, "critic_rollouts"), (2, "generator_update"),
    (1, "generator_update_typo"), (3, "generator_update"), (True, "generator_update"),
    (1.0, "generator_update")])
def test_unknown_or_reordered_stage_fails_closed_without_backend_calls(cycle, stage):
    collector, backend = fixture()
    events = list(backend.events)
    with pytest.raises(ValueError):
        with collector.stage(cycle, stage):
            pytest.fail("invalid stage must not run")
    assert backend.events == events
    assert collector.snapshot()["status"] == "FAIL"
    with pytest.raises(ValueError):
        collector.finish()


def test_duplicate_stage_is_rejected_without_reexecution():
    collector, backend = fixture()
    op = native_graph().operations[0]
    run_stage(collector, backend, op)
    events = list(backend.events)
    with pytest.raises(ValueError, match="duplicate"):
        run_stage(collector, backend, op)
    assert backend.events == events
    assert len(collector.snapshot()["stages"]) == 1
    assert collector.snapshot()["status"] == "FAIL"


@pytest.mark.parametrize("swallow_inner", [False, True])
def test_nested_attempt_invalidates_outer_even_if_body_swallows_error(swallow_inner):
    collector, backend = fixture()
    with pytest.raises(ValueError):
        with collector.stage(1, "generator_update"):
            backend.body()
            try:
                with collector.stage(1, "critic_rollouts"):
                    pytest.fail("nested stage must not run")
            except ValueError:
                if not swallow_inner:
                    raise
    receipt = collector.snapshot()
    assert receipt["status"] == "FAIL"
    assert receipt["stages"][0]["status"] == "FAIL"
    assert receipt["completed_operation_ids"] == []
    assert backend.events.count("reset_peak_memory_stats") == 1


@pytest.mark.parametrize("completed", [0, 1, 13])
def test_finish_rejects_missing_stages_without_backend_calls(completed):
    collector, backend = fixture()
    for op in native_graph().operations[:completed]:
        run_stage(collector, backend, op)
    events = list(backend.events)
    with pytest.raises(ValueError, match="missing"):
        collector.finish()
    assert backend.events == events
    assert len(collector.snapshot()["missing_operation_ids"]) == 14 - completed
    assert collector.snapshot()["status"] == "FAIL"


def test_finish_inside_active_stage_cannot_mark_stage_pass():
    collector, backend = fixture()
    with pytest.raises(ValueError, match="active"):
        with collector.stage(1, "generator_update"):
            backend.body()
            collector.finish()
    assert collector.snapshot()["stages"][0]["status"] == "FAIL"


def test_clock_must_supply_monotonic_integer_nanoseconds():
    times = iter((100, 99))
    collector, backend = fixture(clock=lambda: next(times))
    with pytest.raises(ValueError, match="duration"):
        run_stage(collector, backend, native_graph().operations[0])
    assert collector.snapshot()["stages"][0]["status"] == "FAIL"
    assert collector.snapshot()["stages"][0]["duration_ns"] is None


def test_import_is_torch_free_and_collector_has_no_tensor_transfer_or_profile_api():
    result = subprocess.run([sys.executable, "-c", """
import builtins
import sys
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise AssertionError('CPU_FIXTURE_ONLY: torch import forbidden')
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
from distillschedule.search.wan_stage_calibration import StageCalibrator
assert not any(name == 'torch' or name.startswith('torch.') for name in sys.modules)
print('CPU_FIXTURE_ONLY IMPORT PASS')
"""], text=True, capture_output=True, check=True)
    assert result.stdout.strip() == "CPU_FIXTURE_ONLY IMPORT PASS"
    source = Path(__file__).resolve().parents[1] / "src/distillschedule/search/wan_stage_calibration.py"
    tree = ast.parse(source.read_text())
    forbidden = {"cpu", "numpy", "item", "tolist", "to", "tensor", "as_tensor", "clone",
                 "detach", "profiler", "all_reduce", "empty_cache"}
    assert not {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)} & forbidden
