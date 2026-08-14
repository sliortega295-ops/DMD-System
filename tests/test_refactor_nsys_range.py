from __future__ import annotations

import json

import pytest

from distillschedule.runtime.nsys_range import (
    GanFull5NsightRange,
    StaticFSDP2Full5NsightRange,
    validate_nsys_receipts,
    validate_static_nsys_receipts,
)


class FakeProfiler:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def start(self) -> None:
        self.calls.append("start")

    def stop(self) -> None:
        self.calls.append("stop")


def test_exact_range_starts_at_1000_and_stops_after_1004_commit(tmp_path) -> None:
    api = FakeProfiler()
    receipt = tmp_path / "rank_000.json"
    gate = GanFull5NsightRange(
        api=api, receipt_path=receipt, rank=0, local_rank=0
    )
    gate.before_model_compute(999)
    assert api.calls == []
    gate.before_model_compute(1000)
    assert api.calls == ["start"]
    assert gate.active is True
    assert gate.after_critic_commit(1003) is None
    result = gate.after_critic_commit(1004)
    assert api.calls == ["start", "stop"]
    assert result is not None
    assert result.start_point == "before_first_model_compute"
    assert result.stop_point == "after_step1004_critic_commit"
    assert result.controls_profiler_api is True
    gate.assert_complete()
    payload = json.loads(receipt.read_text())
    assert payload["logical_window"] == [1000, 1004]
    assert payload["diagnostic_only"] is True
    assert payload["formal_speedup_eligible"] is False


def test_non_local_main_rank_records_boundary_without_stopping_nsys(tmp_path) -> None:
    api = FakeProfiler()
    gate = GanFull5NsightRange(
        api=api,
        receipt_path=tmp_path / "rank_001.json",
        rank=1,
        local_rank=1,
    )
    gate.before_model_compute(1000)
    gate.after_critic_commit(1004)
    assert api.calls == []
    payload = json.loads((tmp_path / "rank_001.json").read_text())
    assert payload["controls_profiler_api"] is False


def test_range_fails_closed_on_bad_order() -> None:
    gate = GanFull5NsightRange(api=FakeProfiler(), rank=0, local_rank=0)
    with pytest.raises(RuntimeError, match="before it started"):
        gate.after_critic_commit(1004)

    gate = GanFull5NsightRange(api=FakeProfiler(), rank=0, local_rank=0)
    gate.before_model_compute(1000)
    with pytest.raises(RuntimeError, match="started more than once"):
        gate.before_model_compute(1000)


def test_disabled_range_is_noop() -> None:
    api = FakeProfiler()
    gate = GanFull5NsightRange(
        api=api, rank=0, local_rank=0, enabled=False
    )
    gate.before_model_compute(1000)
    assert gate.after_critic_commit(1004) is None
    assert api.calls == []
    gate.assert_complete()


def test_receipt_bundle_requires_one_controller_per_node(tmp_path) -> None:
    for rank in range(4):
        local_rank = rank % 2
        api = FakeProfiler()
        gate = GanFull5NsightRange(
            api=api,
            receipt_path=tmp_path / f"rank_{rank:03d}.json",
            rank=rank,
            local_rank=local_rank,
        )
        gate.before_model_compute(1000)
        gate.after_critic_commit(1004)
        expected = ["start", "stop"] if local_rank == 0 else []
        assert api.calls == expected
    result = validate_nsys_receipts(
        tmp_path, expected_world_size=4, expected_local_world_size=2
    )
    assert result["status"] == "PASS"
    assert result["controller_ranks"] == [0, 2]
    assert result["diagnostic_only"] is True


def test_receipt_bundle_rejects_missing_rank(tmp_path) -> None:
    gate = GanFull5NsightRange(
        api=FakeProfiler(),
        receipt_path=tmp_path / "rank_000.json",
        rank=0,
        local_rank=0,
    )
    gate.before_model_compute(1000)
    gate.after_critic_commit(1004)
    with pytest.raises(ValueError, match="receipt count"):
        validate_nsys_receipts(
            tmp_path, expected_world_size=2, expected_local_world_size=1
        )


def test_static_no_gan_range_uses_steps_10_to_14(tmp_path) -> None:
    api = FakeProfiler()
    receipt = tmp_path / "rank_000.json"
    gate = StaticFSDP2Full5NsightRange(
        variant="no-gan", api=api, receipt_path=receipt, rank=0, local_rank=0
    )
    gate.before_model_compute(9)
    gate.before_model_compute(10)
    gate.after_critic_commit(13)
    result = gate.after_critic_commit(14)
    assert result is not None
    assert api.calls == ["start", "stop"]
    assert result.start_step == 10
    assert result.stop_step == 14
    assert result.schema_version == "distillschedule.static_fsdp2_nsys_range.v1"
    assert json.loads(receipt.read_text())["logical_window"] == [10, 14]


def test_static_receipt_bundle_requires_world16_and_two_controllers(tmp_path) -> None:
    for rank in range(16):
        local_rank = rank % 8
        gate = StaticFSDP2Full5NsightRange(
            api=FakeProfiler(),
            variant="gan",
            receipt_path=tmp_path / f"rank_{rank:03d}.json",
            rank=rank,
            local_rank=local_rank,
        )
        gate.before_model_compute(1000)
        gate.after_critic_commit(1004)
    result = validate_static_nsys_receipts(tmp_path, variant="gan")
    assert result["status"] == "PASS"
    assert result["controller_ranks"] == [0, 8]
