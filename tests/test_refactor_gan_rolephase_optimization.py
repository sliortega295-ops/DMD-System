from __future__ import annotations

import json

import pytest

from distillschedule.runtime.gan_rolephase_optimization import (
    JsonlStageRecorder,
    PersistentTargetPolicy,
    TransitionSubphase,
    estimate_layouts,
    summarize_service_records,
    validate_zero_engine,
)


def test_forward_target_policy_forbids_timed_construction() -> None:
    PersistentTargetPolicy()
    with pytest.raises(ValueError, match="forward target construction"):
        PersistentTargetPolicy(forward_targets_preconstructed_before_timing=False)
    with pytest.raises(ValueError, match="stable phases"):
        PersistentTargetPolicy(repeated_frozen_streaming_in_stable_phase=True)


def test_transition_recorder_writes_append_only_receipts(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    recorder = JsonlStageRecorder(path, rank=2)
    with recorder.stage(TransitionSubphase.TARGET_PREPARE, direction="to_b"):
        pass
    with recorder.stage(TransitionSubphase.ENGINE_INITIALIZE, direction="to_b"):
        pass
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["stage"] for record in records] == [
        TransitionSubphase.TARGET_PREPARE.value,
        TransitionSubphase.ENGINE_INITIALIZE.value,
    ]
    summary = recorder.summary()
    assert summary["event_count"] == 2
    assert summary["stages"][TransitionSubphase.TARGET_PREPARE.value]["count"] == 1


def test_layout_estimates_are_contract_explicit() -> None:
    estimates = estimate_layouts(
        producer_seconds_per_sample=2.0,
        critic_seconds_per_sample=5.0,
    )
    assert {value.layout.name for value in estimates} == {
        "4G12C-B12",
        "6G10C-B30",
        "8G8C-B16",
    }
    assert sum(value.matched_to_current_b12 for value in estimates) == 1
    assert all(value.predicted_samples_per_second > 0 for value in estimates)


def test_service_summary_does_not_turn_different_batches_into_speedups() -> None:
    ranks = [
        {
            "phase_b_wall_seconds": 10.0,
            "phase_b": {
                "role": "generator",
                "samples": [
                    {
                        "produce_seconds": 2.0,
                        "send_completion_wait_seconds": 0.2,
                        "critic_commit_wait_seconds": 1.5,
                    }
                ],
            },
        },
        {
            "phase_b_wall_seconds": 10.0,
            "phase_b": {
                "role": "critic",
                "samples": [
                    {
                        "critic_compute_commit_seconds": 5.0,
                        "receive_wait_seconds": 0.1,
                    }
                ],
            },
        },
    ]
    summary = summarize_service_records(ranks)
    assert summary["producer_sample_seconds"]["median"] == 2.0
    assert summary["critic_compute_commit_seconds"]["median"] == 5.0
    assert summary["layout_estimates"]
    assert not any(
        estimate["claimable_native_speedup"]
        for estimate in summary["layout_estimates"]
    )


class _Stage3Optimizer:
    parameter_offload = object()


class _Stage3Engine:
    optimizer = _Stage3Optimizer()

    @staticmethod
    def zero_optimization_stage() -> int:
        return 3


def test_zero3_guard_does_not_touch_stage12_only_attribute() -> None:
    receipt = validate_zero_engine(_Stage3Engine(), expected_stage=3)
    assert receipt["stage3_guard_does_not_access_partition_gradients"] is True
    assert receipt["partition_gradients_checked"] is False
