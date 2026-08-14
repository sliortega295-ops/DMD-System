from __future__ import annotations

import copy

import pytest

from distillschedule.runtime.gan_fairness import (
    CRITIC_RANDOM_ROLES,
    FULL5_STEPS,
    GENERATOR_RANDOM_ROLES,
    LOGICAL_SAMPLE_SLOTS,
    GanLogicalSampleAddress,
    compare_gan_fairness_receipts,
    normalize_gan_fairness_receipt,
)


def _sha(seed: int) -> str:
    return f"{seed:064x}"[-64:]


def _phase(seed: int, roles: tuple[str, ...], exits=(1, 0, 1)) -> dict:
    return {
        "trace_operation_sha256": _sha(seed),
        "dit": {
            "stage_schedule_scope": "global_rank0_broadcast_equivalent",
            "stage_exits": list(exits),
            "stage_call_counts": [value + 1 for value in exits],
            "total_dit_forwards": sum(value + 1 for value in exits),
        },
        "consumed": {
            role: _sha(seed * 100 + index + 1)
            for index, role in enumerate(roles)
        },
    }


def _receipt(family: str) -> dict:
    entries = []
    for step in FULL5_STEPS:
        for slot in LOGICAL_SAMPLE_SLOTS:
            address = GanLogicalSampleAddress.for_slot(slot)
            physical = (
                {"rank": address.baseline_rank}
                if family == "baseline"
                else {
                    "wave": address.pipeline_wave,
                    "generator_rank": address.pipeline_generator_rank,
                    "score_rank": address.pipeline_score_rank,
                }
            )
            base = step * 1000 + slot
            entries.append(
                {
                    "logical_step": step,
                    "logical_sample_slot": slot,
                    "physical": physical,
                    "generator": (
                        _phase(base + 1, GENERATOR_RANDOM_ROLES)
                        if step == 1000
                        else None
                    ),
                    "critic": _phase(base + 2, CRITIC_RANDOM_ROLES),
                }
            )
    return {
        "schema_version": "distillschedule.gan_fairness_receipt.v1",
        "family": family,
        "status": "PASS",
        "trace_bundle_hash": "a" * 64,
        "r2_enabled": False,
        "entries": entries,
    }


def test_sample_slot_mapping_matches_world16_to_two_waves() -> None:
    assert GanLogicalSampleAddress.for_slot(0).to_dict() == {
        "logical_sample_slot": 0,
        "baseline_rank": 0,
        "pipeline_wave": 0,
        "pipeline_generator_rank": 0,
        "pipeline_score_rank": 8,
    }
    assert GanLogicalSampleAddress.for_slot(15).to_dict() == {
        "logical_sample_slot": 15,
        "baseline_rank": 15,
        "pipeline_wave": 1,
        "pipeline_generator_rank": 7,
        "pipeline_score_rank": 15,
    }


def test_identical_consumed_random_values_and_dit_work_pass() -> None:
    result = compare_gan_fairness_receipts(
        _receipt("baseline"), _receipt("pipeline")
    )
    assert result["status"] == "PASS"
    assert result["logical_window"] == [1000, 1004]
    assert result["logical_samples_per_step"] == 16
    assert len(result["fairness_contract_digest"]) == 64


def test_same_seed_is_not_enough_when_consumed_noise_changes() -> None:
    baseline = _receipt("baseline")
    pipeline = _receipt("pipeline")
    # The receipts intentionally have no seed field.  Formal equality is based
    # on the consumed operation-keyed tensor identity instead.
    pipeline["entries"][0]["generator"]["consumed"]["generator_dmd_noise"] = "b" * 64
    with pytest.raises(ValueError, match="stochastic tensor or DiT schedule"):
        compare_gan_fairness_receipts(baseline, pipeline)


def test_changed_dit_exit_or_call_count_is_rejected() -> None:
    baseline = _receipt("baseline")
    pipeline = _receipt("pipeline")
    pipeline["entries"][16]["critic"]["dit"]["stage_exits"] = [1, 1, 1]
    pipeline["entries"][16]["critic"]["dit"]["stage_call_counts"] = [2, 2, 2]
    pipeline["entries"][16]["critic"]["dit"]["total_dit_forwards"] = 6
    with pytest.raises(ValueError, match="stochastic tensor or DiT schedule"):
        compare_gan_fairness_receipts(baseline, pipeline)

    malformed = _receipt("pipeline")
    malformed["entries"][0]["critic"]["dit"]["stage_call_counts"] = [99, 1, 2]
    with pytest.raises(ValueError, match="disagrees with exits"):
        normalize_gan_fairness_receipt(malformed)


def test_wrong_pipeline_wave_mapping_is_rejected() -> None:
    pipeline = _receipt("pipeline")
    pipeline["entries"][8]["physical"]["wave"] = 0
    with pytest.raises(ValueError, match="pipeline sample mapping"):
        normalize_gan_fairness_receipt(pipeline)


def test_missing_active_gan_random_role_is_rejected() -> None:
    receipt = _receipt("baseline")
    del receipt["entries"][0]["critic"]["consumed"]["r1_epsilon"]
    with pytest.raises(ValueError, match="role inventory drift"):
        normalize_gan_fairness_receipt(receipt)


def test_rank_local_dit_exception_is_out_of_scope() -> None:
    receipt = _receipt("baseline")
    receipt["entries"][0]["critic"]["dit"]["stage_schedule_scope"] = (
        "rank_local_amplified_gt_zero_history_exception"
    )
    with pytest.raises(ValueError, match="forbids rank-local"):
        normalize_gan_fairness_receipt(receipt)


def test_r2_enabled_is_a_different_workload_contract() -> None:
    receipt = _receipt("baseline")
    receipt["r2_enabled"] = True
    with pytest.raises(ValueError, match="r2_enabled=false"):
        normalize_gan_fairness_receipt(receipt)
