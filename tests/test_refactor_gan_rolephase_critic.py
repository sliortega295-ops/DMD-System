from __future__ import annotations

import pytest

from distillschedule.runtime.gan_rolephase import GanRolePhaseTopology
from distillschedule.runtime.gan_rolephase_critic import (
    CriticReductionContract,
    assert_collective_safe_call_signature,
    build_critic_local_batches,
    expected_slot_order_for_rank,
    validate_per_sample_weights,
)


def test_c12_packs_flex_b2_and_fixed_b1_with_one_call_each() -> None:
    batches = build_critic_local_batches()
    assert [item.global_rank for item in batches] == list(range(4, 16))
    assert [item.local_batch for item in batches] == [2, 2, 2, 2] + [1] * 8
    covered = sorted(slot for item in batches for slot in item.logical_slots)
    assert covered == list(range(16))
    assert_collective_safe_call_signature({rank: 1 for rank in range(4, 16)})
    with pytest.raises(AssertionError, match="exactly one physical model call"):
        assert_collective_safe_call_signature(
            {rank: (2 if rank == 4 else 1) for rank in range(4, 16)}
        )


def test_c12_canonical_slot_order_matches_topology() -> None:
    topology = GanRolePhaseTopology()
    assert expected_slot_order_for_rank(4, topology) == (0, 12)
    assert expected_slot_order_for_rank(5, topology) == (1, 13)
    assert expected_slot_order_for_rank(8, topology) == (4,)
    assert expected_slot_order_for_rank(15, topology) == (11,)


def test_explicit_gradient_reduction_scaling_yields_global_sample_mean() -> None:
    contract = CriticReductionContract()
    assert contract.local_sum_scale("SUM") == pytest.approx(1.0 / 16.0)
    assert contract.local_sum_scale("DP_MEAN") == pytest.approx(12.0 / 16.0)
    with pytest.raises(ValueError, match="explicitly bound"):
        contract.local_sum_scale("AUTO")


def test_logical_sample_weights_remain_exact_batch16_mean() -> None:
    validate_per_sample_weights([1.0 / 16.0] * 16)
    with pytest.raises(AssertionError):
        validate_per_sample_weights([1.0 / 12.0] * 16)
