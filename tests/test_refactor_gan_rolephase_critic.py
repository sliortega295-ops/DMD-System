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


def test_c12_is_uniform_b1_batch12_with_one_call_each() -> None:
    batches = build_critic_local_batches()
    assert [item.global_rank for item in batches] == list(range(4, 16))
    assert [item.local_batch for item in batches] == [1] * 12
    covered = sorted(slot for item in batches for slot in item.logical_slots)
    assert covered == list(range(12))
    assert_collective_safe_call_signature({rank: 1 for rank in range(4, 16)})
    with pytest.raises(AssertionError, match="exactly one physical model call"):
        assert_collective_safe_call_signature(
            {rank: (2 if rank == 4 else 1) for rank in range(4, 16)}
        )


def test_c12_canonical_slot_order_is_one_to_one() -> None:
    topology = GanRolePhaseTopology()
    assert expected_slot_order_for_rank(4, topology) == (0,)
    assert expected_slot_order_for_rank(5, topology) == (1,)
    assert expected_slot_order_for_rank(8, topology) == (4,)
    assert expected_slot_order_for_rank(15, topology) == (11,)


def test_explicit_gradient_reduction_scaling_yields_batch12_mean() -> None:
    contract = CriticReductionContract()
    assert contract.local_loss_scale("SUM") == pytest.approx(1.0 / 12.0)
    assert contract.local_loss_scale("DP_MEAN") == pytest.approx(1.0)
    with pytest.raises(ValueError, match="explicitly bound"):
        contract.local_loss_scale("AUTO")


def test_logical_sample_weights_are_exact_batch12_mean() -> None:
    validate_per_sample_weights([1.0 / 12.0] * 12)
    with pytest.raises(AssertionError):
        validate_per_sample_weights([1.0 / 16.0] * 12)
