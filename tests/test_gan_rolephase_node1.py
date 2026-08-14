from __future__ import annotations

from types import SimpleNamespace

import pytest

from runtime.gan_rolephase_node1 import (
    assert_phase_b_zero1_contract,
    bind_node1_rank,
    phase_b_zero1_config,
)


def test_node1_mapping_preserves_phase_local_batches() -> None:
    bindings = [
        bind_node1_rank(global_rank=rank, local_rank=rank - 8)
        for rank in range(8, 16)
    ]
    assert [item.phase_a_logical_slots for item in bindings] == [
        (lane, lane + 8) for lane in range(8)
    ]
    assert [item.phase_b_logical_slot for item in bindings] == list(range(4, 12))
    assert [item.phase_b_generator_rank for item in bindings] == [0, 1, 2, 3] * 2
    record = bindings[0].as_dict()
    assert record["phase_a"]["gradient_service_calls"] == 2
    assert record["phase_a"]["expected_score_forwards"] == 8
    assert record["phase_b"]["critic_loss_calls_per_step"] == 1
    assert record["phase_b"]["expected_score_forwards_per_step"] == 2


def test_node1_mapping_fails_closed_on_rank_drift() -> None:
    with pytest.raises(ValueError, match="ranks 8..15"):
        bind_node1_rank(global_rank=7, local_rank=7)
    with pytest.raises(ValueError, match="global/local"):
        bind_node1_rank(global_rank=8, local_rank=1)
    with pytest.raises(ValueError, match="world16"):
        bind_node1_rank(global_rank=8, local_rank=0, world_size=8)


def test_phase_b_config_is_dp12_b1_gas1_zero1_cpu_optimizer() -> None:
    config = phase_b_zero1_config()
    assert config["train_micro_batch_size_per_gpu"] == 1
    assert config["gradient_accumulation_steps"] == 1
    assert config["train_batch_size"] == 12
    assert config["gradient_clipping"] == 1.0
    zero = config["zero_optimization"]
    assert zero["stage"] == 1
    assert zero["offload_optimizer"] == {"device": "cpu", "pin_memory": False}


def test_resolved_engine_contract_rejects_zero2_or_wrong_batch() -> None:
    group = object()
    optimizer = SimpleNamespace(
        partition_gradients=False,
        dp_process_group=group,
        clip_grad=1.0,
    )
    engine = SimpleNamespace(
        dp_world_size=lambda: 12,
        train_micro_batch_size_per_gpu=lambda: 1,
        gradient_accumulation_steps=lambda: 1,
        train_batch_size=lambda: 12,
        zero_optimization_stage=lambda: 1,
        optimizer=optimizer,
    )
    assert assert_phase_b_zero1_contract(engine, expected_dp_group=group)[
        "dp_world_size"
    ] == 12
    optimizer.partition_gradients = True
    with pytest.raises(AssertionError, match="ZeRO-2"):
        assert_phase_b_zero1_contract(engine, expected_dp_group=group)
