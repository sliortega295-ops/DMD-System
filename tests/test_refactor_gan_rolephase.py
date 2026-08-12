from __future__ import annotations

import pytest

from distillschedule.runtime.gan_rolephase import (
    ConsumerQ2Window,
    GanFull5RolePhaseState,
    GanRolePhase,
    GanRolePhaseTopology,
    PhaseResidencyContract,
    ProducerPayloadCredits,
    balanced_shards,
    plan_contiguous_reshard,
    rolephase_contract_summary,
)


def test_rolephase_topology_preserves_phase_a_and_switches_only_flex() -> None:
    topology = GanRolePhaseTopology()
    assert topology.phase_a_generator_ranks == tuple(range(8))
    assert topology.phase_a_score_ranks == tuple(range(8, 16))
    assert topology.phase_b_generator_ranks == (0, 1, 2, 3)
    assert topology.phase_b_critic_ranks == tuple(range(4, 16))
    assert topology.flex_ranks == (4, 5, 6, 7)
    assert topology.phase_a_global_batch == 16
    assert topology.phase_b_global_batch == 12


def test_phase_a_keeps_exact_two_wave_batch16_mapping() -> None:
    topology = GanRolePhaseTopology()
    slots = [topology.phase_a_slot(slot) for slot in range(16)]
    assert [item.micro_wave for item in slots] == [0] * 8 + [1] * 8
    assert [item.generator_rank for item in slots[:8]] == list(range(8))
    assert [item.generator_rank for item in slots[8:]] == list(range(8))
    assert [item.score_rank for item in slots[:8]] == list(range(8, 16))
    assert [item.score_rank for item in slots[8:]] == list(range(8, 16))


def test_phase_b_is_exact_4g12c_batch12_b1() -> None:
    topology = GanRolePhaseTopology()
    slots = [topology.phase_b_slot(slot) for slot in range(12)]
    assert sorted(item.logical_slot for item in slots) == list(range(12))
    assert all(item.critic_micro_wave == 0 for item in slots)
    assert sum(item.loss_weight for item in slots) == pytest.approx(1.0)

    for rank in topology.fixed_generator_ranks:
        owned = topology.phase_b_slots_for_generator(rank)
        assert len(owned) == 3
        assert [item.logical_slot for item in owned] == [rank, rank + 4, rank + 8]

    for rank in topology.phase_b_critic_ranks:
        owned = topology.phase_b_slots_for_critic(rank)
        assert len(owned) == 1


def test_phase_b_rejects_old_batch16_slots() -> None:
    topology = GanRolePhaseTopology()
    for slot in range(12, 16):
        with pytest.raises(ValueError, match="\[0,12\)"):
            topology.phase_b_slot(slot)


def test_phase_residency_contract_rejects_repeated_cpu_frozen_streaming() -> None:
    PhaseResidencyContract()
    with pytest.raises(ValueError, match="repeated CPU"):
        PhaseResidencyContract(repeated_cpu_frozen_streaming=True)


def _apply_reshard(total: int, src_ranks: tuple[int, ...], dst_ranks: tuple[int, ...]):
    source_ranges = {item.rank: item for item in balanced_shards(total, src_ranks)}
    destination_ranges = {item.rank: item for item in balanced_shards(total, dst_ranks)}
    canonical = list(range(total))
    source = {
        rank: canonical[shard.start : shard.stop]
        for rank, shard in source_ranges.items()
    }
    destination = {
        rank: [None] * shard.numel
        for rank, shard in destination_ranges.items()
    }
    plan = plan_contiguous_reshard(total, src_ranks, dst_ranks, max_chunk_numel=7)
    for item in plan:
        width = item.numel
        destination[item.dst_rank][item.dst_offset : item.dst_offset + width] = (
            source[item.src_rank][item.src_offset : item.src_offset + width]
        )
    rebuilt = []
    for rank in dst_ranks:
        rebuilt.extend(destination[rank])
    return rebuilt, plan


def test_gpu_reshard_diagnostic_still_exact_for_layout_changes() -> None:
    rebuilt, _ = _apply_reshard(257, tuple(range(8)), tuple(range(4)))
    assert rebuilt == list(range(257))
    rebuilt, _ = _apply_reshard(509, tuple(range(8, 16)), tuple(range(4, 16)))
    assert rebuilt == list(range(509))


def test_producer_payload_credit_returns_on_send_completion_not_critic_commit() -> None:
    credits = ProducerPayloadCredits()
    slot0 = credits.reserve(1000)
    credits.mark_send_posted(slot0, 1000)
    slot1 = credits.reserve(1001)
    credits.mark_send_posted(slot1, 1001)
    assert credits.available == 0
    with pytest.raises(RuntimeError, match="credits exhausted"):
        credits.reserve(1002)
    credits.mark_send_complete(slot0, 1000)
    assert credits.available == 1
    assert credits.reserve(1002) == slot0


def test_consumer_q2_allows_one_batch_ahead_but_not_three_uncommitted() -> None:
    queue = ConsumerQ2Window()
    queue.accept(1000)
    queue.accept(1001)
    with pytest.raises(RuntimeError, match="Q2 window is full"):
        queue.accept(1002)
    queue.begin_consume(1000)
    queue.commit(1000)
    queue.accept(1002)
    queue.begin_consume(1001)
    queue.commit(1001)
    queue.begin_consume(1002)
    queue.commit(1002)
    queue.assert_empty()


def test_full5_rolephase_state_preserves_official_g_c_ema_versions() -> None:
    state = GanFull5RolePhaseState()
    state.flush_phase_a_score_service()
    state.commit_generator()
    assert state.generator_version == 201
    state.begin_handoff_to_b()
    state.finish_handoff_to_b()
    assert state.phase is GanRolePhase.PHASE_B

    for step in range(1000, 1005):
        state.commit_critic(step)
        if step == 1000:
            state.commit_ema_after_step1000()
    state.begin_reverse_handoff()
    state.finish_reverse_handoff()
    state.assert_complete()


def test_contract_summary_exposes_phase_local_batches() -> None:
    summary = rolephase_contract_summary()
    assert summary["phase_a"]["global_batch"] == 16
    assert summary["phase_b"]["global_batch"] == 12
    assert summary["phase_b"]["generator_sample_counts"] == {0: 3, 1: 3, 2: 3, 3: 3}
    assert set(summary["phase_b"]["critic_sample_counts"].values()) == {1}
    assert summary["phase_b"]["queue_depth"] == 2
    assert summary["residency"]["repeated_cpu_frozen_streaming"] is False
