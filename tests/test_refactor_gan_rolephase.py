from __future__ import annotations

import pytest

from distillschedule.runtime.gan_rolephase import (
    ConsumerQ2Window,
    GanFull5RolePhaseState,
    GanRolePhase,
    GanRolePhaseTopology,
    GpuShardedResidencyContract,
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


def test_phase_a_keeps_exact_two_wave_batch16_mapping() -> None:
    topology = GanRolePhaseTopology()
    slots = [topology.phase_a_slot(slot) for slot in range(16)]
    assert [item.micro_wave for item in slots] == [0] * 8 + [1] * 8
    assert [item.generator_rank for item in slots[:8]] == list(range(8))
    assert [item.generator_rank for item in slots[8:]] == list(range(8))
    assert [item.score_rank for item in slots[:8]] == list(range(8, 16))
    assert [item.score_rank for item in slots[8:]] == list(range(8, 16))


def test_phase_b_maps_batch16_to_4g12c_without_sample_loss() -> None:
    topology = GanRolePhaseTopology()
    slots = [topology.phase_b_slot(slot) for slot in range(16)]
    assert sorted(item.logical_slot for item in slots) == list(range(16))
    assert sum(item.loss_weight for item in slots) == pytest.approx(1.0)

    for rank in topology.fixed_generator_ranks:
        owned = topology.phase_b_slots_for_generator(rank)
        assert len(owned) == 4
        assert [item.logical_slot for item in owned] == [rank, rank + 4, rank + 8, rank + 12]

    for rank in topology.flex_ranks:
        assert len(topology.phase_b_slots_for_critic(rank)) == 2
    for rank in topology.fixed_critic_ranks:
        assert len(topology.phase_b_slots_for_critic(rank)) == 1


def test_phase_b_critic_waves_are_12_plus_4() -> None:
    topology = GanRolePhaseTopology()
    slots = [topology.phase_b_slot(slot) for slot in range(16)]
    assert sum(item.critic_micro_wave == 0 for item in slots) == 12
    assert sum(item.critic_micro_wave == 1 for item in slots) == 4
    assert {item.critic_rank for item in slots if item.critic_micro_wave == 1} == set(
        topology.flex_ranks
    )


def test_gpu_residency_contract_rejects_repeated_cpu_frozen_streaming() -> None:
    GpuShardedResidencyContract()
    with pytest.raises(ValueError, match="repeated CPU"):
        GpuShardedResidencyContract(repeated_cpu_frozen_streaming=True)


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


def test_gpu_reshard_plan_exactly_supports_g8_to_g4_and_back() -> None:
    total = 257
    rebuilt, forward = _apply_reshard(total, tuple(range(8)), tuple(range(4)))
    assert rebuilt == list(range(total))
    assert sum(item.numel for item in forward) == total
    rebuilt_back, reverse = _apply_reshard(total, tuple(range(4)), tuple(range(8)))
    assert rebuilt_back == list(range(total))
    assert sum(item.numel for item in reverse) == total


def test_gpu_reshard_plan_exactly_supports_score8_to_critic12_and_back() -> None:
    total = 509
    c8 = tuple(range(8, 16))
    c12 = tuple(range(4, 16))
    rebuilt, forward = _apply_reshard(total, c8, c12)
    assert rebuilt == list(range(total))
    assert sum(item.numel for item in forward) == total
    rebuilt_back, reverse = _apply_reshard(total, c12, c8)
    assert rebuilt_back == list(range(total))
    assert sum(item.numel for item in reverse) == total


def test_producer_payload_credit_returns_on_send_completion_not_critic_commit() -> None:
    credits = ProducerPayloadCredits()
    slot0 = credits.reserve(1000)
    credits.mark_send_posted(slot0, 1000)
    slot1 = credits.reserve(1001)
    credits.mark_send_posted(slot1, 1001)
    assert credits.available == 0
    with pytest.raises(RuntimeError, match="credits exhausted"):
        credits.reserve(1002)

    # No Critic ACK/commit is involved: NCCL send completion is sufficient to
    # reuse the producer payload buffer.
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


def test_contract_summary_exposes_no_cpu_streaming_and_q2() -> None:
    summary = rolephase_contract_summary()
    assert summary["phase_a"]["generator_ranks"] == list(range(8))
    assert summary["phase_b"]["critic_ranks"] == list(range(4, 16))
    assert summary["phase_b"]["queue_depth"] == 2
    assert summary["residency"]["repeated_cpu_frozen_streaming"] is False
