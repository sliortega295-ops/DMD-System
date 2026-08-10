from __future__ import annotations

import pytest

from distillschedule.runtime.gan_pipeline import (
    CriticAccumulationState,
    GanPipelinePhase,
    GanPipelinePlanner,
    GanPipelineQueueState,
    GanPipelineTopology,
    ScoreGradientReply,
    ScoreGradientRequest,
    VersionState,
    expected_final_versions,
    summarize_operations,
)


def test_world16_gan_pipeline_keeps_persistent_8g8score_layout() -> None:
    topology = GanPipelineTopology()
    assert topology.generator_ranks == tuple(range(8))
    assert topology.score_ranks == tuple(range(8, 16))
    for rank in range(8):
        assert topology.paired_score(rank) == rank + 8
        assert topology.paired_generator(rank + 8) == rank


def test_step1000_commits_generator_before_same_step_critic_rollout() -> None:
    planner = GanPipelinePlanner()
    initial = VersionState(generator=200, critic=1000, ema=1000)
    operations = planner.compile(start_step=1000, steps=1, initial=initial)
    phases = [operation.phase for operation in operations]
    assert phases[0] is GanPipelinePhase.FLUSH
    assert phases.count(GanPipelinePhase.GENERATOR_FORWARD) == 2
    assert phases.count(GanPipelinePhase.SCORE_BOUNDARY_GRAD) == 2
    assert phases.count(GanPipelinePhase.GENERATOR_BACKWARD) == 2
    assert phases.count(GanPipelinePhase.GENERATOR_COMMIT) == 1
    assert phases.count(GanPipelinePhase.CRITIC_COMMIT) == 1
    assert phases.count(GanPipelinePhase.EMA_COMMIT) == 1

    g_commit_index = phases.index(GanPipelinePhase.GENERATOR_COMMIT)
    first_critic_rollout = phases.index(GanPipelinePhase.CRITIC_ROLLOUT)
    assert g_commit_index < first_critic_rollout
    assert operations[first_critic_rollout].versions_before.generator == 201
    assert operations[first_critic_rollout].versions_before.critic == 1000


def test_first_gan_active_full5_has_exact_official_versions() -> None:
    initial = VersionState(generator=200, critic=1000, ema=1000)
    operations = GanPipelinePlanner().compile(
        start_step=1000,
        steps=5,
        initial=initial,
    )
    summary = summarize_operations(operations)
    assert summary["generator_commits"] == 1
    assert summary["critic_commits"] == 5
    assert summary["ema_commits"] == 5
    expected = expected_final_versions(start_step=1000, steps=5, initial=initial)
    assert expected == VersionState(generator=201, critic=1005, ema=1005)
    assert operations[-1].versions_after == expected


def test_two_full_cycles_flush_again_before_step1005_generator_update() -> None:
    initial = VersionState(generator=200, critic=1000, ema=1000)
    operations = GanPipelinePlanner().compile(
        start_step=1000,
        steps=10,
        initial=initial,
    )
    step1005 = [operation for operation in operations if operation.logical_step == 1005]
    assert step1005[0].phase is GanPipelinePhase.FLUSH
    assert step1005[0].queue_must_be_empty is True
    assert step1005[0].versions_before == VersionState(201, 1005, 1005)
    assert operations[-1].versions_after == VersionState(202, 1010, 1010)


def test_score_gradient_service_is_read_only_for_critic_state() -> None:
    request = ScoreGradientRequest(
        logical_step=1000,
        micro_wave=0,
        generator_version=200,
        critic_version=1000,
        trace_digest="a" * 64,
    )
    reply = ScoreGradientReply(
        logical_step=1000,
        micro_wave=0,
        generator_version=200,
        critic_before=1000,
        critic_after=1000,
        trace_digest="a" * 64,
    )
    reply.validate_against(request)
    bad = ScoreGradientReply(
        logical_step=1000,
        micro_wave=0,
        generator_version=200,
        critic_before=1000,
        critic_after=1001,
        trace_digest="a" * 64,
    )
    with pytest.raises(AssertionError, match="advanced Critic state"):
        bad.validate_against(request)


def test_critic_global_batch16_requires_four_backwards_and_one_commit() -> None:
    state = CriticAccumulationState()
    state.denoise_backward(0)
    state.gan_backward(0)
    assert not state.can_commit()
    with pytest.raises(RuntimeError, match="before both"):
        state.commit()
    state.denoise_backward(1)
    state.gan_backward(1)
    assert state.can_commit()
    state.commit()
    with pytest.raises(RuntimeError, match="already committed"):
        state.denoise_backward(0)


def test_q2_queue_must_flush_at_generator_barrier() -> None:
    queue = GanPipelineQueueState()
    queue.push(logical_step=1001, micro_wave=0, generator_version=201)
    queue.push(logical_step=1001, micro_wave=1, generator_version=201)
    with pytest.raises(RuntimeError, match="exceeded Q2"):
        queue.push(logical_step=1002, micro_wave=0, generator_version=201)
    assert queue.begin_consume() == (1001, 0, 201)
    assert queue.finish_consume() == (1001, 0, 201)
    assert queue.begin_consume() == (1001, 1, 201)
    assert queue.finish_consume() == (1001, 1, 201)
    queue.assert_empty()
