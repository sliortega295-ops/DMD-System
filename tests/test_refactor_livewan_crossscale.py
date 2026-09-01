from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from distillschedule.adapters.livewan import LiveWanSFDMDAdapter
from distillschedule.core.config import load_plan
from distillschedule.core.contracts import ContractViolation
from distillschedule.workloads.livewan_crossscale import (
    PIPELINE_6G10S_LAYOUT,
    PIPELINE_6G10S_NODE_LOCAL_LAYOUT,
    PIPELINE_6G10S_SCORE_LEADERS,
    PIPELINE_6G10S_SCORE_NODE_GROUPS,
    PIPELINE_8G8S_LAYOUT,
    TRACE_BALANCED_8G8S_SLOT_PAIRS,
    balanced_slots,
    compile_trace_identities,
    full5_routes,
    speedup_receipt,
    trace_sha256,
)
from distillschedule.runtime.livewan_crossscale_worker import (
    OperationTraceRow,
    InputLoader,
    _cfg_velocity_full_context,
    _combine_dmd_gradient,
    _dmd_flow_inputs,
    _batch_seeded_flow_inputs,
    _per_sample_mse,
    _pipeline_slot_routes,
    _pipeline_8x8_slot_pairs,
    _prefetch_score_sample_contexts,
    prepare_score_sample_context,
)


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_g1c5_world16_fsdp2.json"
PIPELINE = ROOT / "configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline8x8.json"
PIPELINE_RUNAHEAD = ROOT / "configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline8x8_runahead.json"
PIPELINE_RUNAHEAD_CRITIC_B2 = ROOT / "configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline8x8_runahead_criticb2.json"
PIPELINE_RUNAHEAD_CRITIC_B2_LOCAL_CONTEXT = ROOT / "configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline8x8_runahead_criticb2_local_context.json"
PIPELINE_RUNAHEAD_CRITIC_B2_LOCAL_CONTEXT_ASYNC32 = ROOT / "configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline8x8_runahead_criticb2_local_context_async32.json"
PIPELINE_RUNAHEAD_CRITIC_B2_LOCAL_CONTEXT_ASYNC32_TRACE_BALANCED = ROOT / "configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline8x8_runahead_criticb2_local_context_async32_trace_balanced.json"
PIPELINE_RUNAHEAD_CRITIC_B2_LOCAL_CONTEXT_ASYNC32_TRACE_BALANCED_FULL_CONTEXT = ROOT / "configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline8x8_runahead_criticb2_local_context_async32_trace_balanced_full_context.json"
PIPELINE_RUNAHEAD_CRITIC_B2_LOCAL_CONTEXT_ASYNC32_TRACE_BALANCED_FULL_CONTEXT_CONTEXT_PREFETCH = ROOT / "configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline8x8_runahead_criticb2_local_context_async32_trace_balanced_full_context_context_prefetch.json"
PIPELINE_RUNAHEAD_CRITIC_B2_LOCAL_CONTEXT_ASYNC32_TRACE_BALANCED_FULL_CONTEXT_LORA_SCALE1 = ROOT / "configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline8x8_runahead_criticb2_local_context_async32_trace_balanced_full_context_lora_scale1.json"
PIPELINE_RUNAHEAD_CRITIC_B2_LOCAL_CONTEXT_ASYNC32_TRACE_BALANCED_FULL_CONTEXT_LORA_SCALE1_MODULATION = ROOT / "configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline8x8_runahead_criticb2_local_context_async32_trace_balanced_full_context_lora_scale1_modulation.json"
PIPELINE_RUNAHEAD_CRITIC_B2_LOCAL_CONTEXT_ASYNC32_TRACE_BALANCED_FULL_CONTEXT_LORA_FUSED_CAST_ADD = ROOT / "configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline8x8_runahead_criticb2_local_context_async32_trace_balanced_full_context_lora_fused_cast_add.json"
PIPELINE_RUNAHEAD_CRITIC_B2_LOCAL_CONTEXT_ASYNC32_TRACE_BALANCED_FULL_CONTEXT_LORA_AUTOCAST_INPUT_ELISION = ROOT / "configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline8x8_runahead_criticb2_local_context_async32_trace_balanced_full_context_lora_autocast_input_elision.json"
PIPELINE_RUNAHEAD_CRITIC_B2_LOCAL_CONTEXT_ASYNC32_TRACE_BALANCED_FULL_CONTEXT_LORA_AUTOCAST_INPUT_ELISION_SPLIT_DMD_SHADOW = ROOT / "configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline8x8_runahead_criticb2_local_context_async32_trace_balanced_full_context_lora_autocast_input_elision_split_dmd_shadow.json"
PIPELINE_RUNAHEAD_CRITIC_B2_LOCAL_CONTEXT_ASYNC32_TRACE_BALANCED_FULL_CONTEXT_LORA_AUTOCAST_INPUT_ELISION_SPLIT_DMD_SHADOW_FUSED_ADAM = ROOT / "configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline8x8_runahead_criticb2_local_context_async32_trace_balanced_full_context_lora_autocast_input_elision_split_dmd_shadow_fused_adam.json"
PIPELINE_RUNAHEAD_CRITIC_B2_LOCAL_CONTEXT_ASYNC32_TRACE_BALANCED_FULL_CONTEXT_LORA_AUTOCAST_INPUT_ELISION_SPLIT_DMD_SHADOW_FUSED_ADAM_SCORE_ROPE = ROOT / "configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline8x8_runahead_criticb2_local_context_async32_trace_balanced_full_context_lora_autocast_input_elision_split_dmd_shadow_fused_adam_score_rope.json"
PIPELINE_RUNAHEAD_CRITIC_B2_LOCAL_CONTEXT_ASYNC32_TRACE_BALANCED_FULL_CONTEXT_LORA_AUTOCAST_INPUT_ELISION_SPLIT_DMD_SHADOW_FUSED_ADAM_SCORE_ROPE_FLASH_ATTN2 = ROOT / "configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline8x8_runahead_criticb2_local_context_async32_trace_balanced_full_context_lora_autocast_input_elision_split_dmd_shadow_fused_adam_score_rope_flash_attn2.json"
PIPELINE_RUNAHEAD_CRITIC_B2_LOCAL_CONTEXT_ASYNC32_TRACE_BALANCED_FULL_CONTEXT_LORA_AUTOCAST_INPUT_ELISION_SPLIT_DMD_SHADOW_FUSED_ADAM_SCORE_ROPE_FLASH_ATTN2_GENERATOR_CROSS_KV = ROOT / "configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline8x8_runahead_criticb2_local_context_async32_trace_balanced_full_context_lora_autocast_input_elision_split_dmd_shadow_fused_adam_score_rope_flash_attn2_generator_cross_kv_cache.json"
PIPELINE_RUNAHEAD_CRITIC_B2_LOCAL_CONTEXT_ASYNC32_TRACE_BALANCED_FULL_CONTEXT_LORA_AUTOCAST_INPUT_ELISION_SPLIT_DMD_SHADOW_FUSED_ADAM_SCORE_ROPE_FLASH_ATTN2_GENERATOR_CROSS_KV_HOST_RUNAHEAD = ROOT / "configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline8x8_runahead_criticb2_local_context_async32_trace_balanced_full_context_lora_autocast_input_elision_split_dmd_shadow_fused_adam_score_rope_flash_attn2_generator_cross_kv_cache_host_runahead.json"
PIPELINE_C20_GENERATOR_FLASH_ATTN2 = ROOT / "configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_c20_generator_flash_attn2.json"
PIPELINE_PHASE_EXPANDED_CRITIC16 = ROOT / "configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline8x8_phase_expanded_critic16.json"
PIPELINE_PHASE_EXPANDED_CRITIC16_FUSED_LORA_ELISION = ROOT / "configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline8x8_phase_expanded_critic16_fused_lora_elision.json"
PIPELINE_SCORE_B2 = ROOT / "configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline8x8_scoreb2.json"
PIPELINE_6G10S = ROOT / "configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline6g10score.json"
PIPELINE_6G10S_NODE_LOCAL = ROOT / "configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline6g10score_node_local.json"
PIPELINE_6G10S_NODE_LOCAL_FUSED = ROOT / "configs/distillschedule/plans/livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline6g10score_node_local_fused.json"


@pytest.fixture(autouse=True)
def plan_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    values = {
        "LIVEWAN_STUDENT_CHECKPOINT": tmp_path / "student",
        "LIVEWAN_CRITIC_CHECKPOINT": tmp_path / "critic",
        "LIVEWAN_TEACHER_CHECKPOINT": tmp_path / "teacher",
        "LIVEWAN_OPERATION_TRACE": tmp_path / "trace.json",
        "LIVEWAN_SYSTEMS_BOOTSTRAP": tmp_path / "bootstrap.json",
        "DISTILLSCHEDULE_OUTPUT_DIR": tmp_path / "output",
        "DISTILLSCHEDULE_PYTHON": "python3",
        "DMD_SYSTEM_ROOT": ROOT,
        "DISTILLSCHEDULE_NODE_RANK": "0",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, str(value))


def test_livewan_plans_compile_and_freeze_only_placement() -> None:
    baseline = load_plan(BASELINE).plan
    pipeline = load_plan(PIPELINE).plan
    adapter = LiveWanSFDMDAdapter()
    adapter.validate(baseline)
    adapter.validate(pipeline)
    assert baseline.workload.to_dict() == pipeline.workload.to_dict()
    assert baseline.hardware.to_dict() == pipeline.hardware.to_dict()
    assert baseline.workload.teacher is not None
    assert "t-wan2.1-14b-bf16" in baseline.profile_key
    assert baseline.metadata["execution_mode"] == "baseline"
    assert pipeline.metadata["execution_mode"] == "pipeline"
    assert baseline.rank_groups[0].ranks == tuple(range(16))
    assert pipeline.rank_groups[0].ranks == tuple(range(8))


def test_score_b2_plan_changes_only_physical_score_batching() -> None:
    b1 = load_plan(PIPELINE).plan
    b2 = load_plan(PIPELINE_SCORE_B2).plan
    adapter = LiveWanSFDMDAdapter()
    adapter.validate(b2)
    assert b1.workload.to_dict() == b2.workload.to_dict()
    assert b1.hardware.to_dict() == b2.hardware.to_dict()
    assert b1.rank_groups == b2.rank_groups
    assert b2.metadata["execution_mode"] == "pipeline"
    assert b2.metadata["score_physical_batch_size"] == 2
    assert b2.metadata["logical_batch_preserved"] is True
    assert b2.metadata["per_sample_rng_preserved"] is True


def test_runahead_plan_changes_only_cross_step_ordering() -> None:
    reference = load_plan(PIPELINE).plan
    candidate = load_plan(PIPELINE_RUNAHEAD).plan
    LiveWanSFDMDAdapter().validate(candidate)
    assert reference.workload.to_dict() == candidate.workload.to_dict()
    assert reference.hardware.to_dict() == candidate.hardware.to_dict()
    assert reference.rank_groups == candidate.rank_groups
    assert reference.training_backends == candidate.training_backends
    assert candidate.metadata["generator_nograd_runahead"] is True
    assert candidate.metadata["runahead_buffers_per_generator_rank"] == 2
    assert candidate.metadata["runahead_steps"] == [1, 2, 3, 4]


def test_runahead_critic_b2_preserves_b1_dmd_and_single_commit() -> None:
    reference = load_plan(PIPELINE_RUNAHEAD).plan
    candidate = load_plan(PIPELINE_RUNAHEAD_CRITIC_B2).plan
    LiveWanSFDMDAdapter().validate(candidate)
    assert reference.workload.to_dict() == candidate.workload.to_dict()
    assert reference.hardware.to_dict() == candidate.hardware.to_dict()
    assert reference.rank_groups == candidate.rank_groups
    assert candidate.metadata["score_physical_batch_size"] == 1
    assert candidate.metadata["dmd_physical_batch_size"] == 1
    assert candidate.metadata["critic_physical_batch_size"] == 2
    assert candidate.metadata["critic_b2_preserves_per_sample_rng"] is True
    assert candidate.metadata["critic_b2_preserves_single_commit_per_step"] is True


def test_local_score_context_changes_only_transport_residency() -> None:
    reference = load_plan(PIPELINE_RUNAHEAD_CRITIC_B2).plan
    candidate = load_plan(PIPELINE_RUNAHEAD_CRITIC_B2_LOCAL_CONTEXT).plan
    LiveWanSFDMDAdapter().validate(candidate)
    assert reference.workload.to_dict() == candidate.workload.to_dict()
    assert reference.hardware.to_dict() == candidate.hardware.to_dict()
    assert reference.rank_groups == candidate.rank_groups
    assert reference.training_backends == candidate.training_backends
    assert candidate.metadata["score_context_transport"] == "trace_local_replay"
    assert candidate.lifecycle.metadata["context_p2p"] is False

    drifted = replace(
        candidate,
        metadata={**candidate.metadata, "score_context_transport": "cuda_p2p_float32"},
    )
    with pytest.raises(ContractViolation, match="immutable trace"):
        LiveWanSFDMDAdapter().validate(drifted)


def test_trace_local_score_context_selects_exact_fp32_prompt() -> None:
    torch = pytest.importorskip("torch")
    loader = object.__new__(InputLoader)
    loader.prompts = {"pos": torch.zeros((4, 512, 4096), dtype=torch.float32)}
    loader.prompts["pos"][2].fill_(2.0)
    row = _trace_row(0, 100)
    row = replace(row, dataset_index=2)
    context = loader.context_from_trace(row, device=torch.device("cpu"))
    assert context.dtype == torch.float32
    assert context.shape == (1, 512, 4096)
    assert torch.equal(context[0], loader.prompts["pos"][2])


def test_6g10score_plan_changes_only_placement_and_exact_physical_batching() -> None:
    b1 = load_plan(PIPELINE).plan
    candidate = load_plan(PIPELINE_6G10S).plan
    adapter = LiveWanSFDMDAdapter()
    adapter.validate(candidate)
    assert b1.workload.to_dict() == candidate.workload.to_dict()
    assert b1.hardware.to_dict() == candidate.hardware.to_dict()
    assert candidate.rank_groups[0].ranks == tuple(range(6))
    assert candidate.rank_groups[1].ranks == tuple(range(6, 16))
    assert candidate.metadata["pool_policy"] == "persistent_6g_10score_teacher_fsdp10"
    assert candidate.metadata["score_physical_batch_size"] == 2
    assert candidate.metadata["logical_batch_preserved"] is True
    assert candidate.metadata["per_sample_rng_preserved"] is True


def test_6g10score_node_local_plan_changes_only_collective_substrate() -> None:
    flat = load_plan(PIPELINE_6G10S).plan
    candidate = load_plan(PIPELINE_6G10S_NODE_LOCAL).plan
    LiveWanSFDMDAdapter().validate(candidate)
    assert flat.workload.to_dict() == candidate.workload.to_dict()
    assert flat.hardware.to_dict() == candidate.hardware.to_dict()
    assert flat.rank_groups == candidate.rank_groups
    assert candidate.metadata["pool_policy"] == (
        "persistent_6g_10score_teacher_node_local_hierarchical"
    )
    assert candidate.metadata["large_score10_collectives"] is False
    assert candidate.training_backends["teacher"].options["mesh_ranks_by_node"] == [
        list(ranks) for ranks in PIPELINE_6G10S_SCORE_NODE_GROUPS
    ]
    assert candidate.training_backends["score"].options["leader_ranks"] == list(
        PIPELINE_6G10S_SCORE_LEADERS
    )


def test_6g10score_node_local_fused_changes_only_generator_adam_kernel() -> None:
    reference = load_plan(PIPELINE_6G10S_NODE_LOCAL).plan
    candidate = load_plan(PIPELINE_6G10S_NODE_LOCAL_FUSED).plan
    LiveWanSFDMDAdapter().validate(candidate)
    assert reference.workload.to_dict() == candidate.workload.to_dict()
    assert reference.hardware.to_dict() == candidate.hardware.to_dict()
    assert reference.rank_groups == candidate.rank_groups
    assert candidate.metadata["pool_policy"] == reference.metadata["pool_policy"]
    assert candidate.metadata["generator_optimizer_impl"] == "fused_adamw"
    assert (
        candidate.training_backends["generator"].options["optimizer_impl"]
        == "fused_adamw"
    )


def test_hierarchical_score_reduction_recovers_same_global_mean() -> None:
    PIPELINE_6G10S_NODE_LOCAL_LAYOUT.validate()
    per_rank = {rank: float(rank - 5) for rank in range(6, 16)}
    local_totals = [
        sum(per_rank[rank] for rank in ranks)
        for ranks in PIPELINE_6G10S_SCORE_NODE_GROUPS
    ]
    hierarchical = sum(local_totals) / len(per_rank)
    flat = sum(per_rank.values()) / len(per_rank)
    assert hierarchical == flat
    assert PIPELINE_6G10S_SCORE_LEADERS == tuple(
        ranks[0] for ranks in PIPELINE_6G10S_SCORE_NODE_GROUPS
    )


def _trace_row(slot: int, offset: int) -> OperationTraceRow:
    return OperationTraceRow(
        logical_step=0,
        logical_slot=slot,
        dataset_index=slot,
        sample_path=f"sample-{slot}.pt",
        sample_sha256=None,
        operation_key=f"step=0/slot={slot}",
        world_frames=3,
        rope_gap=0,
        seeds={
            "rollout": offset + 1,
            "generator_dmd_t": offset + 2,
            "generator_dmd_noise": offset + 3,
            "generator_grad_blocks": offset + 4,
            "critic_t": offset + 5,
            "critic_noise": offset + 6,
        },
        generator_update=True,
        critic_update=True,
    )


def test_score_b2_random_inputs_are_two_exact_b1_streams() -> None:
    torch = pytest.importorskip("torch")
    rows = (_trace_row(0, 100), _trace_row(8, 200))
    x0 = torch.zeros((2, 2, 3), dtype=torch.float32)
    times, noises = _batch_seeded_flow_inputs(
        torch,
        rows=rows,
        x0=x0,
        t_seed_key="generator_dmd_t",
        noise_seed_key="generator_dmd_noise",
    )
    for index, row in enumerate(rows):
        one_t, one_noise = _batch_seeded_flow_inputs(
            torch,
            rows=(row,),
            x0=x0[index : index + 1],
            t_seed_key="generator_dmd_t",
            noise_seed_key="generator_dmd_noise",
        )
        assert torch.equal(times[index : index + 1], one_t)
        assert torch.equal(noises[index : index + 1], one_noise)


def test_score_b2_mean_loss_matches_two_weighted_b1_losses() -> None:
    torch = pytest.importorskip("torch")
    prediction_b2 = torch.tensor([[1.0, 2.0], [4.0, 8.0]], requires_grad=True)
    target = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    losses = _per_sample_mse(torch, prediction_b2, target)
    losses.mean().backward()
    batched_gradient = prediction_b2.grad.detach().clone()

    prediction_b1 = prediction_b2.detach().clone().requires_grad_(True)
    for index in range(2):
        _per_sample_mse(
            torch, prediction_b1[index : index + 1], target[index : index + 1]
        ).mul(0.5).backward()
    assert torch.equal(losses.detach(), torch.tensor([1.0, 14.5]))
    assert torch.equal(batched_gradient, prediction_b1.grad)


def test_critic_only_context_skips_unused_teacher_projection() -> None:
    torch = pytest.importorskip("torch")

    class Projection:
        def __init__(self) -> None:
            self.calls = 0

        def text_embedding(self, value):
            self.calls += 1
            return value + self.calls

    teacher = Projection()
    critic = Projection()
    bundle = SimpleNamespace(teacher=teacher, critic=critic)
    raw = torch.ones((1, 2, 3), dtype=torch.float32)

    critic_only = prepare_score_sample_context(
        torch,
        bundle=bundle,
        raw_context=raw,
        include_teacher=False,
    )
    assert critic_only.teacher is None
    assert teacher.calls == 0
    assert critic.calls == 1

    dmd = prepare_score_sample_context(
        torch,
        bundle=bundle,
        raw_context=raw,
        include_teacher=True,
    )
    assert dmd.teacher is not None
    assert teacher.calls == 1
    assert critic.calls == 2


def test_livewan_adapter_fails_closed_on_semantic_drift() -> None:
    plan = load_plan(BASELINE).plan
    changed_workload = replace(
        plan.workload,
        schedule=replace(plan.workload.schedule, generator_update_every=4),
    )
    with pytest.raises(ContractViolation, match="G1/C5"):
        LiveWanSFDMDAdapter().validate(replace(plan, workload=changed_workload))


def test_pipeline_8x8_routes_form_two_b1_waves() -> None:
    PIPELINE_8G8S_LAYOUT.validate()
    routes = full5_routes()
    assert len(routes) == 80
    for step in range(5):
        rows = [route for route in routes if route.logical_step == step]
        assert sorted(route.logical_slot for route in rows) == list(range(16))
        assert {route.wave for route in rows} == {0, 1}
        assert all(route.score_rank == route.generator_rank + 8 for route in rows)
    assert balanced_slots(tuple(range(8))) == {
        rank: (rank, rank + 8) for rank in range(8)
    }


def test_pipeline_6g10score_routes_are_balanced_bijective_and_real_only() -> None:
    PIPELINE_6G10S_LAYOUT.validate()
    assert balanced_slots(PIPELINE_6G10S_LAYOUT.generator_ranks) == {
        0: (0, 6, 12),
        1: (1, 7, 13),
        2: (2, 8, 14),
        3: (3, 9, 15),
        4: (4, 10),
        5: (5, 11),
    }
    assert balanced_slots(PIPELINE_6G10S_LAYOUT.score_ranks) == {
        6: (0, 10),
        7: (1, 11),
        8: (2, 12),
        9: (3, 13),
        10: (4, 14),
        11: (5, 15),
        12: (6,),
        13: (7,),
        14: (8,),
        15: (9,),
    }
    routes = _pipeline_slot_routes(
        PIPELINE_6G10S_LAYOUT.generator_ranks,
        PIPELINE_6G10S_LAYOUT.score_ranks,
    )
    assert [slot for slot, _, _ in routes] == list(range(16))
    assert len(set(routes)) == 16


def test_pipeline_6g10score_loss_scales_recover_global_sample_mean() -> None:
    values = [float(index + 1) for index in range(16)]
    score_slots = balanced_slots(PIPELINE_6G10S_LAYOUT.score_ranks)
    local_scaled_means = []
    for slots in score_slots.values():
        local_mean = sum(values[slot] for slot in slots) / len(slots)
        scale = len(PIPELINE_6G10S_LAYOUT.score_ranks) * len(slots) / 16
        local_scaled_means.append(local_mean * scale)
    assert sum(local_scaled_means) / len(local_scaled_means) == sum(values) / 16

    generator_slots = balanced_slots(PIPELINE_6G10S_LAYOUT.generator_ranks)
    local_weighted_sums = [
        sum(values[slot] * (len(generator_slots) / 16) for slot in slots)
        for slots in generator_slots.values()
    ]
    assert sum(local_weighted_sums) / len(local_weighted_sums) == sum(values) / 16


def test_operation_trace_identity_is_stable_and_rank_qualified() -> None:
    rows = compile_trace_identities(sample_count=96, seed=20260830)
    assert len(rows) == 80
    assert len({(row.logical_step, row.logical_slot) for row in rows}) == 80
    assert len({row.operation_key for row in rows}) == 80
    first = trace_sha256(rows)
    assert first == trace_sha256(compile_trace_identities(sample_count=96, seed=20260830))
    assert first != trace_sha256(compile_trace_identities(sample_count=96, seed=20260831))


def test_speedup_target_is_exactly_thirty_percent() -> None:
    passing = speedup_receipt(baseline_seconds=130.0, pipeline_seconds=100.0)
    failing = speedup_receipt(baseline_seconds=130.0, pipeline_seconds=100.01)
    assert passing["status"] == "PASS"
    assert passing["target_pipeline_seconds"] == 100.0
    assert failing["status"] == "BELOW_TARGET"


def test_submodules_are_pinned_and_source_prepare_is_non_installing() -> None:
    assert (ROOT / "third_party/livewan/.git").is_file()
    assert (ROOT / "third_party/wan21/.git").is_file()
    modules = (ROOT / ".gitmodules").read_text()
    assert "third_party/livewan" in modules
    assert "third_party/wan21" in modules
    assert subprocess.check_output(
        ["git", "-C", str(ROOT / "third_party/livewan"), "rev-parse", "HEAD"], text=True
    ).strip() == "1eabf43c777bc5b2816926ca998cd4e4c2dfa3d5"
    assert subprocess.check_output(
        ["git", "-C", str(ROOT / "third_party/wan21"), "rev-parse", "HEAD"], text=True
    ).strip() == "9737cba9c1c3c4d04b33fcad41c111989865d315"
    script = (ROOT / "scripts/distillschedule/prepare_livewan_crossscale_source.sh").read_text()
    assert "pip install" not in script
    assert "archive --format=tar" in script
    assert "1eabf43c777bc5b2816926ca998cd4e4c2dfa3d5" in script
    assert "9737cba9c1c3c4d04b33fcad41c111989865d315" in script
    overlay = (
        ROOT / "scripts/distillschedule/prepare_livewan_crossscale_python_overlay.sh"
    ).read_text()
    assert "easydict==1.13" in overlay
    assert "--target" in overlay
    assert "pip install" not in overlay.replace('"$(dirname "${PYTHON}")/pip" install', "")


def test_controller_keeps_remote_evidence_outside_source_checkout() -> None:
    controller = (
        ROOT / "scripts/distillschedule/run_livewan_crossscale_world16.py"
    ).read_text()
    assert '"--remote-output-root", required=True' in controller
    assert 'args.remote_output_root / args.run_id' in controller
    assert 'args.remote_root / "artifacts/runs"' not in controller
    assert (
        '"6g10score-node-local-fused",'
        in controller
    )
    assert '"b1-runahead"' in controller
    assert '"b1-runahead-critic-b2"' in controller
    assert '"b1-runahead-critic-b2-local-context"' in controller
    assert '"b1-runahead-critic-b2-local-context-async32"' in controller
    assert '"b1-runahead-critic-b2-local-context-async32-trace-balanced"' in controller
    assert '"b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context"' in controller
    assert '"b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context-context-prefetch"' in controller
    assert '"b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context-lora-scale1"' in controller
    assert '"b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context-lora-scale1-modulation"' in controller
    assert '"b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context-lora-fused-cast-add"' in controller
    assert '"b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context-lora-autocast-input-elision"' in controller
    assert '"b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context-lora-autocast-input-elision-split-dmd-shadow"' in controller
    assert '"b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context-lora-autocast-input-elision-split-dmd-shadow-fused-adam"' in controller
    assert '"c15-score-rope"' in controller
    assert '"c16-score-flash-attn2"' in controller
    assert '"c18-generator-cross-kv-cache"' in controller
    assert '"c19-host-runahead"' in controller
    assert '"c20-generator-flash-attn2"' in controller
    assert '"phase-expanded-critic16"' in controller
    assert '"capacity", "correctness", "timing", "profile"' in controller
    assert '"LIVEWAN_PIPELINE_VARIANT": args.pipeline_variant' in controller
    assert 'resolved_plan.node0.json' in controller

    transport_probe = (
        ROOT / "scripts/distillschedule/probe_livewan_crossscale_transport.py"
    ).read_text()
    # The C16 variant inherits every transport contract from C15; keeping all
    # six memberships explicit prevents validate-only from silently probing a
    # different slot/reducer/payload contract than the real worker.
    assert transport_probe.count('"c16-score-flash-attn2"') == 6
    assert transport_probe.count('"c18-generator-cross-kv-cache"') == 6
    # C19 inherits the six C18 memberships and adds two explicit Gloo-specific
    # gates (group construction and summary acceptance).
    assert transport_probe.count('"c19-host-runahead"') == 8
    assert transport_probe.count('"c20-generator-flash-attn2"') == 8

    launcher = (
        ROOT / "scripts/distillschedule/launch_livewan_crossscale_world16_node.sh"
    ).read_text()
    assert "livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline8x8_scoreb2.json" in launcher
    assert "livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline6g10score.json" in launcher
    assert "livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline6g10score_node_local.json" in launcher
    assert "livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline8x8_runahead_criticb2_local_context.json" in launcher
    assert "livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline8x8_runahead_criticb2_local_context_async32.json" in launcher
    assert "livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline8x8_runahead_criticb2_local_context_async32_trace_balanced.json" in launcher
    assert "livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline8x8_runahead_criticb2_local_context_async32_trace_balanced_full_context.json" in launcher
    assert "livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline8x8_runahead_criticb2_local_context_async32_trace_balanced_full_context_context_prefetch.json" in launcher
    assert "livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline8x8_runahead_criticb2_local_context_async32_trace_balanced_full_context_lora_scale1.json" in launcher
    assert "livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline8x8_runahead_criticb2_local_context_async32_trace_balanced_full_context_lora_scale1_modulation.json" in launcher
    assert "livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline8x8_phase_expanded_critic16.json" in launcher
    assert "livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline8x8_runahead_criticb2_local_context_async32_trace_balanced_full_context_lora_autocast_input_elision_split_dmd_shadow.json" in launcher
    assert "livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline8x8_runahead_criticb2_local_context_async32_trace_balanced_full_context_lora_autocast_input_elision_split_dmd_shadow_fused_adam.json" in launcher
    assert "livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline8x8_runahead_criticb2_local_context_async32_trace_balanced_full_context_lora_autocast_input_elision_split_dmd_shadow_fused_adam_score_rope.json" in launcher
    assert "livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline8x8_runahead_criticb2_local_context_async32_trace_balanced_full_context_lora_autocast_input_elision_split_dmd_shadow_fused_adam_score_rope_flash_attn2.json" in launcher
    assert "livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline8x8_runahead_criticb2_local_context_async32_trace_balanced_full_context_lora_autocast_input_elision_split_dmd_shadow_fused_adam_score_rope_flash_attn2_generator_cross_kv_cache.json" in launcher
    assert "livewan_wan13b_14b_sf_dmd_g1c5_world16_pipeline8x8_runahead_criticb2_local_context_async32_trace_balanced_full_context_lora_autocast_input_elision_split_dmd_shadow_fused_adam_score_rope_flash_attn2_generator_cross_kv_cache_host_runahead.json" in launcher
    assert "livewan_wan13b_14b_sf_dmd_c20_generator_flash_attn2.json" in launcher
    assert '"${RUN_KIND}" == capacity' in launcher
    assert "baseline_pipeline_variant_mismatch" in launcher
    summarizer = (
        ROOT / "scripts/distillschedule/summarize_livewan_crossscale_world16.py"
    ).read_text()
    assert 'value.add_argument("--plan", required=True' in summarizer
    assert '"physical_call_inventory"' in summarizer
    assert '"logical_call_inventory"' in summarizer
    assert '"persistent_6g_10score_teacher_fsdp10"' in summarizer
    assert '"persistent_6g_10score_teacher_node_local_hierarchical"' in summarizer
    assert '!= "gpu_resident_replica"' in summarizer


def test_cold_loader_avoids_redundant_full_fp32_copies() -> None:
    worker = (
        ROOT / "src/distillschedule/runtime/livewan_crossscale_worker.py"
    ).read_text()
    assert 'mmap=True' in worker
    assert '"complete_stage1_direct_mmap"' in worker
    assert '_wan_model(WanModel, size="14B", config=config).to(torch.bfloat16)' in worker
    assert '_wan_model(WanModel, size="1.3B", config=config).to(torch.bfloat16)' in worker


def test_correctness_state_evidence_is_placement_invariant() -> None:
    worker = (
        ROOT / "src/distillschedule/runtime/livewan_crossscale_worker.py"
    ).read_text()
    metrics = worker.split("def _state_metrics(", 1)[1].split("@dataclass", 1)[0]
    assert "full_tensor()" in metrics
    assert "dist.all_reduce(values" not in metrics
    assert 'run_kind != "correctness"' in worker
    assert '"gradient_before_clip"' in worker
    assert '"gradient_after_clip"' in worker


def test_step0_p2p_posts_gradient_receive_before_second_wave_send() -> None:
    """Default-PG P2P peers must enqueue matching operations in one order."""
    worker = (
        ROOT / "src/distillschedule/runtime/livewan_crossscale_worker.py"
    ).read_text()
    pipeline = worker.split("def _run_pipeline_8x8(", 1)[1]
    generator_path = pipeline.split("for step in range(FULL5_STEPS):", 1)[1].split(
        "else:\n            bundle.critic_optimizer.zero_grad", 1
    )[0]
    step0 = generator_path.split("if step == 0:", 1)[1]
    assert step0.index("dist.recv(real0, src=peer)") < step0.index(
        "sends = _send_pair("
    )
    legacy = step0.split("# The peers must enqueue matching default-group P2P calls.", 1)[1]
    assert legacy.index("dist.recv(grad0, src=peer)") < legacy.index(
        "sends = _send_pair("
    )


def test_score_prioritizes_both_dmd_gradients_before_critic_backward() -> None:
    """The second returned gradient must not sit behind Critic backward0."""
    worker = (
        ROOT / "src/distillschedule/runtime/livewan_crossscale_worker.py"
    ).read_text()
    pipeline = worker.split("def _run_pipeline_8x8(", 1)[1].split(
        "def _run_pipeline_8x8_score_b2(", 1
    )[0]
    score_path = pipeline
    dmd1 = score_path.index('"livewan/pipeline/score/step_0/dmd_wave_1"')
    teacher1 = score_path.index(
        '"livewan/pipeline/score/step_0/teacher_velocity_wave_1"'
    )
    async_send = score_path.index("grad1_work = dist.isend(grad1_send, dst=peer)")
    critic0 = score_path.index(
        'label=f"livewan/pipeline/score/step_{step}/critic_backward_wave_0"'
    )
    critic1 = score_path.index(
        'label=f"livewan/pipeline/score/step_{step}/critic_backward_wave_1"'
    )
    send_wait = score_path.index(
        'label="livewan/pipeline/score/step_0/send_gradient_1_wait"'
    )
    assert teacher1 < dmd1 < async_send < critic0 < critic1 < send_wait


def test_trace_builder_requires_80_distinct_source_samples(tmp_path: Path) -> None:
    data = tmp_path / "latents"
    data.mkdir()
    prompts = tmp_path / "prompts.pt"
    prompts.write_bytes(b"prompt-bundle")
    for index in range(79):
        (data / f"sample_{index:03d}.pt").write_bytes(f"sample-{index}".encode())
    script = ROOT / "scripts/distillschedule/build_livewan_crossscale_trace.py"
    failed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--data",
            str(data),
            "--prompts",
            str(prompts),
            "--output",
            str(tmp_path / "trace.json"),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert failed.returncode != 0
    assert "needs 80 distinct latent samples" in failed.stderr

    (data / "sample_079.pt").write_bytes(b"sample-79")
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--data",
            str(data),
            "--prompts",
            str(prompts),
            "--output",
            str(tmp_path / "trace.json"),
        ],
        check=True,
    )
    trace = json.loads((tmp_path / "trace.json").read_text())
    assert trace["unique_samples_consumed"] == 80
    assert len({row["dataset_index"] for row in trace["rows"]}) == 80


def test_launcher_fails_before_model_load_on_world16_and_p2p_drift() -> None:
    launcher = (
        ROOT / "scripts/distillschedule/launch_livewan_crossscale_world16_node.sh"
    ).read_text()
    assert "probe_world16_topology.py" in launcher
    assert "probe_livewan_crossscale_transport.py" in launcher
    assert 'master-port="$((MASTER_PORT + 1))"' in launcher
    assert 'master-port="$((MASTER_PORT + 2))"' in launcher
    probe = (
        ROOT / "scripts/distillschedule/probe_livewan_crossscale_transport.py"
    ).read_text()
    assert "default_world_initialized_collectively" in probe
    assert "range(2)" in probe
    assert probe.index("dist.recv(gradient, src=peer)") < probe.index(
        "works = (dist.isend(header, dst=peer), dist.isend(payload, dst=peer))"
    )
    assert "deferred_gradient_work = dist.isend(gradient, dst=peer)" in probe
    assert probe.index("deferred_gradient_work = dist.isend") < probe.index(
        "deferred_gradient_work.wait()"
    )
    assert '"observed_gradients"' in probe
    assert '"6g10score-node-local-fused",' in probe
    assert '"b1-runahead"' in probe
    assert '"b1-runahead-critic-b2"' in probe
    assert '"b1-runahead-critic-b2-local-context"' in probe
    assert '"b1-runahead-critic-b2-local-context-async32"' in probe
    assert '"b1-runahead-critic-b2-local-context-async32-trace-balanced"' in probe
    assert '"b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context"' in probe
    assert '"b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context-context-prefetch"' in probe
    assert '"b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context-lora-scale1"' in probe
    assert '"b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context-lora-scale1-modulation"' in probe
    assert '"b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context-lora-fused-cast-add"' in probe
    assert '"b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context-lora-autocast-input-elision"' in probe
    assert '"b1-runahead-critic-b2-local-context-async32-trace-balanced-full-context-lora-autocast-input-elision-split-dmd-shadow"' in probe
    assert "async Score gradient mean mismatch" in probe
    assert '"async_reducer_canary"' in probe
    assert "node-local Teacher all-gather payload mismatch" in probe
    assert "hierarchical Score mean mismatch" in probe
    assert "dist.batch_isend_irecv" in probe


def test_critic_async32_plan_changes_only_gradient_sync_policy() -> None:
    base = load_plan(PIPELINE_RUNAHEAD_CRITIC_B2_LOCAL_CONTEXT).plan.to_dict()
    candidate = load_plan(PIPELINE_RUNAHEAD_CRITIC_B2_LOCAL_CONTEXT_ASYNC32).plan.to_dict()
    base_metadata = dict(base["metadata"])
    candidate_metadata = dict(candidate["metadata"])
    assert candidate_metadata.pop("critic_gradient_sync") == {
        "policy": "grad_ready_async_flat_all_reduce",
        "bucket_mib": 32,
        "group_size": 8,
        "reduction": "sum_then_divide_group_size",
        "overlap_with_backward": True,
    }
    base_metadata["pool_policy"] = candidate_metadata["pool_policy"]
    assert candidate_metadata == base_metadata
    base["id"] = candidate["id"]
    base["metadata"] = base_metadata
    candidate["metadata"] = candidate_metadata
    assert candidate == base


def test_trace_balanced_plan_changes_only_fixed_slot_pairing() -> None:
    base = load_plan(PIPELINE_RUNAHEAD_CRITIC_B2_LOCAL_CONTEXT_ASYNC32).plan
    candidate = load_plan(
        PIPELINE_RUNAHEAD_CRITIC_B2_LOCAL_CONTEXT_ASYNC32_TRACE_BALANCED
    ).plan
    LiveWanSFDMDAdapter().validate(candidate)
    assert base.workload.to_dict() == candidate.workload.to_dict()
    assert base.hardware.to_dict() == candidate.hardware.to_dict()
    assert base.rank_groups == candidate.rank_groups
    assert base.training_backends == candidate.training_backends
    assert base.lifecycle == candidate.lifecycle
    assert tuple(
        tuple(pair) for pair in candidate.metadata["slot_pairs_by_lane"]
    ) == TRACE_BALANCED_8G8S_SLOT_PAIRS
    assert candidate.metadata["slot_pair_policy"] == "trace_balanced_fixed_rank_pairs_v1"
    assert _pipeline_8x8_slot_pairs(candidate.to_dict()) == TRACE_BALANCED_8G8S_SLOT_PAIRS
    routed = [slot for pair in TRACE_BALANCED_8G8S_SLOT_PAIRS for slot in pair]
    assert sorted(routed) == list(range(16))


def test_phase_expanded_critic16_changes_only_registered_schedule_and_placement() -> None:
    reference = load_plan(
        PIPELINE_RUNAHEAD_CRITIC_B2_LOCAL_CONTEXT_ASYNC32_TRACE_BALANCED_FULL_CONTEXT_LORA_SCALE1
    ).plan
    candidate = load_plan(PIPELINE_PHASE_EXPANDED_CRITIC16).plan
    LiveWanSFDMDAdapter().validate(candidate)
    assert reference.workload.to_dict() == candidate.workload.to_dict()
    assert reference.hardware.to_dict() == candidate.hardware.to_dict()
    groups = {group.name: tuple(group.ranks) for group in candidate.rank_groups}
    assert groups["generator"] == tuple(range(8))
    assert groups["score"] == tuple(range(8, 16))
    assert groups["teacher"] == tuple(range(8, 16))
    assert groups["critic"] == tuple(range(16))
    assert candidate.training_backends["generator"].kind.value == "native"
    assert candidate.training_backends["generator"].options == {
        "resident": True,
        "reshard_after_forward": False,
        "gradient_sync": "group_all_reduce_sum_then_divide",
    }
    assert candidate.training_backends["score"].kind.value == "fsdp"
    assert candidate.training_backends["score"].options["mesh_ranks"] == list(
        range(16)
    )
    assert candidate.training_backends["score"].options["reshard_after_forward"] is False
    assert candidate.training_backends["score"].options["explicit_pre_dmd_unshard"] is True
    assert candidate.metadata["score_physical_batch_size"] == 1
    assert candidate.metadata["critic_physical_batch_size"] == 1
    assert candidate.metadata["phase_a_generator_global_batch"] == 16
    assert candidate.metadata["phase_b_critic_global_batch"] == 16
    assert candidate.metadata["phase_b_critic_rank_batch"] == 1
    assert candidate.metadata["variable_rollout_call_counts_require_resident_generator"] is True
    assert candidate.metadata["critic_collective_call_alignment"] == (
        "one_fsdp_forward_backward_per_rank_per_step"
    )
    assert candidate.metadata["generator_nograd_runahead"] is False
    assert candidate.metadata["runahead_buffers_per_generator_rank"] == 0
    assert _pipeline_8x8_slot_pairs(candidate.to_dict()) == (
        TRACE_BALANCED_8G8S_SLOT_PAIRS
    )
    worker = (
        ROOT / "src/distillschedule/runtime/livewan_crossscale_worker.py"
    ).read_text()
    assert "def _run_pipeline_phase_expanded_critic16(" in worker
    assert "world16_fsdp2_keep_unsharded" in worker
    assert "_materialize_fsdp_units" in worker
    assert "bundle.critic_group_size == WORLD_SIZE" in worker
    assert 'run_kind in {"capacity", "correctness", "timing", "profile"}' in worker


def test_phase_expanded_fused_lora_elision_reuses_validated_capacity_recovery() -> None:
    reference = load_plan(PIPELINE_PHASE_EXPANDED_CRITIC16).plan
    candidate = load_plan(PIPELINE_PHASE_EXPANDED_CRITIC16_FUSED_LORA_ELISION).plan
    LiveWanSFDMDAdapter().validate(candidate)
    reference_dict = reference.to_dict()
    candidate_dict = candidate.to_dict()
    assert reference.workload.to_dict() == candidate.workload.to_dict()
    assert reference.hardware.to_dict() == candidate.hardware.to_dict()
    assert reference.rank_groups == candidate.rank_groups
    assert reference.lifecycle.to_dict() == candidate.lifecycle.to_dict()
    assert candidate.training_backends["generator"].options == {
        **reference.training_backends["generator"].options,
        "optimizer_impl": "fused_adamw",
    }
    assert (
        candidate.metadata["capacity_recovery"]
        == "reuse_validated_foreach_sqrt_transient_recovery"
    )
    assert candidate.metadata["generator_optimizer_impl"] == "fused_adamw"
    assert candidate.metadata["critic_lora_kernel"]["name"] == (
        "scale_one_autocast_bf16_input_elision_v1"
    )
    for key in ("workload", "hardware", "rank_groups", "lifecycle"):
        assert candidate_dict[key] == reference_dict[key]


def test_livewan_launcher_scopes_shared_worktree_git_ownership() -> None:
    launcher = (
        ROOT / "scripts/distillschedule/launch_livewan_crossscale_world16_node.sh"
    ).read_text()
    assert 'export SUDO_UID="$(stat -c \'%u\' "${ROOT}")"' in launcher
    assert "git config --global" not in launcher


def test_full_context_plan_changes_only_mask_discovery_metadata() -> None:
    base = load_plan(
        PIPELINE_RUNAHEAD_CRITIC_B2_LOCAL_CONTEXT_ASYNC32_TRACE_BALANCED
    ).plan.to_dict()
    candidate_plan = load_plan(
        PIPELINE_RUNAHEAD_CRITIC_B2_LOCAL_CONTEXT_ASYNC32_TRACE_BALANCED_FULL_CONTEXT
    ).plan
    LiveWanSFDMDAdapter().validate(candidate_plan)
    candidate = candidate_plan.to_dict()
    candidate_metadata = dict(candidate["metadata"])
    assert candidate_metadata.pop("full_context_lens_elision") is True
    assert candidate_metadata.pop("context_length") == 512
    assert (
        candidate_metadata.pop("context_padding_policy")
        == "all_512_tokens_valid_no_padding"
    )
    assert (
        candidate_metadata.pop("attention_mask_equivalence")
        == "ctx_lens_512_equivalent_to_none_at_exact_tensor_length"
    )
    assert candidate_metadata == base["metadata"]
    base["id"] = candidate["id"]
    base["metadata"] = dict(candidate["metadata"])
    assert candidate == base


def test_context_prefetch_plan_changes_only_frozen_projection_schedule() -> None:
    base = load_plan(
        PIPELINE_RUNAHEAD_CRITIC_B2_LOCAL_CONTEXT_ASYNC32_TRACE_BALANCED_FULL_CONTEXT
    ).plan.to_dict()
    candidate_plan = load_plan(
        PIPELINE_RUNAHEAD_CRITIC_B2_LOCAL_CONTEXT_ASYNC32_TRACE_BALANCED_FULL_CONTEXT_CONTEXT_PREFETCH
    ).plan
    LiveWanSFDMDAdapter().validate(candidate_plan)
    candidate = candidate_plan.to_dict()
    metadata = dict(candidate["metadata"])
    assert metadata.pop("score_context_prefetch") is True
    assert metadata.pop("score_context_prefetch_scope") == (
        "all_10_lane_rows_before_first_latent_receive"
    )
    assert metadata.pop("score_context_invariance") == (
        "frozen_teacher_and_critic_text_embeddings_across_full5"
    )
    assert metadata.pop("score_context_calls_preserved") is True
    assert metadata.pop("score_context_prefetch_timing_position") == (
        "inside_full5_before_first_latent_receive"
    )
    assert metadata.pop("score_context_prefetch_rows_per_lane") == 10
    assert metadata.pop("score_context_teacher_rows_per_lane") == 2
    assert metadata == base["metadata"]
    base["id"] = candidate["id"]
    base["metadata"] = dict(candidate["metadata"])
    assert candidate == base


def test_context_prefetch_builds_exact_frozen_call_inventory() -> None:
    import torch

    class CountingEmbedding(torch.nn.Module):
        def __init__(self, width: int) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.eye(width, dtype=torch.bfloat16), requires_grad=False
            )
            self.calls = 0

        def forward(self, value):
            self.calls += 1
            return torch.nn.functional.linear(value, self.weight)

    class Trace:
        @staticmethod
        def row(step, slot):
            return SimpleNamespace(logical_step=step, logical_slot=slot)

    class Loader:
        def __init__(self) -> None:
            self.rows = []

        def context_from_trace(self, row, *, device):
            self.rows.append((row.logical_step, row.logical_slot))
            return torch.full(
                (1, 3, 4),
                float(row.logical_step * 10 + row.logical_slot),
                dtype=torch.float32,
                device=device,
            )

    teacher_embedding = CountingEmbedding(4)
    critic_embedding = CountingEmbedding(4)
    bundle = SimpleNamespace(
        teacher=SimpleNamespace(text_embedding=teacher_embedding),
        critic=SimpleNamespace(text_embedding=critic_embedding),
    )
    loader = Loader()
    contexts, receipt = _prefetch_score_sample_contexts(
        torch,
        trace=Trace(),
        loader=loader,
        bundle=bundle,
        lane_slots=(2, 9),
        device=torch.device("cpu"),
        run_kind="correctness",
    )
    assert set(contexts) == {(step, wave) for step in range(5) for wave in range(2)}
    assert teacher_embedding.calls == 2
    assert critic_embedding.calls == 10
    assert len(loader.rows) == 10
    assert receipt["logical_context_count"] == 10
    assert receipt["teacher_context_count"] == 2
    assert receipt["critic_context_count"] == 10
    assert receipt["gpu_resident_bytes"] > 0


def test_lora_scale_one_plan_changes_only_exact_kernel_metadata() -> None:
    base = load_plan(
        PIPELINE_RUNAHEAD_CRITIC_B2_LOCAL_CONTEXT_ASYNC32_TRACE_BALANCED_FULL_CONTEXT
    ).plan.to_dict()
    candidate_plan = load_plan(
        PIPELINE_RUNAHEAD_CRITIC_B2_LOCAL_CONTEXT_ASYNC32_TRACE_BALANCED_FULL_CONTEXT_LORA_SCALE1
    ).plan
    LiveWanSFDMDAdapter().validate(candidate_plan)
    candidate = candidate_plan.to_dict()
    metadata = dict(candidate["metadata"])
    assert metadata.pop("critic_lora_kernel") == {
        "enabled": True,
        "name": "scale_one_identity_epilogue_v1",
        "scale": 1.0,
        "math_change": "remove_identity_multiply_only",
        "base_input_output_dtype": "bfloat16",
        "lora_parameter_compute_dtype": "float32",
        "source_forward_sha256": (
            "70d00d496aa2bd8ea06d64e6b04af873644eff52539396f74ec586a4cbbe97a3"
        ),
    }
    assert metadata == base["metadata"]
    base["id"] = candidate["id"]
    base["metadata"] = dict(candidate["metadata"])
    assert candidate == base


def test_lora_fused_cast_add_plan_changes_only_exact_kernel_metadata() -> None:
    base = load_plan(
        PIPELINE_RUNAHEAD_CRITIC_B2_LOCAL_CONTEXT_ASYNC32_TRACE_BALANCED_FULL_CONTEXT
    ).plan.to_dict()
    candidate_plan = load_plan(
        PIPELINE_RUNAHEAD_CRITIC_B2_LOCAL_CONTEXT_ASYNC32_TRACE_BALANCED_FULL_CONTEXT_LORA_FUSED_CAST_ADD
    ).plan
    LiveWanSFDMDAdapter().validate(candidate_plan)
    candidate = candidate_plan.to_dict()
    metadata = dict(candidate["metadata"])
    assert metadata.pop("critic_lora_kernel") == {
        "enabled": True,
        "name": "scale_one_fused_cast_add_v1",
        "scale": 1.0,
        "math_change": (
            "remove_identity_multiply_and_fuse_fp32_to_bf16_cast_with_bf16_add"
        ),
        "base_input_output_dtype": "bfloat16",
        "lora_parameter_compute_dtype": "float32",
        "source_forward_sha256": (
            "70d00d496aa2bd8ea06d64e6b04af873644eff52539396f74ec586a4cbbe97a3"
        ),
        "fused_min_numel": 8_000_000,
        "triton_block_size": 256,
    }
    assert metadata == base["metadata"]
    base["id"] = candidate["id"]
    base["metadata"] = dict(candidate["metadata"])
    assert candidate == base


def test_lora_autocast_input_elision_plan_changes_only_exact_kernel_metadata() -> None:
    base = load_plan(
        PIPELINE_RUNAHEAD_CRITIC_B2_LOCAL_CONTEXT_ASYNC32_TRACE_BALANCED_FULL_CONTEXT
    ).plan.to_dict()
    candidate_plan = load_plan(
        PIPELINE_RUNAHEAD_CRITIC_B2_LOCAL_CONTEXT_ASYNC32_TRACE_BALANCED_FULL_CONTEXT_LORA_AUTOCAST_INPUT_ELISION
    ).plan
    LiveWanSFDMDAdapter().validate(candidate_plan)
    candidate = candidate_plan.to_dict()
    metadata = dict(candidate["metadata"])
    assert metadata.pop("critic_lora_kernel") == {
        "enabled": True,
        "name": "scale_one_autocast_bf16_input_elision_v1",
        "scale": 1.0,
        "math_change": (
            "remove_identity_multiply_and_redundant_pre_autocast_fp32_input_cast"
        ),
        "base_input_output_dtype": "bfloat16",
        "lora_parameter_compute_dtype": "float32",
        "source_forward_sha256": (
            "70d00d496aa2bd8ea06d64e6b04af873644eff52539396f74ec586a4cbbe97a3"
        ),
        "autocast_dtype": "bfloat16",
        "required_lora_output_dtype": "bfloat16",
    }
    assert metadata == base["metadata"]
    base["id"] = candidate["id"]
    base["metadata"] = dict(candidate["metadata"])
    assert candidate == base


def test_split_dmd_shadow_plan_changes_only_registered_execution_graph() -> None:
    base = load_plan(
        PIPELINE_RUNAHEAD_CRITIC_B2_LOCAL_CONTEXT_ASYNC32_TRACE_BALANCED_FULL_CONTEXT_LORA_AUTOCAST_INPUT_ELISION
    ).plan.to_dict()
    candidate_plan = load_plan(
        PIPELINE_RUNAHEAD_CRITIC_B2_LOCAL_CONTEXT_ASYNC32_TRACE_BALANCED_FULL_CONTEXT_LORA_AUTOCAST_INPUT_ELISION_SPLIT_DMD_SHADOW
    ).plan
    LiveWanSFDMDAdapter().validate(candidate_plan)
    candidate = candidate_plan.to_dict()
    metadata = dict(candidate["metadata"])
    assert metadata.pop("split_dmd_shadow") == {
        "enabled": True,
        "version": "C0",
        "placement": "generator_rank_frozen_replica",
        "scope": "step0_fake_score_only",
        "teacher_velocity_transport": "cuda_p2p_float32",
        "critic_training_placement": "score_ranks_only",
        "source_trainable_sha_matches_score_c0": True,
        "logical_calls_preserved": True,
    }
    assert metadata == base["metadata"]
    base["id"] = candidate["id"]
    base["metadata"] = dict(candidate["metadata"])
    assert candidate == base


def test_split_dmd_replays_identical_flow_inputs_and_gradient_combine() -> None:
    torch = pytest.importorskip("torch")
    row = _trace_row(0, 0)
    row = replace(
        row,
        seeds={
            **row.seeds,
            "generator_dmd_t": 12345,
            "generator_dmd_noise": 67890,
        },
    )
    x0 = torch.randn(1, 2, 3, 4, 5)
    score_flow = _dmd_flow_inputs(torch, row=row, x0=x0)
    generator_flow = _dmd_flow_inputs(torch, row=row, x0=x0)
    assert torch.equal(score_flow.t, generator_flow.t)
    assert torch.equal(score_flow.z_t, generator_flow.z_t)
    v_real = torch.randn_like(x0)
    v_fake = torch.randn_like(x0)
    actual = _combine_dmd_gradient(
        torch,
        x0=x0,
        flow=generator_flow,
        v_real=v_real,
        v_fake=v_fake,
    )
    x0_real = (score_flow.z_t - score_flow.t_view * v_real).float()
    x0_fake = (score_flow.z_t - score_flow.t_view * v_fake).float()
    normalizer = (x0.float() - x0_real).abs().mean().clamp_min(1e-4)
    expected = torch.nan_to_num((x0_fake - x0_real) / normalizer)
    assert torch.equal(actual, expected)


def test_split_dmd_fused_adam_reuses_only_validated_capacity_recovery() -> None:
    reference = load_plan(
        PIPELINE_RUNAHEAD_CRITIC_B2_LOCAL_CONTEXT_ASYNC32_TRACE_BALANCED_FULL_CONTEXT_LORA_AUTOCAST_INPUT_ELISION_SPLIT_DMD_SHADOW
    ).plan
    candidate = load_plan(
        PIPELINE_RUNAHEAD_CRITIC_B2_LOCAL_CONTEXT_ASYNC32_TRACE_BALANCED_FULL_CONTEXT_LORA_AUTOCAST_INPUT_ELISION_SPLIT_DMD_SHADOW_FUSED_ADAM
    ).plan
    LiveWanSFDMDAdapter().validate(candidate)
    assert reference.workload.to_dict() == candidate.workload.to_dict()
    assert reference.hardware.to_dict() == candidate.hardware.to_dict()
    assert reference.rank_groups == candidate.rank_groups
    assert reference.lifecycle.to_dict() == candidate.lifecycle.to_dict()
    assert candidate.metadata["split_dmd_shadow"] == reference.metadata["split_dmd_shadow"]
    assert candidate.metadata["critic_lora_kernel"] == reference.metadata["critic_lora_kernel"]
    assert candidate.metadata["generator_optimizer_impl"] == "fused_adamw"
    assert candidate.metadata["capacity_recovery"] == (
        "reuse_validated_foreach_sqrt_transient_recovery"
    )
    assert candidate.training_backends["generator"].options == {
        **reference.training_backends["generator"].options,
        "optimizer_impl": "fused_adamw",
    }


def test_score_modulation_plan_changes_only_exact_kernel_metadata() -> None:
    base = load_plan(
        PIPELINE_RUNAHEAD_CRITIC_B2_LOCAL_CONTEXT_ASYNC32_TRACE_BALANCED_FULL_CONTEXT_LORA_SCALE1
    ).plan.to_dict()
    candidate_plan = load_plan(
        PIPELINE_RUNAHEAD_CRITIC_B2_LOCAL_CONTEXT_ASYNC32_TRACE_BALANCED_FULL_CONTEXT_LORA_SCALE1_MODULATION
    ).plan
    LiveWanSFDMDAdapter().validate(candidate_plan)
    candidate = candidate_plan.to_dict()
    metadata = dict(candidate["metadata"])
    assert metadata.pop("score_modulation_kernel") == {
        "enabled": True,
        "name": "score_bf16_modulation_fusion_v1",
        "scope": "score_ranks_only_critic_and_teacher",
        "source_modulate_sha256": (
            "f244152360d40d6745e28bed4add081604312d04fe2f342842a32432f4bd20ad"
        ),
        "math_change": "fuse_exact_bf16_to_fp32_modulation_pointwise_chain",
        "normalized_shift_scale_dtype": "bfloat16",
        "output_dtype": "float32",
        "per_frame": False,
        "triton_block_size": 256,
    }
    assert metadata == base["metadata"]
    base["id"] = candidate["id"]
    base["metadata"] = dict(candidate["metadata"])
    assert candidate == base


def test_score_rope_plan_changes_only_exact_kernel_metadata() -> None:
    reference = load_plan(
        PIPELINE_RUNAHEAD_CRITIC_B2_LOCAL_CONTEXT_ASYNC32_TRACE_BALANCED_FULL_CONTEXT_LORA_AUTOCAST_INPUT_ELISION_SPLIT_DMD_SHADOW_FUSED_ADAM
    ).plan
    candidate_plan = load_plan(
        PIPELINE_RUNAHEAD_CRITIC_B2_LOCAL_CONTEXT_ASYNC32_TRACE_BALANCED_FULL_CONTEXT_LORA_AUTOCAST_INPUT_ELISION_SPLIT_DMD_SHADOW_FUSED_ADAM_SCORE_ROPE
    ).plan
    LiveWanSFDMDAdapter().validate(candidate_plan)
    reference_dict = reference.to_dict()
    candidate = candidate_plan.to_dict()
    metadata = dict(candidate["metadata"])
    assert metadata.pop("score_rope_kernel") == {
        "enabled": True,
        "name": "score_bf16_rope_fusion_v1",
        "scope": "score_ranks_only_critic_and_teacher",
        "source_apply_rope_sha256": (
            "8ab587e9f15cc969659609ec2f9cc79f4d4cf6fba468ae8c54267dc949224482"
        ),
        "math_change": "fuse_exact_fp32_complex_rope_and_immediate_bf16_conversion",
        "input_output_dtype": "bfloat16",
        "table_dtype": "complex64",
        "fp32_rounding_order": "mul_then_first_product_fma_rn",
        "triton_block_size": 512,
    }
    assert metadata == reference_dict["metadata"]
    for key in ("workload", "hardware", "rank_groups", "lifecycle", "training_backends"):
        assert candidate[key] == reference_dict[key]


def test_score_rope_kernel_is_source_pinned_and_fail_closed() -> None:
    source = (
        ROOT / "src/distillschedule/runtime/livewan_rope_kernel.py"
    ).read_text()
    assert "EXPECTED_APPLY_ROPE_SHA256" in source
    assert "8ab587e9f15cc969659609ec2f9cc79f4d4cf6fba468ae8c54267dc949224482" in source
    assert "value.dtype != torch.bfloat16" in source
    assert "table.dtype != torch.complex64" in source
    assert "table.requires_grad" in source
    assert "contract_failure_count" in source
    assert "fma.rn.f32" in source
    assert "There is no eager" in source


def test_score_flash_attn2_plan_changes_only_backend_metadata() -> None:
    reference = load_plan(
        PIPELINE_RUNAHEAD_CRITIC_B2_LOCAL_CONTEXT_ASYNC32_TRACE_BALANCED_FULL_CONTEXT_LORA_AUTOCAST_INPUT_ELISION_SPLIT_DMD_SHADOW_FUSED_ADAM_SCORE_ROPE
    ).plan.to_dict()
    candidate_plan = load_plan(
        PIPELINE_RUNAHEAD_CRITIC_B2_LOCAL_CONTEXT_ASYNC32_TRACE_BALANCED_FULL_CONTEXT_LORA_AUTOCAST_INPUT_ELISION_SPLIT_DMD_SHADOW_FUSED_ADAM_SCORE_ROPE_FLASH_ATTN2
    ).plan
    LiveWanSFDMDAdapter().validate(candidate_plan)
    candidate = candidate_plan.to_dict()
    metadata = dict(candidate["metadata"])
    assert metadata.pop("score_attention_backend") == {
        "enabled": True,
        "name": "score_flash_attn2_unmasked_bf16_v1",
        "scope": "score_ranks_only_critic_and_teacher",
        "source_flash_attention_sha256": (
            "c59c125fd92234d2ddec63f3ae3bd2a2479862faba44fa955fdedad2b8203a59"
        ),
        "artifact_metadata_sha256": (
            "7749dd68b81a544c1973d68c30be5aa316ee1a718ae0228e0368c2cd04365516"
        ),
        "artifact_package_sha256": (
            "0c560f96b857c188c4a8297ff27d1299619957c3459a5e433e132329fbfdf066"
        ),
        "artifact_extension_sha256": (
            "c2acd487c5d1d77b7f6721eabc0fee4f8c5c8f9e682aadf43bfa99b394eecba7"
        ),
        "math_change": (
            "replace_cudnn_sdpa_with_pinned_flash_attn2_under_fixed_parity_gate"
        ),
        "mask_contract": "full_length_unmasked_only",
        "input_output_dtype": "bfloat16",
        "head_dimension": 128,
    }
    assert metadata == reference["metadata"]
    for key in ("workload", "hardware", "rank_groups", "lifecycle", "training_backends"):
        assert candidate[key] == reference[key]


def test_score_flash_attn2_backend_is_source_and_artifact_pinned() -> None:
    source = (
        ROOT / "src/distillschedule/runtime/livewan_attention_backend.py"
    ).read_text()
    for value in (
        "EXPECTED_FLASH_ATTENTION_SHA256",
        "c59c125fd92234d2ddec63f3ae3bd2a2479862faba44fa955fdedad2b8203a59",
        "7749dd68b81a544c1973d68c30be5aa316ee1a718ae0228e0368c2cd04365516",
        "0c560f96b857c188c4a8297ff27d1299619957c3459a5e433e132329fbfdf066",
        "c2acd487c5d1d77b7f6721eabc0fee4f8c5c8f9e682aadf43bfa99b394eecba7",
        "masked or variable-length attention is not registered",
        "q.shape[-1] != 128",
        "contract_failure_count",
    ):
        assert value in source


def test_generator_cross_kv_plan_changes_only_cache_metadata() -> None:
    reference = load_plan(
        PIPELINE_RUNAHEAD_CRITIC_B2_LOCAL_CONTEXT_ASYNC32_TRACE_BALANCED_FULL_CONTEXT_LORA_AUTOCAST_INPUT_ELISION_SPLIT_DMD_SHADOW_FUSED_ADAM_SCORE_ROPE_FLASH_ATTN2
    ).plan.to_dict()
    candidate_plan = load_plan(
        PIPELINE_RUNAHEAD_CRITIC_B2_LOCAL_CONTEXT_ASYNC32_TRACE_BALANCED_FULL_CONTEXT_LORA_AUTOCAST_INPUT_ELISION_SPLIT_DMD_SHADOW_FUSED_ADAM_SCORE_ROPE_FLASH_ATTN2_GENERATOR_CROSS_KV
    ).plan
    LiveWanSFDMDAdapter().validate(candidate_plan)
    candidate = candidate_plan.to_dict()
    metadata = dict(candidate["metadata"])
    assert metadata.pop("generator_cross_attention_kv_cache") == {
        "enabled": True,
        "name": "generator_nograd_text_kv_cache_v1",
        "scope": "generator_ranks_streamer_nograd_only",
        "source_set_text_sha256": (
            "def2452ea9528abb9af2aa166d78afdf9b8b7c5daa52c0fc87bfd63bc9832386"
        ),
        "source_cross_forward_sha256": (
            "0df9365afb37bb8a876d8c8659ca95d1caaffb89bf3e64918b782eb86b0a8253"
        ),
        "source_flash_attention_sha256": (
            "c59c125fd92234d2ddec63f3ae3bd2a2479862faba44fa955fdedad2b8203a59"
        ),
        "math_change": (
            "reuse_exact_cross_kv_projection_within_fixed_weight_no_grad_rollout"
        ),
        "context_contract": "b1_full_512_token_fp32_projected_context",
        "grad_contract": "grad_enabled_checkpoint_forward_and_replay_bypass_cache",
        "block_count": 30,
        "cache_bytes_per_streamer": 141_557_760,
        "max_streamers_per_generator_rank": 2,
    }
    assert metadata == reference["metadata"]
    for key in ("workload", "hardware", "rank_groups", "lifecycle", "training_backends"):
        assert candidate[key] == reference[key]


def test_generator_cross_kv_cache_is_source_pinned_and_grad_bypassed() -> None:
    source = (
        ROOT / "src/distillschedule/runtime/livewan_cross_kv_cache.py"
    ).read_text()
    for value in (
        "EXPECTED_SET_TEXT_SHA256",
        "EXPECTED_CROSS_FORWARD_SHA256",
        "EXPECTED_FLASH_ATTENTION_SHA256",
        "if torch.is_grad_enabled():",
        "return original_forward(self, x, context, context_lens)",
        "_parameter_versions(self) != entry",
        "total_bytes != 141_557_760",
        "contract_failure_count",
    ):
        assert value in source


def test_host_runahead_plan_changes_only_transport_metadata() -> None:
    reference = load_plan(
        PIPELINE_RUNAHEAD_CRITIC_B2_LOCAL_CONTEXT_ASYNC32_TRACE_BALANCED_FULL_CONTEXT_LORA_AUTOCAST_INPUT_ELISION_SPLIT_DMD_SHADOW_FUSED_ADAM_SCORE_ROPE_FLASH_ATTN2_GENERATOR_CROSS_KV
    ).plan.to_dict()
    candidate_plan = load_plan(
        PIPELINE_RUNAHEAD_CRITIC_B2_LOCAL_CONTEXT_ASYNC32_TRACE_BALANCED_FULL_CONTEXT_LORA_AUTOCAST_INPUT_ELISION_SPLIT_DMD_SHADOW_FUSED_ADAM_SCORE_ROPE_FLASH_ATTN2_GENERATOR_CROSS_KV_HOST_RUNAHEAD
    ).plan
    LiveWanSFDMDAdapter().validate(candidate_plan)
    candidate = candidate_plan.to_dict()
    metadata = dict(candidate["metadata"])
    assert metadata.pop("generator_runahead_transport") == {
        "enabled": True,
        "name": "pinned_cpu_gloo_latent_v1",
        "scope": "critic_only_steps_1_4_latent_only",
        "logical_steps": [1, 2, 3, 4],
        "waves_per_step": 2,
        "payload_dtype": "float32",
        "payload_shape": [1, 16, 21, 46, 80],
        "pinned_host": True,
        "background_sender_threads_per_rank": 1,
        "preposted_receives": True,
        "step0_transport": "cuda_nccl_p2p_unchanged",
        "context_transport": "trace_local_replay_unchanged",
        "calls_preserved": True,
    }
    assert metadata == reference["metadata"]
    for key in ("workload", "hardware", "rank_groups", "lifecycle", "training_backends"):
        assert candidate[key] == reference[key]


def test_host_runahead_tags_and_worker_path_are_fail_closed() -> None:
    from distillschedule.runtime.livewan_host_runahead import runahead_tag

    assert [
        runahead_tag(step, wave)
        for step in range(1, 5)
        for wave in range(2)
    ] == list(range(27102, 27110))
    with pytest.raises(ValueError):
        runahead_tag(0, 0)
    with pytest.raises(ValueError):
        runahead_tag(1, 2)
    source = (
        ROOT / "src/distillschedule/runtime/livewan_host_runahead.py"
    ).read_text()
    for value in (
        "ThreadPoolExecutor",
        "pin_memory=True",
        "event.synchronize()",
        "preposted run-ahead receive is missing",
        "Generator run-ahead send inventory is incomplete",
        "step0_transport",
    ):
        assert value in source


def test_generator_flash_attn2_plan_changes_only_backend_metadata() -> None:
    reference = load_plan(
        PIPELINE_RUNAHEAD_CRITIC_B2_LOCAL_CONTEXT_ASYNC32_TRACE_BALANCED_FULL_CONTEXT_LORA_AUTOCAST_INPUT_ELISION_SPLIT_DMD_SHADOW_FUSED_ADAM_SCORE_ROPE_FLASH_ATTN2_GENERATOR_CROSS_KV_HOST_RUNAHEAD
    ).plan.to_dict()
    candidate_plan = load_plan(PIPELINE_C20_GENERATOR_FLASH_ATTN2).plan
    LiveWanSFDMDAdapter().validate(candidate_plan)
    candidate = candidate_plan.to_dict()
    metadata = dict(candidate["metadata"])
    assert metadata.pop("generator_attention_backend") == {
        "enabled": True,
        "name": "generator_flash_attn2_unmasked_exact_cast_v1",
        "scope": "generator_ranks_student_and_frozen_critic_shadow",
        "source_flash_attention_sha256": (
            "c59c125fd92234d2ddec63f3ae3bd2a2479862faba44fa955fdedad2b8203a59"
        ),
        "artifact_metadata_sha256": (
            "7749dd68b81a544c1973d68c30be5aa316ee1a718ae0228e0368c2cd04365516"
        ),
        "artifact_package_sha256": (
            "0c560f96b857c188c4a8297ff27d1299619957c3459a5e433e132329fbfdf066"
        ),
        "artifact_extension_sha256": (
            "c2acd487c5d1d77b7f6721eabc0fee4f8c5c8f9e682aadf43bfa99b394eecba7"
        ),
        "math_change": (
            "replace_cudnn_sdpa_with_pinned_flash_attn2_under_fixed_parity_gate"
        ),
        "mask_contract": "full_length_unmasked_only",
        "raw_input_dtype_contract": [
            "bfloat16/bfloat16/bfloat16",
            "float32/float32/bfloat16",
        ],
        "kernel_compute_dtype": "bfloat16",
        "output_dtype": "preserve_query_dtype",
        "head_dimension": 128,
    }
    assert metadata == reference["metadata"]
    for key in ("workload", "hardware", "rank_groups", "lifecycle", "training_backends"):
        assert candidate[key] == reference[key]


def test_generator_flash_attn2_backend_preserves_pinned_helper_cast_contract() -> None:
    source = (
        ROOT / "src/distillschedule/runtime/livewan_attention_backend.py"
    ).read_text()
    for value in (
        "install_generator_flash_attn2",
        "generator_flash_attn2_unmasked_exact_cast_v1",
        "(torch.float32, torch.float32, torch.bfloat16)",
        "output_dtype = q.dtype",
        "q_kernel = q if q.dtype == torch.bfloat16 else q.to(dtype)",
        "k_kernel = k if k.dtype == torch.bfloat16 else k.to(dtype)",
        "output = output.to(output_dtype)",
        "raw_dtype_histogram",
        "input_cast_count",
        "output_cast_count",
    ):
        assert value in source


def test_score_modulation_kernel_is_source_pinned_and_fail_closed() -> None:
    source = (
        ROOT / "src/distillschedule/runtime/livewan_modulation_kernel.py"
    ).read_text()
    assert "EXPECTED_MODULATE_SHA256" in source
    assert "f244152360d40d6745e28bed4add081604312d04fe2f342842a32432f4bd20ad" in source
    assert "normalized.dtype != torch.bfloat16" in source
    assert "shift.requires_grad" in source
    assert "scale.requires_grad" in source
    assert "if per_frame:" in source
    assert "raise RuntimeError" in source


def test_lora_scale_one_runtime_is_bitwise_exact() -> None:
    import importlib.util
    import torch
    import types

    # Load the pinned, self-contained LoRA module without executing
    # ``wanstreamer.__init__``.  The latter imports the full Wan runtime, which
    # is intentionally absent from the source-only unit-test environment.
    package = types.ModuleType("wanstreamer")
    package.__path__ = [str(ROOT / "third_party/livewan/wanstreamer")]
    sys.modules["wanstreamer"] = package
    spec = importlib.util.spec_from_file_location(
        "wanstreamer.lora", ROOT / "third_party/livewan/wanstreamer/lora.py"
    )
    assert spec is not None and spec.loader is not None
    lora_module = importlib.util.module_from_spec(spec)
    sys.modules["wanstreamer.lora"] = lora_module
    spec.loader.exec_module(lora_module)
    LoRALinear = lora_module.LoRALinear
    from distillschedule.runtime.livewan_lora_kernel import (
        install_scale_one_lora_epilogue,
        uninstall_scale_one_lora_epilogue,
    )

    torch.manual_seed(20260831)
    original = LoRALinear(
        torch.nn.Linear(16, 24, bias=False).to(torch.bfloat16), rank=4
    )
    reference = LoRALinear(
        torch.nn.Linear(16, 24, bias=False).to(torch.bfloat16), rank=4
    )
    reference.load_state_dict(original.state_dict())
    x0 = torch.randn(7, 16, dtype=torch.bfloat16)
    grad = torch.randn(7, 24, dtype=torch.bfloat16)

    x_reference = x0.clone().requires_grad_(True)
    output_reference = reference(x_reference)
    output_reference.backward(grad)
    reference_gradients = (
        x_reference.grad.clone(),
        reference.a.grad.clone(),
        reference.b.grad.clone(),
    )
    try:
        receipt = install_scale_one_lora_epilogue(torch)
        x_candidate = x0.clone().requires_grad_(True)
        output_candidate = original(x_candidate)
        output_candidate.backward(grad)
        assert receipt["class_forward_patched"] is True
        assert torch.equal(output_reference, output_candidate)
        assert torch.equal(reference_gradients[0], x_candidate.grad)
        assert torch.equal(reference_gradients[1], original.a.grad)
        assert torch.equal(reference_gradients[2], original.b.grad)
    finally:
        uninstall_scale_one_lora_epilogue()
        sys.modules.pop("wanstreamer.lora", None)
        sys.modules.pop("wanstreamer", None)


def test_lora_autocast_input_elision_fails_closed_without_cuda_bf16_autocast() -> None:
    import importlib.util
    import torch
    import types

    package = types.ModuleType("wanstreamer")
    package.__path__ = [str(ROOT / "third_party/livewan/wanstreamer")]
    sys.modules["wanstreamer"] = package
    spec = importlib.util.spec_from_file_location(
        "wanstreamer.lora", ROOT / "third_party/livewan/wanstreamer/lora.py"
    )
    assert spec is not None and spec.loader is not None
    lora_module = importlib.util.module_from_spec(spec)
    sys.modules["wanstreamer.lora"] = lora_module
    spec.loader.exec_module(lora_module)
    from distillschedule.runtime.livewan_lora_kernel import (
        install_scale_one_lora_epilogue,
        uninstall_scale_one_lora_epilogue,
    )

    module = lora_module.LoRALinear(
        torch.nn.Linear(16, 24, bias=False).to(torch.bfloat16), rank=4
    )
    try:
        receipt = install_scale_one_lora_epilogue(
            torch, autocast_input_elision=True
        )
        with pytest.raises(RuntimeError, match="autocast input-elision contract"):
            module(torch.randn(7, 16, dtype=torch.bfloat16))
        assert receipt["autocast_input_elision"] is True
        assert receipt["autocast_input_elision_invocation_count"] == 1
        assert receipt["autocast_contract_failure_count"] == 1
    finally:
        uninstall_scale_one_lora_epilogue()
        sys.modules.pop("wanstreamer.lora", None)
        sys.modules.pop("wanstreamer", None)


def test_full_context_cfg_preserves_pinned_b1_timestep_broadcast() -> None:
    import torch

    class FakeDMD:
        calls: list[dict[str, object]] = []

        @classmethod
        def bidirectional_velocity(
            cls, model, z, t, rope, context, context_lens, dtype, time_scale
        ):
            cls.calls.append(
                {
                    "z_shape": tuple(z.shape),
                    "t_shape": tuple(t.shape),
                    "context_shape": tuple(context.shape),
                    "context_lens": context_lens,
                }
            )
            return torch.tensor([1.0, 3.0], dtype=torch.float32).reshape(2, 1)

    result = _cfg_velocity_full_context(
        torch,
        FakeDMD,
        model=object(),
        z=torch.zeros((1, 2), dtype=torch.bfloat16),
        t=torch.tensor([0.5]),
        rope=object(),
        context_positive=torch.zeros((1, 4, 3)),
        context_negative=torch.ones((1, 4, 3)),
        dtype=torch.bfloat16,
        guidance=5.0,
        time_scale=1000.0,
    )
    assert torch.equal(result, torch.tensor([[-7.0]]))
    assert FakeDMD.calls == [
        {
            "z_shape": (2, 2),
            "t_shape": (1,),
            "context_shape": (2, 4, 3),
            "context_lens": None,
        }
    ]


def test_async_reducer_bucket_plan_is_reverse_deterministic() -> None:
    from distillschedule.runtime.livewan_grad_reducer import bucket_named_parameters

    class Parameter:
        def __init__(self, elements: int) -> None:
            self.elements = elements

        def numel(self) -> int:
            return self.elements

        def element_size(self) -> int:
            return 4

    rows = [(f"p{index}", Parameter(elements)) for index, elements in enumerate((2, 3, 4, 5))]
    buckets = bucket_named_parameters(rows, bucket_bytes=24)
    assert [[name for name, _parameter in bucket] for bucket in buckets] == [
        ["p3"],
        ["p2"],
        ["p1", "p0"],
    ]
