from __future__ import annotations

from pathlib import Path

from distillschedule.core.config import load_plan
from distillschedule.core.enums import StepKind
from distillschedule.workloads.dmd import DMDProtocolCompiler, DMDUpdateSchedule


ROOT = Path(__file__).resolve().parents[1]


def test_full5_schedule_is_one_full_plus_four_critic_only() -> None:
    plan = load_plan(ROOT / "configs/distillschedule/plans/helios_14b_384x640_full5.json").plan
    intents = DMDUpdateSchedule(plan.workload.schedule, plan.workload.branches).compile_range(10, 5)
    assert [intent.kind for intent in intents] == [
        StepKind.FULL,
        StepKind.CRITIC_ONLY,
        StepKind.CRITIC_ONLY,
        StepKind.CRITIC_ONLY,
        StepKind.CRITIC_ONLY,
    ]
    assert sum(intent.generator_update for intent in intents) == 1
    assert sum(intent.critic_update for intent in intents) == 5
    assert sum(intent.ema_update for intent in intents) == 5


def test_compiler_tracks_versions_and_shared_score_modes() -> None:
    plan = load_plan(ROOT / "configs/distillschedule/plans/helios_14b_384x640_full5.json").plan
    compiled = DMDProtocolCompiler().compile(plan)
    assert compiled.final_versions == {"generator": 3, "critic": 15, "ema": 15}
    first = compiled.steps[0].operation_ids
    assert first[0] == "step10.generator_rollout"
    assert "step10.score_fake" in first
    assert "step10.score_real_conditional" in first
    assert "step10.generator_update" in first
    assert "step10.critic_update" in first
    assert first[-1] == "step10.ema_update"
    graph = compiled.graph
    score_states = {
        operation.adapter_mode
        for operation in graph.operations
        if operation.logical_step == 10 and operation.physical_state == "score"
    }
    assert score_states == {"base", "critic"}
    assert graph.topological_order()[0] == "step10.generator_rollout"


def test_optional_dmd_branches_are_explicit_operations() -> None:
    from distillschedule.core.plan import ExecutionPlan

    base = load_plan(
        ROOT / "configs/distillschedule/plans/generic_autoregressive_dmd_vllm_critic_only.json"
    ).plan.to_dict()
    base["evidence"]["start_step"] = 10
    base["evidence"]["steps"] = 1
    base["workload"]["branches"].update(
        {"gan": True, "reward": True, "vae_decode": True}
    )
    plan = ExecutionPlan.from_mapping(base)
    compiled = DMDProtocolCompiler().compile(plan)
    kinds = [operation.kind.value for operation in compiled.graph.operations]
    assert "vae_decode" in kinds
    assert "gan_loss" in kinds
    assert "reward_loss" in kinds


def test_generic_schedule_can_place_ema_before_critic() -> None:
    from distillschedule.core.plan import ExecutionPlan

    base = load_plan(
        ROOT / "configs/distillschedule/plans/generic_autoregressive_dmd_vllm_critic_only.json"
    ).plan.to_dict()
    base["evidence"]["start_step"] = 10
    base["evidence"]["steps"] = 1
    base["workload"]["schedule"]["ema_after_critic"] = False
    plan = ExecutionPlan.from_mapping(base)
    step = DMDProtocolCompiler().compile(plan).steps[0]
    assert step.operation_ids.index("step10.ema_update") < step.operation_ids.index(
        "step10.critic_rollout"
    )
