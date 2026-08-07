"""Compile DMD algorithm semantics into a model-independent logical graph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from distillschedule.core.enums import GradientMode, OperationKind, StepKind
from distillschedule.core.graph import Dependency, ExecutionGraph, Operation
from distillschedule.core.plan import ExecutionPlan

from .schedule import DMDStepIntent, DMDUpdateSchedule


@dataclass(frozen=True)
class CompiledDMDStep:
    intent: DMDStepIntent
    operation_ids: tuple[str, ...]


@dataclass(frozen=True)
class CompiledDMDProtocol:
    graph: ExecutionGraph
    steps: tuple[CompiledDMDStep, ...]
    final_versions: Mapping[str, int]


class DMDProtocolCompiler:
    """Build a conservative logical order without embedding model code.

    The compiler captures the contracts that must survive backend changes:
    shared Score modes, Generator/Score optimizer ownership, post-update rollout,
    version barriers, and EMA ordering. Model-specific nested calls remain in an
    adapter rather than being flattened into the core schema.
    """

    def compile(self, plan: ExecutionPlan) -> CompiledDMDProtocol:
        schedule = DMDUpdateSchedule(plan.workload.schedule, plan.workload.branches)
        intents = schedule.compile_range(plan.evidence.start_step, plan.evidence.steps)
        initial = plan.evidence.metadata.get("initial_versions", {})
        generator_version = int(initial.get("generator", 0))
        critic_version = int(initial.get("critic", 0))
        ema_version = int(initial.get("ema", 0))

        operations: list[Operation] = []
        dependencies: list[Dependency] = []
        compiled_steps: list[CompiledDMDStep] = []
        previous_step_tail: str | None = None

        def append(
            step: int,
            suffix: str,
            *,
            role: str,
            kind: OperationKind,
            grad_mode: GradientMode,
            physical_state: str | None,
            adapter_mode: str | None = None,
            required_versions: Mapping[str, int] | None = None,
            produced_versions: Mapping[str, int] | None = None,
            metadata: Mapping[str, Any] | None = None,
        ) -> str:
            operation_id = f"step{step}.{suffix}"
            operations.append(
                Operation(
                    operation_id=operation_id,
                    logical_step=step,
                    role=role,
                    kind=kind,
                    grad_mode=grad_mode,
                    physical_state=physical_state,
                    adapter_mode=adapter_mode,
                    required_versions=dict(required_versions or {}),
                    produced_versions=dict(produced_versions or {}),
                    metadata=dict(metadata or {}),
                )
            )
            return operation_id

        for intent in intents:
            step = intent.logical_step
            ids: list[str] = []
            if intent.generator_update:
                ids.append(
                    append(
                        step,
                        "generator_rollout",
                        role="generator",
                        kind=OperationKind.GENERATOR_ROLLOUT,
                        grad_mode=GradientMode.GRAD,
                        physical_state="generator",
                        adapter_mode="generator",
                        required_versions={"generator": generator_version, "critic": critic_version},
                        metadata={"nested_model_calls_owned_by_adapter": True},
                    )
                )
                if plan.workload.branches.dmd:
                    ids.append(
                        append(
                            step,
                            "score_fake",
                            role="score",
                            kind=OperationKind.SCORE_FAKE,
                            grad_mode=GradientMode.NO_GRAD,
                            physical_state="score",
                            adapter_mode="critic",
                            required_versions={"critic": critic_version},
                        )
                    )
                    ids.append(
                        append(
                            step,
                            "score_real_conditional",
                            role="score",
                            kind=OperationKind.SCORE_REAL_CONDITIONAL,
                            grad_mode=GradientMode.NO_GRAD,
                            physical_state="score",
                            adapter_mode="base",
                            required_versions={"critic": critic_version},
                        )
                    )
                    if plan.workload.branches.classifier_free_guidance:
                        ids.append(
                            append(
                                step,
                                "score_real_unconditional",
                                role="score",
                                kind=OperationKind.SCORE_REAL_UNCONDITIONAL,
                                grad_mode=GradientMode.NO_GRAD,
                                physical_state="score",
                                adapter_mode="base",
                                required_versions={"critic": critic_version},
                            )
                        )
                    ids.append(
                        append(
                            step,
                            "dmd_gradient",
                            role="boundary",
                            kind=OperationKind.DMD_GRADIENT,
                            grad_mode=GradientMode.EXTERNAL_GRAD,
                            physical_state=None,
                            required_versions={"generator": generator_version, "critic": critic_version},
                        )
                    )
                if plan.workload.branches.vae_decode:
                    ids.append(
                        append(
                            step,
                            "vae_decode",
                            role="vae",
                            kind=OperationKind.VAE_DECODE,
                            grad_mode=GradientMode.GRAD,
                            physical_state="vae",
                            required_versions={"generator": generator_version},
                        )
                    )
                if plan.workload.branches.gan:
                    ids.append(
                        append(
                            step,
                            "gan_loss",
                            role="score",
                            kind=OperationKind.GAN_LOSS,
                            grad_mode=GradientMode.GRAD,
                            physical_state="score",
                            adapter_mode="critic",
                            required_versions={"critic": critic_version},
                        )
                    )
                if plan.workload.branches.reward:
                    ids.append(
                        append(
                            step,
                            "reward_loss",
                            role="reward",
                            kind=OperationKind.REWARD_LOSS,
                            grad_mode=GradientMode.EXTERNAL_GRAD,
                            physical_state="reward",
                            required_versions={"generator": generator_version},
                        )
                    )
                ids.append(
                    append(
                        step,
                        "generator_backward",
                        role="generator",
                        kind=OperationKind.GENERATOR_BACKWARD,
                        grad_mode=GradientMode.EXTERNAL_GRAD,
                        physical_state="generator",
                        adapter_mode="generator",
                        required_versions={"generator": generator_version},
                    )
                )
                generator_version += 1
                ids.append(
                    append(
                        step,
                        "generator_update",
                        role="generator",
                        kind=OperationKind.GENERATOR_UPDATE,
                        grad_mode=GradientMode.GRAD,
                        physical_state="generator",
                        adapter_mode="generator",
                        produced_versions={"generator": generator_version},
                    )
                )
                if intent.ema_update and not plan.workload.schedule.ema_after_critic:
                    ema_version += 1
                    ids.append(
                        append(
                            step,
                            "ema_update",
                            role="ema",
                            kind=OperationKind.EMA_UPDATE,
                            grad_mode=GradientMode.NO_GRAD,
                            physical_state="ema",
                            required_versions={"generator": generator_version},
                            produced_versions={"ema": ema_version},
                            metadata={"source_generator_version": generator_version},
                        )
                    )

            if intent.critic_update:
                ids.append(
                    append(
                        step,
                        "critic_rollout",
                        role="rollout",
                        kind=OperationKind.CRITIC_ROLLOUT,
                        grad_mode=GradientMode.NO_GRAD,
                        physical_state="generator",
                        adapter_mode="generator",
                        required_versions={"generator": generator_version},
                        metadata={
                            "rollout_engine": plan.rollout.engine.value,
                            "max_version_staleness": plan.rollout.max_version_staleness,
                        },
                    )
                )
                ids.append(
                    append(
                        step,
                        "critic_forward",
                        role="score",
                        kind=OperationKind.CRITIC_FORWARD,
                        grad_mode=GradientMode.GRAD,
                        physical_state="score",
                        adapter_mode="critic",
                        required_versions={"critic": critic_version},
                    )
                )
                ids.append(
                    append(
                        step,
                        "critic_backward",
                        role="score",
                        kind=OperationKind.CRITIC_BACKWARD,
                        grad_mode=GradientMode.GRAD,
                        physical_state="score",
                        adapter_mode="critic",
                        required_versions={"critic": critic_version},
                    )
                )
                critic_version += 1
                ids.append(
                    append(
                        step,
                        "critic_update",
                        role="score",
                        kind=OperationKind.CRITIC_UPDATE,
                        grad_mode=GradientMode.GRAD,
                        physical_state="score",
                        adapter_mode="critic",
                        produced_versions={"critic": critic_version},
                    )
                )

            if intent.ema_update and (plan.workload.schedule.ema_after_critic or not intent.generator_update):
                ema_version += 1
                ids.append(
                    append(
                        step,
                        "ema_update",
                        role="ema",
                        kind=OperationKind.EMA_UPDATE,
                        grad_mode=GradientMode.NO_GRAD,
                        physical_state="ema",
                        required_versions={"generator": generator_version},
                        produced_versions={"ema": ema_version},
                        metadata={"source_generator_version": generator_version},
                    )
                )

            if not ids:
                ids.append(
                    append(
                        step,
                        "barrier",
                        role="runtime",
                        kind=OperationKind.BARRIER,
                        grad_mode=GradientMode.NO_GRAD,
                        physical_state=None,
                        metadata={"step_kind": StepKind.IDLE.value},
                    )
                )

            for source, target in zip(ids, ids[1:]):
                dependencies.append(Dependency(source=source, target=target))
            if previous_step_tail is not None:
                dependencies.append(Dependency(source=previous_step_tail, target=ids[0], kind="version_barrier"))
            previous_step_tail = ids[-1]
            compiled_steps.append(CompiledDMDStep(intent=intent, operation_ids=tuple(ids)))

        graph = ExecutionGraph(
            operations=tuple(operations),
            dependencies=tuple(dependencies),
            metadata={
                "plan_id": plan.plan_id,
                "profile_key": plan.profile_key,
                "adapter": plan.workload.adapter,
            },
        )
        return CompiledDMDProtocol(
            graph=graph,
            steps=tuple(compiled_steps),
            final_versions={
                "generator": generator_version,
                "critic": critic_version,
                "ema": ema_version,
            },
        )
