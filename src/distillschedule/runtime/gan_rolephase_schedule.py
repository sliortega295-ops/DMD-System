"""Dependency DAG for the phase-aware GAN Full5 wavefront.

The graph encodes the legal overlap instead of relying on loop order.
Phase A now contains the two real Generator/Score boundary waves rather than a
placeholder flush->commit edge. Phase B keeps Critic versions strictly serial
while allowing the no-grad Generator producer to stay one batch ahead.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Mapping


class RolePhaseTaskKind(str, enum.Enum):
    PHASE_A_SCORE_FLUSH = "phase_a_score_flush"
    PHASE_A_G_FORWARD = "phase_a_generator_forward"
    PHASE_A_LATENT_SEND_COMPLETE = "phase_a_latent_send_complete"
    PHASE_A_SCORE_BOUNDARY = "phase_a_score_boundary"
    PHASE_A_G_BACKWARD = "phase_a_generator_backward"
    PHASE_A_GENERATOR_COMMIT = "phase_a_generator_commit"
    HANDOFF_TO_B = "handoff_to_b"
    PRODUCE = "produce"
    RECEIVE_POST = "receive_post"
    SEND_COMPLETE = "send_complete"
    CRITIC_CONSUME = "critic_consume"
    CRITIC_COMMIT = "critic_commit"
    EMA_COMMIT = "ema_commit"
    HANDOFF_TO_A = "handoff_to_a"


@dataclass(frozen=True)
class RolePhaseTask:
    key: str
    kind: RolePhaseTaskKind
    logical_step: int | None
    dependencies: tuple[str, ...]
    resource: str
    micro_wave: int | None = None


class GanRolePhasePlanner:
    """Compile the first GAN-active 1000..1004 cycle into an overlap DAG."""

    @staticmethod
    def _key(step: int, suffix: str) -> str:
        return f"b:{step}:{suffix}"

    @staticmethod
    def _a(suffix: str, wave: int | None = None) -> str:
        return f"a:{suffix}" if wave is None else f"a:{suffix}:w{wave}"

    def _phase_a(self) -> list[RolePhaseTask]:
        return [
            RolePhaseTask(
                key=self._a("score_flush"),
                kind=RolePhaseTaskKind.PHASE_A_SCORE_FLUSH,
                logical_step=1000,
                dependencies=(),
                resource="score8",
            ),
            RolePhaseTask(
                key=self._a("g_forward", 0),
                kind=RolePhaseTaskKind.PHASE_A_G_FORWARD,
                logical_step=1000,
                dependencies=(),
                resource="generator8",
                micro_wave=0,
            ),
            RolePhaseTask(
                key=self._a("latent_send_complete", 0),
                kind=RolePhaseTaskKind.PHASE_A_LATENT_SEND_COMPLETE,
                logical_step=1000,
                dependencies=(self._a("g_forward", 0),),
                resource="phase_a_p2p",
                micro_wave=0,
            ),
            RolePhaseTask(
                key=self._a("score_boundary", 0),
                kind=RolePhaseTaskKind.PHASE_A_SCORE_BOUNDARY,
                logical_step=1000,
                dependencies=(
                    self._a("score_flush"),
                    self._a("latent_send_complete", 0),
                ),
                resource="score8",
                micro_wave=0,
            ),
            # Generator wave1 is released by wave0 send completion, not by the
            # wave0 Score reply. This is the intended Phase-A overlap.
            RolePhaseTask(
                key=self._a("g_forward", 1),
                kind=RolePhaseTaskKind.PHASE_A_G_FORWARD,
                logical_step=1000,
                dependencies=(self._a("latent_send_complete", 0),),
                resource="generator8",
                micro_wave=1,
            ),
            RolePhaseTask(
                key=self._a("latent_send_complete", 1),
                kind=RolePhaseTaskKind.PHASE_A_LATENT_SEND_COMPLETE,
                logical_step=1000,
                dependencies=(self._a("g_forward", 1),),
                resource="phase_a_p2p",
                micro_wave=1,
            ),
            RolePhaseTask(
                key=self._a("score_boundary", 1),
                kind=RolePhaseTaskKind.PHASE_A_SCORE_BOUNDARY,
                logical_step=1000,
                dependencies=(
                    self._a("score_boundary", 0),
                    self._a("latent_send_complete", 1),
                ),
                resource="score8",
                micro_wave=1,
            ),
            # The single Generator engine finishes forward1 before it consumes
            # the wave0 returned latent gradient.
            RolePhaseTask(
                key=self._a("g_backward", 0),
                kind=RolePhaseTaskKind.PHASE_A_G_BACKWARD,
                logical_step=1000,
                dependencies=(
                    self._a("score_boundary", 0),
                    self._a("g_forward", 1),
                ),
                resource="generator8",
                micro_wave=0,
            ),
            RolePhaseTask(
                key=self._a("g_backward", 1),
                kind=RolePhaseTaskKind.PHASE_A_G_BACKWARD,
                logical_step=1000,
                dependencies=(
                    self._a("score_boundary", 1),
                    self._a("g_backward", 0),
                ),
                resource="generator8",
                micro_wave=1,
            ),
            RolePhaseTask(
                key=self._a("g_commit"),
                kind=RolePhaseTaskKind.PHASE_A_GENERATOR_COMMIT,
                logical_step=1000,
                dependencies=(self._a("g_backward", 1),),
                resource="generator8_optimizer",
            ),
            RolePhaseTask(
                key="handoff:to_b",
                kind=RolePhaseTaskKind.HANDOFF_TO_B,
                logical_step=1000,
                dependencies=(
                    self._a("g_commit"),
                    self._a("score_boundary", 1),
                ),
                resource="world16_transition",
            ),
        ]

    def compile_full5(self) -> tuple[RolePhaseTask, ...]:
        tasks: list[RolePhaseTask] = self._phase_a()

        for step in range(1000, 1005):
            produce_deps = ["handoff:to_b"]
            if step >= 1002:
                produce_deps.append(self._key(step - 2, "send_complete"))

            receive_deps = ["handoff:to_b"]
            if step >= 1002:
                receive_deps.append(self._key(step - 2, "commit"))

            tasks.extend(
                [
                    RolePhaseTask(
                        key=self._key(step, "produce"),
                        kind=RolePhaseTaskKind.PRODUCE,
                        logical_step=step,
                        dependencies=tuple(produce_deps),
                        resource="generator4",
                    ),
                    RolePhaseTask(
                        key=self._key(step, "recv_post"),
                        kind=RolePhaseTaskKind.RECEIVE_POST,
                        logical_step=step,
                        dependencies=tuple(receive_deps),
                        resource="critic12_rx",
                    ),
                    RolePhaseTask(
                        key=self._key(step, "send_complete"),
                        kind=RolePhaseTaskKind.SEND_COMPLETE,
                        logical_step=step,
                        dependencies=(
                            self._key(step, "produce"),
                            self._key(step, "recv_post"),
                        ),
                        resource="world_p2p",
                    ),
                ]
            )

            consume_deps = [self._key(step, "send_complete")]
            if step > 1000:
                consume_deps.append(self._key(step - 1, "commit"))
            tasks.append(
                RolePhaseTask(
                    key=self._key(step, "consume"),
                    kind=RolePhaseTaskKind.CRITIC_CONSUME,
                    logical_step=step,
                    dependencies=tuple(consume_deps),
                    resource="critic12_single_physical_call",
                )
            )
            tasks.append(
                RolePhaseTask(
                    key=self._key(step, "commit"),
                    kind=RolePhaseTaskKind.CRITIC_COMMIT,
                    logical_step=step,
                    dependencies=(self._key(step, "consume"),),
                    resource="critic12_optimizer",
                )
            )
            if step == 1000:
                tasks.append(
                    RolePhaseTask(
                        key="b:1000:ema",
                        kind=RolePhaseTaskKind.EMA_COMMIT,
                        logical_step=1000,
                        dependencies=(self._key(1000, "commit"),),
                        resource="generator_ema",
                    )
                )

        tasks.append(
            RolePhaseTask(
                key="handoff:to_a",
                kind=RolePhaseTaskKind.HANDOFF_TO_A,
                logical_step=1005,
                dependencies=(self._key(1004, "commit"), "b:1000:ema"),
                resource="world16_transition",
            )
        )
        self.validate(tasks)
        return tuple(tasks)

    def validate(self, tasks: tuple[RolePhaseTask, ...] | list[RolePhaseTask]) -> None:
        by_key = {task.key: task for task in tasks}
        if len(by_key) != len(tasks):
            raise AssertionError("duplicate role-phase task key")
        for task in tasks:
            for dependency in task.dependencies:
                if dependency not in by_key:
                    raise AssertionError(
                        f"task {task.key} references missing dependency {dependency}"
                    )
        self.topological_order(tuple(tasks))

        g1 = by_key[self._a("g_forward", 1)]
        if self._a("latent_send_complete", 0) not in g1.dependencies:
            raise AssertionError("Generator wave1 lacks wave0 send-complete credit")
        if self._a("score_boundary", 0) in g1.dependencies:
            raise AssertionError("Phase-A regressed to send/wait/reply serialization")

        for step in range(1001, 1005):
            previous_commit = self._key(step - 1, "commit")
            consume = by_key[self._key(step, "consume")]
            produce = by_key[self._key(step, "produce")]
            if previous_commit not in consume.dependencies:
                raise AssertionError("Critic version dependency was dropped")
            if previous_commit in produce.dependencies:
                raise AssertionError("producer regressed to commit-ACK serialization")

        for step in range(1002, 1005):
            receive = by_key[self._key(step, "recv_post")]
            if self._key(step - 2, "commit") not in receive.dependencies:
                raise AssertionError("consumer Q2 backpressure dependency missing")
            produce = by_key[self._key(step, "produce")]
            if self._key(step - 2, "send_complete") not in produce.dependencies:
                raise AssertionError("producer double-buffer reuse dependency missing")

    @staticmethod
    def topological_order(tasks: tuple[RolePhaseTask, ...]) -> tuple[str, ...]:
        by_key = {task.key: task for task in tasks}
        remaining = {key: set(task.dependencies) for key, task in by_key.items()}
        order: list[str] = []
        while remaining:
            ready = sorted(key for key, deps in remaining.items() if not deps)
            if not ready:
                raise AssertionError("role-phase task graph contains a cycle")
            for key in ready:
                order.append(key)
                del remaining[key]
            for deps in remaining.values():
                deps.difference_update(ready)
        return tuple(order)

    def dependency_map(self) -> Mapping[str, tuple[str, ...]]:
        return {task.key: task.dependencies for task in self.compile_full5()}


__all__ = [
    "GanRolePhasePlanner",
    "RolePhaseTask",
    "RolePhaseTaskKind",
]
