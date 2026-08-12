"""Dependency DAG for the phase-aware GAN Full5 wavefront.

The graph encodes the legal overlap instead of relying on loop order:

* Critic compute/commit remains strictly version-serial.
* Generator production for the next logical batch does not wait for the
  previous Critic optimizer commit.
* producer buffer reuse depends on NCCL send completion;
* the consumer Q2 receive window provides backpressure at depth two; and
* the reverse handoff is part of the steady-state cycle boundary.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Mapping


class RolePhaseTaskKind(str, enum.Enum):
    PHASE_A_SCORE_FLUSH = "phase_a_score_flush"
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


class GanRolePhasePlanner:
    """Compile the first GAN-active 1000..1004 cycle into an overlap DAG."""

    @staticmethod
    def _key(step: int, suffix: str) -> str:
        return f"b:{step}:{suffix}"

    def compile_full5(self) -> tuple[RolePhaseTask, ...]:
        tasks: list[RolePhaseTask] = [
            RolePhaseTask(
                key="a:score_flush",
                kind=RolePhaseTaskKind.PHASE_A_SCORE_FLUSH,
                logical_step=1000,
                dependencies=(),
                resource="score8",
            ),
            RolePhaseTask(
                key="a:g_commit",
                kind=RolePhaseTaskKind.PHASE_A_GENERATOR_COMMIT,
                logical_step=1000,
                dependencies=("a:score_flush",),
                resource="generator8",
            ),
            RolePhaseTask(
                key="handoff:to_b",
                kind=RolePhaseTaskKind.HANDOFF_TO_B,
                logical_step=1000,
                dependencies=("a:g_commit",),
                resource="world16_reshard",
            ),
        ]

        for step in range(1000, 1005):
            produce_deps = ["handoff:to_b"]
            # Two producer payload buffers. Reusing buffer i%2 only requires the
            # corresponding send from i-2 to have completed; no optimizer ACK.
            if step >= 1002:
                produce_deps.append(self._key(step - 2, "send_complete"))

            receive_deps = ["handoff:to_b"]
            # Consumer Q2 holds at most current + one-batch-ahead. The receive
            # for i+2 is posted only after i commits and frees one consumer slot.
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
                # Critic step i must observe C_i committed by step i-1.
                consume_deps.append(self._key(step - 1, "commit"))
            tasks.append(
                RolePhaseTask(
                    key=self._key(step, "consume"),
                    kind=RolePhaseTaskKind.CRITIC_CONSUME,
                    logical_step=step,
                    dependencies=tuple(consume_deps),
                    resource="critic12",
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
                resource="world16_reshard",
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

        # Version legality: each Critic consume after 1000 must depend on the
        # immediately preceding commit, but producer work must not.
        for step in range(1001, 1005):
            previous_commit = self._key(step - 1, "commit")
            consume = by_key[self._key(step, "consume")]
            produce = by_key[self._key(step, "produce")]
            if previous_commit not in consume.dependencies:
                raise AssertionError("Critic version dependency was dropped")
            if previous_commit in produce.dependencies:
                raise AssertionError(
                    "producer regressed to commit-ACK serialization"
                )

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
