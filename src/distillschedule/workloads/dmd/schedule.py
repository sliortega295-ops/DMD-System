"""Frequency-aware DMD step classification."""

from __future__ import annotations

from dataclasses import dataclass

from distillschedule.core.enums import StepKind
from distillschedule.core.specs import DMDBranchesSpec, DMDScheduleSpec


@dataclass(frozen=True)
class DMDStepIntent:
    logical_step: int
    kind: StepKind
    generator_update: bool
    critic_update: bool
    ema_update: bool


class DMDUpdateSchedule:
    def __init__(self, schedule: DMDScheduleSpec, branches: DMDBranchesSpec) -> None:
        self.schedule = schedule
        self.branches = branches

    def intent(self, logical_step: int) -> DMDStepIntent:
        if logical_step < 0:
            raise ValueError("logical_step must be non-negative")
        generator_update = logical_step % self.schedule.generator_update_every == 0
        critic_update = logical_step % self.schedule.critic_update_every == 0
        ema_update = (
            self.branches.ema
            and (
                self.schedule.ema_bootstrap_before_start
                or logical_step >= self.schedule.ema_start_step
            )
            and logical_step % self.schedule.ema_update_every == 0
        )
        if generator_update and critic_update:
            kind = StepKind.FULL
        elif critic_update:
            kind = StepKind.CRITIC_ONLY
        elif generator_update:
            kind = StepKind.GENERATOR_ONLY
        else:
            kind = StepKind.IDLE
        return DMDStepIntent(
            logical_step=logical_step,
            kind=kind,
            generator_update=generator_update,
            critic_update=critic_update,
            ema_update=ema_update,
        )

    def compile_range(self, start_step: int, steps: int) -> tuple[DMDStepIntent, ...]:
        if start_step < 0:
            raise ValueError("start_step must be non-negative")
        if steps <= 0:
            raise ValueError("steps must be positive")
        return tuple(self.intent(step) for step in range(start_step, start_step + steps))
