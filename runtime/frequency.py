"""Frequency semantics used by the Stage-3 planner."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Stage3FrequencyStep:
    global_step: int
    generator_update: bool
    dmd_score: bool
    critic_update: bool = True
    ema_update: bool = False

    @property
    def kind(self) -> str:
        return "generator_and_critic" if self.generator_update else "critic_only"


def stage3_frequency_step(
    global_step: int,
    *,
    dfake_gen_update_ratio: int,
    ema_enabled: bool = True,
    ema_started: bool = True,
) -> Stage3FrequencyStep:
    """Return the default Helios Stage-3 branch at ``global_step``.

    The source always trains Critic. Generator/DMD and the post-start EMA transition are
    periodic under ``dfake_gen_update_ratio``.
    """
    if global_step < 0:
        raise ValueError("global_step must be non-negative")
    if dfake_gen_update_ratio <= 0:
        raise ValueError("dfake_gen_update_ratio must be positive")
    generator_update = global_step % dfake_gen_update_ratio == 0
    return Stage3FrequencyStep(
        global_step=global_step,
        generator_update=generator_update,
        dmd_score=generator_update,
        critic_update=True,
        ema_update=ema_enabled and (not ema_started or generator_update),
    )
