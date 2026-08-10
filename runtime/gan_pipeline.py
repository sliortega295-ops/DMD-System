"""Version-safe world16 GAN pipeline contracts.

This module encodes the proposed primary GAN pipeline without reimplementing
Helios model math.  The physical layout stays fixed at 8 Generator ranks plus
8 Score ranks for the whole five-step cycle.  Score ranks act as a frozen
DMD/GAN-G gradient service on Generator-update steps and as the trainable
Critic/GAN-D service on every logical step.

The key invariant is that the pipeline may move tensors and gradients, but it
may not move an optimizer/version boundary.  One logical batch is 16 samples.
With eight ranks per role this is represented as two micro-waves and exactly
one optimizer commit per logical step.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


LATENT_SHAPE = (1, 16, 9, 48, 80)
LATENT_DTYPE = "bfloat16"
GRADIENT_DTYPE = "bfloat16"


class GanPipelinePhase(str, enum.Enum):
    FLUSH = "flush"
    GENERATOR_FORWARD = "generator_forward_autograd"
    SCORE_BOUNDARY_GRAD = "score_boundary_grad"
    GENERATOR_BACKWARD = "generator_backward_accumulate"
    GENERATOR_COMMIT = "generator_optimizer_commit"
    CRITIC_ROLLOUT = "critic_rollout_no_grad"
    CRITIC_DENOISE_BACKWARD = "critic_denoise_backward"
    CRITIC_GAN_BACKWARD = "critic_gan_backward"
    CRITIC_COMMIT = "critic_optimizer_commit"
    EMA_COMMIT = "ema_commit"


@dataclass(frozen=True)
class GanPipelineTopology:
    world_size: int = 16
    generator_ranks: tuple[int, ...] = tuple(range(8))
    score_ranks: tuple[int, ...] = tuple(range(8, 16))

    def __post_init__(self) -> None:
        all_ranks = self.generator_ranks + self.score_ranks
        if self.world_size != 16:
            raise ValueError("primary GAN pipeline is frozen to world16")
        if len(self.generator_ranks) != 8 or len(self.score_ranks) != 8:
            raise ValueError("primary GAN pipeline requires 8G+8Score")
        if len(set(all_ranks)) != self.world_size or set(all_ranks) != set(range(16)):
            raise ValueError("every world16 rank must have exactly one persistent role")

    def paired_score(self, generator_rank: int) -> int:
        if generator_rank not in self.generator_ranks:
            raise ValueError("rank is not a Generator rank")
        return self.score_ranks[self.generator_ranks.index(generator_rank)]

    def paired_generator(self, score_rank: int) -> int:
        if score_rank not in self.score_ranks:
            raise ValueError("rank is not a Score rank")
        return self.generator_ranks[self.score_ranks.index(score_rank)]


@dataclass(frozen=True)
class GanPipelineBatchContract:
    logical_global_batch: int = 16
    local_micro_batch: int = 1
    generator_dp_world_size: int = 8
    score_dp_world_size: int = 8
    micro_waves_per_logical_step: int = 2
    generator_gradient_accumulation_steps: int = 2
    critic_gradient_accumulation_steps: int = 2

    def __post_init__(self) -> None:
        if self.local_micro_batch != 1:
            raise ValueError("the matched GAN contract fixes micro batch to one")
        if self.generator_dp_world_size != 8 or self.score_dp_world_size != 8:
            raise ValueError("the matched GAN contract fixes 8G+8Score")
        if self.micro_waves_per_logical_step != 2:
            raise ValueError("world16 over an eight-rank role requires two micro-waves")
        if self.generator_gradient_accumulation_steps != 2:
            raise ValueError("Generator must accumulate exactly two micro-waves")
        if self.critic_gradient_accumulation_steps != 2:
            raise ValueError("Critic must accumulate exactly two micro-waves")
        generated = (
            self.local_micro_batch
            * self.generator_dp_world_size
            * self.micro_waves_per_logical_step
        )
        consumed = (
            self.local_micro_batch
            * self.score_dp_world_size
            * self.micro_waves_per_logical_step
        )
        if generated != self.logical_global_batch or consumed != self.logical_global_batch:
            raise ValueError("role-local waves do not reconstruct global batch16")


@dataclass(frozen=True)
class VersionState:
    generator: int
    critic: int
    ema: int

    def __post_init__(self) -> None:
        if min(self.generator, self.critic, self.ema) < 0:
            raise ValueError("versions must be non-negative")


@dataclass(frozen=True)
class ScoreGradientRequest:
    logical_step: int
    micro_wave: int
    generator_version: int
    critic_version: int
    trace_digest: str
    latent_shape: tuple[int, ...] = LATENT_SHAPE
    latent_dtype: str = LATENT_DTYPE

    def __post_init__(self) -> None:
        if self.logical_step < 0 or self.micro_wave not in (0, 1):
            raise ValueError("invalid logical step or micro-wave")
        if self.generator_version < 0 or self.critic_version < 0:
            raise ValueError("version must be non-negative")
        if len(self.trace_digest) != 64:
            raise ValueError("trace digest must be SHA256")
        if self.latent_shape != LATENT_SHAPE or self.latent_dtype != LATENT_DTYPE:
            raise ValueError("score boundary requires the official BF16 latent shape")


@dataclass(frozen=True)
class ScoreGradientReply:
    logical_step: int
    micro_wave: int
    generator_version: int
    critic_before: int
    critic_after: int
    trace_digest: str
    gradient_shape: tuple[int, ...] = LATENT_SHAPE
    gradient_dtype: str = GRADIENT_DTYPE
    dmd_loss: float | None = None
    gan_g_loss: float | None = None

    def validate_against(self, request: ScoreGradientRequest) -> None:
        expected = (
            request.logical_step,
            request.micro_wave,
            request.generator_version,
            request.critic_version,
            request.trace_digest,
        )
        observed = (
            self.logical_step,
            self.micro_wave,
            self.generator_version,
            self.critic_before,
            self.trace_digest,
        )
        if observed != expected:
            raise AssertionError("Score gradient reply does not match its request")
        if self.critic_after != self.critic_before:
            raise AssertionError("GAN-G/DMD gradient service advanced Critic state")
        if self.gradient_shape != request.latent_shape:
            raise AssertionError("remote latent gradient shape drift")
        if self.gradient_dtype != GRADIENT_DTYPE:
            raise AssertionError("remote latent gradient dtype drift")


@dataclass(frozen=True)
class PipelineOperation:
    logical_step: int
    phase: GanPipelinePhase
    versions_before: VersionState
    versions_after: VersionState
    micro_wave: int | None = None
    queue_must_be_empty: bool = False
    score_trainable: bool | None = None
    optimizer_commit: str | None = None
    overlap_lane: str | None = None

    def __post_init__(self) -> None:
        if self.micro_wave is not None and self.micro_wave not in (0, 1):
            raise ValueError("micro-wave must be 0, 1, or None")
        if self.phase is GanPipelinePhase.GENERATOR_COMMIT:
            if self.versions_after.generator != self.versions_before.generator + 1:
                raise ValueError("Generator commit must advance G exactly once")
        elif self.versions_after.generator != self.versions_before.generator:
            raise ValueError("non-Generator phase changed G version")
        if self.phase is GanPipelinePhase.CRITIC_COMMIT:
            if self.versions_after.critic != self.versions_before.critic + 1:
                raise ValueError("Critic commit must advance C exactly once")
        elif self.versions_after.critic != self.versions_before.critic:
            raise ValueError("non-Critic phase changed C version")
        if self.phase is GanPipelinePhase.EMA_COMMIT:
            if self.versions_after.ema != self.versions_before.ema + 1:
                raise ValueError("EMA commit must advance EMA exactly once")
        elif self.versions_after.ema != self.versions_before.ema:
            raise ValueError("non-EMA phase changed EMA version")


class CriticAccumulationState:
    """Enforce two micro-waves and one Critic optimizer commit.

    Official low-VRAM GAN performs a denoising backward and a GAN/R1/R2 backward
    before one Critic update.  In the 8Score matched layout, global batch16 needs
    two micro-waves, so one logical step contains four backwards total but still
    exactly one optimizer/scheduler version increment.
    """

    def __init__(self, waves: int = 2) -> None:
        if waves != 2:
            raise ValueError("matched world16 GAN pipeline requires two waves")
        self.waves = waves
        self._denoise: set[int] = set()
        self._gan: set[int] = set()
        self.committed = False

    def denoise_backward(self, wave: int) -> None:
        self._check_wave(wave)
        if wave in self._denoise:
            raise RuntimeError("duplicate denoising backward")
        if wave in self._gan:
            raise RuntimeError("GAN backward arrived before denoising backward")
        self._denoise.add(wave)

    def gan_backward(self, wave: int) -> None:
        self._check_wave(wave)
        if wave not in self._denoise:
            raise RuntimeError("GAN backward requires denoising backward first")
        if wave in self._gan:
            raise RuntimeError("duplicate GAN backward")
        self._gan.add(wave)

    def can_commit(self) -> bool:
        return self._denoise == {0, 1} and self._gan == {0, 1} and not self.committed

    def commit(self) -> None:
        if not self.can_commit():
            raise RuntimeError("Critic commit before both low-VRAM micro-waves completed")
        self.committed = True

    def _check_wave(self, wave: int) -> None:
        if wave not in (0, 1):
            raise ValueError("wave must be 0 or 1")
        if self.committed:
            raise RuntimeError("logical Critic step already committed")


class GanPipelineQueueState:
    """Bounded Q2 producer/consumer queue for Critic rollouts."""

    def __init__(self, capacity: int = 2) -> None:
        if capacity != 2:
            raise ValueError("primary GAN pipeline uses Q2")
        self.capacity = capacity
        self._ready: list[tuple[int, int, int]] = []
        self._consuming: tuple[int, int, int] | None = None

    @property
    def depth(self) -> int:
        return len(self._ready) + int(self._consuming is not None)

    def push(self, *, logical_step: int, micro_wave: int, generator_version: int) -> None:
        item = (logical_step, micro_wave, generator_version)
        if self.depth >= self.capacity:
            raise RuntimeError("GAN pipeline queue exceeded Q2")
        if item in self._ready or item == self._consuming:
            raise RuntimeError("duplicate rollout payload")
        self._ready.append(item)

    def begin_consume(self) -> tuple[int, int, int]:
        if self._consuming is not None:
            raise RuntimeError("Score lane already consuming")
        if not self._ready:
            raise RuntimeError("no rollout available")
        self._consuming = self._ready.pop(0)
        return self._consuming

    def finish_consume(self) -> tuple[int, int, int]:
        if self._consuming is None:
            raise RuntimeError("no active rollout consumption")
        item = self._consuming
        self._consuming = None
        return item

    def assert_empty(self) -> None:
        if self.depth:
            raise AssertionError("Generator-update barrier requires an empty Q2 queue")


class GanPipelinePlanner:
    """Compile exact optimizer/version order for the proposed 8G+8Score pipeline."""

    def __init__(
        self,
        *,
        generator_update_every: int = 5,
        batch: GanPipelineBatchContract = GanPipelineBatchContract(),
    ) -> None:
        if generator_update_every != 5:
            raise ValueError("Helios GAN contract fixes dfake_gen_update_ratio=5")
        self.generator_update_every = generator_update_every
        self.batch = batch

    def compile(
        self,
        *,
        start_step: int,
        steps: int,
        initial: VersionState,
    ) -> tuple[PipelineOperation, ...]:
        if start_step < 0 or steps <= 0:
            raise ValueError("invalid pipeline range")
        current = initial
        result: list[PipelineOperation] = []
        for step in range(start_step, start_step + steps):
            train_generator = step % self.generator_update_every == 0
            if train_generator:
                result.append(
                    PipelineOperation(
                        logical_step=step,
                        phase=GanPipelinePhase.FLUSH,
                        versions_before=current,
                        versions_after=current,
                        queue_must_be_empty=True,
                        overlap_lane="barrier",
                    )
                )
                for wave in range(self.batch.micro_waves_per_logical_step):
                    result.append(
                        PipelineOperation(
                            logical_step=step,
                            phase=GanPipelinePhase.GENERATOR_FORWARD,
                            micro_wave=wave,
                            versions_before=current,
                            versions_after=current,
                            score_trainable=False,
                            overlap_lane="generator",
                        )
                    )
                    result.append(
                        PipelineOperation(
                            logical_step=step,
                            phase=GanPipelinePhase.SCORE_BOUNDARY_GRAD,
                            micro_wave=wave,
                            versions_before=current,
                            versions_after=current,
                            score_trainable=False,
                            overlap_lane="score",
                        )
                    )
                    result.append(
                        PipelineOperation(
                            logical_step=step,
                            phase=GanPipelinePhase.GENERATOR_BACKWARD,
                            micro_wave=wave,
                            versions_before=current,
                            versions_after=current,
                            score_trainable=False,
                            overlap_lane="generator",
                        )
                    )
                after_g = VersionState(current.generator + 1, current.critic, current.ema)
                result.append(
                    PipelineOperation(
                        logical_step=step,
                        phase=GanPipelinePhase.GENERATOR_COMMIT,
                        versions_before=current,
                        versions_after=after_g,
                        score_trainable=False,
                        optimizer_commit="generator",
                        overlap_lane="barrier",
                    )
                )
                current = after_g

            # The Critic rollout for a Generator-update step must use the newly
            # committed G version, exactly matching official train_helios.py.
            for wave in range(self.batch.micro_waves_per_logical_step):
                result.append(
                    PipelineOperation(
                        logical_step=step,
                        phase=GanPipelinePhase.CRITIC_ROLLOUT,
                        micro_wave=wave,
                        versions_before=current,
                        versions_after=current,
                        score_trainable=True,
                        overlap_lane="generator_producer",
                    )
                )
                result.append(
                    PipelineOperation(
                        logical_step=step,
                        phase=GanPipelinePhase.CRITIC_DENOISE_BACKWARD,
                        micro_wave=wave,
                        versions_before=current,
                        versions_after=current,
                        score_trainable=True,
                        overlap_lane="score_consumer",
                    )
                )
                result.append(
                    PipelineOperation(
                        logical_step=step,
                        phase=GanPipelinePhase.CRITIC_GAN_BACKWARD,
                        micro_wave=wave,
                        versions_before=current,
                        versions_after=current,
                        score_trainable=True,
                        overlap_lane="score_consumer",
                    )
                )

            after_c = VersionState(current.generator, current.critic + 1, current.ema)
            result.append(
                PipelineOperation(
                    logical_step=step,
                    phase=GanPipelinePhase.CRITIC_COMMIT,
                    versions_before=current,
                    versions_after=after_c,
                    score_trainable=True,
                    optimizer_commit="critic",
                    overlap_lane="barrier",
                )
            )
            current = after_c
            after_ema = VersionState(current.generator, current.critic, current.ema + 1)
            result.append(
                PipelineOperation(
                    logical_step=step,
                    phase=GanPipelinePhase.EMA_COMMIT,
                    versions_before=current,
                    versions_after=after_ema,
                    score_trainable=None,
                    optimizer_commit="ema",
                    overlap_lane="barrier",
                )
            )
            current = after_ema
        return tuple(result)


def summarize_operations(operations: Iterable[PipelineOperation]) -> dict[str, object]:
    ops = tuple(operations)
    counts: dict[str, int] = {}
    for operation in ops:
        counts[operation.phase.value] = counts.get(operation.phase.value, 0) + 1
    return {
        "operation_count": len(ops),
        "phase_counts": counts,
        "generator_commits": counts.get(GanPipelinePhase.GENERATOR_COMMIT.value, 0),
        "critic_commits": counts.get(GanPipelinePhase.CRITIC_COMMIT.value, 0),
        "ema_commits": counts.get(GanPipelinePhase.EMA_COMMIT.value, 0),
    }


def expected_final_versions(
    *, start_step: int, steps: int, initial: VersionState
) -> VersionState:
    generator_updates = sum(
        step % 5 == 0 for step in range(start_step, start_step + steps)
    )
    return VersionState(
        generator=initial.generator + generator_updates,
        critic=initial.critic + steps,
        ema=initial.ema + steps,
    )


__all__ = [
    "CriticAccumulationState",
    "GanPipelineBatchContract",
    "GanPipelinePhase",
    "GanPipelinePlanner",
    "GanPipelineQueueState",
    "GanPipelineTopology",
    "PipelineOperation",
    "ScoreGradientReply",
    "ScoreGradientRequest",
    "VersionState",
    "expected_final_versions",
    "summarize_operations",
]
