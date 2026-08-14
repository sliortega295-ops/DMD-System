"""Phase-B Critic batch12 execution contract.

Phase B updates only the Critic/Score model, so it does not need to preserve the
Generator-update batch16 used in Phase A.  The first real implementation reuses
the already validated no-GAN 4G+12C shape:

* four no-grad Generator producers emit three samples each;
* twelve Critic ranks each consume exactly one B1 sample;
* every Critic rank enters exactly one physical model call per logical step;
* the global Critic batch is 12; and
* reduction semantics are explicit, but no uneven-rank batching is required.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .gan_rolephase import GanRolePhaseTopology


@dataclass(frozen=True)
class CriticLocalBatch:
    global_rank: int
    logical_slots: tuple[int, ...]
    local_batch: int = 1
    global_batch: int = 12

    def __post_init__(self) -> None:
        if self.global_batch != 12:
            raise ValueError("role-phase Phase-B Critic is frozen to global batch12")
        if self.local_batch != 1 or len(self.logical_slots) != 1:
            raise ValueError("every Phase-B Critic rank must consume exactly one B1 sample")
        if len(set(self.logical_slots)) != 1:
            raise ValueError("Critic local batch must contain one unique logical slot")


@dataclass(frozen=True)
class CriticReductionContract:
    """Explicit batch12 gradient semantics for DP12/B1.

    The desired gradient is (1/12) * sum(sample_grad_i).

    * If the backend performs DP_MEAN over 12 ranks, the ordinary per-rank B1
      mean loss needs no extra scaling.
    * If the backend performs an explicit SUM, each local B1 loss is scaled by
      1/12 before backward.
    """

    dp_world_size: int = 12
    global_batch: int = 12
    local_batch: int = 1
    physical_model_calls_per_rank: int = 1
    logical_optimizer_commits: int = 1

    def __post_init__(self) -> None:
        if (self.dp_world_size, self.global_batch, self.local_batch) != (12, 12, 1):
            raise ValueError("Phase-B Critic requires DP12/B1/global-batch12")
        if self.physical_model_calls_per_rank != 1:
            raise ValueError("all C12 ranks must enter one model call per step")
        if self.logical_optimizer_commits != 1:
            raise ValueError("one logical Critic step must commit exactly once")

    def local_loss_scale(self, backend_dp_reduction: str) -> float:
        mode = str(backend_dp_reduction).upper()
        if mode == "SUM":
            return 1.0 / float(self.global_batch)
        if mode in {"MEAN", "DP_MEAN"}:
            return 1.0
        raise ValueError(
            "backend_dp_reduction must be explicitly bound to SUM or DP_MEAN"
        )

    # Compatibility name used by the inspector/tests from the first prototype.
    def local_sum_scale(self, backend_dp_reduction: str) -> float:
        return self.local_loss_scale(backend_dp_reduction)


def build_critic_local_batches(
    topology: GanRolePhaseTopology = GanRolePhaseTopology(),
) -> tuple[CriticLocalBatch, ...]:
    result: list[CriticLocalBatch] = []
    covered: list[int] = []
    for rank in topology.phase_b_critic_ranks:
        slots = tuple(
            item.logical_slot for item in topology.phase_b_slots_for_critic(rank)
        )
        batch = CriticLocalBatch(rank, slots)
        result.append(batch)
        covered.extend(slots)
    if sorted(covered) != list(range(topology.phase_b_global_batch)):
        raise AssertionError("C12 ranks do not cover logical batch12 exactly")
    if any(item.local_batch != 1 for item in result):
        raise AssertionError("every Phase-B Critic rank must remain B1")
    return tuple(result)


def assert_collective_safe_call_signature(
    calls_per_rank: Mapping[int, int],
    topology: GanRolePhaseTopology = GanRolePhaseTopology(),
) -> None:
    expected = set(topology.phase_b_critic_ranks)
    if set(calls_per_rank) != expected:
        raise AssertionError("C12 call-signature ranks differ from the Critic group")
    bad = {rank: calls for rank, calls in calls_per_rank.items() if int(calls) != 1}
    if bad:
        raise AssertionError(
            "every C12 rank must enter exactly one physical model call per step: "
            f"{bad}"
        )


def expected_slot_order_for_rank(
    rank: int,
    topology: GanRolePhaseTopology = GanRolePhaseTopology(),
) -> tuple[int, ...]:
    return tuple(
        item.logical_slot for item in topology.phase_b_slots_for_critic(rank)
    )


def validate_per_sample_weights(weights: Sequence[float]) -> None:
    if len(weights) != 12:
        raise AssertionError("exactly twelve per-sample weights are required")
    if any(abs(float(value) - 1.0 / 12.0) > 1e-12 for value in weights):
        raise AssertionError("every Phase-B logical sample must retain weight 1/12")
    if abs(sum(float(value) for value in weights) - 1.0) > 1e-12:
        raise AssertionError("Phase-B batch12 weights do not sum to one")


__all__ = [
    "CriticLocalBatch",
    "CriticReductionContract",
    "assert_collective_safe_call_signature",
    "build_critic_local_batches",
    "expected_slot_order_for_rank",
    "validate_per_sample_weights",
]
