"""Batch16-equivalent Phase-B Critic execution contract.

The old 12+4 sketch described sixteen logical samples but did not describe a
collective-safe physical execution.  This module freezes the first admissible
shape for a ZeRO-3/FSDP-style 12-rank Critic backend:

* every Critic rank enters exactly one model forward/backward per logical step;
* Flex ranks 4..7 pack two logical samples into local batch=2;
* fixed Critic ranks 8..15 pack one logical sample into local batch=1;
* the sixteen samples are reduced as one logical global batch; and
* loss/reduction normalization is explicit rather than inferred from rank count.

The model-specific packer and DeepSpeed/FSDP reduction hook live in the real
executor.  This module is dependency-light so CI can fail closed on the batch
math before any 14B process is constructed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .gan_rolephase import GanRolePhaseTopology


@dataclass(frozen=True)
class CriticLocalBatch:
    global_rank: int
    logical_slots: tuple[int, ...]
    local_batch: int
    global_batch: int = 16

    def __post_init__(self) -> None:
        if self.global_batch != 16:
            raise ValueError("role-phase Critic is frozen to global batch16")
        if self.local_batch != len(self.logical_slots):
            raise ValueError("local batch must equal the number of logical slots")
        if self.local_batch not in (1, 2):
            raise ValueError("C12 v1 allows only local batch1 or batch2")
        if len(set(self.logical_slots)) != len(self.logical_slots):
            raise ValueError("logical slot was duplicated within one Critic rank")


@dataclass(frozen=True)
class CriticReductionContract:
    """Explicit reduction semantics for an uneven 12-rank physical batch.

    ``backend_dp_reduction`` is deliberately not guessed.  A real backend must
    bind whether its gradient collective returns SUM or DP_MEAN and use the
    corresponding local loss scale.  Either path yields the same desired
    gradient: (1/16) * sum(sample_grad_i).
    """

    dp_world_size: int = 12
    global_batch: int = 16
    physical_model_calls_per_rank: int = 1
    logical_optimizer_commits: int = 1

    def __post_init__(self) -> None:
        if (self.dp_world_size, self.global_batch) != (12, 16):
            raise ValueError("C12 v1 requires DP12/global-batch16")
        if self.physical_model_calls_per_rank != 1:
            raise ValueError("all C12 ranks must enter one model call per step")
        if self.logical_optimizer_commits != 1:
            raise ValueError("one logical Critic step must commit exactly once")

    def local_sum_scale(self, backend_dp_reduction: str) -> float:
        """Scale a local *sum of per-sample losses* before backward.

        For an explicit DP SUM, each rank contributes local_sum/16.
        For a backend DP MEAN over 12 ranks, each rank contributes
        local_sum*(12/16), because the backend later divides by 12.
        """
        mode = str(backend_dp_reduction).upper()
        if mode == "SUM":
            return 1.0 / float(self.global_batch)
        if mode in {"MEAN", "DP_MEAN"}:
            return float(self.dp_world_size) / float(self.global_batch)
        raise ValueError(
            "backend_dp_reduction must be explicitly bound to SUM or DP_MEAN"
        )


def build_critic_local_batches(
    topology: GanRolePhaseTopology = GanRolePhaseTopology(),
) -> tuple[CriticLocalBatch, ...]:
    result: list[CriticLocalBatch] = []
    covered: list[int] = []
    for rank in topology.phase_b_critic_ranks:
        slots = tuple(
            item.logical_slot for item in topology.phase_b_slots_for_critic(rank)
        )
        batch = CriticLocalBatch(rank, slots, len(slots))
        result.append(batch)
        covered.extend(slots)
    if sorted(covered) != list(range(16)):
        raise AssertionError("C12 local batches do not cover logical batch16 exactly")
    if tuple(item.local_batch for item in result[:4]) != (2, 2, 2, 2):
        raise AssertionError("Flex Critic ranks must each pack two samples")
    if any(item.local_batch != 1 for item in result[4:]):
        raise AssertionError("fixed Critic ranks must each pack one sample")
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
    """Canonical FIFO order used by the receiver and local batch packer."""
    return tuple(
        item.logical_slot for item in topology.phase_b_slots_for_critic(rank)
    )


def validate_per_sample_weights(weights: Sequence[float]) -> None:
    if len(weights) != 16:
        raise AssertionError("exactly sixteen per-sample weights are required")
    if any(abs(float(value) - 1.0 / 16.0) > 1e-12 for value in weights):
        raise AssertionError("every logical sample must retain weight 1/16")
    if abs(sum(float(value) for value in weights) - 1.0) > 1e-12:
        raise AssertionError("logical batch weights do not sum to one")


__all__ = [
    "CriticLocalBatch",
    "CriticReductionContract",
    "assert_collective_safe_call_signature",
    "build_critic_local_batches",
    "expected_slot_order_for_rank",
    "validate_per_sample_weights",
]
