"""Fail-closed schedule contract for the Route-B Helios Full5 experiment.

This module deliberately owns no model math.  It makes the otherwise easy to
conflate quantities explicit:

* a logical Critic update;
* a Critic gradient-accumulation micro-wave; and
* a Generator production round.

Candidate A happens to use two of both per update.  Candidate B does not:
four Generator ranks must each produce three samples for one twelve-rank
Critic micro-wave.  Keeping these axes separate prevents the old one-video /
one-update mistake from reappearing in the executor.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RouteBPhase(str, Enum):
    INITIAL = "INITIAL"
    STEP10A = "STEP10A"
    TRANSITION = "TRANSITION"
    CRITIC = "CRITIC"
    FLUSHED = "FLUSHED"


@dataclass(frozen=True)
class RouteBFull5Contract:
    name: str
    generator_ranks: tuple[int, ...]
    critic_ranks: tuple[int, ...]
    micro_batch_per_critic_rank: int
    gradient_accumulation_steps: int
    logical_critic_updates: int = 5
    initial_generator_version: int = 2
    final_generator_version: int = 3
    initial_critic_version: int = 10
    final_critic_version: int = 15
    initial_ema_step: int = 10
    final_ema_step: int = 15
    gradient_clipping: float = 1.0

    def __post_init__(self) -> None:
        if not self.generator_ranks or not self.critic_ranks:
            raise ValueError("Route-B pools must be non-empty")
        if set(self.generator_ranks) & set(self.critic_ranks):
            raise ValueError("Generator and Critic ranks must be disjoint")
        if set(self.generator_ranks) | set(self.critic_ranks) != set(range(16)):
            raise ValueError("Route-B contract must cover world16 exactly")
        if self.micro_batch_per_critic_rank != 1:
            raise ValueError("the frozen experiment requires Critic MB1")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("GAS must be positive")
        if self.logical_critic_updates != 5:
            raise ValueError("steps 10-14 require exactly five Critic updates")
        if self.final_generator_version != self.initial_generator_version + 1:
            raise ValueError("Step10-A must perform exactly one Generator update")
        if self.final_critic_version != self.initial_critic_version + 5:
            raise ValueError("Full5 must advance C10 to C15")
        if self.final_ema_step != self.initial_ema_step + 5:
            raise ValueError("Full5 must advance EMA10 to EMA15")
        if self.gradient_clipping != 1.0:
            raise ValueError("official Full5 gradient clipping is 1.0")
        if self.global_critic_batch % len(self.generator_ranks):
            raise ValueError("global Critic batch must divide over Generator ranks")

    @property
    def critic_dp_world_size(self) -> int:
        return len(self.critic_ranks)

    @property
    def generator_pool_width(self) -> int:
        return len(self.generator_ranks)

    @property
    def global_critic_batch(self) -> int:
        return (
            self.critic_dp_world_size
            * self.micro_batch_per_critic_rank
            * self.gradient_accumulation_steps
        )

    @property
    def critic_micro_waves_per_update(self) -> int:
        return self.gradient_accumulation_steps

    @property
    def generator_production_rounds_per_update(self) -> int:
        return self.global_critic_batch // self.generator_pool_width

    @property
    def physical_critic_micro_waves(self) -> int:
        return self.logical_critic_updates * self.critic_micro_waves_per_update

    @property
    def physical_generator_rounds(self) -> int:
        return self.logical_critic_updates * self.generator_production_rounds_per_update

    @property
    def logical_samples(self) -> int:
        return self.logical_critic_updates * self.global_critic_batch

    def update_for_critic_micro_wave(self, micro_wave: int) -> int:
        if not 0 <= micro_wave < self.physical_critic_micro_waves:
            raise IndexError(micro_wave)
        return micro_wave // self.critic_micro_waves_per_update

    def micro_index_in_update(self, micro_wave: int) -> int:
        if not 0 <= micro_wave < self.physical_critic_micro_waves:
            raise IndexError(micro_wave)
        return micro_wave % self.critic_micro_waves_per_update

    def critic_version_for_micro_wave(self, micro_wave: int) -> int:
        return self.initial_critic_version + self.update_for_critic_micro_wave(micro_wave)

    def logical_step_for_update(self, update: int) -> int:
        if not 0 <= update < self.logical_critic_updates:
            raise IndexError(update)
        return 10 + update

    def source_trace_rank(self, production_round: int, producer_pool_rank: int) -> int:
        """Map one production slot to the authoritative rank16 trace.

        Each logical update consumes source ranks ``0..global_batch-1``.  A
        producer owns the strided slots ``round * G + producer_pool_rank``.
        """

        if not 0 <= production_round < self.generator_production_rounds_per_update:
            raise IndexError(production_round)
        if not 0 <= producer_pool_rank < self.generator_pool_width:
            raise IndexError(producer_pool_rank)
        source_rank = production_round * self.generator_pool_width + producer_pool_rank
        if source_rank >= self.global_critic_batch:
            raise AssertionError("source-rank mapping escaped the global batch")
        return source_rank

    @classmethod
    def candidate_a(cls) -> "RouteBFull5Contract":
        return cls(
            name="RB16-F5-PIPE-MATCH16",
            generator_ranks=tuple(range(8)),
            critic_ranks=tuple(range(8, 16)),
            micro_batch_per_critic_rank=1,
            gradient_accumulation_steps=2,
        )

    @classmethod
    def candidate_b(cls) -> "RouteBFull5Contract":
        return cls(
            name="RB16-F5-PIPE-2G6C",
            generator_ranks=(0, 1, 8, 9),
            critic_ranks=(2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15),
            micro_batch_per_critic_rank=1,
            gradient_accumulation_steps=1,
        )


@dataclass(frozen=True)
class RouteBCandidateBTopology:
    """Explicit many-consumer mapping for the frozen 4G+12C candidate.

    Candidate B is deliberately *not* represented by ``CriticPoolTopology``:
    that older topology is one-to-one and rejects unequal role widths.  One
    Generator rank owns three production rounds per logical Critic update;
    each round is consumed by a distinct Critic rank.  The mapping is stable
    and node-local for every P2P edge.
    """

    contract: RouteBFull5Contract

    def __post_init__(self) -> None:
        if self.contract != RouteBFull5Contract.candidate_b():
            raise ValueError("Candidate-B topology requires the frozen 4G+12C contract")
        for generator_rank in self.contract.generator_ranks:
            for critic_rank in self.critics_for_generator(generator_rank):
                if generator_rank // 8 != critic_rank // 8:
                    raise ValueError("Candidate-B G->C transport must remain node-local")

    def role(self, global_rank: int) -> str:
        if global_rank in self.contract.generator_ranks:
            return "generator"
        if global_rank in self.contract.critic_ranks:
            return "critic"
        raise ValueError("rank is outside Candidate-B world16")

    def generator_pool_rank(self, global_rank: int) -> int:
        try:
            return self.contract.generator_ranks.index(global_rank)
        except ValueError as error:
            raise ValueError("rank is not in the Candidate-B Generator pool") from error

    def critic_pool_rank(self, global_rank: int) -> int:
        try:
            return self.contract.critic_ranks.index(global_rank)
        except ValueError as error:
            raise ValueError("rank is not in the Candidate-B Critic pool") from error

    def critic_for(self, generator_global_rank: int, production_round: int) -> int:
        generator_pool_rank = self.generator_pool_rank(generator_global_rank)
        if not 0 <= production_round < self.contract.generator_production_rounds_per_update:
            raise IndexError(production_round)
        critic_pool_rank = (
            generator_pool_rank
            * self.contract.generator_production_rounds_per_update
            + production_round
        )
        return self.contract.critic_ranks[critic_pool_rank]

    def producer_for_critic(self, critic_global_rank: int) -> tuple[int, int]:
        critic_pool_rank = self.critic_pool_rank(critic_global_rank)
        generator_pool_rank, production_round = divmod(
            critic_pool_rank,
            self.contract.generator_production_rounds_per_update,
        )
        return (
            self.contract.generator_ranks[generator_pool_rank],
            production_round,
        )

    def critics_for_generator(self, generator_global_rank: int) -> tuple[int, ...]:
        return tuple(
            self.critic_for(generator_global_rank, production_round)
            for production_round in range(
                self.contract.generator_production_rounds_per_update
            )
        )

    @property
    def edges(self) -> tuple[tuple[int, int, int], ...]:
        """Return ``(G rank, C rank, production round)`` in group-create order."""

        return tuple(
            (
                generator_rank,
                self.critic_for(generator_rank, production_round),
                production_round,
            )
            for production_round in range(
                self.contract.generator_production_rounds_per_update
            )
            for generator_rank in self.contract.generator_ranks
        )


class RouteBFull5State:
    """Global schedule-only state used by unit tests and final summarization."""

    def __init__(self, contract: RouteBFull5Contract) -> None:
        self.contract = contract
        self.phase = RouteBPhase.INITIAL
        self.generator_version = contract.initial_generator_version
        self.critic_version = contract.initial_critic_version
        self.ema_step = contract.initial_ema_step
        self.consumed_micro_waves = 0
        self.optimizer_commits = 0
        self.ema_commits = 0
        self._accumulated_in_current_update = 0
        self.queue_empty = True
        self.outstanding_transport = 0

    def begin_step10a(self) -> None:
        if self.phase is not RouteBPhase.INITIAL:
            raise RuntimeError("Step10-A may begin only from INITIAL")
        self.phase = RouteBPhase.STEP10A

    def finish_step10a(self, *, generator_before: int, generator_after: int) -> None:
        if self.phase is not RouteBPhase.STEP10A:
            raise RuntimeError("Step10-A is not active")
        expected = (
            self.contract.initial_generator_version,
            self.contract.final_generator_version,
        )
        if (generator_before, generator_after) != expected:
            raise ValueError("Generator version transition changed")
        self.generator_version = generator_after
        self.phase = RouteBPhase.TRANSITION

    def finish_transition(
        self,
        *,
        generator_version: int,
        critic_version: int,
        ema_step: int,
        target_dp_world_size: int,
    ) -> None:
        if self.phase is not RouteBPhase.TRANSITION:
            raise RuntimeError("target-layout transition is not active")
        observed = (
            generator_version,
            critic_version,
            ema_step,
            target_dp_world_size,
        )
        expected = (
            self.contract.final_generator_version,
            self.contract.initial_critic_version,
            self.contract.initial_ema_step,
            self.contract.critic_dp_world_size,
        )
        if observed != expected:
            raise ValueError(f"target-layout transition mismatch: {observed} != {expected}")
        self.phase = RouteBPhase.CRITIC

    def consume_micro_wave(self, micro_wave: int, *, observed_critic_version: int) -> None:
        if self.phase is not RouteBPhase.CRITIC:
            raise RuntimeError("Critic phase is not active")
        if micro_wave != self.consumed_micro_waves:
            raise ValueError("Critic micro-waves must be consumed in order")
        expected_version = self.contract.critic_version_for_micro_wave(micro_wave)
        if observed_critic_version != expected_version or self.critic_version != expected_version:
            raise ValueError(
                f"micro-wave {micro_wave} consumed at C{observed_critic_version}; "
                f"expected C{expected_version}"
            )
        self.consumed_micro_waves += 1
        self._accumulated_in_current_update += 1

    def commit_optimizer(self, logical_update: int, *, critic_before: int, critic_after: int) -> None:
        if logical_update != self.optimizer_commits:
            raise ValueError("Critic commits must be strictly ordered")
        gas = self.contract.critic_micro_waves_per_update
        if self._accumulated_in_current_update != gas:
            raise RuntimeError(
                f"optimizer commit requires {gas} consumed micro-waves, "
                f"got {self._accumulated_in_current_update}"
            )
        expected_before = self.contract.initial_critic_version + logical_update
        if (critic_before, critic_after) != (expected_before, expected_before + 1):
            raise ValueError("Critic version transition changed")
        self.critic_version = critic_after
        self.optimizer_commits += 1
        self._accumulated_in_current_update = 0

    def commit_ema(
        self,
        logical_update: int,
        *,
        ema_before: int,
        ema_after: int,
        generator_version: int,
    ) -> None:
        if logical_update != self.ema_commits:
            raise ValueError("EMA commits must be strictly ordered")
        if self.optimizer_commits != logical_update + 1:
            raise RuntimeError("EMA must follow its Critic optimizer commit")
        expected_before = self.contract.initial_ema_step + logical_update
        if (ema_before, ema_after) != (expected_before, expected_before + 1):
            raise ValueError("EMA optimization-step recurrence changed")
        if generator_version != self.contract.final_generator_version:
            raise ValueError("EMA read a stale or future Generator version")
        self.ema_step = ema_after
        self.ema_commits += 1

    def assert_flushed(self) -> None:
        c = self.contract
        if self.consumed_micro_waves != c.physical_critic_micro_waves:
            raise AssertionError("Critic micro-wave count changed")
        if (self.optimizer_commits, self.ema_commits) != (5, 5):
            raise AssertionError("Full5 requires five Critic and five EMA commits")
        if self._accumulated_in_current_update:
            raise AssertionError("partial gradient accumulation remains")
        if not self.queue_empty or self.outstanding_transport:
            raise AssertionError("queue or transport did not flush")
        if (
            self.generator_version,
            self.critic_version,
            self.ema_step,
        ) != (
            c.final_generator_version,
            c.final_critic_version,
            c.final_ema_step,
        ):
            raise AssertionError("final G/C/EMA version tuple changed")
        self.phase = RouteBPhase.FLUSHED
