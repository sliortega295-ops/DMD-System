"""Phase-aware GAN scheduling contracts for the 4G + 8C + 4Flex prototype.

This module owns only systems semantics: topology, batch16 mapping, phase
versioning, Q2 credits, and bounded GPU-shard redistribution. It deliberately
does not reimplement Helios DMD/GAN math; PR #7 trace/fairness/compute/version
and wire-header contracts remain authoritative.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence


class GanRolePhase(str, enum.Enum):
    PHASE_A = "phase_a_8g8score"
    HANDOFF_TO_B = "handoff_to_b"
    PHASE_B = "phase_b_4g12critic"
    HANDOFF_TO_A = "handoff_to_a"
    COMPLETE = "complete"


@dataclass(frozen=True)
class PhaseASlot:
    logical_slot: int
    micro_wave: int
    generator_rank: int
    score_rank: int


@dataclass(frozen=True)
class PhaseBSlot:
    logical_slot: int
    critic_micro_wave: int
    generator_rank: int
    critic_rank: int
    loss_weight: float


@dataclass(frozen=True)
class GanRolePhaseTopology:
    """World16 topology preserving PR #7 Phase-A rank placement.

    Phase A: ranks 0..7 = Generator, ranks 8..15 = Score.
    Phase B: ranks 0..3 = resident no-grad Generator producer,
             ranks 4..7 = Flex G->Critic,
             ranks 8..15 = resident Critic.
    """

    world_size: int = 16
    fixed_generator_ranks: tuple[int, ...] = (0, 1, 2, 3)
    flex_ranks: tuple[int, ...] = (4, 5, 6, 7)
    fixed_critic_ranks: tuple[int, ...] = tuple(range(8, 16))
    global_batch: int = 16

    def __post_init__(self) -> None:
        if self.world_size != 16 or self.global_batch != 16:
            raise ValueError("GAN role-phase v1 is frozen to world16/batch16")
        groups = (self.fixed_generator_ranks, self.flex_ranks, self.fixed_critic_ranks)
        if tuple(map(len, groups)) != (4, 4, 8):
            raise ValueError("expected exactly 4 fixed G, 4 Flex and 8 fixed C")
        all_ranks = tuple(rank for group in groups for rank in group)
        if len(set(all_ranks)) != 16 or set(all_ranks) != set(range(16)):
            raise ValueError("role groups must be disjoint and cover ranks 0..15")

    @property
    def phase_a_generator_ranks(self) -> tuple[int, ...]:
        return self.fixed_generator_ranks + self.flex_ranks

    @property
    def phase_a_score_ranks(self) -> tuple[int, ...]:
        return self.fixed_critic_ranks

    @property
    def phase_b_generator_ranks(self) -> tuple[int, ...]:
        return self.fixed_generator_ranks

    @property
    def phase_b_critic_ranks(self) -> tuple[int, ...]:
        return self.flex_ranks + self.fixed_critic_ranks

    def phase_a_slot(self, logical_slot: int) -> PhaseASlot:
        if not 0 <= logical_slot < 16:
            raise ValueError("logical slot must be in [0,16)")
        wave, lane = divmod(logical_slot, 8)
        return PhaseASlot(
            logical_slot=logical_slot,
            micro_wave=wave,
            generator_rank=self.phase_a_generator_ranks[lane],
            score_rank=self.phase_a_score_ranks[lane],
        )

    def phase_b_slot(self, logical_slot: int) -> PhaseBSlot:
        """Map batch16 to 4 producers and 12 Critic consumers exactly once.

        Critic wave0: slots 0..11, one per C rank.
        Critic wave1: slots 12..15, one extra sample per Flex C rank.
        Each G rank produces four samples: lane, lane+4, lane+8, lane+12.
        """
        if not 0 <= logical_slot < 16:
            raise ValueError("logical slot must be in [0,16)")
        generator_rank = self.fixed_generator_ranks[logical_slot % 4]
        if logical_slot < 12:
            critic_wave = 0
            critic_rank = self.phase_b_critic_ranks[logical_slot]
        else:
            critic_wave = 1
            critic_rank = self.flex_ranks[logical_slot - 12]
        return PhaseBSlot(
            logical_slot=logical_slot,
            critic_micro_wave=critic_wave,
            generator_rank=generator_rank,
            critic_rank=critic_rank,
            loss_weight=1.0 / 16.0,
        )

    def phase_b_slots_for_generator(self, rank: int) -> tuple[PhaseBSlot, ...]:
        if rank not in self.fixed_generator_ranks:
            raise ValueError("rank is not a Phase-B Generator")
        return tuple(
            item for item in map(self.phase_b_slot, range(16))
            if item.generator_rank == rank
        )

    def phase_b_slots_for_critic(self, rank: int) -> tuple[PhaseBSlot, ...]:
        if rank not in self.phase_b_critic_ranks:
            raise ValueError("rank is not a Phase-B Critic")
        return tuple(
            item for item in map(self.phase_b_slot, range(16))
            if item.critic_rank == rank
        )

    def phase_groups(self) -> Mapping[str, tuple[int, ...]]:
        """All NCCL groups should be created once before the formal window."""
        return {
            "phase_a_generator": self.phase_a_generator_ranks,
            "phase_a_score": self.phase_a_score_ranks,
            "phase_b_generator": self.phase_b_generator_ranks,
            "phase_b_critic": self.phase_b_critic_ranks,
            "flex": self.flex_ranks,
            "world": tuple(range(16)),
        }


@dataclass(frozen=True)
class GpuShardedResidencyContract:
    """Forbid repeated CPU frozen-backbone streaming in steady-state phases."""

    phase_a_generator_shards: int = 8
    phase_a_score_shards: int = 8
    phase_b_generator_shards: int = 4
    phase_b_critic_shards: int = 12
    frozen_parameter_transport: str = "nccl_p2p_chunked_reshard"
    repeated_cpu_frozen_streaming: bool = False
    cpu_optimizer_offload_allowed: bool = True
    handoff_is_bounded_once_per_direction: bool = True

    def __post_init__(self) -> None:
        geometry = (
            self.phase_a_generator_shards,
            self.phase_a_score_shards,
            self.phase_b_generator_shards,
            self.phase_b_critic_shards,
        )
        if geometry != (8, 8, 4, 12):
            raise ValueError("unexpected role-phase shard geometry")
        if self.repeated_cpu_frozen_streaming:
            raise ValueError("repeated CPU frozen-backbone streaming is forbidden")
        if self.frozen_parameter_transport not in {
            "nccl_p2p_chunked_reshard",
            "nccl_all_to_all_chunked_reshard",
        }:
            raise ValueError("frozen parameter handoff must stay GPU-side")
        if not self.handoff_is_bounded_once_per_direction:
            raise ValueError("parameter handoff must be phase-bounded")


@dataclass(frozen=True)
class ShardRange:
    rank: int
    start: int
    stop: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.stop < self.start:
            raise ValueError("invalid shard range")

    @property
    def numel(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class ReshardTransfer:
    src_rank: int
    dst_rank: int
    global_start: int
    global_stop: int
    src_offset: int
    dst_offset: int

    @property
    def numel(self) -> int:
        return self.global_stop - self.global_start


def balanced_shards(total_numel: int, ranks: Sequence[int]) -> tuple[ShardRange, ...]:
    ranks = tuple(int(rank) for rank in ranks)
    if total_numel <= 0:
        raise ValueError("total_numel must be positive")
    if not ranks or len(set(ranks)) != len(ranks):
        raise ValueError("ranks must be unique and non-empty")
    base, remainder = divmod(total_numel, len(ranks))
    cursor = 0
    result: list[ShardRange] = []
    for index, rank in enumerate(ranks):
        width = base + (1 if index < remainder else 0)
        result.append(ShardRange(rank, cursor, cursor + width))
        cursor += width
    if cursor != total_numel:
        raise AssertionError("balanced shards do not cover payload")
    return tuple(result)


def _split_transfer(
    src: ShardRange,
    dst: ShardRange,
    start: int,
    stop: int,
    max_chunk_numel: int | None,
) -> Iterable[ReshardTransfer]:
    if max_chunk_numel is not None and max_chunk_numel <= 0:
        raise ValueError("max_chunk_numel must be positive")
    cursor = start
    while cursor < stop:
        next_stop = stop if max_chunk_numel is None else min(stop, cursor + max_chunk_numel)
        yield ReshardTransfer(
            src.rank,
            dst.rank,
            cursor,
            next_stop,
            cursor - src.start,
            cursor - dst.start,
        )
        cursor = next_stop


def plan_contiguous_reshard(
    total_numel: int,
    src_ranks: Sequence[int],
    dst_ranks: Sequence[int],
    *,
    max_chunk_numel: int | None = None,
) -> tuple[ReshardTransfer, ...]:
    """Exact canonical flat-shard redistribution without full all-gather."""
    sources = balanced_shards(total_numel, src_ranks)
    destinations = balanced_shards(total_numel, dst_ranks)
    result: list[ReshardTransfer] = []
    for src in sources:
        for dst in destinations:
            start, stop = max(src.start, dst.start), min(src.stop, dst.stop)
            if start < stop:
                result.extend(_split_transfer(src, dst, start, stop, max_chunk_numel))
    validate_reshard_plan(total_numel, src_ranks, dst_ranks, result)
    return tuple(result)


def validate_reshard_plan(
    total_numel: int,
    src_ranks: Sequence[int],
    dst_ranks: Sequence[int],
    transfers: Sequence[ReshardTransfer],
) -> None:
    sources = {item.rank: item for item in balanced_shards(total_numel, src_ranks)}
    destinations = {item.rank: item for item in balanced_shards(total_numel, dst_ranks)}
    if not transfers:
        raise AssertionError("reshard plan is empty")
    by_src: dict[int, list[tuple[int, int]]] = {rank: [] for rank in sources}
    by_dst: dict[int, list[tuple[int, int]]] = {rank: [] for rank in destinations}
    for item in transfers:
        if item.src_rank not in sources or item.dst_rank not in destinations:
            raise AssertionError("reshard rank escaped plan")
        src, dst = sources[item.src_rank], destinations[item.dst_rank]
        if item.numel <= 0:
            raise AssertionError("empty reshard transfer")
        if not (
            src.start <= item.global_start < item.global_stop <= src.stop
            and dst.start <= item.global_start < item.global_stop <= dst.stop
        ):
            raise AssertionError("reshard transfer escaped canonical shards")
        if item.src_offset != item.global_start - src.start:
            raise AssertionError("source-local offset drift")
        if item.dst_offset != item.global_start - dst.start:
            raise AssertionError("destination-local offset drift")
        by_src[item.src_rank].append((item.global_start, item.global_stop))
        by_dst[item.dst_rank].append((item.global_start, item.global_stop))

    def exact(shard: ShardRange, intervals: list[tuple[int, int]]) -> None:
        cursor = shard.start
        for start, stop in sorted(intervals):
            if start != cursor:
                raise AssertionError(f"coverage gap/overlap at {cursor}: next={start}")
            cursor = stop
        if cursor != shard.stop:
            raise AssertionError(f"coverage ended at {cursor}, expected {shard.stop}")

    for rank, shard in sources.items():
        exact(shard, by_src[rank])
    for rank, shard in destinations.items():
        exact(shard, by_dst[rank])
    if sum(item.numel for item in transfers) != total_numel:
        raise AssertionError("reshard moved other than exactly one payload copy")


class BufferState(str, enum.Enum):
    FREE = "free"
    RESERVED = "reserved"
    SEND_POSTED = "send_posted"


@dataclass
class ProducerPayloadCredits:
    """Two producer buffers; NCCL send completion, not Critic ACK, returns credit."""

    capacity: int = 2
    _slots: list[tuple[BufferState, int | None]] = field(init=False)

    def __post_init__(self) -> None:
        if self.capacity != 2:
            raise ValueError("role-phase v1 fixes producer payload credits to Q2")
        self._slots = [(BufferState.FREE, None) for _ in range(2)]

    @property
    def available(self) -> int:
        return sum(state is BufferState.FREE for state, _ in self._slots)

    @property
    def active_sequences(self) -> tuple[int, ...]:
        return tuple(
            sequence for state, sequence in self._slots
            if state is not BufferState.FREE and sequence is not None
        )

    def reserve(self, sequence: int) -> int:
        if sequence in self.active_sequences:
            raise RuntimeError("sequence already owns a producer buffer")
        for index, (state, _) in enumerate(self._slots):
            if state is BufferState.FREE:
                self._slots[index] = (BufferState.RESERVED, int(sequence))
                return index
        raise RuntimeError("producer Q2 credits exhausted")

    def mark_send_posted(self, slot: int, sequence: int) -> None:
        state, owner = self._slots[slot]
        if state is not BufferState.RESERVED or owner != sequence:
            raise RuntimeError("buffer was not reserved by this sequence")
        self._slots[slot] = (BufferState.SEND_POSTED, owner)

    def mark_send_complete(self, slot: int, sequence: int) -> None:
        state, owner = self._slots[slot]
        if state is not BufferState.SEND_POSTED or owner != sequence:
            raise RuntimeError("send is not outstanding for this sequence")
        self._slots[slot] = (BufferState.FREE, None)


class ConsumerBatchState(str, enum.Enum):
    ACCEPTED = "accepted"
    CONSUMING = "consuming"


@dataclass
class ConsumerQ2Window:
    """At most two received logical batches may remain uncommitted."""

    capacity: int = 2
    _active: dict[int, ConsumerBatchState] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.capacity != 2:
            raise ValueError("role-phase v1 fixes consumer queue depth to Q2")

    @property
    def active_sequences(self) -> tuple[int, ...]:
        return tuple(sorted(self._active))

    def accept(self, sequence: int) -> None:
        sequence = int(sequence)
        if sequence in self._active:
            raise RuntimeError("consumer already owns this batch")
        if len(self._active) >= self.capacity:
            raise RuntimeError("consumer Q2 window is full")
        self._active[sequence] = ConsumerBatchState.ACCEPTED

    def begin_consume(self, sequence: int) -> None:
        if self._active.get(sequence) is not ConsumerBatchState.ACCEPTED:
            raise RuntimeError("batch was not accepted")
        self._active[sequence] = ConsumerBatchState.CONSUMING

    def commit(self, sequence: int) -> None:
        if self._active.get(sequence) is not ConsumerBatchState.CONSUMING:
            raise RuntimeError("batch is not being consumed")
        del self._active[sequence]

    def assert_empty(self) -> None:
        if self._active:
            raise AssertionError(f"uncommitted Q2 batches remain: {self.active_sequences}")


@dataclass
class GanFull5RolePhaseState:
    """Fail-closed logical G/C/EMA state for one GAN-active Full5 cycle."""

    generator_version: int = 200
    critic_version: int = 1000
    ema_version: int = 800
    phase: GanRolePhase = GanRolePhase.PHASE_A
    score_gradient_service_flushed: bool = False
    generator_committed: bool = False
    handoff_to_b_complete: bool = False
    critic_committed_steps: list[int] = field(default_factory=list)
    ema_committed: bool = False
    reverse_handoff_complete: bool = False

    def flush_phase_a_score_service(self) -> None:
        if self.phase is not GanRolePhase.PHASE_A:
            raise RuntimeError("Score service may flush only in Phase A")
        self.score_gradient_service_flushed = True

    def commit_generator(self) -> None:
        if self.phase is not GanRolePhase.PHASE_A:
            raise RuntimeError("Generator commit belongs to Phase A")
        if not self.score_gradient_service_flushed:
            raise RuntimeError("Score gradient service must flush before G commit")
        if self.generator_committed:
            raise RuntimeError("Generator already committed")
        self.generator_version += 1
        self.generator_committed = True

    def begin_handoff_to_b(self) -> None:
        if self.phase is not GanRolePhase.PHASE_A:
            raise RuntimeError("unexpected handoff-to-B phase")
        if not self.generator_committed or not self.score_gradient_service_flushed:
            raise RuntimeError("Phase A must be drained before handoff")
        self.phase = GanRolePhase.HANDOFF_TO_B

    def finish_handoff_to_b(self) -> None:
        if self.phase is not GanRolePhase.HANDOFF_TO_B:
            raise RuntimeError("handoff-to-B was not active")
        self.handoff_to_b_complete = True
        self.phase = GanRolePhase.PHASE_B

    def commit_critic(self, logical_step: int) -> None:
        if self.phase is not GanRolePhase.PHASE_B:
            raise RuntimeError("Critic commits belong to Phase B")
        expected = 1000 + len(self.critic_committed_steps)
        if logical_step != expected or logical_step > 1004:
            raise RuntimeError(f"unexpected Critic commit {logical_step}; expected {expected}")
        self.critic_version += 1
        self.critic_committed_steps.append(logical_step)

    def commit_ema_after_step1000(self) -> None:
        if self.phase is not GanRolePhase.PHASE_B:
            raise RuntimeError("EMA commit belongs to Phase B")
        if self.critic_committed_steps != [1000]:
            raise RuntimeError("EMA must follow the step1000 Critic commit exactly")
        if self.ema_committed:
            raise RuntimeError("EMA already committed")
        self.ema_version += 1
        self.ema_committed = True

    def begin_reverse_handoff(self) -> None:
        if self.phase is not GanRolePhase.PHASE_B:
            raise RuntimeError("reverse handoff requires Phase B")
        if self.critic_committed_steps != list(range(1000, 1005)):
            raise RuntimeError("Full5 Critic sequence is incomplete")
        if not self.ema_committed:
            raise RuntimeError("step1000 EMA commit is missing")
        self.phase = GanRolePhase.HANDOFF_TO_A

    def finish_reverse_handoff(self) -> None:
        if self.phase is not GanRolePhase.HANDOFF_TO_A:
            raise RuntimeError("reverse handoff was not active")
        self.reverse_handoff_complete = True
        self.phase = GanRolePhase.COMPLETE

    def assert_complete(self) -> None:
        observed = (
            self.generator_version,
            self.critic_version,
            self.ema_version,
            self.phase,
            self.score_gradient_service_flushed,
            self.generator_committed,
            self.handoff_to_b_complete,
            self.critic_committed_steps,
            self.ema_committed,
            self.reverse_handoff_complete,
        )
        expected = (
            201,
            1005,
            801,
            GanRolePhase.COMPLETE,
            True,
            True,
            True,
            list(range(1000, 1005)),
            True,
            True,
        )
        if observed != expected:
            raise AssertionError(f"role-phase Full5 state drift: {observed} != {expected}")


def rolephase_contract_summary(
    topology: GanRolePhaseTopology = GanRolePhaseTopology(),
    residency: GpuShardedResidencyContract = GpuShardedResidencyContract(),
) -> dict[str, object]:
    return {
        "schema_version": "distillschedule.gan_rolephase.v1",
        "world_size": 16,
        "global_batch": 16,
        "fixed_generator_ranks": list(topology.fixed_generator_ranks),
        "flex_ranks": list(topology.flex_ranks),
        "fixed_critic_ranks": list(topology.fixed_critic_ranks),
        "phase_a": {
            "generator_ranks": list(topology.phase_a_generator_ranks),
            "score_ranks": list(topology.phase_a_score_ranks),
            "logical_waves": 2,
        },
        "phase_b": {
            "generator_ranks": list(topology.phase_b_generator_ranks),
            "critic_ranks": list(topology.phase_b_critic_ranks),
            "critic_sample_counts": {
                rank: len(topology.phase_b_slots_for_critic(rank))
                for rank in topology.phase_b_critic_ranks
            },
            "loss_reduction": "SUM(samples * 1/16)",
            "queue_depth": 2,
        },
        "residency": {
            "phase_a_generator_shards": residency.phase_a_generator_shards,
            "phase_a_score_shards": residency.phase_a_score_shards,
            "phase_b_generator_shards": residency.phase_b_generator_shards,
            "phase_b_critic_shards": residency.phase_b_critic_shards,
            "frozen_parameter_transport": residency.frozen_parameter_transport,
            "repeated_cpu_frozen_streaming": residency.repeated_cpu_frozen_streaming,
            "cpu_optimizer_offload_allowed": residency.cpu_optimizer_offload_allowed,
        },
        "correctness_reuse": [
            "operation-keyed trace",
            "fairness receipt",
            "compute inventory",
            "G/C/EMA version contract",
            "existing GAN request/reply header identities",
        ],
    }


__all__ = [
    "ConsumerQ2Window",
    "GanFull5RolePhaseState",
    "GanRolePhase",
    "GanRolePhaseTopology",
    "GpuShardedResidencyContract",
    "PhaseASlot",
    "PhaseBSlot",
    "ProducerPayloadCredits",
    "ReshardTransfer",
    "ShardRange",
    "balanced_shards",
    "plan_contiguous_reshard",
    "rolephase_contract_summary",
    "validate_reshard_plan",
]
