"""Contracts for the real-14B Critic-only Generator/Student pool.

This module deliberately contains no Helios model code.  It owns the small,
fail-closed pieces that must be identical in the sequential and pipelined
executors:

* the symmetric 4G+4S topology and global-batch contract;
* the fixed-width NCCL header/commit-ack wire format;
* the two-slot producer/consumer state machine; and
* the thin DeepSpeed data-parallel-group adapter required by ZeRO-1.

The adapter is not a model-parallel or ZeRO-3 abstraction.  It only tells
DeepSpeed that Student ranks 4--7, rather than the global eight-rank WORLD
group, form the ZeRO-1 data-parallel group.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, Sequence

import torch


WIRE_MAGIC = 0x4453504F4F4C  # "DSPOOL"
WIRE_SCHEMA_VERSION = 1
X0_SHAPE = (1, 16, 9, 48, 80)
BF16_DTYPE_CODE = 1
HEADER_I64_COUNT = 31
ACK_I64_COUNT = 17


class ChecksumKind(enum.IntEnum):
    """Checksums carried by the fixed-width wire protocol."""

    SHA256 = 1
    GPU_FINGERPRINT_V1 = 2


@dataclass(frozen=True)
class CriticPoolTopology:
    """Global ranks and pair mapping for the first symmetric pool candidate."""

    world_size: int
    generator_ranks: tuple[int, ...]
    student_ranks: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.world_size <= 0:
            raise ValueError("world_size must be positive")
        if len(self.generator_ranks) != len(self.student_ranks):
            raise ValueError("the first pool candidate requires equal-width roles")
        if not self.generator_ranks:
            raise ValueError("role groups cannot be empty")
        all_ranks = self.generator_ranks + self.student_ranks
        if len(set(all_ranks)) != len(all_ranks):
            raise ValueError("Generator and Student ranks must be disjoint")
        if any(rank < 0 or rank >= self.world_size for rank in all_ranks):
            raise ValueError("role rank is outside WORLD")
        if set(all_ranks) != set(range(self.world_size)):
            raise ValueError("every WORLD rank must have exactly one pool role")

    @classmethod
    def single_node_4g4s(cls) -> "CriticPoolTopology":
        return cls(
            world_size=8,
            generator_ranks=(0, 1, 2, 3),
            student_ranks=(4, 5, 6, 7),
        )

    @property
    def role_width(self) -> int:
        return len(self.generator_ranks)

    def paired_student(self, generator_global_rank: int) -> int:
        try:
            index = self.generator_ranks.index(generator_global_rank)
        except ValueError as error:
            raise ValueError("rank is not a Generator rank") from error
        return self.student_ranks[index]

    def paired_generator(self, student_global_rank: int) -> int:
        try:
            index = self.student_ranks.index(student_global_rank)
        except ValueError as error:
            raise ValueError("rank is not a Student rank") from error
        return self.generator_ranks[index]

    def pool_rank(self, global_rank: int) -> int:
        if global_rank in self.generator_ranks:
            return self.generator_ranks.index(global_rank)
        if global_rank in self.student_ranks:
            return self.student_ranks.index(global_rank)
        raise ValueError("rank is outside the pool topology")


@dataclass(frozen=True)
class CriticBatchContract:
    """The optimizer semantics frozen for the symmetric single-node run."""

    local_batch: int = 1
    student_dp_world_size: int = 4
    gradient_accumulation_steps: int = 1
    waves: int = 4

    def __post_init__(self) -> None:
        if min(
            self.local_batch,
            self.student_dp_world_size,
            self.gradient_accumulation_steps,
            self.waves,
        ) <= 0:
            raise ValueError("batch contract fields must be positive")
        if self.gradient_accumulation_steps != 1:
            raise ValueError("Critic-only v1 forbids merging logical updates")

    @property
    def global_batch_per_wave(self) -> int:
        return self.local_batch * self.student_dp_world_size

    @property
    def logical_samples(self) -> int:
        return self.global_batch_per_wave * self.waves

    @property
    def train_batch_size(self) -> int:
        return (
            self.local_batch
            * self.student_dp_world_size
            * self.gradient_accumulation_steps
        )


def _digest_bytes(value: bytes) -> tuple[int, int, int, int]:
    if len(value) != 32:
        raise ValueError("digest must contain exactly 32 bytes")
    return tuple(
        int.from_bytes(value[offset : offset + 8], "little", signed=True)
        for offset in range(0, 32, 8)
    )  # type: ignore[return-value]


def _digest_from_i64(values: Sequence[int]) -> bytes:
    if len(values) != 4:
        raise ValueError("digest representation must contain four int64 values")
    return b"".join(
        int(value).to_bytes(8, "little", signed=True) for value in values
    )


@dataclass(frozen=True)
class CriticWaveHeader:
    """Fixed-width metadata sent before one rank-local ``generated_x0``."""

    run_nonce: int
    cycle: int
    logical_step: int
    wave: int
    producer_global_rank: int
    producer_pool_rank: int
    student_global_rank: int
    student_pool_rank: int
    generator_version: int
    expected_critic_before: int
    expected_critic_after: int
    trace_sha256: bytes
    payload_checksum: bytes
    checksum_kind: ChecksumKind = ChecksumKind.SHA256
    shape: tuple[int, int, int, int, int] = X0_SHAPE
    dtype_code: int = BF16_DTYPE_CODE
    local_batch: int = 1

    def __post_init__(self) -> None:
        if self.wave < 0 or self.logical_step < 0:
            raise ValueError("wave and logical step must be non-negative")
        if self.expected_critic_after != self.expected_critic_before + 1:
            raise ValueError("one wave must advance exactly one Critic version")
        if self.local_batch != 1:
            raise ValueError("the first pool wire protocol fixes local batch to one")
        if self.shape != X0_SHAPE:
            raise ValueError(f"official x0 shape required: {self.shape} != {X0_SHAPE}")
        if self.dtype_code != BF16_DTYPE_CODE:
            raise ValueError("the first pool wire protocol requires BF16 x0")
        _digest_bytes(self.trace_sha256)
        _digest_bytes(self.payload_checksum)

    @property
    def numel(self) -> int:
        result = 1
        for dimension in self.shape:
            result *= dimension
        return result

    @property
    def payload_bytes(self) -> int:
        return self.numel * 2

    def to_tensor(self, device: torch.device | str = "cpu") -> torch.Tensor:
        values = [
            WIRE_MAGIC,
            WIRE_SCHEMA_VERSION,
            self.run_nonce,
            self.cycle,
            self.logical_step,
            self.wave,
            self.producer_global_rank,
            self.producer_pool_rank,
            self.student_global_rank,
            self.student_pool_rank,
            self.generator_version,
            self.expected_critic_before,
            self.expected_critic_after,
            self.local_batch,
            *self.shape,
            self.dtype_code,
            self.numel,
            self.payload_bytes,
            int(self.checksum_kind),
            *_digest_bytes(self.trace_sha256),
            *_digest_bytes(self.payload_checksum),
        ]
        if len(values) != HEADER_I64_COUNT:
            raise AssertionError("wire header width changed")
        return torch.tensor(values, dtype=torch.int64, device=device)

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "CriticWaveHeader":
        if tensor.dtype != torch.int64 or tensor.numel() != HEADER_I64_COUNT:
            raise ValueError("invalid wire header tensor")
        values = [int(value) for value in tensor.detach().cpu().reshape(-1).tolist()]
        if values[0] != WIRE_MAGIC or values[1] != WIRE_SCHEMA_VERSION:
            raise ValueError("wire magic/schema mismatch")
        shape = tuple(values[14:19])
        result = cls(
            run_nonce=values[2],
            cycle=values[3],
            logical_step=values[4],
            wave=values[5],
            producer_global_rank=values[6],
            producer_pool_rank=values[7],
            student_global_rank=values[8],
            student_pool_rank=values[9],
            generator_version=values[10],
            expected_critic_before=values[11],
            expected_critic_after=values[12],
            local_batch=values[13],
            shape=shape,  # type: ignore[arg-type]
            dtype_code=values[19],
            checksum_kind=ChecksumKind(values[22]),
            trace_sha256=_digest_from_i64(values[23:27]),
            payload_checksum=_digest_from_i64(values[27:31]),
        )
        if values[20] != result.numel or values[21] != result.payload_bytes:
            raise ValueError("wire payload size metadata mismatch")
        return result


@dataclass(frozen=True)
class CriticCommitAck:
    """Commit acknowledgement emitted only after one Student optimizer step."""

    run_nonce: int
    cycle: int
    logical_step: int
    wave: int
    producer_global_rank: int
    student_global_rank: int
    observed_generator_version: int
    critic_before: int
    critic_after: int
    optimizer_step_before: int
    optimizer_step_after: int
    scheduler_step_before: int
    scheduler_step_after: int
    status: int = 0
    error_code: int = 0

    def __post_init__(self) -> None:
        if self.critic_after != self.critic_before + 1:
            raise ValueError("commit ack must advance exactly one Critic version")
        if self.optimizer_step_after != self.optimizer_step_before + 1:
            raise ValueError("commit ack must advance exactly one optimizer step")
        if self.scheduler_step_after != self.scheduler_step_before + 1:
            raise ValueError("commit ack must advance exactly one scheduler step")

    def to_tensor(self, device: torch.device | str = "cpu") -> torch.Tensor:
        values = [
            WIRE_MAGIC,
            WIRE_SCHEMA_VERSION,
            self.run_nonce,
            self.cycle,
            self.logical_step,
            self.wave,
            self.producer_global_rank,
            self.student_global_rank,
            self.observed_generator_version,
            self.critic_before,
            self.critic_after,
            self.optimizer_step_before,
            self.optimizer_step_after,
            self.scheduler_step_before,
            self.scheduler_step_after,
            self.status,
            self.error_code,
        ]
        if len(values) != ACK_I64_COUNT:
            raise AssertionError("wire ack width changed")
        return torch.tensor(values, dtype=torch.int64, device=device)

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "CriticCommitAck":
        if tensor.dtype != torch.int64 or tensor.numel() != ACK_I64_COUNT:
            raise ValueError("invalid commit ack tensor")
        values = [int(value) for value in tensor.detach().cpu().reshape(-1).tolist()]
        if values[0] != WIRE_MAGIC or values[1] != WIRE_SCHEMA_VERSION:
            raise ValueError("commit ack magic/schema mismatch")
        return cls(
            run_nonce=values[2],
            cycle=values[3],
            logical_step=values[4],
            wave=values[5],
            producer_global_rank=values[6],
            student_global_rank=values[7],
            observed_generator_version=values[8],
            critic_before=values[9],
            critic_after=values[10],
            optimizer_step_before=values[11],
            optimizer_step_after=values[12],
            scheduler_step_before=values[13],
            scheduler_step_after=values[14],
            status=values[15],
            error_code=values[16],
        )


class ProducerState(str, enum.Enum):
    EMPTY = "empty"
    PRODUCING = "producing"
    READY = "ready"
    SENDING = "sending"


class ConsumerState(str, enum.Enum):
    EMPTY = "empty"
    RECEIVING = "receiving"
    VALIDATED = "validated"
    CONSUMING = "consuming"
    OPTIMIZING = "optimizing"
    COMMITTED = "committed"
    ACK_SENT = "ack_sent"


@dataclass
class CriticOnlyQueueState:
    """Two-slot queue accounting for one paired Generator/Student rank."""

    waves: int = 4
    produced: int = 0
    sent: int = 0
    received: int = 0
    consumed: int = 0
    committed: int = 0
    outstanding_work: int = 0
    producer_state: ProducerState = ProducerState.EMPTY
    producer_wave: int | None = None
    consumer_state: ConsumerState = ConsumerState.EMPTY
    consumer_wave: int | None = None

    def _check_bounds(self) -> None:
        counts = (
            self.produced,
            self.sent,
            self.received,
            self.consumed,
            self.committed,
        )
        if not all(0 <= value <= self.waves for value in counts):
            raise AssertionError("queue count is outside the cycle")
        if not (
            self.produced >= self.sent >= self.received >= self.consumed >= self.committed
        ):
            raise AssertionError("queue events are out of order")
        if self.produced - self.sent > 1:
            raise AssertionError("ProducerSlot holds more than one ready batch")
        if self.sent - self.committed > 1:
            raise AssertionError("Student has more than one uncommitted batch")
        if self.produced - self.committed > 2:
            raise AssertionError("queue depth exceeds the two-slot contract")
        if self.outstanding_work < 0:
            raise AssertionError("outstanding work cannot be negative")

    def begin_produce(self, wave: int) -> None:
        if self.producer_state is not ProducerState.EMPTY:
            raise RuntimeError("ProducerSlot is not empty")
        if wave != self.produced or wave >= self.waves:
            raise RuntimeError("unexpected producer wave")
        self.producer_state = ProducerState.PRODUCING
        self.producer_wave = wave
        self._check_bounds()

    def mark_ready(self, wave: int) -> None:
        if (
            self.producer_state is not ProducerState.PRODUCING
            or self.producer_wave != wave
        ):
            raise RuntimeError("batch was not being produced")
        self.producer_state = ProducerState.READY
        self.produced += 1
        self._check_bounds()

    def begin_send(self, wave: int) -> None:
        if self.producer_state is not ProducerState.READY or self.producer_wave != wave:
            raise RuntimeError("batch is not ready for send")
        if self.sent != self.committed:
            raise RuntimeError("current Student commit ACK is required before next send")
        self.producer_state = ProducerState.SENDING
        self.outstanding_work += 1
        self._check_bounds()

    def finish_send(self, wave: int) -> None:
        if self.producer_state is not ProducerState.SENDING or self.producer_wave != wave:
            raise RuntimeError("batch send is not active")
        self.sent += 1
        self.outstanding_work -= 1
        self.producer_state = ProducerState.EMPTY
        self.producer_wave = None
        self._check_bounds()

    def observe_remote_send(self, wave: int) -> None:
        """Mirror one producer delivery on the paired Student process."""
        if wave != self.produced or self.produced != self.sent:
            raise RuntimeError("unexpected mirrored producer delivery")
        self.produced += 1
        self.sent += 1
        self._check_bounds()

    def begin_receive(self, wave: int) -> None:
        if self.consumer_state is not ConsumerState.EMPTY:
            raise RuntimeError("ConsumerSlot is not empty")
        if wave != self.received or wave >= self.waves:
            raise RuntimeError("unexpected consumer wave")
        self.consumer_state = ConsumerState.RECEIVING
        self.consumer_wave = wave
        self.outstanding_work += 1
        self._check_bounds()

    def finish_receive(self, wave: int) -> None:
        if (
            self.consumer_state is not ConsumerState.RECEIVING
            or self.consumer_wave != wave
        ):
            raise RuntimeError("batch receive is not active")
        self.received += 1
        self.outstanding_work -= 1
        self.consumer_state = ConsumerState.VALIDATED
        self._check_bounds()

    def begin_consume(self, wave: int) -> None:
        if (
            self.consumer_state is not ConsumerState.VALIDATED
            or self.consumer_wave != wave
        ):
            raise RuntimeError("batch has not passed validation")
        self.consumer_state = ConsumerState.CONSUMING
        self._check_bounds()

    def mark_consumed(self, wave: int) -> None:
        if (
            self.consumer_state is not ConsumerState.CONSUMING
            or self.consumer_wave != wave
        ):
            raise RuntimeError("batch is not being consumed")
        self.consumed += 1
        self.consumer_state = ConsumerState.OPTIMIZING
        self._check_bounds()

    def mark_committed(self, wave: int) -> None:
        if (
            self.consumer_state is not ConsumerState.OPTIMIZING
            or self.consumer_wave != wave
        ):
            raise RuntimeError("optimizer commit is not ready")
        self.committed += 1
        self.consumer_state = ConsumerState.COMMITTED
        self._check_bounds()

    def begin_ack(self, wave: int) -> None:
        if (
            self.consumer_state is not ConsumerState.COMMITTED
            or self.consumer_wave != wave
        ):
            raise RuntimeError("commit ACK is not ready")
        self.consumer_state = ConsumerState.ACK_SENT
        self._check_bounds()

    def finish_ack(self, wave: int) -> None:
        if (
            self.consumer_state is not ConsumerState.ACK_SENT
            or self.consumer_wave != wave
        ):
            raise RuntimeError("commit ACK is not active")
        self.consumer_state = ConsumerState.EMPTY
        self.consumer_wave = None
        self._check_bounds()

    def observe_remote_commit(self, wave: int) -> None:
        """Mirror a validated Student ACK on the paired Generator process."""
        if wave != self.committed:
            raise RuntimeError("unexpected mirrored Student commit")
        expected = wave + 1
        if self.sent != expected:
            raise RuntimeError("Student ACK arrived before the wave was sent")
        if not (
            self.received == self.consumed == self.committed == wave
        ):
            raise RuntimeError("mirrored Student counters are inconsistent")
        self.received = expected
        self.consumed = expected
        self.committed = expected
        self._check_bounds()

    def assert_flushed(
        self,
        *,
        generator_before: int,
        generator_after: int,
        critic_before: int,
        critic_after: int,
    ) -> None:
        self._check_bounds()
        expected = (self.waves,) * 5
        actual = (
            self.produced,
            self.sent,
            self.received,
            self.consumed,
            self.committed,
        )
        if actual != expected:
            raise AssertionError(f"cycle is not fully consumed: {actual} != {expected}")
        if (
            self.producer_state is not ProducerState.EMPTY
            or self.producer_wave is not None
        ):
            raise AssertionError("ProducerSlot is not empty at flush")
        if (
            self.consumer_state is not ConsumerState.EMPTY
            or self.consumer_wave is not None
        ):
            raise AssertionError("ConsumerSlot is not empty at flush")
        if self.outstanding_work:
            raise AssertionError("NCCL work remains at flush")
        if generator_after != generator_before:
            raise AssertionError("Generator version advanced in a Critic-only cycle")
        if critic_after != critic_before + self.waves:
            raise AssertionError("Critic version did not advance once per wave")


class StudentDPGroupAdapter:
    """Minimal DeepSpeed ``mpu=`` adapter for a Student-only ZeRO-1 group."""

    def __init__(
        self,
        *,
        data_parallel_group: Any,
        data_parallel_ranks: Sequence[int],
        global_rank: int,
        singleton_model_parallel_group: Any,
    ) -> None:
        ranks = tuple(int(rank) for rank in data_parallel_ranks)
        if global_rank not in ranks:
            raise ValueError("global rank is outside the Student DP group")
        if len(set(ranks)) != len(ranks) or not ranks:
            raise ValueError("Student DP ranks must be unique and non-empty")
        self._dp_group = data_parallel_group
        self._dp_ranks = ranks
        self._global_rank = int(global_rank)
        self._mp_group = singleton_model_parallel_group

    def get_data_parallel_group(self) -> Any:
        return self._dp_group

    def get_data_parallel_rank(self) -> int:
        return self._dp_ranks.index(self._global_rank)

    def get_data_parallel_world_size(self) -> int:
        return len(self._dp_ranks)

    def get_data_parallel_group_ranks(self) -> list[int]:
        return list(self._dp_ranks)

    def get_model_parallel_group(self) -> Any:
        return self._mp_group

    def get_model_parallel_rank(self) -> int:
        return 0

    def get_model_parallel_world_size(self) -> int:
        return 1

    get_tensor_model_parallel_group = get_model_parallel_group
    get_tensor_model_parallel_rank = get_model_parallel_rank
    get_tensor_model_parallel_world_size = get_model_parallel_world_size
    get_pipeline_model_parallel_group = get_model_parallel_group
    get_pipeline_model_parallel_rank = get_model_parallel_rank
    get_pipeline_model_parallel_world_size = get_model_parallel_world_size


def _resolved_engine_int(engine: Any, name: str) -> int:
    value = getattr(engine, name)
    if callable(value):
        value = value()
    return int(value)


def assert_student_zero1_contract(
    engine: Any,
    *,
    expected_dp_group: Any,
    contract: CriticBatchContract = CriticBatchContract(),
) -> dict[str, int]:
    """Fail closed on DeepSpeed's resolved, rather than requested, batch values."""

    resolved = {
        "dp_world_size": _resolved_engine_int(engine, "dp_world_size"),
        "train_micro_batch_size_per_gpu": _resolved_engine_int(
            engine, "train_micro_batch_size_per_gpu"
        ),
        "gradient_accumulation_steps": _resolved_engine_int(
            engine, "gradient_accumulation_steps"
        ),
        "train_batch_size": _resolved_engine_int(engine, "train_batch_size"),
    }
    expected = {
        "dp_world_size": contract.student_dp_world_size,
        "train_micro_batch_size_per_gpu": contract.local_batch,
        "gradient_accumulation_steps": contract.gradient_accumulation_steps,
        "train_batch_size": contract.train_batch_size,
    }
    if resolved != expected:
        raise AssertionError(f"resolved DeepSpeed batch contract {resolved} != {expected}")
    optimizer_group = getattr(engine.optimizer, "dp_process_group", None)
    if optimizer_group is not expected_dp_group:
        raise AssertionError("ZeRO-1 optimizer escaped the Student DP group")
    return resolved


@dataclass(frozen=True)
class EmaExperimentRecord:
    """Explicitly records that early-step EMA is outside this isolated subgraph."""

    input_sha256: str
    output_sha256: str
    optimization_step_before: int
    optimization_step_after: int
    execution: str = "skipped_by_experiment_contract"

    def __post_init__(self) -> None:
        if self.execution != "skipped_by_experiment_contract":
            raise ValueError("Critic-only v1 must explicitly freeze EMA")
        if self.input_sha256 != self.output_sha256:
            raise ValueError("skipped EMA changed its state hash")
        if self.optimization_step_before != self.optimization_step_after:
            raise ValueError("skipped EMA advanced its optimization step")
