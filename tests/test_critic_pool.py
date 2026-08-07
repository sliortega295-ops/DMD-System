from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from runtime.critic_pool import (
    ACK_I64_COUNT,
    HEADER_I64_COUNT,
    BF16_DTYPE_CODE,
    X0_SHAPE,
    CriticBatchContract,
    CriticCommitAck,
    CriticOnlyQueueState,
    CriticPoolTopology,
    CriticWaveHeader,
    EmaExperimentRecord,
    StudentDPGroupAdapter,
    assert_student_zero1_contract,
)


def _digest(byte: int) -> bytes:
    return bytes([byte]) * 32


def _header(wave: int = 0) -> CriticWaveHeader:
    return CriticWaveHeader(
        run_nonce=17, cycle=2, logical_step=11 + wave, wave=wave,
        producer_global_rank=wave, producer_pool_rank=wave,
        student_global_rank=wave + 4, student_pool_rank=wave,
        generator_version=3, expected_critic_before=11 + wave,
        expected_critic_after=12 + wave,
        trace_sha256=_digest(0xA0 + wave), payload_checksum=_digest(0xB0 + wave),
    )


def _ack(wave: int = 0) -> CriticCommitAck:
    return CriticCommitAck(
        run_nonce=17, cycle=2, logical_step=11 + wave, wave=wave,
        producer_global_rank=wave, student_global_rank=wave + 4,
        observed_generator_version=3, critic_before=11 + wave, critic_after=12 + wave,
        optimizer_step_before=11 + wave, optimizer_step_after=12 + wave,
        scheduler_step_before=11 + wave, scheduler_step_after=12 + wave,
    )


def test_symmetric_topology_and_batch_contract() -> None:
    topology = CriticPoolTopology.single_node_4g4s(); contract = CriticBatchContract()
    assert topology.role_width == 4
    assert topology.paired_student(2) == 6
    assert topology.paired_generator(7) == 3
    assert topology.pool_rank(5) == 1
    assert contract.global_batch_per_wave == 4
    assert contract.logical_samples == 16
    assert contract.train_batch_size == 4


@pytest.mark.parametrize("kwargs", [
    {"world_size": 8, "generator_ranks": (0, 1, 2), "student_ranks": (4, 5, 6, 7)},
    {"world_size": 8, "generator_ranks": (0, 1, 2, 3), "student_ranks": (3, 5, 6, 7)},
    {"world_size": 8, "generator_ranks": (0, 1, 2, 3), "student_ranks": (4, 5, 6, 9)},
])
def test_topology_rejects_non_symmetric_or_invalid_roles(kwargs: dict) -> None:
    with pytest.raises(ValueError): CriticPoolTopology(**kwargs)


def test_wave_header_roundtrip_and_official_payload_contract() -> None:
    header = _header(); wire = header.to_tensor(); decoded = CriticWaveHeader.from_tensor(wire)
    assert wire.shape == (HEADER_I64_COUNT,)
    assert decoded == header
    assert decoded.shape == X0_SHAPE
    assert decoded.dtype_code == BF16_DTYPE_CODE
    assert decoded.numel == 552_960
    assert decoded.payload_bytes == 1_105_920


def test_wave_header_rejects_corruption() -> None:
    wire = _header().to_tensor(); wire[0] = 0
    with pytest.raises(ValueError, match="magic"): CriticWaveHeader.from_tensor(wire)
    wire = _header().to_tensor(); wire[21] += 2
    with pytest.raises(ValueError, match="payload size"): CriticWaveHeader.from_tensor(wire)


def test_commit_ack_roundtrip() -> None:
    ack = _ack(); wire = ack.to_tensor()
    assert wire.shape == (ACK_I64_COUNT,)
    assert CriticCommitAck.from_tensor(wire) == ack


def test_two_slot_queue_allows_one_produced_batch_ahead() -> None:
    queue = CriticOnlyQueueState(waves=4)
    for wave in range(4):
        queue.begin_produce(wave); queue.mark_ready(wave); queue.begin_send(wave); queue.finish_send(wave)
        queue.begin_receive(wave); queue.finish_receive(wave); queue.begin_consume(wave)
        if wave + 1 < queue.waves:
            queue.begin_produce(wave + 1); queue.mark_ready(wave + 1)
        queue.mark_consumed(wave); queue.mark_committed(wave); queue.begin_ack(wave); queue.finish_ack(wave)
        if wave + 1 < queue.waves:
            queue.begin_send(wave + 1); queue.finish_send(wave + 1); queue.begin_receive(wave + 1); queue.finish_receive(wave + 1); queue.begin_consume(wave + 1)
            queue.mark_consumed(wave + 1); queue.mark_committed(wave + 1); queue.begin_ack(wave + 1); queue.finish_ack(wave + 1)
            for remaining in range(wave + 2, queue.waves):
                queue.begin_produce(remaining); queue.mark_ready(remaining); queue.begin_send(remaining); queue.finish_send(remaining)
                queue.begin_receive(remaining); queue.finish_receive(remaining); queue.begin_consume(remaining); queue.mark_consumed(remaining)
                queue.mark_committed(remaining); queue.begin_ack(remaining); queue.finish_ack(remaining)
            break
    queue.assert_flushed(generator_before=3, generator_after=3, critic_before=11, critic_after=15)


def test_queue_requires_ack_before_next_send_and_rejects_false_flush() -> None:
    queue = CriticOnlyQueueState(waves=4)
    queue.begin_produce(0); queue.mark_ready(0); queue.begin_send(0); queue.finish_send(0)
    queue.begin_receive(0); queue.finish_receive(0); queue.begin_consume(0)
    queue.begin_produce(1); queue.mark_ready(1)
    with pytest.raises(RuntimeError, match="ACK"): queue.begin_send(1)
    with pytest.raises(AssertionError, match="not fully consumed"):
        queue.assert_flushed(generator_before=3, generator_after=3, critic_before=11, critic_after=15)


def test_split_process_queue_mirrors_delivery_and_commit() -> None:
    producer = CriticOnlyQueueState(waves=1)
    producer.begin_produce(0); producer.mark_ready(0); producer.begin_send(0); producer.finish_send(0)
    consumer = CriticOnlyQueueState(waves=1)
    consumer.observe_remote_send(0); consumer.begin_receive(0); consumer.finish_receive(0); consumer.begin_consume(0)
    consumer.mark_consumed(0); consumer.mark_committed(0); consumer.begin_ack(0); consumer.finish_ack(0)
    producer.observe_remote_commit(0)
    for state in (producer, consumer): state.assert_flushed(generator_before=3, generator_after=3, critic_before=11, critic_after=12)


def test_student_mpu_adapter_only_exposes_student_dp_group() -> None:
    dp_group = object(); mp_group = object()
    adapter = StudentDPGroupAdapter(data_parallel_group=dp_group, data_parallel_ranks=(4, 5, 6, 7), global_rank=6, singleton_model_parallel_group=mp_group)
    assert adapter.get_data_parallel_group() is dp_group
    assert adapter.get_data_parallel_rank() == 2
    assert adapter.get_data_parallel_world_size() == 4
    assert adapter.get_data_parallel_group_ranks() == [4, 5, 6, 7]
    assert adapter.get_model_parallel_group() is mp_group
    assert adapter.get_model_parallel_rank() == 0
    assert adapter.get_model_parallel_world_size() == 1


class _FakeEngine:
    def __init__(self, dp_group: object) -> None:
        self.dp_world_size = 4
        self.train_micro_batch_size_per_gpu = lambda: 1
        self.gradient_accumulation_steps = lambda: 1
        self.train_batch_size = lambda: 4
        self.optimizer = SimpleNamespace(dp_process_group=dp_group)


def test_deepspeed_resolved_batch_contract_is_fail_closed() -> None:
    dp_group = object(); engine = _FakeEngine(dp_group)
    assert assert_student_zero1_contract(engine, expected_dp_group=dp_group) == {
        "dp_world_size": 4, "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1, "train_batch_size": 4,
    }
    engine.train_batch_size = lambda: 8
    with pytest.raises(AssertionError, match="batch contract"): assert_student_zero1_contract(engine, expected_dp_group=dp_group)


def test_early_step_ema_skip_is_explicit_and_immutable() -> None:
    record = EmaExperimentRecord(input_sha256="a" * 64, output_sha256="a" * 64, optimization_step_before=11, optimization_step_after=11)
    assert record.execution == "skipped_by_experiment_contract"
    with pytest.raises(ValueError, match="changed"):
        EmaExperimentRecord(input_sha256="a" * 64, output_sha256="b" * 64, optimization_step_before=11, optimization_step_after=11)
