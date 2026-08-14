import pytest

from runtime.critic_pool import ChecksumKind, CriticWaveHeader
from runtime.routeb_full5_transport import (
    RouteBWaveAck,
    assert_ack_matches_header,
)


def _header() -> CriticWaveHeader:
    return CriticWaveHeader(
        run_nonce=7,
        cycle=0,
        logical_step=10,
        wave=0,
        producer_global_rank=0,
        producer_pool_rank=0,
        student_global_rank=8,
        student_pool_rank=0,
        generator_version=3,
        expected_critic_before=10,
        expected_critic_after=11,
        trace_sha256=b"t" * 32,
        payload_checksum=b"x" * 32,
        checksum_kind=ChecksumKind.SHA256,
    )


def test_nonboundary_and_boundary_ack_roundtrip() -> None:
    header = _header()
    nonboundary = RouteBWaveAck(
        run_nonce=7,
        cycle=0,
        logical_step=10,
        micro_wave=0,
        producer_global_rank=0,
        critic_global_rank=8,
        observed_generator_version=3,
        critic_before=10,
        critic_after=10,
        optimizer_step_before=10,
        optimizer_step_after=10,
        scheduler_step_before=10,
        scheduler_step_after=10,
        optimizer_committed=False,
    )
    decoded = RouteBWaveAck.from_tensor(nonboundary.to_tensor())
    assert decoded == nonboundary
    assert_ack_matches_header(decoded, header, expected_commit=False)

    committed = RouteBWaveAck(
        run_nonce=7,
        cycle=0,
        logical_step=10,
        micro_wave=0,
        producer_global_rank=0,
        critic_global_rank=8,
        observed_generator_version=3,
        critic_before=10,
        critic_after=11,
        optimizer_step_before=10,
        optimizer_step_after=11,
        scheduler_step_before=10,
        scheduler_step_after=11,
        optimizer_committed=True,
    )
    assert_ack_matches_header(committed, header, expected_commit=True)


def test_ack_rejects_version_advance_without_commit() -> None:
    with pytest.raises(ValueError, match="version delta"):
        RouteBWaveAck(
            run_nonce=7,
            cycle=0,
            logical_step=10,
            micro_wave=0,
            producer_global_rank=0,
            critic_global_rank=8,
            observed_generator_version=3,
            critic_before=10,
            critic_after=11,
            optimizer_step_before=10,
            optimizer_step_after=10,
            scheduler_step_before=10,
            scheduler_step_after=10,
            optimizer_committed=False,
        )
