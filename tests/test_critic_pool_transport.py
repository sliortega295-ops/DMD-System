from __future__ import annotations

import pytest
import torch

from runtime.critic_pool import ChecksumKind, CriticCommitAck, CriticWaveHeader
from runtime.critic_pool_transport import (
    CriticPoolP2PTransport,
    assert_ack_matches_header,
    payload_checksum,
    prepare_transport_payload,
    tensor_identity,
    validate_payload,
)


def _header(x0: torch.Tensor, kind: ChecksumKind) -> CriticWaveHeader:
    return CriticWaveHeader(
        run_nonce=77,
        cycle=0,
        logical_step=11,
        wave=0,
        producer_global_rank=0,
        producer_pool_rank=0,
        student_global_rank=4,
        student_pool_rank=0,
        generator_version=3,
        expected_critic_before=11,
        expected_critic_after=12,
        trace_sha256=b"a" * 32,
        payload_checksum=payload_checksum(x0, kind),
        checksum_kind=kind,
    )


@pytest.mark.parametrize(
    "kind", [ChecksumKind.SHA256, ChecksumKind.GPU_FINGERPRINT_V1]
)
def test_payload_checksum_detects_value_change(kind: ChecksumKind) -> None:
    x0 = torch.arange(552_960, dtype=torch.float32).reshape(1, 16, 9, 48, 80)
    x0 = (x0 / 1024).to(torch.bfloat16)
    header = _header(x0, kind)
    validate_payload(x0, header)
    changed = x0.clone()
    changed.reshape(-1)[123] += 1
    with pytest.raises(ValueError, match="checksum"):
        validate_payload(changed, header)


@pytest.mark.parametrize(
    "kind", [ChecksumKind.SHA256, ChecksumKind.GPU_FINGERPRINT_V1]
)
def test_generic_tensor_identity_does_not_require_x0_shape(kind: ChecksumKind) -> None:
    value = torch.arange(37, dtype=torch.float32).reshape(1, 37)
    changed = value.clone()
    changed[0, 3] += 1
    identity = tensor_identity(value, kind)
    assert len(identity) == 32
    assert identity != tensor_identity(changed, kind)


def test_commit_ack_must_match_wave_header() -> None:
    x0 = torch.zeros((1, 16, 9, 48, 80), dtype=torch.bfloat16)
    header = _header(x0, ChecksumKind.SHA256)
    ack = CriticCommitAck(
        run_nonce=77, cycle=0, logical_step=11, wave=0,
        producer_global_rank=0, student_global_rank=4,
        observed_generator_version=3, critic_before=11, critic_after=12,
        optimizer_step_before=11, optimizer_step_after=12,
        scheduler_step_before=11, scheduler_step_after=12,
    )
    assert_ack_matches_header(ack, header)
    wrong = CriticCommitAck(
        run_nonce=77, cycle=0, logical_step=11, wave=0,
        producer_global_rank=0, student_global_rank=4,
        observed_generator_version=4, critic_before=11, critic_after=12,
        optimizer_step_before=11, optimizer_step_after=12,
        scheduler_step_before=11, scheduler_step_after=12,
    )
    with pytest.raises(ValueError, match="mismatch"):
        assert_ack_matches_header(wrong, header)


def test_transport_materializes_non_dense_helios_x0_view() -> None:
    base = torch.arange(552_960 * 2, dtype=torch.float32).reshape(1, 16, 9, 48, 160)
    x0 = (base / 1024).to(torch.bfloat16)[..., ::2]
    assert tuple(x0.shape) == (1, 16, 9, 48, 80)
    assert not x0.is_contiguous()
    payload = prepare_transport_payload(x0)
    header = _header(payload, ChecksumKind.SHA256)
    assert payload.is_contiguous()
    assert torch.equal(payload, x0)
    assert payload_checksum(payload, ChecksumKind.SHA256) == header.payload_checksum


def test_ack_can_use_cpu_gloo_control_plane(monkeypatch: pytest.MonkeyPatch) -> None:
    data_group = object()
    ack_group = object()
    calls: list[tuple[torch.Tensor, int, object]] = []

    class _Work:
        def wait(self) -> None:
            return None

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_backend", lambda group: "gloo")

    def _isend(tensor: torch.Tensor, *, dst: int, group: object) -> _Work:
        calls.append((tensor, dst, group))
        return _Work()

    monkeypatch.setattr(torch.distributed, "isend", _isend)
    transport = CriticPoolP2PTransport(
        group=data_group,
        device="cuda:0",
        ack_group=ack_group,
        ack_device="cpu",
    )
    pending = transport.send_ack(_ack_for_transport_test(), destination_global_rank=0)
    pending.wait()
    assert transport.ack_backend == "gloo"
    assert len(calls) == 1
    assert calls[0][0].device.type == "cpu"
    assert calls[0][1:] == (0, ack_group)


def _ack_for_transport_test() -> CriticCommitAck:
    return CriticCommitAck(
        run_nonce=77, cycle=0, logical_step=11, wave=0,
        producer_global_rank=0, student_global_rank=4,
        observed_generator_version=3, critic_before=11, critic_after=12,
        optimizer_step_before=11, optimizer_step_after=12,
        scheduler_step_before=11, scheduler_step_after=12,
    )
