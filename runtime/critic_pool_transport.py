"""Bounded P2P transport shared by sequential and pipelined pool executors."""
from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from .critic_pool import (
    ACK_I64_COUNT,
    HEADER_I64_COUNT,
    X0_SHAPE,
    ChecksumKind,
    CriticCommitAck,
    CriticWaveHeader,
)
from .operation_trace import tensor_sha256


def tensor_identity(tensor: torch.Tensor, kind: ChecksumKind) -> bytes:
    """Return one fixed-width identity without assuming the x0 wire shape.

    SHA-256 remains the correctness identity.  ``GPU_FINGERPRINT_V1`` reduces
    on the source device and copies only four scalars to CPU; it is a timing
    corruption guard, not cryptographic evidence.
    """

    value = tensor.detach()
    if kind is ChecksumKind.SHA256:
        return bytes.fromhex(tensor_sha256(value))
    if kind is not ChecksumKind.GPU_FINGERPRINT_V1:
        raise ValueError(f"unsupported checksum kind {kind}")
    if value.numel() == 0:
        raise ValueError("cannot fingerprint an empty tensor")

    flat = value.reshape(-1).float()
    scalars = torch.stack(
        (
            flat.sum(dtype=torch.float64),
            flat.abs().sum(dtype=torch.float64),
            flat.square().sum(dtype=torch.float64),
            flat[::4096].sum(dtype=torch.float64),
        )
    ).cpu()
    raw = b"".join(struct.pack("<d", float(item)) for item in scalars)
    return hashlib.sha256(raw).digest()


def payload_checksum(
    tensor: torch.Tensor, kind: ChecksumKind
) -> bytes:
    """Return a fixed 32-byte identity for the official x0 wire payload."""

    if tuple(tensor.shape) != X0_SHAPE or tensor.dtype != torch.bfloat16:
        raise ValueError("transport payload must be official-shape BF16 x0")
    return tensor_identity(tensor, kind)


def validate_payload(
    tensor: torch.Tensor,
    header: CriticWaveHeader,
) -> None:
    if tuple(tensor.shape) != header.shape:
        raise ValueError("received x0 shape differs from its header")
    if tensor.dtype != torch.bfloat16:
        raise ValueError("received x0 dtype is not BF16")
    actual = payload_checksum(tensor, header.checksum_kind)
    if actual != header.payload_checksum:
        raise ValueError("received x0 checksum mismatch")


def prepare_transport_payload(tensor: torch.Tensor) -> torch.Tensor:
    """Validate x0 and return storage that NCCL P2P can legally consume."""

    if tuple(tensor.shape) != X0_SHAPE or tensor.dtype != torch.bfloat16:
        raise ValueError("transport payload must be official-shape BF16 x0")
    payload = tensor.detach()
    if not payload.is_contiguous():
        payload = payload.contiguous()
    if not payload.is_contiguous():
        raise ValueError("P2P payload could not be materialized as dense storage")
    return payload


@dataclass
class PendingP2PSend:
    """Owns NCCL Work handles until the paired tensors are safe to release."""

    works: tuple[Any, ...]
    tensors: tuple[torch.Tensor, ...]
    completed: bool = False

    def wait(self) -> None:
        if self.completed:
            return
        for work in self.works:
            work.wait()
        self.completed = True

    def assert_complete(self) -> None:
        if not self.completed:
            raise AssertionError("P2P send remains outstanding")


class CriticPoolP2PTransport:
    """Header + x0 + commit-ACK transport over one explicit process group."""

    def __init__(
        self,
        *,
        group: Any,
        device: torch.device | str,
        ack_group: Any | None = None,
        ack_device: torch.device | str | None = None,
    ) -> None:
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized")
        self.group = group
        self.device = torch.device(device)
        self.ack_group = group if ack_group is None else ack_group
        self.ack_device = (
            self.device if ack_device is None else torch.device(ack_device)
        )
        ack_backend = str(dist.get_backend(self.ack_group)).lower()
        if ack_backend == "gloo" and self.ack_device.type != "cpu":
            raise ValueError("Gloo ACK control plane requires CPU tensors")
        if ack_backend == "nccl" and self.ack_device.type != "cuda":
            raise ValueError("NCCL ACK control plane requires CUDA tensors")
        self.ack_backend = ack_backend
        self._pending: list[PendingP2PSend] = []

    def send_wave(
        self,
        header: CriticWaveHeader,
        x0: torch.Tensor,
        *,
        destination_global_rank: int,
    ) -> PendingP2PSend:
        payload = prepare_transport_payload(x0)
        validate_payload(payload, header)
        header_tensor = header.to_tensor(self.device)
        if payload.device != self.device:
            raise ValueError("P2P payload is not on the transport device")
        works = (
            dist.isend(
                header_tensor,
                dst=int(destination_global_rank),
                group=self.group,
            ),
            dist.isend(
                payload,
                dst=int(destination_global_rank),
                group=self.group,
            ),
        )
        pending = PendingP2PSend(
            works=works,
            tensors=(header_tensor, payload),
        )
        self._pending.append(pending)
        return pending

    def receive_wave(
        self,
        *,
        source_global_rank: int,
    ) -> tuple[CriticWaveHeader, torch.Tensor]:
        header_tensor = torch.empty(
            HEADER_I64_COUNT, dtype=torch.int64, device=self.device
        )
        dist.recv(
            header_tensor,
            src=int(source_global_rank),
            group=self.group,
        )
        header = CriticWaveHeader.from_tensor(header_tensor)
        payload = torch.empty(header.shape, dtype=torch.bfloat16, device=self.device)
        dist.recv(
            payload,
            src=int(source_global_rank),
            group=self.group,
        )
        validate_payload(payload, header)
        return header, payload

    def send_ack(
        self,
        ack: CriticCommitAck,
        *,
        destination_global_rank: int,
    ) -> PendingP2PSend:
        tensor = ack.to_tensor(self.ack_device)
        pending = PendingP2PSend(
            works=(
                dist.isend(
                    tensor,
                    dst=int(destination_global_rank),
                    group=self.ack_group,
                ),
            ),
            tensors=(tensor,),
        )
        self._pending.append(pending)
        return pending

    def receive_ack(
        self,
        *,
        source_global_rank: int,
    ) -> CriticCommitAck:
        tensor = torch.empty(
            ACK_I64_COUNT, dtype=torch.int64, device=self.ack_device
        )
        dist.recv(
            tensor,
            src=int(source_global_rank),
            group=self.ack_group,
        )
        return CriticCommitAck.from_tensor(tensor)

    def reap_completed(self) -> None:
        retained: list[PendingP2PSend] = []
        for pending in self._pending:
            if pending.completed:
                continue
            retained.append(pending)
        self._pending = retained

    def flush(self) -> None:
        for pending in self._pending:
            pending.wait()
        self._pending.clear()

    def assert_flushed(self) -> None:
        if self._pending:
            raise AssertionError(f"{len(self._pending)} P2P sends remain outstanding")


def assert_ack_matches_header(
    ack: CriticCommitAck,
    header: CriticWaveHeader,
) -> None:
    exact = {
        "run_nonce": (ack.run_nonce, header.run_nonce),
        "cycle": (ack.cycle, header.cycle),
        "logical_step": (ack.logical_step, header.logical_step),
        "wave": (ack.wave, header.wave),
        "producer_global_rank": (
            ack.producer_global_rank,
            header.producer_global_rank,
        ),
        "student_global_rank": (
            ack.student_global_rank,
            header.student_global_rank,
        ),
        "observed_generator_version": (
            ack.observed_generator_version,
            header.generator_version,
        ),
        "critic_before": (
            ack.critic_before,
            header.expected_critic_before,
        ),
        "critic_after": (
            ack.critic_after,
            header.expected_critic_after,
        ),
    }
    mismatches = {
        name: values for name, values in exact.items() if values[0] != values[1]
    }
    if mismatches:
        raise ValueError(f"commit ACK/header mismatch: {mismatches}")
