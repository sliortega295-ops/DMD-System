"""Bidirectional Generator <-> Score P2P transport for GAN score gradients."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from .critic_pool import ChecksumKind
from .critic_pool_transport import PendingP2PSend, tensor_identity


GAN_BOUNDARY_MAGIC = 0x47414E47524144  # "GANGRAD"
GAN_BOUNDARY_SCHEMA = 1
REQUEST_I64_COUNT = 24
REPLY_I64_COUNT = 26
LATENT_SHAPE = (1, 16, 9, 48, 80)


def _digest_i64(value: bytes) -> tuple[int, int, int, int]:
    if len(value) != 32:
        raise ValueError("digest must contain 32 bytes")
    return tuple(
        int.from_bytes(value[offset : offset + 8], "little", signed=True)
        for offset in range(0, 32, 8)
    )  # type: ignore[return-value]


def _i64_digest(values: list[int]) -> bytes:
    if len(values) != 4:
        raise ValueError("digest representation must contain four int64 values")
    return b"".join(int(value).to_bytes(8, "little", signed=True) for value in values)


@dataclass(frozen=True)
class GanGradientRequestHeader:
    run_nonce: int
    logical_step: int
    micro_wave: int
    generator_rank: int
    score_rank: int
    generator_version: int
    critic_version: int
    trace_sha256: bytes
    payload_checksum: bytes
    checksum_kind: ChecksumKind = ChecksumKind.GPU_FINGERPRINT_V1

    def __post_init__(self) -> None:
        if self.logical_step < 0 or self.micro_wave not in (0, 1):
            raise ValueError("invalid GAN gradient request step/wave")
        if min(self.generator_version, self.critic_version) < 0:
            raise ValueError("versions must be non-negative")
        _digest_i64(self.trace_sha256)
        _digest_i64(self.payload_checksum)

    def to_tensor(self, device: torch.device | str) -> torch.Tensor:
        values = [
            GAN_BOUNDARY_MAGIC,
            GAN_BOUNDARY_SCHEMA,
            self.run_nonce,
            self.logical_step,
            self.micro_wave,
            self.generator_rank,
            self.score_rank,
            self.generator_version,
            self.critic_version,
            *LATENT_SHAPE,
            1,
            int(self.checksum_kind),
            *_digest_i64(self.trace_sha256),
            *_digest_i64(self.payload_checksum),
        ]
        if len(values) != REQUEST_I64_COUNT:
            raise AssertionError("GAN request header width changed")
        return torch.tensor(values, dtype=torch.int64, device=device)

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "GanGradientRequestHeader":
        if tensor.dtype != torch.int64 or tensor.numel() != REQUEST_I64_COUNT:
            raise ValueError("invalid GAN request header")
        values = [int(item) for item in tensor.detach().cpu().reshape(-1).tolist()]
        if values[0] != GAN_BOUNDARY_MAGIC or values[1] != GAN_BOUNDARY_SCHEMA:
            raise ValueError("GAN request magic/schema mismatch")
        if tuple(values[9:14]) != LATENT_SHAPE or values[14] != 1:
            raise ValueError("GAN request latent contract drift")
        return cls(
            run_nonce=values[2],
            logical_step=values[3],
            micro_wave=values[4],
            generator_rank=values[5],
            score_rank=values[6],
            generator_version=values[7],
            critic_version=values[8],
            checksum_kind=ChecksumKind(values[15]),
            trace_sha256=_i64_digest(values[16:20]),
            payload_checksum=_i64_digest(values[20:24]),
        )


@dataclass(frozen=True)
class GanGradientReplyHeader:
    run_nonce: int
    logical_step: int
    micro_wave: int
    generator_rank: int
    score_rank: int
    generator_version: int
    critic_before: int
    critic_after: int
    trace_sha256: bytes
    gradient_checksum: bytes
    checksum_kind: ChecksumKind = ChecksumKind.GPU_FINGERPRINT_V1
    status: int = 0

    def __post_init__(self) -> None:
        if self.critic_after != self.critic_before:
            raise ValueError("Score gradient service must not advance Critic version")
        _digest_i64(self.trace_sha256)
        _digest_i64(self.gradient_checksum)

    def to_tensor(self, device: torch.device | str) -> torch.Tensor:
        values = [
            GAN_BOUNDARY_MAGIC,
            GAN_BOUNDARY_SCHEMA,
            self.run_nonce,
            self.logical_step,
            self.micro_wave,
            self.generator_rank,
            self.score_rank,
            self.generator_version,
            self.critic_before,
            self.critic_after,
            *LATENT_SHAPE,
            1,
            int(self.checksum_kind),
            self.status,
            *_digest_i64(self.trace_sha256),
            *_digest_i64(self.gradient_checksum),
        ]
        if len(values) != REPLY_I64_COUNT:
            raise AssertionError("GAN reply header width changed")
        return torch.tensor(values, dtype=torch.int64, device=device)

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "GanGradientReplyHeader":
        if tensor.dtype != torch.int64 or tensor.numel() != REPLY_I64_COUNT:
            raise ValueError("invalid GAN reply header")
        values = [int(item) for item in tensor.detach().cpu().reshape(-1).tolist()]
        if values[0] != GAN_BOUNDARY_MAGIC or values[1] != GAN_BOUNDARY_SCHEMA:
            raise ValueError("GAN reply magic/schema mismatch")
        if tuple(values[10:15]) != LATENT_SHAPE or values[15] != 1:
            raise ValueError("GAN reply gradient contract drift")
        return cls(
            run_nonce=values[2],
            logical_step=values[3],
            micro_wave=values[4],
            generator_rank=values[5],
            score_rank=values[6],
            generator_version=values[7],
            critic_before=values[8],
            critic_after=values[9],
            checksum_kind=ChecksumKind(values[16]),
            status=values[17],
            trace_sha256=_i64_digest(values[18:22]),
            gradient_checksum=_i64_digest(values[22:26]),
        )


def _validate_bf16_latent(tensor: torch.Tensor) -> torch.Tensor:
    if tuple(tensor.shape) != LATENT_SHAPE or tensor.dtype != torch.bfloat16:
        raise ValueError("GAN boundary payload must be official-shape BF16 latent")
    value = tensor.detach()
    return value if value.is_contiguous() else value.contiguous()


class GanGradientP2PTransport:
    """Header + latent request and header + latent-gradient reply over one group."""

    def __init__(self, *, group: Any, device: torch.device | str) -> None:
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized")
        self.group = group
        self.device = torch.device(device)
        self._pending: list[PendingP2PSend] = []

    def send_request(
        self,
        header: GanGradientRequestHeader,
        latent: torch.Tensor,
        *,
        destination_global_rank: int,
    ) -> PendingP2PSend:
        payload = _validate_bf16_latent(latent).to(self.device)
        actual = tensor_identity(payload, header.checksum_kind)
        if actual != header.payload_checksum:
            raise ValueError("GAN request payload checksum mismatch")
        header_tensor = header.to_tensor(self.device)
        pending = PendingP2PSend(
            works=(
                dist.isend(header_tensor, dst=int(destination_global_rank), group=self.group),
                dist.isend(payload, dst=int(destination_global_rank), group=self.group),
            ),
            tensors=(header_tensor, payload),
        )
        self._pending.append(pending)
        return pending

    def receive_request(
        self, *, source_global_rank: int
    ) -> tuple[GanGradientRequestHeader, torch.Tensor]:
        header_tensor = torch.empty(REQUEST_I64_COUNT, dtype=torch.int64, device=self.device)
        dist.recv(header_tensor, src=int(source_global_rank), group=self.group)
        header = GanGradientRequestHeader.from_tensor(header_tensor)
        payload = torch.empty(LATENT_SHAPE, dtype=torch.bfloat16, device=self.device)
        dist.recv(payload, src=int(source_global_rank), group=self.group)
        if tensor_identity(payload, header.checksum_kind) != header.payload_checksum:
            raise ValueError("received GAN request checksum mismatch")
        return header, payload

    def send_reply(
        self,
        header: GanGradientReplyHeader,
        gradient: torch.Tensor,
        *,
        destination_global_rank: int,
    ) -> PendingP2PSend:
        payload = _validate_bf16_latent(gradient).to(self.device)
        if tensor_identity(payload, header.checksum_kind) != header.gradient_checksum:
            raise ValueError("GAN gradient reply checksum mismatch")
        header_tensor = header.to_tensor(self.device)
        pending = PendingP2PSend(
            works=(
                dist.isend(header_tensor, dst=int(destination_global_rank), group=self.group),
                dist.isend(payload, dst=int(destination_global_rank), group=self.group),
            ),
            tensors=(header_tensor, payload),
        )
        self._pending.append(pending)
        return pending

    def receive_reply(
        self,
        *,
        source_global_rank: int,
        request: GanGradientRequestHeader,
    ) -> tuple[GanGradientReplyHeader, torch.Tensor]:
        header_tensor = torch.empty(REPLY_I64_COUNT, dtype=torch.int64, device=self.device)
        dist.recv(header_tensor, src=int(source_global_rank), group=self.group)
        header = GanGradientReplyHeader.from_tensor(header_tensor)
        exact = (
            header.run_nonce,
            header.logical_step,
            header.micro_wave,
            header.generator_rank,
            header.score_rank,
            header.generator_version,
            header.critic_before,
            header.trace_sha256,
        )
        expected = (
            request.run_nonce,
            request.logical_step,
            request.micro_wave,
            request.generator_rank,
            request.score_rank,
            request.generator_version,
            request.critic_version,
            request.trace_sha256,
        )
        if exact != expected or header.status != 0:
            raise ValueError("GAN gradient reply/request mismatch")
        gradient = torch.empty(LATENT_SHAPE, dtype=torch.bfloat16, device=self.device)
        dist.recv(gradient, src=int(source_global_rank), group=self.group)
        if tensor_identity(gradient, header.checksum_kind) != header.gradient_checksum:
            raise ValueError("received GAN gradient checksum mismatch")
        return header, gradient

    def flush(self) -> None:
        for pending in self._pending:
            pending.wait()
        self._pending.clear()

    def assert_flushed(self) -> None:
        if self._pending:
            raise AssertionError("GAN gradient P2P work remains outstanding")


__all__ = [
    "GanGradientP2PTransport",
    "GanGradientReplyHeader",
    "GanGradientRequestHeader",
]
