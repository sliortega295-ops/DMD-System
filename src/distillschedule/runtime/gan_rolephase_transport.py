"""Non-blocking P2P primitives for the phase-aware GAN Critic wavefront.

The existing PR #7 GAN headers/checksums stay authoritative. This module only
changes transport lifetime: receives can be posted ahead of compute and
producer payload credit is returned on send completion rather than waiting for
a Critic optimizer acknowledgement.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from runtime.critic_pool import ChecksumKind
from runtime.critic_pool_transport import PendingP2PSend, tensor_identity
from .gan_pipeline_transport import (
    CRITIC_REQUEST_I64_COUNT,
    LATENT_SHAPE,
    GanCriticRequestHeader,
)


@dataclass
class PendingCriticReceive:
    source_global_rank: int
    header_tensor: torch.Tensor
    payload: torch.Tensor
    works: tuple[Any, Any]
    completed: bool = False

    def wait(self) -> tuple[GanCriticRequestHeader, torch.Tensor]:
        if self.completed:
            raise RuntimeError("Critic receive was already consumed")
        for work in self.works:
            work.wait()
        header = GanCriticRequestHeader.from_tensor(self.header_tensor)
        actual = tensor_identity(self.payload, header.checksum_kind)
        if actual != header.payload_checksum:
            raise ValueError("received role-phase Critic payload checksum mismatch")
        if header.generator_rank != self.source_global_rank:
            raise ValueError("role-phase Critic header/source rank mismatch")
        self.completed = True
        return header, self.payload


class RolePhaseCriticP2PTransport:
    """WORLD-group P2P transport for 4G -> 12C uneven routing.

    Unlike the old pair-only executor, a Phase-B Generator rank routes four
    logical samples to multiple Critic ranks. Using a pre-created WORLD NCCL
    group avoids rebuilding pair groups during the timed handoff.
    """

    def __init__(self, *, group: Any, device: torch.device | str) -> None:
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized")
        self.group = group
        self.device = torch.device(device)
        self._pending_sends: list[PendingP2PSend] = []
        self._pending_receives: list[PendingCriticReceive] = []

    def send(
        self,
        header: GanCriticRequestHeader,
        latent: torch.Tensor,
        *,
        destination_global_rank: int,
    ) -> PendingP2PSend:
        if tuple(latent.shape) != LATENT_SHAPE or latent.dtype != torch.bfloat16:
            raise ValueError("role-phase Critic payload must be official-shape BF16")
        payload = latent.detach()
        if not payload.is_contiguous():
            payload = payload.contiguous()
        if payload.device != self.device:
            raise ValueError("role-phase Critic payload is not on transport device")
        if tensor_identity(payload, header.checksum_kind) != header.payload_checksum:
            raise ValueError("role-phase Critic payload checksum mismatch")
        if header.score_rank != int(destination_global_rank):
            raise ValueError("role-phase Critic header/destination rank mismatch")
        header_tensor = header.to_tensor(self.device)
        pending = PendingP2PSend(
            works=(
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
            ),
            tensors=(header_tensor, payload),
        )
        self._pending_sends.append(pending)
        return pending

    def post_receive(self, *, source_global_rank: int) -> PendingCriticReceive:
        """Pre-post header and fixed-shape payload receives before producer send."""
        header_tensor = torch.empty(
            CRITIC_REQUEST_I64_COUNT,
            dtype=torch.int64,
            device=self.device,
        )
        payload = torch.empty(LATENT_SHAPE, dtype=torch.bfloat16, device=self.device)
        pending = PendingCriticReceive(
            source_global_rank=int(source_global_rank),
            header_tensor=header_tensor,
            payload=payload,
            works=(
                dist.irecv(
                    header_tensor,
                    src=int(source_global_rank),
                    group=self.group,
                ),
                dist.irecv(
                    payload,
                    src=int(source_global_rank),
                    group=self.group,
                ),
            ),
        )
        self._pending_receives.append(pending)
        return pending

    def finish_receive(
        self,
        pending: PendingCriticReceive,
    ) -> tuple[GanCriticRequestHeader, torch.Tensor]:
        header, payload = pending.wait()
        try:
            self._pending_receives.remove(pending)
        except ValueError as error:
            raise RuntimeError("role-phase Critic receive was already reaped") from error
        return header, payload

    def reap_send_completion(self, pending: PendingP2PSend) -> None:
        """Safe point for returning one producer payload-buffer credit."""
        pending.wait()
        try:
            self._pending_sends.remove(pending)
        except ValueError as error:
            raise RuntimeError("role-phase Critic send was already reaped") from error

    def flush(self) -> None:
        for pending in list(self._pending_sends):
            pending.wait()
        self._pending_sends.clear()
        for pending in list(self._pending_receives):
            if not pending.completed:
                pending.wait()
        self._pending_receives.clear()

    def assert_flushed(self) -> None:
        if self._pending_sends or self._pending_receives:
            raise AssertionError(
                "role-phase P2P work remains outstanding: "
                f"send={len(self._pending_sends)} recv={len(self._pending_receives)}"
            )


def make_phase_b_header(
    *,
    run_nonce: int,
    logical_step: int,
    logical_slot: int,
    generator_rank: int,
    critic_rank: int,
    generator_version: int,
    critic_before: int,
    trace_sha256: bytes,
    payload_checksum: bytes,
    checksum_kind: ChecksumKind = ChecksumKind.GPU_FINGERPRINT_V1,
) -> GanCriticRequestHeader:
    """Reuse the PR #7 wire header for one uneven Phase-B logical sample."""
    if not 0 <= logical_slot < 16:
        raise ValueError("logical slot must be in [0,16)")
    return GanCriticRequestHeader(
        run_nonce=run_nonce,
        logical_step=logical_step,
        micro_wave=0 if logical_slot < 12 else 1,
        generator_rank=generator_rank,
        score_rank=critic_rank,
        generator_version=generator_version,
        critic_before=critic_before,
        critic_after=critic_before + 1,
        trace_sha256=trace_sha256,
        payload_checksum=payload_checksum,
        checksum_kind=checksum_kind,
    )


__all__ = [
    "PendingCriticReceive",
    "RolePhaseCriticP2PTransport",
    "make_phase_b_header",
]
