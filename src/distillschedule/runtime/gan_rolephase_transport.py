"""Ordered non-blocking P2P primitives for the phase-aware GAN wavefront.

PR #7 GAN headers/checksums remain authoritative.  Phase B now carries exactly
12 logical samples: one B1 sample for each Critic rank.  This layer keeps the
explicit lifetime and ordering guarantees needed by Q2 pre-posted receives.
"""
from __future__ import annotations

from collections import defaultdict, deque
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


@dataclass(frozen=True)
class CriticRouteIdentity:
    logical_step: int
    logical_slot: int
    micro_wave: int
    source_global_rank: int
    destination_global_rank: int
    generator_version: int
    critic_before: int
    critic_after: int
    trace_sha256: bytes

    def __post_init__(self) -> None:
        if self.logical_step < 0 or not 0 <= self.logical_slot < 12:
            raise ValueError("Phase-B logical step/slot is outside batch12")
        if self.micro_wave != 0:
            raise ValueError("Phase-B batch12 uses one Critic wave only")
        if self.critic_after != self.critic_before + 1:
            raise ValueError("one logical Critic step must advance C exactly once")
        if len(self.trace_sha256) != 32:
            raise ValueError("trace identity must be SHA256 bytes")

    @property
    def edge(self) -> tuple[int, int]:
        return (self.source_global_rank, self.destination_global_rank)

    @property
    def fifo_key(self) -> tuple[int, int, int]:
        return (self.logical_step, self.micro_wave, self.logical_slot)

    def validate_header(self, header: GanCriticRequestHeader) -> None:
        actual = (
            header.logical_step,
            header.micro_wave,
            header.generator_rank,
            header.score_rank,
            header.generator_version,
            header.critic_before,
            header.critic_after,
            header.trace_sha256,
        )
        expected = (
            self.logical_step,
            self.micro_wave,
            self.source_global_rank,
            self.destination_global_rank,
            self.generator_version,
            self.critic_before,
            self.critic_after,
            self.trace_sha256,
        )
        if actual != expected:
            raise ValueError(
                "role-phase Critic header/route identity mismatch: "
                f"actual={actual[:-1]} expected={expected[:-1]}"
            )


@dataclass(eq=False)
class PendingCriticReceive:
    route: CriticRouteIdentity
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
        self.route.validate_header(header)
        actual = tensor_identity(self.payload, header.checksum_kind)
        if actual != header.payload_checksum:
            raise ValueError("received role-phase Critic payload checksum mismatch")
        self.completed = True
        return header, self.payload


class RolePhaseCriticP2PTransport:
    """WORLD-group P2P transport for 4G -> 12C batch12 routing."""

    def __init__(
        self,
        *,
        group: Any,
        device: torch.device | str,
        global_rank: int | None = None,
    ) -> None:
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized")
        self.group = group
        self.device = torch.device(device)
        self.global_rank = int(dist.get_rank() if global_rank is None else global_rank)
        self._pending_sends: dict[int, PendingP2PSend] = {}
        self._pending_receives: dict[int, PendingCriticReceive] = {}
        self._recv_fifo: dict[tuple[int, int], deque[int]] = defaultdict(deque)
        self._last_send_key: dict[tuple[int, int], tuple[int, int, int]] = {}
        self._last_recv_key: dict[tuple[int, int], tuple[int, int, int]] = {}

    @staticmethod
    def _assert_monotonic(
        table: dict[tuple[int, int], tuple[int, int, int]],
        route: CriticRouteIdentity,
        *,
        label: str,
    ) -> None:
        previous = table.get(route.edge)
        if previous is not None and route.fifo_key <= previous:
            raise RuntimeError(
                f"{label} route FIFO regression on edge {route.edge}: "
                f"{route.fifo_key} <= {previous}"
            )
        table[route.edge] = route.fifo_key

    def send(
        self,
        header: GanCriticRequestHeader,
        latent: torch.Tensor,
        *,
        route: CriticRouteIdentity,
    ) -> PendingP2PSend:
        if route.source_global_rank != self.global_rank:
            raise ValueError("send route source is not the local rank")
        route.validate_header(header)
        self._assert_monotonic(self._last_send_key, route, label="send")
        if tuple(latent.shape) != LATENT_SHAPE or latent.dtype != torch.bfloat16:
            raise ValueError("role-phase Critic payload must be official-shape BF16")
        payload = latent.detach()
        if not payload.is_contiguous():
            payload = payload.contiguous()
        if payload.device != self.device:
            raise ValueError("role-phase Critic payload is not on transport device")
        if tensor_identity(payload, header.checksum_kind) != header.payload_checksum:
            raise ValueError("role-phase Critic payload checksum mismatch")
        header_tensor = header.to_tensor(self.device)
        pending = PendingP2PSend(
            works=(
                dist.isend(
                    header_tensor,
                    dst=route.destination_global_rank,
                    group=self.group,
                ),
                dist.isend(
                    payload,
                    dst=route.destination_global_rank,
                    group=self.group,
                ),
            ),
            tensors=(header_tensor, payload),
        )
        self._pending_sends[id(pending)] = pending
        return pending

    def post_receive(self, *, route: CriticRouteIdentity) -> PendingCriticReceive:
        if route.destination_global_rank != self.global_rank:
            raise ValueError("receive route destination is not the local rank")
        self._assert_monotonic(self._last_recv_key, route, label="receive-post")
        header_tensor = torch.empty(
            CRITIC_REQUEST_I64_COUNT,
            dtype=torch.int64,
            device=self.device,
        )
        payload = torch.empty(LATENT_SHAPE, dtype=torch.bfloat16, device=self.device)
        pending = PendingCriticReceive(
            route=route,
            header_tensor=header_tensor,
            payload=payload,
            works=(
                dist.irecv(
                    header_tensor,
                    src=route.source_global_rank,
                    group=self.group,
                ),
                dist.irecv(
                    payload,
                    src=route.source_global_rank,
                    group=self.group,
                ),
            ),
        )
        token = id(pending)
        self._pending_receives[token] = pending
        self._recv_fifo[route.edge].append(token)
        return pending

    def finish_receive(
        self,
        pending: PendingCriticReceive,
    ) -> tuple[GanCriticRequestHeader, torch.Tensor]:
        token = id(pending)
        if self._pending_receives.get(token) is not pending:
            raise RuntimeError("role-phase Critic receive is unknown or already reaped")
        fifo = self._recv_fifo[pending.route.edge]
        if not fifo or fifo[0] != token:
            raise RuntimeError(
                "role-phase Critic receive consumption violated per-edge FIFO"
            )
        header, payload = pending.wait()
        fifo.popleft()
        if not fifo:
            del self._recv_fifo[pending.route.edge]
        del self._pending_receives[token]
        return header, payload

    def reap_send_completion(self, pending: PendingP2PSend) -> None:
        token = id(pending)
        if self._pending_sends.get(token) is not pending:
            raise RuntimeError("role-phase Critic send is unknown or already reaped")
        pending.wait()
        del self._pending_sends[token]

    def flush(self) -> None:
        for pending in list(self._pending_sends.values()):
            pending.wait()
        self._pending_sends.clear()
        for edge in sorted(self._recv_fifo):
            fifo = self._recv_fifo[edge]
            while fifo:
                token = fifo.popleft()
                pending = self._pending_receives[token]
                if not pending.completed:
                    pending.wait()
                del self._pending_receives[token]
        self._recv_fifo.clear()

    def assert_flushed(self) -> None:
        if self._pending_sends or self._pending_receives or self._recv_fifo:
            raise AssertionError(
                "role-phase P2P work remains outstanding: "
                f"send={len(self._pending_sends)} "
                f"recv={len(self._pending_receives)} fifo={len(self._recv_fifo)}"
            )


def make_phase_b_route(
    *,
    logical_step: int,
    logical_slot: int,
    generator_rank: int,
    critic_rank: int,
    generator_version: int,
    critic_before: int,
    trace_sha256: bytes,
) -> CriticRouteIdentity:
    return CriticRouteIdentity(
        logical_step=logical_step,
        logical_slot=logical_slot,
        micro_wave=0,
        source_global_rank=generator_rank,
        destination_global_rank=critic_rank,
        generator_version=generator_version,
        critic_before=critic_before,
        critic_after=critic_before + 1,
        trace_sha256=trace_sha256,
    )


def make_phase_b_header(
    *,
    run_nonce: int,
    route: CriticRouteIdentity,
    payload_checksum: bytes,
    checksum_kind: ChecksumKind = ChecksumKind.GPU_FINGERPRINT_V1,
) -> GanCriticRequestHeader:
    return GanCriticRequestHeader(
        run_nonce=run_nonce,
        logical_step=route.logical_step,
        micro_wave=route.micro_wave,
        generator_rank=route.source_global_rank,
        score_rank=route.destination_global_rank,
        generator_version=route.generator_version,
        critic_before=route.critic_before,
        critic_after=route.critic_after,
        trace_sha256=route.trace_sha256,
        payload_checksum=payload_checksum,
        checksum_kind=checksum_kind,
    )


__all__ = [
    "CriticRouteIdentity",
    "PendingCriticReceive",
    "RolePhaseCriticP2PTransport",
    "make_phase_b_header",
    "make_phase_b_route",
]
