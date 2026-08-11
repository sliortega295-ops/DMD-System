"""Control-plane messages for GAS-aware Route-B pair transport."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from .critic_pool import WIRE_MAGIC, WIRE_SCHEMA_VERSION, CriticWaveHeader


ACK_I64_COUNT = 19


@dataclass(frozen=True)
class RouteBWaveAck:
    run_nonce: int
    cycle: int
    logical_step: int
    micro_wave: int
    producer_global_rank: int
    critic_global_rank: int
    observed_generator_version: int
    critic_before: int
    critic_after: int
    optimizer_step_before: int
    optimizer_step_after: int
    scheduler_step_before: int
    scheduler_step_after: int
    optimizer_committed: bool
    status: int = 0
    error_code: int = 0

    def __post_init__(self) -> None:
        delta = int(self.optimizer_committed)
        if self.critic_after != self.critic_before + delta:
            raise ValueError("Critic version delta disagrees with commit flag")
        if self.optimizer_step_after != self.optimizer_step_before + delta:
            raise ValueError("optimizer-step delta disagrees with commit flag")
        if self.scheduler_step_after != self.scheduler_step_before + delta:
            raise ValueError("scheduler-step delta disagrees with commit flag")

    def to_tensor(self, device: torch.device | str = "cpu") -> torch.Tensor:
        values = [
            WIRE_MAGIC,
            WIRE_SCHEMA_VERSION,
            self.run_nonce,
            self.cycle,
            self.logical_step,
            self.micro_wave,
            self.producer_global_rank,
            self.critic_global_rank,
            self.observed_generator_version,
            self.critic_before,
            self.critic_after,
            self.optimizer_step_before,
            self.optimizer_step_after,
            self.scheduler_step_before,
            self.scheduler_step_after,
            int(self.optimizer_committed),
            self.status,
            self.error_code,
            ACK_I64_COUNT,
        ]
        return torch.tensor(values, dtype=torch.int64, device=device)

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "RouteBWaveAck":
        if tensor.dtype != torch.int64 or tensor.numel() != ACK_I64_COUNT:
            raise ValueError("invalid Route-B ACK tensor")
        values = [int(value) for value in tensor.detach().cpu().reshape(-1)]
        if (
            values[0] != WIRE_MAGIC
            or values[1] != WIRE_SCHEMA_VERSION
            or values[18] != ACK_I64_COUNT
        ):
            raise ValueError("Route-B ACK magic/schema/width mismatch")
        return cls(
            run_nonce=values[2],
            cycle=values[3],
            logical_step=values[4],
            micro_wave=values[5],
            producer_global_rank=values[6],
            critic_global_rank=values[7],
            observed_generator_version=values[8],
            critic_before=values[9],
            critic_after=values[10],
            optimizer_step_before=values[11],
            optimizer_step_after=values[12],
            scheduler_step_before=values[13],
            scheduler_step_after=values[14],
            optimizer_committed=bool(values[15]),
            status=values[16],
            error_code=values[17],
        )


def send_ack(
    ack: RouteBWaveAck,
    *,
    group: Any,
    destination_global_rank: int,
) -> None:
    if str(dist.get_backend(group)).lower() != "gloo":
        raise ValueError("Route-B ACK group must use Gloo CPU transport")
    tensor = ack.to_tensor("cpu")
    work = dist.isend(tensor, dst=int(destination_global_rank), group=group)
    work.wait()


def receive_ack(
    *,
    group: Any,
    source_global_rank: int,
) -> RouteBWaveAck:
    if str(dist.get_backend(group)).lower() != "gloo":
        raise ValueError("Route-B ACK group must use Gloo CPU transport")
    tensor = torch.empty(ACK_I64_COUNT, dtype=torch.int64, device="cpu")
    dist.recv(tensor, src=int(source_global_rank), group=group)
    return RouteBWaveAck.from_tensor(tensor)


def assert_ack_matches_header(
    ack: RouteBWaveAck,
    header: CriticWaveHeader,
    *,
    expected_commit: bool,
) -> None:
    expected_after = header.expected_critic_after if expected_commit else header.expected_critic_before
    exact = {
        "run_nonce": (ack.run_nonce, header.run_nonce),
        "cycle": (ack.cycle, header.cycle),
        "logical_step": (ack.logical_step, header.logical_step),
        "micro_wave": (ack.micro_wave, header.wave),
        "producer_global_rank": (
            ack.producer_global_rank,
            header.producer_global_rank,
        ),
        "critic_global_rank": (
            ack.critic_global_rank,
            header.student_global_rank,
        ),
        "generator_version": (
            ack.observed_generator_version,
            header.generator_version,
        ),
        "critic_before": (ack.critic_before, header.expected_critic_before),
        "critic_after": (ack.critic_after, expected_after),
        "optimizer_committed": (ack.optimizer_committed, expected_commit),
    }
    mismatches = {
        name: value for name, value in exact.items() if value[0] != value[1]
    }
    if mismatches:
        raise ValueError(f"Route-B ACK/header mismatch: {mismatches}")
