"""Collective-safe mapping between unequal producer and consumer role groups."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch


def assigned_source_ranks(
    source_world_size: int, consumer_world_size: int, consumer_rank: int
) -> tuple[int, ...]:
    """Return round-robin producer ranks assigned to one consumer rank."""
    if source_world_size <= 0 or consumer_world_size <= 0:
        raise ValueError("world sizes must be positive")
    if not 0 <= consumer_rank < consumer_world_size:
        raise ValueError(
            f"consumer rank {consumer_rank} outside world size {consumer_world_size}"
        )
    assigned = tuple(
        source_rank
        for source_rank in range(source_world_size)
        if source_rank % consumer_world_size == consumer_rank
    )
    if not assigned:
        raise ValueError(
            "consumer has no source sample; empty ranks require an explicit padding policy"
        )
    return assigned


def padded_source_rounds(
    source_world_size: int, consumer_world_size: int
) -> tuple[tuple[int | None, ...], ...]:
    """Return collective-aligned source slots, padding the final round with None."""
    if source_world_size <= 0 or consumer_world_size <= 0:
        raise ValueError("world sizes must be positive")
    rounds = (source_world_size + consumer_world_size - 1) // consumer_world_size
    return tuple(
        tuple(
            source_rank if source_rank < source_world_size else None
            for consumer_rank in range(consumer_world_size)
            for source_rank in (round_index * consumer_world_size + consumer_rank,)
        )
        for round_index in range(rounds)
    )


def source_owned_seed(base_seed: int, offset: int, source_rank: int) -> int:
    """Derive randomness from producer identity, independent of consumer placement."""
    if source_rank < 0:
        raise ValueError("source rank must be non-negative")
    return int(base_seed) + int(offset) + int(source_rank)


def uneven_dp_loss_scale(
    *, local_samples: int, global_samples: int, consumer_world_size: int
) -> float:
    """Scale a local mean before a data-parallel mean reduction.

    If the backend averages gradients across ``consumer_world_size`` ranks, multiplying
    each local mean by ``local_samples * world / global_samples`` gives every global
    sample weight ``1 / global_samples``.
    """
    if local_samples <= 0 or global_samples <= 0 or consumer_world_size <= 0:
        raise ValueError("sample counts and consumer world size must be positive")
    if local_samples > global_samples:
        raise ValueError("local sample count cannot exceed global sample count")
    return local_samples * consumer_world_size / global_samples


@dataclass(frozen=True)
class SourceBatchLayout:
    """Batch slices for source-rank-owned tensors packed on one consumer."""

    source_ranks: tuple[int, ...]
    batch_sizes: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.source_ranks or len(self.source_ranks) != len(self.batch_sizes):
            raise ValueError("source ranks and batch sizes must be non-empty and aligned")
        if len(set(self.source_ranks)) != len(self.source_ranks):
            raise ValueError("source ranks must be unique")
        if any(rank < 0 for rank in self.source_ranks):
            raise ValueError("source ranks must be non-negative")
        if any(size <= 0 for size in self.batch_sizes):
            raise ValueError("batch sizes must be positive")

    @classmethod
    def from_tensors(cls, tensors: Mapping[int, torch.Tensor]) -> "SourceBatchLayout":
        if not tensors:
            raise ValueError("cannot build a layout from no tensors")
        ranks = tuple(tensors)
        sizes = tuple(int(tensors[rank].shape[0]) for rank in ranks)
        return cls(source_ranks=ranks, batch_sizes=sizes)

    @property
    def local_samples(self) -> int:
        return sum(self.batch_sizes)

    @property
    def slices(self) -> dict[int, slice]:
        result: dict[int, slice] = {}
        start = 0
        for rank, size in zip(self.source_ranks, self.batch_sizes):
            result[rank] = slice(start, start + size)
            start += size
        return result

    def pack(self, tensors: Mapping[int, torch.Tensor]) -> torch.Tensor:
        if tuple(tensors) != self.source_ranks:
            raise ValueError(
                f"tensor source order {tuple(tensors)} != layout {self.source_ranks}"
            )
        for rank, size in zip(self.source_ranks, self.batch_sizes):
            if int(tensors[rank].shape[0]) != size:
                raise ValueError(f"source rank {rank} batch size changed")
        return torch.cat([tensors[rank] for rank in self.source_ranks], dim=0)

    def split(self, tensor: torch.Tensor) -> dict[int, torch.Tensor]:
        if int(tensor.shape[0]) != self.local_samples:
            raise ValueError(
                f"packed batch {tensor.shape[0]} != expected {self.local_samples}"
            )
        return {rank: tensor[index] for rank, index in self.slices.items()}
