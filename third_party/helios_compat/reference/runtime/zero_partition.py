"""Helpers for comparing logical ZeRO partitions across world sizes."""
from __future__ import annotations


def zero_partition_valid_length(
    *, logical_numel: int, partition_numel: int, rank: int
) -> int:
    """Return non-padding elements in one rank's fixed-size ZeRO partition."""
    if logical_numel < 0 or partition_numel < 0 or rank < 0:
        raise ValueError("logical size, partition size, and rank must be non-negative")
    start = rank * partition_numel
    return max(0, min(partition_numel, logical_numel - start))
