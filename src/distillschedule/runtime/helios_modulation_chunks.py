"""Fail-closed control of Helios token-chunk launch granularity.

The pinned Helios transformer performs token-independent modulation, residual,
and cross-attention-normalization work in bounded chunks.  Changing this value
does not change the tensor shapes visible to attention/MLP layers; it only
changes how many otherwise identical chunk launches implement those pointwise
operations.  The helper deliberately recognizes only the pinned 40-block
Helios model and records the prior value on every block.
"""
from __future__ import annotations

from typing import Any


SUPPORTED_HELIOS_MODULATION_CHUNK_TOKENS = (512, 1024, 2048)
EXPECTED_HELIOS_BLOCK_COUNT = 40


def configure_helios_modulation_chunk_tokens(
    model: Any,
    *,
    chunk_tokens: int,
    role: str,
) -> dict[str, Any]:
    """Set one explicit chunk size on every pinned Helios transformer block."""

    value = int(chunk_tokens)
    if value not in SUPPORTED_HELIOS_MODULATION_CHUNK_TOKENS:
        raise ValueError(
            "unsupported Helios modulation chunk size: "
            f"{value}; expected one of {SUPPORTED_HELIOS_MODULATION_CHUNK_TOKENS}"
        )
    if role not in {"generator", "critic"}:
        raise ValueError(f"unsupported Helios role: {role}")

    blocks = [
        module
        for module in model.modules()
        if type(module).__name__ == "HeliosTransformerBlock"
    ]
    if len(blocks) != EXPECTED_HELIOS_BLOCK_COUNT:
        raise RuntimeError(
            "pinned Helios block inventory changed: "
            f"{len(blocks)} != {EXPECTED_HELIOS_BLOCK_COUNT}"
        )

    prior = [
        getattr(block, "_distillschedule_modulation_chunk_tokens", None)
        for block in blocks
    ]
    unexpected = sorted(
        {
            int(item)
            for item in prior
            if item is not None and int(item) != 512
        }
    )
    if unexpected:
        raise RuntimeError(
            "Helios modulation chunk policy was already changed before the "
            f"candidate applied: {unexpected}"
        )

    for block in blocks:
        block._distillschedule_modulation_chunk_tokens = value

    observed = {
        int(getattr(block, "_distillschedule_modulation_chunk_tokens"))
        for block in blocks
    }
    if observed != {value}:
        raise AssertionError(f"Helios modulation chunk assignment drifted: {observed}")
    return {
        "status": "PASS",
        "role": role,
        "requested_chunk_tokens": value,
        "block_count": len(blocks),
        "prior_chunk_tokens": sorted(
            {512 if item is None else int(item) for item in prior}
        ),
        "observed_chunk_tokens": sorted(observed),
        "semantic_scope": (
            "TOKEN_INDEPENDENT_MODULATION_RESIDUAL_AND_CROSS_NORM_LAUNCH_GROUPING"
        ),
    }


__all__ = [
    "EXPECTED_HELIOS_BLOCK_COUNT",
    "SUPPORTED_HELIOS_MODULATION_CHUNK_TOKENS",
    "configure_helios_modulation_chunk_tokens",
]
