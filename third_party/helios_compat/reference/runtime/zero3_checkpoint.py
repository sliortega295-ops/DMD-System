"""Block-local gradient checkpointing compatible with ZeRO-3 placeholders."""
from __future__ import annotations

from typing import Any, Literal

import torch
from torch.utils.checkpoint import checkpoint


GradientCheckpointingPolicy = Literal["strict", "zero3_metadata_tolerant"]


def zero3_metadata_tolerant_checkpoint(
    module: torch.nn.Module,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Checkpoint one block without comparing ZeRO-mutated parameter metadata.

    Non-reentrant PyTorch checkpointing normally compares metadata for tensors
    saved by the original forward and recomputation.  ZeRO-3 repartitions full
    parameters into zero-sized local placeholders between those two executions,
    so that metadata comparison can reject a numerically valid recomputation.

    This adapter keeps non-reentrant, block-local checkpointing and RNG-state
    preservation.  It disables only PyTorch's checkpoint metadata comparison;
    callers must keep this policy opt-in and validate numerical parity for the
    real model/workload before using it in training.
    """
    return checkpoint(
        module.__call__,
        *args,
        use_reentrant=False,
        determinism_check="none",
        preserve_rng_state=True,
        **kwargs,
    )


def configure_block_gradient_checkpointing(
    model: Any,
    *,
    zero_stage: int,
    policy: GradientCheckpointingPolicy,
) -> None:
    """Install the selected block checkpoint policy on a Diffusers model."""
    if policy == "strict":
        model.enable_gradient_checkpointing()
        return
    if policy != "zero3_metadata_tolerant":
        raise ValueError(f"unsupported gradient checkpointing policy: {policy}")
    if zero_stage != 3:
        raise ValueError("zero3_metadata_tolerant checkpointing requires ZeRO-3")
    model.enable_gradient_checkpointing(
        gradient_checkpointing_func=zero3_metadata_tolerant_checkpoint
    )
