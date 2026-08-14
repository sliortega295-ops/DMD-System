"""Helios-facing helpers for the Phase-B DP12/B1 Critic backend.

Phase B now reuses the no-GAN batch12 shape: every Critic rank consumes exactly
one logical sample.  No trace batching, uneven local batches, or special
collective-call alignment is required.  The adapter only binds one B1 payload
to its immutable operation trace and makes the backend gradient reduction
semantics explicit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from runtime.operation_trace import OperationTrace
from distillschedule.runtime.gan_rolephase_critic import CriticReductionContract


@dataclass(frozen=True)
class CriticLocalBatchBundle:
    logical_slot: int
    generated_x0: torch.Tensor
    source: Mapping[str, torch.Tensor]
    operation_trace: OperationTrace

    @property
    def local_batch(self) -> int:
        return 1

    def validate(self) -> None:
        if not 0 <= int(self.logical_slot) < 12:
            raise ValueError("Phase-B logical slot must be in [0,12)")
        if self.generated_x0.ndim == 0 or self.generated_x0.shape[0] != 1:
            raise ValueError("Phase-B generated_x0 must remain B1")
        for name, value in self.source.items():
            if value.ndim and value.shape[0] != 1:
                raise ValueError(f"source tensor {name} must remain B1")


def build_critic_local_batch_bundle(
    *,
    logical_slot: int,
    generated_x0: torch.Tensor,
    source: Mapping[str, torch.Tensor],
    trace: OperationTrace,
) -> CriticLocalBatchBundle:
    bundle = CriticLocalBatchBundle(
        logical_slot=int(logical_slot),
        generated_x0=generated_x0,
        source=source,
        operation_trace=trace,
    )
    bundle.validate()
    return bundle


class ScaledBackwardAccelerator:
    """Bind official Critic backwards to the declared DP12 reduction mode.

    With the expected DeepSpeed DP_MEAN semantics and B1 on all 12 ranks, the
    ordinary per-rank Critic loss already yields the batch12 mean gradient, so
    the scale is exactly 1.  An explicit SUM backend instead uses 1/12.
    """

    def __init__(
        self,
        base: Any,
        *,
        backend_dp_reduction: str,
        contract: CriticReductionContract = CriticReductionContract(),
    ) -> None:
        self._base = base
        self.contract = contract
        self.scale = contract.local_loss_scale(backend_dp_reduction)

    def backward(self, loss: torch.Tensor) -> None:
        self._base.backward(loss * self.scale)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)


__all__ = [
    "CriticLocalBatchBundle",
    "ScaledBackwardAccelerator",
    "build_critic_local_batch_bundle",
]
