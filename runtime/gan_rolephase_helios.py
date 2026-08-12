"""Helios-facing helpers for the uneven C12 Critic backend.

Flex Critic ranks consume two logical samples in one physical model call.  It
is not sufficient to concatenate x0: the operation-keyed timestep/noise/crop/
R1/history inputs must become the same physical batch.  ``BatchedOperationTrace``
provides the narrow OperationTrace interface used by Helios replay helpers and
fails closed if controls or non-batch tensors cannot be merged safely.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from runtime.operation_trace import OperationTrace
from distillschedule.runtime.gan_rolephase_critic import CriticReductionContract


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


class BatchedOperationTrace:
    """Read-only batch view over one or two immutable rank-qualified traces."""

    def __init__(self, traces: Sequence[OperationTrace]) -> None:
        self._traces = tuple(traces)
        if len(self._traces) not in (1, 2):
            raise ValueError("C12 trace batching supports local batch1 or batch2")
        if not self._traces:
            raise ValueError("trace batch cannot be empty")
        payload = {
            "kind": "rolephase_batched_operation_trace_v1",
            "members": [trace.trace_hash for trace in self._traces],
        }
        self._trace_hash = hashlib.sha256(_canonical(payload)).hexdigest()
        self.root = self._traces[0].root
        self.manifest = dict(self._traces[0].manifest)
        self.manifest["trace_hash"] = self._trace_hash
        self.manifest["rolephase_batched_members"] = payload["members"]
        self.manifest["rolephase_local_batch"] = len(self._traces)

    @property
    def trace_hash(self) -> str:
        return self._trace_hash

    @property
    def local_batch(self) -> int:
        return len(self._traces)

    @property
    def members(self) -> tuple[OperationTrace, ...]:
        return self._traces

    def tensor(self, key: str, *, clone: bool = True) -> torch.Tensor:
        values = [trace.tensor(key, clone=False) for trace in self._traces]
        first = values[0]
        if len(values) == 1:
            return first.clone(memory_format=torch.preserve_format) if clone else first
        if any(value.dtype != first.dtype or tuple(value.shape) != tuple(first.shape) for value in values[1:]):
            raise ValueError(f"cannot batch trace tensor with shape/dtype drift: {key}")

        # Stochastic/sample tensors are recorded with batch dimension one.  Cat
        # them into B2 so one physical Helios call consumes both logical traces.
        if first.ndim >= 1 and first.shape[0] == 1:
            result = torch.cat(values, dim=0)
            return result.clone(memory_format=torch.preserve_format) if clone else result

        # Scalars/control-like tensors have no sample dimension.  They may be
        # shared only when values are identical; otherwise batching would hide a
        # semantic difference that Helios cannot represent in one call.
        if all(torch.equal(value, first) for value in values[1:]):
            return first.clone(memory_format=torch.preserve_format) if clone else first
        raise ValueError(
            f"non-batched trace tensor differs across local B2 samples: {key}"
        )

    def operation(self, key: str) -> dict[str, Any]:
        operations = [trace.operation(key) for trace in self._traces]
        reference = operations[0]
        for other in operations[1:]:
            if other.get("component") != reference.get("component"):
                raise ValueError(f"trace operation component drift: {key}")
            if other.get("operation") != reference.get("operation"):
                raise ValueError(f"trace operation kind drift: {key}")
            if other.get("control", {}) != reference.get("control", {}):
                raise ValueError(
                    f"C12 B2 requires identical control/DiT schedule for {key}"
                )
            if other.get("tensor_keys", []) != reference.get("tensor_keys", []):
                raise ValueError(f"trace operation tensor-key drift: {key}")
        return dict(reference)


@dataclass(frozen=True)
class CriticLocalBatchBundle:
    logical_slots: tuple[int, ...]
    generated_x0: torch.Tensor
    source: Mapping[str, torch.Tensor]
    operation_trace: OperationTrace | BatchedOperationTrace

    @property
    def local_batch(self) -> int:
        return len(self.logical_slots)

    def validate(self) -> None:
        if self.local_batch not in (1, 2):
            raise ValueError("C12 bundle supports only local batch1 or batch2")
        if self.generated_x0.shape[0] != self.local_batch:
            raise ValueError("generated_x0 batch does not match logical slots")
        for name, value in self.source.items():
            if value.ndim and value.shape[0] != self.local_batch:
                raise ValueError(f"source tensor {name} batch does not match C12 bundle")
        trace_batch = int(getattr(self.operation_trace, "local_batch", 1))
        if trace_batch != self.local_batch:
            raise ValueError("operation trace batch does not match C12 bundle")


def _cat_batch(values: Sequence[torch.Tensor], *, label: str) -> torch.Tensor:
    if not values:
        raise ValueError(f"cannot batch empty {label}")
    first = values[0]
    if any(value.dtype != first.dtype or tuple(value.shape[1:]) != tuple(first.shape[1:]) for value in values[1:]):
        raise ValueError(f"{label} shape/dtype drift across local samples")
    if any(value.ndim == 0 or value.shape[0] != 1 for value in values):
        raise ValueError(f"{label} must carry an explicit B1 dimension before batching")
    return torch.cat(tuple(values), dim=0)


def build_critic_local_batch_bundle(
    *,
    logical_slots: Sequence[int],
    generated_x0: Sequence[torch.Tensor],
    sources: Sequence[Mapping[str, torch.Tensor]],
    traces: Sequence[OperationTrace],
) -> CriticLocalBatchBundle:
    slots = tuple(int(slot) for slot in logical_slots)
    if not (len(slots) == len(generated_x0) == len(sources) == len(traces)):
        raise ValueError("C12 local bundle inputs have different lengths")
    if len(slots) not in (1, 2):
        raise ValueError("C12 local bundle supports one or two logical samples")
    if len(set(slots)) != len(slots):
        raise ValueError("C12 local bundle duplicated a logical slot")
    keys = tuple(sources[0].keys())
    if any(tuple(source.keys()) != keys for source in sources[1:]):
        raise ValueError("Stage-3 source dictionaries differ across local samples")
    source_batch = {
        key: _cat_batch([source[key] for source in sources], label=f"source/{key}")
        for key in keys
    }
    x0_batch = _cat_batch(list(generated_x0), label="generated_x0")
    trace_batch: OperationTrace | BatchedOperationTrace
    trace_batch = traces[0] if len(traces) == 1 else BatchedOperationTrace(traces)
    bundle = CriticLocalBatchBundle(slots, x0_batch, source_batch, trace_batch)
    bundle.validate()
    return bundle


class ScaledBackwardAccelerator:
    """Scale every official Critic backward before delegating to the engine.

    Helios low-VRAM Critic may call ``accelerator.backward`` more than once
    inside one logical `_critic_loss`.  Applying one constant factor here keeps
    denoise and GAN/R1 terms under the same global-batch16 normalization.
    """

    def __init__(
        self,
        base: Any,
        *,
        local_batch: int,
        backend_dp_reduction: str,
        contract: CriticReductionContract = CriticReductionContract(),
    ) -> None:
        if local_batch not in (1, 2):
            raise ValueError("C12 local batch must be one or two")
        self._base = base
        self.local_batch = int(local_batch)
        self.contract = contract
        # Official per-rank Critic losses are batch means. Convert them to a
        # local sample sum, then apply the backend-specific global normalization.
        self.scale = float(local_batch) * contract.local_sum_scale(
            backend_dp_reduction
        )

    def backward(self, loss: torch.Tensor) -> None:
        self._base.backward(loss * self.scale)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)


__all__ = [
    "BatchedOperationTrace",
    "CriticLocalBatchBundle",
    "ScaledBackwardAccelerator",
    "build_critic_local_batch_bundle",
]
