from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from runtime.operation_trace import OperationTrace
from runtime.gan_rolephase_helios import (
    BatchedOperationTrace,
    build_critic_local_batch_bundle,
)


def _trace(seed: int, *, control_exit: int = 2) -> OperationTrace:
    tensors = {
        "step/1000/critic/noise": torch.full((1, 2, 3), float(seed)),
        "step/1000/critic/crop": torch.tensor([[seed, seed + 1]], dtype=torch.int64),
        "shared/scalar": torch.tensor(7, dtype=torch.int64),
    }
    manifest = {
        "trace_hash": f"{seed:064x}"[-64:],
        "contract": {},
        "operations": [
            {
                "key": "step/1000/critic/rollout",
                "component": "generator",
                "operation": "rollout",
                "tensor_keys": ["step/1000/critic/noise"],
                "control": {"pyramid_stage_call_sequence": [{"denoising_exit": control_exit}]},
            }
        ],
    }
    return OperationTrace(Path(f"/tmp/trace-{seed}"), manifest, tensors)


def test_batched_operation_trace_concatenates_sample_tensors() -> None:
    left, right = _trace(1), _trace(2)
    batched = BatchedOperationTrace((left, right))
    assert batched.local_batch == 2
    noise = batched.tensor("step/1000/critic/noise")
    crop = batched.tensor("step/1000/critic/crop")
    assert tuple(noise.shape) == (2, 2, 3)
    assert tuple(crop.shape) == (2, 2)
    assert batched.tensor("shared/scalar").item() == 7
    assert batched.operation("step/1000/critic/rollout")["control"] == left.operation(
        "step/1000/critic/rollout"
    )["control"]


def test_batched_operation_trace_rejects_control_drift() -> None:
    batched = BatchedOperationTrace((_trace(1, control_exit=2), _trace(2, control_exit=3)))
    with pytest.raises(ValueError, match="identical control"):
        batched.operation("step/1000/critic/rollout")


def test_local_batch_bundle_batches_x0_sources_and_trace_together() -> None:
    traces = [_trace(1), _trace(2)]
    x0 = [torch.ones((1, 2, 3)), torch.full((1, 2, 3), 2.0)]
    sources = [
        {"text": torch.ones((1, 4)), "history": torch.ones((1, 2, 3))},
        {"text": torch.full((1, 4), 2.0), "history": torch.full((1, 2, 3), 2.0)},
    ]
    bundle = build_critic_local_batch_bundle(
        logical_slots=(0, 12),
        generated_x0=x0,
        sources=sources,
        traces=traces,
    )
    assert bundle.local_batch == 2
    assert tuple(bundle.generated_x0.shape) == (2, 2, 3)
    assert tuple(bundle.source["text"].shape) == (2, 4)
    assert getattr(bundle.operation_trace, "local_batch") == 2
