from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from runtime.operation_trace import OperationTrace
from runtime.gan_rolephase_helios import (
    ScaledBackwardAccelerator,
    build_critic_local_batch_bundle,
)


def _trace(seed: int) -> OperationTrace:
    tensors = {
        "step/1000/critic/noise": torch.full((1, 2, 3), float(seed)),
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
                "control": {"pyramid_stage_call_sequence": [{"denoising_exit": 2}]},
            }
        ],
    }
    return OperationTrace(Path(f"/tmp/trace-{seed}"), manifest, tensors)


def test_local_batch_bundle_preserves_one_sample_and_one_trace() -> None:
    trace = _trace(1)
    x0 = torch.ones((1, 2, 3))
    source = {"text": torch.ones((1, 4)), "history": torch.ones((1, 2, 3))}
    bundle = build_critic_local_batch_bundle(
        logical_slot=0,
        generated_x0=x0,
        source=source,
        trace=trace,
    )
    assert bundle.local_batch == 1
    assert bundle.logical_slot == 0
    assert bundle.operation_trace is trace
    assert tuple(bundle.generated_x0.shape) == (1, 2, 3)


def test_local_batch_bundle_rejects_old_b2_shape() -> None:
    trace = _trace(1)
    with pytest.raises(ValueError, match="B1"):
        build_critic_local_batch_bundle(
            logical_slot=0,
            generated_x0=torch.ones((2, 2, 3)),
            source={"text": torch.ones((2, 4))},
            trace=trace,
        )


class _Base:
    def __init__(self) -> None:
        self.seen = None

    def backward(self, loss):
        self.seen = loss


def test_scaled_backward_is_identity_for_dp_mean_batch12() -> None:
    base = _Base()
    adapter = ScaledBackwardAccelerator(base, backend_dp_reduction="DP_MEAN")
    loss = torch.tensor(3.0)
    adapter.backward(loss)
    assert adapter.scale == pytest.approx(1.0)
    assert float(base.seen) == pytest.approx(3.0)


def test_scaled_backward_uses_one_over_12_for_sum_backend() -> None:
    base = _Base()
    adapter = ScaledBackwardAccelerator(base, backend_dp_reduction="SUM")
    adapter.backward(torch.tensor(12.0))
    assert adapter.scale == pytest.approx(1.0 / 12.0)
    assert float(base.seen) == pytest.approx(1.0)
