from __future__ import annotations

from types import SimpleNamespace

import pytest
torch = pytest.importorskip("torch")
from torch.distributed.tensor import Replicate, Shard

from distillschedule.runtime.fsdp2_static import _optimizer_dispatch_warmup


class FakeMesh:
    mesh_dim_names = ("replicate", "shard")


class FakeDTensor:
    def __init__(self, local: torch.Tensor, mesh: FakeMesh) -> None:
        self._local = local
        self.device_mesh = mesh
        self.placements = (Replicate(), Shard(0))
        self.grad = None

    def detach(self) -> "FakeDTensor":
        return self

    def to_local(self) -> torch.Tensor:
        return self._local


class FakeOptimizer:
    def __init__(self, parameters: list[FakeDTensor]) -> None:
        self.parameters = parameters
        self.param_groups = [{"lr": 2e-6, "weight_decay": 1e-3}]
        self.state = {
            parameter: {
                "step": torch.tensor(2.0),
                "exp_avg": FakeDTensor(torch.arange(4, dtype=torch.float32), parameter.device_mesh),
                "exp_avg_sq": FakeDTensor(
                    torch.arange(4, dtype=torch.float32).square(),
                    parameter.device_mesh,
                ),
            }
            for parameter in parameters
        }

    def step(self) -> None:
        assert self.param_groups[0]["lr"] == 0.0
        assert self.param_groups[0]["weight_decay"] == 0.0
        for parameter in self.parameters:
            state = self.state[parameter]
            state["step"].add_(1)
            state["exp_avg"].to_local().zero_()
            state["exp_avg_sq"].to_local().zero_()

    def zero_grad(self, *, set_to_none: bool) -> None:
        assert set_to_none
        for parameter in self.parameters:
            parameter.grad = None


def test_optimizer_dispatch_warmup_restores_state_and_keeps_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mesh = FakeMesh()
    parameters = [
        FakeDTensor(torch.tensor([1.0, 2.0]), mesh),
        FakeDTensor(torch.tensor([3.0, 4.0]), mesh),
    ]
    optimizer = FakeOptimizer(parameters)
    before_parameters = [value.to_local().clone() for value in parameters]
    before_states = [
        (
            optimizer.state[value]["step"].clone(),
            optimizer.state[value]["exp_avg"].to_local().clone(),
            optimizer.state[value]["exp_avg_sq"].to_local().clone(),
        )
        for value in parameters
    ]

    monkeypatch.setattr(torch, "zeros_like", lambda value: SimpleNamespace())
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", lambda values, limit: torch.tensor(0.0))
    receipt = _optimizer_dispatch_warmup(
        optimizer=optimizer,
        trainables=parameters,
        gradient_clip=1.0,
    )

    assert receipt["enabled"] is True
    assert receipt["scope"] == "SETUP_ONLY_BEFORE_FORMAL_TIMER"
    assert receipt["configured_gradient_clip"] == 1.0
    assert receipt["parameter_changed_elements_before_restore"] == 0
    assert receipt["state_changed_elements_after_restore"] == 0
    assert receipt["step_state_restored"] is True
    assert receipt["gradients_cleared"] is True
    assert optimizer.param_groups == [{"lr": 2e-6, "weight_decay": 1e-3}]
    for parameter, expected, state_expected in zip(
        parameters, before_parameters, before_states, strict=True
    ):
        assert torch.equal(parameter.to_local(), expected)
        state = optimizer.state[parameter]
        assert torch.equal(state["step"], state_expected[0])
        assert torch.equal(state["exp_avg"].to_local(), state_expected[1])
        assert torch.equal(state["exp_avg_sq"].to_local(), state_expected[2])
        assert parameter.grad is None


def test_optimizer_dispatch_warmup_fails_closed_outside_node_hsdp() -> None:
    parameter = FakeDTensor(torch.tensor([1.0]), FakeMesh())
    parameter.device_mesh.mesh_dim_names = ("fsdp",)
    optimizer = FakeOptimizer([parameter])
    with pytest.raises(ValueError, match="node-HSDP"):
        _optimizer_dispatch_warmup(
            optimizer=optimizer,
            trainables=[parameter],
            gradient_clip=1.0,
        )
