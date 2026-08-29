from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.distributed.tensor import Replicate, Shard

from distillschedule.runtime.fsdp2_static import _optimizer_dispatch_warmup


ROOT = Path(__file__).resolve().parents[1]


class FakeMesh:
    def __init__(self, mesh_dim_names: tuple[str, ...] = ("replicate", "shard")) -> None:
        self.mesh_dim_names = mesh_dim_names
        self.device_type = "cuda"
        self.mesh = (
            torch.arange(16).reshape(2, 8)
            if mesh_dim_names == ("replicate", "shard")
            else torch.arange(16)
        )


class FakeDTensor:
    def __init__(
        self,
        local: torch.Tensor,
        mesh: FakeMesh,
        placements: tuple[object, ...] = (Replicate(), Shard(0)),
    ) -> None:
        self._local = local
        self.device_mesh = mesh
        self.placements = placements
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
                "exp_avg": FakeDTensor(
                    torch.arange(4, dtype=torch.float32),
                    parameter.device_mesh,
                    parameter.placements,
                ),
                "exp_avg_sq": FakeDTensor(
                    torch.arange(4, dtype=torch.float32).square(),
                    parameter.device_mesh,
                    parameter.placements,
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
    assert receipt["topology"] == "node-hsdp"
    assert receipt["mesh_dim_names"] == ["replicate", "shard"]
    assert receipt["placement_types"] == ["Replicate", "Shard"]
    assert receipt["logical_mesh_rank_count"] == 16
    assert receipt["device_mesh_object_count"] == 1
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


def test_optimizer_dispatch_warmup_accepts_world16_fsdp_and_restores_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mesh = FakeMesh(("fsdp",))
    parameters = [
        FakeDTensor(torch.tensor([1.0, 2.0]), mesh, (Shard(0),)),
        FakeDTensor(torch.tensor([3.0, 4.0]), mesh, (Shard(0),)),
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
    monkeypatch.setattr(
        torch.nn.utils,
        "clip_grad_norm_",
        lambda values, limit: torch.tensor(0.0),
    )
    receipt = _optimizer_dispatch_warmup(
        optimizer=optimizer,
        trainables=parameters,
        gradient_clip=1.0,
    )

    assert receipt["topology"] == "world16-fsdp"
    assert receipt["mesh_dim_names"] == ["fsdp"]
    assert receipt["placement_types"] == ["Shard"]
    assert receipt["logical_mesh_rank_count"] == 16
    assert receipt["device_mesh_object_count"] == 1
    assert receipt["state_changed_elements_after_restore"] == 0
    for parameter, expected, state_expected in zip(
        parameters, before_parameters, before_states, strict=True
    ):
        assert torch.equal(parameter.to_local(), expected)
        state = optimizer.state[parameter]
        assert torch.equal(state["step"], state_expected[0])
        assert torch.equal(state["exp_avg"].to_local(), state_expected[1])
        assert torch.equal(state["exp_avg_sq"].to_local(), state_expected[2])
        assert parameter.grad is None


def test_optimizer_dispatch_warmup_accepts_equivalent_mesh_objects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_mesh = FakeMesh(("fsdp",))
    second_mesh = FakeMesh(("fsdp",))
    parameters = [
        FakeDTensor(torch.tensor([1.0, 2.0]), first_mesh, (Shard(0),)),
        FakeDTensor(torch.tensor([3.0, 4.0]), second_mesh, (Shard(0),)),
    ]
    optimizer = FakeOptimizer(parameters)
    monkeypatch.setattr(torch, "zeros_like", lambda value: SimpleNamespace())
    monkeypatch.setattr(
        torch.nn.utils,
        "clip_grad_norm_",
        lambda values, limit: torch.tensor(0.0),
    )

    receipt = _optimizer_dispatch_warmup(
        optimizer=optimizer,
        trainables=parameters,
        gradient_clip=1.0,
    )

    assert receipt["topology"] == "world16-fsdp"
    assert receipt["logical_mesh_rank_count"] == 16
    assert receipt["device_mesh_object_count"] == 2
    assert receipt["state_changed_elements_after_restore"] == 0


def test_optimizer_dispatch_warmup_fails_closed_on_unknown_mesh() -> None:
    parameter = FakeDTensor(torch.tensor([1.0]), FakeMesh(("tp",)))
    optimizer = FakeOptimizer([parameter])
    with pytest.raises(ValueError, match="world16.*node-HSDP"):
        _optimizer_dispatch_warmup(
            optimizer=optimizer,
            trainables=[parameter],
            gradient_clip=1.0,
        )


def test_world16_optimizer_warmup_is_not_rejected_by_the_node_launcher() -> None:
    launcher = (
        ROOT / "scripts/distillschedule/launch_static_fsdp2_p0_world16_node.sh"
    ).read_text()
    stale_node_hsdp_only_gate = (
        '[[ "${OPTIMIZER_DISPATCH_WARMUP}" != on || '
        '"${MESH_POLICY}" == node-hsdp ]] || exit 2'
    )
    assert stale_node_hsdp_only_gate not in launcher
    assert "DISTILLSCHEDULE_OPTIMIZER_DISPATCH_WARMUP" in launcher
    assert "--optimizer-dispatch-warmup" in launcher
