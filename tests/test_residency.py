from __future__ import annotations

import pytest
import torch
from torch import nn

from runtime.residency import (
    AdapterModeConflictError,
    ModuleResidencyBackend,
    ParameterVersionError,
    ResidencyPolicy,
)


def _fake_backend(policy=ResidencyPolicy.TIME_MULTIPLEX):
    moves = []
    backend = ModuleResidencyBackend(
        accelerator_device="cuda:0",
        host_device="cpu",
        policy=policy,
        move_function=lambda module, target: moves.append((module, target)),
        synchronize_function=lambda device: None,
    )
    return backend, moves


def test_shared_aliases_materialize_one_physical_state_and_reference_count():
    backend, moves = _fake_backend()
    generator = nn.Linear(4, 4)
    score = nn.Linear(4, 4)
    backend.register_state(
        "generator", generator, aliases=("Generator",), initial_device="cuda:0"
    )
    backend.register_state(
        "score",
        score,
        aliases=("RealScore", "FakeScore", "Critic", "GAN"),
        modes=("real", "fake"),
        initial_device="cuda:0",
    )
    backend.initialize("Generator")
    assert moves == [(score, "cpu")]

    with backend.acquire("RealScore", mode="real", expected_version=0):
        assert moves[-1] == (score, "cuda:0")
        with backend.acquire("RealScore", mode="real"):
            assert backend.snapshot()["states"]["score"]["active_alias_counts"] == {
                "RealScore": 2
            }
    snapshot = backend.snapshot()
    assert snapshot["states"]["score"]["active_alias_counts"] == {}
    assert snapshot["transfer"]["count"] == 3


def test_active_state_is_not_offloaded_and_adapter_modes_cannot_conflict():
    backend, moves = _fake_backend()
    generator = nn.Linear(2, 2)
    score = nn.Linear(2, 2)
    backend.register_state(
        "generator", generator, aliases=("Generator",), initial_device="cuda:0"
    )
    backend.register_state(
        "score",
        score,
        aliases=("RealScore", "FakeScore"),
        modes=("real", "fake"),
        initial_device="cpu",
    )
    with backend.acquire("Generator"):
        with backend.acquire("RealScore", mode="real"):
            assert (generator, "cpu") not in moves
            with pytest.raises(AdapterModeConflictError):
                with backend.acquire("FakeScore", mode="fake"):
                    pass
    assert any(event.action == "retain_active" for event in backend.events)


def test_parameter_versions_are_checked_and_monotonic():
    backend, _ = _fake_backend(ResidencyPolicy.RESIDENT)
    module = nn.Linear(2, 2)
    backend.register_state("score", module, aliases=("Critic",), version=0)
    with backend.acquire("Critic", expected_version=0):
        pass
    backend.mark_updated("Critic", 1)
    with pytest.raises(ParameterVersionError):
        with backend.acquire("Critic", expected_version=0):
            pass
    with pytest.raises(ParameterVersionError):
        backend.mark_updated("Critic", 1)
