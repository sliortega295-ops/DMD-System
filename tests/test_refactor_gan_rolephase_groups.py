from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from distillschedule.runtime import gan_rolephase_groups as groups_module


def test_execution_groups_are_created_once_in_worldwide_deterministic_order(
    monkeypatch,
) -> None:
    calls: list[tuple[int, ...]] = []

    fake_dist = SimpleNamespace(
        is_initialized=lambda: True,
        get_world_size=lambda: 16,
        group=SimpleNamespace(WORLD="WORLD"),
        new_group=lambda ranks, backend: calls.append(tuple(ranks)) or tuple(ranks),
    )
    monkeypatch.setattr(groups_module, "dist", fake_dist)

    groups = groups_module.create_rolephase_execution_groups()
    assert calls[:5] == [
        tuple(range(8)),
        tuple(range(8, 16)),
        tuple(range(4)),
        tuple(range(4, 16)),
        tuple(range(4, 8)),
    ]
    assert calls[5:13] == [(lane, lane + 8) for lane in range(8)]
    assert calls[13:] == [(rank,) for rank in range(16)]
    assert groups.phase_a_pair_for_rank(0) == (0, 8)
    assert groups.phase_a_pair_for_rank(8) == (0, 8)
    assert groups.phase_a_pair_for_rank(15) == (7, 15)
    assert groups.singleton_for_rank(12) == (12,)


def test_real_worker_creates_only_exact_phase_b_groups(monkeypatch) -> None:
    calls: list[tuple[str, tuple[int, ...]]] = []
    barriers: list[object] = []

    def new_group(ranks, backend, **_kwargs):
        value = (backend, tuple(ranks))
        calls.append(value)
        return value

    fake_dist = SimpleNamespace(
        is_initialized=lambda: True,
        get_world_size=lambda: 16,
        new_group=new_group,
        barrier=lambda group: barriers.append(group),
    )
    monkeypatch.setattr(groups_module, "dist", fake_dist)

    groups = groups_module.create_phase_b_execution_groups()
    assert calls[0] == ("gloo", tuple(range(16)))
    assert calls[1] == ("nccl", tuple(range(4)))
    assert calls[2] == ("nccl", tuple(range(4, 16)))
    assert calls[3:] == [("nccl", (rank,)) for rank in range(16)]
    assert len(barriers) == len(calls)
    assert groups.generator == ("nccl", tuple(range(4)))
    assert groups.critic == ("nccl", tuple(range(4, 16)))
    assert groups.singleton_for_rank(12) == ("nccl", (12,))
