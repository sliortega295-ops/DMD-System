from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from distillschedule.runtime.fsdp2_nograd_persistence import (
    EXPECTED_HELIOS_FSDP_MODULES,
    generator_nograd_persistent_unshard_scope,
)
from distillschedule.runtime.fsdp2_p0 import StaticFSDP2P0Policy


@dataclass
class _Group:
    is_unsharded: bool = False


class _State:
    def __init__(self) -> None:
        self._fsdp_param_group = _Group()


class _FakeFSDP:
    def __init__(self, children: list["_FakeFSDP"] | None = None) -> None:
        self.children = children or []
        self.state = _State()
        self.policies: list[tuple[bool, bool]] = []
        self.reshard_count = 0

    def modules(self):
        yield self
        for child in self.children:
            yield from child.modules()

    def _get_fsdp_state(self):
        return self.state

    def set_reshard_after_forward(self, value: bool, recurse: bool = True) -> None:
        self.policies.append((value, recurse))

    def reshard(self) -> None:
        self.reshard_count += 1
        self.state._fsdp_param_group.is_unsharded = False


def _tree() -> _FakeFSDP:
    return _FakeFSDP([_FakeFSDP() for _ in range(40)])


def test_policy_is_opt_in_and_restricted_to_world16_b1() -> None:
    assert StaticFSDP2P0Policy().generator_nograd_persistent_unshard is False
    policy = StaticFSDP2P0Policy(generator_nograd_persistent_unshard=True)
    assert policy.id.endswith("generatorpersistunshard")
    with pytest.raises(ValueError, match="world16 FSDP2 B1"):
        StaticFSDP2P0Policy(
            rollout_batch_size=2, generator_nograd_persistent_unshard=True
        )
    with pytest.raises(ValueError, match="world16 FSDP2 B1"):
        StaticFSDP2P0Policy(
            mesh_policy="node-hsdp", generator_nograd_persistent_unshard=True
        )


def test_scope_retains_then_restores_all_41_fsdp_groups() -> None:
    root = _tree()
    empty_cache_calls: list[int] = []
    snapshots = iter(
        {"allocated_bytes": i, "reserved_bytes": i} for i in range(4)
    )
    with generator_nograd_persistent_unshard_scope(
        root,
        enabled=True,
        module_type=_FakeFSDP,
        empty_cache=lambda: empty_cache_calls.append(1),
        memory_snapshot=lambda: next(snapshots),
    ) as receipt:
        modules = list(root.modules())
        assert len(modules) == EXPECTED_HELIOS_FSDP_MODULES
        assert all(value.policies == [(False, False)] for value in modules)
        for value in modules:
            value.state._fsdp_param_group.is_unsharded = True
        receipt["rollout_batch_count"] += 5

    modules = list(root.modules())
    assert len(empty_cache_calls) == 2
    assert receipt["fully_unsharded_before_exit_count"] == 41
    assert receipt["unsharded_after_exit_count"] == 0
    assert receipt["rollout_batch_count"] == 5
    assert receipt["restored"] is True
    assert all(value.policies == [(False, False), (True, False)] for value in modules)
    assert all(value.reshard_count == 1 for value in modules)


def test_scope_fails_closed_if_a_forward_did_not_materialize_every_group() -> None:
    root = _tree()
    with pytest.raises(AssertionError, match="did not materialize every FSDP unit"):
        with generator_nograd_persistent_unshard_scope(
            root,
            enabled=True,
            module_type=_FakeFSDP,
            empty_cache=lambda: None,
            memory_snapshot=lambda: {},
        ):
            for value in list(root.modules())[:-1]:
                value.state._fsdp_param_group.is_unsharded = True
    assert all(
        not value.state._fsdp_param_group.is_unsharded for value in root.modules()
    )


def test_persistent_unshard_flag_is_wired_end_to_end() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = (
        root / "scripts/distillschedule/probe_static_fsdp2_p0_world16.py",
        root / "scripts/distillschedule/run_static_fsdp2_p0_world16.py",
        root / "scripts/distillschedule/launch_static_fsdp2_p0_world16_node.sh",
        root / "scripts/distillschedule/summarize_static_fsdp2_p0_world16.py",
    )
    for path in paths:
        source = path.read_text()
        assert "generator-nograd-persistent-unshard" in source
        assert "generator_nograd_persistent_unshard" in source
    launcher = paths[2].read_text()
    assert "DISTILLSCHEDULE_HELIOS_GENERATOR_NOGRAD_PERSISTENT_UNSHARD" in launcher
    summarizer = paths[3].read_text()
    assert "rank_generator_nograd_persistent_unshard" in summarizer
    hardened = (
        root
        / "scripts/distillschedule/probe_static_fsdp2_p0_world16_hardened.py"
    ).read_text()
    assert "outputs, records, persistence = _original_precompute" in hardened
    assert "return outputs, records, persistence" in hardened
