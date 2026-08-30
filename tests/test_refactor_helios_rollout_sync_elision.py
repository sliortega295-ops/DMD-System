from __future__ import annotations

import inspect
from pathlib import Path

from distillschedule.runtime.fsdp2_p0 import StaticFSDP2P0Policy
from runtime.gst_pool_helios import run_generator_update_trace_rollout
from runtime.static_fsdp2_p0_helios import run_generator_trace_rollout_batch


ROOT = Path(__file__).resolve().parents[1]


def test_rollout_sync_elision_is_conservative_and_explicit() -> None:
    assert StaticFSDP2P0Policy().elide_explicit_rollout_sync is False
    policy = StaticFSDP2P0Policy(elide_explicit_rollout_sync=True)
    assert policy.id.endswith("rolloutnosync")

    update_signature = inspect.signature(run_generator_update_trace_rollout)
    assert update_signature.parameters["empty_cache_before_rollout"].default is True
    assert update_signature.parameters["synchronize_after_rollout"].default is True
    batch_signature = inspect.signature(run_generator_trace_rollout_batch)
    assert batch_signature.parameters["synchronize_after_rollout"].default is True


def test_rollout_sync_elision_is_propagated_end_to_end() -> None:
    paths = (
        "scripts/distillschedule/run_static_fsdp2_p0_world16.py",
        "scripts/distillschedule/launch_static_fsdp2_p0_world16_node.sh",
        "scripts/distillschedule/probe_static_fsdp2_p0_world16.py",
        "scripts/distillschedule/probe_static_fsdp2_p0_world16_hardened.py",
        "scripts/distillschedule/summarize_static_fsdp2_p0_world16.py",
    )
    sources = {path: (ROOT / path).read_text() for path in paths}
    for path, source in sources.items():
        assert "elide-explicit-rollout-sync" in source, path

    launcher = sources[
        "scripts/distillschedule/launch_static_fsdp2_p0_world16_node.sh"
    ]
    assert "DISTILLSCHEDULE_ELIDE_EXPLICIT_ROLLOUT_SYNC" in launcher
    assert "outer_full5_boundary_sync_preserved" in sources[
        "scripts/distillschedule/probe_static_fsdp2_p0_world16.py"
    ]
    assert "rollout fence request drifted" in sources[
        "scripts/distillschedule/summarize_static_fsdp2_p0_world16.py"
    ]


def test_gan_and_critic_rollouts_receive_the_policy() -> None:
    gan_worker = (
        ROOT / "scripts/distillschedule/probe_static_fsdp2_full5_world16.py"
    ).read_text()
    p0_worker = (
        ROOT / "scripts/distillschedule/probe_static_fsdp2_p0_world16.py"
    ).read_text()
    assert 'getattr(args, "elide_explicit_rollout_sync", False)' in gan_worker
    assert "synchronize_after_rollout=not policy.elide_explicit_rollout_sync" in p0_worker
    assert "empty_cache_before_rollout=not policy.elide_explicit_rollout_sync" in p0_worker
