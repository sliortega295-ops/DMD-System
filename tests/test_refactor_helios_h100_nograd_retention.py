"""A separate H100 experiment never rewrites the frozen winner or the math."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from distillschedule.runtime.fsdp2_p0 import StaticFSDP2P0Policy


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "h100_retention_summary",
    ROOT / "scripts/distillschedule/summarize_helios_hardware_p0.py",
)
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)


def test_only_registered_persistence_flag_changes():
    base = SUMMARY.freeze_policy(ROOT, variant="no-gan", arm="best")
    candidate = SUMMARY.freeze_policy(ROOT, variant="no-gan", arm="h100-nograd-retain")
    expected = dict(base["policy"], generator_nograd_persistent_unshard=True)
    expected.pop("id")
    observed = dict(candidate["policy"])
    observed.pop("id")
    assert expected == observed
    assert base["policy"]["generator_nograd_persistent_unshard"] is False
    assert candidate["policy_fingerprint"] != base["policy_fingerprint"]
    assert candidate["winner"].startswith("EXPERIMENT_NOT_PROMOTED:")
    for key in ("contract", "trace_hash", "logical_hash", "mesh_shape", "profile_sha256"):
        assert candidate[key] == base[key]
    assert "--generator-nograd-persistent-unshard" in SUMMARY.policy_arguments(candidate["policy"])
    assert "experiment" not in base


def test_cannot_change_gan_policy():
    with pytest.raises((ValueError, AssertionError), match="No-GAN only"):
        SUMMARY.freeze_policy(ROOT, variant="gan", arm="h100-nograd-retain")


def test_a100_hsdp_guard_remains_closed(monkeypatch):
    monkeypatch.setenv('DISTILLSCHEDULE_HARDWARE_PROFILE', 'a100-16-b16')
    with pytest.raises(ValueError, match='world16 FSDP2 B1'):
        StaticFSDP2P0Policy(mesh_policy='node-hsdp', generator_nograd_persistent_unshard=True)


def test_explicit_h100_mesh_and_default_serialization_are_bounded():
    import dataclasses
    candidate = StaticFSDP2P0Policy(mesh_policy='node-hsdp',
                                  generator_nograd_persistent_unshard=True,
                                  hardware_profile_name='h100-8-b16')
    assert 'hardware_profile_name' not in dataclasses.asdict(candidate)
    for batch in (2, 5):
        with pytest.raises(ValueError, match='world16 FSDP2 B1'):
            StaticFSDP2P0Policy(mesh_policy='node-hsdp', rollout_batch_size=batch,
                               generator_nograd_persistent_unshard=True,
                               hardware_profile_name='h100-8-b16')


def test_capability_receipt_identity_and_missing_ranks_fail_closed(tmp_path):
    path = tmp_path / 'receipt.json'
    path.write_text('{}')
    frozen = SUMMARY.freeze_policy(ROOT, variant='no-gan', arm='h100-nograd-retain')
    with pytest.raises((ValueError, AssertionError), match='identity mismatch'):
        SUMMARY.validate_h100_retention_capability(path, frozen)
    with pytest.raises((ValueError, AssertionError), match='identity mismatch'):
        SUMMARY.validate_h100_retention_capability(tmp_path/'absent.json', frozen)


def test_receipt_flags_are_not_accepted_as_actual_lifecycle_proof(tmp_path):
    # Synthetic invalid CPU fixture, never production GPU evidence.
    path = tmp_path / 'invalid.json'
    row = dict(rank=0, status='PASS_BITWISE_HSDP41_NOGRAD_AND_POST_SCOPE_GRAD_UPDATE',
               mesh={'replicate':2,'shard':4}, torch='2.10.0+cu128',
               retained_actual_allgather_calls=41, baseline_actual_allgather_calls=205,
               persistence={'restored':False,'unsharded_after_exit_count':1,'fsdp_module_count':41},
               source_math_changed=False)
    value = dict(status='PASS_PRIMITIVE_API_ONLY',rank_count=8,launcher_rc=0,
                 cgroup_oom_delta={'oom':0,'oom_kill':0},per_rank=[dict(row,rank=r) for r in range(8)])
    path.write_text(json.dumps(value))
    frozen=SUMMARY.freeze_policy(ROOT,variant='no-gan',arm='h100-nograd-retain')
    frozen['experiment']['capability_evidence']['sha256']=SUMMARY.file_sha256(path)
    with pytest.raises((ValueError, AssertionError), match='lifecycle mismatch'):
        SUMMARY.validate_h100_retention_capability(path, frozen)


@pytest.mark.parametrize("change", [{"mesh_policy": "world16-fsdp"},
                                  {"rollout_batch_size": 2},
                                  {"generator_nograd_static_context_cache": True}])
def test_extra_overrides_fail_closed(monkeypatch, change):
    original = SUMMARY._object
    plan = json.loads((ROOT / "configs/distillschedule/experiments/helios_h100_nograd_retain_v1.json").read_text())
    plan = copy.deepcopy(plan)
    plan["overrides"].update(change)
    monkeypatch.setattr(SUMMARY, "_object", lambda path: plan if path.name == "helios_h100_nograd_retain_v1.json" else original(path))
    with pytest.raises((ValueError, AssertionError), match="unreviewed"):
        SUMMARY.freeze_policy(ROOT, variant="no-gan", arm="h100-nograd-retain")
