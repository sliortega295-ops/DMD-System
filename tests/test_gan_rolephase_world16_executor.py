from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from runtime.critic_pool_checkpoint import (
    LOGICAL_CHECKPOINT_SCHEMA,
    LogicalCheckpointPackage,
)
from scripts.distillschedule import probe_gan_rolephase_world16 as worker
from scripts.distillschedule import validate_gan_rolephase_world16_inputs as input_gate


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def test_remote_score_gradient_is_injected_without_second_mean_scaling() -> None:
    generated = torch.randn(2, 3, requires_grad=True)
    gradient = torch.randn_like(generated)
    loss = worker._remote_gradient_sum_surrogate(generated, gradient)
    loss.backward()
    torch.testing.assert_close(generated.grad, gradient)


def test_phase_a_rendezvous_is_restored_after_global_world(monkeypatch) -> None:
    values = {
        "RANK": "3",
        "WORLD_SIZE": "8",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "41234",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    saved = worker._capture_local_phase_a_rendezvous(local_rank=3)
    for key, value in {
        "RANK": "11",
        "WORLD_SIZE": "16",
        "MASTER_ADDR": "10.0.0.1",
        "MASTER_PORT": "42345",
    }.items():
        monkeypatch.setenv(key, value)
    worker._restore_local_phase_a_rendezvous(saved, master_port=41244)
    expected = dict(values, MASTER_PORT="41244")
    assert {key: worker.os.environ[key] for key in values} == expected


def test_systems_only_loader_keeps_physical_and_logical_versions_distinct(
    tmp_path: Path,
) -> None:
    manifest = {
        "schema_version": LOGICAL_CHECKPOINT_SCHEMA,
        "global_step": 10,
        "components": {
            "generator": {
                "parameter_version": 2,
                "optimizer_step": 2,
                "scheduler_state": {"last_epoch": 2},
            },
            "critic": {
                "parameter_version": 10,
                "optimizer_step": 10,
                "scheduler_state": {"last_epoch": 10},
            },
        },
        "ema": {"optimization_step": 800},
        "systems_only_control": {
            "classification": "systems_only_gan_bootstrap",
            "logical_schedule_versions": {
                "generator": 200,
                "critic": 1000,
                "ema": 800,
            },
            "logical_control_window": list(range(1000, 1005)),
            "official_checkpoint_1000": False,
        },
    }
    manifest["manifest_hash"] = hashlib.sha256(_canonical(manifest)).hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    package = LogicalCheckpointPackage.load_systems_only_gan(tmp_path)
    assert package.component("generator")["optimizer_step"] == 2
    assert package.manifest["systems_only_control"]["logical_schedule_versions"][
        "generator"
    ] == 200


def test_worker_defaults_to_resident_phase_b_and_has_no_implicit_tail4() -> None:
    actions = worker._parser()._actions
    option = next(
        action for action in actions if "--phase-b-streamed-blocks" in action.option_strings
    )
    assert option.default == 0
    assert option.choices == (0, 4)


def test_input_gate_reads_resolution_from_helios_data_config() -> None:
    cfg = SimpleNamespace(
        data_config=SimpleNamespace(single_height=384, single_width=640),
        training_config=SimpleNamespace(
            dfake_gen_update_ratio=5,
            gan_start_step=1000,
            gan_g_weight=0.05,
            gan_d_weight=0.01,
            r1_weight=100.0,
            r2_weight=0.0,
            is_use_gan=True,
            is_use_reward_model=False,
            is_dmd_vae_decode=False,
        ),
    )
    observed = input_gate.official_config_contract(cfg)
    assert observed["single_height"] == 384
    assert observed["single_width"] == 640
    assert observed["gan_start_step"] == 1000


def test_nonmember_group_validation_never_introspects_sentinel(monkeypatch) -> None:
    sentinel = object()
    monkeypatch.setattr(
        worker.dist,
        "GroupMember",
        SimpleNamespace(NON_GROUP_MEMBER=sentinel),
    )

    def forbidden(_group):
        raise AssertionError("NON_GROUP_MEMBER must not be introspected")

    monkeypatch.setattr(worker.dist, "get_process_group_ranks", forbidden)
    assert worker._validated_group_ranks(
        sentinel,
        expected=(4, 5, 6),
        global_rank=0,
    ) == (4, 5, 6)
