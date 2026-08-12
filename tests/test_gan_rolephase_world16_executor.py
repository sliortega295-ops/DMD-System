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
from scripts.distillschedule import summarize_gan_rolephase_world16 as summarizer
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


def test_tail4_streamer_is_bound_to_the_phase_b_critic_component() -> None:
    source = Path(worker.__file__).read_text()
    call = source[source.index("residency = install_frozen_block_streamer(") :]
    call = call[: call.index(")\n        policy =")]
    assert 'logical_component="gan_rolephase_phase_b_critic"' in call


def test_resource_monitor_emits_a_host_memory_value() -> None:
    launcher = Path(worker.__file__).with_name(
        "launch_gan_rolephase_world16_node.sh"
    )
    source = launcher.read_text()
    assert 'host_used="$(awk ' in source
    assert '${host_used}' in source
    assert 'printf \\"%.0f\\"' not in source


def test_only_g201_extraction_owner_is_cold_preconstructed(monkeypatch) -> None:
    roles: list[str] = []

    def build(*, args, role, logical):
        roles.append(role)
        return object(), {"role": role}, {"component": role}

    monkeypatch.setattr(worker, "_build_cpu_model", build)
    monkeypatch.setattr(
        worker.pr6,
        "pin_module_cpu_registered_state",
        lambda _model: {"all_registered_state_pinned": True},
    )
    monkeypatch.setattr(worker.pr6, "_control_barrier", lambda _group: None)
    targets = worker._preconstruct_targets(
        args=SimpleNamespace(),
        global_rank=worker.FLEX_RANKS[0],
        logical=object(),
        control_group=object(),
    )
    assert roles == []
    assert targets["forward_model"] is None
    assert targets["reverse_model"] is None
    assert targets["forward_policy"] == "LAZY_AFTER_PHASE_A_SOURCE_RELEASE"
    assert targets["reverse_policy"] == "LAZY_AFTER_PHASE_B_CRITIC_RELEASE"

    owner = worker._preconstruct_targets(
        args=SimpleNamespace(),
        global_rank=worker.PHASE_B_G_RANKS[0],
        logical=object(),
        control_group=object(),
    )
    assert roles == ["generator"]
    assert owner["forward_model"] is not None
    assert owner["forward_policy"] == "COLD_PRECONSTRUCT_G201_EXTRACTION_OWNER"


def test_flex_forward_target_is_built_after_phase_a_release(monkeypatch) -> None:
    roles: list[str] = []

    def build(*, args, role, logical):
        roles.append(role)
        return object(), {"role": role}, {"component": role}

    monkeypatch.setattr(worker, "_build_cpu_model", build)
    monkeypatch.setattr(
        worker.pr6,
        "pin_module_cpu_registered_state",
        lambda _model: {"timing_scope": "excluded_cold_preparation"},
    )
    targets = {
        "forward_model": None,
        "forward_metadata": None,
        "forward_restore": None,
        "forward_pin": None,
        "forward_policy": "LAZY_AFTER_PHASE_A_SOURCE_RELEASE",
    }
    observed = worker._construct_forward_target_after_phase_a(
        args=SimpleNamespace(),
        global_rank=worker.FLEX_RANKS[0],
        logical=object(),
        targets=targets,
    )
    assert roles == ["critic"]
    assert observed["constructed_in_handoff"] is True
    assert observed["host_preparation"]["timing_scope"] == "included_handoff_to_b"
    assert targets["forward_model"] is not None


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


def test_input_gate_binds_gan_precast_to_logical_bootstrap(tmp_path: Path) -> None:
    marker = {
        "status": "PASS",
        "classification": "systems_only_gan_precast",
        "artifact_hash": "a" * 64,
        "bootstrap_artifact_hash": "b" * 64,
        "head_fp32_sha256": "c" * 64,
        "head_parameter_count": 130,
    }
    (tmp_path / input_gate.GAN_PRECAST_MARKER).write_text(json.dumps(marker))
    observed = input_gate.gan_precast_contract(
        tmp_path, expected_bootstrap_artifact_hash="b" * 64
    )
    assert observed["bootstrap_artifact_hash"] == "b" * 64
    assert observed["head_parameter_count"] == 130


def test_input_gate_rejects_plain_or_wrong_bootstrap_score_precast(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="marker is missing"):
        input_gate.gan_precast_contract(
            tmp_path, expected_bootstrap_artifact_hash="b" * 64
        )
    marker = {
        "status": "PASS",
        "classification": "systems_only_gan_precast",
        "bootstrap_artifact_hash": "c" * 64,
    }
    (tmp_path / input_gate.GAN_PRECAST_MARKER).write_text(json.dumps(marker))
    with pytest.raises(ValueError, match="pre-cast/bootstrap mismatch"):
        input_gate.gan_precast_contract(
            tmp_path, expected_bootstrap_artifact_hash="b" * 64
        )


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


def test_resource_parser_accepts_preserved_scientific_host_bytes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "resource.csv"
    path.write_text(
        "unix_time,cgroup_bytes,host_used_bytes,gpu_used_mib,gpu_util_percent\n"
        "1.0,123,3.28281e+10,1:2:3:4:5:6:7:8,0:0:0:0:0:0:0:0\n"
    )
    value = summarizer.resource_peak(path)
    assert value["host_used_peak_bytes"] == 32_828_100_000
    assert value["per_gpu_peak_mib"] == list(range(1, 9))
