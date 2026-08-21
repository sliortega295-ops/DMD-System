from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from distillschedule.runtime.wan13b_upper_bound import (
    GLOBAL_BATCH,
    LORA_RANK,
    PLACEMENTS,
    WAN13B_BLOCK_COUNT,
    Wan13BBootstrap,
    file_sha256,
)


ROOT = Path(__file__).resolve().parents[1]


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def test_wan13b_contract_is_a_real_model_matched_placement_ablation() -> None:
    value = json.loads(
        (
            ROOT
            / "configs/distillschedule/baselines/wan13b_nogan_world16_upper_bound.json"
        ).read_text()
    )
    assert value["model"]["source"] == "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    assert value["model"]["block_count"] == WAN13B_BLOCK_COUNT == 30
    assert value["model"]["generator_lora_rank"] == LORA_RANK == 256
    assert value["global_batch"] == GLOBAL_BATCH == 16
    assert value["rollout_batch_size"] == 1
    assert value["score_packing"] is False
    assert value["generator_updates"] == 1
    assert value["critic_updates"] == 5
    assert value["ema_update_steps"] == [10, 11, 12, 13, 14]
    assert set(value["placements"]) == set(PLACEMENTS)
    assert value["common_residency"] == {
        "parameter_offload": False,
        "activation_offload": False,
        "optimizer_offload": False,
        "frozen_parameter_streaming": False,
        "model_swap": False,
        "generator_residual_storage": "cuda_out_of_place",
        "critic_residual_storage": "cuda_out_of_place",
        "ema_device": "cuda",
    }


def test_existing_helios14b_fsdp_default_remains_40_blocks() -> None:
    source = (ROOT / "src/distillschedule/runtime/fsdp2_p0.py").read_text()
    assert "expected_block_count: int = 40" in source
    assert "f\"{expected_block_count}_BLOCKS_BOTTOM_UP_THEN_ROOT\"" in source


def test_bootstrap_manifest_is_hash_and_payload_bound(tmp_path: Path) -> None:
    for name in ("generator", "critic"):
        (tmp_path / f"{name}.safetensors").write_bytes(name.encode())
    value = {
        "schema_version": "distillschedule.wan13b_bootstrap.v1",
        "block_count": WAN13B_BLOCK_COUNT,
        "lora_rank": LORA_RANK,
        "components": {
            name: {
                "trainable_payload": f"{name}.safetensors",
                "sha256": file_sha256(tmp_path / f"{name}.safetensors"),
                "parameter_names": [],
            }
            for name in ("generator", "critic")
        },
    }
    value["manifest_hash"] = hashlib.sha256(_canonical(value)).hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps(value))
    loaded = Wan13BBootstrap.load(tmp_path)
    assert loaded.manifest_hash == value["manifest_hash"]
    (tmp_path / "critic.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="critic trainable payload hash changed"):
        Wan13BBootstrap.load(tmp_path)


def test_worker_fails_closed_on_the_frozen_systems_contract() -> None:
    source = (
        ROOT / "scripts/distillschedule/probe_wan13b_upper_bound_world16.py"
    ).read_text()
    assert 'rollout_batch_size=1' in source
    assert 'pack_real_score_pair=False' in source
    assert 'chunked_residual_storage="cuda_out_of_place"' in source
    assert '"parameter_offload": False' in source
    assert '"activation_offload": False' in source
    assert '"optimizer_offload": False' in source
    assert '"generator": 1, "critic": 5, "ema": 5' in source
    assert "expected_block_count=WAN13B_BLOCK_COUNT" in source
    assert "missing_pyramid_patch_seed=WAN13B_PYRAMID_PATCH_SEED" in source


def test_resident_control_syncs_only_present_trainable_gradients() -> None:
    source = (
        ROOT / "src/distillschedule/runtime/wan13b_upper_bound.py"
    ).read_text()
    assert "if parameter.requires_grad and parameter.grad is not None" in source
    assert "dist.all_reduce_coalesced" in source
    assert "gradient.mul_(1.0 / WORLD_SIZE)" in source
    assert "clip_grad_norm_" in source
    assert source.index("_synchronize_trainable_gradients()") < source.index(
        "clip_grad_norm_(trainables"
    )


def test_bootstrap_builder_materializes_one_shared_artifact() -> None:
    source = (
        ROOT / "scripts/distillschedule/build_wan13b_systems_bootstrap.py"
    ).read_text()
    assert "if args.output.exists()" in source
    assert "raise FileExistsError(args.output)" in source
    assert '"official_training_checkpoint": False' in source
    assert '"generator": _component' in source
    assert '"critic": _component' in source
    assert '"model_identity_sha256"' in source


def test_controller_ignores_stale_user_controlmaster_socket() -> None:
    source = (
        ROOT / "scripts/distillschedule/run_wan13b_upper_bound_world16.py"
    ).read_text()
    assert '"ControlMaster=no"' in source
    assert '"ControlPath=none"' in source


def test_public_wan_missing_pyramid_modules_are_explicitly_materialized() -> None:
    loader = (ROOT / "runtime/critic_pool_helios.py").read_text()
    bootstrap = (
        ROOT / "scripts/distillschedule/build_wan13b_systems_bootstrap.py"
    ).read_text()
    assert "missing_pyramid_patch_seed: int | None = None" in loader
    assert 'for name in ("patch_short", "patch_mid", "patch_long")' in loader
    assert 'module.to_empty(device="cpu")' in loader
    assert "module.to(dtype=torch.bfloat16)" in loader
    assert "remaining_meta" in loader
    assert "WAN13B_PYRAMID_PATCH_SEED" in bootstrap


def test_correctness_rollout_evidence_is_persisted_and_receipted() -> None:
    source = (
        ROOT / "scripts/distillschedule/probe_wan13b_upper_bound_world16.py"
    ).read_text()
    body = source.split("def _save_rollout_evidence(", 1)[1].split(
        "def _save_trainable_evidence(", 1
    )[0]
    assert "save_file(" in body
    assert '"sha256": file_sha256(path)' in body
    assert '"tensor_keys": list(state)' in body


def test_profile_mode_is_diagnostic_only_and_uses_one_outer_session_per_node() -> None:
    worker = (
        ROOT / "scripts/distillschedule/probe_wan13b_upper_bound_world16.py"
    ).read_text()
    launcher = (
        ROOT / "scripts/distillschedule/launch_wan13b_upper_bound_world16_node.sh"
    ).read_text()
    controller = (
        ROOT / "scripts/distillschedule/run_wan13b_upper_bound_world16.py"
    ).read_text()
    summarizer = (
        ROOT / "scripts/distillschedule/summarize_wan13b_upper_bound.py"
    ).read_text()
    analyzer = (
        ROOT / "scripts/distillschedule/analyze_wan13b_nsys.py"
    ).read_text()

    assert 'value.add_argument("--profile-mode", action="store_true")' in worker
    assert "if args.timing_mode and args.profile_mode" in worker
    assert 'stop_point="after_step14_ema_update"' in worker
    assert 'torch.cuda.nvtx.range_push("WAN13B_FULL5/world16-fsdp")' in worker
    assert "--profile-mode" in launcher
    assert launcher.count('"${NSYS_BIN}" profile') == 1
    assert "DISTILLSCHEDULE_STATIC_FSDP2_NSYS_CONTROL_API=0" in launcher
    assert "clip analysis to the exact balanced WAN13B_FULL5" in launcher
    assert "--trace-fork-before-exec=true" in launcher
    assert "--wait=primary" in launcher
    assert "expected_stop_point=\"after_step14_ema_update\"" in launcher
    assert "expected_controller_ranks=[]" in launcher
    assert '"profile"' in controller
    assert '"--nsys-bin"' in controller
    assert "2024.6.2.225/bin/nsys" in controller
    assert "nsys_version_not_admissible" in launcher
    assert '"diagnostic_only": args.mode == "profile"' in summarizer
    assert "NCCL interval union minus its intersection" in analyzer
    assert "sum of the 16 per-GPU outer-range walls" in analyzer
