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
