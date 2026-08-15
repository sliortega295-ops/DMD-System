from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from distillschedule.runtime.vllm_omni_rollout import (
    OFFICIAL_X0_SHAPE,
    VLLM_OMNI_BACKEND,
    VLLM_OMNI_HELIOS_BASE_COMMIT,
    VLLM_OMNI_HELIOS_BASE_TAG,
    VLLM_OMNI_HELIOS_LATENT_PATCH_PATH,
    VLLM_OMNI_HELIOS_LATENT_PATCH_SHA256,
    VLLM_OMNI_VERSION_FAMILY,
    VLLM_VERSION_FAMILY,
    VLLMOmniRolloutCapabilities,
    expected_generator_version,
    normalize_capabilities,
    parse_factory_spec,
    rollout_phase_matrix,
    source_has_latent_no_vae_guard,
    validate_rollout_receipt,
)


def test_capability_gate_accepts_exact_019_backport() -> None:
    capabilities = VLLMOmniRolloutCapabilities()
    receipt = capabilities.validate(
        requested_batch_size=5,
        expected_revision=VLLM_OMNI_HELIOS_BASE_COMMIT,
        require_hot_lora_sync=True,
    )
    assert receipt["status"] == "PASS"
    assert receipt["vae_decode_calls"] == 0
    assert receipt["compatibility_patch_sha256"] == VLLM_OMNI_HELIOS_LATENT_PATCH_SHA256
    assert receipt["vllm_omni_version_family"] == "0.19"
    assert receipt["vllm_version_family"] == "0.19"
    assert tuple(capabilities.output_shape) == OFFICIAL_X0_SHAPE


def test_mapping_capability_must_report_source_identity() -> None:
    with pytest.raises(ValueError, match="omitted source identity"):
        normalize_capabilities(
            {
                "backend": VLLM_OMNI_BACKEND,
                "output_type": "latent",
                "skips_vae_decode": True,
            }
        )
    capabilities = VLLMOmniRolloutCapabilities.from_mapping(
        {
            "upstream_revision": VLLM_OMNI_HELIOS_BASE_COMMIT,
            "compatibility_patch_sha256": VLLM_OMNI_HELIOS_LATENT_PATCH_SHA256,
            "vllm_omni_version_family": VLLM_OMNI_VERSION_FAMILY,
            "vllm_version_family": VLLM_VERSION_FAMILY,
        }
    )
    assert capabilities.upstream_revision == VLLM_OMNI_HELIOS_BASE_COMMIT


def test_capability_gate_rejects_missing_training_requirements() -> None:
    capabilities = VLLMOmniRolloutCapabilities(
        compatibility_patch_sha256="0" * 64,
        skips_vae_decode=False,
        exact_operation_trace_replay=False,
        hot_generator_lora_sync=False,
        dedicated_rollout_pool=False,
    )
    with pytest.raises(ValueError, match="invalid vLLM-Omni rollout capability"):
        capabilities.validate(
            requested_batch_size=2,
            expected_revision=VLLM_OMNI_HELIOS_BASE_COMMIT,
            require_hot_lora_sync=True,
        )


def test_phase_matrix_limits_vllm_to_no_grad_rollouts() -> None:
    no_gan = rollout_phase_matrix("no-gan")
    allowed = {row["phase"] for row in no_gan if row["allowed"]}
    forbidden = {row["phase"] for row in no_gan if not row["allowed"]}
    assert "post_generator_critic_input_rollouts" in allowed
    assert "isolated_critic_only_frozen_generator_rollouts" in allowed
    assert "differentiable_generator_update_rollout" in forbidden
    assert "score_fake_real_or_gan_g" in forbidden
    assert expected_generator_version("no-gan") == 3
    assert expected_generator_version("gan") == 201


def test_source_probe_distinguishes_stock_019_from_no_vae_backport() -> None:
    stock_019 = """
    current_latents = real_history_latents[index_slice]
    current_video = self.vae.decode(current_latents)[0]
    if output_type == 'latent':
        output = real_history_latents
    """
    patched = """
    def helper(output_type):
        if output_type == 'latent':
            return video
    if output_type != 'latent':
        current_latents = real_history_latents[index_slice]
        current_video = self.vae.decode(current_latents)[0]
    if output_type == 'latent':
        output = real_history_latents
    """
    assert not source_has_latent_no_vae_guard(stock_019)
    assert source_has_latent_no_vae_guard(patched)


def test_checked_in_patch_and_preparation_script_are_pinned() -> None:
    root = Path(__file__).resolve().parents[1]
    patch = root / VLLM_OMNI_HELIOS_LATENT_PATCH_PATH
    assert patch.is_file()
    assert hashlib.sha256(patch.read_bytes()).hexdigest() == VLLM_OMNI_HELIOS_LATENT_PATCH_SHA256
    preparation = (root / "scripts/distillschedule/prepare_vllm_omni_019_helios.sh").read_text()
    assert VLLM_OMNI_HELIOS_BASE_TAG in preparation
    assert VLLM_OMNI_HELIOS_BASE_COMMIT in preparation
    assert VLLM_OMNI_HELIOS_LATENT_PATCH_SHA256 in preparation


def test_factory_and_rollout_receipt_contracts() -> None:
    assert parse_factory_spec("package.client:create_adapter") == (
        "package.client",
        "create_adapter",
    )
    with pytest.raises(ValueError):
        parse_factory_spec("not-a-factory")
    receipt = validate_rollout_receipt(
        {
            "status": "PASS",
            "backend": VLLM_OMNI_BACKEND,
            "output_type": "latent",
            "vae_decode_calls": 0,
            "logical_steps": [10, 11],
            "generator_version": 3,
            "upstream_revision": VLLM_OMNI_HELIOS_BASE_COMMIT,
            "compatibility_patch_sha256": VLLM_OMNI_HELIOS_LATENT_PATCH_SHA256,
            "exact_operation_trace_replay": True,
            "call_count": 6,
        },
        logical_steps=(10, 11),
        generator_version=3,
        expected_revision=VLLM_OMNI_HELIOS_BASE_COMMIT,
    )
    assert receipt["status"] == "PASS"


def test_checked_in_plans_and_entry_points_use_019_external_backend() -> None:
    root = Path(__file__).resolve().parents[1]
    for name in (
        "helios_nogan_world16_fsdp2_p0_vllm_omni.json",
        "helios_gan_world16_fsdp2_p0_vllm_omni.json",
    ):
        value = json.loads(
            (root / "configs/distillschedule/baselines" / name).read_text()
        )
        topology = value["rollout_topology"]
        assert value["default_candidate"]["rollout_backend"] == VLLM_OMNI_BACKEND
        assert topology["dedicated_gpu_pool_required"] is True
        assert topology["vae_decode_calls"] == 0
        assert topology["vllm_omni_version_family"] == VLLM_OMNI_VERSION_FAMILY
        assert topology["vllm_version_family"] == VLLM_VERSION_FAMILY
        assert topology["reference_upstream_tag"] == VLLM_OMNI_HELIOS_BASE_TAG
        assert topology["reference_upstream_revision"] == VLLM_OMNI_HELIOS_BASE_COMMIT
        assert topology["compatibility_patch"]["sha256"] == VLLM_OMNI_HELIOS_LATENT_PATCH_SHA256
        assert topology["initial_canary"] == {"tp": 1, "replicas": 1}
    worker = (
        root
        / "scripts/distillschedule/probe_static_fsdp2_p0_vllm_omni_world16.py"
    ).read_text()
    launcher = (
        root
        / "scripts/distillschedule/launch_static_fsdp2_p0_vllm_omni_world16_node.sh"
    ).read_text()
    assert "precompute_vllm_omni_critic_rollouts" in worker
    assert "source_trainable_sha256" in worker
    assert "trace_schedule_match" in worker
    assert "--rollout-backend vllm-omni-latent" in launcher
    assert VLLM_OMNI_HELIOS_BASE_COMMIT in launcher
    assert VLLM_OMNI_HELIOS_LATENT_PATCH_SHA256 in launcher
