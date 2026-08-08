from __future__ import annotations

from pathlib import Path

import pytest

from distillschedule.adapters import default_registry
from distillschedule.core.config import load_plan
from distillschedule.core.contracts import ContractViolation
from distillschedule.runtime import DistillExecutor


ROOT = Path(__file__).resolve().parents[1]


def test_helios_adapter_maps_typed_lifecycle_to_legacy_environment() -> None:
    loaded = load_plan(ROOT / "configs/distillschedule/plans/helios_14b_384x640_full5.json")
    compiled = DistillExecutor().compile(loaded, repo_root=ROOT)
    invocation = compiled.adapter_plan.invocation
    assert invocation is not None
    env = invocation.environment
    assert env["DISTILLSCHEDULE_SPLIT_MODULATION"] == "1"
    assert env["DISTILLSCHEDULE_FULL_ACTIVATION_OFFLOAD"] == "1"
    assert env["DISTILLSCHEDULE_ACTIVATION_OFFLOAD_MIN_BYTES"] == str(32 * 1024 * 1024)
    assert env["DISTILLSCHEDULE_FROZEN_BLOCK_STREAMING_BLOCK_COUNT"] == "40"
    assert env["DISTILLSCHEDULE_FROZEN_BLOCK_STREAMING_WINDOW"] == "1"
    assert env["DISTILLSCHEDULE_LOGICAL_START_STEP"] == "0"
    assert invocation.command[0].endswith("accelerate")
    assert invocation.command[1] == "launch"
    assert "--use_deepspeed" in invocation.command
    assert invocation.command[-2] == "--config"
    assert invocation.command[-1].endswith("resolved_helios_config.yaml")
    assert len(invocation.generated_files) == 1


def test_generic_adapter_compiles_vllm_plan_without_importing_vllm() -> None:
    loaded = load_plan(ROOT / "configs/distillschedule/plans/generic_autoregressive_dmd_vllm_critic_only.json")
    compiled = DistillExecutor().compile(loaded, repo_root=ROOT)
    assert compiled.adapter_plan.invocation is None
    assert compiled.adapter_plan.protocol.steps[0].intent.kind.value == "critic_only"
    assert "plan-only" in compiled.adapter_plan.warnings[0]


def test_helios_rejects_direct_vllm_without_model_compatibility() -> None:
    loaded = load_plan(ROOT / "configs/distillschedule/plans/helios_14b_384x640_full5.json")
    plan_dict = loaded.plan.to_dict()
    plan_dict.pop("profile_key", None)
    plan_dict["rollout"] = {
        "engine": "vllm",
        "endpoint": "http://localhost:8000",
        "exact_version": True,
        "max_version_staleness": 0,
    }
    from distillschedule.core.plan import ExecutionPlan

    plan = ExecutionPlan.from_mapping(plan_dict)
    with pytest.raises(ContractViolation, match="not directly compatible"):
        default_registry().get("helios").validate(plan)


def test_helios_adapter_renders_runtime_config_from_plan() -> None:
    from omegaconf import OmegaConf

    loaded = load_plan(ROOT / "configs/distillschedule/plans/helios_14b_384x640_full5.json")
    compiled = DistillExecutor().compile(loaded, repo_root=ROOT)
    invocation = compiled.adapter_plan.invocation
    assert invocation is not None
    rendered = next(iter(invocation.generated_files.values()))
    config = OmegaConf.create(rendered)
    assert config.training_config.max_train_steps == 15
    assert config.training_config.resume_from_checkpoint.endswith("checkpoint-10")
    assert config.training_config.dfake_gen_update_ratio == 5
    assert config.training_config.train_batch_size == 1
    assert config.training_config.latent_window_size == [9]
    assert config.model_config.lora_rank == 256
    assert config.model_config.critic_lora_rank == 256
    assert config.model_config.critic_subfolder == "transformer"
    assert config.model_config.pretrained_model_name_or_path == config.model_config.real_score_model_name_or_path
    assert config.data_config.gan_data_root
    assert config.data_config.single_height == 384
    assert config.data_config.single_width == 640


def test_helios_adapter_rejects_backend_config_drift() -> None:
    from distillschedule.core.plan import ExecutionPlan

    loaded = load_plan(ROOT / "configs/distillschedule/plans/helios_14b_384x640_full5.json")
    raw = loaded.plan.to_dict()
    raw["training_backends"]["generator"]["zero_stage"] = 2
    plan = ExecutionPlan.from_mapping(raw)
    adapter = default_registry().get("helios")
    with pytest.raises(ContractViolation, match="config contains ZeRO-1"):
        adapter.compile(plan, source_dir=loaded.base_dir, repo_root=ROOT)


def test_external_helios_rollout_compiles_but_is_not_silently_run() -> None:
    from distillschedule.core.plan import ExecutionPlan

    loaded = load_plan(ROOT / "configs/distillschedule/plans/helios_14b_384x640_full5.json")
    raw = loaded.plan.to_dict()
    raw["workload"]["metadata"]["external_rollout_compatible"] = True
    raw["rollout"] = {
        "engine": "remote",
        "endpoint": "grpc://rollout.service:9000",
        "protocol": "grpc",
        "exact_version": True,
        "max_version_staleness": 0,
    }
    plan = ExecutionPlan.from_mapping(raw)
    compiled = default_registry().get("helios").compile(
        plan, source_dir=loaded.base_dir, repo_root=ROOT
    )
    assert compiled.invocation is None
    assert any("remote role runner" in warning for warning in compiled.warnings)
