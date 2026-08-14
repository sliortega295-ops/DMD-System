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


WORLD16_PLAN = ROOT / "configs/distillschedule/plans/helios_routea_full5_world16_typed.json"
RA16_F5_PR4_PLAN = ROOT / "configs/distillschedule/plans/helios_ra16_f5_pr4.json"
RA16_F5_PR4_EC_PLAN = ROOT / "configs/distillschedule/plans/helios_ra16_f5_pr4_ec.json"
RA16_F5_PR8_PLAN = ROOT / "configs/distillschedule/plans/helios_ra16_f5_pr8.json"
RA16_F5_PR12_PLAN = ROOT / "configs/distillschedule/plans/helios_ra16_f5_pr12.json"
RA16_F5_GLP_OFFLOAD_PLAN = ROOT / "configs/distillschedule/plans/helios_ra16_f5_glp_offload.json"
RA16_F5_GLP_SCORE4_PLAN = ROOT / "configs/distillschedule/plans/helios_ra16_f5_glp_score4.json"


def _world16_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, node_rank: int) -> None:
    monkeypatch.setenv("DISTILLSCHEDULE_RUN_ID", "typed-world16-test")
    monkeypatch.setenv("DISTILLSCHEDULE_OUTPUT_DIR", str(tmp_path / "run"))
    monkeypatch.setenv("DISTILLSCHEDULE_FROZEN_BLOCK_MMAP_CACHE", str(tmp_path / "frozen"))
    monkeypatch.setenv("DISTILLSCHEDULE_IMMUTABLE_MODEL_MMAP_CACHE", str(tmp_path / "immutable"))
    monkeypatch.setenv("DISTILLSCHEDULE_NODE_RANK", str(node_rank))
    monkeypatch.setenv("DISTILLSCHEDULE_MASTER_ADDR", "10.0.0.1")
    monkeypatch.setenv("DISTILLSCHEDULE_MASTER_PORT", "29677")
    monkeypatch.setenv("HELIOS_RUNTIME_ROOT", str(ROOT))
    monkeypatch.setenv("DISTILLSCHEDULE_PYTHON", "/runtime/python")
    monkeypatch.setenv("DISTILLSCHEDULE_ACCELERATE", "/runtime/accelerate")
    monkeypatch.setenv("DISTILLSCHEDULE_ACCELERATE_CONFIG", str(ROOT / "scripts/accelerate_configs/multi_node_zero1_cpu_optimizer_offload.yaml"))


def test_world16_routea_plan_is_exact_typed_baseline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from omegaconf import OmegaConf

    _world16_env(monkeypatch, tmp_path, 0)
    loaded = load_plan(WORLD16_PLAN)
    compiled = DistillExecutor().compile(loaded, repo_root=ROOT)
    invocation = compiled.adapter_plan.invocation
    assert invocation is not None
    env = invocation.environment
    assert loaded.plan.hardware.world_size == 16
    assert loaded.plan.hardware.local_world_size == 8
    assert (loaded.plan.workload.shape.resolution.height, loaded.plan.workload.shape.resolution.width) == (384, 640)
    assert (loaded.plan.workload.shape.latent.height, loaded.plan.workload.shape.latent.width) == (48, 80)
    assert [(item.height, item.width) for item in loaded.plan.workload.shape.pyramid] == [(12, 20), (24, 40), (48, 80)]
    assert loaded.plan.workload.schedule.ema_start_step == 750
    assert loaded.plan.workload.schedule.ema_bootstrap_before_start is True
    assert compiled.adapter_plan.protocol.final_versions == {"generator": 3, "critic": 15, "ema": 15}
    assert env["DISTILLSCHEDULE_FROZEN_BLOCK_HOST_POLICY"] == "HOST_SHARED_IMMUTABLE"
    assert env["DISTILLSCHEDULE_FROZEN_BLOCK_STREAMING_COMPONENTS"] == "generator,score"
    assert env["DISTILLSCHEDULE_FROZEN_BLOCK_STREAMING_BLOCK_COUNT"] == "40"
    assert env["DISTILLSCHEDULE_FROZEN_BLOCK_STREAMING_WINDOW"] == "1"
    assert env["DISTILLSCHEDULE_FROZEN_BLOCK_STREAMING_INDICES"] == ",".join(map(str, range(40)))
    assert env["DISTILLSCHEDULE_ACTIVATION_OFFLOAD_COMPONENTS"] == "generator_selected"
    assert env["DISTILLSCHEDULE_SWAP_WAVE_SIZE"] == "8"
    assert env["DISTILLSCHEDULE_CONSTRUCTION_WAVE_SIZE"] == "0"
    assert env["DISTILLSCHEDULE_HYBRID_RESIDENCY"] == "1"
    assert env["DISTILLSCHEDULE_B0_COMPUTE_PATH"] == "1"
    assert "critic_train" not in env["DISTILLSCHEDULE_ACTIVATION_OFFLOAD_COMPONENTS"]
    command = list(invocation.command)
    assert command[command.index("--num_machines") + 1] == "2"
    assert command[command.index("--num_processes") + 1] == "16"
    assert command[command.index("--machine_rank") + 1] == "0"
    assert "--use_deepspeed" not in command
    accel_config = Path(command[command.index("--config_file") + 1])
    accel_text = accel_config.read_text()
    assert "deepspeed_multinode_launcher: standard" in accel_text
    assert "zero1_cpu_optimizer_offload.json" in accel_text
    assert command[-1].endswith("resolved_helios_config.node0.yaml")
    rendered = OmegaConf.create(next(iter(invocation.generated_files.values())))
    assert rendered.training_config.resume_from_checkpoint.endswith("checkpoint-10")
    assert rendered.training_config.max_train_steps == 15
    assert rendered.training_config.checkpointing_steps == 15
    assert rendered.training_config.ema_start_step == 750
    assert rendered.training_config.dfake_gen_update_ratio == 5
    assert rendered.training_config.train_batch_size == 1
    assert rendered.data_config.gan_data_root == [
        "/share/project/shuchenweng/DistillSchedule_assets/m0/datasets/syn_feature_rank16"
    ]
    assert rendered.training_config.dmd_generator_deepspeed_config.endswith(
        "zero1_cpu_optimizer_offload.json"
    )


def test_ra16_f5_pr4_has_distinct_identity_and_phase_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _world16_env(monkeypatch, tmp_path, 0)
    monkeypatch.setenv("DISTILLSCHEDULE_RUN_ID", "ra16_f5_pr4_test")
    loaded = load_plan(RA16_F5_PR4_PLAN)
    compiled = DistillExecutor().compile(loaded, repo_root=ROOT)
    invocation = compiled.adapter_plan.invocation
    assert invocation is not None
    env = invocation.environment

    assert loaded.plan.plan_id == "helios-ra16-f5-pr4-v1"
    assert loaded.plan.hardware.reserve_gib == 0
    assert loaded.plan.evidence.metadata["experiment_id"] == "RA16-F5-PR4"
    assert loaded.plan.evidence.metadata["comparison_baseline_experiment_id"] == "RA16-F5-FS40"
    assert compiled.adapter_plan.protocol.final_versions == {
        "generator": 3,
        "critic": 15,
        "ema": 15,
    }
    assert env["DISTILLSCHEDULE_FROZEN_BLOCK_STREAMING_COMPONENTS"] == "generator"
    assert env["DISTILLSCHEDULE_FROZEN_BLOCK_STREAMING_BLOCK_COUNT"] == "4"
    assert env["DISTILLSCHEDULE_FROZEN_BLOCK_STREAMING_INDICES"] == "0,24,1,5"
    assert env["DISTILLSCHEDULE_PHASE_RESIDENT_CRITIC_GENERATOR"] == "1"
    assert env["DISTILLSCHEDULE_ACTIVATION_OFFLOAD_COMPONENTS"] == (
        "generator_selected,critic_train"
    )
    assert "score" not in env["DISTILLSCHEDULE_FROZEN_BLOCK_STREAMING_COMPONENTS"].split(",")


def test_ra16_f5_pr4_ec_changes_only_bounded_allocator_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _world16_env(monkeypatch, tmp_path, 0)
    monkeypatch.setenv("DISTILLSCHEDULE_RUN_ID", "ra16_f5_pr4_ec_test")
    parent = load_plan(RA16_F5_PR4_PLAN)
    recovery = load_plan(RA16_F5_PR4_EC_PLAN)
    compiled = DistillExecutor().compile(recovery, repo_root=ROOT)
    invocation = compiled.adapter_plan.invocation
    assert invocation is not None

    assert recovery.plan.evidence.metadata["experiment_id"] == "RA16-F5-PR4-EC"
    assert recovery.plan.evidence.metadata["parent_experiment_id"] == "RA16-F5-PR4"
    assert recovery.plan.lifecycle == parent.plan.lifecycle
    assert recovery.plan.workload.schedule == parent.plan.workload.schedule
    assert recovery.plan.hardware == parent.plan.hardware
    assert invocation.environment["DISTILLSCHEDULE_EMPTY_CACHE_BEFORE_BACKWARD"] == "1"


def test_ra16_f5_pr8_changes_only_generator_streamed_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _world16_env(monkeypatch, tmp_path, 0)
    monkeypatch.setenv("DISTILLSCHEDULE_RUN_ID", "ra16_f5_pr8_test")
    parent = load_plan(RA16_F5_PR4_PLAN)
    candidate = load_plan(RA16_F5_PR8_PLAN)
    compiled = DistillExecutor().compile(candidate, repo_root=ROOT)
    invocation = compiled.adapter_plan.invocation
    assert invocation is not None
    env = invocation.environment

    assert candidate.plan.plan_id == "helios-ra16-f5-pr8-v1"
    assert candidate.plan.evidence.metadata["experiment_id"] == "RA16-F5-PR8"
    assert candidate.plan.evidence.metadata["parent_experiment_id"] == "RA16-F5-PR4"
    assert candidate.plan.workload.schedule == parent.plan.workload.schedule
    assert candidate.plan.hardware == parent.plan.hardware
    assert candidate.plan.training_backends == parent.plan.training_backends
    assert compiled.adapter_plan.protocol.final_versions == {
        "generator": 3,
        "critic": 15,
        "ema": 15,
    }
    assert env["DISTILLSCHEDULE_FROZEN_BLOCK_STREAMING_COMPONENTS"] == "generator"
    assert env["DISTILLSCHEDULE_FROZEN_BLOCK_STREAMING_BLOCK_COUNT"] == "8"
    assert env["DISTILLSCHEDULE_FROZEN_BLOCK_STREAMING_INDICES"] == (
        "0,24,1,5,8,25,10,32"
    )
    assert env["DISTILLSCHEDULE_FROZEN_BLOCK_STREAMING_WINDOW"] == "1"
    assert env["DISTILLSCHEDULE_PHASE_RESIDENT_CRITIC_GENERATOR"] == "1"
    assert env["DISTILLSCHEDULE_ACTIVATION_OFFLOAD_COMPONENTS"] == (
        "generator_selected,critic_train"
    )
    assert "score" not in env["DISTILLSCHEDULE_FROZEN_BLOCK_STREAMING_COMPONENTS"].split(",")
    assert "DISTILLSCHEDULE_EMPTY_CACHE_BEFORE_BACKWARD" not in env


def test_ra16_f5_pr12_changes_only_generator_streamed_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _world16_env(monkeypatch, tmp_path, 0)
    monkeypatch.setenv("DISTILLSCHEDULE_RUN_ID", "ra16_f5_pr12_test")
    parent = load_plan(RA16_F5_PR8_PLAN)
    candidate = load_plan(RA16_F5_PR12_PLAN)
    compiled = DistillExecutor().compile(candidate, repo_root=ROOT)
    invocation = compiled.adapter_plan.invocation
    assert invocation is not None
    env = invocation.environment

    assert candidate.plan.plan_id == "helios-ra16-f5-pr12-v1"
    assert candidate.plan.evidence.metadata["experiment_id"] == "RA16-F5-PR12"
    assert candidate.plan.evidence.metadata["parent_experiment_id"] == "RA16-F5-PR8"
    assert candidate.plan.workload.schedule == parent.plan.workload.schedule
    assert candidate.plan.hardware == parent.plan.hardware
    assert candidate.plan.training_backends == parent.plan.training_backends
    assert compiled.adapter_plan.protocol.final_versions == {
        "generator": 3,
        "critic": 15,
        "ema": 15,
    }
    assert env["DISTILLSCHEDULE_FROZEN_BLOCK_STREAMING_COMPONENTS"] == "generator"
    assert env["DISTILLSCHEDULE_FROZEN_BLOCK_STREAMING_BLOCK_COUNT"] == "12"
    assert env["DISTILLSCHEDULE_FROZEN_BLOCK_STREAMING_INDICES"] == (
        "0,24,1,5,8,25,10,32,36,15,16,26"
    )
    assert env["DISTILLSCHEDULE_FROZEN_BLOCK_STREAMING_WINDOW"] == "1"
    assert env["DISTILLSCHEDULE_PHASE_RESIDENT_CRITIC_GENERATOR"] == "1"
    assert env["DISTILLSCHEDULE_ACTIVATION_OFFLOAD_COMPONENTS"] == (
        "generator_selected,critic_train"
    )
    assert "score" not in env["DISTILLSCHEDULE_FROZEN_BLOCK_STREAMING_COMPONENTS"].split(",")
    assert "DISTILLSCHEDULE_EMPTY_CACHE_BEFORE_BACKWARD" not in env


def test_ra16_f5_glp_offload_changes_only_typed_engine_state_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _world16_env(monkeypatch, tmp_path, 0)
    monkeypatch.setenv("DISTILLSCHEDULE_RUN_ID", "ra16_f5_glp_offload_test")
    parent = load_plan(RA16_F5_PR12_PLAN)
    candidate = load_plan(RA16_F5_GLP_OFFLOAD_PLAN)
    compiled = DistillExecutor().compile(candidate, repo_root=ROOT)
    invocation = compiled.adapter_plan.invocation
    assert invocation is not None

    assert candidate.plan.plan_id == "helios-ra16-f5-glp-offload-v1"
    assert candidate.plan.evidence.metadata["experiment_id"] == "RA16-F5-GLP-OFFLOAD"
    assert candidate.plan.evidence.metadata["parent_experiment_id"] == "RA16-F5-PR12"
    assert candidate.plan.workload.schedule == parent.plan.workload.schedule
    assert candidate.plan.hardware == parent.plan.hardware
    assert candidate.plan.training_backends == parent.plan.training_backends
    assert candidate.plan.lifecycle.frozen_block_streaming == parent.plan.lifecycle.frozen_block_streaming
    assert candidate.plan.lifecycle.activation_offload == parent.plan.lifecycle.activation_offload
    assert candidate.plan.lifecycle.critic_phase_offload_generator_lp_state
    assert not parent.plan.lifecycle.critic_phase_offload_generator_lp_state
    assert invocation.environment[
        "DISTILLSCHEDULE_CRITIC_PHASE_OFFLOAD_GENERATOR_LP_STATE"
    ] == "1"
    assert invocation.environment["DISTILLSCHEDULE_FROZEN_BLOCK_STREAMING_BLOCK_COUNT"] == "12"
    assert invocation.environment["DISTILLSCHEDULE_ACTIVATION_OFFLOAD_MIN_BYTES"] == str(32 * 1024 * 1024)


def test_glp_offload_rejects_non_zero1_generator_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from distillschedule.core.plan import ExecutionPlan

    _world16_env(monkeypatch, tmp_path, 0)
    loaded = load_plan(RA16_F5_GLP_OFFLOAD_PLAN)
    raw = loaded.plan.to_dict()
    raw["training_backends"]["generator"]["zero_stage"] = 2
    plan = ExecutionPlan.from_mapping(raw)
    with pytest.raises(ContractViolation, match="requires Generator ZeRO-1"):
        default_registry().get("helios").validate(plan)


def test_ra16_f5_glp_score4_changes_only_score_streamed_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _world16_env(monkeypatch, tmp_path, 0)
    monkeypatch.setenv("DISTILLSCHEDULE_RUN_ID", "ra16_f5_glp_score4_test")
    parent = load_plan(RA16_F5_GLP_OFFLOAD_PLAN)
    candidate = load_plan(RA16_F5_GLP_SCORE4_PLAN)
    compiled = DistillExecutor().compile(candidate, repo_root=ROOT)
    invocation = compiled.adapter_plan.invocation
    assert invocation is not None
    env = invocation.environment

    assert candidate.plan.plan_id == "helios-ra16-f5-glp-score4-v1"
    assert candidate.plan.evidence.metadata["experiment_id"] == (
        "RA16-F5-GLP-SCORE4"
    )
    assert candidate.plan.evidence.metadata["parent_experiment_id"] == (
        "RA16-F5-GLP-OFFLOAD"
    )
    assert candidate.plan.workload.schedule == parent.plan.workload.schedule
    assert candidate.plan.hardware == parent.plan.hardware
    assert candidate.plan.training_backends == parent.plan.training_backends
    assert candidate.plan.lifecycle.activation_offload == (
        parent.plan.lifecycle.activation_offload
    )
    assert candidate.plan.lifecycle.critic_phase_offload_generator_lp_state
    assert candidate.plan.lifecycle.frozen_block_streaming.block_indices == (
        parent.plan.lifecycle.frozen_block_streaming.block_indices
    )
    assert candidate.plan.lifecycle.frozen_block_streaming.score_block_indices == (
        36,
        37,
        38,
        39,
    )
    assert env["DISTILLSCHEDULE_FROZEN_BLOCK_STREAMING_COMPONENTS"] == (
        "generator,score"
    )
    assert env["DISTILLSCHEDULE_FROZEN_BLOCK_STREAMING_INDICES"] == (
        "0,24,1,5,8,25,10,32,36,15,16,26"
    )
    assert env["DISTILLSCHEDULE_FROZEN_BLOCK_STREAMING_SCORE_INDICES"] == (
        "36,37,38,39"
    )
    assert env["DISTILLSCHEDULE_FROZEN_BLOCK_STREAMING_WINDOW"] == "1"
    assert env["DISTILLSCHEDULE_CRITIC_PHASE_OFFLOAD_GENERATOR_LP_STATE"] == "1"


def test_score_specific_indices_require_score_streaming_component(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from distillschedule.core.plan import ExecutionPlan

    _world16_env(monkeypatch, tmp_path, 0)
    loaded = load_plan(RA16_F5_GLP_SCORE4_PLAN)
    raw = loaded.plan.to_dict()
    raw["lifecycle"]["frozen_block_streaming"]["components"] = ["generator"]
    plan = ExecutionPlan.from_mapping(raw)
    with pytest.raises(ContractViolation, match="require enabled Score streaming"):
        default_registry().get("helios").validate(plan)


def test_phase_resident_policy_requires_generator_streaming(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from distillschedule.core.plan import ExecutionPlan

    _world16_env(monkeypatch, tmp_path, 0)
    loaded = load_plan(RA16_F5_PR4_PLAN)
    raw = loaded.plan.to_dict()
    raw["lifecycle"]["frozen_block_streaming"]["components"] = ["score"]
    plan = ExecutionPlan.from_mapping(raw)
    with pytest.raises(
        ContractViolation,
        match="requires Generator block streaming",
    ):
        default_registry().get("helios").validate(plan)


def test_world16_routea_node_qualified_invocations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    commands = []
    generated = []
    for rank in (0, 1):
        _world16_env(monkeypatch, tmp_path, rank)
        loaded = load_plan(WORLD16_PLAN)
        compiled = DistillExecutor().compile(loaded, repo_root=ROOT)
        invocation = compiled.adapter_plan.invocation
        assert invocation is not None
        commands.append(invocation.command)
        generated.append(next(iter(invocation.generated_files)))
    assert commands[0][commands[0].index("--machine_rank") + 1] == "0"
    assert commands[1][commands[1].index("--machine_rank") + 1] == "1"
    assert generated[0].name == "resolved_helios_config.node0.yaml"
    assert generated[1].name == "resolved_helios_config.node1.yaml"


def test_world16_routea_requires_valid_node_rank(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _world16_env(monkeypatch, tmp_path, 0)
    monkeypatch.delenv("DISTILLSCHEDULE_NODE_RANK")
    loaded = load_plan(WORLD16_PLAN)
    with pytest.raises(ContractViolation, match="DISTILLSCHEDULE_NODE_RANK"):
        DistillExecutor().compile(loaded, repo_root=ROOT)
    monkeypatch.setenv("DISTILLSCHEDULE_NODE_RANK", "2")
    loaded = load_plan(WORLD16_PLAN)
    with pytest.raises(ContractViolation, match="outside the plan node range"):
        DistillExecutor().compile(loaded, repo_root=ROOT)
