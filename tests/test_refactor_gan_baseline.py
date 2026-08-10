from __future__ import annotations

import json
from pathlib import Path

import pytest

from distillschedule.core.contracts import ContractViolation
from distillschedule.runtime.gan_baseline import (
    EXPECTED_FINAL,
    EXPECTED_INITIAL,
    EXPECTED_STEPS,
    expected_gan_schedule,
    validate_common_rank_synchronized_trace,
    validate_gan_plan,
    validate_rendered_gan_config,
)
from distillschedule.runtime.routea_evidence import object_digest

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/distillschedule/plans/helios_routea_gan_full5_world16_baseline.json"


def _set_plan_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    values = {
        "HELIOS_GAN_DATA_ROOT": tmp_path / "gan_data",
        "DISTILLSCHEDULE_GAN_TRACE_BUNDLE": tmp_path / "trace",
        "DISTILLSCHEDULE_GAN_CHECKPOINT": tmp_path / "checkpoint-1000",
        "DISTILLSCHEDULE_OUTPUT_DIR": tmp_path / "run",
        "DISTILLSCHEDULE_FROZEN_BLOCK_MMAP_CACHE": tmp_path / "frozen",
        "DISTILLSCHEDULE_IMMUTABLE_MODEL_MMAP_CACHE": tmp_path / "immutable",
        "HELIOS_RUNTIME_ROOT": tmp_path / "helios",
        "DISTILLSCHEDULE_PYTHON": tmp_path / "bin/python",
        "DISTILLSCHEDULE_ACCELERATE": tmp_path / "bin/accelerate",
        "DISTILLSCHEDULE_ACCELERATE_CONFIG": tmp_path / "accelerate.yaml",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, str(value))


def test_gan_plan_is_full_offload_world16_and_uses_1000_1004(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_plan_env(monkeypatch, tmp_path)
    result = validate_gan_plan(PLAN)
    assert result["expected_steps"] == list(EXPECTED_STEPS)
    assert result["initial_versions"] == EXPECTED_INITIAL
    assert result["final_versions"] == EXPECTED_FINAL


def test_official_gan_frequency_cycle_is_source_faithful() -> None:
    schedule = expected_gan_schedule()["steps"]
    assert schedule["1000"]["train_generator"] is True
    assert "gan_G_loss" in schedule["1000"]["generator_path"]
    for step in EXPECTED_STEPS:
        item = schedule[str(step)]
        assert item["gan_active"] is True
        assert item["critic_path"][-1] == "single_joint_critic_optimizer_step"
    for step in EXPECTED_STEPS[1:]:
        assert schedule[str(step)]["train_generator"] is False
        assert schedule[str(step)]["generator_path"] == []


def _minimal_rendered_config(checkpoint: Path) -> dict:
    return {
        "data_config": {"single_height": 384, "single_width": 640},
        "training_config": {
            "dmd_is_low_vram_mode": True,
            "is_gan_low_vram_mode": True,
            "is_train_dmd": True,
            "is_dmd_vae_decode": False,
            "is_use_gan": True,
            "gan_start_step": 1000,
            "is_separate_gan_grad": False,
            "is_use_gan_hooks": True,
            "is_use_gan_final": True,
            "gan_cond_map_dim": 768,
            "gan_hooks": [5, 15, 25, 35],
            "gan_g_weight": 5e-2,
            "gan_d_weight": 1e-2,
            "aprox_r1": True,
            "aprox_r2": True,
            "r1_weight": 100.0,
            "r2_weight": 0.0,
            "r1_sigma": 0.1,
            "r2_sigma": 0.1,
            "is_use_gt_history": True,
            "dfake_gen_update_ratio": 5,
            "train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "mixed_precision": "bf16",
            "use_ema": True,
            "is_use_reward_model": False,
            "max_train_steps": 1005,
            "checkpointing_steps": 1005,
            "resume_from_checkpoint": str(checkpoint),
            "dmd_generator_deepspeed_config": "/tmp/zero1.json",
            "dmd_critic_deepspeed_config": "/tmp/zero1.json",
        },
    }


def test_rendered_gan_config_rejects_rewriting_gan_start_step(tmp_path: Path) -> None:
    pytest.importorskip("omegaconf")
    checkpoint = tmp_path / "checkpoint-1000"
    checkpoint.mkdir()
    config = _minimal_rendered_config(checkpoint)
    config["training_config"]["gan_start_step"] = 0
    path = tmp_path / "config.yaml"
    from omegaconf import OmegaConf

    OmegaConf.save(config=config, f=path)
    with pytest.raises(ContractViolation, match="expected 1000"):
        validate_rendered_gan_config(path, checkpoint=checkpoint)


def test_rendered_gan_config_rejects_separate_gan_grad(tmp_path: Path) -> None:
    pytest.importorskip("omegaconf")
    checkpoint = tmp_path / "checkpoint-1000"
    checkpoint.mkdir()
    config = _minimal_rendered_config(checkpoint)
    config["training_config"]["is_separate_gan_grad"] = True
    path = tmp_path / "config.yaml"
    from omegaconf import OmegaConf

    OmegaConf.save(config=config, f=path)
    with pytest.raises(ContractViolation, match="expected False"):
        validate_rendered_gan_config(path, checkpoint=checkpoint)


def _write_trace_bundle(root: Path, *, drift_rank: int | None = None, local_rank: int | None = None) -> None:
    root.mkdir(parents=True)
    entries = []
    for rank in range(16):
        trace_root = root / f"rank{rank:02d}"
        trace_root.mkdir()
        operations = []
        for step in EXPECTED_STEPS:
            exits = [1, 2, 1]
            if rank == drift_rank and step == 1002:
                exits = [1, 3, 1]
            scope = (
                "rank_local_amplified_gt_zero_history_exception"
                if rank == local_rank and step == 1001
                else "global_rank0_broadcast_equivalent"
            )
            operations.append(
                {
                    "key": f"step{step}.generator_rollout",
                    "operation": "stage3_branch",
                    "control": {
                        "global_step": step,
                        "stage_schedule_scope": scope,
                        "initial_synchronized_pyramid_stage_exit": [1, 2, 1],
                        "pyramid_stage_call_sequence": [
                            {"denoising_exit": value} for value in exits
                        ],
                    },
                }
            )
        trace = {
            "schema_version": "distillschedule.operation_trace.v1",
            "classification": "GAN_BASELINE_TEST",
            "source": {"source_authority": f"rank-{rank}", "sample_id": rank},
            "contract": {},
            "payload": {"path": "tensors.pt", "format": "torch.save"},
            "tensors": {},
            "operations": operations,
        }
        trace["trace_hash"] = object_digest(trace)
        trace_path = trace_root / "trace.json"
        trace_path.write_text(json.dumps(trace, indent=2, sort_keys=True) + "\n")
        entries.append(
            {
                "rank": rank,
                "path": trace_root.name,
                "trace_hash": trace["trace_hash"],
            }
        )
    bundle = {
        "schema_version": "distillschedule.operation_trace_bundle.v1",
        "ranks": entries,
    }
    bundle["bundle_hash"] = object_digest(bundle)
    (root / "bundle.json").write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n")


def test_gan_trace_requires_identical_dit_schedule_across_ranks(tmp_path: Path) -> None:
    root = tmp_path / "trace"
    _write_trace_bundle(root)
    result = validate_common_rank_synchronized_trace(root)
    assert result["common_rank_schedule_status"] == "PASS"
    assert result["rank_local_exception_supported"] is False


def test_gan_trace_rejects_cross_rank_dit_schedule_drift(tmp_path: Path) -> None:
    root = tmp_path / "trace"
    _write_trace_bundle(root, drift_rank=7)
    with pytest.raises(ContractViolation, match="differs from rank 0"):
        validate_common_rank_synchronized_trace(root)


def test_gan_trace_rejects_official_rank_local_exception_as_out_of_scope(tmp_path: Path) -> None:
    root = tmp_path / "trace"
    _write_trace_bundle(root, local_rank=3)
    with pytest.raises(ContractViolation, match="out-of-scope rank-local"):
        validate_common_rank_synchronized_trace(root)
