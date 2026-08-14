from __future__ import annotations

import json
from pathlib import Path

from distillschedule.runtime.fsdp2_static import (
    GAN_CONTRACT,
    NO_GAN_CONTRACT,
    configure_fsdp2_checkpointing,
    contract_for_variant,
    fsdp2_metadata_tolerant_checkpoint,
)


ROOT = Path(__file__).resolve().parents[1]


def test_static_fsdp2_contracts_keep_independent_full5_semantics() -> None:
    assert NO_GAN_CONTRACT.logical_steps == (10, 11, 12, 13, 14)
    assert NO_GAN_CONTRACT.global_batch == 16
    assert NO_GAN_CONTRACT.ema_update_steps == (10, 11, 12, 13, 14)
    assert dict(NO_GAN_CONTRACT.final_versions) == {
        "generator": 3,
        "critic": 15,
        "ema": 15,
    }
    assert GAN_CONTRACT.logical_steps == (1000, 1001, 1002, 1003, 1004)
    assert GAN_CONTRACT.global_batch == 16
    assert GAN_CONTRACT.ema_update_steps == (1000,)
    assert dict(GAN_CONTRACT.final_versions) == {
        "generator": 201,
        "critic": 1005,
        "ema": 801,
    }
    assert contract_for_variant("no-gan") is NO_GAN_CONTRACT
    assert contract_for_variant("gan") is GAN_CONTRACT


def test_static_baseline_configs_forbid_transition_and_parameter_streaming() -> None:
    for name in (
        "helios_nogan_world16_fsdp2_static.json",
        "helios_gan_world16_fsdp2_static.json",
    ):
        value = json.loads(
            (ROOT / "configs/distillschedule/baselines" / name).read_text()
        )
        assert value["world_size"] == 16
        assert value["global_batch"] == 16
        assert value["generator_updates"] == 1
        assert value["critic_updates"] == 5
        placement = value["placement"]
        assert placement["generator"] == "FSDP2_FULL_SHARD_WORLD16_STATIC"
        assert placement["score_critic"] == "FSDP2_FULL_SHARD_WORLD16_STATIC"
        assert placement["blockwise_wrap"] == "40_BLOCKS_BOTTOM_UP_THEN_ROOT"
        assert placement["reshard_after_forward"] is True
        assert placement["parameter_offload"] is False
        assert placement["frozen_parameter_streaming"] is False
        assert placement["model_swap"] is False
        assert placement["role_transition"] is False


def test_static_worker_uses_separate_unprofiled_and_profile_modes() -> None:
    source = (
        ROOT / "scripts/distillschedule/probe_static_fsdp2_full5_world16.py"
    ).read_text()
    assert "args.timing_mode and args.profile_mode" in source
    assert "transition_count=0" in source
    assert "model_swap_count=0" in source
    assert "parameter_streaming_count=0" in source
    assert "STATIC_FSDP2_FULL5/" in source
    assert "global_batch=16" not in source  # batch comes from the frozen contract, not an ad-hoc CLI override


def test_static_launcher_disables_shared_filesystem_bytecode_writes() -> None:
    source = (
        ROOT / "scripts/distillschedule/launch_static_fsdp2_full5_world16_node.sh"
    ).read_text()
    assert "export PYTHONDONTWRITEBYTECODE=1" in source


def test_static_nsys_reports_pooled_world16_compute_union() -> None:
    source = (
        ROOT / "scripts/distillschedule/analyze_static_fsdp2_nsys.py"
    ).read_text()
    assert "for rank in range(16)" in source
    assert 'categories[COMPUTE]["pooled_percent"]' in source
    assert "formal_full5_time_comes_from_unprofiled_runs" in source


def test_fsdp2_checkpoint_adapter_only_disables_metadata_check() -> None:
    source = (
        ROOT / "src/distillschedule/runtime/fsdp2_static.py"
    ).read_text()
    assert 'use_reentrant=False' in source
    assert 'determinism_check="none"' in source
    assert "preserve_rng_state=True" in source
    assert "torch.autocast" not in source
    assert "no_grad" not in source

    class Model:
        gradient_checkpointing = True

    model = Model()
    configure_fsdp2_checkpointing(model)
    assert model._gradient_checkpointing_func is fsdp2_metadata_tolerant_checkpoint
