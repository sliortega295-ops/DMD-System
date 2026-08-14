from __future__ import annotations

import json
from pathlib import Path

from distillschedule.runtime.fsdp2_static import (
    GAN_CONTRACT,
    NO_GAN_CONTRACT,
    configure_fsdp2_checkpointing,
    contract_for_variant,
    fsdp2_metadata_tolerant_checkpoint,
    ema_update_reference,
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


def test_static_profile_switches_are_fail_closed_and_default_legacy_safe() -> None:
    source = (
        ROOT / "scripts/distillschedule/launch_static_fsdp2_full5_world16_node.sh"
    ).read_text()
    assert 'PROFILE_DIAGNOSTICS="${DISTILLSCHEDULE_PROFILE_DIAGNOSTICS:-on}"' in source
    assert 'EMA_DEVICE="${DISTILLSCHEDULE_EMA_DEVICE:-cpu}"' in source
    assert '[[ "${PROFILE_DIAGNOSTICS}" == on || "${PROFILE_DIAGNOSTICS}" == off ]]' in source
    assert '[[ "${EMA_DEVICE}" == cpu || "${EMA_DEVICE}" == cuda ]]' in source
    assert 'WORKER+=(--ema-device "${EMA_DEVICE}")' in source
    assert '--no-profile-diagnostics' in source


def test_static_profile_uses_one_outer_node_nsight_session() -> None:
    source = (
        ROOT / "scripts/distillschedule/launch_static_fsdp2_full5_world16_node.sh"
    ).read_text()
    assert "--capture-range=cudaProfilerApi" in source
    assert "--trace-fork-before-exec=true" in source
    assert "--no-python --nnodes=2" in source
    assert "DISTILLSCHEDULE_STATIC_FSDP2_NSYS_EXACT_RANGE=1" in source
    assert "nsys_rank_exec.py" not in source
    assert "static_fsdp2_full5_node${NODE_RANK}" in source


def test_static_controller_summarizer_propagates_remote_pythonpath() -> None:
    source = (ROOT / "scripts/distillschedule/run_static_fsdp2_full5_world16.py").read_text()
    assert "export PYTHONPATH=" in source
    assert "args.remote_root}/src:{args.remote_root}" in source


def test_static_controller_propagates_profile_and_ema_switches() -> None:
    source = (ROOT / "scripts/distillschedule/run_static_fsdp2_full5_world16.py").read_text()
    assert '"--profile-diagnostics"' in source
    assert '"--ema-device"' in source
    assert "DISTILLSCHEDULE_PROFILE_DIAGNOSTICS" in source
    assert "DISTILLSCHEDULE_EMA_DEVICE" in source


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


def test_fsdp2_engine_exposes_official_gan_boundary_hook() -> None:
    source = (
        ROOT / "src/distillschedule/runtime/fsdp2_static.py"
    ).read_text()
    assert "def set_gradient_accumulation_boundary" in source
    assert "still calls ``step()`` exactly once" in source


def test_ema_reference_recurrence_is_stable_for_multiple_updates() -> None:
    import torch

    cpu_shadow = torch.tensor([1.0, -2.0, 3.5], dtype=torch.float32)
    gpu_shadow = cpu_shadow.clone()
    sources = [
        torch.tensor([1.25, -1.5, 3.0], dtype=torch.float32),
        torch.tensor([0.5, -1.0, 2.5], dtype=torch.float32),
        torch.tensor([0.25, -0.25, 2.0], dtype=torch.float32),
        torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32),
    ]
    for source in sources:
        ema_update_reference(cpu_shadow, source, 0.8)
        ema_update_reference(gpu_shadow, source, 0.8)
    assert torch.equal(cpu_shadow, gpu_shadow)
    assert float((cpu_shadow - gpu_shadow).abs().max()) == 0.0


def test_ema_cpu_reference_matches_cuda_shadow_when_available() -> None:
    import pytest
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    cpu_shadow = torch.tensor([1.0, -2.0, 3.5], dtype=torch.float32)
    cuda_shadow = cpu_shadow.cuda()
    sources = [
        torch.tensor([1.25, -1.5, 3.0], dtype=torch.float32),
        torch.tensor([0.5, -1.0, 2.5], dtype=torch.float32),
        torch.tensor([0.25, -0.25, 2.0], dtype=torch.float32),
        torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32),
    ]
    for source in sources:
        ema_update_reference(cpu_shadow, source, 0.8)
        ema_update_reference(cuda_shadow, source.cuda(), 0.8)
    diff = (cpu_shadow - cuda_shadow.cpu()).abs()
    rel = diff / cpu_shadow.abs().clamp_min(1e-12)
    assert float(diff.max()) <= 1e-6
    assert float(rel.max()) <= 1e-6


def test_ema_device_switch_is_explicit_and_default_is_cpu() -> None:
    source = (
        ROOT / "src/distillschedule/runtime/fsdp2_static.py"
    ).read_text()
    assert 'device: str = "cpu"' in source
    assert 'device not in {"cpu", "cuda"}' in source
    assert '"device": self.device' in source


def test_ema_profile_has_separate_transfer_and_update_ranges() -> None:
    source = (ROOT / "src/distillschedule/runtime/fsdp2_static.py").read_text()
    assert '"EMA_CPU_D2H"' in source
    assert '"EMA_GPU_UPDATE"' in source
    assert '"EMA_CPU_UPDATE"' in source
