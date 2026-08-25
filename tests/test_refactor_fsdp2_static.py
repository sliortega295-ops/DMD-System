from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import distillschedule.runtime.fsdp2_static as fsdp2_static

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


def test_partial_checkpoint_bypass_is_sparse_default_off_and_censused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Block:
        def __init__(self, index: int) -> None:
            self.index = index

        def __call__(self, value: int) -> int:
            return value + self.index

    class Model:
        gradient_checkpointing = True

        def __init__(self) -> None:
            self.blocks = [Block(index) for index in range(40)]
            self.patch_embedding = Block(100)

        def named_modules(self):
            yield "", self
            yield "patch_embedding", self.patch_embedding
            for index, block in enumerate(self.blocks):
                yield f"blocks.{index}", block

    checkpointed: list[int] = []

    def fake_checkpoint(module: Block, value: int, **_: object) -> int:
        checkpointed.append(module.index)
        return module(value)

    monkeypatch.setattr(
        fsdp2_static,
        "fsdp2_metadata_tolerant_checkpoint",
        fake_checkpoint,
    )
    model = Model()
    receipt = configure_fsdp2_checkpointing(model, checkpoint_bypass_stride=4)
    outputs = [
        model._gradient_checkpointing_func(block, 100) for block in model.blocks
    ]
    patch_output = model._gradient_checkpointing_func(model.patch_embedding, 100)

    assert outputs == [100 + index for index in range(40)]
    assert patch_output == 200
    assert receipt["bypass_block_indices"] == list(range(0, 40, 4))
    assert receipt["direct_forward_count"] == 10
    assert receipt["checkpoint_forward_count"] == 30
    assert checkpointed == [index for index in range(40) if index % 4] + [100]
    assert set(receipt["block_call_counts"].values()) == {1}
    assert receipt["auxiliary_checkpoint_forward_count"] == 1
    assert receipt["auxiliary_checkpoint_module_counts"] == {"patch_embedding": 1}
    assert receipt["adds_synchronization"] is False
    assert receipt["changes_module_math"] is False


def test_partial_checkpoint_bypass_fails_closed_on_wrong_model_shape() -> None:
    class Model:
        gradient_checkpointing = True
        blocks = [object() for _ in range(39)]

    with pytest.raises(ValueError, match="40-block Helios"):
        configure_fsdp2_checkpointing(Model(), checkpoint_bypass_stride=4)


def test_partial_checkpoint_bypass_supports_four_block_canary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Block:
        def __init__(self, index: int) -> None:
            self.index = index

        def __call__(self, value: int) -> int:
            return value + self.index

    class Model:
        gradient_checkpointing = True

        def __init__(self) -> None:
            self.blocks = [Block(index) for index in range(40)]

        def named_modules(self):
            yield "", self
            for index, block in enumerate(self.blocks):
                yield f"blocks.{index}", block

    checkpointed: list[int] = []

    def fake_checkpoint(module: Block, value: int, **_: object) -> int:
        checkpointed.append(module.index)
        return module(value)

    monkeypatch.setattr(
        fsdp2_static,
        "fsdp2_metadata_tolerant_checkpoint",
        fake_checkpoint,
    )
    model = Model()
    receipt = configure_fsdp2_checkpointing(model, checkpoint_bypass_stride=10)
    outputs = [
        model._gradient_checkpointing_func(block, 100) for block in model.blocks
    ]

    assert outputs == [100 + index for index in range(40)]
    assert receipt["bypass_block_indices"] == [0, 10, 20, 30]
    assert receipt["direct_forward_count"] == 4
    assert receipt["checkpoint_forward_count"] == 36
    assert checkpointed == [index for index in range(40) if index % 10]
    assert set(receipt["block_call_counts"].values()) == {1}


def test_selective_flash_checkpoint_context_reuses_target_op(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        fsdp2_static,
        "SELECTIVE_FLASH_ATTN_OP",
        str(torch.ops.aten.mm.default),
    )

    class MatmulSigmoid:
        def __call__(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
            return torch.sigmoid(left @ right)

    receipt = {
        "selective_flash_context_count": 0,
        "selective_flash_original_save_count": 0,
        "selective_flash_recompute_reuse_count": 0,
    }
    torch.manual_seed(376)
    left = torch.randn(8, 8, requires_grad=True)
    right = torch.randn(8, 8, requires_grad=True)
    output = fsdp2_metadata_tolerant_checkpoint(
        MatmulSigmoid(),
        left,
        right,
        selective_flash_receipt=receipt,
    )
    output.sum().backward()

    assert receipt == {
        "selective_flash_context_count": 1,
        "selective_flash_original_save_count": 1,
        "selective_flash_recompute_reuse_count": 1,
    }
    assert left.grad is not None
    assert right.grad is not None


def test_selective_flash_checkpoint_is_sparse_critic_ready_and_censused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Block:
        def __init__(self, index: int) -> None:
            self.index = index

        def __call__(self, value: int) -> int:
            return value + self.index

    class Model:
        gradient_checkpointing = True

        def __init__(self) -> None:
            self.blocks = [Block(index) for index in range(40)]

        def named_modules(self):
            yield "", self
            for index, block in enumerate(self.blocks):
                yield f"blocks.{index}", block

    selective_calls: list[int] = []

    def fake_checkpoint(
        module: Block,
        value: int,
        *,
        selective_flash_receipt: dict[str, object] | None = None,
        **_: object,
    ) -> int:
        if selective_flash_receipt is not None:
            selective_calls.append(module.index)
        return module(value)

    monkeypatch.setattr(
        fsdp2_static,
        "fsdp2_metadata_tolerant_checkpoint",
        fake_checkpoint,
    )
    model = Model()
    receipt = configure_fsdp2_checkpointing(
        model,
        selective_flash_checkpoint_stride=2,
    )
    outputs = [
        model._gradient_checkpointing_func(block, 100) for block in model.blocks
    ]

    assert outputs == [100 + index for index in range(40)]
    assert receipt["selective_flash_enabled"] is True
    assert receipt["selective_flash_block_indices"] == list(range(0, 40, 2))
    assert receipt["direct_forward_count"] == 0
    assert receipt["checkpoint_forward_count"] == 40
    assert selective_calls == list(range(0, 40, 2))


def test_selective_flash_checkpoint_rejects_conflicting_contexts() -> None:
    class Model:
        gradient_checkpointing = True
        blocks = [object() for _ in range(40)]

        def named_modules(self):
            yield "", self
            for index, block in enumerate(self.blocks):
                yield f"blocks.{index}", block

    with pytest.raises(ValueError, match="mutually exclusive"):
        configure_fsdp2_checkpointing(
            Model(),
            checkpoint_bypass_stride=10,
            selective_flash_checkpoint_stride=2,
        )
    with pytest.raises(ValueError, match="operation-source"):
        configure_fsdp2_checkpointing(
            Model(),
            operation_source=True,
            selective_flash_checkpoint_stride=2,
        )


def test_fsdp2_engine_exposes_official_gan_boundary_hook() -> None:
    source = (
        ROOT / "src/distillschedule/runtime/fsdp2_static.py"
    ).read_text()
    assert "def set_gradient_accumulation_boundary" in source
    assert "still calls ``step()`` exactly once" in source


def test_ema_reference_recurrence_is_stable_for_multiple_updates() -> None:
    torch = pytest.importorskip("torch")

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
    torch = pytest.importorskip("torch")

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
