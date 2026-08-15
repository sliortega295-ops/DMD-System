"""Real-14B Helios primitives shared by the Critic-only pool executors.

This module keeps policy out of the model code.  It constructs the two frozen
physical backbones with the exact Stage-3 LoRA inventories and exposes the two
legal operations used by the isolated Critic-only subgraph:

* a trace-replayed, physically no-grad Generator nested rollout; and
* one trace-replayed Student denoising loss.

Sequential and Pipeline must call these same functions.  The only legal
difference between those policies is the order in which waves are executed.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import torch

from .critic_pool_trace import CriticPoolTrace, CriticPoolTraceEntry
from .dmd_boundary import add_noise, critic_denoising_loss
from .operation_trace import OperationTrace, tensor_sha256
from .precast_checkpoint import (
    HELIOS_KEEP_IN_FP32_MODULES,
    PRECAST_MANIFEST_NAME,
    validate_precast_checkpoint,
)
from .rollout_trace import (
    ForceNoGradCalls,
    LightweightNestedCallAudit,
    NestedCallTracer,
)


PoolRole = Literal["generator", "critic", "teacher"]


TRANSFORMER_ADDITIONAL_KWARGS = {
    "has_multi_term_memory_patch": True,
    "zero_history_timestep": True,
    "restrict_self_attn": False,
    "guidance_cross_attn": True,
    "is_train_restrict_lora": False,
    "restrict_lora": False,
    "restrict_lora_rank": 128,
    "is_amplify_history": False,
    "history_scale_mode": "per_head",
}


def configure_offline_helios() -> None:
    """Disable all remote model/kernel fetches before importing Helios."""

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_ENABLE_PARALLEL_LOADING", "yes")
    os.environ.setdefault("HF_PARALLEL_LOADING_WORKERS", "4")
    import kernels

    def _offline_kernel(*_args: Any, **_kwargs: Any) -> None:
        raise ImportError("Critic-pool real-14B runs disable remote kernel fetch")

    kernels.get_kernel = _offline_kernel


def _all_linear_targets(model: torch.nn.Module) -> list[str]:
    return sorted(
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and "norm" not in name
    )


def build_helios_pool_model(
    *,
    role: PoolRole,
    model_root: str | Path,
    subfolder: str,
    precast_model_root: str | Path | None = None,
    lora_rank: int = 256,
    max_workers: int = 4,
    enable_gan: bool = False,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Build the official BF16 backbone plus the Stage-3 FP32 LoRA inventory."""

    if role not in ("generator", "critic", "teacher"):
        raise ValueError(f"unsupported pool role: {role}")
    if role != "critic" and enable_gan:
        raise ValueError("GAN heads belong to the physical Score/Critic model")
    configure_offline_helios()
    from helios.modules.helios_kernels import (
        replace_all_norms_with_flash_norms,
        replace_rmsnorm_with_fp32,
        replace_rope_with_flash_rope,
    )
    from helios.modules.transformer_helios import HeliosTransformer3DModel
    from peft import LoraConfig

    extra = dict(TRANSFORMER_ADDITIONAL_KWARGS)
    if role in ("critic", "teacher"):
        extra.update(
            {
                "is_use_gan": bool(enable_gan),
                "is_use_gan_hooks": bool(enable_gan),
                "is_use_gan_final": bool(enable_gan),
                "gan_cond_map_dim": 768,
                "gan_hooks": [5, 15, 25, 35],
            }
        )
    if precast_model_root is None:
        precast_manifest_sha256: str | None = None
        load_root = str(model_root)
        load_subfolder: str | None = subfolder
        load_kwargs: dict[str, Any] = {
            "low_cpu_mem_usage": True,
            "torch_dtype": torch.bfloat16,
            "device_map": "cpu",
            "max_workers": max_workers,
        }
    else:
        precast_root = Path(precast_model_root).resolve()
        validate_precast_checkpoint(
            precast_root,
            expected_target_dtype=torch.bfloat16,
            expected_keep_in_fp32_modules=HELIOS_KEEP_IN_FP32_MODULES,
            expected_source_dir=Path(model_root).resolve() / subfolder,
            verify_shard_hashes=False,
        )
        precast_manifest_sha256 = hashlib.sha256(
            (precast_root / PRECAST_MANIFEST_NAME).read_bytes()
        ).hexdigest()
        load_root = str(precast_root)
        load_subfolder = None
        load_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": "cpu",
            "max_workers": max_workers,
            "precast_direct_load": True,
            "precast_expected_source_path": str(
                (Path(model_root).resolve() / subfolder)
            ),
        }
    model = HeliosTransformer3DModel.from_pretrained(
        load_root,
        subfolder=load_subfolder,
        transformer_additional_kwargs=extra,
        **load_kwargs,
    )
    model = replace_rmsnorm_with_fp32(model)
    model = replace_all_norms_with_flash_norms(model)
    replace_rope_with_flash_rope()
    model.requires_grad_(False)

    targets: list[str] = []
    excludes: list[str] = []
    if role != "teacher":
        targets = _all_linear_targets(model)
        excludes = ["down", "up"]
        if role == "generator":
            for name in ("patch_short", "patch_mid", "patch_long"):
                if hasattr(model, name):
                    targets.append(name)
        else:
            patch_modules = {
                "patch_short",
                "patch_mid",
                "patch_long",
                "patch_embedding",
            }
            targets = [
                name
                for name in targets
                if name not in patch_modules
                and not name.startswith("gan_heads.")
                and not name.startswith("gan_final_head.")
            ]
            excludes += sorted(patch_modules) + ["gan_heads", "gan_final_head"]

        model.add_adapter(
            LoraConfig(
                r=lora_rank,
                lora_alpha=float(lora_rank),
                lora_dropout=0.0,
                init_lora_weights="gaussian",
                target_modules=sorted(set(targets)),
                exclude_modules=excludes,
            )
        )
        if role == "critic" and enable_gan:
            # Match upstream GAN ownership: the heads are ordinary trainable
            # Critic parameters, not LoRA targets.  This is deliberately done
            # after add_adapter(), which otherwise leaves only adapter tensors
            # trainable.
            for name, parameter in model.named_parameters():
                if "gan_heads" in name or "gan_final_head" in name:
                    parameter.requires_grad_(True)
        for parameter in model.parameters():
            if parameter.requires_grad and parameter.dtype != torch.float32:
                parameter.data = parameter.data.float()
    model._distillschedule_split_modulation = True
    model._distillschedule_has_adapter = role != "teacher"
    for block in model.blocks:
        block._distillschedule_split_modulation = True
    if role != "teacher":
        model.enable_gradient_checkpointing()
    # Official Stage-3 keeps the shared Score backbone in train mode during
    # no-grad Real/Fake DMD calls.  A physically split Teacher must preserve
    # that module-mode semantic even though all of its parameters are frozen.
    model.train()

    trainable = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    return model, {
        "role": role,
        "backbone_root": str(Path(model_root).resolve()),
        "subfolder": subfolder,
        "loading_mode": (
            "mixed_bf16_precast_direct"
            if precast_model_root is not None
            else "official_bf16_loader"
        ),
        "load_root": load_root,
        "precast_manifest_sha256": precast_manifest_sha256,
        "lora_rank": lora_rank,
        "target_count": len(set(targets)),
        "exclude_modules": excludes,
        "trainable_tensor_count": len(trainable),
        "trainable_numel": sum(value.numel() for value in trainable.values()),
        "trainable_dtype_inventory": sorted(
            {str(value.dtype) for value in trainable.values()}
        ),
        "block_count": len(model.blocks),
        "gan_enabled": bool(enable_gan),
        "gan_trainable_numel": sum(
            value.numel()
            for name, value in trainable.items()
            if "gan_heads" in name or "gan_final_head" in name
        ),
        "split_modulation": True,
        "gradient_checkpointing": role != "teacher",
        "mode": "train",
        "has_adapter": role != "teacher",
    }


@dataclass(frozen=True)
class LoadedPoolEntry:
    pool_entry: CriticPoolTraceEntry
    operation_trace: OperationTrace
    tensors: dict[str, torch.Tensor]


def load_pool_entry(
    pool_trace: CriticPoolTrace,
    *,
    wave: int,
    producer_pool_rank: int,
    device: torch.device | str,
) -> LoadedPoolEntry:
    """Load one immutable rank-qualified trace entry onto its role GPU."""

    entry = pool_trace.entry(wave, producer_pool_rank)
    operation_trace = OperationTrace.load(
        (pool_trace.root / entry.trace_path).resolve()
    )
    tensors = {
        role: operation_trace.tensor(key, clone=True).to(device)
        for role, key in entry.tensor_keys.items()
    }
    if tensors["text_embedding"].ndim == 2:
        tensors["text_embedding"] = tensors["text_embedding"].unsqueeze(0)
    return LoadedPoolEntry(entry, operation_trace, tensors)


def _memory_snapshot(device: torch.device) -> dict[str, float]:
    free, total = torch.cuda.mem_get_info(device)
    gib = float(2**30)
    return {
        "allocated_gib": torch.cuda.memory_allocated(device) / gib,
        "reserved_gib": torch.cuda.memory_reserved(device) / gib,
        "max_allocated_gib": torch.cuda.max_memory_allocated(device) / gib,
        "max_reserved_gib": torch.cuda.max_memory_reserved(device) / gib,
        "driver_free_gib": free / gib,
        "driver_total_gib": total / gib,
    }


@torch.no_grad()
def run_generator_trace_rollout(
    *,
    model: torch.nn.Module,
    loaded: LoadedPoolEntry,
    device: torch.device,
    detailed_profile: bool = True,
    record_output_sha256: bool = True,
    empty_cache_before_rollout: bool = True,
    collect_diagnostics: bool = True,
    diagnostics_scope: Any | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Run the exact one-section, three-stage trace with no physical autograd."""

    from helios.scheduler.scheduling_helios import HeliosScheduler
    from helios.utils.utils_helios_post import run_generator

    tensors = loaded.tensors
    operation = loaded.operation_trace.operation(loaded.pool_entry.rollout_operation)
    control = operation["control"]
    expected_call_count = int(control["call_count"])
    expected_stage_calls = {
        int(item["stage"]): int(item["call_count"])
        for item in control["pyramid_stage_call_sequence"]
    }
    gt_all_data = (
        tensors["entry_noise"],
        tensors["indices_hidden_states"],
        tensors["indices_history_short"],
        tensors["indices_history_mid"],
        tensors["indices_history_long"],
        tensors["history_short"],
        tensors["history_mid"],
        tensors["history_long"],
        tensors["continuous_history"],
    )
    scheduler = HeliosScheduler(
        shift=1.0,
        stages=3,
        stage_range=[0, 1 / 3, 2 / 3, 1],
        gamma=1 / 3,
    )
    sigmas = torch.linspace(
        1.0, 1.0 / 1000, 1000, device=device, dtype=torch.float64
    )
    timesteps = sigmas * 1000
    args = SimpleNamespace(
        training_config=SimpleNamespace(
            base_seq_len=256,
            max_seq_len=4096,
            base_shift=0.5,
            max_shift=1.15,
            saturation_ratio_min=0.3,
            saturation_ratio_max=1.7,
            saturation_ratio_clean_prob=0.1,
        )
    )
    if detailed_profile:
        tracer = NestedCallTracer(
            model,
            base_height=48,
            base_width=80,
            num_stages=3,
            synchronize=lambda: torch.cuda.synchronize(device),
            before_call=lambda: torch.cuda.reset_peak_memory_stats(device),
            memory_snapshot=lambda: _memory_snapshot(device),
            retain_output_gradients=False,
        )
    else:
        tracer = LightweightNestedCallAudit(
            model,
            base_height=48,
            base_width=80,
            num_stages=3,
        )
    physical_call = ForceNoGradCalls(tracer)
    model.enable_adapters()
    model.train()
    if empty_cache_before_rollout:
        torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    (
        generated_x0,
        gradient_mask,
        denoised_from,
        denoised_to,
        consistency_loss,
    ) = run_generator(
        args=args,
        accelerator=SimpleNamespace(device=device, is_main_process=False),
        transformer=physical_call,
        scheduler=scheduler,
        noise=tensors["entry_noise"],
        prompt_embeds=tensors["text_embedding"],
        dmd_is_low_vram_mode=False,
        is_keep_x0=True,
        history_sizes=[16, 2, 1],
        is_enable_stage2=True,
        stage2_num_stages=3,
        stage2_num_inference_steps_list=[2, 2, 2],
        denoising_step_list=[1000, 750, 500, 250],
        last_step_only=False,
        last_section_grad_only=False,
        return_sim_step=False,
        sigmas=sigmas,
        timesteps=timesteps,
        timestep_shift=5.0,
        use_dynamic_shifting=True,
        time_shift_type="linear",
        num_critic_input_frames=9,
        num_rollout_sections=1,
        is_skip_first_section=False,
        is_amplify_first_chunk=True,
        is_corrupt_history_latents=True,
        is_add_saturation=True,
        is_use_gt_history=True,
        gt_all_data=gt_all_data,
        is_dmd_vae_decode=False,
        is_multi_pyramid_stage_backward_simulated=False,
        init_pyramid_stage_flag=2,
        is_consistency_align=False,
        use_kv_cache=False,
        operation_trace=loaded.operation_trace,
        trace_prefix=f"step/{loaded.pool_entry.logical_step}/critic",
    )
    torch.cuda.synchronize(device)
    if list(generated_x0.shape) != [1, 16, 9, 48, 80]:
        raise AssertionError("Generator returned a reduced or malformed x0")
    if generated_x0.dtype != torch.bfloat16:
        raise AssertionError("Generator pool wire contract requires BF16 x0")
    if generated_x0.requires_grad or generated_x0.grad_fn is not None:
        raise AssertionError("Critic-only Generator output retained autograd")
    if len(tracer.calls) != expected_call_count:
        raise AssertionError(
            f"trace call count changed: {len(tracer.calls)} != {expected_call_count}"
        )
    stage_calls = {
        stage: sum(int(event["stage"]) == stage for event in tracer.calls)
        for stage in range(3)
    }
    if stage_calls != expected_stage_calls:
        raise AssertionError(
            f"trace stage calls changed: {stage_calls} != {expected_stage_calls}"
        )
    if any(
        event["grad_enabled"] or event.get("output_requires_grad", False)
        for event in tracer.calls
    ):
        raise AssertionError("a physical Generator call escaped no-grad")
    details = {
        "trace_key": loaded.pool_entry.trace_key,
        "trace_identity_sha256": loaded.pool_entry.identity_sha256,
        "call_count": len(tracer.calls),
        "stage_call_count": stage_calls,
        "calls": tracer.calls,
        "output_shape": list(generated_x0.shape),
        "output_dtype": str(generated_x0.dtype),
        "output_requires_grad": False,
        "gradient_mask_is_none": gradient_mask is None,
        "denoised_timestep_from": (
            float(torch.as_tensor(denoised_from).flatten()[0])
            if collect_diagnostics
            else "omitted_in_clean_profile"
        ),
        "denoised_timestep_to": (
            float(torch.as_tensor(denoised_to).flatten()[0])
            if collect_diagnostics
            else "omitted_in_clean_profile"
        ),
        "consistency_loss": (
            float(torch.as_tensor(consistency_loss))
            if collect_diagnostics
            else "omitted_in_clean_profile"
        ),
        "memory": _memory_snapshot(device),
    }
    if record_output_sha256 and collect_diagnostics:
        if diagnostics_scope is None:
            details["output_sha256"] = tensor_sha256(generated_x0)
        else:
            with diagnostics_scope("DIAG_OUTPUT_SHA_D2H/generator_x0"):
                details["output_sha256"] = tensor_sha256(generated_x0)
    return generated_x0.detach(), details


def prepare_student_batch(
    *,
    generated_x0: torch.Tensor,
    loaded: LoadedPoolEntry,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Construct the exact trace-keyed noisy Student input and target."""

    from helios.utils.utils_base import apply_schedule_shift

    tensors = loaded.tensors
    generated_x0 = generated_x0.to(device=device, dtype=torch.bfloat16).detach()
    sigmas = torch.linspace(
        1.0, 1.0 / 1000, 1000, device=device, dtype=torch.float64
    )
    sigmas = apply_schedule_shift(
        sigmas,
        generated_x0,
        base_seq_len=256,
        max_seq_len=4096,
        base_shift=0.5,
        max_shift=1.15,
        time_shift_type="linear",
    )
    timesteps = sigmas * 1000
    timestep = tensors["critic_timestep"].to(device=device)
    noise = tensors["critic_noise"].to(device=device, dtype=torch.bfloat16)
    noisy = add_noise(generated_x0, noise, timestep, sigmas, timesteps).detach()
    return {
        "hidden_states": noisy,
        "timestep": timestep,
        "encoder_hidden_states": tensors["text_embedding"],
        "indices_hidden_states": tensors["indices_hidden_states"],
        "indices_latents_history_short": tensors["indices_history_short"],
        "indices_latents_history_mid": tensors["indices_history_mid"],
        "indices_latents_history_long": tensors["indices_history_long"],
        "latents_history_short": tensors["history_short"],
        "latents_history_mid": tensors["history_mid"],
        "latents_history_long": tensors["history_long"],
        "critic_noise": noise,
        "generated_x0": generated_x0,
    }


def student_forward_loss(
    engine: Any,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the GAN-disabled official Fake Score/Critic denoising objective."""

    model = engine.module
    model.enable_adapters()
    model.train()
    prediction = engine(
        hidden_states=batch["hidden_states"],
        timestep=batch["timestep"],
        encoder_hidden_states=batch["encoder_hidden_states"],
        indices_hidden_states=batch["indices_hidden_states"],
        indices_latents_history_short=batch["indices_latents_history_short"],
        indices_latents_history_mid=batch["indices_latents_history_mid"],
        indices_latents_history_long=batch["indices_latents_history_long"],
        latents_history_short=batch["latents_history_short"],
        latents_history_mid=batch["latents_history_mid"],
        latents_history_long=batch["latents_history_long"],
        return_dict=False,
    )[0]
    loss = critic_denoising_loss(
        prediction, batch["critic_noise"], batch["generated_x0"]
    )
    if not loss.requires_grad or loss.grad_fn is None or not torch.isfinite(loss):
        raise AssertionError("Student Critic loss is not a finite autograd value")
    return prediction, loss


def fast_tensor_fingerprint(tensor: torch.Tensor) -> str:
    """Low-overhead deterministic identity for gate logs, not formal SHA evidence."""

    value = tensor.detach().float()
    summary = torch.stack(
        (
            value.sum(dtype=torch.float64),
            value.square().sum(dtype=torch.float64),
            value.abs().max().double(),
            torch.tensor(float(value.numel()), device=value.device, dtype=torch.float64),
        )
    )
    return hashlib.sha256(summary.cpu().numpy().tobytes()).hexdigest()
