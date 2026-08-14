"""Narrow adapters into the pinned official Helios GAN loss functions.

This module deliberately contains only argument/state binding shared by the
role-phase worker and its input validator.  It replaces importing the old PR5
pipeline executable, whose topology and optimizer policy are superseded.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from omegaconf import OmegaConf

from .operation_trace import OperationTrace


class HeliosAcceleratorAdapter:
    def __init__(self, device: torch.device, engine: Any | None = None) -> None:
        self.device = device
        self.is_main_process = False
        self.deepspeed_engine_wrapped = (
            SimpleNamespace(engine=engine) if engine is not None else None
        )

    def backward(self, loss: torch.Tensor) -> None:
        if self.deepspeed_engine_wrapped is None:
            loss.backward()
        else:
            self.deepspeed_engine_wrapped.engine.backward(loss)

    @staticmethod
    def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
        return model


def helios_scheduler(device: torch.device) -> Any:
    from helios.scheduler.scheduling_helios import HeliosScheduler

    return HeliosScheduler(
        shift=1.0,
        stages=3,
        stage_range=[0, 1 / 3, 2 / 3, 1],
        gamma=1 / 3,
    )


def load_official_config(path: Path) -> Any:
    from helios.utils.train_config import Args

    return OmegaConf.merge(OmegaConf.structured(Args), OmegaConf.load(path))


@torch.no_grad()
def load_logical_trainables(
    model: torch.nn.Module,
    package: Path,
    component: str,
) -> dict[str, Any]:
    manifest = json.loads((package / "manifest.json").read_text())
    spec = manifest["components"][component]
    payload = spec["payloads"]["module"]
    if payload.get("dtype") != "torch.bfloat16":
        raise AssertionError(f"{component} module payload is not BF16: {payload}")
    raw = torch.from_file(
        str(package / payload["path"]),
        shared=False,
        size=int(payload["bytes"]) // 2,
        dtype=torch.bfloat16,
    )
    trainables = {name: parameter for name, parameter in model.named_parameters()
                  if parameter.requires_grad}
    names = list(spec["parameter_names"])
    if set(trainables) != set(names) or list(trainables) != names:
        missing = sorted(set(names) - set(trainables))
        extra = sorted(set(trainables) - set(names))
        raise AssertionError(
            f"{component} logical inventory/order drift "
            f"missing={missing[:3]} extra={extra[:3]}"
        )
    for name in names:
        record = spec["records"][name]
        value = raw.narrow(
            0, int(record["offset_elements"]), int(record["numel"])
        ).view(tuple(record["shape"]))
        parameter = trainables[name]
        if record["module"]["dtype"] != "torch.bfloat16":
            raise AssertionError(f"{component}/{name} module record is not BF16")
        if parameter.dtype != torch.bfloat16:
            parameter.data = parameter.data.bfloat16()
        parameter.copy_(value)
    del raw
    return spec


def common_official_call(config: Any) -> dict[str, Any]:
    training, validation = config.training_config, config.validation_config
    return {
        "is_keep_x0": True,
        "history_sizes": list(training.history_sizes),
        "is_enable_stage2": bool(training.is_enable_stage2),
        "stage2_num_stages": int(training.stage2_num_stages),
        "stage2_num_inference_steps_list": list(
            validation.stage2_simulated_inference_steps
        ),
        "denoising_step_list": list(training.dmd_denoising_step_list),
        "last_step_only": bool(training.dmd_last_step_only),
        "last_section_grad_only": bool(training.dmd_last_section_grad_only),
        "timestep_shift": float(training.dmd_timestep_shift),
        "use_dynamic_shifting": bool(training.use_dynamic_shifting),
        "time_shift_type": str(training.time_shift_type),
        "num_critic_input_frames": int(training.num_critic_input_frames),
        "num_rollout_sections": 1,
        "is_skip_first_section": bool(
            getattr(training, "is_skip_first_section", False)
        ),
        "is_amplify_first_chunk": bool(training.is_amplify_first_chunk),
        "is_corrupt_history_latents": bool(training.corrupt_history),
        "is_add_saturation": bool(training.is_add_saturation),
        "is_use_gt_history": True,
        "is_dmd_vae_decode": False,
        "is_multi_pyramid_stage_backward_simulated": bool(
            training.is_multi_pyramid_stage_backward_simulated
        ),
        "use_kv_cache": bool(getattr(validation, "use_kv_cache", False)),
    }


def score_official_kwargs(
    config: Any,
    trace: OperationTrace,
    source: dict[str, torch.Tensor],
    step: int,
    negative: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    training = config.training_config
    from helios.utils.utils_base import apply_schedule_shift
    from helios.utils.utils_helios_post import replay_operation_trace_prepared_history

    sigmas = torch.linspace(1.0, 1.0 / 1000, 1000, device=device, dtype=torch.float64)
    sigmas = apply_schedule_shift(
        sigmas,
        torch.empty((1, 16, 9, 48, 80)),
        base_seq_len=training.base_seq_len,
        max_seq_len=training.max_seq_len,
        base_shift=training.base_shift,
        max_shift=training.max_shift,
        time_shift_type=training.time_shift_type,
    )
    history, short, mid, long, indices, indices_short, indices_mid, indices_long = (
        replay_operation_trace_prepared_history(
            trace,
            trace_prefix=f"step/{step}/generator",
            device=device,
            dtype=torch.bfloat16,
        )
    )
    gt_all_data = (
        source["primary_x0"],
        indices,
        indices_short,
        indices_mid,
        indices_long,
        short,
        mid,
        long,
        history,
    )
    return {
        "accelerator": HeliosAcceleratorAdapter(device),
        "scheduler": helios_scheduler(device),
        "prompt_embeds": source["text_embedding"],
        "negative_prompt_embeds": negative,
        "dmd_is_low_vram_mode": False,
        "is_gan_low_vram_mode": True,
        "fake_guidance_scale": float(training.fake_guidance_scale),
        "real_guidance_scale": float(training.real_guidance_scale),
        "is_enable_stage2": True,
        "is_use_gt_history": True,
        "gt_all_data": gt_all_data,
        "is_use_gan": True,
        "is_decouple_dmd": False,
        "ts_schedule": False,
        "ts_schedule_max": False,
        "sigmas": sigmas,
        "timesteps": sigmas * 1000,
        "gradient_mask": None,
        "denoised_timestep_from": None,
        "denoised_timestep_to": None,
        "is_forcing_low_renoise": bool(training.generator_is_forcing_low_renoise),
        "dynamic_alpha": float(training.generator_dynamic_alpha),
        "dynamic_beta": float(training.generator_dynamic_beta),
        "dynamic_sample_type": str(training.generator_dynamic_sample_type),
        "global_step": step,
        "dynamic_step": int(training.generator_dynamic_step),
        "operation_trace": trace,
        "trace_prefix": f"step/{step}/generator",
    }


def critic_official_kwargs(
    config: Any,
    trace: OperationTrace,
    source: dict[str, torch.Tensor],
    step: int,
    generated_x0: torch.Tensor,
    engine: Any,
) -> dict[str, Any]:
    training = config.training_config
    return {
        "args": config,
        "critic_accelerator": HeliosAcceleratorAdapter(generated_x0.device, engine),
        "transformer": None,
        "scheduler": helios_scheduler(generated_x0.device),
        "noise": trace.tensor(f"step/{step}/critic/entry_noise").to(
            generated_x0.device, dtype=torch.bfloat16
        ),
        "prompt_embeds": source["text_embedding"],
        "dmd_is_low_vram_mode": False,
        "is_gan_low_vram_mode": True,
        "gt_history_latents": source["secondary_history"],
        "gt_target_latents": source["secondary_target"],
        "gt_x0_latents": source["secondary_x0"],
        "is_use_gan": True,
        "is_separate_gan_grad": bool(training.is_separate_gan_grad),
        "gan_base_critic_trainable_params": None,
        "gan_extra_critic_trainable_params": None,
        "gan_vae_latents": source["primary_target"],
        "gan_prompt_embeds": source["text_embedding"],
        "gan_d_weight": float(training.gan_d_weight),
        "aprox_r1": bool(training.aprox_r1),
        "aprox_r2": bool(training.aprox_r2),
        "r1_weight": float(training.r1_weight),
        "r2_weight": float(training.r2_weight),
        "r1_sigma": float(training.r1_sigma),
        "r2_sigma": float(training.r2_sigma),
        "dynamic_alpha": float(training.critic_dynamic_alpha),
        "dynamic_beta": float(training.critic_dynamic_beta),
        "dynamic_sample_type": str(training.critic_dynamic_sample_type),
        "global_step": step,
        "dynamic_step": int(training.critic_dynamic_step),
        "operation_trace": trace,
        "trace_prefix": f"step/{step}/critic",
        "precomputed_generated_image_or_video": generated_x0,
        "pipeline_micro_wave": None,
        **common_official_call(config),
    }


__all__ = [
    "HeliosAcceleratorAdapter",
    "common_official_call",
    "critic_official_kwargs",
    "load_logical_trainables",
    "load_official_config",
    "score_official_kwargs",
]
