"""Source-semantic primitives for Helios's detached DMD gradient boundary.

These functions intentionally contain no model/runtime imports.  The real-model probes
use them to bind a Score-produced input gradient to a retained Generator graph, while
unit tests can validate the math without importing the full Helios training module.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def shifted_sigma_schedule(
    *,
    num_train_timesteps: int = 1000,
    timestep_shift: float = 5.0,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reproduce ``_generator_loss``'s DMD sigma/timestep construction."""
    if num_train_timesteps < 1:
        raise ValueError("num_train_timesteps must be positive")
    if timestep_shift <= 0:
        raise ValueError("timestep_shift must be positive")
    sigmas = torch.linspace(
        1.0,
        1.0 / num_train_timesteps,
        num_train_timesteps,
        dtype=torch.float64,
        device=device,
    )
    if timestep_shift > 1:
        sigmas = timestep_shift * sigmas / (1 + (timestep_shift - 1) * sigmas)
    return sigmas, sigmas * num_train_timesteps


def sample_uniform_timestep(
    batch_size: int,
    *,
    num_train_timesteps: int = 1000,
    timestep_shift: float = 5.0,
    min_timestep: int = 0,
    max_timestep: int = 1000,
    min_step: int | None = None,
    max_step: int | None = None,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Default Stage-3 DMD uniform sampler from ``sample_dynamic_timestep``."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    min_step = int(0.02 * num_train_timesteps) if min_step is None else min_step
    max_step = int(0.98 * num_train_timesteps) if max_step is None else max_step
    t = torch.rand(batch_size, device=device, generator=generator) * (1.0 - 0.001) + 0.001
    timestep = min_timestep + t * (max_timestep - min_timestep)
    if timestep_shift > 1:
        normalized = timestep / num_train_timesteps
        timestep = (
            timestep_shift
            * normalized
            / (1 + (timestep_shift - 1) * normalized)
            * num_train_timesteps
        )
    return timestep.clamp(min_step, max_step).round().long()


def add_noise(
    original_samples: torch.Tensor,
    noise: torch.Tensor,
    timestep: torch.Tensor,
    sigmas: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    """Exact tensor formula used by Helios ``add_noise``."""
    sigmas = sigmas.to(noise.device)
    timesteps = timesteps.to(noise.device)
    timestep_id = torch.argmin(
        (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1
    )
    sigma = sigmas[timestep_id].reshape(-1, 1, 1, 1, 1)
    return ((1 - sigma) * original_samples + sigma * noise).type_as(noise)


def flow_prediction_to_x0(
    flow_prediction: torch.Tensor,
    noisy_sample: torch.Tensor,
    timestep: torch.Tensor,
    sigmas: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    """Exact higher-precision conversion used by ``convert_flow_pred_to_x0``."""
    original_dtype = flow_prediction.dtype
    device = flow_prediction.device
    flow_prediction, noisy_sample, sigmas, timesteps = (
        value.double().to(device)
        for value in (flow_prediction, noisy_sample, sigmas, timesteps)
    )
    timestep_id = torch.argmin(
        (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1
    )
    sigma = sigmas[timestep_id].reshape(-1, 1, 1, 1, 1)
    return (noisy_sample - sigma * flow_prediction).to(original_dtype)


def dmd_input_gradient(
    *,
    estimated_clean: torch.Tensor,
    noisy_sample: torch.Tensor,
    timestep: torch.Tensor,
    sigmas: torch.Tensor,
    timesteps: torch.Tensor,
    fake_flow_cond: torch.Tensor,
    real_flow_cond: torch.Tensor,
    real_flow_uncond: torch.Tensor | None,
    fake_guidance_scale: float = 0.0,
    real_guidance_scale: float = 3.0,
    fake_flow_uncond: torch.Tensor | None = None,
    normalization: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Construct the default non-decoupled Helios DMD input gradient."""
    fake_cond = flow_prediction_to_x0(
        fake_flow_cond, noisy_sample, timestep, sigmas, timesteps
    )
    if fake_guidance_scale != 0.0:
        if fake_flow_uncond is None:
            raise ValueError("fake_flow_uncond is required when fake guidance is enabled")
        fake_uncond = flow_prediction_to_x0(
            fake_flow_uncond, noisy_sample, timestep, sigmas, timesteps
        )
        pred_fake = fake_cond + (fake_cond - fake_uncond) * fake_guidance_scale
    else:
        pred_fake = fake_cond

    real_cond = flow_prediction_to_x0(
        real_flow_cond, noisy_sample, timestep, sigmas, timesteps
    )
    if real_guidance_scale != 0.0:
        if real_flow_uncond is None:
            raise ValueError("real_flow_uncond is required when real guidance is enabled")
        real_uncond = flow_prediction_to_x0(
            real_flow_uncond, noisy_sample, timestep, sigmas, timesteps
        )
        pred_real = real_cond + (real_cond - real_uncond) * real_guidance_scale
    else:
        pred_real = real_cond

    gradient = pred_fake - pred_real
    normalizer = torch.ones(
        (estimated_clean.shape[0], 1, 1, 1, 1),
        dtype=gradient.dtype,
        device=gradient.device,
    )
    if normalization:
        normalizer = torch.abs(estimated_clean - pred_real).mean(
            dim=[1, 2, 3, 4], keepdim=True
        )
        gradient = gradient / normalizer
    gradient = torch.nan_to_num(gradient)
    return gradient, {
        "pred_fake": pred_fake,
        "pred_real": pred_real,
        "normalizer": normalizer,
    }


def detached_dmd_surrogate_loss(
    generated: torch.Tensor,
    input_gradient: torch.Tensor,
    gradient_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Helios's detached 0.5*MSE surrogate that injects ``input_gradient``."""
    if generated.shape != input_gradient.shape:
        raise ValueError(
            f"generated/input_gradient shape mismatch: {generated.shape} != {input_gradient.shape}"
        )
    if gradient_mask is None:
        source = generated.double()
        target = (generated.double() - input_gradient.double()).detach()
    else:
        source = generated.double()[gradient_mask]
        target = (generated.double() - input_gradient.double()).detach()[gradient_mask]
    return 0.5 * F.mse_loss(source, target, reduction="mean")


def critic_denoising_loss(
    fake_flow_prediction: torch.Tensor,
    critic_noise: torch.Tensor,
    generated_x0: torch.Tensor,
) -> torch.Tensor:
    """Default GAN-disabled ``_critic_loss`` denoising objective."""
    if not (
        fake_flow_prediction.shape == critic_noise.shape == generated_x0.shape
    ):
        raise ValueError(
            "fake prediction, critic noise, and generated x0 must have equal shapes"
        )
    return F.mse_loss(
        fake_flow_prediction.float(),
        (critic_noise - generated_x0).float(),
        reduction="mean",
    )
