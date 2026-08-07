from __future__ import annotations

import pytest
import torch

from runtime.dmd_boundary import (
    add_noise,
    critic_denoising_loss,
    detached_dmd_surrogate_loss,
    dmd_input_gradient,
    flow_prediction_to_x0,
    sample_uniform_timestep,
    shifted_sigma_schedule,
)


def test_shifted_schedule_matches_helios_formula():
    sigmas, timesteps = shifted_sigma_schedule(num_train_timesteps=4, timestep_shift=5.0)
    base = torch.linspace(1.0, 0.25, 4, dtype=torch.float64)
    expected = 5.0 * base / (1.0 + 4.0 * base)
    torch.testing.assert_close(sigmas, expected)
    torch.testing.assert_close(timesteps, expected * 4)


def test_noise_and_flow_conversion_use_nearest_sigma():
    sigmas = torch.tensor([1.0, 0.5, 0.0], dtype=torch.float64)
    timesteps = torch.tensor([1000.0, 500.0, 0.0], dtype=torch.float64)
    clean = torch.full((1, 1, 1, 1, 1), 2.0)
    noise = torch.full_like(clean, 10.0)
    timestep = torch.tensor([501])
    noisy = add_noise(clean, noise, timestep, sigmas, timesteps)
    torch.testing.assert_close(noisy, torch.full_like(clean, 6.0))
    flow = torch.full_like(clean, 4.0)
    x0 = flow_prediction_to_x0(flow, noisy, timestep, sigmas, timesteps)
    torch.testing.assert_close(x0, torch.full_like(clean, 4.0))


def test_dmd_gradient_cfg_and_normalization():
    shape = (1, 1, 1, 1, 2)
    clean = torch.tensor([[[[[2.0, 4.0]]]]])
    noisy = torch.tensor([[[[[1.0, 3.0]]]]])
    timestep = torch.tensor([500])
    sigmas = torch.tensor([0.5], dtype=torch.float64)
    timesteps = torch.tensor([500.0], dtype=torch.float64)
    fake_flow = torch.full(shape, 2.0)
    real_cond_flow = torch.zeros(shape)
    real_uncond_flow = torch.full(shape, 2.0)
    gradient, aux = dmd_input_gradient(
        estimated_clean=clean,
        noisy_sample=noisy,
        timestep=timestep,
        sigmas=sigmas,
        timesteps=timesteps,
        fake_flow_cond=fake_flow,
        real_flow_cond=real_cond_flow,
        real_flow_uncond=real_uncond_flow,
        real_guidance_scale=3.0,
    )
    fake = noisy - 1.0
    real_cond = noisy
    real_uncond = noisy - 1.0
    real = real_cond + (real_cond - real_uncond) * 3.0
    normalizer = torch.abs(clean - real).mean(dim=[1, 2, 3, 4], keepdim=True)
    torch.testing.assert_close(aux["pred_fake"], fake)
    torch.testing.assert_close(aux["pred_real"], real)
    torch.testing.assert_close(aux["normalizer"], normalizer)
    torch.testing.assert_close(gradient, (fake - real) / normalizer)


def test_detached_surrogate_injects_mean_scaled_input_gradient():
    generated = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    input_gradient = torch.randn_like(generated)
    loss = detached_dmd_surrogate_loss(generated, input_gradient)
    loss.backward()
    torch.testing.assert_close(generated.grad, input_gradient / generated.numel())


def test_uniform_timestep_is_seeded_and_bounded():
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    first = sample_uniform_timestep(32, generator=g1)
    second = sample_uniform_timestep(32, generator=g2)
    assert torch.equal(first, second)
    assert int(first.min()) >= 20
    assert int(first.max()) <= 980


def test_detached_surrogate_through_flow_to_x0():
    flow = torch.randn(1, 1, 1, 1, 4, dtype=torch.float64, requires_grad=True)
    noisy = torch.randn_like(flow)
    gradient = torch.randn_like(flow)
    timestep = torch.tensor([500.0], dtype=torch.float64)
    sigmas = torch.tensor([0.5], dtype=torch.float64)
    x0 = flow_prediction_to_x0(flow, noisy, timestep, sigmas, timestep)
    loss = detached_dmd_surrogate_loss(x0, gradient)
    loss.backward()
    torch.testing.assert_close(flow.grad, -0.5 * gradient / gradient.numel())


def test_critic_denoising_loss_matches_source_target_and_detaches_generator():
    prediction = torch.tensor([1.0, -2.0], requires_grad=True)
    noise = torch.tensor([0.5, 3.0])
    generated = torch.tensor([-0.5, 1.0])

    loss = critic_denoising_loss(prediction, noise, generated)
    expected = torch.mean((prediction.float() - (noise - generated).float()) ** 2)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert prediction.grad is not None
    assert generated.grad is None


def test_critic_denoising_loss_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="equal shapes"):
        critic_denoising_loss(torch.zeros(2), torch.zeros(3), torch.zeros(2))
