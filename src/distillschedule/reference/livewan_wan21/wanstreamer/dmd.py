"""Distribution matching distillation for the block causal student.

    x0_real = teacher(x_t, t)
    x0_fake = critic(x_t, t)
    grad = (x0_fake - x0_real) / normalizer
    loss = 0.5 * ||x0 - stopgrad(x0 - grad)||^2
"""
import torch
import torch.nn.functional as tnnF

from . import blockcausal as bc
from .data import add_noise


def bidirectional_velocity(model, z, t, rope, ctx, ctx_lens, dtype,
                           time_scale=1000.0, grad_checkpoint=False):
    """Run a full bidirectional forward over one clip."""
    return bc.block_forward(model, z, t, 0, rope, ctx, ctx_lens, kv=None,
                            dtype=dtype, time_scale=time_scale,
                            grad_checkpoint=grad_checkpoint)


def cfg_velocity(model, z, t, rope, ctx_pos, ctx_neg, ctx_lens, dtype,
                 guidance, time_scale=1000.0):
    """Teacher velocity with classifier free guidance, cond and uncond batched."""
    if not guidance or guidance == 1.0:
        return bidirectional_velocity(model, z, t, rope, ctx_pos, ctx_lens,
                                      dtype, time_scale)
    z2 = torch.cat([z, z], 0)
    ctx2 = torch.cat([ctx_pos, ctx_neg], 0)
    lens2 = ctx_lens.expand(2)
    v = bidirectional_velocity(model, z2, t, rope, ctx2, lens2, dtype, time_scale)
    vc, vu = v[0:1], v[1:2]
    return vu + guidance * (vc - vu)


@torch.no_grad()
def dmd_gradient(x0, t, z_t, v_real, v_fake):
    """Compute the DMD2 KL gradient with one normalizer for the full clip."""
    while t.dim() < z_t.dim():
        t = t.unsqueeze(-1)
    x0_real = (z_t - t * v_real).float()
    x0_fake = (z_t - t * v_fake).float()
    normalizer = (x0.float() - x0_real).abs().mean().clamp_min(1e-4)
    grad = torch.nan_to_num((x0_fake - x0_real) / normalizer)
    return grad, {'dmd_grad_norm': float(grad.norm()),
                  'dmd_normalizer': float(normalizer),
                  'dmd_real_fake_gap': float((x0_real - x0_fake).abs().mean())}


def dmd_surrogate(x0_hat, grad, scale=1.0):
    """Scalar whose gradient wrt x0_hat is exactly `grad` (times scale)."""
    return scale * 0.5 * tnnF.mse_loss(x0_hat.float(),
                                       (x0_hat.detach() - grad).float())


def critic_loss(model, x0, t, rope, ctx, ctx_lens, dtype, time_scale=1000.0,
                grad_checkpoint=True, noise=None):
    """Flow matching loss for the fake score network on student samples."""
    z_t, target, _ = add_noise(x0, t, noise)
    v = bidirectional_velocity(model, z_t, t, rope, ctx, ctx_lens, dtype,
                               time_scale, grad_checkpoint=grad_checkpoint)
    return tnnF.mse_loss(v.float(), target.float())
