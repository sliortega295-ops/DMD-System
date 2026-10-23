"""Teacher latent dataset and shared flow matching conventions."""
import glob
import os

import torch
from torch.utils.data import Dataset


class TeacherLatents(Dataset):
    """Teacher ODE samples: clean latents [16, F, H, W] + their prompt index."""

    def __init__(self, root, prompts_path, frames=None):
        self.files = sorted(glob.glob(os.path.join(root, '*.pt')))
        if not self.files:
            raise RuntimeError(f'no .pt latents under {root}')
        blob = torch.load(prompts_path, map_location='cpu')
        self.pos = blob['pos']            # [P, 512, 4096] fp16
        self.neg = blob['neg']            # [1, 512, 4096] fp16
        self.prompts = blob['prompts']
        self.frames = frames

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = torch.load(self.files[i], map_location='cpu', weights_only=False)
        lat = d['latents'].float()
        if self.frames is not None:
            lat = lat[:, :self.frames]
        pi = int(d['prompt_idx'])
        return lat, self.pos[pi].float(), pi


def shifted_uniform_t(n, shift=5.0, device='cpu', generator=None):
    """Flow fractions distributed as Wan's inference sigma schedule.

    Wan samples with shift=5, which concentrates steps near t=1. Training with
    plain U(0,1) would underserve exactly the region the student spends most of
    its few steps in, so draw from the same reparameterisation the sampler uses:
        t = shift*u / (1 + (shift-1)*u),  u ~ U(0,1).
    """
    u = torch.rand(n, device=device, generator=generator)
    return shift * u / (1 + (shift - 1) * u)


def add_noise(z0, t, noise=None):
    """Wan's rectified flow convention, matching train_streaming.py:

        z_t    = (1 - t) * z0 + t * noise
        target = noise - z0        (velocity the model predicts)

    t broadcasts over [B] or [B, F]; z0 is [B, C, F, H, W].
    """
    if noise is None:
        noise = torch.randn_like(z0)
    while t.dim() < z0.dim():
        t = t.unsqueeze(-1)
    return (1 - t) * z0 + t * noise, noise - z0, noise


def x0_from_velocity(z_t, v, t):
    """Invert the flow parameterisation: z0 = z_t - t * v."""
    while t.dim() < z_t.dim():
        t = t.unsqueeze(-1)
    return z_t - t * v
