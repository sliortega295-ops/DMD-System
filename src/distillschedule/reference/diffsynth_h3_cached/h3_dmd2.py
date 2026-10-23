"""Native H3 DMD2: shared frozen DiT, independent LoRAs, velocity-token critic.

Only generated AV latents enter these losses. Clean reference chunks are separate
inputs to H3, whose model_fn strips reference predictions from its return values.
"""

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

# Preserve the differentiable attention default used by the cache training tools.
from .cache import prepare_cached_inputs
from .diffusion import FlowMatchScheduler
from .models.minimax_h3_dit import pack_audio, patchify_video
from .packing import model_fn_minimax_h3


@dataclass
class DMD2Config:
    num_inference_steps: int = 8
    video_shift: float = 2.22
    audio_shift: float = 2.22
    critic_updates: int = 5
    student_learning_rate: float = 1e-5
    fake_score_learning_rate: float = 1e-5
    discriminator_learning_rate: float = 4e-7
    weight_decay: float = 0.01
    lora_rank: int = 64
    lora_alpha: float = 64
    lora_targets: str = "attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2"
    audio_loss_weight: float = 0.5
    dmd_loss_weight: float = 1.0
    sigma_min: float = 0.02
    sigma_max: float = 0.98
    score_normalization_eps: float = 1e-5
    teacher_cfg_scale: float = 1.0
    adversarial_enabled: bool = True
    gan_start_step: int = 500
    generator_loss_weight: float = 0.05
    discriminator_loss_weight: float = 0.01
    hidden_channels: int = 256
    num_layers: int = 4
    max_grad_norm: float = 1.0
    critic_max_grad_norm: float = 10.0
    gradient_checkpointing: bool = True

    def __post_init__(self):
        for name in ("num_inference_steps", "critic_updates", "lora_rank", "hidden_channels", "num_layers"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("video_shift", "audio_shift", "student_learning_rate", "fake_score_learning_rate",
                     "discriminator_learning_rate", "lora_alpha", "score_normalization_eps",
                     "max_grad_norm", "critic_max_grad_norm", "dmd_loss_weight"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("audio_loss_weight", "weight_decay", "generator_loss_weight", "discriminator_loss_weight"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0 < self.sigma_min < self.sigma_max < 1:
            raise ValueError("Require 0 < sigma_min < sigma_max < 1")
        if not math.isfinite(self.teacher_cfg_scale) or self.teacher_cfg_scale < 1:
            raise ValueError("teacher_cfg_scale must be finite and >= 1")
        if type(self.gan_start_step) is not int or self.gan_start_step < 0:
            raise ValueError("gan_start_step must be a nonnegative integer")
        if self.adversarial_enabled and min(self.generator_loss_weight, self.discriminator_loss_weight) <= 0:
            raise ValueError("DMD2 requires positive GAN weights; disable adversarial_enabled for DMD only")
        if not self.lora_targets or any(not x.strip() for x in self.lora_targets.split(",")):
            raise ValueError("lora_targets must contain nonempty module suffixes")
        for name in ("adversarial_enabled", "gradient_checkpointing"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")


def make_schedule(config, device="cpu"):
    """K+1 sigma boundaries, including the native H3 final Euler step to zero."""
    schedules = []
    for shift in (config.video_shift, config.audio_shift):
        sigmas, _ = FlowMatchScheduler.set_timesteps_minimax_h3(config.num_inference_steps, shift=shift)
        schedules.append(torch.cat((sigmas, sigmas.new_zeros(1))).to(device))
    return tuple(schedules)


def add_noise(clean, sigma, noise):
    return (1 - sigma) * clean + sigma * noise


def prediction_to_x0(noisy, sigma, velocity):
    return noisy - sigma * velocity


def dmd_gradient(generated, teacher_x0, fake_x0, eps):
    # Each entry is one complete modality/sample, including BOTH stereo channels.
    scale = (generated.float() - teacher_x0.float()).abs().mean().clamp_min(eps)
    return ((fake_x0.float() - teacher_x0.float()) / scale).detach()


def exact_gradient_surrogate(value, gradient):
    # gradient already includes reduction/weights; do not average it a second time.
    return (value.float() * gradient.detach()).sum()


class DualLoRALinear(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.scale = alpha / rank
        self.adapters = nn.ModuleDict()
        for role in ("student", "fake_score"):
            pair = nn.ModuleDict({
                "A": nn.Linear(base.in_features, rank, bias=False, device=base.weight.device, dtype=torch.float32),
                "B": nn.Linear(rank, base.out_features, bias=False, device=base.weight.device, dtype=torch.float32),
            })
            nn.init.zeros_(pair["B"].weight)
            self.adapters[role] = pair
        self.role = "teacher"

    def forward(self, value):
        result = self.base(value)
        if self.role != "teacher":
            pair = self.adapters[self.role]
            update = pair["B"](pair["A"](value.to(pair["A"].weight.dtype)))
            result = result + update.to(result.dtype) * self.scale
        return result


class H3Adapter(nn.Module):
    def __init__(self, dit, config):
        super().__init__()
        self.dit = dit.eval().requires_grad_(False)
        self.config = config
        targets = [x.strip() for x in config.lora_targets.split(",")]
        matches = {target: [] for target in targets}
        selected = []
        for name, layer in dit.named_modules():
            hits = [t for t in targets if name == t or name.endswith("." + t)]
            if hits:
                if not isinstance(layer, nn.Linear):
                    raise ValueError(f"LoRA target {name} is not a Linear layer")
                selected.append((name, layer))
                for hit in hits:
                    matches[hit].append(name)
        if any(not names for names in matches.values()):
            raise ValueError(f"Unmatched H3 LoRA targets: {[t for t, names in matches.items() if not names]}")
        for name, layer in selected:
            parent, _, leaf = name.rpartition(".")
            setattr(dit.get_submodule(parent), leaf, DualLoRALinear(layer, config.lora_rank, config.lora_alpha))
        self.layers = [(name, dit.get_submodule(name)) for name, _ in selected]
        self.select("teacher")

    def role_parameters(self, role):
        return [p for _, layer in self.layers for p in layer.adapters[role].parameters()]

    def select(self, role, trainable=False):
        if role not in ("teacher", "student", "fake_score"):
            raise ValueError(f"Invalid role: {role}")
        for _, layer in self.layers:
            layer.role = role
            for name, adapter in layer.adapters.items():
                adapter.requires_grad_(trainable and name == role)

    def predict(self, latents, sigmas, inputs, checkpoint=False):
        predictor = getattr(self, "_distributed_predict", None)
        if predictor is not None:
            return predictor(latents, sigmas, inputs, checkpoint)
        return self.forward(latents, sigmas, inputs, checkpoint)

    def forward(self, latents, sigmas, inputs, checkpoint=False):
        return model_fn_minimax_h3(
            self.dit, video_latents=latents[0], audio_latents=latents[1],
            timestep_video=sigmas[0] * 1000, timestep_audio=sigmas[1] * 1000,
            use_gradient_checkpointing=checkpoint, **inputs,
        )

    def export_lora(self, role):
        state = {}
        for name, layer in self.layers:
            state[name + ".lora_A.weight"] = layer.adapters[role]["A"].weight.detach().cpu().contiguous()
            # Bake alpha/rank into B for the native DiffSynth LoRA loader.
            state[name + ".lora_B.weight"] = (layer.adapters[role]["B"].weight.detach().cpu() * layer.scale).contiguous()
        return state


class VelocityDiscriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.audio_weight = config.audio_loss_weight
        self.heads = nn.ModuleList()
        for channels in (96, 32):
            layers = []
            for _ in range(config.num_layers - 1):
                layers.extend((nn.Linear(channels, config.hidden_channels), nn.SiLU()))
                channels = config.hidden_channels
            layers.append(nn.Linear(channels, 1))
            self.heads.append(nn.Sequential(*layers))

    def forward(self, velocities):
        # H3 returns generated tokens only: references cannot enter pooling.
        return (self.heads[0](patchify_video(velocities[0]).float()),
                self.heads[1](pack_audio(velocities[1]).float()))


def prepare_batch(raw, cfg_scale=1.0):
    shared, positive = (raw[0], raw[1]) if isinstance(raw, (tuple, list)) else (raw, raw)
    # The reused cache parser rebuilds packing. Never silently discard masks.
    for key in ("denoise_mask_video", "denoise_mask_audio", "padding_mask", "attention_mask", "control_rows"):
        if shared.get(key) is not None or positive.get(key) is not None:
            raise ValueError(f"DMD2 cache entry {key} is unsupported; use T2AV or a clean previous AV chunk")
    positive_inputs = prepare_cached_inputs(raw)
    negative_inputs = None
    if cfg_scale > 1:
        negative = raw[2] if isinstance(raw, (tuple, list)) else {
            "prompt_embeds": raw.get("negative_prompt_embeds"),
            "text_token_tags": raw.get("negative_text_token_tags"),
        }
        if not isinstance(negative, dict) or negative.get("prompt_embeds") is None:
            raise ValueError("teacher_cfg_scale > 1 requires cached negative prompt embeddings")
        # Recover reference shape from positive packing even if the negative text
        # length differs. The parser rebuilds all positions for that text length.
        negative = {**negative, "packed": positive.get("packed")}
        negative_inputs = prepare_cached_inputs((shared, negative, {}))
    return positive_inputs, negative_inputs


def modality_sum(values, config):
    return values[0] + config.audio_loss_weight * values[1]


@torch.no_grad()
def rollout(adapter, real, inputs, schedule, selected_step=None):
    adapter.select("student")
    latents = tuple(torch.randn_like(x) for x in real)
    selected = None
    for index in range(len(schedule[0]) - 1):
        if index == selected_step:
            selected = tuple(x.clone() for x in latents)
        sigmas = tuple(s[index] for s in schedule)
        velocities = adapter.predict(latents, sigmas, inputs)
        latents = tuple(x + (s[index + 1] - s[index]) * v for x, s, v in zip(latents, schedule, velocities))
    return latents, selected


class DMD2Trainer:
    """One complete rollout, N completed critic steps, then one student step.

    step_fn owns backward, distributed gradient averaging, clipping and stepping.
    The selected adapter must stay active until its backward has completed.
    """
    def __init__(self, adapter, discriminator, config, step_fn):
        self.adapter, self.discriminator, self.config, self.step_fn = adapter, discriminator, config, step_fn
        self.schedule = make_schedule(config, next(adapter.parameters()).device)
        self.generator_step = 0
        self.critic_step = 0

    def score_noise(self, real):
        c = self.config
        # Shared AV noise level, shifted logit-normal then clamped, as in plan.
        base = torch.randn((), device=real[0].device).sigmoid()
        sigmas = tuple((shift * base / (1 + (shift - 1) * base)).clamp(c.sigma_min, c.sigma_max)
                       for shift in (c.video_shift, c.audio_shift))
        return sigmas, tuple(torch.randn_like(x) for x in real)

    def update(self, real, inputs, negative=None):
        a, d, c = self.adapter, self.discriminator, self.config
        if c.teacher_cfg_scale > 1 and negative is None:
            raise ValueError("Missing negative condition for teacher CFG")
        selected_step = int(torch.randint(c.num_inference_steps, ()).item())
        generated, selected = rollout(a, real, inputs, self.schedule, selected_step)
        gan = c.adversarial_enabled and self.generator_step >= c.gan_start_step
        critic_losses = []
        for _ in range(c.critic_updates):
            sigmas, noise = self.score_noise(real)
            noisy_fake = tuple(add_noise(x, s, n) for x, s, n in zip(generated, sigmas, noise))
            a.select("fake_score", trainable=True)
            d.requires_grad_(gan)
            fake_v = a.predict(noisy_fake, sigmas, inputs, c.gradient_checkpointing)
            loss = modality_sum([F.mse_loss(v.float(), (n - x).float()) for v, n, x in zip(fake_v, noise, generated)], c)
            if gan:
                noisy_real = tuple(add_noise(x, s, n) for x, s, n in zip(real, sigmas, noise))
                real_v = a.predict(noisy_real, sigmas, inputs, c.gradient_checkpointing)
                real_logits, fake_logits = d(real_v), d(fake_v)
                # Convention in dmd2_plan.md: real negative, fake positive.
                loss = loss + c.discriminator_loss_weight * modality_sum([
                    F.softplus(r).mean() + F.softplus(-f).mean() for r, f in zip(real_logits, fake_logits)], c)
            self.step_fn("critic", loss)
            self.critic_step += 1
            critic_losses.append(loss.detach())

        # Fresh predictions from the latest critic. Neither score branch receives
        # generator parameter gradients; GAN differentiates only its noisy input.
        sigmas, noise = self.score_noise(real)
        noisy_fake = tuple(add_noise(x, s, n) for x, s, n in zip(generated, sigmas, noise))
        with torch.no_grad():
            a.select("teacher")
            teacher_v = a.predict(noisy_fake, sigmas, inputs)
            if c.teacher_cfg_scale > 1:
                negative_v = a.predict(noisy_fake, sigmas, negative)
                teacher_v = tuple(n + c.teacher_cfg_scale * (p - n) for p, n in zip(teacher_v, negative_v))
            teacher_x0 = tuple(prediction_to_x0(x, s, v) for x, s, v in zip(noisy_fake, sigmas, teacher_v))
            a.select("fake_score")
            fake_v = a.predict(noisy_fake, sigmas, inputs)
            fake_x0 = tuple(prediction_to_x0(x, s, v) for x, s, v in zip(noisy_fake, sigmas, fake_v))
            gradients = tuple(dmd_gradient(x, t, f, c.score_normalization_eps) for x, t, f in zip(generated, teacher_x0, fake_x0))
        d.requires_grad_(False)
        gan_value = generated[0].new_zeros((), dtype=torch.float32)
        gan_gradients = None
        if gan:
            leaves = tuple(x.detach().requires_grad_() for x in noisy_fake)
            logits = d(a.predict(leaves, sigmas, inputs, c.gradient_checkpointing))
            gan_value = c.generator_loss_weight * modality_sum([F.softplus(x).mean() for x in logits], c)
            gan_gradients = torch.autograd.grad(gan_value, leaves)

        a.select("student", trainable=True)
        selected_sigmas = tuple(s[selected_step] for s in self.schedule)
        velocities = a.predict(selected, selected_sigmas, inputs, c.gradient_checkpointing)
        next_latents = tuple(x + (s[selected_step + 1] - s[selected_step]) * v
                             for x, s, v in zip(selected, self.schedule, velocities))
        proxies = tuple(final + (nxt - nxt.detach()) for final, nxt in zip(generated, next_latents))
        dmd_loss = c.dmd_loss_weight * modality_sum([
            0.5 * F.mse_loss(proxy.float(), (proxy.float() - gradient).detach())
            for proxy, gradient in zip(proxies, gradients)], c)
        student_loss = dmd_loss
        if gan_gradients is not None:
            # Includes (1-sigma) exactly once, and GAN weight is already in grad.
            student_loss = student_loss + sum(exact_gradient_surrogate(add_noise(p, s, n), g)
                                             for p, s, n, g in zip(proxies, sigmas, noise, gan_gradients))
        self.step_fn("student", student_loss)
        self.generator_step += 1
        return {"dmd_loss": float(dmd_loss.detach()), "gan_loss": float(gan_value.detach()),
                "critic_loss": float(torch.stack(critic_losses).mean()), "gan_active": gan,
                "generator_step": self.generator_step, "critic_step": self.critic_step}
