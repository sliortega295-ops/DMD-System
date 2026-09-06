"""H3 shared-backbone rollout, score, and AdaLN-cache runtime methods."""

from __future__ import annotations

import dataclasses
from typing import Any

import torch

from lightx2v_train.model_zoo.native.minimax_h3 import build_row_timesteps
from lightx2v_train.trainers.dmd.math import dmd_loss_pair, weighted_mse_pair

from .matched_contract import FAKE_ROLE, STUDENT_ROLE, TEACHER_ROLE
from .model import BASE_ROLE, FAKE_ADAPTER, STUDENT_ADAPTER


@dataclasses.dataclass
class PreparedFakeUpdate:
    latent_shape: dict[str, Any]
    condition: dict[str, torch.Tensor]
    generated: tuple[torch.Tensor, torch.Tensor]
    noises: tuple[torch.Tensor, torch.Tensor]
    renoised: tuple[torch.Tensor, torch.Tensor]
    sigmas: tuple[torch.Tensor, torch.Tensor]
    cache_key: tuple[str, int]


class H3A100RuntimeMixin:
    """Methods that execute Student/Fake/Teacher forwards on one module."""

    # ------------------------------------------------------------------
    # RNG and rollout control
    # ------------------------------------------------------------------
    def sample_initial_latents(self, latent_shape):
        # Match upstream LightX2V exactly. Sequence-parallel synchronization,
        # if enabled in a future ablation, remains owned by the upstream helper.
        return super().sample_initial_latents(latent_shape)

    def _sample_synced_int(self, low, high):
        if (
            getattr(self, "matched_compute_enabled", False)
            and int(low) == 0
            and int(high) == int(self.num_inference_steps)
        ):
            return int(self.matched_fixed_end_step_idx)
        return super()._sample_synced_int(low, high)

    def _sample_renoise_sigmas(self):
        # Keep the exact continuous random-sigma objective used by upstream.
        return super()._sample_renoise_sigmas()

    @staticmethod
    def _randn_like_exact(tensor):
        return torch.randn_like(tensor, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Role switching, compute census, and AdaLN precomputation
    # ------------------------------------------------------------------
    def _predict_velocity_role(self, role, latents, sigmas, condition, latent_shape):
        logical_role = {
            STUDENT_ADAPTER: STUDENT_ROLE,
            FAKE_ADAPTER: FAKE_ROLE,
            BASE_ROLE: TEACHER_ROLE,
        }[role]
        if getattr(self, "matched_compute_enabled", False):
            self.matched_cycle_census.note_forward(
                logical_role,
                grad_enabled=bool(torch.is_grad_enabled()),
            )
        with self.shared_model.role_scope(role):
            return super()._predict_velocity(
                self.shared_model, latents, sigmas, condition, latent_shape
            )

    def _adaln_timesteps(self, condition, latent_shape, sigmas):
        layout = self._layout(condition, latent_shape)
        timesteps, _ = build_row_timesteps(layout, sigmas[0], sigmas[1])
        return timesteps.to(self.shared_model.device)

    def run_back_simulation(self, condition, latent_shape, end_step_idx, grad_enabled, xt=None):
        if xt is None:
            xt = self.sample_initial_latents(latent_shape)
        self.shared_model.set_transformer_training(True)
        x0 = None
        backward_key = None
        for step_idx in range(end_step_idx + 1):
            sigmas = (
                self.video_sigmas_cpu[step_idx].to(self.shared_model.device),
                self.audio_sigmas_cpu[step_idx].to(self.shared_model.device),
            )
            key = ("rollout", int(step_idx))
            timesteps = self._adaln_timesteps(condition, latent_shape, sigmas)
            self.shared_model.precompute_adaln(key, timesteps, persistent=True)
            context = torch.enable_grad if grad_enabled and step_idx == end_step_idx else torch.no_grad
            with context(), self.shared_model.adaln_scope(key, persistent=True):
                velocity = self._predict_velocity_role(
                    STUDENT_ADAPTER, xt, sigmas, condition, latent_shape
                )
                x0 = (
                    xt[0] + sigmas[0] * velocity[0],
                    xt[1] + sigmas[1] * velocity[1],
                )
            if grad_enabled and step_idx == end_step_idx:
                backward_key = key
            next_sigmas = (
                self.video_sigmas_cpu[step_idx + 1].to(self.shared_model.device),
                self.audio_sigmas_cpu[step_idx + 1].to(self.shared_model.device),
            )
            xt = (
                (xt[0].float() + (sigmas[0] - next_sigmas[0]) * velocity[0].float()).to(
                    self.latent_dtype
                ),
                (xt[1].float() + (sigmas[1] - next_sigmas[1]) * velocity[1].float()).to(
                    self.latent_dtype
                ),
            )
        if x0 is None:
            raise RuntimeError("H3 rollout produced no x0 prediction")
        return (x0[0].to(self.latent_dtype), x0[1].to(self.latent_dtype)), backward_key

    def forward_student_loss(self, latent_shape, conditions, current_iter=None):
        del current_iter
        condition, _ = conditions
        end_step_idx = self._sample_synced_int(0, self.num_inference_steps)
        generated, backward_key = self.run_back_simulation(
            condition, latent_shape, end_step_idx, grad_enabled=True
        )
        sigmas = self._sample_renoise_sigmas()
        noises = (
            self._randn_like_exact(generated[0]),
            self._randn_like_exact(generated[1]),
        )
        renoised = self._add_noise(generated, noises, sigmas)

        score_key = self._next_score_key()
        timesteps = self._adaln_timesteps(condition, latent_shape, sigmas)
        self.shared_model.precompute_adaln(score_key, timesteps, persistent=False)
        self.shared_model.set_transformer_training(False)
        with torch.no_grad(), self.shared_model.adaln_scope(score_key, persistent=False):
            velocity_fake = self._predict_velocity_role(
                FAKE_ADAPTER, renoised, sigmas, condition, latent_shape
            )
            velocity_teacher = self._predict_velocity_role(
                BASE_ROLE, renoised, sigmas, condition, latent_shape
            )
            x0_fake = (
                renoised[0] + sigmas[0] * velocity_fake[0],
                renoised[1] + sigmas[1] * velocity_fake[1],
            )
            x0_teacher = (
                renoised[0] + sigmas[0] * velocity_teacher[0],
                renoised[1] + sigmas[1] * velocity_teacher[1],
            )
        self.shared_model.drop_adaln_key(score_key)
        loss = dmd_loss_pair(
            generated,
            x0_fake,
            x0_teacher,
            self.video_loss_weight,
            self.audio_loss_weight,
        )
        if backward_key is None:
            raise RuntimeError("Student DMD loss has no rollout backward cache key")
        return {"loss": loss, "dmd": loss.detach(), "backward_key": backward_key}

    @torch.no_grad()
    def prepare_fake_update(self, latent_shape, conditions) -> PreparedFakeUpdate:
        condition, _ = conditions
        end_step_idx = self._sample_synced_int(0, self.num_inference_steps)
        generated, _ = self.run_back_simulation(
            condition, latent_shape, end_step_idx, grad_enabled=False
        )
        generated = (generated[0].detach(), generated[1].detach())
        sigmas = self._sample_renoise_sigmas()
        noises = (
            self._randn_like_exact(generated[0]),
            self._randn_like_exact(generated[1]),
        )
        renoised = self._add_noise(generated, noises, sigmas)
        return PreparedFakeUpdate(
            latent_shape=latent_shape,
            condition=condition,
            generated=generated,
            noises=noises,
            renoised=renoised,
            sigmas=sigmas,
            cache_key=self._next_score_key(),
        )

    def fake_loss(self, item: PreparedFakeUpdate):
        timesteps = self._adaln_timesteps(item.condition, item.latent_shape, item.sigmas)
        self.shared_model.precompute_adaln(item.cache_key, timesteps, persistent=False)
        self.shared_model.set_transformer_training(True)
        with self.shared_model.adaln_scope(item.cache_key, persistent=False):
            velocity_fake = self._predict_velocity_role(
                FAKE_ADAPTER,
                item.renoised,
                item.sigmas,
                item.condition,
                item.latent_shape,
            )
            target = (
                item.generated[0].float() - item.noises[0],
                item.generated[1].float() - item.noises[1],
            )
            loss = weighted_mse_pair(
                velocity_fake,
                target,
                self.video_loss_weight,
                self.audio_loss_weight,
            )
        return loss

    def _next_score_key(self):
        self._score_serial += 1
        return ("score", self._score_serial)
