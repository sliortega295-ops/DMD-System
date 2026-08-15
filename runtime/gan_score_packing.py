"""Source-faithful Score request packing for the official Helios GAN boundary.

The official Generator-side GAN/DMD helper calls ``compute_kl_grad`` through a
module-global symbol.  This module provides a narrow, temporary replacement for
the ordinary (non-decoupled) path that packs adapter-disabled Real-Score
conditional and unconditional requests into one physical B2 forward.  Fake
Score and GAN-G remain separate because their adapter/``gan_mode`` semantics
are different.

The patch is process-local and active only around one official loss call.  It
never changes trace/random inputs, loss equations, gradients, or optimizer
order.  Correctness runs additionally execute the original two B1 Real forwards
and fail closed on multi-metric packed-vs-reference drift.
"""
from __future__ import annotations

import contextlib
import dataclasses
from typing import Any, Iterator

import torch

from distillschedule.runtime.fsdp2_p0 import pack_score_conditions, split_packed_prediction
from distillschedule.runtime.p0_parity import (
    DEFAULT_P0_PARITY,
    require_recorded_parity,
    tensor_parity_metrics,
)


# Compatibility name retained for callers that override only the mean-relative
# threshold.  It is no longer the sole correctness criterion.
PACKED_SCORE_MEAN_REL_TOLERANCE = DEFAULT_P0_PARITY.mean_relative


@dataclasses.dataclass
class PackedGANScoreState:
    parity_check: bool = False
    tolerance: float = PACKED_SCORE_MEAN_REL_TOLERANCE
    physical_fake_forwards: int = 0
    physical_real_forwards: int = 0
    logical_score_requests: int = 0
    parity: dict[str, Any] | None = None

    @property
    def physical_dmd_score_forwards(self) -> int:
        return self.physical_fake_forwards + self.physical_real_forwards


def _parity(reference: torch.Tensor, candidate: torch.Tensor, *, mean_relative: float) -> dict[str, Any]:
    thresholds = dataclasses.replace(DEFAULT_P0_PARITY, mean_relative=float(mean_relative))
    return tensor_parity_metrics(reference, candidate, thresholds=thresholds)


def _repeat_for_requests(value: torch.Tensor | None, copies: int, batch_size: int) -> torch.Tensor | None:
    if value is None:
        return None
    if value.ndim == 0:
        return value.repeat(copies)
    if int(value.shape[0]) == batch_size:
        return torch.cat([value] * copies, dim=0)
    # Helios position/index tensors are shared metadata without a batch axis.
    return value


def _score_call(
    model: Any,
    *,
    hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    condition: torch.Tensor,
    indices_hidden_states: torch.Tensor | None,
    indices_latents_history_short: torch.Tensor | None,
    indices_latents_history_mid: torch.Tensor | None,
    indices_latents_history_long: torch.Tensor | None,
    latents_history_short: torch.Tensor | None,
    latents_history_mid: torch.Tensor | None,
    latents_history_long: torch.Tensor | None,
) -> torch.Tensor:
    return model(
        hidden_states=hidden_states,
        timestep=timestep,
        encoder_hidden_states=condition,
        indices_hidden_states=indices_hidden_states,
        indices_latents_history_short=indices_latents_history_short,
        indices_latents_history_mid=indices_latents_history_mid,
        indices_latents_history_long=indices_latents_history_long,
        latents_history_short=latents_history_short,
        latents_history_mid=latents_history_mid,
        latents_history_long=latents_history_long,
        return_dict=False,
    )[0]


def _packed_real_pair(
    model: Any,
    *,
    hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    prompt_embeds: torch.Tensor,
    negative_prompt_embeds: torch.Tensor,
    indices_hidden_states: torch.Tensor | None,
    indices_latents_history_short: torch.Tensor | None,
    indices_latents_history_mid: torch.Tensor | None,
    indices_latents_history_long: torch.Tensor | None,
    latents_history_short: torch.Tensor | None,
    latents_history_mid: torch.Tensor | None,
    latents_history_long: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    source_batch = int(hidden_states.shape[0])
    if source_batch != 1:
        raise ValueError(f"GAN P0 Real-Score packing currently requires B1, got B{source_batch}")
    batch = {
        "hidden_states": hidden_states,
        "timestep": timestep,
        "indices_hidden_states": indices_hidden_states,
        "indices_latents_history_short": indices_latents_history_short,
        "indices_latents_history_mid": indices_latents_history_mid,
        "indices_latents_history_long": indices_latents_history_long,
        "latents_history_short": latents_history_short,
        "latents_history_mid": latents_history_mid,
        "latents_history_long": latents_history_long,
    }
    # ``pack_score_conditions`` ignores None/non-tensor entries and preserves
    # shared index metadata while duplicating all B1 state tensors.
    packed_batch, packed_condition = pack_score_conditions(
        batch,
        (prompt_embeds, negative_prompt_embeds),
    )
    packed_prediction = _score_call(
        model,
        hidden_states=packed_batch["hidden_states"],
        timestep=packed_batch["timestep"],
        condition=packed_condition,
        indices_hidden_states=packed_batch.get("indices_hidden_states"),
        indices_latents_history_short=packed_batch.get("indices_latents_history_short"),
        indices_latents_history_mid=packed_batch.get("indices_latents_history_mid"),
        indices_latents_history_long=packed_batch.get("indices_latents_history_long"),
        latents_history_short=packed_batch.get("latents_history_short"),
        latents_history_mid=packed_batch.get("latents_history_mid"),
        latents_history_long=packed_batch.get("latents_history_long"),
    )
    return split_packed_prediction(packed_prediction, 2)


def compute_kl_grad_real_pair_packed(
    accelerator: Any,
    scheduler: Any,
    real_fake_score_model: Any,
    noisy_image_or_video: torch.Tensor,
    estimated_clean_image_or_video: torch.Tensor,
    prompt_embeds: torch.Tensor,
    negative_prompt_embeds: torch.Tensor,
    timestep: torch.Tensor,
    sigmas: torch.Tensor,
    timesteps: torch.Tensor,
    fake_guidance_scale: float = 0.0,
    real_guidance_scale: float = 3.0,
    normalization: bool = True,
    is_decouple_dmd: bool = False,
    ca_noisy_image_or_video: torch.Tensor | None = None,
    dm_noisy_image_or_video: torch.Tensor | None = None,
    ca_timestep: torch.Tensor | None = None,
    dm_timestep: torch.Tensor | None = None,
    is_use_gt_history: bool = False,
    gt_all_data: tuple[Any, ...] | None = None,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Official ordinary ``compute_kl_grad`` with one packed Real B2 forward."""

    del scheduler, ca_noisy_image_or_video, dm_noisy_image_or_video, ca_timestep, dm_timestep
    from diffusers.utils.torch_utils import is_compiled_module
    from helios.utils.utils_helios_post import convert_flow_pred_to_x0

    state: PackedGANScoreState | None = getattr(
        compute_kl_grad_real_pair_packed, "_active_state", None
    )
    if state is None:
        raise RuntimeError("packed Helios compute_kl_grad called outside its scoped patch")
    if is_decouple_dmd:
        raise ValueError("GAN P0 Real-Score packing does not support decoupled DMD")
    if float(fake_guidance_scale) != 0.0:
        raise ValueError("GAN P0 Real-Score packing currently requires fake_guidance_scale=0")
    if not is_use_gt_history or gt_all_data is None:
        raise ValueError("GAN P0 Real-Score packing requires official GT-history inputs")

    model = accelerator.unwrap_model(real_fake_score_model)
    model = model._orig_mod if is_compiled_module(model) else model
    (
        _,
        indices_hidden_states,
        indices_latents_history_short,
        indices_latents_history_mid,
        indices_latents_history_long,
        latents_history_short,
        latents_history_mid,
        latents_history_long,
        _,
    ) = gt_all_data

    pred_fake_flow = _score_call(
        real_fake_score_model,
        hidden_states=noisy_image_or_video,
        timestep=timestep,
        condition=prompt_embeds,
        indices_hidden_states=indices_hidden_states,
        indices_latents_history_short=indices_latents_history_short,
        indices_latents_history_mid=indices_latents_history_mid,
        indices_latents_history_long=indices_latents_history_long,
        latents_history_short=latents_history_short,
        latents_history_mid=latents_history_mid,
        latents_history_long=latents_history_long,
    )
    state.physical_fake_forwards += 1
    state.logical_score_requests += 1
    pred_fake_image = convert_flow_pred_to_x0(
        flow_pred=pred_fake_flow,
        xt=noisy_image_or_video,
        timestep=timestep,
        sigmas=sigmas,
        timesteps=timesteps,
    )

    model.disable_adapters()
    try:
        real_cond_flow, real_uncond_flow = _packed_real_pair(
            real_fake_score_model,
            hidden_states=noisy_image_or_video,
            timestep=timestep,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            indices_hidden_states=indices_hidden_states,
            indices_latents_history_short=indices_latents_history_short,
            indices_latents_history_mid=indices_latents_history_mid,
            indices_latents_history_long=indices_latents_history_long,
            latents_history_short=latents_history_short,
            latents_history_mid=latents_history_mid,
            latents_history_long=latents_history_long,
        )
        state.physical_real_forwards += 1
        state.logical_score_requests += 2
        pred_real_cond = convert_flow_pred_to_x0(
            flow_pred=real_cond_flow,
            xt=noisy_image_or_video,
            timestep=timestep,
            sigmas=sigmas,
            timesteps=timesteps,
        )
        pred_real_uncond = convert_flow_pred_to_x0(
            flow_pred=real_uncond_flow,
            xt=noisy_image_or_video,
            timestep=timestep,
            sigmas=sigmas,
            timesteps=timesteps,
        )

        if state.parity_check:
            reference_cond_flow = _score_call(
                real_fake_score_model,
                hidden_states=noisy_image_or_video,
                timestep=timestep,
                condition=prompt_embeds,
                indices_hidden_states=indices_hidden_states,
                indices_latents_history_short=indices_latents_history_short,
                indices_latents_history_mid=indices_latents_history_mid,
                indices_latents_history_long=indices_latents_history_long,
                latents_history_short=latents_history_short,
                latents_history_mid=latents_history_mid,
                latents_history_long=latents_history_long,
            )
            reference_uncond_flow = _score_call(
                real_fake_score_model,
                hidden_states=noisy_image_or_video,
                timestep=timestep,
                condition=negative_prompt_embeds,
                indices_hidden_states=indices_hidden_states,
                indices_latents_history_short=indices_latents_history_short,
                indices_latents_history_mid=indices_latents_history_mid,
                indices_latents_history_long=indices_latents_history_long,
                latents_history_short=latents_history_short,
                latents_history_mid=latents_history_mid,
                latents_history_long=latents_history_long,
            )
            reference_cond = convert_flow_pred_to_x0(
                flow_pred=reference_cond_flow,
                xt=noisy_image_or_video,
                timestep=timestep,
                sigmas=sigmas,
                timesteps=timesteps,
            )
            reference_uncond = convert_flow_pred_to_x0(
                flow_pred=reference_uncond_flow,
                xt=noisy_image_or_video,
                timestep=timestep,
                sigmas=sigmas,
                timesteps=timesteps,
            )
            cond_error = _parity(reference_cond, pred_real_cond, mean_relative=state.tolerance)
            uncond_error = _parity(reference_uncond, pred_real_uncond, mean_relative=state.tolerance)
            reference_real = reference_cond + (
                reference_cond - reference_uncond
            ) * float(real_guidance_scale)
        else:
            cond_error = uncond_error = None
            reference_real = None
    finally:
        model.enable_adapters()

    pred_real = pred_real_cond + (
        pred_real_cond - pred_real_uncond
    ) * float(real_guidance_scale)
    grad = pred_fake_image - pred_real
    if normalization:
        normalizer = torch.abs(estimated_clean_image_or_video - pred_real).mean(
            dim=[1, 2, 3, 4], keepdim=True
        )
        grad = grad / normalizer
    grad = torch.nan_to_num(grad)

    if state.parity_check:
        assert reference_real is not None and cond_error is not None and uncond_error is not None
        reference_grad = pred_fake_image - reference_real
        if normalization:
            reference_normalizer = torch.abs(
                estimated_clean_image_or_video - reference_real
            ).mean(dim=[1, 2, 3, 4], keepdim=True)
            reference_grad = reference_grad / reference_normalizer
        reference_grad = torch.nan_to_num(reference_grad)
        gradient_error = _parity(reference_grad, grad, mean_relative=state.tolerance)
        rows = {
            "real_cond": cond_error,
            "real_uncond": uncond_error,
            "dmd_gradient": gradient_error,
        }
        status = "PASS" if all(value.get("status") == "PASS" for value in rows.values()) else "FAIL"
        state.parity = {
            "status": status,
            "policy": "multi_metric",
            **rows,
        }
        for label, metrics in rows.items():
            require_recorded_parity(metrics, label=f"GAN packed {label}")

    return (
        grad,
        None,
        None,
        {
            "dmdtrain_clean_latent": estimated_clean_image_or_video.detach(),
            "dmdtrain_noisy_latent": noisy_image_or_video.detach(),
            "dmdtrain_pred_real_image": pred_real.detach(),
            "dmdtrain_pred_fake_image": pred_fake_image.detach(),
            "dmdtrain_gradient_norm": torch.mean(torch.abs(grad)).detach(),
            "timestep": timestep.detach(),
        },
    )


@contextlib.contextmanager
def packed_gan_real_score_pair(
    *, parity_check: bool = False, tolerance: float = PACKED_SCORE_MEAN_REL_TOLERANCE
) -> Iterator[PackedGANScoreState]:
    """Temporarily route official Helios DMD through the packed Real pair."""

    import helios.utils.utils_helios_post as helios_post

    if tolerance <= 0:
        raise ValueError("packing parity tolerance must be positive")
    original = helios_post.compute_kl_grad
    state = PackedGANScoreState(parity_check=parity_check, tolerance=float(tolerance))
    if hasattr(compute_kl_grad_real_pair_packed, "_active_state"):
        raise RuntimeError("nested GAN Score packing scopes are not supported")
    compute_kl_grad_real_pair_packed._active_state = state  # type: ignore[attr-defined]
    helios_post.compute_kl_grad = compute_kl_grad_real_pair_packed
    try:
        yield state
    finally:
        helios_post.compute_kl_grad = original
        delattr(compute_kl_grad_real_pair_packed, "_active_state")


__all__ = [
    "PACKED_SCORE_MEAN_REL_TOLERANCE",
    "PackedGANScoreState",
    "compute_kl_grad_real_pair_packed",
    "packed_gan_real_score_pair",
]
