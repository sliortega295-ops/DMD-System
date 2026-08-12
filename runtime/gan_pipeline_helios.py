"""Helios-specific tensor boundary for the 8G+8Score GAN pipeline.

The proposed Generator-update split uses the chain rule rather than copying the
Score graph onto Generator ranks:

    G forward -> x_fake
    Score service computes official DMD + GAN-G on x_fake.detach().requires_grad_()
    Score service returns dL/dx_fake
    Generator rank applies that gradient to its local autograd graph

This is exactly the same boundary exploited by Helios' own
``Gan_D_Loss_With_Cached_Grad`` for GAN-G, generalized to the complete Score-side
DMD+GAN-G objective.  The official Helios function remains the owner of DMD and
GAN math; this module only extracts the latent gradient and verifies that the
Score optimizer state was not touched.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence


@dataclass(frozen=True)
class ScoreBoundaryResult:
    gradient: Any
    total_loss: float | None
    dmd_loss: float | None
    gan_g_loss_raw: float | None
    gan_g_loss_weighted: float | None
    logs: Mapping[str, Any]


@contextmanager
def _temporarily_freeze_parameters(model: Any):
    parameters = list(model.parameters())
    original = [bool(parameter.requires_grad) for parameter in parameters]
    preexisting_grad_ids = {
        index: id(parameter.grad)
        for index, parameter in enumerate(parameters)
        if parameter.grad is not None
    }
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        yield
        for index, parameter in enumerate(parameters):
            if parameter.grad is not None and index not in preexisting_grad_ids:
                raise AssertionError(
                    "Score boundary created a parameter gradient; GAN-G/DMD service must be read-only"
                )
            if (
                parameter.grad is not None
                and index in preexisting_grad_ids
                and id(parameter.grad) != preexisting_grad_ids[index]
            ):
                raise AssertionError("Score boundary replaced an existing parameter gradient")
    finally:
        for parameter, requires_grad in zip(parameters, original):
            parameter.requires_grad_(requires_grad)


def _scalar(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "float"):
        value = value.float()
    if hasattr(value, "mean"):
        value = value.mean()
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def compute_official_score_boundary_grad(
    *,
    score_model: Any,
    generated_latent: Any,
    gan_g_weight: float,
    official_kwargs: Mapping[str, Any],
    compute_distribution_matching_loss: Callable[..., Sequence[Any]] | None = None,
    collect_diagnostics: bool = True,
) -> ScoreBoundaryResult:
    """Return the exact Score-side latent gradient for one Generator micro-wave.

    ``official_kwargs`` must contain the same frozen arguments that upstream
    ``_generator_loss`` passes to ``compute_distribution_matching_loss`` except
    for ``real_fake_score_model`` and ``image_or_video``; those are injected here.
    The helper deliberately does not own noise/timestep sampling.  Formal runs
    must supply those values from the rank-qualified operation trace.
    """

    if gan_g_weight < 0:
        raise ValueError("gan_g_weight must be non-negative")
    forbidden = {"real_fake_score_model", "image_or_video"} & set(official_kwargs)
    if forbidden:
        raise ValueError(
            "official_kwargs must not override pipeline-owned fields: "
            + ", ".join(sorted(forbidden))
        )
    if compute_distribution_matching_loss is None:
        from helios.utils.utils_helios_post import (
            compute_distribution_matching_loss as official_compute,
        )

        compute_distribution_matching_loss = official_compute

    import torch

    leaf = generated_latent.detach().requires_grad_(True)
    with _temporarily_freeze_parameters(score_model), torch.enable_grad():
        outputs = compute_distribution_matching_loss(
            real_fake_score_model=score_model,
            image_or_video=leaf,
            **dict(official_kwargs),
        )
        if len(outputs) != 5:
            raise AssertionError(
                "official compute_distribution_matching_loss return signature changed"
            )
        dmd_loss, _ca_dmd_loss, _dm_dmd_loss, gan_g_loss, logs = outputs
        if not getattr(dmd_loss, "requires_grad", False):
            raise AssertionError("official DMD loss lost latent autograd")
        if not getattr(gan_g_loss, "requires_grad", False):
            raise AssertionError("official GAN-G loss lost latent autograd")
        weighted_gan = gan_g_loss * float(gan_g_weight)
        total = dmd_loss + weighted_gan
        gradient = torch.autograd.grad(
            total,
            leaf,
            retain_graph=False,
            create_graph=False,
            only_inputs=True,
        )[0].detach()

    if tuple(gradient.shape) != tuple(generated_latent.shape):
        raise AssertionError("Score boundary gradient shape changed")
    if gradient.dtype != generated_latent.dtype:
        # Keep the wire simple and avoid a hidden precision contract change.
        gradient = gradient.to(dtype=generated_latent.dtype)
    return ScoreBoundaryResult(
        gradient=gradient,
        total_loss=_scalar(total) if collect_diagnostics else None,
        dmd_loss=_scalar(dmd_loss) if collect_diagnostics else None,
        gan_g_loss_raw=_scalar(gan_g_loss) if collect_diagnostics else None,
        gan_g_loss_weighted=(
            _scalar(weighted_gan) if collect_diagnostics else None
        ),
        logs=(dict(logs) if collect_diagnostics and isinstance(logs, Mapping) else {}),
    )


def apply_remote_score_gradient(
    *,
    generated_latent: Any,
    remote_gradient: Any,
    scale: float = 1.0,
) -> None:
    """Continue Generator backward from a Score-computed latent gradient."""

    import torch

    if tuple(generated_latent.shape) != tuple(remote_gradient.shape):
        raise ValueError("remote gradient shape does not match Generator output")
    if not getattr(generated_latent, "requires_grad", False):
        raise ValueError("Generator output must retain autograd")
    grad = remote_gradient.to(
        device=generated_latent.device,
        dtype=generated_latent.dtype,
        non_blocking=True,
    )
    if scale != 1.0:
        grad = grad * float(scale)
    torch.autograd.backward(generated_latent, grad_tensors=grad)


__all__ = [
    "ScoreBoundaryResult",
    "apply_remote_score_gradient",
    "compute_official_score_boundary_grad",
]
