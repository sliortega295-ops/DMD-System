"""DeepSpeed boundary helper for the proposed GAN 8Score Critic lane.

Upstream Helios low-VRAM GAN directly drives the DeepSpeed engine so the
Critic denoising backward and GAN/R1/R2 backward reach the same optimizer step.
The matched 8Score layout represents global batch16 as two micro-waves, so the
same rule extends to four backwards followed by exactly one engine step.

This helper deliberately owns only boundary/step ordering.  It does not compute
or rescale losses; the caller must use the same loss normalization that the
validated DeepSpeed GAS=2 contract resolves on the target cluster.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GanBackwardOutcome:
    micro_wave: int
    denoising_backward: bool
    gan_backward: bool
    optimizer_committed: bool
    accumulation_boundary_after_gan: bool


def backward_low_vram_gan_micro_wave(
    *,
    engine: Any,
    denoising_loss: Any,
    gan_regularization_loss: Any,
    micro_wave: int,
    micro_waves_per_logical_step: int = 2,
) -> GanBackwardOutcome:
    """Accumulate one GAN Critic micro-wave into a logical batch16 update.

    Expected call sequence for two waves::

        wave0: boundary=False; backward(denoise)
               boundary=False; backward(gan/r1/r2)
        wave1: boundary=False; backward(denoise)
               boundary=True;  backward(gan/r1/r2); step()

    The caller must not invoke another Critic optimizer step for this logical
    batch after the final wave.
    """

    if micro_waves_per_logical_step != 2:
        raise ValueError("primary world16 GAN pipeline requires exactly two micro-waves")
    if micro_wave not in (0, 1):
        raise ValueError("micro_wave must be 0 or 1")
    for name in ("set_gradient_accumulation_boundary", "backward", "step"):
        if not callable(getattr(engine, name, None)):
            raise TypeError(f"DeepSpeed engine lacks required method {name}")

    engine.set_gradient_accumulation_boundary(is_boundary=False)
    engine.backward(denoising_loss)

    final_wave = micro_wave == micro_waves_per_logical_step - 1
    engine.set_gradient_accumulation_boundary(is_boundary=final_wave)
    engine.backward(gan_regularization_loss)
    if final_wave:
        engine.step()

    return GanBackwardOutcome(
        micro_wave=micro_wave,
        denoising_backward=True,
        gan_backward=True,
        optimizer_committed=final_wave,
        accumulation_boundary_after_gan=final_wave,
    )


__all__ = ["GanBackwardOutcome", "backward_low_vram_gan_micro_wave"]
