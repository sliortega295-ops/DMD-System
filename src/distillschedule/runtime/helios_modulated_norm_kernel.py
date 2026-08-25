"""Exact forward epilogue for the pinned Helios FP32 modulated norm.

PyTorch continues to own FP32 LayerNorm and the complete source backward.  The
kernel only replaces the eager forward expression after LayerNorm::

    BF16(FP32_norm * (1 + FP32_scale) + FP32_shift)

The three FP32 operations use explicit round-to-nearest PTX instructions and
the result is written directly to the final BF16 output tile.
"""
from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl


BLOCK_SIZE = 512
NUM_WARPS = 8
SUPPORTED_WIDTH = 5120


@triton.jit
def _helios_modulated_norm_epilogue_kernel(
    normalized_ptr,
    scale_ptr,
    shift_ptr,
    output_ptr,
    n_elements,
    normalized_stride_t,
    scale_stride_t,
    shift_stride_t,
    output_stride_t,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    token_index = offsets // WIDTH
    feature_index = offsets - token_index * WIDTH
    normalized_offsets = token_index * normalized_stride_t + feature_index
    scale_offsets = token_index * scale_stride_t + feature_index
    shift_offsets = token_index * shift_stride_t + feature_index
    output_offsets = token_index * output_stride_t + feature_index

    normalized = tl.load(normalized_ptr + normalized_offsets, mask=mask).to(
        tl.float32
    )
    scale = tl.load(scale_ptr + scale_offsets, mask=mask).to(tl.float32)
    shift = tl.load(shift_ptr + shift_offsets, mask=mask).to(tl.float32)
    one_plus_scale = tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [scale, 1.0],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    product = tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [normalized, one_plus_scale],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    output = tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [product, shift],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    tl.store(output_ptr + output_offsets, output, mask=mask)


def fused_modulated_norm_epilogue(
    normalized: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    """Write one B1 Helios tile, rejecting unregistered layouts."""

    values = (normalized, scale, shift, output)
    if not all(value.is_cuda for value in values):
        raise RuntimeError("Helios modulated-norm fusion requires CUDA tensors")
    if normalized.dtype is not torch.float32:
        raise RuntimeError("Helios modulated-norm input must be FP32")
    if scale.dtype is not torch.float32 or shift.dtype is not torch.float32:
        raise RuntimeError("Helios modulated-norm scale/shift must be FP32")
    if output.dtype is not torch.bfloat16:
        raise RuntimeError("Helios modulated-norm output must be BF16")
    if normalized.ndim != 3 or int(normalized.shape[0]) != 1:
        raise RuntimeError("Helios modulated-norm fusion requires rank-local B1")
    if any(value.shape != normalized.shape for value in values[1:]):
        raise RuntimeError("Helios modulated-norm fusion shape mismatch")
    if int(normalized.shape[2]) != SUPPORTED_WIDTH:
        raise RuntimeError(f"Helios modulation width changed: {normalized.shape[2]}")
    for name, value in (("normalized", normalized), ("output", output)):
        if value.stride(2) != 1 or value.stride(1) != SUPPORTED_WIDTH:
            raise RuntimeError(f"unsupported Helios {name} strides: {value.stride()}")
    for name, value in (("scale", scale), ("shift", shift)):
        if value.stride(2) != 1 or value.stride(1) not in (0, SUPPORTED_WIDTH):
            raise RuntimeError(f"unsupported Helios {name} strides: {value.stride()}")

    n_elements = int(output.numel())
    if n_elements == 0:
        return output
    _helios_modulated_norm_epilogue_kernel[
        (triton.cdiv(n_elements, BLOCK_SIZE),)
    ](
        normalized,
        scale,
        shift,
        output,
        n_elements,
        int(normalized.stride(1)),
        int(scale.stride(1)),
        int(shift.stride(1)),
        int(output.stride(1)),
        WIDTH=SUPPORTED_WIDTH,
        BLOCK=BLOCK_SIZE,
        num_warps=NUM_WARPS,
    )
    return output


def warmup(device: torch.device | str) -> dict[str, Any]:
    """Compile the exact B1 specialization outside the formal timer."""

    normalized = torch.zeros(
        (1, 512, SUPPORTED_WIDTH), device=device, dtype=torch.float32
    )
    scale = torch.zeros_like(normalized)
    shift = torch.zeros_like(normalized)
    output = torch.empty(
        (1, 512, SUPPORTED_WIDTH), device=device, dtype=torch.bfloat16
    )
    fused_modulated_norm_epilogue(normalized, scale, shift, output)
    torch.cuda.synchronize(device)
    return {
        "scope": "SETUP_ONLY_BEFORE_FORMAL_TIMER",
        "formal_timer_includes_warmup": False,
        "shape": [1, 512, SUPPORTED_WIDTH],
        "block_size": BLOCK_SIZE,
        "num_warps": NUM_WARPS,
        "torch_version": str(torch.__version__),
        "triton_version": str(triton.__version__),
    }


__all__ = ["fused_modulated_norm_epilogue", "warmup"]
