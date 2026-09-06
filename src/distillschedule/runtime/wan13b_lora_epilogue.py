"""Exact eager Triton epilogue for one no-grad FP32 Wan LoRA update.

The reference PEFT path first rounds ``update * scaling`` to FP32, then rounds
the FP32 add with the BF16 base result, and finally converts the result to
BF16.  CUDA normally emits separate pointwise kernels for those operations.
This diagnostic candidate preserves the two IEEE round-to-nearest FP32
operations with inline PTX and performs the final BF16 store in one kernel.

This module is imported lazily only when the explicitly gated candidate is
enabled.  The ordinary training path therefore has no Triton dependency or
dispatch change.
"""
from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl


@triton.jit
def _wan13b_lora_epilogue_kernel(
    base_result,
    lora_update,
    output,
    numel,
    scaling,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    base = tl.load(base_result + offsets, mask=mask, other=0.0).to(tl.float32)
    update = tl.load(lora_update + offsets, mask=mask, other=0.0).to(tl.float32)
    scale = tl.full((BLOCK_SIZE,), scaling, tl.float32)
    scaled = tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [update, scale],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    summed = tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [base, scaled],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    tl.store(output + offsets, summed, mask=mask)


def fused_lora_epilogue(
    base_result: torch.Tensor,
    lora_update: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    """Return the reference ``(base + update * scale).to(BF16)`` result.

    Fail closed instead of inserting an unregistered layout or dtype copy.
    Real Wan PEFT Linear outputs are contiguous tensors of identical shape.
    """

    if not base_result.is_cuda or not lora_update.is_cuda:
        raise RuntimeError("Wan fused LoRA epilogue requires CUDA tensors")
    if base_result.dtype is not torch.bfloat16:
        raise RuntimeError("Wan fused LoRA epilogue requires a BF16 base result")
    if lora_update.dtype is not torch.float32:
        raise RuntimeError("Wan fused LoRA epilogue requires an FP32 LoRA update")
    if base_result.shape != lora_update.shape:
        raise RuntimeError("Wan fused LoRA epilogue shape mismatch")
    if not base_result.is_contiguous() or not lora_update.is_contiguous():
        raise RuntimeError("Wan fused LoRA epilogue requires contiguous inputs")

    output = torch.empty_like(base_result)
    numel = int(output.numel())
    if numel == 0:
        return output
    block_size = 256
    grid = (triton.cdiv(numel, block_size),)
    _wan13b_lora_epilogue_kernel[grid](
        base_result,
        lora_update,
        output,
        numel,
        float(scaling),
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return output


def runtime_identity() -> dict[str, Any]:
    return {
        "implementation": "TRITON_INLINE_PTX_MUL_RN_ADD_RN_BF16_STORE",
        "triton_version": str(triton.__version__),
        "torch_version": str(torch.__version__),
        "block_size": 256,
    }


__all__ = ["fused_lora_epilogue", "runtime_identity"]
