"""Diagnostic C14 fused Helios LayerNorm plus modulation kernel.

This module is deliberately not wired into the training runtime yet.  The
locked-A100 canary must prove the numerical and latency gates before C14 may
become an opt-in policy.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


SUPPORTED_WIDTH = 5120
ROW_BLOCK = 8192
NUM_WARPS = 8


@triton.jit
def _fused_bf16_layernorm_modulation_kernel(
    hidden_ptr,
    table_ptr,
    scale_component_ptr,
    shift_component_ptr,
    output_ptr,
    row_count,
    hidden_stride_b,
    hidden_stride_t,
    table_stride_c,
    scale_component_stride_b,
    scale_component_stride_t,
    shift_component_stride_b,
    shift_component_stride_t,
    output_stride_b,
    output_stride_t,
    tokens_per_batch,
    EPS: tl.constexpr,
    SCALE_INDEX: tl.constexpr,
    SHIFT_INDEX: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    feature = tl.arange(0, BLOCK)
    mask = feature < WIDTH
    batch = row // tokens_per_batch
    token = row - batch * tokens_per_batch

    hidden_offsets = (
        batch * hidden_stride_b + token * hidden_stride_t + feature
    )
    hidden = tl.load(hidden_ptr + hidden_offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    mean = tl.sum(tl.where(mask, hidden, 0.0), axis=0) / WIDTH
    centered = hidden - mean
    variance = tl.sum(
        tl.where(mask, centered * centered, 0.0), axis=0
    ) / WIDTH
    normalized = centered * tl.rsqrt(variance + EPS)

    table_scale = tl.load(
        table_ptr + SCALE_INDEX * table_stride_c + feature,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    table_shift = tl.load(
        table_ptr + SHIFT_INDEX * table_stride_c + feature,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    scale_component = tl.load(
        scale_component_ptr
        + batch * scale_component_stride_b
        + token * scale_component_stride_t
        + feature,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    shift_component = tl.load(
        shift_component_ptr
        + batch * shift_component_stride_b
        + token * shift_component_stride_t
        + feature,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    scale = tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [table_scale, scale_component],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    shift = tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [table_shift, shift_component],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
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
    value = tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [product, shift],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    output_offsets = batch * output_stride_b + token * output_stride_t + feature
    tl.store(output_ptr + output_offsets, value, mask=mask)


def fused_bf16_layernorm_modulation(
    hidden: torch.Tensor,
    scale_shift_table: torch.Tensor,
    scale_index: int,
    shift_index: int,
    scale_component: torch.Tensor,
    shift_component: torch.Tensor,
    output: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    """Read BF16 hidden state and write the final BF16 modulated norm tile."""

    values = (
        hidden,
        scale_shift_table,
        scale_component,
        shift_component,
        output,
    )
    if not all(value.is_cuda for value in values):
        raise RuntimeError("C14 fused LayerNorm requires CUDA tensors")
    if hidden.dtype is not torch.bfloat16 or output.dtype is not torch.bfloat16:
        raise RuntimeError("C14 hidden/output must be BF16")
    if scale_shift_table.dtype not in (torch.bfloat16, torch.float32):
        raise RuntimeError("unsupported C14 scale-shift table dtype")
    if scale_component.dtype not in (torch.bfloat16, torch.float32):
        raise RuntimeError("unsupported C14 scale component dtype")
    if shift_component.dtype not in (torch.bfloat16, torch.float32):
        raise RuntimeError("unsupported C14 shift component dtype")
    if hidden.ndim != 3 or int(hidden.shape[0]) not in (1, 3):
        raise RuntimeError("C14 requires registered B1/B3 layouts")
    if hidden.shape != output.shape:
        raise RuntimeError("C14 output shape changed")
    if int(hidden.shape[2]) != SUPPORTED_WIDTH:
        raise RuntimeError(f"C14 width changed: {hidden.shape[2]}")
    if scale_shift_table.ndim != 3 or tuple(scale_shift_table.shape[:2]) != (1, 6):
        raise RuntimeError("C14 scale-shift table layout changed")
    if int(scale_shift_table.shape[2]) != SUPPORTED_WIDTH:
        raise RuntimeError("C14 scale-shift table width changed")
    if not 0 <= int(scale_index) < 6 or not 0 <= int(shift_index) < 6:
        raise RuntimeError("C14 component index changed")
    for name, value in (
        ("scale_component", scale_component),
        ("shift_component", shift_component),
    ):
        if value.shape != hidden.shape:
            raise RuntimeError(f"C14 {name} shape changed")
        if value.stride(2) != 1 or value.stride(1) not in (0, SUPPORTED_WIDTH):
            raise RuntimeError(f"unsupported C14 {name} stride: {value.stride()}")
    for name, value in (("hidden", hidden), ("output", output)):
        if value.stride(2) != 1 or value.stride(1) != SUPPORTED_WIDTH:
            raise RuntimeError(f"unsupported C14 {name} stride: {value.stride()}")
    if scale_shift_table.stride(2) != 1:
        raise RuntimeError("unsupported C14 scale-shift table stride")
    if any(value.stride(0) < 0 for value in values):
        raise RuntimeError("C14 does not support negative strides")
    if float(eps) <= 0:
        raise RuntimeError("C14 LayerNorm epsilon must be positive")

    row_count = int(hidden.shape[0] * hidden.shape[1])
    if row_count == 0:
        return output
    _fused_bf16_layernorm_modulation_kernel[(row_count,)](
        hidden,
        scale_shift_table,
        scale_component,
        shift_component,
        output,
        row_count,
        int(hidden.stride(0)),
        int(hidden.stride(1)),
        int(scale_shift_table.stride(1)),
        int(scale_component.stride(0)),
        int(scale_component.stride(1)),
        int(shift_component.stride(0)),
        int(shift_component.stride(1)),
        int(output.stride(0)),
        int(output.stride(1)),
        int(hidden.shape[1]),
        EPS=float(eps),
        SCALE_INDEX=int(scale_index),
        SHIFT_INDEX=int(shift_index),
        WIDTH=SUPPORTED_WIDTH,
        BLOCK=ROW_BLOCK,
        num_warps=NUM_WARPS,
    )
    return output


__all__ = [
    "fused_bf16_layernorm_modulation",
    "NUM_WARPS",
    "ROW_BLOCK",
    "SUPPORTED_WIDTH",
]
