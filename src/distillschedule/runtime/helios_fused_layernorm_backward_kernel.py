"""Diagnostic C15 fused Helios modulated-LayerNorm backward kernels.

This module is intentionally standalone until the locked-A100 canary proves
the registered B1/B3 gradient and latency gates.  The kernels preserve the
source FP32 LayerNorm math, BF16 hidden-gradient output, and deterministic
timestep-component reductions without materializing full FP32 hidden,
normalized, or grad-normalized tensors.
"""
from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl


SUPPORTED_WIDTH = 5120
ROW_BLOCK = 8192
ROW_NUM_WARPS = 8
REDUCE_FEATURE_BLOCK = 64
REDUCE_TOKEN_BLOCK = 32
REDUCE_NUM_WARPS = 8


@triton.jit
def _fused_bf16_layernorm_modulation_hidden_backward_kernel(
    hidden_ptr,
    grad_output_ptr,
    table_ptr,
    scale_component_ptr,
    hidden_gradient_ptr,
    mean_ptr,
    rstd_ptr,
    hidden_stride_b,
    hidden_stride_t,
    grad_stride_b,
    grad_stride_t,
    table_stride_c,
    component_stride_b,
    component_stride_t,
    hidden_gradient_stride_b,
    hidden_gradient_stride_t,
    stats_stride_b,
    stats_stride_t,
    tokens_per_batch,
    EPS: tl.constexpr,
    SCALE_INDEX: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    feature = tl.arange(0, BLOCK)
    mask = feature < WIDTH
    batch = row // tokens_per_batch
    token = row - batch * tokens_per_batch

    hidden_offsets = batch * hidden_stride_b + token * hidden_stride_t + feature
    hidden = tl.load(hidden_ptr + hidden_offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    mean = tl.sum(tl.where(mask, hidden, 0.0), axis=0) / WIDTH
    centered = hidden - mean
    variance = tl.sum(
        tl.where(mask, centered * centered, 0.0), axis=0
    ) / WIDTH
    rstd = tl.rsqrt(variance + EPS)
    normalized = centered * rstd

    grad_offsets = batch * grad_stride_b + token * grad_stride_t + feature
    grad_output = tl.load(
        grad_output_ptr + grad_offsets, mask=mask, other=0.0
    ).to(tl.float32)
    table_scale = tl.load(
        table_ptr + SCALE_INDEX * table_stride_c + feature,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    scale_component = tl.load(
        scale_component_ptr
        + batch * component_stride_b
        + token * component_stride_t
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
    one_plus_scale = tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [scale, 1.0],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    grad_normalized = tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [grad_output, one_plus_scale],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    grad_sum = tl.sum(tl.where(mask, grad_normalized, 0.0), axis=0)
    grad_normalized_sum = tl.sum(
        tl.where(mask, grad_normalized * normalized, 0.0), axis=0
    )
    hidden_gradient = (
        grad_normalized
        - grad_sum / WIDTH
        - normalized * (grad_normalized_sum / WIDTH)
    ) * rstd

    hidden_gradient_offsets = (
        batch * hidden_gradient_stride_b
        + token * hidden_gradient_stride_t
        + feature
    )
    tl.store(
        hidden_gradient_ptr + hidden_gradient_offsets,
        hidden_gradient,
        mask=mask,
    )
    stats_offset = batch * stats_stride_b + token * stats_stride_t
    tl.store(mean_ptr + stats_offset, mean)
    tl.store(rstd_ptr + stats_offset, rstd)


@triton.jit
def _fused_reduced_component_gradient_kernel(
    hidden_ptr,
    grad_output_ptr,
    mean_ptr,
    rstd_ptr,
    component_gradient_ptr,
    hidden_stride_b,
    hidden_stride_t,
    grad_stride_b,
    grad_stride_t,
    stats_stride_b,
    stats_stride_t,
    output_stride_b,
    output_stride_c,
    token_count,
    SCALE_INDEX: tl.constexpr,
    SHIFT_INDEX: tl.constexpr,
    WIDTH: tl.constexpr,
    FEATURE_BLOCK: tl.constexpr,
    TOKEN_BLOCK: tl.constexpr,
):
    feature_block = tl.program_id(0)
    batch = tl.program_id(1)
    feature = feature_block * FEATURE_BLOCK + tl.arange(0, FEATURE_BLOCK)
    feature_mask = feature < WIDTH
    scale_accumulator = tl.zeros((FEATURE_BLOCK,), dtype=tl.float32)
    shift_accumulator = tl.zeros((FEATURE_BLOCK,), dtype=tl.float32)

    for token_start in tl.range(0, token_count, TOKEN_BLOCK):
        token = token_start + tl.arange(0, TOKEN_BLOCK)
        token_mask = token < token_count
        matrix_mask = token_mask[:, None] & feature_mask[None, :]
        hidden_offsets = (
            batch * hidden_stride_b
            + token[:, None] * hidden_stride_t
            + feature[None, :]
        )
        grad_offsets = (
            batch * grad_stride_b
            + token[:, None] * grad_stride_t
            + feature[None, :]
        )
        hidden = tl.load(
            hidden_ptr + hidden_offsets, mask=matrix_mask, other=0.0
        ).to(tl.float32)
        grad_output = tl.load(
            grad_output_ptr + grad_offsets, mask=matrix_mask, other=0.0
        ).to(tl.float32)
        stats_offsets = batch * stats_stride_b + token * stats_stride_t
        mean = tl.load(mean_ptr + stats_offsets, mask=token_mask, other=0.0)
        rstd = tl.load(rstd_ptr + stats_offsets, mask=token_mask, other=0.0)
        normalized = (hidden - mean[:, None]) * rstd[:, None]
        scale_accumulator += tl.sum(
            tl.where(matrix_mask, grad_output * normalized, 0.0), axis=0
        )
        shift_accumulator += tl.sum(
            tl.where(matrix_mask, grad_output, 0.0), axis=0
        )

    scale_offsets = (
        batch * output_stride_b + SCALE_INDEX * output_stride_c + feature
    )
    shift_offsets = (
        batch * output_stride_b + SHIFT_INDEX * output_stride_c + feature
    )
    tl.store(
        component_gradient_ptr + scale_offsets,
        scale_accumulator,
        mask=feature_mask,
    )
    tl.store(
        component_gradient_ptr + shift_offsets,
        shift_accumulator,
        mask=feature_mask,
    )


@triton.jit
def _fused_tensor4d_component_gradient_kernel(
    hidden_ptr,
    grad_output_ptr,
    mean_ptr,
    rstd_ptr,
    component_gradient_ptr,
    hidden_stride_b,
    hidden_stride_t,
    grad_stride_b,
    grad_stride_t,
    stats_stride_b,
    stats_stride_t,
    output_stride_b,
    output_stride_t,
    output_stride_c,
    tokens_per_batch,
    SCALE_INDEX: tl.constexpr,
    SHIFT_INDEX: tl.constexpr,
    WIDTH: tl.constexpr,
    FEATURE_BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    feature = tl.program_id(1) * FEATURE_BLOCK + tl.arange(0, FEATURE_BLOCK)
    feature_mask = feature < WIDTH
    batch = row // tokens_per_batch
    token = row - batch * tokens_per_batch
    hidden_offsets = batch * hidden_stride_b + token * hidden_stride_t + feature
    grad_offsets = batch * grad_stride_b + token * grad_stride_t + feature
    hidden = tl.load(
        hidden_ptr + hidden_offsets, mask=feature_mask, other=0.0
    ).to(tl.float32)
    grad_output = tl.load(
        grad_output_ptr + grad_offsets, mask=feature_mask, other=0.0
    ).to(tl.float32)
    stats_offset = batch * stats_stride_b + token * stats_stride_t
    mean = tl.load(mean_ptr + stats_offset)
    rstd = tl.load(rstd_ptr + stats_offset)
    normalized = (hidden - mean) * rstd
    output_base = batch * output_stride_b + token * output_stride_t
    tl.store(
        component_gradient_ptr
        + output_base
        + SCALE_INDEX * output_stride_c
        + feature,
        grad_output * normalized,
        mask=feature_mask,
    )
    tl.store(
        component_gradient_ptr
        + output_base
        + SHIFT_INDEX * output_stride_c
        + feature,
        grad_output,
        mask=feature_mask,
    )


def _validate_hidden_inputs(
    hidden: torch.Tensor,
    grad_output: torch.Tensor,
    scale_shift_table: torch.Tensor,
    scale_component: torch.Tensor,
    hidden_gradient: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
) -> None:
    values = (
        hidden,
        grad_output,
        scale_shift_table,
        scale_component,
        hidden_gradient,
        mean,
        rstd,
    )
    if not all(value.is_cuda for value in values):
        raise RuntimeError("C15 fused backward requires CUDA tensors")
    if hidden.dtype is not torch.bfloat16:
        raise RuntimeError("C15 hidden input must be BF16")
    if grad_output.dtype is not torch.bfloat16:
        raise RuntimeError("C15 upstream gradient must be BF16")
    if hidden_gradient.dtype is not torch.bfloat16:
        raise RuntimeError("C15 hidden gradient must be BF16")
    if scale_shift_table.dtype not in (torch.bfloat16, torch.float32):
        raise RuntimeError("unsupported C15 scale-shift table dtype")
    if scale_component.dtype not in (torch.bfloat16, torch.float32):
        raise RuntimeError("unsupported C15 scale component dtype")
    if mean.dtype is not torch.float32 or rstd.dtype is not torch.float32:
        raise RuntimeError("C15 row statistics must be FP32")
    if hidden.ndim != 3 or int(hidden.shape[0]) not in (1, 3):
        raise RuntimeError("C15 requires registered B1/B3 layouts")
    if hidden.shape != grad_output.shape or hidden.shape != hidden_gradient.shape:
        raise RuntimeError("C15 hidden/gradient shape changed")
    if scale_component.shape != hidden.shape:
        raise RuntimeError("C15 scale component shape changed")
    if tuple(mean.shape) != tuple(hidden.shape[:2]) or mean.shape != rstd.shape:
        raise RuntimeError("C15 row-statistic shape changed")
    if int(hidden.shape[2]) != SUPPORTED_WIDTH:
        raise RuntimeError(f"C15 width changed: {hidden.shape[2]}")
    if scale_shift_table.ndim != 3 or tuple(scale_shift_table.shape[:2]) != (1, 6):
        raise RuntimeError("C15 scale-shift table layout changed")
    if int(scale_shift_table.shape[2]) != SUPPORTED_WIDTH:
        raise RuntimeError("C15 scale-shift table width changed")
    for name, value in (
        ("hidden", hidden),
        ("grad_output", grad_output),
        ("hidden_gradient", hidden_gradient),
    ):
        if value.stride(2) != 1 or value.stride(1) != SUPPORTED_WIDTH:
            raise RuntimeError(f"unsupported C15 {name} stride: {value.stride()}")
    # The production PhaseLocal layouts are contiguous or token-expanded.  The
    # registered tensor4d correctness layout selects one of six interleaved
    # components and therefore carries a 6*WIDTH token stride; the kernel uses
    # the explicit stride and can admit it without materialization.
    if scale_component.stride(2) != 1 or scale_component.stride(1) < 0:
        raise RuntimeError(
            f"unsupported C15 scale-component stride: {scale_component.stride()}"
        )
    if mean.stride(1) != 1 or rstd.stride(1) != 1:
        raise RuntimeError("unsupported C15 row-statistic stride")


def fused_bf16_layernorm_modulation_hidden_backward(
    hidden: torch.Tensor,
    grad_output: torch.Tensor,
    scale_shift_table: torch.Tensor,
    scale_index: int,
    scale_component: torch.Tensor,
    hidden_gradient: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    """Compute BF16 hidden gradient and reusable FP32 row statistics."""

    _validate_hidden_inputs(
        hidden,
        grad_output,
        scale_shift_table,
        scale_component,
        hidden_gradient,
        mean,
        rstd,
    )
    if not 0 <= int(scale_index) < 6:
        raise RuntimeError("C15 scale index changed")
    if float(eps) <= 0:
        raise RuntimeError("C15 LayerNorm epsilon must be positive")
    row_count = int(hidden.shape[0] * hidden.shape[1])
    if row_count == 0:
        return hidden_gradient
    _fused_bf16_layernorm_modulation_hidden_backward_kernel[(row_count,)](
        hidden,
        grad_output,
        scale_shift_table,
        scale_component,
        hidden_gradient,
        mean,
        rstd,
        int(hidden.stride(0)),
        int(hidden.stride(1)),
        int(grad_output.stride(0)),
        int(grad_output.stride(1)),
        int(scale_shift_table.stride(1)),
        int(scale_component.stride(0)),
        int(scale_component.stride(1)),
        int(hidden_gradient.stride(0)),
        int(hidden_gradient.stride(1)),
        int(mean.stride(0)),
        int(mean.stride(1)),
        int(hidden.shape[1]),
        EPS=float(eps),
        SCALE_INDEX=int(scale_index),
        WIDTH=SUPPORTED_WIDTH,
        BLOCK=ROW_BLOCK,
        num_warps=ROW_NUM_WARPS,
    )
    return hidden_gradient


def fused_reduced_component_gradient(
    hidden: torch.Tensor,
    grad_output: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    component_gradient: torch.Tensor,
    scale_index: int,
    shift_index: int,
) -> torch.Tensor:
    """Reduce one exact token interval into a Bx6xD component gradient."""

    if component_gradient.ndim != 3:
        raise RuntimeError("C15 reduced component gradient must be rank 3")
    if tuple(component_gradient.shape) != (
        int(hidden.shape[0]),
        6,
        SUPPORTED_WIDTH,
    ):
        raise RuntimeError("C15 reduced component-gradient shape changed")
    if component_gradient.dtype not in (torch.bfloat16, torch.float32):
        raise RuntimeError("unsupported C15 component-gradient dtype")
    if not component_gradient.is_cuda:
        raise RuntimeError("C15 component gradient must be CUDA")
    if int(hidden.shape[1]) <= 0:
        return component_gradient
    grid = (
        triton.cdiv(SUPPORTED_WIDTH, REDUCE_FEATURE_BLOCK),
        int(hidden.shape[0]),
    )
    _fused_reduced_component_gradient_kernel[grid](
        hidden,
        grad_output,
        mean,
        rstd,
        component_gradient,
        int(hidden.stride(0)),
        int(hidden.stride(1)),
        int(grad_output.stride(0)),
        int(grad_output.stride(1)),
        int(mean.stride(0)),
        int(mean.stride(1)),
        int(component_gradient.stride(0)),
        int(component_gradient.stride(1)),
        int(hidden.shape[1]),
        SCALE_INDEX=int(scale_index),
        SHIFT_INDEX=int(shift_index),
        WIDTH=SUPPORTED_WIDTH,
        FEATURE_BLOCK=REDUCE_FEATURE_BLOCK,
        TOKEN_BLOCK=REDUCE_TOKEN_BLOCK,
        num_warps=REDUCE_NUM_WARPS,
    )
    return component_gradient


def fused_tensor4d_component_gradient(
    hidden: torch.Tensor,
    grad_output: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    component_gradient: torch.Tensor,
    scale_index: int,
    shift_index: int,
) -> torch.Tensor:
    """Write per-token scale/shift gradients into a BxTx6xD tensor."""

    expected = (
        int(hidden.shape[0]),
        int(hidden.shape[1]),
        6,
        SUPPORTED_WIDTH,
    )
    if component_gradient.ndim != 4 or tuple(component_gradient.shape) != expected:
        raise RuntimeError("C15 tensor4d component-gradient shape changed")
    if component_gradient.dtype not in (torch.bfloat16, torch.float32):
        raise RuntimeError("unsupported C15 tensor4d gradient dtype")
    if not component_gradient.is_cuda:
        raise RuntimeError("C15 tensor4d gradient must be CUDA")
    row_count = int(hidden.shape[0] * hidden.shape[1])
    if row_count == 0:
        return component_gradient
    grid = (
        row_count,
        triton.cdiv(SUPPORTED_WIDTH, REDUCE_FEATURE_BLOCK),
    )
    _fused_tensor4d_component_gradient_kernel[grid](
        hidden,
        grad_output,
        mean,
        rstd,
        component_gradient,
        int(hidden.stride(0)),
        int(hidden.stride(1)),
        int(grad_output.stride(0)),
        int(grad_output.stride(1)),
        int(mean.stride(0)),
        int(mean.stride(1)),
        int(component_gradient.stride(0)),
        int(component_gradient.stride(1)),
        int(component_gradient.stride(2)),
        int(hidden.shape[1]),
        SCALE_INDEX=int(scale_index),
        SHIFT_INDEX=int(shift_index),
        WIDTH=SUPPORTED_WIDTH,
        FEATURE_BLOCK=REDUCE_FEATURE_BLOCK,
        num_warps=REDUCE_NUM_WARPS,
    )
    return component_gradient


def warmup(
    device: torch.device | str,
    *,
    batch_sizes: tuple[int, ...] = (1, 3),
    tokens: int = 2048,
    eps: float = 1.0e-6,
) -> dict[str, Any]:
    """Compile registered phase-local C15 specializations before formal timing."""

    for batch_size in batch_sizes:
        hidden = torch.zeros(
            (int(batch_size), int(tokens), SUPPORTED_WIDTH),
            device=device,
            dtype=torch.bfloat16,
        )
        grad_output = torch.zeros_like(hidden)
        table = torch.zeros(
            (1, 6, SUPPORTED_WIDTH), device=device, dtype=torch.bfloat16
        )
        component = torch.zeros(
            (int(batch_size), 1, SUPPORTED_WIDTH),
            device=device,
            dtype=torch.bfloat16,
        ).expand_as(hidden)
        hidden_gradient = torch.empty_like(hidden)
        mean = torch.empty(hidden.shape[:2], device=device, dtype=torch.float32)
        rstd = torch.empty_like(mean)
        component_gradient = torch.zeros(
            (int(batch_size), 6, SUPPORTED_WIDTH),
            device=device,
            dtype=torch.bfloat16,
        )
        fused_bf16_layernorm_modulation_hidden_backward(
            hidden,
            grad_output,
            table,
            4,
            component,
            hidden_gradient,
            mean,
            rstd,
            eps=float(eps),
        )
        fused_reduced_component_gradient(
            hidden,
            grad_output,
            mean,
            rstd,
            component_gradient,
            4,
            3,
        )
        del hidden, grad_output, table, component, hidden_gradient
        del mean, rstd, component_gradient
    torch.cuda.synchronize(device)
    return {
        "scope": "SETUP_ONLY_BEFORE_FORMAL_TIMER",
        "formal_timer_includes_warmup": False,
        "shapes": [
            [int(batch), int(tokens), SUPPORTED_WIDTH] for batch in batch_sizes
        ],
        "row_block": ROW_BLOCK,
        "reduce_feature_block": REDUCE_FEATURE_BLOCK,
        "reduce_token_block": REDUCE_TOKEN_BLOCK,
        "torch_version": str(torch.__version__),
        "triton_version": str(triton.__version__),
    }


__all__ = [
    "fused_bf16_layernorm_modulation_hidden_backward",
    "fused_reduced_component_gradient",
    "fused_tensor4d_component_gradient",
    "warmup",
]
