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
    tokens_per_batch,
    normalized_stride_b,
    normalized_stride_t,
    scale_stride_b,
    scale_stride_t,
    shift_stride_b,
    shift_stride_t,
    output_stride_b,
    output_stride_t,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    global_token_index = offsets // WIDTH
    feature_index = offsets - global_token_index * WIDTH
    batch_index = global_token_index // tokens_per_batch
    token_index = global_token_index - batch_index * tokens_per_batch
    normalized_offsets = (
        batch_index * normalized_stride_b
        + token_index * normalized_stride_t
        + feature_index
    )
    scale_offsets = (
        batch_index * scale_stride_b + token_index * scale_stride_t + feature_index
    )
    shift_offsets = (
        batch_index * shift_stride_b + token_index * shift_stride_t + feature_index
    )
    output_offsets = (
        batch_index * output_stride_b
        + token_index * output_stride_t
        + feature_index
    )

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


@triton.jit
def _helios_modulated_norm_materialization_kernel(
    normalized_ptr,
    table_ptr,
    scale_component_ptr,
    shift_component_ptr,
    output_ptr,
    n_elements,
    tokens_per_batch,
    normalized_stride_b,
    normalized_stride_t,
    table_stride_c,
    scale_component_stride_b,
    scale_component_stride_t,
    shift_component_stride_b,
    shift_component_stride_t,
    output_stride_b,
    output_stride_t,
    SCALE_INDEX: tl.constexpr,
    SHIFT_INDEX: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    global_token_index = offsets // WIDTH
    feature_index = offsets - global_token_index * WIDTH
    batch_index = global_token_index // tokens_per_batch
    token_index = global_token_index - batch_index * tokens_per_batch
    normalized = tl.load(
        normalized_ptr
        + batch_index * normalized_stride_b
        + token_index * normalized_stride_t
        + feature_index,
        mask=mask,
    ).to(tl.float32)
    scale_table = tl.load(
        table_ptr + SCALE_INDEX * table_stride_c + feature_index,
        mask=mask,
    ).to(tl.float32)
    shift_table = tl.load(
        table_ptr + SHIFT_INDEX * table_stride_c + feature_index,
        mask=mask,
    ).to(tl.float32)
    scale_component = tl.load(
        scale_component_ptr
        + batch_index * scale_component_stride_b
        + token_index * scale_component_stride_t
        + feature_index,
        mask=mask,
    ).to(tl.float32)
    shift_component = tl.load(
        shift_component_ptr
        + batch_index * shift_component_stride_b
        + token_index * shift_component_stride_t
        + feature_index,
        mask=mask,
    ).to(tl.float32)
    scale = tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [scale_table, scale_component],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    shift = tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [shift_table, shift_component],
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
    tl.store(
        output_ptr
        + batch_index * output_stride_b
        + token_index * output_stride_t
        + feature_index,
        value,
        mask=mask,
    )


def fused_modulated_norm_epilogue(
    normalized: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    """Write one registered B1/B3 Helios tile with batch-aware modulation."""

    values = (normalized, scale, shift, output)
    if not all(value.is_cuda for value in values):
        raise RuntimeError("Helios modulated-norm fusion requires CUDA tensors")
    if normalized.dtype is not torch.float32:
        raise RuntimeError("Helios modulated-norm input must be FP32")
    if scale.dtype is not torch.float32 or shift.dtype is not torch.float32:
        raise RuntimeError("Helios modulated-norm scale/shift must be FP32")
    if output.dtype is not torch.bfloat16:
        raise RuntimeError("Helios modulated-norm output must be BF16")
    if normalized.ndim != 3 or int(normalized.shape[0]) not in (1, 3):
        raise RuntimeError("Helios modulated-norm fusion requires B1/B3")
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
    if any(value.stride(0) < 0 for value in values):
        raise RuntimeError("unsupported negative batch stride in modulated norm")

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
        int(normalized.shape[1]),
        int(normalized.stride(0)),
        int(normalized.stride(1)),
        int(scale.stride(0)),
        int(scale.stride(1)),
        int(shift.stride(0)),
        int(shift.stride(1)),
        int(output.stride(0)),
        int(output.stride(1)),
        WIDTH=SUPPORTED_WIDTH,
        BLOCK=BLOCK_SIZE,
        num_warps=NUM_WARPS,
    )
    return output


def fused_modulated_norm_materialization(
    normalized: torch.Tensor,
    scale_shift_table: torch.Tensor,
    scale_index: int,
    shift_index: int,
    scale_component: torch.Tensor,
    shift_component: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    """Fuse source table additions into the exact registered B1/B3 epilogue."""

    values = (
        normalized,
        scale_shift_table,
        scale_component,
        shift_component,
        output,
    )
    if not all(value.is_cuda for value in values):
        raise RuntimeError("Helios modulation materialization requires CUDA tensors")
    if normalized.dtype is not torch.float32:
        raise RuntimeError("Helios modulated-norm input must be FP32")
    if scale_shift_table.dtype not in (torch.bfloat16, torch.float32):
        raise RuntimeError(
            "unsupported Helios scale-shift table dtype: "
            f"{scale_shift_table.dtype}"
        )
    if scale_component.dtype not in (torch.bfloat16, torch.float32):
        raise RuntimeError("unsupported Helios scale-component dtype")
    if shift_component.dtype not in (torch.bfloat16, torch.float32):
        raise RuntimeError("unsupported Helios shift-component dtype")
    if output.dtype is not torch.bfloat16:
        raise RuntimeError("Helios modulated-norm output must be BF16")
    if normalized.ndim != 3 or int(normalized.shape[0]) not in (1, 3):
        raise RuntimeError("Helios modulation materialization requires B1/B3")
    if scale_shift_table.ndim != 3 or tuple(scale_shift_table.shape[:2]) != (1, 6):
        raise RuntimeError("Helios scale-shift table layout changed")
    if int(normalized.shape[2]) != SUPPORTED_WIDTH:
        raise RuntimeError(f"Helios modulation width changed: {normalized.shape[2]}")
    if int(scale_shift_table.shape[2]) != SUPPORTED_WIDTH:
        raise RuntimeError("Helios scale-shift table width changed")
    if not 0 <= int(scale_index) < 6 or not 0 <= int(shift_index) < 6:
        raise RuntimeError("Helios modulation component index changed")
    for name, value in (
        ("scale_component", scale_component),
        ("shift_component", shift_component),
    ):
        if value.shape != normalized.shape:
            raise RuntimeError(f"Helios {name} shape changed")
        if value.stride(2) != 1:
            raise RuntimeError(f"unsupported Helios {name} stride: {value.stride()}")
    for name, value in (("normalized", normalized), ("output", output)):
        if value.stride(2) != 1 or value.stride(1) != SUPPORTED_WIDTH:
            raise RuntimeError(f"unsupported Helios {name} stride: {value.stride()}")
    if scale_shift_table.stride(2) != 1:
        raise RuntimeError("unsupported Helios scale-shift table stride")
    if any(value.stride(0) < 0 for value in values):
        raise RuntimeError("unsupported negative batch stride in modulation materialization")

    n_elements = int(output.numel())
    if n_elements == 0:
        return output
    _helios_modulated_norm_materialization_kernel[
        (triton.cdiv(n_elements, BLOCK_SIZE),)
    ](
        normalized,
        scale_shift_table,
        scale_component,
        shift_component,
        output,
        n_elements,
        int(normalized.shape[1]),
        int(normalized.stride(0)),
        int(normalized.stride(1)),
        int(scale_shift_table.stride(1)),
        int(scale_component.stride(0)),
        int(scale_component.stride(1)),
        int(shift_component.stride(0)),
        int(shift_component.stride(1)),
        int(output.stride(0)),
        int(output.stride(1)),
        SCALE_INDEX=int(scale_index),
        SHIFT_INDEX=int(shift_index),
        WIDTH=SUPPORTED_WIDTH,
        BLOCK=BLOCK_SIZE,
        num_warps=NUM_WARPS,
    )
    return output


def warmup(
    device: torch.device | str,
    *,
    fuse_materialization: bool = False,
    batch_sizes: tuple[int, ...] = (1, 3),
    chunk_tokens: int = 512,
) -> dict[str, Any]:
    """Compile registered B1/B3 specializations outside the formal timer."""

    if int(chunk_tokens) <= 0:
        raise ValueError("Helios modulated-norm warmup chunk size must be positive")
    for batch_size in batch_sizes:
        if int(batch_size) not in (1, 3):
            raise ValueError(f"unsupported Helios modulated-norm warmup batch: {batch_size}")
        normalized = torch.zeros(
            (int(batch_size), int(chunk_tokens), SUPPORTED_WIDTH),
            device=device,
            dtype=torch.float32,
        )
        output = torch.empty(
            (int(batch_size), int(chunk_tokens), SUPPORTED_WIDTH),
            device=device,
            dtype=torch.bfloat16,
        )
        if fuse_materialization:
            # The live FSDP mixed-precision policy unshards this frozen parameter
            # as BF16.  Its leading dimension stays one and broadcasts across B3.
            table = torch.zeros(
                (1, 6, SUPPORTED_WIDTH), device=device, dtype=torch.bfloat16
            )
            component = torch.zeros(
                (int(batch_size), 1, SUPPORTED_WIDTH),
                device=device,
                dtype=torch.bfloat16,
            ).expand_as(normalized)
            fused_modulated_norm_materialization(
                normalized, table, 1, 2, component, component, output
            )
        else:
            scale = torch.zeros(
                (int(batch_size), 1, SUPPORTED_WIDTH),
                device=device,
                dtype=torch.float32,
            ).expand_as(normalized)
            shift = torch.zeros_like(scale)
            fused_modulated_norm_epilogue(normalized, scale, shift, output)
        del normalized, output
        if fuse_materialization:
            del table, component
        else:
            del scale, shift
    torch.cuda.synchronize(device)
    return {
        "scope": "SETUP_ONLY_BEFORE_FORMAL_TIMER",
        "formal_timer_includes_warmup": False,
        "shapes": [
            [int(batch), int(chunk_tokens), SUPPORTED_WIDTH]
            for batch in batch_sizes
        ],
        "chunk_tokens": int(chunk_tokens),
        "block_size": BLOCK_SIZE,
        "num_warps": NUM_WARPS,
        "fuse_materialization": bool(fuse_materialization),
        "materialization_table_dtype": (
            "torch.bfloat16" if fuse_materialization else None
        ),
        "torch_version": str(torch.__version__),
        "triton_version": str(triton.__version__),
    }


__all__ = [
    "fused_modulated_norm_epilogue",
    "fused_modulated_norm_materialization",
    "warmup",
]
