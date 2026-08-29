"""Exact Triton fusion for one Helios gated-residual tile.

The supported training source computes each 512-token tile as five eager CUDA
kernels: two BF16->FP32 casts, multiply, add, and BF16 copy.  This kernel keeps
the two explicit IEEE FP32 round-to-nearest operations and performs one BF16
store.  It is imported lazily only when the default-off candidate is enabled.

The backward kernel implements the exact frozen-gate derivative used by the
pinned source: ``grad_residual = BF16(grad_output.float() * gate)``.  It does
not replace any path where the modulation table or timestep inputs are
trainable.
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
def _helios_gated_residual_kernel(
    hidden_ptr,
    residual_ptr,
    gate_ptr,
    output_ptr,
    n_elements,
    tokens_per_batch,
    hidden_stride_b,
    hidden_stride_t,
    residual_stride_b,
    residual_stride_t,
    gate_stride_b,
    gate_stride_t,
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
    hidden_offsets = (
        batch_index * hidden_stride_b + token_index * hidden_stride_t + feature_index
    )
    residual_offsets = (
        batch_index * residual_stride_b
        + token_index * residual_stride_t
        + feature_index
    )
    gate_offsets = (
        batch_index * gate_stride_b + token_index * gate_stride_t + feature_index
    )
    output_offsets = (
        batch_index * output_stride_b
        + token_index * output_stride_t
        + feature_index
    )
    hidden = tl.load(hidden_ptr + hidden_offsets, mask=mask).to(tl.float32)
    residual = tl.load(residual_ptr + residual_offsets, mask=mask).to(tl.float32)
    gate = tl.load(gate_ptr + gate_offsets, mask=mask).to(tl.float32)
    product = tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [residual, gate],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    output = tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [hidden, product],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    tl.store(output_ptr + output_offsets, output, mask=mask)


@triton.jit
def _helios_gated_residual_backward_kernel(
    grad_output_ptr,
    gate_ptr,
    grad_residual_ptr,
    n_elements,
    tokens_per_batch,
    grad_stride_b,
    grad_stride_t,
    gate_stride_b,
    gate_stride_t,
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
    grad_offsets = (
        batch_index * grad_stride_b + token_index * grad_stride_t + feature_index
    )
    gate_offsets = (
        batch_index * gate_stride_b + token_index * gate_stride_t + feature_index
    )
    output_offsets = (
        batch_index * output_stride_b
        + token_index * output_stride_t
        + feature_index
    )
    grad_output = tl.load(grad_output_ptr + grad_offsets, mask=mask).to(tl.float32)
    gate = tl.load(gate_ptr + gate_offsets, mask=mask).to(tl.float32)
    grad_residual = tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [grad_output, gate],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    tl.store(grad_residual_ptr + output_offsets, grad_residual, mask=mask)


@triton.jit
def _helios_gated_residual_trainable_temb_backward_kernel(
    grad_output_ptr,
    residual_ptr,
    gate_ptr,
    grad_residual_ptr,
    temb_product_ptr,
    n_elements,
    tokens_per_batch,
    grad_stride_b,
    grad_stride_t,
    residual_stride_b,
    residual_stride_t,
    gate_stride_b,
    gate_stride_t,
    grad_residual_stride_b,
    grad_residual_stride_t,
    temb_product_stride_b,
    temb_product_stride_t,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Emit both exact BF16 backward products from one set of input loads."""

    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    global_token_index = offsets // WIDTH
    feature_index = offsets - global_token_index * WIDTH
    batch_index = global_token_index // tokens_per_batch
    token_index = global_token_index - batch_index * tokens_per_batch
    grad_offsets = (
        batch_index * grad_stride_b + token_index * grad_stride_t + feature_index
    )
    residual_offsets = (
        batch_index * residual_stride_b
        + token_index * residual_stride_t
        + feature_index
    )
    gate_offsets = (
        batch_index * gate_stride_b + token_index * gate_stride_t + feature_index
    )
    grad_residual_offsets = (
        batch_index * grad_residual_stride_b
        + token_index * grad_residual_stride_t
        + feature_index
    )
    temb_product_offsets = (
        batch_index * temb_product_stride_b
        + token_index * temb_product_stride_t
        + feature_index
    )
    grad_output = tl.load(grad_output_ptr + grad_offsets, mask=mask).to(tl.float32)
    residual = tl.load(residual_ptr + residual_offsets, mask=mask).to(tl.float32)
    gate = tl.load(gate_ptr + gate_offsets, mask=mask).to(tl.float32)
    grad_residual = tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [grad_output, gate],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    temb_product = tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [residual, grad_output],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    tl.store(grad_residual_ptr + grad_residual_offsets, grad_residual, mask=mask)
    tl.store(temb_product_ptr + temb_product_offsets, temb_product, mask=mask)


def fused_gated_residual_chunk(
    hidden: torch.Tensor,
    residual: torch.Tensor,
    gate: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    """Write one registered B1/B3 tile with exact batch-aware gate addressing."""

    if not all(value.is_cuda for value in (hidden, residual, gate, output)):
        raise RuntimeError("Helios residual fusion requires CUDA tensors")
    if hidden.dtype is not torch.bfloat16 or residual.dtype is not torch.bfloat16:
        raise RuntimeError("Helios residual fusion requires BF16 hidden/residual")
    if gate.dtype is not torch.float32 or output.dtype is not torch.bfloat16:
        raise RuntimeError("Helios residual fusion requires FP32 gate and BF16 output")
    if hidden.ndim != 3 or int(hidden.shape[0]) not in (1, 3):
        raise RuntimeError("Helios residual fusion is registered only for B1/B3 tiles")
    if hidden.shape != residual.shape or hidden.shape != gate.shape or hidden.shape != output.shape:
        raise RuntimeError("Helios residual fusion shape mismatch")
    if int(hidden.shape[2]) != SUPPORTED_WIDTH:
        raise RuntimeError(f"Helios residual width changed: {hidden.shape[2]}")
    for name, value in (("hidden", hidden), ("residual", residual), ("output", output)):
        if value.stride(2) != 1 or value.stride(1) != SUPPORTED_WIDTH:
            raise RuntimeError(f"unsupported Helios {name} strides: {value.stride()}")
    if gate.stride(2) != 1:
        raise RuntimeError(f"unsupported Helios gate strides: {gate.stride()}")
    if gate.stride(1) not in (0, SUPPORTED_WIDTH):
        raise RuntimeError(f"unsupported Helios gate token stride: {gate.stride(1)}")
    if gate.stride(0) < 0:
        raise RuntimeError(f"unsupported Helios gate batch stride: {gate.stride(0)}")

    n_elements = int(output.numel())
    if n_elements == 0:
        return output
    _helios_gated_residual_kernel[(triton.cdiv(n_elements, BLOCK_SIZE),)](
        hidden,
        residual,
        gate,
        output,
        n_elements,
        int(output.shape[1]),
        int(hidden.stride(0)),
        int(hidden.stride(1)),
        int(residual.stride(0)),
        int(residual.stride(1)),
        int(gate.stride(0)),
        int(gate.stride(1)),
        int(output.stride(0)),
        int(output.stride(1)),
        WIDTH=SUPPORTED_WIDTH,
        BLOCK=BLOCK_SIZE,
        num_warps=NUM_WARPS,
    )
    return output


def fused_gated_residual_backward_chunk(
    grad_output: torch.Tensor,
    gate: torch.Tensor,
    grad_residual: torch.Tensor,
) -> torch.Tensor:
    """Write the frozen-gate residual gradient for one registered B1/B3 tile."""

    if not all(value.is_cuda for value in (grad_output, gate, grad_residual)):
        raise RuntimeError("Helios residual backward fusion requires CUDA tensors")
    if grad_output.dtype is not torch.bfloat16:
        raise RuntimeError("Helios residual backward fusion requires BF16 grad_output")
    if gate.dtype is not torch.float32 or grad_residual.dtype is not torch.bfloat16:
        raise RuntimeError(
            "Helios residual backward fusion requires FP32 gate and BF16 output"
        )
    if grad_output.ndim != 3 or int(grad_output.shape[0]) not in (1, 3):
        raise RuntimeError("Helios residual backward fusion is registered only for B1/B3")
    if grad_output.shape != gate.shape or grad_output.shape != grad_residual.shape:
        raise RuntimeError("Helios residual backward fusion shape mismatch")
    if int(grad_output.shape[2]) != SUPPORTED_WIDTH:
        raise RuntimeError(f"Helios residual width changed: {grad_output.shape[2]}")
    if grad_output.stride(2) != 1 or grad_residual.stride(2) != 1:
        raise RuntimeError(
            "Helios residual backward fusion requires unit feature strides"
        )
    if grad_output.stride(1) != SUPPORTED_WIDTH:
        raise RuntimeError(
            f"unsupported Helios grad_output token stride: {grad_output.stride()}"
        )
    if grad_residual.stride(1) != SUPPORTED_WIDTH:
        raise RuntimeError(
            f"unsupported Helios grad_residual token stride: {grad_residual.stride()}"
        )
    if gate.stride(2) != 1 or gate.stride(1) not in (0, SUPPORTED_WIDTH):
        raise RuntimeError(f"unsupported Helios gate strides: {gate.stride()}")
    if any(value.stride(0) < 0 for value in (grad_output, gate, grad_residual)):
        raise RuntimeError("unsupported negative batch stride in residual backward")

    n_elements = int(grad_residual.numel())
    if n_elements == 0:
        return grad_residual
    _helios_gated_residual_backward_kernel[
        (triton.cdiv(n_elements, BLOCK_SIZE),)
    ](
        grad_output,
        gate,
        grad_residual,
        n_elements,
        int(grad_output.shape[1]),
        int(grad_output.stride(0)),
        int(grad_output.stride(1)),
        int(gate.stride(0)),
        int(gate.stride(1)),
        int(grad_residual.stride(0)),
        int(grad_residual.stride(1)),
        WIDTH=SUPPORTED_WIDTH,
        BLOCK=BLOCK_SIZE,
        num_warps=NUM_WARPS,
    )
    return grad_residual


def fused_gated_residual_trainable_temb_backward_chunk(
    grad_output: torch.Tensor,
    residual: torch.Tensor,
    gate: torch.Tensor,
    grad_residual: torch.Tensor,
    temb_product: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Write exact residual and BF16-before-reduction timestep products."""

    values = (grad_output, residual, gate, grad_residual, temb_product)
    if not all(value.is_cuda for value in values):
        raise RuntimeError("Helios trainable-temb backward fusion requires CUDA tensors")
    if grad_output.dtype is not torch.bfloat16 or residual.dtype is not torch.bfloat16:
        raise RuntimeError("Helios trainable-temb backward fusion requires BF16 inputs")
    if gate.dtype is not torch.float32:
        raise RuntimeError("Helios trainable-temb backward fusion requires an FP32 gate")
    if grad_residual.dtype is not torch.bfloat16 or temb_product.dtype is not torch.bfloat16:
        raise RuntimeError("Helios trainable-temb backward outputs must be BF16")
    if grad_output.ndim != 3 or int(grad_output.shape[0]) not in (1, 3):
        raise RuntimeError("Helios trainable-temb backward fusion requires B1/B3")
    if any(value.shape != grad_output.shape for value in values[1:]):
        raise RuntimeError("Helios trainable-temb backward fusion shape mismatch")
    if int(grad_output.shape[2]) != SUPPORTED_WIDTH:
        raise RuntimeError(f"Helios residual width changed: {grad_output.shape[2]}")
    for name, value in (
        ("grad_output", grad_output),
        ("residual", residual),
        ("grad_residual", grad_residual),
        ("temb_product", temb_product),
    ):
        if value.stride(2) != 1 or value.stride(1) != SUPPORTED_WIDTH:
            raise RuntimeError(f"unsupported Helios {name} strides: {value.stride()}")
    if gate.stride(2) != 1 or gate.stride(1) not in (0, SUPPORTED_WIDTH):
        raise RuntimeError(f"unsupported Helios gate strides: {gate.stride()}")
    if any(value.stride(0) < 0 for value in values):
        raise RuntimeError("unsupported negative batch stride in trainable-temb backward")

    n_elements = int(grad_output.numel())
    if n_elements == 0:
        return grad_residual, temb_product
    _helios_gated_residual_trainable_temb_backward_kernel[
        (triton.cdiv(n_elements, BLOCK_SIZE),)
    ](
        grad_output,
        residual,
        gate,
        grad_residual,
        temb_product,
        n_elements,
        int(grad_output.shape[1]),
        int(grad_output.stride(0)),
        int(grad_output.stride(1)),
        int(residual.stride(0)),
        int(residual.stride(1)),
        int(gate.stride(0)),
        int(gate.stride(1)),
        int(grad_residual.stride(0)),
        int(grad_residual.stride(1)),
        int(temb_product.stride(0)),
        int(temb_product.stride(1)),
        WIDTH=SUPPORTED_WIDTH,
        BLOCK=BLOCK_SIZE,
        num_warps=NUM_WARPS,
    )
    return grad_residual, temb_product


def warmup(
    device: torch.device | str,
    *,
    include_backward: bool = False,
    batch_sizes: tuple[int, ...] = (1, 3),
) -> dict[str, Any]:
    """Compile registered B1/B3 kernel specializations outside the formal timer."""

    for batch_size in batch_sizes:
        if int(batch_size) not in (1, 3):
            raise ValueError(f"unsupported Helios residual warmup batch: {batch_size}")
        hidden = torch.zeros(
            (int(batch_size), 512, SUPPORTED_WIDTH),
            device=device,
            dtype=torch.bfloat16,
        )
        residual = torch.zeros_like(hidden)
        gate = torch.zeros(
            (int(batch_size), 1, SUPPORTED_WIDTH),
            device=device,
            dtype=torch.float32,
        ).expand_as(hidden)
        output = torch.empty_like(hidden)
        fused_gated_residual_chunk(hidden, residual, gate, output)
        if include_backward:
            grad_output = torch.zeros_like(hidden)
            grad_residual = torch.empty_like(hidden)
            fused_gated_residual_backward_chunk(grad_output, gate, grad_residual)
            temb_product = torch.empty_like(hidden)
            fused_gated_residual_trainable_temb_backward_chunk(
                grad_output,
                residual,
                gate,
                grad_residual,
                temb_product,
            )
        del hidden, residual, gate, output
        if include_backward:
            del grad_output, grad_residual, temb_product
    torch.cuda.synchronize(device)
    return {
        "scope": "SETUP_ONLY_BEFORE_FORMAL_TIMER",
        "formal_timer_includes_warmup": False,
        "shapes": [[int(batch), 512, SUPPORTED_WIDTH] for batch in batch_sizes],
        "block_size": BLOCK_SIZE,
        "num_warps": NUM_WARPS,
        "backward_kernel": bool(include_backward),
        "torch_version": str(torch.__version__),
        "triton_version": str(triton.__version__),
    }


__all__ = [
    "fused_gated_residual_backward_chunk",
    "fused_gated_residual_chunk",
    "fused_gated_residual_trainable_temb_backward_chunk",
    "warmup",
]
