"""C27 fused frozen affine LayerNorm for Helios cross-attention tiles.

The production ``_ChunkedFrozenNorm`` path first materializes a full FP32
input, runs the patched FP32-output LayerNorm, casts the result back to BF16,
and copies it into a preallocated tile.  Its backward reconstructs that graph
with native autograd.  This module preserves the row-wise FP32 arithmetic and
frozen affine parameters while reading and writing BF16 directly.

It remains a standalone canary implementation until the locked-A100 parity
and latency gate passes.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
import triton
import triton.language as tl


SUPPORTED_WIDTH = 5120
ROW_BLOCK = 8192
NUM_WARPS = 8


@triton.jit
def _frozen_affine_layernorm_forward_kernel(
    hidden_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    hidden_stride_b,
    hidden_stride_t,
    output_stride_b,
    output_stride_t,
    tokens_per_batch,
    EPS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    feature = tl.arange(0, BLOCK)
    mask = feature < WIDTH
    batch = row // tokens_per_batch
    token = row - batch * tokens_per_batch
    hidden_offsets = batch * hidden_stride_b + token * hidden_stride_t + feature

    # This expression deliberately mirrors Helios' patched layernorm_forward:
    # FP32 load, masked mean/variance, affine FP32 output, then BF16 store.
    hidden = tl.load(hidden_ptr + hidden_offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    mean = tl.sum(hidden, axis=0) / WIDTH
    centered = tl.where(mask, hidden - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / WIDTH
    rstd = tl.math.rsqrt(variance + EPS)
    weight = tl.load(weight_ptr + feature, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + feature, mask=mask, other=0.0).to(tl.float32)
    output = (centered * rstd) * weight + bias

    output_offsets = batch * output_stride_b + token * output_stride_t + feature
    tl.store(output_ptr + output_offsets, output, mask=mask)


@triton.jit
def _frozen_affine_layernorm_backward_kernel(
    hidden_ptr,
    grad_output_ptr,
    weight_ptr,
    grad_hidden_ptr,
    hidden_stride_b,
    hidden_stride_t,
    grad_output_stride_b,
    grad_output_stride_t,
    grad_hidden_stride_b,
    grad_hidden_stride_t,
    tokens_per_batch,
    EPS: tl.constexpr,
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
    mean = tl.sum(hidden, axis=0) / WIDTH
    centered = tl.where(mask, hidden - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / WIDTH
    rstd = tl.math.rsqrt(variance + EPS)
    normalized = centered * rstd

    grad_offsets = (
        batch * grad_output_stride_b + token * grad_output_stride_t + feature
    )
    grad_output = tl.load(
        grad_output_ptr + grad_offsets, mask=mask, other=0.0
    ).to(tl.float32)
    weight = tl.load(weight_ptr + feature, mask=mask, other=0.0).to(tl.float32)
    grad_affine = grad_output * weight
    grad_sum = tl.sum(grad_affine, axis=0)
    grad_norm_sum = tl.sum(grad_affine * normalized, axis=0)
    grad_hidden = (
        grad_affine
        - grad_sum / WIDTH
        - normalized * (grad_norm_sum / WIDTH)
    ) * rstd

    grad_hidden_offsets = (
        batch * grad_hidden_stride_b + token * grad_hidden_stride_t + feature
    )
    tl.store(grad_hidden_ptr + grad_hidden_offsets, grad_hidden, mask=mask)


def _validate(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> None:
    values = (hidden, weight, bias)
    if not all(value.is_cuda for value in values):
        raise RuntimeError("C27 frozen cross norm requires CUDA tensors")
    if hidden.dtype is not torch.bfloat16:
        raise RuntimeError(f"C27 hidden dtype changed: {hidden.dtype}")
    if weight.dtype is not torch.float32 or bias.dtype is not torch.float32:
        raise RuntimeError(
            f"C27 frozen affine dtype changed: {weight.dtype}/{bias.dtype}"
        )
    if hidden.ndim != 3 or int(hidden.shape[0]) not in (1, 3):
        raise RuntimeError(f"C27 requires registered B1/B3 layouts: {hidden.shape}")
    if int(hidden.shape[2]) != SUPPORTED_WIDTH:
        raise RuntimeError(f"C27 width changed: {hidden.shape[2]}")
    if tuple(weight.shape) != (SUPPORTED_WIDTH,) or tuple(bias.shape) != (
        SUPPORTED_WIDTH,
    ):
        raise RuntimeError("C27 affine parameter shape changed")
    if hidden.stride(2) != 1 or hidden.stride(1) != SUPPORTED_WIDTH:
        raise RuntimeError(f"unsupported C27 hidden stride: {hidden.stride()}")
    if weight.stride(0) != 1 or bias.stride(0) != 1:
        raise RuntimeError("C27 affine parameters must be contiguous")
    if weight.requires_grad or bias.requires_grad:
        raise RuntimeError("C27 is only valid for frozen affine parameters")


def _forward(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    _validate(hidden, weight, bias)
    if float(eps) <= 0:
        raise RuntimeError("C27 epsilon must be positive")
    output = torch.empty_like(hidden)
    rows = int(hidden.shape[0] * hidden.shape[1])
    if rows:
        _frozen_affine_layernorm_forward_kernel[(rows,)](
            hidden,
            weight,
            bias,
            output,
            int(hidden.stride(0)),
            int(hidden.stride(1)),
            int(output.stride(0)),
            int(output.stride(1)),
            int(hidden.shape[1]),
            EPS=float(eps),
            WIDTH=SUPPORTED_WIDTH,
            BLOCK=ROW_BLOCK,
            num_warps=NUM_WARPS,
        )
    return output


def _backward(
    hidden: torch.Tensor,
    grad_output: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    if grad_output.shape != hidden.shape or grad_output.dtype is not torch.bfloat16:
        raise RuntimeError("C27 grad-output contract changed")
    if grad_output.stride(2) != 1 or grad_output.stride(1) != SUPPORTED_WIDTH:
        grad_output = grad_output.contiguous()
    grad_hidden = torch.empty_like(hidden)
    rows = int(hidden.shape[0] * hidden.shape[1])
    if rows:
        _frozen_affine_layernorm_backward_kernel[(rows,)](
            hidden,
            grad_output,
            weight,
            grad_hidden,
            int(hidden.stride(0)),
            int(hidden.stride(1)),
            int(grad_output.stride(0)),
            int(grad_output.stride(1)),
            int(grad_hidden.stride(0)),
            int(grad_hidden.stride(1)),
            int(hidden.shape[1]),
            EPS=float(eps),
            WIDTH=SUPPORTED_WIDTH,
            BLOCK=ROW_BLOCK,
            num_warps=NUM_WARPS,
        )
    return grad_hidden


class _FusedFrozenAffineLayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        output = _forward(hidden, weight, bias, eps=float(eps))
        ctx.eps = float(eps)
        ctx.save_for_backward(hidden, weight)
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):
        hidden, weight = ctx.saved_tensors
        grad_hidden = (
            _backward(hidden, grad_output, weight, eps=ctx.eps)
            if ctx.needs_input_grad[0]
            else None
        )
        return grad_hidden, None, None, None


def fused_frozen_affine_layernorm(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    """Apply the C27 frozen affine LayerNorm with hidden-only autograd."""

    return _FusedFrozenAffineLayerNorm.apply(hidden, weight, bias, float(eps))


def runtime_identity() -> dict[str, Any]:
    source = Path(__file__).resolve()
    return {
        "schema_version": "distillschedule.helios_frozen_cross_norm_kernel.v1",
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "width": SUPPORTED_WIDTH,
        "row_block": ROW_BLOCK,
        "num_warps": NUM_WARPS,
        "torch_version": str(torch.__version__),
        "triton_version": str(triton.__version__),
    }


__all__ = [
    "SUPPORTED_WIDTH",
    "fused_frozen_affine_layernorm",
    "runtime_identity",
]
