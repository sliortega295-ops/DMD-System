"""Fused BF16 SwiGLU activation kernels for the pinned MiniMax-H3 workload.

The projection GEMM remains the ordinary Diffusers/PEFT module.  These kernels
only replace ``value * silu(gate)`` and its input-gradient calculation, so the
candidate cannot alter LoRA parameter placement or GEMM selection.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _swiglu_forward_kernel(
    projected,
    output,
    n_outputs,
    inner_size: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < n_outputs
    rows = offsets // inner_size
    columns = offsets - rows * inner_size
    projected_base = rows * (2 * inner_size) + columns

    values = tl.load(projected + projected_base, mask=mask, other=0.0).to(tl.float32)
    gates = tl.load(projected + projected_base + inner_size, mask=mask, other=0.0).to(
        tl.float32
    )
    # Match the eager kernel boundaries: SiLU first produces BF16, then the
    # following multiplication produces a second BF16-rounded result.
    sigmoid = tl.sigmoid(gates)
    activated = (gates * sigmoid).to(tl.bfloat16)
    result = (values * activated.to(tl.float32)).to(tl.bfloat16)
    tl.store(output + offsets, result, mask=mask)


@triton.jit
def _swiglu_backward_kernel(
    grad_output,
    projected,
    grad_projected,
    n_outputs,
    inner_size: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < n_outputs
    rows = offsets // inner_size
    columns = offsets - rows * inner_size
    projected_base = rows * (2 * inner_size) + columns

    gradients = tl.load(grad_output + offsets, mask=mask, other=0.0).to(tl.float32)
    values = tl.load(projected + projected_base, mask=mask, other=0.0).to(tl.float32)
    gates = tl.load(projected + projected_base + inner_size, mask=mask, other=0.0).to(
        tl.float32
    )
    sigmoid = tl.sigmoid(gates)
    activated = (gates * sigmoid).to(tl.bfloat16)

    grad_value = (gradients * activated.to(tl.float32)).to(tl.bfloat16)
    # Eager autograd rounds the multiply feeding SiLUBackward to BF16 before
    # evaluating the derivative.
    grad_activation = (gradients * values).to(tl.bfloat16)
    derivative = sigmoid * (1.0 + gates * (1.0 - sigmoid))
    grad_gate = (grad_activation.to(tl.float32) * derivative).to(tl.bfloat16)

    tl.store(grad_projected + projected_base, grad_value, mask=mask)
    tl.store(grad_projected + projected_base + inner_size, grad_gate, mask=mask)


def _validate(projected: torch.Tensor) -> tuple[int, int]:
    if not projected.is_cuda:
        raise RuntimeError("H3 fused SwiGLU requires a CUDA tensor")
    if projected.dtype != torch.bfloat16:
        raise RuntimeError(
            f"H3 fused SwiGLU requires BF16 projected activations, got {projected.dtype}"
        )
    if projected.ndim < 2 or projected.shape[-1] % 2:
        raise RuntimeError(
            "H3 fused SwiGLU requires projected=[...,2*inner], got "
            f"{tuple(projected.shape)}"
        )
    if not projected.is_contiguous():
        raise RuntimeError("H3 fused SwiGLU requires contiguous projected activations")
    major, minor = torch.cuda.get_device_capability(projected.device)
    if (major, minor) < (8, 0):
        raise RuntimeError(f"H3 fused SwiGLU requires SM80+, got sm_{major}{minor}")
    inner_size = int(projected.shape[-1] // 2)
    n_outputs = int(projected.numel() // 2)
    return n_outputs, inner_size


def fused_swiglu(projected: torch.Tensor) -> torch.Tensor:
    """Return the fused activation result while leaving the projection intact."""

    n_outputs, inner_size = _validate(projected)
    output = torch.empty((*projected.shape[:-1], inner_size), device=projected.device, dtype=projected.dtype)
    block_size = 1024
    _swiglu_forward_kernel[(triton.cdiv(n_outputs, block_size),)](
        projected,
        output,
        n_outputs,
        inner_size=inner_size,
        block_size=block_size,
        num_warps=8,
    )
    return output


def fused_swiglu_backward(
    grad_output: torch.Tensor, projected: torch.Tensor
) -> torch.Tensor:
    """Return the fused gradient with respect to the 2-way projection output."""

    n_outputs, inner_size = _validate(projected)
    expected_shape = (*projected.shape[:-1], inner_size)
    if (
        not grad_output.is_cuda
        or grad_output.dtype != projected.dtype
        or tuple(grad_output.shape) != expected_shape
        or not grad_output.is_contiguous()
    ):
        raise RuntimeError(
            "H3 fused SwiGLU backward gradient mismatch: "
            f"observed={tuple(grad_output.shape)}/{grad_output.dtype}/"
            f"cuda={grad_output.is_cuda}/contiguous={grad_output.is_contiguous()} "
            f"expected={expected_shape}/{projected.dtype}"
        )
    grad_projected = torch.empty_like(projected)
    block_size = 1024
    _swiglu_backward_kernel[(triton.cdiv(n_outputs, block_size),)](
        grad_output,
        projected,
        grad_projected,
        n_outputs,
        inner_size=inner_size,
        block_size=block_size,
        num_warps=8,
    )
    return grad_projected
