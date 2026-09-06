"""Bitwise BF16 pointwise kernels for MiniMax-H3 no-grad blocks.

The upstream block materializes full-sequence AdaLN tables with ``index_select``
and then launches separate BF16 pointwise kernels for every multiply and add.
For the fixed A100 workload those tensors are hundreds of MiB.  These kernels
perform the same BF16 rounding sequence while gathering the six-row AdaLN table
directly, avoiding the temporary full-sequence tensors.

This module is imported lazily only when the opt-in production flag is enabled;
the baseline path therefore does not acquire a Triton dependency at import time.
The optional custom-autograd wrapper reuses the same forward kernels and calls
the exact BF16 input-gradient kernels below.  Grid AdaLN tables are frozen, so
only activation/branch gradients are produced.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _bf16_modulate_exact(value, scale, shift):
    """Return ``value * (1 + scale) + shift`` with eager BF16 rounding.

    A100 does not expose scalar BF16 add/mul instructions.  Eager PyTorch
    converts each BF16 operand to FP32, rounds the result back to BF16 at every
    kernel boundary, then converts it to FP32 for the next operation.  The
    explicit conversion chain below preserves that order inside one kernel.
    Two BF16 lanes are packed into one 32-bit inline-assembly invocation.
    """

    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .b16 vh0,vh1,sh0,sh1,bh0,bh1,h0,h1,o0,o1;
          .reg .f32 vf0,vf1,sf0,sf1,bf0,bf1,t0,t1,p0,p1;
          mov.b32 {vh0,vh1}, $1;
          mov.b32 {sh0,sh1}, $2;
          mov.b32 {bh0,bh1}, $3;
          cvt.f32.bf16 vf0, vh0; cvt.f32.bf16 vf1, vh1;
          cvt.f32.bf16 sf0, sh0; cvt.f32.bf16 sf1, sh1;
          cvt.f32.bf16 bf0, bh0; cvt.f32.bf16 bf1, bh1;
          add.f32 t0, sf0, 1.0; add.f32 t1, sf1, 1.0;
          cvt.rn.bf16.f32 h0, t0; cvt.rn.bf16.f32 h1, t1;
          cvt.f32.bf16 t0, h0; cvt.f32.bf16 t1, h1;
          mul.f32 p0, vf0, t0; mul.f32 p1, vf1, t1;
          cvt.rn.bf16.f32 h0, p0; cvt.rn.bf16.f32 h1, p1;
          cvt.f32.bf16 p0, h0; cvt.f32.bf16 p1, h1;
          add.f32 t0, p0, bf0; add.f32 t1, p1, bf1;
          cvt.rn.bf16.f32 o0, t0; cvt.rn.bf16.f32 o1, t1;
          mov.b32 $0, {o0,o1};
        }
        """,
        constraints="=r,r,r,r",
        args=[value, scale, shift],
        dtype=tl.bfloat16,
        is_pure=True,
        pack=2,
    )


@triton.jit
def _bf16_residual_exact(residual, gate, branch):
    """Return ``residual + gate * branch`` with eager BF16 rounding."""

    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .b16 rh0,rh1,gh0,gh1,bh0,bh1,h0,h1,o0,o1;
          .reg .f32 rf0,rf1,gf0,gf1,bf0,bf1,p0,p1,t0,t1;
          mov.b32 {rh0,rh1}, $1;
          mov.b32 {gh0,gh1}, $2;
          mov.b32 {bh0,bh1}, $3;
          cvt.f32.bf16 rf0, rh0; cvt.f32.bf16 rf1, rh1;
          cvt.f32.bf16 gf0, gh0; cvt.f32.bf16 gf1, gh1;
          cvt.f32.bf16 bf0, bh0; cvt.f32.bf16 bf1, bh1;
          mul.f32 p0, gf0, bf0; mul.f32 p1, gf1, bf1;
          cvt.rn.bf16.f32 h0, p0; cvt.rn.bf16.f32 h1, p1;
          cvt.f32.bf16 p0, h0; cvt.f32.bf16 p1, h1;
          add.f32 t0, rf0, p0; add.f32 t1, rf1, p1;
          cvt.rn.bf16.f32 o0, t0; cvt.rn.bf16.f32 o1, t1;
          mov.b32 $0, {o0,o1};
        }
        """,
        constraints="=r,r,r,r",
        args=[residual, gate, branch],
        dtype=tl.bfloat16,
        is_pure=True,
        pack=2,
    )


@triton.jit
def _bf16_modulate_backward_exact(grad_output, scale):
    """Return ``grad_output * bf16(1 + scale)`` with eager BF16 rounding."""

    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .b16 gh0,gh1,sh0,sh1,h0,h1,o0,o1;
          .reg .f32 gf0,gf1,sf0,sf1,t0,t1,p0,p1;
          mov.b32 {gh0,gh1}, $1;
          mov.b32 {sh0,sh1}, $2;
          cvt.f32.bf16 gf0, gh0; cvt.f32.bf16 gf1, gh1;
          cvt.f32.bf16 sf0, sh0; cvt.f32.bf16 sf1, sh1;
          add.f32 t0, sf0, 1.0; add.f32 t1, sf1, 1.0;
          cvt.rn.bf16.f32 h0, t0; cvt.rn.bf16.f32 h1, t1;
          cvt.f32.bf16 t0, h0; cvt.f32.bf16 t1, h1;
          mul.f32 p0, gf0, t0; mul.f32 p1, gf1, t1;
          cvt.rn.bf16.f32 o0, p0; cvt.rn.bf16.f32 o1, p1;
          mov.b32 $0, {o0,o1};
        }
        """,
        constraints="=r,r,r",
        args=[grad_output, scale],
        dtype=tl.bfloat16,
        is_pure=True,
        pack=2,
    )


@triton.jit
def _bf16_mul_exact(left, right):
    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .b16 lh0,lh1,rh0,rh1,o0,o1;
          .reg .f32 lf0,lf1,rf0,rf1,p0,p1;
          mov.b32 {lh0,lh1}, $1;
          mov.b32 {rh0,rh1}, $2;
          cvt.f32.bf16 lf0, lh0; cvt.f32.bf16 lf1, lh1;
          cvt.f32.bf16 rf0, rh0; cvt.f32.bf16 rf1, rh1;
          mul.f32 p0, lf0, rf0; mul.f32 p1, lf1, rf1;
          cvt.rn.bf16.f32 o0, p0; cvt.rn.bf16.f32 o1, p1;
          mov.b32 $0, {o0,o1};
        }
        """,
        constraints="=r,r,r",
        args=[left, right],
        dtype=tl.bfloat16,
        is_pure=True,
        pack=2,
    )


@triton.jit
def _rmsnorm_modulate_kernel(
    value,
    weight,
    scale,
    shift,
    indices,
    output,
    hidden_size: tl.constexpr,
    eps: tl.constexpr,
    block_size: tl.constexpr,
):
    """Fuse one frozen RMSNorm row with the existing exact BF16 modulation.

    The RMS reduction is accumulated in FP32.  The normalized value is rounded
    to BF16 before modulation, matching the eager RMSNorm output boundary.  The
    modulation helper then preserves its two eager BF16 rounding boundaries.
    """

    row = tl.program_id(0)
    columns = tl.arange(0, block_size)
    mask = columns < hidden_size
    values = tl.load(value + row * hidden_size + columns, mask=mask, other=0.0).to(
        tl.float32
    )
    variance = tl.sum(values * values, axis=0) / hidden_size
    inverse_rms = tl.rsqrt(variance + eps)
    weights = tl.load(weight + columns, mask=mask, other=0.0).to(tl.float32)
    normalized = (values * inverse_rms * weights).to(tl.bfloat16)

    table_row = tl.load(indices + row).to(tl.int64)
    table_offsets = table_row * hidden_size + columns
    scales = tl.load(scale + table_offsets, mask=mask, other=0.0)
    shifts = tl.load(shift + table_offsets, mask=mask, other=0.0)
    result = _bf16_modulate_exact(normalized, scales, shifts)
    tl.store(output + row * hidden_size + columns, result, mask=mask)


@triton.jit
def _modulate_kernel(
    value,
    scale,
    shift,
    indices,
    output,
    n_elements: tl.constexpr,
    hidden_size: tl.constexpr,
    sequence_length: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < n_elements
    rows = offsets // hidden_size
    columns = offsets - rows * hidden_size
    sequence_rows = rows % sequence_length
    table_rows = tl.load(indices + sequence_rows, mask=mask, other=0).to(tl.int64)
    table_offsets = table_rows * hidden_size + columns
    values = tl.load(value + offsets, mask=mask, other=0.0)
    scales = tl.load(scale + table_offsets, mask=mask, other=0.0)
    shifts = tl.load(shift + table_offsets, mask=mask, other=0.0)
    result = _bf16_modulate_exact(values, scales, shifts)
    tl.store(output + offsets, result, mask=mask)


@triton.jit
def _residual_kernel(
    residual,
    gate,
    branch,
    indices,
    output,
    n_elements: tl.constexpr,
    hidden_size: tl.constexpr,
    sequence_length: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < n_elements
    rows = offsets // hidden_size
    columns = offsets - rows * hidden_size
    sequence_rows = rows % sequence_length
    table_rows = tl.load(indices + sequence_rows, mask=mask, other=0).to(tl.int64)
    table_offsets = table_rows * hidden_size + columns
    residual_values = tl.load(residual + offsets, mask=mask, other=0.0)
    gates = tl.load(gate + table_offsets, mask=mask, other=0.0)
    branch_values = tl.load(branch + offsets, mask=mask, other=0.0)
    result = _bf16_residual_exact(residual_values, gates, branch_values)
    tl.store(output + offsets, result, mask=mask)


@triton.jit
def _modulate_backward_kernel(
    grad_output,
    scale,
    indices,
    grad_value,
    n_elements,
    hidden_size: tl.constexpr,
    sequence_length,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < n_elements
    rows = offsets // hidden_size
    columns = offsets - rows * hidden_size
    sequence_rows = rows % sequence_length
    table_rows = tl.load(indices + sequence_rows, mask=mask, other=0).to(tl.int64)
    table_offsets = table_rows * hidden_size + columns
    gradients = tl.load(grad_output + offsets, mask=mask, other=0.0)
    scales = tl.load(scale + table_offsets, mask=mask, other=0.0)
    result = _bf16_modulate_backward_exact(gradients, scales)
    tl.store(grad_value + offsets, result, mask=mask)


@triton.jit
def _residual_branch_backward_kernel(
    grad_output,
    gate,
    indices,
    grad_branch,
    n_elements,
    hidden_size: tl.constexpr,
    sequence_length,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < n_elements
    rows = offsets // hidden_size
    columns = offsets - rows * hidden_size
    sequence_rows = rows % sequence_length
    table_rows = tl.load(indices + sequence_rows, mask=mask, other=0).to(tl.int64)
    table_offsets = table_rows * hidden_size + columns
    gradients = tl.load(grad_output + offsets, mask=mask, other=0.0)
    gates = tl.load(gate + table_offsets, mask=mask, other=0.0)
    result = _bf16_mul_exact(gradients, gates)
    tl.store(grad_branch + offsets, result, mask=mask)


def _validate_common(
    value: torch.Tensor,
    table: torch.Tensor,
    indices: torch.Tensor,
) -> tuple[int, int, int]:
    tensors = (value, table, indices)
    if any(not tensor.is_cuda for tensor in tensors):
        raise RuntimeError("H3 fused pointwise requires CUDA tensors")
    if value.dtype != torch.bfloat16 or table.dtype != value.dtype:
        raise RuntimeError("H3 fused pointwise requires BF16 activation and table")
    if indices.dtype not in {torch.int32, torch.int64}:
        raise RuntimeError(f"H3 fused pointwise requires int32/int64 indices, got {indices.dtype}")
    if value.ndim != 3 or table.ndim != 2 or indices.ndim != 1:
        raise RuntimeError("H3 fused pointwise requires value=[B,S,H], table=[K,H], indices=[S]")
    batch, sequence_length, hidden_size = value.shape
    if sequence_length != indices.numel():
        raise RuntimeError(
            f"H3 fused pointwise sequence mismatch value={sequence_length} indices={indices.numel()}"
        )
    if table.shape[1] != hidden_size:
        raise RuntimeError(
            f"H3 fused pointwise table mismatch table={tuple(table.shape)} hidden={hidden_size}"
        )
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise RuntimeError("H3 fused pointwise requires contiguous tensors")
    n_elements = int(batch * sequence_length * hidden_size)
    if n_elements % 2 != 0:
        raise RuntimeError(f"H3 fused BF16 pack=2 requires an even element count, got {n_elements}")
    major, minor = torch.cuda.get_device_capability(value.device)
    if (major, minor) < (8, 0):
        raise RuntimeError(f"H3 fused pointwise requires SM80+, got sm_{major}{minor}")
    return n_elements, int(sequence_length), int(hidden_size)


def fused_modulate(
    value: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    n_elements, sequence_length, hidden_size = _validate_common(value, scale, indices)
    if shift.shape != scale.shape or shift.dtype != value.dtype or not shift.is_cuda or not shift.is_contiguous():
        raise RuntimeError(
            f"H3 fused modulation shift mismatch scale={tuple(scale.shape)} shift={tuple(shift.shape)}"
        )
    output = torch.empty_like(value)
    block_size = 256
    _modulate_kernel[(triton.cdiv(n_elements, block_size),)](
        value,
        scale,
        shift,
        indices,
        output,
        n_elements=n_elements,
        hidden_size=hidden_size,
        sequence_length=sequence_length,
        block_size=block_size,
    )
    return output


def fused_rmsnorm_modulate(
    value: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    indices: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    """Fuse the pinned B1 H3 RMSNorm and exact BF16 AdaLN modulation."""

    _, sequence_length, hidden_size = _validate_common(value, scale, indices)
    if value.shape[0] != 1:
        raise RuntimeError(
            f"H3 fused RMSNorm modulation is preregistered for B1, got B={value.shape[0]}"
        )
    if hidden_size != 5376:
        raise RuntimeError(
            f"H3 fused RMSNorm modulation requires hidden_size=5376, got {hidden_size}"
        )
    if float(eps) != 1e-5:
        raise RuntimeError(f"H3 fused RMSNorm modulation requires eps=1e-5, got {eps}")
    if (
        weight.ndim != 1
        or weight.numel() != hidden_size
        or weight.dtype != torch.bfloat16
        or not weight.is_cuda
        or weight.device != value.device
        or not weight.is_contiguous()
    ):
        raise RuntimeError(
            "H3 fused RMSNorm modulation requires a contiguous CUDA BF16 "
            f"weight[{hidden_size}], got shape={tuple(weight.shape)} "
            f"dtype={weight.dtype} device={weight.device} contiguous={weight.is_contiguous()}"
        )
    if (
        shift.shape != scale.shape
        or shift.dtype != value.dtype
        or not shift.is_cuda
        or shift.device != value.device
        or not shift.is_contiguous()
    ):
        raise RuntimeError(
            f"H3 fused RMSNorm modulation shift mismatch scale={tuple(scale.shape)} "
            f"shift={tuple(shift.shape)}"
        )
    output = torch.empty_like(value)
    _rmsnorm_modulate_kernel[(sequence_length,)](
        value,
        weight,
        scale,
        shift,
        indices,
        output,
        hidden_size=hidden_size,
        eps=float(eps),
        block_size=8192,
        num_warps=8,
        num_stages=1,
    )
    return output


@torch.no_grad()
def warmup_fused_rmsnorm_modulate(device: torch.device | int) -> None:
    """Compile and launch the only production specialization before timing."""

    value = torch.zeros((1, 1, 5376), device=device, dtype=torch.bfloat16)
    weight = torch.ones((5376,), device=device, dtype=torch.bfloat16)
    scale = torch.zeros((6, 5376), device=device, dtype=torch.bfloat16)
    shift = torch.zeros_like(scale)
    indices = torch.zeros((1,), device=device, dtype=torch.int64)
    fused_rmsnorm_modulate(value, weight, scale, shift, indices, eps=1e-5)
    torch.cuda.synchronize(device)


def fused_residual(
    residual: torch.Tensor,
    gate: torch.Tensor,
    branch: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    n_elements, sequence_length, hidden_size = _validate_common(residual, gate, indices)
    if branch.shape != residual.shape or branch.dtype != residual.dtype or not branch.is_cuda or not branch.is_contiguous():
        raise RuntimeError(
            f"H3 fused residual branch mismatch residual={tuple(residual.shape)} branch={tuple(branch.shape)}"
        )
    output = torch.empty_like(residual)
    block_size = 256
    _residual_kernel[(triton.cdiv(n_elements, block_size),)](
        residual,
        gate,
        branch,
        indices,
        output,
        n_elements=n_elements,
        hidden_size=hidden_size,
        sequence_length=sequence_length,
        block_size=block_size,
    )
    return output


def fused_modulate_backward(
    grad_output: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    n_elements, sequence_length, hidden_size = _validate_common(grad_output, scale, indices)
    grad_value = torch.empty_like(grad_output)
    block_size = 256
    _modulate_backward_kernel[(triton.cdiv(n_elements, block_size),)](
        grad_output,
        scale,
        indices,
        grad_value,
        n_elements=n_elements,
        hidden_size=hidden_size,
        sequence_length=sequence_length,
        block_size=block_size,
    )
    return grad_value


def fused_residual_branch_backward(
    grad_output: torch.Tensor,
    gate: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    n_elements, sequence_length, hidden_size = _validate_common(grad_output, gate, indices)
    grad_branch = torch.empty_like(grad_output)
    block_size = 256
    _residual_branch_backward_kernel[(triton.cdiv(n_elements, block_size),)](
        grad_output,
        gate,
        indices,
        grad_branch,
        n_elements=n_elements,
        hidden_size=hidden_size,
        sequence_length=sequence_length,
        block_size=block_size,
    )
    return grad_branch
