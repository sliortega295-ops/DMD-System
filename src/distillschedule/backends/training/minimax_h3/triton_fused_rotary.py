"""Exact BF16 rotary fusion for MiniMax-H3 attention.

The pinned Diffusers implementation materializes a rotate-half tensor, two
BF16 products, their BF16 sum, and a final concatenation for every Q/K pair.
This kernel preserves those BF16 rounding boundaries while writing the final
``[rotary, pass-through]`` tensor directly.  The optional backward kernel keeps
the same BF16 multiply/add rounding boundaries used by eager autograd.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _bf16_rotary_exact(value, rotated, cosine, sine):
    """Return ``bf16(bf16(value*cos) + bf16(rotated*sin))`` exactly."""

    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .b16 xh0,xh1,rh0,rh1,ch0,ch1,sh0,sh1,h0,h1,j0,j1,o0,o1;
          .reg .f32 xf0,xf1,rf0,rf1,cf0,cf1,sf0,sf1,p0,p1,q0,q1,t0,t1;
          mov.b32 {xh0,xh1}, $1;
          mov.b32 {rh0,rh1}, $2;
          mov.b32 {ch0,ch1}, $3;
          mov.b32 {sh0,sh1}, $4;
          cvt.f32.bf16 xf0, xh0; cvt.f32.bf16 xf1, xh1;
          cvt.f32.bf16 rf0, rh0; cvt.f32.bf16 rf1, rh1;
          cvt.f32.bf16 cf0, ch0; cvt.f32.bf16 cf1, ch1;
          cvt.f32.bf16 sf0, sh0; cvt.f32.bf16 sf1, sh1;
          mul.f32 p0, xf0, cf0; mul.f32 p1, xf1, cf1;
          cvt.rn.bf16.f32 h0, p0; cvt.rn.bf16.f32 h1, p1;
          mul.f32 q0, rf0, sf0; mul.f32 q1, rf1, sf1;
          cvt.rn.bf16.f32 j0, q0; cvt.rn.bf16.f32 j1, q1;
          cvt.f32.bf16 p0, h0; cvt.f32.bf16 p1, h1;
          cvt.f32.bf16 q0, j0; cvt.f32.bf16 q1, j1;
          add.f32 t0, p0, q0; add.f32 t1, p1, q1;
          cvt.rn.bf16.f32 o0, t0; cvt.rn.bf16.f32 o1, t1;
          mov.b32 $0, {o0,o1};
        }
        """,
        constraints="=r,r,r,r,r",
        args=[value, rotated, cosine, sine],
        dtype=tl.bfloat16,
        is_pure=True,
        pack=2,
    )


@triton.jit
def _rotary_kernel(
    hidden_states,
    cosine,
    sine,
    output,
    n_elements,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    valid = offsets < n_elements
    head_columns = offsets % head_dim
    rotary_mask = valid & (head_columns < rotary_dim)

    rows = offsets // head_dim
    # The frozen benchmark contract is B1/rank. Keeping the sequence extent
    # out of constexpr arguments prevents one Triton compilation per
    # rank-qualified packed sequence length.
    sequence_rows = rows // num_heads
    rotary_columns = head_columns
    half = rotary_dim // 2
    rotated_offsets = tl.where(
        rotary_columns < half,
        offsets + half,
        offsets - half,
    )

    values = tl.load(hidden_states + offsets, mask=valid, other=0.0)
    rotated = tl.load(hidden_states + rotated_offsets, mask=rotary_mask, other=0.0)
    rotated = tl.where(rotary_columns < half, -rotated, rotated).to(tl.bfloat16)
    table_offsets = sequence_rows * rotary_dim + rotary_columns
    cos_values = tl.load(cosine + table_offsets, mask=rotary_mask, other=0.0).to(tl.bfloat16)
    sin_values = tl.load(sine + table_offsets, mask=rotary_mask, other=0.0).to(tl.bfloat16)
    rotary_values = _bf16_rotary_exact(
        values.to(tl.bfloat16),
        rotated,
        cos_values,
        sin_values,
    )
    result = tl.where(rotary_mask, rotary_values, values)
    tl.store(output + offsets, result, mask=valid)


@triton.jit
def _rotary_backward_kernel(
    grad_output,
    cosine,
    sine,
    grad_input,
    n_elements,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    valid = offsets < n_elements
    head_columns = offsets % head_dim
    rotary_mask = valid & (head_columns < rotary_dim)

    rows = offsets // head_dim
    sequence_rows = rows // num_heads
    half = rotary_dim // 2
    paired_columns = tl.where(head_columns < half, head_columns + half, head_columns - half)
    paired_offsets = tl.where(head_columns < half, offsets + half, offsets - half)

    direct_grad = tl.load(grad_output + offsets, mask=valid, other=0.0).to(tl.bfloat16)
    paired_grad = tl.load(grad_output + paired_offsets, mask=rotary_mask, other=0.0).to(
        tl.bfloat16
    )
    direct_table = sequence_rows * rotary_dim + head_columns
    paired_table = sequence_rows * rotary_dim + paired_columns
    cos_values = tl.load(cosine + direct_table, mask=rotary_mask, other=0.0).to(tl.bfloat16)
    sin_values = tl.load(sine + paired_table, mask=rotary_mask, other=0.0).to(tl.bfloat16)
    signed_paired_grad = tl.where(head_columns < half, paired_grad, -paired_grad).to(
        tl.bfloat16
    )
    rotary_grad = _bf16_rotary_exact(
        direct_grad,
        signed_paired_grad,
        cos_values,
        sin_values,
    )
    result = tl.where(rotary_mask, rotary_grad, direct_grad)
    tl.store(grad_input + offsets, result, mask=valid)


@triton.jit
def _qk_rmsnorm_rotary_kernel(
    hidden_states,
    weight,
    cosine,
    sine,
    output,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    eps: tl.constexpr,
    block_size: tl.constexpr,
):
    """Fuse one frozen per-head RMSNorm row with exact BF16 rotary."""

    row = tl.program_id(0)
    columns = tl.arange(0, block_size)
    mask = columns < head_dim
    row_base = row * head_dim
    values = tl.load(hidden_states + row_base + columns, mask=mask, other=0.0).to(
        tl.float32
    )
    variance = tl.sum(values * values, axis=0) / head_dim
    inverse_rms = tl.rsqrt(variance + eps)
    weights = tl.load(weight + columns, mask=mask, other=0.0).to(tl.float32)
    normalized = (values * inverse_rms * weights).to(tl.bfloat16)

    half = rotary_dim // 2
    rotary_mask = columns < rotary_dim
    paired_columns = tl.where(columns < half, columns + half, columns - half)
    paired_values = tl.load(
        hidden_states + row_base + paired_columns,
        mask=rotary_mask,
        other=0.0,
    ).to(tl.float32)
    paired_weights = tl.load(weight + paired_columns, mask=rotary_mask, other=0.0).to(
        tl.float32
    )
    paired_normalized = (paired_values * inverse_rms * paired_weights).to(tl.bfloat16)
    rotated = tl.where(columns < half, -paired_normalized, paired_normalized).to(
        tl.bfloat16
    )

    sequence_row = row // num_heads
    table_offsets = sequence_row * rotary_dim + columns
    cos_values = tl.load(cosine + table_offsets, mask=rotary_mask, other=0.0).to(
        tl.bfloat16
    )
    sin_values = tl.load(sine + table_offsets, mask=rotary_mask, other=0.0).to(
        tl.bfloat16
    )
    rotary_values = _bf16_rotary_exact(
        normalized,
        rotated,
        cos_values,
        sin_values,
    )
    result = tl.where(rotary_mask, rotary_values, normalized)
    tl.store(output + row_base + columns, result, mask=mask)


@triton.jit
def _qk_rmsnorm_rotary_backward_kernel(
    grad_output,
    hidden_states,
    weight,
    cosine,
    sine,
    grad_input,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    eps: tl.constexpr,
    block_size: tl.constexpr,
):
    """Fuse exact rotary backward with frozen-affine RMSNorm backward."""

    row = tl.program_id(0)
    columns = tl.arange(0, block_size)
    mask = columns < head_dim
    row_base = row * head_dim

    direct_grad = tl.load(grad_output + row_base + columns, mask=mask, other=0.0).to(
        tl.bfloat16
    )
    half = rotary_dim // 2
    rotary_mask = columns < rotary_dim
    paired_columns = tl.where(columns < half, columns + half, columns - half)
    paired_grad = tl.load(
        grad_output + row_base + paired_columns,
        mask=rotary_mask,
        other=0.0,
    ).to(tl.bfloat16)
    sequence_row = row // num_heads
    direct_table = sequence_row * rotary_dim + columns
    paired_table = sequence_row * rotary_dim + paired_columns
    cos_values = tl.load(cosine + direct_table, mask=rotary_mask, other=0.0).to(
        tl.bfloat16
    )
    sin_values = tl.load(sine + paired_table, mask=rotary_mask, other=0.0).to(
        tl.bfloat16
    )
    signed_paired_grad = tl.where(columns < half, paired_grad, -paired_grad).to(
        tl.bfloat16
    )
    grad_normalized = _bf16_rotary_exact(
        direct_grad,
        signed_paired_grad,
        cos_values,
        sin_values,
    )
    grad_normalized = tl.where(rotary_mask, grad_normalized, direct_grad).to(tl.float32)

    values = tl.load(hidden_states + row_base + columns, mask=mask, other=0.0).to(
        tl.float32
    )
    weights = tl.load(weight + columns, mask=mask, other=0.0).to(tl.float32)
    inverse_rms = tl.rsqrt(tl.sum(values * values, axis=0) / head_dim + eps)
    weighted_grad = grad_normalized * weights
    projection = tl.sum(weighted_grad * values, axis=0) / head_dim
    result = inverse_rms * (
        weighted_grad - values * (inverse_rms * inverse_rms) * projection
    )
    tl.store(grad_input + row_base + columns, result, mask=mask)


def fused_apply_rotary_emb(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply pinned MiniMax-H3 rotary math without intermediate tensors."""

    if not hidden_states.is_cuda or not cos.is_cuda or not sin.is_cuda:
        raise RuntimeError("H3 fused rotary requires CUDA tensors")
    if hidden_states.dtype != torch.bfloat16:
        raise RuntimeError(f"H3 fused rotary requires BF16 hidden states, got {hidden_states.dtype}")
    if hidden_states.ndim != 4 or cos.ndim != 2 or sin.ndim != 2:
        raise RuntimeError(
            "H3 fused rotary requires hidden=[B,S,NH,HD] and cos/sin=[S,R]"
        )
    if cos.shape != sin.shape or cos.device != hidden_states.device or sin.device != hidden_states.device:
        raise RuntimeError("H3 fused rotary cos/sin shape or device mismatch")
    batch, sequence_length, num_heads, head_dim = map(int, hidden_states.shape)
    if batch != 1:
        raise RuntimeError(f"H3 fused rotary is registered only for the frozen B1 contract, got B={batch}")
    rotary_dim = int(cos.shape[1])
    if int(cos.shape[0]) != sequence_length:
        raise RuntimeError(
            f"H3 fused rotary sequence mismatch hidden={sequence_length} cos={cos.shape[0]}"
        )
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2:
        raise RuntimeError(
            f"H3 fused rotary requires even 0 < rotary_dim <= head_dim, got {rotary_dim}/{head_dim}"
        )
    if cos.dtype not in {torch.float32, torch.bfloat16} or sin.dtype != cos.dtype:
        raise RuntimeError(f"H3 fused rotary requires matching FP32/BF16 cos/sin, got {cos.dtype}/{sin.dtype}")
    if not hidden_states.is_contiguous() or not cos.is_contiguous() or not sin.is_contiguous():
        raise RuntimeError("H3 fused rotary requires contiguous hidden/cos/sin tensors")
    n_elements = batch * sequence_length * num_heads * head_dim
    if n_elements % 2:
        raise RuntimeError(f"H3 fused rotary BF16 pack=2 requires even element count, got {n_elements}")
    major, minor = torch.cuda.get_device_capability(hidden_states.device)
    if (major, minor) < (8, 0):
        raise RuntimeError(f"H3 fused rotary requires SM80+, got sm_{major}{minor}")

    output = torch.empty_like(hidden_states)
    block_size = 1024
    _rotary_kernel[(triton.cdiv(n_elements, block_size),)](
        hidden_states,
        cos,
        sin,
        output,
        n_elements=n_elements,
        num_heads=num_heads,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        block_size=block_size,
        num_warps=8,
    )
    return output


def fused_qk_rmsnorm_rotary(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    """Fuse pinned no-grad Q/K RMSNorm with the validated rotary operation."""

    if not hidden_states.is_cuda or not weight.is_cuda or not cos.is_cuda or not sin.is_cuda:
        raise RuntimeError("H3 fused Q/K RMSNorm rotary requires CUDA tensors")
    if hidden_states.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise RuntimeError(
            "H3 fused Q/K RMSNorm rotary requires BF16 hidden/weight, got "
            f"{hidden_states.dtype}/{weight.dtype}"
        )
    if hidden_states.ndim != 4 or weight.ndim != 1 or cos.ndim != 2 or sin.ndim != 2:
        raise RuntimeError(
            "H3 fused Q/K RMSNorm rotary requires hidden=[B,S,NH,HD], "
            "weight=[HD], cos/sin=[S,R]"
        )
    batch, sequence_length, num_heads, head_dim = map(int, hidden_states.shape)
    if (batch, num_heads, head_dim) != (1, 56, 128):
        raise RuntimeError(
            "H3 fused Q/K RMSNorm rotary is preregistered for B1/NH56/HD128, got "
            f"B{batch}/NH{num_heads}/HD{head_dim}"
        )
    if float(eps) != 1e-5:
        raise RuntimeError(f"H3 fused Q/K RMSNorm rotary requires eps=1e-5, got {eps}")
    if weight.numel() != head_dim:
        raise RuntimeError(
            f"H3 fused Q/K RMSNorm rotary weight has {weight.numel()} values, expected {head_dim}"
        )
    if cos.shape != sin.shape or int(cos.shape[0]) != sequence_length:
        raise RuntimeError(
            f"H3 fused Q/K RMSNorm rotary cos/sin mismatch {tuple(cos.shape)}/{tuple(sin.shape)}"
        )
    rotary_dim = int(cos.shape[1])
    if rotary_dim != 96:
        raise RuntimeError(
            f"H3 fused Q/K RMSNorm rotary requires rotary_dim=96, got {rotary_dim}"
        )
    if cos.dtype not in {torch.float32, torch.bfloat16} or sin.dtype != cos.dtype:
        raise RuntimeError(
            f"H3 fused Q/K RMSNorm rotary requires matching FP32/BF16 cos/sin, got "
            f"{cos.dtype}/{sin.dtype}"
        )
    tensors = (hidden_states, weight, cos, sin)
    if any(tensor.device != hidden_states.device for tensor in tensors):
        raise RuntimeError("H3 fused Q/K RMSNorm rotary operands must share one device")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise RuntimeError("H3 fused Q/K RMSNorm rotary requires contiguous operands")
    major, minor = torch.cuda.get_device_capability(hidden_states.device)
    if (major, minor) < (8, 0):
        raise RuntimeError(f"H3 fused Q/K RMSNorm rotary requires SM80+, got sm_{major}{minor}")

    output = torch.empty_like(hidden_states)
    _qk_rmsnorm_rotary_kernel[(sequence_length * num_heads,)](
        hidden_states,
        weight,
        cos,
        sin,
        output,
        num_heads=num_heads,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        eps=float(eps),
        block_size=128,
        num_warps=2,
        num_stages=1,
    )
    return output


def fused_qk_rmsnorm_rotary_backward(
    grad_output: torch.Tensor,
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    """Return the input gradient for the frozen Q/K RMSNorm+rotary chain."""

    if not grad_output.is_cuda or grad_output.dtype != torch.bfloat16:
        raise RuntimeError("H3 fused Q/K backward requires CUDA BF16 grad_output")
    if grad_output.shape != hidden_states.shape:
        raise RuntimeError(
            "H3 fused Q/K backward shape mismatch "
            f"grad={tuple(grad_output.shape)} hidden={tuple(hidden_states.shape)}"
        )
    if not grad_output.is_contiguous():
        raise RuntimeError("H3 fused Q/K backward requires contiguous grad_output")
    if hidden_states.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise RuntimeError(
            "H3 fused Q/K backward requires BF16 hidden/weight, got "
            f"{hidden_states.dtype}/{weight.dtype}"
        )
    batch, sequence_length, num_heads, head_dim = map(int, hidden_states.shape)
    if (batch, num_heads, head_dim) != (1, 56, 128):
        raise RuntimeError(
            "H3 fused Q/K backward is preregistered for B1/NH56/HD128, got "
            f"B{batch}/NH{num_heads}/HD{head_dim}"
        )
    if float(eps) != 1e-5:
        raise RuntimeError(f"H3 fused Q/K backward requires eps=1e-5, got {eps}")
    if weight.numel() != head_dim or cos.shape != sin.shape:
        raise RuntimeError("H3 fused Q/K backward weight or rotary table mismatch")
    rotary_dim = int(cos.shape[1])
    if int(cos.shape[0]) != sequence_length or rotary_dim != 96:
        raise RuntimeError(
            "H3 fused Q/K backward requires cos/sin=[S,96], got "
            f"{tuple(cos.shape)}/{tuple(sin.shape)}"
        )
    tensors = (grad_output, hidden_states, weight, cos, sin)
    if any(tensor.device != hidden_states.device for tensor in tensors):
        raise RuntimeError("H3 fused Q/K backward operands must share one device")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise RuntimeError("H3 fused Q/K backward requires contiguous operands")

    grad_input = torch.empty_like(hidden_states)
    _qk_rmsnorm_rotary_backward_kernel[(sequence_length * num_heads,)](
        grad_output,
        hidden_states,
        weight,
        cos,
        sin,
        grad_input,
        num_heads=num_heads,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        eps=float(eps),
        block_size=128,
        num_warps=2,
        num_stages=1,
    )
    return grad_input


@torch.no_grad()
def warmup_fused_qk_rmsnorm_rotary(device: torch.device | int) -> None:
    hidden = torch.zeros((1, 1, 56, 128), device=device, dtype=torch.bfloat16)
    weight = torch.ones((128,), device=device, dtype=torch.bfloat16)
    cos = torch.ones((1, 96), device=device, dtype=torch.float32)
    sin = torch.zeros_like(cos)
    fused_qk_rmsnorm_rotary(hidden, weight, cos, sin, eps=1e-5)
    torch.cuda.synchronize(device)


@torch.no_grad()
def warmup_fused_qk_rmsnorm_rotary_backward(device: torch.device | int) -> None:
    hidden = torch.zeros((1, 1, 56, 128), device=device, dtype=torch.bfloat16)
    grad = torch.zeros_like(hidden)
    weight = torch.ones((128,), device=device, dtype=torch.bfloat16)
    cos = torch.ones((1, 96), device=device, dtype=torch.float32)
    sin = torch.zeros_like(cos)
    fused_qk_rmsnorm_rotary_backward(
        grad, hidden, weight, cos, sin, eps=1e-5
    )
    torch.cuda.synchronize(device)


def fused_apply_rotary_emb_backward(
    grad_output: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Return the exact hidden-state gradient for :func:`fused_apply_rotary_emb`."""

    # Reuse the forward contract checks without allocating a forward output.
    if not grad_output.is_cuda or not cos.is_cuda or not sin.is_cuda:
        raise RuntimeError("H3 fused rotary backward requires CUDA tensors")
    if grad_output.dtype != torch.bfloat16 or grad_output.ndim != 4:
        raise RuntimeError(
            "H3 fused rotary backward requires BF16 grad_output=[B,S,NH,HD]"
        )
    if cos.shape != sin.shape or cos.device != grad_output.device or sin.device != grad_output.device:
        raise RuntimeError("H3 fused rotary backward cos/sin shape or device mismatch")
    batch, sequence_length, num_heads, head_dim = map(int, grad_output.shape)
    if batch != 1:
        raise RuntimeError(
            f"H3 fused rotary backward is registered only for B1, got B={batch}"
        )
    rotary_dim = int(cos.shape[1])
    if int(cos.shape[0]) != sequence_length:
        raise RuntimeError(
            f"H3 fused rotary backward sequence mismatch grad={sequence_length} cos={cos.shape[0]}"
        )
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2:
        raise RuntimeError(
            f"H3 fused rotary backward requires even 0 < rotary_dim <= head_dim, got {rotary_dim}/{head_dim}"
        )
    if cos.dtype not in {torch.float32, torch.bfloat16} or sin.dtype != cos.dtype:
        raise RuntimeError(
            f"H3 fused rotary backward requires matching FP32/BF16 cos/sin, got {cos.dtype}/{sin.dtype}"
        )
    if not grad_output.is_contiguous() or not cos.is_contiguous() or not sin.is_contiguous():
        raise RuntimeError("H3 fused rotary backward requires contiguous grad/cos/sin")
    n_elements = batch * sequence_length * num_heads * head_dim
    if n_elements % 2:
        raise RuntimeError(
            f"H3 fused rotary backward BF16 pack=2 requires even element count, got {n_elements}"
        )

    grad_input = torch.empty_like(grad_output)
    block_size = 1024
    _rotary_backward_kernel[(triton.cdiv(n_elements, block_size),)](
        grad_output,
        cos,
        sin,
        grad_input,
        n_elements=n_elements,
        num_heads=num_heads,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        block_size=block_size,
        num_warps=8,
    )
    return grad_input
