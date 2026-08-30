"""Exact paired query/key RoPE for the pinned Helios Triton path.

The installed Helios attention processor applies the same rotary table to Q
and K with two independent ``Flash_RoPE_Transposed`` calls.  Each call also
materializes identical contiguous cosine and sine views.  This opt-in wrapper
materializes those views once and invokes the *installed Helios arithmetic
kernel* for Q and K.  Reusing the exact production kernel is intentional: the
first single-launch prototype changed one BF16 rounding bit in roughly 1e-5 of
elements.  The bounded repair keeps source arithmetic/codegen intact while
removing redundant frequency staging.  It remains outside the default
workload unless installed by the separately gated runtime rewrite.
"""
from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl


SUPPORTED_HEAD_DIM = 128
SUPPORTED_HEADS = 40
BLOCK_SIZE = 64
NUM_WARPS = 4


@triton.jit
def _paired_rope_kernel(
    query_ptr,
    key_ptr,
    query_out_ptr,
    key_out_ptr,
    freqs_ptr,
    n_rows,
    n_heads: tl.constexpr,
    query_stride_row: tl.constexpr,
    key_stride_row: tl.constexpr,
    query_out_stride_row: tl.constexpr,
    key_out_stride_row: tl.constexpr,
    freqs_stride_row: tl.constexpr,
    head_dim: tl.constexpr,
    NEGATE_SIN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = (row < n_rows) & (offsets < head_dim // 2)
    freq_row = row // n_heads

    query = query_ptr + row * query_stride_row
    key = key_ptr + row * key_stride_row
    query_out = query_out_ptr + row * query_out_stride_row
    key_out = key_out_ptr + row * key_out_stride_row
    freqs = freqs_ptr + freq_row * freqs_stride_row

    even = offsets * 2
    odd = even + 1
    cos_even = tl.load(freqs + even, mask=mask, other=0.0)
    sin_odd = tl.load(freqs + head_dim + odd, mask=mask, other=0.0)
    if NEGATE_SIN:
        sin_odd = -sin_odd

    query_real = tl.load(query + even, mask=mask, other=0.0)
    query_imag = tl.load(query + odd, mask=mask, other=0.0)
    key_real = tl.load(key + even, mask=mask, other=0.0)
    key_imag = tl.load(key + odd, mask=mask, other=0.0)

    query_even = query_real * cos_even - query_imag * sin_odd
    query_odd = query_real * sin_odd + query_imag * cos_even
    key_even = key_real * cos_even - key_imag * sin_odd
    key_odd = key_real * sin_odd + key_imag * cos_even

    tl.store(query_out + even, query_even, mask=mask)
    tl.store(query_out + odd, query_odd, mask=mask)
    tl.store(key_out + even, key_even, mask=mask)
    tl.store(key_out + odd, key_odd, mask=mask)


def _validate(
    query: torch.Tensor,
    key: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[int, int, int, int]:
    if not query.is_cuda or not key.is_cuda or not freqs_cis.is_cuda:
        raise RuntimeError("paired Helios RoPE requires CUDA tensors")
    if query.dtype is not torch.bfloat16 or key.dtype is not torch.bfloat16:
        raise RuntimeError("paired Helios RoPE requires BF16 query/key")
    if freqs_cis.dtype is not torch.float32:
        raise RuntimeError("paired Helios RoPE requires FP32 rotary frequencies")
    if query.ndim != 4 or tuple(key.shape) != tuple(query.shape):
        raise RuntimeError("paired Helios RoPE requires equal rank-4 query/key")
    batch, sequence, heads, head_dim = map(int, query.shape)
    if batch not in (1, 3):
        raise RuntimeError("paired Helios RoPE is registered only for B1/B3")
    if heads != SUPPORTED_HEADS or head_dim != SUPPORTED_HEAD_DIM:
        raise RuntimeError(
            f"Helios RoPE head layout changed: heads={heads}, dim={head_dim}"
        )
    if tuple(freqs_cis.shape) != (batch, sequence, head_dim * 2):
        raise RuntimeError(
            "Helios rotary frequency layout changed: "
            f"{tuple(freqs_cis.shape)} != {(batch, sequence, head_dim * 2)}"
        )
    for name, value in (("query", query), ("key", key)):
        if not value.is_contiguous():
            raise RuntimeError(f"paired Helios RoPE requires contiguous {name}")
    if not freqs_cis.is_contiguous():
        raise RuntimeError("paired Helios RoPE requires contiguous frequencies")
    return batch, sequence, heads, head_dim


def _launch_direct_pair(
    query: torch.Tensor,
    key: torch.Tensor,
    freqs_cis: torch.Tensor,
    *,
    negate_sin: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, sequence, heads, head_dim = _validate(query, key, freqs_cis)
    query_out = torch.empty_like(query)
    key_out = torch.empty_like(key)
    query_flat = query.reshape(-1, head_dim)
    key_flat = key.reshape(-1, head_dim)
    query_out_flat = query_out.reshape(-1, head_dim)
    key_out_flat = key_out.reshape(-1, head_dim)
    freqs_flat = freqs_cis.reshape(batch * sequence, head_dim * 2)
    n_rows = int(query_flat.shape[0])
    _paired_rope_kernel[(n_rows,)](
        query_flat,
        key_flat,
        query_out_flat,
        key_out_flat,
        freqs_flat,
        n_rows,
        heads,
        int(query_flat.stride(0)),
        int(key_flat.stride(0)),
        int(query_out_flat.stride(0)),
        int(key_out_flat.stride(0)),
        int(freqs_flat.stride(0)),
        head_dim,
        NEGATE_SIN=bool(negate_sin),
        BLOCK=BLOCK_SIZE,
        num_warps=NUM_WARPS,
    )
    return query_out, key_out


class _ExactPairedRoPE(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        query: torch.Tensor,
        key: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, sequence, _, head_dim = _validate(query, key, freqs_cis)
        freqs_flat = freqs_cis.reshape(
            batch * sequence, head_dim * 2
        ).contiguous()
        cos = freqs_flat[:, :head_dim].contiguous()
        sin = freqs_flat[:, head_dim:].contiguous()
        query_out, key_out = _launch_source_pair(query, key, cos, sin)
        ctx.save_for_backward(cos, sin)
        return query_out, key_out

    @staticmethod
    def backward(
        ctx: Any,
        query_grad: torch.Tensor,
        key_grad: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        cos, sin = ctx.saved_tensors
        query_input_grad, key_input_grad = _launch_source_pair(
            query_grad.contiguous(),
            key_grad.contiguous(),
            cos,
            -sin,
        )
        return query_input_grad, key_input_grad, None


def _launch_source_pair(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch the exact installed Helios production kernel for both tensors."""

    from helios.modules.helios_kernels.triton_rope import (
        _apply_rope_transposed_kernel,
    )
    from helios.modules.helios_kernels.utils import (
        calculate_settings,
        torch_gpu_device,
    )

    batch, sequence, heads, head_dim = map(int, query.shape)
    if tuple(key.shape) != tuple(query.shape):
        raise RuntimeError("paired Helios RoPE query/key shape drift")
    if tuple(cos.shape) != (batch * sequence, head_dim):
        raise RuntimeError("paired Helios RoPE cosine shape drift")
    if tuple(sin.shape) != tuple(cos.shape):
        raise RuntimeError("paired Helios RoPE sine shape drift")
    query_flat = query.reshape(-1, head_dim).contiguous()
    key_flat = key.reshape(-1, head_dim).contiguous()
    query_out_flat = torch.empty_like(query_flat)
    key_out_flat = torch.empty_like(key_flat)
    n_rows = int(query_flat.shape[0])
    block_size, num_warps = calculate_settings(head_dim // 2)
    with torch_gpu_device(query.device):
        for value, output in (
            (query_flat, query_out_flat),
            (key_flat, key_out_flat),
        ):
            _apply_rope_transposed_kernel[(n_rows,)](
                value,
                output,
                cos,
                sin,
                heads,
                int(value.stride(0)),
                int(output.stride(0)),
                int(cos.stride(0)),
                head_dim,
                BLOCK_SIZE=block_size,
                num_warps=num_warps,
            )
    return (
        query_out_flat.reshape(batch, sequence, heads, head_dim),
        key_out_flat.reshape(batch, sequence, heads, head_dim),
    )


def exact_paired_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the pinned Helios RoPE independently to Q and K in one launch."""

    return _ExactPairedRoPE.apply(query, key, freqs_cis)


def runtime_identity() -> dict[str, Any]:
    return {
        "implementation": "HELIOS_SOURCE_KERNEL_QK_SHARED_FREQ_STAGING",
        "torch_version": str(torch.__version__),
        "triton_version": str(triton.__version__),
        "heads": SUPPORTED_HEADS,
        "head_dim": SUPPORTED_HEAD_DIM,
        "block_size": BLOCK_SIZE,
        "num_warps": NUM_WARPS,
    }


__all__ = ["exact_paired_rope", "runtime_identity"]
