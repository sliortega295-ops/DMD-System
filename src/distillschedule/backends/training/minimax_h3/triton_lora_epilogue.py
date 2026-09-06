"""Fused no-grad BF16 LoRA-B GEMM, rounded residual add, and BF16 store."""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl


LORA_RANK = 128
BLOCK_M = 64
DEFAULT_BLOCK_N = 64
BLOCK_N = int(os.environ.get("H3_LORA_EPILOGUE_BLOCK_N", DEFAULT_BLOCK_N))
if BLOCK_N not in {64, 128}:
    raise RuntimeError(
        "H3_LORA_EPILOGUE_BLOCK_N must be one of {64, 128}, "
        f"got {BLOCK_N}"
    )
BLOCK_K = 32


@triton.jit
def _lora_epilogue_kernel(
    base,
    projected,
    weight,
    output,
    rows,
    columns,
    stride_base_row,
    stride_base_col,
    stride_projected_row,
    stride_projected_k,
    stride_weight_col,
    stride_weight_k,
    stride_output_row,
    stride_output_col,
    rank: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    group_m: tl.constexpr,
):
    program = tl.program_id(0)
    programs_m = tl.cdiv(rows, block_m)
    programs_n = tl.cdiv(columns, block_n)
    programs_per_group = group_m * programs_n
    group = program // programs_per_group
    first_m = group * group_m
    group_size_m = tl.minimum(programs_m - first_m, group_m)
    program_in_group = program % programs_per_group
    program_m = first_m + (program_in_group % group_size_m)
    program_n = program_in_group // group_size_m

    offsets_m = program_m * block_m + tl.arange(0, block_m)
    offsets_n = program_n * block_n + tl.arange(0, block_n)
    offsets_k = tl.arange(0, block_k)
    projected_ptrs = (
        projected
        + offsets_m[:, None] * stride_projected_row
        + offsets_k[None, :] * stride_projected_k
    )
    weight_ptrs = (
        weight
        + offsets_n[None, :] * stride_weight_col
        + offsets_k[:, None] * stride_weight_k
    )
    accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)
    for k_start in range(0, rank, block_k):
        projected_values = tl.load(
            projected_ptrs,
            mask=(offsets_m[:, None] < rows)
            & (offsets_k[None, :] + k_start < rank),
            other=0.0,
        )
        weight_values = tl.load(
            weight_ptrs,
            mask=(offsets_n[None, :] < columns)
            & (offsets_k[:, None] + k_start < rank),
            other=0.0,
        )
        accumulator = tl.dot(projected_values, weight_values, accumulator)
        projected_ptrs += block_k * stride_projected_k
        weight_ptrs += block_k * stride_weight_k

    base_ptrs = (
        base
        + offsets_m[:, None] * stride_base_row
        + offsets_n[None, :] * stride_base_col
    )
    mask = (offsets_m[:, None] < rows) & (offsets_n[None, :] < columns)
    base_values = tl.load(base_ptrs, mask=mask, other=0.0).to(tl.float32)
    # The pinned eager path materializes the BF16 LoRA-B Linear output before
    # the BF16 residual add. Preserve that intermediate rounding point even
    # though the temporary output tensor itself is eliminated.
    rounded_projection = accumulator.to(tl.bfloat16).to(tl.float32)
    output_ptrs = (
        output
        + offsets_m[:, None] * stride_output_row
        + offsets_n[None, :] * stride_output_col
    )
    tl.store(output_ptrs, rounded_projection + base_values, mask=mask)


def _validate(
    base: torch.Tensor, projected: torch.Tensor, weight: torch.Tensor
) -> tuple[int, int]:
    if not (base.is_cuda and projected.is_cuda and weight.is_cuda):
        raise RuntimeError("H3 LoRA epilogue requires CUDA tensors")
    if not (base.device == projected.device == weight.device):
        raise RuntimeError("H3 LoRA epilogue requires all operands on one device")
    if base.dtype != torch.bfloat16:
        raise RuntimeError(f"H3 LoRA epilogue requires BF16 base, got {base.dtype}")
    if projected.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise RuntimeError(
            "H3 LoRA epilogue requires live BF16 LoRA-A output and LoRA-B weight; "
            f"got projected={projected.dtype} weight={weight.dtype} "
            f"base={base.dtype} shapes="
            f"{tuple(projected.shape)}/{tuple(weight.shape)}/{tuple(base.shape)}"
        )
    if base.ndim < 2 or projected.ndim != base.ndim or weight.ndim != 2:
        raise RuntimeError("H3 LoRA epilogue observed an unsupported operand rank")
    if (
        base.shape[:-1] != projected.shape[:-1]
        or projected.shape[-1] != LORA_RANK
        or weight.shape != (base.shape[-1], LORA_RANK)
    ):
        raise RuntimeError(
            "H3 LoRA epilogue shape mismatch: "
            f"base={tuple(base.shape)} projected={tuple(projected.shape)} "
            f"weight={tuple(weight.shape)}"
        )
    if not (base.is_contiguous() and projected.is_contiguous() and weight.is_contiguous()):
        raise RuntimeError("H3 LoRA epilogue requires contiguous operands")
    major, minor = torch.cuda.get_device_capability(base.device)
    if (major, minor) < (8, 0):
        raise RuntimeError(f"H3 LoRA epilogue requires SM80+, got sm_{major}{minor}")
    return int(base.numel() // base.shape[-1]), int(base.shape[-1])


def fused_lora_b_residual(
    base: torch.Tensor, projected: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """Return eager-equivalent BF16 residual output without LoRA-B staging."""

    rows, columns = _validate(base, projected, weight)
    flat_base = base.reshape(rows, columns)
    flat_projected = projected.reshape(rows, LORA_RANK)
    output = torch.empty_like(flat_base)
    grid = (triton.cdiv(rows, BLOCK_M) * triton.cdiv(columns, BLOCK_N),)
    _lora_epilogue_kernel[grid](
        flat_base,
        flat_projected,
        weight,
        output,
        rows,
        columns,
        flat_base.stride(0),
        flat_base.stride(1),
        flat_projected.stride(0),
        flat_projected.stride(1),
        weight.stride(0),
        weight.stride(1),
        output.stride(0),
        output.stride(1),
        rank=LORA_RANK,
        block_m=BLOCK_M,
        block_n=BLOCK_N,
        block_k=BLOCK_K,
        group_m=8,
        num_warps=8,
        num_stages=3,
    )
    return output.reshape(base.shape)


def warmup_lora_epilogue(device: torch.device | int) -> None:
    """Compile and load the one production specialization outside cycle timing."""

    base = torch.zeros((64, 64), device=device, dtype=torch.bfloat16)
    projected = torch.zeros((64, LORA_RANK), device=device, dtype=torch.bfloat16)
    weight = torch.zeros((64, LORA_RANK), device=device, dtype=torch.bfloat16)
    fused_lora_b_residual(base, projected, weight)
    torch.cuda.synchronize(device)
