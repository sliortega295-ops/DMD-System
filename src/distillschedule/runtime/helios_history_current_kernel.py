"""Exact Helios history-prefix/current-suffix add-and-copy fusion.

The pinned non-NaViT guidance-cross-attention path first adds the attention
result to the current-token suffix and then concatenates the unchanged history
prefix.  This module writes the final BF16 tensor directly in one Triton
kernel.  Its custom backward returns the incoming full gradient directly for
the pre-split hidden tensor and a suffix view for the attention result, avoiding
the source cat/split reconstruction without changing any mathematical value.

The module is imported only by the default-off C25 candidate.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


SUPPORTED_WIDTH = 5120
BLOCK_SIZE = 512
NUM_WARPS = 8


@triton.jit
def _history_current_add_kernel(
    hidden_ptr,
    attention_ptr,
    output_ptr,
    n_elements,
    total_tokens,
    history_tokens,
    current_tokens,
    hidden_stride_b,
    hidden_stride_t,
    attention_stride_b,
    attention_stride_t,
    output_stride_b,
    output_stride_t,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    global_token = offsets // WIDTH
    feature = offsets - global_token * WIDTH
    batch = global_token // total_tokens
    token = global_token - batch * total_tokens
    hidden_offsets = (
        batch * hidden_stride_b + token * hidden_stride_t + feature
    )
    output_offsets = (
        batch * output_stride_b + token * output_stride_t + feature
    )
    hidden = tl.load(hidden_ptr + hidden_offsets, mask=mask).to(tl.float32)
    is_current = token >= history_tokens
    current_token = token - history_tokens
    attention_offsets = (
        batch * attention_stride_b
        + current_token * attention_stride_t
        + feature
    )
    attention = tl.load(
        attention_ptr + attention_offsets,
        mask=mask & is_current,
        other=0.0,
    ).to(tl.float32)
    summed = tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [hidden, attention],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    value = tl.where(is_current, summed, hidden)
    tl.store(output_ptr + output_offsets, value, mask=mask)


def _validate(
    hidden_states: torch.Tensor,
    attention_output: torch.Tensor,
    history_tokens: int,
) -> tuple[int, int, int]:
    if not hidden_states.is_cuda or not attention_output.is_cuda:
        raise RuntimeError("Helios history/current fusion requires CUDA tensors")
    if hidden_states.dtype is not torch.bfloat16 or attention_output.dtype is not torch.bfloat16:
        raise RuntimeError("Helios history/current fusion requires BF16 tensors")
    if hidden_states.ndim != 3 or attention_output.ndim != 3:
        raise RuntimeError("Helios history/current fusion requires rank-3 tensors")
    batch, total_tokens, width = map(int, hidden_states.shape)
    if batch not in (1, 3):
        raise RuntimeError("Helios history/current fusion is registered only for B1/B3")
    if width != SUPPORTED_WIDTH:
        raise RuntimeError(f"Helios hidden width changed: {width}")
    history_tokens = int(history_tokens)
    if history_tokens <= 0 or history_tokens >= total_tokens:
        raise RuntimeError(
            f"invalid history/current split: {history_tokens}/{total_tokens}"
        )
    current_tokens = total_tokens - history_tokens
    if tuple(attention_output.shape) != (batch, current_tokens, width):
        raise RuntimeError(
            "Helios attention output shape changed: "
            f"{tuple(attention_output.shape)} != {(batch, current_tokens, width)}"
        )
    for name, value in (
        ("hidden", hidden_states),
        ("attention", attention_output),
    ):
        if int(value.stride(2)) != 1 or int(value.stride(1)) != width:
            raise RuntimeError(f"unsupported Helios {name} strides: {value.stride()}")
        if int(value.stride(0)) < int(value.shape[1]) * width:
            raise RuntimeError(f"overlapping Helios {name} batch stride: {value.stride()}")
    return batch, total_tokens, current_tokens


def _forward_kernel(
    hidden_states: torch.Tensor,
    attention_output: torch.Tensor,
    history_tokens: int,
) -> torch.Tensor:
    _batch, total_tokens, current_tokens = _validate(
        hidden_states, attention_output, history_tokens
    )
    output = torch.empty(
        hidden_states.shape,
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    n_elements = int(output.numel())
    _history_current_add_kernel[(triton.cdiv(n_elements, BLOCK_SIZE),)](
        hidden_states,
        attention_output,
        output,
        n_elements,
        total_tokens,
        int(history_tokens),
        current_tokens,
        int(hidden_states.stride(0)),
        int(hidden_states.stride(1)),
        int(attention_output.stride(0)),
        int(attention_output.stride(1)),
        int(output.stride(0)),
        int(output.stride(1)),
        WIDTH=SUPPORTED_WIDTH,
        BLOCK=BLOCK_SIZE,
        num_warps=NUM_WARPS,
    )
    return output


class _ExactHistoryCurrentAdd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: object,
        hidden_states: torch.Tensor,
        attention_output: torch.Tensor,
        history_tokens: int,
    ) -> torch.Tensor:
        history_tokens = int(history_tokens)
        result = _forward_kernel(hidden_states, attention_output, history_tokens)
        ctx.history_tokens = history_tokens
        return result

    @staticmethod
    def backward(
        ctx: object, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        history_tokens = int(ctx.history_tokens)
        # The source graph is cat(history, current + attention).  Its exact
        # derivatives are the full upstream tensor for the original hidden
        # input and the current-token suffix for attention.
        return grad_output, grad_output[:, history_tokens:, :], None


def exact_history_current_add(
    hidden_states: torch.Tensor,
    attention_output: torch.Tensor,
    history_tokens: int,
) -> torch.Tensor:
    """Return ``cat(history, current + attention)`` with exact derivatives."""

    return _ExactHistoryCurrentAdd.apply(
        hidden_states, attention_output, int(history_tokens)
    )


__all__ = ["SUPPORTED_WIDTH", "exact_history_current_add"]
