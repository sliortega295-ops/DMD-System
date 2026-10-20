"""Opt-in, shape-qualified BF16 LoRA-B GEMM/add, with two BF16 rounds.

This is the N37 single-tile primitive, not an approximate LoRA merge. The GEMM
result is rounded to BF16 before its FP32 add with the BF16 base result. Different
cuBLAS/Triton accumulation order still requires the unchanged numerical gate.
Only first-order backward is supported; the current GAN R1 is finite difference.
Import is lazy from the explicitly enabled existing PEFT wrapper.
"""
from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl


VALIDATED_LORA_B_SHAPES = ((11040, 5120), (13680, 5120), (11040, 13824))
IMPLEMENTATION = (
    "TRITON_BF16_RANK256_B_GEMM_RN_BF16_THEN_ADD_RN_F32_"
    "BF16_STORE_REFERENCE_MM_BACKWARD"
)


@triton.jit
def _b_add(
    H, B, Base, Y, M: tl.constexpr, N: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    mi = tl.program_id(0) * BM + tl.arange(0, BM)
    ni = tl.program_id(1) * BN + tl.arange(0, BN)
    ki = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0.0, tl.float32)
    for kb in range(0, 256, BK):
        k = ki + kb
        a = tl.load(H + mi[:, None] * 256 + k[None, :],
                    mask=mi[:, None] < M, other=0.0)
        b = tl.load(B + ni[None, :] * 256 + k[:, None],
                    mask=ni[None, :] < N, other=0.0)
        acc += tl.dot(a, b, out_dtype=tl.float32)
    # Required quantization barrier: never replace with beta=1/one BF16 round.
    rounded = acc.to(tl.bfloat16).to(tl.float32)
    base = tl.load(Base + mi[:, None] * N + ni[None, :],
                   mask=(mi[:, None] < M) & (ni[None, :] < N), other=0.0).to(tl.float32)
    value = tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;", "=f,f,f", [rounded, base],
        dtype=tl.float32, is_pure=True, pack=1,
    )
    tl.store(Y + mi[:, None] * N + ni[None, :], value,
             mask=(mi[:, None] < M) & (ni[None, :] < N))


def _raw(
    h: torch.Tensor, b: torch.Tensor, base: torch.Tensor, *, inplace: bool,
    on_kernel_launch: Any | None = None,
) -> torch.Tensor:
    if any(not t.is_cuda or t.dtype != torch.bfloat16 or not t.is_contiguous()
           for t in (h, b, base)):
        raise RuntimeError("N37 requires contiguous CUDA BF16 H/B/base")
    if h.ndim != 2 or b.ndim != 2 or base.ndim != 2:
        raise RuntimeError("N37 raw primitive requires flattened rank2 inputs")
    if h.shape[1] != 256 or b.shape[1] != 256 or base.shape != (h.shape[0], b.shape[0]):
        raise RuntimeError("N37 rank256/output shape mismatch")
    if h.device != b.device or h.device != base.device:
        raise RuntimeError("N37 device mismatch")
    if (int(h.shape[0]), int(b.shape[0])) not in VALIDATED_LORA_B_SHAPES:
        raise RuntimeError("N37 shape was not admitted by the measured primitive")
    if torch.cuda.get_device_capability(h.device) != (8, 0):
        raise RuntimeError("N37 is admitted only for A100 SM80")
    if inplace and torch.is_grad_enabled():
        raise RuntimeError("N37 in-place write is no-grad-only")
    output = base if inplace else torch.empty_like(base)
    _b_add[(triton.cdiv(h.shape[0], 64), triton.cdiv(b.shape[0], 128))](
        h, b, base, output, h.shape[0], b.shape[0], 64, 128, 64,
        num_warps=8, num_stages=3, enable_fp_fusion=False,
    )
    if on_kernel_launch is not None:
        # Before Function.apply's saved-tensor hooks can early-stop replay.
        on_kernel_launch()
    return output


class _BAdd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any, h: torch.Tensor, b: torch.Tensor, base: torch.Tensor,
        on_kernel_launch: Any,
    ) -> torch.Tensor:
        need_h, need_b, _need_base = ctx.needs_input_grad[:3]
        # Match Linear's input/weight gradient requirements. GAN-G freezes B
        # but still propagates dH: do not save H or compute an unneeded dB.
        ctx.saved_h = bool(need_b)
        ctx.saved_b = bool(need_h)
        ctx.save_for_backward(*([h] if need_b else []), *([b] if need_h else []))
        return _raw(h, b, base, inplace=False, on_kernel_launch=on_kernel_launch)

    @staticmethod
    def backward(ctx: Any, g: torch.Tensor) -> tuple[Any, Any, Any, None]:
        if torch.is_grad_enabled():
            raise RuntimeError("N37 supports first-order backward only")
        saved = iter(ctx.saved_tensors)
        h = next(saved) if ctx.saved_h else None
        b = next(saved) if ctx.saved_b else None
        need_h, need_b, need_base = ctx.needs_input_grad[:3]
        d_h = g @ b if need_h else None
        d_b = g.transpose(0, 1) @ h if need_b else None
        return d_h, d_b, g if need_base else None, None


def fused_lora_b_gemm_epilogue(
    hidden: torch.Tensor, weight: torch.Tensor, base_result: torch.Tensor,
    *, on_kernel_launch: Any | None = None,
) -> torch.Tensor:
    """Called from the LoRA-B module's normal __call__, after its hooks.

    Do not copy/cast/pack implicitly. Metadata validation is repeated after
    module hooks so the actual unsharded weight, not only a plan, is checked.
    """
    if hidden.ndim < 2 or base_result.ndim != hidden.ndim:
        raise RuntimeError("N37 leading-dimension contract changed")
    if hidden.shape[:-1] != base_result.shape[:-1]:
        raise RuntimeError("N37 leading shapes mismatch")
    if not hidden.is_contiguous() or not base_result.is_contiguous():
        raise RuntimeError("N37 cannot introduce a layout copy")
    h = hidden.view(-1, hidden.shape[-1])
    base = base_result.view(-1, base_result.shape[-1])
    if torch.is_grad_enabled():
        result = _BAdd.apply(h, weight, base, on_kernel_launch)
    else:
        result = _raw(h, weight, base, inplace=True, on_kernel_launch=on_kernel_launch)
    return result.view(base_result.shape)


def runtime_identity() -> dict[str, Any]:
    return {
        "implementation": IMPLEMENTATION,
        "triton_version": str(triton.__version__), "torch_version": str(torch.__version__),
        "validated_shapes": [list(shape) for shape in VALIDATED_LORA_B_SHAPES],
        "rank": 256, "tile": [64, 128, 64], "warps": 8, "stages": 3,
        "source_primitive_sha256": "ef0923e9d587153f1ddf3c07110a70cfcf059e8017485d503fb7a876a95ca100",
        "first_order_only": True, "parameter_compute_dtype": "torch.bfloat16",
        "master_parameter_cast_or_reduction_changed": False,
        "matmul_flags_observed_not_modified": {
            "allow_bf16_reduced_precision_reduction": bool(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction),
            "allow_fp16_reduced_precision_reduction": bool(torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction),
            "allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        },
    }


__all__ = ["IMPLEMENTATION", "VALIDATED_LORA_B_SHAPES", "fused_lora_b_gemm_epilogue", "runtime_identity"]
