"""Exact, plan-gated LiveWan Score RoPE fusion for persistent Score ranks."""

from __future__ import annotations

import hashlib
import inspect
import textwrap
from typing import Any


EXPECTED_APPLY_ROPE_SHA256 = (
    "8ab587e9f15cc969659609ec2f9cc79f4d4cf6fba468ae8c54267dc949224482"
)


def _source_sha256(value: Any) -> str:
    source = textwrap.dedent(inspect.getsource(value)).strip()
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def install_score_rope_kernel(torch: Any) -> dict[str, Any]:
    """Fuse the pinned FP32 complex RoPE and immediate BF16 conversion.

    The pinned Score path first materializes the full Q/K tensor in FP32,
    rotates it as ``complex64``, materializes the real FP32 result, and then
    immediately casts it back to BF16 for attention.  This specialization reads
    BF16, performs the same FP32 operations in the same rounding order, and
    stores BF16 directly.  Backward applies the conjugate rotation in the same
    order and returns the exact BF16 input gradient.

    Unsupported source or runtime contracts fail closed.  There is no eager
    fallback hidden behind this installation.
    """

    from wanstreamer import blockcausal

    requested_name = "score_bf16_rope_fusion_v1"
    if getattr(blockcausal, "_distillschedule_rope_kernel", False):
        installed = getattr(blockcausal, "_distillschedule_rope_kernel_name", None)
        if installed != requested_name:
            raise RuntimeError(
                f"LiveWan Score RoPE kernel already installed as {installed}, "
                f"requested {requested_name}"
            )
        return getattr(blockcausal, "_distillschedule_rope_receipt")

    actual = _source_sha256(blockcausal.apply_rope)
    if actual != EXPECTED_APPLY_ROPE_SHA256:
        raise RuntimeError(
            "LiveWan blockcausal.apply_rope source changed: "
            f"expected {EXPECTED_APPLY_ROPE_SHA256}, observed {actual}"
        )

    import triton
    import triton.language as tl

    @triton.jit
    def mul_rn(left, right):
        return tl.inline_asm_elementwise(
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            [left, right],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def fma_rn(left, right, addend):
        return tl.inline_asm_elementwise(
            "fma.rn.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            [left, right, addend],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def rope_forward_kernel(
        input_ptr,
        table_ptr,
        output_ptr,
        pair_count,
        sequence: tl.constexpr,
        heads: tl.constexpr,
        dimension_pairs: tl.constexpr,
        table_batched: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pair = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = pair < pair_count
        feature = pair % dimension_pairs
        token_linear = pair // (heads * dimension_pairs)
        token = token_linear % sequence
        batch = token_linear // sequence
        table_batch = batch if table_batched else 0
        table_pair = (table_batch * sequence + token) * dimension_pairs + feature
        index = pair * 2

        input_real = tl.load(input_ptr + index, mask=mask).to(tl.float32)
        input_imag = tl.load(input_ptr + index + 1, mask=mask).to(tl.float32)
        table_real = tl.load(table_ptr + table_pair * 2, mask=mask).to(tl.float32)
        table_imag = tl.load(table_ptr + table_pair * 2 + 1, mask=mask).to(
            tl.float32
        )

        # Match c10::complex<float> multiplication: NVCC contracts the first
        # product with the rounded second product.  The explicit instruction
        # order is bitwise-equal to the pinned eager operator after BF16 store.
        output_real = fma_rn(
            input_real, table_real, -mul_rn(input_imag, table_imag)
        )
        output_imag = fma_rn(
            input_real, table_imag, mul_rn(input_imag, table_real)
        )
        tl.store(output_ptr + index, output_real, mask=mask)
        tl.store(output_ptr + index + 1, output_imag, mask=mask)

    @triton.jit
    def rope_backward_kernel(
        gradient_ptr,
        table_ptr,
        input_gradient_ptr,
        pair_count,
        sequence: tl.constexpr,
        heads: tl.constexpr,
        dimension_pairs: tl.constexpr,
        table_batched: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pair = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = pair < pair_count
        feature = pair % dimension_pairs
        token_linear = pair // (heads * dimension_pairs)
        token = token_linear % sequence
        batch = token_linear // sequence
        table_batch = batch if table_batched else 0
        table_pair = (table_batch * sequence + token) * dimension_pairs + feature
        index = pair * 2

        gradient_real = tl.load(gradient_ptr + index, mask=mask).to(tl.float32)
        gradient_imag = tl.load(gradient_ptr + index + 1, mask=mask).to(tl.float32)
        table_real = tl.load(table_ptr + table_pair * 2, mask=mask).to(tl.float32)
        table_imag = tl.load(table_ptr + table_pair * 2 + 1, mask=mask).to(
            tl.float32
        )

        # Multiply by conjugate(table), retaining the eager rounding order.
        input_gradient_real = fma_rn(
            gradient_real,
            table_real,
            mul_rn(gradient_imag, table_imag),
        )
        input_gradient_imag = fma_rn(
            -gradient_real,
            table_imag,
            mul_rn(gradient_imag, table_real),
        )
        tl.store(input_gradient_ptr + index, input_gradient_real, mask=mask)
        tl.store(input_gradient_ptr + index + 1, input_gradient_imag, mask=mask)

    receipt: dict[str, Any] = {
        "enabled": True,
        "name": requested_name,
        "source_apply_rope_sha256": actual,
        "function_patched": True,
        "install_status": "PASS",
        "triton_version": triton.__version__,
        "triton_block_size": 512,
        "fp32_rounding_order": "mul_then_first_product_fma_rn",
        "invocation_count": 0,
        "backward_invocation_count": 0,
        "element_count": 0,
        "backward_element_count": 0,
        "shape_histogram": {},
        "table_rank_histogram": {},
        "gradient_contiguous_copy_count": 0,
        "contract_failure_count": 0,
    }

    def validate(value: Any, table: Any, *, operation: str) -> tuple[int, int, int, int]:
        try:
            if (
                not value.is_cuda
                or not table.is_cuda
                or value.dtype != torch.bfloat16
                or table.dtype != torch.complex64
                or value.ndim != 4
                or table.ndim not in {3, 4}
                or not value.is_contiguous()
                or not table.is_contiguous()
                or table.requires_grad
            ):
                raise RuntimeError("device, dtype, rank, contiguity, or gradient mismatch")
            batch, sequence, heads, dimension = (int(v) for v in value.shape)
            if dimension % 2:
                raise RuntimeError("odd head dimension")
            expected_shared = (sequence, 1, dimension // 2)
            expected_batched = (batch, sequence, 1, dimension // 2)
            if tuple(table.shape) not in {expected_shared, expected_batched}:
                raise RuntimeError(
                    f"table shape {tuple(table.shape)} not in "
                    f"{{{expected_shared}, {expected_batched}}}"
                )
            return batch, sequence, heads, dimension
        except Exception as error:
            receipt["contract_failure_count"] += 1
            raise RuntimeError(
                f"LiveWan Score RoPE {operation} contract changed: "
                f"value={value.device}/{value.dtype}/{tuple(value.shape)}/"
                f"stride={tuple(value.stride())}/contiguous={value.is_contiguous()}, "
                f"table={table.device}/{table.dtype}/{tuple(table.shape)}/"
                f"stride={tuple(table.stride())}/contiguous={table.is_contiguous()}/"
                f"requires_grad={table.requires_grad}: {error}"
            ) from error

    class ExactScoreRoPE(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, value: Any, table: Any) -> Any:
            batch, sequence, heads, dimension = validate(
                value, table, operation="forward"
            )
            output = torch.empty_like(value)
            pair_count = value.numel() // 2
            rope_forward_kernel[(triton.cdiv(pair_count, 512),)](
                value,
                torch.view_as_real(table),
                output,
                pair_count,
                sequence=sequence,
                heads=heads,
                dimension_pairs=dimension // 2,
                table_batched=table.ndim == 4,
                BLOCK=512,
            )
            ctx.save_for_backward(table)
            receipt["invocation_count"] += 1
            receipt["element_count"] += int(value.numel())
            shape_key = "x".join(str(part) for part in value.shape)
            receipt["shape_histogram"][shape_key] = (
                int(receipt["shape_histogram"].get(shape_key, 0)) + 1
            )
            rank_key = str(int(table.ndim))
            receipt["table_rank_histogram"][rank_key] = (
                int(receipt["table_rank_histogram"].get(rank_key, 0)) + 1
            )
            return output

        @staticmethod
        def backward(ctx: Any, gradient: Any) -> tuple[Any, None]:
            (table,) = ctx.saved_tensors
            if not gradient.is_contiguous():
                receipt["gradient_contiguous_copy_count"] += 1
                gradient = gradient.contiguous()
            _, sequence, heads, dimension = validate(
                gradient, table, operation="backward"
            )
            input_gradient = torch.empty_like(gradient)
            pair_count = gradient.numel() // 2
            rope_backward_kernel[(triton.cdiv(pair_count, 512),)](
                gradient,
                torch.view_as_real(table),
                input_gradient,
                pair_count,
                sequence=sequence,
                heads=heads,
                dimension_pairs=dimension // 2,
                table_batched=table.ndim == 4,
                BLOCK=512,
            )
            receipt["backward_invocation_count"] += 1
            receipt["backward_element_count"] += int(gradient.numel())
            return input_gradient, None

    original = blockcausal.apply_rope

    def fused_apply_rope(value: Any, table: Any) -> Any:
        return ExactScoreRoPE.apply(value, table)

    blockcausal._distillschedule_original_apply_rope = original
    blockcausal.apply_rope = fused_apply_rope
    blockcausal._distillschedule_rope_kernel = True
    blockcausal._distillschedule_rope_kernel_name = requested_name
    blockcausal._distillschedule_rope_receipt = receipt
    return receipt


def uninstall_score_rope_kernel() -> None:
    """Test-only restoration; production workers exit with the patch installed."""

    from wanstreamer import blockcausal

    original = getattr(blockcausal, "_distillschedule_original_apply_rope", None)
    if original is None:
        return
    blockcausal.apply_rope = original
    for name in (
        "_distillschedule_original_apply_rope",
        "_distillschedule_rope_kernel",
        "_distillschedule_rope_kernel_name",
        "_distillschedule_rope_receipt",
    ):
        delattr(blockcausal, name)


__all__ = [
    "EXPECTED_APPLY_ROPE_SHA256",
    "install_score_rope_kernel",
    "uninstall_score_rope_kernel",
]
