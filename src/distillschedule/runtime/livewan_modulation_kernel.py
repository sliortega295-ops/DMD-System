"""Exact, plan-gated LiveWan modulation fusion for persistent Score ranks."""

from __future__ import annotations

import hashlib
import inspect
import textwrap
from typing import Any


EXPECTED_MODULATE_SHA256 = (
    "f244152360d40d6745e28bed4add081604312d04fe2f342842a32432f4bd20ad"
)


def _source_sha256(value: Any) -> str:
    source = textwrap.dedent(inspect.getsource(value)).strip()
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def install_score_modulation_kernel(torch: Any) -> dict[str, Any]:
    """Fuse the pinned BF16 LayerNorm-output modulation chain.

    The upstream operation is::

        normalized = norm(x).float()
        output = normalized * (1 + scale) + shift

    ``norm`` remains an ordinary PyTorch call so its exact backward and module
    semantics are untouched.  The Triton function consumes its BF16 output and
    reproduces the two FP32 pointwise operations in one launch.  Critic/Teacher
    modulation values are frozen, so backward only returns the exact BF16
    gradient for the normalized input.  Every unsupported runtime condition
    fails closed rather than falling back to different math.
    """

    from wanstreamer import blockcausal

    requested_name = "score_bf16_modulation_fusion_v1"
    if getattr(blockcausal, "_distillschedule_modulation_kernel", False):
        installed = getattr(blockcausal, "_distillschedule_modulation_kernel_name", None)
        if installed != requested_name:
            raise RuntimeError(
                f"LiveWan modulation kernel already installed as {installed}, "
                f"requested {requested_name}"
            )
        return getattr(blockcausal, "_distillschedule_modulation_receipt")

    actual = _source_sha256(blockcausal._modulate)
    if actual != EXPECTED_MODULATE_SHA256:
        raise RuntimeError(
            "LiveWan _modulate source changed: "
            f"expected {EXPECTED_MODULATE_SHA256}, observed {actual}"
        )

    import triton
    import triton.language as tl

    @triton.jit
    def modulation_kernel(
        normalized_ptr,
        shift_ptr,
        scale_ptr,
        output_ptr,
        n_elements,
        shift_stride0,
        scale_stride0,
        sequence: tl.constexpr,
        dimension: tl.constexpr,
        modulation_rows: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        feature = offsets % dimension
        token = offsets // dimension
        batch = token // sequence
        shift_index = tl.where(
            modulation_rows == 1, feature, batch * shift_stride0 + feature
        )
        scale_index = tl.where(
            modulation_rows == 1, feature, batch * scale_stride0 + feature
        )
        normalized = tl.load(normalized_ptr + offsets, mask=mask).to(tl.float32)
        scale = tl.load(scale_ptr + scale_index, mask=mask).to(tl.float32)
        shift = tl.load(shift_ptr + shift_index, mask=mask).to(tl.float32)
        one = tl.full((BLOCK,), 1.0, tl.float32)
        # ``1 + scale`` is first materialized in BF16 by pinned eager PyTorch.
        factor = (one + scale).to(tl.bfloat16).to(tl.float32)
        product = tl.inline_asm_elementwise(
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            [normalized, factor],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        value = tl.inline_asm_elementwise(
            "add.rn.f32 $0, $1, $2;",
            "=f,f,f",
            [product, shift],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        tl.store(output_ptr + offsets, value, mask=mask)

    receipt: dict[str, Any] = {
        "enabled": True,
        "name": requested_name,
        "source_modulate_sha256": actual,
        "function_patched": True,
        "install_status": "PASS",
        "triton_version": triton.__version__,
        "triton_block_size": 256,
        "invocation_count": 0,
        "backward_invocation_count": 0,
        "element_count": 0,
        "batch_histogram": {},
        "dimension_histogram": {},
        "observed_contract": None,
    }

    class ExactModulation(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, normalized: Any, shift: Any, scale: Any) -> Any:
            if (
                not normalized.is_cuda
                or not shift.is_cuda
                or not scale.is_cuda
                or normalized.dtype != torch.bfloat16
                or shift.dtype != torch.bfloat16
                or scale.dtype != torch.bfloat16
                or normalized.ndim != 3
                or shift.ndim != 3
                or shift.shape != scale.shape
                or shift.shape[0] not in {1, normalized.shape[0]}
                or shift.shape[1] != 1
                or shift.shape[2] != normalized.shape[2]
                or not normalized.is_contiguous()
                or shift.stride(2) != 1
                or scale.stride(2) != 1
                or shift.requires_grad
                or scale.requires_grad
            ):
                raise RuntimeError(
                    "LiveWan modulation fusion contract changed: "
                    f"normalized={normalized.device}/{normalized.dtype}/"
                    f"{tuple(normalized.shape)}/contiguous={normalized.is_contiguous()}, "
                    f"shift={shift.device}/{shift.dtype}/{tuple(shift.shape)}/"
                    f"stride={tuple(shift.stride())}/requires_grad={shift.requires_grad}, "
                    f"scale={scale.device}/{scale.dtype}/{tuple(scale.shape)}/"
                    f"stride={tuple(scale.stride())}/requires_grad={scale.requires_grad}"
                )
            batch, sequence, dimension = normalized.shape
            output = torch.empty_like(normalized, dtype=torch.float32)
            count = normalized.numel()
            modulation_kernel[(triton.cdiv(count, 256),)](
                normalized,
                shift,
                scale,
                output,
                count,
                shift.stride(0),
                scale.stride(0),
                sequence=sequence,
                dimension=dimension,
                modulation_rows=shift.shape[0],
                BLOCK=256,
            )
            ctx.save_for_backward(scale)
            receipt["invocation_count"] += 1
            receipt["element_count"] += int(count)
            batch_key = str(int(batch))
            dimension_key = str(int(dimension))
            receipt["batch_histogram"][batch_key] = (
                int(receipt["batch_histogram"].get(batch_key, 0)) + 1
            )
            receipt["dimension_histogram"][dimension_key] = (
                int(receipt["dimension_histogram"].get(dimension_key, 0)) + 1
            )
            contract = {
                "normalized_dtype": str(normalized.dtype),
                "shift_dtype": str(shift.dtype),
                "scale_dtype": str(scale.dtype),
                "output_dtype": str(output.dtype),
                "per_frame": False,
            }
            if (
                receipt["observed_contract"] is not None
                and receipt["observed_contract"] != contract
            ):
                raise RuntimeError("LiveWan modulation runtime contract changed within run")
            receipt["observed_contract"] = contract
            return output

        @staticmethod
        def backward(ctx: Any, gradient: Any) -> tuple[Any, None, None]:
            (scale,) = ctx.saved_tensors
            receipt["backward_invocation_count"] += 1
            # Exact eager backward of ``normalized.float() * (1 + scale)``.
            return (gradient * (1 + scale)).to(torch.bfloat16), None, None

    original = blockcausal._modulate

    def fused_modulate(
        x: Any,
        norm: Any,
        shift: Any,
        scale: Any,
        per_frame: bool,
        n_frames: int,
    ) -> Any:
        if per_frame:
            raise RuntimeError("Score modulation fusion does not permit per-frame mode")
        if any(parameter.requires_grad for parameter in norm.parameters()):
            raise RuntimeError("Score modulation fusion requires frozen normalization")
        normalized = norm(x)
        return ExactModulation.apply(normalized, shift, scale)

    blockcausal._distillschedule_original_modulate = original
    blockcausal._modulate = fused_modulate
    blockcausal._distillschedule_modulation_kernel = True
    blockcausal._distillschedule_modulation_kernel_name = requested_name
    blockcausal._distillschedule_modulation_receipt = receipt
    return receipt


def uninstall_score_modulation_kernel() -> None:
    """Test-only restoration; production workers exit with the patch installed."""

    from wanstreamer import blockcausal

    original = getattr(blockcausal, "_distillschedule_original_modulate", None)
    if original is None:
        return
    blockcausal._modulate = original
    for name in (
        "_distillschedule_original_modulate",
        "_distillschedule_modulation_kernel",
        "_distillschedule_modulation_kernel_name",
        "_distillschedule_modulation_receipt",
    ):
        delattr(blockcausal, name)


__all__ = [
    "EXPECTED_MODULATE_SHA256",
    "install_score_modulation_kernel",
    "uninstall_score_modulation_kernel",
]
