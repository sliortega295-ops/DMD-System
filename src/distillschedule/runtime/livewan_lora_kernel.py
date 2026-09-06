"""Plan-gated exact runtime specializations for pinned LiveWan LoRA."""

from __future__ import annotations

import hashlib
import inspect
import textwrap
from typing import Any


EXPECTED_FORWARD_SHA256 = (
    "70d00d496aa2bd8ea06d64e6b04af873644eff52539396f74ec586a4cbbe97a3"
)


def _build_fused_cast_add(torch: Any) -> tuple[Any, dict[str, Any]]:
    """Build the exact FP32-update/BF16-residual epilogue lazily.

    Keeping Triton behind the plan gate means the validated C11/K1 paths do
    not gain a new import or runtime dependency.  The kernel performs exactly
    the two roundings in ``update.to(bfloat16)`` followed by the BF16 residual
    add, while eliminating the intermediate tensor and one kernel launch.
    """

    import triton
    import triton.language as tl

    @triton.jit
    def cast_add_kernel(
        residual_ptr,
        update_ptr,
        output_ptr,
        n_elements: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        residual = tl.load(residual_ptr + offsets, mask=mask).to(tl.float32)
        update = (
            tl.load(update_ptr + offsets, mask=mask)
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        tl.store(
            output_ptr + offsets,
            (residual + update).to(tl.bfloat16),
            mask=mask,
        )

    class FusedCastAdd(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, residual: Any, update: Any) -> Any:
            if (
                not residual.is_cuda
                or not update.is_cuda
                or residual.dtype != torch.bfloat16
                or update.dtype != torch.float32
                or residual.shape != update.shape
                or not residual.is_contiguous()
                or not update.is_contiguous()
            ):
                raise RuntimeError(
                    "LiveWan fused LoRA epilogue contract changed: "
                    f"residual_device={residual.device}, update_device={update.device}, "
                    f"residual_dtype={residual.dtype}, update_dtype={update.dtype}, "
                    f"residual_shape={tuple(residual.shape)}, "
                    f"update_shape={tuple(update.shape)}, "
                    f"residual_contiguous={residual.is_contiguous()}, "
                    f"update_contiguous={update.is_contiguous()}"
                )
            output = torch.empty_like(residual)
            count = residual.numel()
            cast_add_kernel[(triton.cdiv(count, 256),)](
                residual, update, output, n_elements=count, BLOCK=256
            )
            return output

        @staticmethod
        def backward(ctx: Any, gradient: Any) -> tuple[Any, Any]:
            # This is the exact autograd contract of BF16 add followed by the
            # backward of FP32->BF16 ``to``.
            return gradient, gradient.to(torch.float32)

    return FusedCastAdd.apply, {
        "triton_version": triton.__version__,
        "triton_block_size": 256,
    }


def _source_sha256(value: Any) -> str:
    source = textwrap.dedent(inspect.getsource(value)).strip()
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def install_scale_one_lora_epilogue(
    torch: Any,
    *,
    fused_cast_add: bool = False,
    autocast_input_elision: bool = False,
    fused_min_numel: int = 8_000_000,
) -> dict[str, Any]:
    """Install a pinned scale-one LoRA specialization.

    The default keeps the FP32 LoRA operations in source order.  The optional
    input-elision mode is valid only when CUDA BF16 autocast immediately casts
    the nominal FP32 LoRA operands back to BF16; it fails closed otherwise.
    """

    from wanstreamer.lora import LoRALinear

    if fused_cast_add and autocast_input_elision:
        raise RuntimeError("LiveWan LoRA kernel candidates are mutually exclusive")
    requested_name = (
        "scale_one_fused_cast_add_v1"
        if fused_cast_add
        else (
            "scale_one_autocast_bf16_input_elision_v1"
            if autocast_input_elision
            else "scale_one_identity_epilogue_v1"
        )
    )
    if getattr(LoRALinear, "_distillschedule_scale_one_epilogue", False):
        installed_name = getattr(LoRALinear, "_distillschedule_kernel_name", None)
        if installed_name != requested_name:
            raise RuntimeError(
                f"LiveWan LoRA kernel already installed as {installed_name}, "
                f"requested {requested_name}"
            )
        return {
            "enabled": True,
            "name": installed_name,
            "source_forward_sha256": EXPECTED_FORWARD_SHA256,
            "class_forward_patched": True,
            "install_status": "ALREADY_INSTALLED",
        }
    actual = _source_sha256(LoRALinear.forward)
    if actual != EXPECTED_FORWARD_SHA256:
        raise RuntimeError(
            "LiveWan LoRALinear.forward source changed: "
            f"expected {EXPECTED_FORWARD_SHA256}, observed {actual}"
        )
    original = LoRALinear.forward
    fused_epilogue = None
    fused_receipt: dict[str, Any] = {}
    if fused_cast_add:
        if fused_min_numel != 8_000_000:
            raise RuntimeError("LiveWan fused LoRA epilogue threshold changed")
        fused_epilogue, fused_receipt = _build_fused_cast_add(torch)

    runtime_receipt = {
        "autocast_input_elision": autocast_input_elision,
        "autocast_input_elision_invocation_count": 0,
        "autocast_input_dtype_histogram": {},
        "autocast_contract_failure_count": 0,
    }

    def scale_one_forward(self: Any, x: Any) -> Any:
        y = self.base(x)
        if not self.enabled:
            return y
        lora_input = x
        if autocast_input_elision:
            try:
                cuda_autocast_enabled = torch.is_autocast_enabled("cuda")
            except TypeError:  # PyTorch before the device-qualified API.
                cuda_autocast_enabled = torch.is_autocast_enabled()
            if hasattr(torch, "get_autocast_dtype"):
                autocast_dtype = torch.get_autocast_dtype("cuda")
            else:
                autocast_dtype = torch.get_autocast_gpu_dtype()
            input_dtype = str(x.dtype)
            runtime_receipt["autocast_input_elision_invocation_count"] += 1
            histogram = runtime_receipt["autocast_input_dtype_histogram"]
            histogram[input_dtype] = int(histogram.get(input_dtype, 0)) + 1
            contract_ok = (
                x.is_cuda
                and x.dtype in {torch.bfloat16, torch.float32}
                and self.a.dtype == torch.float32
                and self.b.dtype == torch.float32
                and cuda_autocast_enabled
                and autocast_dtype == torch.bfloat16
            )
            if not contract_ok:
                runtime_receipt["autocast_contract_failure_count"] += 1
                raise RuntimeError(
                    "LiveWan autocast input-elision contract changed: "
                    f"input_device={x.device}, input_dtype={x.dtype}, "
                    f"a_dtype={self.a.dtype}, b_dtype={self.b.dtype}, "
                    f"cuda_autocast={cuda_autocast_enabled}, "
                    f"autocast_dtype={autocast_dtype}"
                )
        else:
            lora_input = x.to(self.a.dtype)
        h = torch.nn.functional.linear(lora_input, self.a)
        update_fp32 = torch.nn.functional.linear(h, self.b)
        if autocast_input_elision and (
            h.dtype != torch.bfloat16 or update_fp32.dtype != torch.bfloat16
        ):
            runtime_receipt["autocast_contract_failure_count"] += 1
            raise RuntimeError(
                "LiveWan autocast input-elision output contract changed: "
                f"h_dtype={h.dtype}, update_dtype={update_fp32.dtype}"
            )
        if fused_epilogue is not None and update_fp32.numel() >= fused_min_numel:
            return fused_epilogue(y, update_fp32)
        update = update_fp32.to(y.dtype)
        if self.scale == 1.0:
            return y + update
        return y + update * self.scale

    LoRALinear._distillschedule_original_forward = original
    LoRALinear.forward = scale_one_forward
    LoRALinear._distillschedule_scale_one_epilogue = True
    LoRALinear._distillschedule_kernel_name = requested_name
    runtime_receipt.update({
        "enabled": True,
        "name": requested_name,
        "source_forward_sha256": actual,
        "class_forward_patched": True,
        "install_status": "PASS",
        "fused_cast_add": fused_cast_add,
        "fused_min_numel": fused_min_numel if fused_cast_add else None,
        **fused_receipt,
    })
    return runtime_receipt


def uninstall_scale_one_lora_epilogue() -> None:
    """Test-only restoration; production workers exit with the patch installed."""

    from wanstreamer.lora import LoRALinear

    original = getattr(LoRALinear, "_distillschedule_original_forward", None)
    if original is None:
        return
    LoRALinear.forward = original
    delattr(LoRALinear, "_distillschedule_original_forward")
    delattr(LoRALinear, "_distillschedule_scale_one_epilogue")
    delattr(LoRALinear, "_distillschedule_kernel_name")
