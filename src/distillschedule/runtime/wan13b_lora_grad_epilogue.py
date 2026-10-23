"""Default-off exact LoRA epilogue fusion for grad forward/checkpoint replay.

The existing Generator no-grad specialization is untouched. This candidate
changes only PEFT's scale/add/final-cast pointwise implementation, never its
base/LoRA GEMMs, dropout, parameter dtype, or checkpoint policy.
"""
from __future__ import annotations

import functools
import os
from typing import Any

from .wan13b_lora_cast import PEFT_0181_LINEAR_FORWARD_SHA256, source_sha256


GRAD_EPILOGUE_ENV = "DISTILLSCHEDULE_WAN13B_GRAD_LORA_FUSED_EPILOGUE"
_FUNCTION: Any = None


def grad_epilogue_requested() -> bool:
    value = os.environ.get(GRAD_EPILOGUE_ENV, "0")
    if value not in {"0", "1"}:
        raise ValueError(f"{GRAD_EPILOGUE_ENV} must be 0 or 1")
    return value == "1"


def runtime_precision_contract() -> dict[str, str]:
    """Capture the effective policy outside timers, without changing it."""
    import torch

    return {
        "tf32_cublas_override": os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "NOT_SET"),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": str(torch.backends.cuda.matmul.allow_tf32).lower(),
        "cudnn_allow_tf32": str(torch.backends.cudnn.allow_tf32).lower(),
    }


def validate_grad_epilogue_receipts(build: dict[str, Any], *, requested: bool) -> dict[str, Any]:
    result = {}
    for role in ("generator", "critic"):
        receipt = build[role]["metadata"]["grad_lora_fused_epilogue"]
        if receipt.get("requested") is not requested or receipt.get("enabled") is not requested:
            raise AssertionError(f"{role}: grad epilogue request/runtime mismatch")
        if requested and int(receipt.get("fused_invocation_count", 0)) <= 0:
            raise AssertionError(f"{role}: requested grad epilogue never executed")
        result[role] = receipt
    return result


def make_epilogue_autograd_function(forward_kernel: Any) -> Any:
    """Factory also permits CPU tests with the exact eager forward reference."""
    import torch

    class ExactEpilogue(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, base: Any, update: Any, scale: float) -> Any:
            if base.dtype is not torch.bfloat16 or update.dtype is not torch.float32:
                raise ValueError("grad epilogue requires BF16 base and FP32 update")
            if base.shape != update.shape:
                raise ValueError("grad epilogue shape mismatch")
            ctx.scale = float(scale)
            return forward_kernel(base, update, ctx.scale)

        @staticmethod
        def backward(ctx: Any, grad_output: Any) -> tuple[Any, Any, None]:
            # Native chain: BF16->FP32 cast backward, add backward, scale
            # backward, and promoted BF16-base add's cast back to BF16.
            # The first and last casts compose to identity on this BF16 grad.
            grad_base = grad_output if ctx.needs_input_grad[0] else None
            grad_update = (
                grad_output.float() * ctx.scale if ctx.needs_input_grad[1] else None
            )
            return grad_base, grad_update, None

    return ExactEpilogue


def fused_lora_epilogue_with_grad(base: Any, update: Any, scale: float) -> Any:
    global _FUNCTION
    if _FUNCTION is None:
        from .wan13b_lora_epilogue import fused_lora_epilogue

        _FUNCTION = make_epilogue_autograd_function(fused_lora_epilogue)
    return _FUNCTION.apply(base, update, float(scale))


def install_grad_lora_fused_epilogue(model: Any, *, role: str) -> dict[str, Any]:
    requested = grad_epilogue_requested()
    receipt: dict[str, Any] = {
        "requested": requested,
        "enabled": False,
        "role": role,
        "scope": "GRAD_ENABLED_LINEAR_AND_CHECKPOINT_REPLAY_ONLY",
        "fused_invocation_count": 0,
        "reference_fallback_count": 0,
        "no_grad_passthrough_count": 0,
        "semantic_contract": "SAME_FP32_MUL_ADD_BF16_STORE_AND_NATIVE_INPUT_GRADIENTS",
    }
    model._distillschedule_grad_lora_fused_epilogue = receipt
    if not requested:
        receipt["disabled_reason"] = "NOT_REQUESTED"
        return receipt
    if role not in {"generator", "critic"}:
        raise ValueError("grad LoRA epilogue requires Generator or Critic role")
    import torch
    from peft.tuners.lora.layer import Linear, VARIANT_KWARG_KEYS

    observed = source_sha256(Linear.forward)
    receipt["peft_linear_forward_sha256"] = observed
    if observed != PEFT_0181_LINEAR_FORWARD_SHA256:
        raise RuntimeError("PEFT Linear.forward changed; grad epilogue fails closed")
    linears = [child for child in model.modules() if isinstance(child, Linear)]
    if not linears:
        raise RuntimeError("grad epilogue found no PEFT Linear modules")
    for child in linears:
        original = child.forward

        @functools.wraps(original)
        def dispatch(x: Any, *args: Any, _child: Any = child,
                     _original: Any = original, **kwargs: Any) -> Any:
            if not torch.is_grad_enabled():
                receipt["no_grad_passthrough_count"] += 1
                return _original(x, *args, **kwargs)
            if _child.disable_adapters or _child.merged:
                return _original(x, *args, **kwargs)
            adapters = tuple(_child.active_adapters)
            if (len(adapters) != 1 or adapters[0] not in _child.lora_A
                    or adapters[0] in _child.lora_variant
                    or kwargs.get("adapter_names") is not None
                    or any(kwargs.get(key) is not None for key in VARIANT_KWARG_KEYS)):
                raise RuntimeError("grad epilogue requires one vanilla active adapter")
            _child._check_forward_args(x, *args, **kwargs)
            kwargs.pop("adapter_names", None)
            for key in VARIANT_KWARG_KEYS:
                kwargs.pop(key, None)
            adapter = adapters[0]
            base = _child.base_layer(x, *args, **kwargs)
            cast_x = _child._cast_input_dtype(x, _child.lora_A[adapter].weight.dtype)
            update = _child.lora_B[adapter](
                _child.lora_A[adapter](_child.lora_dropout[adapter](cast_x))
            )
            if (base.dtype is torch.bfloat16 and update.dtype is torch.float32
                    and base.shape == update.shape and base.is_contiguous()
                    and update.is_contiguous()):
                result = fused_lora_epilogue_with_grad(base, update, _child.scaling[adapter])
                receipt["fused_invocation_count"] += 1
                return result
            receipt["reference_fallback_count"] += 1
            return (base + update * _child.scaling[adapter]).to(base.dtype)

        child.forward = dispatch
    receipt.update(enabled=True, instrumented_module_count=len(linears))
    return receipt


__all__ = ["GRAD_EPILOGUE_ENV", "grad_epilogue_requested",
           "runtime_precision_contract", "validate_grad_epilogue_receipts",
           "make_epilogue_autograd_function", "fused_lora_epilogue_with_grad",
           "install_grad_lora_fused_epilogue"]
