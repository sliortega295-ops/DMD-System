"""Exact, opt-in no-grad LoRA epilogue fusion for Helios rollouts.

Only Generator calls made with autograd disabled enter this path.  Gradient
calls and every Critic call retain PEFT's source-pinned implementation.  The
fused kernel preserves PEFT's FP32 multiply, FP32 add, and final BF16 rounding
while replacing three pointwise kernels with one.
"""
from __future__ import annotations

import contextvars
import functools
import hashlib
import inspect
from typing import Any, Iterable


PEFT_0181_LINEAR_FORWARD_SHA256 = (
    "ec0a80f5c5ce05f5dd90027952bb5d29aaf910e9d2ab40486a90e8a03c7e1cd9"
)


def _torch() -> Any:
    import torch

    return torch


def source_sha256(function: Any) -> str:
    return hashlib.sha256(inspect.getsource(function).encode()).hexdigest()


def install_generator_nograd_lora_fused_epilogue(
    model: Any,
    *,
    enabled: bool,
    role: str,
    linear_type: type[Any] | None = None,
    variant_kwarg_keys: Iterable[str] | None = None,
    fused_epilogue: Any | None = None,
) -> dict[str, Any]:
    """Install the bounded Helios Generator no-grad replacement."""

    receipt: dict[str, Any] = {
        "requested": bool(enabled),
        "enabled": False,
        "role": str(role),
        "scope": "GENERATOR_NOGRAD_MODEL_FORWARD_ONLY",
        "model_call_count": 0,
        "fused_invocation_count": 0,
        "fused_element_count": 0,
        "reference_fallback_count": 0,
        "reference_fallback_reasons": {},
        "reference_pointwise_kernel_count_per_invocation": 3,
        "candidate_kernel_count_per_invocation": 1,
        "semantic_contract": (
            "FP32_MUL_RN_THEN_FP32_ADD_RN_THEN_BF16_RN_STORE;"
            "GRAD_AND_CRITIC_PATHS_UNCHANGED"
        ),
    }
    model._distillschedule_helios_nograd_lora_epilogue = receipt
    if not enabled:
        receipt["disabled_reason"] = "NOT_REQUESTED"
        return receipt
    if role != "generator":
        receipt["disabled_reason"] = "GENERATOR_ONLY"
        return receipt

    injected_type = linear_type is not None
    if linear_type is None:
        from peft.tuners.lora.layer import Linear, VARIANT_KWARG_KEYS

        linear_type = Linear
        variant_kwarg_keys = tuple(VARIANT_KWARG_KEYS)
    else:
        variant_kwarg_keys = tuple(variant_kwarg_keys or ())
    observed_sha = source_sha256(linear_type.forward)
    receipt["peft_linear_forward_sha256"] = observed_sha
    receipt["expected_peft_linear_forward_sha256"] = (
        PEFT_0181_LINEAR_FORWARD_SHA256 if not injected_type else "TEST_INJECTED"
    )
    if not injected_type and observed_sha != PEFT_0181_LINEAR_FORWARD_SHA256:
        raise RuntimeError(
            "installed PEFT Linear.forward source changed: "
            f"{observed_sha} != {PEFT_0181_LINEAR_FORWARD_SHA256}"
        )

    if fused_epilogue is None:
        from distillschedule.runtime.helios_lora_epilogue_kernel import (
            fused_lora_epilogue,
            runtime_identity,
        )

        fused_epilogue = fused_lora_epilogue
        receipt["runtime_identity"] = runtime_identity()
    else:
        receipt["runtime_identity"] = {"implementation": "TEST_INJECTED"}

    active_call: contextvars.ContextVar[bool] = contextvars.ContextVar(
        "helios_generator_nograd_lora_fused_epilogue", default=False
    )
    named_linears = [
        (name, child)
        for name, child in model.named_modules()
        if isinstance(child, linear_type)
    ]
    if not named_linears:
        raise RuntimeError("Helios Generator has no PEFT Linear modules")
    receipt["instrumented_module_count"] = len(named_linears)

    for module_name, child in named_linears:
        original_forward = child.forward

        @functools.wraps(original_forward)
        def fused_forward(
            x: Any,
            *args: Any,
            _child: Any = child,
            _module_name: str = module_name,
            _original_forward: Any = original_forward,
            **kwargs: Any,
        ) -> Any:
            del _module_name
            if not active_call.get():
                return _original_forward(x, *args, **kwargs)

            _child._check_forward_args(x, *args, **kwargs)
            adapter_names = kwargs.pop("adapter_names", None)
            variant_kwargs = {
                key: kwargs.pop(key, None) for key in variant_kwarg_keys
            }
            if _child.disable_adapters or adapter_names is not None or _child.merged:
                raise RuntimeError(
                    "fused Helios LoRA epilogue requires enabled, unmerged, "
                    "non-mixed adapters"
                )
            active_adapters = tuple(_child.active_adapters)
            if len(active_adapters) != 1:
                raise RuntimeError("fused Helios LoRA epilogue requires one adapter")
            adapter = active_adapters[0]
            if adapter not in _child.lora_A or adapter in _child.lora_variant:
                raise RuntimeError("Helios LoRA adapter contract changed")
            if any(value is not None for value in variant_kwargs.values()):
                raise RuntimeError("LoRA variant kwargs are outside the fused contract")

            result = _child.base_layer(x, *args, **kwargs)
            result_dtype = result.dtype
            lora_a = _child.lora_A[adapter]
            lora_b = _child.lora_B[adapter]
            cast_x = _child._cast_input_dtype(x, lora_a.weight.dtype)
            update = lora_b(lora_a(_child.lora_dropout[adapter](cast_x)))
            torch = _torch()
            fallback_reason = None
            if result.dtype != torch.bfloat16:
                fallback_reason = f"BASE_DTYPE_{result.dtype}"
            elif update.dtype != torch.float32:
                fallback_reason = f"UPDATE_DTYPE_{update.dtype}"
            elif result.shape != update.shape:
                fallback_reason = "SHAPE_MISMATCH"
            elif not result.is_contiguous() or not update.is_contiguous():
                fallback_reason = "NONCONTIGUOUS"
            if fallback_reason is None:
                result = fused_epilogue(result, update, _child.scaling[adapter])
                receipt["fused_invocation_count"] += 1
                receipt["fused_element_count"] += int(result.numel())
            else:
                result = (result + update * _child.scaling[adapter]).to(result_dtype)
                receipt["reference_fallback_count"] += 1
                reasons = receipt["reference_fallback_reasons"]
                reasons[fallback_reason] = int(reasons.get(fallback_reason, 0)) + 1
            return result

        child.forward = fused_forward

    original_model_forward = model.forward

    @functools.wraps(original_model_forward)
    def model_forward(*args: Any, **kwargs: Any) -> Any:
        torch = _torch()
        if torch.is_grad_enabled():
            return original_model_forward(*args, **kwargs)
        if active_call.get():
            raise RuntimeError("nested Helios no-grad LoRA epilogue scope")
        token = active_call.set(True)
        receipt["model_call_count"] += 1
        try:
            return original_model_forward(*args, **kwargs)
        finally:
            active_call.reset(token)

    model.forward = model_forward
    receipt["enabled"] = True
    return receipt


__all__ = [
    "PEFT_0181_LINEAR_FORWARD_SHA256",
    "install_generator_nograd_lora_fused_epilogue",
    "source_sha256",
]
