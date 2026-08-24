"""Profile-only operator attribution for mixed-dtype Wan LoRA calls.

The production path is deliberately unchanged unless the explicit diagnostic
environment switch is enabled. On one selected rank, a bounded number of real
Generator no-grad Transformer calls replay PEFT 0.18.1's exact vanilla
``Linear.forward`` operator order with same-thread NVTX ranges around the base
linear, BF16-to-FP32 input cast, LoRA A/B, scale/add, and FP32-to-BF16 output
cast.

This module does not synchronize CUDA, copy tensors for diagnostics, or change
any dtype. It exists only to build an auditable NVTX -> CUDA runtime
correlationId -> kernel attribution chain before a custom kernel is considered.
"""
from __future__ import annotations

import contextlib
import contextvars
import functools
import hashlib
import inspect
import os
from typing import Any, Iterable


PEFT_0181_LINEAR_FORWARD_SHA256 = (
    "ec0a80f5c5ce05f5dd90027952bb5d29aaf910e9d2ab40486a90e8a03c7e1cd9"
)
ATTRIBUTION_ENV = "DISTILLSCHEDULE_WAN13B_LORA_OPERATOR_ATTRIBUTION"
TARGET_RANK_ENV = "DISTILLSCHEDULE_WAN13B_LORA_ATTRIBUTION_TARGET_RANK"
TARGET_CALLS_ENV = "DISTILLSCHEDULE_WAN13B_LORA_ATTRIBUTION_TARGET_NOGRAD_CALLS"


def _torch() -> Any:
    import torch

    return torch


def _enabled(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def source_sha256(function: Any) -> str:
    return hashlib.sha256(inspect.getsource(function).encode()).hexdigest()


@contextlib.contextmanager
def _nvtx(torch: Any, label: str):
    torch.cuda.nvtx.range_push(label)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def _shape(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return [int(item) for item in shape]
    except (TypeError, ValueError):
        return None


def _first_tensor_shape(args: Iterable[Any], kwargs: dict[str, Any]) -> list[int] | None:
    for value in (*tuple(args), *tuple(kwargs.values())):
        result = _shape(value)
        if result is not None:
            return result
    return None


def install_lora_operator_attribution(
    model: Any,
    *,
    role: str,
    profile_mode: bool,
    global_rank: int,
    linear_type: type[Any] | None = None,
    variant_kwarg_keys: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Install bounded, default-off NVTX attribution on one real rank.

    ``linear_type`` and ``variant_kwarg_keys`` are injectable only so focused
    CPU tests can validate dispatch without importing the server PEFT package.
    Production calls omit them and are source-pinned to PEFT 0.18.1.
    """

    requested = _enabled(os.environ.get(ATTRIBUTION_ENV))
    target_rank = int(os.environ.get(TARGET_RANK_ENV, "0"))
    target_calls = int(os.environ.get(TARGET_CALLS_ENV, "3"))
    if target_calls <= 0:
        raise ValueError(f"{TARGET_CALLS_ENV} must be positive")
    receipt: dict[str, Any] = {
        "requested": requested,
        "enabled": False,
        "profile_only": True,
        "role": str(role),
        "global_rank": int(global_rank),
        "target_rank": target_rank,
        "target_nograd_model_calls": target_calls,
        "activated_model_call_count": 0,
        "instrumented_module_count": 0,
        "operator_counts": {
            "BASE": 0,
            "INPUT_CAST": 0,
            "DROPOUT": 0,
            "LORA_A": 0,
            "LORA_B": 0,
            "SCALE_ADD": 0,
            "OUTPUT_CAST": 0,
        },
        "model_calls": [],
        "semantic_contract": (
            "PROFILE_ONLY_SAME_OPERATOR_ORDER_NO_SYNC_NO_DIAGNOSTIC_COPY"
        ),
    }
    if not requested:
        receipt["disabled_reason"] = "NOT_REQUESTED"
        return receipt
    if not profile_mode:
        raise RuntimeError("LoRA operator attribution is allowed only in profile mode")
    if role != "generator":
        receipt["disabled_reason"] = "GENERATOR_ONLY"
        return receipt
    if int(global_rank) != target_rank:
        receipt["disabled_reason"] = "NON_TARGET_RANK"
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

    torch = _torch()
    active_call: contextvars.ContextVar[int | None] = contextvars.ContextVar(
        f"wan13b_lora_attribution_rank_{global_rank}", default=None
    )
    named_linears = [
        (name, child)
        for name, child in model.named_modules()
        if isinstance(child, linear_type)
    ]
    if not named_linears:
        raise RuntimeError("LoRA operator attribution found no PEFT Linear modules")

    for module_name, child in named_linears:
        original_forward = child.forward

        @functools.wraps(original_forward)
        def attributed_forward(
            x: Any,
            *args: Any,
            _child: Any = child,
            _module_name: str = module_name,
            _original_forward: Any = original_forward,
            **kwargs: Any,
        ) -> Any:
            call_index = active_call.get()
            if call_index is None:
                return _original_forward(x, *args, **kwargs)

            _child._check_forward_args(x, *args, **kwargs)
            adapter_names = kwargs.pop("adapter_names", None)
            variant_kwargs = {
                key: kwargs.pop(key, None) for key in variant_kwarg_keys
            }
            if _child.disable_adapters or adapter_names is not None or _child.merged:
                raise RuntimeError(
                    "bounded LoRA attribution requires enabled, unmerged, "
                    "non-mixed vanilla adapters"
                )
            active_adapters = tuple(_child.active_adapters)
            if len(active_adapters) != 1:
                raise RuntimeError(
                    "bounded LoRA attribution requires exactly one active adapter"
                )
            active_adapter = active_adapters[0]
            if active_adapter not in _child.lora_A:
                raise RuntimeError("active adapter is absent from LoRA-A")
            if active_adapter in _child.lora_variant:
                raise RuntimeError("LoRA variants are outside the attribution contract")
            if any(value is not None for value in variant_kwargs.values()):
                raise RuntimeError("variant kwargs are outside the attribution contract")

            prefix = f"WAN13B_LORA_OP/call={call_index}/module={_module_name}"
            with _nvtx(torch, f"{prefix}/op=BASE"):
                result = _child.base_layer(x, *args, **kwargs)
            receipt["operator_counts"]["BASE"] += 1
            torch_result_dtype = result.dtype
            lora_a = _child.lora_A[active_adapter]
            lora_b = _child.lora_B[active_adapter]
            dropout = _child.lora_dropout[active_adapter]
            scaling = _child.scaling[active_adapter]
            with _nvtx(torch, f"{prefix}/op=INPUT_CAST"):
                cast_x = _child._cast_input_dtype(x, lora_a.weight.dtype)
            receipt["operator_counts"]["INPUT_CAST"] += 1
            with _nvtx(torch, f"{prefix}/op=DROPOUT"):
                dropped = dropout(cast_x)
            receipt["operator_counts"]["DROPOUT"] += 1
            with _nvtx(torch, f"{prefix}/op=LORA_A"):
                intermediate = lora_a(dropped)
            receipt["operator_counts"]["LORA_A"] += 1
            with _nvtx(torch, f"{prefix}/op=LORA_B"):
                update = lora_b(intermediate)
            receipt["operator_counts"]["LORA_B"] += 1
            with _nvtx(torch, f"{prefix}/op=SCALE_ADD"):
                result = result + update * scaling
            receipt["operator_counts"]["SCALE_ADD"] += 1
            with _nvtx(torch, f"{prefix}/op=OUTPUT_CAST"):
                result = result.to(torch_result_dtype)
            receipt["operator_counts"]["OUTPUT_CAST"] += 1
            return result

        child.forward = attributed_forward

    original_model_forward = model.forward

    @functools.wraps(original_model_forward)
    def model_forward(*args: Any, **kwargs: Any) -> Any:
        if (
            torch.is_grad_enabled()
            or receipt["activated_model_call_count"] >= target_calls
        ):
            return original_model_forward(*args, **kwargs)
        call_index = int(receipt["activated_model_call_count"])
        receipt["activated_model_call_count"] += 1
        receipt["model_calls"].append(
            {
                "call_index": call_index,
                "first_tensor_shape": _first_tensor_shape(args, kwargs),
                "grad_enabled": False,
            }
        )
        token = active_call.set(call_index)
        try:
            with _nvtx(torch, f"WAN13B_LORA_ATTRIBUTION_CALL/{call_index}"):
                return original_model_forward(*args, **kwargs)
        finally:
            active_call.reset(token)

    model.forward = model_forward
    receipt["enabled"] = True
    receipt["instrumented_module_count"] = len(named_linears)
    return receipt


__all__ = [
    "ATTRIBUTION_ENV",
    "PEFT_0181_LINEAR_FORWARD_SHA256",
    "TARGET_CALLS_ENV",
    "TARGET_RANK_ENV",
    "install_lora_operator_attribution",
    "source_sha256",
]
