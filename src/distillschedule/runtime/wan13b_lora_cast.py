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
CAST_CACHE_ENV = "DISTILLSCHEDULE_WAN13B_GENERATOR_NOGRAD_LORA_CAST_CACHE"
FUSED_EPILOGUE_ENV = "DISTILLSCHEDULE_WAN13B_GENERATOR_NOGRAD_LORA_FUSED_EPILOGUE"


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


def install_generator_nograd_lora_cast_cache(
    model: Any,
    *,
    role: str,
    linear_type: type[Any] | None = None,
) -> dict[str, Any]:
    """Reuse an exact FP32 cast for adjacent LoRA calls with identical input.

    Wan attention projections commonly invoke several PEFT Linear modules with
    the exact same BF16 Tensor object. PEFT independently materializes an FP32
    copy for every module because the LoRA trainables are FP32. This bounded
    cache retains at most one converted tensor, only during one Generator
    no-grad Transformer call, and reuses it only when tensor identity, in-place
    version, target dtype, device, shape, and stride all still match.

    Gradient-bearing paths and Critic paths never enter the cache. The returned
    value is the exact result of PEFT's original ``Tensor.to(dtype)`` call; no
    arithmetic, parameter dtype, operator count, or model call is changed.
    """

    requested = _enabled(os.environ.get(CAST_CACHE_ENV))
    receipt: dict[str, Any] = {
        "requested": requested,
        "enabled": False,
        "role": str(role),
        "scope": "GENERATOR_NOGRAD_MODEL_FORWARD_ONLY",
        "cache_capacity": 1,
        "model_call_count": 0,
        "eligible_cast_requests": 0,
        "cache_hit_count": 0,
        "cache_miss_count": 0,
        "passthrough_count": 0,
        "avoided_source_read_bytes": 0,
        "avoided_destination_write_bytes": 0,
        "max_cached_tensor_bytes": 0,
        "semantic_contract": (
            "EXACT_TENSOR_TO_RESULT_REUSE_FOR_IDENTICAL_UNMODIFIED_INPUT;"
            "GRAD_AND_CRITIC_PATHS_UNCHANGED"
        ),
    }
    model._distillschedule_generator_nograd_lora_cast_cache = receipt
    if not requested:
        receipt["disabled_reason"] = "NOT_REQUESTED"
        return receipt
    if role != "generator":
        receipt["disabled_reason"] = "GENERATOR_ONLY"
        return receipt

    injected_type = linear_type is not None
    if linear_type is None:
        from peft.tuners.lora.layer import Linear

        linear_type = Linear
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

    linears = [child for child in model.modules() if isinstance(child, linear_type)]
    if not linears:
        raise RuntimeError("Generator no-grad LoRA cast cache found no PEFT Linear modules")
    receipt["instrumented_module_count"] = len(linears)
    active_cache: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
        "wan13b_generator_nograd_lora_cast_cache", default=None
    )

    for child in linears:
        original_cast = child._cast_input_dtype

        def cached_cast(
            x: Any,
            dtype: Any,
            *,
            _child: Any = child,
            _original_cast: Any = original_cast,
        ) -> Any:
            state = active_cache.get()
            if state is None:
                return _original_cast(x, dtype)
            if not getattr(_child, "cast_input_dtype_enabled", True) or x.dtype == dtype:
                receipt["passthrough_count"] += 1
                return _original_cast(x, dtype)

            receipt["eligible_cast_requests"] += 1
            version = int(getattr(x, "_version", -1))
            signature = (
                dtype,
                x.device,
                tuple(int(value) for value in x.shape),
                tuple(int(value) for value in x.stride()),
                int(x.storage_offset()),
                version,
            )
            cached_input = state.get("input")
            cached_value = state.get("value")
            if (
                cached_input is x
                and cached_value is not None
                and state.get("signature") == signature
                and int(getattr(cached_value, "_version", -1))
                == state.get("value_version")
            ):
                receipt["cache_hit_count"] += 1
                numel = int(x.numel())
                receipt["avoided_source_read_bytes"] += numel * int(x.element_size())
                receipt["avoided_destination_write_bytes"] += numel * int(
                    cached_value.element_size()
                )
                return cached_value

            value = _original_cast(x, dtype)
            state.clear()
            state.update(
                {
                    "input": x,
                    "value": value,
                    "signature": signature,
                    "value_version": int(getattr(value, "_version", -1)),
                }
            )
            receipt["cache_miss_count"] += 1
            receipt["max_cached_tensor_bytes"] = max(
                int(receipt["max_cached_tensor_bytes"]),
                int(value.numel()) * int(value.element_size()),
            )
            return value

        child._cast_input_dtype = cached_cast

    original_model_forward = model.forward

    @functools.wraps(original_model_forward)
    def model_forward(*args: Any, **kwargs: Any) -> Any:
        torch = _torch()
        if torch.is_grad_enabled():
            return original_model_forward(*args, **kwargs)
        if active_cache.get() is not None:
            raise RuntimeError("nested Generator no-grad LoRA cast-cache scope")
        token = active_cache.set({})
        receipt["model_call_count"] += 1
        try:
            return original_model_forward(*args, **kwargs)
        finally:
            state = active_cache.get()
            if state is not None:
                state.clear()
            active_cache.reset(token)

    model.forward = model_forward
    receipt["enabled"] = True
    return receipt


def install_generator_nograd_lora_fused_epilogue(
    model: Any,
    *,
    role: str,
    linear_type: type[Any] | None = None,
    variant_kwarg_keys: Iterable[str] | None = None,
    fused_epilogue: Any | None = None,
) -> dict[str, Any]:
    """Fuse the no-grad LoRA scale/add/output cast without changing FP32 math.

    Only Generator no-grad Transformer calls enter the replacement forward.
    Grad-bearing calls and every Critic call retain the installed PEFT forward.
    The production path is source-pinned to PEFT 0.18.1 and fails closed on
    adapter variants, mixed batches, merged adapters, or unexpected layouts.
    """

    requested = _enabled(os.environ.get(FUSED_EPILOGUE_ENV))
    receipt: dict[str, Any] = {
        "requested": requested,
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
    model._distillschedule_generator_nograd_lora_fused_epilogue = receipt
    if not requested:
        receipt["disabled_reason"] = "NOT_REQUESTED"
        return receipt
    if _enabled(os.environ.get(CAST_CACHE_ENV)):
        raise RuntimeError("cast cache and fused epilogue must be tested separately")
    if _enabled(os.environ.get(ATTRIBUTION_ENV)):
        raise RuntimeError("operator attribution and fused epilogue are separate runs")
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
        from distillschedule.runtime.wan13b_lora_epilogue import (
            fused_lora_epilogue,
            runtime_identity,
        )

        fused_epilogue = fused_lora_epilogue
        receipt["runtime_identity"] = runtime_identity()
    else:
        receipt["runtime_identity"] = {"implementation": "TEST_INJECTED"}

    active_call: contextvars.ContextVar[bool] = contextvars.ContextVar(
        "wan13b_generator_nograd_lora_fused_epilogue", default=False
    )
    named_linears = [
        (name, child)
        for name, child in model.named_modules()
        if isinstance(child, linear_type)
    ]
    if not named_linears:
        raise RuntimeError("Generator no-grad fused epilogue found no PEFT Linear modules")
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
                    "fused LoRA epilogue requires enabled, unmerged, non-mixed adapters"
                )
            active_adapters = tuple(_child.active_adapters)
            if len(active_adapters) != 1:
                raise RuntimeError("fused LoRA epilogue requires one active adapter")
            adapter = active_adapters[0]
            if adapter not in _child.lora_A or adapter in _child.lora_variant:
                raise RuntimeError("fused LoRA epilogue adapter contract changed")
            if any(value is not None for value in variant_kwargs.values()):
                raise RuntimeError("LoRA variant kwargs are outside the fused contract")

            result = _child.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            lora_a = _child.lora_A[adapter]
            lora_b = _child.lora_B[adapter]
            dropout = _child.lora_dropout[adapter]
            graph_controller = getattr(model, "_distillschedule_rollout_cuda_graph", None)
            input_fused = _enabled(os.environ.get("DISTILLSCHEDULE_WAN13B_FUSED_INPUT_LORA_A"))
            use_input_fused = (
                input_fused and graph_controller is not None
                and graph_controller.receipt["active"] and x.dtype == _torch().bfloat16
                and x.shape[-1] == 1536 and x.numel() // 1536 >= 4096
                and tuple(lora_a.weight.shape) == (256, 1536)
                and x.is_contiguous() and lora_a.weight.is_contiguous()
                and lora_a.bias is None and not getattr(dropout, "p", 0)
            )
            if use_input_fused:
                from .wan13b_input_lora_gemm import fused_input_lora_a
                intermediate = fused_input_lora_a(x, lora_a.weight)
                update = lora_b(intermediate)
                receipt["input_gemm_capture_invocation_count"] = receipt.get("input_gemm_capture_invocation_count", 0) + 1
            else:
                cast_x = _child._cast_input_dtype(x, lora_a.weight.dtype)
                update = lora_b(lora_a(dropout(cast_x)))
            fallback_reason = None
            torch = _torch()
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
                result = (result + update * _child.scaling[adapter]).to(
                    torch_result_dtype
                )
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
            raise RuntimeError("nested Generator no-grad fused-epilogue scope")
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
    "ATTRIBUTION_ENV",
    "CAST_CACHE_ENV",
    "FUSED_EPILOGUE_ENV",
    "PEFT_0181_LINEAR_FORWARD_SHA256",
    "TARGET_CALLS_ENV",
    "TARGET_RANK_ENV",
    "install_lora_operator_attribution",
    "install_generator_nograd_lora_cast_cache",
    "install_generator_nograd_lora_fused_epilogue",
    "source_sha256",
]
