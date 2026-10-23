"""Exact, opt-in no-grad LoRA epilogue fusion for Helios rollouts.

Only Generator calls made with autograd disabled enter this path.  Gradient
calls and every Critic call retain PEFT's source-pinned implementation.  The
fused kernel preserves either of the two source-pinned PEFT epilogues observed
under the supported precision policies:

* FP32 update: FP32 multiply, FP32 add, final BF16 rounding;
* BF16 update: BF16 multiply rounding, BF16 add rounding.

The second contract is important for the real Helios FSDP2 path, whose mixed
precision policy materializes both the frozen base and LoRA parameters as
BF16.  It must not be replaced with a one-round FP32 fused expression.
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

LORA_B_GEMM_VALIDATED_SHAPES = ((11040, 5120), (13680, 5120), (11040, 13824))


def _lora_b_gemm_selection(
    hidden: Any, base: Any, weight: Any, torch: Any, *,
    allowed_shapes: Iterable[tuple[int, int]] = LORA_B_GEMM_VALIDATED_SHAPES,
    rank: int = 256,
) -> tuple[bool, str, str]:
    """Metadata-only selection; the kernel rechecks actual post-hook inputs."""
    if hidden.ndim < 2 or base.ndim != hidden.ndim:
        return False, "NDIM", "UNRESOLVED"
    m, n = int(hidden.numel()) // int(hidden.shape[-1]), int(base.shape[-1])
    shape_key = f"M{m}_N{n}_R{int(hidden.shape[-1])}"
    if (m, n) not in allowed_shapes:
        return False, "NOT_SELECTED_SHAPE", shape_key
    if int(hidden.shape[-1]) != rank or tuple(weight.shape) != (n, rank):
        return False, "RANK_OR_WEIGHT_SHAPE", shape_key
    if tuple(hidden.shape[:-1]) != tuple(base.shape[:-1]):
        return False, "LEADING_SHAPE", shape_key
    if any(value.dtype != torch.bfloat16 for value in (hidden, base, weight)):
        return False, "DTYPE", shape_key
    if any(not value.is_cuda for value in (hidden, base, weight)):
        return False, "DEVICE_NOT_CUDA", shape_key
    if hidden.device != base.device or hidden.device != weight.device:
        return False, "DEVICE_MISMATCH", shape_key
    if any(not value.is_contiguous() for value in (hidden, base, weight)):
        return False, "NONCONTIGUOUS", shape_key
    return True, "SELECTED", shape_key


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
    all_role_scale1: bool = False,
    checkpoint_replay_scope: bool = False,
    lora_b_gemm_epilogue: bool = False,
    lora_b_gemm_epilogue_impl: Any | None = None,
    lora_b_gemm_test_shapes: Iterable[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Install one bounded exact Helios LoRA epilogue replacement.

    ``all_role_scale1`` is a separate candidate: it activates for both roles
    and for grad-enabled calls, but only accepts BF16 updates with scaling 1.
    Grad calls use an out-of-place BF16 add; no-grad calls retain the safe
    in-place specialization.  The ordinary Generator-no-grad candidate remains
    available unchanged when this argument is false.
    """

    if checkpoint_replay_scope and (not enabled or not all_role_scale1):
        raise ValueError("LoRA checkpoint scope requires enabled all-role scale1 add")
    if lora_b_gemm_epilogue and (not enabled or not all_role_scale1 or not checkpoint_replay_scope):
        raise ValueError("LoRA-B GEMM requires enabled all-role scale1 and checkpoint scope")
    if lora_b_gemm_test_shapes is not None and (linear_type is None or lora_b_gemm_epilogue_impl is None):
        raise ValueError("LoRA-B test shapes require both injected type and injected implementation")
    receipt: dict[str, Any] = {
        "requested": bool(enabled),
        "enabled": False,
        "role": str(role),
        "scope": (
            "ALL_ROLE_SCALE1_MODEL_FORWARD"
            if all_role_scale1
            else "GENERATOR_NOGRAD_MODEL_FORWARD_ONLY"
        ),
        "model_call_count": 0,
        "fused_invocation_count": 0,
        "fused_grad_invocation_count": 0,
        "fused_nograd_invocation_count": 0,
        "fused_element_count": 0,
        "fused_update_dtype_counts": {},
        "fused_scale_counts": {},
        "reference_fallback_count": 0,
        "reference_fallback_reasons": {},
        "adapter_disabled_bypass_count": 0,
        "checkpoint_replay_scope": bool(checkpoint_replay_scope),
        "checkpoint_wrap_count": 0,
        "checkpoint_scope_execution_count": 0,
        "checkpoint_scope_restore_count": 0,
        "checkpoint_missing_active_scope_count": 0,
        "lora_b_gemm_epilogue_requested": bool(lora_b_gemm_epilogue),
        "lora_b_gemm_epilogue_enabled": False,
        "lora_b_gemm_selected_invocation_count": 0,
        "lora_b_gemm_selected_grad_invocation_count": 0,
        "lora_b_gemm_selected_nograd_invocation_count": 0,
        "lora_b_gemm_fallback_count": 0,
        "lora_b_gemm_fallback_reasons": {},
        "lora_b_gemm_shape_counts": {},
        "lora_b_gemm_selected_shape_counts": {},
        "lora_b_gemm_selected_attempt_count": 0,
        "lora_b_gemm_completed_invocation_count": 0,
        "lora_b_gemm_kernel_call_count": 0,
        "reference_pointwise_kernel_count_per_invocation": {
            "torch.float32": 3,
            "torch.bfloat16": 2,
        },
        "candidate_kernel_count_per_invocation": 1,
        "semantic_contract": (
            "ALL_ROLE_BF16_SCALE1_ADD_FORWARD_AND_BACKWARD_EXACT;"
            "TRAINING_MATH_UNCHANGED"
            if all_role_scale1
            else (
                "FP32_UPDATE:FP32_MUL_RN_THEN_FP32_ADD_RN_THEN_BF16_RN_STORE;"
                "BF16_UPDATE:BF16_MUL_RN_THEN_BF16_ADD_RN;"
                "GRAD_AND_CRITIC_PATHS_UNCHANGED"
            )
        ),
    }
    model._distillschedule_helios_nograd_lora_epilogue = receipt
    if not enabled:
        receipt["disabled_reason"] = "NOT_REQUESTED"
        return receipt
    if role != "generator" and not all_role_scale1:
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

    allowed_b_shapes = tuple(lora_b_gemm_test_shapes or LORA_B_GEMM_VALIDATED_SHAPES)
    injected_b_impl = lora_b_gemm_epilogue_impl is not None
    if lora_b_gemm_epilogue:
        if lora_b_gemm_epilogue_impl is None:
            from distillschedule.runtime.helios_lora_b_gemm_kernel import (
                fused_lora_b_gemm_epilogue, runtime_identity as b_runtime_identity,
            )
            lora_b_gemm_epilogue_impl = fused_lora_b_gemm_epilogue
            receipt["lora_b_gemm_runtime_identity"] = b_runtime_identity()
        else:
            if not injected_type:
                raise ValueError("LoRA-B implementation injection requires injected Linear type")
            receipt["lora_b_gemm_runtime_identity"] = {
                "implementation": "TEST_INJECTED_NOT_GPU_PRODUCTION",
            }
        receipt["lora_b_gemm_validated_shapes"] = [list(shape) for shape in allowed_b_shapes]
        receipt["lora_b_gemm_epilogue_enabled"] = True
        receipt["lora_b_gemm_semantics"] = (
            "BF16_B_GEMM_RN_THEN_FP32_ADD_RN_THEN_BF16_STORE;"
            "REFERENCE_CONDITIONAL_FIRST_ORDER_MM_BACKWARD;NOT_BITWISE_GEMM_CLAIM"
        )

    active_call: contextvars.ContextVar[bool] = contextvars.ContextVar(
        "helios_generator_nograd_lora_fused_epilogue", default=False
    )
    if checkpoint_replay_scope:
        def wrap_checkpoint_function(function: Any) -> Any:
            # Capture only our LoRA scope, not unrelated saved-tensor/CPU-offload
            # contexts.  This callable is what checkpoint actually saves.
            captured_active = active_call.get()
            if not captured_active:
                receipt["checkpoint_missing_active_scope_count"] += 1
                raise RuntimeError("LoRA checkpoint created outside active model scope")
            receipt["checkpoint_wrap_count"] += 1

            @functools.wraps(function)
            def scoped_execution(*args: Any, **kwargs: Any) -> Any:
                token = active_call.set(captured_active)
                receipt["checkpoint_scope_execution_count"] += 1
                try:
                    return function(*args, **kwargs)
                finally:
                    # Non-reentrant replay may exit through PyTorch early-stop.
                    active_call.reset(token)
                    receipt["checkpoint_scope_restore_count"] += 1

            return scoped_execution

        model._distillschedule_lora_checkpoint_function_wrapper = (
            wrap_checkpoint_function
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
        if lora_b_gemm_epilogue:
            for b_module in child.lora_B.values():
                if getattr(b_module, "_distillschedule_lora_b_gemm_wrapper_installed", False):
                    raise RuntimeError("LoRA-B GEMM module already wrapped")
                original_b_forward = b_module.forward

                @functools.wraps(original_b_forward)
                def b_forward(
                    hidden: Any, *args: Any,
                    _distillschedule_lora_base_result: Any = None,
                    _b_module: Any = b_module, _original_b_forward: Any = original_b_forward,
                    **kwargs: Any,
                ) -> Any:
                    if _distillschedule_lora_base_result is None:
                        return _original_b_forward(hidden, *args, **kwargs)
                    if not active_call.get() or args or kwargs:
                        raise RuntimeError("LoRA-B fused call outside bounded model scope")
                    if getattr(_b_module, "bias", None) is not None:
                        raise RuntimeError("LoRA-B fused path requires bias=False")
                    caller_grad_enabled = _torch().is_grad_enabled()
                    shape_key = (
                        f"M{int(hidden.numel()) // int(hidden.shape[-1])}"
                        f"_N{int(_distillschedule_lora_base_result.shape[-1])}"
                        f"_R{int(hidden.shape[-1])}"
                    )

                    def note_launch() -> None:
                        receipt["lora_b_gemm_kernel_call_count"] += 1
                        receipt["lora_b_gemm_selected_invocation_count"] += 1
                        grad_suffix = "grad" if caller_grad_enabled else "nograd"
                        receipt[f"lora_b_gemm_selected_{grad_suffix}_invocation_count"] += 1
                        counts = receipt["lora_b_gemm_selected_shape_counts"]
                        counts[shape_key] = int(counts.get(shape_key, 0)) + 1

                    # This forward is reached via B.__call__, retaining module
                    # pre/post hooks (including any FSDP hooks), never directly.
                    if injected_b_impl:
                        result = lora_b_gemm_epilogue_impl(
                            hidden, _b_module.weight, _distillschedule_lora_base_result,
                        )
                        note_launch()  # Test fixture, not a production kernel claim.
                    else:
                        result = lora_b_gemm_epilogue_impl(
                            hidden, _b_module.weight, _distillschedule_lora_base_result,
                            on_kernel_launch=note_launch,
                        )
                    receipt["lora_b_gemm_completed_invocation_count"] += 1
                    return result

                b_module.forward = b_forward
                b_module._distillschedule_lora_b_gemm_wrapper_installed = True

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

            if _child.disable_adapters:
                receipt["adapter_disabled_bypass_count"] += 1
                return _original_forward(x, *args, **kwargs)

            _child._check_forward_args(x, *args, **kwargs)
            adapter_names = kwargs.pop("adapter_names", None)
            variant_kwargs = {
                key: kwargs.pop(key, None) for key in variant_kwarg_keys
            }
            if adapter_names is not None or _child.merged:
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
            lora_hidden = lora_a(_child.lora_dropout[adapter](cast_x))
            torch = _torch()
            if lora_b_gemm_epilogue:
                selected, reason, shape_key = _lora_b_gemm_selection(
                    lora_hidden, result, lora_b.weight, torch,
                    allowed_shapes=allowed_b_shapes,
                    rank=int(lora_hidden.shape[-1]) if lora_b_gemm_test_shapes is not None else 256,
                )
                if float(_child.scaling[adapter]) != 1.0:
                    raise RuntimeError("LoRA-B GEMM requires scaling=1")
                if float(getattr(_child.lora_dropout[adapter], "p", 0.0)) != 0.0:
                    raise RuntimeError("LoRA-B GEMM requires dropout=0")
                shape_counts = receipt["lora_b_gemm_shape_counts"]
                shape_counts[shape_key] = int(shape_counts.get(shape_key, 0)) + 1
                if selected:
                    receipt["lora_b_gemm_selected_attempt_count"] += 1
                    result = lora_b(
                        lora_hidden, _distillschedule_lora_base_result=result,
                    )
                    grad_suffix = "grad" if torch.is_grad_enabled() else "nograd"
                    receipt["fused_invocation_count"] += 1
                    receipt[f"fused_{grad_suffix}_invocation_count"] += 1
                    receipt["fused_element_count"] += int(result.numel())
                    dtype_counts = receipt["fused_update_dtype_counts"]
                    dtype_counts[str(torch.bfloat16)] = int(dtype_counts.get(str(torch.bfloat16), 0)) + 1
                    scale_counts = receipt["fused_scale_counts"]
                    scale_counts["1.0"] = int(scale_counts.get("1.0", 0)) + 1
                    return result
                receipt["lora_b_gemm_fallback_count"] += 1
                reasons = receipt["lora_b_gemm_fallback_reasons"]
                reasons[reason] = int(reasons.get(reason, 0)) + 1
            update = lora_b(lora_hidden)
            fallback_reason = None
            if result.dtype != torch.bfloat16:
                fallback_reason = f"BASE_DTYPE_{result.dtype}"
            elif all_role_scale1 and update.dtype != torch.bfloat16:
                fallback_reason = f"ALL_ROLE_UPDATE_DTYPE_{update.dtype}"
            elif not all_role_scale1 and update.dtype not in (
                torch.float32,
                torch.bfloat16,
            ):
                fallback_reason = f"UPDATE_DTYPE_{update.dtype}"
            elif all_role_scale1 and float(_child.scaling[adapter]) != 1.0:
                fallback_reason = f"ALL_ROLE_SCALE_{_child.scaling[adapter]!r}"
            elif result.shape != update.shape:
                fallback_reason = "SHAPE_MISMATCH"
            elif not result.is_contiguous() or not update.is_contiguous():
                fallback_reason = "NONCONTIGUOUS"
            if fallback_reason is None:
                if all_role_scale1 and torch.is_grad_enabled():
                    # `update * 1.0` is bitwise the BF16 identity.  Dropping
                    # that node keeps the same AddBackward gradients while
                    # eliminating one pointwise kernel.
                    result = result + update
                else:
                    result = fused_epilogue(
                        result, update, _child.scaling[adapter]
                    )
                receipt["fused_invocation_count"] += 1
                grad_key = (
                    "fused_grad_invocation_count"
                    if torch.is_grad_enabled()
                    else "fused_nograd_invocation_count"
                )
                receipt[grad_key] += 1
                receipt["fused_element_count"] += int(result.numel())
                dtype_counts = receipt["fused_update_dtype_counts"]
                dtype_key = str(update.dtype)
                dtype_counts[dtype_key] = int(dtype_counts.get(dtype_key, 0)) + 1
                scale_counts = receipt["fused_scale_counts"]
                scale_key = repr(float(_child.scaling[adapter]))
                scale_counts[scale_key] = int(scale_counts.get(scale_key, 0)) + 1
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
        if torch.is_grad_enabled() and not all_role_scale1:
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
