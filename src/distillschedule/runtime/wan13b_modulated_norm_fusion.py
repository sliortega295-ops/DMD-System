"""Default-off exact Helios modulated-norm forward epilogue fusion."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch


SUPPORTED_HELIOS_TRANSFORMER_SHA256 = (
    "5d3ffe37cdea3c04a156106f6246c82f91010a1eaf0dd61529cd74083065a886"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def install_exact_modulated_norm_epilogue(
    *,
    enabled: bool,
    role: str,
    fuse_materialization: bool = False,
    fuse_backward: bool = False,
    fuse_layernorm: bool = False,
    fuse_backward_kernel: bool = False,
    transformer_module: Any | None = None,
    fused_epilogue: Any | None = None,
    fused_materialization: Any | None = None,
    fused_layernorm: Any | None = None,
    fused_hidden_backward: Any | None = None,
    fused_reduced_component_gradient: Any | None = None,
    fused_tensor4d_component_gradient: Any | None = None,
) -> dict[str, Any]:
    """Patch the pinned custom Function with independently gated exact paths."""

    receipt: dict[str, Any] = {
        "requested": bool(enabled),
        "enabled": False,
        "role": str(role),
        "scope": "ALL_ROLE_REGISTERED_B1_B3_MODULATED_NORM_FORWARD_EPILOGUE",
        "semantic_contract": (
            (
                "FUSED_BF16_INPUT_FP32_AFFINE_FREE_LAYERNORM_WIDTH1536;"
                if fuse_layernorm
                else "PYTORCH_FP32_LAYERNORM_UNCHANGED;"
            )
            + "ADD_RN_F32_ONE_PLUS_SCALE;"
            "MUL_RN_F32;ADD_RN_F32_SHIFT;BF16_RN_STORE;"
            + (
                (
                    "FUSED_TRITON_LAYERNORM_BACKWARD_AND_DETERMINISTIC_"
                    "COMPONENT_GRADIENTS"
                    if fuse_backward_kernel
                    else "EXPLICIT_NATIVE_LAYERNORM_BACKWARD_AND_COMPONENT_GRADIENTS"
                )
                if fuse_backward
                else "SOURCE_BACKWARD_UNCHANGED"
            )
        ),
        "fused_call_count": 0,
        "fused_chunk_count": 0,
        "fused_element_count": 0,
        "reference_fallback_count": 0,
        "reference_fallback_reasons": {},
        "materialization_requested": bool(fuse_materialization),
        "materialization_contract": (
            "SOURCE_TABLE_PLUS_COMPONENT_ADD_RN_F32_FUSED_BEFORE_EXISTING_EPILOGUE"
        ),
        "materialization_fused_call_count": 0,
        "materialization_fused_chunk_count": 0,
        "backward_requested": bool(fuse_backward),
        "backward_contract": (
            (
                "SOURCE_CHUNK_ORDER;TRITON_FP32_LAYERNORM_BACKWARD;"
                "DETERMINISTIC_BF16_COMPONENT_GRADIENT_ACCUMULATION"
            )
            if fuse_backward_kernel
            else (
                "SOURCE_CHUNK_ORDER;FP32_NATIVE_LAYERNORM_BACKWARD;"
                "BF16_COMPONENT_GRADIENT_ACCUMULATION"
            )
        ),
        "backward_call_count": 0,
        "backward_chunk_count": 0,
        "backward_hidden_element_count": 0,
        "backward_layout_counts": {},
        "backward_kernel_requested": bool(fuse_backward_kernel),
        "backward_kernel_contract": (
            "SOURCE_CHUNK_ORDER;BF16_INPUT_AND_UPSTREAM;FP32_AFFINE_FREE_"
            "LAYERNORM_GRAD_INPUT;DETERMINISTIC_PHASE_LOCAL_OR_TENSOR_"
            "COMPONENT_REDUCTIONS;BF16_GRADIENT_OUTPUT"
        ),
        "backward_kernel_call_count": 0,
        "backward_kernel_chunk_count": 0,
        "backward_hidden_kernel_launch_count": 0,
        "backward_component_kernel_launch_count": 0,
        "backward_tensor4d_kernel_launch_count": 0,
        "backward_kernel_reference_fallback_count": 0,
        "layernorm_requested": bool(fuse_layernorm),
        "layernorm_contract": (
            "BF16_INPUT;FP32_AFFINE_FREE_LAYERNORM_WIDTH1536;"
            "SOURCE_TABLE_COMPONENT_FP32_ADDS;SOURCE_MODULATION_FP32_FORMULA;"
            "BF16_OUTPUT;"
            + (
                "TRITON_FUSED_LAYERNORM_BACKWARD"
                if fuse_backward_kernel
                else "EXPLICIT_NATIVE_LAYERNORM_BACKWARD_UNCHANGED"
            )
        ),
        "layernorm_fused_call_count": 0,
        "layernorm_fused_chunk_count": 0,
        "layernorm_fused_element_count": 0,
        "warmup": {"status": "NOT_RUN"},
    }
    if not enabled:
        receipt["disabled_reason"] = "NOT_REQUESTED"
        return receipt
    if role != "generator":
        receipt["disabled_reason"] = "INSTALL_ONCE_FROM_GENERATOR_BUILD"
        return receipt

    if transformer_module is None:
        from helios.modules import transformer_helios as transformer_module

    if getattr(transformer_module, "_distillschedule_modulated_norm_receipt", None):
        raise RuntimeError("Helios modulated-norm fusion was already installed")
    source_path = Path(transformer_module.__file__).resolve()
    observed_sha = _sha256(source_path)
    receipt["transformer_source"] = str(source_path)
    receipt["transformer_sha256"] = observed_sha
    receipt["expected_transformer_sha256"] = SUPPORTED_HELIOS_TRANSFORMER_SHA256
    if fuse_layernorm and not fuse_materialization:
        raise ValueError("fused LayerNorm requires fused modulation materialization")
    if fuse_layernorm and not fuse_backward:
        raise ValueError("fused LayerNorm requires the explicit backward")
    if fuse_backward_kernel and not (
        fuse_materialization and fuse_backward and fuse_layernorm
    ):
        raise ValueError(
            "fused backward kernel requires materialization, explicit backward, "
            "and fused LayerNorm"
        )
    test_injected = any(
        value is not None
        for value in (
            fused_epilogue,
            fused_materialization,
            fused_layernorm,
            fused_hidden_backward,
            fused_reduced_component_gradient,
            fused_tensor4d_component_gradient,
        )
    )
    if not test_injected and observed_sha != SUPPORTED_HELIOS_TRANSFORMER_SHA256:
        raise RuntimeError(
            "Helios transformer source changed: "
            f"{observed_sha} != {SUPPORTED_HELIOS_TRANSFORMER_SHA256}"
        )
    if fused_epilogue is None:
        from distillschedule.runtime.wan13b_modulated_norm_kernel import (
            fused_modulated_norm_epilogue,
        )

        fused_epilogue = fused_modulated_norm_epilogue
    if fuse_materialization and fused_materialization is None:
        from distillschedule.runtime.wan13b_modulated_norm_kernel import (
            fused_modulated_norm_materialization,
        )

        fused_materialization = fused_modulated_norm_materialization
    if fuse_layernorm and fused_layernorm is None:
        from distillschedule.runtime.wan13b_fused_layernorm_kernel import (
            fused_bf16_layernorm_modulation,
        )

        fused_layernorm = fused_bf16_layernorm_modulation
    if fuse_backward_kernel and (
        fused_hidden_backward is None
        or fused_reduced_component_gradient is None
        or fused_tensor4d_component_gradient is None
    ):
        from distillschedule.runtime.wan13b_fused_layernorm_backward_kernel import (
            fused_bf16_layernorm_modulation_hidden_backward,
            fused_reduced_component_gradient as default_reduced_component_gradient,
            fused_tensor4d_component_gradient as default_tensor4d_component_gradient,
        )

        if fused_hidden_backward is None:
            fused_hidden_backward = fused_bf16_layernorm_modulation_hidden_backward
        if fused_reduced_component_gradient is None:
            fused_reduced_component_gradient = default_reduced_component_gradient
        if fused_tensor4d_component_gradient is None:
            fused_tensor4d_component_gradient = default_tensor4d_component_gradient

    function = transformer_module._TembAwareChunkedNorm
    original_forward = function.forward
    original_backward = function.backward
    materialize = transformer_module._materialize_scale_shift_component
    rebuild_temb = transformer_module._rebuild_temb
    phase_local_type = getattr(
        transformer_module, "PhaseLocalTimestepProjection", ()
    )

    def component(
        temb: Any, index: int, start: int, end: int
    ) -> Any:
        if isinstance(temb, phase_local_type):
            return temb.materialize_component(index, start, end)
        if temb.ndim == 4:
            return temb[:, slice(start, end), index, :]
        return temb[:, index, :].unsqueeze(1)

    def fallback(reason: str, ctx: Any, *args: Any) -> Any:
        receipt["reference_fallback_count"] += 1
        reasons = receipt["reference_fallback_reasons"]
        reasons[reason] = int(reasons.get(reason, 0)) + 1
        return original_forward(ctx, *args)

    @staticmethod
    def forward(
        ctx: Any,
        hidden_states: Any,
        norm: Any,
        scale_shift_table: Any,
        scale_index: int,
        shift_index: int,
        chunk_tokens: int,
        temb_descriptor: Any,
        *temb_inputs: Any,
    ) -> Any:
        common = (
            hidden_states,
            norm,
            scale_shift_table,
            scale_index,
            shift_index,
            chunk_tokens,
            temb_descriptor,
            *temb_inputs,
        )
        if scale_shift_table.requires_grad:
            return fallback("TRAINABLE_SCALE_SHIFT_TABLE", ctx, *common)
        if any(parameter.requires_grad for parameter in norm.parameters()):
            return fallback("TRAINABLE_NORM", ctx, *common)
        if int(chunk_tokens) <= 0:
            return fallback("NONPOSITIVE_CHUNK_SIZE", ctx, *common)
        if hidden_states.ndim != 3 or int(hidden_states.shape[0]) not in (1, 3):
            return fallback("UNSUPPORTED_BATCH", ctx, *common)
        if hidden_states.dtype is not torch.bfloat16:
            return fallback("NON_BF16_HIDDEN", ctx, *common)
        if int(hidden_states.shape[2]) != 1536 and not test_injected:
            return fallback("UNSUPPORTED_WIDTH", ctx, *common)
        if not hidden_states.is_cuda and not test_injected:
            return fallback("NON_CUDA_HIDDEN", ctx, *common)

        ctx.norm = norm
        ctx.scale_shift_table = scale_shift_table
        ctx.scale_index = int(scale_index)
        ctx.shift_index = int(shift_index)
        ctx.chunk_tokens = int(chunk_tokens)
        ctx.temb_descriptor = temb_descriptor
        ctx.temb_input_count = len(temb_inputs)
        ctx.sequence_length = int(hidden_states.shape[1])
        ctx.hidden_dtype = hidden_states.dtype
        ctx.save_for_backward(hidden_states, *temb_inputs)
        temb = rebuild_temb(temb_descriptor, tuple(temb_inputs))
        output = torch.empty_like(hidden_states)
        chunk_count = 0
        for start in range(0, ctx.sequence_length, ctx.chunk_tokens):
            end = min(ctx.sequence_length, start + ctx.chunk_tokens)
            hidden_chunk = hidden_states[:, start:end]
            if fuse_layernorm:
                scale_component = component(temb, ctx.scale_index, start, end)
                shift_component = component(temb, ctx.shift_index, start, end)
                for name, value in (
                    ("scale_component", scale_component),
                    ("shift_component", shift_component),
                ):
                    if value.shape != hidden_chunk.shape:
                        if (
                            value.ndim == 3
                            and value.shape[0] == hidden_chunk.shape[0]
                            and value.shape[1] == 1
                            and value.shape[2] == hidden_chunk.shape[2]
                        ):
                            value = value.expand_as(hidden_chunk)
                        else:
                            raise RuntimeError(
                                f"unsupported exact modulated-norm {name} shape: "
                                f"{tuple(value.shape)} != {tuple(hidden_chunk.shape)}"
                            )
                    if name == "scale_component":
                        scale_component = value
                    else:
                        shift_component = value
                fused_layernorm(
                    hidden_chunk,
                    scale_shift_table,
                    ctx.scale_index,
                    ctx.shift_index,
                    scale_component,
                    shift_component,
                    output[:, start:end],
                    eps=float(norm.eps),
                )
                receipt["layernorm_fused_chunk_count"] += 1
                receipt["layernorm_fused_element_count"] += int(
                    hidden_chunk.numel()
                )
                receipt["materialization_fused_chunk_count"] += 1
            elif fuse_materialization:
                normalized = norm(hidden_chunk.float())
                scale_component = component(temb, ctx.scale_index, start, end)
                shift_component = component(temb, ctx.shift_index, start, end)
                for name, value in (
                    ("scale_component", scale_component),
                    ("shift_component", shift_component),
                ):
                    if value.shape != hidden_chunk.shape:
                        if (
                            value.ndim == 3
                            and value.shape[0] == hidden_chunk.shape[0]
                            and value.shape[1] == 1
                            and value.shape[2] == hidden_chunk.shape[2]
                        ):
                            value = value.expand_as(hidden_chunk)
                        else:
                            raise RuntimeError(
                                f"unsupported exact modulated-norm {name} shape: "
                                f"{tuple(value.shape)} != {tuple(hidden_chunk.shape)}"
                            )
                    if name == "scale_component":
                        scale_component = value
                    else:
                        shift_component = value
                fused_materialization(
                    normalized,
                    scale_shift_table,
                    ctx.scale_index,
                    ctx.shift_index,
                    scale_component,
                    shift_component,
                    output[:, start:end],
                )
                receipt["materialization_fused_chunk_count"] += 1
            else:
                normalized = norm(hidden_chunk.float())
                scale = materialize(
                    scale_shift_table, temb, ctx.scale_index, start, end
                )
                shift = materialize(
                    scale_shift_table, temb, ctx.shift_index, start, end
                )
                for name, value in (("scale", scale), ("shift", shift)):
                    if value.shape != hidden_chunk.shape:
                        if (
                            value.ndim == 3
                            and value.shape[0] == hidden_chunk.shape[0]
                            and value.shape[1] == 1
                            and value.shape[2] == hidden_chunk.shape[2]
                        ):
                            value = value.expand_as(hidden_chunk)
                        else:
                            raise RuntimeError(
                                f"unsupported exact modulated-norm {name} shape: "
                                f"{tuple(value.shape)} != {tuple(hidden_chunk.shape)}"
                            )
                    if name == "scale":
                        scale = value
                    else:
                        shift = value
                fused_epilogue(normalized, scale, shift, output[:, start:end])
            chunk_count += 1
        receipt["fused_call_count"] += 1
        receipt["fused_chunk_count"] += chunk_count
        receipt["fused_element_count"] += int(output.numel())
        if fuse_materialization:
            receipt["materialization_fused_call_count"] += 1
        if fuse_layernorm:
            receipt["layernorm_fused_call_count"] += 1
        return output

    def fused_kernel_backward(ctx: Any, grad_output: Any) -> Any:
        hidden_states, *saved_temb_inputs = ctx.saved_tensors
        hidden_gradient = (
            torch.empty_like(hidden_states) if ctx.needs_input_grad[0] else None
        )
        temb_needs_grad = list(ctx.needs_input_grad[7:])
        if len(temb_needs_grad) != len(saved_temb_inputs):
            raise RuntimeError("modulated-norm timestep gradient arity changed")
        temb_gradients: list[Any | None] = [None for _ in saved_temb_inputs]
        kind, lengths = ctx.temb_descriptor
        if kind == "phase_local":
            if len(lengths) != len(saved_temb_inputs):
                raise RuntimeError("phase-local timestep descriptor arity changed")
        elif kind == "tensor":
            if lengths or len(saved_temb_inputs) != 1:
                raise RuntimeError("tensor timestep descriptor layout changed")
        else:
            raise RuntimeError(f"unsupported timestep descriptor: {kind!r}")
        if any(parameter.requires_grad for parameter in ctx.norm.parameters()):
            raise RuntimeError("fused modulated-norm backward requires frozen norm")
        if ctx.scale_shift_table.requires_grad:
            raise RuntimeError(
                "fused modulated-norm backward requires frozen scale-shift table"
            )

        chunk_count = 0
        hidden_launch_count = 0
        component_launch_count = 0
        tensor4d_launch_count = 0
        if hidden_gradient is not None or any(temb_needs_grad):
            # Row statistics are tiny compared with the eliminated FP32
            # hidden/normalized temporaries and are reused by the deterministic
            # component-gradient kernel.
            mean = torch.empty(
                hidden_states.shape[:2],
                device=hidden_states.device,
                dtype=torch.float32,
            )
            rstd = torch.empty_like(mean)
            hidden_target = (
                hidden_gradient
                if hidden_gradient is not None
                else torch.empty_like(hidden_states)
            )
            for start in range(0, ctx.sequence_length, ctx.chunk_tokens):
                end = min(ctx.sequence_length, start + ctx.chunk_tokens)
                if kind == "phase_local":
                    segment_start = 0
                    for index, (value, length, needs_grad) in enumerate(
                        zip(
                            saved_temb_inputs,
                            lengths,
                            temb_needs_grad,
                            strict=True,
                        )
                    ):
                        segment_end = segment_start + int(length)
                        overlap_start = max(start, segment_start)
                        overlap_end = min(end, segment_end)
                        if overlap_start < overlap_end:
                            hidden_slice = hidden_states[:, overlap_start:overlap_end]
                            grad_slice = grad_output[:, overlap_start:overlap_end]
                            scale_component = (
                                value[:, ctx.scale_index, :]
                                .unsqueeze(1)
                                .expand_as(hidden_slice)
                            )
                            mean_slice = mean[:, overlap_start:overlap_end]
                            rstd_slice = rstd[:, overlap_start:overlap_end]
                            fused_hidden_backward(
                                hidden_slice,
                                grad_slice,
                                ctx.scale_shift_table,
                                ctx.scale_index,
                                scale_component,
                                hidden_target[:, overlap_start:overlap_end],
                                mean_slice,
                                rstd_slice,
                                eps=float(ctx.norm.eps),
                            )
                            hidden_launch_count += 1
                            if needs_grad:
                                chunk_gradient = torch.zeros_like(value)
                                fused_reduced_component_gradient(
                                    hidden_slice,
                                    grad_slice,
                                    mean_slice,
                                    rstd_slice,
                                    chunk_gradient,
                                    ctx.scale_index,
                                    ctx.shift_index,
                                )
                                component_launch_count += 1
                                current = temb_gradients[index]
                                temb_gradients[index] = (
                                    chunk_gradient
                                    if current is None
                                    else current + chunk_gradient
                                )
                        segment_start = segment_end
                else:
                    value = saved_temb_inputs[0]
                    hidden_slice = hidden_states[:, start:end]
                    grad_slice = grad_output[:, start:end]
                    scale_component = (
                        value[:, start:end, ctx.scale_index, :]
                        if value.ndim == 4
                        else value[:, ctx.scale_index, :]
                        .unsqueeze(1)
                        .expand_as(hidden_slice)
                    )
                    mean_slice = mean[:, start:end]
                    rstd_slice = rstd[:, start:end]
                    fused_hidden_backward(
                        hidden_slice,
                        grad_slice,
                        ctx.scale_shift_table,
                        ctx.scale_index,
                        scale_component,
                        hidden_target[:, start:end],
                        mean_slice,
                        rstd_slice,
                        eps=float(ctx.norm.eps),
                    )
                    hidden_launch_count += 1
                    if temb_needs_grad[0]:
                        chunk_gradient = torch.zeros_like(value)
                        if value.ndim == 4:
                            fused_tensor4d_component_gradient(
                                hidden_slice,
                                grad_slice,
                                mean_slice,
                                rstd_slice,
                                chunk_gradient[:, start:end],
                                ctx.scale_index,
                                ctx.shift_index,
                            )
                            layout = "tensor_4d"
                            tensor4d_launch_count += 1
                        elif value.ndim == 3:
                            fused_reduced_component_gradient(
                                hidden_slice,
                                grad_slice,
                                mean_slice,
                                rstd_slice,
                                chunk_gradient,
                                ctx.scale_index,
                                ctx.shift_index,
                            )
                            layout = "tensor_3d"
                            component_launch_count += 1
                        else:
                            raise RuntimeError(
                                "unsupported tensor timestep rank in fused backward: "
                                f"{value.ndim}"
                            )
                        current = temb_gradients[0]
                        temb_gradients[0] = (
                            chunk_gradient
                            if current is None
                            else current + chunk_gradient
                        )
                        counts = receipt["backward_layout_counts"]
                        counts[layout] = int(counts.get(layout, 0)) + 1
                chunk_count += 1

        receipt["backward_call_count"] += 1
        receipt["backward_chunk_count"] += chunk_count
        if hidden_gradient is not None:
            receipt["backward_hidden_element_count"] += int(hidden_states.numel())
        counts = receipt["backward_layout_counts"]
        counts[kind] = int(counts.get(kind, 0)) + chunk_count
        receipt["backward_kernel_call_count"] += 1
        receipt["backward_kernel_chunk_count"] += chunk_count
        receipt["backward_hidden_kernel_launch_count"] += hidden_launch_count
        receipt["backward_component_kernel_launch_count"] += component_launch_count
        receipt["backward_tensor4d_kernel_launch_count"] += tensor4d_launch_count
        return (
            hidden_gradient,
            None,
            None,
            None,
            None,
            None,
            None,
            *temb_gradients,
        )

    @staticmethod
    def backward(ctx: Any, grad_output: Any) -> Any:
        if fuse_backward_kernel:
            return fused_kernel_backward(ctx, grad_output)
        hidden_states, *saved_temb_inputs = ctx.saved_tensors
        hidden_gradient = (
            torch.empty_like(hidden_states) if ctx.needs_input_grad[0] else None
        )
        temb_needs_grad = list(ctx.needs_input_grad[7:])
        if len(temb_needs_grad) != len(saved_temb_inputs):
            raise RuntimeError("modulated-norm timestep gradient arity changed")
        temb_gradients: list[Any | None] = [None for _ in saved_temb_inputs]
        kind, lengths = ctx.temb_descriptor
        if kind == "phase_local":
            if len(lengths) != len(saved_temb_inputs):
                raise RuntimeError("phase-local timestep descriptor arity changed")
        elif kind == "tensor":
            if lengths or len(saved_temb_inputs) != 1:
                raise RuntimeError("tensor timestep descriptor layout changed")
        else:
            raise RuntimeError(f"unsupported timestep descriptor: {kind!r}")
        if any(parameter.requires_grad for parameter in ctx.norm.parameters()):
            raise RuntimeError("explicit modulated-norm backward requires frozen norm")
        if ctx.scale_shift_table.requires_grad:
            raise RuntimeError(
                "explicit modulated-norm backward requires frozen scale-shift table"
            )
        normalized_shape = [int(hidden_states.shape[-1])]
        epsilon = float(ctx.norm.eps)
        chunk_count = 0

        if hidden_gradient is not None or any(temb_needs_grad):
            temb = rebuild_temb(ctx.temb_descriptor, tuple(saved_temb_inputs))
            for start in range(0, ctx.sequence_length, ctx.chunk_tokens):
                end = min(ctx.sequence_length, start + ctx.chunk_tokens)
                hidden_float = hidden_states[:, start:end].float()
                normalized, mean, rstd = torch.native_layer_norm(
                    hidden_float,
                    normalized_shape,
                    None,
                    None,
                    epsilon,
                )
                scale = materialize(
                    ctx.scale_shift_table,
                    temb,
                    ctx.scale_index,
                    start,
                    end,
                )
                grad_float = grad_output[:, start:end].float()
                if hidden_gradient is not None:
                    grad_normalized = grad_float * (1.0 + scale)
                    hidden_chunk, _weight_gradient, _bias_gradient = (
                        torch.ops.aten.native_layer_norm_backward.default(
                            grad_normalized,
                            hidden_float,
                            normalized_shape,
                            mean,
                            rstd,
                            None,
                            None,
                            [True, False, False],
                        )
                    )
                    hidden_gradient[:, start:end].copy_(
                        hidden_chunk.to(hidden_states.dtype)
                    )

                if kind == "phase_local":
                    segment_start = 0
                    for index, (value, length, needs_grad) in enumerate(
                        zip(
                            saved_temb_inputs,
                            lengths,
                            temb_needs_grad,
                            strict=True,
                        )
                    ):
                        segment_end = segment_start + int(length)
                        overlap_start = max(start, segment_start)
                        overlap_end = min(end, segment_end)
                        if needs_grad and overlap_start < overlap_end:
                            local = slice(
                                overlap_start - start, overlap_end - start
                            )
                            chunk_gradient = torch.zeros_like(value)
                            chunk_gradient[:, ctx.scale_index, :].copy_(
                                (
                                    grad_float[:, local]
                                    * normalized[:, local]
                                )
                                .sum(dim=1)
                                .to(value.dtype)
                            )
                            chunk_gradient[:, ctx.shift_index, :].copy_(
                                grad_float[:, local].sum(dim=1).to(value.dtype)
                            )
                            current = temb_gradients[index]
                            temb_gradients[index] = (
                                chunk_gradient
                                if current is None
                                else current + chunk_gradient
                            )
                        segment_start = segment_end
                elif temb_needs_grad[0]:
                    value = saved_temb_inputs[0]
                    chunk_gradient = torch.zeros_like(value)
                    scale_gradient = grad_float * normalized
                    shift_gradient = grad_float
                    if value.ndim == 4:
                        chunk_gradient[:, start:end, ctx.scale_index, :].copy_(
                            scale_gradient.to(value.dtype)
                        )
                        chunk_gradient[:, start:end, ctx.shift_index, :].copy_(
                            shift_gradient.to(value.dtype)
                        )
                        layout = "tensor_4d"
                    elif value.ndim == 3:
                        chunk_gradient[:, ctx.scale_index, :].copy_(
                            scale_gradient.sum(dim=1).to(value.dtype)
                        )
                        chunk_gradient[:, ctx.shift_index, :].copy_(
                            shift_gradient.sum(dim=1).to(value.dtype)
                        )
                        layout = "tensor_3d"
                    else:
                        raise RuntimeError(
                            "unsupported tensor timestep rank in explicit backward: "
                            f"{value.ndim}"
                        )
                    current = temb_gradients[0]
                    temb_gradients[0] = (
                        chunk_gradient
                        if current is None
                        else current + chunk_gradient
                    )
                    counts = receipt["backward_layout_counts"]
                    counts[layout] = int(counts.get(layout, 0)) + 1
                chunk_count += 1

        receipt["backward_call_count"] += 1
        receipt["backward_chunk_count"] += chunk_count
        if hidden_gradient is not None:
            receipt["backward_hidden_element_count"] += int(hidden_states.numel())
        counts = receipt["backward_layout_counts"]
        counts[kind] = int(counts.get(kind, 0)) + chunk_count
        return (
            hidden_gradient,
            None,
            None,
            None,
            None,
            None,
            None,
            *temb_gradients,
        )

    function.forward = forward
    if fuse_backward:
        function.backward = backward
    transformer_module._distillschedule_modulated_norm_receipt = receipt
    receipt["enabled"] = True
    receipt["implementation"] = (
        "TEST_INJECTED"
        if test_injected
        else (
            (
                "TRITON_B1_B3_BF16_INPUT_FP32_LAYERNORM_TABLE_COMPONENT_"
                "AND_MODULATION_EPILOGUE_"
                if fuse_layernorm
                else "TRITON_B1_B3_FP32_TABLE_COMPONENT_AND_MODULATION_EPILOGUE_"
            )
            + (
                (
                    "TRITON_FUSED_LAYERNORM_AND_COMPONENT_BACKWARD"
                    if fuse_backward_kernel
                    else "EXPLICIT_NATIVE_LAYERNORM_BACKWARD"
                )
                if fuse_backward
                else "SOURCE_BACKWARD"
            )
            if fuse_materialization
            else (
                "TRITON_B1_B3_FP32_MODULATION_EPILOGUE_"
                + (
                    (
                        "TRITON_FUSED_LAYERNORM_AND_COMPONENT_BACKWARD"
                        if fuse_backward_kernel
                        else "EXPLICIT_NATIVE_LAYERNORM_BACKWARD"
                    )
                    if fuse_backward
                    else "SOURCE_BACKWARD"
                )
            )
        )
    )
    return receipt


def warmup_exact_modulated_norm_epilogue(
    receipt: dict[str, Any], *, device: Any, chunk_tokens: int = 512
) -> dict[str, Any]:
    if not receipt.get("enabled"):
        return receipt
    if receipt.get("layernorm_requested"):
        from distillschedule.runtime.wan13b_fused_layernorm_kernel import (
            warmup as warmup_layernorm,
        )

        details = warmup_layernorm(device, chunk_tokens=int(chunk_tokens))
        details["fuse_materialization"] = True
        details["fuse_layernorm"] = True
    else:
        from distillschedule.runtime.wan13b_modulated_norm_kernel import warmup

        details = warmup(
            device,
            fuse_materialization=bool(receipt.get("materialization_requested")),
            chunk_tokens=int(chunk_tokens),
        )
        details["fuse_layernorm"] = False
    if receipt.get("backward_kernel_requested"):
        from distillschedule.runtime.wan13b_fused_layernorm_backward_kernel import (
            warmup as warmup_backward,
        )

        details["fused_backward"] = warmup_backward(
            device, tokens=int(chunk_tokens)
        )
    else:
        details["fused_backward"] = {"status": "NOT_REQUESTED"}
    receipt["warmup"] = {"status": "PASS", **details}
    return receipt


__all__ = [
    "SUPPORTED_HELIOS_TRANSFORMER_SHA256",
    "install_exact_modulated_norm_epilogue",
    "warmup_exact_modulated_norm_epilogue",
]
