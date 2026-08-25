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
    transformer_module: Any | None = None,
    fused_epilogue: Any | None = None,
    fused_materialization: Any | None = None,
) -> dict[str, Any]:
    """Patch only the pinned custom Function forward; retain source backward."""

    receipt: dict[str, Any] = {
        "requested": bool(enabled),
        "enabled": False,
        "role": str(role),
        "scope": "ALL_ROLE_B1_MODULATED_NORM_FORWARD_EPILOGUE",
        "semantic_contract": (
            "PYTORCH_FP32_LAYERNORM_UNCHANGED;ADD_RN_F32_ONE_PLUS_SCALE;"
            "MUL_RN_F32;ADD_RN_F32_SHIFT;BF16_RN_STORE;SOURCE_BACKWARD_UNCHANGED"
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
    test_injected = (
        fused_epilogue is not None or fused_materialization is not None
    )
    if not test_injected and observed_sha != SUPPORTED_HELIOS_TRANSFORMER_SHA256:
        raise RuntimeError(
            "Helios transformer source changed: "
            f"{observed_sha} != {SUPPORTED_HELIOS_TRANSFORMER_SHA256}"
        )
    if fused_epilogue is None:
        from distillschedule.runtime.helios_modulated_norm_kernel import (
            fused_modulated_norm_epilogue,
        )

        fused_epilogue = fused_modulated_norm_epilogue
    if fuse_materialization and fused_materialization is None:
        from distillschedule.runtime.helios_modulated_norm_kernel import (
            fused_modulated_norm_materialization,
        )

        fused_materialization = fused_modulated_norm_materialization

    function = transformer_module._TembAwareChunkedNorm
    original_forward = function.forward
    materialize = transformer_module._materialize_scale_shift_component
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
        if hidden_states.ndim != 3 or int(hidden_states.shape[0]) != 1:
            return fallback("NON_B1_HIDDEN", ctx, *common)
        if hidden_states.dtype is not torch.bfloat16:
            return fallback("NON_BF16_HIDDEN", ctx, *common)
        if int(hidden_states.shape[2]) != 5120 and not test_injected:
            return fallback("UNSUPPORTED_WIDTH", ctx, *common)
        if not hidden_states.is_cuda and not test_injected:
            return fallback("NON_CUDA_HIDDEN", ctx, *common)

        rebuild_temb = transformer_module._rebuild_temb
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
            normalized = norm(hidden_chunk.float())
            if fuse_materialization:
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
        return output

    function.forward = forward
    transformer_module._distillschedule_modulated_norm_receipt = receipt
    receipt["enabled"] = True
    receipt["implementation"] = (
        "TEST_INJECTED"
        if test_injected
        else (
            "TRITON_B1_FP32_TABLE_COMPONENT_AND_MODULATION_EPILOGUE_SOURCE_BACKWARD"
            if fuse_materialization
            else "TRITON_B1_FP32_MODULATION_EPILOGUE_SOURCE_BACKWARD"
        )
    )
    return receipt


def warmup_exact_modulated_norm_epilogue(
    receipt: dict[str, Any], *, device: Any
) -> dict[str, Any]:
    if not receipt.get("enabled"):
        return receipt
    from distillschedule.runtime.helios_modulated_norm_kernel import warmup

    receipt["warmup"] = {
        "status": "PASS",
        **warmup(
            device,
            fuse_materialization=bool(receipt.get("materialization_requested")),
        ),
    }
    return receipt


__all__ = [
    "SUPPORTED_HELIOS_TRANSFORMER_SHA256",
    "install_exact_modulated_norm_epilogue",
    "warmup_exact_modulated_norm_epilogue",
]
