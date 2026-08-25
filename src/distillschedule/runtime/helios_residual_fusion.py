"""Default-off exact Helios gated-residual fusions.

The established candidate fuses only the Generator no-grad critic-rollout
scope.  A second, independently gated candidate handles grad-enabled calls
when the AdaLN table is frozen, including the trainable tensor and phase-local
timestep projections observed in the real Helios backward.  The grad path uses
a custom autograd function that preserves the pinned FP32 rounding contract
and never falls back after its forward has been selected.
"""
from __future__ import annotations

import functools
import hashlib
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Iterator


SUPPORTED_HELIOS_TRANSFORMER_SHA256 = (
    "5d3ffe37cdea3c04a156106f6246c82f91010a1eaf0dd61529cd74083065a886"
)


_GENERATOR_NOGRAD_SCOPE: ContextVar[bool] = ContextVar(
    "distillschedule_helios_generator_nograd_residual_scope",
    default=False,
)


@contextmanager
def generator_nograd_residual_fusion_scope() -> Iterator[None]:
    """Mark the already-audited Generator critic-rollout model-call boundary.

    The pinned Helios block tag is not a reliable execution-role discriminator
    after composable FSDP wrapping.  The caller already owns the stronger
    semantic boundary: ``run_generator_trace_rollout_batch`` is decorated with
    ``torch.no_grad`` and invokes only the persistent Generator.  A ContextVar
    keeps the marker local to that call without adding synchronization or tensor
    operations.
    """

    token = _GENERATOR_NOGRAD_SCOPE.set(True)
    try:
        yield
    finally:
        _GENERATOR_NOGRAD_SCOPE.reset(token)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def install_generator_nograd_residual_fusion(
    *,
    enabled: bool,
    exact_grad_enabled: bool = False,
    role: str,
    transformer_module: Any | None = None,
    fused_chunk: Any | None = None,
    fused_backward_chunk: Any | None = None,
    fused_trainable_backward_chunk: Any | None = None,
) -> dict[str, Any]:
    """Patch the pinned module helper for the explicitly requested exact paths."""

    receipt: dict[str, Any] = {
        "requested": bool(enabled),
        "exact_grad_requested": bool(exact_grad_enabled),
        "enabled": False,
        "role": str(role),
        "scope": (
            "GENERATOR_NOGRAD_AND_TRAINABLE_TEMB_EXACT_GRAD_GATED_RESIDUAL"
            if enabled and exact_grad_enabled
            else (
                "TRAINABLE_TEMB_EXACT_GRAD_GATED_RESIDUAL_ONLY"
                if exact_grad_enabled
                else "GENERATOR_NOGRAD_GATED_RESIDUAL_ONLY"
            )
        ),
        "semantic_contract": (
            "BF16_LOADS;MUL_RN_F32;ADD_RN_F32;BF16_RN_STORE;"
            + (
                "TRAINABLE_TEMB_GRAD_RESIDUAL_MUL_RN_F32_BF16_RN_STORE;"
                "EXPLICIT_TENSOR_OR_PHASE_LOCAL_TEMB_GRADIENT;"
                "NO_TRAINABLE_ADALN_TABLE"
                if exact_grad_enabled
                else "GRAD_AND_CRITIC_PATHS_UNCHANGED"
            )
        ),
        "fused_call_count": 0,
        "fused_chunk_count": 0,
        "fused_element_count": 0,
        "scope_active_call_count": 0,
        "grad_bypass_count": 0,
        "grad_scope_active_call_count": 0,
        "grad_fused_call_count": 0,
        "grad_fused_chunk_count": 0,
        "grad_fused_element_count": 0,
        "grad_backward_call_count": 0,
        "grad_backward_chunk_count": 0,
        "grad_backward_element_count": 0,
        "grad_reference_fallback_count": 0,
        "grad_reference_fallback_reasons": {},
        "grad_saved_residual_tensor_count": 0,
        "grad_trainable_temb_call_count": 0,
        "grad_frozen_temb_call_count": 0,
        "grad_temb_descriptor_counts": {},
        "grad_temb_dtype_counts": {},
        "grad_temb_gradient_tensor_count": 0,
        "grad_phase_local_fused_staging_chunk_count": 0,
        "grad_phase_local_fused_staging_element_count": 0,
        "non_generator_bypass_count": 0,
        "non_cuda_storage_bypass_count": 0,
        "reference_fallback_count": 0,
        "reference_fallback_reasons": {},
        "warmup": {"status": "NOT_RUN"},
    }
    if not enabled and not exact_grad_enabled:
        receipt["disabled_reason"] = "NOT_REQUESTED"
        return receipt
    if role != "generator":
        receipt["disabled_reason"] = "GENERATOR_INSTALL_ONLY"
        return receipt

    if transformer_module is None:
        from helios.modules import transformer_helios as transformer_module

    if getattr(transformer_module, "_distillschedule_residual_fusion_receipt", None):
        raise RuntimeError("Helios residual fusion was already installed")
    source_path = Path(transformer_module.__file__).resolve()
    observed_sha = _sha256(source_path)
    receipt["transformer_source"] = str(source_path)
    receipt["transformer_sha256"] = observed_sha
    receipt["expected_transformer_sha256"] = SUPPORTED_HELIOS_TRANSFORMER_SHA256
    if fused_chunk is None and observed_sha != SUPPORTED_HELIOS_TRANSFORMER_SHA256:
        raise RuntimeError(
            "Helios transformer source changed: "
            f"{observed_sha} != {SUPPORTED_HELIOS_TRANSFORMER_SHA256}"
        )

    original = transformer_module._apply_chunked_gated_residual
    materialize = transformer_module._materialize_scale_shift_component
    flatten_temb = transformer_module._flatten_temb_inputs
    rebuild_temb = transformer_module._rebuild_temb
    test_injected = fused_chunk is not None
    if fused_chunk is None:
        from distillschedule.runtime.helios_residual_kernel import (
            fused_gated_residual_backward_chunk,
            fused_gated_residual_chunk,
            fused_gated_residual_trainable_temb_backward_chunk,
        )

        fused_chunk = fused_gated_residual_chunk
        fused_backward_chunk = fused_gated_residual_backward_chunk
        fused_trainable_backward_chunk = (
            fused_gated_residual_trainable_temb_backward_chunk
        )
    elif exact_grad_enabled and fused_backward_chunk is None:
        raise ValueError(
            "exact grad residual fusion tests must inject the backward tile"
        )

    def _fallback(reason: str, *args: Any, **kwargs: Any) -> Any:
        receipt["reference_fallback_count"] += 1
        reasons = receipt["reference_fallback_reasons"]
        reasons[reason] = int(reasons.get(reason, 0)) + 1
        return original(*args, **kwargs)

    def _grad_fallback(reason: str, *args: Any, **kwargs: Any) -> Any:
        receipt["grad_reference_fallback_count"] += 1
        reasons = receipt["grad_reference_fallback_reasons"]
        reasons[reason] = int(reasons.get(reason, 0)) + 1
        return original(*args, **kwargs)

    import torch

    class _ExactTrainableTembResidual(torch.autograd.Function):
        """Exact residual derivative with explicit timestep-input gradients."""

        @staticmethod
        def forward(
            ctx: Any,
            hidden_states: Any,
            residual: Any,
            scale_shift_table: Any,
            component_index: int,
            chunk_tokens: int,
            temb_descriptor: Any,
            logical_component: str,
            residual_storage: str,
            *temb_inputs: Any,
        ) -> Any:
            ctx.scale_shift_table = scale_shift_table
            ctx.component_index = int(component_index)
            ctx.chunk_tokens = int(chunk_tokens)
            ctx.temb_descriptor = temb_descriptor
            ctx.logical_component = str(logical_component)
            ctx.residual_storage = str(residual_storage)
            ctx.hidden_dtype = hidden_states.dtype
            ctx.residual_dtype = residual.dtype
            ctx.sequence_length = int(hidden_states.shape[1])
            ctx.temb_input_count = len(temb_inputs)
            # The residual value is required by dL/d(temb).  This matches the
            # pinned reference autograd path, which also saves the residual for
            # trainable timestep projections.
            ctx.save_for_backward(residual, *temb_inputs)
            receipt["grad_saved_residual_tensor_count"] += 1
            temb_value = rebuild_temb(temb_descriptor, tuple(temb_inputs))
            output = torch.empty_like(hidden_states)
            chunk_count = 0
            for start in range(0, ctx.sequence_length, ctx.chunk_tokens):
                end = min(ctx.sequence_length, start + ctx.chunk_tokens)
                gate = materialize(
                    scale_shift_table,
                    temb_value,
                    ctx.component_index,
                    start,
                    end,
                )
                hidden_chunk = hidden_states[:, start:end]
                if gate.shape != hidden_chunk.shape:
                    if (
                        gate.ndim == 3
                        and gate.shape[0] == hidden_chunk.shape[0]
                        and gate.shape[1] == 1
                        and gate.shape[2] == hidden_chunk.shape[2]
                    ):
                        gate = gate.expand_as(hidden_chunk)
                    else:
                        raise RuntimeError(
                            "unsupported exact residual gate shape: "
                            f"{tuple(gate.shape)} != {tuple(hidden_chunk.shape)}"
                        )
                fused_chunk(
                    hidden_chunk,
                    residual[:, start:end],
                    gate,
                    output[:, start:end],
                )
                chunk_count += 1
            receipt["grad_fused_call_count"] += 1
            receipt["grad_fused_chunk_count"] += chunk_count
            receipt["grad_fused_element_count"] += int(output.numel())
            return output

        @staticmethod
        def backward(ctx: Any, grad_output: Any) -> tuple[Any, ...]:
            residual_value, *saved_temb_inputs = ctx.saved_tensors
            saved_temb_inputs = tuple(saved_temb_inputs)
            hidden_gradient = (
                grad_output.to(ctx.hidden_dtype)
                if ctx.needs_input_grad[0]
                else None
            )
            residual_gradient = (
                torch.empty_like(grad_output, dtype=ctx.residual_dtype)
                if ctx.needs_input_grad[1]
                else None
            )
            temb_needs_grad = tuple(bool(value) for value in ctx.needs_input_grad[8:])
            temb_gradients: list[Any | None] = [None for _ in saved_temb_inputs]
            chunk_count = 0
            if residual_gradient is not None or any(temb_needs_grad):
                temb_value = rebuild_temb(
                    ctx.temb_descriptor,
                    saved_temb_inputs,
                )
                for start in range(0, ctx.sequence_length, ctx.chunk_tokens):
                    end = min(ctx.sequence_length, start + ctx.chunk_tokens)
                    gate = materialize(
                        ctx.scale_shift_table,
                        temb_value,
                        ctx.component_index,
                        start,
                        end,
                    )
                    grad_chunk = grad_output[:, start:end]
                    if gate.shape != grad_chunk.shape:
                        if (
                            gate.ndim == 3
                            and gate.shape[0] == grad_chunk.shape[0]
                            and gate.shape[1] == 1
                            and gate.shape[2] == grad_chunk.shape[2]
                        ):
                            gate = gate.expand_as(grad_chunk)
                        else:
                            raise RuntimeError(
                                "unsupported exact residual gate shape: "
                                f"{tuple(gate.shape)} != {tuple(grad_chunk.shape)}"
                            )
                    descriptor_kind, descriptor_lengths = ctx.temb_descriptor
                    use_fused_phase_local_staging = bool(
                        descriptor_kind == "phase_local"
                        and fused_trainable_backward_chunk is not None
                        and residual_gradient is not None
                        and any(temb_needs_grad)
                        and all(
                            value.dtype is torch.bfloat16
                            for value in saved_temb_inputs
                        )
                    )
                    gate_gradient = None
                    if use_fused_phase_local_staging:
                        gate_gradient = torch.empty_like(grad_chunk)
                        fused_trainable_backward_chunk(
                            grad_chunk,
                            residual_value[:, start:end],
                            gate,
                            residual_gradient[:, start:end],
                            gate_gradient,
                        )
                        receipt["grad_phase_local_fused_staging_chunk_count"] += 1
                        receipt["grad_phase_local_fused_staging_element_count"] += int(
                            gate_gradient.numel()
                        )
                    elif residual_gradient is not None:
                        fused_backward_chunk(
                            grad_chunk,
                            gate,
                            residual_gradient[:, start:end],
                        )
                    if any(temb_needs_grad) and gate_gradient is None:
                        residual_chunk = residual_value[:, start:end]
                        gate_gradient = (
                            residual_chunk.float()
                            * grad_chunk.float()
                        )
                    def accumulate_component_gradient(
                        input_index: int,
                        value: Any,
                        partial: Any,
                        *,
                        token_start: int | None = None,
                        token_end: int | None = None,
                    ) -> None:
                        current = temb_gradients[input_index]
                        first_contribution = current is None
                        if first_contribution:
                            current = torch.zeros_like(value)
                            temb_gradients[input_index] = current
                        if value.ndim == 4:
                            if token_start is None or token_end is None:
                                raise RuntimeError(
                                    "token bounds are required for rank-four timestep gradients"
                                )
                            target = current[
                                :, token_start:token_end, ctx.component_index, :
                            ]
                            # Chunks cover disjoint token ranges, so no arithmetic
                            # is required for the rank-four form.
                            target.copy_(partial)
                        else:
                            target = current[:, ctx.component_index, :]
                            if first_contribution:
                                target.copy_(partial)
                            else:
                                # Preserve the source's chunk-order BF16 additions
                                # without allocating a zero-filled full descriptor
                                # tensor and adding its five untouched components.
                                target.add_(partial)
                        receipt["grad_temb_gradient_tensor_count"] += 1

                    if descriptor_kind == "tensor":
                        if len(saved_temb_inputs) != 1:
                            raise RuntimeError(
                                "tensor timestep descriptor requires one input"
                            )
                        value = saved_temb_inputs[0]
                        if temb_needs_grad[0]:
                            if value.ndim == 4:
                                assert gate_gradient is not None
                                accumulate_component_gradient(
                                    0,
                                    value,
                                    gate_gradient.to(value.dtype),
                                    token_start=start,
                                    token_end=end,
                                )
                            elif value.ndim == 3:
                                assert gate_gradient is not None
                                accumulate_component_gradient(
                                    0,
                                    value,
                                    gate_gradient.sum(dim=1).to(value.dtype),
                                )
                            else:
                                raise RuntimeError(
                                    "unsupported tensor timestep rank: "
                                    f"{value.ndim}"
                                )
                    elif descriptor_kind == "phase_local":
                        if len(descriptor_lengths) != len(saved_temb_inputs):
                            raise RuntimeError(
                                "phase-local timestep descriptor/input mismatch"
                            )
                        segment_start = 0
                        for index, (value, length, needs_grad) in enumerate(
                            zip(
                                saved_temb_inputs,
                                descriptor_lengths,
                                temb_needs_grad,
                                strict=True,
                            )
                        ):
                            segment_end = segment_start + int(length)
                            overlap_start = max(start, segment_start)
                            overlap_end = min(end, segment_end)
                            if needs_grad and overlap_start < overlap_end:
                                local_start = overlap_start - start
                                local_end = overlap_end - start
                                assert gate_gradient is not None
                                accumulate_component_gradient(
                                    index,
                                    value,
                                    # PhaseLocalTimestepProjection expands the
                                    # input before ``component.float()``.  Its
                                    # source backward therefore casts each
                                    # token gradient back to the input dtype
                                    # before ExpandBackward performs the token
                                    # reduction.  Reversing this order changes
                                    # BF16 rounding substantially.
                                    gate_gradient[:, local_start:local_end]
                                    .to(value.dtype)
                                    .sum(dim=1),
                                )
                            segment_start = segment_end
                    else:
                        raise RuntimeError(
                            f"unsupported timestep descriptor: {descriptor_kind!r}"
                        )
                    chunk_count += 1
            receipt["grad_backward_call_count"] += 1
            receipt["grad_backward_chunk_count"] += chunk_count
            if residual_gradient is not None:
                receipt["grad_backward_element_count"] += int(
                    residual_gradient.numel()
                )
            return (
                hidden_gradient,
                residual_gradient,
                None,
                None,
                None,
                None,
                None,
                None,
                *temb_gradients,
            )

    @functools.wraps(original)
    def wrapped(
        hidden_states: Any,
        residual: Any,
        scale_shift_table: Any,
        temb: Any,
        component_index: int,
        chunk_tokens: int,
        logical_component: str = "unknown",
        residual_storage: str = "cpu_inplace",
    ) -> Any:
        common = (
            hidden_states,
            residual,
            scale_shift_table,
            temb,
            component_index,
            chunk_tokens,
            logical_component,
            residual_storage,
        )
        if torch.is_grad_enabled():
            if not exact_grad_enabled:
                receipt["grad_bypass_count"] += 1
                return original(*common)
            receipt["grad_scope_active_call_count"] += 1
            if residual_storage != "cuda_out_of_place":
                return _grad_fallback("NON_CUDA_OUT_OF_PLACE_STORAGE", *common)
            if scale_shift_table.requires_grad:
                return _grad_fallback("TRAINABLE_SCALE_SHIFT_TABLE", *common)
            if chunk_tokens <= 0:
                return _grad_fallback("NONPOSITIVE_CHUNK_SIZE", *common)
            if hidden_states.shape != residual.shape:
                return _grad_fallback("SHAPE_MISMATCH", *common)
            if hidden_states.ndim != 3 or int(hidden_states.shape[0]) != 1:
                return _grad_fallback("NON_B1_HIDDEN", *common)
            if (
                hidden_states.dtype is not torch.bfloat16
                or residual.dtype is not torch.bfloat16
            ):
                return _grad_fallback("NON_BF16_INPUT", *common)
            if (
                not hidden_states.is_contiguous()
                or not residual.is_contiguous()
            ):
                return _grad_fallback("NONCONTIGUOUS_INPUT", *common)
            if (
                (not hidden_states.is_cuda or not residual.is_cuda)
                and not test_injected
            ):
                return _grad_fallback("NON_CUDA_INPUT", *common)
            temb_descriptor, temb_inputs = flatten_temb(temb)
            if not hidden_states.requires_grad and not residual.requires_grad:
                return _grad_fallback("NO_DIFFERENTIABLE_INPUT", *common)
            descriptor_key = str(temb_descriptor[0])
            descriptor_counts = receipt["grad_temb_descriptor_counts"]
            descriptor_counts[descriptor_key] = int(
                descriptor_counts.get(descriptor_key, 0)
            ) + 1
            dtype_counts = receipt["grad_temb_dtype_counts"]
            for value in temb_inputs:
                dtype_key = str(value.dtype)
                dtype_counts[dtype_key] = int(dtype_counts.get(dtype_key, 0)) + 1
            if any(value.requires_grad for value in temb_inputs):
                receipt["grad_trainable_temb_call_count"] += 1
            else:
                receipt["grad_frozen_temb_call_count"] += 1
            return _ExactTrainableTembResidual.apply(
                hidden_states,
                residual,
                scale_shift_table,
                int(component_index),
                int(chunk_tokens),
                temb_descriptor,
                str(logical_component),
                str(residual_storage),
                *temb_inputs,
            )
        if not _GENERATOR_NOGRAD_SCOPE.get():
            receipt["non_generator_bypass_count"] += 1
            return original(*common)
        receipt["scope_active_call_count"] += 1
        if residual_storage != "cuda_out_of_place":
            receipt["non_cuda_storage_bypass_count"] += 1
            return original(*common)
        if scale_shift_table.requires_grad:
            return _fallback("TRAINABLE_SCALE_SHIFT_TABLE", *common)
        if chunk_tokens <= 0:
            return _fallback("NONPOSITIVE_CHUNK_SIZE", *common)
        if hidden_states.ndim != 3 or int(hidden_states.shape[0]) != 1:
            return _fallback("NON_B1_HIDDEN", *common)
        if hidden_states.dtype is not torch.bfloat16 or residual.dtype is not torch.bfloat16:
            return _fallback("NON_BF16_INPUT", *common)
        if (not hidden_states.is_cuda or not residual.is_cuda) and not test_injected:
            return _fallback("NON_CUDA_INPUT", *common)

        sequence_length = int(hidden_states.shape[1])
        output = torch.empty_like(hidden_states)
        chunk_count = 0
        for start in range(0, sequence_length, int(chunk_tokens)):
            end = min(sequence_length, start + int(chunk_tokens))
            gate = materialize(
                scale_shift_table,
                temb,
                int(component_index),
                start,
                end,
            )
            fused_chunk(
                hidden_states[:, start:end],
                residual[:, start:end],
                gate,
                output[:, start:end],
            )
            chunk_count += 1
        receipt["fused_call_count"] += 1
        receipt["fused_chunk_count"] += chunk_count
        receipt["fused_element_count"] += int(output.numel())
        return output

    transformer_module._apply_chunked_gated_residual = wrapped
    transformer_module._distillschedule_residual_fusion_receipt = receipt
    receipt["enabled"] = True
    receipt["implementation"] = (
        (
            "TRITON_B1_TILE_FORWARD_AND_TRAINABLE_TEMB_EXACT_BACKWARD"
            if exact_grad_enabled
            else "TRITON_B1_TILE_MUL_RN_F32_ADD_RN_F32_BF16_STORE"
        )
        if not test_injected
        else "TEST_INJECTED"
    )
    return receipt


def warmup_generator_nograd_residual_fusion(
    receipt: dict[str, Any], *, device: Any
) -> dict[str, Any]:
    if not receipt.get("enabled"):
        return receipt
    from distillschedule.runtime.helios_residual_kernel import warmup

    receipt["warmup"] = {
        "status": "PASS",
        **warmup(
            device,
            include_backward=bool(receipt.get("exact_grad_requested")),
        ),
    }
    return receipt


__all__ = [
    "SUPPORTED_HELIOS_TRANSFORMER_SHA256",
    "generator_nograd_residual_fusion_scope",
    "install_generator_nograd_residual_fusion",
    "warmup_generator_nograd_residual_fusion",
]
