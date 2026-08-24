"""Default-off, Generator-no-grad-only Helios gated-residual fusion."""
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
    role: str,
    transformer_module: Any | None = None,
    fused_chunk: Any | None = None,
) -> dict[str, Any]:
    """Patch the pinned module-level helper while preserving every grad path."""

    receipt: dict[str, Any] = {
        "requested": bool(enabled),
        "enabled": False,
        "role": str(role),
        "scope": "GENERATOR_NOGRAD_GATED_RESIDUAL_ONLY",
        "semantic_contract": (
            "BF16_LOADS;MUL_RN_F32;ADD_RN_F32;BF16_RN_STORE;"
            "GRAD_AND_CRITIC_PATHS_UNCHANGED"
        ),
        "fused_call_count": 0,
        "fused_chunk_count": 0,
        "fused_element_count": 0,
        "scope_active_call_count": 0,
        "grad_bypass_count": 0,
        "non_generator_bypass_count": 0,
        "non_cuda_storage_bypass_count": 0,
        "reference_fallback_count": 0,
        "reference_fallback_reasons": {},
        "warmup": {"status": "NOT_RUN"},
    }
    if not enabled:
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
    test_injected = fused_chunk is not None
    if fused_chunk is None:
        from distillschedule.runtime.helios_residual_kernel import (
            fused_gated_residual_chunk,
        )

        fused_chunk = fused_gated_residual_chunk

    def _fallback(reason: str, *args: Any, **kwargs: Any) -> Any:
        receipt["reference_fallback_count"] += 1
        reasons = receipt["reference_fallback_reasons"]
        reasons[reason] = int(reasons.get(reason, 0)) + 1
        return original(*args, **kwargs)

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
        import torch

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
            receipt["grad_bypass_count"] += 1
            return original(*common)
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
        "TRITON_B1_TILE_MUL_RN_F32_ADD_RN_F32_BF16_STORE"
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

    receipt["warmup"] = {"status": "PASS", **warmup(device)}
    return receipt


__all__ = [
    "SUPPORTED_HELIOS_TRANSFORMER_SHA256",
    "generator_nograd_residual_fusion_scope",
    "install_generator_nograd_residual_fusion",
    "warmup_generator_nograd_residual_fusion",
]
