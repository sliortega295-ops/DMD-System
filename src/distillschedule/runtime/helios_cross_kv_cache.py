"""Exact cross-attention encoder K/V reuse for one Helios no-grad rollout.

The pinned Helios implementation projects the same text conditioning through
every block's cross-attention ``to_k``/``to_v`` (or fused ``to_kv``) on every
DiT call. During one Generator no-grad rollout, the conditioning and model
version are invariant. This module therefore keeps each block's *pre-norm*
encoder K/V on the GPU for that rollout only. Query projection, K norm,
attention, output projection, every DiT call, and every FSDP collective remain
unchanged.

The wrapper is source pinned and disabled by default. It fails closed on a
scope, model, dtype/device/shape, call-count, or lifecycle mismatch.
"""
from __future__ import annotations

import contextvars
import hashlib
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch


EXPECTED_HELIOS_TRANSFORMER_SHA256 = (
    "5d3ffe37cdea3c04a156106f6246c82f91010a1eaf0dd61529cd74083065a886"
)


def _tensor_signature(value: torch.Tensor) -> tuple[Any, ...]:
    return (
        tuple(value.shape),
        str(value.dtype),
        str(value.device),
        tuple(value.stride()),
        int(value.storage_offset()),
        str(value.layout),
    )


class _ScopeState:
    def __init__(
        self,
        *,
        expected_model_calls: int,
        attentions: tuple[Any, ...],
    ) -> None:
        self.expected_model_calls = int(expected_model_calls)
        self.attentions = attentions
        self.allowed_attention_ids = {id(value) for value in attentions}
        self.cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.encoder_signatures: dict[int, tuple[Any, ...]] = {}
        self.cache_versions: dict[int, tuple[int, int]] = {}
        self.cross_projection_call_count = 0
        self.self_projection_call_count = 0
        self.query_projection_count = 0
        self.cache_fill_count = 0
        self.cache_reuse_count = 0
        self.cache_miss_count = 0
        self.encoder_signature_mismatch_count = 0
        self.cached_tensor_mutation_count = 0
        self.grad_enabled_call_count = 0
        self.unexpected_cross_attention_count = 0


_ACTIVE_SCOPE: contextvars.ContextVar[_ScopeState | None] = contextvars.ContextVar(
    "distillschedule_helios_cross_kv_cache_scope", default=None
)
_INSTALL_RECEIPT: dict[str, Any] | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def install_generator_nograd_cross_kv_cache(
    *,
    enabled: bool,
    role: str,
    transformer_module: Any | None = None,
    expected_source_sha256: str = EXPECTED_HELIOS_TRANSFORMER_SHA256,
) -> dict[str, Any]:
    """Source-pin and wrap Helios' module-level Q/K/V projection helper."""

    receipt: dict[str, Any] = {
        "schema_version": "distillschedule.helios_cross_kv_cache.v1",
        "requested": bool(enabled),
        "enabled": False,
        "role": role,
        "scope": "GENERATOR_NOGRAD_ONE_ROLLOUT_ONLY",
        "cached_value": "CROSS_ATTENTION_ENCODER_KV_PRE_NORM",
        "cache_device": "CUDA",
        "changes_model_call_count": False,
        "changes_attention_math": False,
        "changes_fsdp_collectives": False,
        "cross_rollout_reuse": False,
        "rollout_scope_count": 0,
        "cross_projection_call_count": 0,
        "self_projection_call_count": 0,
        "query_projection_count": 0,
        "cache_fill_count": 0,
        "cache_reuse_count": 0,
        "cache_miss_count": 0,
        "encoder_signature_mismatch_count": 0,
        "cached_tensor_mutation_count": 0,
        "grad_enabled_call_count": 0,
        "unexpected_cross_attention_count": 0,
        "cache_resident_bytes_sum": 0,
    }
    if role != "generator":
        receipt["disabled_reason"] = "GENERATOR_INSTALL_ONLY"
        return receipt
    if not enabled:
        receipt["disabled_reason"] = "NOT_REQUESTED"
        return receipt

    if transformer_module is None:
        from helios.modules import transformer_helios as transformer_module

    source_path = Path(transformer_module.__file__).resolve()
    observed_sha = _sha256(source_path)
    receipt.update(
        source_path=str(source_path),
        source_sha256=observed_sha,
        expected_source_sha256=expected_source_sha256,
    )
    if observed_sha != expected_source_sha256:
        raise RuntimeError(
            "Helios cross-attention-KV cache source identity changed: "
            f"{observed_sha} != {expected_source_sha256}"
        )
    if not hasattr(transformer_module, "_get_qkv_projections"):
        raise RuntimeError("pinned Helios Q/K/V projection helper is missing")
    if hasattr(
        transformer_module,
        "_distillschedule_cross_kv_original_get_qkv_projections",
    ):
        raise RuntimeError("Helios cross-attention-KV wrapper was already installed")

    original = transformer_module._get_qkv_projections

    def cached_get_qkv_projections(
        attn: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state = _ACTIVE_SCOPE.get()
        if state is None:
            return original(attn, hidden_states, encoder_hidden_states)

        if torch.is_grad_enabled():
            state.grad_enabled_call_count += 1
            raise RuntimeError(
                "Helios cross-attention-KV cache escaped Generator no-grad"
            )
        if not bool(getattr(attn, "is_cross_attention", False)):
            state.self_projection_call_count += 1
            return original(attn, hidden_states, encoder_hidden_states)

        attention_id = id(attn)
        state.cross_projection_call_count += 1
        if attention_id not in state.allowed_attention_ids:
            state.unexpected_cross_attention_count += 1
            raise RuntimeError(
                "Helios cross-attention-KV scope saw an unregistered attention"
            )
        if encoder_hidden_states is None:
            state.cache_miss_count += 1
            raise RuntimeError(
                "Helios cross-attention-KV projection has no encoder conditioning"
            )

        signature = _tensor_signature(encoder_hidden_states)
        cached = state.cache.get(attention_id)
        if cached is None:
            query, key, value = original(attn, hidden_states, encoder_hidden_states)
            state.query_projection_count += 1
            state.cache[attention_id] = (key, value)
            state.encoder_signatures[attention_id] = signature
            state.cache_versions[attention_id] = (int(key._version), int(value._version))
            state.cache_fill_count += 1
            return query, key, value

        if state.encoder_signatures[attention_id] != signature:
            state.encoder_signature_mismatch_count += 1
            raise RuntimeError(
                "Helios cross-attention encoder signature changed within a rollout"
            )
        key, value = cached
        if state.cache_versions[attention_id] != (
            int(key._version),
            int(value._version),
        ):
            state.cached_tensor_mutation_count += 1
            raise RuntimeError("cached Helios cross-attention K/V was mutated")

        # The pinned helper uses ``to_q`` for cross-attention in both fused and
        # unfused projection modes. Only encoder K/V projection is skipped.
        query = attn.to_q(hidden_states)
        state.query_projection_count += 1
        state.cache_reuse_count += 1
        return query, key, value

    transformer_module._distillschedule_cross_kv_original_get_qkv_projections = (
        original
    )
    transformer_module._get_qkv_projections = cached_get_qkv_projections
    transformer_module._distillschedule_cross_kv_receipt = receipt
    receipt["enabled"] = True
    receipt["wrapped_symbol"] = (
        f"{transformer_module.__name__}._get_qkv_projections"
        if hasattr(transformer_module, "__name__")
        else "_get_qkv_projections"
    )

    global _INSTALL_RECEIPT
    _INSTALL_RECEIPT = receipt
    return receipt


def _cross_attentions(model: Any) -> tuple[Any, ...]:
    blocks = tuple(getattr(model, "blocks", ()))
    if len(blocks) != 40:
        raise RuntimeError(
            f"Helios cross-attention-KV cache requires 40 blocks, got {len(blocks)}"
        )
    attentions = tuple(getattr(block, "attn2", None) for block in blocks)
    if any(attention is None for attention in attentions):
        raise RuntimeError("a Helios block has no cross-attention module")
    if any(
        not bool(getattr(attention, "is_cross_attention", False))
        for attention in attentions
    ):
        raise RuntimeError("a Helios attn2 module is not cross-attention")
    if len({id(attention) for attention in attentions}) != len(attentions):
        raise RuntimeError("Helios cross-attention modules are unexpectedly shared")
    return attentions


@contextmanager
def generator_nograd_cross_kv_cache_scope(
    model: Any,
    *,
    enabled: bool,
    expected_model_calls: int,
) -> Iterator[dict[str, Any]]:
    """Cache exact encoder K/V for one B1 Generator no-grad rollout."""

    if not enabled:
        yield {
            "enabled": False,
            "status": "DISABLED",
            "scope": "GENERATOR_NOGRAD_ONE_ROLLOUT_ONLY",
        }
        return
    if _ACTIVE_SCOPE.get() is not None:
        raise RuntimeError("nested Helios cross-attention-KV scopes are forbidden")
    if torch.is_grad_enabled():
        raise RuntimeError("Helios cross-attention-KV scope requires no-grad")
    if expected_model_calls <= 1:
        raise RuntimeError("cross-attention-KV reuse requires at least two model calls")
    if not _INSTALL_RECEIPT or not _INSTALL_RECEIPT.get("enabled"):
        raise RuntimeError("Helios cross-attention-KV wrapper is not installed")

    attentions = _cross_attentions(model)
    state = _ScopeState(
        expected_model_calls=expected_model_calls,
        attentions=attentions,
    )
    token = _ACTIVE_SCOPE.set(state)
    scope_receipt: dict[str, Any] = {
        "enabled": True,
        "status": "RUNNING",
        "scope": "GENERATOR_NOGRAD_ONE_ROLLOUT_ONLY",
        "cached_value": "CROSS_ATTENTION_ENCODER_KV_PRE_NORM",
        "expected_model_calls": int(expected_model_calls),
        "attention_count": len(attentions),
    }
    error: BaseException | None = None
    try:
        yield scope_receipt
    except BaseException as caught:
        error = caught
        raise
    finally:
        _ACTIVE_SCOPE.reset(token)
        cached_attention_count_before_clear = len(state.cache)
        cache_resident_bytes_before_clear = sum(
            tensor.numel() * tensor.element_size()
            for pair in state.cache.values()
            for tensor in pair
        )
        state.cache.clear()
        state.encoder_signatures.clear()
        state.cache_versions.clear()
        cached_attention_count_after_clear = len(state.cache)
        expected_fills = len(attentions)
        expected_reuses = (expected_model_calls - 1) * len(attentions)
        expected_projection_calls = expected_model_calls * len(attentions)
        scope_receipt.update(
            status="PASS" if error is None else "ERROR_CLEANED",
            cross_projection_call_count=state.cross_projection_call_count,
            self_projection_call_count=state.self_projection_call_count,
            query_projection_count=state.query_projection_count,
            cache_fill_count=state.cache_fill_count,
            cache_reuse_count=state.cache_reuse_count,
            cache_miss_count=state.cache_miss_count,
            encoder_signature_mismatch_count=(
                state.encoder_signature_mismatch_count
            ),
            cached_tensor_mutation_count=state.cached_tensor_mutation_count,
            grad_enabled_call_count=state.grad_enabled_call_count,
            unexpected_cross_attention_count=(
                state.unexpected_cross_attention_count
            ),
            cached_attention_count_before_clear=(
                cached_attention_count_before_clear
            ),
            cached_attention_count_after_clear=cached_attention_count_after_clear,
            cache_resident_bytes_before_clear=cache_resident_bytes_before_clear,
            expected_cache_fill_count=expected_fills,
            expected_cache_reuse_count=expected_reuses,
        )

        install_receipt = _INSTALL_RECEIPT
        assert install_receipt is not None
        install_receipt["rollout_scope_count"] += 1
        for key in (
            "cross_projection_call_count",
            "self_projection_call_count",
            "query_projection_count",
            "cache_fill_count",
            "cache_reuse_count",
            "cache_miss_count",
            "encoder_signature_mismatch_count",
            "cached_tensor_mutation_count",
            "grad_enabled_call_count",
            "unexpected_cross_attention_count",
        ):
            install_receipt[key] += int(getattr(state, key))
        install_receipt["cache_resident_bytes_sum"] += int(
            cache_resident_bytes_before_clear
        )

        if error is None:
            observed = {
                "cross_projection_call_count": state.cross_projection_call_count,
                "self_projection_call_count": state.self_projection_call_count,
                "query_projection_count": state.query_projection_count,
                "cache_fill_count": state.cache_fill_count,
                "cache_reuse_count": state.cache_reuse_count,
                "cache_miss_count": state.cache_miss_count,
                "encoder_signature_mismatch_count": (
                    state.encoder_signature_mismatch_count
                ),
                "cached_tensor_mutation_count": state.cached_tensor_mutation_count,
                "grad_enabled_call_count": state.grad_enabled_call_count,
                "unexpected_cross_attention_count": (
                    state.unexpected_cross_attention_count
                ),
                "cached_attention_count_before_clear": (
                    cached_attention_count_before_clear
                ),
                "cached_attention_count_after_clear": (
                    cached_attention_count_after_clear
                ),
            }
            expected = {
                "cross_projection_call_count": expected_projection_calls,
                "self_projection_call_count": expected_projection_calls,
                "query_projection_count": expected_projection_calls,
                "cache_fill_count": expected_fills,
                "cache_reuse_count": expected_reuses,
                "cache_miss_count": 0,
                "encoder_signature_mismatch_count": 0,
                "cached_tensor_mutation_count": 0,
                "grad_enabled_call_count": 0,
                "unexpected_cross_attention_count": 0,
                "cached_attention_count_before_clear": len(attentions),
                "cached_attention_count_after_clear": 0,
            }
            if observed != expected:
                scope_receipt["status"] = "FAIL"
                raise RuntimeError(
                    "Helios cross-attention-KV lifecycle mismatch: "
                    f"{observed} != {expected}"
                )


__all__ = [
    "EXPECTED_HELIOS_TRANSFORMER_SHA256",
    "generator_nograd_cross_kv_cache_scope",
    "install_generator_nograd_cross_kv_cache",
]
