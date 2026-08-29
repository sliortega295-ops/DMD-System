"""Exact Helios history-KV reuse for one Generator no-grad rollout.

The pinned Helios attention processor already implements an exact cache for the
history-only self-attention prefix.  This module does not change attention math
or skip a Transformer call.  It only controls that existing cache at the
DistillSchedule rollout boundary and records fail-closed lifecycle evidence.
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


class _ScopeState:
    def __init__(self, *, expected_model_calls: int, processor_count: int) -> None:
        self.expected_model_calls = int(expected_model_calls)
        self.processor_count = int(processor_count)
        self.enabled_processor_calls = 0
        self.first_step_calls = 0
        self.cache_fill_count = 0
        self.cache_reuse_count = 0
        self.cache_miss_count = 0
        self.navit_incompatible_count = 0
        self.grad_enabled_call_count = 0
        self.cached_tensor_mutation_count = 0
        self.cache_versions: dict[int, tuple[int, int, int]] = {}


_ACTIVE_SCOPE: contextvars.ContextVar[_ScopeState | None] = contextvars.ContextVar(
    "distillschedule_helios_history_kv_cache_scope", default=None
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _processor_argument(
    args: tuple[Any, ...], kwargs: dict[str, Any], name: str, index: int, default: Any
) -> Any:
    if name in kwargs:
        return kwargs[name]
    return args[index] if len(args) > index else default


def install_generator_nograd_history_kv_cache(
    *,
    enabled: bool,
    role: str,
    transformer_module: Any | None = None,
    expected_source_sha256: str = EXPECTED_HELIOS_TRANSFORMER_SHA256,
) -> dict[str, Any]:
    """Install a source-pinned audit wrapper around ``HeliosAttnProcessor``.

    The wrapper is inert unless :func:`generator_nograd_history_kv_cache_scope`
    is active and the concrete attention processor has been cache-enabled.
    """

    receipt: dict[str, Any] = {
        "schema_version": "distillschedule.helios_history_kv_cache.v1",
        "requested": bool(enabled),
        "enabled": False,
        "role": role,
        "scope": "GENERATOR_NOGRAD_ONE_ROLLOUT_ONLY",
        "cached_value": "RESTRICT_SELF_ATTN_HISTORY_KV_AND_HISTORY_OUTPUT",
        "changes_model_call_count": False,
        "changes_attention_math": False,
        "changes_fsdp_collectives": False,
        "cross_rollout_reuse": False,
        "rollout_scope_count": 0,
        "enabled_processor_calls": 0,
        "first_step_calls": 0,
        "cache_fill_count": 0,
        "cache_reuse_count": 0,
        "cache_miss_count": 0,
        "navit_incompatible_count": 0,
        "grad_enabled_call_count": 0,
        "cached_tensor_mutation_count": 0,
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
            "Helios history-KV cache source identity changed: "
            f"{observed_sha} != {expected_source_sha256}"
        )

    processor_class = transformer_module.HeliosAttnProcessor
    if hasattr(processor_class, "_distillschedule_history_kv_original_call"):
        raise RuntimeError("Helios history-KV audit was already installed")
    original_call = processor_class.__call__

    def audited_call(self: Any, *args: Any, **kwargs: Any) -> Any:
        state = _ACTIVE_SCOPE.get()
        if state is None or not bool(getattr(self, "cache_enabled", False)):
            return original_call(self, *args, **kwargs)
        if torch.is_grad_enabled():
            state.grad_enabled_call_count += 1
            raise RuntimeError("Helios history-KV cache escaped no-grad")

        attention = _processor_argument(args, kwargs, "attn", 0, None)
        first_step = bool(
            _processor_argument(
                args, kwargs, "is_first_denoising_step", 8, False
            )
        )
        enable_navit = bool(
            _processor_argument(args, kwargs, "enable_navit", 7, False)
        )
        if attention is None or not bool(
            getattr(attention, "restrict_self_attn", False)
        ):
            raise RuntimeError(
                "cache-enabled Helios processor escaped restrict-self-attention"
            )
        if enable_navit:
            state.navit_incompatible_count += 1
            raise RuntimeError("Helios history-KV cache is incompatible with NAViT")

        had_cache = getattr(self, "kv_cache", None) is not None
        if had_cache:
            cached = self.kv_cache
            if not isinstance(cached, dict) or any(
                not isinstance(cached.get(key), torch.Tensor)
                for key in ("key_history", "value_history", "history_hidden_states")
            ):
                raise RuntimeError("Helios history-KV cache payload changed")
            observed_versions = tuple(
                int(cached[key]._version)
                for key in ("key_history", "value_history", "history_hidden_states")
            )
            if state.cache_versions.get(id(self)) != observed_versions:
                state.cached_tensor_mutation_count += 1
                raise RuntimeError("cached Helios history K/V was mutated")
        state.enabled_processor_calls += 1
        if first_step:
            state.first_step_calls += 1
        elif had_cache:
            state.cache_reuse_count += 1
        else:
            state.cache_miss_count += 1

        result = original_call(self, *args, **kwargs)
        has_cache = getattr(self, "kv_cache", None) is not None
        if first_step:
            if had_cache:
                raise RuntimeError("history-KV cache was not empty on first step")
            if not has_cache:
                state.cache_miss_count += 1
                raise RuntimeError("history-KV cache did not populate on first step")
            cached = self.kv_cache
            if not isinstance(cached, dict) or any(
                not isinstance(cached.get(key), torch.Tensor)
                for key in ("key_history", "value_history", "history_hidden_states")
            ):
                raise RuntimeError("Helios history-KV cache payload is malformed")
            state.cache_versions[id(self)] = tuple(
                int(cached[key]._version)
                for key in ("key_history", "value_history", "history_hidden_states")
            )
            state.cache_fill_count += 1
        elif had_cache:
            cached = self.kv_cache
            if not isinstance(cached, dict) or any(
                not isinstance(cached.get(key), torch.Tensor)
                for key in ("key_history", "value_history", "history_hidden_states")
            ):
                raise RuntimeError("Helios history-KV cache payload changed")
            observed_versions = tuple(
                int(cached[key]._version)
                for key in ("key_history", "value_history", "history_hidden_states")
            )
            if state.cache_versions.get(id(self)) != observed_versions:
                state.cached_tensor_mutation_count += 1
                raise RuntimeError("cached Helios history K/V was mutated")
        return result

    processor_class._distillschedule_history_kv_original_call = original_call
    processor_class._distillschedule_history_kv_receipt = receipt
    processor_class.__call__ = audited_call
    receipt["enabled"] = True
    receipt["processor_class"] = (
        f"{processor_class.__module__}.{processor_class.__qualname__}"
    )
    return receipt


def _cache_processors(model: Any) -> tuple[Any, ...]:
    blocks = tuple(getattr(model, "blocks", ()))
    if len(blocks) != 40:
        raise RuntimeError(f"Helios history-KV cache requires 40 blocks, got {len(blocks)}")
    processors = tuple(getattr(block.attn1, "processor", None) for block in blocks)
    if any(processor is None for processor in processors):
        raise RuntimeError("a Helios block has no self-attention processor")
    if len({id(processor) for processor in processors}) != len(processors):
        raise RuntimeError("Helios history-KV processors are unexpectedly shared")
    return processors


@contextmanager
def generator_nograd_history_kv_cache_scope(
    model: Any,
    *,
    enabled: bool,
    expected_model_calls: int,
) -> Iterator[dict[str, Any]]:
    """Enable, audit, and clear exact history KV state for one B1 rollout."""

    if not enabled:
        yield {
            "enabled": False,
            "status": "DISABLED",
            "scope": "GENERATOR_NOGRAD_ONE_ROLLOUT_ONLY",
        }
        return
    if _ACTIVE_SCOPE.get() is not None:
        raise RuntimeError("nested Helios history-KV scopes are forbidden")
    if torch.is_grad_enabled():
        raise RuntimeError("Helios history-KV scope requires no-grad")
    if expected_model_calls <= 1:
        raise RuntimeError("history-KV reuse requires at least two model calls")

    processors = _cache_processors(model)
    processor_class = type(processors[0])
    install_receipt = getattr(
        processor_class, "_distillschedule_history_kv_receipt", None
    )
    if not install_receipt or not install_receipt.get("enabled"):
        raise RuntimeError("Helios history-KV audit is not installed")
    if any(type(processor) is not processor_class for processor in processors):
        raise RuntimeError("Helios history-KV processor classes differ across blocks")
    if any(
        bool(getattr(processor, "cache_enabled", False))
        or getattr(processor, "kv_cache", None) is not None
        for processor in processors
    ):
        raise RuntimeError("history-KV state leaked into a new rollout")

    for processor in processors:
        processor.enable_cache()
    state = _ScopeState(
        expected_model_calls=expected_model_calls,
        processor_count=len(processors),
    )
    token = _ACTIVE_SCOPE.set(state)
    scope_receipt: dict[str, Any] = {
        "enabled": True,
        "status": "RUNNING",
        "scope": "GENERATOR_NOGRAD_ONE_ROLLOUT_ONLY",
        "cached_value": "RESTRICT_SELF_ATTN_HISTORY_KV_AND_HISTORY_OUTPUT",
        "expected_model_calls": int(expected_model_calls),
        "processor_count": len(processors),
    }
    error: BaseException | None = None
    try:
        yield scope_receipt
    except BaseException as caught:
        error = caught
        raise
    finally:
        _ACTIVE_SCOPE.reset(token)
        populated_before_clear = sum(
            getattr(processor, "kv_cache", None) is not None
            for processor in processors
        )
        cache_resident_bytes_before_clear = sum(
            tensor.numel() * tensor.element_size()
            for processor in processors
            for tensor in (
                tuple(getattr(processor, "kv_cache", {}).values())
                if isinstance(getattr(processor, "kv_cache", None), dict)
                else ()
            )
            if isinstance(tensor, torch.Tensor)
        )
        for processor in processors:
            processor.disable_cache()
        enabled_after = sum(
            bool(getattr(processor, "cache_enabled", False))
            for processor in processors
        )
        populated_after = sum(
            getattr(processor, "kv_cache", None) is not None
            for processor in processors
        )
        expected_fills = len(processors)
        expected_reuses = (expected_model_calls - 1) * len(processors)
        scope_receipt.update(
            status="PASS" if error is None else "ERROR_CLEANED",
            enabled_processor_calls=state.enabled_processor_calls,
            first_step_calls=state.first_step_calls,
            cache_fill_count=state.cache_fill_count,
            cache_reuse_count=state.cache_reuse_count,
            cache_miss_count=state.cache_miss_count,
            navit_incompatible_count=state.navit_incompatible_count,
            grad_enabled_call_count=state.grad_enabled_call_count,
            cached_tensor_mutation_count=state.cached_tensor_mutation_count,
            populated_before_clear=populated_before_clear,
            cache_enabled_after_scope=enabled_after,
            populated_after_scope=populated_after,
            cache_resident_bytes_before_clear=cache_resident_bytes_before_clear,
            expected_cache_fill_count=expected_fills,
            expected_cache_reuse_count=expected_reuses,
        )
        install_receipt["rollout_scope_count"] += 1
        for key in (
            "enabled_processor_calls",
            "first_step_calls",
            "cache_fill_count",
            "cache_reuse_count",
            "cache_miss_count",
            "navit_incompatible_count",
            "grad_enabled_call_count",
            "cached_tensor_mutation_count",
        ):
            install_receipt[key] += int(getattr(state, key))
        install_receipt["cache_resident_bytes_sum"] += int(
            cache_resident_bytes_before_clear
        )

        if error is None:
            observed = {
                "enabled_processor_calls": state.enabled_processor_calls,
                "first_step_calls": state.first_step_calls,
                "cache_fill_count": state.cache_fill_count,
                "cache_reuse_count": state.cache_reuse_count,
                "cache_miss_count": state.cache_miss_count,
                "navit_incompatible_count": state.navit_incompatible_count,
                "grad_enabled_call_count": state.grad_enabled_call_count,
                "cached_tensor_mutation_count": state.cached_tensor_mutation_count,
                "populated_before_clear": populated_before_clear,
                "cache_enabled_after_scope": enabled_after,
                "populated_after_scope": populated_after,
            }
            expected = {
                "enabled_processor_calls": expected_model_calls * len(processors),
                "first_step_calls": len(processors),
                "cache_fill_count": expected_fills,
                "cache_reuse_count": expected_reuses,
                "cache_miss_count": 0,
                "navit_incompatible_count": 0,
                "grad_enabled_call_count": 0,
                "cached_tensor_mutation_count": 0,
                "populated_before_clear": len(processors),
                "cache_enabled_after_scope": 0,
                "populated_after_scope": 0,
            }
            if observed != expected:
                scope_receipt["status"] = "FAIL"
                raise RuntimeError(
                    f"Helios history-KV lifecycle mismatch: {observed} != {expected}"
                )


__all__ = [
    "EXPECTED_HELIOS_TRANSFORMER_SHA256",
    "generator_nograd_history_kv_cache_scope",
    "install_generator_nograd_history_kv_cache",
]
