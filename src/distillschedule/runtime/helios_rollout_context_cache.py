"""Exact static-context reuse for one Helios Generator no-grad rollout.

The pinned Helios transformer rebuilds four pieces of immutable context on
every physical DiT call in a rollout:

* three history-latent Conv3d patch paths;
* the complete current/history rotary-position tensor;
* history-prefix packing; and
* the PixArt text projection.

Only the noisy current latent and timestep change between those calls.  This
module keeps the first-call static outputs on CUDA until the enclosing rollout
finishes.  It does not cache the current-latent patch or any DiT block output,
and it does not cross a rollout boundary.

The integration is disabled by default, source-pinned, no-grad-only, and
fail-closed on tensor identity/version/layout or lifecycle drift.
"""
from __future__ import annotations

import contextvars
import hashlib
import inspect
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch


EXPECTED_HELIOS_TRANSFORMER_SHA256 = (
    "5d3ffe37cdea3c04a156106f6246c82f91010a1eaf0dd61529cd74083065a886"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contract(value: torch.Tensor) -> tuple[Any, ...]:
    return (
        tuple(value.shape),
        str(value.dtype),
        str(value.device),
        tuple(value.stride()),
        int(value.storage_offset()),
        str(value.layout),
    )


def _current_structure_key(
    value: torch.Tensor,
    *,
    frames: int,
    height: int,
    width: int,
    tokens: int,
) -> tuple[Any, ...]:
    """Return only the fields on which the current-token RoPE depends.

    The noisy latent value and storage allocation deliberately change between
    denoising calls.  They are therefore not an invariant and must not be used
    as a cache key.  The current patch is still recomputed on every call; only
    the position tensor is keyed by its exact post-patch geometry and device.
    """

    return (
        int(value.shape[0]),
        int(frames),
        int(height),
        int(width),
        int(tokens),
        str(value.dtype),
        str(value.device),
        str(value.layout),
    )


def _identity(value: torch.Tensor) -> tuple[Any, ...]:
    return (
        *_contract(value),
        int(value.untyped_storage().data_ptr()),
        int(value._version),
    )


_STATIC_INPUT_NAMES = (
    "indices_hidden_states",
    "indices_latents_history_short",
    "indices_latents_history_mid",
    "indices_latents_history_long",
    "latents_history_short",
    "latents_history_mid",
    "latents_history_long",
)


class _ScopeState:
    def __init__(
        self,
        *,
        model: Any,
        expected_model_calls: int,
        expected_current_layouts: int,
        source_encoder_identity: tuple[Any, ...],
    ) -> None:
        self.model = model
        self.expected_model_calls = int(expected_model_calls)
        self.expected_current_layouts = int(expected_current_layouts)
        self.source_encoder_identity = source_encoder_identity
        self.model_call_count = 0
        self.process_call_count = 0
        self.process_fill_count = 0
        self.process_reuse_count = 0
        self.text_projection_call_count = 0
        self.text_fill_count = 0
        self.text_reuse_count = 0
        self.current_rope_fill_count = 0
        self.current_rope_reuse_count = 0
        self.grad_enabled_call_count = 0
        self.source_encoder_identity_mismatch_count = 0
        self.static_input_identity_mismatch_count = 0
        self.cached_tensor_mutation_count = 0
        self.unsupported_layout_count = 0
        self.static_input_identities: dict[str, tuple[Any, ...] | None] = {}
        self.history_prefix: torch.Tensor | None = None
        self.history_rope: torch.Tensor | None = None
        self.text_projection: torch.Tensor | None = None
        self.history_version: int | None = None
        self.history_rope_version: int | None = None
        self.text_version: int | None = None
        self.current_ropes: dict[tuple[Any, ...], torch.Tensor] = {}
        self.current_rope_versions: dict[tuple[Any, ...], int] = {}


_ACTIVE_SCOPE: contextvars.ContextVar[_ScopeState | None] = contextvars.ContextVar(
    "distillschedule_helios_rollout_context_cache_scope", default=None
)


def _tensor_or_none_identity(value: Any) -> tuple[Any, ...] | None:
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        raise RuntimeError("Helios static rollout input is not a tensor")
    return _identity(value)


def install_generator_nograd_static_context_cache(
    model: Any,
    *,
    enabled: bool,
    role: str,
    transformer_module: Any | None = None,
    expected_source_sha256: str = EXPECTED_HELIOS_TRANSFORMER_SHA256,
) -> dict[str, Any]:
    """Wrap one pinned Generator instance with an opt-in rollout cache."""

    receipt: dict[str, Any] = {
        "schema_version": "distillschedule.helios_rollout_context_cache.v1",
        "requested": bool(enabled),
        "enabled": False,
        "role": str(role),
        "scope": "GENERATOR_NOGRAD_ONE_ROLLOUT_ONLY",
        "cached_values": [
            "HISTORY_PATCH_PREFIX",
            "FULL_ROPE",
            "STRUCTURAL_SHAPE_LISTS",
            "PROJECTED_TEXT",
        ],
        "current_latent_patch_recomputed": True,
        "changes_model_call_count": False,
        "changes_dit_attention_ffn_calls": False,
        "changes_fsdp_collectives": False,
        "cross_rollout_reuse": False,
        "rollout_scope_count": 0,
        "model_call_count": 0,
        "process_call_count": 0,
        "process_fill_count": 0,
        "process_reuse_count": 0,
        "text_projection_call_count": 0,
        "text_fill_count": 0,
        "text_reuse_count": 0,
        "current_rope_fill_count": 0,
        "current_rope_reuse_count": 0,
        "grad_enabled_call_count": 0,
        "source_encoder_identity_mismatch_count": 0,
        "static_input_identity_mismatch_count": 0,
        "cached_tensor_mutation_count": 0,
        "unsupported_layout_count": 0,
        "cache_resident_bytes_sum": 0,
    }
    model._distillschedule_helios_rollout_context_cache = receipt
    if role != "generator":
        receipt["disabled_reason"] = "GENERATOR_ONLY"
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
            "Helios rollout-context source identity changed: "
            f"{observed_sha} != {expected_source_sha256}"
        )
    required = (
        "process_input_hidden_states",
        "gradient_checkpointing_method",
        "patch_embedding",
        "rope",
        "condition_embedder",
    )
    if any(not hasattr(model, name) for name in required):
        raise RuntimeError("Helios Generator static-context surface changed")
    text_embedder = getattr(model.condition_embedder, "text_embedder", None)
    if text_embedder is None or not hasattr(text_embedder, "forward"):
        raise RuntimeError("Helios text projection surface changed")
    if hasattr(model, "_distillschedule_rollout_context_original_process"):
        raise RuntimeError("Helios rollout-context wrapper was already installed")

    original_process = model.process_input_hidden_states
    process_signature = inspect.signature(original_process)
    original_text_forward = text_embedder.forward

    def cached_process(*args: Any, **kwargs: Any) -> Any:
        state = _ACTIVE_SCOPE.get()
        if state is None:
            return original_process(*args, **kwargs)
        if state.model is not model:
            raise RuntimeError("Helios rollout-context scope reached another model")
        if torch.is_grad_enabled():
            state.grad_enabled_call_count += 1
            raise RuntimeError("Helios rollout-context cache escaped no-grad")

        bound = process_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        latents = values["latents"]
        if not isinstance(latents, torch.Tensor) or latents.ndim != 5:
            state.unsupported_layout_count += 1
            raise RuntimeError("rollout-context cache requires tensor B1 latents")
        if int(latents.shape[0]) != 1:
            state.unsupported_layout_count += 1
            raise RuntimeError("rollout-context cache is registered only for B1")
        static_identities = {
            name: _tensor_or_none_identity(values.get(name))
            for name in _STATIC_INPUT_NAMES
        }
        if any(value is None for value in static_identities.values()):
            state.unsupported_layout_count += 1
            raise RuntimeError("rollout-context cache requires all history inputs")

        state.process_call_count += 1
        if state.history_prefix is None:
            output = original_process(*args, **kwargs)
            if not isinstance(output, tuple) or len(output) != 6:
                state.unsupported_layout_count += 1
                raise RuntimeError("Helios process-input return contract changed")
            hidden, rope, heights, widths, temporals, seqs = output
            if not isinstance(hidden, torch.Tensor) or not isinstance(rope, torch.Tensor):
                state.unsupported_layout_count += 1
                raise RuntimeError("Helios process-input tensors are missing")
            seq_list = tuple(int(value) for value in seqs)
            current_tokens = sum(seq_list)
            history_tokens = int(hidden.shape[1]) - current_tokens
            if history_tokens <= 0:
                state.unsupported_layout_count += 1
                raise RuntimeError("Helios rollout has no cacheable history prefix")
            state.static_input_identities = static_identities
            # Clone only the history slice.  Retaining the original view would
            # pin the first call's current-latent patch for the full rollout.
            state.history_prefix = hidden[:, :history_tokens].clone()
            state.history_rope = rope[:, :history_tokens].clone()
            state.history_version = int(state.history_prefix._version)
            state.history_rope_version = int(state.history_rope._version)
            if not (
                len(heights) == len(widths) == len(temporals) == len(seqs) == 1
            ):
                state.unsupported_layout_count += 1
                raise RuntimeError("B1 rollout current input must have one layout")
            current_key = _current_structure_key(
                hidden[:, history_tokens:],
                frames=int(temporals[0]),
                height=int(heights[0]),
                width=int(widths[0]),
                tokens=int(seqs[0]),
            )
            current_rope = rope[:, history_tokens:].clone()
            state.current_ropes[current_key] = current_rope
            state.current_rope_versions[current_key] = int(current_rope._version)
            state.current_rope_fill_count += 1
            state.process_fill_count += 1
            return output

        if static_identities != state.static_input_identities:
            state.static_input_identity_mismatch_count += 1
            raise RuntimeError("static rollout history/index identity changed")
        assert state.history_rope is not None
        if (
            int(state.history_prefix._version) != state.history_version
            or int(state.history_rope._version) != state.history_rope_version
        ):
            state.cached_tensor_mutation_count += 1
            raise RuntimeError("cached rollout history/RoPE tensor was mutated")

        current = model.gradient_checkpointing_method(model.patch_embedding, latents)
        if not isinstance(current, torch.Tensor) or current.ndim != 5:
            state.unsupported_layout_count += 1
            raise RuntimeError("current-latent patch return contract changed")
        _batch, _channels, frames, height, width = current.shape
        current = current.flatten(2).transpose(1, 2)
        current_key = _current_structure_key(
            current,
            frames=int(frames),
            height=int(height),
            width=int(width),
            tokens=int(current.shape[1]),
        )
        current_rope = state.current_ropes.get(current_key)
        if current_rope is None:
            indices = values["indices_hidden_states"]
            if not isinstance(indices, torch.Tensor):
                state.unsupported_layout_count += 1
                raise RuntimeError("current rollout indices are missing")
            current_rope = model.rope(
                frame_indices=indices,
                height=int(height),
                width=int(width),
                device=current.device,
            )
            current_rope = current_rope.flatten(2).transpose(1, 2).clone()
            state.current_ropes[current_key] = current_rope
            state.current_rope_versions[current_key] = int(current_rope._version)
            state.current_rope_fill_count += 1
        else:
            expected_version = state.current_rope_versions[current_key]
            if int(current_rope._version) != expected_version:
                state.cached_tensor_mutation_count += 1
                raise RuntimeError("cached current-stage RoPE tensor was mutated")
            state.current_rope_reuse_count += 1
        state.process_reuse_count += 1
        return (
            torch.cat([state.history_prefix, current], dim=1),
            torch.cat([state.history_rope, current_rope], dim=1),
            [int(height)],
            [int(width)],
            [int(frames)],
            [int(current.shape[1])],
        )

    def cached_text_forward(encoder_hidden_states: Any, *args: Any, **kwargs: Any) -> Any:
        state = _ACTIVE_SCOPE.get()
        if state is None:
            return original_text_forward(encoder_hidden_states, *args, **kwargs)
        if state.model is not model:
            raise RuntimeError("Helios text cache reached another model")
        if torch.is_grad_enabled():
            state.grad_enabled_call_count += 1
            raise RuntimeError("Helios text cache escaped no-grad")
        if not isinstance(encoder_hidden_states, torch.Tensor):
            state.unsupported_layout_count += 1
            raise RuntimeError("Helios text cache requires a tensor input")
        state.text_projection_call_count += 1
        identity = _identity(encoder_hidden_states)
        if identity != state.source_encoder_identity:
            state.source_encoder_identity_mismatch_count += 1
            raise RuntimeError("source text tensor identity changed within rollout")
        if state.text_projection is None:
            state.text_projection = original_text_forward(
                encoder_hidden_states, *args, **kwargs
            )
            if not isinstance(state.text_projection, torch.Tensor):
                state.unsupported_layout_count += 1
                raise RuntimeError("Helios text projection return contract changed")
            state.text_version = int(state.text_projection._version)
            state.text_fill_count += 1
            return state.text_projection
        if int(state.text_projection._version) != state.text_version:
            state.cached_tensor_mutation_count += 1
            raise RuntimeError("cached Helios text projection was mutated")
        state.text_reuse_count += 1
        return state.text_projection

    model._distillschedule_rollout_context_original_process = original_process
    text_embedder._distillschedule_rollout_context_original_forward = (
        original_text_forward
    )
    model.process_input_hidden_states = cached_process
    text_embedder.forward = cached_text_forward
    receipt["enabled"] = True
    return receipt


@contextmanager
def generator_nograd_static_context_cache_scope(
    model: Any,
    *,
    enabled: bool,
    expected_model_calls: int,
    expected_current_layouts: int,
    source_encoder_hidden_states: torch.Tensor | None,
) -> Iterator[dict[str, Any]]:
    """Keep exact immutable context for one B1 Generator rollout."""

    if not enabled:
        yield {
            "enabled": False,
            "status": "DISABLED",
            "scope": "GENERATOR_NOGRAD_ONE_ROLLOUT_ONLY",
        }
        return
    if _ACTIVE_SCOPE.get() is not None:
        raise RuntimeError("nested Helios rollout-context scopes are forbidden")
    if torch.is_grad_enabled():
        raise RuntimeError("Helios rollout-context scope requires no-grad")
    if int(expected_model_calls) <= 1:
        raise RuntimeError("rollout-context reuse requires at least two model calls")
    if not 1 <= int(expected_current_layouts) <= int(expected_model_calls):
        raise RuntimeError("invalid rollout-context current-layout inventory")
    if not isinstance(source_encoder_hidden_states, torch.Tensor):
        raise RuntimeError("rollout-context scope requires source text tensor")
    install_receipt = getattr(
        model, "_distillschedule_helios_rollout_context_cache", None
    )
    if not isinstance(install_receipt, dict) or not install_receipt.get("enabled"):
        raise RuntimeError("Helios rollout-context wrapper is not installed")

    state = _ScopeState(
        model=model,
        expected_model_calls=int(expected_model_calls),
        expected_current_layouts=int(expected_current_layouts),
        source_encoder_identity=_identity(source_encoder_hidden_states),
    )

    def count_model_call(
        _module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        encoder = kwargs.get("encoder_hidden_states")
        if encoder is None and len(args) >= 3:
            encoder = args[2]
        state.model_call_count += 1
        if not isinstance(encoder, torch.Tensor):
            state.source_encoder_identity_mismatch_count += 1
            raise RuntimeError("Generator model call has no source text tensor")
        if _identity(encoder) != state.source_encoder_identity:
            state.source_encoder_identity_mismatch_count += 1
            raise RuntimeError("Generator source text identity changed")

    hook = model.register_forward_pre_hook(count_model_call, with_kwargs=True)
    token = _ACTIVE_SCOPE.set(state)
    scope_receipt: dict[str, Any] = {
        "enabled": True,
        "status": "RUNNING",
        "scope": "GENERATOR_NOGRAD_ONE_ROLLOUT_ONLY",
        "expected_model_calls": int(expected_model_calls),
        "expected_current_layouts": int(expected_current_layouts),
        "source_identity_contract": (
            "EXACT_STORAGE_POINTER_OFFSET_VERSION_DTYPE_SHAPE_STRIDE_DEVICE"
        ),
        "current_structure_contract": (
            "POST_PATCH_B_T_H_W_TOKENS_DTYPE_DEVICE_LAYOUT"
        ),
    }
    error: BaseException | None = None
    try:
        yield scope_receipt
    except BaseException as caught:
        error = caught
        raise
    finally:
        _ACTIVE_SCOPE.reset(token)
        hook.remove()
        cached_values = (
            state.history_prefix,
            state.history_rope,
            state.text_projection,
            *state.current_ropes.values(),
        )
        cache_bytes = sum(
            int(value.numel() * value.element_size())
            for value in cached_values
            if isinstance(value, torch.Tensor)
        )
        versions_ok = True
        for value, expected in (
            (state.history_prefix, state.history_version),
            (state.history_rope, state.history_rope_version),
            (state.text_projection, state.text_version),
        ):
            if isinstance(value, torch.Tensor) and int(value._version) != expected:
                versions_ok = False
                state.cached_tensor_mutation_count += 1
        for key, value in state.current_ropes.items():
            if int(value._version) != state.current_rope_versions[key]:
                versions_ok = False
                state.cached_tensor_mutation_count += 1
        current_layout_count = len(state.current_ropes)
        tensor_entries_before_clear = 3 + current_layout_count
        state.history_prefix = None
        state.history_rope = None
        state.text_projection = None
        state.current_ropes.clear()
        state.current_rope_versions.clear()
        scope_receipt.update(
            status="PASS" if error is None else "ERROR_CLEANED",
            model_call_count=state.model_call_count,
            process_call_count=state.process_call_count,
            process_fill_count=state.process_fill_count,
            process_reuse_count=state.process_reuse_count,
            text_projection_call_count=state.text_projection_call_count,
            text_fill_count=state.text_fill_count,
            text_reuse_count=state.text_reuse_count,
            current_rope_fill_count=state.current_rope_fill_count,
            current_rope_reuse_count=state.current_rope_reuse_count,
            current_layout_count=current_layout_count,
            grad_enabled_call_count=state.grad_enabled_call_count,
            source_encoder_identity_mismatch_count=(
                state.source_encoder_identity_mismatch_count
            ),
            static_input_identity_mismatch_count=(
                state.static_input_identity_mismatch_count
            ),
            cached_tensor_mutation_count=state.cached_tensor_mutation_count,
            unsupported_layout_count=state.unsupported_layout_count,
            cache_resident_bytes_before_clear=cache_bytes,
            cache_entries_before_clear=tensor_entries_before_clear,
            cache_entries_after_clear=0,
        )

        for key in (
            "model_call_count",
            "process_call_count",
            "process_fill_count",
            "process_reuse_count",
            "text_projection_call_count",
            "text_fill_count",
            "text_reuse_count",
            "current_rope_fill_count",
            "current_rope_reuse_count",
            "grad_enabled_call_count",
            "source_encoder_identity_mismatch_count",
            "static_input_identity_mismatch_count",
            "cached_tensor_mutation_count",
            "unsupported_layout_count",
        ):
            install_receipt[key] += int(getattr(state, key))
        install_receipt["rollout_scope_count"] += 1
        install_receipt["cache_resident_bytes_sum"] += cache_bytes

        if error is None:
            expected = {
                "model_call_count": int(expected_model_calls),
                "process_call_count": int(expected_model_calls),
                "process_fill_count": 1,
                "process_reuse_count": int(expected_model_calls) - 1,
                "text_projection_call_count": int(expected_model_calls),
                "text_fill_count": 1,
                "text_reuse_count": int(expected_model_calls) - 1,
                "current_rope_fill_count": int(expected_current_layouts),
                "current_rope_reuse_count": (
                    int(expected_model_calls) - int(expected_current_layouts)
                ),
                "grad_enabled_call_count": 0,
                "source_encoder_identity_mismatch_count": 0,
                "static_input_identity_mismatch_count": 0,
                "cached_tensor_mutation_count": 0,
                "unsupported_layout_count": 0,
            }
            observed = {key: scope_receipt[key] for key in expected}
            if (
                observed != expected
                or current_layout_count != int(expected_current_layouts)
                or not versions_ok
                or cache_bytes <= 0
            ):
                scope_receipt["status"] = "FAIL"
                raise RuntimeError(
                    "Helios rollout-context lifecycle mismatch: "
                    f"observed={observed}, expected={expected}, "
                    f"current_layout_count={current_layout_count}, "
                    f"expected_current_layouts={expected_current_layouts}, "
                    f"versions_ok={versions_ok}, cache_bytes={cache_bytes}"
                )


__all__ = [
    "EXPECTED_HELIOS_TRANSFORMER_SHA256",
    "generator_nograd_static_context_cache_scope",
    "install_generator_nograd_static_context_cache",
]
