"""Opt-in segmented activation checkpointing for MiniMax-H3.

This is deliberately separate from FSDP2 placement.  Every transformer block
remains an individual ``fully_shard`` module; only the autograd checkpoint
boundary is moved from one block to a consecutive group of blocks.  The helper
is installed after the LightX2V setup has applied FSDP2 so that it cannot alter
the communication topology.

The implementation is guarded to the pinned MiniMax-H3 forward contract.  It
duplicates the upstream forward *without changing its math* and replaces only
the block-loop checkpoint granularity.  A source/model mismatch fails closed
instead of silently running a different workload.
"""
from __future__ import annotations

import hashlib
import inspect
import subprocess
import textwrap
import types
from pathlib import Path
from typing import Any


SUPPORTED_SEGMENTS = (2, 4, 5, 8)
DEBUG_REFERENCE_SEGMENT = 1
PINNED_MODEL_NAME = "MiniMaxH3Transformer3DModel"
# These identities come from the prepared MiniMax-H3 environment used by the
# real canaries.  The source hash is over the class's ``forward`` method after
# ``textwrap.dedent``.  A model with the same class name and block count but a
# different forward implementation must not silently use this copied method.
PINNED_DIFFUSERS_VERSION = "0.40.0.dev0"
PINNED_DIFFUSERS_SOURCE_REVISION = "9284607295a09f759aadd65ed08f48b35feea6d9"
PINNED_MINIMAX_FORWARD_SHA256 = "24fce9f0721ede96867a8b52ceb0d80431b9c2f3d576844d2b4481a3a40c0c6b"


def _validate_segment(segment: int, *, allow_debug_reference: bool = False) -> int:
    segment = int(segment)
    allowed = SUPPORTED_SEGMENTS + ((DEBUG_REFERENCE_SEGMENT,) if allow_debug_reference else ())
    if segment not in allowed:
        raise ValueError(
            f"segmented MiniMax checkpoint requires one of {allowed}, got {segment}"
        )
    return segment


def _validate_pinned_diffusers_contract() -> dict[str, str]:
    """Fail closed if the installed MiniMax forward is not the pinned source."""
    import diffusers
    from diffusers import MiniMaxH3Transformer3DModel

    actual_version = str(getattr(diffusers, "__version__", ""))
    if actual_version != PINNED_DIFFUSERS_VERSION:
        raise RuntimeError(
            "segmented MiniMax checkpoint requires Diffusers "
            f"{PINNED_DIFFUSERS_VERSION}, found {actual_version or '<unknown>'}"
        )
    actual_revision = _diffusers_source_revision(diffusers)
    if actual_revision != PINNED_DIFFUSERS_SOURCE_REVISION:
        raise RuntimeError(
            "Diffusers source revision mismatch: "
            f"{actual_revision} != {PINNED_DIFFUSERS_SOURCE_REVISION}"
        )
    try:
        source = textwrap.dedent(inspect.getsource(MiniMaxH3Transformer3DModel.forward))
    except (OSError, TypeError) as exc:
        raise RuntimeError("cannot inspect pinned MiniMaxH3Transformer3DModel.forward") from exc
    actual_forward_sha = hashlib.sha256(source.encode()).hexdigest()
    if actual_forward_sha != PINNED_MINIMAX_FORWARD_SHA256:
        raise RuntimeError(
            "MiniMaxH3Transformer3DModel.forward source mismatch: "
            f"{actual_forward_sha} != {PINNED_MINIMAX_FORWARD_SHA256}"
        )
    return {
        "diffusers_version": actual_version,
        "diffusers_source_revision": actual_revision,
        "minimax_forward_sha256": actual_forward_sha,
    }


def _diffusers_source_revision(diffusers: Any) -> str:
    """Resolve the source revision from the imported source checkout."""
    module_path = Path(getattr(diffusers, "__file__", "")).resolve()
    for parent in (module_path.parent, *module_path.parents):
        git_marker = parent / ".git"
        if not git_marker.exists():
            continue
        result = subprocess.run(
            ["git", "-C", str(parent), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    raise RuntimeError(
        "cannot resolve the Diffusers source revision; the pinned editable "
        "Diffusers source checkout is required"
    )


def _debug_saved_tensor_stats(transformer: Any) -> dict[str, int | None]:
    """Read optional scalar saved-tensor counters without retaining tensors."""
    provider = getattr(transformer, "_distillschedule_saved_tensor_stats", None)
    if callable(provider):
        provider = provider()
    if not isinstance(provider, dict):
        return {
            "saved_tensor_count": None,
            "saved_tensor_bytes": None,
            "saved_tensor_pack_count": None,
            "saved_tensor_pack_bytes": None,
            "saved_tensor_unpack_count": None,
            "saved_tensor_unpack_bytes": None,
        }
    return {
        "saved_tensor_count": provider.get("saved_tensor_count", provider.get("live_count")),
        "saved_tensor_bytes": provider.get("saved_tensor_bytes", provider.get("live_bytes")),
        "saved_tensor_pack_count": provider.get("pack_count"),
        "saved_tensor_pack_bytes": provider.get("pack_bytes"),
        "saved_tensor_unpack_count": provider.get("unpack_count"),
        "saved_tensor_unpack_bytes": provider.get("unpack_bytes"),
    }


def _debug_event(transformer: Any, event: dict[str, Any]) -> None:
    """Invoke an opt-in progress hook without changing the normal path."""
    callback = getattr(transformer, "_distillschedule_segment_debug_hook", None)
    if callback is not None:
        callback(event)


def _is_checkpoint_early_stop(error: BaseException) -> bool:
    """PyTorch non-reentrant checkpoint uses this private control exception."""
    return type(error).__name__ == "_StopRecomputationError"


def _install_transformer_forward(
    transformer: Any,
    segment: int,
    *,
    validate_contract: bool = True,
    debug: bool = False,
) -> None:
    """Install a segmented forward on one already-FSDP-wrapped transformer."""
    if getattr(transformer, "_distillschedule_segmented_checkpoint", None) == segment:
        return
    class_names = tuple(cls.__name__ for cls in transformer.__class__.__mro__)
    # FSDP2 changes the runtime class to ``FSDPMiniMaxH3Transformer3DModel``
    # while retaining the pinned MiniMax class in its MRO.  That wrapper is
    # exactly the desired post-fully_shard insertion point.
    if PINNED_MODEL_NAME not in class_names:
        raise RuntimeError(
            "segmented MiniMax checkpoint only supports the pinned "
            f"{PINNED_MODEL_NAME}, got {transformer.__class__.__name__} (MRO={class_names})"
        )
    blocks = getattr(transformer, "transformer_blocks", None)
    if blocks is None or len(blocks) != 50:
        raise RuntimeError(
            "segmented MiniMax checkpoint requires exactly 50 transformer blocks; "
            f"got {0 if blocks is None else len(blocks)}"
        )

    if validate_contract:
        _validate_pinned_diffusers_contract()

    import torch
    from diffusers.models.modeling_utils import get_parameter_dtype
    from diffusers.models.transformers.transformer_minimax_h3 import (
        MINIMAX_H3_MODALITY_NUM,
        MiniMaxH3TransformerOutput,
    )
    from diffusers.utils import apply_lora_scale

    @apply_lora_scale("attention_kwargs")
    def segmented_forward(
        self: Any,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        timestep_indices: torch.Tensor,
        token_tags: torch.Tensor,
        position_ids: torch.Tensor,
        video_indices: torch.Tensor,
        audio_indices: torch.Tensor,
        text_indices: torch.Tensor,
        attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
    ) -> Any:
        # Keep the upstream validation and input projection semantics verbatim.
        if position_ids.ndim != 2 or position_ids.shape[-1] != 3:
            raise ValueError(f"`position_ids` must be a `(seq_len, 3)` tensor, got {list(position_ids.shape)}.")
        sequence_length = position_ids.shape[0]
        if token_tags.shape != (sequence_length,) or timestep_indices.shape != (sequence_length,):
            raise ValueError(
                "`token_tags` and `timestep_indices` must both be `(seq_len,)` tensors matching `position_ids`, got "
                f"{list(token_tags.shape)} and {list(timestep_indices.shape)} for seq_len={sequence_length}."
            )

        rotary_emb = self.rope(position_ids)
        video_embeds = self.proj_in(hidden_states.to(get_parameter_dtype(self.proj_in)))
        audio_embeds = self.audio_proj_in(audio_hidden_states.to(get_parameter_dtype(self.audio_proj_in)))
        text_embeds = self.context_embedder(encoder_hidden_states.to(get_parameter_dtype(self.context_embedder)))
        text_embeds = self.token_refiner(text_embeds)

        hidden_states = text_embeds.new_zeros((text_embeds.shape[0], sequence_length, text_embeds.shape[-1]))
        hidden_states = hidden_states.index_copy(1, text_indices, text_embeds)
        hidden_states = hidden_states.index_copy(1, video_indices, video_embeds.to(text_embeds.dtype))
        hidden_states = hidden_states.index_copy(1, audio_indices, audio_embeds.to(text_embeds.dtype))

        temb = self.time_proj(timestep)
        temb = self.time_embedder(temb.to(get_parameter_dtype(self.time_embedder)))
        adaln_indices = timestep_indices * MINIMAX_H3_MODALITY_NUM + token_tags

        # FSDP2 still sees and executes each original block.  The only changed
        # object is the checkpoint function around a consecutive group.
        for segment_index, start in enumerate(range(0, len(self.transformer_blocks), segment)):
            end = min(start + segment, len(self.transformer_blocks))
            group = tuple(self.transformer_blocks[start : start + segment])

            def report(
                event: str,
                *,
                block_index: int | None = None,
                error: BaseException | None = None,
                _segment_index: int = segment_index,
                _start: int = start,
                _end: int = end,
            ) -> None:
                if not debug:
                    return
                allocated = reserved = None
                if hidden_states.is_cuda:
                    allocated = int(torch.cuda.memory_allocated(hidden_states.device))
                    reserved = int(torch.cuda.memory_reserved(hidden_states.device))
                _debug_event(
                    self,
                    {
                        "event": event,
                        "segment": _segment_index,
                        "segment_start": _start,
                        "segment_end": _end,
                        "block": block_index,
                        "allocated_bytes": allocated,
                        "reserved_bytes": reserved,
                        **_debug_saved_tensor_stats(self),
                        "error_type": type(error).__name__ if error is not None else None,
                        "error": str(error) if error is not None else None,
                    },
                )

            def run_group(
                value: torch.Tensor,
                group: tuple[Any, ...] = group,
                _start: int = start,
                _report: Any = report,
            ) -> torch.Tensor:
                for offset, block in enumerate(group):
                    _report("block_enter", block_index=_start + offset)
                    try:
                        value = block(value, temb, adaln_indices, rotary_emb)
                    except BaseException as exc:
                        if not _is_checkpoint_early_stop(exc):
                            _report("block_error", block_index=_start + offset, error=exc)
                        raise
                    _report("block_exit", block_index=_start + offset)
                return value

            report("segment_enter")
            try:
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    hidden_states = self._gradient_checkpointing_func(run_group, hidden_states)
                else:
                    hidden_states = run_group(hidden_states)
            except BaseException as exc:
                report("segment_error", error=exc)
                raise
            report("segment_exit")

        hidden_states = self.norm_out(hidden_states, temb, timestep_indices).to(get_parameter_dtype(self.proj_out))
        video_output = self.proj_out(hidden_states).index_select(1, video_indices)
        audio_output = self.audio_proj_out(hidden_states).index_select(1, audio_indices)
        if not return_dict:
            return (video_output, audio_output)
        return MiniMaxH3TransformerOutput(sample=video_output, audio_sample=audio_output)

    transformer.forward = types.MethodType(segmented_forward, transformer)
    transformer._distillschedule_segmented_checkpoint = segment
    transformer._distillschedule_segmented_checkpoint_block_count = len(blocks)


def install_segmented_checkpoint(model: Any, segment: int, *, debug: bool = False) -> dict[str, Any]:
    """Install the policy on a LightX2V model after FSDP2 setup."""
    segment = _validate_segment(segment)
    transformer = getattr(model, "transformer", None)
    if transformer is None and hasattr(model, "denoiser_module"):
        transformer = model.denoiser_module()
    if transformer is None:
        raise RuntimeError("LightX2V model has no transformer/denoiser module")
    contract = _validate_pinned_diffusers_contract()
    _install_transformer_forward(transformer, segment, validate_contract=False, debug=debug)
    if debug:
        transformer._distillschedule_segment_debug_enabled = True
    return {
        "enabled": True,
        "segment": segment,
        "block_count": int(transformer._distillschedule_segmented_checkpoint_block_count),
        "fsdp_granularity": "one_fully_shard_per_transformer_block",
        "activation_offload": False,
        "diffusers_version": contract["diffusers_version"],
        "diffusers_source_revision": contract["diffusers_source_revision"],
        "minimax_forward_sha256": contract["minimax_forward_sha256"],
        "debug_progress": bool(debug),
    }
