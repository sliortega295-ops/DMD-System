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

import types
from typing import Any


SUPPORTED_SEGMENTS = (2, 4, 5, 8)
PINNED_MODEL_NAME = "MiniMaxH3Transformer3DModel"


def _validate_segment(segment: int) -> int:
    segment = int(segment)
    if segment not in SUPPORTED_SEGMENTS:
        raise ValueError(
            f"segmented MiniMax checkpoint requires one of {SUPPORTED_SEGMENTS}, got {segment}"
        )
    return segment


def _install_transformer_forward(transformer: Any, segment: int) -> None:
    """Install a segmented forward on one already-FSDP-wrapped transformer."""
    if getattr(transformer, "_distillschedule_segmented_checkpoint", None) == segment:
        return
    if getattr(transformer.__class__, "__name__", "") != PINNED_MODEL_NAME:
        raise RuntimeError(
            "segmented MiniMax checkpoint only supports the pinned "
            f"{PINNED_MODEL_NAME}, got {transformer.__class__.__name__}"
        )
    blocks = getattr(transformer, "transformer_blocks", None)
    if blocks is None or len(blocks) != 50:
        raise RuntimeError(
            "segmented MiniMax checkpoint requires exactly 50 transformer blocks; "
            f"got {0 if blocks is None else len(blocks)}"
        )

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
        for start in range(0, len(self.transformer_blocks), segment):
            group = tuple(self.transformer_blocks[start : start + segment])

            def run_group(
                value: torch.Tensor,
                group: tuple[Any, ...] = group,
            ) -> torch.Tensor:
                for block in group:
                    value = block(value, temb, adaln_indices, rotary_emb)
                return value

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(run_group, hidden_states)
            else:
                hidden_states = run_group(hidden_states)

        hidden_states = self.norm_out(hidden_states, temb, timestep_indices).to(get_parameter_dtype(self.proj_out))
        video_output = self.proj_out(hidden_states).index_select(1, video_indices)
        audio_output = self.audio_proj_out(hidden_states).index_select(1, audio_indices)
        if not return_dict:
            return (video_output, audio_output)
        return MiniMaxH3TransformerOutput(sample=video_output, audio_sample=audio_output)

    transformer.forward = types.MethodType(segmented_forward, transformer)
    transformer._distillschedule_segmented_checkpoint = segment
    transformer._distillschedule_segmented_checkpoint_block_count = len(blocks)


def install_segmented_checkpoint(model: Any, segment: int) -> dict[str, Any]:
    """Install the policy on a LightX2V model after FSDP2 setup."""
    segment = _validate_segment(segment)
    transformer = getattr(model, "transformer", None)
    if transformer is None and hasattr(model, "denoiser_module"):
        transformer = model.denoiser_module()
    if transformer is None:
        raise RuntimeError("LightX2V model has no transformer/denoiser module")
    _install_transformer_forward(transformer, segment)
    return {
        "enabled": True,
        "segment": segment,
        "block_count": int(transformer._distillschedule_segmented_checkpoint_block_count),
        "fsdp_granularity": "one_fully_shard_per_transformer_block",
        "activation_offload": False,
    }

