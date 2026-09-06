"""Opt-in exact no-grad pointwise fusion for pinned MiniMax-H3 blocks."""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import types
from typing import Any

import torch


PINNED_BLOCK_FORWARD_SHA256 = "ca6d9ca44871d0ed10fceaa4eecd184a7ac907cbd3e3b8fd9f761b91598a944d"
EXPECTED_BLOCK_COUNT = 50
EXPECTED_NOGRAD_BLOCK_CALLS_PER_CYCLE = 25 * EXPECTED_BLOCK_COUNT
EXPECTED_NOGRAD_RMSNORM_MODULATION_CALLS_PER_CYCLE = (
    2 * EXPECTED_NOGRAD_BLOCK_CALLS_PER_CYCLE
)
EXPECTED_HIDDEN_SIZE = 5376
EXPECTED_RMSNORM_EPS = 1e-5


@dataclasses.dataclass
class FusedPointwiseStats:
    fused_nograd_block_calls: int = 0
    fused_grad_block_calls: int = 0
    reference_grad_block_calls: int = 0
    fused_modulation_calls: int = 0
    fused_residual_calls: int = 0
    fused_grad_modulation_calls: int = 0
    fused_grad_residual_calls: int = 0
    fused_grad_modulation_backward_calls: int = 0
    fused_grad_residual_backward_calls: int = 0
    fused_nograd_rmsnorm_modulation_calls: int = 0


@dataclasses.dataclass
class FusedPointwiseRegistration:
    enabled: bool
    grad_enabled: bool
    rmsnorm_modulate_enabled: bool
    source_sha256: str | None
    block_count: int
    stats: FusedPointwiseStats

    def snapshot(self) -> dict[str, int]:
        return dataclasses.asdict(self.stats)

    def receipt(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "grad_enabled": self.grad_enabled,
            "rmsnorm_modulate_enabled": self.rmsnorm_modulate_enabled,
            "source_sha256": self.source_sha256,
            "block_count": self.block_count,
            "stats": self.snapshot(),
        }


def _source_sha256(function) -> str:
    return hashlib.sha256(inspect.getsource(function).encode()).hexdigest()


def _patched_forward(
    self,
    hidden_states: torch.Tensor,
    temb: torch.Tensor,
    adaln_indices: torch.Tensor,
    rotary_emb,
    attention_mask=None,
):
    stats: FusedPointwiseStats = self._h3_fused_pointwise_stats
    grad_enabled = torch.is_grad_enabled()
    if grad_enabled and not self._h3_fused_pointwise_grad_enabled:
        stats.reference_grad_block_calls += 1
        return self._h3_original_forward(
            hidden_states,
            temb,
            adaln_indices,
            rotary_emb,
            attention_mask,
        )
    if grad_enabled:
        # Non-reentrant checkpoint replay may terminate a recomputed block via
        # PyTorch's internal early-stop exception before normal return.  Count
        # entry, not return, exactly like the Grid replay scope audit.
        stats.fused_grad_block_calls += 1

    # Lazy import keeps the baseline/import-only path independent of Triton.
    from .triton_fused_pointwise import (
        fused_modulate,
        fused_residual,
        fused_rmsnorm_modulate,
    )

    if grad_enabled:
        modulate = lambda value, scale, shift, indices: _FusedModulateAutograd.apply(
            value, scale, shift, indices, stats
        )
        residual_update = lambda residual, gate, branch, indices: _FusedResidualAutograd.apply(
            residual, gate, branch, indices, stats
        )
    else:
        modulate = fused_modulate
        residual_update = fused_residual

    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(temb)

    residual = hidden_states
    if not grad_enabled and self._h3_fused_rmsnorm_modulate_enabled:
        norm_hidden_states = fused_rmsnorm_modulate(
            hidden_states,
            self.norm1.weight,
            scale_msa,
            shift_msa,
            adaln_indices,
            eps=self.norm1.eps,
        )
        stats.fused_nograd_rmsnorm_modulation_calls += 1
    else:
        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = modulate(
            norm_hidden_states,
            scale_msa,
            shift_msa,
            adaln_indices,
        )
    if not grad_enabled:
        stats.fused_modulation_calls += 1
    attn_output = self.attn(norm_hidden_states, rotary_emb, attention_mask)
    hidden_states = residual_update(residual, gate_msa, attn_output, adaln_indices)
    if not grad_enabled:
        stats.fused_residual_calls += 1

    residual = hidden_states
    if not grad_enabled and self._h3_fused_rmsnorm_modulate_enabled:
        norm_hidden_states = fused_rmsnorm_modulate(
            hidden_states,
            self.norm2.weight,
            scale_mlp,
            shift_mlp,
            adaln_indices,
            eps=self.norm2.eps,
        )
        stats.fused_nograd_rmsnorm_modulation_calls += 1
    else:
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = modulate(
            norm_hidden_states,
            scale_mlp,
            shift_mlp,
            adaln_indices,
        )
    if not grad_enabled:
        stats.fused_modulation_calls += 1
    ff_output = self.ff(norm_hidden_states)
    hidden_states = residual_update(residual, gate_mlp, ff_output, adaln_indices)
    if not grad_enabled:
        stats.fused_residual_calls += 1
    if not grad_enabled:
        stats.fused_nograd_block_calls += 1
    return hidden_states


class _FusedModulateAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, scale, shift, indices, stats):
        if scale.requires_grad or shift.requires_grad or indices.requires_grad:
            raise RuntimeError("H3 fused pointwise grad path requires frozen AdaLN tables/indices")
        from .triton_fused_pointwise import fused_modulate

        ctx.save_for_backward(scale, indices)
        ctx.stats = stats
        stats.fused_grad_modulation_calls += 1
        return fused_modulate(value, scale, shift, indices)

    @staticmethod
    def backward(ctx, grad_output):
        from .triton_fused_pointwise import fused_modulate_backward

        scale, indices = ctx.saved_tensors
        ctx.stats.fused_grad_modulation_backward_calls += 1
        grad_value = fused_modulate_backward(grad_output.contiguous(), scale, indices)
        return grad_value, None, None, None, None


class _FusedResidualAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, residual, gate, branch, indices, stats):
        if gate.requires_grad or indices.requires_grad:
            raise RuntimeError("H3 fused pointwise grad path requires frozen AdaLN gates/indices")
        from .triton_fused_pointwise import fused_residual

        ctx.save_for_backward(gate, indices)
        ctx.stats = stats
        stats.fused_grad_residual_calls += 1
        return fused_residual(residual, gate, branch, indices)

    @staticmethod
    def backward(ctx, grad_output):
        from .triton_fused_pointwise import fused_residual_branch_backward

        gate, indices = ctx.saved_tensors
        ctx.stats.fused_grad_residual_backward_calls += 1
        grad_output = grad_output.contiguous()
        grad_branch = fused_residual_branch_backward(grad_output, gate, indices)
        # The eager add backward returns the incoming BF16 gradient unchanged
        # for the residual edge.  Returning the same tensor preserves that
        # exact value and avoids a redundant device copy.
        return grad_output, None, grad_branch, None, None


def install_fused_block_pointwise(
    transformer: torch.nn.Module,
    *,
    enabled: bool,
    grad_enabled: bool = False,
    rmsnorm_modulate_enabled: bool = False,
) -> FusedPointwiseRegistration:
    """Patch the 50 pinned block instances without editing shared Diffusers."""

    blocks = list(getattr(transformer, "transformer_blocks", ()))
    if not enabled:
        if grad_enabled or rmsnorm_modulate_enabled:
            raise RuntimeError(
                "H3 fused grad/RMSNorm pointwise requires base pointwise fusion enabled"
            )
        return FusedPointwiseRegistration(
            False, False, False, None, len(blocks), FusedPointwiseStats()
        )

    if len(blocks) != EXPECTED_BLOCK_COUNT:
        raise RuntimeError(
            f"H3 fused pointwise requires {EXPECTED_BLOCK_COUNT} blocks, got {len(blocks)}"
        )
    block_type = type(blocks[0])
    if any(type(block) is not block_type for block in blocks):
        raise RuntimeError("H3 fused pointwise requires one homogeneous pinned block type")
    source_sha256 = _source_sha256(block_type.forward)
    if source_sha256 != PINNED_BLOCK_FORWARD_SHA256:
        raise RuntimeError(
            "H3 block forward source identity mismatch: "
            f"observed={source_sha256} expected={PINNED_BLOCK_FORWARD_SHA256}"
        )

    if rmsnorm_modulate_enabled:
        for block_index, block in enumerate(blocks):
            for norm_name in ("norm1", "norm2"):
                norm = getattr(block, norm_name, None)
                if not isinstance(norm, torch.nn.RMSNorm):
                    raise RuntimeError(
                        f"H3 block {block_index} {norm_name} is {type(norm).__name__}, "
                        "expected torch.nn.RMSNorm"
                    )
                normalized_shape = tuple(norm.normalized_shape)
                if normalized_shape != (EXPECTED_HIDDEN_SIZE,):
                    raise RuntimeError(
                        f"H3 block {block_index} {norm_name} normalized_shape="
                        f"{normalized_shape}, expected {(EXPECTED_HIDDEN_SIZE,)}"
                    )
                if float(norm.eps) != EXPECTED_RMSNORM_EPS:
                    raise RuntimeError(
                        f"H3 block {block_index} {norm_name} eps={norm.eps}, "
                        f"expected {EXPECTED_RMSNORM_EPS}"
                    )
                if norm.weight is None or norm.weight.requires_grad:
                    raise RuntimeError(
                        f"H3 block {block_index} {norm_name} requires one frozen affine weight"
                    )

    stats = FusedPointwiseStats()
    for block in blocks:
        if hasattr(block, "_h3_original_forward"):
            raise RuntimeError("H3 fused pointwise was installed more than once")
        object.__setattr__(block, "_h3_original_forward", block.forward)
        object.__setattr__(block, "_h3_fused_pointwise_stats", stats)
        object.__setattr__(block, "_h3_fused_pointwise_grad_enabled", bool(grad_enabled))
        object.__setattr__(
            block,
            "_h3_fused_rmsnorm_modulate_enabled",
            bool(rmsnorm_modulate_enabled),
        )
        object.__setattr__(block, "forward", types.MethodType(_patched_forward, block))
    return FusedPointwiseRegistration(
        True,
        bool(grad_enabled),
        bool(rmsnorm_modulate_enabled),
        source_sha256,
        len(blocks),
        stats,
    )


def validate_cycle(registration: FusedPointwiseRegistration, start: dict[str, int]) -> dict[str, int]:
    """Fail closed when any no-grad H3 block silently falls back to eager."""

    if not registration.enabled:
        return {}
    current = registration.snapshot()
    delta = {key: int(current[key]) - int(start.get(key, 0)) for key in current}
    expected = {
        "fused_nograd_block_calls": EXPECTED_NOGRAD_BLOCK_CALLS_PER_CYCLE,
        "fused_modulation_calls": 2 * EXPECTED_NOGRAD_BLOCK_CALLS_PER_CYCLE,
        "fused_residual_calls": 2 * EXPECTED_NOGRAD_BLOCK_CALLS_PER_CYCLE,
        "fused_grad_block_calls": 600 if registration.grad_enabled else 0,
        "reference_grad_block_calls": 0 if registration.grad_enabled else 600,
        "fused_grad_modulation_calls": 1200 if registration.grad_enabled else 0,
        "fused_grad_residual_calls": 1200 if registration.grad_enabled else 0,
        "fused_grad_modulation_backward_calls": 600 if registration.grad_enabled else 0,
        "fused_grad_residual_backward_calls": 600 if registration.grad_enabled else 0,
        "fused_nograd_rmsnorm_modulation_calls": (
            EXPECTED_NOGRAD_RMSNORM_MODULATION_CALLS_PER_CYCLE
            if registration.rmsnorm_modulate_enabled
            else 0
        ),
    }
    errors = [f"{key}={delta[key]} expected={value}" for key, value in expected.items() if delta[key] != value]
    if errors:
        raise RuntimeError(f"H3 fused pointwise cycle contract failed: {'; '.join(errors)}")
    return delta
