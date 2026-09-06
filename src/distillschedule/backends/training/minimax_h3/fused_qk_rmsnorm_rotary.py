"""Opt-in no-grad Q/K RMSNorm + rotary fusion for pinned MiniMax-H3."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import inspect
from typing import Any, Callable

import torch

from .fused_rotary import FusedRotaryRegistration


PINNED_PROCESSOR_CALL_SHA256 = (
    "0db0e6aa98d226a24f88079b97abb3f546b7af5f81f5fad19d6e35336a0638d4"
)
EXPECTED_BLOCK_COUNT = 50
EXPECTED_NOGRAD_ATTENTION_CALLS_PER_CYCLE = 25 * EXPECTED_BLOCK_COUNT
EXPECTED_NOGRAD_QK_CALLS_PER_CYCLE = 2 * EXPECTED_NOGRAD_ATTENTION_CALLS_PER_CYCLE
EXPECTED_REFERENCE_GRAD_ATTENTION_CALLS_PER_CYCLE = 6 * EXPECTED_BLOCK_COUNT * 2
EXPECTED_FUSED_GRAD_ATTENTION_CALLS_PER_CYCLE = EXPECTED_REFERENCE_GRAD_ATTENTION_CALLS_PER_CYCLE
EXPECTED_FUSED_GRAD_QK_CALLS_PER_CYCLE = 2 * EXPECTED_FUSED_GRAD_ATTENTION_CALLS_PER_CYCLE
EXPECTED_FUSED_GRAD_QK_BACKWARD_CALLS_PER_CYCLE = 6 * EXPECTED_BLOCK_COUNT * 2


@dataclasses.dataclass
class FusedQKStats:
    fused_nograd_attention_calls: int = 0
    fused_nograd_qk_calls: int = 0
    fused_grad_attention_calls: int = 0
    fused_grad_qk_calls: int = 0
    fused_grad_qk_backward_calls: int = 0
    reference_grad_attention_calls: int = 0


@dataclasses.dataclass
class FusedQKRegistration:
    enabled: bool
    grad_enabled: bool
    source_sha256: str | None
    attention_count: int
    stats: FusedQKStats
    original: Callable[..., torch.Tensor] | None = None

    def snapshot(self) -> dict[str, int]:
        return dataclasses.asdict(self.stats)

    def receipt(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "grad_enabled": self.grad_enabled,
            "source_sha256": self.source_sha256,
            "attention_count": self.attention_count,
            "stats": self.snapshot(),
        }


def _source_sha256(function: Callable[..., Any]) -> str:
    return hashlib.sha256(inspect.getsource(function).encode()).hexdigest()


class _FusedQKRMSNormRotaryAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        eps: float,
        stats: FusedQKStats,
        base_stats: Any,
    ) -> torch.Tensor:
        if weight.requires_grad or cos.requires_grad or sin.requires_grad:
            raise RuntimeError(
                "H3 fused grad Q/K path requires frozen RMSNorm and rotary tables"
            )
        from .triton_fused_rotary import fused_qk_rmsnorm_rotary

        ctx.save_for_backward(hidden_states, weight, cos, sin)
        ctx.eps = float(eps)
        ctx.stats = stats
        ctx.base_stats = base_stats
        return fused_qk_rmsnorm_rotary(hidden_states, weight, cos, sin, eps=eps)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):
        from .triton_fused_rotary import fused_qk_rmsnorm_rotary_backward

        hidden_states, weight, cos, sin = ctx.saved_tensors
        ctx.stats.fused_grad_qk_backward_calls += 1
        ctx.base_stats.fused_grad_backward_calls += 1
        grad_hidden = fused_qk_rmsnorm_rotary_backward(
            grad_output.contiguous(),
            hidden_states,
            weight,
            cos,
            sin,
            eps=ctx.eps,
        )
        return grad_hidden, None, None, None, None, None, None


def install_fused_qk_rmsnorm_rotary(
    transformer: torch.nn.Module,
    *,
    enabled: bool,
    grad_enabled: bool = False,
    base_rotary_registration: FusedRotaryRegistration | None = None,
) -> FusedQKRegistration:
    """Patch the pinned processor class for bounded no-grad and optional grad fusion."""

    blocks = list(getattr(transformer, "transformer_blocks", ()))
    stats = FusedQKStats()
    if grad_enabled and not enabled:
        raise RuntimeError("H3 fused grad Q/K path requires base Q/K fusion enabled")
    if not enabled:
        return FusedQKRegistration(False, False, None, len(blocks), stats)
    if len(blocks) != EXPECTED_BLOCK_COUNT:
        raise RuntimeError(
            f"H3 fused Q/K path requires {EXPECTED_BLOCK_COUNT} blocks, got {len(blocks)}"
        )
    if base_rotary_registration is None or not base_rotary_registration.enabled:
        raise RuntimeError(
            "H3 fused Q/K RMSNorm rotary requires the installed base rotary registration"
        )
    if grad_enabled and not base_rotary_registration.grad_enabled:
        raise RuntimeError(
            "H3 fused grad Q/K path requires the installed grad rotary registration"
        )

    module = importlib.import_module("diffusers.models.transformers.transformer_minimax_h3")
    processor_type = getattr(module, "MiniMaxH3AttnProcessor", None)
    if processor_type is None:
        raise RuntimeError("pinned MiniMax-H3 module has no MiniMaxH3AttnProcessor")
    original = getattr(processor_type, "__call__", None)
    if not callable(original):
        raise RuntimeError("pinned MiniMax-H3 processor has no callable __call__")
    if getattr(original, "_h3_fused_qk_rmsnorm_rotary_wrapper", False):
        raise RuntimeError("H3 fused Q/K RMSNorm rotary was installed more than once")
    source_sha256 = _source_sha256(original)
    if source_sha256 != PINNED_PROCESSOR_CALL_SHA256:
        raise RuntimeError(
            "H3 attention processor source identity mismatch: "
            f"observed={source_sha256} expected={PINNED_PROCESSOR_CALL_SHA256}"
        )

    attentions = []
    for block_index, block in enumerate(blocks):
        attention = getattr(block, "attn", None)
        if attention is None or type(getattr(attention, "processor", None)) is not processor_type:
            raise RuntimeError(f"H3 block {block_index} does not use the pinned attention processor")
        if int(getattr(attention, "heads", -1)) != 56 or int(
            getattr(attention, "head_dim", -1)
        ) != 128:
            raise RuntimeError(
                f"H3 block {block_index} attention geometry is "
                f"heads={getattr(attention, 'heads', None)} "
                f"head_dim={getattr(attention, 'head_dim', None)}, expected 56/128"
            )
        for norm_name in ("norm_q", "norm_k"):
            norm = getattr(attention, norm_name, None)
            if not isinstance(norm, torch.nn.RMSNorm):
                raise RuntimeError(
                    f"H3 block {block_index} {norm_name} is {type(norm).__name__}, "
                    "expected torch.nn.RMSNorm"
                )
            if tuple(norm.normalized_shape) != (128,) or float(norm.eps) != 1e-5:
                raise RuntimeError(
                    f"H3 block {block_index} {norm_name} contract changed: "
                    f"shape={tuple(norm.normalized_shape)} eps={norm.eps}"
                )
            if norm.weight is None or norm.weight.requires_grad:
                raise RuntimeError(
                    f"H3 block {block_index} {norm_name} requires one frozen affine weight"
                )
        attentions.append(attention)

    def wrapped(
        self,
        attn,
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Token-refiner attention has no rotary embedding and is outside this
        # preregistered main-block candidate.
        if rotary_emb is None:
            return original(self, attn, hidden_states, rotary_emb, attention_mask)
        grad_path = torch.is_grad_enabled()
        if grad_path and not grad_enabled:
            # Count entry: checkpoint replay may early-stop before normal return.
            stats.reference_grad_attention_calls += 1
            return original(self, attn, hidden_states, rotary_emb, attention_mask)

        from .triton_fused_rotary import fused_qk_rmsnorm_rotary

        if grad_path:
            stats.fused_grad_attention_calls += 1
        else:
            stats.fused_nograd_attention_calls += 1
        if attn.fused_projections:
            query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            query = attn.to_q(hidden_states)
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))
        if grad_path:
            query = _FusedQKRMSNormRotaryAutograd.apply(
                query,
                attn.norm_q.weight,
                *rotary_emb,
                attn.norm_q.eps,
                stats,
                base_rotary_registration.stats,
            )
            key = _FusedQKRMSNormRotaryAutograd.apply(
                key,
                attn.norm_k.weight,
                *rotary_emb,
                attn.norm_k.eps,
                stats,
                base_rotary_registration.stats,
            )
            stats.fused_grad_qk_calls += 2
            # Preserve the parent exact-rotary physical forward census: these
            # two transforms are now nested in the fused Q/K kernel.
            base_rotary_registration.stats.fused_grad_calls += 2
        else:
            query = fused_qk_rmsnorm_rotary(
                query,
                attn.norm_q.weight,
                *rotary_emb,
                eps=attn.norm_q.eps,
            )
            key = fused_qk_rmsnorm_rotary(
                key,
                attn.norm_k.weight,
                *rotary_emb,
                eps=attn.norm_k.eps,
            )
            stats.fused_nograd_qk_calls += 2
            # The two exact rotary operations are subsumed by this fused kernel, so
            # retain the parent feature's physical-call census rather than making
            # its otherwise-correct fail-closed gate report zero calls.
            base_rotary_registration.stats.fused_nograd_calls += 2
        output = module.dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        output = output.flatten(2, 3).type_as(query)
        output = attn.to_out[0](output)
        output = attn.to_out[1](output)
        return output

    setattr(wrapped, "_h3_fused_qk_rmsnorm_rotary_wrapper", True)
    processor_type.__call__ = wrapped
    return FusedQKRegistration(
        True, grad_enabled, source_sha256, len(attentions), stats, original
    )


def validate_cycle(
    registration: FusedQKRegistration, start: dict[str, int]
) -> dict[str, int]:
    if not registration.enabled:
        return {}
    current = registration.snapshot()
    delta = {key: int(current[key]) - int(start.get(key, 0)) for key in current}
    expected = {
        "fused_nograd_attention_calls": EXPECTED_NOGRAD_ATTENTION_CALLS_PER_CYCLE,
        "fused_nograd_qk_calls": EXPECTED_NOGRAD_QK_CALLS_PER_CYCLE,
        "fused_grad_attention_calls": (
            EXPECTED_FUSED_GRAD_ATTENTION_CALLS_PER_CYCLE
            if registration.grad_enabled
            else 0
        ),
        "fused_grad_qk_calls": (
            EXPECTED_FUSED_GRAD_QK_CALLS_PER_CYCLE if registration.grad_enabled else 0
        ),
        "fused_grad_qk_backward_calls": (
            EXPECTED_FUSED_GRAD_QK_BACKWARD_CALLS_PER_CYCLE
            if registration.grad_enabled
            else 0
        ),
        "reference_grad_attention_calls": (
            0
            if registration.grad_enabled
            else EXPECTED_REFERENCE_GRAD_ATTENTION_CALLS_PER_CYCLE
        ),
    }
    errors = [
        f"{key}={delta[key]} expected={value}"
        for key, value in expected.items()
        if delta[key] != value
    ]
    if errors:
        raise RuntimeError(f"H3 fused Q/K RMSNorm rotary cycle failed: {'; '.join(errors)}")
    return delta
