"""Exact no-grad text K/V reuse for persistent LiveWan Generator ranks.

The block-causal Student sees one immutable projected prompt throughout one
rollout, while its weights cannot change until the later Generator commit.
The pinned upstream implementation nevertheless reruns every block's text
K/V projections for every denoising call.  This module builds those projections
once when ``FewStepStreamer.set_text`` installs the prompt and reuses them only
while autograd is disabled.  Generator checkpoint replay deliberately falls
back to the original function so K/V projection parameter gradients are
unchanged.
"""

from __future__ import annotations

import hashlib
import inspect
import textwrap
from typing import Any


EXPECTED_SET_TEXT_SHA256 = (
    "def2452ea9528abb9af2aa166d78afdf9b8b7c5daa52c0fc87bfd63bc9832386"
)
EXPECTED_CROSS_FORWARD_SHA256 = (
    "0df9365afb37bb8a876d8c8659ca95d1caaffb89bf3e64918b782eb86b0a8253"
)
EXPECTED_FLASH_ATTENTION_SHA256 = (
    "c59c125fd92234d2ddec63f3ae3bd2a2479862faba44fa955fdedad2b8203a59"
)
CACHE_ATTRIBUTE = "_distillschedule_generator_text_kv_v1"


def _source_sha256(value: Any) -> str:
    source = textwrap.dedent(inspect.getsource(value)).strip()
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _parameter_versions(module: Any) -> tuple[int, ...]:
    parameters = (
        tuple(module.k.parameters())
        + tuple(module.v.parameters())
        + tuple(module.norm_k.parameters())
    )
    return tuple(int(parameter._version) for parameter in parameters)


def install_generator_cross_kv_cache(torch: Any) -> dict[str, Any]:
    """Install the plan-gated Generator-only no-grad text K/V cache."""

    from wan.modules import model as model_module
    from wanstreamer.stream import FewStepStreamer

    requested_name = "generator_nograd_text_kv_cache_v1"
    installed = getattr(FewStepStreamer, "_distillschedule_cross_kv_cache", None)
    if installed is not None:
        if installed != requested_name:
            raise RuntimeError(
                f"LiveWan Generator cross K/V cache already installed as {installed}, "
                f"requested {requested_name}"
            )
        return FewStepStreamer._distillschedule_cross_kv_receipt

    original_set_text = FewStepStreamer.set_text
    original_forward = model_module.WanT2VCrossAttention.forward
    observed_sources = {
        "set_text": _source_sha256(original_set_text),
        "cross_forward": _source_sha256(original_forward),
        "flash_attention": _source_sha256(model_module.flash_attention),
    }
    expected_sources = {
        "set_text": EXPECTED_SET_TEXT_SHA256,
        "cross_forward": EXPECTED_CROSS_FORWARD_SHA256,
        "flash_attention": EXPECTED_FLASH_ATTENTION_SHA256,
    }
    if observed_sources != expected_sources:
        raise RuntimeError(
            "LiveWan Generator cross K/V source changed: "
            f"expected={expected_sources}, observed={observed_sources}"
        )

    receipt: dict[str, Any] = {
        "enabled": True,
        "name": requested_name,
        "source_set_text_sha256": EXPECTED_SET_TEXT_SHA256,
        "source_cross_forward_sha256": EXPECTED_CROSS_FORWARD_SHA256,
        "source_flash_attention_sha256": EXPECTED_FLASH_ATTENTION_SHA256,
        "function_patched": True,
        "install_status": "PASS",
        "cache_build_context_count": 0,
        "cache_build_block_count": 0,
        "cache_build_bytes": 0,
        "cache_invalidation_count": 0,
        "cache_hit_count": 0,
        "grad_bypass_count": 0,
        "uncached_fallback_count": 0,
        "contract_failure_count": 0,
        "query_shape_histogram": {},
    }

    def fail(message: str) -> None:
        receipt["contract_failure_count"] += 1
        raise RuntimeError(f"LiveWan Generator cross K/V contract changed: {message}")

    def cached_set_text(self: Any, context: Any, context_lens: Any = None) -> None:
        old_context = getattr(self, "ctx", None)
        if old_context is not None and hasattr(old_context, CACHE_ATTRIBUTE):
            delattr(old_context, CACHE_ATTRIBUTE)
            receipt["cache_invalidation_count"] += 1
        original_set_text(self, context, context_lens)
        if torch.is_grad_enabled():
            fail("set_text must run under torch.no_grad")
        if (
            not context.is_cuda
            or context.dtype != torch.float32
            or context.requires_grad
            or context.ndim != 3
            or tuple(context.shape) != (1, 512, 1536)
            or self.ctx is not context
            or self.ctx_lens is not None
            or len(self.model.blocks) != 30
        ):
            fail(
                "context/device/dtype/shape/length/block inventory mismatch: "
                f"device={context.device}, dtype={context.dtype}, "
                f"requires_grad={context.requires_grad}, shape={tuple(context.shape)}, "
                f"ctx_lens={self.ctx_lens}, blocks={len(self.model.blocks)}"
            )

        cache: dict[int, dict[str, Any]] = {}
        total_bytes = 0
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for block_index, block in enumerate(self.model.blocks):
                cross = block.cross_attn
                if not isinstance(cross, model_module.WanT2VCrossAttention):
                    fail(f"block {block_index} cross-attention type changed")
                if cross.num_heads != 12 or cross.head_dim != 128:
                    fail(f"block {block_index} attention geometry changed")
                key = cross.norm_k(cross.k(context)).view(1, 512, 12, 128)
                value = cross.v(context).view(1, 512, 12, 128)
                if (
                    key.dtype != torch.float32
                    or value.dtype != torch.bfloat16
                    or not key.is_contiguous()
                    or not value.is_contiguous()
                ):
                    fail(f"block {block_index} cached K/V layout changed")
                cache[id(cross)] = {
                    "key": key,
                    "value": value,
                    "parameter_versions": _parameter_versions(cross),
                }
                total_bytes += int(
                    key.numel() * key.element_size()
                    + value.numel() * value.element_size()
                )

        if len(cache) != 30 or total_bytes != 141_557_760:
            fail(f"cache inventory changed: blocks={len(cache)}, bytes={total_bytes}")
        setattr(context, CACHE_ATTRIBUTE, cache)
        receipt["cache_build_context_count"] += 1
        receipt["cache_build_block_count"] += len(cache)
        receipt["cache_build_bytes"] += total_bytes

    def cached_forward(
        self: Any,
        x: Any,
        context: Any,
        context_lens: Any,
    ) -> Any:
        cache = getattr(context, CACHE_ATTRIBUTE, None)
        if cache is None:
            receipt["uncached_fallback_count"] += 1
            return original_forward(self, x, context, context_lens)
        if torch.is_grad_enabled():
            receipt["grad_bypass_count"] += 1
            return original_forward(self, x, context, context_lens)

        entry = cache.get(id(self))
        if entry is None:
            fail("context cache is missing the active block")
        if context_lens is not None:
            fail("cached full-length context unexpectedly supplied context_lens")
        if (
            not x.is_cuda
            or x.dtype != torch.float32
            or x.requires_grad
            or x.ndim != 3
            or x.shape[0] != 1
            or x.shape[2] != 1536
            or context.shape != (1, 512, 1536)
            or _parameter_versions(self) != entry["parameter_versions"]
        ):
            fail(
                "cached query/context or parameter version changed: "
                f"x={x.device}/{x.dtype}/{tuple(x.shape)}/{x.requires_grad}, "
                f"context={tuple(context.shape)}"
            )

        key, value = entry["key"], entry["value"]
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            query = self.norm_q(self.q(x)).view(1, -1, 12, 128)
            output = model_module.flash_attention(
                query, key, value, k_lens=None
            )
            output = self.o(output.flatten(2))
        receipt["cache_hit_count"] += 1
        shape_key = f"{x.shape[0]}x{x.shape[1]}x{x.shape[2]}"
        receipt["query_shape_histogram"][shape_key] = (
            int(receipt["query_shape_histogram"].get(shape_key, 0)) + 1
        )
        return output

    FewStepStreamer._distillschedule_original_set_text = original_set_text
    model_module.WanT2VCrossAttention._distillschedule_original_forward = (
        original_forward
    )
    FewStepStreamer.set_text = cached_set_text
    model_module.WanT2VCrossAttention.forward = cached_forward
    FewStepStreamer._distillschedule_cross_kv_cache = requested_name
    FewStepStreamer._distillschedule_cross_kv_receipt = receipt
    return receipt


def uninstall_generator_cross_kv_cache() -> None:
    """Test-only restoration; production workers exit with the patch installed."""

    from wan.modules import model as model_module
    from wanstreamer.stream import FewStepStreamer

    original_set_text = getattr(
        FewStepStreamer, "_distillschedule_original_set_text", None
    )
    original_forward = getattr(
        model_module.WanT2VCrossAttention,
        "_distillschedule_original_forward",
        None,
    )
    if original_set_text is not None:
        FewStepStreamer.set_text = original_set_text
    if original_forward is not None:
        model_module.WanT2VCrossAttention.forward = original_forward
    for owner, names in (
        (
            FewStepStreamer,
            (
                "_distillschedule_original_set_text",
                "_distillschedule_cross_kv_cache",
                "_distillschedule_cross_kv_receipt",
            ),
        ),
        (
            model_module.WanT2VCrossAttention,
            ("_distillschedule_original_forward",),
        ),
    ):
        for name in names:
            if hasattr(owner, name):
                delattr(owner, name)


__all__ = [
    "install_generator_cross_kv_cache",
    "uninstall_generator_cross_kv_cache",
]
