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

from contextlib import contextmanager
from copy import deepcopy
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
_RETIREMENT_BINDING = "_distillschedule_generator_text_kv_retirement_v1"
_RETIREMENT_CONTEXT = "_distillschedule_generator_text_kv_owner_v1"
RETIREMENT_PRESET = "self_forcing.c51_stream_retire_textkv.v1"
CONTEXT_SHAPE = (1, 512, 1536)
BLOCK_COUNT = 30
NUM_HEADS = 12
HEAD_DIM = 128
CACHE_BYTES = 141_557_760


def payload_retirement_policy() -> dict[str, Any]:
    """Torch-free exact lifecycle identity; not an independently tunable factor."""
    return dict(schema_version="distillschedule.livewan_generator_text_kv_retirement.v1",
        preset=RETIREMENT_PRESET, scope="public_c51_continuation_generator_only",
        boundary="prepare_rollout_after_nograd_scope", payload=CACHE_ATTRIBUTE,
        release="clear_dictionary_tensor_entries_only", marker="retain_empty_until_original_set_text",
        grad="original_forward_before_cache_entry_lookup", retired_nograd="reject",
        next_set_text="original_invalidation_and_unconditional_rebuild",
        failed_rollout="release_built_owned_payload_under_aborted_not_success_counters",
        context_shape=list(CONTEXT_SHAPE), blocks=BLOCK_COUNT, heads=NUM_HEADS, head_dim=HEAD_DIM,
        payload_bytes=CACHE_BYTES, source_set_text_sha256=EXPECTED_SET_TEXT_SHA256,
        source_cross_forward_sha256=EXPECTED_CROSS_FORWARD_SHA256,
        source_flash_attention_sha256=EXPECTED_FLASH_ATTENTION_SHA256)


def _retirement_fail(receipt: dict[str, Any], message: str) -> None:
    receipt["contract_failure_count"] += 1
    receipt["text_kv_payload_retirement"]["contract_failure_count"] += 1
    raise RuntimeError("LiveWan Generator text K/V retirement contract changed: " + message)


def bind_generator_text_kv_retirement(torch: Any, *, streamer: Any, student: Any,
                                      policy: dict[str, Any]) -> dict[str, Any]:
    """Bind only an explicitly admitted continuation's original streamer owner."""
    from wanstreamer.stream import FewStepStreamer

    if policy != payload_retirement_policy():
        raise RuntimeError("LiveWan Generator text K/V retirement policy/version changed")
    receipt = getattr(FewStepStreamer, "_distillschedule_cross_kv_receipt", None)
    if (type(streamer) is not FewStepStreamer or streamer.model is not student
            or getattr(FewStepStreamer, "_distillschedule_cross_kv_cache", None)
            != "generator_nograd_text_kv_cache_v1" or not isinstance(receipt, dict)):
        raise RuntimeError("LiveWan Generator text K/V retirement requires the installed original owner")
    if "text_kv_payload_retirement" not in receipt:
        receipt["text_kv_payload_retirement"] = dict(policy=deepcopy(policy), execution_status="BOUND",
            bound_streamer_count=0, retired_context_count=0, retired_block_count=0,
            retired_payload_bytes=0, aborted_context_count=0, aborted_block_count=0,
            aborted_payload_bytes=0, contract_failure_count=0)
    if (hasattr(streamer, _RETIREMENT_BINDING) or getattr(streamer, "ctx", None) is not None
            or len(student.blocks) != BLOCK_COUNT):
        _retirement_fail(receipt, "duplicate binding, live context or block inventory changed")
    setattr(streamer, _RETIREMENT_BINDING, dict(policy=deepcopy(policy), owner=id(streamer),
        model=id(student), receipt=receipt, active=False))
    receipt["text_kv_payload_retirement"]["bound_streamer_count"] += 1
    return receipt["text_kv_payload_retirement"]


def _retirement_binding(streamer: Any) -> dict[str, Any] | None:
    binding = getattr(streamer, _RETIREMENT_BINDING, None)
    if binding is not None:
        if type(binding) is not dict or type(binding.get("receipt")) is not dict:
            raise RuntimeError("LiveWan Generator text K/V retirement binding/receipt changed")
        from wanstreamer.stream import FewStepStreamer
        receipt = binding["receipt"]
        if (binding.get("policy") != payload_retirement_policy() or binding.get("owner") != id(streamer)
                or binding.get("model") != id(streamer.model)
                or type(streamer) is not FewStepStreamer
                or receipt is not getattr(FewStepStreamer, "_distillschedule_cross_kv_receipt", None)
                or type(binding.get("active")) is not bool):
            _retirement_fail(receipt, "policy/version or streamer/model owner changed")
    return binding


def _retire_generator_text_kv_payload(torch: Any, *, streamer: Any, aborted: bool) -> None:
    """Drop dead no-grad projections, never context, native self-KV or graph state.

    Validation reads tensor metadata and parameter versions only. Logical bytes
    are not driver-free memory, and this performs no CUDA sync/allocator call.
    """
    binding = _retirement_binding(streamer)
    if binding is None:
        raise RuntimeError("LiveWan Generator text K/V retirement requires an explicit bound owner")
    receipt = binding["receipt"]
    if binding["active"]:
        _retirement_fail(receipt, "active rollout cannot retire its no-grad payload")
    context = getattr(streamer, "ctx", None)
    cache = getattr(context, CACHE_ATTRIBUTE, None)
    owner = getattr(context, _RETIREMENT_CONTEXT, None)
    if (type(cache) is not dict or type(owner) is not dict
            or owner.get("policy") != binding["policy"] or owner.get("streamer") != id(streamer)
            or owner.get("model") != id(streamer.model) or owner.get("context") != id(context)
            or owner.get("cache") != id(cache)):
        _retirement_fail(receipt, "missing marker or context/cache ownership/version changed")
    if owner.get("retired") is not False:
        _retirement_fail(receipt, "duplicate retirement of an already retired context")
    if (not context.is_cuda or context.dtype != torch.float32 or context.requires_grad
            or tuple(context.shape) != CONTEXT_SHAPE or streamer.ctx_lens is not None
            or len(streamer.model.blocks) != BLOCK_COUNT):
        _retirement_fail(receipt, "context geometry/device/dtype/length changed")
    blocks = tuple(block.cross_attn for block in streamer.model.blocks)
    if set(cache) != {id(block) for block in blocks} or len(cache) != BLOCK_COUNT:
        _retirement_fail(receipt, "cache block ownership/inventory changed")
    total_bytes = 0
    for block in blocks:
        entry = cache[id(block)]
        if (type(entry) is not dict or set(entry) != {"key", "value", "parameter_versions"}
                or block.num_heads != NUM_HEADS or block.head_dim != HEAD_DIM
                or entry["parameter_versions"] != _parameter_versions(block)):
            _retirement_fail(receipt, "block geometry or parameter version changed")
        for name, dtype in (("key", torch.float32), ("value", torch.bfloat16)):
            tensor = entry[name]
            if (not torch.is_tensor(tensor) or not tensor.is_cuda or tensor.device != context.device
                    or tensor.dtype != dtype or tensor.requires_grad or tensor.grad_fn is not None
                    or tuple(tensor.shape) != (*CONTEXT_SHAPE[:2], NUM_HEADS, HEAD_DIM)
                    or not tensor.is_contiguous()):
                _retirement_fail(receipt, "payload geometry/device/dtype/autograd changed")
            total_bytes += tensor.numel() * tensor.element_size()
    if total_bytes != CACHE_BYTES:
        _retirement_fail(receipt, "payload logical byte inventory changed")
    cache.clear()  # KEEP the dictionary marker: grad replay bypasses before lookup.
    owner["retired"] = True
    counters = receipt["text_kv_payload_retirement"]
    counters["execution_status"] = "ABORTED" if aborted else "EXECUTED"
    prefix = "aborted" if aborted else "retired"
    counters[prefix+"_context_count"] += 1
    counters[prefix+"_block_count"] += len(blocks)
    counters[prefix+"_payload_bytes"] += total_bytes


def retire_generator_text_kv_payload(torch: Any, *, streamer: Any) -> None:
    """Retire one successfully completed no-grad rollout's exact owned payload."""
    _retire_generator_text_kv_payload(torch, streamer=streamer, aborted=False)


@contextmanager
def generator_text_kv_payload_lifetime(torch: Any, streamer: Any):
    """Default-OFF boundary around the unchanged complete no-grad rollout scope."""
    binding = _retirement_binding(streamer)
    if binding is None:
        yield
        return
    if binding["active"]:
        _retirement_fail(binding["receipt"], "nested rollout ownership")
    binding["active"] = True
    try:
        yield
    except BaseException as exc:
        # The failed rollout is not a completed retirement/capacity receipt.
        # Clear a successfully built owned payload without replaying the model
        # or masking the original exception. Incomplete/invalid builds still
        # fail closed and cannot become a successful lifecycle receipt.
        binding["active"] = False
        owner = getattr(getattr(streamer, "ctx", None), _RETIREMENT_CONTEXT, None)
        if type(owner) is dict and owner.get("retired") is False:
            try:
                _retire_generator_text_kv_payload(torch, streamer=streamer, aborted=True)
            except RuntimeError as cleanup:
                if hasattr(exc, "add_note"):  # Python 3.10 keeps the original exception too.
                    exc.add_note(str(cleanup))
        raise
    else:
        binding["active"] = False
        retire_generator_text_kv_payload(torch, streamer=streamer)


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
        retirement = _retirement_binding(self)
        if retirement is not None and not retirement["active"]:
            _retirement_fail(receipt, "set_text outside the bound rollout lifetime")
        old_context = getattr(self, "ctx", None)
        if old_context is not None and hasattr(old_context, CACHE_ATTRIBUTE):
            delattr(old_context, CACHE_ATTRIBUTE)
            if hasattr(old_context, _RETIREMENT_CONTEXT):
                delattr(old_context, _RETIREMENT_CONTEXT)
            receipt["cache_invalidation_count"] += 1
        original_set_text(self, context, context_lens)
        if torch.is_grad_enabled():
            fail("set_text must run under torch.no_grad")
        if (
            not context.is_cuda
            or context.dtype != torch.float32
            or context.requires_grad
            or context.ndim != 3
            or tuple(context.shape) != CONTEXT_SHAPE
            or self.ctx is not context
            or self.ctx_lens is not None
            or len(self.model.blocks) != BLOCK_COUNT
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
                if cross.num_heads != NUM_HEADS or cross.head_dim != HEAD_DIM:
                    fail(f"block {block_index} attention geometry changed")
                key = cross.norm_k(cross.k(context)).view(*CONTEXT_SHAPE[:2], NUM_HEADS, HEAD_DIM)
                value = cross.v(context).view(*CONTEXT_SHAPE[:2], NUM_HEADS, HEAD_DIM)
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

        if len(cache) != BLOCK_COUNT or total_bytes != CACHE_BYTES:
            fail(f"cache inventory changed: blocks={len(cache)}, bytes={total_bytes}")
        setattr(context, CACHE_ATTRIBUTE, cache)
        if retirement is not None:
            setattr(context, _RETIREMENT_CONTEXT, dict(policy=deepcopy(retirement["policy"]),
                streamer=id(self), model=id(self.model), context=id(context), cache=id(cache), retired=False))
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
        owner = getattr(context, _RETIREMENT_CONTEXT, None)
        if cache is None:
            if owner is not None:
                _retirement_fail(receipt, "marked context lost its cache marker")
            receipt["uncached_fallback_count"] += 1
            return original_forward(self, x, context, context_lens)
        if torch.is_grad_enabled():
            receipt["grad_bypass_count"] += 1
            return original_forward(self, x, context, context_lens)

        if owner is not None and owner.get("retired") is True:
            _retirement_fail(receipt, "retired context cannot serve a no-grad consumer")

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
            or x.shape[2] != CONTEXT_SHAPE[2]
            or tuple(context.shape) != CONTEXT_SHAPE
            or _parameter_versions(self) != entry["parameter_versions"]
        ):
            fail(
                "cached query/context or parameter version changed: "
                f"x={x.device}/{x.dtype}/{tuple(x.shape)}/{x.requires_grad}, "
                f"context={tuple(context.shape)}"
            )

        key, value = entry["key"], entry["value"]
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            query = self.norm_q(self.q(x)).view(1, -1, NUM_HEADS, HEAD_DIM)
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
    "payload_retirement_policy",
    "bind_generator_text_kv_retirement",
    "generator_text_kv_payload_lifetime",
    "retire_generator_text_kv_payload",
    "install_generator_cross_kv_cache",
    "uninstall_generator_cross_kv_cache",
]
