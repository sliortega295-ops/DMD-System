"""C49: read-only split-KV tiling for registered Generator no-grad attention."""
from __future__ import annotations

import hashlib
import inspect
from typing import Any, Mapping

VARIANT = "c49-continuous-b16-generator-readonly-splitkv"
CONTRACT = {
    "enabled": True,
    "name": "generator_nograd_fa2_readonly_splitkv2_v1",
    "scope": "generator_ranks_only",
    "num_splits": 2,
    "batch": 1,
    "query_tokens": 2760,
    "key_tokens": [8280, 11040, 13800, 16560, 19320],
    "heads": 12,
    "head_dimension": 128,
    "kernel_dtype": "bfloat16",
    "grad_backend_unchanged": True,
    "reuse_previous_kv": False,
    "append_kv": False,
    "application_calls_unchanged": True,
}
# Filled from the exact hash-verified pinned FA2 source, not an API guess.
EXPECTED_KVCACHE_SOURCE_SHA256 = "bb57b5afc7ac164376fd8c43921488a70f3c6a05568607759a1b497774f6dded"


def enabled(plan: Mapping[str, Any], *, mode: str) -> bool:
    if 'c51_ffn_retention' in plan.get('metadata', {}):
        # Only a fully authenticated C51 child may compose independent C-side reuse.
        from distillschedule.runtime.livewan_c51_ffn_retention import parent_plan
        plan = parent_plan(plan, mode=mode)
    value = plan.get("metadata", {}).get("generator_readonly_splitkv")
    if value is None:
        return False
    if mode != "pipeline" or dict(value) != CONTRACT:
        raise RuntimeError("C49 Generator read-only splitKV contract changed")
    metadata = plan["metadata"]
    if (metadata.get("generator_attention_backend", {}).get("name")
            != "generator_flash_attn2_unmasked_exact_cast_v1"
            or metadata.get("score_attention_backend", {}).get("name")
            != "score_flash_attn3_unmasked_bf16_split1_v1"
            or metadata.get("critic_ffn_down_checkpoint") is not None
            or metadata.get("critic_step1_overlap") is not None
            or metadata.get("critic_lora_kernel", {}).get("name")
            != "scale_one_autocast_bf16_input_elision_v1"
            or metadata.get("step0_wave1_early_delivery") is not None):
        raise RuntimeError("C49 requires the unchanged C38 parent, not other candidates")
    return True


def is_registered_shape(q: Any, k: Any) -> bool:
    return (tuple(q.shape) == (1, 2760, 12, 128)
            and tuple(k.shape) in {
                (1, length, 12, 128) for length in CONTRACT["key_tokens"]})


class ReadOnlySplitKV:
    """Current K/V inputs are consumed in full; no persistent cache is created."""
    def __init__(self, torch: Any, fa2: Any) -> None:
        function = fa2.flash_attn_with_kvcache
        source_hash = hashlib.sha256(inspect.getsource(function).encode()).hexdigest()
        if source_hash != EXPECTED_KVCACHE_SOURCE_SHA256:
            raise RuntimeError("pinned FA2 read-only splitKV API source mismatch")
        self.torch = torch
        self.function = function
        self.receipt = {
            "enabled": True, "name": CONTRACT["name"], "contract": dict(CONTRACT),
            "api_source_sha256": source_hash, "production_installed": True,
            "execution_status": "PASS", "eligible_nograd_calls": 0,
            "reference_calls": 0, "eligible_grad_bypass_calls": 0,
            "contract_failure_count": 0, "shape_histogram": {},
        }

    def execute(self, q: Any, k: Any, v: Any, *, reference: Any,
                reference_options: Mapping[str, Any]) -> Any:
        eligible = is_registered_shape(q, k)
        grad = self.torch.is_grad_enabled() or any(t.requires_grad for t in (q, k, v))
        if not eligible or grad:
            self.receipt["reference_calls"] += 1
            self.receipt["eligible_grad_bypass_calls"] += int(eligible and grad)
            return reference(q, k, v, **reference_options)
        options = reference_options
        if (q.dtype != self.torch.bfloat16 or k.dtype != self.torch.bfloat16
                or v.dtype != self.torch.bfloat16 or tuple(v.shape) != tuple(k.shape)
                or not all(t.is_cuda and t.is_contiguous() for t in (q, k, v))
                or options.get("dropout_p") != 0.0 or options.get("causal") is not False
                or tuple(options.get("window_size", ())) != (-1, -1)
                or options.get("deterministic") is not False
                or options.get("softmax_scale") not in (None, 128 ** (-0.5))):
            self.receipt["contract_failure_count"] += 1
            raise RuntimeError("C49 eligible attention device/dtype/options changed")
        versions = tuple(t._version for t in (q, k, v))
        output = self.function(
            q, k, v, k=None, v=None, rotary_cos=None, rotary_sin=None,
            cache_seqlens=None, cache_batch_idx=None, cache_leftpad=None,
            block_table=None, softmax_scale=options["softmax_scale"],
            causal=False, window_size=(-1, -1), softcap=0.0,
            alibi_slopes=None, num_splits=2, return_softmax_lse=False,
        )
        if (tuple(output.shape) != tuple(q.shape) or output.dtype != q.dtype
                or versions != tuple(t._version for t in (q, k, v))):
            self.receipt["contract_failure_count"] += 1
            raise RuntimeError("C49 output or input metadata contract changed")
        self.receipt["eligible_nograd_calls"] += 1
        key = "q=" + "x".join(map(str, q.shape)) + "|k=" + "x".join(map(str, k.shape))
        self.receipt["shape_histogram"][key] = self.receipt["shape_histogram"].get(key, 0) + 1
        return output


def runtime_receipt_errors(row: Mapping[str, Any], *, rank: int,
                           total_calls: int | None = None) -> list[str]:
    errors = []
    if row.get("enabled") is not True:
        return ["C49 runtime receipt not enabled"]
    if rank >= 8:
        if row.get("name") != CONTRACT["name"] or row.get("contract") != CONTRACT:
            errors.append("C49 Score contract identity missing")
        if row.get("production_installed") is not False:
            errors.append("C49 Score must not install the operator")
        if row.get("execution_status") != "NOT_APPLICABLE":
            errors.append("C49 must not execute on Score ranks")
        if row.get("eligible_nograd_calls", 0) != 0:
            errors.append("C49 unexpected Score splitKV calls")
        return errors
    if (row.get("name") != CONTRACT["name"] or row.get("contract") != CONTRACT
            or row.get("api_source_sha256") != EXPECTED_KVCACHE_SOURCE_SHA256
            or row.get("production_installed") is not True
            or row.get("execution_status") != "PASS"
            or row.get("contract_failure_count") != 0):
        errors.append("C49 actual runtime contract/source not proven")
    count = row.get("eligible_nograd_calls", 0)
    histogram = row.get("shape_histogram", {})
    if not isinstance(count, int) or count <= 0 or sum(histogram.values()) != count:
        errors.append("C49 splitKV actual call inventory missing")
    if row.get("reference_calls", 0) <= 0:
        errors.append("C49 original attention fallback/grad path missing")
    if total_calls is not None and count + row.get("reference_calls", 0) != total_calls:
        errors.append("C49 one-to-one attention call reconciliation failed")
    return errors
