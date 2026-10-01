"""C44 opt-in exact GPU-local frozen FFN-down checkpoint-output cache.

This adds one MUST_SAVE operation to the existing C38 attention SAC context.
It never caches LoRA matmuls, frozen FFN-up, Teacher or no-grad Generator work.
Each checkpoint owns its cache; no value reads, synchronization or CPU offload.
"""
from __future__ import annotations

from typing import Any, Mapping

from distillschedule.runtime.livewan_attention_checkpoint import AttentionCheckpointContexts
from distillschedule.runtime.livewan_score_fa3 import (
    BACKEND_CONTRACT, SAC_CONTRACT, CriticFA3CheckpointProxy, attention_metadata,
    discover_flash_forward, runtime_receipt_errors as attention_receipt_errors,
)

VARIANT = "c44-continuous-b16-critic-ffn-down-cache"
CONTRACT = {
    "enabled": True,
    "name": "critic_frozen_ffn_down_exact_gpu_checkpoint_output_v1",
    "scope": "score_rank_critic_grad_only",
    "operator": "aten.addmm.default",
    "source_blockcausal_sha256": "bde44405af4f38ee05ed911c9ab0f626c41ca14c5ad95539f2a3ccf47ca4ce96",
    "cache_location": "cuda_per_checkpoint",
    "bias_shape": [1536],
    "input_shape": [38640, 8960],
    "transposed_weight_shape": [8960, 1536],
    "dtype": "bfloat16",
    "alpha": 1,
    "beta": 1,
    "frozen_base_only": True,
    "allow_cache_entry_mutation": False,
    "preserve_rng_state": True,
    "application_calls_unchanged": True,
    "lora_and_backward_unchanged": True,
    "cpu_offload": False,
}


def enabled(plan: Mapping[str, Any], *, mode: str) -> bool:
    metadata = plan.get("metadata", {})
    value = metadata.get("critic_ffn_down_checkpoint")
    if value is None:
        return False
    if (not isinstance(value, Mapping) or dict(value) != CONTRACT
        or mode != "pipeline"
        or metadata.get("score_attention_backend") != BACKEND_CONTRACT
        or metadata.get("critic_attention_checkpoint") != SAC_CONTRACT
        or metadata.get("critic_step1_overlap") is not None
        or metadata.get("critic_physical_batch_size") != 2
        or metadata.get("critic_lora_kernel", {}).get("name")
           != "scale_one_autocast_bf16_input_elision_v1"):
        raise RuntimeError("C44 frozen FFN-down cache/C38 parent contract changed")
    return True


def ffn_down_metadata(torch: Any, args: tuple, kwargs: dict) -> tuple | None:
    """Identify the observed frozen addmm using only host tensor metadata."""
    if len(args) != 3:
        raise RuntimeError("C44 addmm schema changed")
    bias, operand, weight = args
    if (tuple(operand.shape) != tuple(CONTRACT["input_shape"])
        or tuple(weight.shape) != tuple(CONTRACT["transposed_weight_shape"])):
        return None
    if (set(kwargs) - {"alpha", "beta"} or kwargs.get("alpha", 1) != 1
        or kwargs.get("beta", 1) != 1
        or tuple(bias.shape) != tuple(CONTRACT["bias_shape"])
        or bias.requires_grad or weight.requires_grad
        or any(not t.is_cuda or t.dtype != torch.bfloat16 for t in args)
        or any(t.device != operand.device for t in args)):
        raise RuntimeError("C44 frozen FFN-down device/dtype/weight/math changed")
    return tuple(
        (tuple(t.shape), tuple(t.stride()), str(t.dtype), str(t.device),
         # Frozen base storage/version must survive recompute unchanged. The
         # activation storage is naturally regenerated and is not compared.
         int(t.data_ptr()) if i != 1 else None,
         int(t._version) if i != 1 else None)
        for i, t in enumerate(args)
    )


class FrozenFFNDownPolicies:
    def __init__(self, torch: Any):
        self.torch = torch
        self.checkpoints: list[dict[str, Any]] = []

    def __call__(self):
        from torch.utils.checkpoint import CheckpointPolicy
        ledger = {"original": [], "replay_calls": 0, "failures": 0,
                  "cached_output_bytes": 0}
        self.checkpoints.append(ledger)

        def policy(ctx: Any, op: Any, *args: Any, **kwargs: Any):
            if op != self.torch.ops.aten.addmm.default:
                return CheckpointPolicy.PREFER_RECOMPUTE
            try:
                metadata = ffn_down_metadata(self.torch, args, kwargs)
                if metadata is None:
                    return CheckpointPolicy.PREFER_RECOMPUTE
                if ctx.is_recompute:
                    i = ledger["replay_calls"]
                    if i >= len(ledger["original"]) or metadata != ledger["original"][i]:
                        raise RuntimeError("C44 FFN-down replay operation identity changed")
                    ledger["replay_calls"] += 1
                else:
                    ledger["original"].append(metadata)
                    bias, operand, weight = args
                    ledger["cached_output_bytes"] += (
                        operand.shape[0] * weight.shape[1] * bias.element_size()
                    )
            except Exception:
                ledger["failures"] += 1
                raise
            return CheckpointPolicy.MUST_SAVE
        return policy

    def receipt(self, *, expected_calls: int) -> dict[str, Any]:
        return {
            "name": CONTRACT["name"], "contract": dict(CONTRACT),
            "enabled": True, "production_installed": True,
            "execution_status": "PASS", "checkpoint_calls": len(self.checkpoints),
            "expected_checkpoint_calls": expected_calls,
            "original_ffn_down_calls": sum(len(r["original"]) for r in self.checkpoints),
            "replay_ffn_down_cache_reads": sum(r["replay_calls"] for r in self.checkpoints),
            "contract_failures": sum(r["failures"] for r in self.checkpoints),
            "all_checkpoint_ledgers_one_to_one": all(
                len(r["original"]) == r["replay_calls"] == 1 for r in self.checkpoints),
            "cumulative_cached_output_bytes": sum(r["cached_output_bytes"] for r in self.checkpoints),
            "bytes_scope": "CUMULATIVE_ORIGINAL_OUTPUTS_NOT_PEAK_LIVE_MEMORY",
            "cache_location": "cuda_per_checkpoint",
            "cpu_offload": False, "allow_cache_entry_mutation": False,
        }


class CriticFFNDownCheckpointProxy(CriticFA3CheckpointProxy):
    def __init__(self, checkpoint_module: Any, torch: Any, flash_module: Any):
        self.module = checkpoint_module
        op, identity = discover_flash_forward(flash_module)
        self.ffn_policies = FrozenFFNDownPolicies(torch)
        self.contexts = AttentionCheckpointContexts(
            torch, op, identity, self_attention_only=True,
            metadata_reader=attention_metadata, receipt_name=SAC_CONTRACT["name"],
            additional_policy_factory=self.ffn_policies)
        self.receipt_errors = attention_receipt_errors

    def ffn_down_receipt(self, *, expected_calls: int) -> dict[str, Any]:
        r = self.ffn_policies.receipt(expected_calls=expected_calls)
        if runtime_receipt_errors(r, rank=8, expected_calls=expected_calls):
            raise RuntimeError(f"C44 FFN-down replay census failed: {r}")
        return r


def runtime_receipt_errors(r: dict[str, Any], *, rank: int, expected_calls: int) -> list[str]:
    if rank < 8:
        return [] if (r.get("enabled") is True
            and r.get("execution_status") == "NOT_APPLICABLE"
            and r.get("checkpoint_calls") == 0) else ["C44 Generator cache changed"]
    if (expected_calls <= 0 or r.get("name") != CONTRACT["name"]
        or r.get("contract") != CONTRACT
        or r.get("execution_status") != "PASS"
        or r.get("production_installed") is not True
        or r.get("contract_failures") != 0
        or r.get("checkpoint_calls") != expected_calls
        or r.get("expected_checkpoint_calls") != expected_calls
        or r.get("original_ffn_down_calls") != expected_calls
        or r.get("replay_ffn_down_cache_reads") != expected_calls
        or r.get("all_checkpoint_ledgers_one_to_one") is not True
        or r.get("cumulative_cached_output_bytes") != expected_calls * 38640 * 1536 * 2
        or r.get("cache_location") != "cuda_per_checkpoint"
        or r.get("allow_cache_entry_mutation") is not False
        or r.get("cpu_offload") is not False):
        return ["C44 frozen FFN-down source/replay/census contract failed"]
    return []
