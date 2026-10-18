"""Default-off C51 child: retain full autograd state for four Critic blocks.

This changes only checkpoint/recompute policy, not original application calls.
Unselected blocks still use the exact SHA-pinned FA3 SAC implementation.
"""
from __future__ import annotations

import copy
from typing import Any, Mapping

from distillschedule.runtime import livewan_replay_ablation as replay
from distillschedule.runtime.livewan_score_fa3 import (
    CriticFA3CheckpointProxy, runtime_receipt_errors as attention_errors,
)

VARIANT = "c53-c51-critic-tail4-retention-v1"
PLAN_ID = "livewan_wan13b_14b_sf_dmd_c53_c51_critic_tail4_retention_v1"
PLAN_FILE = PLAN_ID + ".json"
BLOCKS = (26, 27, 28, 29)
CONTRACT = {
    "schema": "livewan_c51_critic_block_retention_v1",
    "parent_variant": replay.PARENT_VARIANT,
    "parent_source_head": replay.PARENT_HEAD,
    "parent_plan_sources_sha256": replay.PARENT_PLAN_SOURCES_SHA256,
    "source_blockcausal_sha256": "bde44405af4f38ee05ed911c9ab0f626c41ca14c5ad95539f2a3ccf47ca4ce96",
    "factor": "critic_grad_block_autograd_retention",
    "blocks": list(BLOCKS),
    "other_blocks": "unchanged_pinned_fa3_selective_checkpoint",
    "cache_location": "cuda_native_autograd_saved_tensors",
    "release": "native_backward_graph_release_each_critic_update",
    "application_calls_unchanged": True,
    "optimizer_placement_and_schedule_unchanged": True,
    "preserve_rng_state": True,
    "cpu_offload": False,
    "scope": "conditional_on_full_c51_not_layout_only",
}


def validate_effective_policy(plan: Mapping[str, Any]) -> None:
    if (plan.get("id") != PLAN_ID
        or plan.get("metadata", {}).get("critic_block_retention") != CONTRACT):
        raise RuntimeError("C53 full-block retention identity/policy changed")


def parent_plan(plan: Mapping[str, Any], *, mode: str) -> dict:
    validate_effective_policy(plan)
    if mode != "pipeline":
        raise RuntimeError("C53 requires unchanged C51 pipeline role layout")
    # This view is validation-only: the actual worker retains the child policy.
    view = copy.deepcopy(dict(plan))
    view["id"] = replay.PLAN_ID
    view["metadata"].pop("critic_block_retention")
    view["metadata"]["checkpoint_replay_ablation"] = copy.deepcopy(replay.CONTRACT)
    view["metadata"]["critic_attention_checkpoint"] = copy.deepcopy(replay.RECOMPUTE_CONTRACT)
    return replay.parent_plan(view, mode=mode)


def enabled(plan: Mapping[str, Any], *, mode: str) -> bool:
    if "critic_block_retention" not in plan.get("metadata", {}):
        if plan.get("id") == PLAN_ID:
            raise RuntimeError("C53 retention contract missing")
        return False
    parent_plan(plan, mode=mode)
    return True


class CriticBlockRetentionProxy(CriticFA3CheckpointProxy):
    """Preserve the original block callable/hooks; retain only the selected tail."""

    def __init__(self, checkpoint_module: Any, torch: Any, flash_module: Any, *, blocks: Any):
        super().__init__(checkpoint_module, torch, flash_module)
        self.torch = torch
        self.block_ids = {id(block): i for i, block in enumerate(blocks)}
        if len(blocks) != 30 or len(self.block_ids) != 30:
            raise RuntimeError("C53 requires 30 distinct original Critic blocks")
        self.original_calls = [0] * 30
        self.direct_calls = [0] * 30
        self.backward_calls = [0] * 30
        self.failures = 0

    def checkpoint(self, function: Any, *args: Any, **kwargs: Any):
        if (set(kwargs) != {"use_reentrant"} or kwargs["use_reentrant"] is not False
            or not self.torch.is_grad_enabled() or not args
            or id(args[0]) not in self.block_ids):
            self.failures += 1
            raise RuntimeError("C53 original block/checkpoint/RNG call site changed")
        index = self.block_ids[id(args[0])]
        self.original_calls[index] += 1
        if index not in BLOCKS:
            return super().checkpoint(function, *args, **kwargs)
        self.direct_calls[index] += 1
        output = function(*args)  # args[0] is the original module, hooks intact.
        if not isinstance(output, self.torch.Tensor) or not output.requires_grad:
            self.failures += 1
            raise RuntimeError("C53 retained block lost its autograd chain")

        def backward_seen(gradient):
            self.backward_calls[index] += 1
            return None  # Observation only: never replace or modify the gradient.

        output.register_hook(backward_seen)
        return output

    def receipt(self, *, expected_calls: int):
        if expected_calls <= 0 or expected_calls % 30:
            raise RuntimeError("C53 original application block census invalid")
        checkpoint_calls = expected_calls // 30 * (30 - len(BLOCKS))
        result = super().receipt(expected_calls=checkpoint_calls)
        result["block_retention"] = {
            "contract": copy.deepcopy(CONTRACT),
            "production_installed": True,
            "original_block_calls": sum(self.original_calls),
            "direct_retained_calls": sum(self.direct_calls),
            "retained_backward_calls": sum(self.backward_calls),
            "original_calls_by_block": list(self.original_calls),
            "retained_calls_by_block": list(self.direct_calls),
            "backward_calls_by_block": list(self.backward_calls),
            "contract_failures": self.failures,
        }
        if runtime_receipt_errors(result, rank=8, expected_calls=expected_calls):
            raise RuntimeError(f"C53 actual block retention census failed: {result}")
        return result


def runtime_receipt_errors(r: dict, *, rank: int, expected_calls: int) -> list[str]:
    if rank < 8:
        return attention_errors(r, rank=rank, expected_calls=expected_calls)
    if expected_calls <= 0 or expected_calls % 30:
        return ["C53 original application census invalid"]
    forwards = expected_calls // 30
    errors = attention_errors(r, rank=rank, expected_calls=forwards * 26)
    retained = [forwards if i in BLOCKS else 0 for i in range(30)]
    required = {
        "contract": CONTRACT, "production_installed": True,
        "original_block_calls": expected_calls,
        "direct_retained_calls": forwards * len(BLOCKS),
        "retained_backward_calls": forwards * len(BLOCKS),
        "original_calls_by_block": [forwards] * 30,
        "retained_calls_by_block": retained,
        "backward_calls_by_block": retained,
        "contract_failures": 0,
    }
    value = r.get("block_retention", {})
    if any(value.get(k) != v for k, v in required.items()):
        errors.append("C53 retained block forward/backward/source census changed")
    return errors
