"""Finite executable recipes, separate from the locked historical winners.

Names describe mechanisms, not a model-name -> winner lookup. Scheduling and
kernel dependencies are explicit: unsafe cross-products are never generated.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from distillschedule.workloads.best_known.registry import (
    CANONICAL_IDS, fingerprint, load_profile, repository_root,
)
from .contracts import Candidate, PILOTS, require, semantic_contract

WAN, LIVEWAN = PILOTS
LIVEWAN_VARIANTS = (
    ("c24", "score-modulation", "score_modulation", ("pointwise",)),
    ("c25", "score-rmsnorm", "score_rmsnorm", ("pointwise",)),
    ("c26", "critic-attention-checkpoint", "critic_attention_checkpoint", ("compute",)),
    ("c27", "pinned-prompt-staging", "pinned_prompt_staging", ("h2d", "host_gap")),
    ("c28", "rollout-cast-cache", "rollout_cast_cache", ("pointwise",)),
    ("c29", "generator-rms-epilogue", "generator_rms_epilogue", ("pointwise",)),
    ("c30", "generator-rms-critic-checkpoint", "critic_fa2_gpu_checkpoint", ("compute",)),
    ("c31", "generator-rms-cast-square", "rms_cast_square", ("pointwise",)),
    ("c32", "generator-modulation", "generator_modulation", ("pointwise",)),
    ("c33", "generator-gate", "generator_gate", ("pointwise",)),
    ("c34", "residual-layout", "row_major_residual", ("pointwise", "compute")),
)


def candidates(workload: str) -> tuple[Candidate, ...]:
    require(workload in CANONICAL_IDS, f"unknown workload: {workload}")
    if workload == WAN:
        rows = [Candidate("fsdp-reference", WAN, "world16-fsdp", targets=("nccl",))]
        definitions = (
            ("resident", "resident", "fsdp-reference", "replicate_model", ("nccl",), ()),
            ("resident-clean", "resident-clean", "resident", "host_path_cleanup", ("host_gap",), ()),
            ("resident-async128", "resident-async128", "resident-clean", "async_gradient_reduction", ("nccl",), ()),
            ("resident-tree", "resident-async128-tree", "fsdp-reference", "resident_async_tree_recipe", ("nccl",), ()),
            ("resident-chunk2048", "resident-async128-tree-chunk2048", "resident-tree", "modulation_chunk", ("pointwise", "host_gap"), ("modulation_chunk2048",)),
            ("resident-lora-epilogue", "resident-async128-tree-chunk2048", "resident-chunk2048", "lora_epilogue", ("pointwise",), ("modulation_chunk2048", "lora_nograd_epilogue")),
            ("resident-local-fast", "resident-local-fast", "resident-clean", "fused_local_state", ("optimizer",), ("fused_adam_foreach_ema",)),
            ("resident-flat256", "resident-flat256", "resident-local-fast", "flat_gradient_256", ("nccl", "host_gap"), ("fused_adam_foreach_ema",)),
            ("resident-flat512", "resident-flat512", "resident-flat256", "flat_gradient_512", ("nccl",), ("fused_adam_foreach_ema",)),
            ("resident-async256", "resident-async256", "resident-async128", "gradient_bucket_256", ("nccl",), ()),
        )
        from .wan_contract import SEARCH_IDS
        rows += [Candidate(i, WAN, placement, kernels, targets, parent, mechanism,
                           executable=i in SEARCH_IDS)
                 for i, placement, parent, mechanism, targets, kernels in definitions]
        return tuple(rows)
    if workload == LIVEWAN:
        rows = [Candidate("fsdp-reference", LIVEWAN, "world16-fsdp", variant="b1")]
        parents = {"c24": "fsdp-reference", "c25": "c24", "c26": "c25", "c27": "c25", "c28": "c25",
                   "c29": "c28", "c30": "c29", "c31": "c30", "c32": "c31", "c33": "c32", "c34": "c33"}
        for number, suffix, kernel, targets in LIVEWAN_VARIANTS:
            variant = f"{number}-continuous-b16-{suffix}"
            from distillschedule.core.config import _load_with_includes
            path = repository_root() / "configs/distillschedule/plans" / (
                "livewan_wan13b_14b_sf_dmd_" + variant.replace("-", "_") + ".json")
            raw = _load_with_includes(path, stack=(), sources=[])
            kernels = tuple(sorted(v["name"] for v in raw.get("metadata", {}).values()
                                   if isinstance(v, dict) and v.get("enabled") and "name" in v))
            rows.append(Candidate(number, LIVEWAN, "8g8score", kernels, targets,
                                  parents[number], "pipeline_and_registered_dependencies" if number == "c24" else kernel,
                                  variant))
        return tuple(rows)
    profile = load_profile(workload)
    return (Candidate("canonical-reference", workload, profile["topology"].get("placement", "world16-fsdp"),
                      executable=False, mechanism="REGISTERED_CPU_ONLY"),)


def get_candidate(workload: str, candidate_id: str) -> Candidate:
    found = [c for c in candidates(workload) if c.id == candidate_id]
    require(len(found) == 1, f"unregistered candidate: {workload}/{candidate_id}")
    return found[0]


def catalog() -> list[dict[str, Any]]:
    return [{"workload": w, "gpu_search_status": "NOT_RUN", "pilot": w in PILOTS,
             "applicability": {"world_size": 16, "global_batch": 16,
                 "model_and_dtype": "FROZEN_CANONICAL_CONTRACT", "kernel_backend": "REGISTERED_NATIVE_IMPLEMENTATION",
                 "unknown_shape_device_or_backend": "REJECT_BEFORE_LAUNCH"},
             "candidates": [c.to_dict() for c in candidates(w)]} for w in sorted(CANONICAL_IDS)]


def candidate_profile(candidate: Candidate, *, root: Path | None = None) -> dict[str, Any]:
    """Compile only a registered recipe. No arbitrary env override API."""
    root = root or repository_root()
    require(candidate == get_candidate(candidate.workload, candidate.id), "candidate payload was altered")
    p = copy.deepcopy(load_profile(candidate.workload, root=root))
    original_semantics = semantic_contract(p)
    p["display_name"] = f"Search recipe: {candidate.workload}/{candidate.id}"
    runtime, env = p["runtime"], p["runtime"]["environment"]
    if candidate.workload == WAN:
        p["topology"]["placement"] = candidate.placement
        if candidate.id == "fsdp-reference":
            p["topology"].update(model_replication="world16_full_shard",
                                 gradient_sync="reduce_scatter_fp32_sum_postdivide16")
        env["NCCL_ALGO"] = "Tree" if "tree" in candidate.placement else ""
        env["DISTILLSCHEDULE_WAN13B_GENERATOR_NOGRAD_LORA_FUSED_EPILOGUE"] = str(int("lora_nograd_epilogue" in candidate.kernels))
        env["DISTILLSCHEDULE_WAN13B_GENERATOR_NOGRAD_LORA_CAST_CACHE"] = "0"
        runtime["arguments"][-1] = candidate.placement
        env.update(DISTILLSCHEDULE_SEARCH_CANDIDATE_ID=candidate.id,
                   DISTILLSCHEDULE_TRAJECTORY_CYCLES="2",
                   DISTILLSCHEDULE_TRAJECTORY_SEED="20260903",
                   DISTILLSCHEDULE_TWO_CYCLE_SEED="20260903",
                   DISTILLSCHEDULE_TRAJECTORY_ARM_ID="search-" + candidate.id,
                   DISTILLSCHEDULE_TWO_CYCLE_ARM_ID="search-" + candidate.id)
        runtime["native_run_modes"] = {"correctness": "trajectory",
                                      "diagnostic-timing": "two-cycle-timing"}
        runtime["search_two_cycle_ready"] = candidate.executable
        p["contract"]["continuous_cycles"] = 2
        p["contract"]["final_versions"] = {"generator": 4, "critic": 20, "ema": 20}
    elif candidate.workload == LIVEWAN:
        is_baseline = candidate.id == "fsdp-reference"
        env["LIVEWAN_PIPELINE_VARIANT"] = candidate.variant
        env["LIVEWAN_TWO_CYCLE_ARM_ID"] = "search-" + candidate.id
        env["LIVEWAN_TRAJECTORY_ARM_ID"] = "search-" + candidate.id
        runtime["arguments"][3] = "baseline" if is_baseline else "pipeline"
        p["contract"]["variant"] = candidate.variant
        p["topology"]["placement"] = candidate.placement
        if is_baseline:
            p["topology"].update(generator_ranks=list(range(16)), score_ranks=list(range(16)),
                                 teacher_ranks=list(range(16)), student_critic="fsdp2")
        if is_baseline:
            template = "livewan_wan13b_14b_sf_dmd_g1c5_world16_fsdp2.json"
        else:
            template = "livewan_wan13b_14b_sf_dmd_" + candidate.variant.replace("-", "_") + ".json"
        runtime["typed_plan"] = "configs/distillschedule/plans/" + template
        from distillschedule.core.config import _load_with_includes
        raw = _load_with_includes(root / runtime["typed_plan"], stack=(), sources=[])
        runtime["typed_plan_template_sha256"] = fingerprint(raw)
    require(semantic_contract(p) == original_semantics, "candidate changed frozen workload semantics")
    return p


def search_identity(workload: str, *, source_head: str, assets: dict[str, Any],
                    hardware: dict[str, Any]) -> dict[str, str]:
    require(hardware.get("world_size") == 16 and hardware.get("nodes") == 2,
            "v1 pilots require the same 2-node/world16 allocation")
    return {"source_head": source_head, "semantic_fingerprint": fingerprint(semantic_contract(load_profile(workload))),
            "asset_fingerprint": fingerprint(assets), "hardware_fingerprint": fingerprint(hardware)}
