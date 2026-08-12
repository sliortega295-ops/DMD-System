#!/usr/bin/env python3
"""Inspect the static 4G+8C+4Flex GAN role-phase schedule.

The default v2 contract uses the PR6 persistent-pinned phase transition.  The
old canonical GPU-reshard planner remains available only as a diagnostic via
``--show-gpu-reshard``; it is not the first real-execution backend.
"""
from __future__ import annotations

import argparse
import json

from distillschedule.runtime.gan_rolephase import (
    GanRolePhaseTopology,
    plan_contiguous_reshard,
    rolephase_contract_summary,
)
from distillschedule.runtime.gan_rolephase_critic import (
    CriticReductionContract,
    build_critic_local_batches,
)
from distillschedule.runtime.gan_rolephase_transition import (
    transition_contract_summary,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--show-gpu-reshard", action="store_true")
    p.add_argument("--generator-numel", type=int, default=0)
    p.add_argument("--critic-numel", type=int, default=0)
    p.add_argument("--chunk-mib", type=int, default=64)
    return p


def _plan_summary(total_numel: int, src: tuple[int, ...], dst: tuple[int, ...], chunk_mib: int):
    max_chunk_numel = chunk_mib * 1024 * 1024 // 2
    plan = plan_contiguous_reshard(
        total_numel,
        src,
        dst,
        max_chunk_numel=max_chunk_numel,
    )
    cross = sum(item.numel for item in plan if item.src_rank != item.dst_rank)
    return {
        "diagnostic_only": True,
        "total_numel": total_numel,
        "src_ranks": list(src),
        "dst_ranks": list(dst),
        "transfer_count": len(plan),
        "cross_rank_numel": cross,
        "cross_rank_fraction": cross / total_numel,
    }


def main() -> int:
    args = parser().parse_args()
    if args.chunk_mib <= 0:
        raise ValueError("--chunk-mib must be positive")
    topology = GanRolePhaseTopology()
    result = rolephase_contract_summary(topology)
    result["transition"] = transition_contract_summary()
    reduction = CriticReductionContract()
    result["phase_b_critic_local_batches"] = [
        {
            "rank": item.global_rank,
            "logical_slots": list(item.logical_slots),
            "local_batch": item.local_batch,
        }
        for item in build_critic_local_batches(topology)
    ]
    result["phase_b_reduction"] = {
        "dp_world_size": reduction.dp_world_size,
        "global_batch": reduction.global_batch,
        "physical_model_calls_per_rank": reduction.physical_model_calls_per_rank,
        "sum_backend_local_sum_scale": reduction.local_sum_scale("SUM"),
        "dp_mean_backend_local_sum_scale": reduction.local_sum_scale("DP_MEAN"),
        "backend_semantics_must_be_explicit": True,
    }
    result["phase_a_slot_mapping"] = [
        vars(topology.phase_a_slot(slot)) for slot in range(16)
    ]
    result["phase_b_slot_mapping"] = [
        vars(topology.phase_b_slot(slot)) for slot in range(16)
    ]

    if args.show_gpu_reshard:
        result["gpu_reshard"] = {
            "status": "DIAGNOSTIC_ONLY_NOT_DEFAULT_V2_HANDOFF",
        }
        if args.generator_numel:
            result["gpu_reshard"]["generator"] = _plan_summary(
                args.generator_numel,
                topology.phase_a_generator_ranks,
                topology.phase_b_generator_ranks,
                args.chunk_mib,
            )
        if args.critic_numel:
            result["gpu_reshard"]["critic"] = _plan_summary(
                args.critic_numel,
                topology.phase_a_score_ranks,
                topology.phase_b_critic_ranks,
                args.chunk_mib,
            )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
