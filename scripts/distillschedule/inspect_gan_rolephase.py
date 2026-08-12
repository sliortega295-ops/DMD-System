#!/usr/bin/env python3
"""Inspect the static 4G+8C+4Flex GAN role-phase schedule.

This is a CPU-only contract tool. It does not launch Helios or claim GPU
performance. It is intended to preregister topology, batch mapping and the
bounded GPU-shard transfer graph before wiring the real executor.
"""
from __future__ import annotations

import argparse
import json

from distillschedule.runtime.gan_rolephase import (
    GanRolePhaseTopology,
    plan_contiguous_reshard,
    rolephase_contract_summary,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--generator-numel", type=int, default=0)
    p.add_argument("--critic-numel", type=int, default=0)
    p.add_argument("--chunk-mib", type=int, default=64)
    return p


def _plan_summary(total_numel: int, src: tuple[int, ...], dst: tuple[int, ...], chunk_mib: int):
    # BF16 frozen backbones are the dominant payload; convert MiB to BF16 numel.
    max_chunk_numel = chunk_mib * 1024 * 1024 // 2
    plan = plan_contiguous_reshard(
        total_numel,
        src,
        dst,
        max_chunk_numel=max_chunk_numel,
    )
    return {
        "total_numel": total_numel,
        "total_bf16_bytes": total_numel * 2,
        "src_ranks": list(src),
        "dst_ranks": list(dst),
        "chunk_mib": chunk_mib,
        "transfer_count": len(plan),
        "transferred_numel": sum(item.numel for item in plan),
        "cross_rank_numel": sum(
            item.numel for item in plan if item.src_rank != item.dst_rank
        ),
    }


def main() -> int:
    args = parser().parse_args()
    if args.chunk_mib <= 0:
        raise ValueError("--chunk-mib must be positive")
    topology = GanRolePhaseTopology()
    result = rolephase_contract_summary(topology)
    result["phase_a_slot_mapping"] = [
        vars(topology.phase_a_slot(slot)) for slot in range(16)
    ]
    result["phase_b_slot_mapping"] = [
        vars(topology.phase_b_slot(slot)) for slot in range(16)
    ]
    if args.generator_numel:
        result["generator_handoff"] = {
            "to_phase_b": _plan_summary(
                args.generator_numel,
                topology.phase_a_generator_ranks,
                topology.phase_b_generator_ranks,
                args.chunk_mib,
            ),
            "to_phase_a": _plan_summary(
                args.generator_numel,
                topology.phase_b_generator_ranks,
                topology.phase_a_generator_ranks,
                args.chunk_mib,
            ),
        }
    if args.critic_numel:
        result["critic_handoff"] = {
            "to_phase_b": _plan_summary(
                args.critic_numel,
                topology.phase_a_score_ranks,
                topology.phase_b_critic_ranks,
                args.chunk_mib,
            ),
            "to_phase_a": _plan_summary(
                args.critic_numel,
                topology.phase_b_critic_ranks,
                topology.phase_a_score_ranks,
                args.chunk_mib,
            ),
        }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
