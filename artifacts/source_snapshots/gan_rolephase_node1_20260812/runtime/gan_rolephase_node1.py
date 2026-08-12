"""Fail-closed node-1 binding for the GAN 8G/8Score -> 4G/12C runtime.

This module contains no model math.  It binds node 1's eight physical GPUs to
the phase-local logical topology and freezes the only DeepSpeed contract used
there: Phase A is a read-only resident Score service; Phase B is DP12/B1/GAS1
ZeRO-1 with CPU optimizer state.  Keeping this small contract separate makes
it possible to validate rank/sample/batch semantics before constructing 14B
models.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


WORLD_SIZE = 16
NODE1_GLOBAL_RANKS = tuple(range(8, 16))
PHASE_A_SCORE_RANKS = tuple(range(8, 16))
PHASE_B_CRITIC_RANKS = tuple(range(4, 16))


@dataclass(frozen=True)
class Node1RolePhaseBinding:
    global_rank: int
    local_rank: int
    phase_a_lane: int
    phase_a_logical_slots: tuple[int, int]
    phase_b_logical_slot: int
    phase_b_generator_rank: int
    phase_b_critic_rank: int

    def __post_init__(self) -> None:
        if self.global_rank not in NODE1_GLOBAL_RANKS:
            raise ValueError("node1 worker is restricted to global ranks 8..15")
        if not 0 <= self.local_rank < 8:
            raise ValueError("node1 local rank must be in [0,8)")
        if self.global_rank != self.local_rank + 8:
            raise ValueError("node1 global/local rank mapping drift")
        lane = self.global_rank - 8
        if self.phase_a_lane != lane:
            raise ValueError("Phase-A Score lane drift")
        if self.phase_a_logical_slots != (lane, lane + 8):
            raise ValueError("Phase-A batch16 two-wave slot mapping drift")
        slot = self.global_rank - 4
        if self.phase_b_logical_slot != slot:
            raise ValueError("Phase-B node1 slot mapping drift")
        if self.phase_b_generator_rank != slot % 4:
            raise ValueError("Phase-B 4G producer mapping drift")
        if self.phase_b_critic_rank != self.global_rank:
            raise ValueError("Phase-B Critic destination drift")

    def as_dict(self) -> dict[str, Any]:
        return {
            "global_rank": self.global_rank,
            "local_rank": self.local_rank,
            "phase_a": {
                "role": "score_gradient_service",
                "lane": self.phase_a_lane,
                "logical_slots": list(self.phase_a_logical_slots),
                "gradient_service_calls": 2,
                "expected_score_forwards_per_sample": 4,
                "expected_score_forwards": 8,
                "optimizer_commits": 0,
            },
            "phase_b": {
                "role": "critic",
                "logical_slot": self.phase_b_logical_slot,
                "generator_rank": self.phase_b_generator_rank,
                "critic_rank": self.phase_b_critic_rank,
                "local_batch": 1,
                "critic_loss_calls_per_step": 1,
                "expected_score_forwards_per_step": 2,
                "optimizer_commits_per_step": 1,
            },
        }


def bind_node1_rank(
    *,
    global_rank: int,
    local_rank: int,
    world_size: int = WORLD_SIZE,
) -> Node1RolePhaseBinding:
    if int(world_size) != WORLD_SIZE:
        raise ValueError(f"GAN role-phase worker requires world16, got {world_size}")
    global_rank = int(global_rank)
    local_rank = int(local_rank)
    lane = global_rank - 8
    slot = global_rank - 4
    return Node1RolePhaseBinding(
        global_rank=global_rank,
        local_rank=local_rank,
        phase_a_lane=lane,
        phase_a_logical_slots=(lane, lane + 8),
        phase_b_logical_slot=slot,
        phase_b_generator_rank=slot % 4,
        phase_b_critic_rank=global_rank,
    )


def phase_b_zero1_config() -> dict[str, Any]:
    """Return the frozen PR6-style DP12/B1/GAS1 Student config."""

    return {
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1,
        "train_batch_size": 12,
        "bf16": {"enabled": True},
        "fp16": {"enabled": False},
        "communication_data_type": "fp32",
        "gradient_clipping": 1.0,
        "zero_allow_untested_optimizer": True,
        "zero_force_ds_cpu_optimizer": False,
        "zero_optimization": {
            "stage": 1,
            "offload_optimizer": {"device": "cpu", "pin_memory": False},
            "overlap_comm": False,
            "contiguous_gradients": False,
            "reduce_bucket_size": 64 * 1024 * 1024,
            "allgather_bucket_size": 64 * 1024 * 1024,
        },
    }


def _resolved_int(engine: Any, name: str) -> int:
    value = getattr(engine, name)
    return int(value() if callable(value) else value)


def assert_phase_b_zero1_contract(engine: Any, *, expected_dp_group: Any) -> dict[str, int]:
    resolved = {
        "dp_world_size": _resolved_int(engine, "dp_world_size"),
        "train_micro_batch_size_per_gpu": _resolved_int(
            engine, "train_micro_batch_size_per_gpu"
        ),
        "gradient_accumulation_steps": _resolved_int(
            engine, "gradient_accumulation_steps"
        ),
        "train_batch_size": _resolved_int(engine, "train_batch_size"),
    }
    expected = {
        "dp_world_size": 12,
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1,
        "train_batch_size": 12,
    }
    if resolved != expected:
        raise AssertionError(f"Phase-B batch contract drift: {resolved} != {expected}")
    if int(engine.zero_optimization_stage()) != 1:
        raise AssertionError("Phase B requires ZeRO-1")
    if bool(getattr(engine.optimizer, "partition_gradients", False)):
        raise AssertionError("Phase B accidentally enabled ZeRO-2")
    if getattr(engine.optimizer, "dp_process_group", None) is not expected_dp_group:
        raise AssertionError("Phase-B optimizer escaped ranks 4..15")
    if float(getattr(engine.optimizer, "clip_grad", -1.0)) != 1.0:
        raise AssertionError("official gradient clipping changed")
    return resolved


def assert_topology_parity(binding: Node1RolePhaseBinding, topology: Any) -> None:
    """Cross-check the local binding against PR #8's canonical topology."""

    a = tuple(
        topology.phase_a_slot(slot)
        for slot in binding.phase_a_logical_slots
    )
    if any(item.score_rank != binding.global_rank for item in a):
        raise AssertionError("node1 Phase-A mapping differs from PR8 topology")
    b = topology.phase_b_slot(binding.phase_b_logical_slot)
    observed = (b.generator_rank, b.critic_rank, b.critic_micro_wave)
    expected = (
        binding.phase_b_generator_rank,
        binding.phase_b_critic_rank,
        0,
    )
    if observed != expected:
        raise AssertionError(f"node1 Phase-B mapping drift: {observed} != {expected}")


__all__ = [
    "NODE1_GLOBAL_RANKS",
    "Node1RolePhaseBinding",
    "assert_phase_b_zero1_contract",
    "assert_topology_parity",
    "bind_node1_rank",
    "phase_b_zero1_config",
]
