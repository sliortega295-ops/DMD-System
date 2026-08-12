"""Pre-create all world16 process groups used by the GAN role-phase runtime.

Group creation is intentionally outside the formal Full5 timing window. The
runtime switches active groups at the phase boundary but never creates or
tears down NCCL groups during the timed handoff.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch.distributed as dist

from .gan_rolephase import GanRolePhaseTopology


@dataclass(frozen=True)
class GanRolePhaseGroups:
    world: Any
    phase_a_generator: Any
    phase_a_score: Any
    phase_b_generator: Any
    phase_b_critic: Any
    flex: Any


@dataclass(frozen=True)
class GanRolePhaseExecutionGroups:
    """Complete process-group inventory needed by the real worker.

    ``GanRolePhaseGroups`` remains the small phase-topology contract.  The
    execution worker additionally needs eight Phase-A G/Score point-to-point
    groups and one singleton model-parallel group per global rank.  They are
    all created before model construction and in an identical order on every
    WORLD rank; no process group is created at the timed phase boundary.
    """

    core: GanRolePhaseGroups
    phase_a_pairs: tuple[Any, ...]
    singletons: tuple[Any, ...]

    def phase_a_pair_for_rank(
        self,
        rank: int,
        topology: GanRolePhaseTopology = GanRolePhaseTopology(),
    ) -> Any:
        rank = int(rank)
        if rank in topology.phase_a_generator_ranks:
            lane = topology.phase_a_generator_ranks.index(rank)
        elif rank in topology.phase_a_score_ranks:
            lane = topology.phase_a_score_ranks.index(rank)
        else:  # pragma: no cover - topology covers world16, kept fail-closed.
            raise ValueError(f"rank {rank} is outside Phase A")
        return self.phase_a_pairs[lane]

    def singleton_for_rank(self, rank: int) -> Any:
        rank = int(rank)
        if not 0 <= rank < len(self.singletons):
            raise ValueError(f"rank {rank} is outside singleton inventory")
        return self.singletons[rank]


def create_rolephase_groups(
    topology: GanRolePhaseTopology = GanRolePhaseTopology(),
    *,
    backend: str = "nccl",
) -> GanRolePhaseGroups:
    """Create groups in one deterministic order on every WORLD rank."""
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized")
    if dist.get_world_size() != topology.world_size:
        raise RuntimeError(
            f"role-phase runtime requires world16, got {dist.get_world_size()}"
        )

    # Every rank calls new_group for every entry in exactly this order. This is
    # required by torch.distributed and avoids phase-time group creation stalls.
    a_g = dist.new_group(list(topology.phase_a_generator_ranks), backend=backend)
    a_s = dist.new_group(list(topology.phase_a_score_ranks), backend=backend)
    b_g = dist.new_group(list(topology.phase_b_generator_ranks), backend=backend)
    b_c = dist.new_group(list(topology.phase_b_critic_ranks), backend=backend)
    flex = dist.new_group(list(topology.flex_ranks), backend=backend)
    return GanRolePhaseGroups(
        world=dist.group.WORLD,
        phase_a_generator=a_g,
        phase_a_score=a_s,
        phase_b_generator=b_g,
        phase_b_critic=b_c,
        flex=flex,
    )


def create_rolephase_execution_groups(
    topology: GanRolePhaseTopology = GanRolePhaseTopology(),
    *,
    backend: str = "nccl",
) -> GanRolePhaseExecutionGroups:
    """Pre-create phase, Phase-A pair, and singleton groups deterministically."""

    core = create_rolephase_groups(topology, backend=backend)
    pairs = tuple(
        dist.new_group(
            [generator_rank, score_rank],
            backend=backend,
        )
        for generator_rank, score_rank in zip(
            topology.phase_a_generator_ranks,
            topology.phase_a_score_ranks,
            strict=True,
        )
    )
    singletons = tuple(
        dist.new_group([rank], backend=backend)
        for rank in range(topology.world_size)
    )
    if len(pairs) != 8 or len(singletons) != topology.world_size:
        raise AssertionError("role-phase execution-group inventory drift")
    return GanRolePhaseExecutionGroups(
        core=core,
        phase_a_pairs=pairs,
        singletons=singletons,
    )


__all__ = [
    "GanRolePhaseExecutionGroups",
    "GanRolePhaseGroups",
    "create_rolephase_execution_groups",
    "create_rolephase_groups",
]
