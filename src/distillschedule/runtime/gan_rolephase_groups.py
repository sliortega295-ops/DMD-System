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


__all__ = ["GanRolePhaseGroups", "create_rolephase_groups"]
