"""Bounded role-phase transition contract.

The first real GAN role-phase executor intentionally reuses the transition
substrate already validated by the no-GAN strict-v4 runtime instead of making
GPU live-reshard a prerequisite.  Future target modules are constructed on CPU
before formal timing, optionally registered/pinned there, and materialized once
at each role transition.  Repeated frozen-block CPU streaming inside a stable
phase remains forbidden.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RolePhaseTransitionContract:
    backend: str = "pr6_persistent_pinned_target_materialization"
    target_preconstructed_before_timing: bool = True
    target_host_state_pinned_before_timing: bool = True
    steady_state_repeated_cpu_streaming: bool = False
    forward_handoff_timed: bool = True
    reverse_handoff_timed: bool = True
    transition_once_per_direction_per_cycle: bool = True
    preferred_chunk_mib: int = 256
    preferred_staging_slots: int = 2

    def __post_init__(self) -> None:
        if self.backend not in {
            "pr6_persistent_pinned_target_materialization",
            "pr6_bounded_pinned_staging",
        }:
            raise ValueError("role-phase v1 requires the proven PR6 transition family")
        if not self.target_preconstructed_before_timing:
            raise ValueError("target model construction may not enter formal timing")
        if self.steady_state_repeated_cpu_streaming:
            raise ValueError("stable phases may not repeatedly stream frozen parameters")
        if not (self.forward_handoff_timed and self.reverse_handoff_timed):
            raise ValueError("both role transitions belong to steady-state accounting")
        if not self.transition_once_per_direction_per_cycle:
            raise ValueError("handoff must remain phase-bounded")
        if self.preferred_chunk_mib <= 0 or self.preferred_staging_slots <= 0:
            raise ValueError("invalid bounded transition staging geometry")

    @property
    def uses_gpu_live_reshard(self) -> bool:
        return False

    @property
    def requires_locality_aware_reshard_before_first_gpu_run(self) -> bool:
        return False


def transition_contract_summary() -> dict[str, object]:
    value = RolePhaseTransitionContract()
    return {
        "backend": value.backend,
        "target_preconstructed_before_timing": value.target_preconstructed_before_timing,
        "target_host_state_pinned_before_timing": value.target_host_state_pinned_before_timing,
        "steady_state_repeated_cpu_streaming": value.steady_state_repeated_cpu_streaming,
        "forward_handoff_timed": value.forward_handoff_timed,
        "reverse_handoff_timed": value.reverse_handoff_timed,
        "transition_once_per_direction_per_cycle": value.transition_once_per_direction_per_cycle,
        "uses_gpu_live_reshard": value.uses_gpu_live_reshard,
    }


__all__ = ["RolePhaseTransitionContract", "transition_contract_summary"]
