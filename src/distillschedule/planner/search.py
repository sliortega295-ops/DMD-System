"""Memory-pruned ranking and active calibration requests for planner candidates."""
from __future__ import annotations

import dataclasses
from typing import Any, Mapping, Sequence

from .cost_model import MemoryEstimate
from .schema import PlanSpec, PrimitiveKey
from .simulator import PlanSimulator, SimulationResult


@dataclasses.dataclass(frozen=True)
class CandidatePlan:
    plan: PlanSpec
    memory: MemoryEstimate | None = None
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def feasible(self) -> bool:
        return self.memory is None or self.memory.feasible


@dataclasses.dataclass(frozen=True)
class RankedPlan:
    candidate: CandidatePlan
    simulation: SimulationResult | None
    objective_seconds: float | None
    uncertainty_penalty: float
    status: str
    blockers: tuple[str, ...] = ()


def rank_candidates(
    candidates: Sequence[CandidatePlan],
    simulator: PlanSimulator,
    *,
    allow_interpolation: bool = True,
    interpolation_penalty_fraction: float = 0.15,
) -> tuple[RankedPlan, ...]:
    """Rank feasible plans without turning missing measurements into guesses.

    Exact predictions receive zero uncertainty penalty. Every interpolated
    primitive adds a bounded fractional penalty so measured plans are preferred
    when point estimates are close. Missing primitives remain explicit and
    are returned for active calibration rather than assigned a made-up cost.
    """

    if interpolation_penalty_fraction < 0:
        raise ValueError("interpolation penalty cannot be negative")
    ranked: list[RankedPlan] = []
    for candidate in candidates:
        if not candidate.feasible:
            blockers: list[str] = []
            assert candidate.memory is not None
            if not candidate.memory.feasible_gpu:
                blockers.append("GPU_MEMORY")
            if not candidate.memory.feasible_cpu:
                blockers.append("CPU_MEMORY")
            ranked.append(
                RankedPlan(
                    candidate=candidate,
                    simulation=None,
                    objective_seconds=None,
                    uncertainty_penalty=0.0,
                    status="MEMORY_INFEASIBLE",
                    blockers=tuple(blockers),
                )
            )
            continue
        simulation = simulator.simulate(
            candidate.plan,
            allow_interpolation=allow_interpolation,
        )
        if simulation.predicted_seconds is None:
            blockers = tuple(
                f"PRIMITIVE:{key.id}" for key in simulation.coverage.missing
            ) + tuple(
                f"TRANSITION:{name}"
                for name in simulation.coverage.transition_missing
            )
            ranked.append(
                RankedPlan(
                    candidate=candidate,
                    simulation=simulation,
                    objective_seconds=None,
                    uncertainty_penalty=0.0,
                    status="PROFILE_INCOMPLETE",
                    blockers=blockers,
                )
            )
            continue
        interpolated = simulation.coverage.interpolated
        penalty = (
            float(simulation.predicted_seconds)
            * interpolation_penalty_fraction
            * interpolated
        )
        status = (
            "CLAIMABLE"
            if simulation.claimable_prediction
            else "CANARY_REQUIRED"
        )
        ranked.append(
            RankedPlan(
                candidate=candidate,
                simulation=simulation,
                objective_seconds=float(simulation.predicted_seconds) + penalty,
                uncertainty_penalty=penalty,
                status=status,
            )
        )

    def sort_key(value: RankedPlan) -> tuple[int, float, str]:
        category = {
            "CLAIMABLE": 0,
            "CANARY_REQUIRED": 1,
            "PROFILE_INCOMPLETE": 2,
            "MEMORY_INFEASIBLE": 3,
        }[value.status]
        objective = value.objective_seconds
        return category, float("inf") if objective is None else objective, value.candidate.plan.name

    return tuple(sorted(ranked, key=sort_key))


def profile_requests(
    ranked: Sequence[RankedPlan],
    *,
    top_k: int = 5,
) -> tuple[PrimitiveKey, ...]:
    """Return deduplicated primitive requests from the best incomplete plans."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    result: list[PrimitiveKey] = []
    seen: set[str] = set()
    incomplete = [item for item in ranked if item.status == "PROFILE_INCOMPLETE"]
    for item in incomplete[:top_k]:
        simulation = item.simulation
        if simulation is None:
            continue
        for key in simulation.coverage.missing:
            if key.id in seen:
                continue
            seen.add(key.id)
            result.append(key)
    return tuple(result)


__all__ = ["CandidatePlan", "RankedPlan", "profile_requests", "rank_candidates"]
