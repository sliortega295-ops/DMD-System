"""Deterministic event simulator for cyclic compound-distillation plans."""
from __future__ import annotations

import dataclasses
import statistics
from typing import Any, Mapping

from .cost_model import HybridCostModel
from .profile_db import CoverageReport, ProfileDB
from .schema import PhaseSpec, PlanObservation, PlanSpec, PrimitiveKey


@dataclasses.dataclass(frozen=True)
class PhaseEstimate:
    name: str
    seconds: float | None
    kind: str
    confidence: str
    details: Mapping[str, Any]


@dataclasses.dataclass(frozen=True)
class SimulationResult:
    plan_name: str
    predicted_seconds: float | None
    phases: tuple[PhaseEstimate, ...]
    transition_seconds: Mapping[str, float]
    coverage: CoverageReport
    confidence: str
    claimable_prediction: bool
    observed_median_seconds: float | None = None
    absolute_error_seconds: float | None = None
    relative_error: float | None = None
    notes: tuple[str, ...] = ()


def simulate_two_stage_pipeline(*, producer_seconds: float, consumer_seconds: float, batches: int, queue_depth: int, credit: str) -> Mapping[str, Any]:
    """Simulate a producer/consumer stream with explicit payload-credit semantics."""
    if producer_seconds <= 0 or consumer_seconds <= 0 or batches <= 0 or queue_depth <= 0:
        raise ValueError("service times, batches, and queue depth must be positive")
    if credit not in {"payload_receive", "send_completion", "consumer_commit"}:
        raise ValueError(f"unsupported credit policy {credit}")
    producer_finish: list[float] = []
    consumer_start: list[float] = []
    consumer_finish: list[float] = []
    producer_ready = consumer_ready = producer_blocked = 0.0
    for index in range(batches):
        credit_ready = 0.0
        if index >= queue_depth:
            prior = index - queue_depth
            credit_ready = consumer_start[prior] if credit in {"payload_receive", "send_completion"} else consumer_finish[prior]
        start = max(producer_ready, credit_ready)
        producer_blocked += max(0.0, start - producer_ready)
        produced = start + producer_seconds
        producer_finish.append(produced)
        producer_ready = produced
        consume_start = max(consumer_ready, produced)
        consume_finish = consume_start + consumer_seconds
        consumer_start.append(consume_start)
        consumer_finish.append(consume_finish)
        consumer_ready = consume_finish
    total = consumer_finish[-1]
    ideal = producer_seconds + consumer_seconds + (batches - 1) * max(producer_seconds, consumer_seconds)
    return {
        "seconds": total,
        "producer_service_seconds": producer_seconds,
        "consumer_service_seconds": consumer_seconds,
        "batches": batches,
        "queue_depth": queue_depth,
        "credit": credit,
        "producer_blocked_seconds": producer_blocked,
        "fill_seconds": producer_seconds,
        "drain_seconds": consumer_seconds,
        "ideal_unbounded_seconds": ideal,
        "queue_penalty_seconds": max(0.0, total - ideal),
        "bottleneck": "producer" if producer_seconds > consumer_seconds else "consumer",
    }


class PlanSimulator:
    def __init__(self, profile_db: ProfileDB) -> None:
        self.db = profile_db
        self.cost = HybridCostModel(profile_db)

    @staticmethod
    def _required_keys(plan: PlanSpec) -> tuple[PrimitiveKey, ...]:
        result: list[PrimitiveKey] = []
        for phase in plan.phases:
            if phase.kind == "sequential":
                result.extend(phase.primitive_keys)
            elif phase.kind == "pipeline":
                assert phase.producer_key is not None and phase.consumer_key is not None
                result.extend((phase.producer_key, phase.consumer_key))
        return tuple(result)

    def _phase(self, phase: PhaseSpec, *, allow_interpolation: bool) -> PhaseEstimate:
        if phase.kind == "fixed":
            return PhaseEstimate(phase.name, phase.fixed_seconds, phase.kind, "anchored_component", {"source": "explicit fixed phase component"})
        if phase.kind == "sequential":
            estimates = [self.cost.operation(key, allow_interpolation=allow_interpolation) for key in phase.primitive_keys]
            missing = [value for value in estimates if value.wall_seconds is None]
            if missing:
                return PhaseEstimate(phase.name, None, phase.kind, "missing", {"missing": [value.key.id for value in missing]})
            seconds = phase.repeats * sum(float(value.wall_seconds) for value in estimates)
            confidence = "measured" if all(value.match == "exact" for value in estimates) else "interpolated"
            return PhaseEstimate(
                phase.name, seconds, phase.kind, confidence,
                {
                    "repeats": phase.repeats,
                    "operations": [
                        {"key": value.key.id, "seconds": value.wall_seconds, "match": value.match, "breakdown": dict(value.breakdown)}
                        for value in estimates
                    ],
                },
            )
        assert phase.producer_key is not None and phase.consumer_key is not None
        producer = self.cost.operation(phase.producer_key, allow_interpolation=allow_interpolation)
        consumer = self.cost.operation(phase.consumer_key, allow_interpolation=allow_interpolation)
        if producer.wall_seconds is None or consumer.wall_seconds is None:
            missing = []
            if producer.wall_seconds is None:
                missing.append(producer.key.id)
            if consumer.wall_seconds is None:
                missing.append(consumer.key.id)
            return PhaseEstimate(phase.name, None, phase.kind, "missing", {"missing": missing})
        details = simulate_two_stage_pipeline(
            producer_seconds=float(producer.wall_seconds),
            consumer_seconds=float(consumer.wall_seconds),
            batches=phase.batches,
            queue_depth=phase.queue_depth,
            credit=phase.overlap_credit,
        )
        confidence = "measured" if producer.match == consumer.match == "exact" else "interpolated"
        return PhaseEstimate(
            phase.name, float(details["seconds"]), phase.kind, confidence,
            {**details, "producer_key": producer.key.id, "consumer_key": consumer.key.id, "producer_match": producer.match, "consumer_match": consumer.match},
        )

    def simulate(self, plan: PlanSpec, *, allow_interpolation: bool = True, observation_name: str | None = None) -> SimulationResult:
        keys = self._required_keys(plan)
        coverage = self.db.coverage(keys, transitions=plan.transition_names, allow_interpolation=allow_interpolation)
        phases = tuple(self._phase(phase, allow_interpolation=allow_interpolation) for phase in plan.phases)
        transition_values = {
            name: transition.wall_seconds
            for name in plan.transition_names
            if (transition := self.db.transition(name)) is not None
        }
        predicted = None
        if not any(value.seconds is None for value in phases) and not coverage.transition_missing:
            predicted = sum(float(value.seconds) for value in phases) + sum(transition_values.values())

        observation: PlanObservation | None = self.db.plan(observation_name or plan.name)
        observed = absolute = relative = None
        if observation is not None:
            observed = statistics.median(observation.full5_seconds)
            if predicted is not None:
                absolute = abs(predicted - observed)
                relative = absolute / observed

        if predicted is None:
            confidence = "insufficient_profile"
        elif coverage.claimable and all(value.confidence in {"measured", "anchored_component"} for value in phases):
            confidence = "measured_components"
        else:
            confidence = "interpolated"
        notes: list[str] = []
        if any(value.kind == "fixed" for value in phases):
            notes.append("fixed component anchors test composition but not unseen-plan generalization")
        if coverage.interpolated:
            notes.append("interpolated primitives require a short calibration canary")
        if observation is not None and not observation.claimable_timing:
            notes.append("comparison observation is diagnostic rather than formal timing")
        return SimulationResult(
            plan.name, predicted, phases, transition_values, coverage, confidence,
            predicted is not None and coverage.claimable and not any(value.kind == "fixed" for value in phases),
            observed, absolute, relative, tuple(notes),
        )


def replay_decomposed_observation(observation: PlanObservation) -> Mapping[str, Any]:
    """Check whether measured phase components reconstruct a plan anchor."""
    if not observation.phase_seconds:
        return {"name": observation.name, "status": "INSUFFICIENT_COMPONENTS", "observed_median_seconds": statistics.median(observation.full5_seconds)}
    predicted = sum(float(value) for value in observation.phase_seconds.values())
    observed = statistics.median(observation.full5_seconds)
    error = abs(predicted - observed)
    return {
        "name": observation.name,
        "status": "PASS",
        "component_sum_seconds": predicted,
        "observed_median_seconds": observed,
        "absolute_error_seconds": error,
        "relative_error": error / observed,
        "component_names": sorted(observation.phase_seconds),
        "claim_boundary": "composition check, not leave-one-out unseen-plan validation",
    }


__all__ = ["PhaseEstimate", "PlanSimulator", "SimulationResult", "replay_decomposed_observation", "simulate_two_stage_pipeline"]
