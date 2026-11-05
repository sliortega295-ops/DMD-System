"""Profile database and explicit coverage accounting for the v0 planner."""
from __future__ import annotations

import dataclasses
import statistics
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

from .schema import PlanObservation, PlannerProfile, PrimitiveKey, PrimitiveMeasurement, TransitionMeasurement


@dataclasses.dataclass(frozen=True)
class LookupResult:
    measurement: PrimitiveMeasurement | None
    match: str
    distance: float | None
    explanation: str


@dataclasses.dataclass(frozen=True)
class CoverageReport:
    required: int
    exact: int
    interpolated: int
    missing: tuple[PrimitiveKey, ...]
    transition_missing: tuple[str, ...] = ()

    @property
    def fraction(self) -> float:
        return (self.exact + self.interpolated) / self.required if self.required else 1.0

    @property
    def claimable(self) -> bool:
        return not self.missing and not self.transition_missing and self.interpolated == 0


class ProfileDB:
    """Read-only indexed view over measured planner evidence."""

    def __init__(self, profile: PlannerProfile) -> None:
        self.profile = profile
        self._exact: dict[str, list[PrimitiveMeasurement]] = defaultdict(list)
        self._families: dict[tuple[Any, ...], list[PrimitiveMeasurement]] = defaultdict(list)
        for measurement in profile.primitives:
            self._exact[measurement.key.id].append(measurement)
            self._families[self._family(measurement.key)].append(measurement)
        self._transitions: dict[str, list[TransitionMeasurement]] = defaultdict(list)
        for transition in profile.transitions:
            self._transitions[transition.name].append(transition)
        self._plans = {plan.name: plan for plan in profile.plans}
        self._models = {model.role: model for model in profile.models}

    @staticmethod
    def _family(key: PrimitiveKey) -> tuple[Any, ...]:
        return (
            key.workload, key.operation, key.role, key.backend, key.stage,
            key.grad_mode, key.residency, key.activation_policy,
            key.offload_blocks, key.dtype,
        )

    @staticmethod
    def _aggregate(measurements: Sequence[PrimitiveMeasurement]) -> PrimitiveMeasurement:
        first = measurements[0]
        if any(value.key != first.key for value in measurements[1:]):
            raise ValueError("primitive aggregation requires identical keys")

        def median_optional(name: str):
            present = [getattr(value, name) for value in measurements if getattr(value, name) is not None]
            return statistics.median(present) if present else None

        gpu_peak = median_optional("gpu_peak_bytes")
        cpu_peak = median_optional("cpu_peak_bytes")
        activation = median_optional("activation_peak_bytes")
        return PrimitiveMeasurement(
            key=first.key,
            wall_seconds=statistics.median(value.wall_seconds for value in measurements),
            samples=sum(value.samples for value in measurements),
            provenance=";".join(sorted({value.provenance for value in measurements})),
            cuda_device_seconds=median_optional("cuda_device_seconds"),
            compute_seconds=median_optional("compute_seconds"),
            nccl_seconds=median_optional("nccl_seconds"),
            h2d_seconds=median_optional("h2d_seconds"),
            d2h_seconds=median_optional("d2h_seconds"),
            transferred_h2d_bytes=int(statistics.median(value.transferred_h2d_bytes for value in measurements)),
            transferred_d2h_bytes=int(statistics.median(value.transferred_d2h_bytes for value in measurements)),
            collective_bytes=int(statistics.median(value.collective_bytes for value in measurements)),
            gpu_peak_bytes=int(gpu_peak) if gpu_peak is not None else None,
            cpu_peak_bytes=int(cpu_peak) if cpu_peak is not None else None,
            activation_peak_bytes=int(activation) if activation is not None else None,
            parameter_materialization_count=int(statistics.median(value.parameter_materialization_count for value in measurements)),
            confidence="measured",
            metadata={"aggregated_measurement_count": len(measurements)},
        )

    def lookup(self, key: PrimitiveKey, *, allow_interpolation: bool = True) -> LookupResult:
        exact = self._exact.get(key.id)
        if exact:
            return LookupResult(self._aggregate(exact), "exact", 0.0, "exact primitive key")
        if not allow_interpolation:
            return LookupResult(None, "missing", None, "exact primitive is absent")
        candidates = self._families.get(self._family(key), ())
        if not candidates:
            return LookupResult(None, "missing", None, "no measurement in same strict primitive family")

        def distance(value: PrimitiveMeasurement) -> float:
            return (
                abs(value.key.mesh_size - key.mesh_size) / max(key.mesh_size, 1)
                + abs(value.key.local_batch - key.local_batch) / max(key.local_batch, 1)
            )

        best = min(candidates, key=distance)
        d = distance(best)
        if d > 1.0:
            return LookupResult(None, "missing", d, "nearest strict-family sample is too far away")
        scaled = dataclasses.replace(
            best,
            key=key,
            wall_seconds=best.wall_seconds * key.local_batch / best.key.local_batch,
            confidence="derived",
            metadata={
                **dict(best.metadata),
                "interpolated_from": best.key.id,
                "mesh_scaling_applied": False,
                "local_batch_linear_scaling": True,
                "distance": d,
            },
        )
        return LookupResult(scaled, "interpolated", d, "same strict family; local-batch scaling only")

    def transition(self, name: str) -> TransitionMeasurement | None:
        values = self._transitions.get(name)
        if not values:
            return None
        first = values[0]
        return dataclasses.replace(
            first,
            wall_seconds=statistics.median(value.wall_seconds for value in values),
            samples=sum(value.samples for value in values),
            provenance=";".join(sorted({value.provenance for value in values})),
        )

    def plan(self, name: str) -> PlanObservation | None:
        return self._plans.get(name)

    def model(self, role: str):
        return self._models.get(role)

    def coverage(self, keys: Iterable[PrimitiveKey], *, transitions: Iterable[str] = (), allow_interpolation: bool = True) -> CoverageReport:
        key_list = tuple(keys)
        exact = interpolated = 0
        missing: list[PrimitiveKey] = []
        for key in key_list:
            result = self.lookup(key, allow_interpolation=allow_interpolation)
            if result.match == "exact":
                exact += 1
            elif result.match == "interpolated":
                interpolated += 1
            else:
                missing.append(key)
        transition_missing = tuple(name for name in transitions if self.transition(name) is None)
        return CoverageReport(len(key_list), exact, interpolated, tuple(missing), transition_missing)

    def observations_by_workload(self) -> Mapping[str, tuple[PlanObservation, ...]]:
        grouped: dict[str, list[PlanObservation]] = defaultdict(list)
        for value in self.profile.plans:
            grouped[value.workload].append(value)
        return {key: tuple(values) for key, values in grouped.items()}


__all__ = ["CoverageReport", "LookupResult", "ProfileDB"]
