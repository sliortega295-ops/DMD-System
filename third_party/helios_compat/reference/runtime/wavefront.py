"""Lifetime- and version-aware wavefront planning for gradient boundaries.

The planner expands one Generator/Score gradient-boundary template across
microbatches.  It keeps gradient consumption separate from the parameter
version barrier, which lets an executor release replay state before stepping.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from .ir import EdgeType, ExecutionGraph, OperationType
from .planner import HardwareBudget, NoFeasiblePlanError


class WavefrontPolicy(str, Enum):
    SEQUENTIAL = "sequential"
    BATCHED_PIPELINE = "pipeline"
    STREAMING_PIPELINE = "streaming_pipeline"


class CostSelectionMode(str, Enum):
    """How replicated update-time observations affect policy selection.

    ``POINT_ESTIMATE`` preserves the original median-based behavior.
    ``OBSERVED_WORST_CASE`` minimizes the slowest supplied replicate.  The
    latter is an empirical risk objective, not a statistical confidence bound.
    """

    POINT_ESTIMATE = "point_estimate"
    OBSERVED_WORST_CASE = "observed_worst_case"
    PERFORMANCE_BAND_MEMORY = "performance_band_memory"


@dataclass(frozen=True)
class BoundaryAction:
    kind: str
    microbatch: Optional[int] = None
    parameter_version: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "microbatch": self.microbatch,
            "parameter_version": self.parameter_version,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BoundaryAction":
        return cls(
            kind=str(value["kind"]),
            microbatch=(
                None if value.get("microbatch") is None else int(value["microbatch"])
            ),
            parameter_version=int(value.get("parameter_version", 0)),
            metadata=dict(value.get("metadata", {})),
        )


@dataclass(frozen=True)
class BoundaryLiveness:
    max_live_replay_records: int
    max_live_input_gradients: int
    max_live_edges: int


@dataclass(frozen=True)
class WavefrontCandidate:
    policy: WavefrontPolicy
    update_time_ms: float
    required_gpus: int = 2
    observed_update_times_ms: Tuple[float, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.update_time_ms < 0.0:
            raise ValueError("update_time_ms must be non-negative")
        if self.required_gpus <= 0:
            raise ValueError("required_gpus must be positive")
        if any(value < 0.0 for value in self.observed_update_times_ms):
            raise ValueError("observed update times must be non-negative")

    def selection_cost_ms(self, mode: CostSelectionMode | str) -> float:
        mode = CostSelectionMode(mode)
        if mode == CostSelectionMode.OBSERVED_WORST_CASE:
            if not self.observed_update_times_ms:
                raise ValueError(
                    "observed_worst_case selection requires replicate observations"
                )
            return max(self.observed_update_times_ms)
        return self.update_time_ms

    def cost_summary(self) -> Dict[str, Any]:
        observations = self.observed_update_times_ms
        return {
            "point_estimate_ms": self.update_time_ms,
            "observed_update_times_ms": list(observations),
            "observed_min_ms": min(observations) if observations else None,
            "observed_max_ms": max(observations) if observations else None,
        }


@dataclass(frozen=True)
class LifetimeMemoryModel:
    """Conservative peak model calibrated from one-microbatch residency.

    ``base_peak_bytes`` includes one input and one boundary. Additional
    microbatches add input residency, while each live edge beyond the first
    adds replay state plus its returned input gradient.
    """

    base_peak_bytes: int
    input_bytes_per_extra_microbatch: int
    retained_edge_bytes_per_extra_live_edge: int

    def __post_init__(self) -> None:
        if self.base_peak_bytes <= 0:
            raise ValueError("base_peak_bytes must be positive")
        if self.input_bytes_per_extra_microbatch < 0:
            raise ValueError("input residency must be non-negative")
        if self.retained_edge_bytes_per_extra_live_edge < 0:
            raise ValueError("edge residency must be non-negative")

    def estimate_peak_bytes(
        self, microbatches: int, liveness: BoundaryLiveness
    ) -> int:
        if microbatches <= 0:
            raise ValueError("microbatches must be positive")
        return (
            self.base_peak_bytes
            + (microbatches - 1) * self.input_bytes_per_extra_microbatch
            + max(0, liveness.max_live_edges - 1)
            * self.retained_edge_bytes_per_extra_live_edge
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": "single_role",
            "base_peak_bytes": self.base_peak_bytes,
            "input_bytes_per_extra_microbatch": self.input_bytes_per_extra_microbatch,
            "retained_edge_bytes_per_extra_live_edge": self.retained_edge_bytes_per_extra_live_edge,
        }


@dataclass(frozen=True)
class RoleMemoryTerm:
    """Peak components for one physical execution role.

    ``multi_microbatch_overhead_bytes`` captures measured allocator/workspace or
    first-accumulation state that appears once when an update spans more than one
    microbatch. Only ``owns_gradient_lifetime`` roles are charged for the
    policy-dependent live replay/input-gradient window.
    """

    base_peak_bytes: int
    input_bytes_per_extra_microbatch: int = 0
    retained_edge_bytes_per_extra_live_edge: int = 0
    multi_microbatch_overhead_bytes: int = 0
    owns_gradient_lifetime: bool = False

    def __post_init__(self) -> None:
        for name in (
            "base_peak_bytes",
            "input_bytes_per_extra_microbatch",
            "retained_edge_bytes_per_extra_live_edge",
            "multi_microbatch_overhead_bytes",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")

    def estimate_peak_bytes(
        self, microbatches: int, liveness: BoundaryLiveness
    ) -> int:
        if microbatches <= 0:
            raise ValueError("microbatches must be positive")
        edge_count = (
            max(0, liveness.max_live_edges - 1)
            if self.owns_gradient_lifetime
            else 0
        )
        return (
            self.base_peak_bytes
            + (microbatches - 1) * self.input_bytes_per_extra_microbatch
            + (self.multi_microbatch_overhead_bytes if microbatches > 1 else 0)
            + edge_count * self.retained_edge_bytes_per_extra_live_edge
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_peak_bytes": self.base_peak_bytes,
            "input_bytes_per_extra_microbatch": self.input_bytes_per_extra_microbatch,
            "retained_edge_bytes_per_extra_live_edge": self.retained_edge_bytes_per_extra_live_edge,
            "multi_microbatch_overhead_bytes": self.multi_microbatch_overhead_bytes,
            "owns_gradient_lifetime": self.owns_gradient_lifetime,
        }


@dataclass(frozen=True)
class RoleAwareLifetimeMemoryModel:
    """Compose policy-dependent memory per role, then take the hardware peak."""

    roles: Mapping[str, RoleMemoryTerm]

    def __post_init__(self) -> None:
        if not self.roles:
            raise ValueError("at least one role memory term is required")
        if not any(term.owns_gradient_lifetime for term in self.roles.values()):
            raise ValueError("one role must own the gradient lifetime")

    def estimate_role_peak_bytes(
        self, microbatches: int, liveness: BoundaryLiveness
    ) -> Dict[str, int]:
        return {
            role: term.estimate_peak_bytes(microbatches, liveness)
            for role, term in self.roles.items()
        }

    def estimate_peak_bytes(
        self, microbatches: int, liveness: BoundaryLiveness
    ) -> int:
        return max(self.estimate_role_peak_bytes(microbatches, liveness).values())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": "role_aware",
            "roles": {role: term.to_dict() for role, term in self.roles.items()},
        }


@dataclass(frozen=True)
class WavefrontPlan:
    candidate: WavefrontCandidate
    microbatches: int
    actions: Tuple[BoundaryAction, ...]
    liveness: BoundaryLiveness
    predicted_peak_memory_bytes: int
    rejected_candidates: Mapping[str, str]
    predicted_role_peak_memory_bytes: Mapping[str, int] = field(default_factory=dict)
    bottleneck_role: Optional[str] = None
    cost_selection_mode: CostSelectionMode = CostSelectionMode.POINT_ESTIMATE
    selection_cost_ms: Optional[float] = None
    planner_metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def policy(self) -> WavefrontPolicy:
        return self.candidate.policy

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy": self.policy.value,
            "microbatches": self.microbatches,
            "predicted_update_time_ms": self.candidate.update_time_ms,
            "cost_selection_mode": self.cost_selection_mode.value,
            "selection_cost_ms": self.selection_cost_ms,
            "predicted_peak_memory_bytes": self.predicted_peak_memory_bytes,
            "predicted_role_peak_memory_bytes": dict(
                self.predicted_role_peak_memory_bytes
            ),
            "bottleneck_role": self.bottleneck_role,
            "required_gpus": self.candidate.required_gpus,
            "liveness": {
                "max_live_replay_records": self.liveness.max_live_replay_records,
                "max_live_input_gradients": self.liveness.max_live_input_gradients,
                "max_live_edges": self.liveness.max_live_edges,
            },
            "rejected_candidates": dict(self.rejected_candidates),
            "planner_metadata": dict(self.planner_metadata),
            "actions": [action.to_dict() for action in self.actions],
        }


def build_boundary_actions(
    policy: WavefrontPolicy | str,
    microbatches: int,
    *,
    optimizer_action: str = "optimizer_ema",
) -> Tuple[BoundaryAction, ...]:
    """Expand the rank-0 boundary protocol for one fixed parameter version."""

    policy = WavefrontPolicy(policy)
    if microbatches <= 0:
        raise ValueError("microbatches must be positive")
    if optimizer_action not in {"optimizer_update", "optimizer_ema"}:
        raise ValueError("optimizer_action must be optimizer_update or optimizer_ema")
    actions: list[BoundaryAction] = []

    def add(kind: str, microbatch: Optional[int] = None, **metadata: Any) -> None:
        actions.append(
            BoundaryAction(
                kind=kind,
                microbatch=microbatch,
                parameter_version=0,
                metadata=metadata,
            )
        )

    if policy == WavefrontPolicy.SEQUENTIAL:
        for microbatch in range(microbatches):
            add("rollout", microbatch)
            add("send_fake", microbatch)
            add("recv_boundary", microbatch)
        add("begin_accumulation")
        for microbatch in range(microbatches):
            add("replay_backward", microbatch)

    elif policy == WavefrontPolicy.BATCHED_PIPELINE:
        add("rollout", 0)
        add("send_fake", 0)
        for microbatch in range(1, microbatches):
            add("rollout", microbatch, overlaps=f"score_input_gradient:{microbatch - 1}")
            add("recv_boundary", microbatch - 1)
            add("send_fake", microbatch)
        add("recv_boundary", microbatches - 1)
        add("begin_accumulation")
        for microbatch in range(microbatches):
            add("replay_backward", microbatch)

    else:
        add("begin_accumulation")
        add("rollout", 0)
        add("send_fake", 0)
        for microbatch in range(1, microbatches):
            add("rollout", microbatch, overlaps=f"score_input_gradient:{microbatch - 1}")
            add("recv_boundary", microbatch - 1)
            add("send_fake", microbatch)
            add(
                "replay_backward",
                microbatch - 1,
                overlaps=f"score_input_gradient:{microbatch}",
                release_gradient=True,
                release_replay=True,
            )
        add("recv_boundary", microbatches - 1)
        add(
            "replay_backward",
            microbatches - 1,
            release_gradient=True,
            release_replay=True,
        )

    add("version_barrier", expected_replays=microbatches)
    actions.append(
        BoundaryAction(
            kind=optimizer_action,
            parameter_version=1,
            metadata={"previous_parameter_version": 0},
        )
    )
    analyze_boundary_liveness(actions, microbatches=microbatches)
    return tuple(actions)


def analyze_boundary_liveness(
    actions: Iterable[BoundaryAction], *, microbatches: int
) -> BoundaryLiveness:
    replay_records: set[int] = set()
    gradients: set[int] = set()
    replayed: set[int] = set()
    sent: set[int] = set()
    accumulation_started = False
    barrier_seen = False
    max_replay = 0
    max_gradient = 0
    max_edges = 0

    for action in actions:
        microbatch = action.microbatch
        if action.kind == "begin_accumulation":
            if accumulation_started:
                raise ValueError("accumulation begins more than once")
            accumulation_started = True
        elif action.kind == "rollout":
            if microbatch is None or microbatch in replay_records or microbatch in replayed:
                raise ValueError("invalid or duplicate rollout")
            replay_records.add(microbatch)
        elif action.kind == "send_fake":
            if microbatch not in replay_records:
                raise ValueError("fake sent before rollout")
            sent.add(microbatch)
        elif action.kind == "recv_boundary":
            if microbatch not in sent or microbatch in gradients:
                raise ValueError("boundary received before send or more than once")
            gradients.add(microbatch)
        elif action.kind == "replay_backward":
            if not accumulation_started:
                raise ValueError("replay backward precedes accumulation begin")
            if microbatch not in replay_records or microbatch not in gradients:
                raise ValueError("replay backward requires replay state and dL/dx")
            replay_records.remove(microbatch)
            gradients.remove(microbatch)
            replayed.add(microbatch)
        elif action.kind == "version_barrier":
            if replay_records or gradients or len(replayed) != microbatches:
                raise ValueError("version barrier crossed with live gradient state")
            barrier_seen = True
        elif action.kind in {"optimizer_update", "optimizer_ema"}:
            if not barrier_seen or action.parameter_version != 1:
                raise ValueError("optimizer step precedes version barrier")
        else:
            raise ValueError(f"unknown boundary action: {action.kind}")
        max_replay = max(max_replay, len(replay_records))
        max_gradient = max(max_gradient, len(gradients))
        max_edges = max(max_edges, max(len(replay_records), len(gradients)))

    if len(replayed) != microbatches or not barrier_seen:
        raise ValueError("incomplete boundary plan")
    return BoundaryLiveness(max_replay, max_gradient, max_edges)


class LifetimeAwareWavefrontPlanner:
    """Choose a measured policy, then synthesize its edge-lifetime actions."""

    def __init__(
        self,
        candidates: Iterable[WavefrontCandidate],
        memory_model: LifetimeMemoryModel | RoleAwareLifetimeMemoryModel,
        *,
        cost_selection_mode: CostSelectionMode | str = CostSelectionMode.POINT_ESTIMATE,
        relative_slowdown_tolerance: float = 0.0,
    ) -> None:
        self.candidates = tuple(candidates)
        if not self.candidates:
            raise ValueError("at least one wavefront candidate is required")
        policies = [candidate.policy for candidate in self.candidates]
        if len(policies) != len(set(policies)):
            raise ValueError("wavefront candidate policies must be unique")
        self.memory_model = memory_model
        self.cost_selection_mode = CostSelectionMode(cost_selection_mode)
        if relative_slowdown_tolerance < 0.0:
            raise ValueError("relative_slowdown_tolerance must be non-negative")
        if (
            relative_slowdown_tolerance != 0.0
            and self.cost_selection_mode
            != CostSelectionMode.PERFORMANCE_BAND_MEMORY
        ):
            raise ValueError(
                "relative_slowdown_tolerance is only valid for "
                "performance_band_memory selection"
            )
        self.relative_slowdown_tolerance = relative_slowdown_tolerance
        if self.cost_selection_mode == CostSelectionMode.OBSERVED_WORST_CASE:
            missing = [
                candidate.policy.value
                for candidate in self.candidates
                if not candidate.observed_update_times_ms
            ]
            if missing:
                raise ValueError(
                    "observed_worst_case selection requires observations for: "
                    + ", ".join(missing)
                )

    def plan(
        self,
        graph: ExecutionGraph,
        *,
        microbatches: int,
        budget: HardwareBudget,
    ) -> WavefrontPlan:
        graph.validate()
        edge_types = {edge.edge_type for edge in graph.edges}
        if EdgeType.GRADIENT not in edge_types or EdgeType.VERSION not in edge_types:
            raise ValueError("wavefront planning requires gradient and version edges")

        optimizer_action = (
            "optimizer_ema"
            if any(
                node.operation_type == OperationType.EMA_UPDATE
                for node in graph.nodes.values()
            )
            else "optimizer_update"
        )
        feasible: list[
            tuple[
                WavefrontCandidate,
                Tuple[BoundaryAction, ...],
                BoundaryLiveness,
                int,
                Mapping[str, int],
                Optional[str],
            ]
        ] = []
        rejected: Dict[str, str] = {}
        for candidate in self.candidates:
            actions = build_boundary_actions(
                candidate.policy,
                microbatches,
                optimizer_action=optimizer_action,
            )
            liveness = analyze_boundary_liveness(actions, microbatches=microbatches)
            if isinstance(self.memory_model, RoleAwareLifetimeMemoryModel):
                role_peaks = self.memory_model.estimate_role_peak_bytes(
                    microbatches, liveness
                )
                bottleneck_role = max(role_peaks, key=role_peaks.get)
                peak = role_peaks[bottleneck_role]
            else:
                role_peaks = {}
                bottleneck_role = None
                peak = self.memory_model.estimate_peak_bytes(microbatches, liveness)
            if candidate.required_gpus > budget.gpu_count:
                rejected[candidate.policy.value] = (
                    f"requires {candidate.required_gpus} GPUs, budget has {budget.gpu_count}"
                )
            elif peak > budget.gpu_memory_bytes:
                owner = (
                    f" on role {bottleneck_role}" if bottleneck_role is not None else ""
                )
                rejected[candidate.policy.value] = (
                    f"predicted peak {peak}{owner} exceeds budget "
                    f"{budget.gpu_memory_bytes}"
                )
            else:
                feasible.append(
                    (
                        candidate,
                        actions,
                        liveness,
                        peak,
                        role_peaks,
                        bottleneck_role,
                    )
                )
        if not feasible:
            reasons = "; ".join(
                f"{policy}: {reason}" for policy, reason in rejected.items()
            )
            raise NoFeasiblePlanError(f"no wavefront fits the hardware budget ({reasons})")

        fastest_feasible_ms = min(item[0].update_time_ms for item in feasible)
        performance_band = feasible
        if self.cost_selection_mode == CostSelectionMode.PERFORMANCE_BAND_MEMORY:
            threshold_ms = fastest_feasible_ms * (
                1.0 + self.relative_slowdown_tolerance
            )
            performance_band = [
                item for item in feasible if item[0].update_time_ms <= threshold_ms
            ]
            candidate, actions, liveness, peak, role_peaks, bottleneck_role = min(
                performance_band,
                key=lambda item: (
                    item[3],
                    sum(item[4].values()) if item[4] else item[3],
                    item[0].update_time_ms,
                    item[0].policy.value,
                ),
            )
        else:
            candidate, actions, liveness, peak, role_peaks, bottleneck_role = min(
                feasible,
                key=lambda item: (
                    item[0].selection_cost_ms(self.cost_selection_mode),
                    item[3],
                    item[0].update_time_ms,
                    item[0].policy.value,
                ),
            )

        def dominates(left: tuple[Any, ...], right: tuple[Any, ...]) -> bool:
            left_memory = (
                tuple(left[4][role] for role in sorted(left[4]))
                if left[4]
                else (left[3],)
            )
            right_memory = (
                tuple(right[4][role] for role in sorted(right[4]))
                if right[4]
                else (right[3],)
            )
            left_values = (left[0].update_time_ms, *left_memory)
            right_values = (right[0].update_time_ms, *right_memory)
            return all(a <= b for a, b in zip(left_values, right_values)) and any(
                a < b for a, b in zip(left_values, right_values)
            )

        pareto_policies = [
            item[0].policy.value
            for item in feasible
            if not any(
                other is not item and dominates(other, item) for other in feasible
            )
        ]
        selection_cost_ms = candidate.selection_cost_ms(self.cost_selection_mode)
        selected_slowdown_percent = (
            (candidate.update_time_ms / fastest_feasible_ms - 1.0) * 100.0
            if fastest_feasible_ms > 0.0
            else 0.0
        )
        return WavefrontPlan(
            candidate=candidate,
            microbatches=microbatches,
            actions=actions,
            liveness=liveness,
            predicted_peak_memory_bytes=peak,
            rejected_candidates=rejected,
            predicted_role_peak_memory_bytes=role_peaks,
            bottleneck_role=bottleneck_role,
            cost_selection_mode=self.cost_selection_mode,
            selection_cost_ms=selection_cost_ms,
            planner_metadata={
                "candidate_count": len(self.candidates),
                "feasible_count": len(feasible),
                "gradient_edge_count": sum(
                    edge.edge_type == EdgeType.GRADIENT for edge in graph.edges
                ),
                "version_edge_count": sum(
                    edge.edge_type == EdgeType.VERSION for edge in graph.edges
                ),
                "optimizer_action": optimizer_action,
                "memory_model": self.memory_model.to_dict(),
                "cost_model": {
                    "selection_mode": self.cost_selection_mode.value,
                    "selected_score_ms": selection_cost_ms,
                    "fastest_feasible_ms": fastest_feasible_ms,
                    "relative_slowdown_tolerance": self.relative_slowdown_tolerance,
                    "selected_slowdown_percent": selected_slowdown_percent,
                    "performance_band_policies": [
                        item[0].policy.value for item in performance_band
                    ],
                    "pareto_policies": pareto_policies,
                    "memory_tiebreak": (
                        "global_peak_then_sum_role_peaks"
                        if self.cost_selection_mode
                        == CostSelectionMode.PERFORMANCE_BAND_MEMORY
                        else None
                    ),
                    "candidates": {
                        item.policy.value: {
                            **item.cost_summary(),
                            "selection_cost_ms": item.selection_cost_ms(
                                self.cost_selection_mode
                            ),
                        }
                        for item in self.candidates
                    },
                },
            },
        )
