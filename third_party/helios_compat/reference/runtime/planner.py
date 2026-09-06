"""Profile-driven budget planner for the minimal DistillSchedule IR."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .ir import ExecutionGraph, GradientStrategy


@dataclass(frozen=True)
class HardwareBudget:
    gpu_memory_bytes: int
    gpu_count: int = 1
    cpu_memory_bytes: int = 0
    pinned_memory_bytes: int = 0
    h2d_bandwidth_bytes_per_s: float = 0.0
    d2h_bandwidth_bytes_per_s: float = 0.0
    network_bandwidth_bytes_per_s: float = 0.0

    def __post_init__(self) -> None:
        if self.gpu_memory_bytes <= 0 or self.gpu_count <= 0:
            raise ValueError("GPU memory and GPU count must be positive")


@dataclass(frozen=True)
class StrategyProfile:
    name: str
    gradient_strategy: GradientStrategy
    peak_memory_bytes: int
    iteration_time_ms: float
    required_gpus: int = 1
    h2d_bytes: int = 0
    d2h_bytes: int = 0
    recompute_time_ms: float = 0.0
    backend: str = "vanilla"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.peak_memory_bytes < 0 or self.iteration_time_ms < 0:
            raise ValueError("profile memory/time must be non-negative")
        if self.required_gpus <= 0:
            raise ValueError("required_gpus must be positive")


@dataclass(frozen=True)
class PlanAction:
    kind: str
    node: Optional[str] = None
    physical_state: Optional[str] = None
    adapter_mode: Optional[str] = None
    asynchronous: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionPlan:
    profile: StrategyProfile
    actions: Tuple[PlanAction, ...]
    rejected_profiles: Mapping[str, str]
    planner_metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def gradient_strategy(self) -> GradientStrategy:
        return self.profile.gradient_strategy


class NoFeasiblePlanError(RuntimeError):
    pass


class BudgetPlanner:
    """Enumerate measured profiles and emit state-aware execution actions.

    This is deliberately simple: it chooses the fastest measured profile that
    fits the supplied hardware budget. The important MVP property is that plan
    construction follows physical-state identity, so logical Real/Fake/Critic
    nodes do not trigger redundant model materializations.
    """

    def __init__(self, profiles: Iterable[StrategyProfile]) -> None:
        self.profiles = tuple(profiles)
        if not self.profiles:
            raise ValueError("at least one strategy profile is required")
        names = [profile.name for profile in self.profiles]
        if len(names) != len(set(names)):
            raise ValueError("strategy profile names must be unique")

    def plan(self, graph: ExecutionGraph, budget: HardwareBudget) -> ExecutionPlan:
        graph.validate()
        feasible: List[StrategyProfile] = []
        rejected: Dict[str, str] = {}
        for profile in self.profiles:
            if profile.required_gpus > budget.gpu_count:
                rejected[profile.name] = (
                    f"requires {profile.required_gpus} GPUs, budget has {budget.gpu_count}"
                )
            elif profile.peak_memory_bytes > budget.gpu_memory_bytes:
                rejected[profile.name] = (
                    f"peak {profile.peak_memory_bytes} exceeds budget "
                    f"{budget.gpu_memory_bytes}"
                )
            elif (
                profile.h2d_bytes > 0
                and budget.h2d_bandwidth_bytes_per_s <= 0
                and profile.backend == "offload"
            ):
                rejected[profile.name] = "offload profile requires measured H2D bandwidth"
            else:
                feasible.append(profile)
        if not feasible:
            reasons = "; ".join(f"{name}: {reason}" for name, reason in rejected.items())
            raise NoFeasiblePlanError(f"no strategy fits the hardware budget ({reasons})")

        chosen = min(
            feasible,
            key=lambda profile: (
                profile.iteration_time_ms,
                profile.peak_memory_bytes,
                profile.name,
            ),
        )
        actions = self._actions(graph, chosen)
        return ExecutionPlan(
            profile=chosen,
            actions=tuple(actions),
            rejected_profiles=rejected,
            planner_metadata={
                "candidate_count": len(self.profiles),
                "feasible_count": len(feasible),
                "shared_parameter_bytes": graph.parameter_bytes(
                    shared_state_aware=True
                ),
                "logical_model_parameter_bytes": graph.parameter_bytes(
                    shared_state_aware=False
                ),
            },
        )

    @staticmethod
    def _actions(
        graph: ExecutionGraph, profile: StrategyProfile
    ) -> List[PlanAction]:
        order = graph.topological_order()
        last_use: Dict[str, int] = {}
        for index, node_name in enumerate(order):
            state = graph.nodes[node_name].physical_state
            if state is not None:
                last_use[state] = index

        resident: set[str] = set()
        active_mode: Dict[str, Optional[str]] = {}
        actions: List[PlanAction] = []
        boundary_inserted = False
        for index, node_name in enumerate(order):
            node = graph.nodes[node_name]
            state = node.physical_state
            if state is not None and state not in resident:
                actions.append(
                    PlanAction(
                        kind="materialize",
                        physical_state=state,
                        asynchronous=False,
                        metadata={"backend": profile.backend},
                    )
                )
                resident.add(state)
            if (
                state is not None
                and node.adapter_mode is not None
                and active_mode.get(state) != node.adapter_mode
            ):
                actions.append(
                    PlanAction(
                        kind="set_adapter_mode",
                        physical_state=state,
                        adapter_mode=node.adapter_mode,
                    )
                )
                active_mode[state] = node.adapter_mode

            if (
                not boundary_inserted
                and bool(node.metadata.get("gradient_boundary"))
            ):
                actions.append(
                    PlanAction(
                        kind=profile.gradient_strategy.value,
                        node=node_name,
                        metadata={"profile": profile.name},
                    )
                )
                boundary_inserted = True
            actions.append(
                PlanAction(
                    kind="execute",
                    node=node_name,
                    physical_state=state,
                    adapter_mode=node.adapter_mode,
                    metadata={
                        "operation_type": node.operation_type.value,
                        "grad_mode": node.grad_mode.value,
                    },
                )
            )
            if state is not None and last_use[state] == index:
                actions.append(
                    PlanAction(kind="release", physical_state=state)
                )
                resident.remove(state)
                active_mode.pop(state, None)
        return actions
