"""Minimal intermediate representation for multi-component distillation schedules.

The IR intentionally separates logical operations from physical parameter state.
Real-Score, Fake-Score, and Critic may therefore be distinct logical nodes while
sharing one physical backbone and selecting mutually-exclusive adapter modes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


class _StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class OperationType(_StringEnum):
    FORWARD = "forward"
    BACKWARD = "backward"
    OPTIMIZER_STEP = "optimizer_step"
    EMA_UPDATE = "ema_update"
    TRANSFER = "transfer"
    REPLAY = "replay"


class EdgeType(_StringEnum):
    DATA = "data"
    GRADIENT = "gradient"
    STATE = "state"
    VERSION = "version"
    CONTROL = "control"
    CONDITIONAL = "conditional"
    FREQUENCY = "frequency"


class GradMode(_StringEnum):
    NO_GRAD = "no_grad"
    GRAD = "grad"
    EXTERNAL_GRAD = "external_grad"


class GradientStrategy(_StringEnum):
    FULL_GRAPH = "full_graph"
    CACHE_INPUT_GRADIENT = "cache_input_gradient"
    GENERATOR_REPLAY = "generator_replay"


@dataclass(frozen=True)
class PhysicalState:
    """One physical parameter allocation, possibly exposed as many components."""

    name: str
    parameter_bytes: int
    trainable_bytes: int = 0
    optimizer_bytes: int = 0
    ema_bytes: int = 0
    modes: Tuple[str, ...] = ()
    shardable: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in (
            "parameter_bytes",
            "trainable_bytes",
            "optimizer_bytes",
            "ema_bytes",
        ):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must be non-negative")

    @property
    def total_training_bytes(self) -> int:
        return self.parameter_bytes + self.optimizer_bytes + self.ema_bytes


@dataclass(frozen=True)
class OperationNode:
    name: str
    logical_component: str
    physical_state: Optional[str]
    operation_type: OperationType
    grad_mode: GradMode = GradMode.NO_GRAD
    adapter_mode: Optional[str] = None
    update_frequency: int = 1
    duration_ms: float = 0.0
    activation_bytes: int = 0
    temporary_bytes: int = 0
    parameter_version: Optional[str] = None
    condition: Optional[str] = None
    nested_position: Tuple[int, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.update_frequency <= 0:
            raise ValueError("update_frequency must be positive")
        if self.duration_ms < 0:
            raise ValueError("duration_ms must be non-negative")
        if self.activation_bytes < 0 or self.temporary_bytes < 0:
            raise ValueError("memory demands must be non-negative")


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    edge_type: EdgeType
    tensor_bytes: int = 0
    name: Optional[str] = None
    condition: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.tensor_bytes < 0:
            raise ValueError("tensor_bytes must be non-negative")


@dataclass
class ExecutionGraph:
    """Conditional DAG plus a separate physical-state sharing table."""

    physical_states: Dict[str, PhysicalState] = field(default_factory=dict)
    nodes: Dict[str, OperationNode] = field(default_factory=dict)
    edges: List[Edge] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_physical_state(self, state: PhysicalState) -> None:
        if state.name in self.physical_states:
            raise ValueError(f"duplicate physical state: {state.name}")
        self.physical_states[state.name] = state

    def add_node(self, node: OperationNode) -> None:
        if node.name in self.nodes:
            raise ValueError(f"duplicate operation node: {node.name}")
        self.nodes[node.name] = node

    def add_edge(self, edge: Edge) -> None:
        self.edges.append(edge)

    def validate(self) -> None:
        for node in self.nodes.values():
            if node.physical_state is None:
                continue
            if node.physical_state not in self.physical_states:
                raise ValueError(
                    f"node {node.name} references missing physical state "
                    f"{node.physical_state}"
                )
            state = self.physical_states[node.physical_state]
            if (
                node.adapter_mode is not None
                and state.modes
                and node.adapter_mode not in state.modes
            ):
                raise ValueError(
                    f"node {node.name} selects unknown mode {node.adapter_mode!r} "
                    f"for state {state.name}"
                )
        for edge in self.edges:
            if edge.source not in self.nodes or edge.target not in self.nodes:
                raise ValueError(
                    f"edge {edge.source}->{edge.target} references a missing node"
                )
        self.topological_order()  # Raises on a cycle.

    def topological_order(self) -> List[str]:
        indegree = {name: 0 for name in self.nodes}
        outgoing: Dict[str, List[str]] = {name: [] for name in self.nodes}
        for edge in self.edges:
            # Every edge is a scheduling dependency. Conditional/frequency edges
            # remain in the static superset DAG and are filtered by a runtime.
            if edge.source not in indegree or edge.target not in indegree:
                raise ValueError(
                    f"edge {edge.source}->{edge.target} references a missing node"
                )
            indegree[edge.target] += 1
            outgoing[edge.source].append(edge.target)
        ready = [name for name in self.nodes if indegree[name] == 0]
        order: List[str] = []
        while ready:
            name = ready.pop(0)
            order.append(name)
            for target in outgoing[name]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    ready.append(target)
        if len(order) != len(self.nodes):
            cyclic = sorted(name for name, degree in indegree.items() if degree)
            raise ValueError(f"execution graph contains a cycle: {cyclic}")
        return order

    def logical_components_for_state(self, state_name: str) -> Tuple[str, ...]:
        return tuple(
            sorted(
                {
                    node.logical_component
                    for node in self.nodes.values()
                    if node.physical_state == state_name
                }
            )
        )

    def parameter_bytes(self, *, shared_state_aware: bool = True) -> int:
        """Estimate physical bytes with or without logical/physical de-duplication.

        The unaware estimate counts one copy per logical component, not per
        operation, mirroring a model-centric runtime that treats Real Score,
        Fake Score, and Critic as independent models.
        """

        if shared_state_aware:
            used_states = {
                node.physical_state
                for node in self.nodes.values()
                if node.physical_state is not None
            }
            return sum(
                self.physical_states[name].parameter_bytes for name in used_states
            )
        logical_allocations = {
            (node.logical_component, node.physical_state)
            for node in self.nodes.values()
            if node.physical_state is not None
            # Transfer/synchronization operations consume an existing physical
            # state; they are not another naive logical model allocation.
            and node.operation_type != OperationType.TRANSFER
        }
        return sum(
            self.physical_states[state_name].parameter_bytes
            for _, state_name in logical_allocations
        )

    def to_dict(self) -> Dict[str, Any]:
        def enum_value(value: Any) -> Any:
            return value.value if isinstance(value, Enum) else value

        return {
            "metadata": dict(self.metadata),
            "physical_states": {
                name: {
                    "name": state.name,
                    "parameter_bytes": state.parameter_bytes,
                    "trainable_bytes": state.trainable_bytes,
                    "optimizer_bytes": state.optimizer_bytes,
                    "ema_bytes": state.ema_bytes,
                    "modes": list(state.modes),
                    "shardable": state.shardable,
                    "metadata": dict(state.metadata),
                }
                for name, state in self.physical_states.items()
            },
            "nodes": {
                name: {
                    "name": node.name,
                    "logical_component": node.logical_component,
                    "physical_state": node.physical_state,
                    "operation_type": enum_value(node.operation_type),
                    "grad_mode": enum_value(node.grad_mode),
                    "adapter_mode": node.adapter_mode,
                    "update_frequency": node.update_frequency,
                    "duration_ms": node.duration_ms,
                    "activation_bytes": node.activation_bytes,
                    "temporary_bytes": node.temporary_bytes,
                    "parameter_version": node.parameter_version,
                    "condition": node.condition,
                    "nested_position": list(node.nested_position),
                    "metadata": dict(node.metadata),
                }
                for name, node in self.nodes.items()
            },
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "edge_type": enum_value(edge.edge_type),
                    "tensor_bytes": edge.tensor_bytes,
                    "name": edge.name,
                    "condition": edge.condition,
                    "metadata": dict(edge.metadata),
                }
                for edge in self.edges
            ],
        }
