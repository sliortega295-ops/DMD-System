"""CPU-only parameterized-cycle front door, shared with distillctl workloads.

The input format intentionally accepts only synthetic CPU costs. A real model
needs its protocol/shape adapter and source-bound calibration before native
lowering. This command must not manufacture support for new GPU resolutions,
LoRA ranks, DiffSynth, stochastic exits, or cross-role training protocols.
"""
from __future__ import annotations

from typing import Any, Mapping

from distillschedule.runtime.parameterized_cycle import build_parameterized_cycle_graph
from distillschedule.workloads.best_known.registry import fingerprint
from .contracts import require
from .cost_model import CostModel, OperationCost
from .frontier_schedule import schedule_frontier
from .plan_ir import BindingRule, StateAllocation, Lifetime

SCHEMA = "distillschedule.cycle_planner_input.v1"


def plan_cycle(document: Mapping[str, Any], *, max_states: int = 50000,
               beam_width: int = 0, timeout_seconds: float = 60, prune: bool = True) -> dict[str, Any]:
    fields = {"schema_version", "scope", "protocol", "source_steps", "initial_versions", "ema_after",
              "memory_limits", "static_bytes_per_rank", "action_costs", "retained_output_bytes"}
    require(set(document) == fields, "cycle input has missing or unknown fields")
    require(document["schema_version"] == SCHEMA and document["scope"] == "CPU_FIXTURE_ONLY",
            "plan-cycle accepts explicit CPU fixtures only; real execution requires a workload adapter")
    graph = build_parameterized_cycle_graph(source_steps=tuple(document["source_steps"]),
        initial_versions=document["initial_versions"], ema_after=tuple(document["ema_after"]),
        protocol=document["protocol"])
    require(isinstance(document["memory_limits"], Mapping), "memory limits must be a mapping")
    limits = {int(r): b for r, b in document["memory_limits"].items()}
    require(len(limits) == len(document["memory_limits"]), "duplicate normalized rank")
    ranks = tuple(sorted(limits))
    static = document["static_bytes_per_rank"]
    require(type(static) is int and static >= 0, "invalid static allocation")
    raw_costs = document["action_costs"]
    require(set(raw_costs) == {o.operation_id for o in graph.operations}, "every action needs its own explicit cost")
    require(set(document["retained_output_bytes"]) == {f"r{i}" for i in range(len(document["source_steps"]))},
            "each rollout needs explicit retained bytes")
    binding = {"source_head": "0" * 40, **{k: fingerprint(document) for k in (
        "hardware_fingerprint", "environment_fingerprint", "asset_fingerprint",
        "semantic_fingerprint", "precision_fingerprint")}}
    records = []
    bindings = {}
    for op in graph.operations:
        row = raw_costs[op.operation_id]
        require(set(row) == {"duration_ns", "workspace_bytes_per_rank"}, "invalid action cost fields")
        records.append(OperationCost(op.operation_id, (row["duration_ns"],), row["workspace_bytes_per_rank"],
                                     {"kind": "CPU_FIXTURE_ONLY"}))
        bindings[op.operation_id] = BindingRule(op.operation_id, {}, op.operation_id, ranks,
                                               tuple(f"gpu:{r}" for r in ranks), ("base",))
    costs = CostModel(binding, tuple(records), scope="CPU_FIXTURE_ONLY")
    states = (StateAllocation("base", ranks, static, ("native_runtime",), True, "CPU_FIXTURE_ONLY"),)
    lives = tuple(Lifetime("retained:" + key, key, ("c" + key[1:],), ranks, value,
                           allocation_point="producer_exit") for key, value in document["retained_output_bytes"].items())
    result = schedule_frontier(graph, bindings, costs, states=states, lifetimes=lives,
                              memory_limits=limits, max_states=max_states, beam_width=beam_width,
                              timeout_seconds=timeout_seconds, prune=prune)
    result["program"] = graph.to_dict()
    result["input_sha256"] = fingerprint(document)
    result["native_lowering"] = "NOT_IMPLEMENTED_FOR_THIS_CPU_INPUT"
    return result
