"""Source-bound planning at the existing native Wan worker's stage boundaries.

This first native adapter keeps the already implemented clustered G1 -> 5G ->
5(C+EMA) schedule. It composes placement and kernel parameters, but does not
pretend that the worker supports arbitrary graph reordering or pool splits.
Memory input must be a measured current-source layout receipt, not estimated
from parameter counts or historical winner timings.
"""
from __future__ import annotations

from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import Any, Mapping

from distillschedule.core.enums import GradientMode, OperationKind
from distillschedule.core.graph import Dependency, ExecutionGraph, Operation
from distillschedule.workloads.best_known.registry import (
    _object, file_sha256, fingerprint, git_head, load_profile, repository_root,
)
from .contracts import require, semantic_contract
from .plan_ir import BindingRule, PlanningProblem, StateAllocation, nonnegative, validate_binding
from .wan_implementation import PRECISION, WAN, WanImplementation

ADAPTER = "wan_native_clustered_stages_v1"
DOMAINS = {"placement": ("fsdp", "resident"), "chunk_tokens": ("512", "1024", "2048"),
           "lora_epilogue": ("off", "on")}
RANKS = tuple(range(16))


def implementations() -> tuple[WanImplementation, ...]:
    return (WanImplementation("fsdp", 512, False), *(
        WanImplementation("resident", chunk, epilogue)
        for chunk, epilogue in product((512, 1024, 2048), (False, True))))


def native_graph() -> ExecutionGraph:
    profile = load_profile(WAN)
    initial = profile["contract"]["initial_versions"]
    g, c, ema = (initial[k] for k in ("generator", "critic", "ema"))
    operations, dependencies = [], []
    ends = {}
    previous = None

    def append(cycle, suffix, kind, grad, reads, writes, inventory):
        nonlocal previous
        key = f"cycle{cycle}.{suffix}"
        operations.append(Operation(key, 10, "native_runtime", kind, grad,
            adapter_mode="native_scoped_role_boundaries", required_versions=reads, produced_versions=writes,
            metadata={"cycle": cycle, "stage": suffix, "inventory": inventory,
                      "scope": "NATIVE_COMPOSITE_OPERATION_ENVELOPE"}))
        if previous is not None:
            dependencies.append(Dependency(previous, key, "native_order"))
        previous = key
        return key

    for cycle in (1, 2):
        append(cycle, "generator_update", OperationKind.GENERATOR_UPDATE, GradientMode.GRAD,
            {"generator": g, "critic": c}, {"generator": g + 1},
            {"generator_updates": 1, "generator_backward": 1, "score_forwards": 3,
             "rollout": "TRACE_DERIVED_GRAD_ROLLOUT_EXITS_AND_CALLS"})
        g += 1
        append(cycle, "critic_rollouts", OperationKind.CRITIC_ROLLOUT, GradientMode.NO_GRAD,
            {"generator": g}, {}, {"physical_rollout_batches": 5, "local_batch": 1,
                                   "rollout": "FIVE_TRACE_DERIVED_NOGRAD_ROLLOUTS"})
        for index in range(5):
            ends[str(cycle)] = append(cycle, f"critic{index}_update_ema", OperationKind.CRITIC_UPDATE,
                GradientMode.GRAD, {"generator": g, "critic": c, "ema": ema},
                {"critic": c + 1, "ema": ema + 1},
                {"source_step": 10 + index, "critic_forward": 1, "critic_backward": 1,
                 "critic_updates": 1, "ema_updates": 1})
            c, ema = c + 1, ema + 1
    return ExecutionGraph(tuple(operations), tuple(dependencies), {
        "adapter": ADAPTER, "initial_versions": initial, "cycle_ends": ends,
        "final_versions": {"generator": g, "critic": c, "ema": ema},
        "canonical_profile_sha256": fingerprint(profile),
        "native_worker": "scripts/distillschedule/probe_wan13b_two_cycle_timing_world16.py",
        "schedule": "G_UPDATE_THEN_5_ROLLOUTS_THEN_5_CRITIC_UPDATES_WITH_EMA",
        "macro_operation_costs_include_internal_compute_and_communication_overlap": True,
        "retained_outputs": "INCLUDED_IN_CALIBRATED_ADDITIONAL_LIVE_MEMORY_ENVELOPE",
        "end_step_exit_count": "VALIDATED_BY_EXISTING_RUNTIME_AGAINST_IMMUTABLE_OPERATION_TRACE",
    })


def operation_contract(operation: Operation, implementation: WanImplementation) -> dict[str, Any]:
    return {"adapter": ADAPTER, "operation_id": operation.operation_id,
            "stage_inventory": operation.metadata["inventory"], "implementation": implementation.assignment,
            "precision_fingerprint": PRECISION.digest,
            "memory_definition": "MAX_TOTAL_ALLOCATED_DURING_STAGE_MINUS_POST_SETUP_STATIC_BYTES",
            "timing_definition": "SYNCHRONIZED_NATIVE_STAGE_CALIBRATION_NOT_FORMAL_CYCLE_TIMING"}


def cost_key(operation: Operation, implementation: WanImplementation) -> str:
    return fingerprint(operation_contract(operation, implementation))


def capability_report() -> dict[str, Any]:
    from .wan_action_model_adapter import DOMAINS as ACTION_DOMAINS
    from .wan_action_costs import ADAPTER as ACTION_ADAPTER, action_graph
    actions = [replace(impl, rollout_window=window)
               for impl in implementations() for window in range(1, 6)]
    return {"adapter": ADAPTER, "workload": WAN, "decisions": DOMAINS,
            "native_implementations": [{"id": i.id, "assignment": i.assignment,
                "lowered_placement": i.native_placement, "native_gpu_validation": "NOT_RUN"} for i in implementations()],
            "baseline": implementations()[0].assignment,
            "incumbent": WanImplementation("resident", 2048, True).assignment,
            "precision_fingerprint": PRECISION.digest,
            "sharing": "FIXED_CURRENT_BACKEND; NO_UNSUPPORTED_SHARED_GENERATOR_CRITIC",
            "schedule": "LEGACY_MACRO_FIXED_NATIVE_CLUSTERED; SEE_PREPARED_ACTIONS_FOR_GENERATED_WINDOWS",
            "cost_calibration": "REQUIRED", "gpu_execution": "NOT_RUN",
            "prepared_actions": {
                "adapter": ACTION_ADAPTER, "decisions": ACTION_DOMAINS,
                "actions_per_two_cycles": len(action_graph().operations),
                "native_implementations": [dict(id=impl.id, assignment=impl.assignment,
                    lowered_placement=impl.native_placement) for impl in actions],
                "measurement_admission": "SOURCE_BOUND_NATIVE_RECEIPTS_REQUIRED",
                "inter_action_overlap": False, "claimable_speedup": False,
                "cost_scope": "REUSABLE_NATIVE_ACTIONS_WITH_EXPLICIT_RETAINED_OUTPUT_LIFETIMES"}}


def build_problem(layout_path: Path, *, root: Path | None = None) -> PlanningProblem:
    root = root or repository_root()
    layout = _object(layout_path)
    if layout.get("schema_version") == "distillschedule.wan_action_layout.v1":
        from .wan_action_model_adapter import build_problem as build_actions
        return build_actions(layout_path,root=root)
    require(layout.get("schema_version") == "distillschedule.wan_stage_layout.v1", "unsupported native layout receipt")
    require(layout.get("status") == "PASS" and layout.get("profiled") is False, "native layout receipt not passing/unprofiled")
    binding = layout["binding"]
    validate_binding(binding)
    require(binding["source_head"] == git_head(root), "layout calibration belongs to another HEAD")
    require(binding["precision_fingerprint"] == PRECISION.digest, "layout precision differs")
    require(binding["semantic_fingerprint"] == fingerprint(semantic_contract(load_profile(WAN))), "layout workload differs")
    require(layout.get("adapter") == ADAPTER, "native layout adapter mismatch")
    require(layout.get("memory_definition") == "POST_SETUP_STATIC; STAGE_COST_INCLUDES_ALL_ADDITIONAL_LIVE_MEMORY",
            "ambiguous native memory accounting")
    limits = {int(k): v for k, v in layout["memory_limits"].items()}
    require(set(limits) == set(RANKS), "world16 capacity required")
    graph = native_graph()
    states, rules = [], []
    rows = layout["implementations"]
    require(set(rows) == {i.id for i in implementations()}, "native layout implementation inventory incomplete")
    for impl in implementations():
        row = rows[impl.id]
        require(row.get("status") == "PASS", "native implementation layout failed")
        ranks = row["ranks"]
        require(len(ranks) == 16 and {r["rank"] for r in ranks} == set(RANKS), "layout must contain sixteen rank owners")
        require(all(r.get("status") == "PASS" and r.get("source_head") == binding["source_head"]
                    and r.get("precision_fingerprint") == PRECISION.digest for r in ranks), "layout rank source/precision gate failed")
        static = max(nonnegative(r["post_setup_static_bytes"], "static bytes") for r in ranks)
        require(static > 0, "native layout cannot have zero model/state bytes")
        # Max-per-rank is conservative, not a claim of identical allocation.
        states.append(StateAllocation(impl.id, RANKS, static, ("native_runtime",), False, fingerprint(row)))
        for op in graph.operations:
            rules.append(BindingRule(op.operation_id, impl.assignment, cost_key(op, impl), RANKS,
                tuple(f"gpu:{r}" for r in RANKS), (impl.id,), fingerprint(operation_contract(op, impl))))
    forbidden = ({"placement": "fsdp", "chunk_tokens": "1024"},
                 {"placement": "fsdp", "chunk_tokens": "2048"},
                 {"placement": "fsdp", "lora_epilogue": "on"})
    return PlanningProblem(WAN, binding, graph, DOMAINS, tuple(rules), tuple(states), (), limits,
        (WanImplementation("fsdp", 512, False).assignment, WanImplementation("resident", 2048, True).assignment),
        forbidden=forbidden, evidence_scope="MEASURED",
        lowering={"adapter": ADAPTER, "layout_receipt": str(layout_path.resolve()), "layout_sha256": file_sha256(layout_path)})


def validate_native_problem(problem: PlanningProblem) -> None:
    if problem.lowering.get("adapter") == "wan_prepared_actions_v2":
        from .wan_action_model_adapter import validate_native_problem as validate_actions
        return validate_actions(problem)
    require(problem.evidence_scope == "MEASURED", "CPU fixture may not be lowered to native execution")
    require(problem.lowering.get("adapter") == ADAPTER, "unimplemented generated native adapter")
    path = Path(problem.lowering["layout_receipt"])
    require(file_sha256(path) == problem.lowering["layout_sha256"], "native layout receipt changed")
    expected = build_problem(path)
    require(expected.digest == problem.digest, "native operation graph/binding changed; arbitrary graph lowering is forbidden")


def lower_prediction(result: Mapping[str, Any], problem: PlanningProblem, costs: Any,
                     *, run_id: str, assets_path: Path | None = None, mode: str = "correctness") -> dict[str, Any]:
    from .model_search import validate_saved_prediction
    from .execution import resolve_candidate
    validate_native_problem(problem)
    costs.verify_receipts()
    prediction = validate_saved_prediction(result, problem, costs)
    require(prediction["status"] == "PREDICTED_FEASIBLE", "predicted plan does not fit")
    impl = WanImplementation.from_assignment(prediction["assignment"])
    receipt = resolve_candidate(WAN, impl.id, assets_path=assets_path, run_id=run_id, mode=mode)
    receipt["model_plan"] = {"problem": problem.to_dict(), "cost_model": costs.to_dict(), "search_result": dict(result)}
    validate_lowered_plan(receipt)
    return receipt


def validate_lowered_plan(receipt: Mapping[str, Any]) -> None:
    from .cost_model import CostModel
    from .model_search import validate_saved_prediction
    model_plan = receipt["model_plan"]
    problem = PlanningProblem.from_dict(model_plan["problem"])
    validate_native_problem(problem)
    require(receipt.get("integration_head") == problem.binding["source_head"], "lowered source changed")
    require(receipt.get("workload_semantics_hash") == problem.binding["semantic_fingerprint"], "lowered workload semantics changed")
    if receipt.get("asset_config"):
        require(receipt.get("asset_fingerprint") == problem.binding["asset_fingerprint"], "lowered native assets differ from calibration")
    costs = CostModel.from_dict(model_plan["cost_model"])
    costs.verify_receipts()
    predicted = validate_saved_prediction(model_plan["search_result"], problem, costs)
    require(receipt["search_candidate"]["id"] == WanImplementation.from_assignment(predicted["assignment"]).id,
            "native plan did not lower the searched decision vector")
