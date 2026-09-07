"""Model-based planning extension; native execution remains capability-gated."""
from __future__ import annotations

from distillschedule.workloads.best_known.registry import (
    CANONICAL_IDS, _object, fingerprint, git_head, load_profile, repository_root,
    write_new_json,
)
from .contracts import require
from .cost_model import CostModel
from .implementation_library import implementation_inventory
from .model_search import search_plans
from .plan_ir import PlanningProblem


def model_based_search(args):
    require(args.workload in CANONICAL_IDS, "unknown workload")
    require(not args.resume, "model-based execution journal/resume is not yet integrated; preserve the old run")
    require(not args.historical_profile_prior,
            "historical profile hints cannot substitute for source-bound operation calibration")
    root = repository_root()
    profile = load_profile(args.workload)
    require(not (args.problem and args.planning_layout), "use either adapter layout or saved problem, not both")
    if args.problem is None and args.planning_layout is None:
        require(args.dry_run, "model-based search requires an adapter-produced --problem")
        from .wan_model_adapter import capability_report
        from .wan_implementation import WAN
        return {"status": "PASS_CPU_CAPABILITY_INVENTORY", "workload": args.workload,
                "source_head": git_head(root), "canonical_profile_sha256": fingerprint(profile),
                "semantics": profile["semantics"], "implementation_library": implementation_inventory(root),
                "gpu_execution": "NOT_RUN", "claimable_speedup": False,
                "native_lowering": "WAN_GENERATED_COMPOSITION_AFTER_CALIBRATION" if args.workload == WAN else "NOT_IMPLEMENTED_FOR_GENERATED_PLANS",
                "native_adapter": capability_report() if args.workload == WAN else None,
                "next_gate": "ADAPTER_PROBLEM_AND_MATCHED_OPERATION_CALIBRATION_REQUIRED"}
    if args.planning_layout:
        from .wan_model_adapter import build_problem
        from .wan_implementation import WAN
        require(args.workload == WAN, "native layout adapter not yet implemented for this workload")
        problem = build_problem(args.planning_layout)
    else:
        problem = PlanningProblem.from_dict(_object(args.problem))
    require(problem.workload == args.workload, "planning problem workload mismatch")
    if problem.evidence_scope != "CPU_FIXTURE_ONLY":
        require(problem.binding["source_head"] == git_head(root), "planning problem belongs to another source HEAD")
        require(problem.graph.metadata.get("canonical_profile_sha256") == fingerprint(profile),
                "planning graph not bound to the canonical workload contract")
    if args.cost_model is None:
        require(args.dry_run, "model-based search requires --cost-model")
        return {"status": "UNCALIBRATED", "problem_sha256": problem.digest,
                "required_cost_keys": sorted({r.cost_key for r in problem.rules}),
                "gpu_execution": "NOT_RUN", "claimable_speedup": False}
    costs = CostModel.from_dict(_object(args.cost_model))
    costs.verify_receipts()
    result = search_plans(problem, costs, seed=args.search_seed, max_proposals=args.max_proposals,
                          timeout_seconds=args.search_timeout, exhaustive=args.exhaustive)
    from .wan_model_adapter import ADAPTER
    result["native_lowering"] = ("WAN_COMPOSITION_AVAILABLE_AFTER_NATIVE_INPUT_VALIDATION"
        if problem.lowering.get("adapter") == ADAPTER else "NOT_IMPLEMENTED_FOR_GENERATED_PLANS")
    result["native_trial_budget"] = {"total": 8, "baseline": 1, "incumbent": 1,
                                     "new_candidates": 5, "confirmation": 1,
                                     "actual_runs": 0, "status": "PLANNED_MODEL_EXECUTION_CONTROLLER_PENDING"}
    if args.emit_native_plan:
        from .wan_model_adapter import lower_prediction
        plan = lower_prediction(result, problem, costs, run_id=args.run_id,
                                assets_path=args.assets_config)
        write_new_json(args.emit_native_plan, plan)
        result["native_lowering"] = "PASS_CPU_LOWERING_GPU_NOT_RUN"
        result["native_plan_path"] = str(args.emit_native_plan)
    return result
