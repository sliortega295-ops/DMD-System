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
    execute_native = getattr(args, "execute_native", False)
    validation_dir = getattr(args, "validation_dir", None)
    require(not args.resume or execute_native, "resume applies only to native validation journals")
    require(not execute_native or (not args.dry_run and validation_dir and args.assets_config),
            "native execution needs validation-dir/assets-config and cannot be dry-run")
    require(not (execute_native and args.emit_native_plan), "emit-only and execute-native are separate actions")
    require(validation_dir is None or execute_native, "validation-dir requires execute-native")
    require(not args.historical_profile_prior,
            "historical profile hints cannot substitute for source-bound operation calibration")
    budget = getattr(args, "max_trials", 8)
    require(type(budget) is int and 3 <= budget <= 8,
            "model-based total timing budget must be 3..8 including anchors/confirmation")
    root = repository_root()
    profile = load_profile(args.workload)
    from .model_adapters import adapter_for, SUPPORTED_WORKLOADS
    supported = args.workload in SUPPORTED_WORKLOADS
    adapter = adapter_for(args.workload) if supported else None
    require(not (args.problem and args.planning_layout), "use either adapter layout or saved problem, not both")
    if args.problem is None and args.planning_layout is None:
        require(args.dry_run, "model-based search requires an adapter-produced --problem")
        return {"status": "PASS_CPU_CAPABILITY_INVENTORY", "workload": args.workload,
                "source_head": git_head(root), "canonical_profile_sha256": fingerprint(profile),
                "semantics": profile["semantics"], "implementation_library": implementation_inventory(root),
                "gpu_execution": "NOT_RUN", "claimable_speedup": False,
                "native_lowering": "GENERATED_COMPOSITION_AFTER_CALIBRATION" if supported else "NOT_IMPLEMENTED_FOR_GENERATED_PLANS",
                "native_adapter": adapter.capability_report() if supported else None,
                "next_gate": "ADAPTER_PROBLEM_AND_MATCHED_OPERATION_CALIBRATION_REQUIRED"}
    if args.planning_layout:
        require(adapter is not None, "native layout adapter not yet implemented for this workload")
        problem = adapter.build_problem(args.planning_layout)
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
    result["native_lowering"] = ("COMPOSITION_AVAILABLE_AFTER_NATIVE_INPUT_VALIDATION"
        if adapter and problem.lowering.get("adapter") in getattr(adapter,"ADAPTERS",(adapter.ADAPTER,)) else "NOT_IMPLEMENTED_FOR_GENERATED_PLANS")
    result["native_trial_budget"] = {"total": budget, "baseline": 1, "incumbent": 1,
                                     "new_candidates": budget - 3, "confirmation": 1,
                                     "actual_runs": 0, "status": "REQUIRES_EXPLICIT_EXECUTE_NATIVE"}
    if execute_native:
        from .model_validation import ModelValidationSession
        from .execution import NativeTrialRunner
        from distillschedule.workloads.best_known.registry import load_assets
        session = ModelValidationSession(problem=problem, costs=costs, prediction=result,
            output=validation_dir, max_trials=budget, resume=args.resume)
        from .registry import candidate_profile, get_candidate
        assets = load_assets(args.assets_config, candidate_profile(get_candidate(args.workload, session.seeds[0])))
        require(fingerprint(assets["assets"]) == problem.binding["asset_fingerprint"],
                "native validation assets differ from calibration")
        runner = NativeTrialRunner(workload=args.workload, identity=session.identity, assets=assets,
            output=validation_dir, run_prefix=args.run_id, reference_id=session.seeds[0],
            bootstrap_peer=session.seeds[1])
        return session.run(runner)
    if args.emit_native_plan:
        require(adapter is not None, "native adapter not implemented")
        plan = adapter.lower_prediction(result, problem, costs, run_id=args.run_id,
                                assets_path=args.assets_config)
        write_new_json(args.emit_native_plan, plan)
        result["native_lowering"] = "PASS_CPU_LOWERING_GPU_NOT_RUN"
        result["native_plan_path"] = str(args.emit_native_plan)
    return result
