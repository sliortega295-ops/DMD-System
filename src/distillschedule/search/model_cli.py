"""Model-based planning extension; native execution remains capability-gated."""
from __future__ import annotations

from distillschedule.workloads.best_known.registry import (
    CANONICAL_IDS, _object, fingerprint, git_head, load_profile, repository_root,
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
    if args.problem is None:
        require(args.dry_run, "model-based search requires an adapter-produced --problem")
        return {"status": "PASS_CPU_CAPABILITY_INVENTORY", "workload": args.workload,
                "source_head": git_head(root), "canonical_profile_sha256": fingerprint(profile),
                "semantics": profile["semantics"], "implementation_library": implementation_inventory(root),
                "gpu_execution": "NOT_RUN", "claimable_speedup": False,
                "native_lowering": "NOT_IMPLEMENTED_FOR_GENERATED_PLANS",
                "next_gate": "ADAPTER_PROBLEM_AND_MATCHED_OPERATION_CALIBRATION_REQUIRED"}
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
    result["native_lowering"] = "NOT_IMPLEMENTED_FOR_GENERATED_PLANS"
    result["native_trial_budget"] = {"total": 8, "baseline": 1, "incumbent": 1,
                                     "new_candidates": 5, "confirmation": 1,
                                     "actual_runs": 0}
    return result
