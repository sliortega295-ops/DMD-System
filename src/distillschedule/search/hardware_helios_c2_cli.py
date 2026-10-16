"""Unified workloads hardware-helios-c2 branch; CPU control only, no run verb."""
from __future__ import annotations

import argparse
from pathlib import Path
import json

from . import hardware_helios_c2 as adapter
from .model_search import search_plans

c = adapter.control


def main(argv=None):
    parser = argparse.ArgumentParser(prog="distillctl workloads hardware-helios-c2", description=__doc__)
    subs = parser.add_subparsers(dest="action", required=True)
    source = subs.add_parser("prepare-source", help="Create one exact local 8fc-derived provenance commit; no GPU")
    source.add_argument("--parent-root", type=Path, required=True)
    source.add_argument("--destination", type=Path, required=True)
    source.add_argument("--allow-local-provenance-commit", action="store_true")
    source.add_argument("--output", type=Path, required=True)
    bind = subs.add_parser("bind-deployment")
    bind.add_argument("--legacy-deployment", type=Path, required=True)
    bind.add_argument("--source-proof", type=Path, required=True)
    bind.add_argument("--repo-root", type=Path, required=True)
    bind.add_argument("--output", type=Path, required=True)
    rebind = subs.add_parser("rebind-assets", help="Rehash original relocated assets and bind them to the new source; no GPU")
    rebind.add_argument("--source-root", type=Path, required=True)
    rebind.add_argument("--legacy-gate", type=Path, required=True)
    rebind.add_argument("--asset-root", type=Path, required=True)
    rebind.add_argument("--arm", choices=tuple(adapter.runtime.POLICIES), required=True)
    rebind.add_argument("--output", type=Path, required=True)
    inspect = subs.add_parser("inspect")
    inspect.add_argument("--deployment", type=Path, required=True)
    compile = subs.add_parser("compile", help="Explicit arm for fresh calibration; selection comes from search")
    compile.add_argument("--deployment", type=Path, required=True)
    compile.add_argument("--arm", choices=tuple(adapter.runtime.POLICIES), required=True)
    compile.add_argument("--mode", choices=("correctness", "timing"), required=True)
    compile.add_argument("--run-id", required=True)
    compile.add_argument("--correctness", type=Path)
    compile.add_argument("--enable-native-c2", action="store_true")
    compile.add_argument("--output", type=Path, required=True)
    validate = subs.add_parser("validate")
    validate.add_argument("--plan", type=Path, required=True)
    dry = subs.add_parser("dry-run", help="Validate exact source/plan and render launch; never invokes shell/worker")
    dry.add_argument("--plan", type=Path, required=True)
    dry.add_argument("--verify-assets", action="store_true", help="Also perform real native CPU full-hash asset and correctness preflight")
    audit = subs.add_parser("admit", help="Reopen all native raw receipts; old Full5 summaries fail closed")
    audit.add_argument("--plan", type=Path, required=True)
    audit.add_argument("--summary", type=Path, required=True)
    assemble = subs.add_parser("assemble-costs")
    assemble.add_argument("--deployment", type=Path, required=True)
    assemble.add_argument("--observations", type=Path, required=True, help="JSON object with observations: [{plan:{path,sha256},summary:{path,sha256}}]")
    assemble.add_argument("--output-dir", type=Path, required=True)
    search = subs.add_parser("search", help="Use common model_plan hierarchical search and validate selected native binding")
    search.add_argument("--deployment", type=Path, required=True)
    search.add_argument("--model", type=Path, required=True, help="Problem/cost_model saved by assemble-costs")
    search.add_argument("--correctness-map", type=Path, required=True, help="Object mapping native arms to same-arm correctness path/hash refs")
    search.add_argument("--run-id", required=True)
    search.add_argument("--enable-native-c2", action="store_true")
    search.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == "prepare-source":
        result = adapter.prepare_source(args.parent_root, args.destination, allow_local_provenance_commit=args.allow_local_provenance_commit)
    elif args.action == "bind-deployment":
        result = adapter.bind_deployment(c.read(args.legacy_deployment), c.reference(args.source_proof), repo_root=args.repo_root)
    elif args.action == "rebind-assets":
        result = adapter._native_cpu(args.source_root, "rebind-assets", {"legacy_gate": c.reference(args.legacy_gate), "asset_root": str(args.asset_root.resolve()), "arm": args.arm})
    elif args.action == "inspect":
        result = adapter.inspect_deployment(c.read(args.deployment))
    elif args.action == "compile":
        result = adapter.compile_plan(c.read(args.deployment), arm=args.arm, mode=args.mode, run_id=args.run_id,
            explicit_opt_in=args.enable_native_c2, correctness=c.reference(args.correctness) if args.correctness else None)
    elif args.action == "validate":
        plan = adapter.validate_plan(c.read(args.plan))
        result = {"status": "PASS_CPU_SOURCE_AND_MODEL_BINDING", "plan_sha256": plan["plan_sha256"], "gpu_execution": "NOT_RUN"}
    elif args.action == "dry-run":
        result = adapter.dry_run(c.reference(args.plan), verify_assets=args.verify_assets)
    elif args.action == "admit":
        plan = adapter.validate_plan(c.read(args.plan))
        observation = {"plan": c.reference(args.plan), "summary": c.reference(args.summary)}
        result = adapter.audit_plan_run(observation["plan"], observation["summary"])
    elif args.action == "assemble-costs":
        problem, costs = adapter.assemble_problem_and_costs(c.read(args.deployment), c.read(args.observations)["observations"], output_dir=args.output_dir / "components")
        result = {"problem": problem.to_dict(), "cost_model": costs.to_dict(), "gpu_execution": "NOT_RUN", "claimable_speedup": False}
        c.write_new(args.output_dir / "model.json", result)
    else:
        d, model = c.read(args.deployment), c.read(args.model)
        problem, costs = adapter.PlanningProblem.from_dict(model["problem"]), adapter.CostModel.from_dict(model["cost_model"])
        adapter.validate_model(problem, costs, d)
        prediction = search_plans(problem, costs, algorithm="hierarchical")
        c.require(prediction.get("predicted_best"), "UNCALIBRATED: no native C2 candidate has admissible costs")
        arm = prediction["predicted_best"]["assignment"]["arm"]
        result = adapter.lower_selection(prediction, problem, costs, d, run_id=args.run_id,
            correctness=c.read(args.correctness_map)[arm], explicit_opt_in=args.enable_native_c2)
    if getattr(args, "output", None):
        c.write_new(args.output, result)
        print(json.dumps({"output": str(args.output.resolve()), "sha256": c.sha(args.output), "gpu_execution": "NOT_RUN"}))
    else:
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
