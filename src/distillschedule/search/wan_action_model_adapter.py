"""Lower measured Wan components and retained outputs to the common planner.

This consumes content-hashed native calibration fragments. It exposes all
implemented kernel/placement/window choices; missing component costs remain
UNCALIBRATED. A serial placement does not gain hypothetical pipeline overlap.
"""
from dataclasses import replace
from pathlib import Path

from distillschedule.workloads.best_known.registry import (
    _object, file_sha256, fingerprint, git_head, load_profile, repository_root,
)
from .contracts import require, semantic_contract
from .plan_ir import BindingRule, Lifetime, PlanningProblem, StateAllocation, validate_binding
from .wan_action_costs import ADAPTER, MEMORY_DEFINITION, RANKS, action_graph, component_contract
from .wan_calibration_evidence import validate_rank
from .wan_implementation import WAN, PRECISION, WanImplementation
from .wan_model_adapter import implementations

SCHEMA = "distillschedule.wan_action_layout.v1"
DOMAINS = {"placement":("fsdp","resident"),"chunk_tokens":("512","1024","2048"),
           "lora_epilogue":("off","on"),"rollout_window":("1","2","3","4","5")}


def build_problem(layout_path: Path, *, root: Path | None = None) -> PlanningProblem:
    root = root or repository_root()
    layout = _object(layout_path)
    require(layout.get("schema_version") == SCHEMA,"not a native action layout")
    require(layout.get("fragments"),"native component fragments required")
    identity, work, limits = None, None, None
    placements, outputs, evidence = {},{},[]
    for reference in layout["fragments"]:
        path = Path(reference["path"])
        require(file_sha256(path) == reference["sha256"],"native fragment hash changed")
        fragment = _object(path)
        require(fragment.get("schema_version") == "distillschedule.wan_action_layout_fragment.v1"
                and fragment.get("status") == "PASS" and fragment.get("profiled") is False
                and fragment.get("adapter") == ADAPTER
                and fragment.get("memory_definition") == MEMORY_DEFINITION,
                "invalid native action fragment")
        binding = fragment["binding"]
        validate_binding(binding)
        require(identity is None or identity == binding,"fragment source/environment/assets differ")
        identity = binding
        require(identity["source_head"] == git_head(root),"component calibration belongs to another HEAD")
        require(identity["precision_fingerprint"] == PRECISION.digest,"component precision differs")
        require(identity["semantic_fingerprint"] == fingerprint(semantic_contract(load_profile(WAN))),
                "component workload differs")
        local_limits = {int(k):v for k,v in fragment["memory_limits"].items()}
        require(set(local_limits) == set(RANKS),"world16 capacity observations missing")
        limits = local_limits if limits is None else {r:min(limits[r],local_limits[r]) for r in RANKS}
        for key,row in fragment["implementations"].items():
            impl = WanImplementation.parse(key)
            require(impl.rollout_window is not None and row.get("status") == "PASS","legacy/failed fragment")
            ranks = row["ranks"]
            require(len(ranks) == 16 and {r["rank"] for r in ranks} == set(RANKS),"sixteen rank owners required")
            for rank in ranks:
                require(validate_rank(rank,source_head=identity["source_head"],rank=rank["rank"],
                    placement=impl.native_placement) == impl,"native implementation mismatch")
                require(work is None or work == rank["work_signatures"],"physical tensor/call work differs")
                work = rank["work_signatures"]
                placements[impl.placement] = max(placements.get(impl.placement,0),rank["post_setup_static_bytes"])
                for life in rank["output_lifetimes"]:
                    outputs[life["producer"]] = max(outputs.get(life["producer"],0),life["bytes"])
        evidence.append(reference)
    require(set(placements) == {"fsdp","resident"},"matched placement static layouts required")
    require(len(outputs) == 10,"ten measured cross-action outputs required")
    graph = action_graph()
    graph = replace(graph,metadata={**graph.metadata,"canonical_profile_sha256":fingerprint(load_profile(WAN))})
    states = tuple(StateAllocation(f"native:{name}",RANKS,size,("native_runtime",),False,
        fingerprint({"placement":name,"sources":evidence})) for name,size in sorted(placements.items()))
    lives = tuple(Lifetime(key,key,(key.replace(".r",".c"),),RANKS,size,"producer_exit")
                  for key,size in sorted(outputs.items()))
    rules, edges = [],[]
    for base in implementations():
        for window in range(1,6):
            impl = replace(base,rollout_window=window)
            for op in graph.operations:
                contract = component_contract(op,impl,work[op.metadata["action"]])
                rules.append(BindingRule(op.operation_id,impl.assignment,contract.digest,RANKS,
                    tuple(f"gpu:{r}" for r in RANKS),(f"native:{impl.placement}",),contract.digest))
    for window in range(1,6):
        ordered = action_graph(window).topological_order()
        edges.extend({"when":{"rollout_window":str(window)},"source":a,"target":b}
                     for a,b in zip(ordered,ordered[1:]))
    return PlanningProblem(WAN,identity,graph,DOMAINS,tuple(rules),states,lives,limits,
        (WanImplementation("fsdp",512,False,5).assignment,WanImplementation("resident",2048,True,5).assignment),
        forbidden=({"placement":"fsdp","chunk_tokens":"1024"},
                   {"placement":"fsdp","chunk_tokens":"2048"},
                   {"placement":"fsdp","lora_epilogue":"on"}),
        conditional_dependencies=tuple(edges),evidence_scope="MEASURED",
        lowering={"adapter":ADAPTER,"layout_receipt":str(layout_path.resolve()),
                  "layout_sha256":file_sha256(layout_path),"component_sources":evidence,
                  "memory_scope":"MEASURED_ALLOCATOR_ENVELOPES_PLUS_OUTPUT_LIFETIMES; NATIVE_CONFIRMATION_REQUIRED"})


def validate_native_problem(problem):
    require(problem.evidence_scope == "MEASURED" and problem.lowering.get("adapter") == ADAPTER,
            "not a measured native action problem")
    path = Path(problem.lowering["layout_receipt"])
    require(file_sha256(path) == problem.lowering["layout_sha256"],"action layout hash changed")
    require(build_problem(path).digest == problem.digest,"generated action problem/lowering changed")


def assemble_calibrations(results, *, output: Path, expected_head: str, root: Path | None = None):
    """Metadata-only bridge from native collection to the common search CLI.

    A partial component library is valid; unmeasured components remain explicit
    calibration requests. Matching costs are pooled, never overwritten by the
    last recipe that happened to contain the same component.
    """
    from .component_calibration import pool_component_models
    from .cost_model import CostModel
    from distillschedule.workloads.best_known.registry import write_new_json
    root = root or repository_root()
    require(git_head(root) == expected_head, "assembly source HEAD changed")
    require(output.resolve() != root.resolve() and root.resolve() not in output.resolve().parents,
            "native evidence must be outside the source checkout")
    fragments, models, sources, implementation_ids = [], [], [], set()
    for result_path in results:
        result = _object(result_path)
        require(result.get("status") == "PASS_NATIVE_STAGE_CALIBRATION"
                and result.get("rank_count") == 16 and result.get("source_head") == expected_head
                and result.get("formal_timing_eligible") is False, "invalid native calibration result")
        values = {}
        for kind in ("layout_fragment", "operation_costs"):
            reference = result[kind]
            path = Path(reference["path"])
            require(file_sha256(path) == reference["sha256"], "calibration input hash changed")
            values[kind] = _object(path)
        fragment = values["layout_fragment"]
        require(fragment.get("schema_version") == "distillschedule.wan_action_layout_fragment.v1",
                "not an action calibration fragment")
        model = CostModel.from_dict(values["operation_costs"])
        require(model.scope == "MEASURED" and dict(model.binding) == fragment["binding"],
                "fragment/cost binding differs")
        model.verify_receipts()
        require(set(fragment["implementations"]) == {result["implementation_id"]},
                "fragment implementation differs")
        ranks = fragment["implementations"][result["implementation_id"]]["ranks"]
        require(len(ranks) == 16 and {r["rank"] for r in ranks} == set(RANKS), "rank coverage differs")
        impl = WanImplementation.parse(result["implementation_id"])
        for rank in ranks:
            validate_rank(rank, source_head=expected_head, rank=rank["rank"], placement=impl.native_placement)
        grouped = {}
        for i, row in enumerate(ranks[0]["stages"]):
            grouped.setdefault(row["key"], []).append(i)
        require(set(grouped) == {r.key for r in model.records}, "fragment/cost component keys differ")
        for key, indices in grouped.items():
            record = model.get(key)
            rows = [r["stages"][i] for i in indices for r in ranks]
            require(all(s["key"] == key and s["component_contract"] == record.evidence["component_contract"]
                        for s in rows), "fragment/cost component contract differs")
            require(record.samples_ns == tuple(max(r["stages"][i]["duration_ns"] for r in ranks) for i in indices)
                    and record.workspace_bytes_per_rank == max(s["workspace_bytes_per_rank"] for s in rows),
                    "fragment/cost duration or memory differs")
        fragments.append(result["layout_fragment"])
        models.append(model)
        implementation_ids.add(result["implementation_id"])
        sources.append({"path":str(result_path.resolve()), "sha256":file_sha256(result_path)})
    require(models, "native calibration results required")
    output.mkdir(parents=True, exist_ok=False)
    layout = output / "layout.json"
    write_new_json(layout, {"schema_version":SCHEMA, "fragments":fragments, "calibration_results":sources})
    problem = build_problem(layout, root=root)
    costs = pool_component_models(models, output=output / "components")
    costs.bind(problem.binding, problem_scope=problem.evidence_scope)
    write_new_json(output / "problem.json", problem.to_dict())
    write_new_json(output / "cost-model.json", costs.to_dict())
    summary = dict(status="PASS_COMPONENT_LIBRARY_ASSEMBLY", source_head=expected_head,
        calibrated_implementations=len(implementation_ids), component_count=len(costs.records),
        native_actions=len(problem.graph.operations), source_calibrations=sources,
        layout=str(layout.resolve()), problem=str((output / "problem.json").resolve()),
        cost_model=str((output / "cost-model.json").resolve()),
        gpu_execution="NOT_RUN", claimable_speedup=False, next_gate="MODEL_SEARCH_AND_NATIVE_CONFIRMATION")
    write_new_json(output / "summary.json", summary)
    return summary
