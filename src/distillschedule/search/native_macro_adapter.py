"""Restricted native LiveWan/H3 composition, not an arbitrary pipeline scheduler.

LiveWan CUDA clocks cannot align G and Score pools. Its five C-commit windows
are therefore coupled accounting partitions, NOT G/Score operation ownership.
Lookahead, transfers and waits stay inside each calibrated implementation.
No independent G cost, overlap discount or new pool split may be added. Native
receipts, not these accounting edges, prove parameter versions and call census.

H3 observes seven existing native stages; duplicate Fake ranges are preserved.
Both adapters use source-bound operation costs and the common simulator, but
only measured compositions are admitted. Unmeasured decisions remain rejected.
"""
from itertools import product
from pathlib import Path

from distillschedule.core.enums import GradientMode, OperationKind
from distillschedule.core.graph import Dependency, ExecutionGraph, Operation
from distillschedule.workloads.best_known.registry import (
    _object, fingerprint, file_sha256, git_head, load_profile, repository_root,
)
from .contracts import require, semantic_contract
from .plan_ir import BindingRule, PlanningProblem, StateAllocation, validate_binding
from .livewan_implementation import LIVEWAN, LiveWanImplementation
from .h3_implementation import H3, H3Implementation

ADAPTER = "native_coupled_macro_v1"
RANKS = tuple(range(16))
MEMORY = "POST_SETUP_STATIC_PLUS_CONSERVATIVE_GLOBAL_ADDITIONAL_PEAK"


def implementation_type(workload):
    require(workload in (LIVEWAN, H3), "native macro adapter not implemented")
    return LiveWanImplementation if workload == LIVEWAN else H3Implementation


def implementations(workload):
    implementation_type(workload)
    if workload == LIVEWAN:
        from .livewan_implementation import implementations as values
        return values()
    return tuple(H3Implementation(d, n) for d, n in product((1, 2), (64, 128)))


def precision_contract(workload):
    """Pin native precision source/config; do not pretend this is a dtype probe."""
    p = load_profile(workload)
    source = ("src/distillschedule/runtime/livewan_crossscale_worker.py" if workload == LIVEWAN
              else "src/distillschedule/backends/training/minimax_h3/grid_entrypoint.py")
    return dict(workload=workload, native_precision_source_sha256=file_sha256(repository_root()/source),
                dependency_contract=p.get("dependency_contract"),
                model_config=p["runtime"].get("model_config"),
                scope="SOURCE_BOUND_NATIVE_PRECISION_NOT_NEW_DTYPE_POLICY")


def native_graph(workload):
    implementation_type(workload)
    names = ([f"coupled_commit{i}" for i in range(5)] if workload == LIVEWAN else
             ["student_update", "fake_rollout_prepare", *[f"fake{i}_update" for i in range(5)]])
    ops, edges, ends = [], [], {}
    previous = None
    for cycle in (1, 2):
        for i, name in enumerate(names):
            key = f"cycle{cycle}.{name}"
            # Completion counters only: no invented fine-grained G commit
            # timestamp or parameter-read edge inside an opaque native macro.
            last = i == len(names)-1
            ops.append(Operation(key, (cycle-1)*5, "native_runtime", OperationKind.CRITIC_UPDATE,
                GradientMode.GRAD, adapter_mode="native_role_scopes",
                required_versions={"completed_cycles": cycle-1},
                produced_versions={"completed_cycles": cycle} if last else {},
                metadata=dict(cycle=cycle, stage=name, scope="NATIVE_COUPLED_ACCOUNTING_ENVELOPE")))
            if previous is not None:
                edges.append(Dependency(previous, key, "fixed_native_accounting_order"))
            previous = key
        ends[str(cycle)] = previous
    return ExecutionGraph(tuple(ops), tuple(edges), dict(adapter=ADAPTER, workload=workload,
        initial_versions={"completed_cycles": 0}, final_versions={"completed_cycles": 2},
        native_final_versions=dict(generator=2, critic=10, ema=0), cycle_ends=ends,
        canonical_profile_sha256=fingerprint(load_profile(workload)),
        scope="COUPLED_COMMIT_ACCOUNTING_PARTITIONS" if workload == LIVEWAN else "FIXED_NATIVE_STAGE_ENVELOPES",
        parameter_version_proof="WORKLOAD_NATIVE_RECEIPTS_NOT_MACRO_CLOCK_PARTITIONS",
        arbitrary_pool_dag_lowering=False, independent_overlap_discounts=False,
        memory_definition=MEMORY))


def operation_contract(workload, op, impl, *, precision_fingerprint=None):
    return dict(adapter=ADAPTER, workload=workload, operation_id=op.operation_id,
        implementation=impl.assignment, precision_fingerprint=(precision_fingerprint or fingerprint(precision_contract(workload))),
        memory_definition=MEMORY, timing_definition="CONTEXTUAL_CUDA_STREAM_ENVELOPE_NOT_FORMAL_WALL",
        aggregation="MAX_LOCAL_DURATION_PER_PARTITION_NOT_CROSS_DEVICE_CLOCK_UNION")


def cost_key(workload, op, impl):
    return fingerprint(operation_contract(workload, op, impl))


def observed_operations(workload, raw):
    if workload == LIVEWAN:
        marks = raw.get("commit_boundaries", [])
        expected = [(c, s) for c in (1, 2) for s in range(5)]
        require([(m["cycle"], m["step"]) for m in marks] == expected, "native C-commit boundary inventory changed")
        result, previous = [], 0
        for mark in marks:
            stop = mark["time_ns"]
            require(type(stop) is int and stop > previous, "invalid commit boundary clock")
            result.append(dict(cycle=mark["cycle"], duration_ns=stop-previous))
            previous = stop
        return result
    require(workload == H3, "unsupported observation adapter")
    expected = ["h3/student_step", "h3/critic_prepare_5xG"]+["h3/critic_update_F"]*5
    rows = raw.get("rows", [])
    require([(r["cycle"], r["label"]) for r in rows] == [(c,l) for c in (1,2) for l in expected],
            "H3 native seven-stage inventory changed")
    require(all(type(r["duration_ns"]) is int and r["duration_ns"] > 0 for r in rows), "invalid stage clock")
    return rows


def capability_report(workload):
    values = implementations(workload)
    return dict(adapter=ADAPTER, workload=workload,
        native_implementations=[dict(id=i.id, assignment=i.assignment) for i in values],
        precision_fingerprint=fingerprint(precision_contract(workload)),
        arbitrary_pool_dag_lowering=False, gpu_execution="NOT_RUN", cost_calibration="REQUIRED",
        limitation="FIXED_SOURCE_NATIVE_SCHEDULES_ONLY; NO_INDEPENDENT_G_SCORE_OR_LINK_PREDICTION")


def build_problem(layout_path, *, root=None):
    if _object(layout_path).get("schema_version") == "distillschedule.h3_action_layout.v1":
        from .h3_action_model_adapter import build_problem as build_actions
        return build_actions(layout_path, root=root)
    root = root or repository_root()
    layout_path = Path(layout_path)
    layout = _object(layout_path)
    require(layout.get("schema_version") == "distillschedule.native_macro_layout.v1"
            and layout.get("status") == "PASS" and layout.get("profiled") is False
            and layout.get("adapter") == ADAPTER and layout.get("memory_definition") == MEMORY,
            "invalid native macro layout")
    w, binding = layout["workload"], layout["binding"]
    validate_binding(binding)
    require(binding["source_head"] == git_head(root), "native calibration source changed")
    precision = fingerprint(precision_contract(w))
    require(binding["semantic_fingerprint"] == fingerprint(semantic_contract(load_profile(w)))
            and binding["precision_fingerprint"] == precision, "native workload/precision changed")
    all_impls = implementations(w)
    rows = layout["implementations"]
    require(set(rows) <= {i.id for i in all_impls} and len(rows) >= 2, "invalid calibrated implementation inventory")
    graph, states, rules = native_graph(w), [], []
    domains = {k: tuple(sorted({i.assignment[k] for i in all_impls})) for k in all_impls[0].assignment}
    for impl in all_impls:
        if impl.id not in rows:
            continue  # No guessed cost or nearest-recipe substitution.
        row = rows[impl.id]
        require(row.get("status") == "PASS" and len(row["ranks"]) == 16
                and {r["rank"] for r in row["ranks"]} == set(RANKS), "incomplete layout ranks")
        require(all(r["source_head"] == binding["source_head"] and r["status"] == "PASS"
                    and r["post_setup_static_bytes"] > 0 for r in row["ranks"]), "rank layout source failed")
        static = max(r["post_setup_static_bytes"] for r in row["ranks"])
        states.append(StateAllocation(impl.id, RANKS, static, ("native_runtime",), False, fingerprint(row)))
        for op in graph.operations:
            contract_hash = fingerprint(operation_contract(w, op, impl, precision_fingerprint=precision))
            rules.append(BindingRule(op.operation_id, impl.assignment, contract_hash, RANKS,
                tuple(f"gpu:{r}" for r in RANKS), (impl.id,), contract_hash))
    seeds = tuple(implementation_type(w).parse(k).assignment for k in layout["seed_implementations"])
    require(len(seeds) == 2 and seeds[0] != seeds[1]
            and all(implementation_type(w).from_assignment(s).id in rows for s in seeds), "calibrated distinct anchors required")
    limits = {int(k): v for k,v in layout["memory_limits"].items()}
    require(set(limits) == set(RANKS), "world16 capacity limits required")
    return PlanningProblem(w, binding, graph, domains, tuple(rules), tuple(states), (), limits, seeds,
        evidence_scope="MEASURED", lowering=dict(adapter=ADAPTER, layout_receipt=str(layout_path.resolve()),
                                               layout_sha256=file_sha256(layout_path)))


def validate_native_problem(problem):
    if problem.lowering.get("adapter") == "h3_prepared_actions_v2":
        from .h3_action_model_adapter import validate_native_problem as validate_actions
        return validate_actions(problem)
    require(problem.evidence_scope == "MEASURED" and problem.lowering.get("adapter") == ADAPTER,
            "unimplemented native adapter or fixture")
    path = Path(problem.lowering["layout_receipt"])
    require(file_sha256(path) == problem.lowering["layout_sha256"], "native layout changed")
    require(build_problem(path).digest == problem.digest, "unsupported altered native graph")


def lower_prediction(result, problem, costs, *, run_id, assets_path=None, mode="correctness"):
    from .execution import resolve_candidate
    from .model_search import validate_saved_prediction
    validate_native_problem(problem)
    costs.verify_receipts()
    selected = validate_saved_prediction(result, problem, costs)
    require(selected["status"] == "PREDICTED_FEASIBLE", "prediction cannot execute")
    impl = implementation_type(problem.workload).from_assignment(selected["assignment"])
    receipt = resolve_candidate(problem.workload, impl.id, assets_path=assets_path, run_id=run_id, mode=mode)
    receipt["model_plan"] = dict(problem=problem.to_dict(), cost_model=costs.to_dict(), search_result=dict(result))
    validate_lowered_plan(receipt)
    return receipt


def validate_lowered_plan(receipt):
    from .cost_model import CostModel
    from .model_search import validate_saved_prediction
    p = PlanningProblem.from_dict(receipt["model_plan"]["problem"])
    validate_native_problem(p)
    costs = CostModel.from_dict(receipt["model_plan"]["cost_model"])
    costs.verify_receipts()
    selected = validate_saved_prediction(receipt["model_plan"]["search_result"], p, costs)
    impl = implementation_type(p.workload).from_assignment(selected["assignment"])
    require(receipt["integration_head"] == p.binding["source_head"]
            and receipt["workload_semantics_hash"] == p.binding["semantic_fingerprint"]
            and receipt["search_candidate"] == impl.candidate().to_dict(), "native lowering binding changed")
    if receipt.get("asset_config"):
        require(receipt["asset_fingerprint"] == p.binding["asset_fingerprint"], "native assets changed")


def assemble_native(*args, **kwargs):
    from .native_macro_calibration import assemble_native as assemble
    return assemble(*args, **kwargs)
