"""Reconcile actual native calibration receipts; never normalize them as timing."""
from __future__ import annotations

from typing import Any, Mapping
from pathlib import Path

from distillschedule.workloads.best_known.registry import fingerprint
from .contracts import require
from .plan_ir import nonnegative
from .wan_implementation import PRECISION, WAN, WanImplementation
from .wan_model_adapter import ADAPTER, cost_key, native_graph, operation_contract


def validate_rank(value: Mapping[str, Any], *, source_head: str, rank: int,
                  placement: str) -> WanImplementation:
    impl = WanImplementation.parse(value["implementation_id"])
    actions = impl.rollout_window is not None
    from . import wan_action_costs as action_costs
    from .wan_action_calibration import validate_lifetimes
    expected = dict(schema_version="distillschedule.wan_action_rank_calibration.v1" if actions else "distillschedule.wan_stage_rank_calibration.v1",
        status="PASS", source_head=source_head, rank=rank, adapter=action_costs.ADAPTER if actions else ADAPTER, workload=WAN,
        precision_fingerprint=PRECISION.digest, calibration_only=True,
        formal_timing_eligible=False, profiled=False, evidence_scope="MEASURED",
        kind="native_operation", errors=[], missing_operation_ids=[])
    for key, item in expected.items():
        require(value.get(key) == item, f"stage calibration rank{rank}: {key} differs")
    require(impl.assignment == value["implementation"] and impl.native_placement == placement,
            "calibrated implementation differs from executed placement")
    graph = action_costs.action_graph(impl.rollout_window) if actions else native_graph()
    if actions:
        validate_lifetimes(value)
    ids = [op.operation_id for op in graph.operations]
    require(value["native_graph_sha256"] == fingerprint(graph.to_dict()), "calibration graph differs")
    require(value["completed_operation_ids"] == ids, "calibration operation order differs")
    stages = value["stages"]
    require(len(stages) == len(ids), "missing/extra calibration operations")
    static = nonnegative(value["post_setup_static_bytes"], "static bytes")
    require(static > 0 and static == value["post_setup_memory"]["allocated_bytes"], "invalid static allocation")
    snapshots = [value["post_setup_memory"]]
    previous_stop = -1
    for op, stage in zip(graph.operations, stages):
        contract = (action_costs.component_contract(op, impl, value["work_signatures"][op.metadata["action"]]).to_dict()
                    if actions else operation_contract(op, impl))
        require(stage["status"] == "PASS" and stage.get("errors") == [], "failed stage")
        require(stage["operation_id"] == op.operation_id and stage["key"] == fingerprint(contract), "stage cost key differs")
        require(stage["operation_contract"] == contract and stage["operation_contract_sha256"] == fingerprint(contract),
                "stage operation contract differs")
        start, stop, duration = (nonnegative(stage[k], k) for k in ("start_ns", "stop_ns", "duration_ns"))
        require(duration > 0 and stop - start == duration and start >= previous_stop, "stage clock/order invalid")
        previous_stop = stop
        after = stage["memory_after"]
        retained = stage["entry_retained_input_bytes"] if actions else 0
        require(stage["workspace_bytes_per_rank"] == max(0, after["peak_allocated_bytes"] - static - retained),
                "stage workspace accounting differs")
        if actions:
            require(stage.get("component_contract") == contract,"component identity differs")
        snapshots.extend((stage["memory_before"], after))
    total = nonnegative(value["post_setup_memory"]["driver_total_bytes"], "driver capacity")
    for memory in snapshots:
        for key in ("allocated_bytes", "reserved_bytes", "driver_free_bytes", "driver_total_bytes"):
            nonnegative(memory[key], key)
        require(memory["driver_total_bytes"] == total and memory["driver_free_bytes"] <= total,
                "driver capacity changed")
        require(memory["allocated_bytes"] <= memory["reserved_bytes"] <= total, "invalid allocation/reservation")
        if "peak_allocated_bytes" in memory:
            require(memory["allocated_bytes"] <= memory["peak_allocated_bytes"] <= memory["peak_reserved_bytes"] <= total,
                    "invalid stage peak")
    require(value["peak_allocated_bytes"] == max(m.get("peak_allocated_bytes", m["allocated_bytes"]) for m in snapshots),
            "aggregate allocated peak lost across resets")
    require(value["peak_reserved_bytes"] == max(m.get("peak_reserved_bytes", m["reserved_bytes"]) for m in snapshots),
            "aggregate reserved peak lost across resets")
    require(value["minimum_observed_driver_free_bytes"] == min(m["driver_free_bytes"] for m in snapshots),
            "sampled minimum driver free differs")
    return impl


def assemble_native(receipt: dict[str, Any], preparation: dict[str, Any],
                    identity: dict[str, str], runtime_identity_evidence: dict[str, Any]) -> dict[str, Any]:
    """Run beside persistent receipts; export one measured implementation fragment.

    A fragment is NOT a complete planning layout. Unknown implementations stay
    uncalibrated. Stage barriers deliberately disturb cross-stage asynchrony;
    every predicted winner still needs a separate native unprofiled cycle-2 run.
    """
    from .native import checked_summary, read, binding, validate_native_comparison
    from .runtime_identity import verify_measurement_runtime
    from .plan_ir import validate_binding
    from .cost_model import CostModel, OperationCost
    from distillschedule.workloads.best_known.registry import file_sha256, write_new_json

    validate_binding(identity)
    require(receipt["native_mode"] == "two-cycle-calibration" and receipt["mode"] == "calibration",
            "only explicitly calibrated native runs can produce stage costs")
    require(receipt["integration_head"] == identity["source_head"]
            and receipt["asset_fingerprint"] == identity["asset_fingerprint"]
            and receipt["workload_semantics_hash"] == identity["semantic_fingerprint"]
            and identity["precision_fingerprint"] == PRECISION.digest, "calibration binding differs")
    verify_measurement_runtime({"runtime_identity_evidence": runtime_identity_evidence}, identity)
    out, summary = checked_summary(receipt)
    require(summary.get("schema_version") == "distillschedule.wan13b_two_cycle_calibration_summary.v1"
            and summary.get("rank_count") == 16 and summary.get("formal_eligibility") is False
            and summary.get("calibration_only") is True and summary.get("profiled") is False
            and summary.get("speed_claim_denominator") is None and summary["cycle_2"]["claimable"] is False,
            "calibration summary missing/non-diagnostic")
    require(preparation["source_head"] == identity["source_head"]
            and preparation["candidate_id"] == receipt["search_candidate"]["id"], "calibration preparation differs")
    for name in ("correctness", "comparison"):
        require(file_sha256(Path(preparation[name]["path"])) == preparation[name]["sha256"], "calibration gate changed")
    validate_native_comparison(read(Path(preparation["comparison"]["path"])))
    resources = [read(out / f"search-resources.node{i}.json") for i in (0, 1)]
    for i, resource in enumerate(resources):
        require(resource["status"] == "PASS" and resource["node_rank"] == i
                and resource["run_id"] == receipt["run_id"] and len(resource["samples"]) >= 2,
                "calibration resource observer incomplete")
        for counter in ("oom", "oom_kill"):
            values = [s["cgroup_events"][counter] for s in resource["samples"]]
            require(max(values) == min(values), "calibration cgroup OOM counter changed")
    ranks, sources, limits = [], [], {}
    impl = WanImplementation.parse(receipt["search_candidate"]["id"])
    for rank in range(16):
        path = out / "execution/ranks" / f"rank_{rank:02d}.json"
        raw = read(path)
        require(raw["status"] == "PASS" and raw["rank"] == rank and raw["run_id"] == receipt["run_id"]
                and raw["final_versions"] == {"generator": 4, "critic": 20, "ema": 20}
                and raw["finite_status"] == [True, True], "native calibration rank incomplete")
        value = raw["stage_calibration"]
        require(validate_rank(value, source_head=identity["source_head"], rank=rank,
                placement=impl.native_placement) == impl, "calibrated implementation differs")
        require(raw["runtime_contract"]["planner_execution_precision"]["precision_fingerprint"] == PRECISION.digest,
                "observed runtime precision differs")
        ranks.append(value)
        sources.append(binding(path))
        memory = [value["post_setup_memory"], *[s[k] for s in value["stages"] for k in ("memory_before", "memory_after")]]
        non_allocator = max(max(0, m["driver_total_bytes"] - m["driver_free_bytes"] - m["reserved_bytes"]) for m in memory)
        limits[str(rank)] = value["post_setup_memory"]["driver_total_bytes"] - non_allocator - 2 * 2**30
        require(limits[str(rank)] > value["peak_allocated_bytes"], "calibration lacks conservative HBM margin")
    # Write only after every native gate passed. Existing directories are never reused.
    actions = impl.rollout_window is not None
    from . import wan_action_costs as action_costs
    graph = action_costs.action_graph(impl.rollout_window) if actions else native_graph()
    if actions:
        require(all(r["work_signatures"] == ranks[0]["work_signatures"] for r in ranks),
                "rank-global physical work signatures disagree")
    target = out / ("component-costs" if actions else "stage-costs")
    target.mkdir(exist_ok=False)
    provenance = dict(native_summary=binding(out / "bound-summary.json"), rank_receipts=sources,
        resource_receipts=[binding(out / f"search-resources.node{i}.json") for i in (0, 1)],
        runtime_identity_evidence=runtime_identity_evidence, correctness=preparation["correctness"],
        comparison=preparation["comparison"], strict_numerical_gate=preparation["strict_numerical_gate"])
    write_new_json(target / "provenance.json", provenance)
    records, grouped = [], {}
    for index, op in enumerate(graph.operations):
        stages = [r["stages"][index] for r in ranks]
        key = stages[0]["key"]
        require(all(s["key"] == key and s["operation_contract"] == stages[0]["operation_contract"] for s in stages),
                "ranks executed different component contracts")
        grouped.setdefault(key, []).append((op, stages))
    for key, occurrences in grouped.items():
        op, stages = occurrences[0]
        contract = stages[0]["operation_contract"]
        payload = dict(schema_version="distillschedule.operation_calibration.v1", binding=identity,
            key=key, samples_ns=[max(s["duration_ns"] for s in rows) for _,rows in occurrences],
            workspace_bytes_per_rank=max(s["workspace_bytes_per_rank"] for _,rows in occurrences for s in rows), status="PASS",
            profiled=False, kind="native_operation", operation_contract_sha256=fingerprint(contract),
            calibration_only=True, formal_timing_eligible=False, operation_id=op.operation_id,
            aggregation="MAX_RANK_SYNCHRONIZED_LOCAL_STAGE_WALL_NOT_SUM_OF_GPU_KERNELS",
            per_rank_duration_ns=[s["duration_ns"] for s in stages], provenance=binding(target / "provenance.json"),
            occurrences=[{"operation_id":o.operation_id,"per_rank_duration_ns":[s["duration_ns"] for s in rows]}
                         for o,rows in occurrences])
        if actions:
            payload["component_contract"] = contract
        path = target / f"{key if actions else op.operation_id}.json"
        write_new_json(path, payload)
        records.append(OperationCost(payload["key"], tuple(payload["samples_ns"]), payload["workspace_bytes_per_rank"],
            {**{k: payload[k] for k in ("status", "profiled", "kind", "operation_contract_sha256")},
             "source_receipt": str(path), "source_receipt_sha256": file_sha256(path),
             **({"component_contract":contract} if actions else {})}))
    costs = CostModel(identity, tuple(records))
    costs.verify_receipts()
    write_new_json(target / "operation-costs.json", costs.to_dict())
    fragment = dict(schema_version="distillschedule.wan_action_layout_fragment.v1" if actions else "distillschedule.wan_stage_layout_fragment.v1", status="PASS", profiled=False,
        binding=identity, adapter=action_costs.ADAPTER if actions else ADAPTER, memory_limits=limits,
        memory_definition=action_costs.MEMORY_DEFINITION if actions else "POST_SETUP_STATIC; STAGE_COST_INCLUDES_ALL_ADDITIONAL_LIVE_MEMORY",
        capacity_scope="TORCH_ALLOCATOR; MAX_BOUNDARY_OBSERVED_NON_ALLOCATOR_USE; 2GIB_SAFETY_NOT_EXACT_DRIVER_PEAK",
        implementations={impl.id: {"status": "PASS", "ranks": ranks}},
        provenance=binding(target / "provenance.json"), complete_planning_layout=False)
    write_new_json(target / "layout-fragment.json", fragment)
    return dict(status="PASS_NATIVE_STAGE_CALIBRATION", source_head=identity["source_head"],
        run_id=receipt["run_id"], implementation_id=impl.id, rank_count=16, stages_per_rank=len(graph.operations),
        native_summary=summary, layout_fragment=binding(target / "layout-fragment.json"),
        operation_costs=binding(target / "operation-costs.json"), provenance=binding(target / "provenance.json"),
        stage_world_max_seconds={op.operation_id:max(r["stages"][i]["duration_ns"] for r in ranks)/1e9
                                 for i,op in enumerate(graph.operations)},
        formal_timing_eligible=False, complete_planning_layout=False, speedup=None)
