"""Read native evidence beside remote files; emit small hash-bound macro costs.

No model code, fitted speed discounts, kernel-union-as-wall, or GPU launch.
Calibration remains diagnostic even when every native contract passes.
"""
from pathlib import Path

from distillschedule.workloads.best_known.registry import fingerprint, file_sha256, write_new_json
from .contracts import require
from .cost_model import CostModel, OperationCost
from .native_macro_adapter import (ADAPTER, MEMORY, RANKS, LIVEWAN, H3, implementation_type,
    native_graph, observed_operations, operation_contract, precision_contract)


def assemble_native(receipt, preparation, identity, runtime_identity_evidence):
    from .native import binding, read, checked_summary, validate_native_comparison
    from .runtime_identity import verify_measurement_runtime
    from .plan_ir import validate_binding
    validate_binding(identity)
    w = receipt["profile_id"]
    impl = implementation_type(w).parse(receipt["search_candidate"]["id"])
    require(receipt["mode"] == "calibration", "only explicit calibration can emit costs")
    for field, key in (("source_head", "integration_head"), ("asset_fingerprint", "asset_fingerprint"),
                       ("semantic_fingerprint", "workload_semantics_hash")):
        require(identity[field] == receipt[key], "calibration source/assets/semantics changed")
    precision = fingerprint(precision_contract(w))
    require(identity["precision_fingerprint"] == precision, "calibration precision changed")
    verify_measurement_runtime({"runtime_identity_evidence": runtime_identity_evidence}, identity)
    require(preparation["source_head"] == identity["source_head"] and preparation["candidate_id"] == impl.id,
            "calibration preparation changed")
    for key in ("correctness", "comparison"):
        require(file_sha256(Path(preparation[key]["path"])) == preparation[key]["sha256"], "native gate changed")
    compare = read(Path(preparation["comparison"]["path"]))
    if w == H3:
        from .h3_native_bridge import validate_comparison, _checked_native
        validate_comparison(compare)
        out, summary, native, sources = _checked_native(receipt)
    else:
        validate_native_comparison(compare)
        out, summary = checked_summary(receipt)
        paths = [out / "receipts" / f"rank_{r:03d}.json" for r in RANKS]
        native, sources = [read(p) for p in paths], [binding(p) for p in paths]
    require(summary.get("calibration_only") is True, "native summary is not calibration")
    require(summary.get("formal_timing_eligible", summary.get("formal_eligibility")) is False,
            "calibration mislabeled formal timing")
    resources = []
    for n in (0, 1):
        p = out / f"search-resources.node{n}.json"
        r = read(p)
        require(r["status"] == "PASS" and r["node_rank"] == n and r["run_id"] == receipt["run_id"]
                and len(r["samples"]) >= 2, "missing resource evidence")
        for event in ("oom", "oom_kill"):
            counts = [s["cgroup_events"][event] for s in r["samples"]]
            require(max(counts) == min(counts), "cgroup OOM increased")
        teardown = out / "launcher" / f"node_{n}" / "teardown_status.txt"
        flags = dict(line.split("=",1) for line in teardown.read_text().splitlines() if "=" in line)
        require(flags.get("status") == "PASS" and flags.get("gpu_process_count") == "0"
                and flags.get("owner_run_id") == receipt["run_id"], "native teardown failed")
        sources += [binding(p), binding(teardown)]
        resources.append(r)
    require({r["rank"] for r in native} == set(RANKS), "missing native ranks")
    ranks, observations, limits = [], [], {}
    for raw in sorted(native, key=lambda r:r["rank"]):
        rank = raw["rank"]
        require(raw["final_versions"] == dict(generator=2, critic=10, ema=0), "native versions failed")
        if w == LIVEWAN:
            p = out / "operation_calibration" / f"rank_{rank:03d}.json"
            marker = raw.get("operation_calibration", {})
            require(raw["status"] == "PASS" and marker.get("enabled") is True
                    and marker.get("status") == "PASS" and marker.get("path") == str(p)
                    and marker.get("calibration_only") is True
                    and marker.get("formal_timing_eligible") is False,
                    "native rank/calibration run binding failed")
            cal = read(p)
            sources.append(binding(p))
            peak = raw["peak_memory"]["max_reserved_bytes"]
        else:
            cal = raw["stage_calibration"]
            peak = max(c[k]["hbm"]["peak_reserved_bytes"] for c in raw["cycles"] for k in ("memory_before","memory_after"))
        for key, expected in dict(schema_version="distillschedule.async_range_calibration.v1", status="PASS",
            source_head=identity["source_head"], rank=rank, implementation=impl.id, calibration_only=True,
            formal_timing_eligible=False, profiled=False, extra_synchronizations=0, extra_collectives=0,
            tensor_value_reads=0).items():
            require(cal.get(key) == expected, f"rank{rank} calibration {key} mismatch")
        static = cal["post_setup_static_bytes"]
        require(type(static) is int and static > 0, "invalid native static allocation")
        if w != LIVEWAN or impl.placement == "fsdp" or rank >= 8:
            observations.append(observed_operations(w, cal))
        else:
            require(not cal["commit_boundaries"], "G rank cannot claim Score commit clock")
        gpu = [next(g for g in s["gpus"] if g["device"] == rank%8) for s in resources[rank//8]["samples"]]
        require(len({g["total_bytes"] for g in gpu}) == 1, "driver capacity changed")
        limits[str(rank)] = gpu[0]["total_bytes"]
        used = max(peak, *(g["total_bytes"]-g["free_bytes"] for g in gpu))
        require(static <= used <= limits[str(rank)], "invalid driver/allocator evidence")
        ranks.append(dict(rank=rank, status="PASS", source_head=identity["source_head"],
            post_setup_static_bytes=static, peak_reserved_bytes=peak, sampled_driver_or_reserved_peak_bytes=used,
            sampled_min_driver_free_bytes=min(g["free_bytes"] for g in gpu)))
    expected_owners = 8 if w == LIVEWAN and impl.placement == "pipeline" else 16
    require(len(observations) == expected_owners, "native operation clock owners incomplete")
    target = out / "macro-costs"
    target.mkdir(exist_ok=False)
    provenance = dict(native_sources=sources, runtime_identity_evidence=runtime_identity_evidence,
        correctness=preparation["correctness"], comparison=preparation["comparison"],
        strict_numerical_gate=preparation["strict_numerical_gate"])
    write_new_json(target/"provenance.json", provenance)
    graph = native_graph(w)
    static = max(r["post_setup_static_bytes"] for r in ranks)
    additional = max(r["sampled_driver_or_reserved_peak_bytes"] for r in ranks)-static
    records = []
    for i, op in enumerate(graph.operations):
        durations = [r[i]["duration_ns"] for r in observations]
        contract_hash = fingerprint(operation_contract(w, op, impl, precision_fingerprint=precision))
        payload = dict(schema_version="distillschedule.operation_calibration.v1", binding=identity,
            key=contract_hash, samples_ns=[max(durations)], workspace_bytes_per_rank=additional,
            status="PASS", profiled=False, kind="native_operation", calibration_only=True, formal_timing_eligible=False,
            operation_contract_sha256=contract_hash, operation_id=op.operation_id,
            per_owner_duration_ns=durations, owner_ranks=list(range(8,16)) if expected_owners==8 else list(RANKS),
            cost_scope=graph.metadata["scope"], provenance=binding(target/"provenance.json"))
        path=target/f"{op.operation_id}.json"
        write_new_json(path,payload)
        records.append(OperationCost(payload["key"],tuple(payload["samples_ns"]),additional,
            {**{k:payload[k] for k in ("status","profiled","kind","operation_contract_sha256")},
             "source_receipt":str(path),"source_receipt_sha256":file_sha256(path)}))
    costs=CostModel(identity,tuple(records))
    costs.verify_receipts()
    write_new_json(target/"operation-costs.json",costs.to_dict())
    fragment=dict(schema_version="distillschedule.native_macro_layout_fragment.v1", status="PASS", profiled=False,
        adapter=ADAPTER,workload=w,binding=identity,memory_definition=MEMORY,memory_limits=limits,
        implementations={impl.id:dict(status="PASS",ranks=ranks)},provenance=binding(target/"provenance.json"),
        capacity_scope="RESERVED_HIGHWATER_AND_SAMPLED_DRIVER; NO_UNSAMPLED_PEAK_OR_ROBUST_HEADROOM_CLAIM")
    write_new_json(target/"layout-fragment.json",fragment)
    return dict(status="PASS_NATIVE_MACRO_CALIBRATION",source_head=identity["source_head"],run_id=receipt["run_id"],
        implementation_id=impl.id,rank_count=16,operation_clock_owners=expected_owners,
        layout_fragment=binding(target/"layout-fragment.json"),operation_costs=binding(target/"operation-costs.json"),
        provenance=binding(target/"provenance.json"),formal_timing_eligible=False,speedup=None,
        operation_count=len(records),stage_world_max_seconds={op.operation_id:records[i].duration_ns/1e9 for i,op in enumerate(graph.operations)})
