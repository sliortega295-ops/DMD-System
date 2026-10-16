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
    precision = fingerprint(precision_contract(w, h3_request=impl.request if w == H3 else None,
                                              livewan_request=impl.request if w == LIVEWAN else None))
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
        require(compare["integration_head"] == identity["source_head"]
                and impl.id in (compare["reference_candidate_id"], compare["candidate_id"]),
                "H3 comparison does not bind calibration source/candidate")
        out, summary, native, sources = _checked_native(receipt)
    else:
        validate_native_comparison(compare)
        out, summary = checked_summary(receipt)
        paths = [out / "receipts" / f"rank_{r:03d}.json" for r in RANKS]
        native, sources = [read(p) for p in paths], [binding(p) for p in paths]
    require(summary.get("calibration_only") is True, "native summary is not calibration")
    require(summary.get("formal_timing_eligible", summary.get("formal_eligibility")) is False,
            "calibration mislabeled formal timing")
    r9 = w == H3 and "execution_transport" in receipt
    if r9:
        from .r9_native import verify_execution
        sources += verify_execution(receipt)
    resources = []
    for n in (0, 1):
        p = out / f"search-resources.node{n}.json"
        r = read(p)
        require(r["status"] == "PASS" and r["node_rank"] == n and r["run_id"] == receipt["run_id"]
                and len(r["samples"]) >= 2, "missing resource evidence")
        for event in ("oom", "oom_kill"):
            counts = [s["cgroup_events"][event] for s in r["samples"]]
            require(max(counts) == min(counts), "cgroup OOM increased")
        sources.append(binding(p))
        if not r9:
            teardown = out / "launcher" / f"node_{n}" / "teardown_status.txt"
            flags = dict(line.split("=",1) for line in teardown.read_text().splitlines() if "=" in line)
            require(flags.get("status") == "PASS" and flags.get("gpu_process_count") == "0"
                    and flags.get("owner_run_id") == receipt["run_id"], "native teardown failed")
            sources.append(binding(teardown))
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
            from distillschedule.runtime.livewan_geometry import receipt_errors
            require(not receipt_errors(cal, impl.request), "LiveWan calibration request/actual shape differs")
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
    sources = list({ref["path"]: ref for ref in sources}.values())
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


def _read_assembly(results, *, expected_head):
    """Reconcile joined metadata to hashed native observations, never timing hints."""
    from distillschedule.workloads.best_known.registry import _object

    def bound(ref):
        path = Path(ref["path"])
        require(path.is_file() and file_sha256(path) == ref["sha256"], "native calibration source hash changed")
        return _object(path)

    fragments, models, sources = [], [], []
    for path in results:
        path = Path(path)
        result = _object(path)
        require(result.get("status") == "PASS_NATIVE_MACRO_CALIBRATION"
                and result.get("rank_count") == 16 and result.get("source_head") == expected_head
                and result.get("formal_timing_eligible") is False, "invalid native macro calibration result")
        fragment = bound(result["layout_fragment"])
        require(fragment.get("schema_version") == "distillschedule.native_macro_layout_fragment.v1"
                and fragment.get("adapter") == ADAPTER and fragment.get("memory_definition") == MEMORY
                and fragment.get("status") == "PASS" and fragment.get("profiled") is False,
                "invalid native macro calibration fragment")
        cid, workload = result["implementation_id"], fragment["workload"]
        impl = implementation_type(workload).parse(cid)
        require(set(fragment["implementations"]) == {cid}, "calibration fragment implementation changed")
        model = CostModel.from_dict(bound(result["operation_costs"]))
        require(model.scope == "MEASURED" and dict(model.binding) == fragment["binding"]
                and model.binding["source_head"] == expected_head, "native calibration binding differs")
        model.verify_receipts()
        provenance = bound(fragment["provenance"])
        require(fragment["provenance"] == result["provenance"], "native provenance differs")
        bound(provenance["correctness"])
        comparison = bound(provenance["comparison"])
        from .runtime_identity import verify_measurement_runtime
        verify_measurement_runtime({"runtime_identity_evidence": provenance["runtime_identity_evidence"]},
                                   model.binding)
        if workload == H3:
            from .h3_native_bridge import validate_comparison
            validate_comparison(comparison)
            require(comparison["integration_head"] == expected_head
                    and cid in (comparison["reference_candidate_id"], comparison["candidate_id"]),
                    "H3 comparison does not bind assembled source/candidate")
        else:
            from .native import validate_native_comparison
            validate_native_comparison(comparison)
        # Resource/teardown files may be non-JSON; hash every original source.
        observations, native_h3_rows = {}, []
        for ref in provenance["native_sources"]:
            source = Path(ref["path"])
            require(source.is_file() and file_sha256(source) == ref["sha256"], "native source receipt changed")
            if source.suffix != ".json":
                continue
            raw = _object(source)
            if workload == H3 and raw.get("schema_version") == "distillschedule.h3_native_two_cycle.v1":
                native_h3_rows.append(raw)
            cal = raw.get("stage_calibration") if workload == H3 else raw
            if not isinstance(cal, dict) or cal.get("schema_version") != "distillschedule.async_range_calibration.v1":
                continue
            rank = cal["rank"]
            if workload == LIVEWAN:
                from distillschedule.runtime.livewan_geometry import receipt_errors
                require(not receipt_errors(cal, impl.request), "LiveWan source calibration request/actual shape differs")
            require(cal["source_head"] == expected_head and cal["implementation"] == cid,
                    "native calibration rank source/implementation changed")
            if workload == LIVEWAN and impl.placement == "pipeline" and rank < 8:
                require(not cal["commit_boundaries"], "Generator rank cannot supply Score clock")
                continue
            require(rank not in observations, "duplicate native calibration rank")
            observations[rank] = observed_operations(workload, cal)
        owners = tuple(range(8, 16)) if workload == LIVEWAN and impl.placement == "pipeline" else RANKS
        require(set(observations) == set(owners), "native calibration clock owners incomplete")
        if workload == H3:
            from .h3_native import summarize_native_two_cycle
            summary = summarize_native_two_cycle(native_h3_rows, expected_head=expected_head, candidate_id=cid)
            require(summary["calibration_only"], "native component sources are not calibration runs")
        graph = native_graph(workload)
        require(len(model.records) == len(graph.operations), "native macro operation inventory differs")
        for index, op in enumerate(graph.operations):
            key = fingerprint(operation_contract(workload, op, impl,
                              precision_fingerprint=model.binding["precision_fingerprint"]))
            record = model.get(key)
            samples = [observations[r][index]["duration_ns"] for r in owners]
            require(record is not None and record.samples_ns == (max(samples),),
                    "native macro costs differ from actual rank observations")
        fragments.append(fragment)
        models.append(model)
        sources.append(dict(path=str(path.resolve()), sha256=file_sha256(path)))
    require(len(fragments) >= 2, "two distinct native calibration anchors required")
    first = fragments[0]
    require(all(f["workload"] == first["workload"] and f["binding"] == first["binding"] for f in fragments),
            "cannot join calibration from different workloads/requests/assets/hardware/source")
    implementations = {}
    for fragment in fragments:
        require(not (set(implementations) & set(fragment["implementations"])), "duplicate calibrated implementation")
        implementations.update(fragment["implementations"])
    limits = {str(r): min(f["memory_limits"][str(r)] for f in fragments) for r in RANKS}
    layout = dict(schema_version="distillschedule.native_macro_layout.v1", adapter=ADAPTER,
        workload=first["workload"], binding=first["binding"], status="PASS", profiled=False,
        memory_definition=MEMORY, memory_limits=limits, implementations=implementations,
        seed_implementations=list(implementations)[:2], calibration_results=sources)
    records = tuple(record for model in models for record in model.records)
    return layout, CostModel(first["binding"], records)


def validate_assembled_layout(layout):
    references = layout["calibration_results"]
    for ref in references:
        require(file_sha256(Path(ref["path"])) == ref["sha256"], "calibration result hash changed")
    expected, _ = _read_assembly([Path(r["path"]) for r in references],
                               expected_head=layout["binding"]["source_head"])
    require(layout == expected, "assembled native layout differs from source calibrations")


def validate_assembled_costs(layout, costs):
    _, expected = _read_assembly([Path(r["path"]) for r in layout["calibration_results"]],
                                expected_head=layout["binding"]["source_head"])
    require(costs.digest == expected.digest, "selected native costs differ from assembled source observations")


def assemble_calibrations(results, *, output, expected_head, root=None):
    from distillschedule.workloads.best_known.registry import git_head, repository_root
    from .native_macro_adapter import build_problem
    root = root or repository_root()
    require(git_head(root) == expected_head, "assembly source HEAD changed")
    output = Path(output)
    require(output.resolve() != root.resolve() and root.resolve() not in output.resolve().parents,
            "native evidence must be outside the source checkout")
    layout, costs = _read_assembly(results, expected_head=expected_head)
    output.mkdir(parents=True, exist_ok=False)
    path = output / "layout.json"
    write_new_json(path, layout)
    problem = build_problem(path, root=root)
    costs.bind(problem.binding, problem_scope=problem.evidence_scope)
    write_new_json(output / "problem.json", problem.to_dict())
    write_new_json(output / "cost-model.json", costs.to_dict())
    return problem, costs
