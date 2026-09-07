"""Synthetic component-cost planning; these numbers are not GPU measurements."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from test_refactor_wan_action_calibration import fixture, Tensor
from test_refactor_model_planning import BINDING
from distillschedule.search.wan_action_costs import action_graph
from distillschedule.search.wan_action_model_adapter import build_problem
from distillschedule.search.wan_implementation import WanImplementation,PRECISION,WAN
from distillschedule.search.cost_model import CostModel,OperationCost
from distillschedule.search.model_search import search_plans
from distillschedule.search.simulator import simulate
from distillschedule.search.contracts import semantic_contract
from distillschedule.workloads.best_known.registry import file_sha256,fingerprint,load_profile


def planning_fixture(tmp_path,monkeypatch):
    from distillschedule.search import wan_action_model_adapter as m
    monkeypatch.setattr(m,"git_head",lambda root:"a"*40)
    binding={**BINDING,"precision_fingerprint":PRECISION.digest,
             "semantic_fingerprint":fingerprint(semantic_contract(load_profile(WAN)))}
    refs,records=[],{}
    for placement in ("fsdp","resident"):
        backend,collector,_,_=fixture(5)
        impl=WanImplementation(placement,512 if placement=="fsdp" else 2048,placement=="resident",5)
        collector._implementation=impl
        for n,op in enumerate(action_graph(5).operations):
            with collector.stage(op.metadata["cycle"],op.metadata["action"]) as sample:
                backend.body()
            if op.metadata["action"].startswith("r"):
                sample.record_output(Tensor(n+1,20))
        rank=collector.finish()
        rank["evidence_scope"]="MEASURED" # synthetic protocol fixture only
        ranks=[{**deepcopy(rank),"rank":i} for i in range(16)]
        fragment={"schema_version":"distillschedule.wan_action_layout_fragment.v1",
            "status":"PASS","profiled":False,"adapter":m.ADAPTER,"binding":binding,
            "memory_definition":rank["memory_definition"],
            "memory_limits":{str(i):8000 for i in range(16)},
            "implementations":{impl.id:{"status":"PASS","ranks":ranks}}}
        path=tmp_path/f"{placement}.json";path.write_text(json.dumps(fragment))
        refs.append({"path":str(path),"sha256":file_sha256(path)})
        for row in rank["stages"]:
            records[row["key"]]=OperationCost(row["key"],(row["duration_ns"],),row["workspace_bytes_per_rank"],
                {"component_contract":row["component_contract"],"kind":"CPU_FIXTURE_ONLY"})
    layout=tmp_path/"layout.json"
    layout.write_text(json.dumps({"schema_version":"distillschedule.wan_action_layout.v1","fragments":refs}))
    problem=build_problem(layout)
    return replace(problem,evidence_scope="CPU_FIXTURE_ONLY"),CostModel(binding,tuple(records.values()),scope="CPU_FIXTURE_ONLY"),layout


def test_costs_from_window5_predict_heldout_orders_and_explicit_missing_kernels(tmp_path,monkeypatch):
    problem,costs,_=planning_fixture(tmp_path,monkeypatch)
    result=search_plans(problem,costs,max_proposals=60,exhaustive=True)
    assert len(result["ranked"])==10 # 2 calibrated local implementations x 5 orders
    assert result["calibration_requests"] # unmeasured kernel components remain uncalibrated
    a=WanImplementation("resident",2048,True,1).assignment
    b={**a,"rollout_window":"5"}
    one,five=(simulate(problem,x,costs) for x in (a,b))
    assert one["predicted_cycle2_ns"]==five["predicted_cycle2_ns"] # no invented overlap/speedup
    assert one["memory"][0]["peak_bytes"] < five["memory"][0]["peak_bytes"]
    assert one["memory_reconciliation"]==five["memory_reconciliation"]=="PASS"
    assert all(row["operation"].split(".")[1] for row in one["timeline"])


def test_fragment_mutation_and_legacy_layout_fail_closed(tmp_path,monkeypatch):
    _,_,layout=planning_fixture(tmp_path,monkeypatch)
    data=json.loads(layout.read_text());p=tmp_path/"fsdp.json"
    p.write_text(p.read_text()+" ")
    with pytest.raises(ValueError,match="hash"):
        build_problem(layout)
    data["schema_version"]="distillschedule.wan_stage_layout.v1"
    layout.write_text(json.dumps(data))
    with pytest.raises(ValueError):build_problem(layout)


def results_fixture(tmp_path, monkeypatch):
    from test_refactor_component_cost_pool import model
    from distillschedule.search.cost_model import ComponentContract
    _, _, layout = planning_fixture(tmp_path, monkeypatch)
    results = []
    for reference in json.loads(layout.read_text())["fragments"]:
        fragment = json.loads(open(reference["path"]).read())
        impl, data = next(iter(fragment["implementations"].items()))
        grouped = {}
        for index, row in enumerate(data["ranks"][0]["stages"]):
            grouped.setdefault(row["key"], []).append(index)
        records = []
        for key, indices in grouped.items():
            stages = [r["stages"][i] for i in indices for r in data["ranks"]]
            samples = tuple(max(r["stages"][i]["duration_ns"] for r in data["ranks"]) for i in indices)
            cost = model(tmp_path, impl+key, samples=samples,
                workspace=max(s["workspace_bytes_per_rank"] for s in stages),
                contract=ComponentContract.from_dict(stages[0]["component_contract"]),
                binding=fragment["binding"])
            records.extend(cost.records)
        cost_path = tmp_path/f"{impl}-costs.json"
        cost_path.write_text(json.dumps(CostModel(fragment["binding"],tuple(records)).to_dict()))
        path = tmp_path/f"{impl}-result.json"
        path.write_text(json.dumps(dict(status="PASS_NATIVE_STAGE_CALIBRATION", rank_count=16,
            source_head="a"*40, formal_timing_eligible=False, implementation_id=impl,
            layout_fragment=reference, operation_costs={"path":str(cost_path),"sha256":file_sha256(cost_path)})))
        results.append(path)
    return results


def test_assemble_actual_result_protocol_to_pooled_costs_and_problem(tmp_path,monkeypatch):
    from distillschedule.search.wan_action_model_adapter import assemble_calibrations
    from distillschedule.search.plan_ir import PlanningProblem
    result = assemble_calibrations(results_fixture(tmp_path,monkeypatch), output=tmp_path/"assembled",
                                  expected_head="a"*40)
    assert result["gpu_execution"] == "NOT_RUN" and result["claimable_speedup"] is False
    assert result["calibrated_implementations"] == 2
    problem = PlanningProblem.from_dict(json.loads(open(result["problem"]).read()))
    costs = CostModel.from_dict(json.loads(open(result["cost_model"]).read()))
    costs.verify_receipts()
    prediction = search_plans(problem,costs,exhaustive=True,max_proposals=60)
    assert len(prediction["ranked"]) == 10
    assert prediction["calibration_requests"]
    # Every pooled value still resolves to its original calibration source.
    assert all(json.loads(open(r.evidence["source_receipt"]).read())["component_sources"] for r in costs.records)


def test_assembler_rejects_a_cost_model_unrelated_to_its_fragment(tmp_path,monkeypatch):
    from distillschedule.search.wan_action_model_adapter import assemble_calibrations
    results = results_fixture(tmp_path,monkeypatch)
    a,b = [json.loads(p.read_text()) for p in results]
    a["operation_costs"] = b["operation_costs"]
    results[0].write_text(json.dumps(a))
    with pytest.raises(ValueError, match="fragment.*cost"):
        assemble_calibrations(results,output=tmp_path/"rejected",expected_head="a"*40)
