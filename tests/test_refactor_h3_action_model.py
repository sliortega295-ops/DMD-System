"""Synthetic model-search protocol only; never native H3 measurement evidence."""
from copy import deepcopy
from pathlib import Path
import json

import pytest

from test_refactor_h3_action_calibration import fixture, work, Tensor
from test_refactor_model_planning import BINDING
from distillschedule.search import h3_action_model_adapter as m
from distillschedule.search.h3_action_costs import action_graph, PRECISION, H3
from distillschedule.search.model_search import search_plans
from distillschedule.search.simulator import simulate
from distillschedule.search.contracts import semantic_contract
from distillschedule.workloads.best_known.registry import file_sha256, fingerprint, load_profile


def data(tmp_path, monkeypatch):
    b,c,impl=fixture()
    for n,op in enumerate(action_graph(3).operations):
        with c.stage(op.metadata['cycle'],op.metadata['action']) as sample:
            for row in work(op.metadata['action'][0]):c.note_forward(row)
            b.body()
        if op.metadata['action'].startswith('r'): sample.record_output(Tensor(n+1,20))
    row=c.finish()
    # Only the surrounding native receipt validator is stubbed. The collector
    # identity, actual call contracts, storage, hash, cost reconciliation run.
    monkeypatch.setattr(m,'summarize_native_two_cycle',lambda *a,**k:dict(calibration_only=True,run_kind='calibration'))
    monkeypatch.setattr(m,'_verify_run_binding',lambda *a,**k: None)
    monkeypatch.setattr(m,'git_head',lambda _: 'a'*40)
    row['evidence_scope']='MEASURED'
    row['post_setup_memory']['driver_free_bytes']=40*1024**3
    refs=[]
    for rank in range(16):
        value=deepcopy(row);value['rank']=rank
        p=tmp_path/f'rank{rank}.json';p.write_text(json.dumps(dict(rank=rank,stage_calibration=value)))
        refs.append(dict(path=str(p),sha256=file_sha256(p)))
    binding={**BINDING,'precision_fingerprint':PRECISION,
             'semantic_fingerprint':fingerprint(semantic_contract(load_profile(H3)))}
    p=tmp_path/'layout.json'
    p.write_text(json.dumps(dict(schema_version=m.SCHEMA, binding=binding,
                                implementation_id=impl.id,rank_receipts=refs)))
    return p,impl


def test_calibrated_components_search_new_windows_without_invented_speedup(tmp_path,monkeypatch):
    layout,impl=data(tmp_path,monkeypatch)
    problem,costs=m.assemble_calibration(layout,output=tmp_path/'out')
    costs.verify_receipts()
    assert len(problem.graph.operations)==22
    result=search_plans(problem,costs,max_proposals=20,exhaustive=True)
    assert result['status']=='PASS_CPU_SEARCH'
    assert len(result['ranked'])==5 and len(result['rejections'])==15
    assert all(r['status']=='UNCALIBRATED' for r in result['rejections'])
    assert len({r['predicted_cycle2_ns'] for r in result['ranked']})==1
    one=simulate(problem,{**impl.assignment,'rollout_window':'1'},costs)
    five=simulate(problem,{**impl.assignment,'rollout_window':'5'},costs)
    assert one['memory'][0]['peak_bytes'] < five['memory'][0]['peak_bytes']
    assert not result['claimable_speedup']


def test_changed_raw_layout_and_unproven_native_costs_fail_closed(tmp_path,monkeypatch):
    layout,_=data(tmp_path,monkeypatch)
    path=tmp_path/'rank0.json'
    path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError,match='rank hash'):m.build_problem(layout)


def test_changed_actual_call_inventory_cannot_be_repaired_by_rehash(tmp_path,monkeypatch):
    layout,_=data(tmp_path,monkeypatch)
    p=tmp_path/'rank0.json';v=json.loads(p.read_text())
    v['stage_calibration']['work_signatures']['cycle1.g']['application_calls'].pop()
    p.write_text(json.dumps(v));l=json.loads(layout.read_text());l['rank_receipts'][0]['sha256']=file_sha256(p)
    layout.write_text(json.dumps(l))
    with pytest.raises(ValueError,match='inventory'):m.build_problem(layout)


def test_rehashed_fabricated_cost_is_rejected_against_native_observations(tmp_path,monkeypatch):
    from dataclasses import replace
    from distillschedule.search.cost_model import CostModel
    layout,_=data(tmp_path,monkeypatch)
    _,costs=m.assemble_calibration(layout,output=tmp_path/'out')
    record=costs.records[0];path=Path(record.evidence['source_receipt'])
    raw=json.loads(path.read_text());raw['samples_ns'][0]+=100
    path.write_text(json.dumps(raw))
    bad=replace(record,samples_ns=tuple(raw['samples_ns']),
        evidence={**record.evidence,'source_receipt_sha256':file_sha256(path)})
    with pytest.raises(ValueError,match='native rank observations'):
        CostModel(costs.binding,(bad,*costs.records[1:])).verify_receipts()


def test_launch_asset_binding_cannot_be_relabelled(tmp_path,monkeypatch):
    from distillschedule.search import execution
    monkeypatch.setattr(execution,'validated_profile',lambda *a,**k: {})
    path=tmp_path/'plan.json'
    path.write_text(json.dumps(dict(mode='calibration',search_candidate={'id':'joint-h3v2-d2-n128-w3'},
        integration_head='a'*40,asset_fingerprint='b'*64,workload_semantics_hash='c'*64)))
    layout=dict(execution_receipt=dict(path=str(path),sha256=file_sha256(path)),
        implementation_id='joint-h3v2-d2-n128-w3',binding=dict(source_head='a'*40,asset_fingerprint='d'*64,
                                                           semantic_fingerprint='c'*64))
    with pytest.raises(ValueError,match='actual launch'):m._verify_run_binding(layout,[],{},tmp_path)


def test_stripped_optional_metadata_cannot_bypass_h3_cost_provenance(tmp_path,monkeypatch):
    from dataclasses import replace
    from distillschedule.search.cost_model import CostModel
    layout,_=data(tmp_path,monkeypatch)
    _,costs=m.assemble_calibration(layout,output=tmp_path/'out')
    record=costs.records[0];path=Path(record.evidence['source_receipt'])
    raw=json.loads(path.read_text());raw['samples_ns']=[1]
    del raw['component_contract'];del raw['h3_action_layout'];path.write_text(json.dumps(raw))
    evidence={k:v for k,v in record.evidence.items() if k!='component_contract'}
    evidence['source_receipt_sha256']=file_sha256(path)
    bad=replace(record,samples_ns=(1,),evidence=evidence)
    with pytest.raises(ValueError,match='component provenance'):
        CostModel(costs.binding,(bad,*costs.records[1:])).verify_receipts()
