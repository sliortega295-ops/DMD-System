from copy import deepcopy
from dataclasses import replace
import json
import pytest
from distillschedule.open_workload.planning import calibrate,plan,run,load_calibration
from distillschedule.workloads.best_known.registry import write_new_json,fingerprint
from test_open_workload_execution import build


def test_real_measurements_to_dp_to_native_execution(tmp_path):
    w=build(n=2)
    m=calibrate(w,tmp_path/'costs',samples=1)
    assert len(m['variants'])==8 and m['budget']['actual']==9
    p=plan(tmp_path/'costs/calibration.json',memory_limit_bytes=1_000_000)
    write_new_json(tmp_path/'plan.json',p)
    out,session,values=run(w,tmp_path/'plan.json',tmp_path/'run')
    assert out['status']=='PASS_SELECTED_PROGRAM'
    assert out['reference_gate']['status']=='PASS'
    assert session.receipt['dispatcher']['final_versions']['learner']==2
    assert out['gpu_execution']=='NOT_RUN'
    assert out['selected_implementation']==p['selected']['implementation']
    assert out['executed_order']==p['selected']['order']
    assert not out['claimable_speedup'] and out['gpu_execution']=='NOT_RUN'


def test_capacity_can_select_shared_implementation_not_hardcoded_winner(tmp_path):
    w=build(n=1,width=64,length=1)
    m=calibrate(w,tmp_path/'costs',samples=1)
    p=plan(tmp_path/'costs/calibration.json',memory_limit_bytes=10**8)
    rows=p['search']['ranked']
    shared=min(x['memory'][0]['peak_bytes'] for x in rows if x['implementation']['share_immutable'])
    private=min(x['memory'][0]['peak_bytes'] for x in rows if not x['implementation']['share_immutable'])
    assert shared < private
    constrained=plan(tmp_path/'costs/calibration.json',memory_limit_bytes=shared)
    assert constrained['selected']['implementation']['share_immutable'] is True
    write_new_json(tmp_path/'p.json',constrained)
    assert run(w,tmp_path/'p.json',tmp_path/'run')[0]['status']=='PASS_SELECTED_PROGRAM'


def test_changed_parameters_require_new_measurements(tmp_path):
    w=build(n=1);calibrate(w,tmp_path/'costs',samples=1)
    p=plan(tmp_path/'costs/calibration.json',memory_limit_bytes=10**8)
    write_new_json(tmp_path/'p.json',p)
    with pytest.raises(ValueError,match='changed'):
        run(build(n=2),tmp_path/'p.json',tmp_path/'run')


@pytest.mark.parametrize('field',['time','memory','order','implementation'])
def test_rehashed_plan_still_requires_real_costs_and_legality(tmp_path,field):
    w=build(n=1);calibrate(w,tmp_path/'costs',samples=1)
    p=plan(tmp_path/'costs/calibration.json',memory_limit_bytes=10**8)
    s=p['selected']
    if field=='time': s['predicted_makespan_ns']=0
    if field=='memory': s['memory'][0]['peak_bytes']=0
    if field=='order': s['order']=list(reversed(s['order']))
    if field=='implementation': s['implementation']['backend']='invented-new-winner'
    p.pop('plan_sha256');p['plan_sha256']=fingerprint(p);write_new_json(tmp_path/'p.json',p)
    with pytest.raises(ValueError): run(w,tmp_path/'p.json',tmp_path/'run')


def test_small_budget_reports_incomplete_enumeration(tmp_path):
    w=build(n=1)
    m=calibrate(w,tmp_path/'costs',samples=1,max_candidates=1,max_program_runs=2)
    assert not m['inventory']['exhaustive'] and len(m['variants'])==1
    with pytest.raises(ValueError,match='budget'):
        calibrate(w,tmp_path/'fail',samples=2,max_candidates=4,max_program_runs=4)
    assert not (tmp_path/'fail').exists()


def test_plan_cpu_metadata_only(tmp_path,monkeypatch):
    w=build(n=1);calibrate(w,tmp_path/'costs',samples=1)
    import distillschedule.open_workload.torch_backend as backend
    monkeypatch.setattr(backend,'Session',lambda *_:(_ for _ in ()).throw(AssertionError('model in search')))
    p=plan(tmp_path/'costs/calibration.json',memory_limit_bytes=10**8)
    assert p['gpu_execution']=='NOT_RUN_BY_SEARCH'
