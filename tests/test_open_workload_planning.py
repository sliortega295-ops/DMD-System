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
    assert len(m['variants'])==6 and m['budget']['actual']==9
    assert len(m['failed'])==2 and all(row['reason']=='INAPPLICABLE_NATIVE_SELECTIVE_CAPABILITY' for row in m['failed'])
    assert all(row['native_capability_evidence']['active'] for row in m['variants'])
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


def test_serialized_identity_retains_nested_tuple_semantics_for_local_execution(tmp_path):
    """Algorithm-native tuples cross the same JSON boundary as replicated identity."""
    workload = build(n=1)
    workload.identity_extra = {'algorithm': {'sigmas': (1., .5, 0.), 'nested': [(('exact', 7),)]}}
    workload.program = replace(workload.program,
        parameters={**workload.program.parameters, 'algorithm_sigmas': (1., .5, 0.)},
        calls=tuple(replace(call, attributes={**call.attributes, 'algorithm_contract': {'sigmas': (1., .5, 0.)}})
                    for call in workload.program.calls))
    manifest = calibrate(workload, tmp_path/'costs', samples=1, max_candidates=1, max_program_runs=2)
    document = plan(tmp_path/'costs/calibration.json', memory_limit_bytes=10**8)
    assert manifest['identity'] == document['identity'] == json.loads(json.dumps(workload.identity))
    write_new_json(tmp_path/'plan.json', document)
    assert run(workload, tmp_path/'plan.json', tmp_path/'run')[0]['reference_gate']['status'] == 'PASS'


@pytest.mark.parametrize('value', [{1, 2}, float('nan')])
def test_non_json_identity_rejected_before_trial_construction(tmp_path, monkeypatch, value):
    from distillschedule.open_workload.torch_backend import Session
    workload = build(n=1); workload.identity_extra = {'not_json': value}
    def forbidden(*args, **kwargs): raise AssertionError('invalid identity reached native trial construction')
    monkeypatch.setattr(Session, '__init__', forbidden)
    with pytest.raises((TypeError, ValueError)):
        calibrate(workload, tmp_path/'rejected', samples=1, max_candidates=1, max_program_runs=2)
    assert not (tmp_path/'rejected').exists()


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
