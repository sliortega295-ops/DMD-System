"""Small CPU oracles; no generated timing is native evidence."""
from dataclasses import replace
import itertools
import json
from pathlib import Path
import random
import subprocess
import sys

import pytest

from distillschedule.core.graph import Dependency
from distillschedule.search.hierarchical import bound_partial, search_hierarchical
from distillschedule.search.model_search import search_plans, validate_saved_prediction
from distillschedule.search.simulator import simulate
from test_refactor_model_planning import fixture


def oracle(p, costs):
    rows = []
    names = sorted(p.decisions)
    for values in itertools.product(*(p.decisions[k] for k in names)):
        try:
            r = simulate(p, dict(zip(names, values)), costs)
        except ValueError:
            continue
        if r['status'] == 'PREDICTED_FEASIBLE':
            rows.append(r)
    return min((r['predicted_cycle2_ns'] for r in rows), default=None)


@pytest.mark.parametrize('limit', [55, 75, 80, 100, 200])
@pytest.mark.parametrize('prune', [False, True])
def test_hierarchical_matches_all_assignments_with_or_without_pruning(limit, prune):
    p, c, _ = fixture()
    p = replace(p, memory_limits={0: limit, 1: limit}, seeds=())
    r = search_plans(p, c, algorithm='hierarchical', max_proposals=1000, prune=prune)
    assert (r['predicted_best']['predicted_cycle2_ns'] if r['predicted_best'] else None) == oracle(p, c)
    assert r['search_diagnostics']['optimality'] == 'DISCRETE_SIMULATOR_OPTIMAL'
    assert not r['claimable_speedup'] and r['gpu_execution'] == 'NOT_RUN'
    if r['predicted_best']:
        assert validate_saved_prediction(r, p, c) == r['predicted_best']


@pytest.mark.parametrize('seed', range(15))
def test_random_costs_and_memory_match_exhaustive_oracle(seed):
    rng = random.Random(seed)
    p, c, _ = fixture()
    p = replace(p, seeds=(), memory_limits={0:rng.randrange(60,130),1:rng.randrange(60,130)})
    c = replace(c, records=tuple(replace(r, samples_ns=(rng.randrange(1,16),),
                                       workspace_bytes_per_rank=rng.randrange(0,50)) for r in c.records))
    r = search_hierarchical(p,c,max_proposals=1000)
    assert (r['predicted_best']['predicted_cycle2_ns'] if r['predicted_best'] else None) == oracle(p,c)


def test_time_lower_bound_requires_a_proven_cycle_fence():
    p,c,a=fixture()
    assert not bound_partial(p,c,{}).fenced
    assert bound_partial(p,c,{'lookahead':'depth1'}).cycle2_ns == 0
    assert bound_partial(p,c,{'lookahead':'off'}).fenced
    # Verify every partial bound against all valid completions, including the
    # early-cycle2 path which defeats whole-DAG cycle2 subtraction heuristics.
    names=list(p.decisions)
    for vals in itertools.product(*[(None,*p.decisions[k]) for k in names]):
        part={k:v for k,v in zip(names,vals) if v is not None}
        b=bound_partial(p,c,part)
        for values in itertools.product(*(p.decisions[k] for k in names)):
            full=dict(zip(names,values))
            if any(full[k]!=v for k,v in part.items()):continue
            r=simulate(p,full,c)
            if r['status']=='PREDICTED_FEASIBLE':
                assert b.reason is None
                assert b.cycle2_ns <= r['predicted_cycle2_ns']
                assert all(dict(b.memory_bytes)[v['rank']] <= v['peak_bytes'] for v in r['memory'])


def test_equivalent_unused_decision_is_memoized_but_cached_state_is_not():
    p,c,_=fixture()
    p=replace(p,seeds=(),decisions={**p.decisions,'unused':tuple(str(i) for i in range(20))})
    r=search_hierarchical(p,c,max_proposals=5000)
    assert r['search_diagnostics']['equivalent_pruned'] > 0
    assert r['unique_evaluated'] <= 8
    assert r['predicted_best']['predicted_cycle2_ns']==oracle(p,c)
    assert r['search_diagnostics']['structural_decisions']==['sharing']


def test_capacity_and_forbidden_pruning_have_separate_reasons():
    p,c,a=fixture()
    p=replace(p,seeds=(),memory_limits={0:70,1:70},forbidden=({'kernel':'slow','sharing':'off'},))
    r=search_hierarchical(p,c)
    assert r['search_diagnostics']['capacity_pruned'] > 0
    assert r['search_diagnostics']['contract_pruned'] > 0
    assert (r['predicted_best']['predicted_cycle2_ns'] if r['predicted_best'] else None)==oracle(p,c)


def test_missing_costs_are_requested_and_never_count_as_infeasible_gpu_runs():
    p,c,_=fixture()
    r=search_hierarchical(replace(p,seeds=()),replace(c,records=()))
    assert r['status']=='NO_CALIBRATED_FEASIBLE_PLAN'
    assert len(r['calibration_requests'])==4
    assert r['search_diagnostics']['optimality']=='CALIBRATED_SUBSET_OPTIMAL'
    assert all(x['status']=='UNCALIBRATED' for x in r['rejections'])


def test_conditional_cycle_pruned_without_moving_versions():
    p,c,_=fixture()
    p=replace(p,seeds=(),conditional_dependencies=(*p.conditional_dependencies,
        {'when':{'sharing':'on'},'source':'2c','target':'1f'}))
    r=search_hierarchical(p,c)
    assert r['search_diagnostics']['contract_pruned']>0
    assert r['predicted_best']['assignment']['sharing']=='off'
    assert r['predicted_best']['predicted_cycle2_ns']==oracle(p,c)


@pytest.mark.parametrize('kw', [{'beam_width':1},{'max_proposals':1}])
def test_beam_or_budget_truncation_never_claims_optimality(kw):
    p,c,_=fixture()
    r=search_hierarchical(p,c,**kw)
    assert r['search_diagnostics']['optimality']=='BEST_FOUND_ONLY'


def test_opt_in_does_not_change_legacy_search_or_rng():
    p,c,_=fixture()
    rng=random.getstate()
    a=search_plans(p,c,seed=7,max_proposals=30)
    search_hierarchical(p,c)
    b=search_plans(p,c,seed=7,max_proposals=30)
    assert a['journal']==b['journal'] and random.getstate()==rng
    assert a['predicted_best']==b['predicted_best']


@pytest.mark.parametrize('kw',[{'beam_width':True},{'max_proposals':0},{'timeout_seconds':float('nan')},
 {'structural_decisions':['unknown']},{'structural_decisions':['sharing','sharing']}])
def test_invalid_search_options_fail(kw):
    p,c,_=fixture()
    with pytest.raises(ValueError):search_hierarchical(p,c,**kw)


def test_search_result_uses_existing_native_wan_lowerer(tmp_path):
    from test_refactor_wan_model_adapter import calibration_fixture
    from distillschedule.search.wan_model_adapter import lower_prediction, validate_lowered_plan
    from distillschedule.search.wan_implementation import WanImplementation
    p,c=calibration_fixture(tmp_path)
    r=search_hierarchical(p,c)
    assert r['predicted_best']['assignment']==WanImplementation('resident',1024,True).assignment
    plan=lower_prediction(r,p,c,run_id='hierarchical-cpu-fixture-no-gpu')
    validate_lowered_plan(plan)
    assert plan['search_candidate']['id']==WanImplementation('resident',1024,True).id


@pytest.mark.parametrize('workload', ['wan13b_nogan_resident_best','livewan_crossscale_pipeline_best',
 'minimax_h3_grid1000_best','helios14b_nogan_best','helios14b_gan_best'])
def test_all_five_cli_capabilities_accept_hierarchical_without_gpu(workload):
    root=Path(__file__).parents[1]
    p=subprocess.run([sys.executable,str(root/'scripts/distillctl.py'),'workloads','search',
        '--strategy','model-based','--search-algorithm','hierarchical','--workload',workload,'--dry-run'],
        capture_output=True,text=True,cwd=root)
    assert p.returncode==0,p.stderr
    value=json.loads(p.stdout)
    assert value['search_algorithm']=='hierarchical'
    assert value['gpu_execution']=='NOT_RUN'


def test_cli_saved_fixture_replays_and_has_no_native_permission(tmp_path):
    p,c,_=fixture()
    pp=tmp_path/'p.json';cc=tmp_path/'c.json';out=tmp_path/'out.json'
    pp.write_text(json.dumps(p.to_dict()));cc.write_text(json.dumps(c.to_dict()))
    root=Path(__file__).parents[1]
    result=subprocess.run([sys.executable,str(root/'scripts/distillctl.py'),'workloads','search',
        '--strategy','model-based','--search-algorithm','hierarchical','--workload',p.workload,
        '--problem',str(pp),'--cost-model',str(cc),'--output',str(out)],capture_output=True,text=True,cwd=root)
    assert result.returncode==0,result.stderr
    r=json.loads(out.read_text())
    assert r['predicted_best']['predicted_cycle2_ns']==oracle(p,c)
    assert r['native_trial_budget']['actual_runs']==0


@pytest.mark.parametrize('workload',['livewan_crossscale_pipeline_best','minimax_h3_grid1000_best'])
def test_existing_macro_adapter_lowering_with_new_selector(tmp_path,monkeypatch,workload):
    # Reuse the original source-bound synthetic receipt/lowering acceptance.
    # Disable ranking reductions so its full calibrated-subspace assertions
    # still apply. This runs NO model or GPU worker.
    from distillschedule.search import model_search
    from test_refactor_native_macro_adapter import test_partial_calibrated_subspace_never_invents_missing_costs
    original=model_search.search_plans
    def hierarchical(*args,**kwargs):
        kwargs.pop('exhaustive',None)
        kwargs.update(algorithm='hierarchical',prune=False,max_proposals=1000)
        return original(*args,**kwargs)
    monkeypatch.setattr(model_search,'search_plans',hierarchical)
    test_partial_calibrated_subspace_never_invents_missing_costs(tmp_path,workload)


@pytest.mark.parametrize('variant,window',[('no-gan',3),('gan',5)])
def test_existing_helios_component_adapter_with_new_selector(tmp_path,monkeypatch,variant,window):
    from distillschedule.search import model_search
    from test_refactor_helios_search_bridge import test_component_search_cpu_fixture_preserves_native_decisions_and_missing_costs
    original=model_search.search_plans
    def hierarchical(*args,**kwargs):
        kwargs.pop('exhaustive',None)
        kwargs.update(algorithm='hierarchical',prune=False,max_proposals=1000)
        return original(*args,**kwargs)
    monkeypatch.setattr(model_search,'search_plans',hierarchical)
    test_component_search_cpu_fixture_preserves_native_decisions_and_missing_costs(tmp_path,monkeypatch,variant,window)


def test_wan_action_calibration_library_also_works_with_hierarchical(tmp_path,monkeypatch):
    from test_refactor_wan_action_model import planning_fixture
    p,c,_=planning_fixture(tmp_path,monkeypatch)
    r=search_hierarchical(p,c,max_proposals=1000)
    assert r['predicted_best']['predicted_cycle2_ns']==oracle(p,c)
    assert r['calibration_requests'] and not r['claimable_speedup']
