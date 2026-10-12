"""Synthetic costs exercise original-strategy selection; no GPU evidence."""
from dataclasses import replace
import itertools
import pytest
from distillschedule.strategies.wan import OriginalWanTask as Task,OriginalWanStrategy as Strategy,candidates
from distillschedule.strategies.wan_planning import action_graph,component_contract,build_problem,select_plan,validate_plan,assignment
from distillschedule.search.cost_model import CostModel,OperationCost
from distillschedule.search.simulator import simulate
from distillschedule.workloads.best_known.registry import fingerprint


def fixture(task,winner=None,capacity=1000):
    versions=dict(generator=2,critic=10,ema=10)
    graph=action_graph(task,versions)
    binding={k:'a'*64 for k in ('hardware_fingerprint','environment_fingerprint','asset_fingerprint','semantic_fingerprint','precision_fingerprint')}
    binding.update(source_head='b'*40,semantic_fingerprint=task.digest)
    winner=winner or Strategy('resident',2048,True)
    records=[]; memory={}
    for s in candidates(task):
        if s.rollout_window!=1:continue
        memory[s.implementation.id]=100 if s.placement=='fsdp' else 300
        for op in graph.operations:
            c=component_contract(task,op,s)
            ns=10 if s.implementation==winner.implementation else 50
            records.append(OperationCost(c.digest,(ns,),50,{'component_contract':c.to_dict()}))
    costs=CostModel(binding,tuple(records),scope='CPU_FIXTURE_ONLY')
    p=build_problem(task,costs,initial_versions=versions,variant_memory=memory,
        output_bytes={o.operation_id:50 for o in graph.operations if o.metadata['action'].startswith('r')},
        memory_limits={r:capacity for r in range(16)})
    return p,costs

@pytest.mark.parametrize('n',[1,2,5,8])
@pytest.mark.parametrize('winner',[Strategy('resident',2048,True),Strategy('resident',1024,False),Strategy('fsdp')])
def test_selects_existing_optimization_according_to_cost_not_model_name(n,winner):
    t=Task(fake_updates=n);p,c=fixture(t,winner)
    plan=select_plan(t,c,p)
    s=validate_plan(plan,t,p,c)
    assert s.implementation==winner.implementation
    assert plan['environment']['DISTILLSCHEDULE_SEARCH_CANDIDATE_ID']==winner.implementation.id
    assert plan['native_placement']==winner.implementation.native_placement
    assert plan['claimable_speedup'] is False
    oracle=[]
    for candidate in candidates(t):
        r=simulate(p,assignment(candidate),c)
        if r['status']=='PREDICTED_FEASIBLE':oracle.append(r['predicted_cycle2_ns'])
    assert plan['predicted']['predicted_cycle2_ns']==min(oracle)


def test_capacity_can_change_placement_without_changing_algorithm_work():
    t=Task(fake_updates=2);p,c=fixture(t,capacity=250)
    plan=select_plan(t,c,p)
    assert plan['strategy']['placement']=='fsdp'
    assert p.graph.metadata['final_versions']==dict(generator=4,critic=14,ema=14)
    assert plan['search_diagnostics']['capacity_pruned']>0

@pytest.mark.parametrize('field,value',[('fake_updates',1),('resolution',(256,384)),('stage2_steps',(3,3,2))])
def test_new_parameters_cannot_reuse_old_costs(field,value):
    t=Task(fake_updates=2);p,c=fixture(t)
    with pytest.raises(ValueError):select_plan(replace(t,**{field:value}),c,p)


def test_rehashed_wrong_execution_controls_rejected():
    t=Task(fake_updates=1);p,c=fixture(t);plan=select_plan(t,c,p)
    plan['environment']['DISTILLSCHEDULE_SEARCH_CANDIDATE_ID']='joint-wan1-fsdp-c512-e0'
    plan.pop('plan_sha256');plan['plan_sha256']=fingerprint(plan)
    with pytest.raises(ValueError,match='dispatch'):validate_plan(plan,t,p,c)
