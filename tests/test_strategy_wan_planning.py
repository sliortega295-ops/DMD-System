"""Synthetic model costs validate selection, NOT any GPU speedup."""
from dataclasses import replace
import json
from pathlib import Path
import pytest
from distillschedule.workloads.best_known.registry import fingerprint,write_new_json,file_sha256
from distillschedule.search.cost_model import CostModel,OperationCost
from distillschedule.runtime.parameterized_cycle import build_parameterized_cycle_graph
from distillschedule.strategies.wan_task import OriginalWanTask
from distillschedule.strategies.wan_strategy import WanStrategy
from distillschedule.strategies.wan_calibration import component,two_cycles
from distillschedule.strategies.wan_planning import COLLECTION,plan,validate_plan


def collection(tmp_path,task,candidates):
    versions=dict(generator=3,critic=10+task.fake_updates,ema=10+len(task.ema_after))
    graph=build_parameterized_cycle_graph(source_steps=task.source_steps,initial_versions=versions,ema_after=task.ema_after)
    work={op.operation_id:{'shape':list(task.latent_shape),'synthetic':True} for op in graph.operations}
    binding={k:'1'*64 for k in ('hardware_fingerprint','environment_fingerprint','asset_fingerprint','semantic_fingerprint','precision_fingerprint')}
    binding['source_head']='0'*40;variants=[]
    for i,(strategy,duration,static,workspace,limit) in enumerate(candidates):
        records=[]
        for op in graph.operations:
            contract=component(task,op.operation_id,strategy,cycle=2,work=work)
            records.append(OperationCost(contract.digest,(duration,),workspace,{'component_contract':contract.to_dict()}))
        costs=CostModel(binding,tuple(records),scope='CPU_FIXTURE_ONLY')
        p=tmp_path/f'costs{i}.json';write_new_json(p,costs.to_dict())
        variants.append(dict(strategy=strategy.to_dict(),cost_model=str(p),cost_sha256=file_sha256(p),
                static_bytes=static,memory_limit_bytes=limit,output_bytes={f'r{j}':20 for j in range(task.fake_updates)},rank_sources=[]))
    doc=dict(schema_version=COLLECTION,task=task.to_dict(),identity={'synthetic':True},work=work,
             warm_initial_versions=versions,variants=variants,evidence_scope='CPU_FIXTURE_ONLY')
    p=tmp_path/'collection.json';write_new_json(p,doc);return p


@pytest.mark.parametrize('n',[1,3,8])
def test_selects_old_optimized_combination_when_its_costs_win(tmp_path,n):
    t=OriginalWanTask(fake_updates=n,first_step=2*n)
    path=collection(tmp_path,t,[(WanStrategy('fsdp',512,False),100,50,20,200),
                              (WanStrategy('resident',2048,True),50,100,30,200),
                              (WanStrategy('resident',512,False),70,100,30,200)])
    p=plan(t,[path])
    assert p['selected']['strategy']['placement']=='resident'
    assert p['selected']['strategy']['chunk_tokens']==2048
    assert p['selected']['strategy']['lora_epilogue'] is True
    assert len(p['ranked'])==3
    assert p['evidence_scope']=='CPU_FIXTURE_ONLY' and not p['claimable_speedup']


def test_capacity_can_select_fsdp_instead_of_hardcoded_resident(tmp_path):
    t=OriginalWanTask(fake_updates=2,first_step=4)
    path=collection(tmp_path,t,[(WanStrategy('fsdp',512,False),100,50,20,200),
                              (WanStrategy('resident',2048,True),40,180,30,200)])
    p=plan(t,[path]);assert p['selected']['strategy']['placement']=='fsdp'
    assert len(p['rejected'])==1


def test_slower_fused_cost_selects_unfused(tmp_path):
    t=OriginalWanTask(fake_updates=1,first_step=2)
    path=collection(tmp_path,t,[(WanStrategy('resident',512,False),30,50,10,200),
                              (WanStrategy('resident',2048,True),50,50,10,200)])
    assert plan(t,[path])['selected']['strategy']['lora_epilogue'] is False


def test_changed_task_does_not_borrow_measurements(tmp_path):
    t=OriginalWanTask();path=collection(tmp_path,t,[(WanStrategy('fsdp',512,False),100,50,20,200)])
    with pytest.raises(ValueError,match='task/schema'):
        plan(replace(t,video_height=512),[path])


def test_missing_cost_not_replaced_by_known_winner(tmp_path):
    t=OriginalWanTask();path=collection(tmp_path,t,[(WanStrategy('fsdp',512,False),100,50,20,200)])
    result=plan(t,[path]);assert result['selected']['strategy']['placement']=='fsdp'
    assert len(result['uncalibrated'])==6
    write_new_json(tmp_path/'plan.json',result)
    with pytest.raises(ValueError,match='fixture plans'):
        validate_plan(t,tmp_path/'plan.json')


def test_memory_and_schedule_use_one_world16_pool(tmp_path):
    t=OriginalWanTask(fake_updates=3,first_step=6)
    path=collection(tmp_path,t,[(WanStrategy('resident',2048,True),50,100,30,155)])
    p=plan(t,[path]);assert p['selected']['strategy']['rollout_window']==1
    assert [r['rank'] for r in p['selected']['memory']]==list(range(16))
    rows=p['selected']['timeline']
    assert all(rows[i]['end_ns']==rows[i+1]['start_ns'] for i in range(len(rows)-1))


def test_parameterized_calibration_graph_not_still_five():
    from distillschedule.search.plan_ir import validate_versions
    for n in (1,3,8):
        t=OriginalWanTask(fake_updates=n,first_step=n*2)
        g=two_cycles(t,dict(generator=0,critic=0,ema=0),n)
        validate_versions(g)
        assert len(g.operations)==2*(1+3*n)
        assert g.metadata['final_versions']==dict(generator=2,critic=2*n,ema=2*n)
