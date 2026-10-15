"""Fine-grained CPU prediction and real held-out native FIFO execution.

Synthetic durations below test solver decisions only; real calibration tests
never prescribe a winner from noisy CPU timing or claim a GPU speedup.
"""
from datetime import timedelta
from dataclasses import replace
from pathlib import Path
import json
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from distillschedule.open_workload.role_program import RoleTask,PoolLayout,AllocatedWorld
from distillschedule.open_workload.role_transport import RoleFunctions,RolePoolExecutor
from distillschedule.open_workload.role_prediction import graph_for,generator_order,contract,validate_execution
from distillschedule.open_workload.role_planning import calibrate,search,run_selected
from distillschedule.search.cost_model import CostModel,OperationCost
from distillschedule.search.plan_ir import BindingRule,StateAllocation,Lifetime
from distillschedule.search.service import BoundCandidate,select
from distillschedule.workloads.best_known.registry import fingerprint,write_new_json
from test_open_workload_role_transport import (
    make_model,make_opt,sample,generate,guidance,score_train,independent_reference,
)


def candidate(window,limit=10**7):
    task=RoleTask(4,(2,4));layout=PoolLayout((0,),(1,),window);hw=AllocatedWorld((limit,)*2)
    graph,specs=graph_for(task,layout,hw)
    binding={k:'a'*64 for k in ('hardware_fingerprint','environment_fingerprint','asset_fingerprint',
                               'semantic_fingerprint','precision_fingerprint')}
    binding['source_head']='a'*40
    keys={spec[0] for spec in specs.values()}
    records=[];rules={}
    for key in keys:
        c=contract(task,layout,key)
        records.append(OperationCost(c.digest,(20 if key.startswith('score') else 10,),0,
                                     {'component_contract':c.to_dict()}))
    costs=CostModel(binding,tuple(records),'CPU_FIXTURE_ONLY')
    for key,(value,ranks,resources) in specs.items():
        c=contract(task,layout,value)
        rules[key]=BindingRule(key,{},c.digest,ranks,resources,('static',),c.digest)
    states=(StateAllocation('static',(0,1),100,('generator','score'),True,'a'*64),)
    lifetimes=tuple(Lifetime(f'x{i}',f'g{i}',(f'b{i}',),(0,),100) for i in range(4))
    return BoundCandidate(str(window),layout.to_dict(),graph,rules,costs,states,lifetimes,
                          dict(enumerate(hw.capacities)),'TEST_ONLY')


def test_component_graph_predicts_window_benefit_and_lifetime_capacity():
    result=select([candidate(w) for w in (1,2,3,4)],max_states=50000,timeout_seconds=15)
    times={r['implementation']['window']:r['predicted_makespan_ns'] for r in result['ranked']}
    assert times[2]<times[1]
    capacity=select([candidate(w,250) for w in (1,2,3)],timeout_seconds=15)
    assert [r['implementation']['window'] for r in capacity['ranked']]==[1]
    assert len(capacity['rejected'])==2


def test_cost_projection_excludes_window_but_binds_real_pool_and_input():
    task=RoleTask(4,(2,4));a=PoolLayout((0,1),(2,3),1)
    assert contract(task,a,'forward:0').digest==contract(task,replace(a,window=2),'forward:0').digest
    assert contract(task,a,'forward:0').digest!=contract(RoleTask(4,(2,8)),a,'forward:0').digest
    assert contract(task,a,'forward:0').digest!=contract(task,PoolLayout((0,2),(1,3),1),'forward:0').digest
    with pytest.raises(ValueError,match='balanced replicated resident'):
        graph_for(task,PoolLayout((0,),(1,),1,'cpu_offload_prefetch'),
                  AllocatedWorld((10**6,)*2,host_capacities=(10**6,)*2))


def _worker(rank,rdzv,out,updates):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+rdzv,rank=rank,world_size=2,
                            timeout=timedelta(seconds=75))
    try:
        task=RoleTask(4,(2,4),score_updates=updates)
        bindings=RoleFunctions(make_model,make_opt,sample,generate,guidance,
                               score_train if updates else None,data_identity='e'*64)
        hw=AllocatedWorld((2**26,)*2)
        root=Path(out)
        manifest=calibrate(task,hw,bindings,root/'cal',generator_bytes=2000,score_bytes=2000,
                           max_candidates=4,samples=1,max_native_runs=3,cost_mode='components')
        assert manifest['budget']['calibration_runs']==2
        assert manifest['budget']['predicted_candidates']==4
        assert len(manifest['anchors'])==1
        raw=json.loads(Path(manifest['anchors'][0]['native_path']).read_text())
        assert all(r['layout']['window']==1 for r in raw['runs'])
        if rank==0:
            plan=search(root/'cal/calibration.json',timeout_seconds=30)
            assert len(plan['search']['ranked'])==4
            assert any(not p['was_full_cycle_measured'] for p in plan['prediction_provenance'].values())
            assert len(plan['selected']['order'])==21+int(bool(updates))
            # A corrupted raw observation cannot be covered by an old cost hash.
            path=Path(manifest['anchors'][0]['native_path']);data=path.read_bytes()
            path.write_text('{}')
            with pytest.raises(ValueError,match='changed component native receipt'):
                search(root/'cal/calibration.json')
            path.write_bytes(data)
            # Rehashing an edited aggregate must not disguise invented samples:
            # the source per-rank/native-link observations are re-derived.
            manifest_path=root/'cal/calibration.json'
            manifest_bytes=manifest_path.read_bytes()
            anchor=manifest['anchors'][0]
            cost_path=Path(anchor['cost_path']);cost_bytes=cost_path.read_bytes()
            costs=json.loads(cost_bytes);record=costs['records'][0]
            evidence_path=Path(record['evidence']['source_receipt'])
            evidence_bytes=evidence_path.read_bytes();evidence=json.loads(evidence_bytes)
            record['samples_ns'][0]+=123
            evidence['samples_ns']=record['samples_ns']
            evidence_path.write_text(json.dumps(evidence))
            from distillschedule.workloads.best_known.registry import file_sha256
            record['evidence']['source_receipt_sha256']=file_sha256(evidence_path)
            cost_path.write_text(json.dumps(costs));anchor['cost_sha256']=file_sha256(cost_path)
            manifest_path.write_text(json.dumps(manifest))
            with pytest.raises(ValueError,match='original native component'):
                search(manifest_path)
            evidence_path.write_bytes(evidence_bytes);cost_path.write_bytes(cost_bytes)
            manifest_path.write_bytes(manifest_bytes)
            write_new_json(root/'plan.json',plan)
        dist.barrier()
        receipt,module,optimizer=run_selected(task,hw,bindings,root/'plan.json',root/'selected')
        expected=independent_reference(task)['generator' if rank==0 else 'score']
        assert all(torch.allclose(v.detach(),expected[k],atol=1e-7,rtol=1e-6)
                   for k,v in module.state_dict().items())
        assert receipt['component_prediction_check']['status']=='PASS_NATIVE_FIFO_PROJECTION'
        assert not receipt['ranks'][rank]['components']['enabled']
        # Independently execute an explicitly held-out window as a lowering
        # correctness test, NOT mislabeled as another automatic search winner.
        runtime=RolePoolExecutor(task,PoolLayout((0,),(1,),2),hw,bindings,device='cpu')
        try:
            heldout,model,_=runtime.run()
            validate_execution(task,runtime.layout,heldout)
            assert all(torch.allclose(v.detach(),expected[k],atol=1e-7,rtol=1e-6)
                       for k,v in model.state_dict().items())
        finally:runtime.close()
    finally:dist.destroy_process_group()


@pytest.mark.parametrize('updates',[0,1])
def test_one_anchor_predicts_four_windows_then_executes_real_selected_plan(tmp_path,updates):
    mp.spawn(_worker,args=(str(tmp_path/'rdzv'),str(tmp_path),updates),nprocs=2,join=True)
    plan=json.loads((tmp_path/'plan.json').read_text())
    result=json.loads((tmp_path/'selected/execution.json').read_text())
    assert plan['claimable_speedup'] is False and result['claimable_speedup'] is False
    assert result['component_prediction_check']['relative_error']>=0


def test_transfer_cannot_masquerade_as_unreserved_compute_or_wrong_link():
    c=candidate(2);rules=dict(c.rules)
    a=rules['a0'];rules['a0']=replace(a,resources=('dma:0','pair:0:1'))
    with pytest.raises(ValueError,match='both endpoint'):
        select([replace(c,rules=rules)])
    rules['a0']=replace(a,resources=('dma:0','dma:1','pair:0:9'))
    with pytest.raises(ValueError,match='physical link'):
        select([replace(c,rules=rules)])


def test_public_component_mode_unknown_conv3d_has_unmeasured_targets(tmp_path):
    from test_open_workload_role_cli import FACTORY,PARAMETERS,invoke_torchrun
    output=tmp_path/'component-public'
    result=invoke_torchrun(tmp_path,['role-auto','--cost-mode','components',
        '--factory',str(FACTORY)+':build','--parameters',str(PARAMETERS),
        '--output',str(output),'--distributed','gloo','--memory-limit-bytes','10000000',
        '--host-memory-limit-bytes','10000000','--max-candidates','3',
        '--samples','1','--max-native-runs','3','--execute'])
    assert result.returncode==0,result.stdout[-6000:]
    m=json.loads((output/'calibration/calibration.json').read_text())
    assert m['budget']['calibration_runs']==2 and m['budget']['predicted_candidates']==3
    p=json.loads((output/'plan.json').read_text())
    assert len(p['search']['ranked'])==3
    e=json.loads((output/'selected/execution.json').read_text())
    assert e['component_prediction_check']['state_check']['status']=='PASS_SAME_ANCHOR_STATE'
