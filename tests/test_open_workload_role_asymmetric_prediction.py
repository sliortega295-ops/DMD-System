"""Asymmetric native FIFO prediction: budgets/precedence, then real Gloo checks.

Synthetic observations test search mechanics, not A100/H100 performance.
"""
from dataclasses import replace
import pytest

from distillschedule.open_workload.role_program import RoleTask, PoolLayout, AllocatedWorld, enumerate_layouts
from distillschedule.open_workload.role_prediction import (
    admission, graph_for, projection, generator_order, select_anchor_targets,
)
from distillschedule.search.plan_ir import ancestors
from distillschedule.workloads.best_known.registry import fingerprint


@pytest.mark.parametrize('world,cut,n',[(3,1,5),(3,2,5),(4,1,7),(4,3,7),(5,2,7)])
def test_asymmetric_graph_obeys_all_receivers_not_only_producers(world,cut,n):
    task=RoleTask(n,(2,4),score_updates=1)
    hw=AllocatedWorld((10**7,)*world)
    for window in (1,2,3,4):
        layout=PoolLayout(tuple(range(cut)),tuple(range(cut,world)),window)
        graph,specs=graph_for(task,layout,hw)
        before=ancestors(graph)
        assert len(graph.operations)==5*n+2
        for score in layout.score:
            indices=[i for i in range(n) if layout.owners(i)[1]==score]
            for left,right in zip(indices,indices[1:]):
                assert f'r{left}' in before[f'a{right}']
        for producer in layout.generator:
            order=generator_order([i for i in range(n) if layout.owners(i)[0]==producer],window)
            assert all(a in before[b] for a,b in zip(order,order[1:]))
        assert all(f'b{i}' in before['commit_g'] for i in range(n))
        assert 'commit_g' in before['updates_s']
        for i in range(n):
            g,s=layout.owners(i)
            assert specs[f'a{i}'][0]==f'link:{g}:{s}'
            assert specs[f'r{i}'][0]==f'link:{s}:{g}'


def test_receiver_dependency_crosses_different_producers():
    task=RoleTask(5,(2,4));hw=AllocatedWorld((10**7,)*3)
    graph,_=graph_for(task,PoolLayout((0,1),(2,),2),hw)
    before=ancestors(graph)
    assert 'r0' in before['a1']  # Score 2 must not consume G1's input before G0's.
    assert 'r1' in before['a2']


@pytest.mark.parametrize('max_candidates,native_runs,samples',[(4,3,1),(8,8,1),(8,10,2),(1,3,1)])
def test_bounded_anchors_cover_splits_and_leave_confirmation(max_candidates,native_runs,samples):
    task=RoleTask(24,(2,4));hw=AllocatedWorld((10**7,)*16,(0,)*8+(1,)*8)
    inventory=enumerate_layouts(task,hw,generator_bytes=1000,score_bytes=1000,max_candidates=1024)
    selected,omitted=select_anchor_targets(task,hw,inventory['candidates'],
        max_candidates=max_candidates,samples=samples,max_native_runs=native_runs)
    groups={fingerprint(projection(c)) for c in selected}
    assert len(selected)<=max_candidates
    assert len(groups)*(samples+1)+1<=native_runs
    assert omitted
    assert select_anchor_targets(task,hw,inventory['candidates'],max_candidates=max_candidates,
        samples=samples,max_native_runs=native_runs)[0]==selected
    if max_candidates>1:
        assert any(c.window>1 for c in selected)
    if native_runs>=8:
        assert len({len(c.generator) for c in selected})>1


def test_equivalent_windows_are_not_counted_as_new_unseen_schedules():
    task=RoleTask(4,(2,4));hw=AllocatedWorld((10**7,)*4)
    layouts=[PoolLayout((0,1),(2,3),w) for w in range(1,5)]
    selected,omitted=select_anchor_targets(task,hw,layouts,max_candidates=4,samples=1,max_native_runs=3)
    assert [c.window for c in selected]==[1,2]
    assert {r['reason'] for r in omitted}=={'EQUIVALENT_NATIVE_FIFO'}


def test_inactive_ranks_and_unfunded_anchor_fail_before_execution():
    task=RoleTask(2,(2,4));hw=AllocatedWorld((10**7,)*4)
    with pytest.raises(ValueError,match='active replicated resident'):
        admission(task,PoolLayout((0,),(1,2,3),1),hw)
    with pytest.raises(ValueError,match='confirmation exceed budget'):
        select_anchor_targets(RoleTask(4,(2,4)),hw,[PoolLayout((0,1),(2,3),1)],
                              max_candidates=4,samples=2,max_native_runs=3)


def _native_worker(rank,world,rdzv,output,n):
    from datetime import timedelta
    from pathlib import Path
    import json
    import torch
    import torch.distributed as dist
    from distillschedule.open_workload.role_transport import RoleFunctions,RolePoolExecutor
    from distillschedule.open_workload.role_planning import calibrate,search,run_selected
    from distillschedule.open_workload.role_prediction import validate_execution
    from test_open_workload_role_transport import make_model,make_opt,sample,generate,guidance,score_train
    from distillschedule.workloads.best_known.registry import write_new_json
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+rdzv,rank=rank,world_size=world,
                            timeout=timedelta(seconds=90))
    try:
        task=RoleTask(n,(2,4),score_updates=1)
        hw=AllocatedWorld((2**26,)*world)
        fn=RoleFunctions(make_model,make_opt,sample,generate,guidance,score_train,data_identity='b'*64)
        root=Path(output)
        record=calibrate(task,hw,fn,root/'cal',generator_bytes=2000,score_bytes=2000,
            max_candidates=4,samples=1,max_native_runs=5,cost_mode='components')
        assert record['budget']['calibration_runs']==4
        assert record['budget']['predicted_candidates']==4
        raw=[json.loads(Path(a['native_path']).read_text()) for a in record['anchors']]
        assert len(raw)==2 and all(r['layout']['window']==1 for r in raw)
        assert any(len(r['layout']['generator'])!=len(r['layout']['score']) for r in raw)
        if rank==0:
            plan=search(root/'cal/calibration.json',max_states=30000,beam_width=32,timeout_seconds=25)
            assert len(plan['search']['ranked'])==4
            assert len({len(c['implementation']['generator']) for c in plan['search']['ranked']})==2
            write_new_json(root/'plan.json',plan)
        dist.barrier()
        result,model,optimizer=run_selected(task,hw,fn,root/'plan.json',root/'selected')
        assert result['component_prediction_check']['state_check']['status']=='PASS_SAME_ANCHOR_STATE'
        layout=PoolLayout.from_dict(result['layout'])
        expected=_oracle(task)
        role='generator' if rank in layout.generator else 'score'
        _assert_state(model,optimizer,*expected[role])
        # A second explicitly held-out asymmetric window tests native lowering,
        # not a fabricated selected winner. It is outside the calibration budget.
        asymmetric=next(PoolLayout.from_dict(r['layout']) for r in raw
                        if len(r['layout']['generator'])!=len(r['layout']['score']))
        runtime=RolePoolExecutor(task,replace(asymmetric,window=2),hw,fn,device='cpu')
        try:
            heldout,m,o=runtime.run()
            assert validate_execution(task,runtime.layout,heldout)['status']=='PASS_NATIVE_FIFO_PROJECTION'
            _assert_state(m,o,*expected[runtime.role])
        finally:runtime.close()
        if rank==0:
            write_new_json(root/'assertions.json',{'status':'PASS_CPU_NATIVE_ASYMMETRIC',
                'reference':'independent_global_microbatch_params_and_adam',
                'extra_heldout_test_cycles':1,'gpu_execution':'NOT_RUN'})
    finally:dist.destroy_process_group()


def _oracle(task):
    import torch
    from test_open_workload_role_transport import make_model,make_opt,sample,generate,guidance,score_train
    g=make_model('generator','cpu');s=make_model('score','cpu')
    go=make_opt('generator',g);so=make_opt('score',s);latents=[]
    for i in range(task.microbatches):
        latent=generate(g,sample(i,'generator','cpu'))
        remote=latent.detach().clone().requires_grad_(True)
        vjp=torch.autograd.grad(guidance(s,remote,sample(i,'score','cpu')),remote)[0]
        latent.backward(vjp/task.microbatches)
        latents.append(latent.detach())
    go.step();go.zero_grad(set_to_none=True)
    for i,latent in enumerate(latents):
        (score_train(s,latent,sample(i,'score','cpu'))/task.microbatches).backward()
    so.step();so.zero_grad(set_to_none=True)
    return {'generator':(g,go),'score':(s,so)}


def _assert_state(model,optimizer,reference,reference_optimizer):
    import torch
    for key,value in model.state_dict().items():
        assert torch.allclose(value,reference.state_dict()[key],rtol=1e-6,atol=1e-7),key
    for actual,expected in zip(model.parameters(),reference.parameters()):
        a,b=optimizer.state[actual],reference_optimizer.state[expected]
        assert a.keys()==b.keys()
        for key,value in a.items():
            assert torch.allclose(value,b[key],rtol=2e-6,atol=2e-7),key


@pytest.mark.parametrize('world,n',[(3,5),(4,6)])
def test_asymmetric_component_search_selected_and_heldout_native_execution(tmp_path,world,n):
    import json
    import torch.multiprocessing as mp
    mp.spawn(_native_worker,args=(world,str(tmp_path/'rdzv'),str(tmp_path),n),nprocs=world,join=True)
    result=json.loads((tmp_path/'assertions.json').read_text())
    assert result['gpu_execution']=='NOT_RUN'
