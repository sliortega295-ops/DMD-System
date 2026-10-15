"""A planned producer order must be executed, not replaced by a FIFO preset."""
from dataclasses import replace
import pytest
from distillschedule.open_workload.role_program import RoleTask,PoolLayout,AllocatedWorld
from distillschedule.open_workload.role_prediction import graph_for,generator_order
from distillschedule.open_workload.role_dispatch import from_order,validate_dispatch
from distillschedule.search.service import select


def test_frontier_can_flush_before_fixed_window_when_memory_requires():
    from test_open_workload_role_prediction import candidate
    task=RoleTask(4,(2,4));layout=PoolLayout((0,),(1,),2)
    c=candidate(2,250)
    fixed=select([c],timeout_seconds=10)
    assert not fixed['ranked']
    graph,_=graph_for(task,layout,AllocatedWorld((250,)*2),planned_producer=True)
    planned=select([replace(c,graph=graph)],timeout_seconds=10)
    assert planned['ranked']
    dispatch=from_order(task,layout,planned['ranked'][0]['order'])
    assert dispatch['0']!=generator_order(list(range(4)),2)
    assert dispatch['0']==generator_order(list(range(4)),1)


@pytest.mark.parametrize('actions',[
    ['b0','g0','g1','b1'],  # backward without the original graph
    ['g1','g0','b1','b0'],  # forward RNG/data order changed
    ['g0','g1','b1','b0'],  # backward accumulation reordered
    ['g0','b0','g1','commit_g'],  # commit is not a producer action
    ['g0','b0'],  # incomplete training
])
def test_unsafe_dispatch_is_rejected(actions):
    with pytest.raises(ValueError):
        validate_dispatch(RoleTask(2,(2,4)),PoolLayout((0,),(1,),2),{'0':actions})


def test_window_and_sharding_admission_are_not_silently_ignored():
    task=RoleTask(2,(2,4));layout=PoolLayout((0,),(1,),1)
    with pytest.raises(ValueError,match='window exceeded'):
        validate_dispatch(task,layout,{'0':['g0','g1','b0','b1']})
    with pytest.raises(ValueError,match='replicated resident'):
        validate_dispatch(task,replace(layout,activation_policy='cpu_offload_prefetch'),{'0':[]})


def _worker(rank,rdzv,output):
    from datetime import timedelta
    from pathlib import Path
    import json
    import torch
    import torch.distributed as dist
    from distillschedule.open_workload.role_transport import RolePoolExecutor,RoleFunctions
    from distillschedule.open_workload.role_prediction import validate_execution
    from test_open_workload_role_transport import make_model,make_opt,sample,generate,guidance,score_train
    from test_open_workload_role_asymmetric_prediction import _oracle,_assert_state
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+rdzv,rank=rank,world_size=2,
                            timeout=timedelta(seconds=70))
    runtime=None
    try:
        task=RoleTask(4,(2,4),score_updates=1)
        layout=PoolLayout((0,),(1,),3)
        # This sequence is intentionally different from EVERY fixed FIFO
        # window. It is a lowering test, not a claimed measured-cost winner.
        actions=['g0','b0','g1','g2','b1','g3','b2','b3']
        assert all(actions!=generator_order(list(range(4)),w) for w in range(1,5))
        dispatch={'0':actions}
        fn=RoleFunctions(make_model,make_opt,sample,generate,guidance,score_train)
        runtime=RolePoolExecutor(task,layout,AllocatedWorld((2**26,)*2),fn,
                                device='cpu',producer_dispatch=dispatch)
        receipt,module,optimizer=runtime.run()
        assert validate_execution(task,layout,receipt,dispatch=dispatch)['status']=='PASS_NATIVE_FIFO_PROJECTION'
        if rank==0:assert receipt['ranks'][0]['producer_dispatch']==actions
        _assert_state(module,optimizer,*_oracle(task)[runtime.role])
        Path(output,f'{rank}.json').write_text(json.dumps({'ok':True,'gpu':'NOT_RUN'}))
    finally:
        if runtime is not None:runtime.close()
        dist.destroy_process_group()


def test_real_runtime_follows_nonpreset_order_and_preserves_both_adams(tmp_path):
    import torch.multiprocessing as mp
    mp.spawn(_worker,args=(str(tmp_path/'rdzv'),str(tmp_path)),nprocs=2,join=True)


def _never_construct(role,device):
    raise AssertionError('invalid producer dispatch reached model construction')


def _rejection_worker(rank,rdzv,mode):
    from datetime import timedelta
    import torch
    import torch.distributed as dist
    from distillschedule.open_workload.role_transport import RoleFunctions,RolePoolExecutor
    from test_open_workload_role_transport import make_opt,sample,generate,guidance
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+rdzv,rank=rank,world_size=2,
                            timeout=timedelta(seconds=30))
    runner=None
    try:
        task=RoleTask(2,(2,4));layout=PoolLayout((0,),(1,),2)
        dispatch={'0':['g0','b0','g1','b1']}
        if mode=='different-valid-orders' and rank:
            dispatch={'0':['g0','g1','b0','b1']}
        if mode=='unsafe-all-ranks':dispatch={'0':['b0','g0','g1','b1']}
        fn=RoleFunctions(_never_construct,make_opt,sample,generate,guidance)
        runner=RolePoolExecutor(task,layout,AllocatedWorld((2**26,)*2),fn,
                               device='cpu',producer_dispatch=dispatch)
        if mode=='one-rank-mutation' and rank:
            dispatch['0'][:]=['g0','g1','b0','b1']
        pattern={'different-valid-orders':'rank cycle/version mismatch',
                 'unsafe-all-ranks':'role planned producer admission',
                 'one-rank-mutation':'role cycle admission'}[mode]
        with pytest.raises(ValueError,match=pattern):runner.run()
        assert runner.module is None and runner.optimizer is None
        dist.barrier()  # every rank rejected rather than stranding its peer
    finally:
        if runner is not None:runner.close()
        dist.destroy_process_group()


@pytest.mark.parametrize('mode',['different-valid-orders','unsafe-all-ranks','one-rank-mutation'])
def test_dispatch_rejects_collectively_before_model_work(tmp_path,mode):
    import torch.multiprocessing as mp
    mp.spawn(_rejection_worker,args=(str(tmp_path/'rdzv'),mode),nprocs=2,join=True)


def test_projection_rejects_missing_rank_and_ignored_dispatch():
    from distillschedule.open_workload.role_prediction import validate_execution
    task=RoleTask(1,(2,4));layout=PoolLayout((0,),(1,),1)
    dispatch={'0':['g0','b0']}
    rows=[{'rank':0,'role':'generator','components':{'enabled':False},
           'producer_dispatch':['g0','b0'],
           'events':[{'action':'generator_forward_send','microbatch':0},
                     {'action':'generator_vjp_backward','microbatch':0}]},
          {'rank':1,'role':'score','components':{'enabled':False},
           'events':[{'action':'score_receive_vjp_send','microbatch':0}]}]
    receipt={'task_sha256':task.digest,'layout':layout.to_dict(),'ranks':rows}
    assert validate_execution(task,layout,receipt,dispatch=dispatch)['status']=='PASS_NATIVE_FIFO_PROJECTION'
    receipt['ranks']=rows[:1]
    with pytest.raises(ValueError,match='missing/duplicate ranks'):
        validate_execution(task,layout,receipt,dispatch=dispatch)
    receipt['ranks']=rows
    rows[0]['producer_dispatch']=None
    with pytest.raises(ValueError,match='ignored selected producer'):
        validate_execution(task,layout,receipt,dispatch=dispatch)


def test_variable_activation_sizes_need_a_nonpreset_planned_order():
    from test_open_workload_role_prediction import candidate
    task=RoleTask(4,(2,4));hw=AllocatedWorld((290,)*2)
    sizes=[50,100,150,50]
    # Explicitly synthetic component costs: this is a decision/capacity test,
    # not a claimed GPU measurement or measured training speedup.
    durations={'g0':21,'g1':36,'g2':33,'g3':4,
               's0':3,'s1':18,'s2':10,'s3':30,
               'b0':3,'b1':35,'b2':33,'b3':15}
    def build(window,planned=False):
        value=candidate(window,290)
        records={r.key:r for r in value.costs.records}
        for operation,cost in durations.items():
            key=value.rules[operation].cost_key
            records[key]=replace(records[key],samples_ns=(cost,))
        value=replace(value,costs=replace(value.costs,records=tuple(records.values())),
                      lifetimes=tuple(replace(l,bytes_per_rank=sizes[i])
                                      for i,l in enumerate(value.lifetimes)))
        if planned:
            graph,_=graph_for(task,PoolLayout((0,),(1,),window),hw,planned_producer=True)
            value=replace(value,graph=graph,id='planned')
        return value
    fixed=select([build(w) for w in range(1,5)],timeout_seconds=10)
    planned=select([build(4,True)],timeout_seconds=10)
    assert fixed['ranked'] and planned['ranked']
    result=planned['ranked'][0]
    actions=from_order(task,PoolLayout((0,),(1,),4),result['order'])['0']
    assert all(actions!=generator_order(list(range(4)),w) for w in range(1,5))
    assert result['predicted_makespan_ns']<fixed['ranked'][0]['predicted_makespan_ns']
