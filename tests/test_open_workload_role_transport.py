"""Real multiprocess Gloo role pools are compared to an independent global oracle."""
from datetime import timedelta
from pathlib import Path
import json
import pytest
import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp
from distillschedule.open_workload.role_program import RoleTask,AllocatedWorld,PoolLayout
from distillschedule.open_workload.role_transport import RoleFunctions,RolePoolExecutor


def make_model(role,device):
    with torch.random.fork_rng():
        torch.manual_seed(31 if role=='generator' else 53)
        model=nn.Linear(3,4) if role=='generator' else nn.Linear(4,2)
    return model.to(device)

def make_opt(role,model):
    return torch.optim.Adam(model.parameters(),lr=.005)

def sample(i,role,device):
    if role=='generator':
        return (torch.arange(6,dtype=torch.float32,device=device).view(2,3).add(i*.12))
    return torch.arange(4,dtype=torch.float32,device=device).view(2,2).add(i*.17)

def generate(model,x):return model(x)

def guidance(score,x,target):return (score(x)-target).square().mean()

def score_train(score,x,target):return (score(x.detach())-target).square().mean()


def independent_reference(task):
    device=torch.device('cpu');g=make_model('generator',device);s=make_model('score',device)
    go=make_opt('generator',g);so=make_opt('score',s)
    retained=[]
    for i in range(task.microbatches):
        x=generate(g,sample(i,'generator',device))
        remote=x.detach().clone().requires_grad_()
        loss=guidance(s,remote,sample(i,'score',device))
        dx=torch.autograd.grad(loss,remote)[0]
        x.backward(dx/task.microbatches)
        retained.append(x.detach())
    go.step();go.zero_grad(set_to_none=True)
    if task.score_updates:
        for i,x in enumerate(retained):
            (score_train(s,x,sample(i,'score',device))/task.microbatches).backward()
        so.step();so.zero_grad(set_to_none=True)
    return {r:{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            for r,model in (('generator',g),('score',s))}


def n_local(task,layout,rank):
    return sum(layout.owners(i)[0]==rank for i in range(task.microbatches))


def worker(rank,world,layout,task,rendezvous,out):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=world,
                            timeout=timedelta(seconds=50))
    try:
        bindings=RoleFunctions(make_model,make_opt,sample,generate,guidance,
                               score_train if task.score_updates else None)
        hardware=AllocatedWorld((2**25,)*world)
        runtime=RolePoolExecutor(task,layout,hardware,bindings,device='cpu')
        result,module,opt=runtime.run()
        ref=independent_reference(task)
        for name,v in module.state_dict().items():
            assert torch.allclose(v.detach().cpu(),ref[runtime.role][name],rtol=1e-6,atol=1e-7),(
                name,runtime.role,layout)
        assert result['status']=='PASS_ROLE_POOL_EXECUTION'
        assert result['claimable_speedup'] is False
        assert result['ranks'][rank]['generator_version']==1
        if runtime.role=='generator' and n_local(task,layout,rank)>=layout.window:
            assert result['ranks'][rank]['max_inflight']==layout.window
        Path(out,f'rank{rank}.json').write_text(json.dumps({'status':result['status'],
            'pool':result['layout'],'max_inflight':result['ranks'][rank]['max_inflight'],
            'gradients':sum(x['input_gradients'] for x in result['ranks'])}))
    finally:dist.destroy_process_group()


@pytest.mark.parametrize('world,cut,n,window,score_updates',[
    (2,1,2,1,0),(2,1,4,2,0),(2,1,4,2,1),
    (3,1,4,2,1),(3,2,5,2,1),(4,2,5,2,0),
])
def test_native_pool_transport_with_independent_global_baseline(tmp_path,world,cut,n,window,score_updates):
    layout=PoolLayout(tuple(range(cut)),tuple(range(cut,world)),window)
    task=RoleTask(n,(2,4),score_updates=score_updates)
    mp.spawn(worker,args=(world,layout,task,str(tmp_path/'rdzv'),str(tmp_path)),nprocs=world,join=True)
    for rank in range(world):
        receipt=json.loads((tmp_path/f'rank{rank}.json').read_text())
        assert receipt['gradients']==n and receipt['status']=='PASS_ROLE_POOL_EXECUTION'


def test_refuse_update_without_train_objective():
    task=RoleTask(2,(2,4),score_updates=1)
    with pytest.raises(ValueError):RoleFunctions(make_model,make_opt,sample,generate,guidance).validate(task)


def test_bad_grad_and_output_shape_rejected():
    from distillschedule.open_workload.role_transport import _validate_tensor
    task=RoleTask(2,(2,4))
    with pytest.raises(ValueError):_validate_tensor(torch.ones(2,5),task,'bad')
    with pytest.raises(ValueError):_validate_tensor(torch.full((2,4),float('nan')),task,'nan')


@pytest.mark.parametrize('world,cut,updates',[(2,1,0),(2,1,1),(3,2,1)])
def test_cpu_saved_tensor_offload_in_actual_role_pool(tmp_path,world,cut,updates):
    layout=PoolLayout(tuple(range(cut)),tuple(range(cut,world)),2,'cpu_offload_prefetch')
    task=RoleTask(4,(2,4),score_updates=updates)
    mp.spawn(offload_worker,args=(world,layout,task,str(tmp_path/'rdzv'),str(tmp_path)),
             nprocs=world,join=True)
    for rank in layout.generator:
        row=json.loads((tmp_path/f'offload{rank}.json').read_text())
        assert row['packed_tensors']>0 and row['prefetch_tensors']>0
        assert row['unpacked_tensors']>0 and row['peak_host_bytes']>0
        assert row['model_matches_reference']


def offload_worker(rank,world,layout,task,rdzv,out):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+rdzv,rank=rank,world_size=world,
                            timeout=timedelta(seconds=50))
    try:
        hw=AllocatedWorld((2**25,)*world,host_capacities=(2**25,)*world)
        bindings=RoleFunctions(make_model,make_opt,sample,generate,guidance,
                               score_train if task.score_updates else None)
        receipt,module,_=RolePoolExecutor(task,layout,hw,bindings,device='cpu').run()
        baseline=independent_reference(task)
        role='generator' if rank in layout.generator else 'score'
        matches=all(torch.allclose(v.detach(),baseline[role][k],atol=1e-7,rtol=1e-6)
                    for k,v in module.state_dict().items())
        assert matches and receipt['status']=='PASS_ROLE_POOL_EXECUTION'
        Path(out,f'offload{rank}.json').write_text(json.dumps({
            **receipt['ranks'][rank]['offload'],'model_matches_reference':matches}))
    finally:dist.destroy_process_group()


def test_noncontiguous_node_local_pool_preserves_real_vjp_and_adam(tmp_path):
    """A node-local partition must lower to real P2P instead of only planning."""
    task=RoleTask(6,(2,4),score_updates=1)
    hw=AllocatedWorld((2**25,)*4,(0,0,1,1))
    from distillschedule.open_workload.role_program import enumerate_layouts
    candidates=enumerate_layouts(task,hw,generator_bytes=1000,score_bytes=1000,max_candidates=80)
    layout=next(p for p in candidates['candidates'] if p.generator==(0,2)
                and p.score==(1,3) and p.window==2)
    mp.spawn(worker,args=(4,layout,task,str(tmp_path/'rdzv'),str(tmp_path)),nprocs=4,join=True)
    for rank in range(4):
        receipt=json.loads((tmp_path/f'rank{rank}.json').read_text())
        assert receipt['gradients']==task.microbatches
        assert receipt['pool']==layout.to_dict()


def test_node_local_pool_with_activation_offload_matches_reference(tmp_path):
    """Noncontiguous ranks must preserve saved tensors through VJP return."""
    task=RoleTask(6,(2,4),score_updates=1)
    hw=AllocatedWorld((2**25,)*4,(0,0,1,1),(2**25,)*4)
    from distillschedule.open_workload.role_program import enumerate_layouts
    layouts=enumerate_layouts(task,hw,generator_bytes=1000,score_bytes=1000,
                              max_candidates=200,allow_offload=True)['candidates']
    layout=next(p for p in layouts if p.generator==(0,2) and p.score==(1,3)
                and p.window==2 and p.activation_policy=='cpu_offload_prefetch')
    mp.spawn(offload_worker,args=(4,layout,task,str(tmp_path/'rdzv'),str(tmp_path)),
             nprocs=4,join=True)
    for rank in layout.generator:
        row=json.loads((tmp_path/f'offload{rank}.json').read_text())
        assert row['model_matches_reference']
        assert row['packed_tensors']>0 and row['prefetch_tensors']>0
        assert row['unpacked_tensors']>0 and row['peak_host_bytes']>0
