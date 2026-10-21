"""Real trainable Score shards, supplied DMD-like gradients, and repeated Adam.

No full video model or A100/H100 performance is inferred from these fixtures.
"""
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
import json
import pytest
import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp
from distillschedule.open_workload.role_program import RoleTask, PoolLayout, AllocatedWorld, enumerate_layouts
from distillschedule.open_workload.role_transport import RoleFunctions, RolePoolExecutor


class MixedScore(nn.Module):
    def __init__(self):
        super().__init__()
        self.base=nn.Linear(4,4).to(torch.bfloat16).requires_grad_(False)
        self.adapter=nn.Linear(4,4)
    def forward(self,x):
        return self.base(x.to(torch.bfloat16)).float()+self.adapter(x.float())


def make_model(role,device):
    with torch.random.fork_rng():
        torch.manual_seed(107 if role=='generator' else 223)
        m=nn.Linear(3,4) if role=='generator' else nn.Sequential(nn.Linear(4,8),nn.Tanh(),nn.Linear(8,4))
    return m.to(device)


def make_mixed(role,device):
    if role=='generator':return make_model(role,device)
    with torch.random.fork_rng():
        torch.manual_seed(223);m=MixedScore()
    return m.to(device)


def warm_opt(role,model):
    opt=torch.optim.Adam((p for p in model.parameters() if p.requires_grad),lr=.003)
    for p in model.parameters():
        if p.requires_grad:p.grad=torch.full_like(p,.01 if role=='generator' else .02)
    opt.step();opt.zero_grad(set_to_none=True)
    return opt


def data(cycle,index,role,device):
    width=3 if role=='generator' else 4
    return torch.arange(2*width,device=device,dtype=torch.float32).view(2,width)/10+index*.03+cycle*.02


def first_data(index,role,device):return data(0,index,role,device)
def generate(model,x):return model(x)
def constructed(model,latent,target):
    assert not torch.is_grad_enabled()
    return model(latent)-target

def score_train(model,latent,target):return (model(latent)-target).square().mean()
def update_input(ctx):return ctx.retained_latent+.01*ctx.update,ctx.conditions+.02*ctx.update


def reference(task,cycles,mixed):
    factory=make_mixed if mixed else make_model
    g,s=factory('generator','cpu'),factory('score','cpu')
    go,so=warm_opt('generator',g),warm_opt('score',s)
    for cycle in range(cycles):
        xs=[]
        for i in range(task.microbatches):
            value=g(data(cycle,i,'generator','cpu'))
            with torch.no_grad():vjp=s(value.detach())-data(cycle,i,'score','cpu')
            value.backward(vjp/task.microbatches);xs.append(value.detach())
        go.step();go.zero_grad(set_to_none=True)
        for u in range(task.score_updates):
            for i,x in enumerate(xs):
                loss=(s(x+.01*u)-(data(cycle,i,'score','cpu')+.02*u)).square().mean()/task.microbatches
                loss.backward()
            so.step();so.zero_grad(set_to_none=True)
    return {'generator':(g,go),'score':(s,so)}


def full(value):
    from torch.distributed.tensor import DTensor
    if isinstance(value,torch.Tensor):return (value.full_tensor() if isinstance(value,DTensor) else value).detach().cpu()
    if isinstance(value,dict):return {k:full(v) for k,v in value.items()}
    if isinstance(value,list):return [full(v) for v in value]
    return value


def worker(rank,rdzv,out,g_shard,mixed,updates):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+rdzv,rank=rank,world_size=4,
                            timeout=timedelta(seconds=70))
    runner=None
    try:
        task=RoleTask(4,(2,4),score_updates=updates)
        layout=PoolLayout((0,2),(1,3),2,'resident',
                          'fsdp2_keep_forward' if g_shard else 'replicated','fsdp2_train')
        fn=RoleFunctions(make_mixed if mixed else make_model,warm_opt,first_data,generate,None,
                         score_train,score_input_gradient=constructed,sample_at=data,
                         score_update_input=update_input)
        runner=RolePoolExecutor(task,layout,AllocatedWorld((2**26,)*4,(0,0,1,1)),fn,device='cpu')
        ids=None
        for cycle in range(2):
            receipt,model,opt=runner.run()
            if ids is None:ids=(id(model),id(opt))
            assert ids==(id(model),id(opt))
            expected,expected_opt=reference(task,cycle+1,mixed)[runner.role]
            torch.testing.assert_close(full(model.state_dict()),full(expected.state_dict()),rtol=3e-5,atol=3e-6)
            torch.testing.assert_close(full(opt.state_dict()),full(expected_opt.state_dict()),rtol=5e-5,atol=5e-6)
            assert receipt['final_versions']['score']==(cycle+1)*updates
            assert receipt['final_versions']['generator']==cycle+1
            if runner.role=='score':
                from torch.distributed.tensor import DTensor
                assert any(isinstance(p,DTensor) for p in model.parameters())
                assert receipt['ranks'][rank]['sharding']['role']=='score_trainable_sharded'
                if mixed:assert {p.dtype for p in model.parameters()}=={torch.bfloat16,torch.float32}
        Path(out,f'{rank}.json').write_text(json.dumps({'ok':True}))
    finally:
        if runner is not None:runner.close()
        dist.destroy_process_group()


@pytest.mark.parametrize('g_shard,mixed,updates',[(False,False,1),(True,False,3),(False,True,3)])
def test_score_shards_keep_warm_adam_and_supplied_gradient(tmp_path,g_shard,mixed,updates):
    mp.spawn(worker,args=(str(tmp_path/'rdzv'),str(tmp_path),g_shard,mixed,updates),nprocs=4,join=True)
    assert all(json.loads((tmp_path/f'{r}.json').read_text())['ok'] for r in range(4))


def test_trainable_score_candidates_require_collective_safe_updates():
    task=RoleTask(4,(2,4),score_updates=3)
    hw=AllocatedWorld((2**26,)*4,(0,0,1,1))
    candidates=enumerate_layouts(task,hw,generator_bytes=3000,score_bytes=3000,
                                 max_candidates=8,allow_trainable_score_fsdp=True)['candidates']
    assert any(x.score_parallelism=='fsdp2_train' for x in candidates)
    with pytest.raises(ValueError,match='balanced paired calls'):
        PoolLayout((0,),(1,2,3),2,'resident','replicated','fsdp2_train').validate(task,hw)
    assert all(x.score_parallelism!='fsdp2_train' for x in enumerate_layouts(
        task,hw,generator_bytes=3000,score_bytes=3000,max_candidates=8)['candidates'])


def planned_worker(rank,rdzv,folder):
    from distillschedule.open_workload.role_planning import calibrate,search,run_selected
    from distillschedule.workloads.best_known.registry import write_new_json
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+rdzv,rank=rank,world_size=4,
                            timeout=timedelta(seconds=70))
    try:
        task=RoleTask(4,(2,4),score_updates=3)
        hw=AllocatedWorld((2**26,)*4,(0,0,1,1))
        fn=RoleFunctions(make_model,warm_opt,first_data,generate,None,score_train,
                         data_identity='a'*64,score_input_gradient=constructed,
                         sample_at=data,score_update_input=update_input)
        root=Path(folder)
        measured=calibrate(task,hw,fn,root/'cal',generator_bytes=3000,score_bytes=3000,
                           max_candidates=3,samples=1,max_native_runs=4)
        assert any(c['layout']['score_parallelism']=='fsdp2_train' for c in measured['candidates'])
        if rank==0:
            p=search(root/'cal/calibration.json')
            assert any(c['implementation']['score_parallelism']=='fsdp2_train' for c in p['search']['ranked'])
            write_new_json(root/'plan.json',p)
        dist.barrier()
        receipt,model,opt=run_selected(task,hw,fn,root/'plan.json',root/'train',cycles=2)
        role='generator' if rank in receipt['layout']['generator'] else 'score'
        exp,exp_opt=reference(task,2,False)[role]
        torch.testing.assert_close(full(model.state_dict()),full(exp.state_dict()),rtol=3e-5,atol=3e-6)
        torch.testing.assert_close(full(opt.state_dict()),full(exp_opt.state_dict()),rtol=5e-5,atol=5e-6)
        assert receipt['final_versions']['score']==6
        Path(folder,f'planned-{rank}.json').write_text(json.dumps({'ok':True}))
    finally:dist.destroy_process_group()


def test_automatic_trainable_score_matching_calibration_and_selected_execution(tmp_path):
    mp.spawn(planned_worker,args=(str(tmp_path/'rdzv'),str(tmp_path)),nprocs=4,join=True)
    assert all(json.loads((tmp_path/f'planned-{r}.json').read_text())['ok'] for r in range(4))
