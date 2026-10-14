"""Stateful multi-cycle P2P training with an independent global-batch oracle."""
from datetime import timedelta
from pathlib import Path
import json
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from distillschedule.open_workload.role_program import RoleTask,PoolLayout,AllocatedWorld
from distillschedule.open_workload.role_transport import RolePoolExecutor,RoleFunctions
from test_open_workload_role_transport import make_model,make_opt,sample,generate,guidance,score_train


constructed={'model':0,'optimizer':0}

def counted_model(role,device):
    constructed['model']+=1
    model=make_model(role,device)
    return model

def counted_optimizer(role,model):
    constructed['optimizer']+=1
    return make_opt(role,model)

def cycle_data(cycle,index,role,device):
    return sample(index,role,device)+(cycle*.07 if role=='generator' else cycle*.03)

def analytic_input_gradient(model,latent,condition):
    # Independent closed-form gradient for this TEST objective. Production
    # bindings invoke the original algorithm, not this synthetic objective.
    assert not torch.is_grad_enabled()
    value=model(latent)-condition
    return (2*value/value.numel())@model.weight

def tree_equal(a,b):
    if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,rtol=2e-6,atol=1e-7)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:tree_equal(a[key],b[key])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):tree_equal(x,y)
    else:assert a==b


def reference(task,cycles):
    g=make_model('generator','cpu');s=make_model('score','cpu')
    go=make_opt('generator',g);so=make_opt('score',s) if task.score_updates else None
    for cycle in range(cycles):
        values=[]
        for i in range(task.microbatches):
            value=generate(g,cycle_data(cycle,i,'generator','cpu'))
            remote=value.detach().clone().requires_grad_()
            loss=guidance(s,remote,cycle_data(cycle,i,'score','cpu'))
            value.backward(torch.autograd.grad(loss,remote)[0]/task.microbatches)
            values.append(value.detach())
        go.step();go.zero_grad(set_to_none=True)
        if so is not None:
            for i,value in enumerate(values):
                (score_train(s,value,cycle_data(cycle,i,'score','cpu'))/task.microbatches).backward()
            so.step();so.zero_grad(set_to_none=True)
    return {'generator':(g,go),'score':(s,so)}


def worker(rank,world,task,layout,constructed_gradient,rdzv,out):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+rdzv,rank=rank,world_size=world,
                            timeout=timedelta(seconds=80))
    runtime=None
    try:
        constructed.update(model=0,optimizer=0)
        f=RoleFunctions(counted_model,counted_optimizer,sample,generate,
             None if constructed_gradient else guidance,
             score_train if task.score_updates else None,data_identity='c'*64,
             score_input_gradient=analytic_input_gradient if constructed_gradient else None,
             sample_at=cycle_data)
        hw=AllocatedWorld((2**25,)*world,host_capacities=(2**25,)*world)
        runtime=RolePoolExecutor(task,layout,hw,f,device='cpu')
        ids=None
        for cycle in range(3):
            receipt,model,opt=runtime.run()
            if ids is None:ids=(id(model),id(opt))
            assert (id(model),id(opt))==ids
            assert receipt['initial_versions']['generator']==cycle
            assert receipt['final_versions']['generator']==cycle+1
            assert receipt['final_versions']['score']==(cycle+1)*task.score_updates
            assert all(row['cycle']==cycle for row in receipt['ranks'])
            g,o=reference(task,cycle+1)[runtime.role]
            tree_equal(model.state_dict(),g.state_dict())
            if opt is not None:tree_equal(opt.state_dict(),o.state_dict())
        assert constructed['model']==1
        assert constructed['optimizer']==int(runtime.role=='generator' or task.score_updates>0)
        runtime.close();runtime.close()
        with pytest.raises(ValueError,match='closed'):runtime.run()
        Path(out,f'rank{rank}.json').write_text(json.dumps({'cycle':receipt['cycle'],'passes':3}))
    finally:
        if runtime is not None:runtime.close()
        dist.destroy_process_group()


@pytest.mark.parametrize('world,updates,provided,offload',[(2,0,False,False),(2,0,True,True),(3,1,True,False)])
def test_continuous_versions_data_and_adam_without_reinitialization(tmp_path,world,updates,provided,offload):
    task=RoleTask(4,(2,4),score_updates=updates)
    layout=PoolLayout((0,),tuple(range(1,world)),2,'cpu_offload_prefetch' if offload else 'resident')
    mp.spawn(worker,args=(world,task,layout,provided,str(tmp_path/'rdzv'),str(tmp_path)),nprocs=world,join=True)
    assert all(json.loads((tmp_path/f'rank{i}.json').read_text())['passes']==3 for i in range(world))


def test_role_gradient_provider_is_explicit_and_unambiguous():
    task=RoleTask(2,(2,4))
    with pytest.raises(ValueError,match='exactly one'):
        RoleFunctions(make_model,make_opt,sample,generate,guidance,
                      score_input_gradient=analytic_input_gradient).validate(task)
    with pytest.raises(ValueError,match='exactly one'):
        RoleFunctions(make_model,make_opt,sample,generate,None).validate(task)
    RoleFunctions(make_model,make_opt,sample,generate,None,
                  score_input_gradient=analytic_input_gradient).validate(task)
    prior={'generator':4,'score':3,'optimizer_g':4,'gradient_g':12,'optimizer_s':3}
    graph=task.program(initial_versions=prior).compile()
    assert graph.metadata['final_versions']['generator']==5
    assert graph.metadata['final_versions']['score']==3
