"""Changed update cadence must change real Adam commits, not only Program JSON."""
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
import json
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from distillschedule.open_workload.role_program import RoleTask, PoolLayout, AllocatedWorld
from distillschedule.open_workload.role_transport import RoleFunctions, RolePoolExecutor
from distillschedule.open_workload.role_score import ScoreUpdateContext
from test_open_workload_role_transport import make_model, make_opt, sample, generate, guidance, score_train
from test_open_workload_role_continuous import cycle_data, tree_equal


def update_input(ctx):
    # Explicit TEST algorithm: update-indexed perturbation of the same detached
    # G cohort, with changing targets. Not fresh model-generated video rollouts.
    return (ctx.retained_latent + .03*(ctx.update+1),
            ctx.conditions + .05*ctx.update + .02*ctx.cycle)


def reference(task, cycles):
    g, s = make_model('generator', 'cpu'), make_model('score', 'cpu')
    go, so = make_opt('generator', g), make_opt('score', s)
    for cycle in range(cycles):
        retained = []
        for i in range(task.microbatches):
            x = g(cycle_data(cycle, i, 'generator', 'cpu'))
            remote = x.detach().clone().requires_grad_()
            loss = (s(remote)-cycle_data(cycle,i,'score','cpu')).square().mean()
            x.backward(torch.autograd.grad(loss,remote)[0]/task.microbatches)
            retained.append(x.detach())
        go.step(); go.zero_grad(set_to_none=True)
        for u in range(task.score_updates):
            # Independent loop/math, not execute_updates or update_input.
            for i,x in enumerate(retained):
                target = cycle_data(cycle,i,'score','cpu') + .05*u + .02*cycle
                loss = (s(x+.03*(u+1))-target).square().mean()/task.microbatches
                loss.backward()
            so.step(); so.zero_grad(set_to_none=True)
    return {'generator':(g,go),'score':(s,so)}


def worker(rank,world,n,updates,rdzv,folder):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+rdzv,rank=rank,
                            world_size=world,timeout=timedelta(seconds=65))
    runtime = None
    try:
        task = RoleTask(n,(2,4),score_updates=updates)
        layout = PoolLayout((0,),tuple(range(1,world)),2)
        fn = RoleFunctions(make_model,make_opt,sample,generate,guidance,score_train,
                           sample_at=cycle_data,score_update_input=update_input)
        runtime = RolePoolExecutor(task,layout,AllocatedWorld((2**26,)*world),fn,device='cpu')
        initial_ids = None
        for cycle in range(2):
            receipt,model,opt=runtime.run()
            if initial_ids is None: initial_ids=(id(model),id(opt))
            assert (id(model),id(opt))==initial_ids
            expected,expected_opt=reference(task,cycle+1)[runtime.role]
            tree_equal(model.state_dict(),expected.state_dict())
            tree_equal(opt.state_dict(),expected_opt.state_dict())
            assert receipt['final_versions']['generator']==cycle+1
            assert receipt['final_versions']['score']==(cycle+1)*updates
            for r in receipt['ranks']:
                commits=[e for e in r['events'] if e['action']=='score_commit']
                assert len(commits)==(updates if r['role']=='score' else 0)
                if commits:assert [e['score_update'] for e in commits]==list(range(updates))
        Path(folder,f'{rank}.json').write_text(json.dumps({'ok':True}))
    finally:
        if runtime is not None:runtime.close()
        dist.destroy_process_group()


@pytest.mark.parametrize('world,n,updates',[(2,4,1),(2,4,3),(3,5,5)])
def test_repeated_score_commits_match_independent_global_adam(tmp_path,world,n,updates):
    mp.spawn(worker,args=(world,n,updates,str(tmp_path/'rdzv'),str(tmp_path)),nprocs=world,join=True)
    assert all(json.loads((tmp_path/f'{r}.json').read_text())['ok'] for r in range(world))


def test_repetition_requires_explicit_data_semantics_and_identity():
    from distillschedule.open_workload.role_planning import code_identity
    fn=RoleFunctions(make_model,make_opt,sample,generate,guidance,score_train)
    with pytest.raises(ValueError,match='explicit score_update_input'):
        fn.validate(RoleTask(4,(2,4),score_updates=3))
    updated=replace(fn,score_update_input=update_input)
    updated.validate(RoleTask(4,(2,4),score_updates=3))
    assert code_identity(updated)!=code_identity(fn)
    with pytest.raises(ValueError,match='without Score updates'):
        replace(updated,score_train_loss=None).validate(RoleTask(4,(2,4)))
