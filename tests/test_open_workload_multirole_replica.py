"""Unknown two-role programs: actual collectives and Adam, not model stubs."""
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
import json
import pytest
import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp
from distillschedule.workloads.program import Program, Call
from distillschedule.open_workload.torch_backend import Workload, Session, candidates, compare, unwrap_module
from distillschedule.open_workload.replica_contract import ReplicaSemantics
from distillschedule.open_workload.replicated import inventory, make_session, identity


class UnregisteredVideoBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.base = nn.Linear(width, width).requires_grad_(False)
        self.adapter = nn.Linear(width, width, bias=False)

    def forward(self, value):
        return torch.tanh(self.base(value) + self.adapter(value))


def role_a(ctx):
    return ctx.state['role_a'](next(iter(ctx.inputs.values())))


def role_b(ctx):
    return ctx.state['role_b'](next(iter(ctx.inputs.values())))


def frozen_input_gradient(ctx):
    return ctx.state['fixed'](next(iter(ctx.inputs.values())))


def loss(ctx):
    values = list(ctx.inputs.values())
    return ((values[0]-values[1]) if len(values)==2 else values[0]).square().mean()


def backward(ctx):
    next(iter(ctx.inputs.values())).backward()


def update_a(ctx):
    ctx.state['opt_a'].step(); ctx.state['opt_a'].zero_grad(set_to_none=True)


def update_b(ctx):
    ctx.state['opt_b'].step(); ctx.state['opt_b'].zero_grad(set_to_none=True)


def build_workload(frequency, width, frames, *, rank=0, global_batch=False):
    with torch.random.fork_rng():
        torch.manual_seed(987)
        a = UnregisteredVideoBlock(width)
        b = deepcopy(a)
        frozen = deepcopy(a).requires_grad_(False)
        # Share exactly frozen tensors only; trainable role updates are different.
        with torch.no_grad(): b.adapter.weight.add_(.03)
        x = torch.randn(2, frames, width)
    x = torch.cat((x, x+.2)) if global_batch else x+rank*.2
    state = {'role_a':a, 'role_b':b, 'fixed':frozen,
             'opt_a':torch.optim.Adam(a.adapter.parameters(),lr=.001),
             'opt_b':torch.optim.Adam(b.adapter.parameters(),lr=.002),
             'grad_a':None, 'grad_b':None}
    calls = [Call('make','a',('input',),('role_a',),grad='grad',replay_safe=True),
             Call('through_fixed','fixed',('make',),('fixed',),grad='grad',replay_safe=True),
             Call('loss_a','loss',('through_fixed',),grad='grad'),
             Call('differentiate_a','backward',('loss_a',),('grad_a',),('grad_a',),
                  grad='grad',backward_of=('loss_a',)),
             Call('commit_a','update_a',('differentiate_a',),('role_a','opt_a','grad_a'),
                  ('role_a','opt_a','grad_a'))]
    for index in range(frequency):
        calls.extend((
            Call(f'draw_{index}','a',('input',),('role_a',)),
            Call(f'estimate_{index}','b',('input',),('role_b',),grad='grad',replay_safe=True),
            Call(f'loss_b_{index}','loss',(f'estimate_{index}',f'draw_{index}'),grad='grad'),
            Call(f'differentiate_b_{index}','backward',(f'loss_b_{index}',),('grad_b',),('grad_b',),
                 grad='grad',backward_of=(f'loss_b_{index}',)),
            Call(f'commit_b_{index}','update_b',(f'differentiate_b_{index}',),('role_b','opt_b','grad_b'),
                 ('role_b','opt_b','grad_b'))))
    program = Program(tuple(calls),dict.fromkeys(state,0),('input',),(f'loss_b_{frequency-1}',),
                      {'frequency':frequency,'width':width,'frames':frames})
    workload = Workload(program,{'a':role_a,'b':role_b,'fixed':frozen_input_gradient,'loss':loss,
                      'backward':backward,'update_a':update_a,'update_b':update_b},state,{'input':x},
                      immutable_modules=('role_a.base','role_b.base','fixed.base'))
    if not global_batch:
        workload.replica_semantics = ReplicaSemantics({'a':'role_a','b':'role_b'},
            ('differentiate_a',*(f'differentiate_b_{i}' for i in range(frequency))),2)
    return workload


def multirole_worker(rank, rendezvous, output, frequency, width, frames):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2,
                            timeout=timedelta(seconds=45))
    try:
        workload=build_workload(frequency,width,frames,rank=rank)
        original=workload.identity
        oracle=build_workload(frequency,width,frames,global_batch=True)
        baseline=Session(oracle,candidates(oracle,limit=1)['candidates'][0])
        expected=baseline.result_state(baseline.run())['state']
        local_options=candidates(workload,limit=16)['candidates']
        # Cover the full checkpoint and all-shared configuration, not only the
        # enumeration prefix. Choice of mechanisms is a test matrix, not a recipe.
        chosen=[local_options[0],next(o for o in local_options if o['share_immutable'] and
                                     len(o['checkpoint_bindings'])==3)]
        options=[o for o in inventory(workload,limit=64)['candidates'] if o['local'] in chosen
                 and not o['reduction'].startswith('fsdp2_')]
        assert len(options)==6
        for option in options:
            session=make_session(workload,option);outputs=session.run()
            assert compare(expected,session.result_state(outputs)['state'],rtol=3e-5,atol=3e-7)['status']=='PASS'
            assert session.receipt['dispatcher']['final_versions']['role_a']==1
            assert session.receipt['dispatcher']['final_versions']['role_b']==frequency
            assert session.effects['parameter_backward_calls']==frequency+1
            if option['local']['share_immutable']:
                models=[unwrap_module(session.state[k]) for k in ('role_a','role_b','fixed')]
                assert models[0].base.weight is models[1].base.weight is models[2].base.weight
                assert models[0].adapter.weight is not models[1].adapter.weight
                assert session.effects['recompute_calls']>0
        assert workload.identity==original
        Path(output,f'rank{rank}.json').write_text(json.dumps({'variants':len(options),'gpu_execution':'NOT_RUN'}))
    finally:dist.destroy_process_group()


@pytest.mark.parametrize('frequency,width,frames',[(1,4,3),(3,6,5)])
def test_two_trainable_roles_with_new_frequency_and_geometry(tmp_path,frequency,width,frames):
    mp.spawn(multirole_worker,args=(str(tmp_path/'rdzv'),str(tmp_path),frequency,width,frames),
             nprocs=2,join=True)
    assert json.loads((tmp_path/'rank0.json').read_text())['variants']==6


def mismatch_worker(rank,rendezvous,kind):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2,
                            timeout=timedelta(seconds=20))
    try:
        workload=build_workload(1,4,3,rank=rank)
        if kind=='state' and rank:
            with torch.no_grad(): workload.state['role_b'].adapter.weight.add_(.5)
        if kind=='shape' and rank:workload.feeds={'input':workload.feeds['input'][:,:2]}
        try:identity(workload)
        except ValueError as exc:assert 'differs' in str(exc)
        else:raise AssertionError('incompatible rank admitted')
        dist.barrier()
    finally:dist.destroy_process_group()


@pytest.mark.parametrize('kind',['state','shape'])
def test_collective_preflight_fails_on_all_ranks(tmp_path,kind):
    mp.spawn(mismatch_worker,args=(str(tmp_path/'rdzv'),kind),nprocs=2,join=True)
