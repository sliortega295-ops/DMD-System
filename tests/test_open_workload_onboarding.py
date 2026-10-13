"""A new external factory is NOT in the model registry or capability library."""
import json
from pathlib import Path
import pytest
from distillschedule.open_workload.cli import main
from distillschedule.open_workload.planning import source_identity

FACTORY='''
from copy import deepcopy
import torch
from torch import nn
from distillschedule.workloads.program import Program, Call
from distillschedule.open_workload.torch_backend import Workload

def predict(ctx):
    return ctx.state['generator'](next(iter(ctx.inputs.values())))
def frozen_critic(ctx):
    return ctx.state['critic'](next(iter(ctx.inputs.values())))
def loss(ctx):
    return next(iter(ctx.inputs.values())).square().mean()
def backward(ctx):
    next(iter(ctx.inputs.values())).backward()
def step(ctx):
    ctx.state['optimizer'].step();ctx.state['optimizer'].zero_grad(set_to_none=True)

def build(config):
    width=config['width']; frames=config['frames']; updates=config['updates']
    with torch.random.fork_rng():
        torch.manual_seed(321)
        generator=nn.Conv3d(width,width,1)
        critic=nn.Conv3d(width,width,1).requires_grad_(False)
        x=torch.randn(1,width,frames,3,3)
    opt=torch.optim.Adam(generator.parameters(),lr=.001)
    state={'generator':generator,'critic':critic,'optimizer':opt,'grad':None}
    calls=[]
    for i in range(updates):
        calls += [Call(f'g{i}','predict',('noise',),('generator',),grad='grad',replay_safe=True),
                  Call(f'c{i}','critic',(f'g{i}',),('critic',),grad='grad',replay_safe=True),
                  Call(f'l{i}','loss',(f'c{i}',),grad='grad'),
                  Call(f'b{i}','backward',(f'l{i}',),('grad',),('grad',),grad='grad',backward_of=(f'l{i}',)),
                  Call(f'u{i}','step',(f'b{i}',),('generator','optimizer','grad'),('generator','optimizer','grad'))]
    program=Program(tuple(calls),dict.fromkeys(state,0),('noise',),(f'l{updates-1}',),config)
    return Workload(program,{'predict':predict,'critic':frozen_critic,'loss':loss,'backward':backward,'step':step},state,{'noise':x})
'''


@pytest.mark.parametrize('updates,frames,width',[(1,2,2),(2,3,3)])
def test_new_external_video_gan_graph_no_scheduler_changes(tmp_path,updates,frames,width):
    core=source_identity()
    f=tmp_path/'unseen_video.py';f.write_text(FACTORY)
    p=tmp_path/'params.json';p.write_text(json.dumps(dict(updates=updates,frames=frames,width=width)))
    rc=main(['auto','--factory',str(f)+':build','--parameters',str(p),'--output',str(tmp_path/'run'),
             '--samples','1','--max-program-runs','8','--memory-limit-bytes','1000000','--execute'])
    assert rc==0
    report=json.loads((tmp_path/'run/result.json').read_text())
    assert report['reference_gate']['status']=='PASS'
    assert report['dispatcher']['final_versions']['generator']==updates
    assert report['gpu_execution']=='NOT_RUN'
    assert core==source_identity()
    selected=json.loads((tmp_path/'run/plan.json').read_text())
    calibration=json.loads((tmp_path/'run/calibration/calibration.json').read_text())
    backward=next(o for o in calibration['graph']['operations'] if o['operation_id']=='b0')
    assert {'generator','critic'} <= set(backward['required_versions'])
    assert selected['identity']['extra']['factory_parameters']['frames']==frames


def test_public_cli_routes_program_without_model_name(tmp_path,capsys):
    from distillschedule.cli import main as public
    assert public(['workloads','native','program','describe','--factory','does-not-exist.py:build',
                   '--parameters','also-missing.json'])==0
    assert 'NOT_EXECUTED' in capsys.readouterr().out


def test_factory_cannot_return_prewritten_winner(tmp_path):
    f=tmp_path/'invalid.py';f.write_text("def build(config): return {'winner':'fsdp'}")
    p=tmp_path/'p.json';p.write_text('{}')
    assert main(['describe','--factory',str(f)+':build','--parameters',str(p),'--execute'])==2
