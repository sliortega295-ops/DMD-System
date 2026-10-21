"""Small runnable factory: model/algorithm facts only, no strategy or planner.

This is a random-weight API example with a frozen input-differentiable critic,
not a DMD benchmark or evidence of video generation quality.
"""
import torch
from torch import nn
from distillschedule.workloads.program import Program, Call
from distillschedule.open_workload.torch_backend import Workload


def predict(ctx):
    return ctx.state['generator'](next(iter(ctx.inputs.values())))


def critic(ctx):
    return ctx.state['critic'](next(iter(ctx.inputs.values())))


def loss(ctx):
    return next(iter(ctx.inputs.values())).square().mean()


def backward(ctx):
    next(iter(ctx.inputs.values())).backward()


def step(ctx):
    ctx.state['optimizer'].step()
    ctx.state['optimizer'].zero_grad(set_to_none=True)


def build(config):
    width, frames, updates = config['width'], config['frames'], config['updates']
    with torch.random.fork_rng():
        torch.manual_seed(321)
        generator = nn.Conv3d(width, width, 1)
        frozen_critic = nn.Conv3d(width, width, 1).requires_grad_(False)
        x = torch.randn(1, width, frames, 3, 3)
    state = {'generator': generator, 'critic': frozen_critic,
             'optimizer': torch.optim.Adam(generator.parameters(), lr=.001), 'grad': None}
    calls = []
    for i in range(updates):
        calls.extend([
            Call(f'g{i}', 'predict', ('noise',), ('generator',), grad='grad', replay_safe=True),
            Call(f'c{i}', 'critic', (f'g{i}',), ('critic',), grad='grad', replay_safe=True),
            Call(f'l{i}', 'loss', (f'c{i}',), grad='grad'),
            Call(f'b{i}', 'backward', (f'l{i}',), ('grad',), ('grad',), grad='grad', backward_of=(f'l{i}',)),
            Call(f'u{i}', 'step', (f'b{i}',), ('generator', 'optimizer', 'grad'), ('generator', 'optimizer', 'grad')),
        ])
    program = Program(tuple(calls), dict.fromkeys(state, 0), ('noise',), (f'l{updates-1}',), config)
    return Workload(program, {'predict': predict, 'critic': critic, 'loss': loss, 'backward': backward, 'step': step},
                    state, {'noise': x})
