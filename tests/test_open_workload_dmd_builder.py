"""Hook-defined Student1/FakeN semantics, not a pretrained/GPU DMD claim."""
from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from torch import nn

from distillschedule.open_workload.dmd import (
    DMDHook, DMDHooks, build_dmd_program, build_dmd_workload, dmd_inputs,
)
from distillschedule.open_workload.torch_backend import Session, candidates, compare
from distillschedule.workloads.program import Call, Program
from runtime.dmd_boundary import (
    add_noise, critic_denoising_loss, detached_dmd_surrogate_loss,
    dmd_input_gradient, shifted_sigma_schedule,
)


class CrossScaleModel(nn.Module):
    """Different hidden widths, identical latent input/output ABI."""
    def __init__(self, width):
        super().__init__()
        self.up = nn.Conv3d(2, width, 1)
        self.down = nn.Conv3d(width, 2, 1)

    def forward(self, value):
        return self.down(torch.tanh(self.up(value)))


def student_forward(ctx):
    values = dmd_inputs(ctx)
    history = values.get('history')
    if history is not None:
        assert all(not x.requires_grad and x.grad_fn is None for x in history.values())
    previous = history['current'] if history is not None else 0
    return ctx.state['student'](values['feeds']['latent'] + .1 * previous)


def history_pair(ctx):
    values = dmd_inputs(ctx)
    assert not torch.is_grad_enabled()
    current = values['current']
    previous = values.get('previous')
    assert not current.requires_grad and current.grad_fn is None
    return {'previous': previous['current'] if previous is not None else torch.zeros_like(current),
            'current': current}


def noise(ctx):
    values = dmd_inputs(ctx)
    assert not torch.is_grad_enabled()
    assert not values['generated'].requires_grad
    feeds = values['feeds']; cycle = ctx.attributes['cycle']
    epsilon = (feeds['student_noise'][cycle] if ctx.attributes['phase'] == 'student'
               else feeds['fake_noise'][cycle][ctx.attributes['fake_index']])
    return {'noisy': add_noise(values['generated'], epsilon, feeds['timestep'],
                               feeds['sigmas'], feeds['timesteps']),
            'epsilon': epsilon, 'timestep': feeds['timestep']}


def teacher_forward(ctx):
    values = dmd_inputs(ctx)
    assert not torch.is_grad_enabled()
    sample = values['noise']['noisy']; model = ctx.state['teacher']
    return {'cond': model(sample + .07), 'uncond': model(sample)}


def fake_forward(ctx):
    values = dmd_inputs(ctx)
    assert torch.is_grad_enabled() == (ctx.attributes['phase'] == 'fake')
    assert not values['generated'].requires_grad
    return {'cond': ctx.state['fake'](values['noise']['noisy'] + .07)}


def student_loss(ctx):
    values = dmd_inputs(ctx); feeds = values['feeds']
    assert values['generated'].requires_grad
    assert not values['teacher']['cond'].requires_grad
    assert not values['fake']['cond'].requires_grad
    assert all(x.grad_fn is None for x in values.get('history', {}).values())
    # This is the existing source-semantic boundary, not a builder-selected loss.
    with torch.no_grad():
        gradient, _ = dmd_input_gradient(
            estimated_clean=values['generated'].detach(),
            noisy_sample=values['noise']['noisy'], timestep=values['noise']['timestep'],
            sigmas=feeds['sigmas'], timesteps=feeds['timesteps'],
            fake_flow_cond=values['fake']['cond'],
            real_flow_cond=values['teacher']['cond'],
            real_flow_uncond=values['teacher']['uncond'], real_guidance_scale=1.5)
    return detached_dmd_surrogate_loss(values['generated'], gradient)


def fake_loss(ctx):
    values = dmd_inputs(ctx)
    assert values['fake']['cond'].requires_grad
    assert not values['generated'].requires_grad
    assert all(x.grad_fn is None for x in values.get('history', {}).values())
    return critic_denoising_loss(values['fake']['cond'], values['noise']['epsilon'],
                                values['generated'])


def make_workload(fake_updates, policy, *, widths=(3, 5, 7), replica_batch_size=None,
                  with_history=True, initial_history=None):
    with torch.random.fork_rng():
        torch.manual_seed(4321)
        student, fake, teacher = (CrossScaleModel(width) for width in widths)
        teacher.requires_grad_(False).eval()
        shape = (2, 2, 2, 1, 1)
        latent = torch.randn(shape)
        initial = torch.randn(shape)
        student_noise = tuple(torch.randn(shape) for _ in range(2))
        fake_noise = tuple(tuple(torch.randn(shape) for _ in range(fake_updates)) for _ in range(2))
    sigmas, timesteps = shifted_sigma_schedule(num_train_timesteps=8, timestep_shift=1)
    feeds = {'latent': latent, 'initial_history': (initial_history if initial_history is not None else
             {'previous': torch.zeros_like(initial), 'current': initial}),
             'student_noise': student_noise, 'fake_noise': fake_noise,
             'timestep': torch.tensor([3, 5]), 'sigmas': sigmas, 'timesteps': timesteps}
    hooks = DMDHooks(student_forward=DMDHook(student_forward, replay_safe=True),
                     fake_forward=DMDHook(fake_forward, replay_safe=True),
                     teacher_forward=teacher_forward, noise=noise,
                     student_loss=student_loss, fake_loss=fake_loss,
                     history=history_pair if with_history else None)
    return build_dmd_workload(student=student, fake=fake, teacher=teacher,
        student_optimizer=torch.optim.Adam(student.parameters(), lr=.002),
        fake_optimizer=torch.optim.Adam(fake.parameters(), lr=.003),
        hooks=hooks, feeds=feeds, cycles=2, fake_updates=fake_updates,
        fake_sample_policy=policy, history_feed='initial_history' if with_history else None,
        replica_batch_size=replica_batch_size)


def sequential_oracle(workload, fake_updates, policy, *, with_history=True):
    """Independent loop: no Program, builder hooks, Session, or graph executor."""
    state, feeds = deepcopy((dict(workload.state), dict(workload.feeds)))
    student, fake, teacher = (state[key] for key in ('student', 'fake', 'teacher'))
    student_opt, fake_opt = (state[key] for key in ('student_optimizer', 'fake_optimizer'))
    history = feeds['initial_history'] if with_history else None; losses = {}
    for cycle in range(2):
        old_history = history
        previous = old_history['current'] if old_history is not None else 0
        generated = student(feeds['latent'] + .1 * previous)
        sample = generated.detach()
        if with_history:
            history = {'previous': old_history['current'].detach(), 'current': sample}
        with torch.no_grad():
            noisy = add_noise(sample, feeds['student_noise'][cycle], feeds['timestep'],
                              feeds['sigmas'], feeds['timesteps'])
            gradient, _ = dmd_input_gradient(estimated_clean=sample, noisy_sample=noisy,
                timestep=feeds['timestep'], sigmas=feeds['sigmas'], timesteps=feeds['timesteps'],
                fake_flow_cond=fake(noisy + .07), real_flow_cond=teacher(noisy + .07),
                real_flow_uncond=teacher(noisy), real_guidance_scale=1.5)
        loss = detached_dmd_surrogate_loss(generated, gradient)
        losses[f'dmd.c{cycle}.student_loss'] = loss.detach()
        loss.backward(); student_opt.step(); student_opt.zero_grad(set_to_none=True)
        for index in range(fake_updates):
            with torch.no_grad():
                target = (student(feeds['latent'] + .1 * previous)
                          if policy == 'fresh_student' else sample)
                noisy = add_noise(target, feeds['fake_noise'][cycle][index], feeds['timestep'],
                                  feeds['sigmas'], feeds['timesteps'])
            loss = critic_denoising_loss(fake(noisy + .07), feeds['fake_noise'][cycle][index], target)
            losses[f'dmd.c{cycle}.f{index}.loss'] = loss.detach()
            loss.backward(); fake_opt.step(); fake_opt.zero_grad(set_to_none=True)

    def clone(value):
        if isinstance(value, torch.Tensor): return value.detach().clone()
        if isinstance(value, nn.Module): return clone(value.state_dict())
        if isinstance(value, torch.optim.Optimizer): return clone(value.state_dict())
        if isinstance(value, dict): return {str(k): clone(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)): return [clone(v) for v in value]
        return value
    if with_history:
        losses['dmd.c1.history'] = history
    return clone(state), clone(losses)


@pytest.mark.parametrize('fake_updates,policy,widths,with_history', [
    (1, 'reuse_student', (3, 5, 7), True), (3, 'reuse_student', (4, 6, 9), True),
    (1, 'fresh_student', (3, 5, 7), True), (3, 'fresh_student', (4, 6, 9), True),
    (1, 'reuse_student', (3, 5, 7), False), (3, 'fresh_student', (4, 6, 9), False),
])
def test_two_cycles_full_models_and_adam_match_independent_oracle(fake_updates, policy, widths, with_history):
    workload = make_workload(fake_updates, policy, widths=widths, with_history=with_history)
    original = workload.identity
    expected_state, expected_outputs = sequential_oracle(
        workload, fake_updates, policy, with_history=with_history)
    options = candidates(workload, limit=32)['candidates']
    full_checkpoint = next(option for option in options
        if set(option['checkpoint_bindings']) == {'student_forward', 'fake_forward'}
        and option.get('checkpoint_policy', 'recompute') == 'recompute')
    for option in (options[0], full_checkpoint):
        session = Session(workload, option)
        result = session.result_state(session.run())
        assert compare(expected_state, result['state'], rtol=3e-6, atol=3e-7)['status'] == 'PASS'
        assert compare(expected_outputs, result['outputs'], rtol=3e-6, atol=3e-7)['status'] == 'PASS'
        versions = session.receipt['dispatcher']['final_versions']
        assert (versions['student'], versions['fake'], versions['teacher']) == (2, 2 * fake_updates, 0)
        for name, tensor in workload.state['teacher'].state_dict().items():
            assert torch.equal(tensor, session.state['teacher'].state_dict()[name])
        assert any(not torch.equal(value, session.state['student'].state_dict()[name])
                   for name, value in workload.state['student'].state_dict().items())
        assert any(not torch.equal(value, session.state['fake'].state_dict()[name])
                   for name, value in workload.state['fake'].state_dict().items())
        assert all(parameter.grad is None for role in ('student', 'fake', 'teacher')
                   for parameter in session.state[role].parameters())
        assert session.receipt['gpu_execution'] == 'NOT_RUN'
        if option['checkpoint_bindings']:
            assert session.effects['checkpoint_calls'] == 2 * (1 + fake_updates)
            assert session.effects['recompute_calls'] > 0
    assert workload.identity == original


def test_history_and_score_branches_are_explicit_detach_boundaries():
    workload = make_workload(2, 'reuse_student')
    calls = {call.id: call for call in workload.program.calls}
    assert calls['dmd.c0.detach'].grad == calls['dmd.c0.history'].grad == 'no_grad'
    assert 'dmd.c0.history' in calls['dmd.c1.student_forward'].inputs
    operations = {op.operation_id: op for op in workload.program.compile().operations}
    assert 'student' in operations['dmd.c0.student_backward'].required_versions
    assert not {'fake', 'teacher'} & set(operations['dmd.c0.student_backward'].required_versions)
    assert 'fake' in operations['dmd.c0.f0.backward'].required_versions
    assert not {'student', 'teacher'} & set(operations['dmd.c0.f0.backward'].required_versions)


def test_stale_live_student_provenance_is_rejected():
    workload = make_workload(1, 'reuse_student')
    calls = list(workload.program.calls)
    index = next(i for i, call in enumerate(calls) if call.id == 'dmd.c0.student_backward')
    calls.insert(index, Call('unsafe_update', 'unsafe', reads=('student',), writes=('student',)))
    with pytest.raises(ValueError, match='backward/replay after parameter version changed'):
        replace(workload.program, calls=tuple(calls))


def test_external_history_graph_and_trainable_teacher_are_not_admitted():
    workload = make_workload(1, 'reuse_student')
    leaf = torch.ones_like(workload.feeds['latent'], requires_grad=True)
    workload.feeds['initial_history']['current'] = leaf * 2
    with pytest.raises(ValueError, match='external autograd graphs'):
        workload.validate()
    workload = make_workload(1, 'reuse_student')
    workload.state['teacher'].requires_grad_(True)
    with pytest.raises(ValueError, match='immutable module has trainable parameters'):
        workload.validate()


@pytest.mark.parametrize('kind', ['leaf', 'nonleaf'])
def test_initial_history_grad_tensors_are_rejected_at_construction(kind):
    leaf = torch.ones((2, 2, 2, 1, 1), requires_grad=True)
    invalid = leaf if kind == 'leaf' else leaf * 2
    initial_history = {'previous': torch.zeros_like(leaf), 'current': torch.zeros_like(leaf),
                       'nested': [{'latent': invalid}]}
    with pytest.raises(ValueError, match='initial history must contain only detached tensors'):
        make_workload(1, 'reuse_student', initial_history=initial_history)
    assert invalid.requires_grad and leaf.grad is None


def test_detached_nested_initial_history_is_accepted_without_mutation():
    current = torch.ones((2, 2, 2, 1, 1))
    nested = current.clone()
    initial_history = {'previous': torch.zeros_like(current), 'current': current,
                       'nested': [{'latent': nested}]}
    workload = make_workload(1, 'reuse_student', initial_history=initial_history)
    assert workload.validate() is workload
    assert workload.feeds['initial_history'] is initial_history
    assert workload.feeds['initial_history']['nested'][0]['latent'] is nested
    assert not nested.requires_grad and nested.grad_fn is None


def test_replica_contract_is_optional_semantics_not_a_selected_strategy():
    workload = make_workload(2, 'fresh_student')
    assert workload.replica_semantics is None
    # The same standard ABI is sufficient for the existing distributed planner.
    workload = make_workload(2, 'fresh_student', replica_batch_size=2)
    workload.validate()
    assert workload.replica_semantics.module_bindings['fake_forward'] == 'fake'
    assert workload.replica_semantics.local_batch_size == 2


@pytest.mark.parametrize('change', [
    {'cycles': 0}, {'fake_updates': True}, {'fake_sample_policy': 'guess'},
    {'history_feed': 'missing'},
])
def test_invalid_semantic_contract_is_rejected(change):
    hooks = DMDHooks(student_forward, fake_forward, teacher_forward, noise,
                     student_loss, fake_loss, history=history_pair)
    options = dict(hooks=hooks, feed_names=('latent', 'initial_history'), cycles=2,
                   fake_updates=2, fake_sample_policy='reuse_student', history_feed='initial_history')
    with pytest.raises(ValueError):
        build_dmd_program(**{**options, **change})


def test_custom_loss_binding_is_preserved_not_replaced_by_a_builtin_objective():
    workload = make_workload(1, 'reuse_student')
    assert workload.bindings['student_loss'] is student_loss
    assert workload.bindings['fake_loss'] is fake_loss
    assert isinstance(workload.program, Program)


def test_hook_state_effects_are_preserved_in_the_version_graph():
    hooks = DMDHooks(student_forward, fake_forward, teacher_forward,
        DMDHook(noise, reads=('rng_counter',), writes=('rng_counter',)),
        student_loss, fake_loss)
    versions = dict.fromkeys(('student', 'fake', 'teacher', 'student_optimizer',
                              'fake_optimizer', 'student_grad', 'fake_grad'), 0)
    versions['rng_counter'] = 7
    program = build_dmd_program(hooks=hooks, feed_names=('latent',), cycles=2,
        fake_updates=3, fake_sample_policy='reuse_student', initial_versions=versions)
    assert program.compile().metadata['final_versions']['rng_counter'] == 15


def test_builder_rejects_wrong_optimizer_ownership_without_mutating_models():
    workload = make_workload(1, 'reuse_student')
    before = workload.identity
    hooks = DMDHooks(student_forward, fake_forward, teacher_forward, noise,
                     student_loss, fake_loss, history=history_pair)
    with pytest.raises(ValueError, match='optimizer must exactly own'):
        build_dmd_workload(student=workload.state['student'], fake=workload.state['fake'],
            teacher=workload.state['teacher'], student_optimizer=workload.state['student_optimizer'],
            fake_optimizer=workload.state['student_optimizer'], hooks=hooks, feeds=workload.feeds,
            cycles=2, fake_updates=1, fake_sample_policy='reuse_student', history_feed='initial_history')
    assert workload.identity == before
