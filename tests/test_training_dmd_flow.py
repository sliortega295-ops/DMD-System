"""Shape-independent No-GAN flow DMD, not official DMD2/Self-Forcing math."""
from copy import deepcopy

import pytest
import torch
from torch import nn

from distillschedule.training.adapters import ModelComponents
from distillschedule.training.prediction import PredictionSpec


class OtherFlow(nn.Module):
    """A deliberately non-LTX, two-dimensional payload exercises the protocol."""
    def __init__(self, value):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(value, dtype=torch.float32))

    def validate_feeds(self, feeds):
        if set(feeds) != {'latent', 'condition', 'null_condition'}:
            raise ValueError('explicit actual null condition required')

    def sample_template(self, feeds):
        return {'payload': feeds['latent']}

    def rollout(self, feeds, sigmas, gradient_policy):
        self.validate_feeds(feeds)
        value = torch.randn_like(feeds['latent'])
        for index, (sigma, following) in enumerate(zip(sigmas, sigmas[1:])):
            if gradient_policy == 'last_step' and index != len(sigmas) - 2:
                with torch.no_grad():
                    value = value + (following - sigma) * self.denoise({'payload': value},
                        value.new_full((value.shape[0],), sigma), feeds, conditional=True)['payload']
            else:
                value = value + (following - sigma) * self.denoise({'payload': value},
                    value.new_full((value.shape[0],), sigma), feeds, conditional=True)['payload']
        return {'payload': value}

    def denoise(self, sample, sigma, feeds, *, conditional):
        condition = feeds['condition' if conditional else 'null_condition']
        return {'payload': sample['payload'] * self.weight + condition}

    def loss_masks(self, feeds, sample):
        return {'payload': torch.ones_like(sample['payload'], dtype=torch.bool)}

    def apply_conditioning(self, sample, feeds):
        return sample


def policy(**changes):
    return dict(fake_updates=2, fake_sample_policy='reuse_student',
        rollout_sigmas=[1., .75, .5, .25, 0.], gradient_policy='all_steps',
        sigma_min=.1, sigma_max=.9, teacher_cfg=2., fake_cfg=1.,
        student_loss='flow_x0_surrogate', fake_loss='flow_velocity_mse',
        normalizer='per_modality_mean_abs', normalizer_epsilon=1e-5,
        modality_weights={'payload': 1.}, max_grad_norm=1., **changes)


def components():
    from distillschedule.training.dmd import training_state
    student, fake, teacher = OtherFlow(.2), OtherFlow(.3), OtherFlow(.6)
    teacher.requires_grad_(False)
    return ModelComponents(state=training_state(student, fake, teacher,
        student_lr=.01, fake_lr=.02, scheduler_gamma=.95, ema_decay=.9),
        prediction=PredictionSpec('flow'), capabilities=('dmd_flow_v1',)).validate()


def test_generic_algorithm_updates_a_non_ltx_payload_and_declares_actual_versions():
    from distillschedule.training.dmd import build_algorithm
    from distillschedule.open_workload.torch_backend import Session
    model = components()
    feeds = dict(latent=torch.zeros(2, 3), condition=torch.full((2, 3), .15),
                 null_condition=torch.full((2, 3), -.25))
    before = deepcopy(model.state['teacher'].state_dict())
    old = model.state['student'].weight.detach().clone()
    workload = build_algorithm(model, feeds, policy(), 0, 22)
    session = Session(workload, {'backend': 'torch-callable-resident-v1',
        'share_immutable': False, 'checkpoint_bindings': []}, copy_state=False)
    outputs = session.run()
    assert all(torch.isfinite(value) for value in outputs.values())
    assert not torch.equal(old, model.state['student'].weight)
    assert torch.equal(before['weight'], model.state['teacher'].weight)
    assert model.state['dmd_counters'].tolist() == [1, 1, 2]
    assert model.state['student_scheduler'].last_epoch == 1
    assert model.state['fake_scheduler'].last_epoch == 2
    versions = session.receipt['dispatcher']['final_versions']
    assert versions['student'] == 1 and versions['fake'] == 2 and versions['teacher'] == 0
    assert versions['student_scheduler'] == 1 and versions['fake_scheduler'] == 2
    assert versions['student_ema'] == 1
    assert all(parameter.grad is None for name in ('student', 'fake', 'teacher')
               for parameter in model.state[name].parameters())


@pytest.mark.parametrize('change', [dict(normalizer='guess'), dict(gradient_policy='truncate_silently'),
    dict(fake_sample_policy='stale'), dict(fake_cfg=2.), dict(rollout_sigmas=[1., .5, .75, 0.]),
    dict(modality_weights={'missing': 1.}), dict(unknown=True)])
def test_invalid_or_ambiguous_policy_rejected(change):
    from distillschedule.training.dmd import build_algorithm
    model = components()
    feeds = dict(latent=torch.zeros(2, 3), condition=torch.ones(2, 3), null_condition=torch.zeros(2, 3))
    settings = policy(); settings.update(change)
    with pytest.raises(ValueError):
        build_algorithm(model, feeds, settings, 0, 1)


def test_flow_surrogate_is_detached_and_masks_clean_tokens():
    from distillschedule.training.dmd import student_objective, fake_objective
    sample = {'a': torch.tensor([[[2.], [100.]]], requires_grad=True)}
    teacher = {'a': torch.tensor([[[1.], [0.]]], requires_grad=True)}
    fake = {'a': torch.tensor([[[1.5], [-20.]]], requires_grad=True)}
    masks = {'a': torch.tensor([[[True], [False]]])}
    loss = student_objective(sample, teacher, fake, masks, weights={'a': 1.},
        normalizer='per_modality_mean_abs', epsilon=1e-5)
    loss.backward()
    assert sample['a'].grad.tolist() == [[[.5], [0.]]]
    assert teacher['a'].grad is None and fake['a'].grad is None
    velocity = {'a': torch.tensor([[[2.], [900.]]], requires_grad=True)}
    target = {'a': torch.tensor([[[1.], [-900.]]])}
    fake_objective(velocity, target, masks, weights={'a': 1.}).backward()
    assert velocity['a'].grad.tolist() == [[[2.], [0.]]]


@pytest.mark.parametrize('sample_policy', ['reuse_student', 'fresh_student'])
@pytest.mark.parametrize('gradient_policy', ['all_steps', 'last_step'])
def test_explicit_sample_and_gradient_policies_execute_on_other_model(sample_policy, gradient_policy):
    from distillschedule.training.dmd import build_algorithm
    from distillschedule.open_workload.torch_backend import Session
    model = components()
    feeds = dict(latent=torch.zeros(2, 3), condition=torch.full((2, 3), .15), null_condition=torch.full((2, 3), -.25))
    parameters = policy(); parameters.update(fake_sample_policy=sample_policy, gradient_policy=gradient_policy)
    workload = build_algorithm(model, feeds, parameters, 0, 29)
    session = Session(workload, {'backend': 'torch-callable-resident-v1',
        'share_immutable': False, 'checkpoint_bindings': []}, copy_state=False)
    session.run()
    assert model.state['dmd_counters'].tolist() == [1, 1, 2]


_CLOCK_FAULTS = ('cycle_counter', 'student_counter', 'fake_counter', 'student_last_epoch',
    'fake_last_epoch', 'student_step_count', 'fake_step_count', 'role_version',
    'optimizer_version', 'counter_version', 'teacher_version', 'version_type')


def _flow_session(model):
    from distillschedule.training.dmd import build_algorithm
    from distillschedule.open_workload.torch_backend import Session
    feeds = dict(latent=torch.zeros(2, 3), condition=torch.full((2, 3), .15),
                 null_condition=torch.full((2, 3), -.25))
    workload = build_algorithm(model, feeds, policy(), 0, 22)
    return Session(workload, {'backend': 'torch-callable-resident-v1',
        'share_immutable': False, 'checkpoint_bindings': []}, copy_state=False)


def _change_clock(state, versions, fault, *, packed=False):
    def value(name):
        return state[name]['value'] if packed else state[name]
    if fault.endswith('_counter'):
        counters = value('dmd_counters')
        # The original unguarded tensor is also used for proof-first RED runs.
        if isinstance(counters, dict):
            assert counters['schema_version'] == 'distillschedule.dmd_clocks.v1'
            counters = counters['counters']
        counters[{'cycle_counter': 0, 'student_counter': 1, 'fake_counter': 2}[fault]] -= 1
    elif fault.endswith(('_last_epoch', '_step_count')):
        role, field = fault.split('_', 1)
        field = '_step_count' if field == 'step_count' else field
        scheduler = value(role + '_scheduler')
        if packed:
            scheduler[field] -= 1
        else:
            setattr(scheduler, field, getattr(scheduler, field) - 1)
    else:
        name = dict(role_version='fake', optimizer_version='student_optimizer',
            counter_version='dmd_counters', teacher_version='teacher', version_type='student')[fault]
        versions[name] = True if fault == 'version_type' else versions[name] + 1


def _rehash_checkpoint(path, payload):
    import json
    from distillschedule.workloads.best_known.registry import file_sha256
    torch.save(payload, path / 'state.pt')
    manifest = json.loads((path / 'manifest.json').read_text())
    manifest['versions'] = payload['versions']
    manifest['files']['state.pt'] = file_sha256(path / 'state.pt')
    (path / 'manifest.json').write_text(json.dumps(manifest, sort_keys=True))
    (path / '_SUCCESS').write_text(file_sha256(path / 'manifest.json') + '\n')


@pytest.mark.parametrize('fault', _CLOCK_FAULTS)
def test_flow_checkpoint_clocks_reject_inconsistent_restore_before_any_live_mutation(tmp_path, fault):
    from distillschedule.training.checkpoint import pack_state, save_checkpoint, restore_checkpoint
    from test_training_end_to_end import equal
    original = components(); session = _flow_session(original); session.run()
    versions = dict(session.receipt['dispatcher']['final_versions'])
    path = save_checkpoint(tmp_path / 'checkpoint', original.state, identity={}, cycle=1, cursor=1, versions=versions)
    payload = torch.load(path / 'state.pt', weights_only=True)
    _change_clock(payload['state'], payload['versions'], fault, packed=True)
    _rehash_checkpoint(path, payload)
    live = components(); _flow_session(live)
    with torch.no_grad():
        live.state['student'].weight.fill_(-.75)
    before = pack_state(live.state)
    with pytest.raises(ValueError, match='complete-cycle|scheduler clock|logical.*version'):
        restore_checkpoint(path, live.state, identity={})
    assert equal(before, pack_state(live.state)), 'rejection must precede every module/Adam/scheduler/clock restore'


@pytest.mark.parametrize('fault', _CLOCK_FAULTS)
def test_flow_checkpoint_clocks_reject_inconsistent_save_without_publication(tmp_path, fault):
    from distillschedule.training.checkpoint import pack_state, save_checkpoint
    from test_training_end_to_end import equal
    model = components(); session = _flow_session(model); session.run()
    versions = dict(session.receipt['dispatcher']['final_versions'])
    _change_clock(model.state, versions, fault)
    before = pack_state(model.state)
    with pytest.raises(ValueError, match='complete-cycle|scheduler clock|logical.*version'):
        save_checkpoint(tmp_path / 'must-not-publish', model.state, identity={}, cycle=1, cursor=1, versions=versions)
    assert equal(before, pack_state(model.state))
    assert not (tmp_path / 'must-not-publish').exists()
    assert not list(tmp_path.glob('.must-not-publish*'))


@pytest.mark.parametrize('fault', ['student_counter', 'fake_counter', 'student_last_epoch', 'fake_step_count'])
def test_live_flow_clocks_fail_before_the_first_role_update(fault):
    from distillschedule.training.checkpoint import pack_state
    from test_training_end_to_end import equal
    model = components(); session = _flow_session(model)
    _change_clock(model.state, {}, fault)
    before = pack_state(model.state)
    with pytest.raises(ValueError):
        session.run()
    assert equal(before, pack_state(model.state)), 'the first Student update must not run on inconsistent clocks'


def test_flow_clock_schema_and_native_scheduler_arithmetic_remain_explicit():
    from distillschedule.open_workload.torch_backend import tensors
    from distillschedule.training.dmd import DMDClocks, DMDExponentialLR
    model = components(); _flow_session(model)
    counters = model.state['dmd_counters']
    assert type(counters) is DMDClocks
    assert counters.state_dict()['schema_version'] == DMDClocks.SCHEMA
    live = list(tensors(counters))
    assert len(live) == 1 and live[0] is counters.state_dict()['counters']
    for role in ('student', 'fake'):
        scheduler, optimizer = model.state[role + '_scheduler'], model.state[role + '_optimizer']
        assert type(scheduler) is DMDExponentialLR and scheduler.optimizer is optimizer
        reference_optimizer = torch.optim.AdamW([nn.Parameter(torch.ones(()))], lr=optimizer.param_groups[0]['lr'])
        reference = torch.optim.lr_scheduler.ExponentialLR(reference_optimizer, gamma=.95)
        for _ in range(12):
            optimizer.step(); scheduler.step()
            reference_optimizer.step(); reference.step()
            native = scheduler.state_dict()
            assert native.pop('dmd_schema_version') == DMDExponentialLR.SCHEMA
            assert native.pop('dmd_role') == role
            assert native == reference.state_dict()
            assert optimizer.param_groups[0]['lr'] == reference_optimizer.param_groups[0]['lr']
        assert '_dmd_progress' not in scheduler.state_dict() and '_progress' not in counters.state_dict()


def test_flow_complete_cycle_contract_cannot_be_rebound_to_another_ratio():
    from distillschedule.training.dmd import build_algorithm
    from distillschedule.training.checkpoint import pack_state
    from test_training_end_to_end import equal
    model = components(); _flow_session(model)
    feeds = dict(latent=torch.zeros(2, 3), condition=torch.ones(2, 3), null_condition=torch.zeros(2, 3))
    before = pack_state(model.state)
    changed = policy(); changed['fake_updates'] = 3
    with pytest.raises(ValueError, match='clock contract changed'):
        build_algorithm(model, feeds, changed, 0, 22)
    assert equal(before, pack_state(model.state))
