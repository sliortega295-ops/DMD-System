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
