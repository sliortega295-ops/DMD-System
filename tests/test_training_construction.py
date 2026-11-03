"""Small CPU checks for construction-time frozen storage, not model benchmarks."""
from copy import deepcopy
import gc
from types import SimpleNamespace
import weakref

import pytest
import torch
from torch import nn

from distillschedule.open_workload.torch_backend import Session, Workload
from distillschedule.training import dmd
from distillschedule.training.adapters import ModelComponents
from distillschedule.training.construction_sharing import (
    admit_frozen_sharing, frozen_native_subtrees, place_frozen_role,
)
from distillschedule.training.dmd import training_state
from distillschedule.training.low_rank import trainable_scope
from distillschedule.training.session import _construct_model
from distillschedule.workloads.program import Call, Program
from test_training_end_to_end import equal


class Native(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(4, 4)
        self.proj_out = nn.Linear(4, 4)
        self.register_buffer('rotary', torch.tensor([1 + 2j]))
        with torch.no_grad():
            for p in self.parameters():
                p.fill_(0.25)

    def forward(self, value):
        return self.proj_out(self.hidden(value).sin())


PARAMETERS = {'trainable': {'mode': 'lora', 'rank': 2, 'alpha': 2., 'targets': ['proj_out']}}
SHARED = {'mode': 'manual', 'implementation': {
    'backend': 'torch-callable-resident-v1', 'share_immutable': True, 'checkpoint_bindings': []}}


def factory(parameters):
    roles = {}
    with torch.random.fork_rng():
        torch.manual_seed(79)
        for name in ('student', 'fake', 'teacher'):
            native = Native()
            if name == 'fake':
                with torch.no_grad():
                    if parameters.get('different_fake'):
                        for parameter in native.parameters():
                            parameter.add_(.1)
                    elif parameters.get('partial_fake'):
                        native.proj_out.weight.add_(.1)
            if name == 'teacher':
                native.requires_grad_(False).eval()
            else:
                trainable_scope(native, parameters['trainable'])
            identity = {'config': {'width': 4}, 'weights_sha256': (
                'b' if (parameters.get('different_fake') or parameters.get('partial_fake')) and name == 'fake' else 'a') * 64}
            place_frozen_role(native, torch.device('cpu'), identity=identity)
            roles[name] = native
        state = training_state(**roles, student_lr=.001, fake_lr=.001,
                               scheduler_gamma=.99, ema_decay=.99)
    return ModelComponents(state=state, immutable_modules=('teacher',), metadata={})


def construct(parameters=None):
    parameters = deepcopy(PARAMETERS if parameters is None else parameters)
    provider = admit_frozen_sharing(parameters=parameters, request=SimpleNamespace(execution=SHARED),
                                    budget=None, factory=factory)
    assert provider is not None
    components = provider.construct(factory, parameters)
    provider.attach(components)
    return provider, components


def test_reuses_backbone_before_placement_without_merging_trainables_or_buffers():
    provider, components = construct()
    state = components.state
    student, fake, teacher = (state[k] for k in ('student', 'fake', 'teacher'))
    assert student.hidden.weight is fake.hidden.weight is teacher.hidden.weight
    assert student.proj_out.base.weight is fake.proj_out.base.weight is teacher.proj_out.weight
    assert student.proj_out.lora_a is not fake.proj_out.lora_a
    assert student.rotary is not fake.rotary and torch.equal(student.rotary, fake.rotary)
    assert student.rotary.is_complex()
    assert student.hidden.weight is not student.proj_out.base.weight
    receipt = components.construction_receipt
    assert receipt['frozen_parameter_bytes_reused'] == 2 * 40 * 4
    assert receipt['aliased_parameters'] == 8
    assert receipt['placement'] == 'BEFORE_DEVICE_TRANSFER'
    assert set(state['student_ema']) == set(dict(student.named_parameters()))
    provider.close()


def test_runtime_replaced_frozen_parameters_are_reclaimable_before_provider_close():
    provider, components = construct()
    receipt = dict(components.construction_receipt)
    replaced = [weakref.ref(components.state['teacher'].proj_out.weight),
                weakref.ref(components.state['teacher'].proj_out.bias)]
    assert all(parameter() is not None for parameter in replaced)
    try:
        workload = Workload(
            Program((Call('noop', 'noop'),), {name: 0 for name in components.state}),
            {'noop': dmd._commit}, components.state, {},
            immutable_modules=('teacher',)
                + frozen_native_subtrees(components.state['student'], 'student')
                + frozen_native_subtrees(components.state['fake'], 'fake'))
        session = Session(workload, SHARED['implementation'], copy_state=False, persistent=True)
        assert session.effects['shared_parameter_aliases'] > 0
        assert session.state['teacher'].proj_out.weight is session.state['teacher'].hidden.weight
        assert session.state['teacher'].proj_out.bias is session.state['teacher'].hidden.bias
        gc.collect()
        assert all(parameter() is None for parameter in replaced)
        assert not provider.closed
        assert not provider.pool.parameters
        assert components.construction_receipt == receipt
    finally:
        provider.close()


def test_session_constructs_sharing_provider_and_owns_managed_close(monkeypatch):
    providers = []

    def admit(**kwargs):
        provider = admit_frozen_sharing(**kwargs, factory=factory)
        providers.append(provider)
        return provider

    monkeypatch.setattr(factory, 'admit_construction', admit, raising=False)
    reference = __name__ + ':factory'
    config = SimpleNamespace(model={'factory': reference, 'parameters': deepcopy(PARAMETERS)},
                             trainer={'device': 'cpu'}, execution=SHARED)
    components, source = _construct_model(config)
    assert len(providers) == 1
    provider = providers[0]
    with components.managed():
        assert provider.components is components and not provider.closed
        assert source['reference'] == reference
        assert components.construction_receipt['roles'] == 3
        assert components.construction_receipt['aliased_parameters'] == 8
        assert not provider.pool.parameters
    assert provider.closed and provider.components is None
    assert components.state['student'].hidden.weight is components.state['teacher'].hidden.weight


def test_different_checkpoint_is_not_aliased_and_reference_is_unchanged():
    p = deepcopy(PARAMETERS); p['different_fake'] = True
    provider, components = construct(p)
    state = components.state
    assert state['student'].hidden.weight is not state['fake'].hidden.weight
    assert state['student'].hidden.weight is state['teacher'].hidden.weight
    provider.close()
    reference = factory(PARAMETERS).state
    assert reference['student'].hidden.weight is not reference['fake'].hidden.weight


def test_identical_layers_in_different_checkpoints_reuse_exact_values():
    parameters = {**deepcopy(PARAMETERS), 'partial_fake': True}
    provider, components = construct(parameters)
    student, fake = (components.state[name] for name in ('student', 'fake'))
    assert student.hidden.weight is fake.hidden.weight
    assert student.proj_out.base.weight is not fake.proj_out.base.weight
    reference = factory(parameters)
    assert equal(fake.state_dict(), reference.state['fake'].state_dict())
    provider.close()


def test_updates_optimizer_ema_layout_and_rng_equal_the_unshared_reference():
    before = torch.get_rng_state().clone()
    reference = factory(PARAMETERS).state
    provider, candidate = construct()
    assert torch.equal(before, torch.get_rng_state())
    assert equal(reference['student_ema'], candidate.state['student_ema'])
    for state in (reference, candidate.state):
        initial = {name: {key: parameter.detach().clone()
                          for key, parameter in state[name].named_parameters() if parameter.requires_grad}
                   for name in ('student', 'fake')}
        initial_ema = deepcopy(state['student_ema'])
        for _ in range(2):
            value = torch.arange(4, dtype=torch.float32).reshape(1, 4)
            for name in ('student', 'fake'):
                optimizer = state[name + '_optimizer']
                optimizer.zero_grad(set_to_none=True)
                target = state['teacher'](value).detach() + .5
                (state[name](value) - target).square().mean().backward()
                dmd._update(SimpleNamespace(state=state, attributes={
                    'phase': name, 'flow_policy': {'max_grad_norm': 1.}}))
        for name in ('student', 'fake'):
            parameter = state[name].proj_out.lora_b
            assert not torch.equal(parameter, initial[name]['proj_out.lora_b'])
            moments = state[name + '_optimizer'].state[parameter]
            assert torch.count_nonzero(moments['exp_avg']) > 0
            assert torch.count_nonzero(moments['exp_avg_sq']) > 0
        assert set(state['student_ema']) == set(dict(state['student'].named_parameters()))
        assert not equal(initial_ema, state['student_ema'])
    for name in ('student', 'fake', 'teacher', 'student_optimizer', 'fake_optimizer',
                 'student_scheduler', 'fake_scheduler', 'dmd_counters'):
        assert equal(reference[name].state_dict(), candidate.state[name].state_dict())
    assert equal(reference['student_ema'], candidate.state['student_ema'])
    assert torch.equal(before, torch.get_rng_state())
    provider.close()


def test_frozen_subtrees_include_the_backbone_not_only_lora_bases():
    model = Native(); trainable_scope(model, PARAMETERS['trainable'])
    assert frozen_native_subtrees(model, 'student.native') == (
        'student.native.hidden', 'student.native.proj_out.base')


@pytest.mark.parametrize('execution', [
    {'mode': 'reference'}, {'mode': 'auto'},
    {'mode': 'manual', 'implementation': {**SHARED['implementation'], 'share_immutable': False}},
])
def test_other_strategies_keep_the_original_construction_path(execution):
    assert admit_frozen_sharing(parameters=PARAMETERS, request=SimpleNamespace(execution=execution),
                                budget=None, factory=factory) is None


def test_replaced_parameters_fail_closed_and_the_scope_is_released():
    previous = torch.__future__.get_overwrite_module_params_on_conversion()
    try:
        torch.__future__.set_overwrite_module_params_on_conversion(True)
        with pytest.raises(ValueError, match='placement changed a shared frozen Parameter'):
            construct()
    finally:
        torch.__future__.set_overwrite_module_params_on_conversion(previous)
    plain = factory(PARAMETERS).state
    assert plain['student'].hidden.weight is not plain['fake'].hidden.weight


def test_provider_cannot_be_reused_or_substituted():
    provider, components = construct()
    with pytest.raises(ValueError, match='single-use'):
        provider.construct(factory, PARAMETERS)
    provider.close()
    assert components.state['student'].hidden.weight is components.state['teacher'].hidden.weight


def test_reference_placement_keeps_the_native_error_callback():
    native = Native()
    native.unregistered = torch.ones(1, requires_grad=True)

    def require(condition, message):
        if not condition:
            raise ValueError('native-placement: ' + message)

    with pytest.raises(ValueError, match='native-placement: unregistered native trainable tensor'):
        place_frozen_role(native, torch.device('cpu'), identity={}, require=require)
