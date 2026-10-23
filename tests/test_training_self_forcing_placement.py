"""Actual pinned native owners: resident placement and host-state exceptions.

CUDA simulations below test admission only; actual optional CUDA updates live
in tests/gpu/. No simulation is a GPU execution or capacity receipt.
"""
from copy import deepcopy

import pytest
import torch

from distillschedule.training.checkpoint import cpu_tree, rng_state
from distillschedule.training.self_forcing import build_model, build_self_forcing
from distillschedule.training.self_forcing_owner import plain_tensors
from test_training_end_to_end import equal
from test_training_self_forcing_cpu import fixture_assets, model_and_data, source_root


def test_actual_factory_preserves_caller_rng_without_touching_cuda(fixture_assets, monkeypatch):
    config, _, data = model_and_data(fixture_assets)
    before = rng_state(device='cpu')

    def forbidden(*args, **kwargs):
        raise AssertionError('CPU factory touched an unowned CUDA context or RNG')

    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: True)
    for name in ('manual_seed_all', 'manual_seed', 'get_rng_state', 'current_device', '_lazy_init'):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    parameters = dict(config.model['parameters'], seed=39, device='cpu')
    first = build_model(parameters).state['native_cycle']
    second = build_model(parameters).state['native_cycle']
    assert equal(before, rng_state(device='cpu'))
    assert equal(cpu_tree(first.state_dict()), cpu_tree(second.state_dict()))
    first.run(data.cycle_sample(0), cycle=0)
    assert equal(before, rng_state(device='cpu'))
    assert 'cuda' not in first._rng


def test_native_plain_inventory_dtypes_alias_optimizer_and_host_ema(fixture_assets):
    _, components, data = model_and_data(fixture_assets)
    owner = components.state['native_cycle']
    expected = {'denoising_step_list': torch.float32, 'vae.mean': torch.float32, 'vae.std': torch.float32}
    for role in ('generator', 'real_score', 'fake_score'):
        expected[role + '.model.freqs'] = torch.complex128
        for name in ('timesteps', 'sigmas', 'linear_timesteps_weights'):
            expected[role + '.scheduler.' + name] = torch.float32
    # Native conversion promotes schedule arithmetic to FP64 transiently; its
    # persistent schedule stays in the original set_timesteps dtype.
    assert {name: value.dtype for name, value in plain_tensors(owner.model).items()} == expected
    assert owner.model.scheduler is owner.model.generator.scheduler
    assert owner.identity['placement'] == dict(device='cpu', layout='SINGLE_DEVICE_RESIDENT', distributed=False, amp=False)
    owner.run(data.cycle_sample(0), cycle=0)
    assert owner.generator_ema is not None
    assert all(value.device.type == 'cpu' and value.dtype == torch.float32
               for value in owner.generator_ema.shadow.values())
    assert not ({id(value) for value in owner.generator_ema.shadow.values()}
                & {id(value) for value in owner.runtime_tensors()})
    names = {id(p): name for name, p in owner.model.named_parameters()}
    for role, optimizer in owner.optimizers.items():
        assert [[names[id(p)] for p in group['params']] for group in optimizer.param_groups] == owner.ownership[role]
        for parameter, slots in optimizer.state.items():
            assert slots['step'].device.type == 'cpu' and slots['step'].dtype == torch.float32
            assert slots['exp_avg'].device == slots['exp_avg_sq'].device == parameter.device == owner.device
    saved = cpu_tree(owner.state_dict())
    restored = model_and_data(fixture_assets)[1].state['native_cycle']
    restored.load_state_dict(saved)
    assert restored.model.scheduler is restored.model.generator.scheduler
    assert equal(saved, cpu_tree(restored.state_dict()))


def test_public_program_reference_uses_resident_owner_without_trial_clone(fixture_assets, tmp_path, monkeypatch):
    from distillschedule.training.session import build_cycle
    from distillschedule.training.policy import CyclePolicy
    config, components, data = model_and_data(fixture_assets)
    owner = components.state['native_cycle']

    def forbidden(*args, **kwargs):
        raise AssertionError('resident reference path cloned a native owner')

    monkeypatch.setattr(type(owner), '__deepcopy__', forbidden, raising=False)
    workload = build_cycle(config, components, data, build_self_forcing, cycle=0, cursor=0)
    assert workload.device == owner.device
    assert workload.identity_extra['placement'] == owner.identity['placement']
    assert workload.identity_extra['scope'] == 'OFFICIAL_SELF_FORCING_NATIVE_FP32_SINGLE_DEVICE'
    session, outputs = CyclePolicy(config.execution).run(workload, tmp_path / 'reference')
    assert session.state['native_cycle'] is owner and owner.cycle == 1
    assert outputs['self_forcing.native_cycle']['consumed'] == {'generator': 1, 'critic': 5}


def test_cuda_whole_owner_copy_and_trial_guard_is_nonmutating_simulation(fixture_assets):
    _, components, _ = model_and_data(fixture_assets)
    owner = components.state['native_cycle']
    before = cpu_tree(owner.state_dict())
    owner.device = torch.device('cuda:0')  # Admission only, never a CUDA allocation.
    try:
        for mode, implementation in (('auto', {}), ('plan', {}),
                                     ('manual', {'owner_execution': {'native_cycle': {}}})):
            with pytest.raises(ValueError, match='CUDA.*trial.*snapshot'):
                owner.admit_execution(mode, implementation)
        for mode in ('manual', 'reference'):
            owner.admit_execution(mode, {})
        with pytest.raises(ValueError, match='CUDA.*trial.*snapshot'):
            deepcopy(owner)
    finally:
        owner.device = torch.device('cpu')
    assert equal(before, cpu_tree(owner.state_dict()))


@pytest.mark.parametrize('field', ['device', 'dtype', 'alias'])
def test_native_plain_placement_or_alias_drift_rejects_before_hash(fixture_assets, field):
    _, components, _ = model_and_data(fixture_assets)
    owner = components.state['native_cycle']
    original = owner.model.generator.model.freqs
    if field == 'alias':
        owner.model.scheduler = deepcopy(owner.model.scheduler)
    else:
        owner.model.generator.model.freqs = original.to('meta') if field == 'device' else original.to(torch.complex64)
    try:
        with pytest.raises(ValueError, match='device|placement|dtype|alias'):
            owner.state_dict()
    finally:
        owner.model.generator.model.freqs = original
        owner.model.scheduler = owner.model.generator.scheduler
    owner._validate_live()


def test_wrong_feed_device_rejects_without_copy_or_mutation(fixture_assets):
    _, components, data = model_and_data(fixture_assets)
    owner = components.state['native_cycle']; before = cpu_tree(owner.state_dict())
    feeds = data.cycle_sample(0)
    feeds['generator'][0]['prompt_embeds'] = feeds['generator'][0]['prompt_embeds'].to('meta')
    with pytest.raises(ValueError, match='device|placement|CPU'):
        owner.run(feeds, cycle=0)
    assert equal(before, cpu_tree(owner.state_dict()))


def test_factory_failure_after_native_placement_restores_caller_rng(fixture_assets, monkeypatch):
    import random
    import numpy as np
    from distillschedule.training import self_forcing
    config, _, _ = model_and_data(fixture_assets)
    before = rng_state(device='cpu')
    place = self_forcing._place_native_tensors

    def fail(model, device):
        place(model, device)
        random.random(); np.random.rand(); torch.randn(3)
        raise RuntimeError('injected after native placement')

    monkeypatch.setattr(self_forcing, '_place_native_tensors', fail)
    with pytest.raises(RuntimeError, match='injected after native placement'):
        build_model(dict(config.model['parameters'], device='cpu', seed=81))
    assert equal(before, rng_state(device='cpu'))


def test_placement_preserves_explicit_fp64_schedule_and_complex128_rope(fixture_assets):
    from distillschedule.training.self_forcing import _place_native_tensors
    _, components, _ = model_and_data(fixture_assets)
    model = components.state['native_cycle'].model
    for role in ('generator', 'real_score', 'fake_score'):
        scheduler = getattr(model, role).scheduler
        for name in ('sigmas', 'timesteps', 'linear_timesteps_weights'):
            setattr(scheduler, name, getattr(scheduler, name).double())
    model.denoising_step_list = model.denoising_step_list.double()
    before = {name: tensor.clone() for name, tensor in plain_tensors(model).items()}
    _place_native_tensors(model, torch.device('cpu'))
    assert model.scheduler is model.generator.scheduler
    for name, tensor in plain_tensors(model).items():
        assert tensor.dtype == before[name].dtype
        assert torch.equal(tensor, before[name])
    assert model.generator.model.freqs.dtype == torch.complex128


def test_live_native_moment_dtype_and_config_negative_drift_refuse_before_update(fixture_assets):
    _, components, data = model_and_data(fixture_assets)
    owner = components.state['native_cycle']; owner.run(data.cycle_sample(0), cycle=0)
    slots = next(iter(owner.optimizers['critic'].state.values()))
    moment = slots['exp_avg']; slots['exp_avg'] = moment.double()
    try:
        with pytest.raises(ValueError, match='dtype|FP32'):
            owner.state_dict()
    finally:
        slots['exp_avg'] = moment
    negative = owner.config.negative_prompt
    owner.config.negative_prompt = negative + 1
    try:
        with pytest.raises(ValueError, match='negative|condition'):
            owner.state_dict()
    finally:
        owner.config.negative_prompt = negative
    owner._validate_live()


@pytest.mark.parametrize('field', ['placement', 'negative', 'schedule_dtype', 'ema_dtype', 'adam_step_device', 'cuda_rng'])
def test_host_checkpoint_device_and_dtype_rejections_are_transactional(fixture_assets, field):
    _, components, data = model_and_data(fixture_assets)
    owner = components.state['native_cycle']; owner.run(data.cycle_sample(0), cycle=0)
    saved = cpu_tree(owner.state_dict()); bad = deepcopy(saved)
    if field == 'placement':
        bad['identity']['placement']['device'] = 'cuda:0'
    elif field == 'negative':
        bad['negative_embedding'].add_(1)
    elif field == 'schedule_dtype':
        bad['plain']['generator.scheduler.sigmas'] = bad['plain']['generator.scheduler.sigmas'].double()
    elif field == 'ema_dtype':
        key = next(iter(bad['ema'])); bad['ema'][key] = bad['ema'][key].double()
    elif field == 'adam_step_device':
        slots = next(iter(bad['optimizers']['critic']['state'].values()))
        slots['step'] = slots['step'].to('meta')
    elif field == 'cuda_rng':
        bad['rng']['cuda'] = dict(device=0, state=torch.zeros(8, dtype=torch.uint8))
    with pytest.raises(ValueError):
        owner.load_state_dict(bad)
    assert equal(saved, cpu_tree(owner.state_dict()))


@pytest.mark.parametrize('device', ['cpu:0', 'meta', 'mps', 'not-a-device'])
def test_noncanonical_or_unsupported_device_rejects(device):
    from distillschedule.training.self_forcing import _execution_device
    with pytest.raises(ValueError, match='device'):
        _execution_device(device)


def test_single_cuda_device_canonicalization_is_admission_only(monkeypatch):
    from distillschedule.training.self_forcing import _execution_device
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 3)
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 2)
    assert _execution_device('cuda') == _execution_device('cuda:2') == torch.device('cuda:2')
    with pytest.raises(ValueError, match='device index'):
        _execution_device('cuda:3')


def test_unavailable_cuda_is_not_silently_downgraded_or_initialized(monkeypatch):
    from distillschedule.training.self_forcing import _execution_device
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)

    def forbidden(*args, **kwargs):
        raise AssertionError('unavailable CUDA admission tried to initialize a device')

    monkeypatch.setattr(torch.cuda, '_lazy_init', forbidden)
    monkeypatch.setattr(torch.cuda, 'current_device', forbidden)
    with pytest.raises(ValueError, match='CUDA device is unavailable'):
        _execution_device('cuda:0')


@pytest.mark.parametrize('mode', ['default_dtype', 'autocast'])
def test_ambient_non_fp32_compute_is_rejected(fixture_assets, mode):
    from distillschedule.training.self_forcing import _arguments
    config, _, _ = model_and_data(fixture_assets)
    if mode == 'autocast':
        with torch.autocast('cpu', dtype=torch.bfloat16), pytest.raises(ValueError, match='FP32|autocast'):
            _arguments(config.model['parameters'])
    else:
        original = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            with pytest.raises(ValueError, match='FP32'):
                _arguments(config.model['parameters'])
        finally:
            torch.set_default_dtype(original)


@pytest.mark.parametrize('key,value', [('WORLD_SIZE', '2'), ('LOCAL_WORLD_SIZE', '2'), ('RANK', '1'), ('LOCAL_RANK', '1')])
def test_multirank_environment_is_not_a_singleton(fixture_assets, monkeypatch, key, value):
    from distillschedule.training.self_forcing import _arguments
    config, _, _ = model_and_data(fixture_assets)
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError, match='process/rank'):
        _arguments(config.model['parameters'])


def test_runtime_records_tf32_determinism_and_factory_does_not_change_policy(fixture_assets):
    from distillschedule.training.self_forcing import runtime_identity
    config, _, _ = model_and_data(fixture_assets)
    before = runtime_identity()
    owner = build_model(dict(config.model['parameters'], device='cpu')).state['native_cycle']
    assert owner.identity['runtime'] == runtime_identity() == before
    assert before['cpu_policy']['float32_matmul_precision'] == torch.get_float32_matmul_precision()
    assert before['cpu_policy']['deterministic_algorithms'] == torch.are_deterministic_algorithms_enabled()
    assert before['cuda_policy'] == dict(matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32, cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_benchmark=torch.backends.cudnn.benchmark)


@pytest.mark.parametrize('field', ['matmul_tf32', 'cudnn_tf32', 'cudnn_deterministic',
                                  'float32_matmul_precision', 'deterministic_algorithms'])
def test_live_runtime_precision_policy_drift_is_rejected_without_overriding_caller(fixture_assets, field):
    _, components, _ = model_and_data(fixture_assets)
    owner = components.state['native_cycle']; before = cpu_tree(owner.state_dict())
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    cudnn_deterministic = torch.backends.cudnn.deterministic
    precision = torch.get_float32_matmul_precision()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        if field == 'matmul_tf32':
            torch.backends.cuda.matmul.allow_tf32 = not matmul_tf32
        elif field == 'cudnn_tf32':
            torch.backends.cudnn.allow_tf32 = not cudnn_tf32
        elif field == 'cudnn_deterministic':
            torch.backends.cudnn.deterministic = not cudnn_deterministic
        elif field == 'float32_matmul_precision':
            torch.set_float32_matmul_precision('high' if precision == 'highest' else 'highest')
        elif field == 'deterministic_algorithms':
            torch.use_deterministic_algorithms(not deterministic, warn_only=warn_only)
        with pytest.raises(ValueError, match='runtime|precision'):
            owner.state_dict()
        # Validation refuses; it must not rewrite the caller's selected policy.
        if field == 'cudnn_tf32':
            assert torch.backends.cudnn.allow_tf32 is not cudnn_tf32
        elif field == 'matmul_tf32':
            assert torch.backends.cuda.matmul.allow_tf32 is not matmul_tf32
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.set_float32_matmul_precision(precision)
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    assert equal(before, cpu_tree(owner.state_dict()))
