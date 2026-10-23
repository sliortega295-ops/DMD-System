"""Actual original tiny owners: placement, isolated RNG and trial boundaries.

CUDA guard simulations below are admission tests, never CUDA execution proof.
The optional real CUDA lifecycle lives in tests/gpu/.
"""
from copy import deepcopy

import pytest
import torch

from distillschedule.training.checkpoint import cpu_tree, rng_state
from distillschedule.training.sd import build_sd15, build_sdxl
from test_training_dmd2_cpu import model_and_feeds, source_root
from test_training_dmd2_owner_execution import native_assets
from test_training_end_to_end import equal


def test_actual_factory_keeps_outer_rng_and_never_seeds_unowned_cuda(native_assets, monkeypatch):
    config, components, _ = model_and_feeds(native_assets)
    parameters = dict(config.model['parameters'], seed=39, device='cpu')
    factory = build_sdxl if components.state['native_cycle'].args.sdxl else build_sd15
    before = rng_state()
    def forbidden(*args, **kwargs):
        raise AssertionError('factory touched CUDA RNG outside its owned device')
    monkeypatch.setattr(torch.cuda, 'manual_seed_all', forbidden)
    first = factory(parameters).state['native_cycle']
    second = factory(parameters).state['native_cycle']
    assert equal(before, rng_state()), 'construction must not advance the caller RNG'
    assert equal(first.state_dict(), second.state_dict())
    assert first.device == torch.device('cpu')


def test_actual_factory_failure_restores_outer_rng_after_native_placement(native_assets, monkeypatch):
    import random
    import numpy as np
    from distillschedule.training import sd
    config, components, _ = model_and_feeds(native_assets)
    parameters = dict(config.model['parameters'], seed=89, device='cpu')
    factory = build_sdxl if components.state['native_cycle'].args.sdxl else build_sd15
    place = sd._place_native_tensors
    def fail_after_placement(model, device):
        place(model, device)
        random.random(); np.random.rand(); torch.randn(3)
        raise RuntimeError('injected after actual native placement')
    monkeypatch.setattr(sd, '_place_native_tensors', fail_after_placement)
    before = rng_state()
    with pytest.raises(RuntimeError, match='injected after actual native placement'):
        factory(parameters)
    assert equal(before, rng_state())


def test_explicit_cpu_rng_capture_and_seed_do_not_inspect_cuda(monkeypatch):
    from distillschedule.training.session import _seed
    def forbidden(*args, **kwargs):
        raise AssertionError('CPU RNG requested an unrelated CUDA generator')
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: True)
    monkeypatch.setattr(torch.cuda, 'get_rng_state', forbidden)
    monkeypatch.setattr(torch.cuda, 'manual_seed_all', forbidden)
    _seed(73, device='cpu')
    value = rng_state(device='cpu')
    assert 'cuda' not in value
    assert torch.equal(value['torch'], torch.Generator().manual_seed(73).get_state())


def test_actual_owner_validates_feed_device_before_reading_tensor_values(native_assets):
    _, components, feeds = model_and_feeds(native_assets)
    owner = components.state['native_cycle']
    before = cpu_tree(owner.state_dict())
    bad = deepcopy(feeds)
    bad['generator'][0]['text_input_ids_one'] = bad['generator'][0]['text_input_ids_one'].to('meta')
    with pytest.raises(ValueError, match='device|placement'):
        owner.run(bad, cycle=0)
    assert equal(before, cpu_tree(owner.state_dict()))


def test_actual_owner_rejects_plain_tensor_device_drift_before_hashing(native_assets):
    _, components, _ = model_and_feeds(native_assets)
    owner = components.state['native_cycle']
    original = owner.model.denoising_step_list
    owner.model.denoising_step_list = original.to('meta')
    try:
        with pytest.raises(ValueError, match='device|placement'):
            owner.state_dict()
    finally:
        owner.model.denoising_step_list = original
    owner._validate_live()


def test_cuda_owner_copy_guard_is_preallocation_and_nonmutating_admission_simulation(native_assets):
    _, components, _ = model_and_feeds(native_assets)
    owner = components.state['native_cycle']
    before = cpu_tree(owner.state_dict())
    # Deliberately simulate only the placement guard on a CPU actual owner.
    # No CUDA tensor, context, model construction or native update occurs.
    owner.device = torch.device('cuda:0')
    try:
        for mode, implementation in (('auto', {}), ('plan', {}),
                                     ('manual', {'owner_execution': {'native_cycle': {}}})):
            with pytest.raises(ValueError, match='CUDA.*trial.*snapshot'):
                owner.admit_execution(mode, implementation)
        with pytest.raises(ValueError, match='CUDA.*trial.*snapshot|CUDA.*snapshot.*trial'):
            deepcopy(owner)
    finally:
        owner.device = torch.device('cpu')
    assert equal(before, cpu_tree(owner.state_dict()))


def test_actual_owner_host_checkpoint_restores_moments_and_parameter_ownership(native_assets):
    config, components, feeds = model_and_feeds(native_assets)
    owner = components.state['native_cycle']; owner.run(feeds, cycle=0)
    saved = cpu_tree(owner.state_dict())
    from distillschedule.training.session import construct
    restored = construct(config)[0].state['native_cycle']
    restored.load_state_dict(saved)
    assert restored.model.alphas_cumprod is restored.model.guidance_model.alphas_cumprod
    assert restored.model.noise_scheduler.alphas_cumprod is restored.model.guidance_model.alphas_cumprod
    for namespace in (restored.model, restored.model.guidance_model, restored.model.noise_scheduler):
        assert all(value.device == restored.device for value in vars(namespace).values()
                   if isinstance(value, torch.Tensor))
    assert all(t.device == restored.device for t in restored.runtime_tensors())
    names = {id(parameter): name for name, parameter in restored.model.named_parameters()}
    for role, optimizer in restored.optimizers.items():
        assert [[names[id(p)] for p in group['params']] for group in optimizer.param_groups] == restored.ownership[role]
        for parameter, slots in optimizer.state.items():
            assert slots['step'].device.type == 'cpu'
            assert slots['exp_avg'].device == slots['exp_avg_sq'].device == parameter.device
    assert equal(saved, cpu_tree(restored.state_dict()))


@pytest.mark.parametrize('device', ('cpu:0', 'meta', 'mps', 'not-a-device'))
def test_non_single_device_placement_is_rejected(device):
    from distillschedule.training.sd import _execution_device
    with pytest.raises(ValueError, match='device'):
        _execution_device(device)


def test_single_cuda_context_selection_is_scoped_admission_simulation(monkeypatch):
    from contextlib import contextmanager
    from types import SimpleNamespace
    from distillschedule.training.distributed import context
    events = []
    @contextmanager
    def selected(device):
        events.append(('enter', str(device)))
        try:
            yield
        finally:
            events.append(('exit', str(device)))
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device', selected)
    monkeypatch.setenv('WORLD_SIZE', '1')
    with pytest.raises(RuntimeError, match='injected'):
        with context(SimpleNamespace(trainer={'device': 'cuda:2'})):
            assert events == [('enter', 'cuda:2')]
            raise RuntimeError('injected')
    assert events == [('enter', 'cuda:2'), ('exit', 'cuda:2')]
