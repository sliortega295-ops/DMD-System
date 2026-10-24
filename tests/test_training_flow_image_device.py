"""Native image placement gates, without optional models or a CUDA context."""
from copy import deepcopy

import pytest
import torch


def test_native_tensor_placement_preserves_complex_rope_and_parameter_owners():
    from distillschedule.training.single_device import place_module, validate_placement
    module = torch.nn.Linear(2, 2)
    module.frequency = torch.polar(torch.ones(2), torch.tensor([.2, .9]))
    before = module.frequency.clone(); owner = id(module.weight)
    place_module(module, torch.device('cpu'))
    assert id(module.weight) == owner
    assert module.frequency.is_complex() and torch.equal(module.frequency, before)
    assert 'frequency' in module.state_dict() and 'frequency' not in vars(module)
    validate_placement(module, torch.device('cpu'))
    with pytest.raises(ValueError, match='device'):
        validate_placement(module, torch.device('cuda:0'))
    module.unowned = {'cache': torch.ones(1)}
    with pytest.raises(ValueError, match='unowned'):
        place_module(module, torch.device('cpu'))
    assert not torch.cuda.is_initialized()


def test_ema_restoration_validates_every_tensor_before_mutation():
    from distillschedule.training.single_device import ResidentEMA
    value = ResidentEMA(a=torch.tensor([1., 2.]), b=torch.tensor([3.]))
    owners = {k: id(t) for k, t in value.items()}
    saved = deepcopy(value.state_dict())
    invalid = dict(a=torch.tensor([4., 5.]), b=torch.tensor([float('nan')]))
    with pytest.raises(ValueError, match='EMA'):
        value.load_state_dict(invalid)
    assert all(torch.equal(saved[k], t) for k, t in value.items())
    value.load_state_dict(dict(a=torch.tensor([4., 5.]), b=torch.tensor([6.])))
    assert {k: id(t) for k, t in value.items()} == owners
    assert list(value.runtime_tensors()) == list(value.values())
    assert value['b'].item() == 6.


@pytest.mark.parametrize('family', ['flux2-klein-base', 'qwen-image'])
@pytest.mark.parametrize('device_field', ['model', 'trainer'])
@pytest.mark.parametrize('execution', [
    {'mode': 'auto'}, {'mode': 'manual', 'implementation': {}},
    {'mode': 'plan', 'plan': 'missing-plan.json'}, {'mode': 'reference', 'order': []},
])
def test_cuda_clone_policies_refused_by_metadata_before_native_import(family, device_field, execution):
    from distillschedule.training.catalog import describe, validate_compatibility
    factory = describe()['models'][family]['factory']
    model = dict(factory=factory, parameters={})
    trainer = dict(device='cpu')
    if device_field == 'model':
        model['parameters']['device'] = 'cuda:0'
    else:
        trainer['device'] = 'cuda:0'
    with pytest.raises(ValueError, match='CUDA execution requires'):
        validate_compatibility(model, dict(factory='distillschedule.training.dmd:build_algorithm'), trainer, execution)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('family', ['flux2-klein-base', 'qwen-image'])
def test_single_cuda_reference_catalog_and_invalid_device_refusal(family):
    from distillschedule.training.catalog import describe, validate_compatibility
    factory = describe()['models'][family]['factory']
    algorithm = dict(factory='distillschedule.training.dmd:build_algorithm')
    for device in ('cpu', 'cuda', 'cuda:0', 'cuda:15'):
        validate_compatibility(dict(factory=factory, parameters={'device': device}), algorithm,
                               dict(device=device), dict(mode='reference'))
    for device in ('cpu:0', 'cuda:-1', 'cuda:bad', 'mps', 'cuda:0:1'):
        with pytest.raises(ValueError, match='incompatible'):
            validate_compatibility(dict(factory=factory, parameters={'device': device}), algorithm,
                                   dict(device=device), dict(mode='reference'))
    assert not torch.cuda.is_initialized()
