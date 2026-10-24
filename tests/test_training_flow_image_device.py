"""Native image placement gates, without optional models or a CUDA context."""
from copy import deepcopy

import pytest
import torch


CUDA_MANUAL = dict(mode='manual', implementation=dict(backend='torch-callable-resident-v1',
                                                     share_immutable=True, checkpoint_bindings=[]))
CUDA_UNSUPPORTED = [
    {'mode': 'auto'}, {'mode': 'manual', 'implementation': {}},
    {'mode': 'plan', 'plan': 'missing-plan.json'}, {'mode': 'calibrate'},
    {'mode': 'reference', 'order': []}, {'mode': 'reference', 'implementation': {}},
    {'mode': 'reference', 'plan': 'missing-plan.json'},
    *[dict(CUDA_MANUAL, **{key: value}) for key, value in (
        ('order', []), ('order', ['original']), ('plan', 'missing-plan.json'),
        ('adapter', 'native'), ('max_candidates', 1), ('memory_limit_bytes', 1))],
    *[dict(CUDA_MANUAL, implementation=dict(CUDA_MANUAL['implementation'], **{key: value}))
      for key, value in (('share_immutable', False), ('share_immutable', 1),
                        ('backend', 'other'), ('checkpoint_bindings', ['student_forward']),
                        ('checkpoint_bindings', ()), ('checkpoint_policy', 'recompute'),
                        ('owner_execution', {}))],
    *[dict(CUDA_MANUAL, implementation={key: value for key, value in CUDA_MANUAL['implementation'].items()
                                      if key != missing}) for missing in CUDA_MANUAL['implementation']],
]


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
@pytest.mark.parametrize('execution', CUDA_UNSUPPORTED)
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
def test_single_cuda_reference_or_exact_manual_catalog_and_invalid_device_refusal(family):
    from distillschedule.training.catalog import describe, validate_compatibility
    factory = describe()['models'][family]['factory']
    algorithm = dict(factory='distillschedule.training.dmd:build_algorithm')
    for device in ('cpu', 'cuda', 'cuda:0', 'cuda:15'):
        for execution in (dict(mode='reference'), CUDA_MANUAL, dict(CUDA_MANUAL, adapter='program')):
            validate_compatibility(dict(factory=factory, parameters={'device': device}), algorithm,
                                   dict(device=device), execution)
    for device in ('cpu:0', 'cuda:-1', 'cuda:bad', 'mps', 'cuda:0:1'):
        with pytest.raises(ValueError, match='incompatible'):
            validate_compatibility(dict(factory=factory, parameters={'device': device}), algorithm,
                                   dict(device=device), dict(mode='reference'))
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('family', ['ltx25', 'flux2-klein-base', 'qwen-image', 'wan22-ti2v5b-t2v', 'hunyuan15-t2v'])
def test_exact_manual_public_config_and_replaced_request_gate(family, tmp_path, monkeypatch):
    from dataclasses import replace
    from distillschedule.training.config import TrainingConfig
    from distillschedule.training import session
    value = dict(schema_version='distillschedule.training.v1', model=dict(name=family, parameters={}),
                 algorithm=dict(name='dmd'), data=dict(prepared='MISSING'), execution=CUDA_MANUAL,
                 trainer=dict(device='cuda:0', cycles=1, output_dir=str(tmp_path / 'output')))
    config = TrainingConfig.from_dict(value, source=tmp_path / 'request.json')
    def forbidden(*args, **kwargs):
        raise AssertionError('data, model or RNG preceded policy admission')
    for name in ('training_data', '_seed', '_construct_model'):
        monkeypatch.setattr(session, name, forbidden)
    for execution in CUDA_UNSUPPORTED:
        with pytest.raises(ValueError, match='CUDA execution requires'):
            session.construct(replace(config, execution=execution))
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('module', ['ltx', 'flux', 'qwen_image', 'wan22', 'hunyuan_video'])
def test_adapter_admission_uses_same_exact_manual_policy(module, monkeypatch):
    # Admission-only witnesses: no synthetic stand-in model is executed here.
    import importlib
    from types import SimpleNamespace
    api = importlib.import_module('distillschedule.training.' + module)
    owner = importlib.import_module('distillschedule.training.flow_image') if module in ('flux', 'qwen_image') else api
    monkeypatch.setattr(owner, '_admit_device' if module == 'ltx' else 'admit_fp32',
                        lambda *args, **kwargs: torch.device('cuda:0'))
    schedule = [1., .75, .5, .25, 0.]
    state = {name: SimpleNamespace(device=torch.device('cuda:0'), binding='test', generation_sigmas=schedule)
             for name in ('student', 'fake', 'teacher', 'media')}
    request = SimpleNamespace(trainer=dict(device='cuda:0'), execution=CUDA_MANUAL, data={},
        algorithm=dict(factory='distillschedule.training.dmd:build_algorithm',
                       parameters=dict(teacher_cfg=1, fake_cfg=1, fake_updates=5, rollout_sigmas=schedule)))
    if module == 'ltx':
        from distillschedule.data.ltx_inputs import SCHEMA
        metadata = dict(schema_version=SCHEMA)
    else:
        metadata = dict(schema_version=owner.INPUT_SCHEMA, family=api.FAMILY, binding='test')
    api.admit_training(state=state, request=request, prepared_identity='test', prepared_manifest=dict(metadata=metadata))
    for execution in CUDA_UNSUPPORTED:
        request.execution = execution
        with pytest.raises(ValueError, match='CUDA.*trial.*snapshot'):
            api.admit_training(state=state, request=request, prepared_identity='test', prepared_manifest=dict(metadata=metadata))
    assert not torch.cuda.is_initialized()
