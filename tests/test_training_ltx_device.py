"""LTX single-device admission tests do not require upstream assets or CUDA."""
from copy import deepcopy

import pytest
import torch


def _parameters():
    return dict(source_root='must-not-load', assets='must-not-load', device='cpu', seed=19,
        trainable=dict(mode='full'), student_lr=1e-4, fake_lr=2e-4,
        scheduler_gamma=.98, ema_decay=.9, generation_sigmas=[1., .75, .5, .25, 0.])


def _forbid_source(monkeypatch):
    import distillschedule.training.ltx as ltx
    def forbidden(*args, **kwargs):
        raise AssertionError('native source/model allocation reached before admission')
    monkeypatch.setattr(ltx, 'original', forbidden)
    return ltx


@pytest.mark.parametrize('device_field', ['trainer', 'model'])
@pytest.mark.parametrize('execution', [
    {'mode': 'auto'}, {'mode': 'manual', 'implementation': {}},
    {'mode': 'plan', 'plan': 'must-not-load.json'},
])
def test_ltx_public_request_refuses_cuda_trials_before_any_ownership(
        monkeypatch, tmp_path, device_field, execution):
    from distillschedule.training.config import TrainingConfig
    from distillschedule.training.session import construct
    import distillschedule.training.session as session
    _forbid_source(monkeypatch)
    def forbidden(*args, **kwargs):
        raise AssertionError('data, RNG, model, or device touched before policy admission')
    for name in ('training_data', '_seed', '_construct_model'):
        monkeypatch.setattr(session, name, forbidden)
    request = dict(schema_version='distillschedule.training.v1',
        model={'name': 'ltx25', 'parameters': _parameters()},
        data={'prepared': 'must-not-load.json'}, algorithm={'name': 'dmd'},
        execution={'mode': 'reference'}, trainer={'cycles': 1, 'seed': 19,
            'device': 'cpu', 'output_dir': str(tmp_path / 'output')})
    cpu = TrainingConfig.from_dict(request, source=tmp_path / 'request.json')
    request['execution'] = execution
    target = request['trainer'] if device_field == 'trainer' else request['model']['parameters']
    target['device'] = 'cuda:0'
    with pytest.raises(ValueError, match='CUDA execution requires.*reference'):
        TrainingConfig.from_dict(request, source=tmp_path / 'request.json')
    # Direct dataclass callers and programmatically replaced requests must not
    # bypass the same metadata gate at the actual public construction boundary.
    from dataclasses import replace
    changed = replace(cpu, execution=execution,
        trainer=request['trainer'], model={**cpu.model, 'parameters': request['model']['parameters']})
    with pytest.raises(ValueError, match='CUDA execution requires.*reference'):
        construct(changed)


def test_ltx_execution_constraints_are_declarative_and_cpu_discovery_stays_open(tmp_path):
    from distillschedule.training.catalog import resolve, validate_compatibility
    declaration = resolve('model', 'ltx25')
    assert declaration['execution_constraints']['cuda']['modes'] == ['reference']
    for mode in ('reference', 'auto', 'manual', 'plan'):
        validate_compatibility({'factory': declaration['factory']}, {}, {'device': 'cpu'}, {'mode': mode})
    validate_compatibility({'factory': declaration['factory']}, {}, {'device': 'cuda:0'}, {'mode': 'reference'})
    # Plugin factories without an execution declaration retain their own owner
    # admission; a generic planner must not branch on LTX's model name.
    validate_compatibility({'factory': 'external.adapter:build'}, {}, {'device': 'cuda:0'}, {'mode': 'auto'})


@pytest.mark.parametrize('key,value', [('WORLD_SIZE', '2'), ('LOCAL_WORLD_SIZE', '2'),
                                      ('RANK', '1'), ('LOCAL_RANK', '1')])
def test_ltx_rank_environment_rejected_before_source_or_allocation(monkeypatch, key, value):
    ltx = _forbid_source(monkeypatch)
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError, match='single|one actual process|rank|distributed'):
        ltx.build_model(_parameters())


def test_ltx_active_distributed_owner_rejected_before_source_or_allocation(monkeypatch):
    ltx = _forbid_source(monkeypatch)
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)
    with pytest.raises(ValueError, match='distributed'):
        ltx.build_model(_parameters())


def test_ltx_autocast_rejected_before_source_or_allocation(monkeypatch):
    ltx = _forbid_source(monkeypatch)
    with torch.autocast('cpu', dtype=torch.bfloat16), pytest.raises(ValueError, match='FP32|autocast'):
        ltx.build_model(_parameters())


def test_ltx_default_precision_rejected_before_source_or_allocation(monkeypatch):
    ltx = _forbid_source(monkeypatch)
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        with pytest.raises(ValueError, match='FP32'):
            ltx.build_model(_parameters())
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize('device', ['meta', 'cpu:0', 'not-a-device'])
def test_ltx_invalid_device_rejected_before_source_or_allocation(monkeypatch, device):
    ltx = _forbid_source(monkeypatch)
    with pytest.raises(ValueError, match='device'):
        ltx.build_model(dict(_parameters(), device=device))


def test_ltx_unavailable_cuda_rejected_before_source_or_allocation(monkeypatch):
    ltx = _forbid_source(monkeypatch)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(ValueError, match='CUDA device is unavailable'):
        ltx.build_model(dict(_parameters(), device='cuda:0'))


def test_ltx_inference_autocast_rejected_before_source_or_allocation(monkeypatch):
    ltx = _forbid_source(monkeypatch)
    parameters = dict(source_root='must-not-load', architecture={}, media_architecture={},
        binding='00' * 32, trainable=dict(mode='full'), generation_sigmas=[1., 0.])
    with torch.autocast('cpu', dtype=torch.bfloat16), pytest.raises(ValueError, match='FP32|autocast'):
        ltx.build_inference_model(parameters)


def test_ltx_cuda_clone_refused_before_traversing_native_state():
    from distillschedule.training.ltx import LTXRole
    # An empty shell is only a guard witness: no substituted model is executed.
    # The explicitly declared device must be checked before visiting any native
    # child or attempting a whole-role allocation.
    role = LTXRole.__new__(LTXRole)
    torch.nn.Module.__init__(role)
    role.device = torch.device('cuda:0')
    with pytest.raises(ValueError, match='CUDA.*trial.*snapshot'):
        deepcopy(role)


def test_ltx_plain_tensor_ownership_checkpoint_and_negative_placement():
    from distillschedule.training.ltx_media import place_module, validate_placement
    module = torch.nn.Linear(2, 2)
    module.plain_frequency = torch.arange(2, dtype=torch.float32)
    module._private_frequency = torch.arange(3, dtype=torch.float32)
    place_module(module, torch.device('cpu'))
    assert {'plain_frequency', '_private_frequency'} <= module.state_dict().keys()
    assert 'plain_frequency' not in vars(module) and '_private_frequency' not in vars(module)
    restored = torch.nn.Linear(2, 2)
    restored.plain_frequency = torch.zeros(2)
    restored._private_frequency = torch.zeros(3)
    place_module(restored, torch.device('cpu'))
    restored.load_state_dict(deepcopy(module.state_dict()), strict=True)
    assert torch.equal(restored.plain_frequency, module.plain_frequency)
    with pytest.raises(ValueError, match='device'):
        validate_placement(module, torch.device('cuda:0'))
    module.undeclared = torch.ones(1)
    with pytest.raises(ValueError, match='unowned'):
        validate_placement(module, torch.device('cpu'))


def test_ltx_unregistered_tensor_container_is_not_silently_left_on_cpu():
    from distillschedule.training.ltx_media import place_module
    module = torch.nn.Linear(2, 2)
    module.hidden_cache = {'value': torch.zeros(1)}
    with pytest.raises(ValueError, match='explicit placement adapter'):
        place_module(module, torch.device('cpu'))


@pytest.mark.parametrize('fault', ['shape', 'dtype', 'missing', 'nonfinite'])
def test_ltx_resident_ema_restore_preserves_live_tensor_ownership(fault):
    from distillschedule.training.ltx import ResidentEMA
    from distillschedule.training.checkpoint import pack_state, unpack_state
    ema = ResidentEMA({'weight': torch.arange(4, dtype=torch.float32), 'second': torch.ones(2)})
    state = {'student_ema': ema}
    saved = pack_state(state)
    assert saved['student_ema']['kind'] == 'stateful'
    original = ema['weight']; original.add_(10.)
    unpack_state(saved, state)
    assert state['student_ema'] is ema and ema['weight'] is original
    assert torch.equal(ema['weight'], torch.arange(4, dtype=torch.float32))
    bad = deepcopy(saved)
    bad['student_ema']['value']['weight'].add_(100.)  # A valid earlier entry must not be partially applied.
    if fault == 'shape':
        bad['student_ema']['value']['second'] = torch.zeros(5)
    elif fault == 'dtype':
        bad['student_ema']['value']['second'] = torch.zeros(2, dtype=torch.float64)
    elif fault == 'missing':
        del bad['student_ema']['value']['second']
    else:
        bad['student_ema']['value']['second'][1] = float('nan')
    with pytest.raises(ValueError, match='shape|dtype|keys|finite'):
        unpack_state(bad, state)
    assert torch.equal(ema['weight'], torch.arange(4, dtype=torch.float32))
