"""Native video device contracts without optional upstream models or CUDA."""
from copy import deepcopy
from dataclasses import replace
import importlib

import pytest
import torch
from test_training_flow_image_device import CUDA_MANUAL, CUDA_UNSUPPORTED


FAMILIES = [('wan22-ti2v5b-t2v', 'wan22'), ('hunyuan15-t2v', 'hunyuan_video')]


@pytest.mark.parametrize('execution_mode', ['reference', 'manual'])
def test_native_video_acceptance_writer_uses_execution_not_input_mode(tmp_path, execution_mode):
    """Execute the real test receipt writer without importing CUDA fixtures.

    This is metadata-only SYNTHETIC_TEST_ONLY evidence, not a GPU qualification.
    A completed raw-input loop must not overwrite the execution-policy identity.
    """
    import ast
    import json
    from pathlib import Path
    from types import SimpleNamespace

    path = Path(__file__).parent / 'gpu/test_training_flow_video_lifecycle.py'
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == 'test_native_cuda_video_new_process_resume_export_new_inputs_and_cpu_relocation')
    writer = function.body[-1]
    assert 'acceptance.json' in ast.unparse(writer)
    namespace = dict(json=json, tmp_path=tmp_path, native_assets={'module': 'CPU_WRITER_FIXTURE'},
                     config=SimpleNamespace(execution={'mode': execution_mode}),
                     mode='raw', input_mode='raw', execution_mode=execution_mode,
                     rows=[{'sample_id': 'SYNTHETIC_TEST_ONLY'}], whole={'final_versions': {}})
    exec(compile(ast.Module(body=[writer], type_ignores=[]), str(path), 'exec'), namespace)
    receipt = json.loads((tmp_path / 'acceptance.json').read_text())
    assert receipt['evidence_kind'] == 'SYNTHETIC_TEST_ONLY'
    assert receipt['execution_mode'] == execution_mode
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('family,module', FAMILIES)
@pytest.mark.parametrize('device_field', ['model', 'trainer'])
@pytest.mark.parametrize('execution', CUDA_UNSUPPORTED)
def test_cuda_trial_refusal_is_metadata_only(family, module, device_field, execution):
    from distillschedule.training.catalog import describe, validate_compatibility
    model = dict(factory=describe()['models'][family]['factory'], parameters={})
    trainer = dict(device='cpu')
    (model['parameters'] if device_field == 'model' else trainer)['device'] = 'cuda:0'
    with pytest.raises(ValueError, match='CUDA execution requires'):
        validate_compatibility(model, dict(factory='distillschedule.training.dmd:build_algorithm'), trainer, execution)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('family,module', FAMILIES)
@pytest.mark.parametrize('device_field', ['model', 'trainer'])
@pytest.mark.parametrize('execution', [
    {'mode': 'auto'}, {'mode': 'manual', 'implementation': {}}, {'mode': 'plan', 'plan': 'MISSING'},
])
def test_public_construct_refuses_trials_before_any_model_data_or_rng(family, module, device_field, execution, tmp_path, monkeypatch):
    from distillschedule.training.config import TrainingConfig
    from distillschedule.training import session
    value = dict(schema_version='distillschedule.training.v1', model=dict(name=family, parameters={'assets': 'MISSING'}),
                 algorithm=dict(name='dmd'), data=dict(prepared='MISSING'), execution=dict(mode='reference'),
                 trainer=dict(cycles=1, output_dir=str(tmp_path / 'output'), device='cpu'))
    cpu = TrainingConfig.from_dict(value, source=tmp_path / 'request.json')
    def forbidden(*args, **kwargs):
        raise AssertionError('data, RNG or native construction preceded policy admission')
    for name in ('training_data', '_seed', '_construct_model'):
        monkeypatch.setattr(session, name, forbidden)
    value['execution'] = execution
    (value['model']['parameters'] if device_field == 'model' else value['trainer'])['device'] = 'cuda:0'
    with pytest.raises(ValueError, match='CUDA execution requires'):
        TrainingConfig.from_dict(value, source=tmp_path / 'request.json')
    with pytest.raises(ValueError, match='CUDA execution requires'):
        session.construct(replace(cpu, execution=execution, trainer=value['trainer'],
                                  model={**cpu.model, 'parameters': value['model']['parameters']}))
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('family,module', FAMILIES)
def test_catalog_admits_only_single_cpu_cuda_reference_or_exact_manual(family, module):
    from distillschedule.training.catalog import describe, validate_compatibility
    entry = describe()['models'][family]
    algorithm = dict(factory='distillschedule.training.dmd:build_algorithm')
    assert entry['validation']['gpu'] == entry['validation']['quality'] == 'NOT_RUN'
    for device in ('cpu', 'cuda', 'cuda:0', 'cuda:15'):
        for execution in (dict(mode='reference'), CUDA_MANUAL):
            validate_compatibility(dict(factory=entry['factory'], parameters={'device': device}),
                                   algorithm, dict(device=device), execution)
    for device in ('cpu:0', 'cuda:-1', 'cuda:bad', 'mps', 'cuda:0:1'):
        with pytest.raises(ValueError, match='incompatible'):
            validate_compatibility(dict(factory=entry['factory'], parameters={'device': device}),
                                   algorithm, dict(device=device), dict(mode='reference'))
    assert not torch.cuda.is_initialized()


def _row(module):
    shape = [1, 4, 2, 2, 2]
    width = 2
    length = 226 if module == 'wan22' else 1000
    mask = torch.zeros(1, length, dtype=torch.bool); mask[:, :1] = True
    context = torch.zeros(1, length, width); context[:, :1] = 1.
    row = dict(latents={'video': torch.zeros(shape)}, latent_shape=torch.tensor(shape),
               context=context, context_mask=mask, null_context=context.clone(), null_context_mask=mask.clone(),
               encoder_identity=torch.tensor([0] * 32, dtype=torch.uint8))
    empty = (row['null_context'], row['null_context_mask'])
    if module == 'wan22':
        config = dict(in_channels=4, text_dim=width, rope_max_seq_len=16)
    else:
        config = dict(out_channels=4, text_embed_dim=width, text_embed_2_dim=width)
        row.update(context_2=torch.zeros(1, 256, width), context_mask_2=torch.zeros(1, 256, dtype=torch.bool),
                   null_context_2=torch.zeros(1, 256, width), null_context_mask_2=torch.zeros(1, 256, dtype=torch.bool))
        empty += (row['null_context_2'], row['null_context_mask_2'])
    return row, config, empty


@pytest.mark.parametrize('family,module', FAMILIES)
def test_prepared_feature_device_contract_and_owned_native_tensors(family, module):
    api = importlib.import_module('distillschedule.training.' + module)
    inputs = importlib.import_module('distillschedule.data.' + module + '_inputs')
    from distillschedule.training.single_device import place_module, validate_placement
    row, config, empty = _row(module)
    role = api.Role(torch.nn.Linear(2, 2), config=config, media_config={}, binding='00' * 32,
                    empty=empty, trainable={}, schedule=[1., .75, .5, .25, 0.])
    assert role.device == torch.device('cpu')
    # Native complex/integral buffers retain dtype and exact values; parameter
    # identity is not replaced after optimizer ownership has been established.
    role.native.frequency = torch.polar(torch.ones(2), torch.tensor([.2, .9]))
    role.native.register_buffer('positions', torch.arange(2))
    before = role.native.frequency.clone(); owner = id(role.native.weight)
    place_module(role, role.device)
    validate_placement(role, role.device)
    assert id(role.native.weight) == owner and torch.equal(role.native.frequency, before)
    assert role.native.frequency.is_complex() and role.native.positions.dtype == torch.int64
    assert role.validate_feeds(row) is row
    assert inputs.validate_row(row, config=config, media_config={}, binding='00' * 32, empty=empty,
                               device=torch.device('cpu')) is row
    with pytest.raises(ValueError, match='device'):
        inputs.validate_row(row, config=config, media_config={}, binding='00' * 32, empty=empty, device='cuda:0')
    for key in ('latent_shape', 'encoder_identity', 'context_mask'):
        mixed = dict(row, **{key: row[key].to('meta')})
        with pytest.raises(ValueError, match='device'):
            role.validate_feeds(mixed)
    stale = deepcopy(row); stale['encoder_identity'][0] ^= 1
    with pytest.raises(ValueError, match='binding'):
        role.validate_feeds(stale)
    wrong_null = deepcopy(row); wrong_null['null_context'][0, 0, 0] += 1
    with pytest.raises(ValueError, match='empty-prompt'):
        role.validate_feeds(wrong_null)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('family,module', FAMILIES)
@pytest.mark.parametrize('stage', ['training', 'inference'])
def test_unsupported_runtime_rejected_before_native_import_or_assets(family, module, stage, monkeypatch):
    api = importlib.import_module('distillschedule.training.' + module)
    parameters = dict(assets='MISSING', device='cpu', trainable={'mode': 'lora'}, student_lr=.001,
                      fake_lr=.001, scheduler_gamma=.99, ema_decay=.9, generation_sigmas=[1., .75, .5, .25, 0.])
    build = api.build_model
    if stage == 'inference':
        parameters = {key: {} for key in api._DESCRIPTOR}
        build = api.build_inference_model
    # The source must reject a known-invalid owner without touching assets or
    # optional upstream packages, regardless of runtime availability.
    for key, value in (('WORLD_SIZE', '2'), ('LOCAL_WORLD_SIZE', '2'), ('RANK', '1'), ('LOCAL_RANK', '1')):
        with monkeypatch.context() as patch:
            patch.setenv(key, value)
            with pytest.raises(ValueError, match='one actual process/rank|initialized torchrun'):
                build(parameters)
    with monkeypatch.context() as patch:
        patch.setattr(torch.distributed, 'is_initialized', lambda: True)
        patch.setattr(torch.distributed, 'get_world_size', lambda: 1)
        with pytest.raises(ValueError, match='distributed|initialized torchrun'):
            build(parameters)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        with pytest.raises(ValueError, match='FP32'):
            build(parameters)
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        with pytest.raises(ValueError, match='FP32'):
            build(parameters)
    finally:
        torch.set_default_dtype(previous)
    with monkeypatch.context() as patch:
        patch.setattr(torch.cuda, 'is_available', lambda: False)
        with pytest.raises(ValueError, match='CUDA device is unavailable'):
            build(dict(parameters, device='cuda:0'))
    with pytest.raises(ValueError):
        build(dict(parameters, seed=True))
    assert not torch.cuda.is_initialized()
