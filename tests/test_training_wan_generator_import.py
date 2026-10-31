"""Initialization admission/transaction tests; no upstream/GPU qualification."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from safetensors.torch import save_file

from distillschedule.training import wan_generator_import as api


def pin(path):
    import hashlib
    return dict(path=str(path.resolve()), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + '\n')
    return pin(path)


def request_files(root):
    """Metadata-only fixture, NOT an admitted native Generator architecture."""
    root.mkdir(parents=True)
    base_config = write(root/'config.json', {'example': 'METADATA_ONLY'})
    base_weights = root/'base.safetensors'
    save_file({'weight': torch.ones(2, 2)}, base_weights)
    payload = root/'final_generator_trainables.safetensors'
    save_file({'weight': torch.full((2, 2), .031)}, payload,
              metadata={'component': 'generator', 'purpose': 'cross-placement-final-state-parity'})
    spec = pin(payload)
    receipt = dict(status='COMPLETED_ORIGINAL_WAN_WORK', rank=0, finite=True,
        identity=dict(source_head='a'*40, source_fingerprint='b'*64, assets=dict(model_files={
            'config.json': base_config['sha256'], 'base.safetensors': pin(base_weights)['sha256']})),
        task={'metadata_fixture_only': True}, initial_versions=dict(generator=2, critic=10, ema=10),
        final_versions=dict(generator=4, critic=20, ema=20),
        cycles=[dict(cycle=i, versions=dict(generator=i+2, critic=5*(i+2), ema=5*(i+2))) for i in (1, 2)],
        numerical_comparison='REQUIRES_MATCHED_REFERENCE', quality_validation='NOT_RUN',
        state_evidence=dict(generator=dict(spec, writer_rank=0, status='PASS', tensor_count=1)))
    return dict(schema_version=api.SCHEMA, kind=api.KIND, source=api.source_identity(),
        parameters=dict(evidence_kind=api.FIXTURE, task=receipt['task'], budget=dict(max_loaded_bytes=2**20)),
        base=dict(config=base_config, weights=[pin(base_weights)], indexes=[],
                  pyramid=dict(kind='COMPLETE_SAFE_BASE'), key_policy=dict(kind=api.NATIVE_KEYS)),
        trainables=dict(payload=spec, parameter_names=['weight']),
        codec=dict(config=base_config, weights=[pin(base_weights)], provenance={}),
        origin=dict(contract=api.ORIGIN_CONTRACT, receipt=write(root/'rank_00.json', receipt),
                    source_head='a'*40, source_fingerprint='b'*64))


def test_metadata_derives_old_origin_without_claiming_current_resume(tmp_path):
    request = request_files(tmp_path/'origin')
    checked, provenance = api.validate_request(request)
    assert checked == request and checked is not request
    assert provenance['origin_source_head'] == 'a'*40
    assert provenance['origin_cycle'] == 2
    assert provenance['origin_versions'] == dict(generator=4, critic=20, ema=20)
    assert provenance['training_cycles_executed'] == 0
    assert provenance['training_resume'] == 'NOT_SUPPORTED_INITIALIZATION_ONLY'
    assert provenance['origin_contract'] == api.ORIGIN_CONTRACT
    assert provenance['origin_numerical_comparison'] == 'REQUIRES_MATCHED_REFERENCE'
    assert provenance['origin_quality_validation'] == 'NOT_RUN'
    assert 'origin_numerical_status' not in provenance
    assert str(tmp_path) not in json.dumps(provenance)


@pytest.mark.parametrize('fault', ['source', 'receipt_hash', 'wrong_head', 'status', 'finite',
    'writer', 'component', 'versions', 'cycles', 'base_inventory', 'base_symlink',
    'duplicate_names', 'unexpected_field', 'pyramid_seed', 'oversize', 'origin_contract', 'origin_fingerprint'])
def test_invalid_metadata_fails_before_any_publication(tmp_path, fault):
    request = request_files(tmp_path/'origin')
    receipt_path = Path(request['origin']['receipt']['path'])
    receipt = json.loads(receipt_path.read_text())
    if fault == 'source': request['source']['importer_sha256'] = '0'*64
    elif fault == 'receipt_hash': request['origin']['receipt']['sha256'] = '0'*64
    elif fault == 'wrong_head': request['origin']['source_head'] = 'b'*40
    elif fault == 'origin_contract': request['origin']['contract'] = 'unknown-producer'
    elif fault == 'origin_fingerprint': request['origin']['source_fingerprint'] = 'c'*64
    elif fault == 'status': receipt['status'] = 'PARTIAL_ORIGINAL_WAN_WORK'
    elif fault == 'finite': receipt['finite'] = False
    elif fault == 'writer': receipt['state_evidence']['generator']['writer_rank'] = 1
    elif fault == 'component': receipt['state_evidence'] = {'ema': receipt['state_evidence']['generator']}
    elif fault == 'versions': receipt['final_versions']['generator'] += 1
    elif fault == 'cycles': receipt['cycles'][1]['cycle'] = 9
    elif fault == 'base_inventory': receipt['identity']['assets']['model_files']['extra'] = 'a'*64
    elif fault == 'base_symlink':
        original = Path(request['base']['weights'][0]['path'])
        alias = original.with_name('alias.safetensors'); alias.symlink_to(original)
        request['base']['weights'][0]['path'] = str(alias)
    elif fault == 'duplicate_names': request['trainables']['parameter_names'] *= 2
    elif fault == 'unexpected_field': request['resume_from'] = '/old/checkpoint'
    elif fault == 'pyramid_seed': request['base']['pyramid'] = dict(kind='SOURCE_SEEDED_WAN_MISSING_PATCHES', seed=1)
    elif fault == 'oversize': request['parameters']['budget']['max_loaded_bytes'] = 1
    if fault in ('status', 'finite', 'writer', 'component', 'versions', 'cycles', 'base_inventory'):
        request['origin']['receipt'] = write(receipt_path, receipt)
    with pytest.raises((ValueError, FileNotFoundError)):
        api.validate_request(request)
    assert not (tmp_path/'export').exists()


def test_trainable_validation_stages_all_tensors_before_mutating(tmp_path):
    model = torch.nn.Linear(2, 2)
    before = deepcopy(model.state_dict())
    path = tmp_path/'generator.safetensors'
    values = {name: torch.full_like(value, .031) for name, value in model.named_parameters()}
    save_file(values, path, metadata={'component': 'generator'})
    spec = dict(payload=pin(path), parameter_names=list(values))
    staged = api.stage_trainables(model, spec)
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())
    api.apply_trainables(model, staged)
    assert all(torch.equal(value, values[name]) for name, value in model.state_dict().items())


@pytest.mark.parametrize('fault', ['extra', 'missing', 'shape', 'dtype', 'nonfinite', 'ema', 'order'])
def test_invalid_trainables_leave_live_parameters_untouched(tmp_path, fault):
    model = torch.nn.Linear(2, 2)
    before = deepcopy(model.state_dict())
    values = {name: torch.full_like(value, .031) for name, value in model.named_parameters()}
    names = list(values)
    if fault == 'extra': values['unexpected'] = torch.ones(1)
    elif fault == 'missing': values.pop('bias')
    elif fault == 'shape': values['bias'] = torch.ones(3)
    elif fault == 'dtype': values['bias'] = values['bias'].bfloat16()
    elif fault == 'nonfinite': values['bias'][0] = float('nan')
    elif fault == 'order': names.reverse()
    path = tmp_path/'generator.safetensors'
    save_file(values, path, metadata={'component': 'ema' if fault == 'ema' else 'generator'})
    with pytest.raises(ValueError):
        api.stage_trainables(model, dict(payload=pin(path), parameter_names=names))
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())


def test_public_unified_cli_dispatch_is_initialization_only(tmp_path, monkeypatch, capsys):
    from distillschedule.cli import main
    calls = []
    def initialize(request, output):
        calls.append((request, output))
        return dict(status='EXPORTED', kind=api.KIND, training_cycles_executed=0,
                    scope='SYNTHETIC_DISPATCH_ONLY_NOT_UPSTREAM_EXECUTION')
    monkeypatch.setattr(api, 'import_generator_weights', initialize)
    request, output = tmp_path/'request.json', tmp_path/'portable'
    request.write_text(json.dumps(dict(schema_version=api.SCHEMA)))
    assert main(['import-weights', str(request), '--output', str(output)]) == 0
    assert calls == [(request, output)]
    assert json.loads(capsys.readouterr().out)['training_cycles_executed'] == 0


def test_json_duplicate_or_nonfinite_refuses(tmp_path):
    for text in ('{"kind":1,"kind":2}', '{"x":NaN}'):
        path = tmp_path/'bad.json'; path.write_text(text)
        with pytest.raises(ValueError): api.read_request(path)


@pytest.mark.parametrize('fault', [None, 'missing', 'extra', 'shape', 'integer', 'nonfinite',
    'config', 'duplicate', 'index_owner', 'cast_overflow'])
def test_complete_base_staging_never_mutates_owner(tmp_path, fault):
    model = torch.nn.Linear(2, 2).to(torch.bfloat16)
    before = deepcopy(model.state_dict())
    values = {n: torch.full(t.shape, .125, dtype=torch.float32) for n, t in before.items()}
    architecture = dict(in_features=2, out_features=2, bias=True, device=None, dtype=None)
    config = write(tmp_path/'config.json', architecture)
    index = []
    if fault == 'missing': values.pop('bias')
    elif fault == 'extra': values['extra'] = torch.ones(1)
    elif fault == 'shape': values['bias'] = torch.ones(3)
    elif fault == 'integer': values['bias'] = torch.ones(2, dtype=torch.int64)
    elif fault == 'nonfinite': values['bias'][0] = float('nan')
    elif fault == 'cast_overflow': values['bias'][0] = torch.finfo(torch.float32).max
    elif fault == 'config': config = write(tmp_path/'config.json', dict(in_features=3, out_features=2))
    path = tmp_path/'base.safetensors'; save_file(values, path)
    weights = [pin(path)]
    if fault == 'duplicate': weights *= 2
    elif fault == 'index_owner': index = [write(tmp_path/'index.json', dict(weight_map={n: 'wrong.safetensors' for n in values}))]
    request = dict(parameters=dict(architecture=architecture, evidence_kind=api.FIXTURE),
        base=dict(config=config, weights=weights, indexes=index, pyramid=dict(kind='COMPLETE_SAFE_BASE'), key_policy=dict(kind=api.NATIVE_KEYS)))
    if fault is None:
        staged, missing, dtypes = api._base_values(model, request)
        assert not missing and set(dtypes.values()) == {'F32'}
        assert all(t.dtype == torch.bfloat16 for t in staged.values())
    else:
        with pytest.raises(ValueError): api._base_values(model, request)
    assert all(torch.equal(before[n], t) for n, t in model.state_dict().items())


def test_plain_torch_stage_shared_publisher_and_fresh_process_strict_restore(tmp_path):
    """Real shared producer/consumer; explicitly NOT the upstream W1 factory."""
    from distillschedule.training.adapters import ModelComponents
    from distillschedule.training.inference import publish_export
    from distillschedule.training.plugins import load_callable
    model = torch.nn.Linear(2, 2)
    payload = tmp_path/'trained.safetensors'
    values = {n: torch.full_like(t, .031) for n, t in model.named_parameters()}
    save_file(values, payload, metadata=dict(component='generator'))
    api.apply_trainables(model, api.stage_trainables(model, dict(payload=pin(payload), parameter_names=list(values))))
    _, source = load_callable('torch.nn:Linear')
    metadata = dict(scope='PLAIN_TORCH_SHARED_EXPORT_ONLY_NOT_UPSTREAM_W1')
    export = tmp_path/'portable'
    manifest = publish_export(ModelComponents(state=dict(generator=model), metadata=metadata), export,
        model=dict(factory='torch.nn:Linear', source=source, parameters=dict(in_features=2, out_features=2), metadata=metadata),
        identity=dict(model=source, model_metadata=metadata, weights_initialization=dict(kind=api.KIND)), cycle=0)
    assert manifest['kind'] == 'INFERENCE_WEIGHTS_NOT_RESUMABLE_CHECKPOINT'
    script = '''import sys,torch
from distillschedule.training.adapters import ModelComponents
from distillschedule.training.inference import read_export,restore_modules
root,value=read_export(sys.argv[1]);model=torch.nn.Linear(2,2)
restore_modules(ModelComponents(state={'generator':model}),root,value)
assert value['checkpoint_cycle']==0
assert torch.allclose(model(torch.ones(1,2)),torch.full((1,2),.093))
assert not torch.cuda.is_initialized()
print('PLAIN_TORCH_SHARED_EXPORT_STRICT_FRESH_PROCESS_PASS_NOT_UPSTREAM_W1')
'''
    result = subprocess.run([sys.executable, '-B', '-c', script, str(export)],
        env=dict(os.environ, WORLD_SIZE='1', CUDA_VISIBLE_DEVICES=''), text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert 'FRESH_PROCESS_PASS_NOT_UPSTREAM_W1' in result.stdout


def test_source_policy_is_explicit_not_private_commit_allowlist_and_raw_evidence_survives(tmp_path):
    from distillschedule.workloads.best_known.registry import fingerprint
    request = request_files(tmp_path/'origin')
    request['parameters']['evidence_kind'] = api.REAL
    receipt_path = Path(request['origin']['receipt']['path'])
    receipt = json.loads(receipt_path.read_text())
    receipt['identity']['task_sha256'] = fingerprint(receipt['task'])
    receipt['numerical_comparison'] = 'ORIGINAL_EXTERNAL_STATUS_NOT_INTERPRETED'
    receipt['quality_validation'] = 'ORIGINAL_QUALITY_STATUS_NOT_INTERPRETED'
    request['origin']['receipt'] = write(receipt_path, receipt)
    _, provenance = api.validate_request(request)
    assert provenance['origin_source_head'] == 'a'*40
    assert provenance['origin_numerical_comparison'] == receipt['numerical_comparison']
    assert provenance['origin_quality_validation'] == receipt['quality_validation']
    assert 'REVIEW' not in provenance.values() and 'origin_numerical_status' not in provenance


@pytest.mark.parametrize('fault', [None, 'collision', 'suffix', 'policy_source', 'native_policy'])
def test_original_wan_single_base_alias_is_explicit_source_bound_not_suffix_map(tmp_path, fault):
    from distillschedule.workloads.best_known.registry import HELIOS_SHA
    class Bare(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm_out = torch.nn.Module()
            self.norm_out.register_parameter('scale_shift_table', torch.nn.Parameter(torch.zeros(2, 3, dtype=torch.bfloat16)))
    model = Bare(); before = deepcopy(model.state_dict())
    values = dict(scale_shift_table=torch.ones(2, 3))
    policy = dict(kind=api.WAN_KEYS, source_sha256=HELIOS_SHA)
    if fault == 'collision': values['norm_out.scale_shift_table'] = torch.ones(2, 3)
    elif fault == 'suffix': values = {'some_prefix.scale_shift_table': torch.ones(2, 3)}
    elif fault == 'policy_source': policy['source_sha256'] = '0'*64
    elif fault == 'native_policy': policy = dict(kind=api.NATIVE_KEYS)
    path = tmp_path/'base.safetensors'; save_file(values, path)
    request = dict(parameters=dict(architecture={}, evidence_kind=api.FIXTURE),
        base=dict(config=write(tmp_path/'config.json', {}), weights=[pin(path)], indexes=[],
            pyramid=dict(kind='COMPLETE_SAFE_BASE'), key_policy=policy))
    if fault is None:
        staged, missing, _ = api._base_values(model, request)
        assert set(staged) == {'norm_out.scale_shift_table'} and not missing
    else:
        with pytest.raises(ValueError): api._base_values(model, request)
    assert all(torch.equal(before[n], t) for n, t in model.state_dict().items())


def test_admitted_trainables_own_storage_not_mutable_input_mmap(tmp_path):
    import struct
    model = torch.nn.Linear(2, 2)
    values = {n: torch.full_like(t, .031) for n, t in model.named_parameters()}
    path = tmp_path/'generator.safetensors'
    save_file(values, path, metadata=dict(component='generator'))
    staged = api.stage_trainables(model, dict(payload=pin(path), parameter_names=list(values)))
    with path.open('r+b') as stream:
        stream.seek(-4, 2); stream.write(struct.pack('<f', .75)); stream.flush(); os.fsync(stream.fileno())
    assert all(torch.equal(staged[n], t) for n, t in values.items())


@pytest.mark.parametrize('budget_name', ['max_export_bytes_per_rank', 'max_export_bytes_all_ranks'])
def test_import_refuses_export_transient_budget_before_live_construction(tmp_path, monkeypatch, budget_name):
    """Only admission-size metadata is synthetic; not a W1 runtime proof."""
    from distillschedule.training import wan_native_inference as native
    request = request_files(tmp_path/'origin')
    size = 64
    request['parameters']['budget'].update(max_export_bytes_per_rank=3*size, max_export_bytes_all_ranks=3*size)
    request['parameters']['budget'][budget_name] = 3*size-1
    path = tmp_path/'initialization.json'; write(path, request)
    probe = torch.nn.Linear(2, 2, bias=False)
    monkeypatch.setattr(native, '_admit', lambda parameters: (deepcopy(parameters), probe, None, size))
    def forbidden(*args, **kwargs):
        raise AssertionError('export budget guard reached codec/live construction or publication')
    monkeypatch.setattr(native, '_load_codec', forbidden)
    output = tmp_path/'portable'
    with pytest.raises(ValueError, match='export.*capacity'):
        api.import_generator_weights(path, output)
    assert not output.exists() and not output.with_name(output.name+'.inference.json').exists()


@pytest.mark.parametrize('field', [None, 'rope_max_seq_len', 'unknown_native_config_field'])
def test_base_nonconstructor_rope_config_compatibility_is_exact(tmp_path, field):
    """Metadata-only admission RED/control, not a substitute for original G."""
    model = torch.nn.Linear(2, 2)
    before = deepcopy(model.state_dict())
    architecture = dict(in_features=2, out_features=2, bias=True, device=None, dtype=None)
    raw = dict(architecture, _class_name='METADATA_ONLY')
    if field is not None:
        raw[field] = 1024
    config = write(tmp_path/'config.json', raw)
    raw_bytes = Path(config['path']).read_bytes()
    path = tmp_path/'base.safetensors'
    values = {name: torch.full_like(t, .125) for name, t in before.items()}
    save_file(values, path)
    request = dict(parameters=dict(architecture=architecture, evidence_kind=api.FIXTURE),
        base=dict(config=config, weights=[pin(path)], indexes=[],
            pyramid=dict(kind='COMPLETE_SAFE_BASE'), key_policy=dict(kind=api.NATIVE_KEYS)))
    if field == 'unknown_native_config_field':
        with pytest.raises(ValueError, match='unknown pinned base constructor field'):
            api._base_values(model, request)
    else:
        staged, missing, _ = api._base_values(model, request)
        assert not missing and all(torch.equal(staged[n], t) for n, t in values.items())
        provenance = {}
        api._base_values(model, request, provenance=provenance)
        assert provenance == ({'base_non_constructor_config': {'rope_max_seq_len': 1024}}
            if field == 'rope_max_seq_len' else {})
    assert all(torch.equal(before[n], t) for n, t in model.state_dict().items())
    assert Path(config['path']).read_bytes() == raw_bytes and pin(Path(config['path'])) == config
