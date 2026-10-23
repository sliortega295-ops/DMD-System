"""Generic export-input contract with an unknown, explicitly trusted test plugin.

The scalar fixture is SYNTHETIC_TEST_ONLY extension-contract evidence, not a
model/media/quality claim. Actual native acceptance lives in the adjacent test.
"""
import json
from pathlib import Path

import pytest

PLUGIN = '''
import hashlib
import random
from pathlib import Path
import numpy as np
import torch
from distillschedule.training.adapters import ModelComponents
from distillschedule.data.manifest import source_identity, require

def encode(state, record, parameters):
    require(parameters['mode'] == 'prompts', 'unsupported test mode')
    require(set(record) == {'id', 'prompt'} and isinstance(record['prompt'], str), 'unknown test record')
    prompt = record['prompt']
    random.random(); np.random.rand(); torch.rand(1)
    if prompt == 'fail':
        raise ValueError('explicit encoder failure')
    if prompt == 'mutate':
        with torch.no_grad(): state['student'].weight.add_(1)
    if prompt == 'replace':
        state['student'] = torch.nn.Linear(1, 1)
    metadata = dict(schema_version='test.unknown_native_inputs.v1', binding='one', evidence_kind='SYNTHETIC_TEST_ONLY')
    if prompt == 'inconsistent': metadata['binding'] = 'two'
    if prompt == 'reserved': metadata['_export_inputs'] = {}
    sources = {'text': {'kind':'inline_text','sha256':hashlib.sha256(prompt.encode()).hexdigest()}}
    if prompt == 'bad_source': sources['text']['sha256'] = 'wrong'
    if prompt == 'change_raw':
        path = Path(parameters['source_dir'])/'raw.bin'
        sources['image'] = source_identity(path)
        path.write_bytes(b'changed during encoding')
    if prompt == 'change_export':
        path = Path(parameters['source_dir'])/'export'/'unrecorded.txt'
        path.write_text('changed inventory')
    result = dict(features={'scalar':torch.tensor([float(len(prompt))])}, metadata=metadata, sources=sources)
    if prompt == 'bad_result': result.pop('sources')
    if prompt == 'nonfinite': result['features']['scalar'].fill_(float('nan'))
    return result

def build(parameters):
    require(set(parameters) <= {'seed', 'device', 'variant'}, 'unknown test parameters')
    require(parameters.get('device', 'cpu') == 'cpu', 'CPU only test plugin')
    random.random(); np.random.rand(); torch.rand(1)
    student = torch.nn.Linear(1, 1)
    return ModelComponents(state={'student':student}, metadata={'fixture':'unknown-family', 'variant':parameters['variant']},
        encode_input=None if parameters['variant'] == 'missing' else encode)
'''


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


@pytest.fixture
def exported(tmp_path):
    from distillschedule.training.inference import publish_export
    from distillschedule.training.plugins import load_callable
    path = tmp_path / 'unknown_adapter.py'
    path.write_text(PLUGIN)

    def create(variant='normal'):
        build, source = load_callable(str(path) + ':build')
        parameters = dict(seed=17, variant=variant)
        components = build(parameters).validate()
        manifest = publish_export(components, tmp_path / 'export',
            model=dict(factory=source['reference'], source=source, parameters=parameters, metadata=dict(components.metadata)),
            identity=dict(model=source, model_metadata=dict(components.metadata)), cycle=1)
        return manifest
    return tmp_path, path, create


def request(root, prompts=('new prompt',)):
    path = root / 'inputs.json'
    write_json(path, dict(schema_version='distillschedule.export_inputs.v1', weights='export', mode='prompts',
                         records=[dict(id=f'new-{i}', prompt=prompt) for i, prompt in enumerate(prompts)]))
    return path


def resign(root, mutate):
    from distillschedule.workloads.best_known.registry import file_sha256
    path = root / 'export/export.json'
    manifest = json.loads(path.read_text())
    mutate(manifest)
    write_json(path, manifest)
    (root / 'export/_SUCCESS').write_text(file_sha256(path) + '\n')


def assert_clean(root):
    assert not (root / 'prepared').exists()
    assert not list(root.glob('.prepared*')), 'failed preparation must remove staging and lock'


def test_unknown_plugin_capability_is_generic_and_preserves_rng_and_input(exported):
    from distillschedule.data.export_inputs import prepare_inputs
    from distillschedule.data import PreparedDataset
    from distillschedule.training.checkpoint import rng_state
    from test_training_end_to_end import equal
    root, plugin, create = exported
    manifest = create()
    path = request(root, ('new prompt', 'another new prompt'))
    before_bytes, before_rng = path.read_bytes(), rng_state(device='cpu')
    prepared = prepare_inputs(path, root / 'prepared')
    assert equal(before_rng, rng_state(device='cpu')) and path.read_bytes() == before_bytes
    dataset = PreparedDataset(prepared)
    assert dataset.validate(sources=True)['status'] == 'PASS'
    assert [dataset.sample_id(i) for i in range(len(dataset))] == ['new-0', 'new-1']
    assert dataset.sample(0)['scalar'].tolist() == [10.]
    value = dataset.manifest
    assert value['metadata']['binding'] == 'one'
    assert value['metadata']['_export_inputs']['files'] == manifest['files']
    encoder = value['provenance']['encoder']
    assert encoder['source_path'] == str(plugin)
    assert encoder['parameters']['export_manifest_sha256']
    assert encoder['parameters']['model_source'] == manifest['model']['source']
    assert str(root / 'export/weights.pt') in [p['path'] for p in value['provenance']['documents']]


def test_missing_optional_capability_refuses_without_training_fallback(exported):
    from distillschedule.data.export_inputs import prepare_inputs
    root, _, create = exported
    create('missing')
    with pytest.raises(ValueError, match='encode_input|capability'):
        prepare_inputs(request(root), root / 'prepared')
    assert_clean(root)


@pytest.mark.parametrize('prompt,message', [
    ('fail', 'encoder failure'), ('mutate', 'mutated'), ('replace', 'mutated'),
    ('inconsistent', 'metadata'), ('reserved', 'reserved'), ('bad_source', 'SHA256'),
    ('change_raw', 'source.*changed|hash changed'), ('change_export', 'unrecorded'),
    ('bad_result', 'encoder.*result|encode_input.*return'),
    ('nonfinite', 'nonfinite'),
])
def test_callback_failures_are_atomic_and_rng_is_restored(exported, prompt, message):
    from distillschedule.data.export_inputs import prepare_inputs
    from distillschedule.training.checkpoint import rng_state
    from test_training_end_to_end import equal
    root, _, create = exported
    create()
    (root / 'raw.bin').write_bytes(b'original inventory')
    path = request(root, ('normal first row', prompt))
    before = rng_state(device='cpu')
    before_input = path.read_bytes()
    with pytest.raises(ValueError, match=message):
        prepare_inputs(path, root / 'prepared')
    assert equal(before, rng_state(device='cpu')) and path.read_bytes() == before_input
    assert_clean(root)


@pytest.mark.parametrize('change,message', [
    ('source', 'source'), ('factory', 'source|factory'), ('metadata', 'metadata'),
    ('layout', 'layout'), ('weights_hash', 'hash'), ('extra_file', 'unrecorded'),
    ('unsafe_weights', 'safe.*weights'), ('nonfinite_weights', 'nonfinite'), ('dtype', 'layout'),
])
def test_export_integrity_and_identity_refuse_transactionally(exported, change, message):
    import torch
    from distillschedule.data.export_inputs import prepare_inputs
    from distillschedule.workloads.best_known.registry import file_sha256
    root, plugin, create = exported
    create()
    path = request(root)
    if change == 'source':
        plugin.write_text(plugin.read_text() + '\n# source changed\n')
    elif change == 'factory':
        resign(root, lambda value: value['model'].update(factory=str(plugin) + ':encode'))
    elif change == 'metadata':
        def modify(value):
            value['model']['metadata']['fixture'] = 'changed'
            value['identity']['model_metadata']['fixture'] = 'changed'
        resign(root, modify)
    elif change == 'layout':
        def modify(value):
            value['model']['modules']['student']['state']['items']['weight']['shape'] = [2, 1]
        resign(root, modify)
    elif change == 'extra_file':
        (root / 'export/unrecorded.txt').write_text('extra')
    else:
        weights = root / 'export/weights.pt'
        if change in ('weights_hash', 'unsafe_weights'):
            weights.write_bytes(b'not torch weights')
        else:
            values = torch.load(weights, weights_only=True)
            if change == 'dtype':
                values['student']['weight'] = values['student']['weight'].double()
            else:
                values['student']['weight'].fill_(float('nan'))
            torch.save(values, weights)
        if change != 'weights_hash':
            resign(root, lambda value: value['files'].update({'weights.pt': file_sha256(weights)}))
    with pytest.raises(ValueError, match=message):
        prepare_inputs(path, root / 'prepared')
    assert_clean(root)


@pytest.mark.parametrize('change,message', [
    ('schema', 'schema'), ('unknown_field', 'fields|schema'), ('duplicate_id', 'duplicate'),
    ('unknown_record', 'unknown.*record'), ('empty', 'nonempty'), ('url', 'local'), ('mode', 'mode'),
])
def test_bad_input_schema_record_and_mode_refuse(exported, change, message):
    from distillschedule.data.export_inputs import prepare_inputs
    root, _, create = exported
    create()
    path = request(root)
    value = json.loads(path.read_text())
    if change == 'schema': value['schema_version'] = 'wrong'
    if change == 'unknown_field': value['assets'] = 'forbidden-training-bundle'
    if change == 'duplicate_id': value['records'] *= 2
    if change == 'unknown_record': value['records'][0]['context'] = [0.]
    if change == 'empty': value['records'] = []
    if change == 'url': value['weights'] = 'https://invalid.example/export'
    if change == 'mode': value['mode'] = 'preencoded'
    write_json(path, value)
    with pytest.raises(ValueError, match=message):
        prepare_inputs(path, root / 'prepared')
    assert_clean(root)


def test_output_exists_refuses_before_import_and_preserves_existing_data(tmp_path):
    from distillschedule.data.export_inputs import prepare_inputs
    output = tmp_path / 'prepared'
    output.mkdir()
    (output / 'owned.txt').write_text('unchanged')
    with pytest.raises(FileExistsError, match='already exists'):
        prepare_inputs(tmp_path / 'nonexistent.json', output)
    assert (output / 'owned.txt').read_text() == 'unchanged'


def test_duplicate_manifest_keys_refuse(exported):
    from distillschedule.data.export_inputs import prepare_inputs
    root, _, create = exported
    create()
    path = request(root)
    path.write_text(path.read_text().replace('"mode": "prompts"', '"mode": "raw", "mode": "prompts"'))
    with pytest.raises(ValueError, match='duplicate'):
        prepare_inputs(path, root / 'prepared')
    assert_clean(root)


def test_output_inside_export_is_refused_without_modifying_inventory(exported):
    from distillschedule.data.export_inputs import prepare_inputs
    from distillschedule.training.inference import read_export
    root, _, create = exported
    manifest = create()
    with pytest.raises(ValueError, match='export inventory'):
        prepare_inputs(request(root), root / 'export/prepared')
    assert read_export(root / 'export')[1] == manifest


def test_encoding_callback_is_an_optional_validated_model_capability():
    from distillschedule.training.adapters import ModelComponents
    assert ModelComponents(state={'owned': 1}).validate().encode_input is None
    with pytest.raises(ValueError, match='encode_input.*callable'):
        ModelComponents(state={'owned': 1}, encode_input='not callable').validate()
