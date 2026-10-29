"""Mandatory exact public factory/sampler/codec CPU chain, no runtime shim.

R225_REQUIRE_UPSTREAM=1 turns missing dependencies into failure, never SKIP.
Tiny/random full original classes prove wiring, not 1.3B/native training quality.
"""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from test_training_livewan_generator_import import fixture_request, pin, write_json, same
from distillschedule.training import livewan_generator_import as importer


@pytest.fixture
def qualified_case(tmp_path):
    missing = [name for name in ('torch', 'diffusers', 'peft', 'transformers', 'omegaconf', 'safetensors')
               if importlib.util.find_spec(name) is None]
    if missing:
        message = 'Actual public W4 runtime dependencies missing: ' + ', '.join(missing)
        if os.environ.get('R225_REQUIRE_UPSTREAM') == '1':
            pytest.fail(message)
        pytest.skip(message)
    return fixture_request(tmp_path / 'OLD-origin-files', qualified_runtime=True)


def prepare(root, feeds):
    import torch
    from distillschedule.data.preparation import import_preencoded
    from distillschedule.data.manifest import PREENCODED_SCHEMA
    root.mkdir(parents=True)
    payload = root / 'new-clean-world-positive-text.pt'; torch.save(feeds, payload)
    source = write_json(root / 'inputs.json', dict(schema_version=PREENCODED_SCHEMA,
        records=[dict(id='NEW-INDEPENDENT-WORLD-AND-TEXT', features={k: v for k, v in pin(payload).items() if k != 'bytes'})]))
    return import_preencoded(source, root / 'prepared')


def test_actual_public_factory_import_export_fresh_process_new_inputs_real_decode(qualified_case, tmp_path, monkeypatch):
    import torch
    from distillschedule.training import livewan_native_inference as native, livewan_public
    from distillschedule.training.checkpoint import rng_state
    from distillschedule.training.inference import read_export, restore_modules
    case = qualified_case
    def forbidden(*args, **kwargs):
        raise AssertionError('training owner/optimizer must not be constructed')
    monkeypatch.setattr(livewan_public.LiveWanCPUOwner, '__init__', forbidden)
    monkeypatch.setattr(torch.optim.AdamW, '__init__', forbidden)
    before = rng_state()
    result = importer.import_generator_weights(case['path'], tmp_path / 'portable')
    assert same(before, rng_state())
    assert result['initialization_only'] is True and result['training_cycles_executed'] == 0
    assert result['checkpoint_cycle'] == 0 and result['quality_validation'] == 'NOT_RUN'
    root, manifest = read_export(tmp_path / 'portable')
    origin = manifest['identity']['weights_initialization']
    assert origin['origin_numerical_status'] == 'FAIL_EXTREME_NUMERICAL_DRIFT'
    assert origin['origin_execution_status'] == {'selected': 'PASS', 'reference': 'PASS'}
    assert origin['origin_cycle'] == 100 and origin['origin_versions'] == {'generator': 100, 'critic': 500, 'ema': 0}
    assert origin['positional_state']['old_runtime_values_captured'] is False
    assert manifest['model']['factory'] == native.FACTORY
    consumer = native.build_inference_model(manifest['model']['parameters'])
    generator = torch.Generator().manual_seed(3571)
    feeds = dict(world=torch.randn(1, 16, 6, 4, 4, generator=generator),
                 context=torch.randn(1, 512, 4096, generator=generator))
    with pytest.raises(ValueError, match='strict restored'):
        consumer.generate(consumer.state, feeds, {})
    restore_modules(consumer, root, manifest)
    assert same(consumer.state['generator'].model.state_dict(), case['student'].state_dict())
    assert torch.equal(consumer.state['generator'].model.freqs, case['student'].freqs)
    before = rng_state()
    expected = consumer.generate(consumer.state, feeds, {})
    assert same(before, rng_state())
    assert torch.equal(expected['latents'][:, :, :6], feeds['world'])
    decoded = consumer.decode(expected, {})
    assert list(decoded['video'].shape) == [81, 32, 32, 3]
    assert decoded['decoder']['latent_clamp'] is False
    torch.save(dict(generated=expected, decoded=decoded), tmp_path / 'expected.pt')
    prepared = prepare(tmp_path / 'NEW-inputs', feeds)
    # Delete EVERY original weight/config/receipt/source path before the
    # new OS process imports the unchanged public consumer.
    shutil.rmtree(case['root'])
    script = r'''
from pathlib import Path
import json,sys,torch
from distillschedule.training import livewan_public
from distillschedule.training.config import load_training_config
from distillschedule.training.session import generate_from_export
from test_training_livewan_generator_import import same
torch.set_num_threads(1)
def forbidden(*a,**k): raise AssertionError('training/Adam/old state entered')
livewan_public.LiveWanCPUOwner.__init__=forbidden
torch.optim.AdamW.__init__=forbidden
root=Path(sys.argv[1]);assert not (root/'OLD-origin-files').exists()
config=load_training_config(root/'portable.inference.json')
result=generate_from_export(config,root/'portable',root/'generated',prepared=Path(sys.argv[2]),output_kind='video')
actual=torch.load(result['artifact'],weights_only=True)
expected=torch.load(root/'expected.pt',weights_only=True)
assert same(actual['generated'],expected['generated']) and same(actual['decoded'],expected['decoded'])
assert result['checkpoint_cycle']==0 and len(list((root/'generated/frames').glob('*.png')))==81
assert not torch.cuda.is_initialized()
print(json.dumps(dict(status='PASS_REAL_ORIGINAL_PUBLIC_W4_IMPORT_EXPORT_FRESH_GENERATE_DECODE_CPU_TINY_ONLY',
    video_frames=81,checkpoint_cycle=0,training_cycles=0,native_1_3B='NOT_RUN',quality='NOT_RUN')))
'''
    result = subprocess.run([sys.executable, '-B', '-c', script, str(tmp_path), str(prepared)],
        text=True, capture_output=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'PASS_REAL_ORIGINAL_PUBLIC_W4_IMPORT_EXPORT_FRESH_GENERATE_DECODE_CPU_TINY_ONLY' in result.stdout


@pytest.mark.parametrize('fault', ['parameter_budget', 'export_budget', 'import_transient_budget', 'reserved_config', 'existing_export'])
def test_actual_public_entry_refuses_capacity_and_output_collision_before_loading_student(qualified_case, tmp_path, monkeypatch, fault):
    from distillschedule.training import livewan_native_inference as native
    case = qualified_case
    request = deepcopy(case['request'])
    output = tmp_path / 'never-published'
    if fault == 'parameter_budget': request['parameters']['budget']['max_parameters'] = 1
    elif fault == 'export_budget': request['parameters']['budget']['max_primary_export_bytes'] = 1
    elif fault == 'import_transient_budget':
        _, _, _, sizes = native._probe(request['parameters'])
        request['parameters']['budget']['max_loaded_bytes'] = sizes['working_bytes'] + request['student']['payload']['bytes']
    elif fault == 'reserved_config': output.with_name(output.name + '.inference.json').write_text('USER_OWNED')
    else: output.mkdir(); (output / 'USER_OWNED').write_text('keep')
    path = write_json(tmp_path / 'bad-request.json', request)
    def forbidden(*args, **kwargs):
        raise AssertionError('Student load ran before capacity/collision admission')
    monkeypatch.setattr(importer, 'stage_student', forbidden)
    with pytest.raises(ValueError): importer.import_generator_weights(path, output)
    assert not (output / '_SUCCESS').exists()
    if fault == 'reserved_config': assert output.with_name(output.name + '.inference.json').read_text() == 'USER_OWNED'
    if fault == 'existing_export': assert (output / 'USER_OWNED').read_text() == 'keep'


def test_actual_import_reauthenticates_old_receipt_after_load_and_leaves_no_export(qualified_case, tmp_path, monkeypatch):
    case = qualified_case
    real_stage = importer.stage_student
    def mutate(*args, **kwargs):
        values = real_stage(*args, **kwargs)
        Path(case['request']['origin']['selected_summary']['path']).write_text('{}')
        return values
    monkeypatch.setattr(importer, 'stage_student', mutate)
    with pytest.raises(ValueError): importer.import_generator_weights(case['path'], tmp_path / 'no-export')
    assert not (tmp_path / 'no-export').exists()


@pytest.mark.parametrize('frames', [3, 6, 9, 21])
def test_existing_native_consumer_only_accepts_explicit_world3_6_9_never_crops_history(qualified_case, tmp_path, frames):
    import torch
    from distillschedule.training import livewan_native_inference as native
    from distillschedule.training.inference import restore_modules
    case = qualified_case
    manifest = importer.import_generator_weights(case['path'], tmp_path / 'portable')
    consumer = native.build_inference_model(manifest['model']['parameters'])
    restore_modules(consumer, tmp_path / 'portable', manifest)
    generator = torch.Generator().manual_seed(264)
    feeds = dict(world=torch.randn(1, 16, frames, 4, 4, generator=generator),
                 context=torch.randn(1, 512, 4096, generator=generator))
    if frames == 21:
        with pytest.raises(ValueError, match='world3/6/9'): consumer.generate(consumer.state, feeds, {})
    else:
        actual = consumer.generate(consumer.state, feeds, {})
        assert list(actual['latents'].shape) == [1, 16, 21, 4, 4]
        assert torch.equal(actual['latents'][:, :, :frames], feeds['world'])
    with pytest.raises(ValueError): consumer.generate(consumer.state, feeds, {'window_frames': 12})
    with pytest.raises(ValueError, match='not a resumable'):
        consumer.admit_training()
