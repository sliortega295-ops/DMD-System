"""Public export-only inputs; actual tiny native models, SYNTHETIC_TEST_ONLY.

No test publishes features by hand. Both preparation and generation run through
the public CLI from a copied package after deleting every training asset/state.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(os.environ.get('DMD_MODERN_FLOW_CPU') != '1',
    reason='requires the opt-in isolated modern-flow pinned CPU stack')

CASES = [
    ('flux2-klein-base', 'flow_image', [1, 4, 4, 4], 'image', 8,
     'distillschedule.flow_image_inputs.v1', 'flow_image_inputs'),
    ('qwen-image', 'flow_image', [1, 4, 4, 4], 'image', 8,
     'distillschedule.flow_image_inputs.v1', 'flow_image_inputs'),
    ('wan22-ti2v5b-t2v', 'wan22_dmd', [1, 4, 2, 2, 2], 'video', 32,
     'distillschedule.wan22_t2v_inputs.v1', 'wan22_inputs'),
    ('hunyuan15-t2v', 'hunyuan_video_dmd', [1, 4, 2, 4, 4], 'video', 16,
     'distillschedule.hunyuan15_t2v_inputs.v1', 'hunyuan_video_inputs'),
]


@pytest.mark.parametrize('family,recipe,shape,kind,size,legacy_schema,legacy_module', CASES)
def test_public_new_inputs_after_assets_and_all_checkpoints_deleted(
        tmp_path, family, recipe, shape, kind, size, legacy_schema, legacy_module):
    import numpy as np
    from PIL import Image
    import torch
    from distillschedule.data import PreparedDataset
    from distillschedule.data.manifest import source_identity
    from distillschedule.training.config import load_training_config
    from distillschedule.training.plugins import load_callable
    from distillschedule.training.session import _load_for_output
    from test_training_end_to_end import equal

    repo = Path(__file__).resolve().parents[1]
    build, _ = load_callable(str(repo / 'recipes' / recipe / 'create_cpu_fixture.py') + ':build_fixture')
    original = tmp_path / 'training-assets'
    assets = build(original, family, cycles=1) if recipe == 'flow_image' else build(original, cycles=1)
    portable = tmp_path / 'portable'
    portable.mkdir()
    shutil.copytree(repo / 'src/distillschedule', portable / 'package/distillschedule')
    shutil.copytree(repo / 'runtime', portable / 'package/runtime')
    # No cwd imports and no canonical-checkout entries in the child search path.
    dependencies = [str(Path(p).resolve()) for p in os.environ['PYTHONPATH'].split(os.pathsep)
                    if p and not Path(p).resolve().is_relative_to(repo)]
    env = {**os.environ, 'PYTHONPATH': os.pathsep.join([str(portable / 'package'), *dependencies]),
           'CUDA_VISIBLE_DEVICES': '', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}

    def cli(*args, success=True):
        command = [sys.executable, '-m', 'distillschedule', *map(str, args)]
        result = subprocess.run(command, cwd=portable, env=env, text=True, capture_output=True)
        with (tmp_path / 'cli.log').open('a') as stream:
            stream.write(json.dumps(dict(command=command, cwd=str(portable), returncode=result.returncode)) + '\n')
            stream.write(result.stdout + result.stderr + '\n')
        if success:
            assert result.returncode == 0, result.stdout + result.stderr
            return json.loads(result.stdout)
        assert result.returncode != 0, result.stdout
        return result

    trained = cli('train', assets['config'], '--cycles', 1, '--output', tmp_path / 'training')
    current = _load_for_output(load_training_config(assets['config']), trained['checkpoint'])[0]
    training_feeds = PreparedDataset(assets['prepared']).sample(0)
    with torch.no_grad():
        sigma = torch.tensor([.5])
        g = current.state['student'].denoise(training_feeds['latents'], sigma, training_feeds, conditional=True)
        teacher = current.state['teacher'].denoise(training_feeds['latents'], sigma, training_feeds, conditional=True)
    assert not equal(g, teacher), 'updated unmerged-LoRA G must differ from the frozen Teacher'
    training_context = training_feeds['context']
    exported = cli('export', assets['config'], '--checkpoint', trained['checkpoint'], '--output', portable / 'export')
    assert set(exported['model']['modules']) == {'student', 'media'}
    shutil.rmtree(original)
    shutil.rmtree(tmp_path / 'training')
    assert not original.exists() and not (tmp_path / 'training').exists()
    assert not list(tmp_path.rglob('state.pt'))

    # Characterize the existing public plugin: it still requires ASSET_SCHEMA.
    legacy = dict(schema_version=legacy_schema, family=family, assets=str(assets['assets']), mode='prompts',
                  records=[dict(id='NEW_AFTER_DELETION', prompt='a violet kite', latent_shape=shape)])
    (portable / 'legacy.json').write_text(json.dumps(legacy))
    refused = cli('data', 'prepare', 'legacy.json', '--output', 'legacy-output',
                  '--preparer', f'distillschedule.data.{legacy_module}:prepare_inputs', success=False)
    assert str(original) in refused.stderr and not (portable / 'legacy-output').exists()

    source = dict(schema_version='distillschedule.export_inputs.v1', weights='export', mode='prompts',
                  records=[dict(id='NEW_AFTER_DELETION', prompt='a violet kite', latent_shape=shape)])
    (portable / 'new.json').write_text(json.dumps(source))
    prepared_result = cli('data', 'prepare', 'new.json', '--output', 'new-prepared',
                          '--preparer', 'distillschedule.data.export_inputs:prepare_inputs')
    prepared = Path(prepared_result['manifest'])
    dataset = PreparedDataset(prepared)
    assert dataset.validate(sources=True)['status'] == 'PASS'
    feeds = dataset.sample(0)
    assert not equal(feeds['context'], training_context), 'new conditions must not be copied from training data'
    native_text = current.state['media'].encode_text(['a violet kite'])
    text_fields = ('context', 'context_mask', 'context_2', 'context_mask_2')[:len(native_text)]
    assert equal(tuple(feeds[key] for key in text_fields), native_text), 'new conditions must be actual native encoder outputs'
    assert equal(tuple(feeds['null_' + key] for key in text_fields), current.state['media'].encode_text(['']))
    assert str(original) not in json.dumps(dataset.manifest), 'new provenance must name the export, not deleted assets'
    export_identity = dataset.manifest['metadata']['_export_inputs']
    assert export_identity['weights'] == str(portable / 'export')
    assert export_identity['files'] == exported['files']

    inference = dict(schema_version='distillschedule.training.v1',
        model=dict(factory=exported['model']['factory'], parameters=exported['model']['parameters']),
        algorithm=dict(name='dmd'), data=dict(prepared=str(prepared)), execution=dict(mode='reference'),
        trainer=dict(cycles=1, seed=909, device='cpu', output_dir=str(portable / 'unused')))
    (portable / 'inference.json').write_text(json.dumps(inference))
    actual = cli('generate', 'inference.json', '--weights', 'export', '--prepared', prepared,
                 '--output', 'generation', '--output-kind', kind)
    for value in current.state.values():
        if isinstance(value, torch.nn.Module):
            value.eval()
    torch.random.default_generator.manual_seed(909)
    with torch.no_grad():
        generated = current.generate(current.state, feeds, {})
        expected = dict(generated=generated, decoded=current.decode(generated, {}))
    artifact = torch.load(actual['artifact'], weights_only=True)
    assert equal(expected, artifact), 'decoded output must equal native CURRENT Student, not Teacher/base/EMA'
    if kind == 'image':
        assert Path(actual['image']['path']).is_file() and artifact['decoded']['image'].shape == (8, 8, 3)
    else:
        assert len(list(Path(actual['video']['frames']).glob('*.png'))) == 5
        assert artifact['decoded']['video'].shape == (5, size, size, 3)

    # Also exercise actual newly created raw RGB media, not only a geometry template.
    pixels = ((np.arange(size * size * 3).reshape(size, size, 3) * 11 + 37) % 256).astype('uint8')
    if kind == 'image':
        raw = portable / 'new-image.png'
        Image.fromarray(pixels).save(raw)
    else:
        raw = portable / 'new-frames'
        raw.mkdir()
        for i in range(5):
            Image.fromarray(np.roll(pixels, i, axis=0)).save(raw / f'{i:04d}.png')
    raw_identity = source_identity(raw, frames=kind == 'video')
    source.update(mode='raw', records=[dict(id='NEW_RAW_AFTER_DELETION', prompt='a teal bird',
        **{kind: dict(path=raw.name, sha256=raw_identity['sha256'])})])
    (portable / 'raw.json').write_text(json.dumps(source))
    raw_result = cli('data', 'prepare', 'raw.json', '--output', 'raw-prepared',
                     '--preparer', 'distillschedule.data.export_inputs:prepare_inputs')
    raw_dataset = PreparedDataset(raw_result['manifest'])
    assert raw_dataset.validate(sources=True)['status'] == 'PASS'
    raw_feeds = raw_dataset.sample(0)
    assert list(raw_feeds['latent_shape']) == list(feeds['latent_shape'])
    if kind == 'image':
        native_pixels = torch.from_numpy(pixels.copy()).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1
        native_latents = current.state['media'].encode_image(native_pixels)[0]
    else:
        from distillschedule.data.decoding import _video
        native_latents = current.state['media'].encode_video(_video(raw))
    assert torch.equal(raw_feeds['latents'][kind], native_latents), 'raw latents must come from the actual restored native VAE'
    raw_actual = cli('generate', 'inference.json', '--weights', 'export', '--prepared', raw_result['manifest'],
                     '--output', 'raw-generation', '--output-kind', kind)
    assert raw_actual['sample_id'] == 'NEW_RAW_AFTER_DELETION'
    # Native guards still own record, condition, geometry and tokenizer limits.
    from copy import deepcopy
    from distillschedule.data.export_inputs import prepare_inputs
    from distillschedule.workloads.best_known.registry import file_sha256
    prompt_source = json.loads((portable / 'new.json').read_text())
    invalid = {}
    invalid['geometry'] = deepcopy(prompt_source)
    invalid['geometry']['records'][0]['latent_shape'][1] += 1
    invalid['unknown_condition'] = deepcopy(prompt_source)
    invalid['unknown_condition']['records'][0]['context'] = [0.]
    invalid['missing_prompt'] = deepcopy(prompt_source)
    invalid['missing_prompt']['records'][0].pop('prompt')
    invalid['overflow'] = deepcopy(prompt_source)
    invalid['overflow']['records'][0]['prompt'] = 'a very long bird ' * 2000
    invalid['preencoded'] = deepcopy(prompt_source)
    invalid['preencoded']['mode'] = 'preencoded'
    invalid['raw_hash'] = deepcopy(source)
    invalid['raw_hash']['records'][0][kind]['sha256'] = '0' * 64
    for label, value in invalid.items():
        bad_source = portable / f'bad-{label}.json'
        bad_source.write_text(json.dumps(value))
        output = portable / f'bad-{label}-output'
        with pytest.raises(ValueError):
            prepare_inputs(bad_source, output)
        assert not output.exists() and not list(portable.glob(f'.{output.name}*'))
    # A self-consistently resigned manifest cannot override the native runtime.
    shutil.copytree(portable / 'export', portable / 'wrong-runtime-export')
    wrong_path = portable / 'wrong-runtime-export/export.json'
    wrong = json.loads(wrong_path.read_text())
    wrong['model']['parameters']['runtime']['versions']['transformers'] = 'wrong'
    wrong_path.write_text(json.dumps(wrong))
    (wrong_path.parent / '_SUCCESS').write_text(file_sha256(wrong_path) + '\n')
    wrong_source = deepcopy(prompt_source)
    wrong_source['weights'] = 'wrong-runtime-export'
    (portable / 'wrong-runtime.json').write_text(json.dumps(wrong_source))
    with pytest.raises(ValueError, match='source/runtime|runtime.*changed'):
        prepare_inputs(portable / 'wrong-runtime.json', portable / 'wrong-runtime-output')
    assert not (portable / 'wrong-runtime-output').exists()
    assert not torch.cuda.is_initialized()
    (tmp_path / 'export-input-proof.json').write_text(json.dumps(dict(status='PASS_CPU', family=family,
        evidence_kind='SYNTHETIC_TEST_ONLY', current_student_differs_from_teacher=True,
        original_assets_and_all_checkpoints_actually_deleted=True, copied_package_nonrepo_cwd=True,
        public_cli_new_prompt_after_deletion=True, public_cli_new_raw_media_after_deletion=True,
        native_refusals=sorted(invalid) + ['runtime'],
        no_hand_published_features=True, exact_current_student_decoded_output=True,
        prepared=str(prepared), raw_prepared=raw_result['manifest'], actual=actual, raw_actual=raw_actual,
        gpu_execution='NOT_RUN', pretrained_execution='NOT_RUN', quality_validation='NOT_RUN'), indent=2) + '\n')
