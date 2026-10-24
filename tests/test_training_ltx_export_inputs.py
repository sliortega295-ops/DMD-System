"""Actual native codecs from current-G export, with no training asset fallback."""
import gc
import json
from pathlib import Path

import pytest
import torch

from distillschedule.data import PreparedDataset
from distillschedule.data.export_inputs import SCHEMA
from distillschedule.training.checkpoint import rng_state
from distillschedule.training.config import load_training_config
from distillschedule.training.session import construct
from test_training_end_to_end import equal, run_cli
from test_training_ltx_cpu import assets, source_root


def _new_raw_media(root):
    import wave
    import numpy as np
    from PIL import Image
    root.mkdir(); frames = root / 'frames'; frames.mkdir()
    grid = np.arange(8 * 8 * 3).reshape(8, 8, 3)
    for index in range(3):
        Image.fromarray(((grid * 7 + index * 29 + 11) % 256).astype('uint8')).save(frames / f'{index:03d}.png')
    Image.fromarray(((grid * 11 + 73) % 256).astype('uint8')).save(root / 'image.png')
    time = np.arange(6400) / 16000
    samples = np.stack((.1 * np.sin(2 * np.pi * 440 * time), .2 * np.cos(2 * np.pi * 550 * time)), axis=1)
    with wave.open(str(root / 'audio.wav'), 'wb') as stream:
        stream.setnchannels(2); stream.setsampwidth(2); stream.setframerate(16000)
        stream.writeframes(np.rint(samples * 32767).astype('<i2').tobytes())
    (root / 'prompt.txt').write_text('a dog rests')
    return dict(id='fresh-raw-av', image=str(root / 'image.png'), video=str(frames), audio=str(root / 'audio.wav'),
                text={'path': str(root / 'prompt.txt')}, metadata={'geometry': {'fps': 8., 'conditioning': {
                    'video_prefix_latent_frames': 1, 'audio_prefix_latent_frames': 1}}})


def test_actual_export_callback_owns_native_media(assets):
    components, _, _, _ = construct(load_training_config(assets['lora_config']))
    projection = components.inference_projection(components, {})
    assert projection.components.encode_input is not None
    assert projection.components.state['media'] is components.state['media']
    assert projection.components.state['student'] is components.state['student']


@pytest.mark.parametrize('config_key', ['config', 'lora_config'])
def test_current_g_export_only_raw_prompt_prepare_and_av_generate(assets, tmp_path, config_key, monkeypatch):
    from distillschedule.data.ltx_inputs import prepare_inputs as prepare_native, SCHEMA as NATIVE_SCHEMA
    from distillschedule.data.ltx_inputs import encode_record
    from distillschedule.data.decoding import decode_record
    from distillschedule.data.manifest import raw_record
    from distillschedule.data.export_inputs import prepare_inputs as prepare_export
    from distillschedule.training.session import _load_for_output, _generate_values
    config = load_training_config(assets[config_key])
    trained = run_cli('train', config.source, '--cycles', 1, '--output', tmp_path / 'trained')
    exported = run_cli('export', config.source, '--checkpoint', trained['checkpoint'], '--output', tmp_path / 'export')
    prompt = dict(id='unseen-native-prompt', prompt='a dog rests', geometry=dict(
        fps=8., video_shape=[1, 128, 2, 2, 2], audio_shape=[1, 8, 11, 16],
        conditioning=dict(video_prefix_latent_frames=0, audio_prefix_latent_frames=0)))
    old_source = tmp_path / 'before-retiring-assets.json'
    old_source.write_text(json.dumps(dict(schema_version=NATIVE_SCHEMA, mode='prompts',
        assets=str(Path(assets['assets']) / 'assets.json'), source_root=config.model['parameters']['source_root'],
        records=[prompt])))
    old_prepared = prepare_native(old_source, tmp_path / 'before-retiring-prepared')
    expected_features = PreparedDataset(old_prepared).sample(0)
    components = _load_for_output(config, trained['checkpoint'])[0]
    asset_manifest = json.loads((Path(assets['assets']) / 'assets.json').read_text())
    training_row = PreparedDataset(assets['prepared']).sample(0)
    training_record = json.loads((Path(assets['root']) / 'inputs.json').read_text())['records'][0]
    training_sources = raw_record(training_record, Path(assets['root']))[1]['media']
    inference = dict(schema_version='distillschedule.training.v1',
        model=dict(factory=exported['model']['factory'], parameters=exported['model']['parameters']),
        algorithm=dict(name='dmd'), data=dict(prepared=str(tmp_path / 'new-prompts/manifest.json')),
        execution=dict(adapter='program', mode='reference'),
        trainer=dict(cycles=1, seed=config.trainer['seed'], device='cpu', output_dir=str(tmp_path / 'unused')))
    request = tmp_path / 'inference.json'; request.write_text(json.dumps(inference))
    retired = []
    # Keep external source and portable export. Retire even original raw media;
    # create genuinely new image/history, frames, WAV and text AFTER retirement.
    for path in (Path(assets['assets']), Path(assets['prepared']).parent, Path(assets['root']) / 'raw',
                 tmp_path / 'trained', Path(old_prepared).parent):
        hidden = path.with_name(path.name + '-retired')
        path.rename(hidden); retired.append((hidden, path))
    try:
        raw = _new_raw_media(tmp_path / 'new-media')
        resolved, identities = raw_record(raw, tmp_path)
        for name in ('image', 'video', 'audio'):
            assert identities['media'][name]['sha256'] != training_sources[name]['sha256']
        # Independent original preparation API, with the already loaded frozen
        # native codecs. No exported callback or unavailable asset file is used.
        expected_raw = encode_record(decode_record(resolved), dict(mode='raw', source_dir=str(tmp_path),
            source_root=config.model['parameters']['source_root'], asset_manifest=asset_manifest),
            media=components.state['media'])
        for name in ('video', 'audio'):
            assert not torch.equal(expected_raw['latents'][name], training_row['latents'][name])
            assert bool(expected_raw['conditioning_mask'][name].any())
        expected = {mode: _generate_values(config, components, row, {}, output_kind='audio_video')
                    for mode, row in [('prompts', expected_features), ('raw', expected_raw)]}
        del components
        gc.collect()
        outputs, generated_outputs = {}, {}
        for mode, record, expected_row in [('prompts', prompt, expected_features), ('raw', raw, expected_raw)]:
            source = tmp_path / (mode + '.json')
            source.write_text(json.dumps(dict(schema_version=SCHEMA, weights=str(tmp_path / 'export'),
                                             mode=mode, records=[record])))
            before = rng_state(device='cpu')
            # Public CLI, separate process, actual restored encoder callback.
            result = run_cli('data', 'prepare', source, '--output', tmp_path / ('new-' + mode),
                             '--preparer', 'distillschedule.data.export_inputs:prepare_inputs')
            prepared = result['manifest']
            dataset = PreparedDataset(prepared)
            assert equal(expected_row, dataset.sample(0)), 'same exact native codec/conditioning/geometry ABI'
            assert dataset.validate(sources=True)['source_validation'] == 'PASS'
            assert equal(before, rng_state(device='cpu'))
            assert dataset.manifest['metadata']['_export_inputs']['weights'] == str(tmp_path / 'export')
            outputs[mode] = prepared
            generated = run_cli('generate', request, '--weights', tmp_path / 'export', '--prepared', prepared,
                                '--output', tmp_path / ('new-av-' + mode), '--output-kind', 'audio_video')
            assert generated['sample_id'] == record['id']
            assert equal(expected[mode], torch.load(generated['artifact'], weights_only=True))
            assert Path(generated['audio']['path']).is_file()
            assert len(list(Path(generated['video']['frames']).glob('*.png'))) == 3
            generated_outputs[mode] = generated
        assert PreparedDataset(outputs['raw']).manifest['provenance']['sources'][0]['media'] == identities['media']
        # Unlike parent-state checks around CLI subprocesses, this runs actual
        # restored native encoding in the observed RNG process.
        before = rng_state(device='cpu')
        in_process = prepare_export(tmp_path / 'raw.json', tmp_path / 'in-process-raw')
        assert equal(before, rng_state(device='cpu'))
        assert equal(expected_raw, PreparedDataset(in_process).sample(0))
        from distillschedule.training import ltx_media
        original_sha = ltx_media.file_sha256
        for helper in ('ltx_inputs.py', 'ltx_media.py', 'ltx_source.py'):
            with monkeypatch.context() as patch:
                patch.setattr(ltx_media, 'file_sha256', lambda path: '0' * 64
                              if Path(path).name == helper else original_sha(path))
                output = tmp_path / ('invalid-' + helper)
                # Factory ltx.py is unchanged: its helper hash must still bind
                # exported conditions before any codec executes or publishes.
                patch.setattr(ltx_media.NativeMedia, 'encode_text',
                              lambda *a, **kw: pytest.fail('stale codec must not execute'))
                with pytest.raises(ValueError, match='export model metadata changed'):
                    prepare_export(tmp_path / 'raw.json', output)
                assert not output.exists()
        (tmp_path / 'acceptance-proof.json').write_text(json.dumps(dict(
            scope='ACTUAL_TINY_NATIVE_CPU_CURRENT_G_EXPORT_NOT_PRETRAINED_QUALITY',
            config_key=config_key, complete_update_counts=trained['final_versions'],
            new_features_exact=True, new_current_g_av_exact=True,
            training_assets_features_checkpoints_absent=True,
            new_raw_source_hashes=identities['media'], native_in_process_rng_preserved=True,
            changed_encoding_helpers_refused=True,
            export=exported, prepared=outputs, generated=generated_outputs), indent=2))
    finally:
        for hidden, original in reversed(retired):
            hidden.rename(original)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('fault', ['missing_geometry', 'invent_history', 'wrong_binding', 'preencoded', 'missing_media'])
def test_export_inputs_reject_missing_conditions_without_mutation(assets, tmp_path, fault):
    components, _, _, _ = construct(load_training_config(assets['lora_config']))
    projection = components.inference_projection(components, {})
    callback = projection.components.encode_input
    assert callback is not None
    parameters = dict(mode='prompts', source_dir=str(tmp_path), model_metadata=dict(projection.components.metadata))
    record = dict(id='new', prompt='a dog rests', geometry=dict(fps=8.,
        video_shape=[1, 128, 2, 2, 2], audio_shape=[1, 8, 11, 16],
        conditioning=dict(video_prefix_latent_frames=0, audio_prefix_latent_frames=0)))
    if fault == 'missing_geometry':
        del record['geometry']['video_shape']
    elif fault == 'invent_history':
        record['geometry']['conditioning']['video_prefix_latent_frames'] = 1
    elif fault == 'wrong_binding':
        parameters['model_metadata']['media_binding'] = '0' * 64
    elif fault == 'preencoded':
        parameters['mode'] = 'preencoded'
    else:
        parameters['mode'] = 'raw'
        record = dict(id='new', text={'value': 'a dog rests'}, video=str(tmp_path / 'MISSING'),
                      metadata={'geometry': record['geometry']})
    from distillschedule.training.validation import _guard
    before = _guard(projection.components.state); outer = rng_state(device='cpu')
    with pytest.raises(ValueError):
        callback(projection.components.state, record, parameters)
    assert _guard(projection.components.state) == before and equal(outer, rng_state(device='cpu'))
