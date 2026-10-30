"""Actual pinned LTX-2.5 classes, native raw media, DMD and fresh-process lifecycle.

Optional source only: absent LTX_SOURCE_ROOT skips. Once set, missing optional
dependencies are failures, not hidden skips. No GPU or pretrained-quality claim.
"""
from copy import deepcopy
import json
import os
from pathlib import Path

import pytest
import torch

from distillschedule.data import PreparedDataset
from distillschedule.training.config import load_training_config
from distillschedule.training.session import construct, build_cycle
from test_training_end_to_end import equal, run_cli

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def source_root():
    value = os.environ.get('LTX_SOURCE_ROOT')
    if not value:
        pytest.skip('set LTX_SOURCE_ROOT for actual upstream native LTX acceptance')
    return Path(value).resolve(strict=True)


@pytest.fixture(scope='module')
def assets(tmp_path_factory, source_root):
    from distillschedule.training.plugins import load_callable
    builder, _ = load_callable(str(REPO / 'recipes/ltx25/create_cpu_fixture.py') + ':build_fixture')
    return builder(tmp_path_factory.mktemp('native-ltx') / 'fixture', source_root, cycles=3)


def test_real_native_media_text_and_modality_abi(assets):
    from distillschedule.training.ltx_source import original
    config = load_training_config(assets['config'])
    components, dataset, algorithm, _ = construct(config)
    assert dataset.validate(sources=True)['source_validation'] == 'PASS'
    api = original(config.model['parameters']['source_root'])
    assert type(components.state['student'].native) is api.model.LTXModel
    assert components.prediction.kind == 'flow'
    media = components.state['media']
    assert type(media.video_encoder).__name__ == 'VideoEncoder'
    assert type(media.video_decoder).__name__ == 'ConvVideoDecoder'
    assert type(media.text_encoder.model).__name__ == 'Gemma4UnifiedForConditionalGeneration'
    assert type(media.text_processor.feature_extractor).__name__ == 'FeatureExtractorV2'
    feeds = dataset.sample(0)
    assert feeds['latents']['video'].shape == (1, 8, 128)
    assert feeds['latents']['audio'].shape == (1, 11, 128)
    assert feeds['positions']['video'].shape == (1, 3, 8, 2)
    assert feeds['positions']['audio'].shape == (1, 1, 11, 2)
    assert feeds['conditioning_mask']['video'].sum() == 4
    assert feeds['conditioning_mask']['audio'].sum() == 1
    actual_empty = media.encode_text([''])
    assert torch.equal(feeds['null_contexts']['video'], actual_empty['video_context'])
    assert torch.equal(feeds['null_contexts']['audio'], actual_empty['audio_context'])
    assert not torch.equal(feeds['contexts']['video'], feeds['contexts']['audio'])
    assert torch.count_nonzero(feeds['null_contexts']['video']) > 0
    seen = []
    hook = components.state['student'].native.register_forward_pre_hook(
        lambda m, a, k: seen.append(k), with_kwargs=True)
    workload = build_cycle(config, components, dataset, algorithm, cycle=0, cursor=0)
    from distillschedule.open_workload.torch_backend import Session
    session = Session(workload, {'backend': 'torch-callable-resident-v1',
        'share_immutable': False, 'checkpoint_bindings': []}, copy_state=False)
    old = {k: p.detach().clone() for k, p in components.state['student'].named_parameters()}
    teacher = deepcopy(components.state['teacher'].state_dict())
    output = session.run(); hook.remove()
    assert len(seen) == 4, 'four actual native Euler forwards for Student'
    for item in seen:
        assert type(item['video'].timesteps) is api.modality.TokenTimesteps
        assert type(item['audio'].timesteps) is api.modality.TokenTimesteps
        assert item['video'].timesteps.values[0, :4].tolist() == [0.] * 4
        assert item['audio'].timesteps.values[0, :1].tolist() == [0.]
    assert any(not torch.equal(p, old[k]) for k, p in components.state['student'].named_parameters())
    assert equal(teacher, components.state['teacher'].state_dict())
    assert components.state['dmd_counters'].tolist() == [1, 1, 5]
    assert session.receipt['dispatcher']['final_versions']['fake'] == 5
    assert all(bool(torch.isfinite(value)) for value in output.values())
    assert torch.cuda.is_initialized() is False


@pytest.mark.parametrize('config_key', ['config', 'lora_config'])
def test_fresh_process_exact_resume_and_current_generator_av_export(assets, tmp_path, config_key):
    config = assets[config_key]
    partial = run_cli('train', config, '--cycles', 1, '--output', tmp_path / 'partial')
    resumed = run_cli('train', config, '--cycles', 3, '--resume', partial['checkpoint'], '--output', tmp_path / 'resumed')
    whole = run_cli('train', config, '--cycles', 3, '--output', tmp_path / 'whole')
    a = torch.load(Path(resumed['checkpoint']) / 'state.pt', weights_only=True)
    b = torch.load(Path(whole['checkpoint']) / 'state.pt', weights_only=True)
    assert equal(a, b), 'all state, RNG, named Adam, schedulers, EMA, data and logical versions must be exact'
    clocks = a['state']['dmd_counters']
    assert clocks['kind'] == 'stateful' and clocks['class'] == 'distillschedule.training.dmd.DMDClocks'
    assert clocks['value']['schema_version'] == 'distillschedule.dmd_clocks.v1'
    assert clocks['value']['counters'].tolist() == [3, 3, 15]
    assert a['state']['student_scheduler']['value']['last_epoch'] == 3
    assert a['state']['fake_scheduler']['value']['last_epoch'] == 15
    assert set(a['state']) >= {'student', 'fake', 'teacher', 'student_optimizer', 'fake_optimizer', 'student_ema'}
    generated = run_cli('evaluate', config, '--checkpoint', resumed['checkpoint'], '--output', tmp_path / 'evaluation')
    original = torch.load(generated['artifact'], weights_only=True)
    assert original['decoded']['video'].shape == (3, 8, 8, 3)
    assert original['decoded']['audio']['samples'].shape == (9840, 2)
    exported = run_cli('export', config, '--checkpoint', resumed['checkpoint'], '--output', tmp_path / 'export')
    assert set(exported['model']['modules']) == {'student', 'media'}
    assert 'assets' not in exported['model']['parameters']
    from distillschedule.data.ltx_inputs import prepare_inputs, SCHEMA
    new_source = tmp_path / 'NEW-prompts.json'
    new_manifest = dict(schema_version=SCHEMA, mode='prompts',
        source_root=os.environ['LTX_SOURCE_ROOT'], assets=str(Path(assets['assets']) / 'assets.json'),
        records=[dict(id='new-native-prompt', prompt='a dog rests', geometry=dict(
            fps=8., video_shape=[1, 128, 2, 2, 2], audio_shape=[1, 8, 11, 16],
            conditioning=dict(video_prefix_latent_frames=0, audio_prefix_latent_frames=0)))])
    new_source.write_text(json.dumps(new_manifest))
    new_prepared = prepare_inputs(new_source, tmp_path / 'NEW-prepared')
    from distillschedule.training.session import _load_for_output
    trained = _load_for_output(load_training_config(config), resumed['checkpoint'])[0]
    for module in trained.state.values():
        if isinstance(module, torch.nn.Module):
            module.eval()
    torch.manual_seed(909)
    with torch.no_grad():
        expected = trained.generate(trained.state, PreparedDataset(new_prepared).sample(0), {})
        expected = {'generated': expected, 'decoded': trained.decode(expected, {})}
    inference = dict(schema_version='distillschedule.training.v1',
        model=dict(factory=exported['model']['factory'], parameters=exported['model']['parameters']),
        algorithm=dict(name='dmd'), data=dict(prepared=str(new_prepared)),
        execution=dict(adapter='program', mode='reference'),
        trainer=dict(cycles=1, seed=909, device='cpu', output_dir=str(tmp_path / 'unused')))
    path = tmp_path / 'inference.json'; path.write_text(json.dumps(inference))
    # The portable artifact has CURRENT G, all native codecs/text and tokenizer
    # configuration. It must not reload a training base checkpoint or teacher.
    assets_path = Path(assets['assets'])
    moved = assets_path.with_name(assets_path.name + '-offline')
    assets_path.rename(moved)
    try:
        import shutil
        shutil.copytree(resumed['checkpoint'], tmp_path / 'verified-checkpoint')
        evidence = tmp_path / 'restart-evidence'; evidence.mkdir()
        for directory in ('partial', 'resumed', 'whole'):
            shutil.copyfile(tmp_path / directory / 'metrics.jsonl', evidence / (directory + '-metrics.jsonl'))
            shutil.rmtree(tmp_path / directory)
        portable = run_cli('generate', path, '--weights', tmp_path / 'export', '--prepared', new_prepared,
                          '--output', tmp_path / 'generated', '--output-kind', 'audio_video')
    finally:
        moved.rename(assets_path)
    output = torch.load(portable['artifact'], weights_only=True)
    assert equal(expected, output)
    assert not torch.equal(original['generated']['video_latents'], output['generated']['video_latents'])
    assert Path(portable['audio']['path']).is_file()
    assert len(list(Path(portable['video']['frames']).glob('*.png'))) == 3
    (tmp_path / 'acceptance-proof.json').write_text(json.dumps(dict(
        schema_version='distillschedule.ltx25_cpu_acceptance_proof.v1',
        config_key=config_key, exact_payload_equality=True, compared_payload_fields=sorted(a),
        final_versions=whole['final_versions'], final_counters=clocks['value']['counters'].tolist(),
        clock_schema=clocks['value']['schema_version'], scheduler_schema='distillschedule.dmd_exponential_lr.v1',
        verified_checkpoint=str(tmp_path / 'verified-checkpoint'), new_prepared=str(new_prepared),
        export=str(tmp_path / 'export'), generated=portable, generation_matches_current_restored_student=True,
        original_training_bases_hidden_and_training_checkpoints_removed_during_portable_generation=True,
        runtime_scope='FP32_CPU1_REAL_NATIVE_CLASSES_RANDOM_WEIGHTS', quality_validation='NOT_RUN',
        gpu_execution='NOT_RUN'), indent=2))


def test_preencoded_exact_prompt_intake_and_no_invented_conditions(assets, tmp_path):
    from distillschedule.data.ltx_inputs import prepare_inputs, SCHEMA
    from distillschedule.training.ltx_media import file_sha256
    original = PreparedDataset(assets['prepared']).sample(0)
    feature = tmp_path / 'native.pt'; torch.save(original, feature)
    source = tmp_path / 'preencoded.json'
    value = dict(schema_version=SCHEMA, mode='preencoded', source_root=os.environ['LTX_SOURCE_ROOT'],
        assets=str(Path(assets['assets']) / 'assets.json'), records=[dict(id='native-exact',
        features=dict(path=feature.name, sha256=file_sha256(feature)))])
    source.write_text(json.dumps(value))
    prepared = prepare_inputs(source, tmp_path / 'preencoded')
    assert equal(original, PreparedDataset(prepared).sample(0))
    value.update(mode='prompts', records=[dict(id='absent-condition', prompt='a cat', geometry=dict(
        video_shape=[1, 128, 2, 2, 2], audio_shape=[1, 8, 11, 16], fps=8.,
        conditioning=dict(video_prefix_latent_frames=1, audio_prefix_latent_frames=0)))])
    source.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='invent clean conditioning'):
        prepare_inputs(source, tmp_path / 'must-not-publish')
    assert not (tmp_path / 'must-not-publish').exists()
    value['records'][0]['geometry'].pop('video_shape')
    value['records'][0]['geometry']['conditioning']['video_prefix_latent_frames'] = 0
    source.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='explicit native latent shape'):
        prepare_inputs(source, tmp_path / 'must-not-publish')


@pytest.mark.parametrize('prefix', [1, 2])
def test_image_only_clean_prefix_uses_only_actual_encoded_frames(assets, tmp_path, prefix):
    from distillschedule.data.ltx_inputs import prepare_inputs, SCHEMA
    root = Path(assets['root'])
    value = dict(schema_version=SCHEMA, mode='raw', source_root=os.environ['LTX_SOURCE_ROOT'],
        assets=str(Path(assets['assets']) / 'assets.json'), records=[dict(
            id='image-only', image=str(root / 'raw/image.png'), text={'value': 'a cat runs'},
            metadata=dict(geometry=dict(fps=8., video_shape=[1, 128, 2, 2, 2],
                audio_shape=[1, 8, 11, 16], conditioning=dict(
                    video_prefix_latent_frames=prefix, audio_prefix_latent_frames=0))))])
    source = tmp_path / 'image-only.json'; source.write_text(json.dumps(value))
    output = tmp_path / 'prepared'
    if prefix == 2:
        with pytest.raises(ValueError, match='image-backed'):
            prepare_inputs(source, output)
        assert not output.exists() and not list(tmp_path.glob('.prepared*'))
    else:
        prepared = prepare_inputs(source, output)
        row = PreparedDataset(prepared).sample(0)
        reference = PreparedDataset(assets['prepared']).sample(0)
        assert row['conditioning_mask']['video'].tolist() == [[True] * 4 + [False] * 4]
        assert not row['conditioning_mask']['audio'].any()
        assert torch.equal(row['conditioning']['video'][:, :4], reference['conditioning']['video'][:, :4])
        assert torch.count_nonzero(row['conditioning']['video'][:, :4]) > 0
        assert not row['conditioning']['video'][:, 4:].any()
        assert not row['latents']['video'].any()


@pytest.fixture(scope='module')
def complete_native_checkpoint(assets, tmp_path_factory):
    from distillschedule.training.checkpoint import save_checkpoint
    from distillschedule.training.policy import CyclePolicy
    config = load_training_config(assets['config'])
    model, dataset, algorithm, identity = construct(config)
    root = tmp_path_factory.mktemp('ltx-clock-checkpoint')
    workload = build_cycle(config, model, dataset, algorithm, cycle=0, cursor=0)
    policy = CyclePolicy(config.execution)
    session, _ = policy.run(workload, root)
    path = save_checkpoint(root / 'complete', model.state, identity=identity, cycle=1, cursor=1,
        versions=session.receipt['dispatcher']['final_versions'], policy=policy.bound)
    return config, path


@pytest.mark.parametrize('entrypoint', ['restore', 'output'])
@pytest.mark.parametrize('fault', ['counter', 'scheduler', 'version'])
def test_native_inconsistent_checkpoint_rejected_before_restore_or_output_mutation(
        complete_native_checkpoint, tmp_path, monkeypatch, entrypoint, fault):
    import shutil
    import distillschedule.training.session as lifecycle
    from distillschedule.training.checkpoint import pack_state, restore_checkpoint
    from test_training_dmd_flow import _rehash_checkpoint
    config, source = complete_native_checkpoint
    path = tmp_path / 'inconsistent'; shutil.copytree(source, path)
    payload = torch.load(path / 'state.pt', weights_only=True)
    if fault == 'counter':
        payload['state']['dmd_counters']['value']['counters'][2] = 4  # G1/F5, not four Fake updates.
    elif fault == 'scheduler':
        payload['state']['fake_scheduler']['value']['_step_count'] = 5  # Native count must be 5 + 1.
    else:
        payload['versions']['fake'] = 4
    _rehash_checkpoint(path, payload)
    model, dataset, algorithm, identity = construct(config)
    build_cycle(config, model, dataset, algorithm, cycle=0, cursor=0)
    with torch.no_grad():
        next(model.state['student'].parameters()).fill_(-.75)
    before = pack_state(model.state)
    if entrypoint == 'output':
        def output_construct(actual_config, *, include_validation):
            assert actual_config is config and include_validation is False
            return model, dataset, algorithm, identity
        monkeypatch.setattr(lifecycle, 'construct', output_construct)
    with pytest.raises(ValueError, match='complete-cycle|scheduler clock|logical.*version'):
        if entrypoint == 'restore':
            restore_checkpoint(path, model.state, identity=identity)
        else:
            lifecycle._load_for_output(config, path)
    assert equal(before, pack_state(model.state)), 'native AV roles/Adam/schedulers must remain untouched'
    assert not torch.cuda.is_initialized()


def test_native_unapplicable_controls_and_incomplete_checkpoint_fail_closed(assets, tmp_path, monkeypatch):
    from distillschedule.training.ltx import build_model
    from distillschedule.training.checkpoint import read_manifest
    config = load_training_config(assets['config'])
    parameters = dict(config.model['parameters'], device='cuda')
    with monkeypatch.context() as patch:
        patch.setattr(torch.cuda, 'is_available', lambda: False)
        with pytest.raises(ValueError, match='CUDA device is unavailable'):
            build_model(parameters)
    model, dataset, algorithm, identity = construct(config)
    with pytest.raises(ValueError, match='not complete'):
        read_manifest(tmp_path / 'incomplete', identity=identity)
    workload = build_cycle(config, model, dataset, algorithm, cycle=0, cursor=0)
    from distillschedule.open_workload.torch_backend import Session
    with pytest.raises(ValueError, match='checkpoint|replay|not admitted'):
        Session(workload, {'backend': 'torch-callable-resident-v1', 'share_immutable': False,
                          'checkpoint_bindings': ['student_forward']}, copy_state=False)
    assert model.state['dmd_counters'].tolist() == [0, 0, 0]
    assert torch.cuda.is_initialized() is False


def test_bad_native_conditions_shapes_and_mismatched_provenance_rejected(assets):
    components, dataset, _, _ = construct(load_training_config(assets['config']))
    student = components.state['student']
    good = dataset.sample(0)
    for mutation in ('null', 'shape', 'positions', 'binding', 'mask'):
        feeds = deepcopy(good)
        if mutation == 'null':
            feeds['null_contexts']['video'].zero_()
        elif mutation == 'shape':
            feeds['latents']['audio'] = feeds['latents']['audio'][..., :64]
        elif mutation == 'positions':
            feeds['positions']['video'][:, 0] += .1
        elif mutation == 'binding':
            feeds['encoder_identity'][0] ^= 1
        else:
            feeds['context_mask'] = torch.ones_like(feeds['context_mask'], dtype=torch.float32)
        with pytest.raises(ValueError):
            student.validate_feeds(feeds)


def test_actual_lora_separate_parameters_gradients_and_frozen_native_base(assets, tmp_path):
    from distillschedule.training.ltx import build_model, TrainableLowRankLinear
    from distillschedule.training.dmd import build_algorithm
    config = load_training_config(assets['config'])
    parameters = dict(config.model['parameters'])
    parameters['trainable'] = dict(mode='lora', rank=2, alpha=2., targets=[
        'transformer_blocks.0.attn1.to_q', 'transformer_blocks.0.audio_attn1.to_q'])
    components = build_model(parameters)
    wrapper = components.state['student'].native.transformer_blocks[0].attn1.to_q
    assert type(wrapper) is TrainableLowRankLinear
    x = torch.randn(1, 2, wrapper.base.in_features)
    assert torch.equal(wrapper(x), wrapper.base(x)), 'zero B is an actual unmerged baseline-preserving adapter'
    before = deepcopy(components.state['student'].state_dict())
    workload = build_algorithm(components, PreparedDataset(assets['prepared']).sample(0),
                               dict(config.algorithm['parameters']), 0, 987)
    from distillschedule.open_workload.torch_backend import Session
    session = Session(workload, {'backend': 'torch-callable-resident-v1',
        'share_immutable': False, 'checkpoint_bindings': []}, copy_state=False)
    session.run()
    changed = []
    for name, p in components.state['student'].named_parameters():
        if not torch.equal(p, before[name]):
            changed.append(name)
        if p.requires_grad:
            assert name.endswith(('.lora_a', '.lora_b'))
        else:
            assert torch.equal(p, before[name])
    assert changed and all(name.endswith('.lora_b') for name in changed)
    assert sum(p.numel() for p in wrapper.parameters() if p.requires_grad) == 2 * (wrapper.base.in_features + wrapper.base.out_features)
