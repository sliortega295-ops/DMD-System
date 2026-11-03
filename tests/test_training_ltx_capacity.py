"""Explicit all-component CPU budgets; actual native meta witnesses, never estimates."""
from copy import deepcopy
import json
from pathlib import Path
import runpy
import subprocess
import sys

import pytest
import torch

from test_training_ltx_cpu import assets, source_root
from test_training_ltx_original_checkpoint import native_checkpoint, seventeen_layer_checkpoint
from test_training_end_to_end import equal, run_cli

REPO = Path(__file__).resolve().parents[1]
ROLE_KEYS = ('student', 'fake', 'teacher')
MEDIA_KEYS = ('video_encoder', 'video_decoder', 'audio_encoder', 'audio_decoder', 'vocoder',
              'text_encoder', 'text_processor')


@pytest.fixture(scope='module')
def budget_assets(tmp_path_factory, source_root):
    from distillschedule.training.ltx_source import original
    from distillschedule.training.ltx_media import NativeMedia, _video_config, constructor, file_sha256
    recipe = runpy.run_path(str(REPO / 'recipes/ltx25/create_cpu_fixture.py'))
    files = recipe['build_fixture'](tmp_path_factory.mktemp('budget-ltx') / 'fixture', source_root, cycles=2)
    architecture, media_architecture = recipe['tiny_architectures']()
    architecture['num_layers'] = 17
    # Still tiny, but this actual decoder also exceeds the legacy 16-block cap.
    media_architecture['video_decoder']['decoder_blocks'] = [['res_x', 1]] * 16 + [['compress_all', {'multiplier': 1}]]
    api = original(source_root)
    manifest_path = Path(files['assets']) / 'assets.json'
    manifest = json.loads(manifest_path.read_text())
    counts = {}
    with torch.random.fork_rng(devices=[]), torch.device('cpu'):
        torch.manual_seed(1123)
        # Actual upstream constructor, not the adapter whose old cap is under test.
        config = deepcopy(architecture); config['model_type'] = api.model.LTXModelType(config['model_type'])
        config['rope_type'] = api.source.import_module('ltx_core.model.transformer.rope').LTXRopeType(config['rope_type'])
        native = api.model.LTXModel(**config).float()
        for name, parameter in native.named_parameters():
            if 'scale_shift_table' in name:
                torch.nn.init.normal_(parameter, std=.02)
        count = sum(p.numel() for p in native.parameters())
        assert 0 < count < 3_000_000
        for role in ROLE_KEYS:
            path = manifest_path.parent / (role + '.pt'); torch.save(native.state_dict(), path)
            counts[role] = count
            manifest['roles'][role] = dict(config=architecture, weights=dict(path=path.name, sha256=file_sha256(path)))
        _, original_media = recipe['tiny_architectures']()
        media = NativeMedia(api.source, original_media)
        decoder = api.source.import_module('ltx_core.model.video_vae.conv_video_decoder')
        media.video_decoder = constructor(decoder.ConvVideoDecoder,
            _video_config(api.source, media_architecture['video_decoder']),
            ('decoder_blocks', 'patch_size', 'base_channels', 'norm_layer', 'decoder_spatial_padding_mode'))
        for name in MEDIA_KEYS:
            module = getattr(media, name)
            path = manifest_path.parent / (name + '.pt'); torch.save(module.state_dict(), path)
            counts[name] = sum(p.numel() for p in module.parameters())
            manifest['media'][name] = dict(config=media_architecture[name], weights=dict(path=path.name, sha256=file_sha256(path)))
    manifest['cpu_parameter_limits'] = counts
    manifest_path.write_text(json.dumps(manifest))
    return dict(files=files, api=api, architecture=architecture, media_architecture=media_architecture,
                manifest=manifest, counts=counts)


def request(budget_assets):
    value = json.loads(Path(budget_assets['files']['config']).read_text())
    return dict(value['model']['parameters'], device='cpu', seed=17)


def manifest_copy(budget_assets, tmp_path, value):
    # Asset paths remain confined to the original fixture; only the manifest differs.
    root = Path(budget_assets['files']['assets'])
    path = root / (tmp_path.name + '.json')
    path.write_text(json.dumps(value))
    return path


def forbid_materialization(monkeypatch):
    original = torch.nn.Module.register_parameter
    seen = []
    def parameter(self, name, value):
        if value is not None:
            assert value.is_meta, 'CPU parameter materialized before complete budget admission'
            seen.append(type(self).__name__)
        return original(self, name, value)
    def payload(*args, **kwargs):
        pytest.fail('weight payload reached before complete budget admission')
    monkeypatch.setattr(torch.nn.Module, 'register_parameter', parameter)
    monkeypatch.setattr(torch, 'load', payload)
    return seen


def test_ltx_budget_opt_in_admits_real_meta_sized_seventeen_layer_bundle(budget_assets):
    from distillschedule.training.ltx import build_model
    model = build_model(request(budget_assets))
    assert model.metadata['cpu_parameter_limits'] == budget_assets['counts']
    assert model.state['student'].architecture['num_layers'] == 17
    assert type(model.state['student'].native) is budget_assets['api'].model.LTXModel
    for name in ROLE_KEYS:
        assert sum(p.numel() for p in model.state[name].native.parameters()) == budget_assets['counts'][name]
        assert model.state[name].cpu_parameter_limits == budget_assets['counts']
    for name in MEDIA_KEYS:
        assert sum(p.numel() for p in getattr(model.state['media'], name).parameters()) == budget_assets['counts'][name]
    assert all(not t.is_meta and bool(torch.isfinite(t).all()) for name in (*ROLE_KEYS, 'media')
               for t in (*model.state[name].parameters(), *model.state[name].buffers()))
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('component', (*ROLE_KEYS, *MEDIA_KEYS))
def test_ltx_all_component_limits_refuse_before_payload_or_concrete(budget_assets, tmp_path, monkeypatch, component):
    from distillschedule.training.ltx import build_model
    value = deepcopy(budget_assets['manifest'])
    value['cpu_parameter_limits'][component] -= 1
    path = manifest_copy(budget_assets, tmp_path, value)
    seen = forbid_materialization(monkeypatch)
    with pytest.raises(ValueError, match='CPU parameter budget.*' + component):
        build_model(dict(request(budget_assets), assets=str(path)))
    assert seen, 'the refusal must use real native meta constructors'
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('bad', [None, {}, True, [], 'limits', 'missing', 'unknown', 'bool_value', 0, -1, 1.5, '10000000'])
def test_ltx_invalid_limit_maps_reject_before_any_constructor(budget_assets, tmp_path, monkeypatch, bad):
    from distillschedule.training.ltx import build_model
    value = deepcopy(budget_assets['manifest'])
    limits = value['cpu_parameter_limits']
    if bad == 'missing':
        limits.pop('text_processor')
    elif bad == 'unknown':
        limits['surprise'] = 1
    elif bad == 'bool_value':
        limits['text_processor'] = True
    elif type(bad) in (int, float) or bad == '10000000':
        limits['text_processor'] = bad
    else:
        value['cpu_parameter_limits'] = bad
    path = manifest_copy(budget_assets, tmp_path, value)
    def forbidden(*args, **kwargs):
        pytest.fail('invalid parameter map reached a native constructor or payload')
    monkeypatch.setattr(torch.nn.Module, 'register_parameter', forbidden)
    monkeypatch.setattr(torch, 'load', forbidden)
    with pytest.raises(ValueError, match='cpu_parameter_limits'):
        build_model(dict(request(budget_assets), assets=str(path)))


def test_ltx_legacy_seventeen_layer_refusal_is_unchanged(source_root):
    from distillschedule.training.ltx import native_transformer
    from distillschedule.training.ltx_source import original
    architecture, _ = runpy.run_path(str(REPO / 'recipes/ltx25/create_cpu_fixture.py'))['tiny_architectures']()
    architecture['num_layers'] = 17
    with pytest.raises(ValueError, match='bounded explicit native architecture'):
        native_transformer(original(source_root).source, architecture)


@pytest.mark.parametrize('component', ('student', *MEDIA_KEYS))
def test_ltx_portable_limits_reject_before_concrete_owner(budget_assets, monkeypatch, component):
    from distillschedule.training.ltx import build_inference_model
    parameters = dict(source_root=str(budget_assets['api'].source.root),
        architecture=budget_assets['architecture'], media_architecture=budget_assets['media_architecture'],
        binding='00' * 32, trainable=dict(mode='full'), generation_sigmas=[1., 0.],
        cpu_parameter_limits=deepcopy(budget_assets['counts']))
    parameters['cpu_parameter_limits'][component] -= 1
    seen = forbid_materialization(monkeypatch)
    with pytest.raises(ValueError, match='CPU parameter budget.*' + component):
        build_inference_model(parameters)
    assert seen


def test_ltx_portable_missing_budget_cannot_fall_back_to_tiny_defaults(budget_assets):
    from distillschedule.training.ltx import build_inference_model
    parameters = dict(source_root=str(budget_assets['api'].source.root),
        architecture=budget_assets['architecture'], media_architecture=budget_assets['media_architecture'],
        binding='00' * 32, trainable=dict(mode='full'), generation_sigmas=[1., 0.])
    with pytest.raises(ValueError, match='bounded explicit native CPU video codec'):
        build_inference_model(parameters)


@pytest.mark.parametrize('component', ['teacher', 'text_processor'])
def test_ltx_original_role_all_component_budget_precedes_any_payload(
        budget_assets, native_checkpoint, tmp_path, monkeypatch, component):
    from distillschedule.training.ltx import build_model
    from distillschedule.training.ltx_media import file_sha256
    checkpoint, _, count = seventeen_layer_checkpoint(native_checkpoint, Path(budget_assets['files']['assets']))
    manifest = deepcopy(budget_assets['manifest'])
    for name in ROLE_KEYS:
        manifest['roles'][name] = dict(format='ltx_original_safetensors', checkpoint=dict(
            path=checkpoint.name, sha256=file_sha256(checkpoint)), cpu_max_parameters=count, dtype_policy='strict_fp32')
        manifest['cpu_parameter_limits'][name] = count
    manifest['cpu_parameter_limits'][component] -= 1
    path = manifest_copy(budget_assets, tmp_path, manifest)
    seen = forbid_materialization(monkeypatch)
    loader = budget_assets['api'].source.import_module('ltx_core.loader.sft_loader').SafetensorsModelStateDictLoader
    def payload(*args, **kwargs):
        pytest.fail('original safetensors payload reached before all-component admission')
    monkeypatch.setattr(loader, 'load', payload)
    with pytest.raises(ValueError, match='CPU parameter budget.*' + component):
        build_model(dict(request(budget_assets), assets=str(path)))
    assert seen


@pytest.mark.parametrize('precision', ['fp32', 'bf16-mixed'])
@pytest.mark.parametrize('config_key', ['config', 'lora_config'])
def test_ltx_equal_budgets_preserve_native_initial_rng_loss_gradient_and_update(assets, tmp_path, precision, config_key):
    from distillschedule.training.config import load_training_config
    from distillschedule.training.session import construct, build_cycle, _construct_model
    from distillschedule.training.checkpoint import pack_state, rng_state
    from distillschedule.open_workload.torch_backend import Session
    from test_training_ltx_precision import native_prediction
    value = json.loads(Path(assets[config_key]).read_text())
    value['model']['parameters']['compute_precision'] = precision
    path = tmp_path / 'request.json'; path.write_text(json.dumps(value))
    control = load_training_config(path)
    model, _, _, _ = construct(control)
    counts = {name: sum(p.numel() for p in model.state[name].native.parameters()) for name in ROLE_KEYS}
    counts.update({name: sum(p.numel() for p in getattr(model.state['media'], name).parameters()) for name in MEDIA_KEYS})
    # Budgets apply to the native base, before the existing trainable LoRA scope.
    if config_key == 'lora_config':
        for role in ROLE_KEYS:
            counts[role] = sum(p.numel() for p in model.state['teacher'].native.parameters())
    manifest_path = Path(assets['assets']) / (tmp_path.name + '.json')
    manifest = json.loads((Path(assets['assets']) / 'assets.json').read_text())
    manifest['cpu_parameter_limits'] = counts
    manifest_path.write_text(json.dumps(manifest))
    value['model']['parameters']['assets'] = str(manifest_path)
    path = tmp_path / 'budget-request.json'; path.write_text(json.dumps(value))
    runs = []
    for config in (control, load_training_config(path)):
        torch.random.default_generator.manual_seed(419)
        before = rng_state(device='cpu')
        model, _ = _construct_model(config)
        assert equal(before, rng_state(device='cpu')), 'sizing/construction may not consume caller RNG'
        initial = deepcopy(pack_state(model.state))
        from distillschedule.data import PreparedDataset
        data = PreparedDataset(assets['prepared'])
        row = data.sample(0); role = model.state['student']; sigma = torch.tensor([.625])
        prediction = role.denoise(row['latents'], sigma, row, conditional=True)
        with role.compute_context():
            native = native_prediction(role, row, sigma)
        assert equal(prediction, {key: tensor.float() for key, tensor in zip(('video', 'audio'), native)})
        loss = sum(t.square().mean() for t in prediction.values())
        loss.backward()
        gradients = {name:p.grad.detach().clone() for name,p in role.named_parameters() if p.grad is not None}
        model.state['student_optimizer'].zero_grad(set_to_none=True)
        from distillschedule.training.dmd import build_algorithm
        workload = build_cycle(config, model, data, build_algorithm, cycle=0, cursor=0)
        session = Session(workload, dict(backend='torch-callable-resident-v1', share_immutable=False,
            checkpoint_bindings=[]), copy_state=False)
        output = session.run()
        metadata = deepcopy(model.metadata)
        metadata.pop('cpu_parameter_limits', None)
        metadata['asset_identity'].pop('cpu_parameter_counts', None)
        metadata['asset_identity'].pop('manifest')
        runs.append(dict(initial=initial, prediction={k:v.detach().clone() for k,v in prediction.items()},
            loss=loss.detach(), gradients=gradients, output=output, after=deepcopy(pack_state(model.state)),
            rng=rng_state(device='cpu'), metadata=metadata))
    assert equal(*runs), 'an equal budget must not change any native value, gradient, update, RNG or logical identity'
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('config_key', ['config', 'lora_config'])
def test_ltx_budgeted_native_new_process_resume_and_current_student_av_export(
        budget_assets, native_checkpoint, tmp_path, config_key):
    from distillschedule.training.ltx_media import file_sha256
    from distillschedule.data.ltx_inputs import prepare_inputs
    from distillschedule.data import PreparedDataset
    from distillschedule.training.plugins import load_callable
    from distillschedule.training.inference import read_export, restore_modules
    files = budget_assets['files']
    manifest = deepcopy(budget_assets['manifest'])
    root = Path(files['root']); assets_path = Path(files['assets'])
    if config_key == 'config':
        # Full training also exercises actual original-format native 17-layer import.
        checkpoint, _, count = seventeen_layer_checkpoint(native_checkpoint, assets_path)
        for name in ROLE_KEYS:
            manifest['roles'][name] = dict(format='ltx_original_safetensors', checkpoint=dict(
                path=checkpoint.name, sha256=file_sha256(checkpoint)), cpu_max_parameters=count, dtype_policy='strict_fp32')
            manifest['cpu_parameter_limits'][name] = count
    manifest_path = manifest_copy(budget_assets, tmp_path, manifest)
    inputs = json.loads((root / 'inputs.json').read_text()); inputs['assets'] = str(manifest_path)
    source = root / (tmp_path.name + '-inputs.json'); source.write_text(json.dumps(inputs))
    prepared = prepare_inputs(source, tmp_path / 'prepared')
    value = json.loads(Path(files[config_key]).read_text())
    value['model']['parameters']['assets'] = str(manifest_path)
    value['data']['prepared'] = str(prepared)
    config = tmp_path / 'train.json'; config.write_text(json.dumps(value))
    first = run_cli('train', config, '--cycles', 1, '--output', tmp_path / 'first')
    changed = deepcopy(manifest); changed['cpu_parameter_limits']['teacher'] += 1
    changed_path = assets_path / (tmp_path.name + '-changed-limits.json'); changed_path.write_text(json.dumps(changed))
    wrong = deepcopy(value); wrong['model']['parameters']['assets'] = str(changed_path)
    wrong_path = tmp_path / 'changed-limits.json'; wrong_path.write_text(json.dumps(wrong))
    rejected = subprocess.run([sys.executable, '-m', 'distillschedule', 'train', str(wrong_path),
        '--cycles', '2', '--resume', first['checkpoint'], '--output', str(tmp_path / 'wrong-resume')],
        capture_output=True, text=True, timeout=180)
    assert rejected.returncode != 0 and 'identity' in rejected.stdout + rejected.stderr
    assert not (tmp_path / 'wrong-resume').exists()
    resumed = run_cli('train', config, '--cycles', 2, '--resume', first['checkpoint'], '--output', tmp_path / 'resumed')
    whole = run_cli('train', config, '--cycles', 2, '--output', tmp_path / 'whole')
    a = torch.load(Path(resumed['checkpoint']) / 'state.pt', weights_only=True)
    b = torch.load(Path(whole['checkpoint']) / 'state.pt', weights_only=True)
    assert equal(a, b), 'exact current roles, Adam, EMA, clocks, data and RNG continuation required'
    assert a['state']['dmd_counters']['value']['counters'].tolist() == [2, 2, 10]
    exported = run_cli('export', config, '--checkpoint', resumed['checkpoint'], '--output', tmp_path / 'export')
    parameters = exported['model']['parameters']
    assert parameters['cpu_parameter_limits'] == manifest['cpu_parameter_limits']
    assert parameters['architecture']['num_layers'] == 17
    assert set(exported['model']['modules']) == {'student', 'media'} and 'assets' not in parameters
    factory, _ = load_callable(exported['model']['factory'])
    current = factory(parameters)
    export_root, receipt = read_export(tmp_path / 'export')
    restore_modules(current, export_root, receipt)
    # Native nonpersistent buffers do not appear in weights.pt; the real
    # constructor must initialize them, not an unchecked meta/to_empty shell.
    buffers = [(module, name, getattr(module, name)) for module in current.state['media'].modules()
               for name in module._non_persistent_buffers_set]
    assert buffers and all(not value.is_meta and bool(torch.isfinite(value).all()) for _, _, value in buffers)
    encoded = current.state['media'].encode_text(['a dog rests'])
    assert all(bool(torch.isfinite(value).all()) for value in encoded.values())
    for name in ('student', 'media'):
        assert equal(current.state[name].state_dict(), a['state'][name]['value'])
    new_source = tmp_path / 'new-inputs.json'
    new_source.write_text(json.dumps(dict(schema_version='distillschedule.export_inputs.v1', weights=str(export_root),
        mode='prompts', records=[dict(id='budget-new-prompt', prompt='a dog rests', geometry=dict(
            fps=8., video_shape=[1, 128, 2, 2, 2], audio_shape=[1, 8, 11, 16],
            conditioning=dict(video_prefix_latent_frames=0, audio_prefix_latent_frames=0)))])))
    hidden = assets_path.with_name(assets_path.name + '-hidden')
    assets_path.rename(hidden)
    try:
        new_prepared = run_cli('data', 'prepare', new_source, '--output', tmp_path / 'new-prepared',
            '--preparer', 'distillschedule.data.export_inputs:prepare_inputs')['manifest']
        inference = dict(schema_version='distillschedule.training.v1',
            model=dict(factory=exported['model']['factory'], parameters=parameters), algorithm=dict(name='dmd'),
            data=dict(prepared=new_prepared), execution=dict(mode='reference'),
            trainer=dict(cycles=1, seed=909, device='cpu', output_dir=str(tmp_path / 'unused')))
        request_path = tmp_path / 'inference.json'; request_path.write_text(json.dumps(inference))
        generated = run_cli('generate', request_path, '--weights', export_root, '--prepared', new_prepared,
            '--output', tmp_path / 'new-av', '--output-kind', 'audio_video')
    finally:
        hidden.rename(assets_path)
    for module in current.state.values():
        module.eval()
    torch.random.default_generator.manual_seed(909)
    with torch.no_grad():
        expected = current.generate(current.state, PreparedDataset(new_prepared).sample(0), {})
        decoded = current.decode(expected, {})
    actual = torch.load(generated['artifact'], weights_only=True)
    assert equal(expected, actual['generated']) and equal(decoded, actual['decoded'])
    assert decoded['video'].shape == (3, 8, 8, 3) and decoded['audio']['samples'].shape == (9840, 2)
    (tmp_path / 'acceptance-proof.json').write_text(json.dumps(dict(
        schema_version='distillschedule.ltx25_budgeted_cpu_proof.v1', config_key=config_key,
        native_layers=17, cpu_parameter_limits=parameters['cpu_parameter_limits'],
        source=budget_assets['api'].identity, complete_cycle_state_rng_adam_ema_exact=True,
        changed_limits_resume_refused=True, current_student_export_and_new_process_av_exact=True,
        training_assets_hidden_during_export_only_preparation_and_generation=True,
        nonpersistent_media_buffer_count=len(buffers), final_counters=[2, 2, 10],
        resumed_checkpoint=resumed['checkpoint'], checkpoint_sha256=file_sha256(Path(resumed['checkpoint']) / 'state.pt'),
        whole_checkpoint_sha256=file_sha256(Path(whole['checkpoint']) / 'state.pt'),
        export=str(export_root), generated=generated, native_original_roles=config_key == 'config',
        quality_validation='NOT_RUN', gpu_execution='NOT_RUN'), indent=2))
    assert not torch.cuda.is_initialized()
