"""Real local SD preprocessing must match the owner, not merely its shapes."""
from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest
import torch

from distillschedule.data import PreparedDataset
from distillschedule.data.sd_inputs import SCHEMA, prepare_inputs
from distillschedule.training.checkpoint import rng_state
from distillschedule.training.config import load_training_config
from distillschedule.training.session import construct, _construct_model
from distillschedule.workloads.best_known.registry import file_sha256
from test_training_dmd2_cpu import REPO, source_root, model_and_feeds
from test_training_end_to_end import equal


@pytest.fixture(params=('sd15', 'sdxl'))
def binding_assets(tmp_path, source_root, request):
    from distillschedule.training.plugins import load_callable
    builder, _ = load_callable(str(REPO / 'recipes/dmd2_sd15/create_cpu_fixture.py') + ':build_fixture')
    return builder(tmp_path / request.param, source_root, variant=request.param, cycles=1, ratio=2)


def _source(files, root, *, assets=None, raw=False, latents=False, declaration=True):
    from PIL import Image
    root.mkdir(parents=True, exist_ok=True)
    variant = json.loads(Path(files['config']).read_text())['model']['name']
    value = dict(schema_version=SCHEMA, variant=variant, model_id=str(assets or files['assets']),
                 records=[dict(id='new-image', prompt='first small sailboat')])
    if raw:
        image = root / 'pixels.png'; Image.new('RGB', (16, 16), (17, 53, 129)).save(image)
        value['records'][0]['images'] = [dict(path=str(image), sha256=file_sha256(image))]
        value['vae_encoding'] = dict(resolution=[16, 16], latent_shape=[4, 8, 8],
                                    posterior='mode', seed=193, dtype='float32')
    if latents:
        path = root / 'native.pt'; torch.save(torch.linspace(-1, 1, 256).reshape(1, 4, 8, 8), path)
        value['records'][0]['latents'] = dict(path=str(path), sha256=file_sha256(path))
        if declaration:
            value['preencoded_vae_compatibility'] = 'PREENCODED_LATENTS_MATCH_LOCAL_VAE'
    source = root / 'inputs.json'; source.write_text(json.dumps(value))
    return source


def _changed_bundle(files, root, component):
    from safetensors.torch import load_file, save_file
    assets = root / 'bundle-B'; shutil.copytree(files['assets'], assets)
    if component == 'tokenizer':
        # SDXL must bind the second tokenizer as well as the first.
        folder = 'tokenizer_2' if (assets / 'tokenizer_2').exists() else 'tokenizer'
        path = assets / folder / 'vocab.json'; vocabulary = json.loads(path.read_text())
        vocabulary['f'], vocabulary['s'] = vocabulary['s'], vocabulary['f']
        path.write_text(json.dumps(vocabulary))
        from transformers import CLIPTokenizer
        a = CLIPTokenizer.from_pretrained(Path(files['assets']) / folder, local_files_only=True)
        b = CLIPTokenizer.from_pretrained(assets / folder, local_files_only=True)
        assert len(a) == len(b) and a('first small sailboat').input_ids != b('first small sailboat').input_ids
    elif component == 'vae_weights':
        path = assets / 'vae/diffusion_pytorch_model.safetensors'
        weights = load_file(str(path)); next(iter(weights.values())).add_(.125)
        save_file(weights, str(path), metadata={'format': 'pt'})
    else:
        path = assets / 'vae/config.json'; config = json.loads(path.read_text())
        config['scaling_factor'] *= 2; path.write_text(json.dumps(config))
    return assets


@pytest.mark.parametrize('component', ('tokenizer', 'vae_weights', 'vae_scaling'))
def test_same_shaped_different_bundle_rejected_before_admission_update_or_generation(binding_assets, tmp_path, component):
    source = _source(binding_assets, tmp_path / 'raw', raw=component != 'tokenizer')
    prepared = prepare_inputs(source, tmp_path / 'prepared-A')
    dataset = PreparedDataset(prepared)
    other = _changed_bundle(binding_assets, tmp_path, component)
    value = json.loads(Path(binding_assets['config']).read_text())
    value['model']['parameters']['model_id'] = str(other)
    path = tmp_path / 'bundle-B.json'; path.write_text(json.dumps(value))
    config = load_training_config(path)
    components, _ = _construct_model(config)
    owner = components.state['native_cycle']
    compatible = PreparedDataset(prepare_inputs(_source(binding_assets, tmp_path / 'matching-B', assets=other,
                                                        raw=component != 'tokenizer'), tmp_path / 'prepared-B'))
    before = deepcopy(owner.state_dict()); before_rng = rng_state()
    # Independently exercise every native stream, not just generator metadata.
    for stream in config.data['streams']:
        manifest = {'streams': {name: compatible.manifest for name in config.data['streams']}}
        manifest['streams'][stream] = dataset.manifest
        with pytest.raises(ValueError, match='binding|preprocessing'):
            components.admit_training(state=components.state, request=config,
                                      prepared_identity='test', prepared_manifest=manifest)
    feeds = {name: [dataset.sample(0) for _ in range(spec['samples_per_cycle'])]
             for name, spec in config.data['streams'].items()}
    with pytest.raises(ValueError, match='binding|preprocessing'):
        owner.run(feeds, cycle=0)
    with pytest.raises(ValueError, match='binding|preprocessing'):
        components.generate(components.state, dataset.sample(0), {})
    from distillschedule.training.sd import project_current_generator
    projected = project_current_generator(components, {}).components
    with pytest.raises(ValueError, match='binding|preprocessing'):
        projected.generate(projected.state, dataset.sample(0), {})
    assert equal(owner.state_dict(), before) and equal(rng_state(), before_rng)
    assert owner.native_step == owner.generator_updates == owner.guidance_updates == 0


def test_direct_native_encoder_cannot_forge_actual_tokenizer_binding(binding_assets, tmp_path):
    from distillschedule.data.sd_inputs import encode_record
    parameters = PreparedDataset(binding_assets['prepared']).manifest['provenance']['encoder']['parameters']
    other = _changed_bundle(binding_assets, tmp_path, 'tokenizer')
    parameters['model_id'] = str(other)
    with pytest.raises(ValueError, match='binding|preprocessing'):
        encode_record(dict(id='forged', prompt='first small sailboat'), parameters)


@pytest.mark.parametrize('corruption', ('missing', 'forged', 'origin'))
def test_missing_or_forged_row_binding_fails_before_native_update(binding_assets, corruption):
    _, components, feeds = model_and_feeds(binding_assets)
    owner = components.state['native_cycle']; before = deepcopy(owner.state_dict()); before_rng = rng_state()
    row = feeds['generator'][0]
    if corruption == 'missing':
        row.pop('preprocessing_binding', None)
    elif corruption == 'forged':
        row['preprocessing_binding'] = torch.zeros(32, dtype=torch.uint8)
    else:
        row['latent_origin'] = torch.tensor(2, dtype=torch.uint8)
    with pytest.raises(ValueError, match='binding|preprocessing'):
        owner.run(feeds, cycle=0)
    from distillschedule.training.sd import project_current_generator
    projection = project_current_generator(components, {}).components
    for current in (components, projection):
        with pytest.raises(ValueError, match='binding|preprocessing'):
            current.generate(current.state, row, {})
    assert equal(owner.state_dict(), before) and equal(rng_state(), before_rng)


@pytest.mark.parametrize('corruption', ('missing', 'forged', 'provenance'))
def test_missing_or_forged_manifest_binding_is_rejected(binding_assets, corruption):
    config, components, _ = model_and_feeds(binding_assets)
    dataset = PreparedDataset(binding_assets['prepared']); manifest = {'streams': {
        name: dataset.manifest for name in config.data['streams']}}
    sample = manifest['streams']['generator']
    if corruption == 'missing':
        sample['metadata'].pop('preprocessing_assets', None)
    elif corruption == 'forged':
        sample['metadata']['preprocessing_assets'] = {'variant': 'forged'}
    else:
        sample['provenance']['encoder']['parameters']['preprocessing_assets'] = {'variant': 'forged'}
    with pytest.raises(ValueError, match='binding|preprocessing'):
        components.admit_training(state=components.state, request=config,
                                  prepared_identity=dataset.identity, prepared_manifest=manifest)


def test_preencoded_latents_require_honest_explicit_vae_assertion(binding_assets, tmp_path):
    source = _source(binding_assets, tmp_path / 'external', latents=True, declaration=False)
    with pytest.raises(ValueError, match='compatibility|assertion'):
        prepare_inputs(source, tmp_path / 'undeclared')
    assert not (tmp_path / 'undeclared').exists()
    value = json.loads(source.read_text())
    value['preencoded_vae_compatibility'] = 'PREENCODED_LATENTS_MATCH_LOCAL_VAE'
    source.write_text(json.dumps(value))
    dataset = PreparedDataset(prepare_inputs(source, tmp_path / 'declared'))
    row = dataset.sample(0)
    assert row['latent_origin'].item() == 2
    assert torch.equal(row['images'], torch.load(tmp_path / 'external/native.pt', weights_only=True))
    parameters = dataset.manifest['provenance']['encoder']['parameters']
    assert parameters['preencoded_vae_compatibility'] == value['preencoded_vae_compatibility']
    assert 'vae_encoding' not in parameters and 'vae_provenance' not in parameters
    assert any(item['path'].endswith('/vae/config.json')
               for item in dataset.manifest['provenance']['documents'])


def test_byte_identical_relocated_assets_are_accepted_for_actual_native_update(binding_assets, tmp_path):
    relocated = tmp_path / 'relocated'; shutil.copytree(binding_assets['assets'], relocated)
    prepared = prepare_inputs(_source(binding_assets, tmp_path / 'raw-relocated', assets=relocated, raw=True),
                              tmp_path / 'relocated-prepared')
    value = json.loads(Path(binding_assets['config']).read_text())
    for spec in value['data']['streams'].values():
        spec['prepared'] = str(prepared); spec['sampling'] = {'kind': 'epochs', 'shuffle': False}
    path = tmp_path / 'relocated.json'; path.write_text(json.dumps(value))
    config = load_training_config(path)
    components, data, _, _ = construct(config)
    owner = components.state['native_cycle']; result = owner.run(data.cycle_sample(0), cycle=0)
    assert result['generator_updates'] == 1 and result['guidance_updates'] == owner.ratio


def test_external_evaluation_and_export_generation_enforce_binding_without_old_assets(binding_assets, tmp_path):
    from distillschedule.training.session import train, evaluate, export, generate_from_export
    config = load_training_config(binding_assets['config'])
    trained = train(config)
    exported = export(config, trained['checkpoint'], tmp_path / 'export')
    other = _changed_bundle(binding_assets, tmp_path, 'tokenizer')
    bad = prepare_inputs(_source(binding_assets, tmp_path / 'bad-source', assets=other), tmp_path / 'bad-prepared')
    with pytest.raises(ValueError, match='binding|preprocessing'):
        evaluate(config, trained['checkpoint'], tmp_path / 'bad-evaluate', prepared=bad)
    assert not (tmp_path / 'bad-evaluate').exists()
    inference = dict(schema_version='distillschedule.training.v1',
        model=dict(factory=exported['model']['factory'], parameters=exported['model']['parameters']),
        algorithm=dict(name='dmd2'), data=dict(prepared=str(bad)),
        execution=dict(adapter='program', mode='reference'),
        trainer=dict(cycles=1, seed=909, device='cpu', output_dir=str(tmp_path / 'unused')))
    path = tmp_path / 'inference.json'; path.write_text(json.dumps(inference))
    inference_config = load_training_config(path)
    with pytest.raises(ValueError, match='binding|preprocessing'):
        generate_from_export(inference_config, tmp_path / 'export', tmp_path / 'bad-generate',
                             prepared=bad, output_kind='image')
    assert not (tmp_path / 'bad-generate').exists()
    # Keep byte-identical tokenizers only. Prepare genuinely new input AFTER
    # the original base weights, inputs and checkpoint have been deleted.
    tokenizer_only = tmp_path / 'portable-tokenizers'; tokenizer_only.mkdir()
    for folder in ('tokenizer', 'tokenizer_2'):
        original = Path(binding_assets['assets']) / folder
        if original.exists():
            shutil.copytree(original, tokenizer_only / folder)
    fresh_source = _source(binding_assets, tmp_path / 'fresh-source', assets=tokenizer_only)
    fresh = prepare_inputs(fresh_source, tmp_path / 'fresh-before-removal')
    expected = evaluate(config, trained['checkpoint'], tmp_path / 'expected', prepared=fresh)
    expected_values = torch.load(expected['artifact'], weights_only=True)
    shutil.rmtree(binding_assets['assets'])
    shutil.rmtree(Path(binding_assets['prepared']).parent)
    shutil.rmtree(Path(trained['checkpoint']).parent)
    (Path(binding_assets['root']) / 'prompts.json').unlink()
    fresh = prepare_inputs(fresh_source, tmp_path / 'fresh-after-removal')
    generated = generate_from_export(inference_config, tmp_path / 'export', tmp_path / 'generated',
                                     prepared=fresh, output_kind='image')
    values = torch.load(generated['artifact'], weights_only=True)
    assert values['decoded']['image'].shape == (16, 16, 3)
    # Same current G, inference seed and real input must produce exact output.
    from dataclasses import replace
    same_seed = replace(inference_config, trainer=dict(inference_config.trainer, seed=config.trainer['seed']))
    repeated = generate_from_export(same_seed, tmp_path / 'export', tmp_path / 'same-seed',
                                    prepared=fresh, output_kind='image')
    assert equal(torch.load(repeated['artifact'], weights_only=True), expected_values)
