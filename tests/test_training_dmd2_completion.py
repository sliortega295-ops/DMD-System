"""Offline actual-VAE intake and immutable native-owner admission contracts."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from distillschedule.data import PreparedDataset
from distillschedule.data.sd_inputs import SCHEMA, prepare_inputs
from distillschedule.training.checkpoint import rng_state
from distillschedule.workloads.best_known.registry import file_sha256, fingerprint
from test_training_dmd2_cpu import fixture_assets, source_root, model_and_feeds
from test_training_end_to_end import equal


def _raw_source(files, root, *, posterior='sample'):
    import numpy as np
    from PIL import Image
    images = []
    for index in range(2):
        path = root / f'image-{index}.png'
        pixels = (np.arange(16 * 16 * 3).reshape(16, 16, 3) + index * 37).astype('uint8')
        Image.fromarray(pixels).save(path)
        images.append(dict(path=str(path), sha256=file_sha256(path)))
    value = dict(schema_version=SCHEMA, variant=json.loads(Path(files['config']).read_text())['model']['name'],
        model_id=files['assets'], vae_encoding=dict(resolution=[16, 16], latent_shape=[4, 8, 8],
            posterior=posterior, seed=193, dtype='float32'),
        records=[dict(id='unique-image-batch', prompt=['first image', 'second image'], images=images)])
    source = root / 'raw.json'; source.write_text(json.dumps(value))
    return source, value


@pytest.mark.parametrize('posterior', ('mode', 'sample'))
def test_raw_rgb_uses_actual_local_vae_native_scaling_and_independent_rng(fixture_assets, tmp_path, posterior):
    import numpy as np
    from PIL import Image
    from diffusers import AutoencoderKL
    source, value = _raw_source(fixture_assets, tmp_path, posterior=posterior)
    before = rng_state()
    prepared = prepare_inputs(source, tmp_path / 'prepared-rgb')
    assert equal(before, rng_state()), 'offline encoding must not consume caller/training RNG'
    dataset = PreparedDataset(prepared)
    assert dataset.validate(sources=True)['source_validation'] == 'PASS'
    row = dataset.sample(0)
    assert row['text_input_ids_one'].shape == (2, 1, 8)
    if value['variant'] == 'sdxl':
        assert row['text_input_ids_two'].shape == (2, 1, 8)
    with torch.random.fork_rng(devices=[]):
        vae = AutoencoderKL.from_pretrained(Path(fixture_assets['assets']) / 'vae', local_files_only=True)
    vae.eval().requires_grad_(False)
    pixels = torch.stack([torch.from_numpy(np.array(Image.open(spec['path']))).permute(2, 0, 1)
                          for spec in value['records'][0]['images']]).float().div(127.5).sub(1)
    with torch.no_grad():
        distribution = vae.encode(pixels).latent_dist
        if posterior == 'mode':
            expected = distribution.mode()
        else:
            seed = int(fingerprint(dict(seed=193, record_id='unique-image-batch'))[:16], 16) % 2**63
            expected = distribution.sample(generator=torch.Generator(device='cpu').manual_seed(seed))
        expected = expected * vae.config.scaling_factor
    assert torch.equal(row['images'], expected)
    assert row['images'].shape == (2, 4, 8, 8), 'fixture VAE scale is 2, not assumed production scale 8'
    provenance = dataset.manifest['provenance']
    assert any(item['path'].endswith('/vae/config.json') for item in provenance['documents'])
    assert provenance['encoder']['parameters']['vae_encoding'] == value['vae_encoding']


def test_live_math_settings_rejected_before_update_or_save(fixture_assets):
    _, components, feeds = model_and_feeds(fixture_assets)
    owner = components.state['native_cycle']
    before_model = deepcopy(owner.model.state_dict())
    before_optimizers = {name: deepcopy(value.state_dict()) for name, value in owner.optimizers.items()}
    owner.args.real_guidance_scale += 1
    for operation in (lambda: owner.run(feeds, cycle=0), owner.state_dict):
        with pytest.raises(ValueError, match='settings'):
            operation()
    assert equal(owner.model.state_dict(), before_model)
    assert equal({name: value.state_dict() for name, value in owner.optimizers.items()}, before_optimizers)
    assert owner.native_step == 0


def test_native_physical_checkpoint_flags_are_not_mathematical_settings(fixture_assets):
    _, components, _ = model_and_feeds(fixture_assets)
    owner = components.state['native_cycle']
    before_args = deepcopy(vars(owner.args))
    # This is an admission distinction only, not activation of a strategy or a
    # claim of checkpoint execution parity (owned by a separate capability unit).
    owner.model.gradient_checkpointing = not owner.model.gradient_checkpointing
    owner.model.guidance_model.gradient_checkpointing = not owner.model.guidance_model.gradient_checkpointing
    owner._validate_live()
    assert vars(owner.args) == before_args
    owner.args.gradient_checkpointing = not owner.args.gradient_checkpointing
    with pytest.raises(ValueError, match='settings'):
        owner._validate_live()


@pytest.mark.parametrize('corruption', ('teacher', 'clip', 'vae', 'scheduler', 'alphas',
                                      'unconditional', 'denoising_table', 'cached_cfg', 'identity_settings'))
def test_live_frozen_constants_rejected_after_legitimate_native_update(fixture_assets, corruption):
    _, components, feeds = model_and_feeds(fixture_assets)
    owner = components.state['native_cycle']
    owner.run(feeds, cycle=0)
    valid = deepcopy(owner.state_dict())
    assert all(p.requires_grad for p in owner.model.guidance_model.real_unet.parameters())
    with torch.no_grad():
        if corruption in ('teacher', 'clip', 'vae'):
            module = {'teacher': owner.model.guidance_model.real_unet,
                      'clip': owner.model.text_encoder, 'vae': owner.model.vae}[corruption]
            next(module.parameters()).add_(1)
        elif corruption == 'scheduler':
            owner.model.noise_scheduler.betas.add_(.01)
        elif corruption == 'alphas':
            owner.model.alphas_cumprod = owner.model.alphas_cumprod.clone().add_(.01)
        elif corruption == 'unconditional':
            if owner.args.sdxl:
                owner.uncond_embedding = torch.zeros(1, 8, 16)
            else:
                owner.uncond_embedding.add_(1)
        elif corruption == 'denoising_table':
            owner.model.denoising_step_list[0] -= 1
        elif corruption == 'cached_cfg':
            owner.model.guidance_model.real_guidance_scale += 1
        else:
            owner.identity['settings']['dm_loss_weight'] += 1
    before = deepcopy(owner.model.state_dict())
    before_rng = rng_state()
    for operation in (lambda: owner.run(feeds, cycle=1), owner.state_dict,
                      lambda: owner.load_state_dict(valid)):
        with pytest.raises(ValueError, match='frozen|conditioning|scheduler|settings|constants'):
            operation()
    assert equal(owner.model.state_dict(), before)
    assert equal(before_rng, rng_state())
    assert owner.native_step == owner.ratio


def test_local_bundle_and_generator_checkpoint_hashing_is_streaming(fixture_assets, monkeypatch, tmp_path):
    from distillschedule.training.sd import local_bundle_identity, build_sdxl
    root = Path(fixture_assets['assets'])
    expected = local_bundle_identity(root, sdxl=bool((root / 'tokenizer_2').exists()))
    read_bytes = Path.read_bytes
    checkpoint = tmp_path / 'generator.pt'
    from safetensors.torch import load_file
    torch.save(load_file(str(root / 'unet/diffusion_pytorch_model.safetensors')), checkpoint)
    def no_payload_read(path):
        if path.is_relative_to(root) or path == checkpoint:
            raise AssertionError('local model/checkpoint hashing must stream, not read_bytes')
        return read_bytes(path)
    monkeypatch.setattr(Path, 'read_bytes', no_payload_read)
    assert local_bundle_identity(root, sdxl=bool((root / 'tokenizer_2').exists())) == expected
    if (root / 'tokenizer_2').exists():
        parameters = json.loads(Path(fixture_assets['config']).read_text())['model']['parameters']
        parameters['settings'].update(num_denoising_step=1, conditioning_timestep=399,
                                     denoising=False, backward_simulation=False)
        parameters['generator_checkpoint'] = str(checkpoint)
        owner = build_sdxl(parameters).state['native_cycle']
        assert owner.identity['generator_checkpoint']['sha256'] == file_sha256(checkpoint)


@pytest.mark.parametrize('corruption', ('resolution', 'latent_shape', 'dtype', 'posterior', 'seed',
    'hash', 'missing', 'batch', 'empty', 'rgb', 'latents_and_images', 'missing_policy'))
def test_raw_intake_fails_closed_without_publishing(fixture_assets, tmp_path, corruption):
    from PIL import Image
    source, value = _raw_source(fixture_assets, tmp_path)
    policy, record = value['vae_encoding'], value['records'][0]
    if corruption == 'resolution':
        policy['resolution'] = [15, 16]
    elif corruption == 'latent_shape':
        policy['latent_shape'] = [4, 2, 2]
    elif corruption in ('dtype', 'posterior', 'seed'):
        policy[corruption] = {'dtype': 'float16', 'posterior': 'implicit', 'seed': True}[corruption]
    elif corruption == 'hash':
        record['images'][0]['sha256'] = '0' * 64
    elif corruption == 'missing':
        Path(record['images'][0]['path']).unlink()
    elif corruption == 'batch':
        record['prompt'].pop()
    elif corruption == 'empty':
        record['images'] = []
    elif corruption == 'rgb':
        path = Path(record['images'][0]['path'])
        Image.new('L', (16, 16)).save(path)
        record['images'][0]['sha256'] = file_sha256(path)
    elif corruption == 'latents_and_images':
        record['latents'] = dict(path='unused', sha256='0' * 64)
    else:
        value.pop('vae_encoding')
    source.write_text(json.dumps(value))
    with pytest.raises((ValueError, FileNotFoundError)):
        prepare_inputs(source, tmp_path / 'bad-prepared')
    assert not (tmp_path / 'bad-prepared').exists()


def test_raw_geometry_is_exact_not_implicitly_resized(fixture_assets, tmp_path):
    from PIL import Image
    source, value = _raw_source(fixture_assets, tmp_path)
    path = Path(value['records'][0]['images'][0]['path'])
    Image.new('RGB', (18, 16)).save(path)
    value['records'][0]['images'][0]['sha256'] = file_sha256(path)
    source.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='geometry|resolution'):
        prepare_inputs(source, tmp_path / 'wrong-geometry')
    assert not (tmp_path / 'wrong-geometry').exists()


@pytest.mark.parametrize('corruption', ('model', 'image', 'source', 'nonfinite', 'added_asset'))
def test_raw_native_encoder_detects_changed_inputs_before_publish(fixture_assets, tmp_path, monkeypatch, corruption):
    from diffusers import AutoencoderKL
    source, value = _raw_source(fixture_assets, tmp_path)
    original = AutoencoderKL.encode
    def native_then_corrupt(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if corruption == 'nonfinite':
            result.latent_dist.mean.fill_(float('nan'))
        elif corruption == 'added_asset':
            (Path(fixture_assets['assets']) / 'vae/unpinned-new-file.json').write_text('{}')
        else:
            target = {'model': Path(fixture_assets['assets']) / 'vae/config.json',
                      'image': Path(value['records'][0]['images'][0]['path']), 'source': source}[corruption]
            with target.open('ab') as stream:
                stream.write(b' ')
        return result
    monkeypatch.setattr(AutoencoderKL, 'encode', native_then_corrupt)
    with pytest.raises(ValueError, match='changed|hash|finite'):
        prepare_inputs(source, tmp_path / 'changed-prepared')
    assert not (tmp_path / 'changed-prepared').exists()


@pytest.mark.parametrize('scaling', (0, float('nan'), float('inf')))
def test_native_vae_scaling_factor_must_be_finite_and_positive(fixture_assets, tmp_path, scaling):
    source, _ = _raw_source(fixture_assets, tmp_path)
    config_path = Path(fixture_assets['assets']) / 'vae/config.json'
    config = json.loads(config_path.read_text()); config['scaling_factor'] = scaling
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='scaling_factor'):
        prepare_inputs(source, tmp_path / 'bad-scaling')
    assert not (tmp_path / 'bad-scaling').exists()


def test_posterior_sample_is_record_keyed_and_order_independent(fixture_assets, tmp_path):
    source, value = _raw_source(fixture_assets, tmp_path)
    value['records'].append(dict(deepcopy(value['records'][0]), id='different-id'))
    source.write_text(json.dumps(value))
    a = PreparedDataset(prepare_inputs(source, tmp_path / 'first'))
    value['records'].reverse(); source.write_text(json.dumps(value))
    b = PreparedDataset(prepare_inputs(source, tmp_path / 'reordered'))
    assert torch.equal(a.sample(0)['images'], b.sample(1)['images'])
    assert torch.equal(a.sample(1)['images'], b.sample(0)['images'])
    assert not torch.equal(a.sample(0)['images'], a.sample(1)['images'])


def test_raw_preparation_uses_no_teacher_fake_or_clip_model(fixture_assets, tmp_path, monkeypatch):
    from diffusers import UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTextModelWithProjection
    def forbidden(*args, **kwargs):
        raise AssertionError('offline intake must not construct Teacher/Fake/CLIP model')
    for cls in (UNet2DConditionModel, CLIPTextModel, CLIPTextModelWithProjection):
        monkeypatch.setattr(cls, 'from_pretrained', forbidden)
    source, _ = _raw_source(fixture_assets, tmp_path)
    dataset = PreparedDataset(prepare_inputs(source, tmp_path / 'vae-only'))
    assert dataset.sample(0)['images'].shape == (2, 4, 8, 8)


def test_sdxl_raw_intake_requires_actual_second_local_tokenizer(fixture_assets, tmp_path):
    source, value = _raw_source(fixture_assets, tmp_path)
    if value['variant'] == 'sd15':
        # SD15 correctly has exactly one native tokenizer and no fallback path.
        assert not (Path(fixture_assets['assets']) / 'tokenizer_2').exists()
        return
    import shutil
    shutil.rmtree(Path(fixture_assets['assets']) / 'tokenizer_2')
    with pytest.raises(ValueError, match='tokenizer'):
        prepare_inputs(source, tmp_path / 'missing-tokenizer')
    assert not (tmp_path / 'missing-tokenizer').exists()


def test_runtime_tensors_exposes_plain_native_conditioning_not_host_metadata(fixture_assets):
    _, components, _ = model_and_feeds(fixture_assets)
    owner = components.state['native_cycle']
    from distillschedule.open_workload.torch_backend import tensors
    live = list(tensors(owner))
    for value in (owner.model.denoising_step_list, owner.model.noise_scheduler.betas,
                  owner.model.noise_scheduler.timesteps):
        assert any(tensor is value for tensor in live)
    if owner.args.sdxl:
        assert any(tensor is owner.model.add_time_ids for tensor in live)
    else:
        assert any(tensor is owner.uncond_embedding for tensor in live)
    assert not any(tensor is owner._rng['torch'] for tensor in live)


def test_real_gan_and_single_step_templates_preserve_native_streams():
    from test_training_dmd2_cpu import REPO
    gan = json.loads((REPO / 'recipes/dmd2_sd15/training.gan.json').read_text())
    single = json.loads((REPO / 'recipes/dmd2_sdxl/training.single_step.json').read_text())
    assert {name: value['samples_per_cycle'] for name, value in gan['data']['streams'].items()} == {
        'generator': 1, 'guidance': 9, 'real': 10}
    settings = gan['model']['parameters']['settings']
    assert settings['cls_on_clean_image'] and settings['gen_cls_loss']
    assert settings['gen_cls_loss_weight'] > 0 and settings['guidance_cls_loss_weight'] > 0
    assert {name: value['samples_per_cycle'] for name, value in single['data']['streams'].items()} == {
        'generator': 1, 'guidance': 4}
    parameters = single['model']['parameters']; settings = parameters['settings']
    assert parameters['generator_checkpoint']
    assert settings['conditioning_timestep'] == 399 and settings['num_denoising_step'] == 1
    assert settings['denoising'] is False and settings['backward_simulation'] is False
    for config in (gan, single):
        assert config['trainer']['device'] == 'cpu' and config['execution']['adapter'] == 'program'
