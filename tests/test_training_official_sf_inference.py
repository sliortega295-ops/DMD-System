"""Codec-boundary regression; CPU probes do not qualify FSDP2 sampling.

The final test uses the pinned actual WanVAE and existing random native assets.
Setting SELF_FORCING_SOURCE_ROOT makes missing dependencies errors, not skips.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test_training_self_forcing_cpu import fixture_assets, source_root


def _decode_probe():
    """Interface-only recorder, never a numerical replacement for WanVAE."""
    vae = torch.nn.Module()
    vae.model = torch.nn.Linear(1, 1).requires_grad_(False)
    calls = []

    def decode_to_pixel(latent, use_cache=False):
        assert latent.dtype == next(vae.model.parameters()).dtype
        calls.append((latent, use_cache))
        return latent

    vae.decode_to_pixel = decode_to_pixel
    return vae, calls


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('use_cache', [False, True])
def test_inference_decode_boundary_preserves_input_and_cache_argument(dtype, use_cache):
    from distillschedule.training.self_forcing import _InferenceVAEDecodeBoundary
    vae, calls = _decode_probe()
    latent = torch.arange(16, dtype=dtype).reshape(1, 1, 16, 1, 1)
    before = latent.clone()
    decoder = _InferenceVAEDecodeBoundary(vae)
    rng = torch.random.get_rng_state().clone()
    actual = decoder.decode_to_pixel(latent, use_cache=use_cache)
    assert decoder.vae is vae
    assert calls == [(actual, use_cache)]
    assert actual.dtype == torch.float32 and torch.equal(actual, before.float())
    assert (actual is latent) is (dtype == torch.float32)
    assert latent.dtype == dtype and torch.equal(latent, before)
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert all(parameter.dtype == torch.float32 for parameter in vae.parameters())


@pytest.mark.parametrize('sharded', [False, True])
def test_generate_preserves_sampler_dtype_latents_rng_and_original_vae(monkeypatch, sharded):
    """Exercise public orchestration only, without CUDA or a sharding claim."""
    from distillschedule.training import self_forcing as sf
    from distillschedule.training.self_forcing_fsdp import SelfForcingFSDP2Owner
    generator = torch.nn.Linear(1, 1)
    generator.model = SimpleNamespace(text_len=2, text_dim=3)
    vae, calls = _decode_probe()
    state = dict(generator=generator, vae=vae)
    dtype = torch.bfloat16 if sharded else torch.float32
    seen = {}

    class PipelineProbe:
        def __init__(self, args, device, generator, text_encoder, vae):
            self.vae, self.text_encoder = vae, text_encoder
            self.denoising_step_list = torch.tensor(args.denoising_step_list)
            seen['steps'] = args.denoising_step_list

        def configure_geometry(self, *, shape):
            seen['shape'] = shape

        def inference(self, *, noise, text_prompts, return_latents):
            assert return_latents
            seen['noise'] = noise
            seen['condition'] = self.text_encoder(text_prompts)['prompt_embeds']
            seen['latents'] = noise.clone()
            video = self.vae.decode_to_pixel(seen['latents'], use_cache=False)
            return video, seen['latents']

    settings = dict(runtime={'interface_fixture': True}, source_root='unused', shape=[1, 21, 16, 4, 4],
                    settings={'denoising_step_list': [1000, 750, 500, 250]})
    monkeypatch.setattr(sf, 'runtime_identity', lambda: settings['runtime'])
    monkeypatch.setattr(sf, 'original', lambda _: SimpleNamespace(InferencePipeline=PipelineProbe))
    owner = None
    if sharded:
        # Only route through the real owner's isinstance gate. No CPU FSDP
        # admission, native forward, or distributed numerical behavior is faked.
        owner = SelfForcingFSDP2Owner.__new__(SelfForcingFSDP2Owner)
        owner.model = SimpleNamespace(generator=generator, vae=vae)
        owner.dtype = dtype
        owner._validate_live = lambda: None
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(909)
        expected_noise = torch.randn(settings['shape'], dtype=dtype)
        expected_rng = torch.random.get_rng_state().clone()
        torch.manual_seed(909)
        generated = sf._generate(state, {'prompt_embeds': torch.ones(1, 2, 3)}, {}, settings,
                                 sharded_owner=owner)
        assert torch.equal(torch.random.get_rng_state(), expected_rng)
    assert state['vae'] is vae
    assert seen['noise'].dtype == seen['condition'].dtype == generated['latents'].dtype == dtype
    assert generated['latents'] is seen['latents']
    assert torch.equal(seen['noise'], expected_noise)
    assert torch.equal(generated['latents'], expected_noise)
    assert len(calls) == 1 and calls[0][1] is False
    assert calls[0][0].dtype == generated['video'].dtype == torch.float32
    assert torch.equal(generated['video'], expected_noise.float())
    assert seen['steps'] == [1000, 750, 500, 250] and seen['shape'] == settings['shape']
    assert all(parameter.dtype == torch.float32 for parameter in vae.parameters())


def test_actual_native_vae_bf16_latents_match_explicit_fp32_decode(fixture_assets, source_root, monkeypatch):
    from distillschedule.training.self_forcing import original, _vae_wrapper, _InferenceVAEDecodeBoundary
    api = original(source_root)
    root = Path(fixture_assets['assets'])
    asset = json.loads((root / 'assets.json').read_text())['vae']
    vae, _ = _vae_wrapper(api, asset['config'])
    vae.model.load_state_dict(torch.load(root / asset['weights']['path'], map_location='cpu', weights_only=True))
    assert type(vae) is api.wrapper.WanVAEWrapper and type(vae.model) is api.vae.WanVAE_
    weights = {name: tensor.clone() for name, tensor in vae.state_dict().items()}
    normalization = (vae.mean.clone(), vae.std.clone())
    latent = torch.linspace(-1, 1, 3 * 16 * 4 * 4).reshape(1, 3, 16, 4, 4).to(torch.bfloat16)
    before = latent.clone()
    rng = torch.random.get_rng_state().clone()
    decoder = _InferenceVAEDecodeBoundary(vae)
    native_decode, native_cached_decode = vae.model.decode, vae.model.cached_decode
    calls = []

    def decode(*args, **kwargs):
        calls.append(False)
        return native_decode(*args, **kwargs)

    def cached_decode(*args, **kwargs):
        calls.append(True)
        return native_cached_decode(*args, **kwargs)

    monkeypatch.setattr(vae.model, 'decode', decode)
    monkeypatch.setattr(vae.model, 'cached_decode', cached_decode)
    with torch.no_grad():
        # Reproduce the exact native input/bias mismatch before the adapter.
        with pytest.raises(RuntimeError, match='Input type.*bias type'):
            vae.decode_to_pixel(latent, use_cache=False)
        for use_cache in (False, True):
            vae.model.clear_cache()
            expected = vae.decode_to_pixel(latent.float(), use_cache=use_cache)
            vae.model.clear_cache()
            calls.clear()
            actual = decoder.decode_to_pixel(latent, use_cache=use_cache)
            assert calls == [use_cache]
            assert actual.dtype == torch.float32 and torch.equal(actual, expected)
        vae.model.clear_cache()
    assert decoder.vae is vae and latent.dtype == torch.bfloat16 and torch.equal(latent, before)
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert all(parameter.dtype == torch.float32 for parameter in vae.parameters())
    assert all(torch.equal(weights[name], tensor) for name, tensor in vae.state_dict().items())
    assert torch.equal(vae.mean, normalization[0]) and torch.equal(vae.std, normalization[1])
