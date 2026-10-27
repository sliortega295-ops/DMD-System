"""Actual upstream guidance/UNet; small head is a labeled CPU test fixture.

Production admission still requires the unmodified 1280-channel SD15 head.
This exercises the packaged hooks, native losses, gradients, RNG and Adam;
it does not stand in for a full-size CUDA run.
"""
from contextlib import ExitStack
from copy import deepcopy
from types import MethodType, SimpleNamespace

import pytest
import torch

from test_training_dmd2_cpu import source_root


def _guidance(source):
    from diffusers import UNet2DConditionModel, DDPMScheduler
    from distillschedule.training.sd import original
    api = original(source)
    module = api.guidance.SDGuidance.__new__(api.guidance.SDGuidance)
    torch.nn.Module.__init__(module)
    module.fake_unet = UNet2DConditionModel(sample_size=8, in_channels=4, out_channels=4,
        down_block_types=('CrossAttnDownBlock2D', 'DownBlock2D'),
        up_block_types=('UpBlock2D', 'CrossAttnUpBlock2D'), block_out_channels=(8, 16),
        layers_per_block=1, cross_attention_dim=8, attention_head_dim=2, norm_num_groups=4)
    module.fake_unet.forward = MethodType(api.unet.classify_forward, module.fake_unet)
    module.cls_pred_branch = torch.nn.Sequential(torch.nn.Conv2d(16, 16, 4),
        torch.nn.GroupNorm(4, 16), torch.nn.SiLU(), torch.nn.Conv2d(16, 1, 1))
    module.scheduler = DDPMScheduler(num_train_timesteps=1000)
    module.register_buffer('alphas_cumprod', module.scheduler.alphas_cumprod)
    module.diffusion_gan = module.gradient_checkpointing = module.cls_on_clean_image = True
    module.diffusion_gan_max_timestep = module.num_train_timesteps = 1000
    module.sdxl = False
    module.network_context_manager = torch.autocast('cpu', dtype=torch.bfloat16)
    return module


def _owner(guidance, ratio):
    accelerator = SimpleNamespace(unwrap_model=lambda value: value)
    return SimpleNamespace(model=SimpleNamespace(guidance_model=guidance,
        accelerator=accelerator), accelerator=accelerator, ratio=ratio)


def test_real_native_guidance_bundle_rebuilds_after_commits_and_drains(source_root):
    from distillschedule.training.official_optimizations.dmd2 import Bundle
    from distillschedule.training.official_optimizations._native.dmd2_activation_retention import ActivationRetention
    from distillschedule.training.official_optimizations._native.dmd2_classifier_pack import PackedClassifier
    from distillschedule.training.official_optimizations._native.dmd2_update_cast_scope import UpdateCastScope
    torch.set_num_threads(1)
    torch.manual_seed(10987)
    models = [_guidance(source_root)]
    models.append(deepcopy(models[0]))
    optimizers = [torch.optim.AdamW(module.parameters(), lr=1e-4) for module in models]
    generator = torch.Generator().manual_seed(882)
    image = torch.randn(1, 4, 8, 8, generator=generator)
    text = torch.randn(1, 3, 8, generator=generator)
    feeds = dict(image=image, text_embedding=text, uncond_embedding=text * .8,
        real_train_dict=dict(images=image * .7, text_embedding=text * .9, unet_added_conditions=None))
    for _ in range(2):
        with ExitStack() as stack:
            activation = ActivationRetention(_owner(models[0], 2))
            stack.callback(activation.close)
            classifier = PackedClassifier(models[0], torch, cpu_test_only=True)
            stack.callback(classifier.close)
            # Compare direct composition of all measured source mechanisms
            # with the public Bundle, not cached versus uncached arithmetic.
            cast = UpdateCastScope(models[0], torch, cpu_test_only=True)
            stack.callback(cast.close)
            bundle = Bundle(_owner(models[1], 2), cpu_test_only=True)
            stack.callback(bundle.close)
            for _ in range(2):
                rng = torch.get_rng_state().clone()
                outputs, grads = [], []
                for model, optimizer in zip(models, optimizers):
                    torch.set_rng_state(rng)
                    losses, _ = model.guidance_forward(**feeds)
                    sum(losses.values()).backward()
                    outputs.append((losses, torch.get_rng_state().clone()))
                    grads.append(torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None]))
                    optimizer.step()
                assert torch.equal(outputs[0][1], outputs[1][1])
                assert outputs[0][0].keys() == outputs[1][0].keys()
                for key in outputs[0][0]:
                    torch.testing.assert_close(outputs[0][0][key], outputs[1][0][key], rtol=0, atol=0)
                torch.testing.assert_close(*grads, rtol=0, atol=0)
                for key in ('exp_avg', 'exp_avg_sq'):
                    moments = [torch.cat([optimizer.state[p][key].flatten() for p in model.parameters()])
                               for model, optimizer in zip(models, optimizers)]
                    torch.testing.assert_close(*moments, rtol=0, atol=0)
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)
            receipt = bundle.receipt()
            assert receipt['cast']['clear_calls'] == 4
            assert receipt['classifier']['physical_packed_trunk_calls'] == 2
            assert receipt['transient_state_empty']
        assert not torch.is_autocast_enabled('cpu')
    assert not torch.cuda.is_initialized()


def test_bundle_exception_restores_patch_and_autocast_state(source_root):
    from distillschedule.training.official_optimizations.dmd2 import Bundle
    model = _guidance(source_root)
    def fail(*args, **kwargs):
        raise RuntimeError('injected native failure')
    model.guidance_forward = fail
    pair = model.compute_guidance_clean_cls_loss
    bundle = Bundle(_owner(model, 2), cpu_test_only=True)
    try:
        with pytest.raises(RuntimeError, match='injected native failure'):
            model.guidance_forward()
        assert not bundle.cast.active and bundle.cast.record['exited'] == 1
    finally:
        bundle.close()
    assert model.guidance_forward is fail and model.compute_guidance_clean_cls_loss == pair
    assert not torch.is_autocast_enabled('cpu') and not torch.cuda.is_initialized()


def test_mixed_vae_inference_projection_round_trip(source_root, tmp_path):
    from distillschedule.training.plugins import load_callable
    from distillschedule.training import sd
    from test_training_dmd2_cpu import model_and_feeds, REPO
    builder, _ = load_callable(str(REPO / 'recipes/dmd2_sd15/create_cpu_fixture.py') + ':build_fixture')
    files = builder(tmp_path / 'sd15', source_root, variant='sd15', cycles=1)
    _, components, feeds = model_and_feeds(files)
    owner = components.state['native_cycle']
    # CPU codec qualification only: production optimized admission remains CUDA/world16.
    owner.model.vae.half()
    view = SimpleNamespace(model=owner.model, args=owner.args, source=owner.source,
        identity={**owner.identity, 'optimization_profile': {'id': sd.__name__}})
    parameters = sd._inference_parameters(view)
    assert parameters['precision'] == dict(generator='float32', text_encoder='float32', vae='float16')
    state = dict(generator=owner.model.feedforward_model, text_encoder=owner.model.text_encoder,
                 vae=owner.model.vae)
    projected = sd._inference_components(state, parameters)
    restored = sd.build_inference_model(parameters)
    for name, module in state.items():
        restored.state[name].load_state_dict(module.state_dict())
        module.eval()
        restored.state[name].eval()  # Same mode as public _generate_values.
        for key, value in module.state_dict().items():
            assert torch.equal(value, restored.state[name].state_dict()[key])
    assert next(restored.state['vae'].parameters()).dtype == torch.float16
    rng = torch.get_rng_state().clone()
    with torch.no_grad():
        output = projected.generate(state, feeds, {})
        image = projected.decode(output, {})
        after_generation = torch.get_rng_state().clone()
        torch.set_rng_state(rng)
        replay = restored.generate(restored.state, feeds, {})
        decoded = restored.decode(replay, {})
        assert torch.equal(torch.get_rng_state(), after_generation)
        # Codec equality uses the exact same input on both restored VAEs.
        assert torch.equal(image['image'], restored.decode(output, {})['image'])
    # Independently constructed UNets can reassociate CPU floating operations;
    # trajectory equivalence is not this dtype/weight/codec round-trip test.
    assert output['latents'].shape == replay['latents'].shape
    assert output['latents'].dtype == replay['latents'].dtype == torch.float32
    assert torch.isfinite(output['latents']).all() and torch.isfinite(replay['latents']).all()
    assert torch.equal(image['image'], decoded['image'])
    assert image['image'].dtype == torch.uint8 and not torch.cuda.is_initialized()
