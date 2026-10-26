"""Actual external DMD2 and random tiny Diffusers integration, never substitutes.

Set DMD2_SOURCE_ROOT to the explicit pinned source root. Missing dependencies
are errors (not skips), so an offline acceptance run cannot silently pass.
"""
import ast
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from distillschedule.data import PreparedDataset
from distillschedule.training.config import load_training_config
from distillschedule.training.session import construct
from distillschedule.training.checkpoint import rng_state, restore_rng, cpu_tree
from test_training_end_to_end import equal, run_cli

REPO = Path(__file__).resolve().parents[1]
ORACLE_SHA = '55a8fca2e4fbf2f36c0c3a205f6fa18e92bfa7be06c8249561ce7780a4c68f7f'


@pytest.fixture(scope='module')
def source_root():
    value = os.environ.get('DMD2_SOURCE_ROOT')
    if not value:
        pytest.skip('optional upstream source: set DMD2_SOURCE_ROOT for native DMD2 acceptance')
    return Path(value).resolve(strict=True)


@pytest.fixture(params=('sd15', 'sdxl'))
def fixture_assets(tmp_path, source_root, request):
    from distillschedule.training.plugins import load_callable
    builder, _ = load_callable(str(REPO / 'recipes/dmd2_sd15/create_cpu_fixture.py') + ':build_fixture')
    return builder(tmp_path / request.param, source_root, variant=request.param, cycles=3)


def model_and_feeds(files):
    config = load_training_config(files['config'])
    components, data, _, _ = construct(config)
    return config, components, data.cycle_sample(0)


def test_actual_original_classes_and_native_input_geometry(fixture_assets):
    config, components, feeds = model_and_feeds(fixture_assets)
    owner = components.state['native_cycle']
    from diffusers import UNet2DConditionModel, AutoencoderKL
    from transformers import CLIPTextModel
    from distillschedule.training.sd import original
    api = original(config.model['parameters']['source_root'])
    assert type(owner.model) is api.unified.SDUniModel
    assert type(owner.model.guidance_model) is api.guidance.SDGuidance
    assert type(owner.model.feedforward_model) is UNet2DConditionModel
    assert type(owner.model.vae) is AutoencoderKL
    assert components.prediction.kind == 'epsilon'
    assert not any(name == 'main' or name.startswith('main.') for name in sys.modules)
    assert 'matplotlib' not in sys.modules
    # Diffusers 0.37.1 itself imports scipy.stats from its single-file scheduler
    # loader. The source-port requirement is to skip upstream utils' unrelated
    # dependencies, not to forbid legitimate third-party numerical imports.
    # A fresh-process import-provenance guard tests that boundary below.
    if owner.args.sdxl:
        assert type(owner.model.text_encoder) is api.text.SDXLTextEncoder
        assert owner.model.denoising_step_list.tolist() == [999, 749, 499, 249]
        row = feeds['denoising'][0]
        embedding, pooled = owner.model.text_encoder(row)
        assert embedding.shape == (1, 8, 16) and pooled.shape == (1, 8)
        assert owner.model.add_time_ids.tolist() == [[16., 16., 0., 0., 16., 16.]]
    else:
        assert type(owner.model.text_encoder) is CLIPTextModel
        from transformers import CLIPTokenizer
        tokenizer = CLIPTokenizer.from_pretrained(Path(fixture_assets['assets']) / 'tokenizer')
        ids = tokenizer([''], max_length=8, padding='max_length', return_tensors='pt').input_ids
        assert torch.equal(owner.uncond_embedding, owner.model.text_encoder(ids)[0])
    assert all(not any(name.startswith('guidance_model.real_unet.') for name in group)
               for group in owner.ownership['guidance'])


def test_original_utility_imports_skip_non_numerical_dependencies(source_root):
    import subprocess
    script = r'''import builtins, pathlib, sys, traceback
import torch
from distillschedule.training.sd import original
utility = str(pathlib.Path(sys.argv[1]).resolve() / 'main/utils.py')
native_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in ('scipy', 'matplotlib'):
        if any(frame.filename == utility for frame in traceback.extract_stack()):
            raise AssertionError('unrelated dependency imported by pinned native utils: ' + name)
    return native_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
api = original(sys.argv[1])
assert api.source.identity['definitions']['main.utils'] == ['get_x0_from_noise', 'NoOpContext', 'DummyNetwork']
assert all(getattr(api.utils, name).__module__ == api.utils.__name__
           for name in api.source.identity['definitions']['main.utils'])
assert not any(name == 'main' or name.startswith('main.') for name in sys.modules)
assert 'matplotlib' not in sys.modules and not torch.cuda.is_initialized()
'''
    result = subprocess.run([sys.executable, '-c', script, str(source_root)],
                            capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr


def _native_method(root):
    path = root / 'oracle/train_sd.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == ORACLE_SHA
    tree = ast.parse(path.read_bytes(), filename=str(path))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Trainer')
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'train_one_step')
    # Exact original update statements, excluding subsequent logging only.
    method.body = [node for node in method.body if node.lineno <= 408]
    for node in method.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'visual' for target in node.targets):
            node.value = ast.Constant(False)  # Disable visualization only, never loss/rollout/update math.
    method.body.append(ast.Return(value=ast.Call(func=ast.Name(id='locals', ctx=ast.Load()), args=[], keywords=[])))
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    scope = {'torch': torch}
    exec(compile(module, str(path), 'exec'), scope)
    return scope['train_one_step']


def _oracle(owner, feeds, source_root):
    """Run the source's own trainer statements, independently of owner.run."""
    snapshots, norms, caches, gradients = [], [], [], []
    class Accelerator:
        device = owner.model.accelerator.device

        def backward(self, loss):
            loss.backward()

        def clip_grad_norm_(self, parameters, maximum):
            parameters = list(parameters)
            names = ({id(parameter): name for name, parameter in owner.model.feedforward_model.named_parameters()}
                     if len(norms) % (owner.ratio + 1) == 0 else
                     {id(parameter): name for name, parameter in owner.model.guidance_model.named_parameters()})
            gradients.append({names[id(parameter)]: None if parameter.grad is None else parameter.grad.clone()
                              for parameter in parameters})
            result = torch.nn.utils.clip_grad_norm_(parameters, maximum)
            norms.append(float(result)); return result

    trainer = SimpleNamespace(model=owner.model, args=owner.args, accelerator=Accelerator(), step=0,
        batch_size=owner.args.batch_size, latent_channel=4, latent_resolution=owner.args.latent_resolution,
        wandb_iters=100, dfake_gen_update_ratio=owner.ratio, sdxl=owner.args.sdxl,
        denoising=owner.args.denoising, cls_on_clean_image=owner.args.cls_on_clean_image,
        uncond_embedding=owner.uncond_embedding, max_grad_norm=owner.args.max_grad_norm,
        gen_cls_loss=owner.args.gen_cls_loss, gen_cls_loss_weight=owner.args.gen_cls_loss_weight,
        guidance_cls_loss_weight=owner.args.guidance_cls_loss_weight,
        optimizer_generator=owner.optimizers['generator'], optimizer_guidance=owner.optimizers['guidance'],
        scheduler_generator=owner.schedulers['generator'], scheduler_guidance=owner.schedulers['guidance'],
        dataloader=iter(deepcopy(feeds['generator'])), guidance_dataloader=iter(deepcopy(feeds.get('guidance', []))),
        denoising_dataloader=iter(deepcopy(feeds.get('denoising', []))),
        real_dataloader=iter(deepcopy(feeds.get('real', []))))
    def capture(module, args, kwargs):
        if kwargs.get('guidance_turn'):
            caches.append(cpu_tree(kwargs['guidance_data_dict']))
    hook = owner.model.register_forward_pre_hook(capture, with_kwargs=True)
    native = _native_method(source_root)
    restore_rng(owner._rng)
    for index in range(owner.ratio):
        trainer.step = index
        values = native(trainer)
        snapshots.append({name: float(value.detach()) for name, value in {
            **values['generator_loss_dict'], **values['guidance_loss_dict']}.items()})
    hook.remove()
    return dict(losses=snapshots, caches=caches, gradients=gradients, norms=norms,
                rng=rng_state(device=owner.model.accelerator.device))


def test_direct_source_native_iterations_match_owner_losses_gradients_cache_and_clocks(fixture_assets, source_root):
    config, components, feeds = model_and_feeds(fixture_assets)
    owner = components.state['native_cycle']
    reference = deepcopy(owner)
    before_teacher = deepcopy(owner.model.guidance_model.real_unet.state_dict())
    outer_rng = rng_state()
    expected = _oracle(reference, feeds, source_root)
    restore_rng(outer_rng)
    events = []
    owner.observer = lambda event, value: events.append((event, cpu_tree(value)))
    result = owner.run(feeds, cycle=0)
    assert equal(outer_rng, rng_state()), 'Program/outer RNG must not replace or consume native RNG'
    assert [row['losses'] for row in result['iterations']] == expected['losses']
    assert equal(owner.model.state_dict(), reference.model.state_dict())
    assert equal({role: optimizer.state_dict() for role, optimizer in owner.optimizers.items()},
                 {role: optimizer.state_dict() for role, optimizer in reference.optimizers.items()})
    assert equal(owner._rng, expected['rng'])
    assert [value for event, value in events if event == 'guidance_input_cache']
    assert equal([value for event, value in events if event == 'guidance_input_cache'], expected['caches'])
    assert equal([value for event, value in events if event.endswith('_gradients')], expected['gradients'])
    assert equal([value for event, value in events if event == 'pre_generator_commit_cache'], expected['caches'])
    assert [event for event, value in events][:4] == ['pre_generator_commit_cache', 'generator_gradients',
                                                     'generator_committed', 'guidance_input_cache']
    assert all(not value['image'].requires_grad for value in expected['caches'])
    assert owner.native_step == owner.ratio and owner.generator_updates == 1 and owner.guidance_updates == owner.ratio
    assert result['scheduler_steps'] == dict(generator=owner.ratio, guidance=owner.ratio)
    assert result['ema_updates'] == 0
    assert equal(owner.model.guidance_model.real_unet.state_dict(), before_teacher)
    assert all(parameter.requires_grad for parameter in owner.model.guidance_model.real_unet.parameters())
    assert all(parameter.grad is None for parameter in owner.model.parameters())
    restored = construct(config)[0].state['native_cycle']
    restored.load_state_dict(deepcopy(owner.state_dict()))
    assert equal(restored.state_dict(), owner.state_dict())


def test_common_program_new_process_restart_and_inference_only_current_export(fixture_assets, tmp_path):
    config = Path(fixture_assets['config'])
    partial = run_cli('train', config, '--cycles', 1, '--output', tmp_path / 'partial')
    resumed = run_cli('train', config, '--cycles', 3, '--resume', partial['checkpoint'], '--output', tmp_path / 'resumed')
    full = run_cli('train', config, '--cycles', 3, '--output', tmp_path / 'full')
    a = torch.load(Path(resumed['checkpoint']) / 'state.pt', weights_only=True)
    b = torch.load(Path(full['checkpoint']) / 'state.pt', weights_only=True)
    assert equal(a, b)
    export = run_cli('export', config, '--checkpoint', resumed['checkpoint'], '--output', tmp_path / 'export')
    assert export['model']['factory'] == 'distillschedule.training.sd:build_inference_model'
    assert set(export['model']['modules']) == {'generator', 'text_encoder', 'vae'}
    weights = torch.load(tmp_path / 'export/weights.pt', weights_only=True)
    assert set(weights) == {'generator', 'text_encoder', 'vae'}
    assert 'model_id' not in export['model']['parameters']
    from distillschedule.data.sd_inputs import prepare_inputs, SCHEMA
    fresh_source = tmp_path / 'NEW-prompts.json'
    fresh_source.write_text(json.dumps(dict(schema_version=SCHEMA,
        variant=export['model']['parameters']['variant'], model_id=fixture_assets['assets'],
        records=[dict(id='new-zebra', prompt='new zebra on the moon')])) )
    fresh_inputs = prepare_inputs(fresh_source, tmp_path / 'NEW-prepared')
    inference_config = dict(schema_version='distillschedule.training.v1',
        model=dict(factory=export['model']['factory'], parameters=export['model']['parameters']),
        algorithm=dict(name='dmd2'), data=dict(prepared=str(fresh_inputs)),
        execution=dict(adapter='program', mode='reference'),
        trainer=dict(cycles=1, seed=909, device='cpu', output_dir=str(tmp_path / 'unused')))
    path = tmp_path / 'inference.json'; path.write_text(json.dumps(inference_config))
    from distillschedule.training.session import _load_for_output
    trained = _load_for_output(load_training_config(config), resumed['checkpoint'])[0]
    torch.manual_seed(909)
    with torch.no_grad():
        expected = trained.generate(trained.state, PreparedDataset(fresh_inputs).sample(0), {})
        expected = trained.decode(expected, {})
    # Relocate/remove all training base weights and original training inputs.
    assets = Path(fixture_assets['assets']); assets.rename(assets.with_name('REMOVED_BASE_BUNDLE'))
    import shutil
    shutil.rmtree(Path(fixture_assets['prepared']).parent)
    (Path(fixture_assets['root']) / 'prompts.json').unlink()
    for directory in ('partial', 'resumed', 'full'):
        shutil.rmtree(tmp_path / directory)
    generated = run_cli('generate', path, '--weights', tmp_path / 'export', '--prepared', fresh_inputs,
                        '--output', tmp_path / 'new-image', '--output-kind', 'image')
    assert Path(generated['artifact']).is_file()
    result = torch.load(generated['artifact'], weights_only=True)
    assert result['decoded']['image'].shape == (16, 16, 3)
    assert result['decoded']['image'].dtype == torch.uint8
    assert equal(result['decoded'], expected)


@pytest.mark.parametrize('corruption', ('moment_dtype', 'moment_missing', 'ownership', 'scheduler',
                                      'native_partial', 'teacher_mask', 'teacher_tensor', 'uncond'))
def test_restore_rejects_corruption_before_mutating_live_owner(fixture_assets, corruption):
    _, components, feeds = model_and_feeds(fixture_assets)
    owner = components.state['native_cycle']
    owner.run(feeds, cycle=0)
    before = deepcopy(owner.state_dict()); bad = deepcopy(before)
    if corruption == 'moment_dtype':
        next(iter(bad['optimizers']['guidance']['state'].values()))['exp_avg'] = next(
            iter(bad['optimizers']['guidance']['state'].values()))['exp_avg'].double()
    elif corruption == 'moment_missing':
        bad['optimizers']['generator']['state'].pop(next(iter(bad['optimizers']['generator']['state'])))
    elif corruption == 'ownership':
        bad['optimizer_ownership']['guidance'][0][0] = 'guidance_model.real_unet.teacher'
    elif corruption == 'scheduler':
        bad['schedulers']['generator']['last_epoch'] -= 1
    elif corruption == 'native_partial':
        bad['native_step'] -= 1
    elif corruption == 'teacher_mask':
        name = next(name for name in bad['requires_grad'] if name.startswith('guidance_model.real_unet.'))
        bad['requires_grad'][name] = False
    elif corruption == 'teacher_tensor':
        name = next(name for name in bad['model'] if name.startswith('guidance_model.real_unet.'))
        bad['model'][name].add_(1)
    elif corruption == 'uncond':
        if owner.args.sdxl:
            bad['uncond_embedding'] = torch.zeros(1, 8, 16)
        else:
            bad['uncond_embedding'].add_(1)
    with pytest.raises(ValueError):
        owner.load_state_dict(bad)
    assert equal(owner.state_dict(), before)
    with pytest.raises(ValueError, match='outer Program'):
        owner.validate_checkpoint_progress(before, dict(cycle=1, cursor=2, versions=dict(native_cycle=1)))


def test_native_sdxl_backward_prefix_and_overwritten_999_rng_draws(fixture_assets):
    _, components, feeds = model_and_feeds(fixture_assets)
    owner = components.state['native_cycle']
    if not owner.args.sdxl:
        # SD1.5 correctly rejects original SDXL-only denoising.
        with pytest.raises(AssertionError, match='only supported for SDXL'):
            owner.model.prepare_denoising_data(feeds['generator'][0], None, torch.randn(1, 4, 8, 8))
        return
    from distillschedule.training.sd import original
    model = owner.model; model.eval()
    api = original(owner.source.root)
    row = feeds['denoising'][0]
    noise = torch.linspace(-1, 1, 256).reshape(1, 4, 8, 8)
    selected_all = set()
    for seed in range(30):
        torch.manual_seed(seed)
        actual = model.prepare_denoising_data(deepcopy(row), None, noise)
        actual_rng = rng_state()
        torch.manual_seed(seed)
        # The first indices are deliberately consumed even though backward
        # simulation overwrites them. Prefix noise is uncorrelated with noise.
        torch.randint(0, 4, (1,), dtype=torch.long)
        embedding, pooled = model.text_encoder(row)
        prefix = torch.randn_like(noise)
        selected = int(torch.randint(0, 4, (1,), dtype=torch.long))
        selected_all.add(selected)
        clean = prefix
        conditions = dict(time_ids=model.add_time_ids, text_embeds=pooled)
        with torch.no_grad():
            for constant in model.denoising_step_list[:selected]:
                times = torch.full((1,), int(constant), dtype=torch.long)
                predicted = model.feedforward_model(prefix, times, embedding, added_cond_kwargs=conditions).sample
                clean = api.utils.get_x0_from_noise(prefix, predicted.double(), model.alphas_cumprod.double(), times).float()
                prefix = model.noise_scheduler.add_noise(clean, torch.randn_like(clean), times - 250).to(prefix.dtype)
        times = torch.full((1,), int(model.denoising_step_list[selected]), dtype=torch.long)
        expected = model.noise_scheduler.add_noise(clean, noise, times)
        expected[times == 999] = noise[times == 999]
        assert torch.equal(actual[0], times) and torch.equal(actual[-1], expected)
        assert equal(actual_rng, rng_state())
        if selected == 0:
            assert torch.equal(actual[-1], noise), 't999 must overwrite the add_noise result with pure outer noise'
        if len(selected_all) == 4:
            break
    assert selected_all == {0, 1, 2, 3}


def test_native_cfg_unconditional_embeddings_and_time_id_order(fixture_assets):
    _, components, feeds = model_and_feeds(fixture_assets)
    owner = components.state['native_cycle']; captured = []
    def record(module, args, kwargs):
        captured.append((cpu_tree(args), cpu_tree(kwargs)))
    handle = owner.model.guidance_model.real_unet.register_forward_pre_hook(record, with_kwargs=True)
    owner.run(feeds, cycle=0); handle.remove()
    args, kwargs = captured[0]
    assert args[0].shape[0] == 2 and args[2].shape[0] == 2
    if owner.args.sdxl:
        assert torch.equal(args[2][:1], torch.zeros_like(args[2][:1]))
        conditions = kwargs['added_cond_kwargs']
        assert torch.equal(conditions['text_embeds'][:1], torch.zeros_like(conditions['text_embeds'][:1]))
        assert conditions['time_ids'].tolist() == [[16., 16., 0., 0., 16., 16.]] * 2
    else:
        assert torch.equal(args[2][:1], owner.uncond_embedding)
        assert kwargs['added_cond_kwargs'] is None


@pytest.mark.parametrize('sdxl', (False, True))
def test_actual_unmodified_1280_channel_gan_head_forward_backward(source_root, tmp_path, sdxl):
    from distillschedule.training.plugins import load_callable
    from distillschedule.training.sd import original, _arguments, _LocalAccelerator
    builder, _ = load_callable(str(REPO / 'recipes/dmd2_sd15/create_cpu_fixture.py') + ':build_fixture')
    files = builder(tmp_path / 'head', source_root, variant='sdxl' if sdxl else 'sd15', cycles=1)
    parameters = load_training_config(files['config']).model['parameters']
    args = _arguments(parameters, sdxl=sdxl); args.cls_on_clean_image = True
    guidance = original(source_root).guidance.SDGuidance(args, _LocalAccelerator(torch.device('cpu')))
    head = guidance.cls_pred_branch
    assert sum(parameter.numel() for parameter in head.parameters()) == (104874241 if sdxl else 52437761)
    # The original classify_forward also runs on the actual tiny UNet; its 16
    # bottleneck channels are NOT silently stretched to the production 1280.
    embedding = torch.randn(1, 8, 16 if sdxl else 8)
    conditions = dict(time_ids=torch.tensor([[16., 16., 0., 0., 16., 16.]]), text_embeds=torch.randn(1, 8)) if sdxl else None
    representations = guidance.fake_unet(torch.randn(1, 4, 8, 8), torch.zeros(1, dtype=torch.long), embedding,
        added_cond_kwargs=conditions, classify_mode=True)
    assert representations[-1].shape[1] == 16
    representation = torch.randn(1, 1280, 32 if sdxl else 8, 32 if sdxl else 8, requires_grad=True)
    logits = head(representation)
    assert logits.shape == (1, 1, 1, 1)
    torch.nn.functional.softplus(-logits).mean().backward()
    assert representation.grad is not None and bool(torch.isfinite(representation.grad).all())
    assert all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) for parameter in head.parameters())


def test_actual_sdxl_lora_ratio_one_and_inference_projection(source_root, tmp_path):
    from distillschedule.training.plugins import load_callable
    from distillschedule.training.sd import build_inference_model
    builder, _ = load_callable(str(REPO / 'recipes/dmd2_sd15/create_cpu_fixture.py') + ':build_fixture')
    files = builder(tmp_path / 'lora', source_root, variant='sdxl', cycles=1, ratio=1, lora=True)
    _, components, feeds = model_and_feeds(files)
    owner = components.state['native_cycle']
    assert set(feeds) == {'generator', 'denoising'}
    assert all('lora_' in name for name, parameter in owner.model.feedforward_model.named_parameters() if parameter.requires_grad)
    owner.run(feeds, cycle=0)
    owner.validate_state_dict(owner.state_dict())
    projection = components.inference_projection(components, {})
    clone = build_inference_model(deepcopy(projection.parameters))
    for name, module in clone.state.items():
        module.load_state_dict(projection.components.state[name].state_dict(), strict=True)
        module.eval(); projection.components.state[name].eval()
    with torch.no_grad():
        torch.manual_seed(123); a = clone.generate(clone.state, feeds['generator'][0], {})
        torch.manual_seed(123); b = projection.components.generate(projection.components.state, feeds['generator'][0], {})
    assert equal(a, b)


def test_sdxl_single_step_explicit_checkpoint_native399_parity_and_missing_checkpoint_rejection(source_root, tmp_path):
    from distillschedule.training.plugins import load_callable
    from distillschedule.training.sd import build_sdxl
    from safetensors.torch import load_file
    builder, _ = load_callable(str(REPO / 'recipes/dmd2_sd15/create_cpu_fixture.py') + ':build_fixture')
    files = builder(tmp_path / 'single', source_root, variant='sdxl', cycles=1, ratio=1)
    value = json.loads(Path(files['config']).read_text())
    value['model']['parameters']['settings'].update(num_denoising_step=1, conditioning_timestep=399,
                                                   denoising=False, backward_simulation=False)
    with pytest.raises(ValueError, match='conditioning399|requires an explicit actual generator checkpoint'):
        build_sdxl(value['model']['parameters'])
    checkpoint = tmp_path / 'EXPLICIT_RANDOM_TINY_TEST_CHECKPOINT.pt'
    torch.save(load_file(str(Path(files['assets']) / 'unet/diffusion_pytorch_model.safetensors')), checkpoint)
    value['model']['parameters']['generator_checkpoint'] = str(checkpoint)
    value['data']['streams'].pop('denoising')
    Path(files['config']).write_text(json.dumps(value))
    _, components, feeds = model_and_feeds(files)
    owner = components.state['native_cycle']; reference = deepcopy(owner)
    outer = rng_state(); expected = _oracle(reference, feeds, source_root); restore_rng(outer)
    actual = owner.run(feeds, cycle=0)
    assert actual['iterations'][0]['denoising_timesteps'] == [399]
    assert actual['iterations'][0]['losses'] == expected['losses'][0]
    assert equal(owner.model.state_dict(), reference.model.state_dict())
    assert equal(owner._rng, expected['rng'])


def test_prompt_preparation_preserves_exact_latents_and_rejects_invalid_layout(fixture_assets, tmp_path):
    from distillschedule.data.sd_inputs import prepare_inputs, validate_row, SCHEMA
    variant = json.loads(Path(fixture_assets['config']).read_text())['model']['name']
    latent = torch.linspace(-2, 2, 256).reshape(1, 4, 8, 8)
    path = tmp_path / 'native.pt'; torch.save(latent, path)
    source = tmp_path / 'new-source.json'
    value = dict(schema_version=SCHEMA, variant=variant, model_id=fixture_assets['assets'],
        preencoded_vae_compatibility='PREENCODED_LATENTS_MATCH_LOCAL_VAE',
        records=[dict(id='native', prompt='new native prompt', latents=dict(path=str(path),
                         sha256=hashlib.sha256(path.read_bytes()).hexdigest()))])
    source.write_text(json.dumps(value))
    prepared = prepare_inputs(source, tmp_path / 'native-prepared')
    data = PreparedDataset(prepared)
    assert data.validate(sources=True)['source_validation'] == 'PASS'
    row = data.sample(0); assert torch.equal(row['images'], latent)
    for bad in (dict(row, images=latent.double()),
                dict(row, text_input_ids_one=row['text_input_ids_one'].squeeze(1)),
                dict(row, text_input_ids_one=row['text_input_ids_one'].float())):
        with pytest.raises(ValueError, match='SD input'):
            validate_row(bad, sdxl=variant == 'sdxl', binding=data.manifest['metadata']['preprocessing_assets'])
    value['records'][0]['latents']['sha256'] = '0' * 64
    source.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        prepare_inputs(source, tmp_path / 'bad-input')
    assert not (tmp_path / 'bad-input').exists()
