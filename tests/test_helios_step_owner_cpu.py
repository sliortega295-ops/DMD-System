"""Full original model + true PEFT + original native-step CPU proof only.

Separate Diffusers 0.37 group. The tiny deterministic frozen output is NOT
pretrained, canonical GAN-start-1000 execution, native world16, or public W3.
"""
from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path
import random
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from distillschedule.training.checkpoint import pack_state, rng_state
from distillschedule.workloads.best_known.registry import repository_root


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def _config():
    from distillschedule.strategies.wan_capture import native_primitives
    native_primitives()
    from helios.utils.train_config import Args
    from omegaconf import OmegaConf
    args = OmegaConf.structured(Args)
    args.training_config.update(dict(is_train_dmd=True, is_use_gan=True,
        is_use_gan_final=True, is_use_gan_hooks=False, gan_cond_map_dim=32,
        is_gan_low_vram_mode=True, is_separate_gan_grad=True,
        is_enable_stage2=True, stage2_num_stages=3, has_multi_term_memory_patch=True,
        history_sizes=[4, 2, 1], latent_window_size=[9], num_critic_input_frames=9,
        dmd_denoising_step_list=[1000, 500], dmd_timestep_shift=1.,
        dmd_num_latent_sections_min=1, dmd_num_latent_sections_max=1,
        use_ema=True, ema_start_step=2, ema_decay=.99, gan_start_step=3,
        mixed_precision='no', learning_rate=1e-4, critic_learning_rate=1e-4,
        lr_scheduler='constant_with_warmup', lr_warmup_steps=2, max_train_steps=20,
        is_use_gt_history=True, use_gt_history_ratio=.6,
        noise_corrupt_clean_prob_history=1., gradient_checkpointing=False,
        r1_weight=.01, r2_weight=.01))
    args.model_config.update(dict(lora_rank=2, lora_alpha=2., critic_lora_rank=2,
        critic_lora_alpha=2., lora_target_modules=['to_q', 'to_k', 'to_v']))
    args.validation_config.stage2_simulated_inference_steps = [1, 1, 1]
    args.validation_config.use_kv_cache = False
    return args


def _trainer_binding(model, args):
    # ONLY this original name-binding block is compiled. No model/head/loss AST
    # extraction: the actual full transformer/provider and helper are imported.
    path = repository_root() / 'third_party/helios_compat/reference/train_helios.py'
    candidates = [n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.If)
        and any(isinstance(a, ast.Assign) and any(isinstance(k, ast.Name)
            and k.id == 'gan_critic_trainable_params' for k in a.targets) for a in n.body)]
    assert len(candidates) == 1
    namespace = dict(args=args, real_score_model=model, critic_trainable_modules=['gan_heads', 'gan_final_head'])
    exec(compile(ast.Module(body=candidates, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['gan_base_critic_trainable_params'], namespace['gan_extra_critic_trainable_params']


def _build(*, seed=71, horizon=20, mutate=None, edit_owners=None):
    from peft import LoraConfig
    from distillschedule.adapters.helios.reference_model import load_reference_model_class
    from distillschedule.runtime.fsdp2_static import FSDP2EngineAdapter
    from distillschedule.strategies.wan_capture import native_primitives
    from runtime.sparse_ema import SparseShardEMA
    from distillschedule.training.helios_step_owner import HeliosStepOwner
    torch.set_num_threads(1)
    args = _config()
    if mutate:
        mutate(args)
    cls = load_reference_model_class(backend='torch-sdpa')
    engines = []
    for role, adapter_seed in [('generator', seed), ('critic', seed + 17)]:
        torch.manual_seed(131)
        model = cls(patch_size=(1, 2, 2), has_multi_term_memory_patch=True,
            num_attention_heads=2, attention_head_dim=16, in_channels=16, out_channels=16,
            text_dim=4, freq_dim=16, ffn_dim=64, num_layers=1, rope_dim=(4, 6, 6),
            is_use_gan=role == 'critic', is_use_gan_final=role == 'critic',
            is_use_gan_hooks=False, gan_cond_map_dim=32)
        with torch.no_grad():
            model.proj_out.weight.normal_(mean=0., std=.02, generator=torch.Generator().manual_seed(1310))
        model.requires_grad_(False)
        torch.manual_seed(adapter_seed)
        model.add_adapter(LoraConfig(r=2, lora_alpha=2, lora_dropout=0.,
            init_lora_weights='gaussian', target_modules=['to_q', 'to_k', 'to_v'],
            exclude_modules=[] if role == 'generator' else
                ['patch_short','patch_mid','patch_long','patch_embedding','gan_heads','gan_final_head']))
        if role == 'critic':
            for name, parameter in model.named_parameters():
                if name.startswith('gan_final_head.'):
                    parameter.requires_grad_(True)
        from diffusers.optimization import get_scheduler
        c = args.training_config
        lr = c.learning_rate if role == 'generator' else c.critic_learning_rate
        optimizer = native_primitives().base.get_optimizer(args, SimpleNamespace(print=print),
            [{'params': [p for p in model.parameters() if p.requires_grad], 'lr': lr}], use_deepspeed=False)
        lr_scheduler = get_scheduler(c.lr_scheduler, optimizer=optimizer,
            num_warmup_steps=c.lr_warmup_steps, num_training_steps=c.max_train_steps,
            num_cycles=c.lr_num_cycles, power=c.lr_power)
        engines.append(FSDP2EngineAdapter(model, optimizer,
            lr_scheduler,
            physical_step=0, global_batch=1, dp_world_size=1))
    g, c = engines
    if edit_owners:
        edit_owners(g, c)
    base, extra = _trainer_binding(c.module, args)
    ema = SparseShardEMA(g.module.named_parameters(), decay=args.training_config.ema_decay,
        update_after_step=args.training_config.ema_start_step,
        base_model='DETERMINISTIC_TINY_FROZEN_FIXTURE_NOT_PRETRAINED')
    scheduler = lambda: native_primitives().scheduler(shift=1., stages=3,
        stage_range=[0, 1/3, 2/3, 1], gamma=1/3)
    source_rng = torch.Generator().manual_seed(811)
    draw = lambda shape: torch.randn(shape, generator=source_rng)
    inputs = []
    for t in range(horizon):
        def history():
            return dict(gt_history_latents=draw((1, 16, 7, 16, 16)),
                gt_target_latents=draw((1, 16, 9, 16, 16)),
                gt_x0_latents=draw((1, 16, 1, 16, 16)))
        inputs.append(dict(identity=f'tiny-fixture-distinct-source-{t}',
            prompt_embeds=draw((1, 2, 4)), gan_prompt_embeds=draw((1, 2, 4)),
            gan_vae_latents=draw((1, 16, 9, 16, 16)),
            generator_history=history(), critic_history=history()))
    torch.manual_seed(773)
    random.seed(773)
    import numpy as np
    np.random.seed(773)
    return HeliosStepOwner(generator=g, critic=c, ema=ema, args=args,
        generator_scheduler=scheduler(), critic_scheduler=scheduler(),
        negative_prompt_embeds=draw((1, 2, 4)), noise_shape=(1, 16, 9, 16, 16),
        inputs=inputs, cohort='TINY_FIXTURE_GAN3_EMA2_NOT_CANONICAL',
        gan_base_critic_trainable_params=base, gan_extra_critic_trainable_params=extra,
        frozen_parameter_names={n for n, p in c.module.named_parameters() if not p.requires_grad},
        backend='torch-sdpa')


def test_original_owner_runs_five_atomic_steps_with_real_g_and_all_five_c_rollouts():
    owner = _build()
    original_rng = rng_state()
    source_calls = []
    generator_entries = []
    critic_active = False
    generator_rollouts = []
    def profile(frame, event, arg):
        nonlocal critic_active
        if frame.f_code is owner.native.post._critic_loss.__code__:
            if event == 'call':
                critic_active = True
                source_calls.append(('critic', frame.f_locals['global_step']))
                assert frame.f_locals['precomputed_generated_image_or_video'] is None
                assert frame.f_locals['transformer'] is owner.generator.module
            elif event == 'return':
                critic_active = False
        if event == 'call' and frame.f_code is owner.native.post._generator_loss.__code__:
            source_calls.append(('generator', frame.f_locals['global_step']))
            generator_entries.append((frame.f_locals['noise'].detach().clone(),
                {n: p.requires_grad for n,p in owner.critic.module.named_parameters()}))
    handle = owner.generator.module.register_forward_pre_hook(
        lambda *args: generator_rollouts.append(owner.global_step) if critic_active else None)
    try:
        sys.setprofile(profile)
        results = [owner.step() for _ in range(5)]
    finally:
        sys.setprofile(None)
        handle.remove()
    assert owner.global_step == 5
    assert owner.versions == dict(generator=1, critic=5, ema=2)
    assert [r['train_generator'] for r in results] == [True, False, False, False, False]
    assert [r['use_gan'] for r in results] == [False, False, False, True, True]
    assert all(torch.isfinite(torch.tensor(r['critic_loss'])) for r in results)
    assert equal(original_rng, rng_state()), 'native owned RNG escaped into the outer Program scope'
    assert owner.generator.dp_world_size == owner.critic.dp_world_size == 1
    assert source_calls == [('generator', 0), *[('critic', t) for t in range(5)]]
    expected_entry = torch.randn(owner.noise_shape, generator=torch.Generator().manual_seed(773))
    assert torch.equal(generator_entries[0][0], expected_entry), 'original trainer entry noise draw was elided'
    assert all(generator_entries[0][1][n] for n in owner.base | owner.extra), 'preGAN C was unconditionally frozen'
    assert set(generator_rollouts) == set(range(5))
    assert all(generator_rollouts.count(t) >= 3 for t in range(5)), 'missing full original three-stage C rollout'
    assert len({r['critic_logs']['critictrain_latent']['sha256'] for r in results}) == 5
    assert [r['input_identity'] for r in results] == [r['identity'] for r in owner.inputs[:5]]
    assert results[0]['generator_events'] == [['backward', True], ['commit', True]]
    assert all(r['critic_events'] == ([['backward', False], ['backward', True], ['commit', True]]
        if r['use_gan'] else [['backward', True], ['commit', True]]) for r in results)
    assert any(r['use_gt_history'] for r in results) and not all(r['use_gt_history'] for r in results)
    c_named = dict(owner.critic.module.named_parameters())
    assert all(not c_named[n].requires_grad for n in owner.base)
    assert all(c_named[n].requires_grad for n in owner.extra)
    assert not torch.cuda.is_initialized()


def _cycle(owner):
    return [owner.step() for _ in range(5)]


def test_canonical_original_curriculum_policy_is_not_the_old_systems_bootstrap():
    from distillschedule.training.helios_step_owner import expected_clocks
    assert expected_clocks(0) == dict(generator=0, critic=0, ema=0)
    assert expected_clocks(1000) == dict(generator=200, critic=1000, ema=800)
    assert expected_clocks(1005) == dict(generator=201, critic=1005, ema=801)
    assert expected_clocks(1010) == dict(generator=202, critic=1010, ema=802)


def test_native_rng_remains_continuous_inside_repeated_program_random_scopes():
    from distillschedule.open_workload.torch_backend import random_scope
    control = _build(horizon=5)
    expected = _cycle(control)
    scoped = _build(horizon=5)
    results = []
    for _ in range(5):
        # Even an identical outer call seed cannot reset the owner's genuine
        # advancing Python/Torch/NumPy native stream between atomic calls.
        with random_scope(123, 'repeated-outer-call', torch.device('cpu')):
            before = rng_state()
            results.append(scoped.step())
            assert equal(before, rng_state())
    assert equal(expected, results)
    assert equal(pack_state({'native_step': control}), pack_state({'native_step': scoped}))


def test_dynamic_timestep_and_gt_helper_options_reach_actual_source_calls():
    def config(args):
        args.training_config.use_dynamic_shifting = True
        args.training_config.use_gt_history_ratio = 1.
        args.training_config.generator_dynamic_sample_type = 'beta'
        args.training_config.critic_dynamic_sample_type = 'beta'
        args.training_config.generator_dynamic_step = 7
        args.training_config.critic_dynamic_step = 9
        args.training_config.generator_is_forcing_low_renoise = True
    owner = _build(horizon=10, mutate=config)
    calls = []
    def profile(frame, event, arg):
        if event == 'call' and frame.f_code in (owner.native.post._generator_loss.__code__,
                                                owner.native.post._critic_loss.__code__):
            f = frame.f_locals
            is_g = frame.f_code is owner.native.post._generator_loss.__code__
            assert f['global_step'] == owner.global_step
            assert f['use_dynamic_shifting'] is True and f['dynamic_sample_type'] == 'beta'
            assert f['dynamic_step'] == (7 if is_g else 9)
            assert f['is_use_gt_history'] is True and f['gt_history_latents'] is not None
            assert f['is_add_saturation'] is False
            if is_g:
                flags = dict(owner.critic.module.named_parameters())
                assert all(flags[n].requires_grad is (not f['is_use_gan']) for n in owner.base | owner.extra)
            calls.append((is_g, f['global_step']))
    try:
        sys.setprofile(profile)
        _cycle(owner)
        _cycle(owner)
    finally:
        sys.setprofile(None)
    assert [t for g,t in calls if g] == [0,5]
    assert [t for g,t in calls if not g] == list(range(10))


def test_fresh_process_checkpoint_exact_next_cycle_all_original_owners(tmp_path):
    owner = _build()
    results = [_cycle(owner) for _ in range(3)]
    assert owner.versions == dict(generator=3, critic=15, ema=4)
    # Actual Adam state: no synthetic precomputed C samples or hand-set clocks.
    for role in ('generator', 'critic'):
        engine = getattr(owner, role)
        assert engine.optimizer.state
        for p, slots in engine.optimizer.state.items():
            assert slots['step'] > 0
            assert torch.count_nonzero(slots['exp_avg']) > 0
            assert torch.count_nonzero(slots['exp_avg_sq']) > 0
    assert results[1][0]['generator_loss'] != 0
    checkpoint = owner.save(tmp_path / 'checkpoint')
    next_receipts = _cycle(owner)
    expected = dict(state=pack_state({'native_step': owner}), rng=rng_state(), receipts=next_receipts)
    torch.save(expected, tmp_path / 'expected.pt')
    script = r'''
from pathlib import Path
import sys, torch
from test_helios_step_owner_cpu import _build, _cycle, equal
from distillschedule.training.checkpoint import pack_state, rng_state
root = Path(sys.argv[1])
fresh = _build(seed=999)
parameters = {r: {n: id(p) for n,p in getattr(fresh,r).module.named_parameters()}
              for r in ('generator','critic')}
hooks = []
hook_handles = [next(iter(getattr(fresh,r).optimizer.param_groups[0]['params'])).register_hook(
    lambda grad: hooks.append(True)) for r in ('generator','critic')]
fresh.restore(root/'checkpoint')
for r in parameters:
    assert parameters[r] == {n: id(p) for n,p in getattr(fresh,r).module.named_parameters()}
receipts = _cycle(fresh)
expected = torch.load(root/'expected.pt', weights_only=True)
assert equal(expected['receipts'], receipts), 'actual original loss/timestep/input receipts drifted'
assert equal(expected['state'], pack_state({'native_step':fresh})), 'native trainables/Adam/flags/EMA/RNG drifted'
assert equal(expected['rng'], rng_state()), 'outer checkpoint RNG drifted'
assert fresh.versions == dict(generator=4,critic=20,ema=5)
assert hooks and not torch.cuda.is_initialized()
for handle in hook_handles: handle.remove()
print('EXACT_FRESH_PROCESS_ORIGINAL_G_PLUS_FIVE_FULL_C_CHECKPOINT_PASS')
'''
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    process = subprocess.run([sys.executable, '-c', script, str(tmp_path)], cwd=repository_root(),
        env=env, text=True, capture_output=True, timeout=180)
    assert process.returncode == 0, process.stdout + process.stderr
    assert 'EXACT_FRESH_PROCESS' in process.stdout
    evidence = os.environ.get('HELIOS_STEP_OWNER_EVIDENCE')
    if evidence:
        import hashlib
        import json
        from distillschedule.training.helios_step_owner import _content
        report = dict(scope='FULL_ORIGINAL_TINY_CPU_PEFT_GAN3_EMA2_NOT_PRETRAINED_OR_PUBLIC_W3',
            torch=str(torch.__version__), runtime=owner.identity['runtime'],
            source=owner.identity['source'], native_owner_sources=owner.identity['native_owner_sources'],
            identity_sha256=hashlib.sha256(json.dumps(owner.identity, sort_keys=True).encode()).hexdigest(),
            final_state_sha256=hashlib.sha256(json.dumps(_content(expected['state']), sort_keys=True).encode()).hexdigest(),
            global_step=owner.global_step, versions=owner.versions,
            actual_original_generator_calls=4, actual_original_full_critic_calls=20,
            fresh_process_next_cycle_exact=True, actual_live_parameter_and_hooks_preserved=True,
            real_adam_nonzero_moments=True, cuda_initialized=torch.cuda.is_initialized(),
            fresh_process_stdout=process.stdout.strip(),
            checkpoint_manifest_sha256=hashlib.sha256((checkpoint/'manifest.json').read_bytes()).hexdigest(),
            receipts=[r for cycle in results for r in cycle] + next_receipts)
        Path(evidence).write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')


def _raw(owner):
    # Do not use the owner's guards to inspect state after intentionally
    # invalidating a live policy; unchanged means unchanged, not normalized.
    return deepcopy(dict(native=pack_state(owner._views), rng=rng_state(), native_rng=owner._rng,
                         global_step=owner.global_step, identity=owner.identity))


@pytest.mark.parametrize('change', ['config', 'input', 'negative', 'frozen', 'ownership',
    'phase', 'progress', 'source', 'adapter', 'cache', 'gradient', 'accumulation', 'names'])
def test_live_admission_rejects_before_model_update_or_rng_mutation(tmp_path, monkeypatch, change):
    from distillschedule.adapters.helios import reference_model
    owner = _build(horizon=5)
    if change == 'config':
        owner.args.training_config.gan_start_step = 1
    elif change == 'input':
        owner.inputs[4]['gan_vae_latents'].add_(1)
    elif change == 'negative':
        owner.negative.add_(1)
    elif change == 'frozen':
        with torch.no_grad():
            owner.generator.module.proj_out.weight.add_(1)
    elif change == 'ownership':
        group = owner.generator.optimizer.param_groups[0]
        group['params'] = list(reversed(group['params']))
    elif change == 'phase':
        named = dict(owner.critic.module.named_parameters())
        for name in owner.base:
            named[name].requires_grad = False
    elif change == 'progress':
        owner.global_step = 5
    elif change == 'source':
        monkeypatch.setattr(reference_model, '_MANIFEST_SHA256', '0' * 64)
    elif change == 'adapter':
        owner.generator.module.disable_adapters()
    elif change == 'cache':
        owner.generator.module.blocks[0].attn1.processor.cache_enabled = True
    elif change == 'gradient':
        p = owner.generator.optimizer.param_groups[0]['params'][0]
        p.grad = torch.ones_like(p)
    elif change == 'accumulation':
        owner.critic._gradient_accumulation_boundary = False
    elif change == 'names':
        owner.base = set()
    # Some native serialization views themselves reject invalid phases or
    # ownership. Capture the raw tensors independently in those cases.
    tensors = lambda: {r: {n: p.detach().clone() for n,p in getattr(owner,r).module.named_parameters()}
                       for r in ('generator','critic')}
    before, outer, native_rng = tensors(), rng_state(), deepcopy(owner._rng)
    counters = owner.versions.copy()
    for action in (owner.step, lambda: owner.save(tmp_path / 'refused')):
        with pytest.raises(ValueError):
            action()
        assert equal(before, tensors()) and owner.versions == counters
        assert equal(outer, rng_state()) and equal(native_rng, owner._rng)
    assert not (tmp_path/'refused').exists()


@pytest.fixture(scope='module')
def completed_state():
    owner = _build()
    _cycle(owner)
    return deepcopy(owner.state_dict())


@pytest.mark.parametrize('change', ['missing', 'source', 'ownership', 'progress', 'version', 'phase',
    'ema', 'rng', 'scheduler', 'optimizer', 'cache', 'horizon', 'scheduler_horizon', 'adapters', 'dtype', 'input',
    'adam_missing_slot', 'adam_step'])
def test_checkpoint_stages_all_owners_before_any_live_mutation(completed_state, change):
    owner = _build(seed=999)
    value = deepcopy(completed_state)
    native = value['native']
    if change == 'missing':
        del native['critic.optimizer']
    elif change == 'source':
        value['identity']['source']['source']['content_tree_sha256'] = '0' * 64
    elif change == 'ownership':
        native['critic.optimizer']['parameters'][0].reverse()
    elif change == 'progress':
        value['global_step'] = 10
    elif change == 'version':
        value['versions']['generator'] = 200
    elif change == 'phase':
        weights = native['critic.weights']
        weights['phase'] = 'ready'
        for n in owner.base:
            weights['requires_grad'][n] = True
    elif change == 'ema':
        native['ema']['optimization_step'] = 800
    elif change == 'rng':
        del value['rng']['torch']
    elif change == 'scheduler':
        native['generator.scheduler']['last_epoch'] = 9
    elif change == 'optimizer':
        native['critic.optimizer']['optimizer']['param_groups'][0]['weight_decay'] = .31
    elif change == 'cache':
        native['critic.noise_scheduler']['state']['last_sample'] = torch.ones(1)
    elif change == 'horizon':
        value['identity']['config']['validation_config']['stage2_simulated_inference_steps'] = [2,2,2]
    elif change == 'scheduler_horizon':
        native['critic.noise_scheduler']['state']['num_inference_steps'] = 7
    elif change == 'adapters':
        value['adapters']['critic']['active'] = ['foreign']
    elif change == 'dtype':
        name = next(iter(native['critic.weights']['tensors']))
        native['critic.weights']['tensors'][name] = native['critic.weights']['tensors'][name].double()
    elif change == 'input':
        value['identity']['inputs'][4]['identity'] = 'foreign-source'
    elif change == 'adam_missing_slot':
        slots = native['critic.optimizer']['optimizer']['state']
        del slots[next(iter(slots))]
    elif change == 'adam_step':
        slots = native['critic.optimizer']['optimizer']['state']
        slots[next(iter(slots))]['step'].fill_(1)
    before = _raw(owner)
    with pytest.raises(ValueError):
        owner.load_state_dict(value)
    assert equal(before, _raw(owner))


def test_only_full_five_step_cycles_publish_and_mid_backward_refuses(tmp_path):
    owner = _build(horizon=5)
    probes = []
    original = owner.critic.backward
    def backward(loss):
        original(loss)
        with pytest.raises(ValueError, match='outstanding native step'):
            owner.save(tmp_path/'mid-backward')
        probes.append(owner.critic._gradient_accumulation_boundary)
    owner.critic.backward = backward
    owner.step()
    with pytest.raises(ValueError, match='five-native-step cycle'):
        owner.save(tmp_path/'partial')
    for _ in range(4):
        owner.step()
    assert probes == [True, True, True, False, True, False, True]
    owner.save(tmp_path/'complete')
    assert (tmp_path/'complete/_SUCCESS').is_file()
    assert not (tmp_path/'mid-backward').exists() and not (tmp_path/'partial').exists()
    with pytest.raises(ValueError, match='exhausted'):
        owner.step()


def test_partial_runtime_failure_preserves_outer_rng_but_never_replays_update(tmp_path):
    owner = _build(horizon=5)
    outer = rng_state()
    def fail(loss):
        raise RuntimeError('explicit critic backward failure')
    owner.critic.backward = fail
    with pytest.raises(RuntimeError, match='explicit critic backward failure'):
        owner.step()
    assert owner.generator.step_calls == 1 and owner.critic.step_calls == 0
    assert owner.global_step == 0 and equal(outer, rng_state())
    for action in (owner.step, lambda: owner.save(tmp_path/'broken')):
        with pytest.raises(ValueError, match='failed owner'):
            action()
    assert owner.generator.step_calls == 1 and not (tmp_path/'broken').exists()


@pytest.mark.parametrize('branch', ['is_use_reward_model', 'is_use_ode_regression', 'use_error_recycling',
    'is_smoothness_loss', 'corrupt_model_input', 'is_multi_pyramid_stage_backward_simulated', 'is_use_gan_hooks'])
def test_unsupported_original_branches_reject_at_binding_without_update(branch):
    with pytest.raises(ValueError, match='unsupported'):
        _build(horizon=5, mutate=lambda args: setattr(args.training_config, branch, True))


def test_native_engine_world_size_metadata_is_explicit_without_changing_original_default():
    from distillschedule.runtime.fsdp2_static import FSDP2EngineAdapter
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
    assert FSDP2EngineAdapter(model, optimizer, scheduler, physical_step=0).dp_world_size == 16
    assert FSDP2EngineAdapter(model, optimizer, scheduler, physical_step=0, dp_world_size=1).dp_world_size == 1
    for size in (0, -1, True, 1.5, '1'):
        with pytest.raises(ValueError, match='positive integer'):
            FSDP2EngineAdapter(model, optimizer, scheduler, physical_step=0, dp_world_size=size)


@pytest.mark.parametrize('field,value', [('mixed_precision','bf16'), ('train_batch_size',2),
    ('gradient_accumulation_steps',2)])
def test_cpu_owner_rejects_unbound_precision_batch_or_accumulation_before_updates(field, value):
    with pytest.raises(ValueError, match='CPU1|precision|batch|accumulation'):
        _build(horizon=5, mutate=lambda args: setattr(args.training_config, field, value))


def test_native_optimizer_construction_must_match_original_role_config():
    def wrong(g,c):
        c.optimizer.param_groups[0]['betas'] = (.1, .2)
    with pytest.raises(ValueError, match='optimizer.*config'):
        _build(horizon=5, edit_owners=wrong)


def test_cpu_owner_refuses_unadmitted_actual_process_group_before_updates(monkeypatch):
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)
    monkeypatch.setattr(torch.distributed, 'get_world_size', lambda: 2)
    with pytest.raises(ValueError, match='CPU1|topology|process group'):
        _build(horizon=5)


def test_full_owner_nonseparate_gan_policy_runs_full_c_and_restores_exactly(tmp_path):
    config = lambda args: setattr(args.training_config, 'is_separate_gan_grad', False)
    owner = _build(horizon=10, mutate=config)
    receipts = _cycle(owner)
    assert owner._views['critic.weights'].state_dict()['phase'] == 'ready'
    named = dict(owner.critic.module.named_parameters())
    assert all(named[n].requires_grad for n in owner.base | owner.extra)
    assert all(r['critic_events'] == [['backward', False], ['backward', True], ['commit', True]]
               for r in receipts if r['use_gan'])
    owner.save(tmp_path/'nonseparate')
    expected = _cycle(owner)
    fresh = _build(seed=999, horizon=10, mutate=config)
    fresh.restore(tmp_path/'nonseparate')
    assert equal(expected, _cycle(fresh))
    assert equal(pack_state({'native_step':owner}), pack_state({'native_step':fresh}))


def test_legitimate_cold_zero_dmd_gradient_is_not_rejected_or_replaced():
    owner = _build(horizon=5, mutate=lambda args: setattr(args.training_config, 'real_guidance_scale', 0.))
    receipt = owner.step()
    assert receipt['generator_loss'] == 0.
    assert owner.versions == dict(generator=1, critic=1, ema=1)
    assert owner.generator.optimizer.state
    assert all(torch.count_nonzero(slot['exp_avg']) == 0 for slot in owner.generator.optimizer.state.values())
