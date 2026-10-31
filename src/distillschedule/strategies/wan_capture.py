"""Capture NEW, operation-qualified inputs using original model-free primitives.

Reads native pre-encoded clips, never old trace tensors. Native data preparation,
history augmentation, scheduler and timestep sampling remain authoritative.
The new seed defines a paired systems experiment, not continuation of a previous
training RNG stream. No model execution or speedup is claimed by capture.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace

from distillschedule.core.contracts import require
from distillschedule.workloads.best_known.registry import repository_root, file_sha256, write_new_json
from .wan_inputs import WanInputSources, load_feature, preflight_sources
from .wan_task import OriginalWanTask, TRACE_KEY, NATIVE_HISTORY_POLICY


def native_primitives(*, source_only=False):
    """Import actual vendored helpers, without constructing models or CUDA state."""
    if source_only:
        from distillschedule.adapters.helios.reference_helpers import load_reference_primitives
        return load_reference_primitives(backend='torch-sdpa')
    source = repository_root() / 'third_party/helios_compat/source'
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    names = ('helios.scheduler.scheduling_helios', 'helios.utils.utils_helios_base',
             'helios.utils.utils_helios_post', 'helios.utils.utils_base', 'helios.dataset.dataloader_dmd')
    modules = [importlib.import_module(name) for name in names]
    require(all(source.resolve() in Path(m.__file__).resolve().parents for m in modules),
            'native imports resolve to a different Helios checkout', path='wan.capture')
    return SimpleNamespace(scheduler=modules[0].HeliosScheduler, history=modules[1],
                           post=modules[2], base=modules[3], dataset=modules[4].BucketedFeatureDataset)


@contextmanager
def input_stream(seed, *identity):
    """Isolate native CPU/Python RNG draws; never perturb a caller's RNG stream."""
    import torch
    digest = hashlib.sha256(json.dumps([seed, *identity], separators=(',', ':')).encode()).digest()
    derived = int.from_bytes(digest[:8], 'big') % (2**63)
    python_state = random.getstate()
    try:
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(derived)
            random.seed(derived)
            yield
    finally:
        random.setstate(python_state)


def read_training_config(path):
    import yaml
    value = yaml.safe_load(Path(path).read_text())
    require(isinstance(value, dict) and isinstance(value.get('training_config'), dict),
            'original training_config mapping required', path='wan.capture')
    c = value['training_config']
    template = repository_root() / 'configs/distillschedule/templates/helios_stage3_384x640.yaml'
    reference = yaml.safe_load(template.read_text())['training_config']
    input_keys = {
        'is_random_drop', 'random_drop_i2v_ratio', 'random_drop_v2v_ratio', 'random_drop_t2v_ratio',
        'corrupt_mode_history', 'corrupt_mode_prob_history', 'is_frame_independent_corrupt_history',
        'is_chunk_independent_corrupt_history', 'noise_corrupt_ratio_history_short',
        'noise_corrupt_ratio_history_mid', 'noise_corrupt_ratio_history_long',
        'noise_corrupt_clean_prob_history', 'downsample_min_corrupt_ratio_history',
        'downsample_max_corrupt_ratio_history',
        *(role+'_'+key for role in ('generator', 'critic') for key in
          ('dynamic_alpha', 'dynamic_beta', 'dynamic_sample_type', 'dynamic_step')),
    }
    require(set(c) <= set(reference) | {'random_drop_i2v_ratio'} and set(reference) <= set(c),
            'capture accepts the original training_config, not unknown or missing algorithm settings', path='wan.capture')
    changed = [key for key in reference if key not in input_keys and c[key] != reference[key]]
    require(not changed, 'non-input training changes are not implemented by this capture adapter: '+str(changed),
            path='wan.capture')
    # One-section GT-history protocol; branch policy is explicit on the task.
    require(c.get('is_use_gt_history') is True and c.get('is_enable_stage2') is True
            and c.get('stage2_num_stages') == 3 and not c.get('is_use_gan', False)
            and not c.get('is_multi_pyramid_stage_backward_simulated', False)
            and not c.get('is_dmd_vae_decode', False) and not c.get('dmd_last_step_only', False),
            'capture does not substitute another training protocol', path='wan.capture')
    require(c.get('use_dynamic_shifting') is True and c.get('time_shift_type', 'linear') == 'linear',
            'original worker uses linear resolution-dependent schedule shifts', path='wan.capture')
    return c


def prepare_history(native, task, sample, config):
    import torch
    x0, history, target = (x.unsqueeze(0).to(dtype=torch.bfloat16).clone() for x in sample)
    values = native.history.prepare_stage1_clean_input_from_latents(
        history, target, x0, latent_window_size=task.latent_shape[2], history_sizes=[16, 2, 1],
        is_random_drop=config['is_random_drop'], random_drop_i2v_ratio=config.get('random_drop_i2v_ratio', 0),
        random_drop_v2v_ratio=config['random_drop_v2v_ratio'],
        random_drop_t2v_ratio=config['random_drop_t2v_ratio'], is_keep_x0=True,
        dtype=torch.bfloat16, device='cpu')
    _, current, short_i, mid_i, long_i, short, mid, long = values
    # Original _generator_loss/_critic_loss capture continuous history BEFORE corruption.
    continuous = torch.cat([long, mid, short[:, :, 1:]], dim=2).clone()
    short, mid, long = native.history.corrupt_history_latents(
        short, mid, long, task.latent_shape[2], is_keep_x0=True,
        corrupt_mode=config['corrupt_mode_history'], noise_mode_prob=config['corrupt_mode_prob_history'],
        is_frame_independent=config['is_frame_independent_corrupt_history'],
        is_chunk_independent=config['is_chunk_independent_corrupt_history'],
        corrupt_ratio_1x=config['noise_corrupt_ratio_history_short'],
        corrupt_ratio_2x=config['noise_corrupt_ratio_history_mid'],
        corrupt_ratio_4x=config['noise_corrupt_ratio_history_long'],
        noise_corrupt_clean_prob=config['noise_corrupt_clean_prob_history'],
        downsample_min_corrupt_ratio=config['downsample_min_corrupt_ratio_history'],
        downsample_max_corrupt_ratio=config['downsample_max_corrupt_ratio_history'])
    # Original GT-history losses corrupt history here; saturation is only
    # applied inside run_generator's non-GT preparation branch. Never add it
    # merely because the shared training configuration has is_add_saturation.
    require(not all(bool(x.sum() == 0) for x in (short, mid, long))
            or task.stage_schedule_policy in ('benchmark_v4_strict_global', NATIVE_HISTORY_POLICY),
            'all-zero history uses the unsupported rank-local amplified branch; '
            'no silent resampling or disabling of dropout is allowed', path='wan.capture')
    return {'continuous_history': continuous, 'history_short': short, 'history_mid': mid,
            'history_long': long, 'indices_hidden_states': current, 'indices_history_short': short_i,
            'indices_history_mid': mid_i, 'indices_history_long': long_i}


def stage_control(native, task, seed, cohort, step, phase, *, all_history_zero=False, rank=None):
    """Native scheduler times depend only on the requested stage shapes/horizons."""
    import torch
    require(type(all_history_zero) is bool, 'explicit history predicate required', path='wan.capture')
    local_history = task.stage_schedule_policy == NATIVE_HISTORY_POLICY
    require(not local_history or type(rank) is int and 0 <= rank < task.world_size,
            'native amplified schedule requires the actual source rank', path='wan.capture')
    require(not all_history_zero or task.stage_schedule_policy in ('benchmark_v4_strict_global', NATIVE_HISTORY_POLICY),
            'amplified history requires an explicit task policy', path='wan.capture')
    with input_stream(seed, cohort, step, phase, 'rank0-stage-exits'):
        exits = [native.post.generate_and_sync_flag(SimpleNamespace(device='cpu', is_main_process=True),
                  horizon, last_step_only=False, is_sync=False) for horizon in task.stage2_inference_steps]
    initial = list(exits)
    if local_history and all_history_zero:
        # Upstream draws the normal synchronized exits even when it later
        # replaces them by one rank-local draw per amplified stage. Keep both
        # decisions; never pad work or broadcast rank zero's local exception.
        with input_stream(seed, cohort, rank, step, phase, 'rank-local-amplified-exits'):
            exits = [native.post.generate_and_sync_flag(
                SimpleNamespace(device='cpu', is_main_process=rank == 0),
                2*horizon, last_step_only=False, is_sync=False)
                for horizon in task.stage2_inference_steps]
    scheduler = native.scheduler(shift=1., stages=3, stage_range=[0, 1/3, 2/3, 1], gamma=1/3)
    sequence = []
    for stage, (horizon, hw, exit_index) in enumerate(zip(task.stage2_inference_steps, task.stage_spatial_shapes, exits)):
        # The historical strict-global benchmark fixes exits, not the native
        # zero-history scheduler budget. Retain its doubled timestep grid and
        # all sampled history/noise; do not resample or disable history dropout.
        budget = horizon * (2 if all_history_zero else 1)
        scheduler.set_timesteps(budget+1, stage, device='cpu')
        scheduler.timesteps = scheduler.timesteps[:-1]
        scheduler.sigmas = torch.cat([scheduler.sigmas[:-2], scheduler.sigmas[-1:]])
        shape = (*task.latent_shape[:3], *hw)
        sigmas, _ = native.base.apply_schedule_shift(scheduler.sigmas, torch.empty(shape, device='meta'),
                         sigmas_two=scheduler.sigmas_per_stage[stage], base_seq_len=256,
                         max_seq_len=4096, base_shift=.5, max_shift=1.15, time_shift_type='linear')
        times = scheduler.timesteps_per_stage[stage]
        times = times.min() + sigmas[:-1] * (times.max()-times.min())
        require(len(times) == budget, 'native stage scheduler contract changed', path='wan.capture')
        sequence.append({'stage': stage, 'denoising_exit': exit_index, 'call_count': exit_index+1,
                         'denoising_step_budget': budget,
                         'latent_shape': list(shape), 'timestep': times[:exit_index+1].tolist()})
    result = {'pyramid_stage_call_sequence': sequence, 'call_count': sum(x+1 for x in exits),
            'initial_synchronized_pyramid_stage_exit': initial,
            'stage_schedule_scope': 'global_rank0_broadcast_equivalent'}
    if local_history:
        result.update(stage_schedule_policy=task.stage_schedule_policy,
                      stage_schedule_scope='native_rank0_initial_with_rank_local_amplified',
                      all_history_zero=all_history_zero, rank=rank)
    if task.stage_schedule_policy == 'benchmark_v4_strict_global':
        result.update(stage_schedule_policy=task.stage_schedule_policy, all_history_zero=all_history_zero,
                      limitation='STRICT_GLOBAL_BENCHMARK_NOT_UPSTREAM_RANK_LOCAL_AMPLIFIED_DISTRIBUTION')
    return result


def capture(task, sources, training_config, output):
    """Produce an immutable rank16 trace from new task-shaped source features."""
    import torch
    from runtime.operation_trace import OperationTraceWriter, STAGE3_SOURCE_KEYS, write_operation_trace_bundle_manifest
    from runtime.official_baseline import stable_block_noise
    from .wan_trace import load_original_trace
    from .wan_planning import source_fingerprint
    require(isinstance(task, OriginalWanTask) and isinstance(sources, WanInputSources), 'typed inputs required', path='wan.capture')
    inventory = preflight_sources(task, sources)
    config = read_training_config(training_config)
    config_sha = file_sha256(Path(training_config))
    native = native_primitives()
    root = Path(output).resolve(); root.mkdir(parents=True, exist_ok=False)
    source_code = source_fingerprint(); entries = []
    try:
        for row in inventory['files']:
            rank = row['rank']; latent, text, mask = load_feature(task, row)
            dataset = native.dataset.__new__(native.dataset)
            dataset.base_seed, dataset._epoch = sources.seed, sources.cohort
            primary, secondary = dataset.prepare_stage1_latent(latent, rank, return_secondary=True)
            writer = OperationTraceWriter(classification='NEW_ORIGINAL_WAN_OPERATION_QUALIFIED_INPUTS',
                source={'feature': row, 'cohort': sources.cohort, 'seed': sources.seed,
                        'source_fingerprint': source_code, 'training_config_sha256': config_sha,
                        'rng_scope': 'new_native_CPU_streams_not_prior_training_continuation'})
            for group, sample in (('primary', primary), ('secondary', secondary)):
                for name, tensor in zip(('x0', 'history', 'target'), sample):
                    writer.add_tensor(f'sample/0/{group}/{name}', tensor.to(torch.bfloat16), role='source_'+name)
            writer.add_tensor('sample/0/text_embedding', text.to(torch.bfloat16), role='text_embedding')
            writer.add_tensor('sample/0/attention_mask', mask, role='attention_mask')
            for step in task.source_steps:
                writer.add_operation(f'step/{step}/branch', component='training', operation='stage3_branch', control={
                    'global_step': step, 'dfake_gen_update_ratio': task.fake_updates,
                    'train_generator': step == task.first_step, 'train_critic': True,
                    'gan': False, 'reward': False, 'vae_decode': False})
                for phase in (('generator', 'critic') if step == task.first_step else ('critic',)):
                    prefix = f'step/{step}/{phase}'
                    with input_stream(sources.seed, sources.cohort, rank, step, phase, 'history'):
                        tensors = prepare_history(native, task, primary if phase == 'generator' else secondary, config)
                    for stream in ('entry_noise', 'rollout_noise', 'dmd_noise' if phase == 'generator' else 'critic_noise'):
                        with input_stream(sources.seed, sources.cohort, rank, step, phase, stream):
                            tensors[stream] = torch.randn(task.latent_shape, dtype=torch.bfloat16)
                    for stage in (1, 2):
                        h, w = task.stage_spatial_shapes[stage]
                        with input_stream(sources.seed, sources.cohort, rank, step, phase, 'block', stage):
                            # This is the existing stable PSD gamma=1/3 primitive, not iid replacement noise.
                            noise = stable_block_noise(16*task.latent_shape[2]*(h//2)*(w//2))
                        tensors[f'block_noise/{stage}'] = noise.view(1,16,task.latent_shape[2],h//2,w//2,2,2).permute(
                            0,1,2,3,5,4,6).reshape(1,16,task.latent_shape[2],h,w).to(torch.bfloat16)
                    with input_stream(sources.seed, sources.cohort, rank, step, phase, 'score-time'):
                        _, mu = native.base.apply_schedule_shift(torch.tensor([.5], dtype=torch.float64),
                            torch.empty(task.latent_shape, device='meta'), return_mu=True)
                        key = 'generator' if phase == 'generator' else 'critic'
                        tensors['dmd_timestep' if phase == 'generator' else 'critic_timestep'] = native.post.sample_dynamic_timestep(
                            B=1, timestep_shift=mu, dynamic_alpha=config[key+'_dynamic_alpha'],
                            dynamic_beta=config[key+'_dynamic_beta'], dynamic_sample_type=config[key+'_dynamic_sample_type'],
                            global_step=step, dynamic_step=config[key+'_dynamic_step'], device='cpu')
                    for key, tensor in tensors.items():
                        writer.add_tensor(prefix+'/'+key, tensor, role=key)
                    keys = [prefix+'/'+k for k in tensors] + ['sample/0/text_embedding', 'sample/0/attention_mask']
                    writer.add_operation(prefix+'/rollout', component=phase, operation='native_pyramid_rollout',
                        tensor_keys=keys, control=stage_control(native, task, sources.seed, sources.cohort, step, phase,
                            rank=rank,
                            all_history_zero=all(bool(tensors[k].sum() == 0) for k in
                                ('history_short', 'history_mid', 'history_long'))))
                    writer.add_operation(prefix+('/dmd' if phase == 'generator' else '/train'),
                        component=phase, operation='native_score_inputs', tensor_keys=keys, control={'global_step':step})
            trace = writer.write(root/f'rank_{rank:02d}', contract={TRACE_KEY:task.to_dict(),
                'resolution':[task.video_height,task.video_width], 'full_latent_shape':list(task.latent_shape),
                'stage_spatial_shapes':[list(x) for x in task.stage_spatial_shapes], 'dtype':'bfloat16',
                'text_shape':[1,*text.shape], 'dfake_gen_update_ratio':task.fake_updates})
            entries.append({'rank':rank, 'path':f'rank_{rank:02d}', 'trace_hash':trace.trace_hash})
        require(file_sha256(Path(training_config)) == config_sha and sources.inventory()['digest'] == inventory['digest']
                and source_fingerprint() == source_code, 'capture inputs/source changed', path='wan.capture')
        bundle = write_operation_trace_bundle_manifest(root, entries)
        checked = load_original_trace(task, root, expected_hash=bundle['bundle_hash'])
        result = {'status':'PASS_CAPTURED_ORIGINAL_INPUTS', 'task':task.to_dict(), 'input_inventory':inventory,
            'training_config_sha256':config_sha, 'source_fingerprint':source_code,
            'trace_bundle':str(root), 'expected_trace_hash':bundle['bundle_hash'],
            'rank_validation':checked.rank_control_validation, 'gpu_execution':'NOT_RUN',
            'claimable_speedup':False, 'rng_continuation':'NOT_CLAIMED',
            'scope':'native_preencoded_features_and_native_input_sampling_only'}
        write_new_json(root/'capture.json', result)
        return result
    except BaseException as exc:
        write_new_json(root/'capture-failure.json', {'status':'FAIL_CAPTURE', 'error':str(exc), 'task':task.to_dict()})
        raise
