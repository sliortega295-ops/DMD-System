"""Actual official Self-Forcing Wan, explicitly local assets and common Program.

Importing this factory is lightweight: models/Torch are loaded only on explicit
construction. One explicit CPU/CUDA device, FP32 MATH attention and cached
21-frame T2V is the source scope; actual CUDA/pretrained acceptance is NOT_RUN.
FSDP, AMP, I2V and >21-frame decode/reencode training are NOT admitted.
"""
from __future__ import annotations

from copy import deepcopy
from contextlib import contextmanager, nullcontext
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

from .adapters import ModelComponents, InferenceProjection
from .prediction import PredictionSpec
from .self_forcing_source import REVISION, REPOSITORY, SOURCE_FILES, original, require
from .single_device import execution_device, device_scope as _device_scope, construction_rng as _construction_rng

ASSET_SCHEMA = 'distillschedule.official_self_forcing_assets.v1'


def _sha(path):
    from distillschedule.workloads.best_known.registry import file_sha256
    return file_sha256(Path(path))


def _json(path):
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, 'duplicate JSON key: ' + key)
            value[key] = item
        return value
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs)


def runtime_identity():
    import importlib
    import importlib.metadata
    import platform
    import torch
    result = {'python': platform.python_version(), 'torch': str(torch.__version__),
              **{name: importlib.metadata.version(name) for name in ('diffusers', 'einops', 'numpy', 'transformers')}}
    result['entry_sources'] = {name: _sha(importlib.import_module(name).__file__) for name in
                               ('torch', 'diffusers', 'einops', 'numpy', 'transformers')}
    result['cpu_policy'] = dict(threads=torch.get_num_threads(), mkldnn=torch.backends.mkldnn.enabled,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        float32_matmul_precision=torch.get_float32_matmul_precision())
    # These are policy reads, not CUDA initialization or global overrides.
    # FP32 parameter storage/MATH SDPA is not a claim that TF32 is disabled.
    result['cuda_policy'] = dict(matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32, cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_benchmark=torch.backends.cudnn.benchmark)
    return result


def _bridge_identity(*, inference=False):
    root = Path(__file__).resolve().parent
    names = ['self_forcing.py', 'self_forcing_source.py', 'single_device.py', 'upstream.py']
    if not inference:
        names += ['self_forcing_owner.py']
    files = {name: _sha(root / name) for name in names}
    files['data/self_forcing_inputs.py'] = _sha(root.parent / 'data/self_forcing_inputs.py')
    return files


def _execution_device(value):
    return execution_device(value, require=require)


def _admit_device(parameters):
    import os
    import torch
    device = _execution_device(parameters.get('device', 'cpu'))
    require(not torch.distributed.is_initialized(), 'distributed ownership is not admitted')
    require(all(int(os.environ.get(key, default)) == int(default) for key, default in
                (('WORLD_SIZE', '1'), ('LOCAL_WORLD_SIZE', '1'), ('RANK', '0'), ('LOCAL_RANK', '0'))),
            'one actual process/rank required')
    require(torch.get_default_dtype() == torch.float32 and not torch.is_autocast_enabled('cpu')
            and not torch.is_autocast_enabled('cuda'), 'explicit FP32 with no autocast required')
    return device


def _arguments(parameters):
    require(isinstance(parameters, dict) and set(parameters) <= {'source_root', 'assets', 'settings', 'seed', 'device'}
            and isinstance(parameters.get('source_root'), str) and isinstance(parameters.get('assets'), str),
            'source_root and explicit local assets manifest required')
    _admit_device(parameters)
    require(type(parameters.get('seed', 0)) is int and 0 <= parameters.get('seed', 0) < 2**63,
            'invalid construction seed')
    settings = dict(mixed_precision=False, denoising_loss_type='flow', num_train_timestep=1000,
        denoising_step_list=[1000, 750, 500, 250], warp_denoising_step=True,
        num_frame_per_block=3, same_step_across_blocks=True, last_step_only=False,
        num_training_frames=21, context_noise=0, independent_first_frame=False,
        i2v=False, backward_simulation=True, gradient_checkpointing=False,
        real_guidance_scale=3., fake_guidance_scale=0., timestep_shift=1., scheduler_shift=8.,
        ts_schedule=True, ts_schedule_max=False, min_score_timestep=0,
        batch_size=1, image_or_video_shape=[1, 21, 16, 60, 104], dfake_gen_update_ratio=5,
        lr=1e-6, lr_critic=1e-6, beta1=0., beta2=.999, beta1_critic=0., beta2_critic=.999,
        weight_decay=.01, max_grad_norm_generator=10., max_grad_norm_critic=10.,
        ema_weight=.99, ema_start_step=200, train_iters=1000000, negative_prompt='')
    # The shipped recipe is the pinned YAML merge, not constructor getattrs.
    # The wrapper schedule shift AND score-time shift are 5, whereas the
    # WanDiffusionWrapper constructor's unrelated fallback is 8.
    official = official_recipe_settings(parameters['source_root'])
    settings.update({key: deepcopy(official[key]) for key in settings if key in official})
    settings.update(real_guidance_scale=official.get('real_guidance_scale', official['guidance_scale']),
        fake_guidance_scale=official.get('fake_guidance_scale', 0.),
        scheduler_shift=official['model_kwargs']['timestep_shift'],
        mixed_precision=False, gradient_checkpointing=False)
    changes = parameters.get('settings', {})
    require(isinstance(changes, dict) and set(changes) <= set(settings), 'unknown native Self-Forcing settings')
    settings.update(deepcopy(changes))
    for key in ('mixed_precision', 'warp_denoising_step', 'same_step_across_blocks', 'last_step_only',
                'independent_first_frame', 'i2v', 'backward_simulation', 'gradient_checkpointing',
                'ts_schedule', 'ts_schedule_max'):
        require(type(settings[key]) is bool, 'boolean setting required: ' + key)
    require(settings['mixed_precision'] is False and settings['denoising_loss_type'] == 'flow'
            and settings['num_train_timestep'] == 1000 and settings['num_training_frames'] == 21
            and settings['independent_first_frame'] is False and settings['i2v'] is False
            and settings['backward_simulation'] is True,
            'resident scope is native flow/T2V/21-frame cached rollout; no AMP/I2V/flex or longer-frame substitution')
    for key in ('batch_size', 'num_frame_per_block', 'dfake_gen_update_ratio', 'train_iters'):
        require(type(settings[key]) is int and settings[key] > 0, 'positive native integer required: ' + key)
    require(settings['batch_size'] == 1 and type(settings['num_train_timestep']) is int
            and type(settings['num_training_frames']) is int,
            'qualified CPU scope requires batch1 and integer native temporal clocks')
    require(21 % settings['num_frame_per_block'] == 0, 'native min21 must divide into whole frame blocks')
    shape = settings['image_or_video_shape']
    require(isinstance(shape, list) and len(shape) == 5 and all(type(n) is int and n > 0 for n in shape)
            and shape[:3] == [settings['batch_size'], 21, 16], 'native BFCHW/16-channel/21-frame shape required')
    steps = settings['denoising_step_list']
    require(isinstance(steps, list) and steps and all(type(n) is int and 0 < n <= 1000 for n in steps)
            and all(a > b for a, b in zip(steps, steps[1:])), 'strict descending nonzero native denoising steps required')
    for key in ('ema_start_step', 'context_noise', 'min_score_timestep'):
        require(type(settings[key]) is int and 0 <= settings[key] < (1000 if key != 'ema_start_step' else 2**63),
                'invalid native integer: ' + key)
    for key in ('lr', 'lr_critic', 'max_grad_norm_generator', 'max_grad_norm_critic', 'scheduler_shift', 'timestep_shift'):
        require(type(settings[key]) in (int, float) and 0 < settings[key] < float('inf'), 'finite positive ' + key + ' required')
    for key in ('beta1', 'beta2', 'beta1_critic', 'beta2_critic', 'ema_weight'):
        require(type(settings[key]) in (int, float) and 0 <= settings[key] < 1., 'native decay must be in [0,1): ' + key)
    for key in ('weight_decay', 'real_guidance_scale', 'fake_guidance_scale'):
        require(type(settings[key]) in (int, float) and 0 <= settings[key] < float('inf'), 'invalid ' + key)
    require(isinstance(settings['negative_prompt'], str), 'negative prompt metadata must be text')
    return SimpleNamespace(**settings)


def official_recipe_settings(source_root):
    """Original default_config + self_forcing_dmd YAML, bytes pinned first."""
    from distillschedule.core.config import load_document
    root = Path(source_root).expanduser().resolve(strict=True)
    result = {}
    for name in ('configs/default_config.yaml', 'configs/self_forcing_dmd.yaml'):
        path = root / name
        require(path.resolve().is_relative_to(root) and _sha(path) == SOURCE_FILES[name],
                'original native recipe source changed: ' + name)
        value, _ = load_document(path)
        require(isinstance(value, dict), 'native recipe must be a YAML mapping')
        result.update(value)
    return result


def _asset_file(root, spec):
    require(isinstance(spec, dict) and set(spec) == {'path', 'sha256'} and isinstance(spec['path'], str),
            'local asset requires path and SHA256')
    path = (root / Path(spec['path']).expanduser()).resolve(strict=True)
    require(path.is_file() and path.is_relative_to(root) and _sha(path) == spec['sha256'],
            'local asset missing, escapes bundle, or changed SHA256')
    return path


def _native_config(cls, spec):
    signature = inspect.signature(cls.__init__)
    defaults = {key: deepcopy(parameter.default) for key, parameter in signature.parameters.items()
                if key != 'self' and parameter.default is not inspect.Parameter.empty}
    require(isinstance(spec, dict) and set(spec) <= set(defaults), 'unknown native architecture setting')
    defaults.update(deepcopy(spec))
    # ConfigMixin persists tuples as JSON lists; normalize at the public seam.
    return json.loads(json.dumps(defaults))


def _wrapper(api, config, *, causal, shape, scheduler_shift):
    import torch
    cls = api.causal.CausalWanModel if causal else api.wan.WanModel
    config = _native_config(cls, config)
    require(config['model_type'] == 't2v' and config['in_dim'] == config['out_dim'] == 16
            and config['patch_size'][0] == 1 and shape[3] % config['patch_size'][1] == 0
            and shape[4] % config['patch_size'][2] == 0, 'actual T2V Wan architecture/patch geometry required')
    require(config['dim'] % config['num_heads'] == 0 and (config['dim'] // config['num_heads']) % 2 == 0,
            'native RoPE head width is invalid')
    if causal:
        require(type(config['local_attn_size']) is int and (config['local_attn_size'] == -1
                or config['local_attn_size'] > 0) and type(config['sink_size']) is int and config['sink_size'] >= 0,
                'invalid native local-attention/sink frame geometry')
    wrapper = api.wrapper.WanDiffusionWrapper.__new__(api.wrapper.WanDiffusionWrapper)
    torch.nn.Module.__init__(wrapper)
    wrapper.model = cls(**config).eval()
    wrapper.uniform_timestep = not causal
    wrapper.scheduler = api.scheduler.FlowMatchScheduler(shift=scheduler_shift, sigma_min=0., extra_one_step=True)
    wrapper.scheduler.set_timesteps(1000, training=True)
    wrapper.seq_len = shape[1] * (shape[3] // config['patch_size'][1]) * (shape[4] // config['patch_size'][2])
    wrapper.post_init()
    wrapper.clip_grad_norm_ = __import__('types').MethodType(_clip_grad_norm, wrapper)
    return wrapper, config


def _clip_grad_norm(wrapper, maximum):
    import torch
    return torch.nn.utils.clip_grad_norm_(wrapper.parameters(), maximum)


def _vae_wrapper(api, config):
    import ast
    import torch
    wrapper = api.wrapper.WanVAEWrapper.__new__(api.wrapper.WanVAEWrapper)
    torch.nn.Module.__init__(wrapper)
    # Take the native Wan normalization lists from the verified constructor;
    # only its hardcoded pretrained-file allocation is bypassed.
    tree = ast.parse((api.source.root / 'utils/wan_wrapper.py').read_bytes())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'WanVAEWrapper')
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    for name in ('mean', 'std'):
        statement = next(n for n in init.body if isinstance(n, ast.Assign)
                         and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
        setattr(wrapper, name, torch.tensor(ast.literal_eval(statement.value), dtype=torch.float32))
    config = _native_config(api.vae.WanVAE_, config)
    require(config['z_dim'] == 16, 'native VAE must have the actual sixteen-channel Wan latent convention')
    wrapper.model = api.vae.WanVAE_(**config).eval().requires_grad_(False)
    return wrapper, config


def preencoded_conditioner(device='cpu'):
    import torch
    device = torch.device(device)
    class NativeEmbeddingBoundary(torch.nn.Module):
        def forward(self, text_prompts):
            require(isinstance(text_prompts, list) and text_prompts
                    and all(type(tensor) is torch.Tensor and tensor.ndim == 2
                            and tensor.dtype == torch.float32 and tensor.device == device
                            for tensor in text_prompts),
                    'this explicit preencoded boundary accepts native LC embeddings, not prompt strings')
            return {'prompt_embeds': torch.stack(text_prompts)}
    return NativeEmbeddingBoundary().eval()


def _place_native_tensors(model, device):
    """Place the known native plain tensors without changing their dtypes.

The scheduler object shared by DMD and G stays aliased. Module.to alone misses
the complex128 RoPE, schedule and VAE normalization tensors. This helper also
accepts the inference-only G/VAE namespace, never inventing missing roles.
    """
    for role in ('generator', 'real_score', 'fake_score', 'vae'):
        wrapper = getattr(model, role, None)
        if wrapper is None:
            continue
        wrapper.to(device)
        if role == 'vae':
            for name in ('mean', 'std'):
                setattr(wrapper, name, getattr(wrapper, name).to(device))
        else:
            wrapper.model.freqs = wrapper.model.freqs.to(device)
            for name in ('timesteps', 'sigmas', 'linear_timesteps_weights'):
                setattr(wrapper.scheduler, name, getattr(wrapper.scheduler, name).to(device))
    if hasattr(model, 'denoising_step_list'):
        model.denoising_step_list = model.denoising_step_list.to(device)


def build_model(parameters):
    args = _arguments(parameters)
    device = _admit_device(parameters)
    with _construction_rng(device, parameters.get('seed', 0)):
        return _build_placed(parameters, args, device)


@contextmanager
def _native_checkpoint_enable(api, children):
    """Private constructor-only bridge for the pinned old Diffusers setter ABI.

    Native BaseModel/DMD still decides which wrappers to enable. Invoke the
    source's original (module, value) setter, as the old enable API did. Do not
    replace native forward/checkpoint bodies or any shared Diffusers method.
    """
    from types import MethodType
    roots = [(children['generator'].model, api.causal.CausalWanModel),
             (children['fake_score'].model, api.wan.WanModel)]
    api.source.validate()
    for root, cls in roots:
        require(type(root) is cls and 'enable_gradient_checkpointing' not in vars(root)
                and root._set_gradient_checkpointing.__func__ is cls._set_gradient_checkpointing
                and list(inspect.signature(cls._set_gradient_checkpointing).parameters) == ['self', 'module', 'value'],
                'native legacy checkpoint setter changed')

    def enable(root, gradient_checkpointing_func=None):
        require(gradient_checkpointing_func is None, 'native checkpoint callback override is not admitted')
        root.apply(lambda module: root._set_gradient_checkpointing(module, value=True))

    try:
        for root, _ in roots:
            root.enable_gradient_checkpointing = MethodType(enable, root)
        yield
    finally:
        for root, _ in roots:
            if 'enable_gradient_checkpointing' in vars(root):
                del root.enable_gradient_checkpointing


def _build_placed(parameters, args, device):
    import torch
    from .self_forcing_owner import SelfForcingOwner
    api = original(parameters['source_root'])
    manifest = Path(parameters['assets']).expanduser().resolve(strict=True)
    root = manifest.parent
    bundle = _json(manifest)
    require(isinstance(bundle, dict) and set(bundle) == {'schema_version', 'evidence_kind', 'generator',
            'real_score', 'fake_score', 'vae', 'negative_prompt_embeds'} and bundle['schema_version'] == ASSET_SCHEMA
            and bundle['evidence_kind'] in ('RANDOM_TINY_ACTUAL_NATIVE_CLASSES_NOT_PRETRAINED',
                                           'EXPLICIT_LOCAL_NATIVE_WEIGHTS_QUALITY_NOT_VERIFIED'),
            'unsupported explicit native asset manifest')
    children, configs, assets = {}, {}, {}
    for role in ('generator', 'real_score', 'fake_score', 'vae'):
        spec = bundle[role]
        require(isinstance(spec, dict) and set(spec) == {'config', 'weights'}, 'native role needs config and pinned weights')
        if role == 'vae':
            child, configs[role] = _vae_wrapper(api, spec['config'])
        else:
            child, configs[role] = _wrapper(api, spec['config'], causal=role == 'generator',
                shape=args.image_or_video_shape, scheduler_shift=args.scheduler_shift)
        path = _asset_file(root, spec['weights'])
        weights = torch.load(path, map_location='cpu', weights_only=True)
        from .checkpoint import _validate_layout
        _validate_layout(child.model.state_dict(), weights, 'native local ' + role)
        require(all(type(tensor) is torch.Tensor and tensor.device.type == 'cpu'
                    and tensor.layout == torch.strided and bool(tensor.isfinite().all())
                    for tensor in weights.values()), 'native weights must be finite dense CPU tensors; no implicit precision cast')
        child.model.load_state_dict(weights, strict=True)
        child.requires_grad_(role in ('generator', 'fake_score'))
        children[role] = child
        assets[role] = {'path': str(path), 'sha256': _sha(path)}
    generator = children['generator'].model
    require(all(model.text_dim == generator.text_dim and model.text_len == generator.text_len
                for model in (children['real_score'].model, children['fake_score'].model)),
            'native generator/score conditioning dimensions differ')
    require(generator.local_attn_size == -1 or generator.local_attn_size
            >= args.num_frame_per_block + configs['generator']['sink_size'],
            'local attention window cannot hold the native block plus sink frames')
    negative_path = _asset_file(root, bundle['negative_prompt_embeds'])
    negative = torch.load(negative_path, map_location='cpu', weights_only=True)
    from distillschedule.data.self_forcing_inputs import validate_row
    validate_row({'prompt_embeds': negative}, shape=(1, generator.text_len, generator.text_dim))
    assets['negative_prompt_embeds'] = {'path': str(negative_path), 'sha256': _sha(negative_path)}

    def initialize_models(model, native_args, device):
        for name, module in children.items():
            setattr(model, name, module)
        model.text_encoder = preencoded_conditioner(device)
        model.scheduler = model.generator.get_scheduler()
        model.scheduler.timesteps = model.scheduler.timesteps.to(device)

    def initialize_pipeline(model):
        api.base.SelfForcingModel._initialize_inference_pipeline(model)
        model.inference_pipeline.configure_geometry(shape=model.args.image_or_video_shape)

    bridge = type('NativeDMDWithExplicitLocalAssets', (api.dmd.DMD,),
                 {'_initialize_models': initialize_models, '_initialize_inference_pipeline': initialize_pipeline})
    with _native_checkpoint_enable(api, children) if args.gradient_checkpointing else nullcontext():
        model = bridge(args, device).eval()
    _place_native_tensors(model, device)  # Before native AdamW owns Parameters.
    negative = negative.to(device)
    metadata = dict(schema_version='distillschedule.official_self_forcing_wan.v1',
        upstream=dict(repository=REPOSITORY, revision=REVISION, license='Apache-2.0', **api.source.identity),
        bridge=_bridge_identity(), boundary_port=api.boundary_identity, runtime=runtime_identity(),
        assets=dict(manifest=str(manifest), sha256=_sha(manifest), files=assets, evidence_kind=bundle['evidence_kind']),
        architecture=configs, settings=deepcopy(vars(args)),
        precision='FP32_CPU1_SDPA_MATH_REFERENCE' if device.type == 'cpu' else 'FP32_CUDA1_SDPA_MATH_REFERENCE',
        precision_policy='FP32_PARAMETERS_NATIVE_FP64_COMPLEX128; TF32_POLICY_RECORDED_NOT_OVERRIDDEN',
        placement=dict(device=str(device), layout='SINGLE_DEVICE_RESIDENT', distributed=False, amp=False),
        conditioning='EXACT_PREENCODED_NATIVE_PROMPT_EMBEDS; RAW_PROMPTS_USE_EXPLICIT_LOCAL_NATIVE_T5_PREPARATION',
        cpu_execution='ACTUAL_RANDOM_TINY_NATIVE_CLASSES_ONLY', gpu_execution='NOT_RUN', quality_validation='NOT_RUN',
        placement_limit='COMPUTE_RESIDENT; NATIVE_EMA_SHADOW_CPU_FP32; RNG_AND_ADAM_STEP_HOST',
        cuda_trial_snapshot='UNAVAILABLE_AUTO_PLAN_AND_OWNER_EXECUTION_REJECTED_BEFORE_TRIAL',
        native_recipe=dict(default='configs/default_config.yaml', override='configs/self_forcing_dmd.yaml',
            reference_exceptions=['mixed_precision=false'] + ([] if args.gradient_checkpointing else
                ['gradient_checkpointing=false']) + ['single_process_no_FSDP',
                'explicit_local_weights_and_preencoded_conditioning']),
        lifecycle=f'COMMON_PROGRAM_ATOMIC_NATIVE_G1_C{args.dfake_gen_update_ratio}; CURRENT_G_EXPORT_NOT_EMA')
    if args.gradient_checkpointing:
        metadata['checkpoint_contract'] = dict(generator=True, fake_reference=True,
            generator_policy='SOURCE_ALGORITHM_IDENTITY_NOT_A_PHYSICAL_OPTION',
            generator_on_off_equivalence='NOT_EQUIVALENT_CACHE_AND_GRADIENT_PATH',
            enable_compatibility='INSTANCE_LOCAL_ORIGINAL_LEGACY_SETTER_CONSTRUCTION_ONLY')
    owner = SelfForcingOwner(model, args, api.source, negative, identity=metadata)
    return ModelComponents(state={'native_cycle': owner}, metadata=metadata, prediction=PredictionSpec('flow'),
        capabilities=('video', 'causal', 'kv_cache', 'flow', 'official-self-forcing'), generate=generate,
        decode=_decode, inference_projection=project_current_generator, admit_training=admit_training,
        collect_metrics=collect_metrics).validate()


def admit_training(*, state, request, prepared_identity, prepared_manifest):
    from .self_forcing_owner import stream_counts
    owner = state['native_cycle']
    require(_admit_device(request.trainer) == owner.device, 'trainer/model device placement must match')
    owner.admit_execution(request.execution.get('mode', 'reference'), request.execution.get('implementation', {}))
    require(request.execution.get('adapter', 'program') == 'program'
            and request.algorithm['factory'] == 'distillschedule.training.self_forcing:build_self_forcing'
            and not request.algorithm.get('parameters'), 'common official Self-Forcing Program algorithm required')
    counts = stream_counts(owner.args)
    streams = request.data.get('streams', {})
    require(set(streams) == set(counts) and all(streams[name].get('samples_per_cycle', 1) == count
                for name, count in counts.items()), 'six independent native prompt draws per G1/C5 cycle required')
    require('distributed' not in request.trainer and request.trainer['cycles'] * owner.ratio <= owner.args.train_iters,
            'unsupported distributed policy or native iteration budget')
    manifests = prepared_manifest.get('streams', {})
    require(set(manifests) == set(counts), 'physical native input stream manifests required')
    ids = [{record['id'] for record in manifests[role]['records']} for role in ('generator', 'critic')]
    require(not ids[0] & ids[1], 'G and C require independent prompt records, not a shared cached G batch')


def _native_cycle(ctx):
    return ctx.state['native_cycle'].run(ctx.inputs, cycle=ctx.attributes['cycle'])


def build_self_forcing(components, feeds, parameters, cycle, seed):
    from distillschedule.open_workload.torch_backend import Workload
    from distillschedule.workloads.program import Program, Call
    from .self_forcing_owner import SelfForcingOwner
    require(not parameters and set(components.state) == {'native_cycle'}
            and type(components.state['native_cycle']) is SelfForcingOwner, 'actual native Self-Forcing owner required')
    owner = components.state['native_cycle']; owner._validate_live(); owner.validate_feeds(feeds)
    require(type(cycle) is int and cycle >= 0, 'invalid Program cycle')
    call = Call(id='self_forcing.native_cycle', binding='official_self_forcing_cycle', inputs=tuple(sorted(feeds)),
        reads=('native_cycle',), writes=('native_cycle',), grad='external_grad', replay_safe=False,
        attributes={'cycle': cycle})
    return Workload(program=Program((call,), initial_versions={'native_cycle': cycle}, feeds=tuple(sorted(feeds)),
        outputs=(call.id,)), bindings={'official_self_forcing_cycle': _native_cycle}, state=components.state,
        feeds=feeds, seed=seed, identity_extra=dict(scope='OFFICIAL_SELF_FORCING_NATIVE_FP32_SINGLE_DEVICE',
            placement=deepcopy(owner.identity['placement']), upstream=owner.source.identity, ratio=owner.ratio))


def collect_metrics(outputs):
    return {f'native_{index}/{name}': value for index, row in
            enumerate(outputs['self_forcing.native_cycle']['iterations']) for name, value in row['losses'].items()}


def _inference_parameters(owner):
    return dict(source_root=str(owner.source.root), upstream=deepcopy(owner.source.identity),
        bridge=_bridge_identity(inference=True), runtime=runtime_identity(),
        architecture={role: deepcopy(owner.identity['architecture'][role]) for role in ('generator', 'vae')},
        shape=deepcopy(owner.args.image_or_video_shape), settings={key: deepcopy(getattr(owner.args, key)) for key in
            ('denoising_step_list', 'warp_denoising_step', 'num_frame_per_block', 'independent_first_frame',
             'context_noise', 'scheduler_shift')}, seed=0)


def _inference_components(state, parameters):
    def generation(state, feeds, settings):
        return _generate(state, feeds, settings, parameters)

    def refuse(**kwargs):
        raise ValueError('official Self-Forcing inference-only export cannot train')

    return ModelComponents(state=state, metadata=dict(schema_version='distillschedule.official_self_forcing_inference.v1',
        parameters=deepcopy({key: value for key, value in parameters.items() if key != 'device'}),
        selection='CURRENT_G_NOT_EMA', training_assets='NOT_REQUIRED', gpu_execution='NOT_RUN', quality_validation='NOT_RUN'),
        prediction=PredictionSpec('flow'), capabilities=('video', 'flow'), generate=generation, decode=_decode,
        admit_training=refuse).validate()


def project_current_generator(components, effective_training_parameters):
    owner = components.state['native_cycle']; owner._validate_live()
    parameters = _inference_parameters(owner)
    return InferenceProjection(_inference_components(dict(generator=owner.model.generator, vae=owner.model.vae), parameters),
        'distillschedule.training.self_forcing:build_inference_model', parameters).validate()


def build_inference_model(parameters):
    """Only architecture G + actual VAE; no teacher, fake, Adam, EMA or data."""
    device = _admit_device(parameters)
    require(set(parameters) <= {'source_root', 'upstream', 'bridge', 'runtime', 'architecture', 'shape',
                               'settings', 'seed', 'device'}, 'unknown inference construction setting')
    require(parameters['bridge'] == _bridge_identity(inference=True)
            and parameters['runtime'] == runtime_identity(), 'inference bridge/dependency identity changed')
    api = original(parameters['source_root'])
    require(parameters['upstream'] == api.source.identity, 'inference source lock changed')
    require(set(parameters['architecture']) == {'generator', 'vae'}, 'inference architecture is current G/VAE only')
    shape = parameters['shape']
    require(isinstance(shape, list) and len(shape) == 5 and shape[:3] == [1, 21, 16]
            and all(type(n) is int and n > 0 for n in shape), 'inference scope requires one native BFCHW/21-frame sample')
    require(set(parameters['settings']) == {'denoising_step_list', 'warp_denoising_step', 'num_frame_per_block',
            'independent_first_frame', 'context_noise', 'scheduler_shift'}
            and parameters['settings']['independent_first_frame'] is False,
            'inference must retain the exported native causal T2V schedule')
    with _construction_rng(device, parameters.get('seed', 0)):
        generator, _ = _wrapper(api, parameters['architecture']['generator'], causal=True,
            shape=parameters['shape'], scheduler_shift=parameters['settings']['scheduler_shift'])
        vae, _ = _vae_wrapper(api, parameters['architecture']['vae'])
        _place_native_tensors(SimpleNamespace(generator=generator, vae=vae), device)
    generator.eval(); vae.eval()
    return _inference_components(dict(generator=generator, vae=vae), parameters)


def generate(state, feeds, parameters):
    owner = state['native_cycle']; owner._validate_live()
    return _generate(dict(generator=owner.model.generator, vae=owner.model.vae), feeds, parameters,
                     _inference_parameters(owner))


def _generate(state, feeds, parameters, settings):
    import torch
    import sys
    from contextlib import redirect_stdout
    from distillschedule.data.self_forcing_inputs import validate_row
    require(not parameters, 'generation uses every official exported denoising step; no random-exit override')
    require(settings['runtime'] == runtime_identity(), 'inference numerical runtime/precision policy changed')
    row = feeds['generator'][0] if 'generator' in feeds else feeds
    generator = state['generator']
    device = next(generator.parameters()).device
    _admit_device({'device': str(device)})
    require(all(t.device == device and (not t.is_floating_point() or t.dtype == torch.float32)
                for module in state.values() for t in (*module.parameters(), *module.buffers())),
            'inference module device/FP32 placement changed')
    validate_row(row, shape=(settings['shape'][0], generator.model.text_len, generator.model.text_dim), device=device)
    api = original(settings['source_root'])
    generator.eval(); state['vae'].eval()
    args = SimpleNamespace(**settings['settings'])
    # Native diagnostics go to stderr so distillctl's JSON stdout stays valid.
    with _device_scope(device), torch.no_grad(), redirect_stdout(sys.stderr):
        noise = torch.randn(settings['shape'], dtype=torch.float32, device=device)
        pipeline = api.InferencePipeline(args, device, generator=generator,
            text_encoder=preencoded_conditioner(device), vae=state['vae'])
        pipeline.denoising_step_list = pipeline.denoising_step_list.to(device)
        pipeline.configure_geometry(shape=settings['shape'])
        video, latents = pipeline.inference(noise=noise, text_prompts=list(row['prompt_embeds'].unbind(0)),
                                           return_latents=True)
    return {'latents': latents, 'video': video}


def _decode(generated, parameters):
    import torch
    video = generated['video']
    require(video.ndim == 5 and video.shape[0] == 1 and video.shape[2] == 3,
            'public decoded video requires one native BFCHW RGB sample')
    return {'video': (video[0].clamp(0, 1) * 255.).round().to(torch.uint8).permute(0, 2, 3, 1).contiguous()}
