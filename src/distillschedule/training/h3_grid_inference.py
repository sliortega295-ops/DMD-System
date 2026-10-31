"""Independent W5 Student, with its exact four rollout AdaLN table entries.

Grid training removes the original AdaLN projections. Exporting its remaining
weights as an ordinary H3 model would be incomplete. This consumer keeps the
same lookup representation, not reconstructed projections or W6's sampler.
Imports are lazy; external source and model payloads are never downloaded.
"""
from copy import deepcopy
from contextlib import nullcontext
import inspect
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

from distillschedule.workloads.best_known.registry import file_sha256
from .adapters import ModelComponents
from .host_memory import admit_host_memory

SCHEMA = 'distillschedule.h3_grid_current_student_inference.v1'
FACTORY = 'distillschedule.training.h3_grid_inference:build_inference_model'
FIXTURE = 'RANDOM_ORIGINAL_CONVERTED_H3_GRID_FIXTURE_NOT_TRAINED'
NATIVE = 'LOCAL_CURRENT_CONVERTED_H3_GRID_STUDENT_UNVERIFIED_QUALITY'
PRECISION = 'EXACT_NATIVE_STORAGE_UNMERGED_STUDENT_CPU_SDPA_FP32_LATENTS'
SCHEDULE = dict(num_inference_steps=4, video_shift=6.0, audio_shift=3.0, latent_dtype='fp32')
LORA = dict(r=128, lora_alpha=128, init_lora_weights='gaussian',
            target_modules=['to_q', 'to_k', 'to_v', 'to_out.0', 'ff.net.0.proj', 'ff.net.2'])
MODEL_SHA = '1926b1bc15a5bebda05e3dc8cde1b3955d56641ba7f8f9d6e90c8f78c66ae30c'
PACKING = 'lightx2v_train/lightx2v_train/model_zoo/native/minimax_h3/packing.py'
TRAINER = 'lightx2v_train/lightx2v_train/trainers/dmd/minimax_h3_trainer.py'
MODEL = 'lightx2v_train/lightx2v_train/model_zoo/minimax_h3_t2av.py'
SOURCE_FILES = {
    PACKING: 'dd02542c46cb3bc310a53197e893c76c563745e7e669bc770158fed19c044ae0',
    TRAINER: 'a940de8f73c0374c2ac5bf6b6bf6518d13ee0101b676e0625776396075f57b8a',
    MODEL: 'a224c5cf173b8ae40136eca92478c351a5492779783128c43625aec3da3bfa83',
}
_Carrier = None
_Projection = None


def require(ok, message):
    if not ok:
        raise ValueError('H3 Grid output: ' + message)


def plain(value):
    return json.loads(json.dumps(value, allow_nan=False))


def runtime_identity():
    import importlib
    import importlib.metadata
    result = {name: dict(version=importlib.metadata.version(name),
                       entry_sha256=file_sha256(Path(importlib.import_module(name).__file__)))
            for name in ('torch', 'diffusers', 'peft', 'numpy', 'safetensors', 'transformers')}
    # Version/entry files alone do not bind edited attention or LoRA helpers.
    helpers = ('diffusers.models.attention_dispatch', 'diffusers.models.attention',
        'diffusers.models.modeling_utils', 'diffusers.models.embeddings', 'diffusers.loaders.peft',
        'peft.tuners.lora.layer', 'peft.tuners.tuners_utils')
    result['helpers'] = {name: file_sha256(Path(importlib.import_module(name).__file__)) for name in helpers}
    return result


def source_identity():
    from . import upstream,h3_grid_codec
    return dict(model_sha256=MODEL_SHA, files=SOURCE_FILES,
                consumer_sha256=file_sha256(Path(__file__)),
                codec_bridge_sha256=file_sha256(Path(h3_grid_codec.__file__)),
                importer_sha256=file_sha256(Path(upstream.__file__)))


def source_api(source):
    """Unmodified upstream definitions, without registry/cluster initialization.

    Only methods used below execute. Their original constructors, train/loss
    methods and the synthetic base class are never used as a training backend.
    """
    import torch
    from .upstream import PinnedSource
    require(isinstance(source, dict) and set(source) == {'root', 'files', 'terms'}
            and source['files'] == SOURCE_FILES and Path(source['root']).is_absolute(),
            'explicit original LightX2V source closure required')
    terms = source['terms']
    require(isinstance(terms, dict) and set(terms) == {'origin', 'license', 'accepted'}
            and all(isinstance(terms[k], str) and terms[k].strip() for k in ('origin', 'license'))
            and terms['accepted'] is True, 'operator-supplied source terms must be accepted')
    lock = PinnedSource(source['root'], SOURCE_FILES)
    packing = PinnedSource(source['root'], {PACKING: SOURCE_FILES[PACKING]}).import_module(
        PACKING[:-3].replace('/', '.'))
    decorate = lambda name: lambda cls: cls
    names = {k: getattr(packing, k) for k in ('audio_latent_num_frames', 'video_latent_num_frames',
                                            'build_packed_sequence', 'build_row_timesteps')}
    trainer_name, model_name = (path[:-3].replace('/', '.') for path in (TRAINER, MODEL))
    loader = PinnedSource(source['root'], {p: SOURCE_FILES[p] for p in (TRAINER, MODEL)},
        definitions={trainer_name: ['_shift_sigma', 'MiniMaxH3T2AVDmdTrainer'],
                     model_name: ['MiniMaxH3T2AVModel']},
        definition_globals={trainer_name: dict(torch=torch, os=os, TRAINER_REGISTER=decorate,
                VideoDmdTrainer=object, broadcast_sequence_parallel_value=lambda x: x, **names),
            model_name: dict(torch=torch, MODEL_REGISTER=decorate, BaseModel=object,
                             nullcontext=nullcontext)})
    return SimpleNamespace(packing=packing, trainer=loader.import_module(trainer_name).MiniMaxH3T2AVDmdTrainer,
                           model=loader.import_module(model_name).MiniMaxH3T2AVModel, lock=lock)


def model_class():
    import diffusers
    from diffusers import MiniMaxH3Transformer3DModel
    cls = MiniMaxH3Transformer3DModel
    require(diffusers.__version__ == '0.40.0.dev0'
            and file_sha256(Path(inspect.getfile(cls))) == MODEL_SHA,
            'the pinned converted H3 Diffusers0.40 model is required; no native W6 substitution')
    return cls


def schedule(api):
    import torch
    base = torch.linspace(1.0, 0.0, 5, dtype=torch.float32)
    module = inspect.getmodule(api.trainer)
    return module._shift_sigma(base, 6.0), module._shift_sigma(base, 3.0)


def timestep_bits(timesteps):
    import torch
    value = timesteps.detach().to(device='cpu', dtype=torch.float32).contiguous().reshape(-1)
    require(value.numel() in (1, 2) and bool(torch.isfinite(value).all()), 'one/two finite original AV timesteps required')
    bits = value.view(torch.int32).tolist()
    # Exact existing Grid step0 ABI: repeated rows, not a new discretization.
    return bits * 2 if len(bits) == 1 else bits


def rollout_bits(api):
    import torch
    video, audio = schedule(api)
    return [timestep_bits(torch.unique(torch.stack((1-video[i], 1-audio[i])), sorted=True)) for i in range(4)]


def current_student_view(transformer, controller, api, *, max_table_bytes):
    """Read-only state view plus just the four original mmap rollout entries.

    DTensors stay DTensors: the existing all-owner exporter, not this helper,
    must materialize them. No role selection, cache staging or LoRA merge runs.
    """
    import torch
    adapter = transformer.peft_config['student']
    require(adapter.r == adapter.lora_alpha == 128 and adapter.lora_dropout == 0.0
        and adapter.bias == 'none' and not adapter.use_dora and not adapter.use_rslora
        and not adapter.fan_in_fan_out and set(adapter.target_modules) == set(LORA['target_modules'])
        and not any(getattr(m, 'merged_adapters', ()) for m in transformer.modules()),
        'original unmerged Student LoRA128 required; merged weights cannot be exported as unmerged')
    shape = (4, len(transformer.transformer_blocks), 6, 6, transformer.config.hidden_size)
    require(type(max_table_bytes) is int and math.prod(shape)*2 <= max_table_bytes,
            'explicit rollout-table staging capacity required')
    payload = controller.payload
    require(payload['schema_version'] == 'h3_a100.adaln_grid.v1' and payload['grid_size'] == 1000
        and payload['num_blocks'] == shape[1] and payload['hidden_size'] == shape[-1]
        and payload['rows_per_entry'] == payload['modulation_chunks'] == 6
        and payload['video_shift'] == 6.0 and payload['audio_shift'] == 3.0,
        'matching original Grid1000 table identity required')
    bits = rollout_bits(api)
    indices = [controller._entry_by_timestep_bits[tuple(pair)] for pair in bits]
    require(len(set(indices)) == 4 and all(type(i) is int and 1000 <= i < payload['num_entries'] for i in indices),
            'all four explicit original rollout entries are required')
    table = controller._table
    require(table.device.type == 'cpu' and table.dtype == torch.bfloat16
        and tuple(table.shape) == (payload['num_entries'], *shape[1:]), 'original frozen BF16 CPU mmap required')
    selected = {k: v for k, v in transformer.state_dict().items() if '.fake.' not in k}
    # Native RoPE is nonpersistent and may have been dtype-cast by the owner.
    # Capture its ACTUAL value too, without changing the live module's flags.
    selected['rope.inv_freq'] = transformer.rope.inv_freq
    require(not any('.adaln_proj.' in k for k in selected)
        and any('.lora_A.student.' in k for k in selected) and any('.lora_B.student.' in k for k in selected),
        'current unmerged Grid Student required; raw AdaLN or missing Student is not this ABI')
    tables = torch.stack([table[i] for i in indices])
    require(bool(torch.isfinite(tables).all()), 'nonfinite frozen rollout entry')
    return {**{'transformer.'+k: v.detach() for k, v in selected.items()}, 'rollout_modulations': tables}, {
        'entry_indices': indices, 'timestep_bits': bits, 'table_bytes': tables.numel()*2,
        'state': 'FROZEN_REMAINING_BASE_CURRENT_UNMERGED_STUDENT_EXACT_ROLLOUT_TABLES',
        'consolidation': 'NOT_PERFORMED_ALL_DTENSOR_OWNERS_STILL_REQUIRED'}


def geometry(api, value, *, text_dim):
    require(isinstance(value, dict) and set(value) == {'height', 'width', 'frames', 'text_tokens'}
            and all(type(n) is int and n > 0 for n in value.values())
            and value['height'] % 32 == value['width'] % 32 == 0
            and value['height'] <= 4096 and value['width'] <= 4096
            and value['frames'] <= 2048 and value['text_tokens'] <= 65536,
            'explicit valid original full-sequence geometry required')
    owner = object.__new__(api.trainer)
    owner.model = SimpleNamespace(vae_spatial_scale_factor=16, patch_size=(1, 2, 2),
                                  video_latent_channels=24, audio_latent_channels=32)
    owner.dmd_config = dict(height=value['height'], width=value['width'], num_frames=value['frames'])
    shape = owner._latent_shape(None)
    return shape, dict(prompt_embeds=[1, value['text_tokens'], text_dim], text_token_tags=[value['text_tokens']])


def parameters(*, source, architecture, geometry, weight_layout, seed, evidence_kind, budget):
    return plain(dict(schema_version=SCHEMA, precision=PRECISION, implementation=source_identity(),
        runtime=runtime_identity(), source=source, architecture=architecture, geometry=geometry,
        weight_layout=weight_layout, seed=seed, evidence_kind=evidence_kind, budget=budget,
        lora=LORA, schedule=SCHEDULE))


def _request(value):
    import torch
    require(isinstance(value, dict) and value.get('device', 'cpu') == 'cpu', 'single-process CPU inference only')
    p = plain({k: v for k, v in value.items() if k != 'device'})
    require(set(p) == {'schema_version', 'precision', 'implementation', 'runtime', 'source', 'architecture',
        'geometry', 'weight_layout', 'seed', 'evidence_kind', 'budget', 'lora', 'schedule'}
        and p['schema_version'] == SCHEMA and p['precision'] == PRECISION
        and p['implementation'] == source_identity() and p['runtime'] == runtime_identity()
        and str(torch.__version__).split('+')[0] == '2.10.0', 'exact schema/source/runtime/precision required')
    require(p['evidence_kind'] in (FIXTURE, NATIVE) and type(p['seed']) is int and 0 <= p['seed'] < 2**63
        and p['lora'] == LORA and p['schedule'] == SCHEDULE, 'original unmerged LoRA128/four-step6/3 schedule required')
    limits = dict(max_parameters=40_000_000_000, max_loaded_bytes=512*2**30,
        max_attention_elements=100_000_000_000, max_input_elements=64_000_000, host_reserve_bytes=512*2**30)
    require(isinstance(p['budget'], dict) and set(p['budget']) == set(limits)
        and all(type(p['budget'][k]) is int and 0 < p['budget'][k] <= n for k, n in limits.items()),
        'explicit bounded parameter/working-set/attention/host capacity required')
    require(isinstance(p['weight_layout'], dict) and p['weight_layout'] and all(
        isinstance(k, str) and (k.startswith('transformer.') or k == 'rollout_modulations')
        and isinstance(v, dict) and set(v) == {'shape', 'dtype'} and isinstance(v['shape'], list)
        and all(type(n) is int and n > 0 for n in v['shape'])
        and v['dtype'] in ('torch.float32', 'torch.bfloat16') for k, v in p['weight_layout'].items()),
        'explicit complete native storage layout required')
    require(not any('.fake.' in k or '.adaln_proj.' in k for k in p['weight_layout'])
        and any('.lora_A.student.' in k for k in p['weight_layout'])
        and any('.lora_B.student.' in k for k in p['weight_layout']),
        'only current unmerged Student plus remaining frozen base and exact rollout tables are exported')
    return p


def _classes():
    global _Carrier, _Projection
    if _Carrier is None:
        import torch
        class GridProjection(torch.nn.Module):
            def __init__(self, owner, index):
                super().__init__()
                object.__setattr__(self, 'owner', owner)
                self.index = index

            def forward(self, temb):
                require(self.owner.selected is not None, 'rollout table key is not active')
                return tuple(self.owner.rollout_modulations[self.owner.selected, self.index].unbind(0))

        class GridStudentWeights(torch.nn.Module):
            def __init__(self, transformer, table, bits):
                super().__init__()
                self.transformer = transformer
                self.register_buffer('rollout_modulations', table)
                self.bits, self.selected, self.restored = bits, None, False

            def forward(self, **kwargs):
                require(self.selected is None, 'nested Grid inference forward')
                bits = timestep_bits(kwargs['timestep'])
                require(bits in self.bits, 'timestep outside the original four-step rollout table')
                self.selected = self.bits.index(bits)
                try:
                    return self.transformer(**kwargs)
                finally:
                    self.selected = None
        _Carrier, _Projection = GridStudentWeights, GridProjection
    return _Carrier, _Projection


def _meta(p, cls, api):
    import torch
    from peft import LoraConfig
    defaults = {k: v.default for k, v in inspect.signature(cls.__init__).parameters.items() if k != 'self'}
    arch = p['architecture']
    require(isinstance(arch, dict) and set(arch) == set(defaults), 'complete converted-model architecture required')
    expected = plain(defaults)
    if p['evidence_kind'] == NATIVE:
        require(arch == expected and all(type(arch[k]) is type(v) for k, v in expected.items()),
                'production requires the original full50block converted architecture')
    else:
        small = {'num_attention_heads', 'attention_head_dim', 'hidden_size', 'num_layers', 'num_refiner_layers',
                 'ffn_dim', 'freq_dim', 'time_embed_hidden_dim', 'time_embed_dim', 'rope_freq_dim'}
        require(all(type(arch[k]) is int and 0 < arch[k] <= 256 for k in small)
            and arch['num_layers'] <= 2 and arch['num_refiner_layers'] <= 2
            and arch['hidden_size'] <= 64 and 6*arch['rope_freq_dim'] <= arch['attention_head_dim']
            and arch['freq_dim'] % 2 == 0
            and all(arch[k] == expected[k] and type(arch[k]) is type(expected[k]) for k in set(expected)-small),
            'bounded actual original-class fixture required, never reduced production')
    shape, inputs = geometry(api, p['geometry'], text_dim=arch['text_dim'])
    Carrier, Projection = _classes()
    bits = rollout_bits(api)
    with torch.device('meta'):
        transformer = cls(**arch)
        transformer.add_adapter(LoraConfig(**deepcopy(LORA)), adapter_name='student')
        carrier = Carrier(transformer, torch.empty((4, arch['num_layers'], 6, 6, arch['hidden_size']),
                                                 dtype=torch.bfloat16), bits)
        for i, block in enumerate(transformer.transformer_blocks):
            block.adaln_proj = Projection(carrier, i)
        transformer.rope._non_persistent_buffers_set.discard('inv_freq')
    state = carrier.state_dict(keep_vars=True)
    require(set(state) == set(p['weight_layout']), 'complete current Student/table keys required')
    for name, tensor in state.items():
        spec = p['weight_layout'][name]
        require(list(tensor.shape) == spec['shape'], 'original state shape mismatch')
        if name == 'rollout_modulations':
            require(spec['dtype'] == 'torch.bfloat16', 'native rollout table must remain BF16')
        parent, _, leaf = name.rpartition('.')
        module = carrier.get_submodule(parent)
        dtype = {'torch.float32': torch.float32, 'torch.bfloat16': torch.bfloat16}[spec['dtype']]
        value = torch.empty(tensor.shape, device='meta', dtype=dtype)
        if leaf in module._parameters:
            module._parameters[leaf] = torch.nn.Parameter(value, requires_grad=False)
        else:
            module._buffers[leaf] = value
    require(set(dict(carrier.named_buffers())) == {'rollout_modulations', 'transformer.rope.inv_freq'},
            'unexpected model buffers need an explicit export lifetime')
    carrier.requires_grad_(False).eval()
    count = sum(t.numel() for t in state.values())
    size = sum(math.prod(v['shape'])*(2 if v['dtype'] == 'torch.bfloat16' else 4) for v in p['weight_layout'].values())
    tokens = shape['video_tokens'][1] + shape['audio_tokens'][1] + p['geometry']['text_tokens']
    attention = arch['num_attention_heads'] * tokens**2
    elements = sum(math.prod(s) for s in inputs.values()) + sum(math.prod(shape[k]) for k in ('video_tokens', 'audio_tokens'))
    needed = 2*size + 4*(attention + 12*tokens*arch['hidden_size'] + 4*elements)
    b = p['budget']
    require(count <= b['max_parameters'] and attention <= b['max_attention_elements']
        and elements <= b['max_input_elements'] and needed <= b['max_loaded_bytes'], 'model/input/attention working-set capacity exceeded')
    return carrier, dict(parameters=count, weight_bytes=size, host_bytes=needed, attention_elements=attention)


def _metadata(p):
    return dict(schema_version=SCHEMA, parameters=deepcopy(p),
        scope='CURRENT_CONVERTED_GRID_STUDENT_CPU_REFERENCE_PACKED_AV',
        semantics='approximate_grid1000', selection='CURRENT_UNMERGED_STUDENT_NOT_EMA',
        rollout='ORIGINAL_LIGHTX2V_FOUR_STEP_DATA_WARD_VELOCITY',
        adaLN='EXACT_FOUR_FROZEN_ROLLOUT_ENTRIES_NOT_RECONSTRUCTED_PROJECTIONS',
        media='PACKED_TOKENS; OPTIONAL_ORIGINAL_CODEC_EXACT_PATCH_AND_STEREO_ABI',
        evidence_kind=p['evidence_kind'], quality_validation='NOT_RUN', training_resume='INFERENCE_ONLY')


def _loaded(module, incompatible):
    import torch
    require(not incompatible.missing_keys and not incompatible.unexpected_keys
        and all(not t.is_meta and t.device.type == 'cpu' and bool(torch.isfinite(t).all())
                for t in module.state_dict().values()), 'complete finite CPU inference state required')
    module.restored = True


def build_inference_model(value):
    import torch
    from .checkpoint import rng_state, restore_rng
    require(int(os.environ.get('WORLD_SIZE', '1')) == 1, 'one CPU inference process required')
    p = _request(value)
    before = rng_state(device='cpu')
    try:
        api = source_api(p['source'])
        carrier, sizes = _meta(p, model_class(), api)
        admit_host_memory(sizes['host_bytes'], p['budget']['host_reserve_bytes'])
        carrier.to_empty(device='cpu')
        # The snapshot explicitly includes the otherwise-nonpersistent RoPE
        # buffer. Strict restore, not a guessed reconstruction, initializes it.
        carrier.register_load_state_dict_post_hook(_loaded)
        state={'student':carrier,'inference':p}
        def decode_output(generated,options):
            require(isinstance(options,dict) and set(options)=={'decoder_manifest'},
                    'explicit original decoder_manifest required; use --output-kind latents otherwise')
            from .h3_grid_codec import decode_generated
            return decode_generated(generated,manifest=options['decoder_manifest'],source=p['source'],geometry=p['geometry'])
        return ModelComponents(state=state,metadata=_metadata(p),generate=generate,
                               decode=decode_output,admit_training=refuse_training).validate()
    finally:
        restore_rng(before)


def export_components(value, *, weights):
    """Serialize an explicit current Student snapshot, never a LoRA merge.

    The native owner is responsible for complete restore and all-rank/capacity
    admission before supplying these CPU weights. This function is not itself
    the native training checkpoint loader or an FSDP consolidation routine.
    """
    import torch
    from .checkpoint import rng_state, restore_rng
    p = _request(value)
    before = rng_state(device='cpu')
    try:
        carrier, _ = _meta(p, model_class(), source_api(p['source']))
        require(isinstance(weights, dict) and set(weights) == set(p['weight_layout'])
            and all(isinstance(t, torch.Tensor) and not t.requires_grad and t.device.type == 'cpu'
                and list(t.shape) == p['weight_layout'][k]['shape'] and str(t.dtype) == p['weight_layout'][k]['dtype']
                and bool(torch.isfinite(t).all()) for k, t in weights.items()), 'exact complete current native CPU snapshot required')
        carrier.load_state_dict(weights, strict=True, assign=True)
        return ModelComponents(state={'student': carrier}, metadata=_metadata(p)).validate()
    finally:
        restore_rng(before)


def generate(state, feeds, options):
    import torch
    from .checkpoint import rng_state, restore_rng
    p, carrier = state['inference'], state['student']
    require(isinstance(options,dict) and set(options)<={'decoder_manifest'} and carrier.restored and carrier.selected is None,
            'strictly restored Student and original schedule; no overrides')
    require(p['runtime'] == runtime_identity() and p['implementation'] == source_identity()
        and not carrier.training and all(not t.requires_grad and t.device.type == 'cpu' for t in carrier.parameters()),
        'inference owner/source/runtime changed')
    api = source_api(p['source'])
    shape, inputs = geometry(api, p['geometry'], text_dim=p['architecture']['text_dim'])
    if 'decoder_manifest' in options:
        from .h3_grid_codec import admit_binding
        admit_binding(options['decoder_manifest'],source=p['source'],geometry=p['geometry'])
    require(isinstance(feeds, dict) and set(feeds) == set(inputs), 'new cached text-only condition required')
    for name, tensor in feeds.items():
        dtype = torch.int64 if name == 'text_token_tags' else torch.bfloat16
        require(isinstance(tensor, torch.Tensor) and tensor.device.type == 'cpu'
            and list(tensor.shape) == inputs[name] and tensor.dtype == dtype
            and (bool((tensor == 1).all()) if name == 'text_token_tags' else bool(torch.isfinite(tensor).all())),
            'original cached condition shape/dtype/tags/finite mismatch')
    model = object.__new__(api.model)
    model.device, model.running_dtype = torch.device('cpu'), torch.bfloat16
    model.use_autocast, model.transformer = False, carrier
    model.patch_size = (1, 2, 2)
    owner = object.__new__(api.trainer)
    owner.model, owner._layout_cache, owner.latent_dtype = model, {}, torch.float32
    owner.video_sigmas_cpu, owner.audio_sigmas_cpu = schedule(api)
    condition = model.prepare_text_condition(feeds)
    before = rng_state(device='cpu')
    try:
        torch.manual_seed(p['seed'])
        # This calls the original sample_initial_latents and run_back_simulation,
        # including train() and x0's data-ward sign/FP32 Euler update order.
        with torch.no_grad():
            video, audio = owner.run_back_simulation(condition, shape, 3, False)
    finally:
        carrier.eval()
        restore_rng(before)
    schema = {name: dict(layout='BLD', shape=list(shape[key]), dtype='torch.float32')
              for name, key in (('video_tokens', 'video_tokens'), ('audio_tokens', 'audio_tokens'))}
    require(all(t.dtype == torch.float32 and list(t.shape) == schema[k]['shape'] and bool(torch.isfinite(t).all())
                for k, t in (('video_tokens', video), ('audio_tokens', audio))), 'original packed AV output changed')
    return dict(video_tokens=video, audio_tokens=audio, latent_schema=schema)


def refuse_training(*args, **kwargs):
    raise ValueError('Grid inference weights are not a resumable training checkpoint')
