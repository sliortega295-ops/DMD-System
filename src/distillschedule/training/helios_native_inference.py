"""Current W2 Student output using the original owner and shared video codec.

Training is restored by helios_worker, never by an inference reconstruction.
Only the independent inference target is frozen. No training asset is needed
after publication of the existing portable named-module artifact.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
import os
from pathlib import Path

import torch

from distillschedule.search.helios_request import OriginalHeliosRequest
from distillschedule.workloads.best_known.registry import file_sha256, repository_root
from . import distributed
from .host_memory import admit_host_memory as _host_admit
from .adapters import InferenceProjection, ModelComponents
from . import wan_native_inference as video

SCHEMA = 'distillschedule.helios_native_generator_inference.v1'
EXPORT_SCHEMA = 'distillschedule.helios_native_output_request.v1'
FACTORY = 'distillschedule.training.helios_native_inference:build_inference_model'
PRECISION = 'HELIOS_NATIVE_BF16_BASE_FP32_LORA_UNMERGED'


def require(ok, message):
    if not ok:
        raise ValueError('native Helios output: ' + message)


def inference_source():
    root = repository_root()
    files = ['src/distillschedule/training/helios_native_inference.py',
             'src/distillschedule/training/helios_worker.py',
             'src/distillschedule/training/helios_lifecycle.py',
             'src/distillschedule/search/helios_request.py',
             'src/distillschedule/search/resources.py',
             'src/distillschedule/training/host_memory.py']
    return dict(video=video.inference_source(), files={name: file_sha256(root / name) for name in files})


def _budget(value):
    bounds = dict(max_parameters=20_000_000_000, max_loaded_bytes=512 * 2**30,
        max_primary_export_bytes=256 * 2**30, max_transient_tensor_bytes=8 * 2**30,
        max_activation_elements=4_000_000_000, max_attention_elements=16_000_000_000,
        host_reserve_bytes=256 * 2**30, device_reserve_bytes=32 * 2**30)
    require(isinstance(value, dict) and set(value) == set(bounds)
        and all(type(value[k]) is int and 0 < value[k] <= limit for k, limit in bounds.items()),
        'explicit bounded model, export, transient and host/device reserve budgets required')



def _device_admit(device, needed, reserve):
    if torch.device(device).type == 'cuda':
        free, _ = torch.cuda.mem_get_info(device)
        require(needed + reserve <= free, 'owned-device headroom cannot hold materialization plus reserve')


def _native(value, task):
    from .helios_inference import SHIFT_KEYS
    require(isinstance(value, dict) and set(value) == {'args', 'scheduler', 'rollout'},
            'complete original native sampler controls required')
    args = value['args']
    require(isinstance(args, dict) and set(args) == set(SHIFT_KEYS)
        and type(args['base_seq_len']) is int and type(args['max_seq_len']) is int
        and 0 < args['base_seq_len'] < args['max_seq_len'] <= 65536
        and all(type(args[k]) in (int, float) and 0 < args[k] <= 100
                for k in ('base_shift', 'max_shift')),
        'original dynamic-shift controls changed')
    require(value['scheduler'] == video.NATIVE['scheduler'] and value['rollout'] == video._rollout(task),
            'original three-stage one-section inference controls changed')


def fresh_noise_inputs(p):
    return _fresh_noise_inputs(p, variant='no-gan')


def _fresh_noise_inputs(p, *, variant):
    task = OriginalHeliosRequest.from_dict(p['task'])
    require(task.variant == variant and p['noise_protocol'] == video.noise_protocol()
        and p['geometry']['noise_shape'] == list(task.latent_shape), 'W2 fresh inference identity changed')
    _native(p['native'], task)
    return video._noise_inputs(task, p['seed'])


def _probe(parameters, *, schema=SCHEMA, variant='no-gan', source_factory=None):
    """Admit source, geometry and meta inventory before a real model allocation."""
    from .checkpoint import rng_state, restore_rng
    require(isinstance(parameters, dict) and set(parameters) <= {'schema_version', 'precision', 'source',
        'runtime', 'architecture', 'peft', 'task', 'native', 'noise_protocol', 'geometry', 'budget',
        'seed', 'evidence_kind', 'codec', 'device'}, 'unknown W2 portable parameters')
    p = video.plain({k: v for k, v in parameters.items() if k != 'device'})
    require(set(p) == {'schema_version', 'precision', 'source', 'runtime', 'architecture', 'peft', 'task',
        'native', 'noise_protocol', 'geometry', 'budget', 'seed', 'evidence_kind', 'codec'}
        and p['schema_version'] == schema and p['precision'] == PRECISION, 'explicit native Helios inference ABI required')
    require(p['source'] == (source_factory or inference_source)() and p['runtime'] == video.runtime_identity(),
            'portable W2 source/runtime changed')
    task = OriginalHeliosRequest.from_dict(p['task'])
    require(task.variant == variant and p['evidence_kind'] in video.EVIDENCE_KINDS
        and p['codec']['evidence_kind'] == p['evidence_kind']
        and type(p['seed']) is int and 0 <= p['seed'] < 2**63, 'W2 variant/evidence/seed changed')
    _budget(p['budget'])
    geometry = p['geometry']
    require(isinstance(geometry, dict) and set(geometry) == {'noise_shape', 'prompt_shape', 'video_shape', 'fps'}
        and geometry['noise_shape'] == list(task.latent_shape)
        and geometry['video_shape'] == [task.video_frames, task.video_height, task.video_width, 3]
        and isinstance(geometry['prompt_shape'], list) and len(geometry['prompt_shape']) == 3
        and geometry['prompt_shape'][0] == 1 and geometry['prompt_shape'][2] == 4096
        and type(geometry['prompt_shape'][1]) is int and 0 < geometry['prompt_shape'][1] <= 4096
        and type(geometry['fps']) is int and 1 <= geometry['fps'] <= 120,
        'native W2 video/prompt geometry/fps changed')
    require(p['noise_protocol'] == video.noise_protocol(), 'fresh exact-PSD inference noise ABI changed')
    _native(p['native'], task)
    require(isinstance(p['peft'], dict) and type(p['peft'].get('r')) is int
        and 0 < p['peft']['r'] <= 1024, 'bounded actual W2 LoRA rank required')
    codec_cls = video._admit_codec(p['codec'])
    before = rng_state()
    try:
        g = video._generator(p, meta=True, lora_rank=p['peft']['r'])
        with torch.device('meta'):
            codec = codec_cls(**p['codec']['architecture'])
        tensors = list(g.state_dict().values()) + list(codec.state_dict().values())
        tensors += [v for module in (g, codec) for name, v in module.named_buffers()
                    if name not in module.state_dict()]
        count = sum(v.numel() for v in tensors)
        size = sum(v.numel() * v.element_size() for v in tensors)
        transient = max(v.numel() * v.element_size() for v in tensors)
        pixels = math.prod(geometry['video_shape'][:-1])
        activation = max(8 * math.prod(task.latent_shape), 2 * pixels * p['codec']['architecture']['base_dim'])
        tokens = math.prod(task.latent_shape[2:]) // 4
        attention = max(tokens * tokens * p['architecture']['num_attention_heads'],
                        math.prod(task.latent_shape[-2:]) ** 2)
        construction = max(12 * count, 3 * size)
        require(count <= p['budget']['max_parameters'] and construction <= p['budget']['max_loaded_bytes']
            and transient <= p['budget']['max_transient_tensor_bytes']
            and activation <= p['budget']['max_activation_elements']
            and attention <= p['budget']['max_attention_elements'], 'meta model/codec/attention capacity exceeded')
        return p, g, codec, dict(state_bytes=size, transient_bytes=transient, host_bytes=construction,
                                device_bytes=size + 4 * (activation + attention))
    finally:
        restore_rng(before)


def _metadata(p):
    role = 'W3' if p['task']['variant'] == 'gan' else 'W2'
    return dict(schema_version=p['schema_version'], scope='NATIVE_' + role + '_CURRENT_G_DECODED_VIDEO', precision=PRECISION,
        parameters=deepcopy(p), generator_selection='CURRENT_FULL_UNMERGED_NOT_EMA',
        evidence_kind=p['evidence_kind'], training_resume='NOT_SUPPORTED_PORTABLE_INFERENCE_ONLY',
        quality_validation='NOT_RUN', claimable_speedup=False)


def build_inference_model(parameters):
    return _build_inference_model(parameters, probe=_probe, generate_callback=generate)


def _build_inference_model(parameters, *, probe, generate_callback, refuse=video.refuse_training):
    device = video._environment(parameters.get('device', 'cpu'))
    p, _, _, sizes = probe(parameters)
    host_bytes = sizes['host_bytes'] + (sizes['device_bytes'] if device.type == 'cpu' else 0)
    _host_admit(host_bytes, p['budget']['host_reserve_bytes'])
    _device_admit(device, sizes['device_bytes'], p['budget']['device_reserve_bytes'])
    g = video._generator(p, meta=False, lora_rank=p['peft']['r'])
    codec = video._codec_class()(**p['codec']['architecture'])
    g.requires_grad_(False).eval(); g.disable_gradient_checkpointing()
    codec.requires_grad_(False).eval()
    state = dict(generator=video.CurrentGenerator(g.to(device)), codec=codec.to(device), inference=p)
    def decode(generated, options):
        require(not options, 'codec overrides are not supported')
        return video.decode(state, generated)
    return ModelComponents(state=state, metadata=_metadata(p), generate=generate_callback, decode=decode,
                           admit_training=refuse).validate()


def generate(state, feeds, options):
    return video._generate(state, feeds, options, prepare_noise=fresh_noise_inputs)


def _project(view, codec, request, *, schema=SCHEMA, variant='no-gan', source_factory=None, factory=FACTORY):
    from distillschedule.open_workload.torch_backend import unwrap_module
    from .helios_inference import SHIFT_KEYS
    g = unwrap_module(view.generator.module)
    require(set(g.peft_config) == {'default'} and all(not getattr(m, 'merged', False) for m in g.modules()),
            'CURRENT unmerged default G required; no merged shadow or EMA substitution')
    peft, task = g.peft_config['default'], OriginalHeliosRequest.from_dict(view.request)
    p = video.plain(dict(schema_version=schema, precision=PRECISION, source=(source_factory or inference_source)(),
        runtime=video.runtime_identity(), architecture={k: v for k, v in dict(g.config).items() if not k.startswith('_')},
        peft=dict(r=peft.r, lora_alpha=peft.lora_alpha, lora_dropout=peft.lora_dropout,
                  init_lora_weights=peft.init_lora_weights, target_modules=sorted(peft.target_modules),
                  exclude_modules=sorted(peft.exclude_modules)), task=task.to_dict(),
        native=dict(args={k: getattr(view.args.training_config, k) for k in SHIFT_KEYS},
                    scheduler=deepcopy(video.NATIVE['scheduler']), rollout=video._rollout(task)),
        noise_protocol=video.noise_protocol(), geometry=dict(noise_shape=list(task.latent_shape),
            prompt_shape=request['prompt_shape'], video_shape=[task.video_frames, task.video_height, task.video_width, 3],
            fps=request['fps']), budget=request['budget'], seed=request['seed'], evidence_kind=request['evidence_kind'],
        codec=video.codec_parameters(codec, evidence_kind=request['evidence_kind'], provenance=request['codec']['provenance'])))
    p, probe, codec_probe, sizes = _probe(p, schema=schema, variant=variant, source_factory=source_factory)
    for live, expected, label in ((g, probe, 'G'), (codec, codec_probe, 'codec')):
        actual, target = live.state_dict(), expected.state_dict()
        require(actual.keys() == target.keys() and all(actual[k].shape == target[k].shape
            and actual[k].dtype == target[k].dtype for k in target), label + ' full state/dtype layout changed')
    rank, _ = distributed.rank_world()
    local_world = int(os.environ.get('LOCAL_WORLD_SIZE', '1'))
    require(0 < local_world <= 8, 'explicit supported local owner count required')
    codec_bytes = sum(v.numel() * v.element_size() for v in codec.state_dict().values())
    primary_bytes = 3 * sizes['state_bytes']
    require(primary_bytes <= p['budget']['max_primary_export_bytes'], 'primary full export exceeds declared capacity')
    # All local codec owners plus the primary's complete gathered state. No
    # nonwriter full CPU G copy is charged or allocated by this path.
    node = rank // local_world
    _host_admit(3 * codec_bytes * local_world + (primary_bytes if node == 0 else 0), p['budget']['host_reserve_bytes'])
    _device_admit(next(g.parameters()).device, sizes['transient_bytes'], p['budget']['device_reserve_bytes'])
    components = ModelComponents(state=dict(generator=video.CurrentGenerator(g), codec=codec), metadata=_metadata(p)).validate()
    return InferenceProjection(components, factory, p).validate()


class NativeOutputRequest:
    export_schema = EXPORT_SCHEMA
    factory = FACTORY

    def __init__(self, value, output):
        self.value, self.output = video.plain(value), Path(output).expanduser().resolve()
        self.config_path = self.output.with_name(self.output.name + '.inference.json')
        self.codec, self.result = None, None

    def admit(self):
        def local():
            p = self.value
            require(isinstance(p, dict) and set(p) == {'schema_version', 'seed', 'prompt_shape', 'fps', 'budget', 'codec', 'evidence_kind'}
                and p['schema_version'] == self.export_schema, 'explicit native Helios trainer.export policy required')
            _budget(p['budget'])
            require(type(p['seed']) is int and 0 <= p['seed'] < 2**63 and p['evidence_kind'] in video.EVIDENCE_KINDS
                and type(p['fps']) is int and 1 <= p['fps'] <= 120
                and isinstance(p['prompt_shape'], list) and len(p['prompt_shape']) == 3
                and p['prompt_shape'][0] == 1 and p['prompt_shape'][2] == 4096
                and type(p['prompt_shape'][1]) is int and 0 < p['prompt_shape'][1] <= 4096,
                'independent inference seed/BF16 prompt/fps/evidence changed')
            require(not self.output.exists() and not self.config_path.exists(), 'new portable output/config required')
            # The explicit shard inventory bounds codec reading before its
            # constructor runs; _load_codec then validates actual meta shapes.
            pins = p['codec'].get('weights') if isinstance(p['codec'], dict) else None
            require(isinstance(pins, list) and pins and all(isinstance(v, dict) and type(v.get('bytes')) is int and v['bytes'] > 0
                for v in pins), 'pinned bounded codec shards required')
            codec_bytes = sum(v['bytes'] for v in pins)
            local_world = int(os.environ.get('LOCAL_WORLD_SIZE', '1'))
            require(0 < local_world <= 8 and 3 * codec_bytes <= p['budget']['max_loaded_bytes'], 'codec local capacity exceeded')
            _host_admit(3 * codec_bytes * local_world, p['budget']['host_reserve_bytes'])
            self.codec, portable = video._load_codec(p['codec'], p['budget'], p['evidence_kind'])
            return dict(request=p, codec=portable, output=str(self.output), config_path=str(self.config_path))
        value = distributed.checked(local, 'W2 native output codec/capacity admission')
        require(all(row == value for row in distributed.gather(value)), 'ranks disagree on W2 codec/output request')

    def project(self, view):
        return _project(view, self.codec, self.value)

    def publish(self, view, *, manifest, identity):
        from .inference import publish_export
        from .plugins import load_callable
        projection = distributed.checked(lambda: self.project(view), 'native Helios CURRENT G projection')
        _, source = distributed.checked(lambda: load_callable(self.factory), 'Helios inference factory source')
        metadata = dict(projection.components.metadata)
        exported = publish_export(projection.components, self.output,
            model=dict(factory=self.factory, source=source, parameters=projection.parameters, metadata=metadata),
            identity=dict(model=source, model_metadata=metadata, training_provenance=identity),
            cycle=manifest['cycle'], primary_only=True)
        def write_config():
            import uuid
            temporary = self.config_path.with_name('.' + self.config_path.name + '.tmp-' + uuid.uuid4().hex)
            config = video.standalone_config(exported, self.config_path)
            config['algorithm']['factory'] = self.factory.rpartition(':')[0] + ':refuse_training'
            try:
                temporary.write_text(json.dumps(config, indent=2, sort_keys=True) + '\n')
                os.link(temporary, self.config_path)
            finally:
                temporary.unlink(missing_ok=True)
        distributed.primary(write_config, 'W2 standalone inference config publication')
        self.result = {**exported, 'inference_config': str(self.config_path),
                       'output_transport': 'PRIMARY_ONLY_CPU_STAGING_ALL_OWNERS_FULL_TENSOR'}
        return self.result


def refuse_training(*args, **kwargs):
    raise ValueError('portable W2 inference weights are not a training checkpoint')


def export_native(config, checkpoint, output):
    from .config import TrainingConfig
    from .native import _admit, _helios_stream, _helios_safe_stream
    from . import helios_worker
    require(_helios_stream(config, 'W2'), 'W2 output requires its original complete fresh-stream checkpoint')
    raw = config.to_dict()
    request = raw['trainer'].pop('export', None)
    require(isinstance(request, dict), 'explicit W2 trainer.export codec/output policy required')
    raw['trainer'].update(checkpoint_every=0, resume_from=str(Path(checkpoint).expanduser().resolve()),
        output_dir=str(Path(output).expanduser().resolve().with_name(Path(output).name + '.native-owner')))
    clean = TrainingConfig.from_dict(raw, source=config.source)
    callback = NativeOutputRequest(request, output)
    runner = helios_worker.run_safe_stream if _helios_safe_stream(clean, 'W2') else helios_worker.run_reference
    def admit():
        result = _admit(clean)
        require(not result[1]['identity']['dirty'], 'clean committed source required')
        return result
    status = runner(clean, admit=admit, native_output=callback)
    require(status == 0 and callback.result is not None, 'original W2 owner did not publish the requested output')
    return callback.result
