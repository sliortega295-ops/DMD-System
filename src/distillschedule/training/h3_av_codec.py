"""Exact external H3 decoders, independent from Student/optimizer ownership.

Original classes/methods execute through PinnedSource; source and weights are
operator-supplied, never fetched or redistributed here. Decoder-only parameter
subtrees avoid allocating unused encoders. This is CPU FP32, not a CUDA backend.
"""
import ast
from copy import deepcopy
import importlib.metadata
import math
from pathlib import Path
from types import SimpleNamespace

from distillschedule.workloads.best_known.registry import file_sha256, fingerprint
from .adapters import ModelComponents
from .host_memory import admit_host_memory

SCHEMA = 'distillschedule.original_h3_av_decoder.v1'
PRECISION = 'CPU_FP32_ORIGINAL_H3_DECODE'
FIXTURE = 'RANDOM_ORIGINAL_H3_CODEC_NOT_PRETRAINED'
LOCAL = 'LOCAL_ORIGINAL_H3_CODEC_UNVERIFIED_QUALITY'
SOURCE_FILES = {
    'diffsynth/core/attention/attention.py': '1126b2e19184299cbfc611dcf66e46d0fd1a7bfa1d616f2ed38bc60ba19679ae',
    'diffsynth/models/minimax_h3_video_vae.py': '3dd4da5c90135c904fa5f1d4f70dd43e2571e394ce437bb168a5110f71076bf4',
    'diffsynth/models/minimax_h3_audio_vae.py': 'ce6dc03b73a0596df22a6aa89e6db9183017bd5d0e2e99b547a6bfee53a7ef2c',
}
VIDEO_NATIVE = dict(patch_size=16, patch_size_t=4, in_channels=24, out_channels=3,
    num_layers=36, heads=32, dim_head=64, norm_affine=True, qk_norm_type='rms_norm',
    qk_norm_affine=False, rope_theta=100.0, rope_dim_ratio=.75, bias=True, eps=1e-5, num_register_tokens=4)


def require(ok, message):
    if not ok:
        raise ValueError('original H3 codec: ' + message)


def _pin(pin):
    require(isinstance(pin, dict) and set(pin) == {'path', 'bytes', 'sha256'}
        and isinstance(pin['path'], str) and Path(pin['path']).is_absolute()
        and type(pin['bytes']) is int and pin['bytes'] > 0
        and isinstance(pin['sha256'], str) and len(pin['sha256']) == 64
        and all(c in '0123456789abcdef' for c in pin['sha256']), 'exact local safe weight pin required')


def validate_source(source):
    """Shared original codec source identity, without importing the models."""
    require(isinstance(source, dict) and set(source) == {'root', 'files', 'provenance'}
        and isinstance(source['root'], str) and Path(source['root']).is_absolute()
        and source['files'] == SOURCE_FILES, 'exact external original source closure required')
    provenance = source['provenance']
    require(isinstance(provenance, dict) and set(provenance) == {'origin', 'license', 'terms_accepted'}
        and all(isinstance(provenance[k], str) and provenance[k].strip() for k in ('origin', 'license'))
        and provenance['terms_accepted'] is True, 'operator must supply and accept external source terms')


def validate_manifest(p):
    require(isinstance(p, dict) and set(p) == {'schema_version', 'precision', 'storage_dtype', 'evidence_kind',
        'source', 'video_decoder', 'geometry', 'weights', 'budget'}
        and p['schema_version'] == SCHEMA and p['precision'] == PRECISION
        and p['storage_dtype'] in ('float32', 'bfloat16_to_float32') and p['evidence_kind'] in (FIXTURE, LOCAL),
        'explicit native codec schema/precision/storage/evidence required')
    fingerprint(p)
    validate_source(p['source'])
    video = p['video_decoder']
    require(isinstance(video, dict) and set(video) == set(VIDEO_NATIVE), 'complete original video decoder architecture required')
    expected = deepcopy(VIDEO_NATIVE)
    if p['evidence_kind'] == FIXTURE:
        require(all(type(video[k]) is int and 1 <= video[k] <= 2 for k in ('num_layers', 'heads'))
            and video['dim_head'] == 16 and type(video['dim_head']) is int, 'bounded original-class tiny decoder required')
        expected.update({k: video[k] for k in ('num_layers', 'heads', 'dim_head')})
    require(video == expected and all(type(video[k]) is type(v) for k, v in expected.items()),
            'original native decoder architecture changed; reduced production decoder refused')
    g = p['geometry']
    require(isinstance(g, dict) and set(g) == {'video_latent_shape', 'audio_latent_shape', 'video_shape',
        'audio_shape', 'fps', 'sample_rate'}, 'complete AV geometry required')
    for name in ('video_latent_shape', 'audio_latent_shape', 'video_shape', 'audio_shape'):
        require(isinstance(g[name], list) and all(type(n) is int and n > 0 for n in g[name]), 'positive integer AV shape required')
    v, a = g['video_latent_shape'], g['audio_latent_shape']
    require(len(v) == 5 and v[:2] == [1, 24] and 3 <= v[2] <= 2048 and max(v[3:]) <= 256
        and len(a) == 3 and a[:2] == [2, 32] and a[2] <= 1_000_000
        and len(g['video_shape']) == 4 and g['video_shape'][1:] == [16*v[3], 16*v[4], 3]
        and g['audio_shape'] == [800*a[2], 2] and type(g['fps']) is int and 1 <= g['fps'] <= 120
        and type(g['sample_rate']) is int and g['sample_rate'] == 32000,
        'original video BCTHW/stereo SCT, pixel geometry and 32kHz audio required; no resize/resampling')
    limits = dict(max_parameters=3_000_000_000, max_loaded_bytes=256*2**30,
        max_activation_elements=4_000_000_000, max_attention_elements=8_000_000_000, host_reserve_bytes=128*2**30)
    budget = p['budget']
    require(isinstance(budget, dict) and set(budget) == set(limits)
        and all(type(budget[k]) is int and 0 < budget[k] <= v for k, v in limits.items()),
        'explicit bounded decoder/working-set/host reserve capacity required')
    require(isinstance(p['weights'], dict) and set(p['weights']) == {'video', 'audio'}, 'both native codec weight inventories required')
    for pins in p['weights'].values():
        require(isinstance(pins, list) and 0 < len(pins) <= 128, 'bounded safetensor shard inventory required')
        for pin in pins:
            _pin(pin)
    return deepcopy(p)


def source_api(source):
    """Original definitions, explicit native SDPA lane, no import-time kernel probes."""
    import inspect
    import os
    from typing import List
    import einops
    import numpy as np
    import torch
    from torchvision.transforms import Normalize
    from .upstream import PinnedSource
    root = Path(source['root'])
    require(root.resolve() == root and root.is_dir(), 'existing canonical external source root required')
    lock = PinnedSource(root, SOURCE_FILES)
    paths = list(SOURCE_FILES)
    def names(path):
        return [n.name for n in ast.parse((root/path).read_bytes()).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    def constants(path):
        values = {}
        for n in ast.parse((root/path).read_bytes()).body:
            if (isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                    and n.targets[0].id.startswith(('_VIDEO_LATENTS_', '_AUDIO_LATENTS_'))):
                values[n.targets[0].id] = ast.literal_eval(n.value)
        return values
    modules = [p[:-3].replace('/', '.') for p in paths]
    attention_loader = PinnedSource(root, {paths[0]: SOURCE_FILES[paths[0]]},
        definitions={modules[0]: names(paths[0])}, definition_globals={modules[0]: dict(torch=torch, os=os,
        inspect=inspect, rearrange=einops.rearrange, repeat=einops.repeat,
        ATTENTION_IMPLEMENTATION='torch', TORCH_SUPPORT_GQA=False)})
    attention = attention_loader.import_module(modules[0])
    shared = dict(math=math, torch=torch, nn=torch.nn, F=torch.nn.functional, np=np, Parameter=torch.nn.Parameter,
        List=List, Normalize=Normalize, rearrange=einops.rearrange, repeat=einops.repeat,
        attention_forward=attention.attention_forward)
    loader = PinnedSource(root, {p: SOURCE_FILES[p] for p in paths[1:]},
        definitions={m: names(p) for m, p in zip(modules[1:], paths[1:])},
        definition_globals={m: {**shared, **constants(p)} for m, p in zip(modules[1:], paths[1:])})
    video, audio = [loader.import_module(m) for m in modules[1:]]
    lock.validate()
    return SimpleNamespace(video=video, audio=audio, attention=attention, lock=lock)


def meta_decoders(p, api):
    """Allocate no concrete model tensors, keep original decode methods intact."""
    import torch
    from .checkpoint import rng_state, restore_rng
    before = rng_state(device='cpu')
    try:
        with torch.device('meta'):
            video, audio = api.video.MiniMaxH3VideoVAE(), api.audio.MiniMaxH3AudioVAE()
            if p['video_decoder'] != VIDEO_NATIVE:
                video.decoder = api.video.ViT3DDecoder(**p['video_decoder'])
            original = {role: {n: (tuple(t.shape), t.dtype) for n, t in model.state_dict().items()}
                        for role, model in (('video', video), ('audio', audio))}
            del video.encoder; del video.quant_conv
            for name in ('encoder', 'mean_proj', 'logs_proj', 'pre_block'):
                delattr(audio, name)
        return {'video': video, 'audio': audio}, original
    finally:
        restore_rng(before)


def _admit_geometry(p, models):
    g, v = p['geometry'], models['video']
    latent_t = g['video_latent_shape'][2]
    pseudo = latent_t + v.token_drop
    pad = (v.tokens_chunk_size - pseudo % v.tokens_chunk_size) % v.tokens_chunk_size
    chunks = (pseudo + pad) // v.tokens_chunk_size - int(v.token_drop > 0)
    # The SAME original frame planner admits shape; no approximate time/latent conversion.
    dummy = SimpleNamespace(shape=g['video_latent_shape'])
    # Native decode_temporal pads before invoking the planner.
    dummy.shape = [*dummy.shape[:2], latent_t + pad, *dummy.shape[3:]]
    _, _, frames = v._decode_temporal_output_frame_plan(dummy, chunks, pad)
    require(chunks > 0 and frames == g['video_shape'][0], 'native temporal frame plan disagrees with declared output')
    tokens = min(latent_t + pad, v.tokens_chunk_size + v.token_overlap) * min(g['video_latent_shape'][3], 16) * min(g['video_latent_shape'][4], 16)
    attention = p['video_decoder']['heads'] * (tokens + 5)**2
    activation = (math.prod(g['video_shape']) + 32*math.prod(g['audio_shape'])
                  + 8*tokens*p['video_decoder']['heads']*p['video_decoder']['dim_head'])
    count = sum(t.numel() for m in models.values() for t in m.state_dict().values())
    required = 12*count + 8*(activation + attention)
    b = p['budget']
    require(count <= b['max_parameters'] and required <= b['max_loaded_bytes']
        and activation <= b['max_activation_elements'] and attention <= b['max_attention_elements'],
        'decoder parameters/staging/activation/attention capacity exceeded')
    return dict(parameters=count, required_bytes=required, activation_elements=activation, attention_elements=attention)


def _load_weights(model, pins, original, storage):
    """Admit all safe headers first; read ONLY needed decoder tensors, shard by shard."""
    import torch
    from safetensors import safe_open
    selected = model.state_dict()
    admitted, paths = {}, set()
    allowed = ('F32',) if storage == 'float32' else ('BF16', 'F32')
    for pin in pins:
        path = Path(pin['path'])
        require(path.resolve() == path and path.is_file() and path not in paths
            and path.stat().st_size == pin['bytes'] and file_sha256(path) == pin['sha256'], 'codec weight pin changed/missing')
        require(path.suffix == '.safetensors', 'safe native codec tensors required')
        paths.add(path)
        with safe_open(path, framework='pt', device='cpu') as handle:
            for name in handle.keys():
                header = handle.get_slice(name)
                require(name not in admitted and name in original and tuple(header.get_shape()) == original[name][0]
                    and header.get_dtype() in allowed, 'native codec safe header/key/shape/storage mismatch')
                admitted[name] = path
    require(set(admitted) in (set(selected), set(original)), 'complete full-native or decoder-only safe state required')
    state = {}
    for path in paths:
        with safe_open(path, framework='pt', device='cpu') as handle:
            for name, owner in admitted.items():
                if owner == path and name in selected:
                    value = handle.get_tensor(name).float()
                    require(value.dtype == selected[name].dtype and bool(torch.isfinite(value).all()), 'nonfinite codec weight')
                    state[name] = value
    model.load_state_dict(state, strict=True, assign=True)
    require(all(t.device.type == 'cpu' and not t.is_meta and t.dtype == torch.float32 for t in model.state_dict().values()),
        'decoder has uninitialized/non-CPU/non-FP32 tensors')
    model.requires_grad_(False).eval()


def decode(state, generated, options):
    import torch
    require(not options, 'decoder does not accept sampling/geometry/precision overrides')
    p = state['spec']
    require(isinstance(generated, dict) and set(generated) >= {'video_latents', 'audio_latents'}, 'named original video/audio latents required')
    video, audio = generated['video_latents'], generated['audio_latents']
    for tensor, shape in ((video, p['geometry']['video_latent_shape']), (audio, p['geometry']['audio_latent_shape'])):
        require(type(tensor) is torch.Tensor and tensor.device.type == 'cpu' and tensor.dtype == torch.float32
            and list(tensor.shape) == shape and not tensor.requires_grad and bool(torch.isfinite(tensor).all()),
            'finite exact CPU FP32 native latent layout required')
    state['api'].lock.validate()
    # Native methods own mean/std, overlap, tiling and the audio clamp.
    with torch.no_grad():
        pixels = state['video_codec'].decode_video(video, dtype=torch.float32)
        waveform = state['audio_codec'].decode_audio(audio, dtype=torch.float32)
    require(list(pixels.shape) == [1, 3, *p['geometry']['video_shape'][:3]] and bool(torch.isfinite(pixels).all()),
            'actual native video decoder returned wrong geometry/nonfinite output')
    require(list(waveform.shape) == [1, 2, p['geometry']['audio_shape'][0]], 'actual native stereo audio geometry changed')
    frames = (pixels[0].permute(1, 2, 3, 0).clamp(0, 1) * 255).round().to(torch.uint8)
    samples = waveform[0].t().contiguous()
    require(bool(torch.isfinite(samples).all()) and bool((samples.abs() <= 1).all()), 'native audio samples are nonfinite/out of range')
    state['api'].lock.validate()
    return dict(video=frames, audio=dict(samples=samples, sample_rate=32000),
        decoder=dict(schema_version=SCHEMA, video_fps=p['geometry']['fps'], latent_clamp=False,
            normalization='ORIGINAL_DECODE_VIDEO_AND_DECODE_AUDIO_MEAN_STD_ONCE', duration_policy='NO_CROP_NO_RESAMPLE'))


def build_decoder(parameters):
    from .h3_cached_assets import read_json, checked_pin
    from .single_device import admit_fp32
    from .checkpoint import rng_state, restore_rng
    import torch
    require(isinstance(parameters, dict) and set(parameters) == {'manifest'}, 'explicit pinned decoder manifest required')
    pin = parameters['manifest']; path = checked_pin(pin)
    p = validate_manifest(read_json(path))
    admit_fp32({'device': 'cpu'}, require=require)
    require(str(torch.__version__).split('+')[0] == '2.10.0', 'validated original codec Torch2.10 runtime required')
    before = rng_state(device='cpu')
    try:
        api = source_api(p['source'])
        models, original = meta_decoders(p, api)
        sizes = _admit_geometry(p, models)
        memory = admit_host_memory(sizes['required_bytes'], p['budget']['host_reserve_bytes'])
        for role, model in models.items():
            _load_weights(model, p['weights'][role], original[role], p['storage_dtype'])
        checked_pin(pin); api.lock.validate()
        state = dict(video_codec=models['video'], audio_codec=models['audio'], spec=p, api=api)
        return ModelComponents(state=state, decode=lambda generated, options: decode(state, generated, options),
            metadata=dict(schema_version=SCHEMA, purpose='DECODE_ONLY_NOT_STUDENT_OR_TRAINING', precision=PRECISION,
                evidence_kind=p['evidence_kind'], source=deepcopy(p['source']), codec_manifest_sha256=file_sha256(path),
                runtime=dict(torch=str(torch.__version__), torchvision=importlib.metadata.version('torchvision')),
                decoded_geometry=deepcopy(p['geometry']), capacity={**sizes, **memory},
                quality_validation='NOT_RUN')).validate()
    finally:
        restore_rng(before)


def build_fl2va_decoder(parameters):
    """Explicit raw FL2VA import; the canonical v1 factory stays unchanged."""
    from .h3_av_weight_import import build_decoder as build_imported_decoder
    return build_imported_decoder(parameters)
