"""Native SD tokens, unchanged pinned latents, or explicit local RGB encoding.

No resize, crop, VAE posterior policy or latent conversion is implicit.
Prepared token batches have B,1,L layout, as in the original DMD2 loaders.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import inspect
import json
from pathlib import Path
import random

from .manifest import source_identity, validate_source
from .preparation import _publish
from .tensors import tensor_tree
from distillschedule.workloads.best_known.registry import fingerprint

SCHEMA = 'distillschedule.sd_inputs.v1'


def _require(condition, message):
    if not condition:
        raise ValueError('SD input: ' + message)


def validate_row(row, *, sdxl, model=None, images=False):
    import torch
    keys = {'text_input_ids_one', 'text_input_ids_two'} if sdxl else {'text_input_ids_one'}
    _require(isinstance(row, dict) and keys <= set(row) <= keys | {'images'}, 'native token fields changed')
    batch = None
    for name in sorted(keys):
        tokens = row[name]
        _require(isinstance(tokens, torch.Tensor) and tokens.dtype == torch.int64 and tokens.ndim == 3
                and tokens.shape[1] == 1 and tokens.shape[0] > 0 and tokens.shape[2] > 0
                and not tokens.requires_grad and bool((tokens >= 0).all()), 'tokens must be native nonnegative int64 B1L')
        batch = tokens.shape[0] if batch is None else batch
        _require(tokens.shape[0] == batch, 'dual CLIP batch sizes differ')
        if model is not None:
            encoder = model.text_encoder
            if sdxl:
                encoder = encoder.text_encoder_one if name.endswith('one') else encoder.text_encoder_two
            _require(tokens.shape[2] == encoder.config.max_position_embeddings
                    and bool((tokens < encoder.config.vocab_size).all())
                    and tokens.shape[0] == model.args.batch_size, 'tokens differ from actual CLIP/batch geometry')
    _require(not images or 'images' in row, 'native real/denoising VAE latents required')
    if 'images' in row:
        latents = row['images']
        _require(isinstance(latents, torch.Tensor) and latents.dtype == torch.float32 and latents.ndim == 4
                and latents.shape[0] == batch and latents.shape[1] == 4 and not latents.requires_grad
                and bool(torch.isfinite(latents).all()), 'images must be finite native FP32 B4HW VAE latents')
        if model is not None:
            _require(tuple(latents.shape) == (batch, 4, model.args.latent_resolution, model.args.latent_resolution),
                    'latent geometry differs from native model settings')
    return row


@contextmanager
def _encoding_rng_scope():
    """Even local pretrained construction may initialize tensors internally."""
    import numpy as np
    import torch
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def _encoding_policy(value):
    _require(isinstance(value, dict) and set(value) == {
        'resolution', 'latent_shape', 'posterior', 'seed', 'dtype'}, 'explicit VAE encoding policy required')
    for name, size in (('resolution', 2), ('latent_shape', 3)):
        _require(isinstance(value[name], list) and len(value[name]) == size
                 and all(type(n) is int and n > 0 for n in value[name]), 'invalid declared VAE ' + name)
    _require(value['posterior'] in ('mode', 'sample'), 'posterior must explicitly be mode or sample')
    _require(type(value['seed']) is int and 0 <= value['seed'] < 2**63, 'explicit independent encoding seed required')
    _require(value['dtype'] == 'float32', 'offline VAE encoding supports CPU FP32 only')
    return deepcopy(value)


def _local_files(root, folder):
    directory = root / folder
    _require(directory.is_dir(), 'local encoder/tokenizer folder missing: ' + folder)
    files = sorted(p for p in directory.rglob('*') if p.is_file())
    _require(bool(files), 'local encoder/tokenizer files missing: ' + folder)
    _require(all(file.resolve().is_relative_to(root) for file in files), 'encoder/tokenizer symlink escapes bundle')
    return [source_identity(file) for file in files]


def _load_vae(root, policy):
    import math
    import torch
    from diffusers import AutoencoderKL
    with _encoding_rng_scope():
        vae = AutoencoderKL.from_pretrained(root / 'vae', local_files_only=True, torch_dtype=torch.float32)
    vae.eval().requires_grad_(False)
    config = vae.config
    factor = 2 ** (len(config.block_out_channels) - 1)
    height, width = policy['resolution']
    _require(config.in_channels == 3 and config.out_channels == 3 and config.latent_channels == 4,
             'VAE must have native RGB and DMD2 four-channel latent geometry')
    _require(height % factor == 0 and width % factor == 0
             and policy['latent_shape'] == [config.latent_channels, height // factor, width // factor],
             'declared resolution/latent geometry differs from actual native VAE')
    _require(type(config.scaling_factor) in (float, int) and math.isfinite(config.scaling_factor)
             and config.scaling_factor > 0, 'native VAE scaling_factor must be finite and positive')
    _require(all(p.dtype == torch.float32 and bool(torch.isfinite(p).all()) for p in vae.parameters()),
             'native VAE weights must be finite FP32')
    return vae, dict(model_class='diffusers.AutoencoderKL', spatial_scale=factor,
                     latent_channels=config.latent_channels, scaling_factor=config.scaling_factor)


def _pinned_reference(spec, base, *, name):
    from .manifest import sha256
    _require(isinstance(spec, dict) and set(spec) == {'path', 'sha256'}
             and isinstance(spec['path'], str) and bool(spec['path']) and '://' not in spec['path'],
             name + ' requires a local path and SHA256')
    sha256(spec['sha256'])
    identity = source_identity((base / Path(spec['path']).expanduser()).resolve(strict=True))
    _require(identity['sha256'] == spec['sha256'], name + ' SHA256 mismatch')
    return identity


def prepare_inputs(source_manifest, output_dir):
    from transformers import CLIPTokenizer
    path = Path(source_manifest).expanduser().resolve(strict=True)
    document = source_identity(path)
    value = json.loads(path.read_text())
    validate_source(document)
    _require(isinstance(value, dict) and {'schema_version', 'variant', 'model_id', 'records'} <= set(value)
            <= {'schema_version', 'variant', 'model_id', 'records', 'vae_encoding'}
            and value['schema_version'] == SCHEMA and value['variant'] in ('sd15', 'sdxl'), 'unsupported source manifest')
    root = (path.parent / Path(value['model_id']).expanduser()).resolve(strict=True)
    _require(root.is_dir(), 'explicit local tokenizer bundle required')
    sdxl = value['variant'] == 'sdxl'
    tokenizer_sources = {Path(inspect.getfile(cls)) for cls in CLIPTokenizer.__mro__ if cls is not object}
    documents = [document, *[source_identity(file) for file in sorted(tokenizer_sources)]]
    asset_files = {}
    tokenizers = []
    for folder in (('tokenizer', 'tokenizer_2') if sdxl else ('tokenizer',)):
        asset_files[folder] = _local_files(root, folder)
        documents.extend(asset_files[folder])
        tokenizers.append(CLIPTokenizer.from_pretrained(root / folder, local_files_only=True))
    rows = value['records']
    _require(isinstance(rows, list) and bool(rows), 'nonempty prompt records required')
    sources, seen = [], set()
    for row in rows:
        _require(isinstance(row, dict) and {'id', 'prompt'} <= set(row) <= {'id', 'prompt', 'latents', 'images'}
                and isinstance(row['id'], str) and row['id'] and row['id'] not in seen, 'invalid/duplicate prompt record')
        _require(not ('latents' in row and 'images' in row), 'latents and raw images are mutually exclusive')
        seen.add(row['id'])
        prompts = row['prompt'] if isinstance(row['prompt'], list) else [row['prompt']]
        _require(bool(prompts) and all(isinstance(prompt, str) for prompt in prompts), 'prompt must be text or text batch')
        media = {'text': documents[0]}
        if 'latents' in row:
            spec = row['latents']
            identity = _pinned_reference(spec, path.parent, name='native latent')
            documents.append(identity)
            media['image'] = identity
        if 'images' in row:
            _require('vae_encoding' in value, 'raw images require explicit VAE encoding policy')
            _require(isinstance(row['images'], list) and len(row['images']) == len(prompts),
                     'raw image batch must align with nonempty prompt batch')
            for spec in row['images']:
                documents.append(_pinned_reference(spec, path.parent, name='raw image'))
            # Every image in a batch is independently pinned in documents.
            media['image'] = documents[-1]
        sources.append({'id': row['id'], 'media': media})

    encoder_parameters = dict(model_id=str(root), variant=value['variant'], source_dir=str(path.parent))
    vae, vae_metadata = None, None
    if 'vae_encoding' in value:
        _require(any('images' in row for row in rows), 'unused VAE encoding policy')
        policy = _encoding_policy(value['vae_encoding'])
        asset_files['vae'] = _local_files(root, 'vae')
        documents.extend(asset_files['vae'])
        vae, vae_metadata = _load_vae(root, policy)
        from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
        from diffusers.utils.torch_utils import randn_tensor
        native_sources = {Path(inspect.getfile(obj)) for obj in
            ({type(child) for child in vae.modules()} | {DiagonalGaussianDistribution, randn_tensor})}
        documents.extend(source_identity(file) for file in sorted(native_sources))
        import importlib.metadata
        vae_metadata['runtime'] = {name: importlib.metadata.version(name)
                                   for name in ('torch', 'diffusers', 'transformers', 'numpy', 'Pillow')}
        encoder_parameters.update(vae_encoding=policy, vae_provenance=vae_metadata)
    encoder_identity = source_identity(Path(__file__).resolve())
    encoder = dict(reference='distillschedule.data.sd_inputs:encode_record',
        sha256=encoder_identity['sha256'], source_path=encoder_identity['path'],
        parameters=encoder_parameters, parameters_sha256=fingerprint(encoder_parameters))

    def produce(row):
        result = tensor_tree(encode_record(row, encoder_parameters, tokenizers=tokenizers, vae=vae))
        if row['id'] == rows[-1]['id']:
            # Check the inventory as well as individual hashes: adding a new
            # shard or repointing a symlink must not publish unpinned assets.
            _require(all(_local_files(root, folder) == pins for folder, pins in asset_files.items()),
                     'local encoder/tokenizer files changed during preparation')
        return result

    return _publish(output_dir, rows=[{**row, 'metadata': {'prompt': row['prompt']}} for row in rows],
        provenance=dict(mode='raw', documents=documents, sources=sources, encoder=encoder),
        metadata=dict(schema_version=SCHEMA, variant=value['variant'], layout='NATIVE_B1L_AND_OPTIONAL_B4HW'), produce=produce)


def encode_record(record, parameters, *, tokenizers=None, vae=None):
    """Native CLIP tokens and explicit VAE posterior; old latents are untouched."""
    import torch
    from transformers import CLIPTokenizer
    sdxl = parameters['variant'] == 'sdxl'
    _require(not ('latents' in record and 'images' in record), 'latents and raw images are mutually exclusive')
    if tokenizers is None:
        root = Path(parameters['model_id'])
        tokenizers = [CLIPTokenizer.from_pretrained(root / folder, local_files_only=True)
            for folder in (('tokenizer', 'tokenizer_2') if sdxl else ('tokenizer',))]
    _require(len(tokenizers) == (2 if sdxl else 1), 'missing native SDXL second tokenizer')
    prompts = record['prompt'] if isinstance(record['prompt'], list) else [record['prompt']]
    tokens = {name: tokenizer(prompts, padding='max_length', max_length=tokenizer.model_max_length,
              truncation=True, return_tensors='pt').input_ids.unsqueeze(1)
              for name, tokenizer in zip(('text_input_ids_one', 'text_input_ids_two'), tokenizers)}
    if 'latents' in record:
        spec = record['latents']
        path = (Path(parameters['source_dir']) / Path(spec['path']).expanduser()).resolve(strict=True)
        identity = source_identity(path)
        _require(identity['sha256'] == spec['sha256'], 'latent bytes changed during preparation')
        tokens['images'] = torch.load(path, map_location='cpu', weights_only=True)
        validate_source(identity)
    if 'images' in record:
        import numpy as np
        from PIL import Image
        policy = _encoding_policy(parameters.get('vae_encoding'))
        _require(isinstance(record['images'], list) and len(record['images']) == len(prompts)
                 and len(prompts) > 0, 'raw image batch must align with nonempty prompt batch')
        if vae is None:
            vae, _ = _load_vae(Path(parameters['model_id']), policy)
        pixels = []
        for spec in record['images']:
            identity = _pinned_reference(spec, Path(parameters['source_dir']), name='raw image')
            with Image.open(identity['path']) as image:
                _require(image.mode == 'RGB' and getattr(image, 'n_frames', 1) == 1,
                         'raw images must be single-frame eight-bit RGB, no implicit conversion')
                _require([image.height, image.width] == policy['resolution'],
                         'raw image geometry must exactly match declared resolution; no resize/crop')
                pixels.append(torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1))
            validate_source(identity)
        rgb = torch.stack(pixels).float().div(127.5).sub(1)
        seed = int(fingerprint(dict(seed=policy['seed'], record_id=record['id']))[:16], 16) % 2**63
        generator = torch.Generator(device='cpu').manual_seed(seed)
        with _encoding_rng_scope(), torch.no_grad():
            posterior = vae.encode(rgb).latent_dist
            latents = posterior.mode() if policy['posterior'] == 'mode' else posterior.sample(generator=generator)
            tokens['images'] = latents * vae.config.scaling_factor
        _require(tokens['images'].dtype == torch.float32
                 and list(tokens['images'].shape) == [len(prompts), *policy['latent_shape']],
                 'actual VAE output differs from declared FP32 latent geometry')
    validate_row(tokens, sdxl=sdxl)
    return tokens
