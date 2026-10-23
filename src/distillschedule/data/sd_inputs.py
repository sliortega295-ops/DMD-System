"""Native SD prompt tokens and optional hash-bound, preencoded VAE latents.

No resize, VAE sampling, image synthesis or latent conversion is implicit.
Prepared token batches have B,1,L layout, as in the original DMD2 loaders.
"""
from __future__ import annotations

import json
from pathlib import Path

from .manifest import source_identity, validate_source
from .preparation import _publish
from .tensors import tensor_tree

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


def prepare_inputs(source_manifest, output_dir):
    from transformers import CLIPTokenizer
    path = Path(source_manifest).expanduser().resolve(strict=True)
    value = json.loads(path.read_text())
    _require(isinstance(value, dict) and set(value) == {'schema_version', 'variant', 'model_id', 'records'}
            and value['schema_version'] == SCHEMA and value['variant'] in ('sd15', 'sdxl'), 'unsupported source manifest')
    root = (path.parent / Path(value['model_id']).expanduser()).resolve(strict=True)
    _require(root.is_dir(), 'explicit local tokenizer bundle required')
    sdxl = value['variant'] == 'sdxl'
    documents = [source_identity(path)]
    tokenizers = []
    for folder in (('tokenizer', 'tokenizer_2') if sdxl else ('tokenizer',)):
        files = sorted(p for p in (root / folder).rglob('*') if p.is_file())
        _require(bool(files), 'local tokenizer files missing')
        for file in files:
            _require(file.resolve().is_relative_to(root), 'tokenizer symlink escapes bundle')
            documents.append(source_identity(file))
        tokenizers.append(CLIPTokenizer.from_pretrained(root / folder, local_files_only=True))
    rows = value['records']
    _require(isinstance(rows, list) and bool(rows), 'nonempty prompt records required')
    sources, seen = [], set()
    for row in rows:
        _require(isinstance(row, dict) and {'id', 'prompt'} <= set(row) <= {'id', 'prompt', 'latents'}
                and isinstance(row['id'], str) and row['id'] and row['id'] not in seen, 'invalid/duplicate prompt record')
        seen.add(row['id'])
        prompts = row['prompt'] if isinstance(row['prompt'], list) else [row['prompt']]
        _require(bool(prompts) and all(isinstance(prompt, str) for prompt in prompts), 'prompt must be text or text batch')
        media = {'text': documents[0]}
        if 'latents' in row:
            spec = row['latents']
            _require(isinstance(spec, dict) and set(spec) == {'path', 'sha256'}, 'latents require a path and SHA256')
            identity = source_identity((path.parent / Path(spec['path']).expanduser()).resolve(strict=True))
            _require(identity['sha256'] == spec['sha256'], 'native latent SHA256 mismatch')
            documents.append(identity)
            media['image'] = identity
        sources.append({'id': row['id'], 'media': media})

    from distillschedule.workloads.best_known.registry import fingerprint
    encoder_parameters = dict(model_id=str(root), variant=value['variant'], source_dir=str(path.parent))
    encoder_identity = source_identity(Path(__file__).resolve())
    encoder = dict(reference='distillschedule.data.sd_inputs:encode_record',
        sha256=encoder_identity['sha256'], source_path=encoder_identity['path'],
        parameters=encoder_parameters, parameters_sha256=fingerprint(encoder_parameters))

    def produce(row):
        return tensor_tree(encode_record(row, encoder_parameters, tokenizers=tokenizers))

    return _publish(output_dir, rows=[{**row, 'metadata': {'prompt': row['prompt']}} for row in rows],
        provenance=dict(mode='raw', documents=documents, sources=sources, encoder=encoder),
        metadata=dict(schema_version=SCHEMA, variant=value['variant'], layout='NATIVE_B1L_AND_OPTIONAL_B4HW'), produce=produce)


def encode_record(record, parameters, *, tokenizers=None):
    """Explicit prompt-token encoder; optional latent tensors are unmodified."""
    import torch
    from transformers import CLIPTokenizer
    sdxl = parameters['variant'] == 'sdxl'
    if tokenizers is None:
        root = Path(parameters['model_id'])
        tokenizers = [CLIPTokenizer.from_pretrained(root / folder, local_files_only=True)
            for folder in (('tokenizer', 'tokenizer_2') if sdxl else ('tokenizer',))]
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
    validate_row(tokens, sdxl=sdxl)
    return tokens
