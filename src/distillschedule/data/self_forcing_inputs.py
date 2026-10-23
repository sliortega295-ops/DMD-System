"""Exact official Wan prompt embeddings or prompt-only native local T5 intake.

Prepared BLC tensors are never reshaped, normalized, resized or synthesized.
The optional prompt encoder runs the original WanTextEncoder.forward and
T5Encoder layers from the same external source lock, with explicit local-only
weights/tokenizer and only the CUDA device property ported to CPU placement.
"""
from __future__ import annotations

from copy import deepcopy
import inspect
from pathlib import Path

from .manifest import source_identity, validate_source
from .preparation import _publish
from .tensors import tensor_tree

SCHEMA = 'distillschedule.official_self_forcing_inputs.v1'


def validate_row(row, *, shape=None):
    import torch
    from distillschedule.training.self_forcing_source import require
    require(isinstance(row, dict) and set(row) == {'prompt_embeds'}, 'native condition is exactly {prompt_embeds: BLC tensor}')
    value = row['prompt_embeds']
    require(type(value) is torch.Tensor and value.dtype == torch.float32 and value.device.type == 'cpu'
            and value.layout == torch.strided and value.ndim == 3 and all(n > 0 for n in value.shape)
            and not value.requires_grad and bool(value.isfinite().all()), 'finite detached dense CPU FP32 BLC condition required')
    require(shape is None or tuple(value.shape) == tuple(shape), 'condition differs from actual native Wan context geometry')
    return row


def build_local_text_encoder(spec, *, base_dir):
    import torch
    from distillschedule.training.self_forcing_source import original, require
    from distillschedule.training.self_forcing import _asset_file
    require(isinstance(spec, dict) and {'source_root', 'config', 'weights', 'tokenizer', 'text_len'} <= set(spec)
            <= {'source_root', 'config', 'weights', 'tokenizer', 'text_len', 'tokenizer_use_fast'},
            'native local T5 requires source/config/weights/tokenizer/text_len')
    root = Path(base_dir).resolve(strict=True)
    api = original(spec['source_root'])
    config = deepcopy(spec['config'])
    signature = inspect.signature(api.t5.T5Encoder.__init__)
    allowed = {name for name in signature.parameters if name != 'self'}
    required = {name for name, parameter in signature.parameters.items() if name != 'self'
                and parameter.default is inspect.Parameter.empty}
    require(isinstance(config, dict) and required <= set(config) <= allowed
            and type(config['vocab']) is int and config['vocab'] > 0, 'explicit actual native T5Encoder architecture required')
    require(type(spec['text_len']) is int and spec['text_len'] > 0, 'positive native text context length required')
    require(type(spec.get('tokenizer_use_fast', True)) is bool, 'tokenizer_use_fast must be explicit boolean')
    checkpoint = _asset_file(root, spec['weights'])
    tokenizer_root = (root / Path(spec['tokenizer']).expanduser()).resolve(strict=True)
    require(tokenizer_root.is_dir() and tokenizer_root.is_relative_to(root), 'explicit local tokenizer inside encoder bundle required')
    tokenizer_files = sorted(path for path in tokenizer_root.rglob('*') if path.is_file())
    require(tokenizer_files and all(path.resolve().is_relative_to(root) for path in tokenizer_files),
            'tokenizer is empty or a symlink escapes encoder bundle')
    tokens = api.source.import_module('wan.modules.tokenizers')
    cls = type('LocalNativeWanTextEncoder', (api.wrapper.WanTextEncoder,),
               {'device': property(lambda self: next(self.text_encoder.parameters()).device)})
    encoder = cls.__new__(cls); torch.nn.Module.__init__(encoder)
    encoder.text_encoder = api.t5.T5Encoder(**config).eval().requires_grad_(False)
    from distillschedule.training.checkpoint import _validate_layout
    weights = torch.load(checkpoint, map_location='cpu', weights_only=True)
    _validate_layout(encoder.text_encoder.state_dict(), weights, 'native local T5')
    require(all(type(tensor) is torch.Tensor and tensor.layout == torch.strided
                and bool(tensor.isfinite().all()) for tensor in weights.values()),
            'local native T5 requires finite dense state with no implicit precision cast')
    encoder.text_encoder.load_state_dict(weights, strict=True)
    encoder.tokenizer = tokens.HuggingfaceTokenizer(name=str(tokenizer_root), seq_len=spec['text_len'],
        clean='whitespace', local_files_only=True, use_fast=spec.get('tokenizer_use_fast', True))
    require(encoder.tokenizer.vocab_size <= config['vocab'], 'tokenizer vocabulary exceeds local native T5 vocabulary')
    encoder.eval().requires_grad_(False)
    documents = [source_identity(checkpoint)] + [source_identity(path) for path in tokenizer_files]
    documents += [source_identity(api.source.root / name) for name in api.source.files]
    return encoder, documents


def encode_text(text_prompts, spec, *, base_dir):
    """Raw strings -> original local Wan/T5 numerical conditioning, no fetch."""
    import torch
    from distillschedule.training.self_forcing_source import require
    prompts = [text_prompts] if isinstance(text_prompts, str) else text_prompts
    require(isinstance(prompts, list) and prompts and all(isinstance(text, str) for text in prompts),
            'nonempty prompt string batch required')
    encoder, documents = build_local_text_encoder(spec, base_dir=base_dir)
    with torch.no_grad():
        result = encoder(text_prompts=prompts)
    validate_row(result)
    for document in documents:
        validate_source(document)
    return result


def encode_record(record, parameters, *, encoder=None):
    import torch
    from distillschedule.training.self_forcing_source import require
    from distillschedule.training.self_forcing import _asset_file
    root = Path(parameters['source_dir'])
    if parameters['mode'] == 'preencoded':
        path = _asset_file(root, record['embedding'])
        row = {'prompt_embeds': torch.load(path, map_location='cpu', weights_only=True)}
    else:
        prompts = [record['prompt']] if isinstance(record['prompt'], str) else record['prompt']
        require(isinstance(prompts, list) and prompts and all(isinstance(item, str) for item in prompts),
                'prompt must be a string or nonempty string batch')
        if encoder is None:
            encoder, _ = build_local_text_encoder(parameters['encoder'], base_dir=root)
        with torch.no_grad():
            row = encoder(text_prompts=prompts)
    return validate_row(row)


def prepare_inputs(source_manifest, output_dir):
    from distillschedule.training.self_forcing_source import require
    from distillschedule.training.self_forcing import _json, _asset_file
    from distillschedule.workloads.best_known.registry import fingerprint
    path = Path(source_manifest).expanduser().resolve(strict=True)
    source = _json(path)
    require(isinstance(source, dict) and {'schema_version', 'mode', 'records'} <= set(source)
            <= {'schema_version', 'mode', 'records', 'encoder'} and source['schema_version'] == SCHEMA
            and source['mode'] in ('preencoded', 'prompts'), 'unsupported official Self-Forcing input manifest')
    require(('encoder' in source) == (source['mode'] == 'prompts'), 'prompt-only intake requires an explicit actual local native T5 encoder')
    documents = [source_identity(path)]
    encoder = None
    if source['mode'] == 'prompts':
        encoder, encoder_documents = build_local_text_encoder(source['encoder'], base_dir=path.parent)
        documents += encoder_documents
    records, seen, sources = source['records'], set(), []
    require(isinstance(records, list) and records, 'nonempty native prompt records required')
    for row in records:
        key = 'embedding' if source['mode'] == 'preencoded' else 'prompt'
        require(isinstance(row, dict) and set(row) == {'id', key} and isinstance(row['id'], str)
                and row['id'] and row['id'] not in seen, 'invalid or duplicate native input record')
        seen.add(row['id'])
        media = {'text': documents[0]}
        if key == 'embedding':
            identity = source_identity(_asset_file(path.parent, row[key]))
            documents.append(identity); media['image'] = identity
        sources.append({'id': row['id'], 'media': media})
    parameters = dict(mode=source['mode'], source_dir=str(path.parent))
    if encoder is not None:
        parameters['encoder'] = deepcopy(source['encoder'])
    code = source_identity(Path(__file__).resolve())
    identity = dict(reference='distillschedule.data.self_forcing_inputs:encode_record',
        sha256=code['sha256'], source_path=code['path'], parameters=parameters,
        parameters_sha256=fingerprint(parameters))
    runtime = None
    if encoder is not None:
        import importlib.metadata
        runtime = {name: importlib.metadata.version(name) for name in
                   ('torch', 'transformers', 'tokenizers', 'sentencepiece', 'ftfy')}
    return _publish(output_dir, rows=records,
        provenance=dict(mode='raw', documents=documents, sources=sources, encoder=identity),
        metadata=dict(schema_version=SCHEMA, layout='EXACT_NATIVE_FP32_BLC', conditioning=source['mode'],
                      latent_or_rollout_inputs='NONE', encoder_runtime=runtime),
        produce=lambda row: tensor_tree(encode_record(row, parameters, encoder=encoder)))
