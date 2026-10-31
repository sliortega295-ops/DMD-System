"""Native flow-image feature plugin: raw RGB, real prompt-only, or hash-bound rows.

No global CLI switch, fabricated text, synthetic null context, implicit resize,
shape conversion, or approximate cache. Zero latents in prompt-only mode are
explicit geometry/noise TEMPLATES, never conditional features or training images.
"""
from pathlib import Path

from .manifest import source_identity, validate_source, raw_record, resolve_path
from .preparation import _publish
from .tensors import tensor_tree

SCHEMA = 'distillschedule.flow_image_inputs.v1'


def family_api(name):
    # Data plugin dispatch only; each family retains its own native ABI. No
    # planner or training-controller knows these names.
    import importlib
    from distillschedule.training.flow_image import require
    references = {'flux2-klein-base': 'flux', 'qwen-image': 'qwen_image'}
    require(name in references, 'unsupported native image family/variant')
    return importlib.import_module('distillschedule.training.' + references[name])


def _tensor(value, *, dtype, shape=None, device='cpu'):
    import torch
    return (type(value) is torch.Tensor and value.device == torch.device(device) and value.dtype == dtype
            and value.layout == torch.strided and not value.requires_grad
            and (shape is None or tuple(value.shape) == tuple(shape))
            and (not value.is_floating_point() or bool(torch.isfinite(value).all())))


def validate_row(row, *, api, config, media_config, binding, empty, device='cpu'):
    import torch
    from functools import partial
    from torch.utils._pytree import tree_leaves
    from distillschedule.training.flow_image import require
    device = torch.device(device)
    require(device.type in ('cpu', 'cuda') and all(type(value) is torch.Tensor and value.device == device
            for value in tree_leaves(row)), 'every native image feature must be on the owned device')
    tensor = partial(_tensor, device=device)
    shape = row.get('latent_shape') if isinstance(row, dict) else None
    require(tensor(shape, dtype=torch.int64, shape=(4,)) and bool((shape > 0).all()), 'explicit positive native BCHW geometry required')
    b, c, h, w = shape.tolist()
    require(b == 1, 'this admitted T2I route requires batch-one native BCHW geometry')
    require(h % 2 == w % 2 == 0 and c * 4 == config['in_channels'], 'incompatible native VAE/packing geometry')
    require(isinstance(row.get('latents'), dict) and set(row['latents']) == {'image'}
            and tensor(row['latents']['image'], dtype=torch.float32, shape=(b, h//2 * (w//2), c*4)),
            'finite detached FP32 native image BNC payload required')
    for context_key, mask_key in (('context', 'context_mask'), ('null_context', 'null_context_mask')):
        context, mask = row.get(context_key), row.get(mask_key)
        require(tensor(context, dtype=torch.float32) and context.ndim == 3 and context.shape[0] == b
                and context.shape[1] == media_config['text_max_length'] and context.shape[2] == config['joint_attention_dim'],
                'missing/invalid actual native text context or text width')
        require(tensor(mask, dtype=torch.bool, shape=context.shape[:2]) and bool(mask.any(dim=1).all())
                and (context.shape[1] < 2 or bool((mask[:, 1:].int() <= mask[:, :-1].int()).all())),
                'native text mask must be nonempty BL bool, right padded, 1=valid')
    require(torch.equal(row['null_context'], empty[0].expand(b, -1, -1))
            and torch.equal(row['null_context_mask'], empty[1].expand(b, -1)),
            'unconditional context is not the actual native empty-prompt encoding')
    identity = row.get('encoder_identity')
    require(tensor(identity, dtype=torch.uint8, shape=(32,)) and bytes(identity.tolist()).hex() == binding,
            'wrong family/source/config/codec/dtype binding; stale conditions refused')
    metadata = api.geometry(row, shape=shape.tolist(), context=row['context'], null_context=row['null_context'])
    keys = {'latents', 'context', 'context_mask', 'null_context', 'null_context_mask', 'latent_shape', 'encoder_identity'}
    require(set(row) == keys | set(metadata), 'unknown/missing fields; edit/reference images and causal caches are not supported')
    for name, expected in metadata.items():
        require(tensor(row[name], dtype=expected.dtype, shape=expected.shape) and torch.equal(row[name], expected.to(device)),
                'stale/heterogeneous/non-native image or text geometry: ' + name)
    # Klein native text encoding intentionally retains every padded hidden state;
    # the native forward has no arbitrary text-mask input.
    if 'txt_ids' in metadata:
        require(bool(row['context_mask'].all()) and bool(row['null_context_mask'].all()), 'klein native forward does not admit arbitrary text masks')
    return row


def encode_record(record, parameters, *, api=None, media=None, config=None):
    import torch
    from distillschedule.training.flow_image import require, load_media_bundle, json_object
    keys = {'family', 'mode', 'source_dir', 'binding'}
    require(isinstance(parameters, dict) and keys <= parameters.keys()
            and set(parameters) <= keys | {'assets'} and (media is not None or 'assets' in parameters),
            'explicit native encoding source/family/mode/binding required')
    api = family_api(parameters['family']) if api is None else api
    if media is None:
        media, bundle, roots, runtime, binding, media_config, empty, documents = load_media_bundle(parameters['assets'], api)
        config = json_object(roots['student']/'config.json')
    require(media.binding == parameters['binding'], 'native encoding assets/source binding changed')
    mode = parameters['mode']; base = Path(parameters['source_dir'])
    require(mode in ('raw', 'prompts', 'preencoded'), 'unsupported native input encoding mode')
    if mode == 'preencoded':
        spec = record['features']
        require(isinstance(spec, dict) and set(spec) == {'path', 'sha256'}, 'preencoded rows require exact file identity')
        path = (base / spec['path']).resolve(strict=True)
        identity = source_identity(path)
        require(identity['sha256'] == spec['sha256'], 'preencoded feature SHA256 changed')
        try:
            row = torch.load(path, map_location='cpu', weights_only=True)
        except Exception as exc:
            raise ValueError('invalid safe native image features') from exc
    else:
        require(isinstance(record.get('prompt'), str), 'real native prompt required, including an explicit empty string')
        context, mask = media.encode_text([record['prompt']])
        null_context, null_mask = media.encode_text([''])
        if mode == 'raw':
            from PIL import Image
            import numpy as np
            spec = record['image']
            require(isinstance(spec, dict) and set(spec) == {'path', 'sha256'}, 'raw RGB requires path and SHA256')
            path = (base / spec['path']).resolve(strict=True)
            require(source_identity(path)['sha256'] == spec['sha256'], 'raw image SHA256 changed')
            with Image.open(path) as image:
                require(image.mode == 'RGB', 'raw input must be RGB; implicit alpha/color conversion forbidden')
                pixels = torch.from_numpy(np.array(image).copy()).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1
            tokens, shape = media.encode_image(pixels)
        else:
            shape = record.get('latent_shape')
            require(isinstance(shape, list) and len(shape) == 4 and all(type(x) is int and x > 0 for x in shape)
                    and shape[0] == 1 and shape[2] % 2 == shape[3] % 2 == 0 and shape[1] * 4 == config['in_channels'],
                    'prompt-only generation needs explicit native batch-one VAE BCHW geometry')
            b, c, h, w = shape
            tokens = torch.zeros(b, h//2 * (w//2), c*4, dtype=torch.float32)
        row = dict(latents={'image': tokens}, context=context, context_mask=mask,
            null_context=null_context, null_context_mask=null_mask, latent_shape=torch.tensor(shape, dtype=torch.int64),
            encoder_identity=torch.tensor(list(bytes.fromhex(media.binding)), dtype=torch.uint8))
        row.update(api.geometry(row, shape=shape, context=context, null_context=null_context))
    empty = media.encode_text([''])
    return validate_row(row, api=api, config=config, media_config=media.architecture, binding=media.binding, empty=empty)


def export_record_sources(record, parameters, *, media_kind):
    """Shared native record admission, independent of any model-name selector."""
    from distillschedule.training.flow_image import require
    require(isinstance(parameters, dict) and set(parameters) == {'mode', 'source_dir', 'model_metadata'},
            'explicit export mode/source/model metadata required')
    mode = parameters['mode']
    require(mode in ('raw', 'prompts'), 'export native input mode must be raw or prompts; preencoded/training assets refused')
    expected = {'id', 'prompt', media_kind if mode == 'raw' else 'latent_shape'}
    require(isinstance(record, dict) and set(record) == expected and isinstance(record['id'], str)
            and record['id'] and isinstance(record['prompt'], str), 'unknown/missing native export record or conditions')
    _, source = raw_record(dict(id=record['id'], text={'value': record['prompt']}), Path(parameters['source_dir']))
    if mode == 'raw':
        spec = record[media_kind]
        require(isinstance(spec, dict) and set(spec) == {'path', 'sha256'}, 'raw media requires explicit path and SHA256')
        path = resolve_path(spec['path'], Path(parameters['source_dir']))
        identity = source_identity(path, frames=media_kind == 'video' and path.is_dir())
        require(identity['sha256'] == spec['sha256'], 'raw native input SHA256 changed')
        source['media'][media_kind] = identity
    return source['media']


def export_metadata(api, media, parameters, *, schema):
    from copy import deepcopy
    from distillschedule.training.flow_image import require
    metadata = parameters['model_metadata']
    require(metadata['binding'] == media.binding, 'export native encoder binding changed')
    return dict(schema_version=schema, family=api.FAMILY, binding=media.binding,
        evidence_kind=metadata['evidence_kind'], mode=parameters['mode'], prediction='FLOW_EPSILON_MINUS_X0',
        encoder_contract=dict(codec_policy=api.CODEC_POLICY, media_config=deepcopy(media.architecture),
                              runtime=deepcopy(metadata['runtime'])),
        gpu_execution='NOT_RUN', quality_validation='NOT_RUN')


def encode_export_input(state, record, parameters):
    """Use only restored current-native codecs; never reopen an asset bundle."""
    role, media = state['student'], state['media']
    sources = export_record_sources(record, parameters, media_kind='image')
    row = encode_record(record, dict(family=role.api.FAMILY, mode=parameters['mode'],
        source_dir=parameters['source_dir'], binding=media.binding), api=role.api, media=media, config=role.architecture)
    role.validate_feeds(row)
    return dict(features=row, metadata=export_metadata(role.api, media, parameters, schema=SCHEMA), sources=sources)


def prepare_inputs(source_manifest, output_dir):
    from distillschedule.training.flow_image import require, json_object, load_media_bundle
    path = Path(source_manifest).expanduser().resolve(strict=True)
    manifest = json_object(path)
    require(set(manifest) == {'schema_version', 'family', 'assets', 'mode', 'records'}
            and manifest['schema_version'] == SCHEMA and manifest['mode'] in ('raw', 'prompts', 'preencoded'),
            'explicit native image preparation schema/mode required')
    api = family_api(manifest['family'])
    assets = (path.parent / manifest['assets']).resolve(strict=True)
    media, bundle, roots, runtime, binding, media_config, empty, files = load_media_bundle(assets, api)
    config = json_object(roots['student'] / 'config.json'); api.validate_architecture(config, media_config)
    rows = manifest['records']; require(isinstance(rows, list) and rows, 'nonempty native image records required')
    documents = [source_identity(path), *(source_identity(p) for p in files)]
    documents += [source_identity(Path(__file__)), source_identity(Path(api.__file__))]
    # Native source identities are folded into binding; provenance also validates
    # the actual files before publication instead of trusting a version string.
    import inspect
    documents += [source_identity(Path(inspect.getfile(cls))) for cls in api.classes()]
    sources, seen = [], set()
    keys = {'raw': {'id', 'prompt', 'image'}, 'prompts': {'id', 'prompt', 'latent_shape'}, 'preencoded': {'id', 'features'}}
    for row in rows:
        require(isinstance(row, dict) and set(row) == keys[manifest['mode']] and isinstance(row['id'], str)
                and row['id'] and row['id'] not in seen, 'invalid/duplicate native T2I row or unsupported edit/cache fields')
        seen.add(row['id'])
        if manifest['mode'] == 'prompts':
            origin = source_identity(path)
        else:
            spec = row['features'] if manifest['mode'] == 'preencoded' else row['image']
            require(isinstance(spec, dict) and set(spec) == {'path', 'sha256'}, 'explicit file SHA256 required')
            origin = source_identity((path.parent / spec['path']).resolve(strict=True))
            require(origin['sha256'] == spec['sha256'], 'input SHA256 mismatch')
            documents.append(origin)
        media_name = 'features' if manifest['mode'] == 'preencoded' else ('image' if manifest['mode'] == 'raw' else 'text')
        sources.append({'id': row['id'], 'media': {media_name: origin}})
    # Encoding is inference-only and isolated from training noise/RNG ownership.
    import torch
    from distillschedule.workloads.best_known.registry import fingerprint
    preencoded = manifest['mode'] == 'preencoded'
    code = source_identity(Path(__file__))
    parameters = dict(family=api.FAMILY, assets=str(assets), mode=manifest['mode'], source_dir=str(path.parent), binding=binding)
    encoder = dict(reference='distillschedule.data.flow_image_inputs:encode_record', sha256=code['sha256'],
        source_path=code['path'], parameters=parameters, parameters_sha256=fingerprint(parameters))
    with torch.random.fork_rng(devices=[]):
        result = _publish(output_dir, rows=rows, provenance=dict(mode='preencoded' if preencoded else 'raw',
            documents=documents, sources=sources, encoder=None if preencoded else encoder),
            metadata=dict(schema_version=SCHEMA, family=api.FAMILY, binding=binding, evidence_kind=bundle['evidence_kind'],
                mode=manifest['mode'], prediction='FLOW_EPSILON_MINUS_X0', quality_validation='NOT_RUN', gpu_execution='NOT_RUN'),
            produce=lambda row: tensor_tree(encode_record(row, parameters, api=api, media=media, config=config)))
    for identity in documents:
        validate_source(identity)
    return result
