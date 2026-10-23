"""Exact native LTX AV tokens, physical coordinates and real Gemma4 contexts.

Raw image/video/audio/text, prompt-only WITH explicit latent shapes, or complete
hash-bound preencoded rows. Never resize, resample, pad clips, invent context,
guess FPS or reuse an encoding cache. The common prepared dataset is the output.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from .manifest import raw_record, source_identity
from .preparation import _publish
from .tensors import tensor_tree

SCHEMA = 'distillschedule.ltx25_inputs.v1'


def native_positions(source, video_shape, audio_shape, fps, media_architecture):
    import torch
    from distillschedule.training.ltx_source import require
    patch = source.import_module('ltx_core.components.patchifiers')
    types = source.import_module('ltx_core.types')
    video = media_architecture['video_encoder']; audio = media_architecture['audio_encoder']
    scales = types.SpatioTemporalScaleFactors.from_blocks(video['encoder_blocks'], video['patch_size'])
    video_patchifier = patch.VideoLatentPatchifier(patch_size=1)
    audio_api = source.import_module('ltx_core.model.audio_vae.audio_vae')
    audio_patchifier = patch.AudioPatchifier(patch_size=1, sample_rate=audio['sample_rate'],
        hop_length=audio['mel_hop_length'], is_causal=audio['is_causal'],
        audio_latent_downsample_factor=audio_api.LATENT_DOWNSAMPLE_FACTOR)
    require(type(fps) in (int, float) and 0 < fps < float('inf'), 'explicit finite positive FPS required')
    video_position = patch.get_pixel_coords(video_patchifier.get_patch_grid_bounds(types.VideoLatentShape(*video_shape)),
                                            scales, causal_fix=True).float()
    video_position[:, 0] /= fps
    audio_position = audio_patchifier.get_patch_grid_bounds(types.AudioLatentShape(*audio_shape))
    return {'video': video_position, 'audio': audio_position.float()}


def _mask(mask, batch, length):
    import torch
    from distillschedule.training.ltx_source import require
    require(isinstance(mask, torch.Tensor) and mask.device.type == 'cpu' and not mask.requires_grad,
            'text context mask must be an explicit detached CPU tensor')
    if mask.dtype in (torch.bool, torch.int64):
        require(mask.shape == (batch, length) and bool(((mask == 0) | (mask == 1)).all())
                and bool(mask.any(dim=1).all()), 'binary text mask must be BL, 1=valid, with a valid token')
    else:
        require(mask.dtype == torch.float32 and mask.shape == (batch, 1, 1, length)
                and bool(((mask == 0) | (mask == -torch.finfo(torch.float32).max)).all())
                and bool((mask == 0).any(dim=-1).all()), 'floating text mask must be native additive B11L, 0/-finfo.max')


def validate_row(row, *, source, media_architecture, binding, context_dimensions=None, null_context=None):
    import torch
    from distillschedule.training.ltx_source import require
    keys = {'latents', 'conditioning', 'conditioning_mask', 'positions', 'contexts', 'context_mask',
            'null_contexts', 'null_context_mask', 'video_shape', 'audio_shape', 'fps', 'encoder_identity'}
    require(isinstance(row, dict) and row.keys() == keys, 'complete explicit native AV/conditioning/geometry fields required')
    for name, length in (('video_shape', 5), ('audio_shape', 4)):
        shape = row[name]
        require(type(shape) is torch.Tensor and shape.dtype == torch.int64 and shape.shape == (length,)
                and shape.device.type == 'cpu' and bool((shape > 0).all()), 'explicit positive native latent shape required: ' + name)
    vs, au = row['video_shape'].tolist(), row['audio_shape'].tolist()
    require(vs[0] == au[0] and vs[1] == 128 and au[1] == media_architecture['audio_encoder']['z_channels']
            and au[3] == media_architecture['audio_encoder']['mel_bins']
                // (2 ** (len(media_architecture['audio_encoder']['ch_mult']) - 1))
            and au[1] * au[3] == 128, 'native patch geometry must produce actual video/audio width128 tokens')
    fps = row['fps']
    require(type(fps) is torch.Tensor and fps.dtype == torch.float32 and fps.shape == () and fps.device.type == 'cpu'
            and bool(torch.isfinite(fps)) and float(fps) > 0, 'explicit native FPS scalar required')
    expected = {'video': (vs[0], vs[2] * vs[3] * vs[4], 128), 'audio': (au[0], au[2], 128)}
    for field in ('latents', 'conditioning', 'conditioning_mask', 'positions', 'contexts', 'null_contexts'):
        require(isinstance(row[field], dict) and set(row[field]) == {'video', 'audio'}, 'both native AV modalities required: ' + field)
    for field in ('latents', 'conditioning'):
        for name, shape in expected.items():
            value = row[field][name]
            require(type(value) is torch.Tensor and value.dtype == torch.float32 and tuple(value.shape) == shape
                    and value.device.type == 'cpu' and value.layout == torch.strided and not value.requires_grad
                    and bool(torch.isfinite(value).all()), 'finite detached exact native FP32 BNC payload required: ' + field + '/' + name)
    for name, shape in expected.items():
        mask = row['conditioning_mask'][name]
        require(type(mask) is torch.Tensor and mask.dtype == torch.bool and tuple(mask.shape) == shape[:2]
                and mask.device.type == 'cpu', 'clean conditioning mask must be actual native per-token BN boolean')
    require(any(not bool(mask.all()) for mask in row['conditioning_mask'].values()), 'all tokens are clean conditioning; no DMD loss remains')
    positions = native_positions(source, vs, au, float(fps), media_architecture)
    for name, value in positions.items():
        require(type(row['positions'][name]) is torch.Tensor and row['positions'][name].dtype == torch.float32
                and torch.equal(row['positions'][name], value), 'stale/non-native coordinates or FPS: ' + name)
    length = None
    for field in ('contexts', 'null_contexts'):
        for name, value in row[field].items():
            require(type(value) is torch.Tensor and value.dtype == torch.float32 and value.ndim == 3
                    and value.shape[0] == vs[0] and value.shape[1] > 0 and value.shape[2] > 0
                    and value.device.type == 'cpu' and not value.requires_grad and bool(torch.isfinite(value).all()),
                    'actual native projected Gemma contexts required: ' + field + '/' + name)
            length = value.shape[1] if length is None else length
            require(value.shape[1] == length and (context_dimensions is None or value.shape[2] == context_dimensions[name]),
                    'native Gemma/context dimensions disagree with the actual transformer')
    _mask(row['context_mask'], vs[0], length); _mask(row['null_context_mask'], vs[0], length)
    identity = row['encoder_identity']
    require(type(identity) is torch.Tensor and identity.dtype == torch.uint8 and identity.shape == (32,)
            and bytes(identity.tolist()).hex() == binding, 'native source/codec binding changed; stale cache refused')
    if null_context is not None:
        for name in ('video', 'audio'):
            require(torch.equal(row['null_contexts'][name], null_context[name].expand(vs[0], -1, -1)),
                    'unconditional features are not the actual native empty-prompt encoding')
        # Equivalent binary vs additive masks are allowed, never floating 0/1.
        binary = row['null_context_mask']
        binary = binary[:, 0, 0] == 0 if binary.dtype == torch.float32 else binary.bool()
        require(torch.equal(binary, null_context['mask'].bool().expand(vs[0], -1)),
                'native empty-prompt mask changed')
    return row


def encode_record(record, parameters, *, media=None):
    """Common raw-encoder plugin, or direct use by the specialized preparation."""
    import torch
    from distillschedule.training.ltx_media import load_media, media_binding
    from distillschedule.training.ltx_source import original, require
    api = original(parameters['source_root'])
    if media is None:
        media, manifest, _ = load_media(api.source, parameters['assets'])
    else:
        manifest = parameters['asset_manifest']
    binding = media_binding(manifest, api.source)
    mode = parameters.get('mode', 'raw')
    if mode == 'preencoded':
        from distillschedule.training.ltx_media import asset_file
        path = asset_file(parameters['source_dir'], record['features'])
        row = torch.load(path, map_location='cpu', weights_only=True)
    else:
        prompt = record.get('text', record.get('prompt'))
        require(isinstance(prompt, str), 'raw text prompt required; no synthetic condition fallback')
        geometry = record.get('metadata', {}).get('geometry', record.get('geometry'))
        require(isinstance(geometry, dict) and {'fps', 'conditioning'} <= geometry.keys()
                and geometry.keys() <= {'fps', 'conditioning', 'video_shape', 'audio_shape'}, 'explicit native geometry/FPS/conditioning policy required')
        policy = geometry['conditioning']
        require(isinstance(policy, dict) and set(policy) == {'video_prefix_latent_frames', 'audio_prefix_latent_frames'}
                and all(type(n) is int and n >= 0 for n in policy.values()), 'explicit nonnegative native conditioning-prefix lengths required')
        latents = {}
        if mode == 'raw' and 'video' in record:
            latents['video'] = media.encode_video(record['video'])
        if mode == 'raw' and 'audio' in record:
            latents['audio'] = media.encode_audio(record['audio'])
        for name in ('video', 'audio'):
            if name not in latents:
                shape = geometry.get(name + '_shape')
                require(isinstance(shape, list) and all(type(n) is int and n > 0 for n in shape)
                        and len(shape) == (5 if name == 'video' else 4), 'prompt-only/missing media requires explicit native latent shape: ' + name)
                require(policy[name + '_prefix_latent_frames'] == 0 or name == 'video' and 'image' in record,
                        'cannot invent clean conditioning for absent raw media')
                latents[name] = torch.zeros(shape, dtype=torch.float32)
            elif name + '_shape' in geometry:
                require(list(latents[name].shape) == geometry[name + '_shape'], 'encoded media shape differs from explicit geometry')
        require(latents['video'].shape[0] == latents['audio'].shape[0] == 1, 'raw/prompt fixture intake currently requires batch one')
        image_latent = media.encode_video(record['image'].unsqueeze(0)) if 'image' in record else None
        if image_latent is not None:
            require(policy['video_prefix_latent_frames'] >= 1 and image_latent.shape[3:] == latents['video'].shape[3:],
                    'image must be explicit first-frame conditioning at the same native geometry')
        patch = api.patchifiers
        video_shape, audio_shape = list(latents['video'].shape), list(latents['audio'].shape)
        tokens = {'video': patch.VideoLatentPatchifier(1).patchify(latents['video']),
                  'audio': patch.AudioPatchifier(1).patchify(latents['audio'])}
        condition = {k: x.clone() for k, x in tokens.items()}
        if image_latent is not None:
            image_tokens = patch.VideoLatentPatchifier(1).patchify(image_latent)
            condition['video'][:, :image_tokens.shape[1]] = image_tokens
        masks = {}
        for name, shape in (('video', video_shape), ('audio', audio_shape)):
            count = policy[name + '_prefix_latent_frames']
            require(count <= shape[2], 'conditioning prefix exceeds actual native latent duration')
            count *= shape[3] * shape[4] if name == 'video' else 1
            masks[name] = torch.arange(tokens[name].shape[1]).unsqueeze(0) < count
        contexts, empty = media.encode_text([prompt]), media.encode_text([''])
        fps = torch.tensor(geometry['fps'], dtype=torch.float32)
        row = dict(latents=tokens, conditioning=condition, conditioning_mask=masks,
            positions=native_positions(api.source, video_shape, audio_shape, float(fps), media.architecture),
            contexts={'video': contexts['video_context'], 'audio': contexts['audio_context']},
            context_mask=contexts['context_mask'],
            null_contexts={'video': empty['video_context'], 'audio': empty['audio_context']},
            null_context_mask=empty['context_mask'], video_shape=torch.tensor(video_shape, dtype=torch.int64),
            audio_shape=torch.tensor(audio_shape, dtype=torch.int64), fps=fps,
            encoder_identity=torch.tensor(list(bytes.fromhex(binding)), dtype=torch.uint8))
    empty = media.encode_text([''])
    return validate_row(row, source=api.source, media_architecture=media.architecture, binding=binding,
        context_dimensions={'video': media.architecture['text_processor']['video_dim'],
                            'audio': media.architecture['text_processor']['audio_dim']},
        null_context={'video': empty['video_context'], 'audio': empty['audio_context'], 'mask': empty['context_mask']})


def prepare_inputs(source_manifest, output_dir):
    from .decoding import decode_record
    from .manifest import validate_source
    from distillschedule.training.ltx_media import json_object, load_media, asset_file
    from distillschedule.training.ltx_source import original, require
    from distillschedule.workloads.best_known.registry import fingerprint
    path = Path(source_manifest).expanduser().resolve(strict=True)
    value = json_object(path)
    require(set(value) == {'schema_version', 'mode', 'source_root', 'assets', 'records'}
            and value['schema_version'] == SCHEMA and value['mode'] in ('raw', 'prompts', 'preencoded'),
            'unsupported native input manifest; select raw, prompts or preencoded explicitly')
    api = original((path.parent / value['source_root']).resolve(strict=True))
    asset_path = (path.parent / value['assets']).resolve(strict=True)
    media, manifest, _ = load_media(api.source, asset_path)
    documents = [source_identity(path), source_identity(asset_path)]
    documents += [source_identity(asset_file(asset_path.parent, row['weights'])) for row in manifest['media'].values()]
    documents += [source_identity(api.source.root / name) for name in api.source.files]
    rows, sources, seen = [], [], set()
    require(isinstance(value['records'], list) and value['records'], 'nonempty native input records required')
    for row in value['records']:
        require(isinstance(row, dict) and isinstance(row.get('id'), str) and row['id'] and row['id'] not in seen,
                'invalid/duplicate native input record')
        seen.add(row['id'])
        if value['mode'] == 'preencoded':
            require(set(row) == {'id', 'features'}, 'preencoded native record requires id and exact feature file')
            identity = source_identity(asset_file(path.parent, row['features']))
            rows.append(row); sources.append({'id': row['id'], 'media': {'features': identity}})
            documents.append(identity)
        elif value['mode'] == 'prompts':
            require(set(row) == {'id', 'prompt', 'geometry'}, 'prompt-only record requires text and explicit AV latent shapes')
            rows.append(row); sources.append({'id': row['id'], 'media': {'text': documents[0]}})
        else:
            require(set(row) <= {'id', 'image', 'video', 'audio', 'text', 'metadata'} and 'text' in row,
                    'raw native record needs text and explicit media/metadata only')
            resolved, identity = raw_record(row, path.parent)
            rows.append(resolved); sources.append(identity)
    parameters = dict(mode=value['mode'], source_dir=str(path.parent), source_root=str(api.source.root),
                      assets=str(asset_path), asset_manifest=manifest)
    code = source_identity(Path(__file__))
    encoder = dict(reference='distillschedule.data.ltx_inputs:encode_record', sha256=code['sha256'],
        source_path=code['path'], parameters=parameters, parameters_sha256=fingerprint(parameters))
    def produce(row):
        decoded = decode_record(row) if value['mode'] == 'raw' else row
        return tensor_tree(encode_record(decoded, parameters, media=media))
    preencoded = value['mode'] == 'preencoded'
    result = _publish(output_dir, rows=rows, provenance=dict(mode='preencoded' if preencoded else 'raw', documents=documents,
        sources=sources, encoder=None if preencoded else encoder), metadata=dict(schema_version=SCHEMA, mode=value['mode'],
        prediction='flow_epsilon_minus_x0', layout='NATIVE_BNC_AV_WIDTH128',
        text='REAL_LOCAL_GEMMA4_FEATURE_EXTRACTOR_V2_DUAL_CONNECTORS', quality_validation='NOT_RUN'), produce=produce)
    for identity in documents:
        validate_source(identity)
    return result
