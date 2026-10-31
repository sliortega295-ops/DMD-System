"""Opt-in original FL2VA codec state import, not an alternate decoder/backend.

The operator supplies pinned codec, converter and config sources plus weights.
Only the associated official converters relabel tensors; original architecture,
decode arithmetic, FP32/storage, finite, strict-load and capacity gates remain.
No upstream source, model weights or experiment-specific identities are bundled.
"""
import ast
from copy import deepcopy
import importlib.metadata
from pathlib import Path
from types import SimpleNamespace

from distillschedule.workloads.best_known.registry import file_sha256, fingerprint
from . import h3_av_codec as codec

SCHEMA = 'distillschedule.original_h3_fl2va_decoder.v1'
FORMAT = 'ORIGINAL_FL2VA_SAFETENSORS'
CONVERTER_FILES = {
    'diffsynth/utils/state_dict_converters/minimax_h3_video_vae.py': '5035fea7e141afca89cf45ac71c0a10251868cba6cd3886edeb8cd214672a7f8',
    'diffsynth/utils/state_dict_converters/minimax_h3_audio_vae.py': '03db95337d3752148b7b51a6de1307f585f9ba877a777712432eb125f098495f',
    'diffsynth/configs/model_configs.py': '009d524def453d1d08d8aa051c582546ddabc27fc2649f7ecd6f403bea886b1d',
}
ASSOCIATIONS = {
    'video': ('24b80900992e2024fab17c991c57da23', 'MiniMaxH3VideoVAE'),
    'audio': ('db383f1c8960837b94059f7722e6cb11', 'MiniMaxH3AudioVAE'),
}
require = codec.require


def validate_manifest(value):
    require(isinstance(value, dict) and set(value) == {'schema_version', 'weight_format',
        'codec_manifest', 'converter_source'} and value['schema_version'] == SCHEMA
        and value['weight_format'] == FORMAT, 'explicit original FL2VA import schema/format required')
    fingerprint(value)
    codec._pin(value['codec_manifest'])
    source = value['converter_source']
    require(isinstance(source, dict) and set(source) == {'root', 'files', 'provenance'}
        and isinstance(source['root'], str) and Path(source['root']).is_absolute()
        and source['files'] == CONVERTER_FILES, 'exact compatible official converter/config source required')
    provenance = source['provenance']
    require(isinstance(provenance, dict) and set(provenance) == {'origin', 'license', 'terms_accepted'}
        and all(isinstance(provenance[k], str) and provenance[k].strip() for k in ('origin', 'license'))
        and provenance['terms_accepted'] is True, 'operator must supply and accept converter source terms')
    return deepcopy(value)


def converter_api(source):
    """Exact original definitions and their official model/config association."""
    from .upstream import PinnedSource
    root = Path(source['root'])
    require(root.resolve() == root and root.is_dir()
        and all((root/name).resolve() == root/name for name in CONVERTER_FILES),
        'canonical existing converter source files required')
    definitions = {f'diffsynth.utils.state_dict_converters.minimax_h3_{role}_vae': [cls+'StateDictConverter']
                   for role, (_, cls) in ASSOCIATIONS.items()}
    lock = PinnedSource(root, CONVERTER_FILES, definitions=definitions)
    tree = ast.parse((root/'diffsynth/configs/model_configs.py').read_bytes())
    series = [n.value for n in tree.body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'minimax_h3_series' for t in n.targets)]
    require(len(series) == 1 and isinstance(series[0], ast.List), 'original H3 model associations missing')
    rows = [ast.literal_eval(n) for n in series[0].elts]
    functions = {}
    for role, (model_hash, cls) in ASSOCIATIONS.items():
        module = f'diffsynth.utils.state_dict_converters.minimax_h3_{role}_vae'
        association = dict(model_hash=model_hash, model_name=f'minimax_h3_{role}_vae',
            model_class=f'diffsynth.models.minimax_h3_{role}_vae.{cls}',
            state_dict_converter=module+'.'+cls+'StateDictConverter')
        require([r for r in rows if r.get('model_hash') == model_hash] == [association],
                'original FL2VA codec/converter association changed')
        functions[role] = getattr(lock.import_module(module), cls+'StateDictConverter')
    lock.validate()
    return SimpleNamespace(functions=functions, lock=lock)


def _raw_layout(role, original, convert):
    layout = {}
    for target, info in original.items():
        candidate, marker = target.removesuffix('.weight'), object()
        mapped = convert({candidate: marker})
        raw = candidate if set(mapped) == {target} and mapped[target] is marker else target
        exact = convert({raw: marker})
        require(set(exact) == {target} and exact[target] is marker and raw not in layout,
                'official converter must be an unambiguous identity-preserving relabel')
        layout[raw] = (target, info)
    if role == 'video':
        # The pinned original decoder appends zeros_like and never owns this
        # raw parameter. Only this official converter's explicit exclusion is admitted.
        shape, dtype = original['decoder.register_tokens.weight']
        require(convert({'decoder.mask_token': object()}) == {}, 'official mask exclusion changed')
        layout['decoder.mask_token'] = (None, ((1, 1, shape[-1]), dtype))
    return layout


def _plan_headers(role, headers, original, selected, storage, converters):
    require(role in ASSOCIATIONS and storage in ('float32', 'bfloat16_to_float32'), 'unknown codec role/storage')
    converters.lock.validate()
    convert = converters.functions[role]
    full, needed = _raw_layout(role, original, convert), _raw_layout(role, selected, convert)
    require(set(headers) in (set(full), set(needed)), 'complete full-native or decoder-only raw FL2VA state required')
    expected = full if set(headers) == set(full) else needed
    allowed = ('F32',) if storage == 'float32' else ('BF16', 'F32')
    for raw, header in headers.items():
        require(tuple(header['shape']) == expected[raw][1][0] and header['dtype'] in allowed,
                'raw safe header/key/shape/storage mismatch: '+raw)
    markers = {name: object() for name in headers}
    reverse = {id(value): name for name, value in markers.items()}
    mapped = {}
    for target, value in convert(markers).items():
        raw = reverse.get(id(value))
        require(raw is not None and expected[raw][0] == target and target not in mapped,
                'official converter lost/changed/aliased a tensor')
        mapped[target] = raw
    require(set(mapped) in (set(original), set(selected)), 'complete converted native layout required')
    omitted = sorted(set(headers)-set(mapped.values()))
    require(omitted == (['decoder.mask_token'] if role == 'video' else []), 'unrecognized official exclusion')
    return dict(mapping=mapped, summary=dict(role=role, raw_keys=len(headers), converted_keys=len(mapped),
        selected_keys=len(selected), layout='FULL_NATIVE' if set(mapped) == set(original) else 'DECODER_ONLY',
        renamed={raw: target for target, raw in mapped.items() if raw != target}, excluded=omitted,
        mapping_sha256=fingerprint(mapped), storage_dtype=storage, converter_arithmetic='NONE'))


def _inspect_headers(role, model, pins, original, storage, converters, *, verify_payload=True):
    from safetensors import safe_open
    from .h3_cached_assets import checked_pin
    headers, owners, paths = {}, {}, []
    for pin in pins:
        codec._pin(pin)
        path = Path(pin['path'])
        require(path.resolve() == path and path.is_file() and path.suffix == '.safetensors'
            and path.stat().st_size == pin['bytes'] and path not in paths, 'codec weight path/size/shard changed')
        if verify_payload:
            checked_pin(pin)
        paths.append(path)
        with safe_open(path, framework='pt', device='cpu') as handle:
            for raw in handle.keys():
                require(raw not in headers, 'duplicate raw codec weight across shards')
                h = handle.get_slice(raw)
                headers[raw] = dict(shape=h.get_shape(), dtype=h.get_dtype())
                owners[raw] = path
    selected = {n: (tuple(t.shape), t.dtype) for n, t in model.state_dict().items()}
    plan = _plan_headers(role, headers, original, selected, storage, converters)
    return dict(**plan, owners=owners, paths=paths)


def _load_weights(role, model, pins, original, storage, converters):
    import torch
    from safetensors import safe_open
    plan = _inspect_headers(role, model, pins, original, storage, converters)
    selected, state = model.state_dict(), {}
    for path in plan['paths']:
        with safe_open(path, framework='pt', device='cpu') as handle:
            for target, raw in plan['mapping'].items():
                if target in selected and plan['owners'][raw] == path:
                    value = handle.get_tensor(raw).float()
                    require(value.dtype == selected[target].dtype and bool(torch.isfinite(value).all()), 'nonfinite codec weight: '+raw)
                    state[target] = value
    model.load_state_dict(state, strict=True, assign=True)
    require(all(t.device.type == 'cpu' and not t.is_meta and t.dtype == torch.float32
        for t in model.state_dict().values()), 'uninitialized/non-CPU/non-FP32 decoder')
    model.requires_grad_(False).eval()
    converters.lock.validate()
    return plan['summary']


def _bound(parameters):
    from .h3_cached_assets import checked_pin, read_json
    require(isinstance(parameters, dict) and set(parameters) == {'manifest'}, 'explicit pinned FL2VA import manifest required')
    path = checked_pin(parameters['manifest'])
    spec = validate_manifest(read_json(path))
    p = codec.validate_manifest(read_json(checked_pin(spec['codec_manifest'])))
    return path, spec, p, converter_api(spec['converter_source'])


def inspect_headers(parameters):
    """CPU meta/header preflight only, not payload SHA/load/host admission."""
    import torch
    from .checkpoint import rng_state, restore_rng
    from .single_device import admit_fp32
    from .h3_cached_assets import checked_pin
    path, spec, p, converters = _bound(parameters)
    admit_fp32({'device': 'cpu'}, require=require)
    require(str(torch.__version__).split('+')[0] == '2.10.0', 'validated original codec Torch2.10 runtime required')
    before = rng_state(device='cpu')
    try:
        api = codec.source_api(p['source'])
        models, original = codec.meta_decoders(p, api)
        sizes = codec._admit_geometry(p, models)
        rows = [_inspect_headers(role, model, p['weights'][role], original[role], p['storage_dtype'],
            converters, verify_payload=False)['summary'] for role, model in models.items()]
        checked_pin(parameters['manifest']); checked_pin(spec['codec_manifest'])
        api.lock.validate(); converters.lock.validate()
        return dict(status='HEADERS_ONLY_NOT_WEIGHT_HASH_LOAD_OR_DECODE', manifest_sha256=file_sha256(path),
            weight_format=FORMAT, converter_source=deepcopy(spec['converter_source']), roles=rows, capacity=sizes,
            full_weight_payload_sha_verified=False, concrete_model_weights_loaded=False,
            host_admission='NOT_RUN_HEADER_ONLY', quality_validation='NOT_RUN')
    finally:
        restore_rng(before)


def build_decoder(parameters):
    import torch
    from .adapters import ModelComponents
    from .checkpoint import rng_state, restore_rng
    from .h3_cached_assets import checked_pin
    from .single_device import admit_fp32
    path, spec, p, converters = _bound(parameters)
    admit_fp32({'device': 'cpu'}, require=require)
    require(str(torch.__version__).split('+')[0] == '2.10.0', 'validated original codec Torch2.10 runtime required')
    before = rng_state(device='cpu')
    try:
        api = codec.source_api(p['source'])
        models, original = codec.meta_decoders(p, api)
        sizes = codec._admit_geometry(p, models)
        memory = codec.admit_host_memory(sizes['required_bytes'], p['budget']['host_reserve_bytes'])
        rows = [_load_weights(role, model, p['weights'][role], original[role], p['storage_dtype'], converters)
                for role, model in models.items()]
        checked_pin(parameters['manifest']); checked_pin(spec['codec_manifest'])
        api.lock.validate(); converters.lock.validate()
        state = dict(video_codec=models['video'], audio_codec=models['audio'], spec=p, api=api)

        def decode(generated, options):
            converters.lock.validate()
            result = codec.decode(state, generated, options)
            converters.lock.validate()
            return result

        return ModelComponents(state=state, decode=decode, metadata=dict(schema_version=SCHEMA,
            original_decoder_schema=codec.SCHEMA, purpose='DECODE_ONLY_NOT_STUDENT_OR_TRAINING',
            precision=codec.PRECISION, evidence_kind=p['evidence_kind'], source=deepcopy(p['source']),
            codec_manifest_sha256=spec['codec_manifest']['sha256'], import_manifest_sha256=file_sha256(path),
            weight_import=dict(format=FORMAT, converter_source=deepcopy(spec['converter_source']), roles=rows,
                helper_source=dict(path=str(Path(__file__).resolve()), sha256=file_sha256(Path(__file__))),
                codec_source=dict(path=str(Path(codec.__file__).resolve()), sha256=file_sha256(Path(codec.__file__))),
                full_weight_payload_sha_verified=True),
            runtime=dict(torch=str(torch.__version__), torchvision=importlib.metadata.version('torchvision'),
                         CPU_threads=torch.get_num_threads()),
            decoded_geometry=deepcopy(p['geometry']), capacity={**sizes, **memory}, quality_validation='NOT_RUN')).validate()
    finally:
        restore_rng(before)
