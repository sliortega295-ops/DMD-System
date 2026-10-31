"""Opt-in W1 Generator weight initialization, never full-state continuation.

The immutable native final-trainable receipt owns the OLD training provenance.
The existing W1 factory/publisher owns the CURRENT inference identity. No native
worker, checkpoint restore, optimizer, Teacher, EMA or training data is entered.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

SCHEMA = 'distillschedule.wan_generator_weights_initialization.v1'
KIND = 'GENERATOR_WEIGHTS_INITIALIZATION_ONLY'
FIXTURE = 'FULL_ORIGINAL_TINY_CPU_FIXTURE_NOT_PRETRAINED_OR_QUALITY'
REAL = 'LOCAL_PRETRAINED_NATIVE_FEATURES_UNVERIFIED_QUALITY'
ORIGIN_CONTRACT = 'distillschedule.original_wan_rank0_final_trainables.v1'
PYRAMID_SEED = 20260819
NATIVE_KEYS = 'EXACT_NATIVE_STATE_KEYS'
WAN_KEYS = 'ORIGINAL_WAN_FROM_PRETRAINED_KEY_REMAP_V1'


def require(ok, message):
    if not ok:
        raise ValueError('Wan Generator initialization: ' + message)


def _sha(path):
    from distillschedule.workloads.best_known.registry import file_sha256
    return file_sha256(path)


def source_identity():
    """Own this importer as well as the existing inference implementation."""
    return dict(importer_sha256=_sha(Path(__file__)),
                inference_factory_sha256=_sha(Path(__file__).with_name('wan_native_inference.py')))


def read_request(path):
    from .helios_public_assets import read_json
    path = Path(path)
    require(path.is_file() and path.resolve() == path.absolute()
            and path.stat().st_size <= 4*2**20, 'bounded canonical local request required')
    return read_json(path)


def _pin(value, *, maximum=64*2**30):
    from .helios_public_assets import checked_pin
    require(isinstance(value, dict) and set(value) == {'path', 'bytes', 'sha256'}
            and type(value['bytes']) is int and 0 < value['bytes'] <= maximum
            and isinstance(value['sha256'], str) and len(value['sha256']) == 64
            and set(value['sha256']) <= set('0123456789abcdef'), 'invalid/bounded file pin')
    return checked_pin(value)


def _portable_pin(value):
    return dict(name=Path(value['path']).name, bytes=value['bytes'], sha256=value['sha256'])


def _versions(value):
    return (isinstance(value, dict) and set(value) == {'generator', 'critic', 'ema'}
            and all(type(v) is int and v >= 0 for v in value.values())
            and value['critic'] == value['ema'] == 5*value['generator'])


def validate_request(value):
    """Authenticate bounded files/real producer metadata; no Torch import/load.

    Header, shape, dtype, complete base and finite-value admission still follows
    before any live tensor copy. This function alone is NOT export admission.
    """
    from .helios_public_assets import read_json
    require(isinstance(value, dict) and set(value) == {'schema_version', 'kind', 'source',
        'parameters', 'base', 'trainables', 'codec', 'origin'} and value['schema_version'] == SCHEMA
        and value['kind'] == KIND, 'explicit initialization-only schema required; not resume')
    request = deepcopy(value)
    require(request['source'] == source_identity(), 'CURRENT importer/inference source changed')
    p, base, spec, codec, origin = (request[k] for k in ('parameters', 'base', 'trainables', 'codec', 'origin'))
    require(isinstance(p, dict) and 'device' not in p and p.get('evidence_kind') in (FIXTURE, REAL),
            'explicit CPU initialization/native inference evidence required')
    require(isinstance(base, dict) and set(base) == {'config', 'weights', 'indexes', 'pyramid', 'key_policy'}
        and isinstance(base['weights'], list) and 0 < len(base['weights']) <= 32
        and isinstance(base['indexes'], list) and len(base['indexes']) <= 1,
        'complete bounded frozen-base inventory required')
    from distillschedule.workloads.best_known.registry import HELIOS_SHA
    require(base['key_policy'] in ({'kind': NATIVE_KEYS}, {'kind': WAN_KEYS, 'source_sha256': HELIOS_SHA}),
            'explicit exact native keys or source-bound original Wan loader key policy required')
    require(base['pyramid'] in ({'kind': 'COMPLETE_SAFE_BASE'},
        {'kind': 'SOURCE_SEEDED_WAN_MISSING_PATCHES', 'seed': PYRAMID_SEED}),
        'only complete base or original Wan missing-pyramid policy allowed')
    require(isinstance(spec, dict) and set(spec) == {'payload', 'parameter_names'}
        and isinstance(spec['parameter_names'], list) and 0 < len(spec['parameter_names']) <= 20000
        and all(isinstance(n, str) and n for n in spec['parameter_names'])
        and len(set(spec['parameter_names'])) == len(spec['parameter_names']), 'exact ordered trainable names required')
    require(isinstance(codec, dict) and set(codec) == {'config', 'weights', 'provenance'}
        and isinstance(codec['weights'], list) and 0 < len(codec['weights']) <= 32,
        'explicit existing native codec policy required')
    require(isinstance(origin, dict) and set(origin) == {'contract', 'receipt', 'source_head', 'source_fingerprint'}
        and origin['contract'] == ORIGIN_CONTRACT
        and isinstance(origin['source_head'], str) and len(origin['source_head']) == 40
        and set(origin['source_head']) <= set('0123456789abcdef')
        and isinstance(origin['source_fingerprint'], str) and len(origin['source_fingerprint']) == 64
        and set(origin['source_fingerprint']) <= set('0123456789abcdef'),
        'explicit original producer contract/source/receipt pins required')
    limit = p.get('budget', {}).get('max_loaded_bytes')
    require(type(limit) is int and 0 < limit <= 64*2**30, 'explicit bounded CPU load budget required')
    pins = [base['config'], *base['weights'], *base['indexes'], spec['payload'],
            codec['config'], *codec['weights']]
    require(all(isinstance(row, dict) and type(row.get('bytes')) is int for row in pins)
            and sum(row['bytes'] for row in pins) <= limit, 'file inventory exceeds CPU load budget')
    _pin(base['config'], maximum=2**20); _pin(codec['config'], maximum=2**20)
    for row in [*base['weights'], *base['indexes'], spec['payload'], *codec['weights']]:
        _pin(row, maximum=limit)
    receipt = read_json(_pin(origin['receipt'], maximum=4*2**20))
    require(receipt.get('status') == 'COMPLETED_ORIGINAL_WAN_WORK' and receipt.get('finite') is True
        and type(receipt.get('rank')) is int and receipt['rank'] == 0
        and receipt.get('identity', {}).get('source_head') == origin['source_head']
        and receipt['identity'].get('source_fingerprint') == origin['source_fingerprint'],
        'original complete finite rank0/source receipt required')
    evidence = receipt.get('state_evidence', {}).get('generator', {})
    require(evidence.get('status') == 'PASS' and type(evidence.get('writer_rank')) is int
        and evidence['writer_rank'] == 0 and type(evidence.get('tensor_count')) is int
        and evidence['tensor_count'] == len(spec['parameter_names'])
        and all(evidence.get(k) == spec['payload'][k] for k in ('path', 'bytes', 'sha256')),
        'payload is not the receipt-attested CURRENT Generator trainables')
    cycles, initial, final = (receipt.get(k) for k in ('cycles', 'initial_versions', 'final_versions'))
    require(isinstance(cycles, list) and cycles and _versions(initial) and _versions(final)
        and all(isinstance(row, dict) and type(row.get('cycle')) is int and row['cycle'] > 0 for row in cycles),
        'original native cycle/version evidence required')
    end = cycles[-1]['cycle']
    require([row['cycle'] for row in cycles] == list(range(end-len(cycles)+1, end+1))
        and end >= len(cycles)
        and all(row.get('versions') == {role: initial[role]+i*ratio for role, ratio in
            (('generator', 1), ('critic', 5), ('ema', 5))} for i, row in enumerate(cycles, 1))
        and cycles[-1]['versions'] == final, 'original progress/versions disagree; never infer completion')
    require(receipt.get('task') == p.get('task')
        and all(isinstance(receipt.get(k), str) and receipt[k].strip()
                for k in ('quality_validation', 'numerical_comparison')),
        'original task/quality/numerical evidence changed')
    if p['evidence_kind'] != FIXTURE:
        from distillschedule.workloads.best_known.registry import fingerprint
        require(receipt['identity'].get('task_sha256') == fingerprint(receipt['task']),
                'original task digest changed')
    model_pins = [base['config'], *base['weights'], *base['indexes']]
    names = [Path(row['path']).name for row in model_pins]
    require(len(set(names)) == len(names)
        and {name: row['sha256'] for name, row in zip(names, model_pins)} ==
            receipt['identity'].get('assets', {}).get('model_files'), 'original frozen-base file inventory changed')
    provenance = dict(schema_version=SCHEMA, kind=KIND, importer_source=deepcopy(request['source']),
        origin_contract=origin['contract'], origin_source_head=origin['source_head'],
        origin_source_fingerprint=origin['source_fingerprint'], origin_receipt=_portable_pin(origin['receipt']),
        origin_cycle=end, origin_initial_versions=deepcopy(initial), origin_versions=deepcopy(final),
        selected_role='generator', selected_writer_rank=0,
        trainables=_portable_pin(spec['payload']), base=[_portable_pin(row) for row in model_pins],
        frozen_pyramid_policy=deepcopy(base['pyramid']), base_key_policy=deepcopy(base['key_policy']),
        origin_numerical_comparison=receipt['numerical_comparison'],
        origin_quality_validation=receipt['quality_validation'],
        origin_input_scope=receipt.get('measurement_scope', 'UNVERIFIED_SOURCE_SHAPED_INPUTS'),
        training_cycles_executed=0, training_resume='NOT_SUPPORTED_INITIALIZATION_ONLY')
    return request, provenance


def stage_trainables(model, spec):
    """Validate the whole exact FP32 payload without changing any Parameter."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file
    path = _pin(spec['payload'])
    selected = {name: p for name, p in model.named_parameters() if p.requires_grad}
    require(list(selected) == spec['parameter_names'], 'ordered Generator trainable inventory changed')
    with safe_open(path, framework='pt', device='cpu') as handle:
        require((handle.metadata() or {}).get('component') == 'generator'
            and set(handle.keys()) == set(selected), 'exact Generator component/tensor keys required; never EMA')
        for name, parameter in selected.items():
            view = handle.get_slice(name)
            require(view.get_dtype() == 'F32' and tuple(view.get_shape()) == tuple(parameter.shape)
                and parameter.dtype == torch.float32, 'Generator trainable shape/FP32 dtype changed: '+name)
    # Own the admitted bytes: CPU safetensors may otherwise retain a mutable
    # input mmap after its hash check. Never let later file writes change G.
    values = {name: t.clone() for name, t in load_file(path, device='cpu').items()}
    require(all(type(t) is torch.Tensor and t.layout == torch.strided and t.dtype == torch.float32
        and bool(t.isfinite().all()) for t in values.values()), 'nonfinite/unsupported Generator trainable values')
    _pin(spec['payload'])
    return values


def apply_trainables(model, values):
    """Copy only already-admitted trainables, never optimizer/clocks/RNG/EMA."""
    import torch
    selected = {name: p for name, p in model.named_parameters() if p.requires_grad}
    require(selected.keys() == values.keys() and all(p.shape == values[n].shape
        and p.dtype == values[n].dtype == torch.float32 and bool(values[n].isfinite().all())
        for n, p in selected.items()), 'staged trainable inventory changed before copy')
    with torch.no_grad():
        for name, parameter in selected.items():
            parameter.copy_(values[name])


def _base_values(model, request, *, native_mixed=False, provenance=None):
    """Stage complete base before PEFT; only explicit original loader alias."""
    import inspect
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file
    from .helios_public_assets import read_json
    from .wan_native_inference import NON_CONSTRUCTOR_CONFIG_FIELDS
    from runtime.critic_pool_helios import TRANSFORMER_ADDITIONAL_KWARGS
    spec, p = request['base'], request['parameters']
    config = read_json(_pin(spec['config'], maximum=2**20))
    signature = {k: v for k, v in inspect.signature(type(model).__init__).parameters.items() if k != 'self'}
    require(isinstance(config, dict) and all(k.startswith('_') or k in signature
        or k in NON_CONSTRUCTOR_CONFIG_FIELDS for k in config),
            'unknown pinned base constructor field')
    effective = {k: config.get(k, v.default) for k, v in signature.items()}
    # Reproduce the original from_pretrained explicit native overrides; all
    # remaining omitted defaults come from the exact admitted original class.
    effective.update({k: v for k, v in TRANSFORMER_ADDITIONAL_KWARGS.items() if k in signature})
    from .helios_inference import _json as plain
    require(not any(v is inspect.Parameter.empty for v in effective.values())
        and plain(effective) == p['architecture'], 'pinned base config/effective original constructor changed')
    expected, shapes, dtypes, owners = model.state_dict(), {}, {}, {}
    policy = spec['key_policy']
    from distillschedule.workloads.best_known.registry import HELIOS_SHA
    require(policy in ({'kind': NATIVE_KEYS}, {'kind': WAN_KEYS, 'source_sha256': HELIOS_SHA}),
            'unadmitted base key policy')
    def base_key(name):
        # Exact original from_pretrained compatibility at the source-bound
        # transformer: scale_shift_table only. Never touch trainable names.
        return 'norm_out.scale_shift_table' if policy['kind'] == WAN_KEYS and name == 'scale_shift_table' else name
    paths = []
    for row in spec['weights']:
        path = _pin(row); paths.append(path)
        with safe_open(path, framework='pt', device='cpu') as handle:
            label = (handle.metadata() or {}).get('evidence_kind')
            require(label in (None, p['evidence_kind']), 'base evidence label disagrees with request')
            for name in handle.keys():
                view = handle.get_slice(name)
                key = base_key(name)
                require(key not in shapes and key in expected and view.get_dtype() in ('F32', 'BF16')
                    and tuple(view.get_shape()) == tuple(expected[key].shape), 'base duplicate/alias/key/shape/dtype changed: '+name)
                shapes[key], dtypes[name] = tuple(view.get_shape()), view.get_dtype()
                owners[name] = path.name
    require(policy['kind'] == NATIVE_KEYS or 'scale_shift_table' in owners
            and 'norm_out.scale_shift_table' not in owners, 'original raw Wan alias policy requires its exact old key')
    missing = set(expected)-set(shapes)
    allowed = {f'{name}.{key}' for name in ('patch_short', 'patch_mid', 'patch_long') for key in ('weight', 'bias')}
    require(missing == (set() if spec['pyramid']['kind'] == 'COMPLETE_SAFE_BASE' else allowed),
            'complete raw base required; only original three missing pyramid patches are allowed')
    for row in spec['indexes']:
        index = read_json(_pin(row))
        mapping = index.get('weight_map')
        require(isinstance(mapping, dict) and mapping == owners,
                'base shard index inventory/ownership changed')
    values = {}
    for path in paths:
        values.update({base_key(name): t for name, t in load_file(path, device='cpu').items()})
    require(all(bool(t.isfinite().all()) for t in values.values()), 'nonfinite frozen-base weights')
    # This is the original from_pretrained BF16 + keep-FP32 policy, not a
    # permissive cast of malformed LoRA. Every source dtype/hash stays recorded.
    target_dtypes = {name: (torch.float32 if any(k in name for k in model._keep_in_fp32_modules)
                     else torch.bfloat16) if native_mixed else expected[name].dtype for name in values}
    values = {name: t.to(device='cpu', dtype=target_dtypes[name], copy=True) for name, t in values.items()}
    require(all(bool(t.isfinite().all()) for t in values.values()), 'frozen-base precision conversion is nonfinite')
    for row in [spec['config'], *spec['weights'], *spec['indexes']]:
        _pin(row)
    non_constructor = {k: config[k] for k in NON_CONSTRUCTOR_CONFIG_FIELDS if k in config and k not in signature}
    if provenance is not None and non_constructor:
        provenance['base_non_constructor_config'] = plain(non_constructor)
    return values, missing, dtypes


def _initialize_base(model, request, provenance):
    """Materialize an unpublished meta owner only after whole-base admission."""
    import torch
    from runtime.critic_pool_helios import _materialize_missing_pyramid_patches
    require(all(p.device.type == 'meta' and p.dtype == torch.float32 for p in model.parameters()),
            'original FP32 meta constructor required before pretrained mixed-precision loading')
    values, missing, dtypes = _base_values(model, request, native_mixed=True, provenance=provenance)
    result = model.load_state_dict(values, strict=not missing, assign=True)
    require(set(result.missing_keys) == missing and not result.unexpected_keys, 'base assignment layout changed')
    if missing:
        actual = _materialize_missing_pyramid_patches(model, PYRAMID_SEED)
        require(actual == ['patch_short', 'patch_mid', 'patch_long'], 'original pyramid materialization changed')
    state = model.state_dict()
    nonpersistent = {n: t for n, t in model.named_buffers() if n not in state}
    require(set(nonpersistent) == {'rope.'+n for n, _ in model.rope.named_buffers()}
        and not list(model.rope.parameters()), 'unadmitted nonpersistent original base state')
    model.rope = type(model.rope)(rope_dim=model.config.rope_dim, theta=model.config.rope_theta)
    require(all(t.device.type == 'cpu' and bool(t.isfinite().all())
        for t in [*model.parameters(), *model.buffers()]), 'uninitialized/nonfinite base remains')
    restored = model.state_dict()
    require(all(torch.equal(restored[n], t) for n, t in values.items()), 'base changed during construction')
    provenance['base_source_dtypes'] = dtypes


def import_generator_weights(path, output):
    """Public unified CLI endpoint: authenticated initialization -> safe export."""
    request, provenance = validate_request(read_request(path))
    import torch
    from . import wan_native_inference as native
    from .checkpoint import rng_state, restore_rng
    from .inference import publish_export
    from .plugins import load_callable
    from .adapters import ModelComponents
    require(not torch.cuda.is_initialized() and not torch.distributed.is_initialized(),
            'CPU initialization requires no CUDA or distributed owner')
    native._environment('cpu')
    destination = Path(output).expanduser().resolve()
    config_path = destination.with_name(destination.name+'.inference.json')
    require(not destination.exists() and not config_path.exists(), 'new export/config paths required')
    before = rng_state()  # isolate CURRENT constructor RNG, never restore OLD training RNG
    try:
        p, probe, _, size = native._admit(request['parameters'])
        require(3*size <= p['budget']['max_export_bytes_per_rank']
            and 3*size <= p['budget']['max_export_bytes_all_ranks'],
            'single-owner full-tensor export and transient host capacity exceeded')
        # Authenticate the payload header/values before constructing a live G.
        values = stage_trainables(probe, request['trainables'])
        codec = native._load_codec(request['codec'], p['budget'], p['evidence_kind'])[0]
        require(native.codec_parameters(codec, evidence_kind=p['evidence_kind'],
            provenance=request['codec']['provenance']) == p['codec'], 'codec policy/parameters disagree')
        g = native._generator(p, meta=False,
            base_initializer=lambda model: _initialize_base(model, request, provenance))
        apply_trainables(g, values)
        del values
        components = ModelComponents(state=dict(generator=native.CurrentGenerator(g), codec=codec),
            metadata=native._metadata(p)).validate()
        # The whole original receipt/source/file inventory is checked again
        # before publishing; mutations during loading cannot become provenance.
        validate_request(request)
        _, source = load_callable(native.FACTORY)
        manifest = publish_export(components, destination,
            model=dict(factory=native.FACTORY, source=source, parameters=p, metadata=components.metadata),
            identity=dict(model=source, model_metadata=components.metadata, weights_initialization=provenance),
            cycle=0)
        # Existing create-only standalone config; no training source/base paths.
        import os
        import uuid
        temporary = config_path.with_name('.'+config_path.name+'.tmp-'+uuid.uuid4().hex)
        try:
            temporary.write_text(json.dumps(native.standalone_config(manifest, config_path),
                indent=2, sort_keys=True, allow_nan=False)+'\n')
            os.link(temporary, config_path)
        finally:
            temporary.unlink(missing_ok=True)
        return {**manifest, 'inference_config': str(config_path), 'training_cycles_executed': 0,
                'initialization_only': True}
    finally:
        restore_rng(before)
