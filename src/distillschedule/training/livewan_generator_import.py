"""Opt-in complete W4 Student weights initialization, never C51 continuation.

The original assembled-payload contract owns OLD training/source/numerical
evidence. The existing public factory and publisher own CURRENT inference.
No checkpoint owner, training input, optimizer, EMA, Teacher or Critic is loaded.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

SCHEMA = 'distillschedule.livewan_generator_weights_initialization.v1'
KIND = 'GENERATOR_WEIGHTS_INITIALIZATION_ONLY'
ORIGIN_CONTRACT = 'distillschedule.original_livewan_assembled_student.v1'
PAYLOAD_SCOPE = 'ACTUAL_TRAINED_PARAMETERS_FOR_PAIRED_NATIVE_GENERATION'
ROPE_POLICY = 'RECONSTRUCTED_DETERMINISTIC_NONTRAINING_POSITIONAL_STATE'


def require(ok, message):
    if not ok:
        raise ValueError('LiveWan Generator initialization: ' + message)


def _sha(path):
    from distillschedule.workloads.best_known.registry import file_sha256
    return file_sha256(path)


def source_identity():
    """Read-only current admission/publication closure, with no Torch import."""
    import distillschedule
    root = Path(distillschedule.__file__).parent
    files = ['training/' + name + '.py' for name in (
        'livewan_native_inference', 'livewan_public', 'wan_native_inference',
        'helios_public_assets', 'helios_inference', 'wan_inference', 'wan21_codec',
        'inference', 'adapters', 'checkpoint', 'distributed', 'plugins', 'host_memory',
        'config', 'session')]
    files += ['reference/livewan_wan21/__init__.py',
        'reference/livewan_wan21/source_manifest.json', 'open_workload/torch_backend.py',
        'search/resources.py', 'workloads/best_known/registry.py']
    return dict(importer_sha256=_sha(Path(__file__)),
                helpers={name: _sha(root / name) for name in files})


def read_request(path):
    from .helios_public_assets import read_json
    path = Path(path)
    require(path.is_file() and path.resolve() == path.absolute()
            and path.stat().st_size <= 4*2**20, 'bounded canonical local request required')
    return read_json(path)


def _hex(value, length):
    return isinstance(value, str) and len(value) == length and set(value) <= set('0123456789abcdef')


def _pin(value, *, maximum):
    from .helios_public_assets import checked_pin
    require(type(value) is dict and set(value) == {'path', 'bytes', 'sha256'}
        and type(value['bytes']) is int and 0 < value['bytes'] <= maximum
        and _hex(value['sha256'], 64), 'complete bounded file pin required')
    return checked_pin(value)


def _reference(value):
    """Historical reference only: never read large original checkpoint shards."""
    require(type(value) is dict and set(value) == {'path', 'sha256'}
        and isinstance(value['path'], str) and Path(value['path']).is_absolute()
        and '://' not in value['path'] and _hex(value['sha256'], 64),
        'exact historical path/hash reference required')


def _summary(value, assembly):
    cycle, head = assembly['cycle'], assembly['source_head']
    require(type(value) is dict and value.get('schema_version') == 'distillschedule.livewan_trajectory_summary.v1'
        and value.get('status') == 'PASS' and value.get('errors') == []
        and value.get('head') == value.get('expected_head') == head
        and type(value.get('heads')) is dict and value['heads']
        and all(h == head for h in value['heads'].values())
        and type(value.get('cycles')) is int and value['cycles'] == cycle
        and type(value.get('final_versions')) is dict
        and value['final_versions'] == dict(generator=cycle, critic=5*cycle, ema=0)
        and all(type(v) is int for v in value['final_versions'].values()),
        'original complete execution summary/source/G-C-EMA counts required')
    count = value.get('rank_cycle_receipts')
    require(type(count) is int and count > 0 and count == value.get('expected_rank_cycle_receipts')
        and count % cycle == 0 and count // cycle <= 1024
        and isinstance(value.get('run_root'), str) and Path(value['run_root']).is_absolute(),
        'original rank-cycle census/run root required')
    return count // cycle


def validate_request(value):
    """Authenticate ORIGINAL receipts and bounded files, not a resume manifest.

    This is stdlib-only admission. Exact model headers, values and capacity
    still have to pass before any concrete Student mutation or publication.
    """
    from .helios_public_assets import read_json
    require(type(value) is dict and set(value) == {'schema_version', 'kind', 'source',
        'parameters', 'student', 'codec', 'origin'} and value['schema_version'] == SCHEMA
        and value['kind'] == KIND, 'explicit versioned initialization-only schema required; not resume')
    request = deepcopy(value)
    require(request['source'] == source_identity(), 'CURRENT importer/inference closure changed')
    p, spec, codec, origin = (request[k] for k in ('parameters', 'student', 'codec', 'origin'))
    require(type(p) is dict and 'device' not in p and p.get('evidence_kind') in (
        'NATIVE_C51_CURRENT_STUDENT_UNVERIFIED_QUALITY',
        'FULL_ORIGINAL_TINY_CPU_FIXTURE_NOT_PRETRAINED_OR_QUALITY'),
        'explicit existing native CPU inference parameters required')
    require(type(p.get('budget')) is dict, 'explicit bounded inference capacity policy required')
    limit = p['budget'].get('max_loaded_bytes')
    require(type(limit) is int and 0 < limit <= 256*2**30, 'bounded CPU initialization load budget required')
    require(type(spec) is dict and set(spec) == {'payload'}, 'one complete original Student payload required')
    require(type(codec) is dict and set(codec) == {'config', 'weights', 'provenance'}
        and isinstance(codec['weights'], list) and 0 < len(codec['weights']) <= 32,
        'explicit existing original FP32 safe codec policy required')
    fields = {'assembly', 'selected_summary', 'reference_summary', 'numerical_comparison', 'model_source'}
    require(type(origin) is dict and set(origin) == fields | {'contract'}
        and origin['contract'] == ORIGIN_CONTRACT, 'explicit original assembled-Student contract required')
    inventory = [spec['payload'], codec['config'], *codec['weights'], *(origin[k] for k in sorted(fields))]
    require(all(type(pin) is dict and type(pin.get('bytes')) is int for pin in inventory)
        and sum(pin['bytes'] for pin in inventory) <= limit, 'complete source-file inventory exceeds load budget')
    _pin(spec['payload'], maximum=limit)
    _pin(codec['config'], maximum=2**20)
    for pin in codec['weights']:
        _pin(pin, maximum=limit)
    documents = {k: read_json(_pin(origin[k], maximum=4*2**20)) for k in fields - {'model_source'}}
    original_model = _pin(origin['model_source'], maximum=2**20)
    from distillschedule.reference import livewan_wan21 as provider
    provider.source_identity()  # authenticates source_manifest and every numerical/license byte
    packaged_model = Path(provider.__file__).parent / 'wan/modules/model.py'
    require(_sha(original_model) == _sha(packaged_model),
            'entire original WanModel/rope_params source must equal the admitted packaged original')
    assembly = documents['assembly']
    require(type(assembly) is dict and assembly.get('status') == 'ASSEMBLED_CPU_WEIGHTS_NOT_GENERATION_QUALITY'
        and all(assembly.get(k) == spec['payload'][k] for k in ('path', 'bytes', 'sha256'))
        and _hex(assembly.get('source_head'), 40)
        and type(assembly.get('cycle')) is int and 0 < assembly['cycle'] <= 1_000_000
        and type(assembly.get('numel')) is int and assembly['numel'] > 0
        and type(assembly.get('parameter_tensors')) is int and 0 < assembly['parameter_tensors'] <= 20000
        and assembly.get('replicas_averaged') is False and assembly.get('missing_parameters_replaced') is False
        and assembly.get('CUDA_initialized') is False
        and assembly.get('parameter_namespace') == 'ORIGINAL_NATIVE_CausalBlock_AND_CausalHead_CANONICALIZER',
        'unaltered original complete CPU assembly receipt required; never EMA/LoRA/merged/base completion')
    _reference(assembly.get('input_binding'))
    selected, reference = documents['selected_summary'], documents['reference_summary']
    world = _summary(selected, assembly)
    require(_summary(reference, assembly) == world, 'original paired execution census changed')
    refs = assembly.get('checkpoint_sources')
    require(isinstance(refs, list) and len(refs) == world, 'complete historical rank checkpoint references required')
    for rank, ref in enumerate(refs):
        _reference(ref)
        expected = Path(selected['run_root']) / 'training_checkpoints' / ('cycle_' + str(assembly['cycle'])) / f'rank_{rank:02d}.pt'
        require(ref['path'] == str(expected), 'historical selected run/rank checkpoint inventory changed')
    comparison = documents['numerical_comparison']
    require(type(comparison) is dict and comparison.get('source_head') == assembly['source_head']
        and comparison.get('cycles') == assembly['cycle']
        and comparison.get('rank_cycle_receipts_per_arm') == selected['rank_cycle_receipts']
        and comparison.get('thresholds_changed') is False
        and all(isinstance(comparison.get(k), str) and comparison[k].strip() for k in ('status', 'state_scope')),
        'original paired numerical evidence/source/cycle/census required; do not promote execution PASS')
    provenance = dict(schema_version=SCHEMA, kind=KIND, importer_source=deepcopy(request['source']),
        origin_contract=ORIGIN_CONTRACT, origin_files={k: deepcopy(origin[k]) for k in sorted(fields)},
        origin=documents, origin_source_head=assembly['source_head'],
        origin_input_binding=deepcopy(assembly['input_binding']), origin_cycle=assembly['cycle'],
        origin_versions=deepcopy(selected['final_versions']), origin_world_size=world,
        origin_execution_status=dict(selected=selected['status'], reference=reference['status']),
        origin_numerical_status=comparison['status'],
        comparison_binding='ORIGINAL_COMPARATOR_SOURCE_CYCLE_CENSUS_NOT_FULL_STATE_EQUALITY',
        selected_role='FULL_STUDENT_NOT_EMA_NOT_LORA_NOT_MERGED', student=deepcopy(spec['payload']),
        positional_state=dict(policy=ROPE_POLICY, source=deepcopy(origin['model_source']),
            old_runtime_values_captured=False, formula='ORIGINAL_WanModel.__init__ rope_params(1024, head_dim splits)'),
        current_cycle=0, training_cycles_executed=0, training_resume='NOT_SUPPORTED_INITIALIZATION_ONLY',
        current_numerical_validation='NOT_RUN', quality_validation='NOT_RUN', default_promotion=False)
    return request, provenance


def _payload_header(value, model, request, *, device):
    import torch
    from .helios_public_assets import read_json
    require(type(value) is dict and set(value) == {'model', 'step', 'generator_updates', 'source_head',
        'input_binding', 'scope'}, 'exact original assembled full-Student payload header required')
    assembly = read_json(_pin(request['origin']['assembly'], maximum=4*2**20))
    require(type(value['step']) is int and value['step'] == 5*assembly['cycle']
        and type(value['generator_updates']) is int and value['generator_updates'] == assembly['cycle']
        and value['source_head'] == assembly['source_head'] and value['input_binding'] == assembly['input_binding']
        and value['scope'] == PAYLOAD_SCOPE, 'original payload source/input/G100-C500 header disagrees with receipt')
    expected = dict(model.named_parameters())
    require(set(expected) == set(model.state_dict()) and not list(model.named_buffers()),
            'original full Student must have parameter-only persistent state; RoPE is reconstructed separately')
    state = value['model']
    require(type(state) is dict and set(state) == set(expected)
        and len(state) == assembly['parameter_tensors']
        and sum(t.numel() for t in expected.values()) == assembly['numel'],
        'complete exact original Student parameter inventory required; no alias/prefix/base fill')
    for name, tensor in state.items():
        require(type(tensor) is torch.Tensor and tensor.layout == torch.strided
            and tensor.device.type == device and tensor.dtype == expected[name].dtype == torch.float32
            and tensor.shape == expected[name].shape and tensor.is_contiguous()
            and not tensor.requires_grad and tensor.grad_fn is None,
            'exact detached dense FP32 Student shape/dtype/storage required: ' + name)
    return state


def stage_student(model, request):
    """Safe metadata first, then whole finite owned tensors; no live mutation."""
    import torch
    pin = request['student']['payload']
    path = _pin(pin, maximum=request['parameters']['budget']['max_loaded_bytes'])
    try:
        header = torch.load(path, map_location='meta', mmap=True, weights_only=True)
        _payload_header(header, model, request, device='meta')
        del header
        raw = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
        state = _payload_header(raw, model, request, device='cpu')
        # Own the values rather than retaining mutable input-file mmap storage.
        values = {name: tensor.clone() for name, tensor in state.items()}
    except Exception as exc:
        raise ValueError('LiveWan Generator initialization: invalid safe complete Student payload') from exc
    require(all(bool(t.isfinite().all()) for t in values.values()), 'nonfinite original Student parameter values')
    _pin(pin, maximum=request['parameters']['budget']['max_loaded_bytes'])
    return values


def restore_student(view, values):
    """Copy complete admitted parameters and explicitly REBUILD absent RoPE."""
    import torch
    from distillschedule.reference import livewan_wan21 as provider
    student = view.model
    expected = student.state_dict()
    require(set(values) == set(expected) and all(type(t) is torch.Tensor and t.device.type == 'cpu'
        and t.dtype == expected[n].dtype == torch.float32 and t.layout == torch.strided
        and t.shape == expected[n].shape and bool(t.isfinite().all()) for n, t in values.items()),
        'complete finite staged Student changed before copy')
    api = provider.load_cpu(backend='torch-sdpa-cpu', precision='float32')
    # This is exactly the source-bound constructor expression, not an old
    # runtime snapshot, not a cast/zero-fill and not a tiny-model substitute.
    d = student.dim // student.num_heads
    with torch.device('cpu'):
        freqs = torch.cat([api.model.rope_params(1024, d - 4*(d//6)),
            api.model.rope_params(1024, 2*(d//6)), api.model.rope_params(1024, 2*(d//6))], dim=1)
    require(freqs.dtype == torch.complex128 and freqs.shape == student.freqs.shape
        and bool(freqs.isfinite().all()), 'original deterministic RoPE reconstruction changed')
    student.load_state_dict(values, strict=True)
    student.freqs = freqs
    view.rope_frequencies = freqs
    student.requires_grad_(False).eval()
    restored = student.state_dict()
    require(all(torch.equal(t, restored[n]) for n, t in values.items()),
            'exact full Student values changed during copy')


def import_generator_weights(path, output):
    """Explicit initialization -> existing public factory -> shared publisher."""
    request, provenance = validate_request(read_request(path))
    import torch
    from . import livewan_native_inference as native
    from .wan_native_inference import _load_codec, standalone_config
    from .checkpoint import rng_state, restore_rng
    from .host_memory import admit_host_memory
    from .inference import publish_export
    from .plugins import load_callable
    from .livewan_public import topology
    require(not torch.cuda.is_initialized() and not torch.distributed.is_initialized(),
            'CPU initialization requires no CUDA or distributed owner')
    topology()
    destination = Path(output).expanduser().absolute()
    require(destination.resolve() == destination, 'canonical create-only output path required')
    config_path = destination.with_name(destination.name + '.inference.json')
    require(not any(p.exists() or p.is_symlink() for p in (destination, config_path)),
            'new export/config paths required; overwrite forbidden')
    before = rng_state()  # preserve CURRENT caller RNG; OLD training RNG is never read
    try:
        p, probe, _, sizes = native._probe(request['parameters'])
        # In addition to the factory's full Student/codec/KV/MATH capacity,
        # charge mmap pages, owned admission values and the live copy peak.
        working = sizes['working_bytes'] + 3*request['student']['payload']['bytes']
        require(working <= p['budget']['max_loaded_bytes'], 'complete initialization/transient load capacity exceeded')
        admit_host_memory(working, p['budget']['host_reserve_bytes'])
        values = stage_student(probe, request)
        codec_evidence = p['codec']['evidence_kind']
        codec, portable_codec = _load_codec(request['codec'], p['budget'], codec_evidence)
        require(portable_codec == p['codec'], 'actual pinned codec disagrees with public inference parameters')
        factory, source = load_callable(native.FACTORY)
        components = factory(p)  # reuse exact unchanged public consumer/factory
        with components.managed():
            restore_student(components.state['generator'], values)
            del values
            components.state['codec'] = codec
            # Re-authenticate the complete file/origin/current closure after
            # loading and before any create-only portable publication.
            require(validate_request(request)[1] == provenance, 'origin changed during initialization')
            manifest = publish_export(components, destination,
                model=dict(factory=native.FACTORY, source=source, parameters=p, metadata=components.metadata),
                identity=dict(model=source, model_metadata=components.metadata, weights_initialization=provenance), cycle=0)
        import os
        import uuid
        temporary = config_path.with_name('.' + config_path.name + '.tmp-' + uuid.uuid4().hex)
        try:
            temporary.write_text(json.dumps(standalone_config(manifest, config_path),
                indent=2, sort_keys=True, allow_nan=False) + '\n')
            os.link(temporary, config_path)  # exclusive publication, never overwrite a racing writer
        finally:
            temporary.unlink(missing_ok=True)
        return {**manifest, 'inference_config': str(config_path), 'training_cycles_executed': 0,
                'initialization_only': True}
    finally:
        restore_rng(before)
