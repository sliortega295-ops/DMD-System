"""Rank-qualified metadata admission for the ORIGINAL W6 cache/parser.

No tensor values are serialized and no tensor is transformed here. In particular,
rank 6's native 94-frame audio is neither cropped nor generalized to other ranks.
The CPU preflight and live Session call the same original parser and validator.
"""
import inspect
import json
from pathlib import Path

from distillschedule.core.contracts import require
from distillschedule.workloads.best_known.registry import SHA256_RE, file_sha256, repository_root
from .h3_ar_assets import checked
from .h3_ar_task import INPUT_CONTRACT_SHA256, SOURCE_RECEIPT

CONTRACT_PATH = 'configs/distillschedule/original_h3_ar/rank_inputs_v2.json'
CONTRACT_SCHEMA = 'distillschedule.original_h3_ar_rank_inputs.v2'
VALIDATION_SCHEMA = 'distillschedule.original_h3_ar_prepared_input.v2'


def input_contract():
    path = repository_root()/CONTRACT_PATH
    require(file_sha256(path) == INPUT_CONTRACT_SHA256,
            'rank-input contract file changed', path='h3_ar.inputs')
    result = json.loads(path.read_text())
    require(result['schema_version'] == CONTRACT_SCHEMA and result['native_source_receipt_sha256'] == SOURCE_RECEIPT
            and [r['rank'] for r in result['rows']] == list(range(16)),
            'not the original rank-qualified native input contract', path='h3_ar.inputs')
    return result


def expected_input(task, rank):
    require(task.input_contract_sha256 == INPUT_CONTRACT_SHA256 and type(rank) is int and 0 <= rank < 16,
            'explicit original rank/input contract required', path='h3_ar.inputs')
    return input_contract()['rows'][rank]


def tensor_metadata(value, torch):
    """Only shape/dtype/structure; never touch values, devices, or RNG."""
    if isinstance(value, torch.Tensor):
        return dict(shape=list(value.shape), dtype=str(value.dtype))
    if isinstance(value, dict):
        return {k: tensor_metadata(v, torch) for k, v in value.items()
                if isinstance(v, (torch.Tensor, dict, tuple, list))}
    if isinstance(value, (tuple, list)):
        return [tensor_metadata(v, torch) for v in value]
    return None


def _input_identity(task, rank, identity):
    """Relocation changes the directory, never the original sample bytes/name."""
    expected = expected_input(task, rank)['input_identity']
    require(isinstance(identity, dict) and isinstance(identity.get('path'), str)
            and bool(identity['path']) and Path(identity['path']).name == expected['path']
            and identity.get('sha256') == expected['sha256'],
            'rank-qualified original cache filename/SHA mismatch', path='h3_ar.inputs')
    return {k: identity[k] for k in ('path', 'sha256')}


def expected_validation(task, rank, input_identity=None):
    row = expected_input(task, rank)
    identity = row['input_identity'] if input_identity is None else _input_identity(task, rank, input_identity)
    return dict(schema_version=VALIDATION_SCHEMA, input_contract_sha256=task.input_contract_sha256,
                **{**row, 'input_identity': identity})


def validate_prepared_input(task, rank, input_identity, raw, positive, negative, torch):
    row = expected_input(task, rank)
    _input_identity(task, rank, input_identity)
    require(tensor_metadata(raw, torch) == row['raw_tensors']
            and tensor_metadata(positive, torch) == row['positive_tensors']
            and tensor_metadata(negative, torch) == row['negative_tensors'],
            'rank-qualified original prepared shape/dtype/packed metadata mismatch', path='h3_ar.inputs')
    scalars = {k: v for k, v in positive.items() if not isinstance(v, (torch.Tensor, dict, list, tuple))}
    require(scalars == row['positive_scalars'] and all(type(v) is float for v in scalars.values()),
            'original prepared clean-reference scalar conditioning mismatch', path='h3_ar.inputs')
    return expected_validation(task, rank, input_identity)


def validate_input_receipt(task, rank, input_identity, build):
    expected = expected_validation(task, rank, input_identity)
    require({k: input_identity.get(k) for k in ('path', 'sha256')} == expected['input_identity']
            and build.get('input_shapes') == expected['input_shapes']
            and build.get('input_validation') == expected,
            'native build lacks matching rank-qualified input/parser contract', path='h3_ar.inputs')


def validate_parser(native):
    pins = input_contract()['parser_sources']
    for function, name in ((native.prepare_batch, 'h3_dmd2.py'), (native.prepare_cached_inputs, 'train_ours.py')):
        path = Path(inspect.getfile(function)).resolve(strict=True)
        require(file_sha256(path) == pins['examples/minimax_h3/model_training/'+name],
                'original prepare_batch/parser source changed', path='h3_ar.inputs')


def prepare_rank_input(task, cache_manifest, native, rank, torch, *, teacher_cfg_scale=1.0):
    expected_input(task, rank)
    contract = input_contract()
    # The caller's explicit path/SHA binds the relocated manifest itself; its
    # rows must still describe every original sample. The archived manifest's
    # hash remains provenance, not a requirement to reproduce author paths.
    cache = json.loads(checked(cache_manifest).read_text())
    require(cache.get('status') == 'PASS_CACHE_SOURCE_AND_SCHEMA' and len(cache.get('rows', [])) == 16,
            'native previous/current cache manifest rejected', path='h3_ar.inputs')
    for index, row in enumerate(cache['rows']):
        _input_identity(task, index, row)
    identity = cache['rows'][rank]
    validate_parser(native)
    raw = torch.load(checked({k: identity[k] for k in ('path', 'sha256')}), map_location='cpu', weights_only=True)
    # ORIGINAL deterministic parser: do not add padding, cropping, dtype casts,
    # sample substitution, or a replacement input-preparation implementation.
    require(type(teacher_cfg_scale) is float and teacher_cfg_scale == 1.0,
            'original teacher CFG scale changed', path='h3_ar.inputs')
    positive, negative = native.prepare_batch(raw, teacher_cfg_scale)
    witness = validate_prepared_input(task, rank, identity, raw, positive, negative, torch)
    return positive, negative, identity, witness


def preflight_native_inputs(task, cache_manifest, native, torch):
    require(not torch.cuda.is_initialized(), 'native input CPU preflight must not initialize CUDA', path='h3_ar.inputs')
    rows = []
    for rank in range(16):
        positive, negative, identity, witness = prepare_rank_input(task, cache_manifest, native, rank, torch)
        rows.append(witness)
        del positive, negative
    require(not torch.cuda.is_initialized(), 'original parser initialized CUDA during CPU preflight', path='h3_ar.inputs')
    return dict(status='PASS_ALL16_ORIGINAL_PREPARE_BATCH_SAME_SESSION_VALIDATOR_CPU',
                input_contract_sha256=task.input_contract_sha256, cache_manifest=cache_manifest,
                parser_sources=input_contract()['parser_sources'], rows=rows, GPU_started=0, CUDA_initialized=False)


def validate_preflight(task, proof):
    contract = input_contract()
    require(isinstance(proof, dict) and isinstance(proof.get('rows'), list) and len(proof['rows']) == 16,
            'all16 CPU input proof required', path='h3_ar.inputs')
    cache_manifest = proof.get('cache_manifest', {})
    require(isinstance(cache_manifest, dict) and set(cache_manifest) == {'path', 'sha256'}
            and isinstance(cache_manifest['path'], str) and bool(cache_manifest['path'])
            and isinstance(cache_manifest['sha256'], str) and SHA256_RE.fullmatch(cache_manifest['sha256']),
            'explicit cache manifest path/SHA proof required', path='h3_ar.inputs')
    require(proof == dict(status='PASS_ALL16_ORIGINAL_PREPARE_BATCH_SAME_SESSION_VALIDATOR_CPU',
                input_contract_sha256=task.input_contract_sha256, cache_manifest=cache_manifest,
                parser_sources=contract['parser_sources'], rows=[expected_validation(task, r, proof['rows'][r].get('input_identity')) for r in range(16)],
                GPU_started=0, CUDA_initialized=False),
            'CPU original16 parser/session input proof missing or changed', path='h3_ar.inputs')
