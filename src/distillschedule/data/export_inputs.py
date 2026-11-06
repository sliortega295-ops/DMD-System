"""Prepare new inputs using only an explicitly trusted portable inference export.

No family dispatch, training bundle fallback, guessed geometry or conditions.
The restored model's optional encode_input callback owns its native contract.
Like other plugin entrypoints, the chosen export/factory must be trusted code.
"""
from copy import deepcopy
from contextlib import ExitStack
import inspect
import json
import os
from pathlib import Path

from distillschedule.training.adapters import ModelComponents
from distillschedule.training.checkpoint import restore_rng, rng_state
from distillschedule.training.inference import VerifiedExport
from distillschedule.training.plugins import load_callable, resolve_reference
from distillschedule.training.validation import _guard, _modules
from distillschedule.workloads.best_known.registry import fingerprint

from .manifest import (json_value, nonempty, require, source_identity,
                       source_schema, validate_source)
from .preparation import _new_output, _publish
from .tensors import tensor_tree

SCHEMA = 'distillschedule.export_inputs.v1'


def _document(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate export input field: ' + key)
            result[key] = value
        return result
    value = json.loads(path.read_text(), object_pairs_hook=unique)
    require(isinstance(value, dict) and set(value) == {'schema_version', 'weights', 'mode', 'records'}
            and value['schema_version'] == SCHEMA, 'exact export input schema/fields required')
    json_value(value)
    nonempty(value['mode'], 'export input mode')
    rows = value['records']
    require(isinstance(rows, list) and rows, 'nonempty export input records required')
    seen = set()
    for row in rows:
        require(isinstance(row, dict), 'export input record must be an object')
        identifier = nonempty(row.get('id'), 'record id')
        require(identifier not in seen, 'duplicate export input record id: ' + identifier)
        seen.add(identifier)
    return value


def prepare_inputs(source_manifest, output_dir):
    """Existing ``data prepare --preparer`` plugin; CPU, single-process only.

    Preserve Python/NumPy/Torch CPU RNG even on refusal. Modules are constructed
    locally, restored in full, and guarded read-only without a second model copy.
    Nothing is published until every row, source and export passes validation.
    """
    import torch
    from distillschedule.training.distributed import rank_world

    output = _new_output(output_dir)  # Existing user data wins, even over plugin imports.
    require(rank_world()[1] == 1 and int(os.environ.get('WORLD_SIZE', '1')) == 1,
            'export input preparation is CPU single-process only')
    require(not torch.is_autocast_enabled('cpu') and not torch.is_autocast_enabled('cuda'),
            'export input preparation does not admit autocast')
    path = Path(source_manifest).expanduser().resolve(strict=True)
    document = source_identity(path)
    request = _document(path)
    weights = nonempty(request['weights'], 'export weights path')
    require('://' not in weights, 'export weights path must be local')
    with ExitStack() as lifetime:
        verified = lifetime.enter_context(VerifiedExport(path.parent / Path(weights).expanduser()))
        root, manifest = verified.root, verified.manifest
        require(not output.is_relative_to(root), 'prepared output must not modify the export inventory')
        model = manifest['model']
        reference = resolve_reference(model['factory'], root)
        require(reference == model['factory'], 'export factory must be an explicit resolved trusted reference')
        before_rng = rng_state(device='cpu')
        restored_rng = False
        try:
            with ExitStack() as model_lifetime:
                factory, source = load_callable(reference)
                require(source == model['source'], 'export model factory source changed')
                factory_file = Path(inspect.getfile(factory)).resolve()
                verified.check_unchanged()
                components = factory({**deepcopy(model['parameters']), 'device': 'cpu'})
                require(isinstance(components, ModelComponents), 'export model factory must return ModelComponents')
                require(components.close is None or callable(components.close), 'model close must be callable')
                model_lifetime.enter_context(components.managed())
                components.validate()
                require(fingerprint(dict(components.metadata)) == fingerprint(model['metadata']),
                        'export model metadata changed')
                require(components.encode_input is not None, 'export adapter has no encode_input capability; no training asset fallback')
                verified.restore_modules(components)
                callback = components.encode_input
                callback_file = Path(inspect.getfile(callback if inspect.isfunction(callback) or inspect.ismethod(callback)
                                                     else type(callback))).resolve()
                code = source_identity(callback_file)
                parameters = dict(mode=request['mode'], source_dir=str(path.parent), model_metadata=deepcopy(model['metadata']))
                export_identity = dict(schema_version=SCHEMA, weights=str(root),
                    export_manifest_sha256=verified.manifest_sha256,
                    files=deepcopy(manifest['files']), model_source=deepcopy(source))
                documents = [document, source_identity(Path(__file__)), source_identity(factory_file), code,
                             source_identity(root / 'export.json'), source_identity(root / '_SUCCESS'),
                             *(dict(kind='file', path=str(root / name), sha256=sha)
                               for name, sha in manifest['files'].items())]
                encoder_parameters = {**deepcopy(parameters), **deepcopy(export_identity)}
                provenance = dict(mode='raw', documents=documents, sources=[], encoder=dict(
                    reference=callback.__module__ + ':' + getattr(callback, '__qualname__', type(callback).__qualname__),
                    source_path=code['path'], sha256=code['sha256'], parameters=encoder_parameters,
                    parameters_sha256=fingerprint(encoder_parameters)))
                # These are new local inference owners, never aliases of a live trainer.
                for module in _modules(components.state):
                    module.training = False
                before_state = _guard(components.state)
                metadata = {}

                def produce(row):
                    try:
                        with torch.no_grad():
                            result = callback(components.state, deepcopy(row), deepcopy(parameters))
                    finally:
                        require(_guard(components.state) == before_state, 'encode_input mutated exported model state or ownership')
                    require(isinstance(result, dict) and set(result) == {'features', 'metadata', 'sources'},
                            'encode_input must return exact features/metadata/sources encoder result')
                    native = result['metadata']
                    require(isinstance(native, dict) and bool(native), 'encode_input metadata must be a nonempty object')
                    require('_export_inputs' not in native, 'encode_input metadata uses reserved _export_inputs key')
                    native = json_value(native)
                    expected_metadata = {**native, '_export_inputs': export_identity}
                    require(not metadata or metadata == expected_metadata, 'encoder metadata changed across input records')
                    metadata.update(expected_metadata)
                    sources = result['sources']
                    require(isinstance(sources, dict) and sources and set(sources) <= {'image', 'video', 'audio', 'text'},
                            'encode_input must return explicit raw input source identities')
                    for identity in sources.values():
                        source_schema(identity)
                    provenance['sources'].append(dict(id=row['id'], media=deepcopy(sources)))
                    # Recheck the entire inventory, not only the original weight files.
                    verified.check_unchanged()
                    features, schema = tensor_tree(result['features'])
                    from torch.utils._pytree import tree_leaves
                    require(all(bool(torch.isfinite(value).all()) for value in tree_leaves(features)
                                if value.is_floating_point() or value.is_complex()), 'encoder returned nonfinite input features')
                    return features, schema

                # Preserve all recorded source identities. Already admitted export
                # bytes are checked by live inode/version tokens, while unrelated
                # raw inputs and encoder source retain their full source checks.
                export_paths = {str(root / name) for name in (*manifest['files'], 'export.json', '_SUCCESS')}
                admitted = {row['path']: row for row in documents if row['path'] in export_paths}

                def validate_document(identity):
                    if identity['path'] in admitted:
                        require(identity == admitted[identity['path']], 'export source identity changed')
                        verified.check_unchanged()
                    else:
                        validate_source(identity)

                def finish():
                    nonlocal restored_rng
                    # Cleanup can fail or modify files too. Finish both model
                    # and admitted-file lifetimes before the create-only rename.
                    model_lifetime.close()
                    lifetime.close()
                    restore_rng(before_rng)
                    restored_rng = True

                return _publish(output_dir, rows=request['records'], provenance=provenance,
                                metadata=metadata, produce=produce, validate_document=validate_document,
                                before_publish=finish)
        finally:
            if not restored_rng:
                restore_rng(before_rng)
