"""Prepare new inputs using only an explicitly trusted portable inference export.

No family dispatch, training bundle fallback, guessed geometry or conditions.
The restored model's optional encode_input callback owns its native contract.
Like other plugin entrypoints, the chosen export/factory must be trusted code.
"""
from copy import deepcopy
import inspect
import json
import os
from pathlib import Path

from distillschedule.training.adapters import ModelComponents
from distillschedule.training.checkpoint import restore_rng, rng_state
from distillschedule.training.inference import read_export, restore_modules
from distillschedule.training.plugins import load_callable, resolve_reference
from distillschedule.training.validation import _guard, _modules
from distillschedule.workloads.best_known.registry import fingerprint

from .manifest import (json_value, nonempty, require, resolve_path, source_identity,
                       source_schema)
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
    root, manifest = read_export(resolve_path(request['weights'], path.parent))
    require(not output.is_relative_to(root), 'prepared output must not modify the export inventory')
    model = manifest['model']
    reference = resolve_reference(model['factory'], root)
    require(reference == model['factory'], 'export factory must be an explicit resolved trusted reference')
    before_rng = rng_state(device='cpu')
    try:
        factory, source = load_callable(reference)
        require(source == model['source'], 'export model factory source changed')
        factory_file = Path(inspect.getfile(factory)).resolve()
        require(read_export(root)[1] == manifest, 'export changed before construction')
        components = factory({**deepcopy(model['parameters']), 'device': 'cpu'})
        require(isinstance(components, ModelComponents), 'export model factory must return ModelComponents')
        components.validate()
        require(fingerprint(dict(components.metadata)) == fingerprint(model['metadata']),
                'export model metadata changed')
        require(components.encode_input is not None, 'export adapter has no encode_input capability; no training asset fallback')
        restore_modules(components, root, manifest)
        callback = components.encode_input
        callback_file = Path(inspect.getfile(callback if inspect.isfunction(callback) or inspect.ismethod(callback)
                                             else type(callback))).resolve()
        code = source_identity(callback_file)
        parameters = dict(mode=request['mode'], source_dir=str(path.parent), model_metadata=deepcopy(model['metadata']))
        export_identity = dict(schema_version=SCHEMA, weights=str(root),
            export_manifest_sha256=source_identity(root / 'export.json')['sha256'],
            files=deepcopy(manifest['files']), model_source=deepcopy(source))
        documents = [document, source_identity(Path(__file__)), source_identity(factory_file), code,
                     source_identity(root / 'export.json'), source_identity(root / '_SUCCESS'),
                     *(source_identity(root / name) for name in manifest['files'])]
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
            require(read_export(root)[1] == manifest, 'export changed during input encoding')
            features, schema = tensor_tree(result['features'])
            from torch.utils._pytree import tree_leaves
            require(all(bool(torch.isfinite(value).all()) for value in tree_leaves(features)
                        if value.is_floating_point() or value.is_complex()), 'encoder returned nonfinite input features')
            return features, schema

        return _publish(output_dir, rows=request['records'], provenance=provenance, metadata=metadata, produce=produce)
    finally:
        restore_rng(before_rng)
