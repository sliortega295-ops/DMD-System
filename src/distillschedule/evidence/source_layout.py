"""Resolve current integrated file ownership without rewriting old import receipts."""
from __future__ import annotations

import copy
import json
from pathlib import Path, PurePosixPath
import re

SCHEMA = "distillschedule.source_layout_lock.v1"


def _relative(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or '\\' in value:
        raise ValueError(f"unsafe source path: {value}")
    return value


def reviewed_imports(root: Path) -> list[dict]:
    original = json.loads((root / 'docs/workloads/source_import_map.json').read_text())['files']
    path = root / 'docs/workloads/source_layout.lock.json'
    current = json.loads(path.read_text())
    if current.get('schema_version') != SCHEMA:
        raise ValueError('unsupported source layout lock')
    overrides = current['files']
    indexed = {r['target']: r for r in original}
    if not set(overrides) <= set(indexed):
        raise ValueError('layout override is not in the original import map')
    result = []
    destinations = set()
    for row in original:
        entry = copy.deepcopy(row)
        change = overrides.get(row['target'])
        if change:
            if change['original_integrated_sha256'] != row['integrated_sha256']:
                raise ValueError(f"origin receipt changed: {row['target']}")
            if not re.fullmatch('[0-9a-f]{40}', change['source_commit']):
                raise ValueError('source override needs an exact reviewed commit')
            if not re.fullmatch('[0-9a-f]{64}', change['integrated_sha256']):
                raise ValueError('source override needs SHA256')
            entry['target'] = _relative(change['target'])
            entry['integrated_sha256'] = change['integrated_sha256']
        if entry['target'] in destinations:
            raise ValueError('duplicate integrated target')
        destinations.add(entry['target'])
        result.append(entry)
    return result
