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
    # New integrations append a hash chain instead of rewriting any prior
    # import/override receipt. An absent chain is exactly the legacy result.
    overlays = current.get('integration_overlays', [])
    if not isinstance(overlays, list):
        raise ValueError('source integration overlays must be an ordered list')
    effective = {entry['target']: entry for entry in result}
    seen = {target: {entry['integrated_sha256']} for target, entry in effective.items()}
    fields = {'schema_version', 'target', 'previous_sha256', 'integrated_sha256', 'source_commit', 'change'}
    for overlay in overlays:
        if (not isinstance(overlay, dict) or set(overlay) != fields
                or overlay['schema_version'] != 'distillschedule.source_integration_overlay.v1'):
            raise ValueError('unsupported source integration overlay')
        if not isinstance(overlay['target'], str):
            raise ValueError('source integration overlay target must be a relative path')
        target = _relative(overlay['target'])
        if target not in effective:
            raise ValueError('source integration overlay target is not imported')
        if (not isinstance(overlay['source_commit'], str)
                or not re.fullmatch('[0-9a-f]{40}', overlay['source_commit'])
                or int(overlay['source_commit'], 16) == 0
                or not isinstance(overlay['change'], str) or not overlay['change'].strip()):
            raise ValueError('source integration overlay needs exact source commit and change')
        previous, updated = overlay['previous_sha256'], overlay['integrated_sha256']
        if (not isinstance(previous, str) or not isinstance(updated, str)
                or not re.fullmatch('[0-9a-f]{64}', previous) or not re.fullmatch('[0-9a-f]{64}', updated)
                or effective[target]['integrated_sha256'] != previous or updated in seen[target]):
            raise ValueError('source integration overlay hash chain is discontinuous, conflicting or repeated')
        effective[target]['integrated_sha256'] = updated
        seen[target].add(updated)
    return result
