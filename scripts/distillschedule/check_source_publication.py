#!/usr/bin/env python3
"""Verify the small, immutable source/evidence publication; never import CUDA."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path


def verify(root: Path) -> dict:
    root = root.resolve()
    manifest_path = root / 'artifacts/publications/20261019/source_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['schema'] != 'dmd.github_source_snapshot.v1':
        raise ValueError('unknown publication schema')
    seen = set()
    for row in manifest['files']:
        path = (root / row['path']).resolve()
        if not path.is_relative_to(root) or row['path'] in seen:
            raise ValueError('unsafe or duplicate publication path')
        seen.add(row['path'])
        data = path.read_bytes()
        if len(data) != row['bytes'] or hashlib.sha256(data).hexdigest() != row['sha256']:
            raise ValueError('publication bytes/hash changed: ' + row['path'])
        if len(data) >= 1_000_000:
            raise ValueError('large raw artifact is not a publication summary')
    for row in manifest['branches']:
        if len(row['head']) != 40 or any(c not in '0123456789abcdef' for c in row['head']):
            raise ValueError('invalid source commit identity')
        if row['main_merged'] is not False:
            raise ValueError('publication must not imply main merge')
    rows = json.loads((root / 'artifacts/publications/20261019/evidence/planner/six_workload_native_match.json').read_text())['rows']
    if {row['id'] for row in rows} != {'W1','W2','W3','W4','W5','W6'} or len(rows) != 6:
        raise ValueError('six workloads must remain distinct')
    unresolved = next(row for row in rows if row['id'] == 'W5')
    if unresolved['manual_C2'] is not None or unresolved['selected_C2'] is not None:
        raise ValueError('unresolved H3 numerical gate must not become a timing result')
    return {'status': 'PASS_PUBLICATION_INTEGRITY_ONLY', 'files': len(seen),
            'source_branches': len(manifest['branches']), 'GPU_started': 0}


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parents[2]), indent=2))
