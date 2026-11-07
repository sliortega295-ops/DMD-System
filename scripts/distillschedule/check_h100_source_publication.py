#!/usr/bin/env python3
"""Check H100 publication bytes and scope; no Torch, CUDA, SSH or launcher import."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re


def verify(root: Path) -> dict:
    root = root.resolve()
    manifest = json.loads((root / 'artifacts/results/h100_sources.json').read_text())
    if manifest['schema'] != 'dmd.h100_measured_results.v2':
        raise ValueError('unknown H100 publication schema')
    if set(manifest) != {'schema', 'source_refs', 'workloads', 'files', 'provenance'}:
        raise ValueError('H100 result data cannot change the stop boundary or authorize execution')
    seen = set()
    for row in manifest['files']:
        path = (root / row['path']).resolve()
        if not path.is_relative_to(root) or row['path'] in seen:
            raise ValueError('unsafe or duplicate H100 publication path')
        seen.add(row['path'])
        data = path.read_bytes()
        if len(data) >= 1_000_000:
            raise ValueError('large raw payload is outside this publication')
        if len(data) != row['bytes'] or hashlib.sha256(data).hexdigest() != row['sha256']:
            raise ValueError('H100 archive bytes changed: ' + row['path'])
    for ref in manifest['source_refs']:
        if not re.fullmatch('[0-9a-f]{40}', ref['head']):
            raise ValueError('invalid frozen H100 source identity')
        if ref['repository'] not in {'DMD-System', 'H3-DMD-for-A100'}:
            raise ValueError('unexpected source repository')
    rows = manifest['workloads']
    if len(rows) != 3 or {r['id'] for r in rows} != {'W1', 'W3', 'W6'}:
        raise ValueError('published H100 measurements must remain distinct')
    for row in rows:
        for key in ('measured_training_head', 'published_head'):
            if not re.fullmatch('[0-9a-f]{40}', row[key]):
                raise ValueError('invalid measured/published H100 source')
        if type(row['replications']) is not int or row['replications'] < 1:
            raise ValueError('measurements require their recorded replication count')
        if row['physical_ranks'] != 8 or row['independent_logical_slots'] != 16 or row['profiled'] is not False:
            raise ValueError('H100 measured resource/timer scope changed')
    if seen != {'artifacts/results/h100.csv'}:
        raise ValueError('H100 measured rows require their complete result table')
    by_id = {r['id']: r for r in rows}
    if by_id['W6']['attention_backend'] != 'FA2':
        raise ValueError('W6 measured attention backend must remain FA2')
    return dict(status='PASS_H100_ARCHIVE_INTEGRITY_ONLY', files=len(seen), workloads=len(rows),
                source_refs=len(manifest['source_refs']), GPU_started=0, SSH_calls=0)


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parents[2]), indent=2))
