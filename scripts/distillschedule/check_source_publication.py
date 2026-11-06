#!/usr/bin/env python3
"""Verify the current measured-result data and its source bindings, without CUDA."""
from __future__ import annotations
import csv
import hashlib
import json
import math
from pathlib import Path
import re


def verify(root: Path) -> dict:
    root = root.resolve()
    manifest_path = root / 'artifacts/results/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['schema'] != 'dmd.public_results.v1':
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
    required = {'artifacts/results/a100.csv', 'artifacts/results/h100.csv',
                'artifacts/results/official.csv', 'artifacts/results/gan_nomerge_retime.json',
                'artifacts/results/h100_sources.json',
                'artifacts/results/planner.json'}
    if seen != required or not re.fullmatch('[0-9a-f]{40}', manifest['source_revision']):
        raise ValueError('current results require the complete source-bound data set')
    with (root / 'artifacts/results/a100.csv').open(newline='') as stream:
        measured = list(csv.DictReader(stream))
    if [r['id'] for r in measured] != ['W1', 'W2', 'W3', 'W4', 'W5', 'W6']:
        raise ValueError('six measured workloads must remain distinct')
    for row in measured:
        baseline, system, ratio = (float(row[k]) for k in
                                 ('baseline_reference_s', 'system_s', 'reference_ratio'))
        if (not all(math.isfinite(v) and v > 0 for v in (baseline, system, ratio))
                or not math.isclose(ratio, baseline / system, rel_tol=1e-12)):
            raise ValueError('invalid measured time or ratio')
    gan = json.loads((root / 'artifacts/results/gan_nomerge_retime.json').read_text())
    if (gan['merge_counts'] != [0] * 16 or gan['reported_C2_s'] != 116.254
            or float(measured[2]['system_s']) != gan['reported_C2_s']):
        raise ValueError('current W3 result requires unmerged weights')
    rows = json.loads((root / 'artifacts/results/planner.json').read_text())['rows']
    if {row['id'] for row in rows} != {'W1','W2','W3','W4','W5','W6'} or len(rows) != 6:
        raise ValueError('six workloads must remain distinct')
    unresolved = next(row for row in rows if row['id'] == 'W5')
    if unresolved['manual_C2'] is not None or unresolved['selected_C2'] is not None:
        raise ValueError('unresolved H3 numerical gate must not become a timing result')
    return {'status': 'PASS_PUBLICATION_INTEGRITY_ONLY', 'files': len(seen),
            'source_branches': len(manifest['branches']), 'GPU_started': 0}


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parents[2]), indent=2))
