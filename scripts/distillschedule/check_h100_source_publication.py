#!/usr/bin/env python3
"""Check H100 publication bytes and scope; no Torch, CUDA, SSH or launcher import."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re


def verify(root: Path) -> dict:
    root = root.resolve()
    manifest = json.loads((root / 'artifacts/publications/20261019/h100/source_manifest.json').read_text())
    if manifest['schema'] != 'dmd.h100_source_snapshot.v1':
        raise ValueError('unknown H100 publication schema')
    if manifest['GPU_started'] != 0 or manifest['SSH_calls'] != 0 or manifest['merged_main'] is not False:
        raise ValueError('publication is not a resumed H100 campaign or main integration')
    if manifest['current_campaign_state'] != 'USER_STOPPED_NO_AUTOMATIC_RESUME':
        raise ValueError('H100 stop boundary must remain explicit')
    if manifest['native_math_or_defaults_changed'] is not False:
        raise ValueError('archive must not change native math or defaults')
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
    if len(rows) != 6 or {r['id'] for r in rows} != {'W1', 'W2', 'W3', 'W4', 'W5', 'W6'}:
        raise ValueError('six H100 workloads must remain distinct')
    for row in rows:
        for key in ('measured_training_head', 'published_head'):
            if not re.fullmatch('[0-9a-f]{40}', row[key]):
                raise ValueError('invalid measured/published H100 source')
        if row['formal_speedup_eligible'] is not False or row['main_merged'] is not False:
            raise ValueError('publication cannot promote diagnostic results')
        if row['physical_ranks'] != 8 or row['independent_logical_slots'] != 16 or row['profiled'] is not False:
            raise ValueError('H100 measured resource/timer scope changed')
        for p in row['evidence_paths']:
            if p not in seen:
                raise ValueError('H100 measured row lacks its source evidence')
    by_id = {r['id']: r for r in rows}
    if by_id['W1']['latest_norm_end_to_end'] != 'NOT_RUN':
        raise ValueError('untimed W1 norm port is not a measured faster endpoint')
    if by_id['W5']['semantics'] != 'GRID1000_APPROXIMATE_NOT_EXACT_CONTINUOUS':
        raise ValueError('W5 approximate Grid is not continuous DMD')
    if by_id['W6']['attention_backend'] != 'FA2' or by_id['W6']['fa3_status'] != 'FAIL_UNWAIVED_NOT_ACCEPTED':
        raise ValueError('W6 rejected FA3 path cannot become the measured endpoint')
    return dict(status='PASS_H100_ARCHIVE_INTEGRITY_ONLY', files=len(seen), workloads=6,
                source_refs=len(manifest['source_refs']), GPU_started=0, SSH_calls=0)


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parents[2]), indent=2))
