"""Resolve current integrated file ownership without rewriting old import receipts."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile

SCHEMA = "distillschedule.source_layout_lock.v1"
COMPACT_SCHEMA = "distillschedule.source_layout_lock.v2"


def _relative(value: str) -> str:
    if not isinstance(value, str) or not value or value == '.':
        raise ValueError(f"unsafe source path: {value}")
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or '\\' in value or path.as_posix() != value:
        raise ValueError(f"unsafe source path: {value}")
    return value


def _compact_imports(root: Path, original: list[dict], current: dict) -> list[dict]:
    """Resolve original ownership plus one current override per imported file."""
    overrides, retired = current.get('files'), current.get('retired_imports')
    if (not isinstance(overrides, dict) or not isinstance(retired, list)
            or not all(isinstance(path, str) for path in retired)
            or len(set(retired)) != len(retired)):
        raise ValueError('invalid compact source layout')
    indexed = {row['target']: row for row in original}
    if len(indexed) != len(original) or not set(overrides) <= set(indexed):
        raise ValueError('duplicate import or unknown source override')
    result, destinations = [], set()
    for row in original:
        entry = copy.deepcopy(row)
        _relative(entry['target'])
        if not re.fullmatch('[0-9a-f]{64}', entry['integrated_sha256']):
            raise ValueError('original import needs SHA256')
        change = overrides.get(row['target'])
        if row['target'] in overrides:
            required = {'target', 'integrated_sha256', 'source_commit'}
            if (not isinstance(change, dict) or not required <= change.keys()
                    or set(change) - required - {'reviewed_source_sha256'}):
                raise ValueError('invalid compact source override')
            if (not isinstance(change['source_commit'], str)
                    or not re.fullmatch('[0-9a-f]{40}', change['source_commit'])
                    or int(change['source_commit'], 16) == 0):
                raise ValueError('source override needs an exact reviewed commit')
            if (not isinstance(change['integrated_sha256'], str)
                    or not re.fullmatch('[0-9a-f]{64}', change['integrated_sha256'])
                    or ('reviewed_source_sha256' in change
                        and change['reviewed_source_sha256'] != change['integrated_sha256'])):
                raise ValueError('source override needs SHA256')
            entry.update(target=_relative(change['target']),
                         integrated_sha256=change['integrated_sha256'])
        if entry['target'] in destinations:
            raise ValueError('duplicate integrated target')
        destinations.add(entry['target'])
        result.append(entry)
    if not set(retired) <= destinations:
        raise ValueError('retired import must identify an imported source')
    for target in retired:
        if (root / _relative(target)).exists():
            raise ValueError('retired import is still active')
    return [entry for entry in result if entry['target'] not in retired]


def update_layout(root: Path, targets: list[str], source_commit: str) -> None:
    """Update only explicitly selected imported files; leave origin receipts alone."""
    root = Path(root).resolve()
    if (not isinstance(source_commit, str) or not re.fullmatch('[0-9a-f]{40}', source_commit)
            or int(source_commit, 16) == 0 or not targets or len(set(targets)) != len(targets)):
        raise ValueError('explicit source paths and exact reviewed commit required')
    path = root / 'docs/workloads/source_layout.lock.json'
    previous = path.read_bytes()
    current = json.loads(previous)
    if current.get('schema_version') != COMPACT_SCHEMA:
        raise ValueError('source updates require compact source layout v2')
    active = {row['target'] for row in reviewed_imports(root)}
    original = json.loads((root / 'docs/workloads/source_import_map.json').read_text())['files']
    origin_paths = {current['files'].get(row['target'], row)['target']: row['target'] for row in original}
    for target in targets:
        source = root / _relative(target)
        if target not in active or source.is_symlink() or not source.resolve().is_relative_to(root):
            raise ValueError(f'not an active imported source: {target}')
        origin = origin_paths[target]
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        row = dict(current['files'].get(origin, {}), target=target,
                   integrated_sha256=digest, source_commit=source_commit)
        if 'reviewed_source_sha256' in row:
            row['reviewed_source_sha256'] = digest
        current['files'][origin] = row
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(json.dumps(current, indent=2) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
            if path.read_bytes() != previous:
                raise ValueError('source layout changed during update')
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--check', action='store_true')
    action.add_argument('--update', nargs='+', metavar='PATH')
    parser.add_argument('--source-commit')
    args = parser.parse_args()
    if args.update:
        update_layout(args.repo, args.update, args.source_commit)
        print(json.dumps({'updated': args.update}))
    else:
        rows = reviewed_imports(args.repo)
        for row in rows:
            if hashlib.sha256((args.repo / row['target']).read_bytes()).hexdigest() != row['integrated_sha256']:
                raise ValueError('source identity mismatch: ' + row['target'])
        print(json.dumps({'status': 'PASS', 'files_checked': len(rows)}))


def reviewed_imports(root: Path) -> list[dict]:
    original = json.loads((root / 'docs/workloads/source_import_map.json').read_text())['files']
    path = root / 'docs/workloads/source_layout.lock.json'
    current = json.loads(path.read_text())
    if current.get('schema_version') == COMPACT_SCHEMA:
        return _compact_imports(root, original, current)
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
    retired = current.get('retired_imports', {})
    if not isinstance(retired, dict) or not set(retired) <= set(effective):
        raise ValueError('retired import must identify an original integrated source')
    for target, expected in retired.items():
        if (expected != effective[target]['integrated_sha256']
                or (root / _relative(target)).exists()):
            raise ValueError('retired import hash changed or file is still active')
    return [entry for entry in result if entry['target'] not in retired]


if __name__ == '__main__':
    main()
