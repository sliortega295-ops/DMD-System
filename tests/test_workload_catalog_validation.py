"""Reject ambiguous or executable metadata without changing native contracts."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from distillschedule.workloads import catalog


def write_catalog(tmp_path: Path, value: dict) -> Path:
    path = tmp_path / 'catalog.json'
    path.write_text(json.dumps(value), encoding='utf-8')
    return path


@pytest.mark.parametrize('command', [
    ['workloads', 'native', 'validate', '--execute=true'],
    ['workloads', 'candidates', '--execute-native=1'],
    ['workloads', 'run', '--dry-run'],
    ['workloads', 'native', 'program', 'run'],
    'workloads candidates',
    ['workloads', 'candidates', None],
    ['workloads', 'candidates', ''],
    [],
])
def test_only_structured_read_only_commands_are_admitted(tmp_path, command):
    value = catalog.load_catalog()
    value['workloads'][0]['inspect_args'] = command
    with pytest.raises(ValueError, match='read-only inspection'):
        catalog.load_catalog(write_catalog(tmp_path, value))


@pytest.mark.parametrize('relative', [
    '/tmp/outside.py', '../outside.py', 'docs/../../outside.py',
    r'docs\outside.py', 'C:/outside.py', './docs/example.md', '', '.',
])
def test_catalog_paths_cannot_escape_the_checkout(tmp_path, relative):
    value = catalog.load_catalog()
    value['workloads'][0]['runbook'] = relative
    with pytest.raises(ValueError, match='path'):
        catalog.load_catalog(write_catalog(tmp_path, value))


def test_duplicate_slug_is_rejected(tmp_path):
    value = catalog.load_catalog()
    value['workloads'][1]['slug'] = value['workloads'][0]['slug']
    with pytest.raises(ValueError, match='unique'):
        catalog.load_catalog(write_catalog(tmp_path, value))


@pytest.mark.parametrize('value_to_use', [None, 123, 'main', 'a' * 39])
def test_source_index_must_be_an_exact_commit(tmp_path, value_to_use):
    value = catalog.load_catalog()
    value['source_index'] = value_to_use
    with pytest.raises(ValueError, match='source index'):
        catalog.load_catalog(write_catalog(tmp_path, value))


def test_evidence_sha_is_required(tmp_path):
    value = catalog.load_catalog()
    relative = next(iter(value['evidence_files']))
    value['evidence_files'][relative] = 'not-a-sha256'
    with pytest.raises(ValueError, match='evidence file'):
        catalog.load_catalog(write_catalog(tmp_path, value))


def test_checkout_verification_rejects_external_symlinks(tmp_path, monkeypatch):
    root = tmp_path / 'checkout'
    root.mkdir()
    outside = tmp_path / 'external'
    outside.mkdir()
    (root / 'docs').symlink_to(outside, target_is_directory=True)
    value = catalog.load_catalog()
    value['evidence_files'] = {'docs/file.json': '0' * 64}
    monkeypatch.setattr(catalog, 'load_catalog', lambda: value)
    with pytest.raises(ValueError, match='outside checkout'):
        catalog.verify_checkout(root)


def test_h100_description_does_not_label_an_a100_inspection_scope():
    row = catalog.describe('W6', hardware='h100')
    assert row['inspection_hardware_scope'] == 'h100_separately_pinned_source'
    assert row['inspection_command'] is None
    assert row['h100']['measured_source'].startswith('0419b96b')


def test_unknown_hardware_rejected_without_guessing():
    with pytest.raises(ValueError, match='hardware'):
        catalog.describe('W1', hardware='b200')
