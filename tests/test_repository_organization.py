"""Six-protocol discovery, immutable provenance and source-only organization."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from distillschedule.workloads.catalog import IDS, describe, load_catalog, verify_checkout
from distillschedule.evidence.source_layout import reviewed_imports

ROOT = Path(__file__).resolve().parents[1]


def test_all_six_protocols_are_distinct_from_five_legacy_profiles():
    from distillschedule.workloads.best_known.registry import CANONICAL_IDS
    assert tuple(row['id'] for row in load_catalog()['workloads']) == IDS
    assert len(CANONICAL_IDS) == 5
    assert 'Grid1000' in describe('W5')['protocol']
    assert '8-step' in describe('W6')['protocol']
    assert describe('W6')['legacy_profile'] is None


def test_livewan_explicit_c51_not_legacy_default():
    row = describe('W4')
    assert '--include-livewan-c51' in row['inspect_args']
    from distillschedule.workloads.best_known.registry import load_profile
    assert 'c34_' in load_profile(row['legacy_profile'], root=ROOT)['runtime']['typed_plan']


def test_newer_measured_and_published_hardware_sources_remain_separate():
    row = describe('W1', hardware='h100')
    assert row['h100']['measured_source'].startswith('7a668bcb')
    assert row['h100']['published_source'].startswith('97b8ecc9')
    assert row['discovery_only'] and row['native_execution'] == 'NOT_RUN'
    assert row['inspection_command'] is None  # Never suggest the A100 command as an H100 launch.


def test_w5_failure_and_no_new_hardware_result_are_preserved():
    row = describe('W5')
    assert 'FAIL_PAIR_NOT_RUN' in row['a100_saved_plan_evidence']['status']
    assert 'NOT_A_NEW_GPU_VALIDATION' in row['h100']['integration']


def test_catalog_sources_and_evidence_files_exist_and_match():
    assert verify_checkout(ROOT)['status'] == 'PASS_SOURCE_CATALOG'


@pytest.mark.parametrize('args,expected', [
    (['--help'], 'workloads'),
    (['workloads','list'], 'W6'),
    (['workloads','list','--legacy'], 'livewan_crossscale_pipeline_best'),
    (['workloads','describe','--workload','W4'], '--include-livewan-c51'),
    (['workloads','describe','--workload','W6','--hardware','h100'], '0419b96b'),
])
def test_public_cli(args, expected):
    result = subprocess.run([sys.executable, 'scripts/distillctl.py', *args],
                            cwd=ROOT, capture_output=True, text=True, check=True)
    assert expected in result.stdout


def test_discovery_does_not_import_torch_or_call_subprocess(capsys, monkeypatch):
    from distillschedule.workloads.catalog import main
    def forbidden(*a, **k):
        raise AssertionError('metadata discovery launched a subprocess')
    monkeypatch.setattr(subprocess, 'run', forbidden)
    assert main(['list','--json']) == 0
    assert len(json.loads(capsys.readouterr().out)) == 6


@pytest.mark.parametrize('workload', IDS)
def test_read_only_inspection_commands_use_real_existing_adapters(workload):
    row = describe(workload)
    result = subprocess.run([sys.executable, 'scripts/distillctl.py', *row['inspect_args']],
                            cwd=ROOT, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data.get('gpu_execution') == 'NOT_RUN'


def test_invalid_catalog_rejected(tmp_path):
    value = load_catalog()
    value['workloads'].pop()
    path=tmp_path/'bad.json';path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='W1 through W6'):
        load_catalog(path)


def test_execution_permission_not_allowed_in_discovery(tmp_path):
    value=load_catalog();value['workloads'][0]['inspect_args'].append('--execute')
    path=tmp_path/'bad.json';path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='authorize execution'):
        load_catalog(path)


def test_current_source_overlay_is_exact_not_a_disabled_check():
    for row in reviewed_imports(ROOT):
        path=ROOT/row['target']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row['integrated_sha256'], row['target']


def test_overlay_cannot_change_origin_hash(tmp_path):
    directory=tmp_path/'docs/workloads';directory.mkdir(parents=True)
    (directory/'source_import_map.json').write_bytes((ROOT/'docs/workloads/source_import_map.json').read_bytes())
    lock=json.loads((ROOT/'docs/workloads/source_layout.lock.json').read_text())
    next(iter(lock['files'].values()))['original_integrated_sha256']='0'*64
    (directory/'source_layout.lock.json').write_text(json.dumps(lock))
    with pytest.raises(ValueError,match='origin receipt changed'):
        reviewed_imports(tmp_path)


def test_moves_have_one_owner_and_keep_runtime_paths():
    manifest=json.loads((ROOT/'docs/architecture/repository_organization.json').read_text())
    assert len({r['to'] for r in manifest['relocations']})==len(manifest['relocations'])
    for row in manifest['relocations']:
        assert not (ROOT/row['from']).exists()
        assert (ROOT/row['to']).is_file()
        assert not row['from'].startswith(('src/','runtime/','configs/','third_party/'))
    for relative in manifest['preserved_launcher_closure']:
        assert (ROOT/relative).is_file()
    assert (ROOT/'runtime/host_admission.py').is_file()


def test_package_version_and_nested_root_detection(monkeypatch,tmp_path):
    import distillschedule
    from distillschedule.cli import _default_repo_root
    assert f'version = "{distillschedule.__version__}"' in (ROOT/'pyproject.toml').read_text()
    (tmp_path/'pyproject.toml').touch()
    nested=tmp_path/'src/distillschedule/nested';nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    assert _default_repo_root()==tmp_path
