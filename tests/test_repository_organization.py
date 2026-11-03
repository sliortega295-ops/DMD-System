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
    assert 'continuous sigma' in describe('W5')['protocol']
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
    assert row['a100_saved_plan_evidence']['status'] == 'NOT_RUN'
    assert row['name'].endswith('H3 Continuous)')
    assert 'NOT_A_NEW_GPU_VALIDATION' in row['h100']['integration']


def test_catalog_sources_and_evidence_files_exist_and_match():
    assert verify_checkout(ROOT)['status'] == 'PASS_SOURCE_CATALOG'
    for workload in ('W1', 'W4', 'W6'):
        assert describe(workload)['runbook'] == 'docs/runbooks/native_planner_workflow.md'


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


def test_retired_imports_preserve_hashes_and_cannot_hide_active_files(tmp_path):
    directory = tmp_path / 'docs/workloads'; directory.mkdir(parents=True)
    for name in ('source_import_map.json', 'source_layout.lock.json'):
        (directory / name).write_bytes((ROOT / 'docs/workloads' / name).read_bytes())
    lock_path = directory / 'source_layout.lock.json'
    lock = json.loads(lock_path.read_text())
    assert lock['retired_imports']
    target = next(iter(lock['retired_imports']))
    active = tmp_path / target; active.parent.mkdir(parents=True); active.write_text('not retired')
    with pytest.raises(ValueError, match='still active'):
        reviewed_imports(tmp_path)
    active.unlink()
    lock['retired_imports'][target] = '0' * 64
    lock_path.write_text(json.dumps(lock))
    with pytest.raises(ValueError, match='hash changed'):
        reviewed_imports(tmp_path)


def test_public_guides_recipes_and_defaults_do_not_embed_author_paths():
    files = [ROOT / 'README.md']
    for folder in ('docs', 'recipes', 'configs'):
        files.extend(p for p in (ROOT / folder).rglob('*') if p.suffix in {'.md', '.json', '.yaml', '.yml'})
    prefixes = ('/home/lyy/', '/home/shuchenweng/', '/share/project/shuchenweng/', '/mnt/bn/yufan')
    matches = []
    for path in files:
        text = path.read_text()
        matches.extend((str(path.relative_to(ROOT)), prefix) for prefix in prefixes if prefix in text)
    assert not matches


@pytest.mark.parametrize('coordinator', ['wan13b_fair', 'static_fsdp2_p0', 'livewan_crossscale'])
@pytest.mark.parametrize('environment', [None, 'envs/model runtime', '/operator/model runtime'])
def test_native_coordinator_and_collector_share_configurable_python(tmp_path, monkeypatch,
                                                                  coordinator, environment):
    import runpy
    import shlex

    path = ROOT / 'scripts/distillschedule' / ('run_' + coordinator + '_world16.py')
    module = runpy.run_path(str(path))
    source = tmp_path / 'source tree'
    argv = ['--run-id', 'CPU_PATH_TEST_ONLY', '--expected-head', '1' * 40,
            '--remote-root', str(source)]
    if coordinator == 'livewan_crossscale':
        argv += ['--hosts', 'cpu-node0,cpu-node1', '--run-kind', 'validate',
                 '--execution-mode', 'baseline']
        for name in ('remote-output-root', 'student-checkpoint', 'critic-checkpoint',
                     'teacher-checkpoint', 'trace', 'bootstrap', 'source-root',
                     'source-receipt', 'python-overlay'):
            argv += ['--' + name, str(tmp_path / name)]
        command_name, summary_name = 'command', 'summarize'
    else:
        argv += ['--mode', 'validate', '--helios-runtime-root', str(tmp_path / 'runtime')]
        if coordinator == 'wan13b_fair':
            argv += ['--placement', 'world16-fsdp', '--model-root', str(tmp_path / 'model'),
                     '--bootstrap', str(tmp_path / 'bootstrap'), '--bootstrap-hash', 'a' * 64]
        else:
            argv += ['--variant', 'no-gan', '--mesh-policy', 'world16-fsdp',
                     '--rollout-batch-size', '1', '--local-evidence-root', str(tmp_path / 'evidence')]
        command_name, summary_name = '_command', '_summarize'
    if environment is not None:
        argv += ['--pyenv', environment]
    args = module['parser']().parse_args(argv)
    expected = Path(environment or '.venv')
    if not expected.is_absolute():
        expected = source / expected
    remote = shlex.split(module[command_name](args, 0, '127.0.0.1')[-1])[0]
    assert 'DISTILLSCHEDULE_PYENV=' + shlex.quote(str(expected)) in remote
    captured = []
    class StopBeforeSSH(Exception):
        pass
    def capture(command, **kwargs):
        captured.append(command)
        raise StopBeforeSSH
    monkeypatch.setattr(subprocess, 'run', capture)
    with pytest.raises(StopBeforeSSH):
        module[summary_name](args, tmp_path / 'local')
    assert len(captured) == 1
    remote = shlex.split(captured[0][-1])[0]
    assert shlex.quote(str(expected / 'bin/python')) in remote
    assert '/tmp/distillschedule-helios-official' not in path.read_text()


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
