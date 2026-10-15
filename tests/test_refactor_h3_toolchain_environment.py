"""Explicit H3 child-toolchain environment; no Torch import or CUDA context."""
import json
import shutil
import subprocess

import pytest

from distillschedule.search.execution import resolve_candidate, validated_profile
from distillschedule.search.h3_implementation import H3, H3Implementation
from distillschedule.workloads.best_known import registry


def _plan(tmp_path, mode='correctness'):
    assets = json.loads((registry.repository_root() /
        'configs/distillschedule/best_known/assets_examples' / (H3 + '.json')).read_text())
    assets['assets']['diffusers_source'] = dict(path='/fixture/diffusers',
        identity_file='identity.json', sha256='a' * 64)
    if mode != 'correctness':
        assets['correctness'] = dict(path='/fixture/correctness.json', sha256='b' * 64)
    path = tmp_path / 'assets.json'
    path.write_text(json.dumps(assets))
    return resolve_candidate(H3, H3Implementation().id, assets_path=path,
                             run_id='CPU_ONLY_TOOLCHAIN', mode=mode)


@pytest.mark.parametrize('inherited', [None, '', '/untrusted/bin:/usr/local/cuda/bin'])
@pytest.mark.parametrize('mode', ['correctness', 'diagnostic-timing', 'calibration'])
def test_resolved_path_is_explicit_not_inherited(tmp_path, monkeypatch, inherited, mode):
    # Source metadata needs git; capture it before removing the entire caller
    # PATH. Exercise real lowering, not a hand-written child environment.
    head = registry.git_head(registry.repository_root())
    layout = registry.source_layout(registry.repository_root(), H3Implementation().lower_profile())
    monkeypatch.setattr(registry, 'git_head', lambda _: head)
    monkeypatch.setattr(registry, 'source_layout', lambda *a, **k: layout)
    if inherited is None:
        monkeypatch.delenv('PATH', raising=False)
    else:
        monkeypatch.setenv('PATH', inherited)
    plan = _plan(tmp_path, mode)
    registry.validate_resolved_asset_binding(plan, plan['fully_resolved_profile'])
    for node in plan['nodes']:
        assert node['environment']['PATH'] == '/usr/bin:/bin'


@pytest.mark.parametrize('path', [None, '', '/usr/bin:/bin:/untrusted', '/usr/local/cuda/bin', '.:/bin'])
def test_serialized_toolchain_path_tampering_fails_closed(tmp_path, path):
    plan = _plan(tmp_path)
    if path is None:
        plan['nodes'][0]['environment'].pop('PATH')
    else:
        plan['nodes'][0]['environment']['PATH'] = path
    with pytest.raises(ValueError, match='toolchain PATH'):
        validated_profile(plan, root=registry.repository_root())


def test_real_cpu_compile_link_execute_with_no_inherited_environment(tmp_path, monkeypatch):
    plan = _plan(tmp_path)
    env = dict(plan['nodes'][0]['environment'], CUDA_VISIBLE_DEVICES='')
    compiler = shutil.which('gcc', path=env['PATH'])
    linker = shutil.which('ld', path=env['PATH'])
    if not compiler or not linker:
        pytest.skip('CPU toolchain unavailable; real compile gate remains required on each execution node')
    # Neither a shell nor the parent process supplies a hidden default PATH.
    # GCC invokes its real linker here, unlike an import-only qualification.
    monkeypatch.setenv('PATH', '/nonexistent/inherited/toolchain')
    source, binary = tmp_path / 'probe.c', tmp_path / 'probe'
    source.write_text('int main(void) { return 0; }\n')
    subprocess.run([linker, '--version'], env=env, check=True, capture_output=True, timeout=10)
    subprocess.run([compiler, str(source), '-o', str(binary)], env=env,
                   check=True, capture_output=True, timeout=30)
    subprocess.run([str(binary)], env=env, check=True, capture_output=True, timeout=10)


def test_canonical_profile_bytes_are_not_rewritten(tmp_path):
    profile = registry.load_profile(H3)
    assert 'PATH' not in profile['runtime']['environment']
    assert _plan(tmp_path)['nodes'][0]['environment']['PATH'] == '/usr/bin:/bin'
