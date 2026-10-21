"""CPU-only dependency identity/lowering tests; no imports of Torch or CUDA."""
import copy
import json
import os
from pathlib import Path

import pytest

from distillschedule.search.execution import NativeTrialRunner, resolve_candidate, validated_profile
from distillschedule.search.h3_implementation import H3, H3Implementation
from distillschedule.workloads.best_known import dispatch, registry


@pytest.fixture
def assets(tmp_path):
    profile = H3Implementation().lower_profile()
    value = json.loads((registry.repository_root() / 'configs/distillschedule/best_known/assets_examples' /
                        (H3 + '.json')).read_text())
    value['assets']['diffusers_source'] = dict(path='/fixture/diffusers', identity_file='identity.json', sha256='a' * 64)
    for index, node in enumerate(value['nodes']):
        node['python_path_overlays'] = []
        for name in ('dependency_a', 'dependency_b'):
            path = tmp_path / f'node{index}' / name
            path.mkdir(parents=True)
            (path / 'identity.txt').write_text(f'CPU_FIXTURE_ONLY_{index}_{name}')
            node['python_path_overlays'].append(dict(path=str(path), identity_file='identity.txt',
                                                    sha256=registry.file_sha256(path / 'identity.txt')))
    registry.validate_asset_config(value, profile)
    return value


def resolve(assets, tmp_path):
    path = tmp_path / 'assets.json'
    path.write_text(json.dumps(assets))
    return resolve_candidate(H3, H3Implementation().id, assets_path=path, run_id='dependency-cpu-only')


def test_default_identity_and_python_path_stay_compatible(assets, tmp_path, monkeypatch):
    for node in assets['nodes']:
        node.pop('python_path_overlays')
    monkeypatch.setenv('PYTHONPATH', '/untrusted/inherited/scripts')
    plain = resolve(assets, tmp_path)
    assert plain['asset_fingerprint'] == registry.fingerprint(assets['assets'])
    assert '/untrusted' not in plain['nodes'][0]['environment']['PYTHONPATH']
    for node in assets['nodes']:
        node['python_path_overlays'] = []
    explicit_empty = resolve(assets, tmp_path)
    assert explicit_empty['asset_fingerprint'] == plain['asset_fingerprint']
    assert [n['environment'] for n in explicit_empty['nodes']] == [n['environment'] for n in plain['nodes']]


def test_exact_per_node_order_and_serialized_asset_binding(assets, tmp_path):
    plan = resolve(assets, tmp_path)
    validated_profile(plan, root=registry.repository_root())
    assert plan['asset_fingerprint'] != registry.fingerprint(assets['assets'])
    for node, spec in zip(plan['nodes'], assets['nodes']):
        assert node['environment']['PYTHONPATH'].split(os.pathsep)[:2] == [o['path'] for o in spec['python_path_overlays']]
        assert len(registry.verify_python_path_overlays(spec)) == 2


@pytest.mark.parametrize('change', ['relative', 'root', 'traversal', 'path_separator', 'nul', 'missing_identity',
    'absolute_identity', 'identity_traversal', 'empty_identity', 'bad_hash', 'extra_key', 'not_list', 'duplicate'])
def test_overlay_schema_fail_closed(assets, change):
    overlay = assets['nodes'][0]['python_path_overlays'][0]
    if change == 'relative': overlay['path'] = 'relative'
    if change == 'root': overlay['path'] = '/'
    if change == 'traversal': overlay['path'] = '/allowed/../scripts'
    if change == 'path_separator': overlay['path'] = '/allowed:/scripts'
    if change == 'nul': overlay['path'] = '/allowed\0'
    if change == 'missing_identity': overlay.pop('identity_file')
    if change == 'absolute_identity': overlay['identity_file'] = '/outside/file'
    if change == 'identity_traversal': overlay['identity_file'] = '../outside'
    if change == 'empty_identity': overlay['identity_file'] = ''
    if change == 'bad_hash': overlay['sha256'] = 'unverified'
    if change == 'extra_key': overlay['environment'] = {'H3_OVERRIDE': '1'}
    if change == 'not_list': assets['nodes'][0]['python_path_overlays'] = 'implicit'
    if change == 'duplicate': assets['nodes'][0]['python_path_overlays'].append(copy.deepcopy(overlay))
    with pytest.raises(ValueError):
        registry.validate_asset_config(assets, H3Implementation().lower_profile())


@pytest.mark.parametrize('change', ['sha', 'path', 'order', 'environment', 'payload'])
def test_serialized_plan_or_real_identity_tampering_rejected(assets, tmp_path, change):
    plan = resolve(assets, tmp_path)
    overlay = plan['asset_config']['nodes'][0]['python_path_overlays'][0]
    if change == 'sha': overlay['sha256'] = 'b' * 64
    if change == 'path': overlay['path'] += '-different'
    if change == 'order': plan['asset_config']['nodes'][0]['python_path_overlays'].reverse()
    if change == 'environment': plan['nodes'][0]['environment']['PYTHONPATH'] = '/injected/scripts'
    if change == 'payload':
        (Path(overlay['path']) / overlay['identity_file']).write_text('CHANGED_CPU_FIXTURE')
        with pytest.raises(ValueError, match='fingerprint mismatch'):
            registry.verify_python_path_overlays(plan['asset_config']['nodes'][0])
    else:
        with pytest.raises(ValueError):
            validated_profile(plan, root=registry.repository_root())


def test_local_gate_requires_explicit_node_and_checks_local_overlay_only(assets, tmp_path, monkeypatch):
    plan = resolve(assets, tmp_path)
    # Only unrelated native assets/upstream setup are absent from this CPU
    # fixture. Exercise the real serialized and local dependency hash gates.
    monkeypatch.setattr(dispatch.subprocess, 'check_output', lambda *a, **k: '')
    monkeypatch.setattr(dispatch, 'git_head', lambda *a: plan['integration_head'])
    monkeypatch.setattr(dispatch, 'source_layout', lambda *a, **k: {'CPU_FIXTURE': True})
    monkeypatch.setattr(dispatch, 'verify_assets', lambda *a: [])
    with pytest.raises(ValueError, match='explicit local node_rank'):
        dispatch.validate_local(plan, root=registry.repository_root())
    remote = assets['nodes'][1]['python_path_overlays'][0]
    (Path(remote['path']) / remote['identity_file']).unlink()
    assert len(dispatch.validate_local(plan, root=registry.repository_root(), node_rank=0)['python_path_overlays']) == 2
    with pytest.raises(ValueError, match='fingerprint mismatch'):
        dispatch.validate_local(plan, root=registry.repository_root(), node_rank=1)


def test_old_asset_cost_identity_cannot_admit_new_dependencies(assets, tmp_path):
    identity = dict(source_head=registry.git_head(registry.repository_root()),
                    asset_fingerprint=registry.fingerprint(assets['assets']))
    with pytest.raises(ValueError, match='asset identity mismatch'):
        NativeTrialRunner(workload=H3, identity=identity, assets=assets, output=tmp_path,
            run_prefix='CPU_ONLY', reference_id='joint-h3v1-d2-n128', bootstrap_peer='joint-h3v1-d1-n128')


def test_cost_lowering_rejects_changed_overlay_identity(tmp_path):
    from test_refactor_native_macro_routing import _calibration_fixture
    from distillschedule.search.native_macro_adapter import lower_prediction
    problem, costs, prediction, path = _calibration_fixture(tmp_path, H3)
    assets = json.loads(path.read_text())
    assets['nodes'][0]['python_path_overlays'] = [dict(path='/fixture/dependency', identity_file='version.txt', sha256='a' * 64)]
    path.write_text(json.dumps(assets))
    with pytest.raises(ValueError, match='assets changed'):
        lower_prediction(prediction, problem, costs, run_id='CPU_ONLY', assets_path=path)


def test_other_launchers_cannot_silently_drop_dependencies(assets):
    with pytest.raises(ValueError, match='only by the H3 native launcher'):
        registry.validate_asset_config(assets, registry.load_profile('helios14b_nogan_best'))
