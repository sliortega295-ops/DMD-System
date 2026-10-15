"""CPU control-plane regressions; mocked fixtures never prove GPU capacity."""
import ast
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('h100_cli_summary',
    ROOT / 'scripts/distillschedule/summarize_helios_hardware_p0.py')
SUMMARY = importlib.util.module_from_spec(spec)
spec.loader.exec_module(SUMMARY)
KEY = 'DISTILLSCHEDULE_HELIOS_H100_RETENTION_API_RECEIPT'


def capability_spy(monkeypatch, tmp_path):
    monkeypatch.setenv('DISTILLSCHEDULE_HARDWARE_PROFILE', 'h100-8-b16')
    path = tmp_path / 'control_plane_only_fixture.json'
    monkeypatch.setenv(KEY, str(path))
    calls = []

    def check(observed, frozen):
        assert observed == path
        assert frozen['arm'] == 'h100-nograd-retain'
        assert frozen['variant'] == 'no-gan' and frozen['mesh_shape'] == [2, 4]
        calls.append((observed, frozen))
        return {'status': 'UNIT_TEST_SENTINEL_NOT_GPU_EVIDENCE'}

    monkeypatch.setattr(SUMMARY, 'validate_h100_retention_capability', check)
    return calls


def test_registered_request_uses_existing_capability_validator(monkeypatch, tmp_path):
    calls = capability_spy(monkeypatch, tmp_path)
    result = SUMMARY.validate_registered_h100_retention_request('no-gan', 'node-hsdp', 1)
    assert len(calls) == 1 and result['status'] == 'UNIT_TEST_SENTINEL_NOT_GPU_EVIDENCE'


@pytest.mark.parametrize('variant,mesh,batch', [
    ('no-gan', 'world16-fsdp', 1), ('no-gan', 'node-hsdp', 2),
    ('no-gan', 'node-hsdp', 5), ('gan', 'node-hsdp', 1),
])
def test_unregistered_requests_still_fail_closed(monkeypatch, tmp_path, variant, mesh, batch):
    calls = capability_spy(monkeypatch, tmp_path)
    with pytest.raises(ValueError):
        SUMMARY.validate_registered_h100_retention_request(variant, mesh, batch)
    assert calls == []


def test_a100_and_missing_capability_remain_rejected(monkeypatch, tmp_path):
    calls = capability_spy(monkeypatch, tmp_path)
    monkeypatch.setenv('DISTILLSCHEDULE_HARDWARE_PROFILE', 'a100-16-b16')
    with pytest.raises(ValueError):
        SUMMARY.validate_registered_h100_retention_request('no-gan', 'node-hsdp', 1)
    monkeypatch.setenv('DISTILLSCHEDULE_HARDWARE_PROFILE', 'h100-8-b16')
    monkeypatch.delenv(KEY)
    with pytest.raises(ValueError):
        SUMMARY.validate_registered_h100_retention_request('no-gan', 'node-hsdp', 1)
    assert calls == []


def actual_guard(script):
    tree = ast.parse((ROOT / 'scripts/distillschedule' / script).read_text())
    found = [node for node in ast.walk(tree) if isinstance(node, ast.If)
             and 'args.generator_nograd_persistent_unshard' in ast.unparse(node.test)
             and 'args.variant' in ast.unparse(node.test)]
    assert len(found) == 1
    return compile(ast.fix_missing_locations(ast.Module(body=found, type_ignores=[])), script, 'exec')


@pytest.mark.parametrize('script,flag', [
    ('run_static_fsdp2_p0_world16.py', 'on'),
    ('probe_static_fsdp2_p0_world16.py', True),
])
def test_actual_controller_and_worker_guards_delegate(monkeypatch, tmp_path, script, flag):
    calls = capability_spy(monkeypatch, tmp_path)
    monkeypatch.setitem(sys.modules, 'summarize_helios_hardware_p0', SUMMARY)
    args = SimpleNamespace(generator_nograd_persistent_unshard=flag,
                           variant='no-gan', mesh_policy='node-hsdp', rollout_batch_size=1)
    exec(actual_guard(script), {'args': args})
    assert len(calls) == 1


@pytest.mark.parametrize('script,flag', [
    ('run_static_fsdp2_p0_world16.py', 'on'),
    ('probe_static_fsdp2_p0_world16.py', True),
])
def test_legacy_gan_guard_path_does_not_change(monkeypatch, tmp_path, script, flag):
    calls = capability_spy(monkeypatch, tmp_path)
    args = SimpleNamespace(generator_nograd_persistent_unshard=flag,
                           variant='gan', mesh_policy='world16-fsdp', rollout_batch_size=1)
    exec(actual_guard(script), {'args': args})
    assert calls == []


def test_actual_worker_environment_preserves_only_validated_capability():
    tree = ast.parse((ROOT / 'scripts/distillschedule/summarize_helios_hardware_p0.py').read_text())
    prepare = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name == 'prepare_launch')
    blocks = [node for node in prepare.body if isinstance(node, ast.If)
              and ast.unparse(node.test) == 'retention_capability is not None']
    assert len(blocks) == 1, 'launcher scrub must re-export the validated receipt path'
    code = compile(ast.fix_missing_locations(ast.Module(body=blocks, type_ignores=[])),
                   'actual_worker_environment', 'exec')
    env = {}
    exec(code, {'environment': env, 'retention_capability': {'path': '/owned/validated.json'}})
    assert env == {KEY: '/owned/validated.json'}
    env = {}
    exec(code, {'environment': env, 'retention_capability': None})
    assert env == {}
