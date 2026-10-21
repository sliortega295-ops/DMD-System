"""Opt-in manual reference admission; CPU fixtures are not timing evidence."""
import ast
import inspect
import json
from pathlib import Path

import pytest

from distillschedule.cli import main
from distillschedule.strategies import wan_worker
from distillschedule.strategies.wan_strategy import WanStrategy
from distillschedule.strategies.wan_task import OriginalWanTask


def assets():
    return wan_worker.OriginalWanAssets(
        model_root='/synthetic/model', bootstrap='/synthetic/bootstrap',
        expected_bootstrap_hash='a' * 64, trace_bundle='/synthetic/trace',
        expected_trace_hash='b' * 64, negative_prompt='/synthetic/negative')


def test_manual_cli_without_execute_cannot_load_assets_or_allocate(tmp_path, monkeypatch, capsys):
    task = tmp_path / 'task.json'
    task.write_text(json.dumps(OriginalWanTask().to_dict()))
    monkeypatch.setattr(wan_worker, 'run_worker', lambda *a, **k: pytest.fail('no execution authority'))
    out = tmp_path / 'absent-output'
    assert main(['workloads', 'native', 'benchmark-manual', '--task', str(task),
                 '--assets', 'absent', '--implementation', 'absent', '--output', str(out)]) == 0
    assert json.loads(capsys.readouterr().out)['native_execution'].startswith('NOT_RUN')
    assert not out.exists()


def test_manual_cli_reaches_same_worker_with_explicit_mode(tmp_path, monkeypatch, capsys):
    task = tmp_path / 'task.json'; task.write_text(json.dumps(OriginalWanTask().to_dict()))
    spec = tmp_path / 'assets.json'; spec.write_text(json.dumps(assets().__dict__))
    strategy = WanStrategy('resident', 2048, True, 5)
    impl = tmp_path / 'manual.json'; impl.write_text(json.dumps(strategy.to_dict()))
    calls = []
    def worker(t, a, **kw):
        calls.append((t, a, kw))
        return {'status': 'SYNTHETIC_TEST_ONLY', 'GPU': 'NOT_RUN'}
    monkeypatch.setattr(wan_worker, 'run_worker', worker)
    assert main(['workloads', 'native', 'benchmark-manual', '--task', str(task),
                 '--assets', str(spec), '--implementation', str(impl), '--cycles', '2',
                 '--output', str(tmp_path / 'manual'), '--execute']) == 0
    assert len(calls) == 1
    assert calls[0][2] == dict(output=tmp_path / 'manual', mode='manual', strategy=strategy,
                              plan_path=None, cycles=2)
    assert json.loads(capsys.readouterr().out)['status'] == 'SYNTHETIC_TEST_ONLY'


@pytest.fixture
def world16(monkeypatch):
    for k, v in {'WORLD_SIZE': '16', 'LOCAL_WORLD_SIZE': '8', 'RANK': '0', 'LOCAL_RANK': '0'}.items():
        monkeypatch.setenv(k, v)


@pytest.mark.parametrize('changed', [dict(plan_path=Path('/synthetic/selected.json')),
                                    dict(cycles=3), dict(stream_path='/synthetic/stream'),
                                    dict(strategy=None)])
def test_manual_rejects_plan_stream_and_non_two_cycle_before_any_output(tmp_path, world16, changed):
    kwargs = dict(output=tmp_path / 'must-not-exist', mode='manual',
                  strategy=WanStrategy('resident', 2048, True, 5), cycles=2)
    kwargs.update(changed)
    with pytest.raises(ValueError, match='manual reference'):
        wan_worker.run_worker(OriginalWanTask(), assets(), **kwargs)
    assert not kwargs['output'].exists()


def test_saved_plan_and_calibration_admission_not_weakened(tmp_path, world16):
    for mode, strategy in [('run', WanStrategy('resident', 2048, True, 5)),
                           ('calibrate', WanStrategy('resident', 2048, True, 5))]:
        with pytest.raises(ValueError):
            wan_worker.run_worker(OriginalWanTask(), assets(), output=tmp_path / mode,
                                  mode=mode, strategy=strategy, cycles=2)
        assert not (tmp_path / mode).exists()


def test_manual_uses_uninstrumented_common_timer_and_never_fabricates_plan():
    source = inspect.getsource(wan_worker.run_worker)
    assert "mode='run' if mode == 'manual' else mode" in source
    assert "operation_order=None if document is None else tuple(document['selected']['order'])" in source
    assert "plan_sha256=None if document is None else document['plan_sha256']" in source
    assert "execution_kind='MANUAL_REFERENCE_DIRECT_NATIVE_ORDER'" in source
    timer = inspect.getsource(wan_worker.execute_cycles)
    positions = {n.targets[0].id: n.lineno for n in ast.walk(ast.parse(timer))
                 if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
    assert positions['prepared'] < positions['start'] < positions['duration'] < positions['record']
