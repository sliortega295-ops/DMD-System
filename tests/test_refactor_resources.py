"""CPU observer budget/strict-failure tests; never touch a GPU or cgroup."""
import json
import subprocess
from unittest.mock import Mock

from distillschedule.search.resources import ResourceObserver


def test_driver_query_has_bounded_startup_tolerance_without_dropping_samples(tmp_path, monkeypatch):
    from distillschedule.search import resources
    calls = []
    def query(argv, **kwargs):
        calls.append(kwargs)
        return '\n'.join(f'{i}, GPU-{i}, 40960, 40000' for i in range(8))
    monkeypatch.setattr(resources.subprocess, 'check_output', query)
    monkeypatch.setattr(resources.Path, 'read_text', lambda p: 'oom 0\noom_kill 0\n' if p.name == 'memory.events' else '1234')
    observer = ResourceObserver(tmp_path/'observer.json', 'cpu-test', 0)
    observer.sample()
    assert calls[0]['timeout'] == 30
    assert not observer.errors and len(observer.rows) == 1 and len(observer.rows[0]['gpus']) == 8


def test_shutdown_join_covers_full_query_budget(tmp_path):
    observer = ResourceObserver(tmp_path/'observer.json', 'cpu-test', 0)
    thread = Mock()
    thread.is_alive.return_value = False
    observer.thread = thread
    observer.sample = Mock()
    observer.__exit__(None, None, None)
    thread.join.assert_called_once_with(timeout=32)


def test_timeout_stays_failed_not_silently_partial_pass(tmp_path, monkeypatch):
    from distillschedule.search import resources
    def query(*args, **kwargs):
        raise subprocess.TimeoutExpired('nvidia-smi', kwargs['timeout'])
    monkeypatch.setattr(resources.subprocess, 'check_output', query)
    observer = ResourceObserver(tmp_path/'observer.json', 'cpu-test', 0)
    thread = Mock()
    thread.is_alive.return_value = False
    observer.thread = thread
    observer.sample()
    observer.__exit__(None, None, None)
    result = json.loads(observer.path.read_text())
    assert result['status'] == 'FAIL' and len(result['errors']) == 2 and not result['samples']


def test_incomplete_device_inventory_stays_failed(tmp_path, monkeypatch):
    from distillschedule.search import resources
    monkeypatch.setattr(resources.subprocess, 'check_output', lambda *a, **k: '0, GPU-0, 40960, 40000\n')
    observer = ResourceObserver(tmp_path/'observer.json', 'cpu-test', 0)
    observer.sample()
    assert not observer.rows and observer.errors == ['observer requires all eight allocated GPUs']
