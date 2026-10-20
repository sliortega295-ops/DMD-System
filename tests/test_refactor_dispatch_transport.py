"""CPU-only native-dispatch transport tests; no SSH, GPU, or model execution."""
import base64
import io
import json
from pathlib import Path
import random
import shlex
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from distillschedule.workloads.best_known import dispatch

TTY = 'argv-json-tty-v1'
PREFIX = 'DMD_NATIVE_DISPATCH_JSON='


def _frame(action='_node', returncode=0):
    payload = json.dumps({'action': action, 'returncode': returncode}).encode()
    return PREFIX + base64.b64encode(payload).decode() + '\n'


def _receipt(tmp_path, modes=(TTY, TTY)):
    specs = [{'host': f'allocated-node{n}', 'ssh_port': 2200 + n,
              'address_family': 'ipv4', 'repo_root': f'/source/node{n}',
              'python_env': '/pinned/env'} for n in range(2)]
    for spec, mode in zip(specs, modes):
        if mode is not None:
            spec['rpc_transport'] = mode
    # Existing compiler copies SSH options, not rpc_transport. The saved
    # asset config is the authoritative transport selection for old plans too.
    nodes = [{'host': s['host'], 'ssh_port': s['ssh_port'],
              'address_family': s['address_family'], 'repo_root': s['repo_root'],
              'python': '/pinned/env/bin/python',
              'environment': {'DECLARED_NATIVE_FLAG': 'canonical'},
              'command': ['bash', s['repo_root'] + '/scripts/canonical.sh']}
             for s in specs]
    return {'nodes': nodes, 'asset_config': {'nodes': specs,
                'run_root': str(tmp_path / 'remote'), 'timeout_seconds': 60},
            'run_id': 'CPU_ONLY_NOT_GPU', 'integration_head': 'a' * 40,
            'profile_id': 'wan13b_nogan_resident_best'}


class _Input(io.StringIO):
    def close(self):
        self.payload = self.getvalue()
        super().close()


class _Process:
    def __init__(self, returncode=0, *, wait_timeout=False):
        self.returncode = returncode
        self.stdin = _Input()
        self.terminated = self.killed = False
        self.waits = []
        self.wait_timeout = wait_timeout

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        self.waits.append(timeout)
        if self.wait_timeout and timeout is not None:
            raise subprocess.TimeoutExpired('mock-owned-ssh', timeout)
        return self.returncode


class _Transport:
    def __init__(self, receipt, processes=None, *, node_frame=True, collect_frame=True,
                 collect_error=None, cancel_error=None, fail_start=None):
        self.receipt = receipt
        self.processes = processes or [_Process(), _Process()]
        self.node_frame = node_frame
        self.collect_frame = collect_frame
        self.collect_error = collect_error
        self.cancel_error = cancel_error
        self.fail_start = fail_start
        self.starts = []
        self.collections = []
        self.cancels = []

    def popen(self, command, **kwargs):
        n = len(self.starts)
        self.starts.append((command, kwargs))
        if n == self.fail_start:
            raise OSError('CPU_ONLY startup failure')
        if self.node_frame and '-tt' in command:
            kwargs['stdout'].write('gateway banner\r\nworker logs unchanged\r\n' + _frame())
        return self.processes[n]

    def run(self, command, **kwargs):
        if 'cancel_remote_stage.sh' in command[-1]:
            self.cancels.append((command, kwargs))
            if self.cancel_error:
                raise self.cancel_error
        else:
            self.collections.append((command, kwargs))
            if self.collect_error:
                raise self.collect_error
            if '-tt' in command and self.collect_frame:
                kwargs['stdout'].write('collector output\r\n' + _frame('_collect'))
        return SimpleNamespace(returncode=0)

    def execute(self, tmp_path):
        with patch.object(dispatch.subprocess, 'Popen', side_effect=self.popen), \
             patch.object(dispatch.subprocess, 'run', side_effect=self.run):
            return dispatch.run_resolved(self.receipt, coordinator_output=tmp_path / 'coordinator')


@pytest.mark.parametrize('modes', [(TTY, TTY), (TTY, None), (None, TTY), (None, None)])
def test_dispatch_routes_each_saved_asset_transport_without_changing_default_stdin(tmp_path, modes):
    receipt = _receipt(tmp_path, modes)
    transport = _Transport(receipt)
    assert transport.execute(tmp_path) == 0
    for n, ((command, kwargs), proc) in enumerate(zip(transport.starts, transport.processes)):
        assert command[command.index('-p') + 1] == str(2200 + n)
        assert '-4' in command and '-oControlMaster=no' in command
        if modes[n] == TTY:
            assert command[-2] == receipt['nodes'][n]['host']
            assert '-n' in command and '-tt' in command and command[-1].endswith('; exit')
            assert kwargs['stdin'] == subprocess.DEVNULL
            assert not proc.stdin.closed
        else:
            assert '-tt' not in command and '-n' not in command
            assert kwargs['stdin'] == subprocess.PIPE and proc.stdin.closed
            assert json.loads(proc.stdin.payload) == {'receipt': receipt, 'node_rank': n}
        assert kwargs['stdout'].closed and proc.waits == [10]
        assert not proc.terminated and not proc.killed
    command, kwargs = transport.collections[0]
    if modes[0] == TTY:
        assert '-tt' in command and '-n' in command and command[-1].endswith('; exit')
        assert kwargs['stdin'] == subprocess.DEVNULL and kwargs['stdout'].closed
        assert kwargs.get('input') is None
    else:
        assert '-tt' not in command and json.loads(kwargs['input']) == receipt
    assert not transport.cancels
    result = json.loads((tmp_path / 'coordinator/dispatch-result.json').read_text())
    assert result['status'] == 'DISPATCH_AND_NATIVE_SUMMARY_COMPLETE'
    assert result['formal_speedup_eligible'] is False


@pytest.mark.parametrize('action', ['_node', '_collect'])
def test_framed_request_rehydrates_exact_payload_in_source_worker_and_streams_output(tmp_path, action):
    root = tmp_path / "source with 'quotes'"
    worker = root / 'scripts/distillschedule/run_best_known_workload.py'
    worker.parent.mkdir(parents=True)
    result = tmp_path / 'request.json'
    worker.write_text('import json,os,sys\nfrom pathlib import Path\n'
        f'assert sys.argv[1:]==[{action!r}]\n'
        'assert os.environ["PYTHONDONTWRITEBYTECODE"]=="1"\n'
        f'assert os.environ["LD_LIBRARY_PATH"]=={str(Path(sys.executable).parent.parent / "lib")!r}\n'
        f'Path({str(result)!r}).write_text(json.dumps(json.load(sys.stdin)))\n'
        'print("worker stdout stays streamed",end="",flush=True)\nraise SystemExit(0)\n')
    node = {'repo_root': str(root), 'python': sys.executable, 'host': 'never-contacted'}
    payload = {'unicode': '独立样本', 'quoted': "'\";exit", 'values': [0, 1]}
    command, stdin_payload = dispatch._worker_request(node, action, payload, transport=TTY)
    assert stdin_payload is None and '-n' in command and '-tt' in command
    # Execute only the exact generated shell bootstrap, never SSH or a model.
    local = subprocess.run(['bash', '-c', command[-1]], capture_output=True, text=True, timeout=10)
    assert local.returncode == 0, local.stderr
    assert json.loads(result.read_text()) == payload
    assert 'worker stdout stays streamed' in local.stdout
    log = tmp_path / 'worker.log'
    log.write_text('gateway banner\r\n' + local.stdout.replace('\n', '\r\n'))
    dispatch._check_worker_frame(log, action)


@pytest.mark.parametrize('action', ['_node', '_collect'])
def test_framed_request_matches_native_rpc_interpreter_bootstrap_and_preserves_stdin(tmp_path, action):
    node = {'repo_root': str(tmp_path / "source with 'quotes'"),
            'python': "/packed env with 'quotes'/bin/python", 'host': 'never-contacted'}
    payload = {'fixture': 'CPU_ONLY'}
    command, stdin_payload = dispatch._worker_request(node, action, payload, transport=TTY)
    words = shlex.split(command[-1][:-len('; exit')])
    assert words[:5] == ['env', 'PYTHONDONTWRITEBYTECODE=1',
        "LD_LIBRARY_PATH=/packed env with 'quotes'/lib", node['python'], '-c']
    assert stdin_payload is None
    default, default_input = dispatch._worker_request(node, action, payload, transport='stdin')
    worker = str(Path(node['repo_root']) / 'scripts/distillschedule/run_best_known_workload.py')
    assert default == [*dispatch.ssh_command(node), shlex.join([node['python'], worker, action])]
    assert default_input == json.dumps(payload)


@pytest.mark.parametrize('failure', ['raise SystemExit(23)', 'raise RuntimeError("CPU_ONLY failure")'])
def test_framed_request_preserves_worker_failure_and_never_prints_success_frame(tmp_path, failure):
    worker = tmp_path / 'scripts/distillschedule/run_best_known_workload.py'
    worker.parent.mkdir(parents=True)
    worker.write_text('print("owned worker entered",flush=True)\n' + failure + '\n')
    node = {'repo_root': str(tmp_path), 'python': sys.executable, 'host': 'never-contacted'}
    command, _ = dispatch._worker_request(node, '_node', {'node_rank': 0}, transport=TTY)
    local = subprocess.run(['bash', '-c', command[-1]], capture_output=True, text=True, timeout=10)
    assert local.returncode == (23 if 'SystemExit' in failure else 1)
    assert PREFIX not in local.stdout


@pytest.mark.parametrize('action', ['_search', '_cancel', 'arbitrary-executor'])
def test_dispatch_bootstrap_rejects_any_non_native_worker_action(tmp_path, action):
    with pytest.raises(ValueError, match='worker action'):
        dispatch._worker_request({'host': 'never-contacted'}, action, {}, transport=TTY)


def test_dispatch_rejects_unknown_second_transport_before_launch_or_evidence(tmp_path):
    receipt = _receipt(tmp_path, (TTY, 'arbitrary-shell'))
    transport = _Transport(receipt)
    with pytest.raises(ValueError, match='transport'):
        transport.execute(tmp_path)
    assert not transport.starts and not (tmp_path / 'coordinator').exists()


def test_framed_dispatch_rejects_oversized_argv_payload_before_launch(tmp_path):
    receipt = _receipt(tmp_path)
    receipt['uncompressible_cpu_fixture'] = random.Random(0).randbytes(120_000).hex()
    _, stdin_payload = dispatch._worker_request(receipt['nodes'][0], '_collect', receipt,
                                                transport='stdin')
    assert json.loads(stdin_payload) == receipt  # The default retains no new argv bound.
    transport = _Transport(receipt)
    with pytest.raises(ValueError, match='argv transport bound'):
        transport.execute(tmp_path)
    assert not transport.starts and not (tmp_path / 'coordinator').exists()


@pytest.mark.parametrize('output', ['no frame\n', PREFIX + '%%%\n',
    _frame() + _frame(), _frame('_collect'), _frame(returncode=1), _frame(returncode=False)])
def test_framed_dispatch_missing_invalid_duplicate_or_wrong_exit_frames_fail_closed(tmp_path, output):
    path = tmp_path / 'stdout.log'
    path.write_text(output)
    with pytest.raises(ValueError):
        dispatch._check_worker_frame(path, '_node')


def _assert_owned_cancels(transport, receipt, modes):
    assert len(transport.cancels) == 2
    remote = Path(receipt['asset_config']['run_root']) / receipt['run_id']
    for n, (command, kwargs) in enumerate(transport.cancels):
        assert str(remote / f'owned-node{n}.pid') in command[-1]
        assert receipt['nodes'][n]['repo_root'] + '/scripts/distillschedule/cancel_remote_stage.sh' in command[-1]
        assert 'pkill' not in command[-1] and 'killall' not in command[-1]
        assert kwargs['timeout'] == 60 and kwargs['check'] is False
        if modes[n] == TTY:
            assert '-n' in command and '-tt' in command and command[-1].endswith('; exit')
            assert kwargs['stdin'] == subprocess.DEVNULL
        else:
            assert '-tt' not in command


def test_successful_ssh_without_worker_completion_is_rejected_and_owned_stages_cancelled(tmp_path):
    modes = (TTY, TTY)
    receipt = _receipt(tmp_path, modes)
    transport = _Transport(receipt, node_frame=False)
    with pytest.raises(ValueError, match='exactly one framed'):
        transport.execute(tmp_path)
    _assert_owned_cancels(transport, receipt, modes)
    assert not transport.collections
    assert not (tmp_path / 'coordinator/dispatch-result.json').exists()
    assert all(kwargs['stdout'].closed for _, kwargs in transport.starts)


@pytest.mark.parametrize('output', ['', PREFIX + '%%%\n', _frame() + _frame(), _frame('_collect')])
@pytest.mark.parametrize('completed_node', [0, 1])
def test_completed_framed_peer_is_rejected_before_waiting_for_running_peer(tmp_path, output, completed_node):
    receipt = _receipt(tmp_path)
    processes = [_Process(0 if i == completed_node else None) for i in (0, 1)]
    transport = _Transport(receipt, processes, node_frame=False)
    spawn = transport.popen
    def popen(command, **kwargs):
        process = spawn(command, **kwargs)
        if len(transport.starts) - 1 == completed_node:
            kwargs['stdout'].write(output)
        return process
    with patch.object(transport, 'popen', side_effect=popen), \
         patch.object(dispatch.time, 'monotonic', side_effect=[0, 100]), \
         patch.object(dispatch.time, 'sleep') as sleep:
        with pytest.raises(ValueError):
            transport.execute(tmp_path)
    assert not sleep.called
    _assert_owned_cancels(transport, receipt, (TTY, TTY))
    assert processes[1 - completed_node].terminated and processes[1 - completed_node].waits == [10]
    assert not transport.collections
    assert all(kwargs['stdout'].closed for _, kwargs in transport.starts)


@pytest.mark.parametrize('peer_pending', [True, False])
def test_framed_peers_are_validated_once_including_instant_exit_race(tmp_path, peer_pending):
    receipt = _receipt(tmp_path)
    processes = [_Process(0), _Process(None if peer_pending else 0)]
    transport = _Transport(receipt, processes)
    check = dispatch._check_worker_frame
    peer_state_at_first_check = []
    def check_frame(path, action):
        if path.name == 'ssh-node0.log':
            peer_state_at_first_check.append(processes[1].poll())
        return check(path, action)
    def finish_peer(_):
        processes[1].returncode = 0
    with patch.object(dispatch, '_check_worker_frame', side_effect=check_frame) as checked, \
         patch.object(dispatch.time, 'monotonic', return_value=0), \
         patch.object(dispatch.time, 'sleep', side_effect=finish_peer):
        assert transport.execute(tmp_path) == 0
    assert peer_state_at_first_check == [None if peer_pending else 0]
    paths = [call.args[0].name for call in checked.call_args_list]
    assert paths == ['ssh-node0.log', 'ssh-node1.log', 'ssh-collect.log']
    assert not transport.cancels


def test_successful_collector_ssh_without_completion_is_rejected_and_owned_stages_cancelled(tmp_path):
    receipt = _receipt(tmp_path)
    transport = _Transport(receipt, collect_frame=False)
    with pytest.raises(ValueError, match='exactly one framed'):
        transport.execute(tmp_path)
    _assert_owned_cancels(transport, receipt, (TTY, TTY))
    assert len(transport.collections) == 1
    assert transport.collections[0][1]['stdout'].closed
    assert not (tmp_path / 'coordinator/dispatch-result.json').exists()


@pytest.mark.parametrize('mode', ['stdin', TTY])
def test_cancel_transport_preserves_exact_owned_pid_argument_and_script_exit(tmp_path, mode):
    root = tmp_path / "source with 'quotes'"
    cancel = root / 'scripts/distillschedule/cancel_remote_stage.sh'
    cancel.parent.mkdir(parents=True)
    captured = tmp_path / 'owned-argument.txt'
    # A harmless local fixture, not the real cancellation script or any SSH.
    cancel.write_text(f'printf "%s" "$1" > {shlex.quote(str(captured))}\nexit 26\n')
    pid = tmp_path / "owned path with 'quotes'.pid"
    node = {'repo_root': str(root), 'host': 'never-contacted'}
    command = dispatch._cancel_command(node, pid, transport=mode)
    local = subprocess.run(['bash', '-c', command[-1]], capture_output=True, text=True, timeout=10)
    assert local.returncode == 26
    assert captured.read_text() == str(pid)


@pytest.mark.parametrize('error', [subprocess.CalledProcessError(17, 'mock-collect'),
                                  subprocess.TimeoutExpired('mock-collect', 300)])
def test_collector_failure_or_timeout_retains_failure_and_cancels_only_owned_stages(tmp_path, error):
    modes = (TTY, None)
    receipt = _receipt(tmp_path, modes)
    transport = _Transport(receipt, collect_error=error)
    with pytest.raises(type(error)):
        transport.execute(tmp_path)
    _assert_owned_cancels(transport, receipt, modes)
    assert transport.collections[0][1]['stdout'].closed
    assert not (tmp_path / 'coordinator/dispatch-result.json').exists()


@pytest.mark.parametrize('modes', [(TTY, TTY), (None, None)])
def test_peer_failure_preserves_owned_cancellation_and_local_reaping(tmp_path, modes):
    receipt = _receipt(tmp_path, modes)
    processes = [_Process(17), _Process(None)]
    transport = _Transport(receipt, processes)
    terminate = processes[1].terminate
    def terminate_after_owned_cancel():
        assert len(transport.cancels) == 2
        assert 'owned-node1.pid' in transport.cancels[-1][0][-1]
        terminate()
    with patch.object(processes[1], 'terminate', side_effect=terminate_after_owned_cancel):
        with pytest.raises(RuntimeError, match='peer exited'):
            transport.execute(tmp_path)
    _assert_owned_cancels(transport, receipt, modes)
    assert not processes[0].terminated and processes[1].terminated
    assert all(p.waits == [10] for p in processes)
    assert all(kwargs['stdout'].closed for _, kwargs in transport.starts)
    assert not transport.collections


@pytest.mark.parametrize('error', [OSError('mock-cancel'), subprocess.TimeoutExpired('mock-cancel', 60)])
def test_cancel_transport_failure_still_terminates_kills_and_reaps_owned_ssh(tmp_path, error):
    receipt = _receipt(tmp_path)
    processes = [_Process(17), _Process(None, wait_timeout=True)]
    transport = _Transport(receipt, processes, cancel_error=error)
    with pytest.raises(RuntimeError, match='peer exited'):
        transport.execute(tmp_path)
    _assert_owned_cancels(transport, receipt, (TTY, TTY))
    assert processes[1].terminated and processes[1].killed
    assert processes[1].waits == [10, None]
    assert all(kwargs['stdout'].closed for _, kwargs in transport.starts)


def test_bounded_dispatch_timeout_cancels_both_owned_stages(tmp_path):
    receipt = _receipt(tmp_path)
    processes = [_Process(None), _Process(None)]
    transport = _Transport(receipt, processes)
    with patch.object(dispatch.time, 'monotonic', side_effect=[0, 100]):
        with pytest.raises(TimeoutError, match='bounded workload timeout'):
            transport.execute(tmp_path)
    _assert_owned_cancels(transport, receipt, (TTY, TTY))
    assert all(p.terminated and p.waits == [10] for p in processes)


def test_second_spawn_failure_closes_all_logs_and_cancels_first_owned_stage(tmp_path):
    receipt = _receipt(tmp_path)
    processes = [_Process(None), _Process()]
    transport = _Transport(receipt, processes, fail_start=1)
    with pytest.raises(OSError, match='startup failure'):
        transport.execute(tmp_path)
    assert len(transport.cancels) == 1 and 'owned-node0.pid' in transport.cancels[0][0][-1]
    assert processes[0].terminated and processes[0].waits == [10]
    assert all(kwargs['stdout'].closed for _, kwargs in transport.starts)


def test_execute_node_still_validates_then_uses_clean_environment_and_owned_wrapper(tmp_path, monkeypatch):
    receipt = _receipt(tmp_path)
    receipt['nodes'][0]['repo_root'] = str(tmp_path / 'source')
    node = receipt['nodes'][0]
    events = []
    monkeypatch.setenv('DISTILLSCHEDULE_UNDECLARED_PERFORMANCE_FLAG', 'must-not-inherit')
    monkeypatch.setenv('LD_LIBRARY_PATH', '/untrusted/library')
    monkeypatch.setenv('PYTHONPATH', '/untrusted/python')
    def validate(value, *, root, node_rank):
        assert value is receipt and root == Path(node['repo_root']) and node_rank == 0
        events.append('validated')
        return {'status': 'CPU_ONLY'}
    def call(command, **kwargs):
        assert events == ['validated']
        events.append('owned-wrapper')
        expected_pid = Path(receipt['asset_config']['run_root']) / receipt['run_id'] / 'owned-node0.pid'
        assert command == ['bash', node['repo_root'] + '/scripts/distillschedule/remote_stage_wrapper.sh',
                           str(expected_pid), *node['command']]
        assert kwargs['cwd'] == Path(node['repo_root'])
        assert kwargs['env'] == dispatch.clean_environment(node)
        assert kwargs['env']['DECLARED_NATIVE_FLAG'] == 'canonical'
        assert 'DISTILLSCHEDULE_UNDECLARED_PERFORMANCE_FLAG' not in kwargs['env']
        assert 'LD_LIBRARY_PATH' not in kwargs['env'] and 'PYTHONPATH' not in kwargs['env']
        return 23
    monkeypatch.setattr(dispatch, 'validate_local', validate)
    monkeypatch.setattr(dispatch.subprocess, 'call', call)
    assert dispatch.execute_node({'receipt': receipt, 'node_rank': 0}) == 23
    result = Path(receipt['asset_config']['run_root']) / receipt['run_id'] / 'node-exit.0.json'
    assert json.loads(result.read_text())['return_code'] == 23


@pytest.mark.parametrize('tamper', ['launcher', 'environment'])
def test_local_canonical_launcher_and_environment_checks_still_reject_tampering(tmp_path, monkeypatch, tamper):
    receipt = _receipt(tmp_path)
    profile = {'id': receipt['profile_id'], 'contract': {},
               'runtime': {'environment': {'DECLARED_NATIVE_FLAG': 'canonical'},
                           'launcher': 'scripts/canonical.sh'}}
    receipt.update(mode='correctness', profile_sha256=dispatch.fingerprint(profile))
    if tamper == 'launcher':
        receipt['nodes'][0]['command'][1] = '/unregistered/launcher.sh'
    else:
        receipt['nodes'][0]['environment']['DECLARED_NATIVE_FLAG'] = 'override'
    monkeypatch.setattr(dispatch, 'load_profile', lambda *a, **kw: profile)
    monkeypatch.setattr(dispatch, 'validate_resolved_asset_binding', lambda *a: None)
    monkeypatch.setattr(dispatch, 'git_head', lambda *a: receipt['integration_head'])
    monkeypatch.setattr(dispatch.subprocess, 'check_output', lambda *a, **kw: '')
    with pytest.raises(ValueError, match='noncanonical launcher|canonical environment override'):
        dispatch.validate_local(receipt, root=tmp_path, node_rank=0)


def test_host_ssh_environment_drops_python_loader_overrides_without_mutating_parent(monkeypatch):
    monkeypatch.setenv('LD_LIBRARY_PATH', '/isolated-python/lib')
    monkeypatch.setenv('LD_PRELOAD', '/isolated-python/lib/override.so')
    monkeypatch.setenv('SSH_AUTH_SOCK', '/caller/agent')
    actual = dispatch.ssh_process_environment()
    assert 'LD_LIBRARY_PATH' not in actual and 'LD_PRELOAD' not in actual
    assert actual['SSH_AUTH_SOCK'] == '/caller/agent'
    import os
    assert os.environ['LD_LIBRARY_PATH'] == '/isolated-python/lib'


@pytest.mark.parametrize('fail', [False, True])
def test_each_dispatch_ssh_including_collection_and_cancel_has_host_safe_loader(tmp_path, monkeypatch, fail):
    monkeypatch.setenv('LD_LIBRARY_PATH', '/isolated-python/lib')
    monkeypatch.setenv('LD_PRELOAD', '/isolated-python/lib/override.so')
    receipt = _receipt(tmp_path)
    transport = _Transport(receipt, processes=[_Process(17 if fail else 0), _Process()])
    if fail:
        with pytest.raises((RuntimeError, ValueError)):
            transport.execute(tmp_path)
        assert transport.cancels
    else:
        assert transport.execute(tmp_path) == 0
        assert transport.collections
    for _, kwargs in transport.starts + transport.collections + transport.cancels:
        assert 'LD_LIBRARY_PATH' not in kwargs['env'] and 'LD_PRELOAD' not in kwargs['env']
    # Remote Python keeps its independently declared environment in argv.
    assert 'LD_LIBRARY_PATH=/pinned/env/lib' in transport.starts[0][0][-1]
