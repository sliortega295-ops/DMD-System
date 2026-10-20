#!/usr/bin/env python3
"""Default CPU-only preparation, then a locked immutable H3 three-arm campaign.

An explicit --execute and a real, hash-qualified common bootstrap are required.
The existing H3 launcher/entrypoint do the model math. This coordinator supplies
ownership, bounded lifecycle and small receipts, never alternate model code.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import statistics
import sys
import subprocess
import time
import traceback
from h100_campaign_lock_adapter import open_campaign_lock

ROOT = Path('/home/jiangzhikun/yongyan_liu/Helios-H100')
CAMPAIGN = ROOT / 'research_artifacts/five_workload_h100_20260930'
PREPARED = CAMPAIGN / 'h100_resident_baselines_20261014_r1/artifacts/h3_three_arm_prepare_r3'
GUARD = Path('/home/jiangzhikun/yongyan_liu/Helios-H100/research_artifacts/five_workload_h100_20260930/h100_resident_baselines_20261014_r1/scripts/guard_h3_three_arm_bootstrap_r3.py')
GUARD_SHA = 'fe5de111e544410c3b9b3144bab1b26bb6bef945b726b0298f52bc8f79070e78'
LAUNCHER_SHA = '88fd660a33433e3eba05a981f0119a0002f3f992b2a6bcb3dbf138aa4a05cd46'
VERIFIER = CAMPAIGN / 'h3_common_admission_prepare_r1/scripts/verify_h3_assets_r1.py'
VERIFIER_SHA = '7a3cf8e794d6e7c58941d7a0f95474462802fc859f0160ba64699919a7271430'
ASSET_RECEIPT = CAMPAIGN / 'h3_full_asset_gate_r1/asset_receipt.json'
ASSET_RECEIPT_SHA = '74686b2b09cd0542a9bf5c4975641700e2f292be12bdaf4a00fa55883823b5d2'
BOUNDARY = 'H100_ONLY_COMMON_START_NOT_A100_INITIAL_STATE_REPLAY'


def load_guard():
    sys.path.insert(0, str(PREPARED / 'scripts'))
    from h3_h100_recording_entry_r1 import file_sha
    if file_sha(PREPARED / 'scripts/h3_h100_recording_entry_r1.py') != '953d104bf24db98a5919794a89b1f7d4d3b0130b731ec431d014b6867fab4660':
        raise ValueError('actual reviewed original observer source SHA changed')
    if file_sha(GUARD) != GUARD_SHA:
        raise ValueError('actual audited guard source SHA changed')
    spec = importlib.util.spec_from_file_location('h3_campaign_guard', GUARD)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def write_new(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True); handle.write('\n')


def proc_record(pid):
    try:
        data = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()
        return {'pid': int(pid), 'state': data[0], 'ppid': int(data[1]), 'start_ticks': data[19]}
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None


def owned_alive(record):
    actual = proc_record(record['pid'])
    return bool(actual and actual['start_ticks'] == record['start_ticks'] and actual['state'] != 'Z')


def discover_owned_children(owned):
    """Only descendants of a still-live exact PID/start-tick handle."""
    records = [proc_record(int(path.name)) for path in Path('/proc').iterdir() if path.name.isdecimal()]
    records = [record for record in records if record]
    for _ in range(4):
        parents = {pid for pid, record in owned.items() if owned_alive(record)}
        additions = [record for record in records if record['ppid'] in parents and record['pid'] not in owned]
        if not additions:
            break
        owned.update({record['pid']: record for record in additions})


def cleanup_owned(owned):
    """Never take PIDs from nvidia-smi and never signal a reused/foreign PID."""
    discover_owned_children(owned)
    for signum, seconds in ((signal.SIGTERM, 30), (signal.SIGKILL, 10)):
        for record in list(owned.values()):
            if owned_alive(record):
                try: os.kill(record['pid'], signum)
                except ProcessLookupError: pass
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline and any(owned_alive(row) for row in owned.values()):
            time.sleep(0.2)
    return [row for row in owned.values() if owned_alive(row)]


def aggregate_arm(rows, expected_adapter_shas=None, expected_sample_order=None):
    if len(rows) != 8 or sorted(row.get('rank') for row in rows) != list(range(8)):
        raise ValueError('eight actual unique physical rank receipts required')
    identities = []
    for row in rows:
        rank = row['rank']; census = row['actual_census']
        if (row['status'] != 'PASS_REAL_WORKER_CENSUS_REPLAY_UPDATES_FINITE'
                or row['logical_ranks'] != [rank, rank + 8]
                or census['forward_counts'] != {'student': 48, 'fake': 12, 'teacher': 2}
                or census['grad_forward_counts'] != {'student': 2, 'fake': 10, 'teacher': 0}
                or census['backward_counts'] != {'student': 2, 'fake': 10}
                or row['actual_optimizer_commits'] != {'student': 1, 'fake': 5}
                or row['actual_scheduler_commits'] != {'student': 1, 'fake': 5}
                or census['fixed_end_step_idx'] != 3):
            raise ValueError('actual rank-qualified logical/physical update census mismatch')
        samples = census['samples']
        expected_stages = [stage for stage in ('student', 'fake_0', 'fake_1', 'fake_2', 'fake_3', 'fake_4') for _ in range(2)]
        if ([sample['stage'] for sample in samples] != expected_stages
                or [sample['logical_rank'] for sample in samples] != [rank, rank + 8] * 6):
            raise ValueError('actual sample stage/rank order changed')
        if expected_sample_order is not None:
            for ordinal, sample in enumerate(samples):
                expected = expected_sample_order[str(sample['logical_rank'])][ordinal // 2]
                if sample.get('source_identity') != expected['identity']:
                    raise ValueError('actual consumed sample differs from immutable bootstrap order')
        expected_grid = {'checkpoint_wrap_count': 600, 'scoped_execution_count': 1200,
                         'cache_hit_count': 1200, 'cache_miss_count': 0, 'missing_key_count': 0}
        if (row.get('Grid_replay_delta') != expected_grid or row.get('activation_policy') != 'none'
                or row.get('registered_adaln_parameters') != 0
                or row.get('dropped_adaln_parameter_numel', 0) < 12_000_000_000):
            raise ValueError('actual Grid replay/parameter-residency receipt incomplete')
        states = row.get('local_state_finite_post_timer', {})
        if set(states) != {'student_parameters', 'fake_parameters', 'student_Adam', 'fake_Adam'} or any(
                value.get('status') != 'PASS_ACTUAL_LOCAL_SHARDS_FINITE' for value in states.values()):
            raise ValueError('actual trainable/Adam finite gate incomplete')
        if expected_adapter_shas is not None:
            observed = {value['role']: value['file_sha256'] for value in row['initial_adapter_actual_load_receipts']}
            if observed != expected_adapter_shas:
                raise ValueError('rank did not load the common bootstrap adapter identity')
        identities.extend(sample['identity'] for sample in samples)
        timer = row['timer']
        if not all(math.isfinite(timer[key]) for key in ('start_perf_seconds', 'stop_perf_seconds', 'rank_wall_seconds')):
            raise ValueError('nonfinite exact original timer')
        if timer['rank_wall_seconds'] <= 0 or abs(timer['stop_perf_seconds'] - timer['start_perf_seconds'] - timer['rank_wall_seconds']) > 1e-6:
            raise ValueError('original timer start/stop/duration do not reconcile')
    if len(identities) != 96 or len(set(identities)) != 96:
        raise ValueError('actual96 independent logical samples required')
    wall = max(row['timer']['stop_perf_seconds'] for row in rows) - min(row['timer']['start_perf_seconds'] for row in rows)
    return {'status': 'PASS_MATCHED_SEMANTIC_DIAGNOSTIC', 'physical_ranks': 8, 'logical_slots': 16,
            'sample_identities': 96, 'Full5_definition': 'ONE Student1+Fake5 cycle',
            'full5_seconds': wall, 'max_rank_seconds': max(row['timer']['rank_wall_seconds'] for row in rows),
            'denominator': 'same-node max original stop_perf minus min original start_perf; not process duration',
            'logical_examples_per_second': 96 / wall, 'GPU_seconds_per_logical_example': wall * 8 / 96,
            'peak_allocated_bytes': max(row['memory_after_timer_before_extra_checks']['peak_allocated'] for row in rows),
            'peak_reserved_bytes': max(row['memory_after_timer_before_extra_checks']['peak_reserved'] for row in rows),
            'strict_cross_arm_numerical_parity': 'NOT_RUN', 'claim_boundary': 'DIAGNOSTIC_ONLY_APPROXIMATE_GRID1000_H100_COMMON_START'}


def rehash_assets_under_held_lock(guard, out, previous):
    """Use the existing source-checked verifier APIs, without nested flock."""
    if guard.sha256_file(VERIFIER) != VERIFIER_SHA:
        raise ValueError('actual verifier source SHA mismatch')
    spec = importlib.util.spec_from_file_location('h3_asset_verifier', VERIFIER)
    verifier = importlib.util.module_from_spec(spec); spec.loader.exec_module(verifier)
    index = CAMPAIGN / 'h3_common_admission_prepare_r1/manifests/h3_transfer_index_r2.json'
    plan = verifier.verify_index(index, guard.INDEX_SHA, guard.ASSETS)
    rows = []
    with (out / 'manifests/fresh_asset_hashes.jsonl').open('x') as journal:
        for value in plan['files']:
            row = verifier.verify_file(value, guard.ASSETS); rows.append(row)
            journal.write(json.dumps(row, sort_keys=True) + '\n'); journal.flush()
    checks = verifier.validate_metadata(guard.ASSETS, rows)
    identities = {key: verifier.group_identity(rows, group) for key, group in
                  (('model_identity_sha256', 'model'), ('prompt_cache_identity_sha256', 'cache'),
                   ('grid_identity_sha256', 'table'), ('FA3_identity_sha256', 'fa3'))}
    if any(identities[key] != previous[key] for key in identities):
        raise ValueError('actual model/cache/Grid/FA3 payload changed since common-start admission')
    write_new(out / 'manifests/fresh_asset_identity.json', {'status': 'PASS_ACTUAL_122_FILE_REHASH',
               'file_count': len(rows), 'checks': checks, **identities})


def launch_text(text):
    """Only replace repo-path discovery and the explicit recording entry module."""
    original = 'REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)'
    replacement = ': "${H3_FROZEN_REPO_ROOT:?Set the exact audited source checkout}"\nREPO_ROOT="$H3_FROZEN_REPO_ROOT"'
    entry = '-m h3_a100.grid_entrypoint --config "${CONFIG}"'
    if text.count(original) != 1 or text.count(entry) != 1:
        raise ValueError('actual frozen launcher differs; do not reconstruct it from memory')
    return text.replace(original, replacement).replace(entry, '-m h3_h100_recording_entry_r1 --config "${CONFIG}"')


def run_arm(guard, parent, arm, mode, bootstrap, bootstrap_sha, adapter_shas, sample_order, gpu, ordinal):
    out = parent / f'{ordinal:02d}_{mode}_{arm}'; out.mkdir(exist_ok=False)
    for name in ('logs', 'manifests', 'rank_receipts', 'training'):
        (out / name).mkdir()
    policy = guard.read_hashed(guard.PREPARED / 'policy_freeze.json', guard.POLICY_SHA)
    policy_arm = 'candidate' if arm == 'candidate' else 'control'
    env = guard.make_environment(dict(policy[policy_arm + '_environment']), out, ','.join(row['uuid'] for row in gpu))
    config_relative = ('configs/minimax_h3_grid1000_h100_resident_b16.yaml'
                       if arm == 'resident' else guard.CONFIG_RELATIVE)
    env.pop('H3_BOOTSTRAP_LOCK_HELD', None)
    env.update({'PYTHONPATH': str(PREPARED / 'scripts') + ':' + env['PYTHONPATH'],
                'H3_PYTHON': str(guard.PYTHON), 'H3_FROZEN_REPO_ROOT': str(guard.REPO),
                'CONFIG': str(guard.REPO / config_relative), 'MASTER_PORT': '29630',
                'H3_OUTPUT_DIR': str(out / 'training'), 'H3_RUN_GPU_LOCK_HELD': '1',
                'H3_LOGICAL_RANK_REPLAY_MANIFEST': str(bootstrap),
                'H3_RECEIPT_RUN_ID': parent.name + '_' + out.name,
                'H3_RECEIPT_OUTPUT': str(out / 'rank_receipts')})
    env['TORCH_EXTENSIONS_DIR'] = str(ROOT / 'cache/h3-h100-matched/torch_extensions')
    env['TRITON_CACHE_DIR'] = str(ROOT / 'cache/h3-h100-matched/triton')
    before = guard.host_snapshot()
    write_new(out / 'manifests/before.json', before)
    command = ['bash', str(parent / 'launch_recorded.sh')]
    write_new(out / 'manifests/command.json', {'command': command, 'cwd': str(guard.REPO), 'bootstrap_sha256': bootstrap_sha, 'config_sha256': guard.sha256_file(guard.REPO / config_relative), 'placement': 'resident' if arm == 'resident' else 'fsdp2'})
    write_new(out / 'manifests/environment.json', {key: value for key, value in env.items()
              if key.startswith(('H3_', 'CUDA_', 'PYTHON', 'LIGHT', 'MINIMAX', 'HF_', 'TRANSFORMERS_', 'TRITON_', 'TORCH_')) or key in ('CONFIG', 'MASTER_PORT')})
    owned = {}; child = None; started = time.monotonic(); host_peak = before['cgroup_current']
    result = {'arm': arm, 'mode': mode, 'status': 'FAIL_ARM', 'GPU': 'NOT_RUN',
              'bootstrap_sha256': bootstrap_sha, 'common_start_boundary': BOUNDARY}
    try:
        apps = guard.run_text(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], timeout=20)
        if apps.strip():
            raise BlockingIOError('resource changed between arms; no foreign GPU process is ours')
        guard.verify_clean_sources()
        with (out / 'logs/launcher.log').open('x') as log, (out / 'logs/host_samples.jsonl').open('x') as samples:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     env=env, cwd=guard.REPO, start_new_session=True)
            result['GPU'] = 'LAUNCHED_REAL_ENTRY'
            record = proc_record(child.pid)
            if not record:
                raise RuntimeError('owned launcher handle disappeared before registration')
            owned[child.pid] = record
            write_new(out / 'manifests/launch.json', record)
            deadline = time.monotonic() + 7200
            while child.poll() is None:
                discover_owned_children(owned)
                current = guard.host_snapshot(); host_peak = max(host_peak, current['cgroup_current'])
                samples.write(json.dumps({'monotonic': time.monotonic(), **current}) + '\n'); samples.flush()
                if time.monotonic() >= deadline:
                    raise TimeoutError('bounded real H3 arm exceeded7200s')
                if current['cgroup_events']['oom'] != before['cgroup_events']['oom'] or current['cgroup_events']['oom_kill'] != before['cgroup_events']['oom_kill']:
                    raise RuntimeError('observed cgroup OOM counter changed; not presumed CUDA OOM')
                time.sleep(10)
            result['launcher_exit_code'] = child.returncode
        if child.returncode:
            raise RuntimeError('real entrypoint failed; inspect preserved rank traceback/launcher log')
        rows = [json.loads((out / f'rank_receipts/rank_{rank:03d}.json').read_text()) for rank in range(8)]
        result.update(aggregate_arm(rows, adapter_shas, sample_order), rank_receipt_count=8)
    except BaseException as error:
        result.update(status='FAIL_ARM', error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    finally:
        remaining = cleanup_owned(owned) if owned else []
        if child is not None:
            with contextlib.suppress(subprocess.TimeoutExpired): child.wait(timeout=10)
        after = None; apps = None
        teardown_errors = []
        try:
            after = guard.host_snapshot()
        except Exception as error:
            teardown_errors.append('host snapshot: ' + str(error))
        try:
            apps = guard.run_text(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], timeout=20)
            (out / 'logs/gpu_contexts_after.txt').write_text(apps)
        except Exception as error:
            teardown_errors.append('GPU context observation: ' + str(error))
        oom_delta = None if after is None else {key: after['cgroup_events'][key] - before['cgroup_events'][key]
                                                for key in ('oom', 'oom_kill')}
        result.update(owned_processes=owned, owned_alive_after_cleanup=remaining,
                      cgroup_current_observed_peak=host_peak, before=before, after=after,
                      cgroup_oom_delta=oom_delta, teardown_observation_errors=teardown_errors,
                      driver_free_minimum='NOT_MEASURED; per-rank setup/end snapshots only',
                      host_peak_basis='10-second cgroup.current samples, not per-rank RSS peak',
                      process_wall_including_setup_and_checks=time.monotonic() - started,
                      profiler='NOT_RUN', payload_exported_locally=False)
        if remaining or teardown_errors or (apps and apps.strip()) or (oom_delta and any(oom_delta.values())):
            result['status'] = 'FAIL_TEARDOWN_OR_CGROUP_GATE'
        try:
            guard.verify_clean_sources()
        except Exception as error:
            result.update(status='FAIL_SOURCE_DRIFT_AT_TEARDOWN', source_error=str(error))
        write_new(out / 'summary.json', result)
    if not result['status'].startswith('PASS_'):
        raise RuntimeError('arm stopped fail-closed: ' + str(out / 'summary.json'))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--bootstrap', type=Path)
    parser.add_argument('--bootstrap-sha256')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,127}', args.run_id):
        parser.error('unsafe new run ID')
    if args.execute and (not args.bootstrap or not args.bootstrap_sha256):
        parser.error('--execute requires a real immutable bootstrap and its exact SHA')
    guard = load_guard()
    out = CAMPAIGN / args.run_id; out.mkdir(exist_ok=False)
    (out / 'logs').mkdir(); (out / 'manifests').mkdir()
    result = {'run_id': args.run_id, 'written_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'status': 'FAIL_PREPARATION', 'source_head': guard.HEAD, 'GPU': 'NOT_RUN',
              'common_start_boundary': BOUNDARY, 'H3_full5_definition': 'ONE Student1+Fake5 cycle',
              'strict_cross_arm_numerical_parity': 'NOT_RUN', 'profiler': 'NOT_RUN'}
    exit_code = 1
    def interrupted(signum, _frame):
        raise InterruptedError('owned campaign interrupted by signal ' + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        guard.validate_authorization(guard.read_hashed(guard.PREPARED / 'authorization.json', guard.AUTH_SHA))
        policy = guard.read_hashed(guard.PREPARED / 'policy_freeze.json', guard.POLICY_SHA)
        guard.validate_policy(policy); guard.verify_clean_sources(); guard.validate_private_data_layer()
        guard.validate_dataset_receipt(guard.read_hashed(guard.PREPARED / 'real_cache_CPU_receipt.json', guard.DATASET_RECEIPT_SHA))
        asset = guard.read_hashed(ASSET_RECEIPT, ASSET_RECEIPT_SHA); guard.validate_asset_receipt(asset)
        launcher = guard.REPO / 'scripts/launch_h100_grid1000.sh'
        if guard.sha256_file(launcher) != LAUNCHER_SHA:
            raise ValueError('original launcher source hash changed')
        (out / 'launch_recorded.sh').write_text(launch_text(launcher.read_text()))
        subprocess.run(['bash', '-n', str(out / 'launch_recorded.sh')], check=True)
        write_new(out / 'manifests/launch_adapter.json', {'original_source_sha256': LAUNCHER_SHA,
                  'recorded_launcher_sha256': guard.sha256_file(out / 'launch_recorded.sh'),
                  'changes': ['repo path explicitly remains the original frozen checkout',
                              'explicit observer delegates to original grid_entrypoint.main; model/trainer/sampler unchanged'],
                  'default_disabled': True, 'same_in_both_arms': True})
        if not args.execute:
            cpu_env = guard.make_environment(policy['control_environment'], out)
            cpu_env['PYTHONPATH'] = str(PREPARED / 'scripts') + ':' + cpu_env['PYTHONPATH']
            source = '''import json,torch
from h3_a100 import grid_entrypoint,trainer
from h3_h100_recording_entry_r1 import Observer,observe_training,GATES
cls=trainer.MiniMaxH3A100DmdTrainer
names=('setup','_start_hardware_cycle_timer','_stop_hardware_cycle_timer','_student_step','_fake_steps')+GATES
before={name:getattr(cls,name) for name in names};loader=trainer.load_initial_adapter_exact
observer=Observer()
with observe_training(cls,trainer,observer):
 assert all(getattr(cls,name).__wrapped__ is function for name,function in before.items())
 assert trainer.load_initial_adapter_exact.__wrapped__ is loader
assert all(getattr(cls,name) is function for name,function in before.items())
assert trainer.load_initial_adapter_exact is loader
assert not torch.cuda.is_initialized() and torch.cuda.device_count()==0
print(json.dumps({'status':'PASS_ACTUAL_H3_WRAPPER_BINDING_NO_MODEL_GPU','original_sources':observer.sources,'CUDA_initialized':False}))
'''
            actual = guard.run_text([guard.PYTHON, '-c', source], env=cpu_env, timeout=180)
            (out / 'logs/actual_CPU_binding.json').write_text(actual)
            result.update(status='PASS_CPU_LIFECYCLE_PREPARATION_GPU_BOOTSTRAP_NOT_RUN',
                          actual_bootstrap='NOT_RUN', capacity='NOT_RUN', timing='NOT_RUN')
            exit_code = 0
        else:
            with contextlib.ExitStack() as stack:
                for path in (ROOT / 'locks/helios_asset_transfer.lock', ROOT / 'locks/helios_world8_gpu.lock', Path('/tmp/distillschedule_gpu_node.lock')):
                    lock = stack.enter_context(open_campaign_lock(path))
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                inventory = guard.run_text(['nvidia-smi', '--query-gpu=index,name,uuid,driver_version,memory.total,memory.used,memory.free', '--format=csv,noheader,nounits'], timeout=20)
                apps = guard.run_text(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], timeout=20)
                gpu = guard.validate_gpu_audit(inventory, apps)
                host = guard.host_snapshot()
                available = host['mem_available_bytes']
                if host['cgroup_max'] != 'max':
                    available = min(available, max(0, int(host['cgroup_max']) - host['cgroup_current'])
                                    + host['cgroup_memory_stat'].get('inactive_file', 0))
                if available < 650 * 1024 ** 3:
                    raise BlockingIOError('eight CPU-first loaders require bounded650GiB construction headroom')
                with socket.socket() as sock: sock.bind(('127.0.0.1', 29630))
                actual = guard.read_hashed(args.bootstrap, args.bootstrap_sha256)
                guard.validate_bootstrap(actual, asset)
                result['runtime_active_grid_table'] = guard.validate_active_table()
                result['original_asset_receipt_grid_scope'] = 'PRESERVED_A100_REFERENCE_NOT_ACTIVE_H100_TABLE'
                adapter_shas = {role: actual['files'][key]['sha256'] for role, key in actual['initial_trainable_files'].items()}
                rehash_assets_under_held_lock(guard, out, asset)
                # Committed replay parser verifies every actual artifact SHA before model load.
                cpu_env = guard.make_environment(policy['control_environment'], out)
                source = 'from h3_a100.hardware_profile import load_replay_manifest;load_replay_manifest(__import__("sys").argv[1],seed=20260817);print("PASS_ACTUAL_IMMUTABLE_BOOTSTRAP_FILES")'
                guard.run_text([guard.PYTHON, '-c', source, args.bootstrap], env=cpu_env, timeout=180)
                write_new(out / 'manifests/ownership_before.json', {'gpu': gpu, 'host': host, 'GPU_context_count': 0})
                result.update(GPU='RUNNING', bootstrap_sha256=args.bootstrap_sha256)
                arms = []
                failed_arms = []
                for name in ('resident', 'control', 'candidate'):
                    for mode in ('correctness', 'timing'):
                        ordinal = len(arms) + len(failed_arms)
                        try:
                            item = run_arm(guard, out, name, mode, args.bootstrap, args.bootstrap_sha256,
                                           adapter_shas, actual['sample_order'], gpu, ordinal)
                            arms.append(item)
                        except RuntimeError:
                            failure_path = out / f'{ordinal:02d}_{mode}_{name}' / 'summary.json'
                            failure = json.loads(failure_path.read_text())
                            log = (failure_path.parent / 'logs/launcher.log').read_text(errors='replace')
                            if ('CUDA out of memory' in log or 'torch.OutOfMemoryError' in log) and (
                                    not failure.get('owned_alive_after_cleanup') and
                                    not any((failure.get('cgroup_oom_delta') or {}).values()) and
                                    not failure.get('teardown_observation_errors')):
                                failure_type = 'CUDA_OOM'
                            else:
                                # Other integration/semantic/environment failures need review,
                                # never call a launcher or census failure a capacity failure.
                                raise
                            failed_arms.append({'arm': name, 'mode': mode, 'failure_type': failure_type,
                                                'summary': str(failure_path), 'timing': 'NOT_RUN',
                                                'workload_contract_changed': False})
                            write_new(out / 'manifests' / f'capacity_{name}.json', failed_arms[-1])
                            break
                raw = {name: [item['full5_seconds'] for item in arms
                              if item['arm'] == name and item['mode'] == 'timing']
                       for name in ('resident', 'control', 'candidate')}
                measured = {name: values[0] for name, values in raw.items() if values}
                ratios = {name: measured[name] / measured['candidate']
                          for name in ('resident', 'control') if name in measured and 'candidate' in measured}
                result.update(status=('PASS_THREE_ARM_EXECUTION_WITH_RECORDED_CAPACITY_FAILURES' if failed_arms
                                      else 'PASS_H100_THREE_ARM_SINGLE_TIMINGS_DIAGNOSTIC'),
                              arms=arms, failed_arms=failed_arms, raw_seconds=raw,
                              seconds=measured, single_pair_ratios=ratios, replications=1,
                              CV='NOT_MEASURED_SINGLE_TRIAL_PER_ARM', cold_setup_and_checks_excluded=True,
                              formal_strict_speedup='NOT_CLAIMABLE_NUMERICAL_PARITY_NOT_RUN')
                exit_code = 0
    except BlockingIOError as error:
        result.update(status='NOT_RUN_RESOURCE_OR_LOCK_GATE', error=str(error)); exit_code = 75
    except BaseException as error:
        result.update(status='FAIL_PREPARATION_OR_MATRIX', error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    finally:
        write_new(out / 'status.json', result)
        print(json.dumps({key: value for key, value in result.items() if key != 'arms'}, sort_keys=True))
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
