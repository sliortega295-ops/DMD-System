#!/usr/bin/env python3
"""Campaign-specific, default-CPU-only guard for the authorized H100 start.

This calls the existing committed builder. It adds no model/trainer API. An
explicit --execute is required to load the real CPU checkpoint and obtain RNG
states on ONE owned GPU. Cooperating GPU and transfer locks remain held until
the child is gone and teardown evidence has been collected.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time
from h100_campaign_lock_adapter import open_campaign_lock

ROOT = Path('/home/jiangzhikun/yongyan_liu/Helios-H100')
EVIDENCE = ROOT / 'research_artifacts/five_workload_h100_20260930'
PREPARED = EVIDENCE / 'h100_resident_baselines_20261014_r1/artifacts/h3_three_arm_prepare_r3'
REPO = ROOT / 'repos/H3-DMD-for-A100-h100-three-arm-20261014-r1'
LIGHT = ROOT / 'repos/h3-lightx2v-d034a6b0-20261004-r2'
DIFF = ROOT / 'repos/h3-diffusers-92846072-20261004-r2'
OVERLAY = ROOT / 'envs/h3-grid-overlay-20261004/site-packages'
DATA = ROOT / 'envs/h3-data-overlay-20261004-r1/site-packages'
DATA_PLAN_SHA = 'a0cdaec4032e2ba27b35f5f801e40a1cc4f4519d2debba8bffd1a0b3f00006bf'
DATASET_RECEIPT_SHA = '432c744ca47dab5f32ea9c9462d076522b48d0c8136b392182ccd4c9d714dd2e'
PYTHON = ROOT / 'envs/five-workload-h100-20260930/bin/python'
ASSETS = ROOT / 'assets/five_workload_h100_20260930/h3_grid_frozen_r1'
HEAD = '44045c63638ff45e144f6fc7f57284057ae6ce0b'
BOOTSTRAP_SOURCE_HEAD = 'a4c289757d0ecc8c32eb6ea27021f22cd04d94fd'
LIGHT_HEAD = 'd034a6b0ecaa31aa07c81aeb7bbe69b225f1d7be'
DIFF_HEAD = '9284607295a09f759aadd65ed08f48b35feea6d9'
CONFIG_RELATIVE = 'configs/minimax_h3_grid1000_h100_8_b16.yaml'
CONFIG_SHA = 'b001f2f517b854c34ca68cccce23422499d95b7a3ad27072b895a502498ca6be'
POLICY_SHA = json.loads((PREPARED / 'source_policy_binding.json').read_text())['policy_sha256']
AUTH_SHA = '67b93ce3afe44f8691c25edb8a27f33ffada86dd3e2dd5aba26e66e7274f0049'
INDEX_SHA = '5886a6f81e65b2393d7aec8c9021dee154d7e40967ae95899744adae98e2091f'
BOUNDARY = 'H100_ONLY_COMMON_START_NOT_A100_INITIAL_STATE_REPLAY'
GIB = 1024 ** 3


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_hashed(path, expected):
    if not re.fullmatch('[0-9a-f]{64}', expected or '') or sha256_file(path) != expected:
        raise ValueError('evidence SHA mismatch: ' + str(path))
    return json.loads(Path(path).read_text())


def validate_authorization(value):
    if (value.get('status') != 'EXPLICIT_USER_AUTHORIZATION_RECEIVED'
            or value.get('comparison_boundary') != BOUNDARY
            or value.get('a100_initial_state_reproduced') is not False
            or value.get('workload_semantics_change_authorized') is not False):
        raise ValueError('fresh-start authorization does not preserve the comparison boundary')


def validate_policy(value):
    if (value.get('h3_head') != HEAD or value.get('h100_config_sha256') != CONFIG_SHA
            or value.get('comparison_boundary') != BOUNDARY
            or value.get('physical_world') != 8 or value.get('logical_global_batch') != 16
            or value.get('GAS') != 2 or value.get('B1_per_call') is not True
            or value.get('optimizer_commits_per_cycle') != {'student': 1, 'fake': 5, 'ema': 0}):
        raise ValueError('frozen H3 policy identity mismatch')
    env = value['control_environment']
    required = {'H3_MAX_ITERS': '1', 'H3_BENCHMARK_SEED': '20260817',
                'H3_ACTIVATION_CHECKPOINT_SEGMENT': '1', 'H3_ACTIVATION_POLICY': 'none',
                'H3_ACTIVATION_OFFLOAD': '0', 'H3_ATTN_BACKEND': '_flash_3_hub'}
    if any(env.get(key) != expected for key, expected in required.items()):
        raise ValueError('bootstrap cannot silently change the frozen workload policy')
    return dict(env)


def validate_asset_receipt(value):
    expected = {'status': 'PASS_FULL_HASH_H3_ASSETS', 'index_sha256': INDEX_SHA,
                'file_count': 122, 'total_bytes': 86_533_781_312,
                'comparison_boundary': BOUNDARY, 'model_root': str(ASSETS / 'model'),
                'prompt_cache_root': str(ASSETS / 'prompt_cache'),
                'grid_manifest': str(ASSETS / 'grid_table/adaln_grid1000.json'),
                'FA3_root': str(ASSETS / 'fa3')}
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        raise ValueError('actual complete H3 asset admission is required')
    for key in ('model_identity_sha256', 'prompt_cache_identity_sha256',
                'grid_identity_sha256', 'FA3_identity_sha256'):
        if not re.fullmatch('[0-9a-f]{64}', str(value.get(key, ''))):
            raise ValueError('missing full-hash content identity: ' + key)
    if value.get('checks', {}).get('cache_samples') != 96:
        raise ValueError('actual 96-sample cache admission is missing')


def validate_gpu_audit(inventory, apps):
    rows = list(csv.reader(inventory.splitlines(), skipinitialspace=True))
    if (len(rows) != 8 or [int(row[0]) for row in rows] != list(range(8))
            or any(len(row) != 7 or 'H100' not in row[1] for row in rows)
            or len({row[2] for row in rows}) != 8):
        raise ValueError('exact eight H100 inventory is required')
    if apps.strip():
        raise BlockingIOError('GPU contexts are nonzero; no foreign process may be signaled')
    if any(row[3] != '535.183.06' or int(row[4]) < 79_000 for row in rows):
        raise ValueError('hardware/driver substrate drift; revalidate instead of guessing')
    if any(int(row[6]) < 2 * 1024 for row in rows):
        raise BlockingIOError('idle-driver memory margin is below the RNG-only gate')
    return [{'index': int(row[0]), 'name': row[1], 'uuid': row[2], 'driver': row[3],
             'total_mib': int(row[4]), 'used_mib': int(row[5]), 'free_mib': int(row[6])}
            for row in rows]


def make_environment(control, output, gpu_uuid=None):
    # Only the child process is changed; never alter shell startup or packages.
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(('H3_', 'DISTILLSCHEDULE_', 'HELIOS_', 'NCCL_', 'CUDA_'))
           and key not in ('PYTHONPATH', 'PYTORCH_CUDA_ALLOC_CONF', 'PYTORCH_ALLOC_CONF')}
    env.update(control)
    env.update({'PYTHONPATH': ':'.join(map(str, (REPO, DATA, OVERLAY, DIFF / 'src', LIGHT,
                                               LIGHT / 'lightx2v_train'))),
                'PYTHONDONTWRITEBYTECODE': '1', 'CUDA_VISIBLE_DEVICES': gpu_uuid or '',
                'LIGHTX2V_ROOT': str(LIGHT), 'H3_HARDWARE_PROFILE': 'h100-8-b16',
                'MINIMAX_H3_MODEL_PATH': str(ASSETS / 'model'),
                'H3_PROMPT_CACHE': str(ASSETS / 'prompt_cache'),
                'H3_ADALN_GRID_MANIFEST': str(EVIDENCE / 'h3_grid_h100_native_table_20261014_r1/adaln_grid1000.json'),
                'H3_FLASH_ATTN3_LOCAL_PATH': str(ASSETS / 'fa3'),
                'H3_LOGICAL_RANK_REPLAY_MANIFEST': str(output / 'bootstrap/manifest.json'),
                'H3_OUTPUT_DIR': str(output / 'training_not_run'),
                'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                'HF_HOME': str(ROOT / 'cache/h3-common-bootstrap'),
                'TORCH_HOME': str(ROOT / 'cache/h3-common-bootstrap/torch'),
                'OMP_NUM_THREADS': '4', 'MALLOC_ARENA_MAX': '2'})
    if gpu_uuid:
        env['H3_BOOTSTRAP_LOCK_HELD'] = '1'
    return env


def run_text(argv, *, timeout=60, env=None):
    result = subprocess.run(list(map(str, argv)), capture_output=True, text=True,
                            check=True, timeout=timeout, env=env)
    return result.stdout


def verify_clean_sources():
    for path, head in ((REPO, HEAD), (LIGHT, LIGHT_HEAD), (DIFF, DIFF_HEAD)):
        if run_text(['git', '-C', path, 'rev-parse', 'HEAD']).strip() != head:
            raise ValueError('actual clean checkout HEAD mismatch: ' + str(path))
        if run_text(['git', '-C', path, 'status', '--porcelain', '--untracked-files=all']).strip():
            raise ValueError('dirty source is not admissible: ' + str(path))
    if sha256_file(REPO / CONFIG_RELATIVE) != CONFIG_SHA:
        raise ValueError('official H100 config SHA mismatch')


def cgroup_events():
    return {key: int(value) for key, value in (line.split() for line in
            Path('/sys/fs/cgroup/memory.events').read_text().splitlines())}


def host_snapshot():
    meminfo = {line.split(':')[0]: line.split(':')[1].strip()
               for line in Path('/proc/meminfo').read_text().splitlines()}
    current = int(Path('/sys/fs/cgroup/memory.current').read_text())
    maximum = Path('/sys/fs/cgroup/memory.max').read_text().strip()
    stats = {k: int(v) for k, v in (line.split() for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())}
    return {'cgroup_memory_stat': stats, 'mem_available_bytes': int(meminfo['MemAvailable'].split()[0]) * 1024,
            'cgroup_current': current, 'cgroup_max': maximum,
            'cgroup_peak': (Path('/sys/fs/cgroup/memory.peak').read_text().strip() if Path('/sys/fs/cgroup/memory.peak').exists() else 'NOT_AVAILABLE_ON_THIS_KERNEL'),
            'cgroup_events': cgroup_events(), 'disk_free_bytes': shutil.disk_usage(ROOT).free}


CPU_SMOKE = """import json,torch,diffusers,peft,transformers,inspect,hashlib,imageio,importlib.metadata as m
assert imageio.__version__=='2.37.3' and 'h3-data-overlay-20261004-r1' in imageio.__file__
from diffusers import MiniMaxH3Transformer3DModel
from lightx2v_train.runtime import load_config
from h3_a100 import fresh_h100_bootstrap
c=load_config(__import__('sys').argv[1])
assert torch.__version__=='2.10.0+cu128' and torch.version.cuda=='12.8'
assert diffusers.__version__=='0.40.0.dev0' and peft.__version__=='0.18.1'
assert c['training']['gradient_accumulation_iters']==2 and c['data']['train']['batch_size']==1
assert c['training']['dmd']['num_inference_steps']==4 and c['training']['dmd']['fake_update_ratio']==5
assert c['training']['a100']['matched_compute']['fixed_end_step_idx']==3
assert c['training']['dmd']['height']==768 and c['training']['dmd']['width']==1344
assert c['training']['dmd']['num_frames']==124
assert c['training']['a100']['activation_policy']['name']=='none'
assert c['training']['student']['lora']['rank']==128 and c['training']['fake']['lora']['rank']==128
assert hashlib.sha256(inspect.getsource(MiniMaxH3Transformer3DModel.forward).encode()).hexdigest()=='ae200f2868ad388f1f40fccb7529f39ec3bc00c999725babef1f6d73d05361d4'
assert not torch.cuda.is_initialized() and torch.cuda.device_count()==0
print(json.dumps({'status':'PASS_CPU_SOURCE_CONFIG_IMPORTS','torch':torch.__version__,
 'cuda':torch.version.cuda,'diffusers':diffusers.__version__,'peft':peft.__version__,
 'transformers':transformers.__version__,'python':__import__('sys').executable,
 'model_forward':'NOT_RUN','CUDA_initialized':False,'resolved_config':c,
 'private_data_import':{'imageio':imageio.__version__,'path':imageio.__file__,
   'numpy':m.version('numpy'),'pillow':m.version('pillow')}}))
"""


def validate_bootstrap(value, asset):
    if (value.get('format') != 'h3-h100-common-bootstrap-v1' or value.get('status') != 'PASS'
            or value.get('source_head') != BOOTSTRAP_SOURCE_HEAD or value.get('comparison_boundary') != BOUNDARY
            or value.get('a100_initial_state_reproduced') is not False
            or value.get('cycle_count') != 1 or value.get('logical_world_size') != 16
            or value.get('initial_optimizer_updates') != {'student': 0, 'fake': 0}
            or value.get('initial_scheduler_updates') != {'student': 0, 'fake': 0}
            or value.get('config_sha256') != CONFIG_SHA):
        raise ValueError('actual bootstrap does not preserve the newly authorized start')
    for key in ('model_identity_sha256', 'prompt_cache_identity_sha256'):
        if value.get(key) != asset.get(key):
            raise ValueError('actual bootstrap changed a fully verified asset identity')



def validate_dataset_receipt(value):
    expected = {'status': 'PASS_ACTUAL_96_SAMPLE_B1_CPU_DATASET_AND_SAMPLER',
        'cache_samples': 96, 'actual_unique_sample_identities': 96,
        'actual_unique_dataset_indices': 96, 'logical_slots': 16, 'samples_per_slot': 6,
        'repeated_order': 'PASS', 'caller_Python_NumPy_Torch_CPU_RNG_unchanged': 'PASS',
        'finite_CPU_conditions': 'PASS', 'CUDA_initialized': False, 'GPU': 'NOT_RUN',
        'comparison_boundary': BOUNDARY,
        'observed_sample_order_identity_sha256': 'bab6d5ea49228f561dd5915072f18d6d7e08f73665b57071a5e2fc696cfb6865'}
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        raise ValueError('actual complete CPU cached-data admission is required')
    if value.get('dependency_identity', {}).get('imageio') != '2.37.3':
        raise ValueError('private cached-data dependency version mismatch')


def validate_private_data_layer():
    source = read_hashed(DATA.parent / 'source_manifest.json', DATA_PLAN_SHA)
    if (source.get('status') != 'PASS_SOURCE_PACKAGE_SNAPSHOT'
            or source.get('package') != 'imageio' or source.get('version') != '2.37.3'
            or source.get('sha256') != 'bd8954e349c892ffa5a944b887333a6c3de719c2701b4ae9ea0172f4a31bde9f'
            or len(source.get('file_inventory', {})) != 65):
        raise ValueError('private data-layer source snapshot mismatch')
    for relative, row in source['file_inventory'].items():
        path = DATA / relative
        if (path.is_symlink() or not path.resolve().is_relative_to(DATA.resolve())
                or not path.is_file() or path.stat().st_size != row['bytes']
                or sha256_file(path) != row['sha256']):
            raise ValueError('private data-layer file content mismatch: ' + relative)
    return {'status': 'PASS_ACTUAL_PRIVATE_IMAGEIO_FILES', 'file_count': 65,
            'package': 'imageio', 'version': '2.37.3', 'manifest_sha256': DATA_PLAN_SHA,
            'activation': 'process PYTHONPATH only; shared environment unchanged'}


def validate_active_table():
    table = EVIDENCE / 'h3_grid_h100_native_table_20261014_r1'
    record = json.loads((table / 'runtime_table_receipt.json').read_text())
    if record.get('status') != 'PASS_H100_NATIVE_GRID1000_BITWISE_VALIDATION':
        raise ValueError('new H100-native table bitwise gate has not passed')
    for name in ('manifest', 'binary'):
        row = record[name]
        if sha256_file(row['path']) != row['sha256']:
            raise ValueError('actual active H100 table payload changed')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--asset-receipt', type=Path)
    parser.add_argument('--asset-receipt-sha256')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,127}', args.run_id):
        raise ValueError('unsafe or ambiguous new run ID')
    if args.execute and (not args.asset_receipt or not args.asset_receipt_sha256):
        parser.error('--execute requires the actual complete asset receipt and its SHA256')
    output = EVIDENCE / args.run_id
    output.mkdir(exist_ok=False)
    (output / 'logs').mkdir(); (output / 'manifests').mkdir()
    report = {'run_id': args.run_id, 'written_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'comparison_boundary': BOUNDARY, 'a100_initial_state_reproduced': False,
              'source_head': HEAD, 'training': 'NOT_RUN', 'model_GPU_load': 'NOT_RUN',
              'capacity': 'NOT_RUN', 'correctness': 'NOT_RUN', 'timing': 'NOT_RUN',
              'common_bootstrap': 'NOT_RUN', 'workload_semantics': 'approximate_grid1000_IN_BOTH_ARMS',
              'script_sha256': sha256_file(__file__)}
    child = None
    exit_code = 1
    def interrupted(signum, _frame):
        raise InterruptedError('owned bootstrap controller interrupted by ' + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        validate_authorization(read_hashed(PREPARED / 'authorization.json', AUTH_SHA))
        control = validate_policy(read_hashed(PREPARED / 'policy_freeze.json', POLICY_SHA))
        report['private_data_layer'] = validate_private_data_layer()
        dataset = read_hashed(PREPARED / 'real_cache_CPU_receipt.json', DATASET_RECEIPT_SHA)
        validate_dataset_receipt(dataset)
        report['actual_CPU_dataset_receipt_sha256'] = DATASET_RECEIPT_SHA
        verify_clean_sources()
        env_cpu = make_environment(control, output)
        actual = run_text([PYTHON, '-c', CPU_SMOKE, REPO / CONFIG_RELATIVE],
                          timeout=150, env=env_cpu)
        (output / 'logs/cpu_import_config.log').write_text(actual)
        imports = json.loads(actual)
        (output / 'manifests/resolved_config.json').write_text(json.dumps(imports.pop('resolved_config'), indent=2)+'\n')
        report['CPU_import_config'] = imports
        if not args.execute:
            report['status'] = 'PASS_CPU_PREPARATION_BOOTSTRAP_NOT_RUN'
            exit_code = 0
        else:
            asset = read_hashed(args.asset_receipt, args.asset_receipt_sha256)
            validate_asset_receipt(asset)
            report['runtime_active_grid_table'] = validate_active_table()
            report['original_asset_receipt_grid_scope'] = 'PRESERVED_A100_REFERENCE; NOT_THE_ACTIVE_H100_TRAINING_TABLE'
            report['asset_receipt_sha256'] = args.asset_receipt_sha256
            with contextlib.ExitStack() as stack:
                for path in (ROOT / 'locks/helios_asset_transfer.lock',
                             ROOT / 'locks/helios_world8_gpu.lock', Path('/tmp/distillschedule_gpu_node.lock')):
                    lock = stack.enter_context(open_campaign_lock(path))
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                inventory = run_text(['nvidia-smi', '--query-gpu=index,name,uuid,driver_version,memory.total,memory.used,memory.free',
                                      '--format=csv,noheader,nounits'], timeout=20)
                apps = run_text(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], timeout=20)
                (output / 'logs/gpu_inventory_before.csv').write_text(inventory)
                (output / 'logs/gpu_contexts_before.txt').write_text(apps)
                gpu = validate_gpu_audit(inventory, apps)
                before = host_snapshot(); report['before'] = before; report['GPU_before'] = gpu
                available = before['mem_available_bytes']
                if before['cgroup_max'] != 'max':
                    # Only inactive file cache is treated as potentially reclaimable;
                    # never count pinned/shmem or drop any shared cache explicitly.
                    available = min(available, max(0, int(before['cgroup_max']) - before['cgroup_current'])
                                    + before['cgroup_memory_stat'].get('inactive_file', 0))
                report['construction_headroom_basis'] = 'min(host MemAvailable, cgroup free + inactive_file); not anonymous RSS; no cache eviction command'
                report['construction_headroom_bytes'] = available
                if available < 256 * GIB or before['disk_free_bytes'] < 32 * GIB:
                    raise BlockingIOError('insufficient bounded CPU-construction/disk headroom')
                env = make_environment(control, output, gpu[0]['uuid'])
                command = [str(PYTHON), str(REPO / 'scripts/build_h100_common_bootstrap.py'),
                           '--config', str(REPO / CONFIG_RELATIVE), '--asset-receipt', str(args.asset_receipt),
                           '--authorization-receipt', str(PREPARED / 'authorization.json'),
                           '--owned-root', str(ROOT), '--output', str(output / 'bootstrap'), '--cycles', '1']
                (output / 'manifests/command.json').write_text(json.dumps(command, indent=2)+'\n')
                (output / 'manifests/environment.json').write_text(json.dumps({key: value for key, value in env.items()
                    if key.startswith(('H3_', 'CUDA_', 'HF_', 'TRANSFORMERS_', 'MINIMAX_', 'LIGHTX2V_', 'PYTHON'))}, indent=2)+'\n')
                started = time.monotonic()
                with (output / 'logs/bootstrap.log').open('x') as log:
                    try:
                        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                                 env=env, cwd=REPO, start_new_session=True)
                        report['owned_child'] = {'pid': child.pid, 'pgid': child.pid,
                            'start_ticks': Path(f'/proc/{child.pid}/stat').read_text().split(') ', 1)[1].split()[19]}
                        (output / 'manifests/launch.json').write_text(json.dumps(report['owned_child'], indent=2)+'\n')
                        code = child.wait(timeout=3600)
                    finally:
                        # Only the dedicated session created above; never NVML-discovered PIDs.
                        if child is not None and child.poll() is None:
                            os.killpg(child.pid, signal.SIGTERM)
                            try: child.wait(timeout=30)
                            except subprocess.TimeoutExpired:
                                os.killpg(child.pid, signal.SIGKILL); child.wait(timeout=10)
                        after = host_snapshot(); report['after'] = after
                        apps_after = run_text(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], timeout=20)
                        (output / 'logs/gpu_contexts_after.txt').write_text(apps_after)
                        report['GPU_contexts_after_count'] = len(apps_after.strip().splitlines()) if apps_after.strip() else 0
                        report['cgroup_oom_delta'] = {key: after['cgroup_events'].get(key, 0)-before['cgroup_events'].get(key, 0)
                                                   for key in ('oom', 'oom_kill')}
                        report['bootstrap_wall_seconds_not_training'] = time.monotonic()-started
                report['builder_exit_code'] = code
                if any(report['cgroup_oom_delta'].values()):
                    raise RuntimeError('FAIL_CGROUP_OOM_DELTA; do not relabel a kill as CUDA OOM')
                if code or report['GPU_contexts_after_count']:
                    raise RuntimeError('builder failure or nonempty teardown; do not signal foreign contexts')
                value = json.loads((output / 'bootstrap/manifest.json').read_text())
                validate_bootstrap(value, asset)
                verify_clean_sources()
                report.update(status='PASS_NEW_IMMUTABLE_H100_COMMON_START', common_bootstrap='PASS',
                              bootstrap_manifest=str(output / 'bootstrap/manifest.json'),
                              bootstrap_manifest_sha256=sha256_file(output / 'bootstrap/manifest.json'))
                exit_code = 0
    except BlockingIOError as error:
        report.update(status='NOT_RUN_RESOURCE_OR_LOCK_GATE', error=str(error), error_type=type(error).__name__)
        exit_code = 75
    except Exception as error:
        report.update(status='FAIL_COMMON_BOOTSTRAP_PREPARATION', error=str(error), error_type=type(error).__name__)
    finally:
        (output / 'manifests/result.json').write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
        print(json.dumps({key: value for key, value in report.items() if key not in ('before', 'after', 'GPU_before')}, sort_keys=True))
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
