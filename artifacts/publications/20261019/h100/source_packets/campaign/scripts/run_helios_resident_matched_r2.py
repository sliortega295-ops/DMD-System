#!/usr/bin/env python3
"""Serial, source-bound H100 launch orchestration; no model implementation.

Delegate all math, trace, receipt, and timing gates to the reviewed native
launchers. Keep numerical REVIEW/NOT_RUN separate from execution PASS.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time
import traceback

ROOT = Path('/home/jiangzhikun/yongyan_liu/Helios-H100')
RUNS = ROOT / 'research_artifacts/five_workload_h100_20260930'
PREP = RUNS / 'h100_resident_baselines_20261014_r1'
REPO = ROOT / 'repos/DMD-System-h100-helios-resident-20261014-r2'
HEAD = '8e9828b5128ef8cbac165ad3f406d4fdfb03b0eb'
PYTHON = ROOT / 'envs/five-workload-h100-20260930/bin/python'
ASSETS = ROOT / 'assets/five_workload_h100_20260930'
ARMS = [('baseline', 'candidate')]
LEASE_FDS = (7, 8, 9)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def dump(path: Path, data, *, immutable=True):
    if immutable:
        with path.open('x') as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write('\n')
    else:
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
        os.replace(tmp, path)


def events():
    return dict((k, int(v)) for k, v in (
        row.split() for row in Path('/sys/fs/cgroup/memory.events').read_text().splitlines()))


def clean_env():
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('DISTILLSCHEDULE_', 'LIVEWAN_', 'H3_', 'HELIOS_', 'NCCL_'))}
    env.update(DISTILLSCHEDULE_INHERITED_GPU_LOCKS='1',
               DISTILLSCHEDULE_PYTHON=str(PYTHON),
               DISTILLSCHEDULE_RUN_DEADLINE_SECONDS='1800',
               DISTILLSCHEDULE_HELIOS_ROOT=str(ROOT / 'repos/frozen-helios-wan-a100-5d3ffe37'),
               PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    return env


def plan(workload):
    return dict(workload=workload, exact_head=HEAD, repo=str(REPO),
                physical_ranks=8, logical_slots=list(range(16)), logical_batch=16,
                rollout='B1; independent original trace row per logical slot',
                order=[list(x) for x in ARMS],
                correctness='fresh both-arm execution gate before timing',
                numerical_policy='unchanged existing thresholds; REVIEW/NOT_RUN not strict PASS',
                profiled=False, GPU_status='NOT_RUN',
                contract=('same-process two cycles; only cycle2 timed'
                          if workload in ('wan', 'livewan') else 'one GAN G1/C5/EMA1 Full5' if workload == 'gan' else 'one No-GAN G1/C5/EMA5 Full5'),
                lock_scope='inherited asset/project/node FDs held through audit/run/teardown')


class Runner:
    def __init__(self, workload):
        self.workload = workload
        self.out = PREP / 'artifacts' / f'helios_{workload}_resident_matched_r2'
        self.out.mkdir(parents=True, exist_ok=False)
        (self.out / 'logs').mkdir()
        self.ordinal = 0
        self.records = []
        dump(self.out / 'manifest.json', plan(workload))

    def status(self, phase, state, **fields):
        dump(self.out / 'status.json', dict(run_id=f'h100_resident_{self.workload}_matched_20261014_r1',
             phase=phase, status=state, updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
             exact_head=HEAD, strict_claimable=False, **fields), immutable=False)

    def invoke(self, label, command, env=None, deadline=2100, gpu=False):
        self.ordinal += 1
        env = clean_env() if env is None else dict(env)
        logpath = self.out / 'logs' / f'{self.ordinal:02d}_{label}.log'
        self.status(label, 'RUNNING')
        dump(self.out / f'command_{self.ordinal:02d}.json',
             dict(label=label, argv=list(map(str, command)), timeout_seconds=deadline,
                  gpu_workload=gpu, inherited_fds=list(LEASE_FDS)))
        before = events()
        if gpu:
            apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                            '--format=csv,noheader'], text=True, timeout=20)
            if apps.strip():
                raise RuntimeError('GPU contexts present before new owned arm; never signal foreign work')
        started = time.monotonic()
        with logpath.open('x') as log:
            child = subprocess.Popen(list(map(str, command)), env=env, stdout=log,
                                     stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                     pass_fds=LEASE_FDS, start_new_session=True)
            try:
                ticks = Path(f'/proc/{child.pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
            except FileNotFoundError:
                if child.poll() is None:
                    raise
                ticks = 'ALREADY_EXITED_BEFORE_HANDLE_SNAPSHOT_CPU_COMMAND'
            dump(self.out / f'owned_handle_{self.ordinal:02d}.json',
                 dict(pid=child.pid, pgid=child.pid, start_ticks=ticks, label=label))
            timed_out = False
            interrupted = False
            try:
                rc = child.wait(timeout=deadline)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    rc = child.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    rc = child.wait(timeout=15)
            except (KeyboardInterrupt, InterruptedError):
                interrupted = True
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    rc = child.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    rc = child.wait(timeout=15)
            finally:
                # Only our own start_new_session process group, never global pkill.
                try:
                    os.killpg(child.pid, 0)
                except ProcessLookupError:
                    pass
                else:
                    os.killpg(child.pid, signal.SIGTERM)
                    time.sleep(2)
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
        after = events()
        delta = {k: after[k] - before[k] for k in ('oom', 'oom_kill')}
        apps = ''
        if gpu:
            apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                            '--format=csv,noheader'], text=True, timeout=20).strip()
        health = dict(status='PASS' if rc == 0 and not timed_out and not interrupted and not apps and not any(delta.values()) else 'FAIL',
                      exit_code=rc, timed_out=timed_out, interrupted=interrupted, cgroup_oom_delta=delta,
                      gpu_contexts_after=apps.splitlines() if gpu else 'NOT_QUERIED_CPU_STEP',
                      process_wall_seconds=time.monotonic() - started,
                      wall_scope='startup/construction/run/teardown; NOT the native training timer',
                      log=str(logpath), log_sha256=sha(logpath))
        dump(self.out / f'health_{self.ordinal:02d}.json', health)
        if health['status'] != 'PASS':
            raise RuntimeError(f'{label}: {health}')

    def admit(self):
        actual = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
        dirty = subprocess.check_output(['git', '-C', str(REPO), 'status', '--porcelain'], text=True).strip()
        if actual != HEAD or dirty:
            raise ValueError('exact clean source gate failed')
        self.invoke('inherited_lease_verification',
                    [PYTHON, REPO / 'scripts/distillschedule/verify_inherited_gpu_locks.py',
                     '--project-root', ROOT, '--asset-fd', '7'], deadline=30)
        gate = json.loads((PREP / 'manifests/helios_cpu_gate_r2.json').read_text())
        if gate['status'] != 'PASS' or gate['head'] != HEAD:
            raise ValueError('committed CPU gate failed')

    def record_trial(self, pair, arm, run_id, summary, seconds, summary_path):
        if summary['status'] != 'PASS' or seconds <= 0:
            raise ValueError('native summary is not PASS')
        row = dict(pair=pair, arm=arm, run_id=run_id, seconds=seconds,
                   summary_status=summary['status'], summary=str(summary_path), summary_sha256=sha(summary_path),
                   profiled=False, strict_claimable=False)
        self.records.append(row)
        dump(self.out / f'trial_{pair}_{arm}.json', row)

    def finish(self):
        rows = {x['arm']: x for x in self.records}
        if len(self.records) != 2 or set(rows) != {'baseline', 'candidate'}:
            raise ValueError('one complete matched timing pair required')
        result = {**plan(self.workload), 'status': 'PASS_CONTROLLED_DIAGNOSTIC',
                  'GPU_status': 'PASS_COMPLETE_ONE_MATCHED_PAIR', 'records': self.records,
                  'baseline_seconds': rows['baseline']['seconds'],
                  'candidate_seconds': rows['candidate']['seconds'],
                  'single_pair_ratio': rows['baseline']['seconds'] / rows['candidate']['seconds'],
                  'replications': 1, 'CV': 'NOT_MEASURED_SINGLE_PAIR',
                  'strict_claimable': False, 'strict_full_state_equivalence': 'NOT_RUN',
                  'source_bound_numerical_report': 'unchanged comparator; REVIEW remains REVIEW'}
        dump(self.out / 'result.json', result)
        self.status('complete_one_native_pair', result['status'])

    def helios(self):
        variant = self.workload
        env = clean_env()
        if variant == 'no-gan':
            api = self.out / 'API_canary'
            api_env = dict(env, PYTHONPATH=f'{REPO}/src:{REPO}', DISTILLSCHEDULE_HARDWARE_PROFILE='h100-8-b16',
                           API_CANARY_OUTPUT=str(api), API_CANARY_HEAD=HEAD)
            self.invoke('real_world8_optimizer_API_canary_NOT_CAPACITY',
                [PYTHON,'-m','torch.distributed.run','--standalone','--nproc-per-node=8',
                 PREP/'scripts/zero1_api_canary_r2.py'],api_env,deadline=120,gpu=True)
            receipts=[json.loads((api/f'rank_{rank:02d}.json').read_text()) for rank in range(8)]
            if any(x['status']!='PASS_API_ONLY' or x['source_head']!=HEAD or x['route_audit']['routed_calls']!=1 for x in receipts):
                raise ValueError('actual optimizer route canary failed; never launch 14B')
        env.update(HELIOS_RUNTIME_ROOT=str(ROOT / 'repos/frozen-helios-wan-a100-5d3ffe37'),
                   DISTILLSCHEDULE_ASSET_ROOT=str(ASSETS), HF_HOME=str(ASSETS / 'frozen_a100_wan/hf_home_official_torch210'),
                   DISTILLSCHEDULE_MASTER_PORT='28791',
                   PYTHONPATH=f'{REPO}/src:{REPO}:{ROOT}/repos/frozen-helios-wan-a100-5d3ffe37')
        cpu_env = dict(env, CUDA_VISIBLE_DEVICES='')
        gate_dir = self.out / 'asset_gate'
        self.invoke('fresh_helios_asset_full_hash', [PYTHON,
                    RUNS / 'helios_gate_inputs_r1/scripts/build_helios_h100_asset_gate_r1.py',
                    '--repo-root', REPO, '--asset-root', ASSETS, '--expected-head', HEAD,
                    '--plan', RUNS / 'helios_gate_inputs_r1/manifests/helios_asset_transfer_gzip_plan_r1.json',
                    '--plan-sha256', '629869a547249af8c7616ae40f816d5bb41df3b8692bbec09e253397cd0bfd04',
                    '--shared-source', RUNS / 'helios_gate_inputs_r1/manifests/helios_shared_source_full_sha_r1.json',
                    '--shared-source-sha256', '73da6db0e90b5f1ddaea55b44fc0c7c2ecc61953a65b7b43cde076270184b0d6',
                    '--variant', variant, '--output-dir', gate_dir], cpu_env, deadline=1200)
        gate = gate_dir / 'asset_gate.json'
        launcher = REPO / 'scripts/distillschedule/launch_helios_hardware_p0_node.sh'
        arms = {'baseline': 'resident', 'candidate': 'best'}
        correctness = {}
        for mode in ('validate', 'correctness'):
            for arm, native in arms.items():
                rid = f'helios_h100_resident_{variant}_{arm}_{mode}_20261014_r2'
                self.invoke(rid, ['bash', launcher, ROOT, rid, variant, mode, native, HEAD, gate, sha(gate)], env,
                            deadline=2400, gpu=True)
                p = RUNS / rid / 'summary.json'; x = json.loads(p.read_text())
                if x['status'] != 'PASS' or x['physical_rank_count'] != 8 or x['logical_slots'] != list(range(16)):
                    raise ValueError('actual eight-rank/sixteen-slot receipt failed')
                if mode == 'correctness':
                    expected = {'generator': 3, 'critic': 15, 'ema': 15} if variant == 'no-gan' else {'generator': 201, 'critic': 1005, 'ema': 801}
                    if not x['execution_correctness_admitted'] or x['logical_final_versions'] != expected:
                        raise ValueError('actual commit/version gate failed')
                    correctness[arm] = p
        for pair, order in enumerate(ARMS):
            for arm in order:
                rid = f'helios_h100_resident_{variant}_{arm}_timing_pair{pair}_20261014_r2'
                child_env = dict(env, DISTILLSCHEDULE_HELIOS_H100_CORRECTNESS_SUMMARY=str(correctness[arm]),
                                 DISTILLSCHEDULE_HELIOS_H100_CORRECTNESS_SHA256=sha(correctness[arm]))
                self.invoke(rid, ['bash', launcher, ROOT, rid, variant, 'timing', arms[arm], HEAD, gate, sha(gate)],
                            child_env, deadline=2400, gpu=True)
                p = RUNS / rid / 'summary.json'; x = json.loads(p.read_text())
                if x['expected_head'] != HEAD or x['physical_rank_count'] != 8:
                    raise ValueError('timing source/rank drift')
                self.record_trial(pair, arm, rid, x, x['full5_seconds_max_rank'], p)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workload', choices=('no-gan', 'gan'), required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(plan(args.workload), indent=2))
        return 0
    def interrupted(signum, _frame):
        raise InterruptedError(f'owned campaign interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    runner = Runner(args.workload)
    try:
        runner.admit()
        runner.helios()
        runner.finish()
    except BaseException as error:
        dump(runner.out / 'failure.json', dict(status='FAIL', error=repr(error),
             traceback=traceback.format_exc(), completed_trials=runner.records, strict_claimable=False))
        runner.status('failed', 'FAIL', error=repr(error))
        raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
