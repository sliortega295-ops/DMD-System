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
REPO = ROOT / 'repos/DMD-System-h100-livewan-resident-20261014-r1'
HEAD = '0997fbe636eb39b8b50d57e28f87f2446cf439e3'
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
                          if workload in ('wan', 'livewan') else 'one No-GAN G1/C5/EMA5 Full5'),
                lock_scope='inherited asset/project/node FDs held through audit/run/teardown')


class Runner:
    def __init__(self, workload):
        self.workload = workload
        self.out = PREP / 'artifacts' / f'{workload}_resident_matched_r2'
        self.out.mkdir(parents=True, exist_ok=False)
        (self.out / 'logs').mkdir()
        self.ordinal = 0
        self.records = []
        dump(self.out / 'manifest.json', plan(workload))

    def status(self, phase, state, **fields):
        dump(self.out / 'status.json', dict(run_id=f'h100_resident_{self.workload}_matched_20261014_r2',
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
        gate = json.loads((PREP / 'manifests/livewan_cpu_gate.json').read_text())
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

    def livewan(self):
        # Reuse the registered, process-local asset environment; never source shell startup.
        loader = RUNS / 'scripts/livewan_h100_environment_r1.sh'
        code = "import json,os; print(json.dumps({k:v for k,v in os.environ.items() if k.startswith(('LIVEWAN_','DISTILLSCHEDULE_')) or k in ('PATH','PYTHONPATH','HF_HOME','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','OMP_NUM_THREADS','PYTHONDONTWRITEBYTECODE','TOKENIZERS_PARALLELISM')}))"
        values = subprocess.check_output(['bash', '-c', f'source "$1"; "$2" -c "$3"', '_', str(loader), str(PYTHON), code], text=True, timeout=30)
        env = clean_env(); env.update(json.loads(values))
        env.update(DMD_SYSTEM_ROOT=str(REPO), DISTILLSCHEDULE_INHERITED_GPU_LOCKS='1',
                   DISTILLSCHEDULE_ASSET_RELOCATION_RECEIPT=str(PREP / 'manifests/livewan_relocation_r1.json'),
                   DISTILLSCHEDULE_ASSET_RELOCATION_SHA256=sha(PREP / 'manifests/livewan_relocation_r1.json'),
                   DISTILLSCHEDULE_CAMPAIGN_ROOT=str(ROOT),
                   PYTHONPATH=f'{env["LIVEWAN_PYTHON_OVERLAY"]}:{REPO}/src:{REPO}:{REPO}/third_party/livewan:{env["LIVEWAN_WAN21_SOURCE"]}')
        preflight = self.out / 'CPU_preflight'
        self.invoke('livewan_latest_CPU_preflight', [PYTHON, PREP / 'scripts/livewan_resident_CPU_preflight_r2.py'],
                    dict(env, CUDA_VISIBLE_DEVICES='', DISTILLSCHEDULE_OUTPUT_DIR=str(preflight)), deadline=1200)
        self.out.joinpath('CPU_preflight/receipt.json').read_text()
        launcher = REPO / 'scripts/distillschedule/launch_livewan_crossscale_world16_node.sh'
        native_arm = {'baseline': 'baseline', 'candidate': 'pipeline'}
        variant = 'c34-continuous-b16-residual-layout'

        def run_arm(arm, kind, rid):
            run = RUNS / rid
            child_env = dict(env, DISTILLSCHEDULE_OUTPUT_DIR=str(run),
                             LIVEWAN_PIPELINE_VARIANT=variant if arm == 'candidate' else 'b1',
                             LIVEWAN_BASELINE_PLACEMENT='full-resident' if arm == 'baseline' else 'legacy-hybrid',
                             LIVEWAN_TEACHER_PARAMETER_OFFLOAD='off',
                             LIVEWAN_TRAJECTORY_CYCLES='2', LIVEWAN_TRAJECTORY_SEED='20260830',
                             LIVEWAN_TRAJECTORY_ARM_ID=f'h100-{native_arm[arm]}-{variant if arm == "candidate" else "b1"}',
                             LIVEWAN_TWO_CYCLE_SEED='20260830', LIVEWAN_EXPECTED_HEAD=HEAD)
            child_env['LIVEWAN_TWO_CYCLE_ARM_ID'] = child_env['LIVEWAN_TRAJECTORY_ARM_ID']
            self.invoke(rid, ['bash', launcher, '0', rid, kind, native_arm[arm], HEAD, '127.0.0.1', '29732'],
                        child_env, deadline=3600, gpu=True)
            summary = run / ('trajectory_summary.json' if kind == 'trajectory' else 'timing_summary.json')
            cmd = [PYTHON, REPO / f'scripts/distillschedule/summarize_livewan_{"trajectory" if kind == "trajectory" else "two_cycle_timing"}_world16.py',
                   '--run-root', run, '--expected-head', HEAD, '--hardware-profile', 'h100-8-b16', '--output', summary]
            if kind == 'trajectory':
                cmd += ['--cycles', '2']
            else:
                cmd += ['--execution-mode', native_arm[arm], '--pipeline-variant', child_env['LIVEWAN_PIPELINE_VARIANT'],
                        '--trace', env['LIVEWAN_OPERATION_TRACE']]
            self.invoke(f'{rid}_summarize', cmd, child_env, deadline=300)
            x = json.loads(summary.read_text())
            if x['status'] != 'PASS':
                raise ValueError('LiveWan actual receipt failed')
            return run, summary, x

        sides = {}
        for arm in ('baseline', 'candidate'):
            sides[arm] = run_arm(arm, 'trajectory', f'livewan_h100_resident_{arm}_correctness_20261014_r2')
        comparison = self.out / 'trajectory_comparison.json'
        self.invoke('livewan_unchanged_numerical_comparator',
                    [PYTHON, REPO / 'scripts/distillschedule/compare_livewan_two_cycle_trajectories.py',
                     '--baseline-run-root', sides['baseline'][0], '--pipeline-run-root', sides['candidate'][0],
                     '--baseline-summary', sides['baseline'][1], '--pipeline-summary', sides['candidate'][1],
                     '--expected-head', HEAD, '--pipeline-variant', variant, '--hardware-profile', 'h100-8-b16',
                     '--output', comparison], env, deadline=1200)
        gate = self.out / 'timing_gate.json'
        self.invoke('livewan_fresh_matched_timing_gate',
                    [PYTHON, REPO / 'scripts/distillschedule/build_livewan_two_cycle_timing_gate.py',
                     '--baseline-trajectory-summary', sides['baseline'][1], '--pipeline-trajectory-summary', sides['candidate'][1],
                     '--trajectory-comparison', comparison, '--expected-head', HEAD, '--pipeline-variant', variant,
                     '--hardware-profile', 'h100-8-b16', '--output', gate], env, deadline=300)
        baseline_gate = self.out / 'resident_timing_gate.json'
        comparison_value = json.loads(comparison.read_text())
        baseline_summary = json.loads(sides['baseline'][1].read_text())
        from distillschedule.runtime.livewan_baseline_placement import BaselinePlacement, validate_timing_gate
        from distillschedule.runtime.livewan_hardware import livewan_hardware
        spec = BaselinePlacement('full-resident', 'off')
        value = dict(schema_version='distillschedule.livewan_baseline_placement_timing_gate.v1',
            status='PASS', claimable_for_two_cycle_timing=True, cycles=2,
            final_versions={'generator':2, 'critic':10, 'ema':0}, expected_head=HEAD,
            baseline_placement=spec.receipt(), hardware_profile=livewan_hardware('h100-8-b16').to_dict(),
            identity=baseline_summary['identity'], seed=baseline_summary['seed'], arm_id=baseline_summary['arm_id'],
            numerical_policy='REJECT_EXTREME_ONLY_RECORD_ORDINARY_REVIEW',
            correctness_summary={'path':str(sides['baseline'][1]), 'sha256':sha(sides['baseline'][1])},
            comparison={'path':str(comparison), 'sha256':sha(comparison)},
            common_matched_gate={'path':str(gate),'sha256':sha(gate)})
        validate_timing_gate(value,spec=spec,expected_head=HEAD,identity=baseline_summary['identity'],
                            arm_id=baseline_summary['arm_id'],seed=baseline_summary['seed'])
        dump(baseline_gate,value)
        env.update(LIVEWAN_PAIR_CORRECTNESS_RECEIPT=str(gate), LIVEWAN_PAIR_CORRECTNESS_SHA256=sha(gate))
        for pair, order in enumerate(ARMS):
            for arm in order:
                rid = f'livewan_h100_resident_{arm}_timing_pair{pair}_20261014_r2'
                env.update(LIVEWAN_PAIR_CORRECTNESS_RECEIPT=str(baseline_gate if arm == 'baseline' else gate),
                           LIVEWAN_PAIR_CORRECTNESS_SHA256=sha(baseline_gate if arm == 'baseline' else gate))
                _, summary_path, x = run_arm(arm, 'two-cycle-timing', rid)
                if x['final_versions'] != {'generator': 2, 'critic': 10, 'ema': 0} or x['rank_count'] != 8:
                    raise ValueError('LiveWan actual rank/version gate failed')
                self.record_trial(pair, arm, rid, x, x['cycle_2']['seconds'], summary_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workload', choices=('livewan',), required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(plan(args.workload), indent=2))
        return 0
    def interrupted(signum, _frame):
        raise InterruptedError(f'owned campaign interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    os.environ['DISTILLSCHEDULE_HARDWARE_PROFILE'] = 'h100-8-b16'
    import sys
    sys.path.insert(0,str(REPO / 'src'))
    runner = Runner(args.workload)
    try:
        runner.admit()
        runner.livewan()
        runner.finish()
    except BaseException as error:
        dump(runner.out / 'failure.json', dict(status='FAIL', error=repr(error),
             traceback=traceback.format_exc(), completed_trials=runner.records, strict_claimable=False))
        runner.status('failed', 'FAIL', error=repr(error))
        raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
