"""Separate native W6 Resident CPU-boundary capacity arm, with owned lease."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys

import run_w6_native_three_arm_r3 as base

ROOT = base.ROOT
OWNER = base.OWNER
base.REPO = ROOT / 'repos/DMD-System-h100-w6-native-resident-boundary-20261014-r1'
base.HEAD = 'c0aeb5389c63d39ff86681ee452001b352f80c8e'
base.WORKER = base.REPO / 'scripts/distillschedule/probe_h3_diffsynth_h100_world8.py'
base.ACTIVATE = OWNER / 'scripts/h3_w6_resident_boundary_environment_r1.sh'
base.WORKER_EXTRA_ARGS = ('--checkpoint-boundary-cpu',)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--tiny-only', action='store_true')
    args = parser.parse_args()
    parent = base.CAMPAIGN / args.run_id
    parent.mkdir(exist_ok=False)
    base.source_gate()
    assert os.environ.get('DISTILLSCHEDULE_INHERITED_GPU_LOCKS') == '1'
    subprocess.run([str(base.PYTHON), str(ROOT/'repos/DMD-System-h100-helios-b16-20260930/scripts/distillschedule/verify_inherited_gpu_locks.py'),
                    '--project-root', str(ROOT), '--asset-fd', '7'], check=True, pass_fds=(7,8,9))
    inputs = json.loads(base.INPUT.read_text())
    assert inputs['status'] == 'PASS_CPU_NATIVE_16_REAL_ROWS_MODEL_LAYOUT'
    assert base.sha(inputs['cache_manifest']) == inputs['cache_manifest_sha256']
    assert base.sha(inputs['config_path']) == inputs['config_sha256']
    report = {'run_id': args.run_id, 'head': base.HEAD, 'status': 'FAIL_PRESERVED',
              'arm': 'resident-boundary-cpu', 'results': [],
              'activation_policy': 'native_per_checkpoint_input_pinned_cpu',
              'sigma': 'UNCHANGED_CONTINUOUS_SHIFTED_LOGIT_NORMAL',
              'parameter_offload': False, 'model_swap': False,
              'pure_placement_comparison': False, 'replications': 1,
              'strict_cross_arm_numerical_parity': 'NOT_RUN'}
    modes = ('correctness',) if args.tiny_only else ('correctness','timing')
    try:
        for mode in modes:
            summary, out = base.arm(parent, 'resident', mode, inputs, args.tiny_only)
            report['results'].append({'path': str(out), **summary})
            if not summary['status'].startswith('PASS_'):
                report['failure_receipts'] = [str(p) for p in (out/'rank_receipts').glob('*_failure.json')]
                break
            rows = [json.loads((out/f'rank_receipts/rank_{i:02d}.json').read_text()) for i in range(8)]
            for row in rows:
                assert row['activation_offload'] is True and not row['parameter_offload']
                for cycle in row['cycles']:
                    policy = cycle['checkpoint_boundary']
                    expected = 8 if args.tiny_only else 200
                    assert policy['enabled'] and policy['counters']['checkpoint_calls'] == expected
                    assert policy['counters']['D2H_count'] == policy['counters']['H2D_count']
                    assert policy['counters']['D2H_bytes'] == policy['counters']['H2D_bytes']
                    assert policy['counters']['D2H_bytes'] > 0
                    assert policy['recompute_hook_scope'] == 'outside_offload_wrapper'
            if mode == modes[-1]:
                report['status'] = 'PASS_TINY_NATIVE_BOUNDARY_ONLY' if args.tiny_only else 'PASS_REAL_NATIVE_RESIDENT_BOUNDARY_CAPACITY_AND_SINGLE_TIMING'
                report['single_timing_seconds'] = None if args.tiny_only else summary['timed_C2_seconds']
    except BaseException as exc:
        report.update(error_type=type(exc).__name__, error=str(exc))
    finally:
        base.write(parent/'result.json', report)
    print(json.dumps({k:v for k,v in report.items() if k != 'results'}), flush=True)
    return 0 if report['status'].startswith('PASS_') else 1


if __name__ == '__main__':
    raise SystemExit(main())
