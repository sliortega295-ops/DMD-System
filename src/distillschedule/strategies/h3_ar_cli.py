"""W6 task adapter on the existing distillctl workloads native front door."""
import argparse
import json
from pathlib import Path
import sys

from distillschedule.workloads.best_known.registry import write_new_json
from .h3_ar_task import H3ARTask, H3ARImplementation


def main(argv=None):
    parser = argparse.ArgumentParser(prog='distillctl workloads native [original H3 AR]')
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('validate', 'prepare-assets', 'calibrate', 'assemble', 'plan', 'run', 'benchmark-manual'):
        p = sub.add_parser(name)
        p.add_argument('--task', type=Path, required=True)
        p.add_argument('--output', type=Path, required=name != 'validate')
        if name == 'prepare-assets':
            p.add_argument('--native-manifest', type=Path, required=True)
            p.add_argument('--checkpoint-audit', type=Path,
                           help='reuse stat-bound owner audit, or hash once before reserving GPUs')
        if name in ('calibrate', 'run', 'benchmark-manual'):
            p.add_argument('--assets', type=Path, required=True)
            p.add_argument('--capacity-policy', type=Path, required=True)
            p.add_argument('--execute', action='store_true')
            if name == 'run':
                p.add_argument('--plan', type=Path, required=True)
            else:
                p.add_argument('--implementation', choices=('zero3-native', 'zero3-ap'), required=True)
        if name == 'assemble':
            p.add_argument('--rank-receipts', nargs='+', type=Path, required=True)
        if name == 'plan':
            p.add_argument('--calibration', nargs='+', type=Path, required=True)
            p.add_argument('--memory-limit-bytes', type=int)
            p.add_argument('--max-states', type=int, default=50000)
            p.add_argument('--beam-width', type=int, default=0)
            p.add_argument('--timeout', type=float, default=60)
    args = parser.parse_args(argv)
    try:
        task = H3ARTask.from_dict(json.loads(args.task.read_text()))
        if args.command == 'validate' or (args.command in ('calibrate', 'run', 'benchmark-manual') and not args.execute):
            result = dict(task=task.to_dict(), census=task.census, implementations=[H3ARImplementation(i).to_dict()
                for i in ('zero3-native', 'zero3-ap')],
                native_execution='NOT_RUN_REQUIRES_EXPLICIT_EXECUTE', gpu_execution='NOT_RUN',
                calibration='fresh same-source world16 two-native-cycle receipts required; historical seconds are not costs',
                scope='whole-native-update only; no arbitrary stage/layout/precision search', claimable_speedup=False)
            if args.command == 'validate' and args.output:
                write_new_json(args.output, result)
        elif args.command == 'prepare-assets':
            from .h3_ar_assets import prepare_from_native_manifest
            audit = args.checkpoint_audit or args.output.parent/'checkpoint-audit.json'
            result = prepare_from_native_manifest(args.native_manifest, checkpoint_audit_path=audit).to_dict()
            write_new_json(args.output, result)
        elif args.command == 'assemble':
            from .h3_ar_planning import assemble
            result = assemble(task, args.rank_receipts, args.output)
        elif args.command == 'plan':
            from .h3_ar_planning import plan
            result = plan(task, args.calibration, memory_limit_bytes=args.memory_limit_bytes,
                max_states=args.max_states, beam_width=args.beam_width, timeout_seconds=args.timeout)
            write_new_json(args.output, result)
        else:
            from .h3_ar_assets import H3ARAssets
            from .h3_ar_worker import run_worker
            result = run_worker(task, H3ARAssets.from_dict(json.loads(args.assets.read_text())), args.output,
                mode='manual' if args.command == 'benchmark-manual' else args.command,
                capacity_policy=json.loads(args.capacity_policy.read_text()),
                implementation=H3ARImplementation(args.implementation) if args.command != 'run' else None,
                plan_path=getattr(args, 'plan', None))
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
        return 0
    except (ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 2
