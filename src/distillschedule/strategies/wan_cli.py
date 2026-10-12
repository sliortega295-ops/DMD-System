"""Original-protocol front door inside distillctl workloads native.

Each original task uses the SAME worker; there is no per-size launcher or hidden
Euler fallback. GPU work requires --execute and an explicit torchrun environment.
"""
from __future__ import annotations
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys

from distillschedule.workloads.best_known.registry import write_new_json
from .wan_task import OriginalWanTask
from .wan_strategy import WanStrategy, capability_report
from .wan_worker import OriginalWanAssets


def main(argv=None):
    parser = argparse.ArgumentParser(prog='distillctl workloads native [original-Wan]')
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('validate', 'calibrate', 'assemble', 'plan', 'run', 'prepare-campaign', 'campaign'):
        p = sub.add_parser(name)
        p.add_argument('--task', type=Path, required=True)
        p.add_argument('--output', type=Path, required=name not in ('validate','campaign'))
        if name in ('calibrate', 'run'):
            p.add_argument('--assets', type=Path, required=True)
            p.add_argument('--execute', action='store_true')
            if name == 'calibrate':
                p.add_argument('--implementation', type=Path, required=True,
                               help='admitted primitive strategy JSON from validate, not a recipe name')
            else:
                p.add_argument('--plan', type=Path, required=True)
                p.add_argument('--cycles', type=int, default=2)
        if name == 'prepare-campaign':
            p.add_argument('--assets', type=Path, required=True)
            p.add_argument('--without-fusion', action='store_true')
            p.add_argument('--max-launches', type=int, default=8)
        if name == 'campaign':
            p.add_argument('--campaign', type=Path, required=True)
            p.add_argument('--node-rank', type=int, required=True)
            p.add_argument('--master-addr', required=True)
            p.add_argument('--master-port', type=int, required=True)
            p.add_argument('--execute', action='store_true')
            p.add_argument('--resume', action='store_true')
        if name == 'assemble':
            p.add_argument('--rank-receipts', type=Path, nargs='+', required=True)
            p.add_argument('--reserve-bytes', type=int, default=2**31)
        if name == 'plan':
            p.add_argument('--calibration', type=Path, nargs='+', required=True)
            p.add_argument('--max-states', type=int, default=50000)
            p.add_argument('--beam-width', type=int, default=0)
            p.add_argument('--timeout', type=float, default=60)
            p.add_argument('--memory-limit-bytes', type=int)
    args = parser.parse_args(argv)
    try:
        task = OriginalWanTask.from_dict(json.loads(args.task.read_text()))
        try:
            importlib.metadata.version('triton'); has_triton = True
        except importlib.metadata.PackageNotFoundError:
            has_triton = False
        if args.command == 'validate' or (args.command in {'calibrate', 'run'} and not args.execute):
            result = capability_report(task, triton_available=has_triton)
            result['native_execution'] = 'NOT_RUN_REQUIRES_EXPLICIT_EXECUTE'
            if args.command == 'validate' and args.output:
                write_new_json(args.output, result)
        elif args.command == 'prepare-campaign':
            from .wan_campaign import prepare
            assets = OriginalWanAssets.from_dict(json.loads(args.assets.read_text()))
            result = prepare(task, assets, args.output, include_fusion=not args.without_fusion,
                             max_launches=args.max_launches)
        elif args.command == 'campaign':
            from .wan_campaign import run, verify
            from distillschedule.core.contracts import require
            require(verify(args.campaign)['task_sha256'] == task.digest, 'campaign/task mismatch', path='wan.campaign')
            result = run(args.campaign, node_rank=args.node_rank, master_addr=args.master_addr,
                         master_port=args.master_port, execute=args.execute, resume=args.resume)
        elif args.command == 'assemble':
            from .wan_planning import assemble
            result = assemble(task, args.rank_receipts, args.output, reserve_bytes=args.reserve_bytes)
        elif args.command == 'plan':
            from .wan_planning import plan
            result = plan(task, args.calibration, max_states=args.max_states, beam_width=args.beam_width,
                          timeout_seconds=args.timeout, memory_limit_bytes=args.memory_limit_bytes)
            write_new_json(args.output, result)
        else:
            from .wan_worker import run_worker
            assets = OriginalWanAssets.from_dict(json.loads(args.assets.read_text()))
            strategy = WanStrategy.from_dict(json.loads(args.implementation.read_text())) if args.command == 'calibrate' else None
            result = run_worker(task, assets, output=args.output, mode=args.command, strategy=strategy,
                                plan_path=getattr(args, 'plan', None), cycles=getattr(args, 'cycles', 2))
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
        return 0
    except (ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 2
