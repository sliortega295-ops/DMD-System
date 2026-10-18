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
    for name in ('validate', 'calibrate', 'assemble', 'plan', 'run', 'benchmark-manual', 'prepare-campaign', 'campaign', 'capture-inputs', 'prepare-task', 'auto-run', 'prepare-stream', 'train-stream'):
        p = sub.add_parser(name)
        p.add_argument('--task', type=Path, required=True)
        p.add_argument('--output', type=Path, required=name not in ('validate','campaign'))
        if name == 'validate':
            p.add_argument('--include-incumbent-bundle', action='store_true')
            p.add_argument('--include-critic-retention', action='store_true',
                           help='Opt-in bounded retainC26-29 inventory; needs incumbent capability and fresh calibration')
        if name in ('calibrate', 'run', 'train-stream', 'benchmark-manual'):
            p.add_argument('--assets', type=Path, required=True)
            p.add_argument('--execute', action='store_true')
            if name != 'train-stream':
                p.add_argument('--record-memory-highwater', action='store_true',
                               help='host-only opt-in accounting across native peak-counter resets')
            if name in ('calibrate', 'benchmark-manual'):
                p.add_argument('--implementation', type=Path, required=True,
                               help='admitted primitive strategy JSON from validate, not a recipe name')
                if name == 'benchmark-manual':
                    p.add_argument('--cycles', type=int, default=2)
            else:
                p.add_argument('--plan', type=Path, required=True)
                if name == 'run':
                    p.add_argument('--cycles', type=int, default=2)
                else:
                    p.add_argument('--stream', type=Path, required=True)
        if name in ('capture-inputs', 'prepare-task', 'auto-run'):
            p.add_argument('--sources', type=Path, required=True)
            p.add_argument('--training-config', type=Path, required=True)
            p.add_argument('--execute', action='store_true')
            if name in ('prepare-task', 'auto-run'):
                p.add_argument('--model-assets', type=Path, required=True)
                p.add_argument('--without-fusion', action='store_true')
                p.add_argument('--max-launches', type=int, default=8)
                p.add_argument('--resume', action='store_true')
        if name == 'prepare-stream':
            p.add_argument('--sources', type=Path, required=True)
            p.add_argument('--training-config', type=Path, required=True)
            p.add_argument('--execute', action='store_true')
            p.add_argument('--resume', action='store_true')
        if name == 'auto-run':
            p.add_argument('--node-rank', type=int, required=True)
            p.add_argument('--master-addr', required=True)
            p.add_argument('--master-port', type=int, required=True)
            p.add_argument('--preparation-timeout', type=float, default=7200)
            p.add_argument('--launch-timeout', type=float, default=7200)
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
        if args.command == 'validate' or (args.command in {'calibrate', 'run', 'train-stream', 'benchmark-manual'} and not args.execute):
            result = capability_report(task, triton_available=has_triton,
                incumbent_bundle_available=getattr(args, "include_incumbent_bundle", False),
                critic_retention_available=getattr(args, "include_critic_retention", False))
            result['native_execution'] = 'NOT_RUN_REQUIRES_EXPLICIT_EXECUTE'
            if args.command == 'validate' and args.output:
                write_new_json(args.output, result)
        elif args.command == 'prepare-stream':
            from .wan_stream import WanCohortSources, prepare_stream
            result = prepare_stream(task, WanCohortSources.from_dict(json.loads(args.sources.read_text())),
                                    args.training_config, args.output, execute=args.execute, resume=args.resume)
        elif args.command == 'train-stream':
            from .wan_stream import load_stream
            from .wan_worker import run_worker
            stream = load_stream(task, args.stream)
            assets = OriginalWanAssets.from_dict(json.loads(args.assets.read_text()))
            result = run_worker(task, assets, output=args.output, mode='run', plan_path=args.plan,
                                cycles=len(stream['cohorts']), stream_path=args.stream)
        elif args.command in ('capture-inputs', 'prepare-task', 'auto-run'):
            from .wan_inputs import WanInputSources, preflight_sources
            from .wan_capture import capture, read_training_config
            sources = WanInputSources.from_dict(json.loads(args.sources.read_text()))
            if args.command == 'auto-run':
                from .wan_autoplan import run as auto_run
                result = auto_run(task, sources, json.loads(args.model_assets.read_text()),
                    args.training_config, args.output, node_rank=args.node_rank,
                    master_addr=args.master_addr, master_port=args.master_port,
                    include_fusion=not args.without_fusion, max_launches=args.max_launches,
                    execute=args.execute, resume=args.resume,
                    preparation_timeout=args.preparation_timeout, launch_timeout=args.launch_timeout)
            elif args.command == 'prepare-task':
                from .wan_prepare import prepare
                result = prepare(task, sources, json.loads(args.model_assets.read_text()),
                                 args.training_config, args.output, execute=args.execute, resume=args.resume,
                                 include_fusion=not args.without_fusion, max_launches=args.max_launches)
            elif args.execute:
                result = capture(task, sources, args.training_config, args.output)
            else:
                read_training_config(args.training_config)
                result = preflight_sources(task, sources)
                result['capture'] = 'NOT_RUN_REQUIRES_EXPLICIT_EXECUTE'
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
            strategy = WanStrategy.from_dict(json.loads(args.implementation.read_text())) if args.command in ('calibrate', 'benchmark-manual') else None
            mode = 'manual' if args.command == 'benchmark-manual' else args.command
            result = run_worker(task, assets, output=args.output, mode=mode, strategy=strategy,
                                plan_path=getattr(args, 'plan', None), cycles=getattr(args, 'cycles', 2),
                                **({'record_memory_highwater': True} if args.record_memory_highwater else {}))
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
        return 0
    except (ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 2
