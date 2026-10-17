"""distillctl workloads native: explicit validate/calibrate/plan/run commands."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from .task import NativeTask
from .planning import capabilities, calibrate, plan, run
from distillschedule.workloads.best_known.registry import write_new_json


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv == ['list-adapters']:
        print(json.dumps(dict(adapters=[
            dict(schema_version='distillschedule.native_flow_task.v1', scope='new explicit flow protocol; not historical W6'),
            dict(schema_version='distillschedule.original_wan_task.v1', scope='original Wan pyramid no-GAN'),
            dict(schema_version='distillschedule.original_h3_ar_task.v1', scope='W6 original DiffSynth previous/current AV DMD2')],
            canonical_profiles='UNCHANGED_FIVE_FROZEN_PROFILES', gpu_execution='NOT_RUN'), indent=2))
        return 0
    if argv and argv[0] == "program":
        from distillschedule.open_workload.cli import main as program_main
        return program_main(argv[1:])
    routing = argparse.ArgumentParser(add_help=False)
    routing.add_argument("--task", type=Path)
    known, _ = routing.parse_known_args(argv)
    if known.task is not None:
        try:
            raw = json.loads(known.task.read_text())
            if isinstance(raw, dict) and raw.get("schema_version") == "distillschedule.original_wan_task.v1":
                from distillschedule.strategies.wan_cli import main as original_wan_main
                return original_wan_main(argv)
            if isinstance(raw, dict) and raw.get("schema_version") == "distillschedule.original_h3_ar_task.v1":
                from distillschedule.strategies.h3_ar_cli import main as original_h3_ar_main
                return original_h3_ar_main(argv)
        except (OSError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
    parser=argparse.ArgumentParser(prog="distillctl workloads native",
        description="Task-schema routed native adapters; use list-adapters for read-only discovery.")
    sub=parser.add_subparsers(dest="command",required=True)
    for name in ("validate","calibrate","plan","run"):
        p=sub.add_parser(name)
        p.add_argument("--task",type=Path,required=True)
        p.add_argument("--output",type=Path,required=name!="validate")
        if name in ("calibrate","run"):
            p.add_argument("--execute",action="store_true",help="explicitly authorize local native model execution")
        if name=="plan":
            p.add_argument("--calibration",type=Path,required=True)
            p.add_argument("--max-states",type=int,default=50000)
            p.add_argument("--beam-width",type=int,default=0)
            p.add_argument("--timeout",type=float,default=60)
        if name=="run":
            p.add_argument("--plan",type=Path,required=True)
            p.add_argument("--cycles",type=int,default=2)
    args=parser.parse_args(argv)
    try:
        task=NativeTask.from_dict(json.loads(args.task.read_text()))
        if args.command=="validate" or (args.command in ("calibrate","run") and not args.execute):
            result=capabilities(task)
            result["native_execution"]="NOT_RUN_REQUIRES_EXPLICIT_EXECUTE"
            if args.command=="validate" and args.output:write_new_json(args.output,result)
        elif args.command=="calibrate":result=calibrate(task,args.output)
        elif args.command=="plan":
            result=plan(task,args.calibration,max_states=args.max_states,beam_width=args.beam_width,timeout_seconds=args.timeout)
            write_new_json(args.output,result)
        else:result=run(task,args.plan,args.output,cycles=args.cycles)
        print(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))
        return 0
    except (ValueError,TypeError,KeyError,OSError,RuntimeError) as exc:
        print(f"ERROR: {exc}",file=sys.stderr)
        return 2
