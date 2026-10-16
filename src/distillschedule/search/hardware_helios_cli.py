"""Read-only CPU bridge CLI. No run/resume/GPU command is implemented."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from .hardware_helios import ARMS, MODES, audit_completed_run, compile_native_plan, inspect_native_source


def main(argv=None):
    parser = argparse.ArgumentParser(prog="distillctl workloads hardware-helios",
        description="Compile the pinned native Full5 command or audit raw receipts. Does not execute GPU work or resume the stopped campaign.")
    sub = parser.add_subparsers(dest="action", required=True)
    inspect = sub.add_parser("inspect-source")
    inspect.add_argument("--source-root", type=Path, required=True)
    compile_ = sub.add_parser("compile")
    compile_.add_argument("--deployment", type=Path, required=True)
    compile_.add_argument("--source-root", type=Path)
    compile_.add_argument("--arm", choices=ARMS, required=True)
    compile_.add_argument("--mode", choices=MODES, required=True)
    compile_.add_argument("--run-id", required=True)
    compile_.add_argument("--correctness", type=Path, help="JSON object containing the native correctness summary path and SHA256")
    compile_.add_argument("--output", type=Path, required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("--plan", type=Path, required=True)
    audit.add_argument("--source-root", type=Path)
    audit.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.action == "inspect-source":
            result = inspect_native_source(args.source_root)
        elif args.action == "compile":
            result = compile_native_plan(json.loads(args.deployment.read_text()), arm=args.arm,
                mode=args.mode, run_id=args.run_id, source_root=args.source_root,
                correctness=json.loads(args.correctness.read_text()) if args.correctness else None)
        else:
            result = audit_completed_run(json.loads(args.plan.read_text()), source_root=args.source_root)
        if args.action != "inspect-source":
            with args.output.open("x") as stream:
                json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
                stream.write("\n")
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (ValueError, TypeError, KeyError, OSError, subprocess.SubprocessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
