"""Canonical workload CLI, called by distillctl workloads or the thin script."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from .registry import list_profiles, load_profile, resolve_profile, verify_assets, write_new_json, repository_root


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv == ["_collect"]:
        from .evidence import collect
        return collect(json.load(sys.stdin))
    if argv == ["_node"]:
        from .dispatch import execute_node
        return execute_node(json.load(sys.stdin))
    parser = argparse.ArgumentParser(prog="distillctl workloads")
    subs = parser.add_subparsers(dest="action", required=True)
    subs.add_parser("list", help="List exactly five canonical workloads without Torch or GPU imports")
    for action in ("explain", "validate", "render", "run"):
        sub = subs.add_parser(action)
        sub.add_argument("--workload", required=True)
        sub.add_argument("--assets-config", type=Path)
        sub.add_argument("--run-id", default="source-inspection")
        sub.add_argument("--mode", default="correctness")
        if action == "render":
            sub.add_argument("--output", type=Path, required=True)
        if action == "run":
            sub.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.action == "list":
            for p in list_profiles():
                print(f'{p["id"]}\t{p["semantics"]}\t{p["display_name"]}')
            return 0
        receipt = resolve_profile(args.workload, assets_path=args.assets_config, run_id=args.run_id, mode=args.mode)
        if args.action == "explain":
            print(json.dumps(receipt, indent=2, sort_keys=True))
        elif args.action == "render":
            if args.assets_config is None:
                parser.error("render requires --assets-config")
            write_new_json(args.output, receipt)
            print(args.output)
        elif args.action == "validate":
            if args.assets_config:
                receipt["asset_checks"] = verify_assets(load_profile(args.workload), receipt["asset_config"])
                receipt["validation"]["asset_verification"] = "PASS_LOCAL_IDENTITY_FILES"
            # Source-only verification never invokes torchrun, NCCL or model imports.
            print(json.dumps(receipt, indent=2, sort_keys=True))
        else:
            if args.assets_config is None:
                parser.error("run requires --assets-config")
            from .dispatch import run_resolved
            return run_resolved(receipt, dry_run=args.dry_run)
        return 0
    except (ValueError, KeyError, OSError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
