"""Canonical workload CLI, called by distillctl workloads or the thin script."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from .registry import list_profiles, load_profile, resolve_profile, verify_assets, write_new_json, repository_root


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "hardware-helios-c2":
        from distillschedule.search.hardware_helios_c2_cli import main as helios_c2_main
        return helios_c2_main(argv[1:])
    if argv and argv[0] == "hardware-helios":
        from distillschedule.search.hardware_helios_cli import main as helios_hardware_main
        return helios_hardware_main(argv[1:])
    if argv and argv[0] == "hardware":
        from distillschedule.search.hardware_wan_cli import main as hardware_main
        return hardware_main(argv[1:])
    if argv and argv[0] == "r9-owner":
        from distillschedule.search.r9_transport import main as r9_main
        return r9_main(argv[1:])
    if argv and argv[0] in {"candidates", "calibrate", "assemble-calibration"}:
        from distillschedule.search.workflow_cli import main as workflow_main
        return workflow_main(argv)
    if argv and argv[0] in {"profile", "search"}:
        from distillschedule.search.cli import main as search_main
        return search_main(argv)
    if argv == ["_collect"]:
        from .evidence import collect
        return collect(json.load(sys.stdin))
    if argv == ["_node"]:
        from .dispatch import execute_node
        return execute_node(json.load(sys.stdin))
    if argv == ["_search"]:
        from distillschedule.search.native import handle_request
        print(json.dumps(handle_request(json.load(sys.stdin)), sort_keys=True, allow_nan=False))
        return 0
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
            sub.add_argument("--search-plan", type=Path, help="Execute a registered, source-bound search plan")
            sub.add_argument("--r9-transport", type=Path, help="Opt-in reviewed R9 deployment for a saved H3 plan")
            sub.add_argument("--coordinator-output", type=Path, help="Fresh local directory for search-plan replay logs")
    args = parser.parse_args(argv)
    try:
        if args.action == "list":
            for p in list_profiles():
                print(f'{p["id"]}\t{p["semantics"]}\t{p["display_name"]}')
            return 0
        if args.action == "run" and args.search_plan:
            from distillschedule.search.execution import run_plan, rebind_plan
            from .registry import _object, _require
            receipt = _object(args.search_plan)
            _require(receipt.get("profile_id") == args.workload, "search-plan workload differs")
            if args.r9_transport:
                from distillschedule.search.r9_transport import rebind_r9_plan
                rebind_plan = rebind_r9_plan
            if not args.dry_run:
                _require(args.coordinator_output is not None, "replay requires a fresh coordinator-output and run-id")
                receipt = rebind_plan(receipt, run_id=args.run_id, assets_path=args.assets_config,
                                      coordinator_output=args.coordinator_output)
            else:
                _require(args.assets_config is None, "dry-run inspects the bound plan; assets override forbidden")
            return run_plan(receipt, dry_run=args.dry_run,
                            coordinator_output=args.coordinator_output / "dispatch" if args.coordinator_output else None,
                            transport_config=_object(args.r9_transport) if args.r9_transport else None)
        if args.action == "run" and args.r9_transport:
            raise ValueError("R9 transport requires a registered serialized --search-plan")
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
