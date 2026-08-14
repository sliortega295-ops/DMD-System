#!/usr/bin/env python3
"""Build and finalize immutable Route-A world16 evidence contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from distillschedule.core.contracts import ContractViolation
from distillschedule.runtime.routea_evidence import (
    atomic_json,
    build_node_contract,
    finalize_node_contracts,
    load_json_mapping,
    sha256_file,
    validate_final_contract,
)


def _asset(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("asset must use NAME=PATH")
    name, path = value.split("=", 1)
    if not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("asset must use non-empty NAME=PATH")
    return name.strip(), path.strip()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    node = subparsers.add_parser("node", help="build one node-qualified contract")
    node.add_argument("--node-rank", required=True, type=int, choices=(0, 1))
    node.add_argument("--plan", required=True, type=Path)
    node.add_argument("--repo-root", required=True, type=Path)
    node.add_argument("--compiled-preflight", required=True, type=Path)
    node.add_argument("--rendered-config", required=True, type=Path)
    node.add_argument("--accelerate-config", required=True, type=Path)
    node.add_argument("--trace-bundle", required=True, type=Path)
    node.add_argument("--checkpoint", required=True, type=Path)
    node.add_argument("--asset", action="append", default=[], type=_asset)
    node.add_argument("--output", required=True, type=Path)

    finalize = subparsers.add_parser(
        "finalize", help="require matching node0/node1 normalized contracts"
    )
    finalize.add_argument("--node-contract", action="append", required=True, type=Path)
    finalize.add_argument("--output", required=True, type=Path)

    check = subparsers.add_parser("check", help="validate a finalized contract")
    check.add_argument("contract", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "node":
            assets = dict(args.asset)
            payload = build_node_contract(
                node_rank=args.node_rank,
                plan_path=args.plan,
                repo_root=args.repo_root,
                compiled_preflight_path=args.compiled_preflight,
                rendered_config_path=args.rendered_config,
                accelerate_config_path=args.accelerate_config,
                trace_bundle_path=args.trace_bundle,
                checkpoint_path=args.checkpoint,
                assets=assets,
            )
            output = atomic_json(args.output, payload)
            result = {
                "status": "PASS",
                "node_rank": args.node_rank,
                "output": str(output),
                "sha256": sha256_file(output),
                "shared_digest": payload["shared_digest"],
            }
        elif args.command == "finalize":
            if len(args.node_contract) != 2:
                raise ContractViolation(
                    "exactly two --node-contract values are required", path="nodes"
                )
            payload = finalize_node_contracts(args.node_contract)
            output = atomic_json(args.output, payload)
            result = {
                "status": "PASS",
                "output": str(output),
                "sha256": sha256_file(output),
                "contract_digest": payload["contract_digest"],
                "shared_digest": payload["shared_digest"],
            }
        else:
            payload = load_json_mapping(args.contract)
            validate_final_contract(payload)
            result = {
                "status": "PASS",
                "contract": str(args.contract.resolve()),
                "sha256": sha256_file(args.contract),
                "contract_digest": payload["contract_digest"],
            }
    except (ContractViolation, OSError, ValueError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
