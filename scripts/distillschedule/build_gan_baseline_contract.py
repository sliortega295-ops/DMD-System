#!/usr/bin/env python3
"""Build/finalize the immutable official-semantics GAN Route-A contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from distillschedule.core.contracts import ContractViolation
from distillschedule.runtime.gan_baseline import (
    atomic_json,
    build_node_contract,
    finalize_node_contracts,
    load_json_mapping,
    sha256_file,
    validate_final_contract,
)


def asset(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("asset must be NAME=PATH")
    name, path = value.split("=", 1)
    if not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("asset must use non-empty NAME=PATH")
    return name.strip(), path.strip()


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="command", required=True)
    node = sub.add_parser("node")
    node.add_argument("--node-rank", type=int, choices=(0, 1), required=True)
    node.add_argument("--plan", type=Path, required=True)
    node.add_argument("--compiled-preflight", type=Path, required=True)
    node.add_argument("--rendered-config", type=Path, required=True)
    node.add_argument("--accelerate-config", type=Path, required=True)
    node.add_argument("--trace-bundle", type=Path, required=True)
    node.add_argument("--checkpoint", type=Path, required=True)
    node.add_argument("--runtime-identity", type=Path, required=True)
    node.add_argument("--helios-root", type=Path, required=True)
    node.add_argument("--asset", action="append", default=[], type=asset)
    node.add_argument("--output", type=Path, required=True)

    final = sub.add_parser("finalize")
    final.add_argument("--node-contract", action="append", type=Path, required=True)
    final.add_argument("--output", type=Path, required=True)

    check = sub.add_parser("check")
    check.add_argument("contract", type=Path)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "node":
            payload = build_node_contract(
                node_rank=args.node_rank,
                plan_path=args.plan,
                compiled_preflight_path=args.compiled_preflight,
                rendered_config_path=args.rendered_config,
                accelerate_config_path=args.accelerate_config,
                trace_bundle_path=args.trace_bundle,
                checkpoint_path=args.checkpoint,
                runtime_identity_path=args.runtime_identity,
                helios_root=args.helios_root,
                assets=dict(args.asset),
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
            payload = finalize_node_contracts(args.node_contract)
            output = atomic_json(args.output, payload)
            result = {
                "status": "PASS",
                "output": str(output),
                "sha256": sha256_file(output),
                "contract_digest": payload["contract_digest"],
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
    except (ContractViolation, OSError, ValueError, KeyError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
