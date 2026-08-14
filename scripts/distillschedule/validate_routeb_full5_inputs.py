#!/usr/bin/env python3
"""Low-cost fail-closed input gate for Route-B Candidate A."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from runtime.critic_pool_checkpoint import LogicalCheckpointPackage
from runtime.operation_trace import load_operation_trace_bundle_manifest
from runtime.routeb_full5 import RouteBFull5Contract
from runtime.routeb_full5_trace import RouteBFull5Trace


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _native_inventory(root: Path, component: str) -> dict[str, Any]:
    parent = root if component == "generator" else root / "critic"
    checkpoint = parent / "pytorch_model"
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    optimizer = sorted(checkpoint.glob("bf16_zero_pp_rank_*_mp_rank_00_optim_states.pt"))
    expected = [
        f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt"
        for rank in range(8)
    ]
    if [path.name for path in optimizer] != expected:
        raise ValueError(f"{component} native ZeRO-1 shard inventory changed")
    model = checkpoint / "mp_rank_00_model_states.pt"
    if not model.is_file() or model.stat().st_size <= 0:
        raise ValueError(f"{component} native model state is missing")
    return {
        "component": component,
        "checkpoint": str(checkpoint.resolve()),
        "optimizer_shards": len(optimizer),
        "optimizer_bytes": sum(path.stat().st_size for path in optimizer),
        "model_state_bytes": model.stat().st_size,
    }


def _logical_payload_inventory(package: LogicalCheckpointPackage) -> dict[str, Any]:
    return {
        component: {
            name: {
                "bytes": int(payload["bytes"]),
                "sha256": str(payload["sha256"]),
            }
            for name, payload in package.component(component)["payloads"].items()
        }
        for component in ("generator", "critic")
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-bundle", type=Path, required=True)
    parser.add_argument("--expected-trace-bundle-hash", required=True)
    parser.add_argument("--logical-package", type=Path, required=True)
    parser.add_argument("--expected-logical-manifest-hash", required=True)
    parser.add_argument("--native-world8-root", type=Path, required=True)
    parser.add_argument("--generator-root", type=Path, required=True)
    parser.add_argument("--precast-generator-root", type=Path, required=True)
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--precast-score-root", type=Path, required=True)
    parser.add_argument("--negative-prompt-asset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    for path in (
        args.trace_bundle,
        args.logical_package,
        args.native_world8_root,
        args.generator_root,
        args.precast_generator_root,
        args.score_root,
        args.precast_score_root,
        args.negative_prompt_asset,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    bundle = load_operation_trace_bundle_manifest(args.trace_bundle, verify=True)
    if bundle is None or bundle["bundle_hash"] != args.expected_trace_bundle_hash:
        raise ValueError("execution trace bundle identity changed")
    package = LogicalCheckpointPackage.load(args.logical_package)
    if package.manifest_hash != args.expected_logical_manifest_hash:
        raise ValueError("logical G2/C10/EMA10 manifest identity changed")
    trace = RouteBFull5Trace(
        args.trace_bundle, RouteBFull5Contract.candidate_a()
    )
    if trace.loaded_trace_count:
        raise AssertionError("manifest preflight eagerly materialized tensor payloads")

    import_record = json.loads(
        (args.native_world8_root / "distillschedule_logical_import.json").read_text()
    )
    if import_record.get("status") != "PASS" or int(import_record.get("global_step", -1)) != 10:
        raise ValueError("native world8 logical import did not pass at step 10")
    imported_package = Path(str(import_record["package"]))
    imported = LogicalCheckpointPackage.load(imported_package)
    requested_payloads = _logical_payload_inventory(package)
    imported_payloads = _logical_payload_inventory(imported)
    if requested_payloads != imported_payloads:
        raise ValueError("native checkpoint was retargeted from different logical tensors")
    requested_ema = {
        item["path"]: item["sha256"] for item in package.manifest["ema"]["files"]
    }
    imported_ema = {
        item["path"]: item["sha256"] for item in imported.manifest["ema"]["files"]
    }
    if requested_ema != imported_ema:
        raise ValueError("native checkpoint and requested package use different EMA10")

    result = {
        "schema_version": "distillschedule.routeb_full5_input_gate.v1",
        "status": "PASS",
        "candidate": RouteBFull5Contract.candidate_a().__dict__,
        "logical_manifest_hash": package.manifest_hash,
        "state_provenance_trace_hash": package.manifest["input_trace_hash"],
        "execution_trace_bundle_hash": trace.bundle_hash,
        "routeb_trace_binding_hash": trace.binding_hash,
        "trace_rank_count": 16,
        "trace_entry_count": len(trace.entries),
        "trace_payloads_loaded_during_preflight": trace.loaded_trace_count,
        "rank_control_validation": trace.rank_control_validation,
        "native_import": {
            "record": str(
                (args.native_world8_root / "distillschedule_logical_import.json").resolve()
            ),
            "source_package": str(imported_package.resolve()),
            "source_manifest_hash": imported.manifest_hash,
            "logical_payloads_exact": True,
            "ema_payloads_exact": True,
        },
        "native_inventory": [
            _native_inventory(args.native_world8_root, "generator"),
            _native_inventory(args.native_world8_root, "critic"),
        ],
        "asset_paths": {
            "generator": str(args.generator_root.resolve()),
            "precast_generator": str(args.precast_generator_root.resolve()),
            "score": str(args.score_root.resolve()),
            "precast_score": str(args.precast_score_root.resolve()),
            "negative_prompt": str(args.negative_prompt_asset.resolve()),
        },
    }
    _atomic_json(args.output, result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
