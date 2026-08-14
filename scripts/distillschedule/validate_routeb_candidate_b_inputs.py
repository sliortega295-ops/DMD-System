#!/usr/bin/env python3
"""Low-cost immutable-input gate for RB16-F5-PIPE-2G6C."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from runtime.critic_pool_checkpoint import CandidateBTransitionPackage, LogicalCheckpointPackage
from runtime.routeb_full5 import RouteBFull5Contract
from runtime.routeb_full5_trace import RouteBFull5Trace


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-bundle", type=Path, required=True)
    parser.add_argument("--expected-trace-bundle-hash", required=True)
    parser.add_argument("--logical-package", type=Path, required=True)
    parser.add_argument("--expected-logical-manifest-hash", required=True)
    parser.add_argument("--transition-package", type=Path, required=True)
    parser.add_argument("--expected-transition-package-hash", required=True)
    parser.add_argument("--expected-source-transition-manifest-sha256", required=True)
    parser.add_argument("--generator-root", type=Path, required=True)
    parser.add_argument("--precast-generator-root", type=Path, required=True)
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--precast-score-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.generator_root, args.precast_generator_root, args.score_root, args.precast_score_root):
        path.resolve(strict=True)
    logical = LogicalCheckpointPackage.load(args.logical_package)
    if logical.manifest_hash != args.expected_logical_manifest_hash:
        raise ValueError("Candidate-B logical package hash changed")
    transition = CandidateBTransitionPackage.load(
        args.transition_package,
        expected_transition_manifest_sha256=args.expected_source_transition_manifest_sha256,
        expected_critic_logical_manifest_hash=logical.manifest_hash,
        verify_generator_payload_sha256=False,
    )
    if transition.manifest_hash != args.expected_transition_package_hash:
        raise ValueError("Candidate-B transition package hash changed")
    trace = RouteBFull5Trace(args.trace_bundle, RouteBFull5Contract.candidate_b())
    if trace.bundle_hash != args.expected_trace_bundle_hash:
        raise ValueError("Candidate-B trace bundle hash changed")
    result = {
        "schema_version": "distillschedule.routeb_candidate_b_inputs.v1",
        "status": "PASS",
        "experiment_id": "RB16-F5-PIPE-2G6C",
        "trace_bundle_hash": trace.bundle_hash,
        "trace_binding_hash": trace.binding_hash,
        "trace_entries": len(trace.entries),
        "logical_manifest_hash": logical.manifest_hash,
        "transition_package_hash": transition.manifest_hash,
        "source_transition_manifest_sha256": args.expected_source_transition_manifest_sha256,
        "model_paths": {
            "generator": str(args.generator_root.resolve()),
            "precast_generator": str(args.precast_generator_root.resolve()),
            "score": str(args.score_root.resolve()),
            "precast_score": str(args.precast_score_root.resolve()),
        },
    }
    if result["trace_entries"] != 60:
        raise AssertionError("Candidate-B trace entry count changed")
    _atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
