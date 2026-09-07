"""CPU-only native adapters, executed beside remote assets and receipts.

Only registered actions are admitted. This is not an arbitrary command/RPC
server: it runs over the existing SSH stdin transport and exits after one
request. Large traces/anchors/weights never traverse the controller link.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from distillschedule.workloads.best_known.registry import (
    file_sha256, fingerprint, git_head, repository_root, write_new_json, load_assets, validate_asset_config,
)
from .contracts import require
from .execution import resolve_candidate, validated_profile
from .registry import WAN, LIVEWAN, candidate_profile, get_candidate


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"not a JSON object: {path}")
    return value


def binding(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": file_sha256(path)}


def validate_native_comparison(value: dict[str, Any]) -> None:
    """Consume the native comparator schema, not synthetic PASS/REVIEW labels.

    Historical status names contain ``50C`` even for a two-cycle comparison.
    The typed inventory, not that label, determines the actual cycle count.
    Neither accepted coarse result establishes strict numerical equivalence.
    """
    for key, expected in {
        "schema_version": "distillschedule.trajectory_coarse_comparison.v1",
        "comparison_kind": "semantic", "expected_cycles": 2,
        "expected_world_size": 16, "complete": True,
    }.items():
        require(value.get(key) == expected, f"native comparison {key} mismatch")
    require(value.get("contract", {}).get("status") == "PASS"
            and not value["contract"].get("mismatches"), "trajectory semantic comparison failed")
    require(value.get("status") in {
        "TRAJECTORY_MATCHED_50C_SINGLE_SEED", "TRAJECTORY_REVIEW_50C_COMPLETED",
    }, "native numerical gate failed")


def checked_summary(receipt: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    validated_profile(receipt, root=repository_root())
    out = Path(receipt["asset_config"]["run_root"]) / receipt["run_id"]
    summary = read(out / "bound-summary.json")
    expected = {k: receipt[k] for k in ("integration_head", "profile_sha256", "asset_fingerprint")}
    require(all(summary.get("integration_binding", {}).get(k) == v for k, v in expected.items()),
            "native summary does not bind this execution")
    require(summary.get("status") == "PASS" and not summary.get("errors"), "native summary failed")
    if receipt["profile_id"] == LIVEWAN and receipt["native_mode"] == "trajectory":
        # This native writer counts rank-cycle rows, not ranks. Do not invent
        # rank_count or accept 32 rows without checking the sixteen owners.
        required = {"schema_version": "distillschedule.livewan_trajectory_summary.v1",
                    "cycles": 2, "rank_cycle_receipts": 32, "expected_rank_cycle_receipts": 32,
                    "final_versions": {"generator": 2, "critic": 10, "ema": 0}}
        require(all(summary.get(k) == v for k, v in required.items()),
                "LiveWan native trajectory summary inventory incomplete")
        require(summary.get("rank_count", 16) == 16, "contradictory native rank coverage")
        for rank in range(16):
            path = out / "trajectory" / f"rank_{rank:03d}.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            require(len(rows) == 2 and all(
                row.get("rank") == rank and row.get("cycle") == cycle
                and row.get("world_size") == 16 and row.get("expected_total_cycles") == 2
                for cycle, row in enumerate(rows, 1)),
                f"LiveWan native rank{rank} two-cycle receipt coverage incomplete")
    else:
        require(summary.get("rank_count") == 16, "native summary rank coverage incomplete")
    return out, summary


def _command(node: dict[str, Any], args: list[str], log: Path) -> None:
    from distillschedule.workloads.best_known.dispatch import clean_environment
    # The validated packed environment needs its own libstdc++ for sqlite/ICU.
    # Process-local only; no shared environment mutation.
    env = clean_environment(node)
    env["LD_LIBRARY_PATH"] = str(Path(node["python"]).parent.parent / "lib")
    with log.open("x") as stream:
        subprocess.run([node["python"], *args], cwd=node["repo_root"], env=env,
                       stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=600)


def prepare_pair(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    if reference["profile_id"] == "minimax_h3_grid1000_best":
        from .h3_native_bridge import prepare_pair as h3_pair
        return h3_pair(reference, candidate)
    from .wan_implementation import PREFIX, WanImplementation
    ref_id = reference["search_candidate"]["id"]
    new_ref = reference["profile_id"] == WAN and ref_id.startswith(PREFIX)
    require(ref_id == "fsdp-reference" or ref_id == "joint-livewan1-fsdp-m0-g0-r0"
            or (new_ref and WanImplementation.parse(ref_id).placement == "fsdp"),
            "unregistered baseline")
    require(new_ref == candidate["search_candidate"]["id"].startswith(PREFIX),
            "generated pair cannot mix precision contracts with legacy measurements")
    require(reference["mode"] == candidate["mode"] == "correctness", "pair must use correctness runs")
    for key in ("profile_id", "integration_head", "asset_fingerprint", "workload_semantics_hash"):
        require(reference[key] == candidate[key], f"pair {key} mismatch")
    left, _ = checked_summary(reference)
    right, _ = checked_summary(candidate)
    out = right / "search-gate"
    out.mkdir(exist_ok=False)
    node = candidate["nodes"][0]
    root = Path(node["repo_root"]) / "scripts/distillschedule"
    workload = candidate["profile_id"]
    comparison = out / "trajectory_comparison.json"
    if workload == WAN:
        _command(node, [str(root / "compare_wan13b_trajectory_world16.py"),
            "--reference-run-root", str(left), "--candidate-run-root", str(right),
            "--cycles", "2", "--no-plot", "--output-dir", str(out / "comparison")], out / "compare.log")
        source = out / "comparison/trajectory_comparison.json"
        # Workload-owned writer names are explicit, never a glob fallback.
        require(source.is_file(), f"native comparison output missing: {source}")
        comparison = source
        raw_gate = None
    else:
        require(workload == LIVEWAN, "CPU-only workload")
        variant = candidate["fully_resolved_profile"]["contract"]["variant"]
        _command(node, [str(root / "compare_livewan_two_cycle_trajectories.py"),
            "--baseline-run-root", str(left), "--pipeline-run-root", str(right),
            "--baseline-summary", str(left / "bound-summary.json"),
            "--pipeline-summary", str(right / "bound-summary.json"),
            "--expected-head", candidate["integration_head"], "--pipeline-variant", variant,
            "--output", str(comparison)], out / "compare.log")
        gate_path = out / "native-gate.json"
        _command(node, [str(root / "build_livewan_two_cycle_timing_gate.py"),
            "--baseline-trajectory-summary", str(left / "bound-summary.json"),
            "--pipeline-trajectory-summary", str(right / "bound-summary.json"),
            "--trajectory-comparison", str(comparison),
            "--expected-head", candidate["integration_head"], "--pipeline-variant", variant,
            "--output", str(gate_path)], out / "build-gate.log")
        raw_gate = read(gate_path)
        require(raw_gate.get("status") == "PASS", "native matched gate failed")
    cv = read(comparison)
    # Coarse trajectory PASS is not DEFAULT_P0_PARITY / strict equivalence.
    # Keep the raw thresholds and metrics. No relaxation or relabeling.
    validate_native_comparison(cv)
    ready = {}
    for receipt, run in ((reference, left), (candidate, right)):
        cid = receipt["search_candidate"]["id"]
        gate = copy.deepcopy(raw_gate) if raw_gate is not None else read(run / "bound-summary.json")
        gate["integration_binding"] = {k: receipt[k] for k in ("integration_head", "profile_sha256", "asset_fingerprint")}
        gate["integration_binding"]["profile_id"] = workload
        path = out / f"{cid}-bound-gate.json"
        write_new_json(path, gate)
        assets = copy.deepcopy(receipt["asset_config"])
        assets["correctness"] = binding(path)
        if workload == WAN:
            assets["trajectory_comparison"] = binding(comparison)
        ready[cid] = {"source_head": receipt["integration_head"], "candidate_id": cid,
            "semantics_status": "PASS", "numerical_status": cv["status"],
            "strict_numerical_gate": "NOT_ESTABLISHED_BY_COARSE_TRAJECTORY",
            "comparison": binding(comparison), "correctness": binding(run / "bound-summary.json"),
            "assets": assets}
        write_new_json(out / f"{cid}-preparation.json", ready[cid])
    return ready


def handle_request(payload: dict[str, Any]) -> dict[str, Any]:
    root = repository_root()
    require(git_head(root) == payload["source_head"], "remote search source changed")
    dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True)
    require(not dirty.strip(), "remote search checkout is dirty")
    action = payload.get("action")
    if action == "runtime_identity":
        from .runtime_identity import collect_planned_node_identity
        return collect_planned_node_identity(payload["receipt"], node_rank=payload["node_rank"])
    if action == "resolve":
        candidate = get_candidate(payload["workload"], payload["candidate_id"])
        require(candidate.executable, "candidate native two-cycle adapter unavailable")
        assets = validate_asset_config(payload["assets"], candidate_profile(candidate))
        output = Path(assets["run_root"]) / "search-inputs"
        require(output.is_absolute() and output != Path("/"), "invalid output root")
        output.mkdir(parents=True, exist_ok=True)
        # Content-addressed configs can be reused but never overwritten.
        path = output / (fingerprint(assets) + ".json")
        if path.exists():
            require(read(path) == assets, "source-bound asset config changed")
        else:
            write_new_json(path, assets)
        load_assets(path, candidate_profile(candidate))
        return resolve_candidate(candidate.workload, candidate.id, root=root, assets_path=path,
                                 run_id=payload["run_id"], mode=payload["mode"])
    if action == "prepare_pair":
        return prepare_pair(payload["reference"], payload["candidate"])
    if action == "measurement":
        if payload["receipt"]["profile_id"] == "minimax_h3_grid1000_best":
            from .h3_native_bridge import normalize_native
            return normalize_native(payload["receipt"], payload["preparation"], payload["identity"])
        from .measurement import normalize_native
        return normalize_native(payload["receipt"], payload["preparation"], payload["identity"])
    if action == "wan_stage_calibration":
        from .wan_calibration_evidence import assemble_native
        return assemble_native(payload["receipt"], payload["preparation"], payload["identity"],
                               payload["runtime_identity_evidence"])
    if action == "macro_calibration":
        from .native_macro_adapter import assemble_native
        return assemble_native(payload["receipt"], payload["preparation"], payload["identity"],
                               payload["runtime_identity_evidence"])
    raise ValueError(f"unsupported search action: {action}")
