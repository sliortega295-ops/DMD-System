#!/usr/bin/env python3
"""Summarize one typed Route-A world16 run without upgrading partial evidence to PASS."""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

EXPECTED_STEPS = (10, 11, 12, 13, 14)
EXPECTED_RANKS = tuple(range(16))
GIB = 1024**3


class EvidenceError(RuntimeError):
    """Raised when an evidence file is present but structurally invalid."""


@dataclass(frozen=True)
class JsonlRead:
    events: tuple[Mapping[str, Any], ...]
    ignored_partial_tail: bool


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl_tail_safe(path: Path) -> JsonlRead:
    """Read JSONL and ignore only one unterminated malformed final record."""

    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    events: list[Mapping[str, Any]] = []
    ignored_partial_tail = False
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError as exc:
            is_last_line = index == len(lines) - 1
            is_unterminated = not line.endswith(("\n", "\r"))
            if is_last_line and is_unterminated:
                ignored_partial_tail = True
                break
            raise EvidenceError(
                f"{path}: malformed JSONL record at line {index + 1}: {exc.msg}"
            ) from exc
        if not isinstance(value, Mapping):
            raise EvidenceError(f"{path}: line {index + 1} is not a JSON object")
        events.append(value)
    return JsonlRead(tuple(events), ignored_partial_tail)


def parse_key_value_manifest(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
    ):
        if not line.strip():
            continue
        if "=" not in line:
            raise EvidenceError(f"{path}: line {line_number} is not key=value")
        key, value = line.split("=", 1)
        if not key:
            raise EvidenceError(f"{path}: line {line_number} has an empty key")
        result[key] = value
    return result


def parse_memory_events(text: str) -> dict[str, int]:
    tokens = text.replace(";", " ").split()
    result: dict[str, int] = {}
    for index in range(0, len(tokens) - 1, 2):
        try:
            result[tokens[index]] = int(tokens[index + 1])
        except ValueError:
            continue
    return result


def parse_resource_csv(path: Path) -> dict[str, object]:
    rows: list[tuple[float, int, int]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"unix_time", "cgroup_bytes", "host_used_bytes"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise EvidenceError(f"{path}: resource CSV is missing required columns")
        for line_number, row in enumerate(reader, start=2):
            try:
                rows.append(
                    (
                        float(row["unix_time"]),
                        int(row["cgroup_bytes"]),
                        int(row["host_used_bytes"]),
                    )
                )
            except (TypeError, ValueError) as exc:
                raise EvidenceError(
                    f"{path}: malformed numeric value at line {line_number}"
                ) from exc
    return {
        "samples": len(rows),
        "span_seconds": rows[-1][0] - rows[0][0] if rows else None,
        "cgroup_peak_bytes": max((row[1] for row in rows), default=None),
        "host_used_peak_bytes": max((row[2] for row in rows), default=None),
    }


def _int_or_none(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def node_summary(run_dir: Path, node_rank: int) -> tuple[dict[str, object], list[str]]:
    launcher_dir = run_dir / "launcher" / f"node_{node_rank}"
    manifest_path = launcher_dir / "launch_manifest.txt"
    resource_path = launcher_dir / "resource.csv"
    console_path = launcher_dir / "console.log"
    errors: list[str] = []
    if not manifest_path.is_file():
        return {
            "node_rank": node_rank,
            "status": "MISSING",
            "final_return_code": None,
            "oom_delta": None,
            "oom_kill_delta": None,
        }, [f"missing {manifest_path}"]
    try:
        manifest = parse_key_value_manifest(manifest_path)
    except EvidenceError as exc:
        return {
            "node_rank": node_rank,
            "status": "INVALID",
            "final_return_code": None,
            "oom_delta": None,
            "oom_kill_delta": None,
        }, [str(exc)]
    before = parse_memory_events(manifest.get("memory_events_before", ""))
    after = parse_memory_events(manifest.get("memory_events_after", ""))
    final_return_code = _int_or_none(manifest.get("final_return_code"))
    if final_return_code is None:
        errors.append(f"{manifest_path}: final_return_code is missing or invalid")
    try:
        resource = parse_resource_csv(resource_path) if resource_path.is_file() else None
        if resource is None:
            errors.append(f"missing {resource_path}")
    except EvidenceError as exc:
        resource = None
        errors.append(str(exc))
    identity_keys = (
        "dmd_system_head",
        "dmd_system_plan_sha256",
        "resolved_plan_digest",
        "helios_head",
        "helios_entrypoint_sha256",
        "python_path",
        "python_sha256",
        "accelerate_path",
        "accelerate_sha256",
        "deepspeed_path",
        "deepspeed_sha256",
        "accelerate_config_sha256",
        "deepspeed_config_sha256",
    )
    identities = {key: manifest.get(key) for key in identity_keys}
    missing_identities = [key for key, value in identities.items() if not value]
    if missing_identities:
        errors.append(
            f"{manifest_path}: missing identity fields {', '.join(missing_identities)}"
        )
    summary: dict[str, object] = {
        "node_rank": node_rank,
        "status": "COMPLETE" if final_return_code is not None else "PARTIAL",
        "host": manifest.get("host"),
        "final_return_code": final_return_code,
        "oom_delta": after.get("oom", 0) - before.get("oom", 0),
        "oom_kill_delta": after.get("oom_kill", 0) - before.get("oom_kill", 0),
        "resource": resource,
        "identities": identities,
        "manifest_sha256": sha256_file(manifest_path),
        "console_sha256": sha256_file(console_path) if console_path.is_file() else None,
    }
    if not console_path.is_file():
        errors.append(f"missing {console_path}")
    return summary, errors


def rank_summary(path: Path) -> tuple[dict[str, object], list[str]]:
    errors: list[str] = []
    try:
        rank = int(path.stem.split("_")[-1])
    except ValueError as exc:
        raise EvidenceError(f"cannot parse rank from {path.name}") from exc
    read = read_jsonl_tail_safe(path)
    events = read.events
    counts = collections.Counter(str(event.get("phase")) for event in events)
    starts = {
        int(event["global_step"]): float(event["time_unix"])
        for event in events
        if event.get("phase") == "training_step_start"
        and "global_step" in event
        and "time_unix" in event
    }
    ends = {
        int(event["global_step"]): float(event["time_unix"])
        for event in events
        if event.get("phase") == "training_step_end"
        and "global_step" in event
        and "time_unix" in event
    }
    validated = [
        event for event in events if event.get("phase") == "operation_trace_input_validated"
    ]
    versions = [
        event for event in events if event.get("phase") == "parameter_versions_initialized"
    ]
    activation = [
        event for event in events if event.get("phase") == "activation_offload_final"
    ]
    streamers = [
        event for event in events if event.get("phase") == "frozen_block_streamer_final"
    ]
    initial_versions = [
        [
            int(event.get("generator_version", -1)),
            int(event.get("critic_version", -1)),
        ]
        for event in versions
    ]
    summary: dict[str, object] = {
        "rank": rank,
        "counts": dict(counts),
        "steps": sorted(ends),
        "step_start_unix": starts,
        "step_end_unix": ends,
        "trace_hashes": sorted(str(event.get("trace_hash")) for event in validated),
        "trace_authorities": sorted(
            str(event.get("source_authority")) for event in validated
        ),
        "initial_versions": initial_versions,
        "derived_final_versions": {"generator": 3, "critic": 15, "ema": 15},
        "training_complete": counts["training_complete_skip_final_save"] == 1,
        "activation_live_final": [
            int(event.get("stats", {}).get("live_cpu_bytes", -1))
            for event in activation
        ],
        "streamer_resident_final": [
            int(event.get("stats", {}).get("resident_bytes", -1))
            for event in streamers
        ],
        "streamer_materialized_bytes": sum(
            int(event.get("stats", {}).get("materialized_bytes", 0))
            for event in streamers
        ),
        "streamer_materialize_count": sum(
            int(event.get("stats", {}).get("materialize_count", 0))
            for event in streamers
        ),
        "cuda_peak_allocated": max(
            (int(event.get("cuda", {}).get("max_allocated_bytes", 0)) for event in events),
            default=0,
        ),
        "cuda_peak_reserved": max(
            (int(event.get("cuda", {}).get("max_reserved_bytes", 0)) for event in events),
            default=0,
        ),
        "ignored_partial_tail": read.ignored_partial_tail,
        "sha256": sha256_file(path),
    }
    exact = (
        summary["steps"] == list(EXPECTED_STEPS)
        and summary["training_complete"] is True
        and counts["generator_optimizer_step"] == 1
        and counts["critic_optimizer_step"] == 5
        and counts["ema_step"] == 5
        and initial_versions == [[2, 10]]
        and len(validated) == 1
        and len(versions) == 1
        and len(activation) == 1
        and len(streamers) == 2
        and summary["activation_live_final"] == [0]
        and summary["streamer_resident_final"] == [0, 0]
    )
    summary["contract_exact"] = exact
    if not exact:
        errors.append(f"{path}: rank contract did not match Full5 G2/C10 -> G3/C15/EMA15")
    return summary, errors


def checkpoint_inventory(path: Path) -> dict[str, object]:
    if not path.is_dir():
        return {"exists": False, "file_count": 0, "total_bytes": 0, "top_level": []}
    files = [candidate for candidate in path.rglob("*") if candidate.is_file()]
    return {
        "exists": True,
        "file_count": len(files),
        "total_bytes": sum(candidate.stat().st_size for candidate in files),
        "top_level": sorted(candidate.name for candidate in path.iterdir()),
    }


def load_topology_probe(path: Path) -> tuple[dict[str, object] | None, list[str]]:
    if not path.is_file():
        return None, [f"missing {path}"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, [f"{path}: invalid topology probe: {exc}"]
    if not isinstance(payload, Mapping) or payload.get("status") != "PASS":
        return dict(payload) if isinstance(payload, Mapping) else None, [
            f"{path}: topology probe did not PASS"
        ]
    summary = payload.get("summary")
    if not isinstance(summary, Mapping):
        return dict(payload), [f"{path}: topology summary is missing"]
    if summary.get("world_size") != 16 or summary.get("nodes") != 2:
        return dict(payload), [f"{path}: topology dimensions do not match world16"]
    return dict(payload), []


def load_baseline_window(path: Path) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise EvidenceError(f"{path}: baseline result must be a JSON object")
    for container_name in ("timing_summary", "timing"):
        container = payload.get(container_name)
        if isinstance(container, Mapping):
            value = container.get("global_training_window_seconds")
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
                return float(value)
    raise EvidenceError(f"{path}: no positive global_training_window_seconds")


def timing_summary(
    ranks: Iterable[Mapping[str, Any]],
    *,
    baseline_result: Path | None,
    relative_band: float,
) -> tuple[dict[str, object], list[str]]:
    starts: dict[int, list[float]] = {step: [] for step in EXPECTED_STEPS}
    ends: dict[int, list[float]] = {step: [] for step in EXPECTED_STEPS}
    for rank in ranks:
        rank_starts = rank.get("step_start_unix", {})
        rank_ends = rank.get("step_end_unix", {})
        if isinstance(rank_starts, Mapping) and isinstance(rank_ends, Mapping):
            for step in EXPECTED_STEPS:
                if step in rank_starts:
                    starts[step].append(float(rank_starts[step]))
                if step in rank_ends:
                    ends[step].append(float(rank_ends[step]))
    if not all(len(starts[step]) == 16 and len(ends[step]) == 16 for step in EXPECTED_STEPS):
        return {
            "status": "INCOMPLETE",
            "global_training_window_seconds": None,
            "relative_delta_vs_baseline": None,
        }, ["timing evidence does not cover all 16 ranks and five steps"]
    step_wall = {
        str(step): max(ends[step]) - min(starts[step]) for step in EXPECTED_STEPS
    }
    window = max(max(values) for values in ends.values()) - min(
        min(values) for values in starts.values()
    )
    result: dict[str, object] = {
        "status": "NOT_EVALUATED",
        "global_step_wall_span_seconds": step_wall,
        "global_training_window_seconds": window,
        "official_weighted_step_seconds": window / len(EXPECTED_STEPS),
        "samples_per_second": 80 / window if window > 0 else None,
        "baseline_result": str(baseline_result) if baseline_result else None,
        "baseline_window_seconds": None,
        "relative_delta_vs_baseline": None,
        "relative_band": relative_band,
    }
    if baseline_result is None:
        return result, []
    try:
        baseline = load_baseline_window(baseline_result)
    except (OSError, json.JSONDecodeError, EvidenceError) as exc:
        result["status"] = "INVALID_BASELINE"
        return result, [str(exc)]
    delta = window / baseline - 1
    result["baseline_window_seconds"] = baseline
    result["relative_delta_vs_baseline"] = delta
    result["status"] = "PASS" if abs(delta) <= relative_band else "OUTSIDE_BAND"
    return result, []


def reserve_admission(
    nodes: Iterable[Mapping[str, Any]], *, node_memory_gib: float, reserve_gib: float
) -> dict[str, object]:
    limit = int((node_memory_gib - reserve_gib) * GIB)
    per_node: list[dict[str, object]] = []
    evaluated = True
    for node in nodes:
        resource = node.get("resource")
        peak = resource.get("host_used_peak_bytes") if isinstance(resource, Mapping) else None
        if not isinstance(peak, int):
            evaluated = False
            per_node.append(
                {"node_rank": node.get("node_rank"), "host_used_peak_bytes": None, "admitted": None}
            )
        else:
            per_node.append(
                {
                    "node_rank": node.get("node_rank"),
                    "host_used_peak_bytes": peak,
                    "admitted": peak <= limit,
                }
            )
    status = "NOT_EVALUATED"
    if evaluated:
        status = "PASS" if all(item["admitted"] for item in per_node) else "FAIL"
    return {
        "status": status,
        "node_memory_gib": node_memory_gib,
        "reserve_gib": reserve_gib,
        "admission_limit_bytes": limit,
        "per_node": per_node,
        "note": "Reserve admission is reported independently from functional success.",
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def summarize(
    *,
    run_id: str,
    root: Path,
    run_dir: Path | None = None,
    output: Path | None = None,
    baseline_result: Path | None = None,
    relative_band: float = 0.10,
    node_memory_gib: float = 760,
    reserve_gib: float = 64,
) -> dict[str, object]:
    run = run_dir or root / "artifacts" / "runs" / run_id
    evidence_errors: list[str] = []
    nodes: list[dict[str, object]] = []
    for node_rank in (0, 1):
        node, errors = node_summary(run, node_rank)
        nodes.append(node)
        evidence_errors.extend(errors)
    topology, topology_errors = load_topology_probe(run / "topology_probe.json")
    evidence_errors.extend(topology_errors)
    if len(nodes) == 2:
        left = nodes[0].get("identities")
        right = nodes[1].get("identities")
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            shared_identity_keys = (
                "dmd_system_head",
                "dmd_system_plan_sha256",
                "resolved_plan_digest",
                "helios_head",
                "helios_entrypoint_sha256",
                "python_sha256",
                "accelerate_sha256",
                "deepspeed_sha256",
                "deepspeed_config_sha256",
            )
            mismatched = [
                key for key in shared_identity_keys if left.get(key) != right.get(key)
            ]
            if mismatched:
                evidence_errors.append(
                    "node identity mismatch: " + ", ".join(mismatched)
                )

    rank_files = sorted(
        (run / "phase_trace").glob("rank_*.jsonl"),
        key=lambda candidate: int(candidate.stem.split("_")[-1]),
    ) if (run / "phase_trace").is_dir() else []
    ranks: list[dict[str, object]] = []
    for path in rank_files:
        try:
            rank, errors = rank_summary(path)
            ranks.append(rank)
            evidence_errors.extend(errors)
        except (EvidenceError, OSError, ValueError) as exc:
            evidence_errors.append(str(exc))
    observed_ranks = sorted(
        int(rank["rank"]) for rank in ranks if isinstance(rank.get("rank"), int)
    )
    if observed_ranks != list(EXPECTED_RANKS):
        evidence_errors.append(
            f"phase trace rank set mismatch; observed={observed_ranks}, expected={list(EXPECTED_RANKS)}"
        )

    checkpoint = checkpoint_inventory(run / "checkpoint-15")
    checkpoint_ok = bool(checkpoint["exists"] and checkpoint["file_count"] > 0)
    if not checkpoint_ok:
        evidence_errors.append("checkpoint-15 inventory is missing or empty")

    node_complete = all(node.get("status") == "COMPLETE" for node in nodes)
    any_oom = any(
        (node.get("oom_delta") or 0) > 0 or (node.get("oom_kill_delta") or 0) > 0
        for node in nodes
    )
    any_nonzero_return = any(
        isinstance(node.get("final_return_code"), int)
        and node.get("final_return_code") != 0
        for node in nodes
    )
    rank_contract_exact = (
        observed_ranks == list(EXPECTED_RANKS)
        and len(ranks) == 16
        and all(rank.get("contract_exact") is True for rank in ranks)
    )
    topology_pass = isinstance(topology, Mapping) and topology.get("status") == "PASS"

    if any_oom:
        functional_status = "OOM"
    elif not node_complete or len(ranks) < 16:
        functional_status = "PARTIAL"
    elif any_nonzero_return:
        functional_status = "CRASH"
    elif not topology_pass or not rank_contract_exact or not checkpoint_ok:
        functional_status = "CONTRACT_FAIL"
    elif evidence_errors:
        functional_status = "EVIDENCE_INVALID"
    else:
        functional_status = "PASS"

    timing, timing_errors = timing_summary(
        ranks,
        baseline_result=baseline_result,
        relative_band=relative_band,
    )
    evidence_errors.extend(timing_errors)
    if functional_status != "PASS":
        overall_status = functional_status
        timing["claimable"] = False
    elif timing["status"] == "PASS":
        overall_status = "PASS"
        timing["claimable"] = True
    elif timing["status"] == "OUTSIDE_BAND":
        overall_status = "FUNCTIONAL_PASS_TIMING_OUTSIDE_BAND"
        timing["claimable"] = False
    else:
        overall_status = "FUNCTIONAL_PASS_TIMING_NOT_EVALUATED"
        timing["claimable"] = False

    payload: dict[str, object] = {
        "schema_version": "distillschedule.typed_routea_world16.revalidation.v2",
        "run_id": run_id,
        "status": overall_status,
        "functional_status": functional_status,
        "claim_scope": (
            "One-run diagnostic consistency validation of the typed DMD-System wrapper "
            "against the frozen Route-A world16 baseline; never a formal speedup."
        ),
        "contract": {
            "world_size": 16,
            "nodes": 2,
            "local_world_size": 8,
            "resolution": [384, 640],
            "latent": [1, 16, 9, 48, 80],
            "steps": list(EXPECTED_STEPS),
            "initial_versions": {"generator": 2, "critic": 10, "ema": 10},
            "final_versions": {"generator": 3, "critic": 15, "ema": 15},
            "zero_stage": 1,
            "optimizer_offload": "cpu",
        },
        "nodes": nodes,
        "topology_probe": topology,
        "rank_count": len(ranks),
        "observed_ranks": observed_ranks,
        "ranks": ranks,
        "checkpoint_15": checkpoint,
        "timing": timing,
        "reserve_admission": reserve_admission(
            nodes, node_memory_gib=node_memory_gib, reserve_gib=reserve_gib
        ),
        "gpu": {
            "max_allocated_bytes": max(
                (int(rank["cuda_peak_allocated"]) for rank in ranks), default=0
            ),
            "max_reserved_bytes": max(
                (int(rank["cuda_peak_reserved"]) for rank in ranks), default=0
            ),
        },
        "lifecycle": {
            "max_materialized_bytes_per_rank": max(
                (int(rank["streamer_materialized_bytes"]) for rank in ranks), default=0
            ),
            "max_materialize_count_per_rank": max(
                (int(rank["streamer_materialize_count"]) for rank in ranks), default=0
            ),
        },
        "evidence_errors": sorted(set(evidence_errors)),
        "limitations": [
            "Single revalidation launch; not a repeated formal timing distribution.",
            "Cold construction is excluded from the five-step comparison.",
            "The typed executor still invokes pinned Helios model math through a compatibility adapter.",
            "Functional success and 64-GiB host-reserve admission are independent outcomes.",
            "Partial or malformed evidence is never promoted to a performance result.",
        ],
    }
    destination = output or root / "artifacts" / "manifests" / f"{run_id}_result.json"
    _atomic_json(destination, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--baseline-result",
        type=Path,
        default=(
            Path(os.environ["DISTILLSCHEDULE_BASELINE_RESULT"])
            if os.environ.get("DISTILLSCHEDULE_BASELINE_RESULT")
            else None
        ),
    )
    parser.add_argument("--relative-band", type=float, default=0.10)
    parser.add_argument("--node-memory-gib", type=float, default=760)
    parser.add_argument("--reserve-gib", type=float, default=64)
    args = parser.parse_args()
    if not 0 <= args.relative_band < 1:
        parser.error("--relative-band must be in [0, 1)")
    if args.reserve_gib < 0 or args.node_memory_gib <= args.reserve_gib:
        parser.error("node memory must be positive and larger than reserve")
    payload = summarize(
        run_id=args.run_id,
        root=args.root.resolve(),
        run_dir=args.run_dir.resolve() if args.run_dir else None,
        output=args.output.resolve() if args.output else None,
        baseline_result=args.baseline_result.resolve() if args.baseline_result else None,
        relative_band=args.relative_band,
        node_memory_gib=args.node_memory_gib,
        reserve_gib=args.reserve_gib,
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "functional_status": payload["functional_status"],
                "rank_count": payload["rank_count"],
                "timing_status": payload["timing"]["status"],
            },
            sort_keys=True,
        )
    )
    if payload["status"] == "PASS":
        return 0
    if (
        payload["status"] == "FUNCTIONAL_PASS_TIMING_NOT_EVALUATED"
        and payload["timing"]["status"] == "NOT_EVALUATED"
    ):
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
