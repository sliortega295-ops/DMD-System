#!/usr/bin/env python3
"""Strictly summarize a typed Route-A world16 run.

Attempt 003 remains a legacy PARTIAL: its launcher manifest is not valid key=value
and it has no in-run topology/evidence contract.  This summarizer does not repair or
reinterpret that artifact.  Attempt 004 and later pass only with exact preregistered
trace, batch/backend/config and observed G/C/EMA evidence.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from distillschedule.runtime.routea_evidence import (
    EXPECTED_STEPS,
    directory_inventory,
    load_json_mapping,
    object_digest,
    sha256_file,
    validate_final_contract,
)

DEFAULT_OLD_RESULT = Path(
    "/share/project/shuchenweng/yongyan_liu/DistillSchedule/artifacts/manifests/"
    "task7_full5_routea_exact_b0_gs_fullstream_wave8_gate_result_001.json"
)
EXPECTED_RANKS = tuple(range(16))
EXPECTED_INITIAL = {"generator": 2, "critic": 10, "ema": 10}
EXPECTED_TRAJECTORY = {
    10: {"generator": 3, "critic": 11, "ema": 11},
    11: {"generator": 3, "critic": 12, "ema": 12},
    12: {"generator": 3, "critic": 13, "ema": 13},
    13: {"generator": 3, "critic": 14, "ema": 14},
    14: {"generator": 3, "critic": 15, "ema": 15},
}
CUDA_OOM = re.compile(
    r"CUDA out of memory|torch\.OutOfMemoryError|CUDACachingAllocator.*out of memory",
    re.IGNORECASE,
)


class EvidenceError(RuntimeError):
    pass


@dataclass(frozen=True)
class JsonlReadResult:
    """Tail-safe JSONL result with tuple-compatible unpacking.

    The named fields keep evidence handling readable for callers, while
    ``__iter__`` preserves the earlier ``events, truncated = ...`` API used by
    the strict rank summarizer.
    """

    events: list[dict[str, Any]]
    ignored_partial_tail: bool

    def __iter__(self):
        yield self.events
        yield self.ignored_partial_tail


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def parse_strict_kv(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(errors="replace").splitlines(), 1):
        if not raw.strip():
            continue
        if "=" not in raw:
            raise EvidenceError(
                f"{path}: line {line_number} is not key=value; legacy/raw CSV manifests are non-claimable"
            )
        key, value = raw.split("=", 1)
        if not key or key.strip() != key or any(char.isspace() for char in key):
            raise EvidenceError(f"{path}: invalid manifest key on line {line_number}")
        if key in result:
            raise EvidenceError(f"{path}: duplicate manifest key {key!r}")
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


def integer(value: str | None, *, default: int | None = None) -> int | None:
    if value is None or value in {"", "unavailable", "max"}:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def read_jsonl_tail_safe(path: Path) -> JsonlReadResult:
    data = path.read_bytes()
    lines = data.splitlines(keepends=True)
    events: list[dict[str, Any]] = []
    truncated_tail = False
    for index, raw in enumerate(lines):
        if not raw.strip():
            continue
        complete_line = raw.endswith((b"\n", b"\r"))
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            if index == len(lines) - 1 and not complete_line:
                truncated_tail = True
                continue
            raise EvidenceError(f"{path}: malformed JSONL complete record: {exc}") from exc
        if not isinstance(value, dict):
            raise EvidenceError(f"{path}: JSONL record must be an object")
        events.append(value)
    return JsonlReadResult(events=events, ignored_partial_tail=truncated_tail)


def resource_summary(path: Path) -> dict[str, Any]:
    rows: list[tuple[float, int, int]] = []
    if not path.is_file():
        return {"status": "MISSING", "samples": 0}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"unix_time", "cgroup_bytes", "host_used_bytes"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise EvidenceError(f"{path}: resource CSV header is invalid")
        for row in reader:
            try:
                rows.append(
                    (
                        float(row["unix_time"]),
                        int(row["cgroup_bytes"]),
                        int(row["host_used_bytes"]),
                    )
                )
            except (TypeError, ValueError):
                continue
    return {
        "status": "MEASURED" if rows else "EMPTY",
        "samples": len(rows),
        "span_seconds": rows[-1][0] - rows[0][0] if rows else None,
        "cgroup_peak_bytes": max((row[1] for row in rows), default=None),
        "host_used_peak_bytes": max((row[2] for row in rows), default=None),
    }


def gpu_inventory(path: Path, expected_sha: str, expected_rows: int) -> dict[str, Any]:
    if not path.is_file():
        raise EvidenceError(f"GPU inventory is missing: {path}")
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha:
        raise EvidenceError(f"GPU inventory SHA mismatch: {path}")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected_rows:
        raise EvidenceError(
            f"GPU inventory row count {len(rows)} != expected {expected_rows}"
        )
    required = {"index", "uuid", "name", "memory.total [MiB]", "memory.free [MiB]"}
    if not rows or not required.issubset(rows[0]):
        raise EvidenceError(f"GPU inventory columns are invalid: {path}")
    return {"path": str(path), "sha256": actual_sha, "rows": rows}


def event_value(event: Mapping[str, Any], *keys: str) -> Any:
    containers: list[Mapping[str, Any]] = [event]
    for key in ("metadata", "versions", "stats", "state"):
        value = event.get(key)
        if isinstance(value, Mapping):
            containers.append(value)
            nested = value.get("versions")
            if isinstance(nested, Mapping):
                containers.append(nested)
    for key in keys:
        for container in containers:
            if key in container:
                return container[key]
    return None


def event_step(event: Mapping[str, Any]) -> int | None:
    value = event_value(
        event,
        "global_step",
        "logical_step",
        "optimization_step",
        "optimizer_step",
        "step",
    )
    if isinstance(value, bool):
        return None
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def component_version(event: Mapping[str, Any], component: str) -> int | None:
    aliases = {
        "generator": ("generator_version", "g_version", "generator"),
        "critic": ("critic_version", "c_version", "critic"),
        "ema": ("ema_version", "ema"),
    }[component]
    value = event_value(event, *aliases)
    if isinstance(value, Mapping):
        value = value.get("version")
    if isinstance(value, bool):
        return None
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def truthy_predicate(event: Mapping[str, Any]) -> bool | None:
    value = event_value(
        event,
        "ema_update_enabled",
        "ema_predicate",
        "predicate_result",
        "updated",
        "applied",
    )
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "pass", "enabled", "applied"}:
            return True
        if normalized in {"0", "false", "fail", "disabled", "skipped"}:
            return False
    return None


def phases(events: Iterable[Mapping[str, Any]], names: Sequence[str]) -> list[Mapping[str, Any]]:
    accepted = set(names)
    return [event for event in events if event.get("phase") in accepted]


def _require_single(events: Sequence[Mapping[str, Any]], description: str) -> Mapping[str, Any]:
    if len(events) != 1:
        raise EvidenceError(f"{description}: expected exactly one event, observed {len(events)}")
    return events[0]


def validate_trace_binding(
    events: Sequence[Mapping[str, Any]], expected: Mapping[str, Any], rank: int
) -> dict[str, Any]:
    validated = phases(events, ("operation_trace_input_validated",))
    event = _require_single(validated, f"rank {rank} trace validation")
    observed_hash = event_value(event, "trace_hash")
    if observed_hash != expected.get("trace_hash"):
        raise EvidenceError(
            f"rank {rank}: trace hash {observed_hash!r} != preregistered {expected.get('trace_hash')!r}"
        )
    observed_rank = event_value(event, "trace_rank", "rank", "global_rank")
    if observed_rank is not None and int(observed_rank) != rank:
        raise EvidenceError(f"rank {rank}: trace event was qualified for rank {observed_rank}")
    authority = event_value(event, "source_authority", "trace_authority")
    aliases = [str(value) for value in expected.get("source_authority_aliases", [])]
    if authority is None or str(authority) not in aliases:
        raise EvidenceError(
            f"rank {rank}: source authority {authority!r} is not preregistered; aliases={aliases}"
        )
    observed_operation = event_value(event, "operation_fingerprint")
    if observed_operation is not None and observed_operation != expected.get(
        "operation_fingerprint"
    ):
        raise EvidenceError(f"rank {rank}: operation fingerprint mismatch")
    return {
        "trace_hash": observed_hash,
        "source_authority": str(authority),
        "mapping_fingerprint": expected.get("mapping_fingerprint"),
        "source_fingerprint": expected.get("source_fingerprint"),
        "operation_fingerprint": expected.get("operation_fingerprint"),
        "operation_fingerprint_binding": (
            "observed_and_trace_hash_bound"
            if observed_operation is not None
            else "trace_hash_commits_preregistered_trace_manifest_and_operations"
        ),
    }


def validate_version_trajectory(
    events: Sequence[Mapping[str, Any]], rank: int
) -> dict[str, Any]:
    errors: list[str] = []
    initial_events = phases(events, ("parameter_versions_initialized",))
    if len(initial_events) != 1:
        errors.append(
            f"initial version event count={len(initial_events)}, expected 1"
        )
        initial = {component: None for component in EXPECTED_INITIAL}
    else:
        initial = {
            component: component_version(initial_events[0], component)
            for component in EXPECTED_INITIAL
        }
        if initial != EXPECTED_INITIAL:
            errors.append(f"initial versions={initial}, expected={EXPECTED_INITIAL}")

    component_phases = {
        "generator": ("generator_optimizer_step",),
        "critic": ("critic_optimizer_step",),
        "ema": ("ema_step", "ema_optimizer_step", "ema_update"),
    }
    expected_steps = {
        "generator": [10],
        "critic": list(EXPECTED_STEPS),
        "ema": list(EXPECTED_STEPS),
    }
    expected_versions = {
        "generator": {10: 3},
        "critic": {step: EXPECTED_TRAJECTORY[step]["critic"] for step in EXPECTED_STEPS},
        "ema": {step: EXPECTED_TRAJECTORY[step]["ema"] for step in EXPECTED_STEPS},
    }
    observed_updates: dict[str, list[dict[str, Any]]] = {}
    for component, accepted_phases in component_phases.items():
        component_events = phases(events, accepted_phases)
        updates = [
            {
                "step": event_step(event),
                "version": component_version(event, component),
                "predicate": truthy_predicate(event) if component == "ema" else None,
                "phase": event.get("phase"),
            }
            for event in component_events
        ]
        observed_updates[component] = updates
        if len(updates) != len(expected_steps[component]):
            errors.append(
                f"{component} update count={len(updates)}, expected={len(expected_steps[component])}"
            )
        steps = [update["step"] for update in updates]
        if steps != expected_steps[component]:
            errors.append(
                f"{component} update steps={steps}, expected={expected_steps[component]}"
            )
        for update in updates:
            step = update["step"]
            if step in expected_versions[component]:
                expected_version = expected_versions[component][step]
                if update["version"] != expected_version:
                    errors.append(
                        f"{component} step {step} version={update['version']}, expected={expected_version}"
                    )
            if component == "ema" and update["predicate"] is not True:
                errors.append(
                    f"EMA step {step} lacks an observed true bootstrap/decay predicate"
                )

    final_candidates = phases(
        events,
        (
            "parameter_versions_final",
            "parameter_versions_committed",
            "training_complete_skip_final_save",
            "training_complete",
        ),
    )
    final_event: Mapping[str, Any] | None = None
    for event in reversed(final_candidates):
        values = {
            component: component_version(event, component)
            for component in EXPECTED_INITIAL
        }
        if all(value is not None for value in values.values()):
            final_event = event
            break
    if final_event is None:
        final = {component: None for component in EXPECTED_INITIAL}
        errors.append("missing final observed G/C/EMA version record")
    else:
        final = {
            component: component_version(final_event, component)
            for component in EXPECTED_INITIAL
        }
        if final != EXPECTED_TRAJECTORY[14]:
            errors.append(
                f"final observed versions={final}, expected={EXPECTED_TRAJECTORY[14]}"
            )

    # The trajectory below is derived only for diagnostics.  Exact acceptance is
    # determined by the observed component events and final record above.
    derived: dict[str, dict[str, int]] = {}
    current = dict(EXPECTED_INITIAL)
    for step in EXPECTED_STEPS:
        if step == 10:
            current["generator"] = 3
        current["critic"] += 1
        current["ema"] += 1
        derived[str(step)] = dict(current)
    if derived != {str(step): EXPECTED_TRAJECTORY[step] for step in EXPECTED_STEPS}:
        errors.append("internal derived trajectory disagrees with frozen contract")

    return {
        "status": "OBSERVED_EXACT" if not errors else "INCOMPLETE_OR_MISMATCH",
        "errors": errors,
        "initial_observed": initial,
        "updates_observed": observed_updates,
        "final_observed": final,
        "expected_trajectory": {
            str(step): value for step, value in EXPECTED_TRAJECTORY.items()
        },
        "derived_trajectory_diagnostic_only": derived,
        "derived_values_are_claimable": False,
    }


def rank_summary(path: Path, expected_trace: Mapping[str, Any]) -> dict[str, Any]:
    rank = int(path.stem.split("_")[-1])
    events, truncated_tail = read_jsonl_tail_safe(path)
    counts = collections.Counter(str(event.get("phase")) for event in events)
    starts = {
        int(event_step(event)): float(event_value(event, "time_unix"))
        for event in events
        if event.get("phase") == "training_step_start"
        and event_step(event) is not None
        and event_value(event, "time_unix") is not None
    }
    ends = {
        int(event_step(event)): float(event_value(event, "time_unix"))
        for event in events
        if event.get("phase") == "training_step_end"
        and event_step(event) is not None
        and event_value(event, "time_unix") is not None
    }
    errors: list[str] = []
    if sorted(ends) != list(EXPECTED_STEPS) or sorted(starts) != list(EXPECTED_STEPS):
        errors.append(
            f"step coverage starts={sorted(starts)} ends={sorted(ends)}, expected={list(EXPECTED_STEPS)}"
        )
    if truncated_tail:
        errors.append("phase trace has an incomplete final JSONL tail")
    try:
        trace = validate_trace_binding(events, expected_trace, rank)
    except EvidenceError as exc:
        trace = {"status": "FAIL", "error": str(exc)}
        errors.append(str(exc))
    versions = validate_version_trajectory(events, rank)
    errors.extend(str(error) for error in versions["errors"])

    activation = phases(events, ("activation_offload_final",))
    streamers = phases(events, ("frozen_block_streamer_final",))
    activation_live = [
        int(event_value(event, "live_cpu_bytes") or 0) for event in activation
    ]
    streamer_resident = [
        int(event_value(event, "resident_bytes") or 0) for event in streamers
    ]
    if activation_live != [0]:
        errors.append(f"activation offload final live bytes={activation_live}, expected=[0]")
    if streamer_resident != [0, 0]:
        errors.append(
            f"streamer final resident bytes={streamer_resident}, expected=[0, 0]"
        )
    complete = counts.get("training_complete_skip_final_save", 0) == 1 or counts.get(
        "training_complete", 0
    ) == 1
    if not complete:
        errors.append("training completion event count is not exactly one")
    return {
        "rank": rank,
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "tail_truncated": truncated_tail,
        "event_count": len(events),
        "counts": dict(counts),
        "step_start_unix": starts,
        "step_end_unix": ends,
        "trace": trace,
        "versions": versions,
        "lifecycle": {
            "activation_live_final": activation_live,
            "streamer_resident_final": streamer_resident,
            "streamer_materialized_bytes": sum(
                int(event_value(event, "materialized_bytes") or 0)
                for event in streamers
            ),
            "streamer_materialize_count": sum(
                int(event_value(event, "materialize_count") or 0)
                for event in streamers
            ),
        },
        "cuda": {
            "max_allocated_bytes": max(
                (
                    int(
                        (event.get("cuda") or {}).get("max_allocated_bytes", 0)
                    )
                    for event in events
                    if isinstance(event.get("cuda"), Mapping)
                ),
                default=0,
            ),
            "max_reserved_bytes": max(
                (
                    int((event.get("cuda") or {}).get("max_reserved_bytes", 0))
                    for event in events
                    if isinstance(event.get("cuda"), Mapping)
                ),
                default=0,
            ),
        },
        "raw_sha256": sha256_file(path),
    }


def node_summary(run: Path, node_rank: int, contract_sha: str) -> dict[str, Any]:
    launcher = run / "launcher" / f"node_{node_rank}"
    manifest_path = launcher / "launch_manifest.txt"
    manifest = parse_strict_kv(manifest_path)
    if manifest.get("schema_version") != "distillschedule.typed_routea_world16.node_launch.v3":
        raise EvidenceError(
            f"node {node_rank}: unsupported/legacy launcher schema {manifest.get('schema_version')!r}"
        )
    if manifest.get("node_rank") != str(node_rank):
        raise EvidenceError(f"node {node_rank}: manifest rank mismatch")
    if manifest.get("evidence_contract_sha256") != contract_sha:
        raise EvidenceError(f"node {node_rank}: evidence contract SHA mismatch")
    if manifest.get("pre_model_gate") != "PASS":
        raise EvidenceError(f"node {node_rank}: pre-model gate did not pass")
    for key in (
        "runtime_identity_path",
        "gpu_inventory_path",
        "accelerate_config_path",
        "node_contract_path",
        "rendered_config_path",
        "topology_path",
    ):
        if key not in manifest:
            raise EvidenceError(f"node {node_rank}: missing manifest field {key}")
    identity_path = Path(manifest["runtime_identity_path"])
    if sha256_file(identity_path) != manifest.get("runtime_identity_sha256"):
        raise EvidenceError(f"node {node_rank}: runtime identity SHA mismatch")
    identity = load_json_mapping(identity_path)
    if bool((identity.get("helios") or {}).get("dirty")):
        raise EvidenceError(f"node {node_rank}: Helios source was dirty")
    inventory = gpu_inventory(
        Path(manifest["gpu_inventory_path"]),
        manifest["gpu_inventory_sha256"],
        int(manifest["gpu_inventory_rows"]),
    )
    before = parse_memory_events(manifest.get("memory_events_before", ""))
    after = parse_memory_events(manifest.get("memory_events_after", ""))
    console_path = launcher / "console.log"
    console_text = console_path.read_text(errors="replace") if console_path.is_file() else ""
    final_rc = integer(manifest.get("final_return_code"), default=-1)
    oom_delta = after.get("oom", 0) - before.get("oom", 0)
    oom_kill_delta = after.get("oom_kill", 0) - before.get("oom_kill", 0)
    cuda_oom = bool(CUDA_OOM.search(console_text))
    memory_failure = cuda_oom or oom_delta > 0 or oom_kill_delta > 0
    if final_rc != 0 and memory_failure:
        memory_failure = True
    return {
        "node_rank": node_rank,
        "host": manifest.get("host"),
        "final_return_code": final_rc,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "console_path": str(console_path),
        "console_sha256": sha256_file(console_path) if console_path.is_file() else None,
        "runtime_identity": identity,
        "gpu_inventory": inventory,
        "resource": resource_summary(launcher / "resource.csv"),
        "memory_observation": {
            "acceptance_policy": "actual_oom_only",
            "host_reserve_gate": False,
            "cgroup_memory_current_before": integer(
                manifest.get("cgroup_memory_current_before")
            ),
            "cgroup_memory_current_after": integer(
                manifest.get("cgroup_memory_current_after")
            ),
            "cgroup_memory_peak_before": integer(
                manifest.get("cgroup_memory_peak_before")
            ),
            "cgroup_memory_peak_after": integer(
                manifest.get("cgroup_memory_peak_after")
            ),
            "cgroup_memory_max": integer(manifest.get("cgroup_memory_max")),
            "oom_delta": oom_delta,
            "oom_kill_delta": oom_kill_delta,
            "cuda_oom_signature": cuda_oom,
            "memory_failure": memory_failure,
            "high_pressure_without_oom_is_failure": False,
        },
        "topology_sha256": manifest.get("topology_sha256"),
        "rendered_config_sha256": manifest.get("rendered_config_sha256"),
        "node_contract_sha256": manifest.get("node_contract_sha256"),
    }


def timing_summary(ranks: Sequence[Mapping[str, Any]], old_result: Path) -> dict[str, Any]:
    starts = {step: [] for step in EXPECTED_STEPS}
    ends = {step: [] for step in EXPECTED_STEPS}
    for rank in ranks:
        for step in EXPECTED_STEPS:
            start = rank["step_start_unix"].get(step)
            end = rank["step_end_unix"].get(step)
            if start is not None:
                starts[step].append(float(start))
            if end is not None:
                ends[step].append(float(end))
    complete = all(len(starts[step]) == 16 and len(ends[step]) == 16 for step in EXPECTED_STEPS)
    if not complete:
        return {"status": "NOT_CLAIMABLE", "reason": "incomplete 16-rank timing"}
    step_wall = {
        str(step): max(ends[step]) - min(starts[step]) for step in EXPECTED_STEPS
    }
    window = max(max(values) for values in ends.values()) - min(
        min(values) for values in starts.values()
    )
    baseline_window = None
    if old_result.is_file():
        old = load_json_mapping(old_result)
        timing = old.get("timing_summary") or old.get("timing")
        if isinstance(timing, Mapping):
            value = timing.get("global_training_window_seconds")
            if value is not None:
                baseline_window = float(value)
    relative = window / baseline_window - 1 if baseline_window else None
    return {
        "status": "MEASURED_DIAGNOSTIC",
        "global_step_wall_span_seconds": step_wall,
        "global_training_window_seconds": window,
        "weighted_step_seconds": window / 5,
        "samples_per_second": 80 / window,
        "baseline_global_training_window_seconds": baseline_window,
        "relative_delta_vs_baseline": relative,
        "within_preregistered_10pct_diagnostic_band": (
            abs(relative) <= 0.10 if relative is not None else None
        ),
        "formal_speedup_claim": False,
    }


def topology_summary(path: Path) -> dict[str, Any]:
    value = load_json_mapping(path)
    records = value.get("records")
    if isinstance(records, list):
        global_ranks = sorted(
            int(record["global_rank"] if "global_rank" in record else record["rank"])
            for record in records
        )
        hosts = sorted({str(record["hostname"]) for record in records})
        per_host = collections.Counter(str(record["hostname"]) for record in records)
        local_by_host = {
            host: sorted(
                int(record["local_rank"])
                for record in records
                if str(record["hostname"]) == host
            )
            for host in hosts
        }
    else:
        global_ranks = sorted(int(item) for item in value.get("global_ranks", []))
        hosts = sorted(str(item) for item in value.get("hosts", []))
        per_host = collections.Counter(value.get("ranks_per_host", {}))
        local_by_host = value.get("local_ranks_by_host", {})
    errors: list[str] = []
    if value.get("status") != "PASS":
        errors.append("topology manifest status is not PASS")
    if global_ranks != list(EXPECTED_RANKS):
        errors.append(f"global ranks={global_ranks}")
    if len(hosts) != 2:
        errors.append(f"hosts={hosts}")
    for host in hosts:
        count = per_host[host] if isinstance(per_host, collections.Counter) else 8
        if count != 8:
            errors.append(f"host {host} rank count={count}")
        observed_local = [int(item) for item in local_by_host.get(host, [])]
        if observed_local and observed_local != list(range(8)):
            errors.append(f"host {host} local ranks={observed_local}")
    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "path": str(path),
        "sha256": sha256_file(path),
        "global_ranks": global_ranks,
        "hosts": hosts,
        "raw": value,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--evidence-contract", type=Path)
    parser.add_argument("--baseline-result", type=Path, default=DEFAULT_OLD_RESULT)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    run = root / "artifacts/runs" / args.run_id
    output = args.output or root / "artifacts/manifests" / f"{args.run_id}_result.json"
    contract_path = args.evidence_contract or run / "preregistration/evidence_contract.json"

    evidence_errors: list[str] = []
    contract: dict[str, Any] | None = None
    try:
        contract = load_json_mapping(contract_path)
        validate_final_contract(contract)
    except Exception as exc:
        evidence_errors.append(f"evidence contract: {exc}")

    nodes: list[dict[str, Any]] = []
    contract_sha = sha256_file(contract_path) if contract_path.is_file() else ""
    for node_rank in (0, 1):
        try:
            nodes.append(node_summary(run, node_rank, contract_sha))
        except Exception as exc:
            evidence_errors.append(f"node {node_rank}: {exc}")

    topology: dict[str, Any] = {"status": "MISSING"}
    topology_path = run / "preregistration/topology.json"
    try:
        topology = topology_summary(topology_path)
        if topology["status"] != "PASS":
            evidence_errors.extend(topology["errors"])
        if nodes and len({node.get("topology_sha256") for node in nodes}) != 1:
            evidence_errors.append("node topology SHA records differ")
        if nodes and nodes[0].get("topology_sha256") != topology.get("sha256"):
            evidence_errors.append("node/topology manifest SHA mismatch")
    except Exception as exc:
        evidence_errors.append(f"topology: {exc}")

    ranks: list[dict[str, Any]] = []
    if contract is not None:
        expected_entries = {
            int(entry["rank"]): entry
            for entry in contract["shared"]["trace"]["ranks"]
        }
        files = sorted(
            (run / "phase_trace").glob("rank_*.jsonl"),
            key=lambda path: int(path.stem.split("_")[-1]),
        )
        observed_ranks = [int(path.stem.split("_")[-1]) for path in files]
        if observed_ranks != list(EXPECTED_RANKS):
            evidence_errors.append(
                f"phase traces cover ranks {observed_ranks}, expected {list(EXPECTED_RANKS)}"
            )
        for path in files:
            rank = int(path.stem.split("_")[-1])
            try:
                ranks.append(rank_summary(path, expected_entries[rank]))
            except Exception as exc:
                evidence_errors.append(f"rank {rank}: {exc}")
    else:
        files = sorted((run / "phase_trace").glob("rank_*.jsonl"))

    rank_errors = [
        f"rank {rank['rank']}: {error}"
        for rank in ranks
        for error in rank.get("errors", [])
    ]
    evidence_errors.extend(rank_errors)

    checkpoint: dict[str, Any] = {"status": "MISSING"}
    checkpoint_path = run / "checkpoint-15"
    try:
        inventory = directory_inventory(checkpoint_path)
        checkpoint = {"status": "PASS", **inventory.to_dict()}
        if inventory.file_count == 0:
            evidence_errors.append("checkpoint-15 is empty")
    except Exception as exc:
        evidence_errors.append(f"checkpoint-15: {exc}")

    memory_failures = [
        node["node_rank"]
        for node in nodes
        if node["memory_observation"]["memory_failure"]
    ]
    nonzero_nodes = [
        node["node_rank"] for node in nodes if node.get("final_return_code") != 0
    ]
    if nodes and len({object_digest(node["runtime_identity"]) for node in nodes}) != 1:
        # Hostnames and absolute executable paths may legitimately differ.  Compare
        # the training-relevant identities separately instead of blindly requiring
        # the whole JSON object to match.
        identities = [
            {
                "python_sha": node["runtime_identity"]["python"]["sha256"],
                "accelerate_sha": node["runtime_identity"]["accelerate"]["sha256"],
                "deepspeed_sha": node["runtime_identity"]["deepspeed"]["sha256"],
                "helios_head": node["runtime_identity"]["helios"]["head"],
                "helios_entrypoint_sha": node["runtime_identity"]["helios"][
                    "entrypoint_sha256"
                ],
            }
            for node in nodes
        ]
        if identities[0] != identities[1]:
            evidence_errors.append("node runtime/source identities differ")

    full_attempt004_shape = (
        contract is not None
        and len(nodes) == 2
        and len(ranks) == 16
        and topology.get("status") == "PASS"
    )
    legacy_partial = not full_attempt004_shape
    if memory_failures:
        status = "OOM"
    elif nonzero_nodes:
        status = "CRASH"
    elif evidence_errors:
        status = "PARTIAL" if legacy_partial else "FAIL"
    else:
        status = "PASS"

    timing = (
        timing_summary(ranks, args.baseline_result)
        if status == "PASS"
        else {
            "status": "NOT_CLAIMABLE",
            "reason": "functional/evidence contract did not pass",
        }
    )
    payload: dict[str, Any] = {
        "schema_version": "distillschedule.typed_routea_world16.revalidation.v2",
        "run_id": args.run_id,
        "status": status,
        "claimable": status == "PASS",
        "claim_scope": (
            "one-run functional acceptance and diagnostic consistency of the typed Route-A world16 path; not a formal repeated speedup"
            if status == "PASS"
            else "non-claimable partial/failure evidence"
        ),
        "legacy_attempt003_policy": {
            "status": "PARTIAL",
            "reason": "invalid mixed key=value/CSV launcher manifest and no in-run topology/evidence contract",
            "retroactive_repair_allowed": False,
        },
        "evidence_contract": {
            "path": str(contract_path),
            "sha256": contract_sha or None,
            "contract_digest": contract.get("contract_digest") if contract else None,
            "shared_digest": contract.get("shared_digest") if contract else None,
        },
        "frozen_contract": {
            "world_size": 16,
            "micro_batch_per_gpu": 1,
            "gradient_accumulation_steps": 1,
            "global_batch": 16,
            "resolution": [384, 640],
            "latent": [1, 16, 9, 48, 80],
            "steps": list(EXPECTED_STEPS),
            "zero_stage": 1,
            "optimizer_offload": "cpu",
            "parameter_offload": "none",
            "initial_versions": EXPECTED_INITIAL,
            "final_versions": EXPECTED_TRAJECTORY[14],
        },
        "evidence_errors": evidence_errors,
        "nodes": nodes,
        "topology": topology,
        "rank_count": len(ranks),
        "rank_status": {str(rank["rank"]): rank["status"] for rank in ranks},
        "ranks": ranks,
        "checkpoint_15": checkpoint,
        "memory_observation": {
            "normative_gate": "actual_oom_only",
            "host_reserve_pass_fail_removed": True,
            "memory_failure_nodes": memory_failures,
            "node_observations": [node["memory_observation"] for node in nodes],
        },
        "timing": timing,
        "gpu": {
            "max_allocated_bytes": max(
                (rank["cuda"]["max_allocated_bytes"] for rank in ranks), default=0
            ),
            "max_reserved_bytes": max(
                (rank["cuda"]["max_reserved_bytes"] for rank in ranks), default=0
            ),
        },
        "lifecycle": {
            "max_materialized_bytes_per_rank": max(
                (
                    rank["lifecycle"]["streamer_materialized_bytes"]
                    for rank in ranks
                ),
                default=0,
            ),
            "max_materialize_count_per_rank": max(
                (
                    rank["lifecycle"]["streamer_materialize_count"]
                    for rank in ranks
                ),
                default=0,
            ),
        },
        "limitations": [
            "Single diagnostic-instrumented acceptance run; formal timing requires matched repeated Route-A/Route-B trials.",
            "The typed executor still invokes the pinned external Helios model-math compatibility entrypoint.",
            "Host-memory pressure and cgroup peak/max are observational unless an actual OOM/oom_kill or memory-attributed nonzero exit occurs.",
            "Any derived version trajectory is diagnostic only; PASS requires observed component versions, EMA predicates and a final observed record.",
        ],
    }
    atomic_json(output, payload)
    print(
        json.dumps(
            {
                "output": str(output),
                "status": status,
                "claimable": payload["claimable"],
                "rank_count": len(ranks),
                "error_count": len(evidence_errors),
                "window": timing.get("global_training_window_seconds"),
            },
            sort_keys=True,
        )
    )
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
