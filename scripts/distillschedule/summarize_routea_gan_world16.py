#!/usr/bin/env python3
"""Strict acceptance summarizer for the official-semantics GAN Route-A baseline."""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from distillschedule.runtime.gan_baseline import (
    EXPECTED_FINAL,
    EXPECTED_INITIAL,
    EXPECTED_STEPS,
    EXPECTED_TRAJECTORY,
    load_json_mapping,
    sha256_file,
    validate_final_contract,
)
from distillschedule.runtime.routea_evidence import directory_inventory

RANKS = tuple(range(16))
OOM_RE = re.compile(r"CUDA out of memory|torch\.OutOfMemoryError|CUDACachingAllocator.*out of memory", re.I)


class EvidenceError(RuntimeError):
    pass


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def strict_kv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line_no, raw in enumerate(path.read_text(errors="replace").splitlines(), 1):
        if not raw.strip():
            continue
        if "=" not in raw:
            raise EvidenceError(f"{path}: line {line_no} is not key=value")
        key, value = raw.split("=", 1)
        if not key or key.strip() != key or any(ch.isspace() for ch in key) or key in out:
            raise EvidenceError(f"{path}: invalid/duplicate key at line {line_no}")
        out[key] = value
    return out


def memory_events(text: str) -> dict[str, int]:
    tokens = text.replace(";", " ").split()
    out: dict[str, int] = {}
    for i in range(0, len(tokens) - 1, 2):
        try: out[tokens[i]] = int(tokens[i + 1])
        except ValueError: pass
    return out


def as_int(value: Any, default: int | None = None) -> int | None:
    if value is None or value in {"", "unavailable", "max"} or isinstance(value, bool):
        return default
    try: return int(value)
    except (TypeError, ValueError): return default


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], bool]:
    lines = path.read_bytes().splitlines(keepends=True)
    events: list[dict[str, Any]] = []
    partial_tail = False
    for i, raw in enumerate(lines):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw.decode())
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            if i == len(lines) - 1 and not raw.endswith((b"\n", b"\r")):
                partial_tail = True
                continue
            raise EvidenceError(f"{path}: malformed complete JSONL record: {exc}") from exc
        if not isinstance(value, dict):
            raise EvidenceError(f"{path}: JSONL record is not an object")
        events.append(value)
    return events, partial_tail


def event_value(event: Mapping[str, Any], *keys: str) -> Any:
    containers = [event]
    for outer in ("metadata", "versions", "stats", "state"):
        value = event.get(outer)
        if isinstance(value, Mapping):
            containers.append(value)
            if isinstance(value.get("versions"), Mapping): containers.append(value["versions"])
    for key in keys:
        for container in containers:
            if key in container: return container[key]
    return None


def step_of(event: Mapping[str, Any]) -> int | None:
    return as_int(event_value(event, "global_step", "logical_step", "optimization_step", "optimizer_step", "step"))


def version_of(event: Mapping[str, Any], component: str) -> int | None:
    aliases = {
        "generator": ("generator_version", "g_version", "generator"),
        "critic": ("critic_version", "c_version", "critic"),
        "ema": ("ema_version", "ema"),
    }[component]
    value = event_value(event, *aliases)
    if isinstance(value, Mapping): value = value.get("version")
    return as_int(value)


def phase(events: Sequence[Mapping[str, Any]], *names: str) -> list[Mapping[str, Any]]:
    allowed = set(names)
    return [event for event in events if event.get("phase") in allowed]


def ema_applied(event: Mapping[str, Any]) -> bool | None:
    value = event_value(event, "ema_update_enabled", "ema_predicate", "predicate_result", "updated", "applied")
    if isinstance(value, bool): return value
    if isinstance(value, (int, float)): return bool(value)
    if isinstance(value, str):
        lowered = value.lower().strip()
        if lowered in {"1", "true", "enabled", "applied", "pass"}: return True
        if lowered in {"0", "false", "disabled", "skipped", "fail"}: return False
    return None


def validate_trace(events: Sequence[Mapping[str, Any]], expected: Mapping[str, Any], rank: int) -> dict[str, Any]:
    items = phase(events, "operation_trace_input_validated")
    if len(items) != 1:
        raise EvidenceError(f"rank{rank}: trace validation count={len(items)}, expected1")
    event = items[0]
    trace_hash = event_value(event, "trace_hash")
    if trace_hash != expected.get("trace_hash"):
        raise EvidenceError(f"rank{rank}: trace hash mismatch")
    observed_rank = as_int(event_value(event, "trace_rank", "rank", "global_rank"))
    if observed_rank is not None and observed_rank != rank:
        raise EvidenceError(f"rank{rank}: trace qualified for rank{observed_rank}")
    authority = event_value(event, "source_authority", "trace_authority")
    aliases = [str(x) for x in expected.get("source_authority_aliases", [])]
    if authority is None or str(authority) not in aliases:
        raise EvidenceError(f"rank{rank}: source authority mismatch")
    return {"trace_hash": trace_hash, "source_authority": str(authority)}


def validate_versions(events: Sequence[Mapping[str, Any]], rank: int) -> dict[str, Any]:
    errors: list[str] = []
    init_events = phase(events, "parameter_versions_initialized")
    initial = {key: None for key in EXPECTED_INITIAL}
    if len(init_events) == 1:
        initial = {key: version_of(init_events[0], key) for key in EXPECTED_INITIAL}
    if initial != EXPECTED_INITIAL:
        errors.append(f"initial={initial}, expected={EXPECTED_INITIAL}")

    specs = {
        "generator": (("generator_optimizer_step",), [1000]),
        "critic": (("critic_optimizer_step",), list(EXPECTED_STEPS)),
        "ema": (("ema_step", "ema_optimizer_step", "ema_update"), list(EXPECTED_STEPS)),
    }
    observed: dict[str, Any] = {}
    for component, (names, expected_steps) in specs.items():
        updates = [
            {
                "step": step_of(event),
                "version": version_of(event, component),
                "predicate": ema_applied(event) if component == "ema" else None,
            }
            for event in phase(events, *names)
        ]
        observed[component] = updates
        if [item["step"] for item in updates] != expected_steps:
            errors.append(f"{component} steps={[item['step'] for item in updates]}, expected={expected_steps}")
        for item in updates:
            step = item["step"]
            if step in EXPECTED_TRAJECTORY and item["version"] != EXPECTED_TRAJECTORY[step][component]:
                errors.append(f"{component}@{step} version={item['version']} expected={EXPECTED_TRAJECTORY[step][component]}")
            if component == "ema" and item["predicate"] is not True:
                errors.append(f"EMA@{step} predicate not observed true")

    final = {key: None for key in EXPECTED_FINAL}
    for event in reversed(phase(events, "parameter_versions_final", "parameter_versions_committed", "training_complete_skip_final_save", "training_complete")):
        values = {key: version_of(event, key) for key in EXPECTED_FINAL}
        if all(value is not None for value in values.values()):
            final = values
            break
    if final != EXPECTED_FINAL:
        errors.append(f"final={final}, expected={EXPECTED_FINAL}")
    return {"status": "OBSERVED_EXACT" if not errors else "MISMATCH", "errors": errors, "initial": initial, "updates": observed, "final": final}


def summarize_rank(path: Path, expected_trace: Mapping[str, Any]) -> dict[str, Any]:
    rank = int(path.stem.split("_")[-1])
    events, partial = read_jsonl(path)
    errors: list[str] = []
    starts = {step_of(e): float(event_value(e, "time_unix")) for e in events if e.get("phase") == "training_step_start" and step_of(e) is not None and event_value(e, "time_unix") is not None}
    ends = {step_of(e): float(event_value(e, "time_unix")) for e in events if e.get("phase") == "training_step_end" and step_of(e) is not None and event_value(e, "time_unix") is not None}
    if sorted(starts) != list(EXPECTED_STEPS) or sorted(ends) != list(EXPECTED_STEPS):
        errors.append(f"step coverage starts={sorted(starts)} ends={sorted(ends)}")
    if partial: errors.append("incomplete final JSONL tail")
    try: trace = validate_trace(events, expected_trace, rank)
    except EvidenceError as exc: trace = {"status":"FAIL","error":str(exc)}; errors.append(str(exc))
    versions = validate_versions(events, rank); errors.extend(versions["errors"])
    activation = [as_int(event_value(e, "live_cpu_bytes"), -1) for e in phase(events, "activation_offload_final")]
    streamers = [as_int(event_value(e, "resident_bytes"), -1) for e in phase(events, "frozen_block_streamer_final")]
    if activation != [0]: errors.append(f"activation final={activation}, expected=[0]")
    if streamers != [0, 0]: errors.append(f"streamer final={streamers}, expected=[0,0]")
    complete = len(phase(events, "training_complete_skip_final_save", "training_complete")) == 1
    if not complete: errors.append("training completion event count !=1")
    return {
        "rank": rank,
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "trace": trace,
        "versions": versions,
        "step_start_unix": starts,
        "step_end_unix": ends,
        "lifecycle": {"activation_live_final": activation, "streamer_resident_final": streamers},
        "raw_sha256": sha256_file(path),
    }


def resource_summary(path: Path) -> dict[str, Any]:
    if not path.is_file(): return {"status":"MISSING","samples":0}
    rows = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try: rows.append((float(row["unix_time"]), int(row["cgroup_bytes"]), int(row["host_used_bytes"])))
            except (KeyError, TypeError, ValueError): pass
    return {
        "status": "MEASURED" if rows else "EMPTY",
        "samples": len(rows),
        "cgroup_peak_bytes": max((r[1] for r in rows), default=None),
        "host_used_peak_bytes": max((r[2] for r in rows), default=None),
    }


def summarize_node(run: Path, node_rank: int, contract_sha: str) -> dict[str, Any]:
    launcher = run / "launcher" / f"node_{node_rank}"
    manifest = strict_kv(launcher / "launch_manifest.txt")
    if manifest.get("schema_version") != "distillschedule.typed_routea_gan_world16.node_launch.v1":
        raise EvidenceError(f"node{node_rank}: launch schema mismatch")
    if manifest.get("node_rank") != str(node_rank) or manifest.get("evidence_contract_sha256") != contract_sha or manifest.get("pre_model_gate") != "PASS":
        raise EvidenceError(f"node{node_rank}: manifest contract/pre-model gate mismatch")
    runtime_path = Path(manifest.get("runtime_identity_path", ""))
    if not runtime_path.is_file() or sha256_file(runtime_path) != manifest.get("runtime_identity_sha256"):
        raise EvidenceError(f"node{node_rank}: runtime identity mismatch")
    runtime = load_json_mapping(runtime_path)
    if bool((runtime.get("helios") or {}).get("dirty")):
        raise EvidenceError(f"node{node_rank}: dirty Helios checkout")
    before, after = memory_events(manifest.get("memory_events_before", "")), memory_events(manifest.get("memory_events_after", ""))
    console = launcher / "console.log"
    console_text = console.read_text(errors="replace") if console.is_file() else ""
    memory_failure = bool(OOM_RE.search(console_text)) or after.get("oom",0) > before.get("oom",0) or after.get("oom_kill",0) > before.get("oom_kill",0)
    return {
        "node_rank": node_rank,
        "host": manifest.get("host"),
        "final_return_code": as_int(manifest.get("final_return_code"), -1),
        "runtime_identity": runtime,
        "resource": resource_summary(launcher / "resource.csv"),
        "memory_observation": {
            "policy": "actual_oom_only",
            "host_reserve_gate": False,
            "memory_failure": memory_failure,
            "oom_delta": after.get("oom",0)-before.get("oom",0),
            "oom_kill_delta": after.get("oom_kill",0)-before.get("oom_kill",0),
            "cgroup_memory_peak_after": as_int(manifest.get("cgroup_memory_peak_after")),
        },
    }


def validate_topology(path: Path) -> dict[str, Any]:
    payload = load_json_mapping(path)
    summary, records = payload.get("summary"), payload.get("records")
    if payload.get("status") != "PASS" or not isinstance(summary, Mapping) or not isinstance(records, list):
        raise EvidenceError("topology probe malformed/not PASS")
    if (summary.get("world_size"), summary.get("nodes"), summary.get("local_world_size")) != (16, 2, 8):
        raise EvidenceError("topology dimensions mismatch")
    ranks = sorted(as_int(r.get("rank"), -1) for r in records if isinstance(r, Mapping))
    if ranks != list(RANKS): raise EvidenceError(f"topology ranks={ranks}, expected0..15")
    return payload


def timing(ranks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    starts, ends = {s:[] for s in EXPECTED_STEPS}, {s:[] for s in EXPECTED_STEPS}
    for rank in ranks:
        for step in EXPECTED_STEPS:
            if rank["step_start_unix"].get(step) is not None: starts[step].append(float(rank["step_start_unix"][step]))
            if rank["step_end_unix"].get(step) is not None: ends[step].append(float(rank["step_end_unix"][step]))
    if not all(len(starts[s]) == 16 and len(ends[s]) == 16 for s in EXPECTED_STEPS):
        return {"status":"NOT_CLAIMABLE","formal_speedup_claim":False}
    window = max(max(v) for v in ends.values()) - min(min(v) for v in starts.values())
    return {
        "status":"MEASURED_DIAGNOSTIC",
        "global_training_window_seconds":window,
        "weighted_step_seconds":window/5,
        "samples_per_second":80/window if window > 0 else None,
        "formal_speedup_claim":False,
    }


def summarize(run_id: str, root: Path, output: Path | None = None) -> dict[str, Any]:
    run = root / "artifacts/runs" / run_id
    destination = output or root / "artifacts/manifests" / f"{run_id}_result.json"
    contract_path = run / "preregistration/evidence_contract.json"
    errors: list[str] = []
    if not contract_path.is_file():
        result = {"schema_version":"distillschedule.routea_gan_world16.result.v1","run_id":run_id,"status":"PARTIAL","errors":[f"missing {contract_path}"]}
        atomic_json(destination, result); return result
    try:
        contract = load_json_mapping(contract_path); validate_final_contract(contract)
    except Exception as exc:
        contract = {}; errors.append(f"evidence contract invalid: {exc}")
    contract_sha = sha256_file(contract_path)

    nodes = []
    for rank in (0,1):
        try: nodes.append(summarize_node(run, rank, contract_sha))
        except Exception as exc: errors.append(str(exc))
    try: topology = validate_topology(run / "preregistration/topology.json")
    except Exception as exc: topology = None; errors.append(str(exc))

    expected_traces: dict[int, Mapping[str, Any]] = {}
    shared = contract.get("shared") if isinstance(contract, Mapping) else None
    trace_contract = shared.get("trace") if isinstance(shared, Mapping) else None
    if isinstance(trace_contract, Mapping) and isinstance(trace_contract.get("ranks"), list):
        expected_traces = {int(item["rank"]):item for item in trace_contract["ranks"] if isinstance(item, Mapping)}
    rank_files = sorted((run/"phase_trace").glob("rank_*.jsonl"), key=lambda p:int(p.stem.split("_")[-1])) if (run/"phase_trace").is_dir() else []
    ranks = []
    for path in rank_files:
        rank = int(path.stem.split("_")[-1])
        if rank not in expected_traces: errors.append(f"rank{rank}: no preregistered trace"); continue
        try:
            item = summarize_rank(path, expected_traces[rank]); ranks.append(item); errors.extend(item["errors"])
        except Exception as exc: errors.append(str(exc))
    observed = sorted(item["rank"] for item in ranks)
    if observed != list(RANKS): errors.append(f"phase ranks={observed}, expected0..15")

    checkpoint = directory_inventory(run/"checkpoint-1005") if (run/"checkpoint-1005").is_dir() else None
    if checkpoint is None or checkpoint.file_count == 0: errors.append("checkpoint-1005 missing/empty")
    source_schedule = shared.get("official_schedule") if isinstance(shared, Mapping) else None
    if not isinstance(source_schedule, Mapping): errors.append("official GAN schedule binding missing")
    memory_failure = any(node.get("memory_observation",{}).get("memory_failure") is True for node in nodes)
    nonzero = any(node.get("final_return_code") != 0 for node in nodes)
    exact = len(ranks)==16 and all(rank.get("status")=="PASS" for rank in ranks)
    if memory_failure: status="OOM"
    elif len(nodes)<2 or len(ranks)<16: status="PARTIAL"
    elif nonzero: status="CRASH"
    elif errors or not exact or topology is None: status="CONTRACT_FAIL"
    else: status="PASS"

    result = {
        "schema_version":"distillschedule.routea_gan_world16.result.v1",
        "run_id":run_id,
        "status":status,
        "claim_scope":"official Helios GAN-on Route-A full-offload baseline; first PASS is functional/diagnostic, not yet a formal speedup denominator",
        "official_semantics":{
            "logical_steps":list(EXPECTED_STEPS),
            "initial_versions":EXPECTED_INITIAL,
            "final_versions":EXPECTED_FINAL,
            "gan_start_step":1000,
            "critic_optimizer_updates":5,
            "generator_optimizer_updates":1,
            "ema_updates":5,
            "low_vram_joint_critic_backward_one_optimizer_step":True,
            "schedule_binding":source_schedule,
        },
        "offload_baseline":{
            "generator_frozen_blocks":"0-39 streamed window1",
            "score_frozen_blocks":"0-39 streamed window1",
            "optimizer_offload":"cpu",
            "prefetch_distance":0,
            "pipeline_overlap":False,
            "vae":"official GAN low-VRAM temporary GPU residency then CPU",
        },
        "nodes":nodes,
        "topology":topology,
        "observed_ranks":observed,
        "ranks":ranks,
        "checkpoint_1005":checkpoint.to_dict() if checkpoint else None,
        "timing":timing(ranks),
        "errors":sorted(set(errors)),
    }
    result["timing"]["claimable"] = status == "PASS"
    atomic_json(destination, result)
    return result


def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--run-id",required=True); p.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[2]); p.add_argument("--output",type=Path)
    a=p.parse_args(); result=summarize(a.run_id,a.root.expanduser().resolve(),a.output.expanduser().resolve() if a.output else None)
    print(json.dumps({"status":result["status"],"run_id":result["run_id"],"rank_count":len(result.get("ranks",[])),"timing_status":result.get("timing",{}).get("status")},sort_keys=True))
    return 0 if result["status"]=="PASS" else 2


if __name__ == "__main__": raise SystemExit(main())
