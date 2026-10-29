"""Read-only C51 scalar phase evidence, never cycle/checkpoint admission.

Bindings are supplied explicitly by the caller. PID evidence is only membership
in that caller's namespace PID census: no process-birth or liveness inference is
made. Host boundary samples establish neither CUDA intervals nor memory minima.
This module imports no Torch, native worker, model, observer, launcher or owner.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import uuid

from distillschedule.workloads.best_known.registry import fingerprint


SCHEMA = "distillschedule.livewan_c51_phase_receipts.v1"
FLAG = "DISTILLSCHEDULE_LIVEWAN_C51_PHASE_RECEIPTS"
MEMORY_FIELDS = {"allocated_bytes", "reserved_bytes", "max_allocated_bytes", "max_reserved_bytes",
                 "driver_free_bytes", "driver_total_bytes"}
LABELS = dict(schema_version=SCHEMA, status="DIAGNOSTIC_ONLY_PARTIAL_HISTORY",
              peak_scope="ORIGINAL_GLOBAL_COUNTERS_NO_PHASE_RESETS", unsynchronized_host_samples=True,
              diagnostic_only=True, formal_timing_eligible=False, claimable_speedup=False,
              native_cuda_numerical_qualification="NOT_RUN", capacity_qualification="NOT_RUN")
ROW_FIELDS = {"sequence", "phase", "cycle", "global_step", "wave", "wave_scope", "physical_versions",
              "monotonic_ns", "wall_time_ns", "memory", "memory_origin", "timing_class", "observation"}
PID_SCOPE = "CALLER_NAMESPACE_PID_MEMBERSHIP_NOT_BIRTH_IDENTITY"
MAX_DOCUMENT_BYTES = 4*1024*1024


def _require(value, message):
    if not value:
        raise ValueError("C51 phase evidence: " + message)


def _hash(value, length=64):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{"+str(length)+"}", value) is not None


def _versions(value):
    _require(type(value) is dict and set(value) == {"generator", "critic", "ema"}
             and all(type(v) is int and v >= 0 for v in value.values()), "invalid physical clocks")
    return value


def _physical_versions(rank, cycle):
    # The producer's physical role clock, not a new update/cycle acceptance test.
    return {"generator": cycle if rank < 8 else 0, "critic": 0 if rank < 8 else 5*cycle, "ema": 0}


def _uuid(value):
    _require(type(value) is str, "missing GPU UUID")
    text = value.lower().removeprefix("gpu-")
    try:
        parsed = uuid.UUID(text)
    except ValueError:
        raise ValueError("C51 phase evidence: unresolved GPU UUID") from None
    _require(str(parsed) == text and parsed.int != 0, "unresolved GPU UUID")
    return text


def _expected_identity(*, training_identity, environment, run_id, binding, expected_pid,
                       namespace_pids, start_cycle, end_cycle, construction_versions):
    _require(type(environment) is dict and environment.get(FLAG) == "v1"
             and environment.get("DISTILLSCHEDULE_RUN_ID") == run_id, "caller must bind explicit v1 flag and stable run ID")
    _require(type(run_id) is str and re.fullmatch(r"[A-Za-z0-9._-]{1,128}", run_id) is not None
             and run_id.strip("."), "invalid explicit run ID")
    identity = training_identity
    _require(type(identity) is dict and identity.get("schema_version") == "distillschedule.livewan_c51_continuation.v1"
             and _hash(identity.get("source_head"), 40) and _hash(identity.get("assets_sha256"))
             and _hash(identity.get("input_sha256")), "invalid caller source/asset/input identity")
    sources = identity.get("source_files")
    _require(type(sources) is dict and sources, "complete caller source inventory required")
    for name, digest in sources.items():
        _require(type(name) is str and bool(name) and not PurePosixPath(name).is_absolute()
                 and ".." not in PurePosixPath(name).parts and "\\" not in name and _hash(digest), "invalid caller source file")
    horizon, cadence = identity.get("horizon"), identity.get("checkpoint_every")
    _require(type(horizon) is int and 2 <= horizon <= 100 and type(cadence) is int and cadence >= 0
             and type(start_cycle) is int and type(end_cycle) is int and 0 <= start_cycle < end_cycle <= horizon,
             "invalid declared segment/horizon/checkpoint cadence")
    _require(type(binding) is dict and set(binding) == {"rank", "node_rank", "cuda_device", "physical_device", "uuid", "physical_inventory"},
             "explicit original device binding required")
    rank = binding["rank"]
    _require(type(rank) is int and 0 <= rank < 16 and all(type(binding[k]) is int for k in ("node_rank", "cuda_device", "physical_device"))
             and binding["node_rank"] == rank//8 and binding["cuda_device"] == binding["physical_device"] == rank % 8,
             "caller physical rank/device differs")
    inventory = binding["physical_inventory"]
    _require(type(inventory) is dict and set(inventory) == {str(i) for i in range(8)}
             and all(_uuid(v) == v for v in inventory.values()) and len(set(inventory.values())) == 8
             and _uuid(binding["uuid"]) == inventory[str(rank % 8)], "caller GPU UUID/inventory differs")
    _require(type(expected_pid) is int and expected_pid > 0
             and type(namespace_pids) in (list, tuple, set, frozenset) and namespace_pids
             and all(type(pid) is int and pid > 0 for pid in namespace_pids)
             and expected_pid in namespace_pids, "caller namespace PID membership missing; birth identity is not inferred")
    _versions(construction_versions)
    return dict(run_id=run_id, training_identity_sha256=fingerprint(identity), source_head=identity["source_head"],
                source_files=sources, source_files_sha256=fingerprint(sources), assets_sha256=identity["assets_sha256"],
                input_sha256=identity["input_sha256"], rank=rank, node_rank=rank//8, local_rank=rank % 8,
                pid=expected_pid, gpu_uuid=binding["uuid"], physical_device=binding["physical_device"],
                physical_inventory=inventory, role="generator" if rank < 8 else "score",
                start_cycle=start_cycle, requested_stop_cycle=end_cycle)


def _safe_path(path, output_root, rank):
    root, path = Path(output_root), Path(path)
    _require(root.is_absolute() and path.is_absolute() and ".." not in root.parts and ".." not in path.parts,
             "absolute non-escaping output/path required")
    _require(path == root/"capacity/phase-receipts"/f"rank-{rank:05d}.json", "receipt path escapes declared output/rank")
    _require(not any(p.is_symlink() for p in (path, *path.parents)), "symlink in phase receipt path")
    return path


def _object(pairs):
    value = {}
    for key, entry in pairs:
        _require(key not in value, "duplicate JSON field")
        value[key] = entry
    return value


def _nonfinite(value):
    raise ValueError("C51 phase evidence: nonfinite JSON " + value)


def _document(path):
    try:
        before = path.stat()
        _require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= MAX_DOCUMENT_BYTES, "invalid/bounded phase document required")
        with path.open("rb") as stream:
            raw = stream.read(MAX_DOCUMENT_BYTES+1)
        after = path.stat()
        fields = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)
        _require(fields(before) == fields(after) and len(raw) == before.st_size, "phase document changed during read")
        _require(not any(p.is_symlink() for p in (path, *path.parents)), "symlink appeared during phase read")
        value = json.loads(raw, object_pairs_hook=_object, parse_constant=_nonfinite)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("C51 phase evidence: unreadable/incomplete phase document") from error
    return value, raw


def _phases(rank, start, end, cadence, construction_versions):
    rows = [(name, start, construction_versions) for name in ("construction_entry", "construction_exit", "setup_ready")]
    rows.append(("public_bind_exit", start, _physical_versions(rank, start)))
    for cycle in range(start+1, end+1):
        rows.extend((("full5_entry", cycle, _physical_versions(rank, cycle-1)),
                     ("full5_exit", cycle, _physical_versions(rank, cycle))))
        if cadence and cycle % cadence == 0:
            rows.extend((name, cycle, _physical_versions(rank, cycle)) for name in ("checkpoint_pack_entry", "checkpoint_pack_exit"))
    return rows


def _observations(value, *, rank, start, end, cadence, construction_versions):
    rows = value["observations"]
    expected = _phases(rank, start, end, cadence, construction_versions)
    limit = 4+4*(end-start)
    _require(type(value["observation_limit"]) is int and value["observation_limit"] == limit
             and type(rows) is list and 0 < len(rows) <= len(expected) <= limit, "invalid/bounded phase history length")
    previous = None
    for sequence, row in enumerate(rows):
        _require(type(row) is dict and set(row) == ROW_FIELDS, "phase row fields differ")
        phase, cycle, versions = expected[sequence]
        _require(type(row["sequence"]) is int and row["sequence"] == sequence and row["phase"] == phase
                 and type(row["cycle"]) is int and row["cycle"] == cycle
                 and _versions(row["physical_versions"]) == versions, "phase ordering/sequence/physical clock differs")
        step = 5*(cycle-1) if phase == "full5_entry" else 5*cycle
        _require(type(row["global_step"]) is int and row["global_step"] == step and row["wave"] is None
                 and row["wave_scope"] == "WHOLE_FULL5_BOTH_WAVES_NOT_PER_WAVE_SAMPLING", "phase step/wave scope differs")
        _require(type(row["monotonic_ns"]) is int and row["monotonic_ns"] >= 0
                 and type(row["wall_time_ns"]) is int and row["wall_time_ns"] > 0
                 and (previous is None or row["monotonic_ns"] >= previous), "invalid rank-local host timestamps")
        previous = row["monotonic_ns"]
        memory = row["memory"]
        _require(type(memory) is dict and set(memory) == MEMORY_FIELDS
                 and all(type(v) is int and v >= 0 for v in memory.values())
                 and memory["driver_free_bytes"] <= memory["driver_total_bytes"], "invalid original six scalar memory fields")
        origin = "EXISTING_SETUP_MEMORY_NO_REQUERY" if phase == "setup_ready" else "ORIGINAL_MEMORY_SNAPSHOT_HELPER"
        _require(row["memory_origin"] == origin and row["timing_class"] == "DIAGNOSTIC_NONPERFORMANCE"
                 and row["observation"] == "HOST_BOUNDARY_REACHED_NOT_CYCLE_ACCEPTANCE", "phase evidence origin/qualification differs")
    return len(rows) == len(expected)


def _qualifications():
    return dict(diagnostic_only=True, formal_timing_eligible=False, claimable_speedup=False,
                native_cuda_numerical_qualification="NOT_RUN", capacity_qualification="NOT_RUN",
                endpoint_qualification="NOT_ESTABLISHED", cleanup_qualification="NOT_EVALUATED_BY_PHASE_PARSER",
                pid_evidence_scope=PID_SCOPE,
                environment_evidence_scope="CALLER_DECLARED_OPT_IN_NOT_AN_OBSERVED_RUNTIME_ENVIRONMENT_RECEIPT")


def read_phase_receipt(path, *, output_root, training_identity, environment, run_id, binding, expected_pid,
                       namespace_pids, start_cycle, end_cycle, construction_versions):
    """Validate one actual published file and return diagnostic-only evidence.

    ``construction_versions`` must be explicitly bound to the real bootstrap;
    resumed construction is not relabeled with the later restored role clock.
    An interrupted valid prefix is evidence, not an accepted cycle or checkpoint.
    """
    expected = _expected_identity(training_identity=training_identity, environment=environment, run_id=run_id,
        binding=binding, expected_pid=expected_pid, namespace_pids=namespace_pids,
        start_cycle=start_cycle, end_cycle=end_cycle, construction_versions=construction_versions)
    path = _safe_path(path, output_root, binding["rank"])
    value, raw = _document(path)
    _require(type(value) is dict and set(value) == set(LABELS) | {"identity", "observations", "observation_limit"}, "phase document fields differ")
    for key, label in LABELS.items():
        _require(type(value[key]) is type(label) and value[key] == label, "phase document qualification differs: " + key)
    actual = value["identity"]
    _require(type(actual) is dict and set(actual) == set(expected)
             and all(type(actual[k]) is int for k in ("rank", "node_rank", "local_rank", "pid", "physical_device", "start_cycle", "requested_stop_cycle"))
             and actual == expected, "source/asset/input/training/run/rank/UUID/segment/PID binding differs")
    complete = _observations(value, rank=binding["rank"], start=start_cycle, end=end_cycle,
                             cadence=training_identity["checkpoint_every"], construction_versions=construction_versions)
    return dict(schema_version="distillschedule.livewan_c51_phase_evidence.v1", status="VALID_DIAGNOSTIC_HISTORY",
                path=str(path), receipt_sha256=hashlib.sha256(raw).hexdigest(), identity=actual,
                history_state="COMPLETE_DECLARED_PHASE_CHAIN" if complete else "INTERRUPTED_PREFIX",
                observation_count=len(value["observations"]), observations=value["observations"], **_qualifications())


def read_phase_receipts(output_root, *, training_identity, environment, run_id, start_cycle, end_cycle, rank_bindings):
    """Require exactly16 published rank histories; never admit an endpoint."""
    _require(type(rank_bindings) is dict and all(type(rank) is int for rank in rank_bindings)
             and set(rank_bindings) == set(range(16)), "exact16 caller rank bindings required")
    root = Path(output_root)
    directory = root/"capacity/phase-receipts"
    _safe_path(directory/"rank-00000.json", root, 0)
    _require(directory.is_dir() and {p.name for p in directory.glob("*.json")} == {f"rank-{rank:05d}.json" for rank in range(16)},
             "missing/extra published phase rank files")
    rows = []
    for rank in range(16):
        arguments = rank_bindings[rank]
        _require(type(arguments) is dict and set(arguments) == {"binding", "expected_pid", "namespace_pids", "construction_versions"}
                 and type(arguments["binding"]) is dict and arguments["binding"].get("rank") == rank, "caller rank binding fields differ")
        rows.append(read_phase_receipt(directory/f"rank-{rank:05d}.json", output_root=root, training_identity=training_identity,
                    environment=environment, run_id=run_id, start_cycle=start_cycle, end_cycle=end_cycle, **arguments))
    inventories = [row["identity"]["physical_inventory"] for row in rows]
    _require(all(inventories[rank] == inventories[rank//8*8] for rank in range(16))
             and len({_uuid(row["identity"]["gpu_uuid"]) for row in rows}) == 16, "all16 physical UUID inventories differ/overlap")
    return dict(schema_version="distillschedule.livewan_c51_phase_evidence_set.v1", status="DIAGNOSTIC_HISTORY_SET_ONLY",
                rank_count=16, all_phase_chains_complete=all(row["history_state"] == "COMPLETE_DECLARED_PHASE_CHAIN" for row in rows),
                ranks=rows, **_qualifications())
