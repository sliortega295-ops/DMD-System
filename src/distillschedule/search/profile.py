"""Bounded, per-device SQLite import for search observations (never GPU MFU).

An input manifest supplies exact rank/PID/device/NVTX identities from native
receipts. Unknown ownership and incomplete captures fail closed. Large reports
are analyzed where they reside; only normalized observations need be moved.
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import re
import sqlite3
import time
from typing import Any

from distillschedule.profiling.nsight_activity import (
    ActivityInterval, Interval, analyze_windows, clip_intervals_to_windows,
    interval_union_ns, sqlite_schema, _classify_copy,
)
from distillschedule.workloads.best_known.registry import file_sha256, fingerprint
from .contracts import PROFILE_SCHEMA, finite_number, require


def _quoted(name: str) -> str:
    require(bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name)), "unsafe SQLite identifier")
    return '"' + name + '"'


def _strings(db: sqlite3.Connection, ids: set[int]) -> dict[int, str]:
    result = {}
    ordered = sorted(ids)
    for offset in range(0, len(ordered), 500):
        batch = ordered[offset:offset + 500]
        marks = ",".join("?" for _ in batch)
        result.update(db.execute(f"SELECT id,value FROM StringIds WHERE id IN ({marks})", batch))
    return result


def _doctor(source: Path, specs: list[dict[str, Any]]) -> dict[str, Any]:
    bindings = [r.get("nsys_ai_doctor") for r in specs]
    require(bindings and all(x == bindings[0] for x in bindings), "inconsistent doctor receipt bindings")
    binding = bindings[0]
    require(isinstance(binding, dict) and set(binding) == {"path", "sha256"}, "hashed nsys-ai doctor receipt required")
    path = Path(binding["path"])
    require(path.is_file() and file_sha256(path) == binding["sha256"], "doctor receipt hash mismatch")
    value = json.loads(path.read_text())
    require(value.get("producer") == "nsys-ai" and bool(value.get("producer_version")), "not an nsys-ai doctor receipt")
    require(Path(value.get("profile_path", "")).resolve() == source.resolve(), "doctor belongs to another SQLite")
    health = [x for s in value.get("sections", []) if s["name"] == "Profile health" for x in s["checks"]]
    require(health and not any(x["status"] in {"fail", "error"} for x in health), "nsys-ai profile health failed")
    for name in ("Schema compatibility", "GPUs", "NVTX events"):
        require(any(x["name"] == name and x["status"] == "ok" for x in health), f"doctor {name} not PASS")
    return {**binding, "producer_version": value["producer_version"],
            "warnings": [x for x in health if x["status"] == "warn"]}


def _activity(db: sqlite3.Connection, spec: dict[str, Any], window: Interval) -> tuple[list[ActivityInterval], dict[str, Any]]:
    schema = sqlite_schema(db)
    table = next((t for t in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL") if t in schema), None)
    require(table is not None and "StringIds" in schema, "missing kernel/StringIds tables")
    cols = schema[table]
    require({"start", "end", "deviceId", "globalPid"}.issubset(cols), "kernel rank/device/process mapping unavailable")
    name_col = next((c for c in ("demangledName", "shortName", "name") if c in cols), None)
    require(name_col is not None, "kernel names unavailable")
    parameters = (spec["device_id"], spec["global_pid"], window.start_ns, window.end_ns)
    rows = list(db.execute(
        f"SELECT start,end,{_quoted(name_col)} FROM {_quoted(table)} "
        "WHERE deviceId=? AND globalPid=? AND end>? AND start<?", parameters))
    require(bool(rows), f"rank {spec['rank']} has no kernels in its exact capture")
    ids = {r[2] for r in rows if isinstance(r[2], int)}
    names = _strings(db, ids)
    result, unresolved, pointwise = [], [], []
    for start, end, name_id in rows:
        name = names.get(name_id, "") if isinstance(name_id, int) else str(name_id or "")
        if end <= start:
            continue
        interval = Interval(int(start), int(end))
        if not name or "UNRESOLVED" in name.upper():
            unresolved.append(interval)
        category = "nccl" if "nccl" in name.lower() else "compute"
        result.append(ActivityInterval(interval, category, name))
        if category == "compute" and re.search(r"elementwise|vectorized|copy|cast|cat|norm|modulat|residual|gate", name, re.I):
            pointwise.append(interval)
    denom = interval_union_ns(clip_intervals_to_windows((a.interval for a in result), (window,)))
    unknown_ns = interval_union_ns(clip_intervals_to_windows(unresolved, (window,)))
    require(denom > 0 and unknown_ns / denom < .01, "unresolved kernel duration >=1%")
    # Each row is kept once. Do not silently double-count a second memcpy view.
    copy_table = "CUPTI_ACTIVITY_KIND_MEMCPY"
    require(copy_table in schema, "memcpy table absent; cannot prove zero transfer")
    require({"start", "end", "deviceId", "globalPid", "copyKind", "bytes"}.issubset(schema[copy_table]),
            "memcpy direction/device/process fields unavailable")
    for start, end, kind, size in db.execute(
            "SELECT start,end,copyKind,bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY "
            "WHERE deviceId=? AND globalPid=? AND end>? AND start<?", parameters):
        if end > start:
            result.append(ActivityInterval(Interval(int(start), int(end)), _classify_copy(kind, {}), str(kind), int(size)))
    return result, {"unresolved_kernel_count": len(unresolved), "unresolved_kernel_union_ns": unknown_ns,
                    "pointwise_name_family_union_ns": interval_union_ns(clip_intervals_to_windows(pointwise, (window,))),
                    "pointwise_attribution": "NAME_FAMILY_HINT_NOT_OPERATOR_CORRELATION"}


def _validate_window(db: sqlite3.Connection, rank: dict[str, Any], item: dict[str, Any]) -> Interval:
    schema = sqlite_schema(db)
    require("NVTX_EVENTS" in schema, "NVTX absent")
    cols = schema["NVTX_EVENTS"]
    require({"start", "end", "globalTid"}.issubset(cols), "NVTX issuing-thread identity missing")
    require(item.get("coverage") == "exact", "inferred phase windows are not exact search evidence")
    text_cols = [c for c in ("text", "textId") if c in cols]
    require(bool(text_cols), "NVTX label missing")
    tid = int(item["global_tid"])
    require(tid // (1 << 24) == int(rank["global_pid"]) // (1 << 24), "NVTX/kernel PID mapping mismatch")
    matches = db.execute("SELECT " + ",".join(_quoted(c) for c in text_cols) +
                         " FROM NVTX_EVENTS WHERE start=? AND end=? AND globalTid=?",
                         (item["start_ns"], item["end_ns"], tid)).fetchall()
    names = _strings(db, {x for row in matches for x in row if isinstance(x, int)})
    require(any(item["nvtx_text"] == names.get(x, x) for row in matches for x in row),
            "phase bounds do not match an exact native NVTX range")
    return Interval(int(item["start_ns"]), int(item["end_ns"]))


def parse_profile(manifest: dict[str, Any]) -> dict[str, Any]:
    require(manifest.get("schema_version") == "distillschedule.search.profile_input.v1", "profile input schema mismatch")
    require(manifest.get("cycle") in (1, 2), "profile must identify the actual captured cycle")
    ranks = manifest.get("ranks", [])
    require(len(ranks) == 16 and {r["rank"] for r in ranks} == set(range(16)), "16-rank profile coverage required")
    require(len({(r["node"], r["device_id"]) for r in ranks}) == 16, "duplicate GPU/rank mapping")
    require({r["node"] for r in ranks} == {0, 1}, "two node profiles required")
    require(all({r["device_id"] for r in ranks if r["node"] == n} == set(range(8)) for n in (0, 1)), "8/8 GPUs per node required")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rank in ranks:
        groups[rank["sqlite"]].append(rank)
    output, sources = [], []
    timeout_seconds = manifest.get("sqlite_timeout_seconds", 300)
    require(type(timeout_seconds) is int and 1 <= timeout_seconds <= 900, "SQLite deadline must be 1..900 seconds per node")
    for path, specs in sorted(groups.items()):
        source = Path(path)
        require(source.is_file(), f"missing profile {source}")
        digest = file_sha256(source)
        require(all(r["sqlite_sha256"] == digest for r in specs), "SQLite source hash mismatch")
        require(all(r.get("parser_status") == "PASS" and not r.get("parser_errors") for r in specs), "Nsight processing failure")
        doctor = _doctor(source, specs)
        sources.append({"path": str(source), "sha256": digest, "nsys_ai_doctor": doctor})
        with sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True) as db:
            db.execute("PRAGMA query_only=ON")
            deadline = time.monotonic() + timeout_seconds
            db.set_progress_handler(lambda: int(time.monotonic() >= deadline), 10000)
            for rank in specs:
                require(time.monotonic() < deadline, "bounded SQLite analysis timed out; preserve input, do not reprofile automatically")
                capture = _validate_window(db, rank, rank["capture"])
                activities, quality = _activity(db, rank, capture)
                require(bool(rank.get("phases")), "phase observations missing")
                phase_windows: list[Interval] = []
                for phase in rank["phases"]:
                    window = _validate_window(db, rank, phase)
                    require(window.clip(capture) == window, "phase outside cycle2 capture")
                    require(all(window.clip(w) is None for w in phase_windows),
                            "search cost phases must be disjoint; nested NVTX would double-count work")
                    phase_windows.append(window)
                    measure = analyze_windows(activities, (window,))
                    compute = [a.interval for a in activities if a.category == "compute"]
                    nccl = [a.interval for a in activities if a.category == "nccl"]
                    c_ns = interval_union_ns(clip_intervals_to_windows(compute, (window,)))
                    n_ns = interval_union_ns(clip_intervals_to_windows(nccl, (window,)))
                    combined = interval_union_ns(clip_intervals_to_windows(compute + nccl, (window,)))
                    overlap = c_ns + n_ns - combined
                    phase_quality = {**quality, "pointwise_name_family_union_ns": interval_union_ns(
                        clip_intervals_to_windows((a.interval for a in activities if a.category == "compute"
                            and re.search(r"elementwise|vectorized|copy|cast|cat|norm|modulat|residual|gate", a.name, re.I)), (window,)))}
                    output.append({"rank": rank["rank"], "node": rank["node"], "device_id": rank["device_id"],
                        "phase": phase["name"], "role": phase["role"], "occurrence": phase.get("occurrence", 0),
                        "metrics": measure, "nccl_hidden_by_compute_ns": overlap,
                        "nccl_not_covered_by_compute_ns": n_ns - overlap,
                        "critical_path_wait": "UNRESOLVED_REQUIRES_DEPENDENCY_EVIDENCE", "quality": phase_quality})
    result = {"schema_version": PROFILE_SCHEMA, "status": "PASS", "cycle": manifest["cycle"],
              "identity": manifest["identity"], "hardware": manifest["hardware"],
              "sources": sources, "rows": output,
              "denominator": "PER_RANK_PHASE_WALL_NOT_GLOBAL_WALL",
              "claim_boundary": "DIAGNOSTIC_TEMPORAL_COVERAGE_NOT_MFU_OR_REMOVABLE_TIME"}
    validate_profile(result)
    return result


def validate_profile(value: dict[str, Any]) -> None:
    require(value.get("schema_version") == PROFILE_SCHEMA and value.get("status") == "PASS", "invalid profile")
    require(value.get("cycle") in (1, 2), "actual captured cycle missing")
    require(bool(value.get("sources")), "profile provenance missing")
    require({row["rank"] for row in value.get("rows", [])} == set(range(16)), "incomplete profile rank coverage")
    require(value["identity"]["hardware_fingerprint"] == fingerprint(value["hardware"]), "profile hardware fingerprint mismatch")
    for row in value["rows"]:
        metrics = row["metrics"]
        wall = finite_number(metrics["wall_ns"], "phase_wall", positive=True)
        cats = metrics["categories"]
        for item in cats.values():
            require(finite_number(item["union_ns"], "activity union") <= wall, "activity exceeds phase wall")
        require(cats["active"]["union_ns"] + cats["no_traced_gpu_activity"]["union_ns"] == wall,
                "active/complement reconciliation failed")


def feature_weights(profile: dict[str, Any]) -> dict[str, float]:
    """Rank GPU-seconds used only for ranking hints, never latency subtraction."""
    validate_profile(profile)
    weights: dict[str, float] = defaultdict(float)
    for row in profile["rows"]:
        metrics = row["metrics"]
        cats = metrics["categories"]
        weights["compute"] += cats["compute"]["union_ns"] / 1e9
        weights["nccl"] += row["nccl_not_covered_by_compute_ns"] / 1e9
        for category in ("h2d", "d2h"):
            weights[category] += cats[category]["union_ns"] / 1e9
        # This is uncertainty worth inspecting, not proof of CPU idle.
        weights["host_gap"] += cats["no_traced_gpu_activity"]["union_ns"] / 1e9
        weights["pointwise"] += row["quality"].get("pointwise_name_family_union_ns", 0) / 1e9
        if "optim" in row["phase"].lower() or "ema" in row["phase"].lower():
            weights["optimizer"] += metrics["wall_ns"] / 1e9
    return dict(weights)
