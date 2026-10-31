#!/usr/bin/env python3
"""CPU-only DMD optimization arithmetic and evidence gate. Never launches workloads.

Stdlib only, Python >=3.10. No network, package install, CUDA import, repo writes,
remote mutation or shell execution. 'inspect' runs read-only git commands.
The gate checks declared contracts and artifacts; it cannot certify their truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

MAX_JSON = 8 * 1024 * 1024
MAX_EVIDENCE = 32 * 1024 * 1024  # Point at small reports, not multi-GB raw traces.
LAYERS = {"M", "L", "Q", "K"}
ACCEPTED_CONTROL = "accepted_unprofiled"


class InputError(ValueError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise InputError(message)


def number(value: Any, name: str, *, minimum: float = 0, positive: bool = False) -> float:
    require(type(value) in (int, float), f"{name}: finite number required (not bool/null)")
    value = float(value)
    require(math.isfinite(value), f"{name}: non-finite number")
    require(value > minimum if positive else value >= minimum, f"{name}: below allowed range")
    return value


def text(value: Any, name: str) -> str:
    require(isinstance(value, str) and bool(value.strip()), f"{name}: nonblank text required")
    return value


def unique_pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load(path: Path) -> Any:
    require(path.is_file(), f"not a regular file: {path}")
    require(path.stat().st_size <= MAX_JSON, f"JSON exceeds {MAX_JSON} bytes")
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_pairs,
                       parse_constant=lambda x: (_ for _ in ()).throw(InputError(f"invalid constant: {x}")))
    # Exponent overflow such as 1e999 is accepted by Python's parser without this check.
    def check(x: Any) -> None:
        if isinstance(x, float):
            require(math.isfinite(x), "non-finite JSON number")
        elif isinstance(x, dict):
            for v in x.values():
                check(v)
        elif isinstance(x, list):
            for v in x:
                check(v)
    check(value)
    require(isinstance(value, dict), "top-level JSON must be an object")
    return value


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def sha_file(path: Path, limit: int = MAX_EVIDENCE) -> str:
    require(path.is_file(), f"not a regular evidence file: {path}")
    require(path.stat().st_size <= limit, f"use a small evidence report, not raw large data: {path}")
    with path.open("rb") as f:
        h = hashlib.sha256()
        size = 0
        for block in iter(lambda: f.read(1024 * 1024), b""):
            size += len(block)
            require(size <= limit, f"evidence grew beyond limit: {path}")
            h.update(block)
    return h.hexdigest()


def target(baseline: Any, control: Any, speedup: Any) -> dict[str, float]:
    b = number(baseline, "baseline_seconds", positive=True)
    c = number(control, "control_seconds", positive=True)
    s = number(speedup, "target_speedup", minimum=1)
    t = b / s
    return {"baseline_seconds": b, "control_seconds": c, "target_seconds": t,
            "target_speedup": s, "achieved_speedup": b / c,
            "remaining_saving_seconds": max(0, c - t),
            "required_speedup_from_control": c / t,
            "target_latency_reduction_percent": 100 * (1 - 1 / s)}


def interval(value: Any, name: str) -> tuple[float, float, float]:
    require(isinstance(value, dict) and set(value) == {"low", "central", "high"},
            f"{name}: exactly low/central/high required")
    a, b, c = (number(value[k], f"{name}.{k}") for k in ("low", "central", "high"))
    require(a <= b <= c, f"{name}: expected low <= central <= high")
    return a, b, c


def estimate(op: dict[str, Any], control_seconds: float) -> dict[str, Any] | None:
    e = op.get("estimate")
    if e is None:
        return None
    require(isinstance(e, dict), "estimate must be object or null")
    kind = e.get("kind")
    require(kind in {"serial_amdahl", "e2e_counterfactual", "probe_extrapolation", "upper_bound_only"},
            "unknown estimate.kind")
    text(e.get("basis"), "estimate.basis")
    if kind == "serial_amdahl":
        text(e.get("path_assumptions"), "serial_amdahl.path_assumptions")
        t = number(e.get("critical_component_seconds"), "critical_component_seconds")
        require(t <= control_seconds, "component duration exceeds control wall time")
        factors = interval(e.get("local_speedup"), "local_speedup")
        require(factors[0] >= 1, "local speedup must be >=1; express added cost in overhead")
        gross = tuple(t * (1 - 1 / s) for s in factors)
        cap = t
    else:
        gross = interval(e.get("gross_saving_seconds"), "gross_saving_seconds")
        cap = e.get("conditional_saving_cap_seconds")
        if cap is not None:
            cap = number(cap, "conditional_saving_cap_seconds")
            require(cap <= control_seconds, "saving cap exceeds control wall time")
    require(gross[2] <= control_seconds, "gross savings exceed control wall time")
    if cap is not None:
        require(gross[2] <= cap + 1e-9, "estimate exceeds its declared saving cap")
    overhead = interval(e.get("added_exposed_cost_seconds"), "added_exposed_cost_seconds")
    net = (gross[0] - overhead[2], gross[1] - overhead[1], gross[2] - overhead[0])
    require(net[2] < control_seconds, "predicted runtime must be positive")
    return {"kind": kind, "net_saving_seconds": dict(zip(("low", "central", "high"), net)),
            "predicted_runtime_seconds": {"low": control_seconds-net[2],
                                          "central": control_seconds-net[1],
                                          "high": control_seconds-net[0]},
            "conditional_zero_cost_cap_seconds": cap,
            "estimate_not_measurement": True}


def check_evidence(items: Any, base: Path, label: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(items, list) or not items:
        return [f"{label}: evidence required"]
    for i, item in enumerate(items):
        try:
            require(isinstance(item, dict), "evidence row must be object")
            path = Path(text(item.get("path"), "evidence.path"))
            if not path.is_absolute():
                path = base / path
            expected = text(item.get("sha256"), "evidence.sha256").lower()
            require(len(expected) == 64 and all(x in "0123456789abcdef" for x in expected),
                    "invalid sha256")
            text(item.get("locator"), "evidence.locator")
            require(sha_file(path) == expected, "evidence digest mismatch")
        except (InputError, OSError) as exc:
            errors.append(f"{label}[{i}]: {exc}")
    return errors


def campaign_target(c: dict[str, Any]) -> dict[str, float]:
    require(isinstance(c, dict), "campaign must be an object")
    for field in ("baseline", "control", "target"):
        require(isinstance(c.get(field), dict), f"campaign.{field} must be an object")
    require(c.get("schema_version") == 1, "campaign.schema_version must be 1")
    require(c.get("search_mode") in {"target-seeking", "incremental-best"}, "bad search_mode")
    text(c.get("analysis_id"), "campaign.analysis_id")
    return target(c["baseline"]["seconds"], c["control"]["seconds"], c["target"]["speedup"])


def rank(c: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    t = campaign_target(c)
    require(data.get("schema_version") == 1, "opportunities.schema_version must be 1")
    require(isinstance(data, dict), "opportunities document must be an object")
    ops = data.get("opportunities")
    require(isinstance(ops, list), "opportunities must be an array")
    require(all(isinstance(o, dict) for o in ops), "every opportunity must be an object")
    ids = [text(o.get("id"), "opportunity.id") for o in ops]
    require(len(ids) == len(set(ids)), "duplicate opportunity id")
    rows = []
    for op in ops:
        require(op.get("layer") in LAYERS, f"{op['id']}: unknown layer")
        require(op.get("kind", "ordinary") in {"ordinary", "enabler"}, "bad opportunity.kind")
        require(isinstance(op.get("enables", []), list) and all(isinstance(x, str) for x in op.get("enables", [])), "enables must be a string array")
        for dep in op.get("enables", []):
            require(dep in ids and dep != op["id"], f"{op['id']}: invalid enables target")
        e = estimate(op, t["control_seconds"])
        cost = op.get("cost") or {}
        require(isinstance(cost, dict), "opportunity.cost must be an object")
        eng = cost.get("engineer_hours")
        if eng is not None:
            eng = number(eng, "engineer_hours", positive=True)
        for field in ("gpu_hours", "full_runs"):
            if cost.get(field) is not None:
                number(cost[field], field)
        if cost.get("full_runs") is not None:
            require(type(cost["full_runs"]) is int, "full_runs must be integer")
        stale = op.get("analysis_id") != c["analysis_id"] or op.get("control_id") != c["control"]["id"]
        rows.append({"id": op["id"], "layer": op["layer"], "kind": op.get("kind", "ordinary"),
                     "stale": stale, "estimate": e, "cost": cost,
                     "saving_seconds_per_engineer_hour": None if eng is None or e is None else
                     e["net_saving_seconds"]["central"] / eng})
    # Detect enabler cycles: they cannot justify one another forever.
    graph = {o["id"]: o.get("enables", []) for o in ops}
    def visit(node: str, active: set[str], done: set[str]) -> None:
        require(node not in active, "enabler dependency cycle")
        if node in done:
            return
        active.add(node)
        for nxt in graph[node]:
            visit(nxt, active, done)
        active.remove(node)
        done.add(node)
    done: set[str] = set()
    for node in graph:
        visit(node, set(), done)
    ranked = sorted((r for r in rows if r["estimate"] is not None),
                    key=lambda r: r["estimate"]["net_saving_seconds"]["central"], reverse=True)
    unknown = [r for r in rows if r["estimate"] is None]
    return {"target": t, "estimated_candidates": ranked, "unknown_candidates": unknown,
            "warning": "Ranking is by declared central E2E savings, not a proof or automatic admission. "
                       "Unknown high-leverage routes need discriminating probes; gains are NOT additive."}


def gate(c: dict[str, Any], data: dict[str, Any], candidate_id: str,
         campaign_dir: Path, opportunities_dir: Path) -> dict[str, Any]:
    ranking = rank(c, data)
    matches = [o for o in data["opportunities"] if o["id"] == candidate_id]
    require(bool(matches), f"candidate not found: {candidate_id}")
    op = matches[0]
    row = next(r for r in ranking["estimated_candidates"] + ranking["unknown_candidates"] if r["id"] == candidate_id)
    invalid: list[str] = []
    if c.get("claim_class") not in {"strict", "approximate", "diagnostic_review"}:
        invalid.append("declare claim_class: strict / approximate / diagnostic_review")
    for obj, keys in ((c, ("workload_id", "semantic_contract", "timer_scope", "hardware_id")),
                      (c["baseline"], ("id", "source_identity")),
                      (c["control"], ("id", "source_identity")),
                      (op, ("mechanism", "semantic_risks", "selection_reason"))):
        for key in keys:
            try:
                text(obj.get(key), key)
            except InputError as exc:
                invalid.append(str(exc))
    if row["stale"]:
        invalid.append("candidate analysis/control identity is stale")
    for name in ("baseline", "control"):
        for key in ("timer_scope", "semantic_contract", "hardware_id", "claim_class"):
            if c[name].get(key) != c.get(key):
                invalid.append(f"{name}.{key} differs from the comparison contract")
        if c[name].get("evidence_grade") != ACCEPTED_CONTROL:
            invalid.append(f"{name} is not accepted_unprofiled; historical/synthetic/provisional is not admission")
        invalid.extend(check_evidence(c[name].get("evidence"), campaign_dir, name))
    invalid.extend(check_evidence(op.get("evidence"), opportunities_dir, "candidate"))
    if op.get("kind") == "enabler" and op.get("enabler_probe_passed") is True:
        invalid.extend(check_evidence(op.get("enabler_probe_evidence"), opportunities_dir, "enabler_probe"))
    out = {"candidate": candidate_id, "claim_class": c.get("claim_class"), "target": ranking["target"], "estimate": row["estimate"],
           "permission": "This result grants no new execution, GPU, semantic-change or publication permission."}
    if invalid:
        return {**out, "status": "INVALID", "reasons": invalid}
    if c.get("search_mode") == "target-seeking" and ranking["target"]["remaining_saving_seconds"] == 0:
        return {**out, "status": "DEFER", "reasons": ["declared target already met; validate goal before expanding work"]}
    probe: list[str] = []
    e = row["estimate"]
    if e is None or e["kind"] == "upper_bound_only":
        probe.append("a bound or unknown is not a realistic E2E forecast")
    for key in ("question", "stop_rule"):
        if not (op.get("probe") or {}).get(key):
            probe.append(f"bounded probe.{key} missing")
    for dimension in ("M", "L", "Q", "K"):
        survey = (c.get("survey") or {}).get(dimension, {})
        if survey.get("status") not in {"checked", "ruled_out"} or not survey.get("evidence_note"):
            probe.append(f"{dimension} scan incomplete: investigate rather than start ordinary optimization")
    if op.get("kind") == "enabler":
        if (op.get("estimate") or {}).get("scope") != "enabled_portfolio":
            probe.append("enabler needs a joint enabled_portfolio forecast, including its own cost; direct kernel gain is not enough")
        if not op.get("enables") or not op.get("capacity_argument"):
            probe.append("enabler lacks downstream opportunity and capacity/dependency argument")
        # Cheap feasibility first; a memory claim is not proof of the enabled schedule.
        if op.get("enabler_probe_passed") is not True:
            probe.append("enabler requires a bounded capacity/dependency probe first")
    budget = c.get("budget") or {}
    cost = row["cost"]
    for suffix in ("gpu_hours", "full_runs"):
        maximum = budget.get("max_" + suffix)
        used = budget.get("used_" + suffix)
        needed = cost.get(suffix)
        if maximum is None or used is None or needed is None:
            probe.append(f"{suffix} budget/cost unknown; no expensive launch")
            continue
        maximum = number(maximum, "max_" + suffix)
        used = number(used, "used_" + suffix)
        needed = number(needed, "cost." + suffix)
        if suffix == "full_runs":
            require(all(type(x) is int for x in (budget['max_full_runs'], budget['used_full_runs'], cost['full_runs'])),
                    "full-run budgets must be integers")
        if used + needed > maximum + 1e-9:
            return {**out, "status": "DEFER", "reasons": [f"{suffix} budget exhausted or candidate exceeds remaining budget"]}
    if cost.get("engineer_hours") is None:
        probe.append("engineering cost unknown")
    if probe:
        return {**out, "status": "PROBE_ONLY", "reasons": probe}
    assert e is not None
    net = e["net_saving_seconds"]
    if net["high"] <= 0:
        return {**out, "status": "DEFER", "reasons": ["even optimistic net E2E saving is nonpositive"]}
    policy = c.get("policy") or {}
    noise = number(policy.get("noise_floor_seconds", 0), "noise_floor_seconds")
    fraction = number(policy.get("small_gain_fraction_of_gap", .1), "small_gain_fraction_of_gap")
    require(fraction <= 1, "small_gain_fraction_of_gap must be <=1")
    threshold = max(noise, ranking["target"]["remaining_saving_seconds"] * fraction)
    alternatives = [r for r in ranking["estimated_candidates"] if r["id"] != candidate_id
                    and not r["stale"] and r["estimate"]["kind"] != "upper_bound_only"
                    and r["estimate"]["net_saving_seconds"]["central"] >= threshold]
    if net["high"] <= noise:
        return {**out, "status": "DEFER", "reasons": ["optimistic saving does not exceed the declared noise floor"]}
    unknown = [r["id"] for r in ranking["unknown_candidates"]
               if r["id"] != candidate_id and not r["stale"]]
    if op.get("kind", "ordinary") == "ordinary" and net["high"] < threshold and unknown:
        return {**out, "status": "PROBE_ONLY", "reasons": [
            "resolve unknown alternatives before prioritizing a small standalone experiment; "
            "this does not reject an already measured small gain"],
            "alternatives_to_probe": unknown}
    if op.get("kind", "ordinary") == "ordinary" and net["high"] < threshold and alternatives:
        return {**out, "status": "DEFER", "reasons": ["small standalone gain while material alternatives remain"],
                "alternatives_to_verify": [a["id"] for a in alternatives], "materiality_threshold_seconds": threshold}
    return {**out, "status": "ADMIT_BOUNDED_EXPERIMENT",
            "reasons": ["declared records pass arithmetic, artifact and budget checks; verify causal assumptions independently"],
            "selected_cost": cost, "materiality_threshold_seconds": threshold}


def git_read(repo: Path, *args: str) -> str:
    p = subprocess.run(["git", "--no-optional-locks", "-C", str(repo), *args], check=False, capture_output=True,
                       text=True, timeout=20)
    require(p.returncode == 0, f"read-only git command failed: {' '.join(args)}: {p.stderr.strip()}")
    return p.stdout.strip()


def bind_ledger(c: dict[str, Any], path: Path) -> dict[str, Any]:
    """Use persisted consumed AND reserved costs, never a reset JSON counter."""
    from dmd_campaign import connection, status
    with connection(path) as db:
        state = status(db)
    c = json.loads(json.dumps(c))
    for key in ("campaign_id", "workload_id", "semantic_contract", "hardware_id", "timer_scope", "claim_class"):
        require(c.get(key) == state["contract"].get(key), key + " differs from ledger")
    for name in ("baseline", "control"):
        frozen = state["contract"]["baseline"] if name == "baseline" else state["control"]
        for key in ("id", "source_identity", "seconds"):
            require(c[name].get(key) == frozen.get(key), name + "." + key + " differs from ledger")
    budget = state["budget"]
    c["budget"] = {"max_" + k: budget["max_" + k] for k in ("gpu_hours", "full_runs")}
    c["budget"].update({"used_" + k: budget["used_" + k] + budget["reserved_" + k]
                        for k in ("gpu_hours", "full_runs")})
    return c


def inspect(repo: Path) -> dict[str, Any]:
    repo = repo.resolve()
    head = git_read(repo, "rev-parse", "HEAD")
    require(Path(git_read(repo, "rev-parse", "--show-toplevel")).resolve() == repo,
            "--repo must be the Git checkout root")
    status = git_read(repo, "status", "--porcelain", "--untracked-files=normal")
    folder = repo / "configs/distillschedule/best_known"
    native = (repo / "src/distillschedule").is_dir()
    locks = load(folder / "registry.lock.json").get("profiles", {}) if native and (folder / "registry.lock.json").is_file() else {}
    require(isinstance(locks, dict), "registry profiles must be an object")
    profiles = []
    for name, expected in sorted(locks.items()):
        require(isinstance(name, str) and name.replace('_', '').isalnum(), "unsafe profile id")
        path = folder / f"{name}.json"
        value = load(path)
        profiles.append({"id": name, "lock_matches": digest(value) == expected,
                         "semantics": value.get("semantics"), "allowed_run_modes": value.get("allowed_run_modes"),
                         "contract": value.get("contract"), "hardware": value.get("hardware"),
                         "topology": value.get("topology"), "memory": value.get("memory"),
                         "launcher": value.get("runtime", {}).get("launcher"),
                         "validation": value.get("validation"),
                         "historical_is_current_measurement": False})
    anchors = ["AGENTS.md", "CONTRIBUTING.md", "pyproject.toml", "docs/guides/support_matrix.md",
               "src/distillschedule/workloads/catalog.json", "docs/workloads/BEST_KNOWN_WORKLOADS.md",
               "src/distillschedule/workloads/best_known/registry.py",
               "docs/runbooks/nsight_activity_overlay.md"]
    return {"repo": str(repo), "head": head, "git_status": status, "clean_status": not status,
            "mode": "dmd-system" if native else "external-source",
            "anchors": {p: sha_file(repo/p) if (repo/p).is_file() else None for p in anchors},
            "profiles": profiles, "gpu_validation_performed": False,
            "warning": "Read-only metadata inspection, not numerical validation. Re-read active runbooks. "
                       "Dirty/untracked source and external assets require explicit experiment identities."}


def dag_model(data: dict[str, Any]) -> dict[str, Any]:
    nodes = data.get("nodes")
    require(isinstance(nodes, list) and nodes, "DAG requires nonempty nodes")
    nd: dict[str, Any] = {}
    require(all(isinstance(n, dict) for n in nodes), "every node must be an object")
    for n in nodes:
        key = text(n.get("id"), "node.id")
        require(key not in nd, "duplicate node id")
        number(n.get("seconds"), f"{key}.seconds")
        require(isinstance(n.get("depends_on", []), list) and all(isinstance(d, str) for d in n.get("depends_on", [])), "depends_on must be a string array")
        nd[key] = n
    changes = data.get("candidate_seconds", {})
    require(isinstance(changes, dict) and set(changes) <= set(nd), "unknown candidate node")
    for key, value in changes.items():
        number(value, f"candidate.{key}")
    def solve(overrides: dict[str, Any]) -> tuple[float, list[str]]:
        memo: dict[str, tuple[float, list[str]]] = {}
        visiting: set[str] = set()
        def end(key: str) -> tuple[float, list[str]]:
            require(key in nd, f"unknown dependency: {key}")
            require(key not in visiting, "cyclic DAG")
            if key in memo:
                return memo[key]
            visiting.add(key)
            choices = [end(d) for d in nd[key].get("depends_on", [])]
            prior, path = max(choices, key=lambda x: x[0]) if choices else (0., [])
            result = (prior + overrides.get(key, nd[key]["seconds"]), path + [key])
            visiting.remove(key)
            memo[key] = result
            return result
        return max((end(k) for k in nd), key=lambda x: x[0])
    b, bp = solve({})
    c, cp = solve(changes)
    return {"base_model_seconds": b, "candidate_model_seconds": c, "model_saving_seconds": b-c,
            "base_critical_path": bp, "candidate_critical_path": cp,
            "assumptions": data.get("assumptions", "not supplied"),
            "classification": "user-supplied fixed-order DAG calculation; NOT measurement or an automatic physical bound",
            "warning": "Capacity, contention, dynamic arrivals and unlisted resource serialization are NOT modeled."}


def compare(data: dict[str, Any]) -> dict[str, Any]:
    for key in ("timer_scope", "semantic_contract", "hardware_id"):
        text(data.get(key), key)
    require(data.get("unprofiled") is True, "comparison input must explicitly state unprofiled")
    a, b = data.get("control_seconds"), data.get("candidate_seconds")
    require(isinstance(a, list) and isinstance(b, list) and a and b, "nonempty independent-run arrays required")
    a = [number(x, "control sample", positive=True) for x in a]
    b = [number(x, "candidate sample", positive=True) for x in b]
    ma, mb = statistics.median(a), statistics.median(b)
    result = {"control_n": len(a), "candidate_n": len(b), "control_median_seconds": ma,
              "candidate_median_seconds": mb, "median_ratio_speedup": ma/mb,
              "latency_reduction_percent": 100*(1-mb/ma),
              "control_range_seconds": [min(a), max(a)], "candidate_range_seconds": [min(b), max(b)],
              "formal_acceptance": "NOT_DECIDED",
              "warning": "Descriptive statistics only. No significance test or numerical acceptance. "
                         "Inputs must be independent complete runs, not correlated steps or cherry-picked timings."}
    if data.get("paired") is True:
        require(len(a) == len(b), "paired sample counts differ")
        result["paired_saving_seconds"] = [x-y for x, y in zip(a, b)]
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("inspect", help="read local Git/source metadata; no GPU or network")
    s.add_argument("--repo", type=Path, required=True)
    s = sub.add_parser("target", help="normalize target and calculate remaining gap")
    s.add_argument("--baseline-seconds", type=float, required=True)
    s.add_argument("--control-seconds", type=float)
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--speedup", type=float)
    g.add_argument("--latency-reduction", type=float, help="fraction, e.g. 0.5 means 2x")
    for command in ("rank", "gate"):
        s = sub.add_parser(command)
        s.add_argument("--campaign", type=Path, required=True)
        s.add_argument("--opportunities", type=Path, required=True)
        s.add_argument("--ledger", type=Path, help="bind identities and use persistent consumed/reserved budget")
        if command == "gate":
            s.add_argument("--candidate", required=True)
    for command in ("dag", "compare"):
        sub.add_parser(command).add_argument("--input", type=Path, required=True)
    a = p.parse_args()
    try:
        code = 0
        if a.command == "inspect":
            out = inspect(a.repo)
        elif a.command == "target":
            speedup = a.speedup
            if a.latency_reduction is not None:
                r = number(a.latency_reduction, "latency_reduction")
                require(r < 1, "latency_reduction must be in [0,1)")
                speedup = 1/(1-r)
            out = target(a.baseline_seconds, a.control_seconds if a.control_seconds is not None else a.baseline_seconds, speedup)
        elif a.command in ("rank", "gate"):
            c, ops = load(a.campaign), load(a.opportunities)
            if a.ledger:
                c = bind_ledger(c, a.ledger)
            out = rank(c, ops) if a.command == "rank" else gate(c, ops, a.candidate,
                       a.campaign.resolve().parent, a.opportunities.resolve().parent)
            out["budget_source"] = "persistent_ledger" if a.ledger else "declared_json_only"
            if a.command == "gate":
                code = {"ADMIT_BOUNDED_EXPERIMENT": 0, "INVALID": 2, "PROBE_ONLY": 3, "DEFER": 4}[out["status"]]
        elif a.command == "dag":
            out = dag_model(load(a.input))
        else:
            out = compare(load(a.input))
        print(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False))
        return code
    except (InputError, ValueError, TypeError, KeyError, AttributeError, OSError, OverflowError, RecursionError, subprocess.SubprocessError, sqlite3.Error) as exc:
        print(json.dumps({"status": "INVALID", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
