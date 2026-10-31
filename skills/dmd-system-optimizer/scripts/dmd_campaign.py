#!/usr/bin/env python3
"""Persistent experiment accounting and scoped result decisions; Python stdlib.

This helper records declared runner receipts. It neither launches nor isolates
GPU work and cannot independently prove that a checker or timing record is true.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
import sqlite3
import statistics
import sys

from dmd_perf import check_evidence, digest, load, number, require, text

LEVELS = {"V0", "V1", "V2", "V3"}
OUTCOMES = {"pass", "incorrect", "no_benefit", "inconclusive", "error", "interrupted"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def refs(items, base):
    errors = check_evidence(items, base, "receipt")
    require(not errors, "; ".join(errors))
    return [{**r, "path": str((base / r["path"]).resolve())} for r in items]


def integer(value, label, minimum=0):
    require(type(value) is int and value >= minimum, label + ": invalid integer")
    return value


@contextmanager
def connection(path, create=False):
    path = path.resolve()
    require(create or path.is_file(), "campaign missing; initialize explicitly")
    if create:
        path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=30, isolation_level=None)
    db.row_factory = sqlite3.Row
    try:
        db.execute("BEGIN IMMEDIATE")
        yield db
        db.commit()
    except BaseException:
        db.rollback()
        raise
    finally:
        db.close()


def get(db, key):
    row = db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    require(row is not None, "missing campaign " + key)
    return json.loads(row[0])


def set_value(db, key, value):
    db.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, canonical(value)))


def init(path, contract, base):
    c = json.loads(canonical(contract))
    require(c.get("schema_version") == 1, "contract schema_version must be 1")
    for key in ("campaign_id", "workload_id", "semantic_contract", "hardware_id", "timer_scope"):
        text(c.get(key), key)
    require(c.get("claim_class") in {"strict", "approximate", "diagnostic_review"}, "invalid claim_class")
    b = c["baseline"]
    for key in ("id", "source_identity"):
        text(b.get(key), "baseline." + key)
    number(b.get("seconds"), "baseline.seconds", positive=True)
    b["evidence"] = refs(b.get("evidence"), base)
    c["validators"] = refs(c.get("validators"), base)
    for key in ("max_gpu_hours", "max_full_runs"):
        number(c["budget"].get(key), key)
    integer(c["budget"]["max_full_runs"], "max_full_runs")
    routes = c["validation_by_risk"]
    require(isinstance(routes, dict) and bool(routes), "validation_by_risk required")
    for risk, levels in routes.items():
        text(risk, "risk")
        require(isinstance(levels, list) and set(levels) <= LEVELS and {"V0", "V1"} <= set(levels),
                "each risk route must include V0 and V1 and use only V0..V3")
    t = c["timing"]
    integer(t["repeat_count"], "repeat_count", 2)
    integer(t["max_candidates"], "max_candidates", 1)
    number(t["minimum_effect_seconds"], "minimum_effect_seconds")
    require(0 < number(t["alpha"], "alpha") <= .1, "alpha must be in (0,0.1]")
    require(2 ** -t["repeat_count"] <= t["alpha"] / t["max_candidates"],
            "repeat_count cannot resolve the declared multiplicity-adjusted alpha even with all wins")
    require(0 <= number(t["max_control_relative_drift"], "max_control_relative_drift") < 1, "bad drift limit")
    with connection(path, create=True) as db:
        db.execute("CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        db.execute("CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY, proposal TEXT NOT NULL, settlement TEXT)")
        db.execute("CREATE TABLE IF NOT EXISTS evaluations(candidate TEXT PRIMARY KEY, input_hash TEXT NOT NULL, result TEXT NOT NULL)")
        old = db.execute("SELECT value FROM meta WHERE key='contract'").fetchone()
        if old:
            require(old[0] == canonical(c), "campaign already exists with a different frozen contract")
        else:
            set_value(db, "contract", c)
            set_value(db, "control", b)
        return status(db)


def status(db):
    c = get(db, "contract")
    used = {"gpu_hours": 0., "full_runs": 0}
    held = {"gpu_hours": 0., "full_runs": 0}
    runs = []
    for row in db.execute("SELECT * FROM runs ORDER BY rowid"):
        proposal = json.loads(row["proposal"])
        settlement = json.loads(row["settlement"]) if row["settlement"] else None
        for key in used:
            (used if settlement else held)[key] += (settlement or proposal)[key]
        runs.append({"run_id": row["id"], "proposal": proposal, "settlement": settlement})
    budget = {**c["budget"], **{"used_" + k: v for k, v in used.items()},
              **{"reserved_" + k: v for k, v in held.items()}}
    budget["exceeded"] = any(used[k] + held[k] > c["budget"]["max_" + k] + 1e-9 for k in used)
    return {"contract_sha256": digest(c), "contract": c, "control": get(db, "control"),
            "budget": budget, "runs": runs, "gpu_enforcement": False}


def reserve(db, run_id, proposal):
    text(run_id, "run_id")
    p = json.loads(canonical(proposal))
    state = status(db)
    c = state["contract"]
    for key in ("candidate_id", "source_identity", "control_id", "risk"):
        text(p.get(key), key)
    require(p["control_id"] == state["control"]["id"], "stale control")
    require(p["risk"] in c["validation_by_risk"], "risk route was not frozen")
    require(p.get("purpose") in {"probe", "validation", "timing"}, "declare purpose")
    number(p.get("gpu_hours"), "gpu_hours")
    integer(p.get("full_runs"), "full_runs")
    old = db.execute("SELECT proposal FROM runs WHERE id=?", (run_id,)).fetchone()
    if old:
        require(old[0] == canonical(p), "run id already reserved differently")
        return state
    require(not db.execute("SELECT 1 FROM evaluations WHERE candidate=?", (p["candidate_id"],)).fetchone(),
            "candidate already assessed; a new experiment needs a new identity")
    same = [r for r in state["runs"] if r["proposal"]["candidate_id"] == p["candidate_id"]]
    for r in same:
        require(all(r["proposal"][k] == p[k] for k in ("source_identity", "control_id", "risk")),
                "candidate identity/risk cannot change within its registered experiment")
    if p["purpose"] == "timing":
        candidates = {r["proposal"]["candidate_id"] for r in state["runs"]
                      if r["proposal"]["purpose"] == "timing"} | {p["candidate_id"]}
        require(len(candidates) <= c["timing"]["max_candidates"], "candidate confirmation budget exhausted")
        require(sum(r["proposal"]["purpose"] == "timing" for r in same) < c["timing"]["repeat_count"],
                "fixed repeat count reached; no optional stopping")
    for k in ("gpu_hours", "full_runs"):
        require(state["budget"]["used_" + k] + state["budget"]["reserved_" + k] + p[k]
                <= c["budget"]["max_" + k] + 1e-9, k + " budget unavailable")
    db.execute("INSERT INTO runs VALUES (?,?,NULL)", (run_id, canonical(p)))
    return status(db)


def settle(db, run_id, settlement, base):
    row = db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
    require(row is not None, "run was not reserved")
    s = json.loads(canonical(settlement))
    require(s.get("outcome") in OUTCOMES, "invalid outcome")
    number(s.get("gpu_hours"), "actual gpu_hours")
    integer(s.get("full_runs"), "actual full_runs")
    s["evidence"] = refs(s.get("evidence"), base)
    if row["settlement"]:
        require(row["settlement"] == canonical(s), "run already settled differently")
    else:
        # Record overruns and failures honestly. Subsequent reservations will fail.
        db.execute("UPDATE runs SET settlement=? WHERE id=?", (canonical(s), run_id))
    return status(db)


def assess(db, result, base):
    state = status(db)
    c, control = state["contract"], state["control"]
    for key in ("workload_id", "semantic_contract", "hardware_id", "timer_scope", "claim_class"):
        require(result.get(key) == c[key], key + " differs from frozen contract")
    require(result.get("contract_sha256") == digest(c), "frozen contract hash mismatch")
    refs(c["baseline"]["evidence"], base)
    refs(c["validators"], base)
    refs(result.get("evidence"), base)
    candidate = text(result.get("candidate_id"), "candidate_id")
    source = text(result.get("source_identity"), "source_identity")
    same = [r for r in state["runs"] if r["proposal"]["candidate_id"] == candidate]
    require(bool(same), "candidate has no registered experiment")
    p = same[0]["proposal"]
    require(source == p["source_identity"] and result.get("control_id") == p["control_id"], "candidate/control identity changed")
    require(all(r["settlement"] for r in same), "unsettled experiments remain; reconcile before judging")
    for r in same:
        refs(r["settlement"]["evidence"], base)
    # Repeating a decision is idempotent even after promotion; receipts are still checked above.
    previous = db.execute("SELECT * FROM evaluations WHERE candidate=?", (candidate,)).fetchone()
    if previous:
        require(previous["input_hash"] == digest(result), "candidate already assessed on different evidence")
        return json.loads(previous["result"])
    require(p["control_id"] == control["id"], "control advanced; re-register the comparison")
    checks = result.get("checks")
    require(isinstance(checks, dict), "checks object required")
    required = c["validation_by_risk"][p["risk"]]
    failed = any(value == "FAIL" for value in checks.values()) or any(
        r["settlement"]["outcome"] == "incorrect" for r in same)
    correct = all(checks.get(level) == "PASS" for level in required) and not failed
    out = {"candidate_id": candidate, "required_validation": required, "claim_class": c["claim_class"],
           "validation_scope": text(result.get("validation_scope"), "validation_scope"),
           "retention": "DISCARD" if failed else "REVIEW", "default_enable": False,
           "performance": "NOT_EVALUATED", "exploration_priority": "SEPARATE_DECISION",
           "automatic_gpu_validation": False}
    if correct:
        require(result.get("unprofiled") is True and result.get("repeat_unit") == "independent_processes",
                "formal timing requires independent unprofiled processes")
        times = result.get("timings")
        require(isinstance(times, list) and len(times) == c["timing"]["repeat_count"],
                "provide exactly the frozen repeat count")
        timing_runs = {r["run_id"]: r for r in same if r["proposal"]["purpose"] == "timing"}
        require(len({t["run_id"] for t in times}) == len(times) and
                {t["run_id"] for t in times} == set(timing_runs), "include every registered timing run once")
        require(all(r["settlement"]["outcome"] == "pass" for r in timing_runs.values()),
                "failed/interrupted timing cannot be promoted")
        process_ids, savings, candidates = set(), [], []
        drift_ok = True
        for t in times:
            ids = t.get("process_ids")
            require(isinstance(ids, list) and len(ids) == 3 and all(isinstance(x, str) and x.strip() for x in ids),
                    "three process invocation ids required per control/candidate/control block")
            require(len(set(ids)) == 3 and not process_ids.intersection(ids), "reused timing process")
            process_ids.update(ids)
            before = number(t.get("control_before"), "control_before", positive=True)
            after = number(t.get("control_after"), "control_after", positive=True)
            candidate_time = number(t.get("candidate"), "candidate time", positive=True)
            drift_ok &= abs(before - after) / min(before, after) <= c["timing"]["max_control_relative_drift"]
            savings.append(min(before, after) - candidate_time)
            candidates.append(candidate_time)
        wins = sum(x > c["timing"]["minimum_effect_seconds"] for x in savings)
        n = len(savings)
        probability = sum(math.comb(n, k) for k in range(wins, n + 1)) / 2**n
        threshold = c["timing"]["alpha"] / c["timing"]["max_candidates"]
        benefit = drift_ok and probability <= threshold
        out.update(performance="BENEFIT" if benefit else "INVALID_DRIFT" if not drift_ok else "INCONCLUSIVE",
                   conservative_paired_savings=savings, candidate_median_seconds=statistics.median(candidates),
                   one_sided_sign_p=probability, adjusted_alpha=threshold,
                   inference="fixed-sample sign test of positive saving above the declared effect floor; not proof of long-run quality")
        if benefit:
            out["retention"] = "KEEP"
            out["default_enable"] = (result.get("compatibility") == "PASS" and result.get("fallback") == "PASS"
                                     and c["claim_class"] != "diagnostic_review")
        elif result.get("capacity_evidence"):
            refs(result["capacity_evidence"], base)
            out["retention"] = "KEEP_OPTIONAL"
    db.execute("INSERT INTO evaluations VALUES (?,?,?)", (candidate, digest(result), canonical(out)))
    return out


def promote(db, result, base):
    require(db.execute("SELECT 1 FROM evaluations WHERE candidate=?", (result.get("candidate_id"),)).fetchone(),
            "assess and persist the decision before promoting it")
    decision = assess(db, result, base)
    require(decision["default_enable"], "candidate is not eligible for default/control promotion")
    control = get(db, "control")
    if control["id"] == result["candidate_id"]:
        return {"status": "ALREADY_PROMOTED", "decision": decision}
    require(control["id"] == result["control_id"], "stale control; another result was promoted")
    set_value(db, "control", {"id": result["candidate_id"], "source_identity": result["source_identity"],
                             "seconds": decision["candidate_median_seconds"],
                             "evidence": refs(result["evidence"], base),
                             "claim_class": decision["claim_class"],
                             "validation_scope": decision["validation_scope"]})
    return {"status": "PROMOTED", "decision": decision, "original_baseline_unchanged": True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("init", "status", "reserve", "settle", "assess", "promote"):
        s = sub.add_parser(name)
        s.add_argument("--state", type=Path, required=True)
        if name != "status":
            s.add_argument("--input", type=Path, required=True)
        if name in {"reserve", "settle"}:
            s.add_argument("--run-id", required=True)
    a = p.parse_args()
    try:
        if a.command == "init":
            out = init(a.state, load(a.input), a.input.resolve().parent)
        else:
            with connection(a.state) as db:
                if a.command == "status":
                    out = status(db)
                elif a.command == "reserve":
                    out = reserve(db, a.run_id, load(a.input))
                else:
                    data, base = load(a.input), a.input.resolve().parent
                    out = (settle(db, a.run_id, data, base) if a.command == "settle"
                           else assess(db, data, base) if a.command == "assess" else promote(db, data, base))
        print(json.dumps(out, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except (ValueError, TypeError, KeyError, OSError, sqlite3.Error) as exc:
        print(json.dumps({"status": "INVALID", "error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
