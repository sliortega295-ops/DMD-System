#!/usr/bin/env python3
"""Persistent experiment accounting and scoped result decisions; Python stdlib.

This helper parses bound runner receipts. It neither launches nor isolates GPU
work and cannot independently prove runner ownership or scientific adequacy.
"""
from __future__ import annotations

import argparse
import hashlib
from contextlib import contextmanager
import json
import math
from pathlib import Path
import sqlite3
import statistics
import sys

from dmd_perf import check_evidence, digest, load, number, require, text
import dmd_receipts

LEVELS = {"V0", "V1", "V2", "V3"}
OUTCOMES = {"pass", "incorrect", "no_benefit", "inconclusive", "error", "interrupted"}
REPLACEABLE = {"launch_failure", "environment_interruption", "invalid_timer_window"}


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
        if db.execute("SELECT 1 FROM sqlite_master WHERE name='evaluations'").fetchone():
            revision_schema(db)
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


def revision_schema(db):
    db.execute("CREATE TABLE IF NOT EXISTS decisions(candidate TEXT, revision INTEGER, input_hash TEXT, "
               "input_json TEXT, result TEXT, PRIMARY KEY(candidate,revision))")
    db.execute("INSERT OR IGNORE INTO decisions SELECT candidate,1,input_hash,NULL,result FROM evaluations")
    db.execute("CREATE TABLE IF NOT EXISTS memory_outbox(id TEXT PRIMARY KEY, store TEXT, record TEXT, delivered INTEGER DEFAULT 0)")


def binding(proposal, run_id):
    return {"run_id": run_id, **{k: proposal[k] for k in dmd_receipts.BINDING if k != "run_id"}}


def invalid_sample(run):
    return (run.get("settlement") or {}).get("sample", {}).get("status") == "invalid"


def init(path, contract, base):
    c = json.loads(canonical(contract))
    require(c.get("schema_version") in {1, 2}, "contract schema_version must be 1 or 2")
    for key in ("campaign_id", "workload_id", "semantic_contract", "hardware_id", "timer_scope"):
        text(c.get(key), key)
    require(c.get("claim_class") in {"strict", "approximate", "diagnostic_review"}, "invalid claim_class")
    b = c["baseline"]
    for key in ("id", "source_identity"):
        text(b.get(key), "baseline." + key)
    if b.get("seconds") is not None or c.get("timing") is not None:
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
    if t is not None:
        integer(t["repeat_count"], "repeat_count", 2)
        integer(t["max_candidates"], "max_candidates", 1)
        number(t["minimum_effect_seconds"], "minimum_effect_seconds")
        require(0 < number(t["alpha"], "alpha") <= .1, "alpha must be in (0,0.1]")
        require(2 ** -t["repeat_count"] <= t["alpha"] / t["max_candidates"],
                "repeat_count cannot resolve the declared multiplicity-adjusted alpha even with all wins")
        require(0 <= number(t["max_control_relative_drift"], "max_control_relative_drift") < 1, "bad drift limit")
        integer(t.get("max_attempts", t["repeat_count"]), "max_attempts", t["repeat_count"])
        require(set(t.get("replaceable_failures", [])) <= REPLACEABLE, "invalid replacement policy")
    if c["schema_version"] == 2:
        policy = c["receipt_policy"]
        require(isinstance(policy.get("checks"), dict) and {"V0", "V1"} <= policy["checks"].keys(),
                "freeze checker selection for V0/V1")
        require(set().union(*map(set, routes.values())) <= policy["checks"].keys(), "risk route has no frozen checker selection")
        for spec in policy["checks"].values():
            text(spec.get("scope"), "checker scope")
            require(isinstance(spec.get("tests"), list) and spec["tests"]
                    and all(isinstance(n, str) and n.strip() for n in spec["tests"]), "freeze required test names")
        for ref in policy["checkers"].values():
            refs([ref], base)
            ref["path"] = str((base / ref["path"]).resolve())
            text(ref.get("version"), "checker version")
        require("junit" in policy["checkers"], "validation adapter required")
        require(c.get("measurement_kind") in {"observed", "synthetic"}, "declare measurement kind")
    with connection(path, create=True) as db:
        db.execute("CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        db.execute("CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY, proposal TEXT NOT NULL, settlement TEXT)")
        db.execute("CREATE TABLE IF NOT EXISTS evaluations(candidate TEXT PRIMARY KEY, input_hash TEXT NOT NULL, result TEXT NOT NULL)")
        revision_schema(db)
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
        runs.append({"run_id": row["id"], "attempt_id": row["id"], "proposal": proposal, "settlement": settlement})
    budget = {**c["budget"], **{"used_" + k: v for k, v in used.items()},
              **{"reserved_" + k: v for k, v in held.items()}}
    budget["exceeded"] = any(used[k] + held[k] > c["budget"]["max_" + k] + 1e-9 for k in used)
    return {"contract_sha256": digest(c), "contract": c, "control": get(db, "control"),
            "budget": budget, "runs": runs, "gpu_enforcement": False,
            "decisions": [json.loads(r[0]) for r in db.execute("SELECT result FROM decisions ORDER BY rowid")],
            "memory_pending": db.execute("SELECT count(*) FROM memory_outbox WHERE delivered=0").fetchone()[0]}


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
    previous = db.execute("SELECT result FROM evaluations WHERE candidate=?", (p["candidate_id"],)).fetchone()
    if previous:
        require(json.loads(previous[0])["retention"] != "DISCARD", "incorrect candidate is terminal; fix source under a new identity")
    same = [r for r in state["runs"] if r["proposal"]["candidate_id"] == p["candidate_id"]]
    for r in same:
        require(all(r["proposal"].get(k) == p.get(k) for k in
                    ("source_identity", "config_identity", "input_identity", "control_id", "risk")),
                "candidate identity/risk cannot change within its registered experiment")
    if c["schema_version"] == 2:
        binding(p, run_id)
        for key in dmd_receipts.BINDING:
            text(binding(p, run_id)[key], key)
        comparators = {b["id"]: b["source_identity"] for b in (c["baseline"], state["control"])}
        require(comparators.get(p["comparator_id"]) == p["comparator_source_identity"], "unknown comparator source")
        if p["purpose"] == "timing":
            require(p["comparator_id"] == p["control_id"], "timing must compare current control")
    if p["purpose"] == "timing":
        require(c["timing"] is not None, "speed confirmation disabled in this frozen contract")
        candidates = {r["proposal"]["candidate_id"] for r in state["runs"]
                      if r["proposal"]["purpose"] == "timing"} | {p["candidate_id"]}
        require(len(candidates) <= c["timing"]["max_candidates"], "candidate confirmation budget exhausted")
        attempts = [r for r in same if r["proposal"]["purpose"] == "timing"]
        require(sum(not invalid_sample(r) for r in attempts) < c["timing"]["repeat_count"],
                "fixed repeat count reached; no optional stopping")
        require(len(attempts) < c["timing"].get("max_attempts", c["timing"]["repeat_count"]), "timing attempt cap reached")
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
    c, p = get(db, "contract"), json.loads(row["proposal"])
    parsed = []
    for receipt in s.get("receipts", []):
        parsed.append(dmd_receipts.verify(receipt, c["receipt_policy"], binding(p, run_id), base))
    if p["purpose"] == "timing" and s.get("sample", {}).get("status") == "invalid":
        require(s["outcome"] in {"error", "interrupted"}, "only failed attempts can be excluded")
        require(s["sample"].get("reason") in c["timing"].get("replaceable_failures", []), "replacement reason not predeclared")
        require(any(r["format"] == "attempt-json" and r["sample"] == s["sample"] and r["outcome"] == s["outcome"]
                    for r in parsed), "invalid sample requires matching runner failure receipt")
    elif "sample" in s:
        require(s["sample"] == {"status": "valid"}, "invalid sample declaration")
    if c["schema_version"] == 2 and p["purpose"] == "timing" and s["outcome"] in {"pass", "no_benefit"}:
        require(sum(r["format"] == "aba-json" for r in parsed) == 1, "timing needs one parsed runner receipt")
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
    if c["schema_version"] == 2:
        require(result.get("validation_scope") == p["validation_scope"], "validation scope differs from reserved scope")
    require(all(r["settlement"] for r in same), "unsettled experiments remain; reconcile before judging")
    for r in same:
        refs(r["settlement"]["evidence"], base)
    # Repeating a decision is idempotent even after promotion; receipts are still checked above.
    parsed = []
    for r in same:
        for receipt in r["settlement"].get("receipts", []):
            parsed.append(dmd_receipts.verify(receipt, c["receipt_policy"], binding(r["proposal"], r["run_id"]), base))
    previous = db.execute("SELECT * FROM decisions WHERE candidate=? ORDER BY revision DESC LIMIT 1", (candidate,)).fetchone()
    identical = db.execute("SELECT result FROM decisions WHERE candidate=? AND input_hash=? ORDER BY revision DESC LIMIT 1", (candidate, digest(result))).fetchone()
    if identical and previous["input_json"] is None:
        return {**json.loads(identical[0]), "default_enable": False, "evidence_assurance": "LEGACY_DECLARATION_ONLY",
                "historical_decision_only": True}
    if identical and json.loads(identical[0]).get("attempt_ids") == [r["run_id"] for r in same]:
        return json.loads(identical[0])
    if previous:
        require(previous["input_json"] is not None, "legacy decision lacks saved inputs; cannot revise historical judgment")
        old = json.loads(previous["input_json"])
        require(json.loads(previous["result"])["retention"] != "DISCARD", "incorrect candidate is terminal")
        # Revisions add validation, never replace observed timings or completed checks.
        for key, value in old.items():
            if key == "checks":
                require(all(result.get(key, {}).get(k) in ({"PASS", "FAIL"} if v == "PASS" else {v})
                            for k, v in value.items() if v in {"PASS", "FAIL"}),
                        "completed checks are immutable")
            elif key in {"evidence", "capacity_evidence"}:
                require(all(v in result.get(key, []) for v in value), "old evidence cannot be removed")
            elif key in {"compatibility", "fallback", "composition"} and value in {"UNKNOWN", "NOT_RUN"}:
                continue
            elif key in {"compatibility", "fallback", "composition"} and value == "PASS" and result.get(key) == "FAIL":
                continue
            elif key == "timings":
                require(all(row in result.get("timings", []) for row in value),
                        "candidate already assessed on different evidence: recorded timings cannot be changed or removed")
            elif key in {"unprofiled", "repeat_unit"} and not old.get("timings"):
                continue
            elif key == "latency_tradeoff" and value is None:
                continue
            else:
                require(result.get(key) == value, "candidate already assessed on different evidence: " + key)
    require(p["control_id"] == control["id"], "control advanced; re-register the comparison")
    checks = result.get("checks")
    require(isinstance(checks, dict), "checks object required")
    require(all(v in {"PASS", "FAIL", "UNKNOWN", "NOT_RUN"} for v in checks.values()), "invalid declared check status")
    required = c["validation_by_risk"][p["risk"]]
    actual, baseline_checks = {}, {}
    for receipt in parsed:
        if receipt["format"] != "junit":
            continue
        for k, v in receipt["checks"].items():
            # Any failure remains a failure across attempts. NOT_RUN can later gain evidence.
            actual[k] = "FAIL" if "FAIL" in {actual.get(k), v} else "PASS" if "PASS" in {actual.get(k), v} else "NOT_RUN"
            if receipt["binding"]["comparator_id"] == c["baseline"]["id"]:
                baseline_checks[k] = "FAIL" if "FAIL" in {baseline_checks.get(k), v} else "PASS" if "PASS" in {baseline_checks.get(k), v} else "NOT_RUN"
    if c["schema_version"] == 2:
        for k, v in {**checks, **{k: result.get(k, "UNKNOWN") for k in ("compatibility", "fallback", "composition")}}.items():
            if v in {"PASS", "FAIL"}:
                require(actual.get(k) == v, "declared " + k + " conflicts with parsed checker body or is missing")
        checks = actual
    failed = any(value == "FAIL" for key, value in checks.items() if key not in
                 {"capacity", "compatibility", "fallback", "composition"}) or any(
        r["settlement"]["outcome"] == "incorrect" for r in same)
    correct = all(checks.get(level) == "PASS" for level in required) and not failed
    out = {"candidate_id": candidate, "required_validation": required, "claim_class": c["claim_class"],
           "validation_scope": text(result.get("validation_scope"), "validation_scope"),
           "retention": "DISCARD" if failed else "REVIEW", "default_enable": False,
           "performance": "NOT_EVALUATED", "exploration_priority": "SEPARATE_DECISION",
           "automatic_gpu_validation": False, "decision_revision": previous["revision"] + 1 if previous else 1,
           "attempt_ids": [r["run_id"] for r in same], "measurement_kind": c.get("measurement_kind", "declared_legacy"),
           "evidence_assurance": "PARSED_RUNNER_OUTPUT" if c["schema_version"] == 2 else "DECLARATION_ONLY",
           "original_baseline_validated": all(baseline_checks.get(k) == "PASS" for k in required)}
    out["validator_scopes"] = {k: v["scope"] for k, v in c.get("receipt_policy", {}).get("checks", {}).items()}
    if c["schema_version"] == 1:
        correct = False  # Historical declarations are readable, but cannot earn new KEEP/default decisions.
    times = result.get("timings", [])
    timing_runs = {r["run_id"]: r for r in same if r["proposal"]["purpose"] == "timing" and not invalid_sample(r)}
    if times:
        require(c["timing"] is not None, "speed confirmation disabled in this frozen contract")
        require(isinstance(times, list) and len(times) <= c["timing"]["repeat_count"], "timings exceed frozen repeat count")
        require(len({t["run_id"] for t in times}) == len(times) and {t["run_id"] for t in times} == set(timing_runs),
                "include every valid registered timing run once")
        parsed_times = {r["binding"]["run_id"]: r for r in parsed if r["format"] == "aba-json"}
        if c["schema_version"] == 2:
            require(all(t["run_id"] in parsed_times and parsed_times[t["run_id"]]["timing"] == t for t in times),
                    "timing declaration conflicts with parsed runner body")
            require(all(parsed_times[t["run_id"]]["measurement_kind"] == c["measurement_kind"] for t in times),
                    "synthetic/observed timing qualification differs")
    complete_timing = bool(times) and len(times) == c["timing"]["repeat_count"]
    out["confirmation"] = {"state": "DISABLED" if c["timing"] is None else
                           "COMPLETE" if complete_timing else "INCOMPLETE" if times else "NOT_EVALUATED",
                           "samples_recorded": len(times),
                           "samples_required": c["timing"]["repeat_count"] if c["timing"] else None}
    if correct and result.get("capacity_evidence"):
        refs(result["capacity_evidence"], base)
        out["capacity_benefit"] = "ESTABLISHED" if checks.get("capacity") == "PASS" else "NOT_ESTABLISHED"
        if checks.get("capacity") == "PASS":
            text(result.get("latency_tradeoff"), "capacity retention needs measured or explicitly unknown latency_tradeoff")
            out.update(retention="KEEP_OPTIONAL", latency_tradeoff=result["latency_tradeoff"])
    if times:
        require(result.get("unprofiled") is True and result.get("repeat_unit") == "independent_processes",
                "formal timing requires independent unprofiled processes")
        require(all(r["settlement"]["outcome"] in {"pass", "no_benefit"} for r in timing_runs.values()),
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
        benefit = False
        if correct and not complete_timing:
            out["performance"] = "INCOMPLETE"
        if correct and complete_timing:
            wins = sum(x > c["timing"]["minimum_effect_seconds"] for x in savings)
            n = len(savings)
            probability = sum(math.comb(n, k) for k in range(wins, n + 1)) / 2**n
            threshold = c["timing"]["alpha"] / c["timing"]["max_candidates"]
            benefit = drift_ok and probability <= threshold
            out.update(performance="BENEFIT" if benefit else "INVALID_DRIFT" if not drift_ok else "INCONCLUSIVE",
                       conservative_paired_savings=savings, candidate_median_seconds=statistics.median(candidates),
                       one_sided_sign_p=probability, adjusted_alpha=threshold,
                       inference="fixed-sample sign test of positive saving above the declared effect floor; not proof of long-run quality")
        if benefit and correct:
            out["retention"] = "KEEP"
            out["default_enable"] = (all(checks.get(k) == "PASS" for k in ("compatibility", "fallback", "composition"))
                                     and out["original_baseline_validated"]
                                     and c["claim_class"] != "diagnostic_review")
    out["needs_validation"] = [k for k in required if checks.get(k) != "PASS"]
    out["can_append_validation"] = out["retention"] != "DISCARD"
    db.execute("INSERT INTO decisions VALUES (?,?,?,?,?)", (candidate, out["decision_revision"], digest(result), canonical(result), canonical(out)))
    db.execute("INSERT OR REPLACE INTO evaluations VALUES (?,?,?)", (candidate, digest(result), canonical(out)))
    return out


def promote(db, result, base):
    require(db.execute("SELECT 1 FROM evaluations WHERE candidate=?", (result.get("candidate_id"),)).fetchone(),
            "assess and persist the decision before promoting it")
    decision = assess(db, result, base)
    latest = db.execute("SELECT input_hash FROM evaluations WHERE candidate=?", (result["candidate_id"],)).fetchone()
    require(latest[0] == digest(result), "only the latest decision can be promoted")
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


def finish(db, bundle, store, base):
    """Commit settlements, decision and a deterministic Memory outbox atomically."""
    for row in bundle.get("settlements", []):
        settle(db, row["run_id"], row["settlement"], base)
    result = bundle["result"]
    decision = assess(db, result, base)
    state = status(db)
    metadata = bundle["experience"]
    for key in ("summary", "pattern", "retry", "failure_scope"):
        text(metadata.get(key), "experience." + key)
    for key in ("conditions", "correctness_conditions", "performance_conditions"):
        require(isinstance(metadata.get(key), dict), "experience." + key + " required")
    evidence = refs(result["evidence"], base)
    for run in state["runs"]:
        if run["run_id"] in decision["attempt_ids"]:
            evidence.extend(run["settlement"]["evidence"])
            for receipt in run["settlement"].get("receipts", []):
                evidence.extend([receipt["raw"], receipt["checker"]])
    evidence = list({canonical(e): e for e in evidence}.values())
    ident = "campaign-" + digest({"contract": state["contract_sha256"], "candidate": result["candidate_id"],
                                  "revision": decision["decision_revision"]})[:32]
    outcome = {"KEEP": "success", "KEEP_OPTIONAL": "success", "DISCARD": "incorrect"}.get(decision["retention"],
              "inconclusive" if decision["performance"] in {"INCONCLUSIVE", "INVALID_DRIFT"} else "incomplete")
    record = {k: metadata[k] for k in ("summary", "pattern", "retry", "failure_scope", "conditions",
              "correctness_conditions", "performance_conditions")}
    record.update(id=ident, kind="experience", outcome=outcome,
                  strength="observation" if decision["evidence_assurance"] == "PARSED_RUNNER_OUTPUT" else "unverified",
                  evidence=evidence, decision=decision,
                  provenance={"campaign_id": state["contract"]["campaign_id"], "contract_sha256": state["contract_sha256"],
                              "candidate_id": result["candidate_id"], "source_identity": result["source_identity"],
                              "decision_revision": decision["decision_revision"], "measurement_kind": decision["measurement_kind"]})
    proposal = next(r["proposal"] for r in state["runs"] if r["proposal"]["candidate_id"] == result["candidate_id"])
    record["provenance"].update({k: proposal.get(k) for k in ("config_identity", "input_identity", "control_id")})
    record["conditions"] = {**record["conditions"], "measurement_kind": decision["measurement_kind"]}
    for key in ("source", "supersedes", "reuse_decision"):
        if key in metadata:
            record[key] = metadata[key]
    if "reuse_decision" in record:
        from dmd_reuse import validate
        reuse = record["reuse_decision"]
        validate(reuse, base)
        require(reuse["candidate_id"] == result["candidate_id"] and reuse["source_identity"] == result["source_identity"],
                "reuse decision identity differs from assessed candidate")
        if decision["retention"] in {"KEEP", "KEEP_OPTIONAL"}:
            require(reuse["status"] == "VERIFIED", "retained reuse needs execution and rollback evidence")
        for field in ("path_evidence", "numerical_evidence", "rollback_evidence"):
            if field in reuse:
                reuse[field] = refs(reuse[field], base)
    if "source" in record:
        record["source"]["path"] = str((base / record["source"]["path"]).resolve())
    target = str(store.resolve())
    # Connect successive judgments for this exact frozen candidate automatically.
    # This is an evidence update, not a claim that another workload is superseded.
    prior_records = [(json.loads(row["record"]), row["store"]) for row in db.execute("SELECT store,record FROM memory_outbox")]
    prior_records = [(r, s) for r, s in prior_records if r["provenance"]["contract_sha256"] == state["contract_sha256"]
                     and r["provenance"]["candidate_id"] == result["candidate_id"]
                     and r["provenance"]["decision_revision"] < decision["decision_revision"]]
    if prior_records:
        prior, prior_store = max(prior_records, key=lambda pair: pair[0]["provenance"]["decision_revision"])
        target_id = prior["id"] if prior_store == target else "private-" + hashlib.sha256(prior_store.encode()).hexdigest()[:12] + ":" + prior["id"]
        from dmd_knowledge import correction_edges
        record["supersedes"] = correction_edges(record) + [{"id": target_id, "kind": "decision_update", "conditions": {},
            "reason": "Additional evidence produced a later decision for the same frozen candidate; earlier judgment remains in history."}]
    old = db.execute("SELECT store,record FROM memory_outbox WHERE id=?", (ident,)).fetchone()
    if old:
        require(old["store"] == target and old["record"] == canonical(record), "decision already queued to Memory differently")
    else:
        db.execute("INSERT INTO memory_outbox(id,store,record) VALUES (?,?,?)", (ident, target, canonical(record)))
    return {"decision": decision, "memory_id": ident, "memory_status": "QUEUED"}


def flush_memory(path):
    """Crash between file publish and delivered flag is safe: put is idempotent."""
    from dmd_knowledge import put
    published = []
    with connection(path) as db:
        for row in db.execute("SELECT * FROM memory_outbox WHERE delivered=0 ORDER BY rowid").fetchall():
            published.append(put(Path(row["store"]), json.loads(row["record"]), path.resolve().parent))
            db.execute("UPDATE memory_outbox SET delivered=1 WHERE id=?", (row["id"],))
    return published


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("init", "status", "reserve", "settle", "assess", "promote", "finish", "flush-memory"):
        s = sub.add_parser(name)
        s.add_argument("--state", type=Path, required=True)
        if name not in {"status", "flush-memory"}:
            s.add_argument("--input", type=Path, required=True)
        if name in {"reserve", "settle"}:
            s.add_argument("--run-id", required=True)
        if name == "finish":
            s.add_argument("--store", type=Path, required=True)
    a = p.parse_args()
    try:
        if a.command == "init":
            out = init(a.state, load(a.input), a.input.resolve().parent)
        elif a.command == "flush-memory":
            out = {"memory": flush_memory(a.state)}
        else:
            with connection(a.state) as db:
                if a.command == "status":
                    out = status(db)
                elif a.command == "reserve":
                    out = reserve(db, a.run_id, load(a.input))
                else:
                    data, base = load(a.input), a.input.resolve().parent
                    out = (settle(db, a.run_id, data, base) if a.command == "settle"
                           else finish(db, data, a.store, base) if a.command == "finish"
                           else assess(db, data, base) if a.command == "assess" else promote(db, data, base))
            if a.command == "finish":
                out.update(memory=flush_memory(a.state), memory_status="PUBLISHED")
        print(json.dumps(out, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except (ValueError, TypeError, KeyError, OSError, sqlite3.Error) as exc:
        print(json.dumps({"status": "INVALID", "error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
