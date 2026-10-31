"""Private local records and correction history. No CLI, network or model imports."""
from __future__ import annotations

from collections import deque
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import warnings as runtime_warnings

from dmd_conditions import condition_match, normalize_context
from dmd_perf import InputError, check_evidence, load, require, sha_file, text

OUTCOMES = {"success", "no_benefit", "inconclusive", "incorrect", "environment",
            "implementation_failure", "incomplete", "unverified"}


def qualified_id(store, ident):
    return "private-" + hashlib.sha256(str(Path(store).resolve()).encode()).hexdigest()[:12] + ":" + ident


def decision_view(value):
    """Native and legacy labels are display data, not parsed checker decisions."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        return {"retention": value, "evidence_assurance": "DECLARATION_ONLY"}
    return {} if value is None else {"evidence_assurance": "MALFORMED_LEGACY_DECISION", "raw": value}


def _validate_reuse(record):
    reuse = record.get("reuse_decision")
    require(reuse is None or isinstance(reuse, dict), "reuse_decision must be an object")
    if reuse and reuse.get("implementation_id") is not None:
        text(reuse["implementation_id"], "reuse_decision.implementation_id")


def put(store: Path, record: dict, base: Path) -> dict:
    """Immutable, idempotent records; a correction gets a new id and supersedes."""
    record = json.loads(json.dumps(record, allow_nan=False))
    ident = text(record.get("id"), "id")
    require(bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}", ident)), "unsafe record id")
    require(record.get("kind") in {"experience", "implementation", "history"}, "invalid kind")
    text(record.get("summary"), "summary")
    require(record.get("outcome") in OUTCOMES, "invalid outcome")
    require(record.get("strength") in {"hypothesis", "observation", "mechanism", "unverified"},
            "invalid strength")
    require(isinstance(record.get("conditions"), dict), "conditions object required")
    require(record.get("decision") is None or isinstance(record["decision"], (dict, str)),
            "decision must be an object or a native decision label string")
    if "pattern" in record:
        text(record["pattern"], "pattern")
    for key in ("conditions", "correctness_conditions", "performance_conditions"):
        if key in record:
            require(isinstance(record[key], dict), key + " must be an object")
            record[key] = normalize_context(record[key])
    _validate_reuse(record)
    edges = correction_edges(record)  # Validate before publishing an immutable record.
    require(all(e["id"] not in {ident, qualified_id(store, ident)} for e in edges),
            "a record cannot supersede itself")
    require(isinstance(record.get("provenance"), dict) and bool(record["provenance"]), "provenance required")
    refs = record.get("evidence", [])
    require(isinstance(refs, list), "evidence must be an array")
    for row in refs:
        require(isinstance(row, dict), "invalid evidence row")
        row["path"] = str((base / Path(text(row.get("path"), "evidence.path"))).resolve())
    if record.get("source"):
        record["source"]["path"] = str((base / Path(record["source"]["path"])).resolve())
    if record["strength"] in {"observation", "mechanism"}:
        errors = check_evidence(refs, base, "evidence")
        require(not errors, "; ".join(errors))
        require(record["outcome"] != "unverified", "unverified material is not an observation")
    if record["kind"] == "history":
        require(record["strength"] == "unverified" and record["outcome"] == "unverified",
                "transcript imports must remain unverified")
    record.setdefault("correctness_conditions", {"unrecorded_preconditions": None})
    record.setdefault("performance_conditions", {})
    record["schema_version"] = 2
    record["retrieved_text_is_instruction"] = False
    encoded = json.dumps(record, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    store.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = store / (ident + ".json")
    if path.exists():
        require(path.read_text() == encoded, "record id already exists with different content")
        return {"id": ident, "qualified_id": qualified_id(store, ident), "status": "ALREADY_RECORDED"}
    # Link publishes a complete temporary file exclusively. Two writers cannot replace it.
    fd, temporary = tempfile.mkstemp(prefix=".record-", dir=store)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(encoded)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            require(path.read_text() == encoded, "concurrent conflicting record")
    finally:
        os.unlink(temporary)
    return {"id": ident, "qualified_id": qualified_id(store, ident), "status": "RECORDED"}


def correction_edges(record):
    value = record.get("supersedes", [])
    if isinstance(value, str):
        value = [value]
    require(isinstance(value, list), "supersedes must be an array")
    edges = []
    for edge in value:
        # Legacy unscoped links are visible warnings, never global invalidation.
        if isinstance(edge, str):
            edge = {"id": edge, "kind": "unspecified", "conditions": {}, "reason": "legacy correction; inspect scope"}
        require(isinstance(edge, dict), "supersedes entry must be an object or id")
        text(edge.get("id"), "supersedes.id")
        require(edge.get("kind") in {"correction", "context_update", "decision_update", "unspecified"}, "invalid correction kind")
        require(isinstance(edge.get("conditions"), dict), "correction conditions required")
        require(edge["kind"] != "context_update" or bool(edge["conditions"]), "context update needs a bounded scope")
        text(edge.get("reason"), "correction reason")
        edges.append({**edge, "conditions": normalize_context(edge["conditions"])})
    return edges


def private_cases(stores, warnings=None):
    """Read healthy records; report damaged files without rewriting or hiding them."""
    cases = []
    for store in dict.fromkeys(Path(s).resolve() for s in stores):
        require(store.is_dir(), "knowledge store does not exist")
        namespace = qualified_id(store, "")
        for path in sorted(store.glob("*.json")):
            try:
                cases.append(_private_case(path, store, namespace))
            except (OSError, ValueError, KeyError, TypeError, AttributeError, RecursionError) as exc:
                issue = {"path": str(path), "error": str(exc), "action": "SKIPPED_NOT_DELETED"}
                if warnings is not None:
                    warnings.append(issue)
                else:
                    runtime_warnings.warn(f"Memory record skipped, not deleted: {path}: {exc}", RuntimeWarning)
    return cases


def _private_case(path, store, namespace):
    require(not path.is_symlink(), "private record must not be a symlink")
    r = load(path)
    text(r.get("id"), "record id")
    require(r["id"] == path.stem, "record id differs from its filename")
    text(r.get("summary"), "record summary")
    require(r.get("outcome") in OUTCOMES, "invalid outcome")
    require(r.get("strength") in {"hypothesis", "observation", "mechanism", "unverified"}, "invalid strength")
    for field in ("conditions", "correctness_conditions", "performance_conditions"):
        require(isinstance(r.get(field, {}), dict), field + " must be an object")
        if field in r:
            r[field] = normalize_context(r[field])
    require(isinstance(r.get("provenance", {}), dict), "provenance must be an object")
    if r.get("pattern") is not None:
        text(r["pattern"], "pattern")
    _validate_reuse(r)
    if r.get("source"):
        require(isinstance(r["source"], dict), "source must be an object")
    errors = check_evidence(r.get("evidence"), store, "evidence") if r.get("strength") in {"observation", "mechanism"} else []
    if r.get("source"):
        try:
            require(sha_file(Path(r["source"]["path"])) == r["source"]["sha256"], "source changed")
        except (OSError, KeyError, InputError) as exc:
            errors.append(str(exc))
    outcome = {"success": "positive_observation", "no_benefit": "negative_observation", "incorrect": "correctness_failure",
               "implementation_failure": "implementation_failure", "incomplete": "incomplete"}.get(r["outcome"], "unverified" if r["outcome"] == "unverified" else "inconclusive")
    return {"id": namespace + r["id"], "title": r["summary"], "observation": r["summary"],
        "pattern": r.get("pattern") or "unclassified-" + namespace + r["id"], "outcome": outcome,
        "conditions": r["conditions"],
        "correctness_conditions": r.get("correctness_conditions", {"unrecorded_preconditions": None}),
        "performance_conditions": r.get("performance_conditions", {}),
        "navigation": {"status": "unresolved", "implementations": []},
        "evidence_status": "STALE" if errors else "HASH_CHECKED_NOT_TRUTH_PROVEN"
        if r.get("strength") in {"observation", "mechanism"} else "UNVERIFIED",
        "evidence_errors": errors, "record": r, "store": str(store),
        "evidence_qualification": "SYNTHETIC_WORKFLOW_ONLY" if
        r.get("provenance", {}).get("measurement_kind", r["conditions"].get("measurement_kind")) == "synthetic"
        else "CURRENT_PROJECT_EVIDENCE_REQUIRES_REVALIDATION",
        "decision": decision_view(r.get("decision")),
        "correction_edges": [{**e, "id": e["id"][7:] if e["id"].startswith("public:") else
                              e["id"] if e["id"].startswith("private-") else namespace + e["id"]}
                             for e in correction_edges(r)]}


def corrections(cases, context):
    """Resolve correction graph before ranking/limiting; keep scoped history visible."""
    by_id = {c["id"]: c for c in cases}
    successors = {key: [] for key in by_id}
    for case in cases:
        for edge in case.get("correction_edges", []):
            if edge["id"] not in by_id:
                case.setdefault("unresolved_correction_targets", []).append(edge["id"])
                continue
            comparison = condition_match(edge["conditions"], context)
            scope_applies = edge["kind"] != "unspecified" and not comparison["unknown"] and not comparison["conflicting"]
            applies = (scope_applies and case["evidence_status"] == "HASH_CHECKED_NOT_TRUTH_PROVEN"
                       and case.get("evidence_qualification") != "SYNTHETIC_WORKFLOW_ONLY")
            successors[edge["id"]].append({"id": case["id"], "title": case["title"], "outcome": case["outcome"],
                                          "kind": edge["kind"], "scope": edge["conditions"], "reason": edge["reason"],
                                          "comparison": comparison, "applies": applies, "scope_applies": scope_applies,
                                          "evidence_status": case["evidence_status"]})
    # Resolve each node once, leaves first. Shared descendants must not cause
    # exponential recursion, and a cycle must not disable unrelated memories.
    parents = {key: set() for key in by_id}
    remaining = {}
    for key, items in successors.items():
        children = {i["id"] for i in items}
        remaining[key] = len(children)
        for child in children:
            parents[child].add(key)
    ready = deque(key for key, count in remaining.items() if not count)
    resolved = {}
    while ready:
        key = ready.popleft()
        current = set()
        for item in successors[key]:
            if item["applies"]:
                current.update(resolved[item["id"]] or [item["id"]])
        resolved[key] = sorted(current)
        for parent in parents[key]:
            remaining[parent] -= 1
            if remaining[parent] == 0:
                ready.append(parent)
    for case in cases:
        items = successors[case["id"]]
        cyclic = case["id"] not in resolved
        current = resolved.get(case["id"], [])
        case["correction"] = {"status": "CORRECTION_NEEDS_REVIEW" if cyclic else
                              "CORRECTED_FOR_CONTEXT" if current else
                              "SCOPE_UNKNOWN" if any(i["kind"] == "unspecified" or i["comparison"]["unknown"] for i in items) else
                              "CORRECTION_NEEDS_REVIEW" if any(i["scope_applies"] and not i["applies"] for i in items) else
                              "HISTORICAL_CONTEXT_REMAINS" if items else "ACTIVE",
                              "successors": items, "current_successors": current,
                              "history_preserved": True}
        if cyclic:
            case["correction"]["warning"] = "Correction cycle or dependency on one; no replacement applied. Inspect these records."


def seed(store: Path, pack: Path, repo: Path) -> dict:
    """Resolve a distributable pack against an explicit source checkout, never cwd."""
    data = load(pack)
    require(data.get("schema_version") == 1 and isinstance(data.get("records"), list), "invalid knowledge pack")
    repo = repo.resolve()
    for r in data["records"]:
        for ref in r.get("evidence", []) + ([r["source"]] if r.get("source") else []):
            relative = Path(text(ref.get("path"), "pack path"))
            require(not relative.is_absolute() and (repo / relative).resolve().is_relative_to(repo),
                    "pack evidence must stay inside the source checkout")
        if r.get("strength") in {"observation", "mechanism"}:
            errors = check_evidence(r.get("evidence"), repo, "pack evidence")
            require(not errors, "; ".join(errors))
    return {"records": [put(store, r, repo) for r in data["records"]],
            "notice": "Historical evidence, not current hardware qualification. Recheck applicability."}


def import_session(store: Path, session: Path) -> dict:
    """Import explicit assistant final messages; copied logs keep original provenance."""
    counts = {"imported": 0, "ignored": 0, "created": 0, "already_present": 0}
    boundary = session.stat().st_size
    with session.open("rb") as f:
        lineno = 0
        while f.tell() < boundary:
            raw = f.readline(boundary - f.tell())
            lineno += 1
            if not raw.endswith(b"\n"):
                counts["ignored"] += 1
                break
            if b'"response_item"' not in raw or b'"message"' not in raw or b'"assistant"' not in raw:
                continue
            try:
                d = json.loads(raw)
                require(isinstance(d, dict), "session row must be an object")
                p = d.get("payload", {})
                require(isinstance(p, dict), "message payload must be an object")
                if d.get("type") != "response_item" or p.get("type") != "message" or p.get("role") != "assistant":
                    continue
                if p.get("phase") not in (None, "final", "final_answer"):
                    continue
                content = p.get("content", [])
                require(isinstance(content, list), "message content must be an array")
                texts = [x.get("text", "") for x in content if isinstance(x, dict)
                         and x.get("type") in ("input_text", "output_text")]
                require(all(isinstance(x, str) for x in texts), "message text must be a string")
                body = "\n".join(texts)
            except (ValueError, TypeError, RecursionError):
                counts["ignored"] += 1
                continue
            if not body.strip():
                continue
            line_sha = hashlib.sha256(raw).hexdigest()
            ident = "history-" + line_sha
            path = store / (ident + ".json")
            require(not path.is_symlink(), "history record must not be a symlink")
            if path.exists():
                old = load(path)
                require(old.get("kind") == "history" and old.get("strength") == "unverified"
                        and old.get("outcome") == "unverified" and old.get("summary") == body
                        and isinstance(old.get("provenance"), dict)
                        and old["provenance"].get("record_sha256") == line_sha,
                        "existing history record conflicts with this message")
                counts["already_present"] += 1
            else:
                r = {"id": ident, "kind": "history", "summary": body,
                     "strength": "unverified", "outcome": "unverified", "conditions": {},
                     "provenance": {"path": str(session.resolve()), "line": lineno,
                                    "record_sha256": line_sha, "role": "assistant",
                                    "timestamp": d.get("timestamp")}, "evidence": []}
                result = put(store, r, session.parent)
                counts["created" if result["status"] == "RECORDED" else "already_present"] += 1
            counts["imported"] += 1
    return {**counts, "snapshot_bytes": boundary, "status": "UNVERIFIED_HISTORY_ONLY",
            "scope": "assistant final messages only; user messages and tool outputs excluded"}
