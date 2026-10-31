#!/usr/bin/env python3
"""Pinned case-to-code navigation. Reads Python syntax, never imports target code."""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import subprocess

from dmd_perf import load, require, text

CATALOG = Path(__file__).resolve().parents[1] / "assets/implementation-map.json"
STATUSES = {"linked", "method_only", "unresolved"}
RELATION_ACTIONS = {
    "historical_implementation": "INSPECT_HISTORICAL_IMPLEMENTATION",
    "reusable_candidate": "VALIDATE_CURRENT_REUSE",
    "adaptation_example": "ADAPT_BEFORE_USE",
    "diagnostic_tool": "DIAGNOSTIC_ONLY",
    "conceptual_reference": "CONCEPT_ONLY",
    # Read older private records without upgrading their broad relationship.
    "current_related": "INSPECT_RELATIONSHIP",
    "historical_reference": "INSPECT_RELATIONSHIP",
}
RELATIONS = set(RELATION_ACTIONS)


def relation_summary(link):
    return {"relationship_action": RELATION_ACTIONS[link["relationship"]],
            "direct_reuse_verified": False}


def symbol_span(source, symbol):
    """Return a qualified Python definition's span, without executing the module."""
    if symbol == "@module":
        return 1, len(source.splitlines())
    def visit(body, prefix=""):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = prefix + node.name
                if name == symbol:
                    return node.lineno, node.end_lineno
                found = visit(node.body, name + ".")
                if found:
                    return found
        return None
    return visit(ast.parse(source).body)


def callsite_matches(source, ref):
    return any(isinstance(n, ast.Call) and n.lineno == ref["focus_start"] and n.end_lineno == ref["focus_end"]
               and ((isinstance(n.func, ast.Name) and n.func.id == ref["focus_call"])
                    or (isinstance(n.func, ast.Attribute) and n.func.attr == ref["focus_call"]))
               for n in ast.walk(ast.parse(source)))


def validate_catalog(path=CATALOG):
    from dmd_knowledge import validate_reference
    data = load(path)
    require(data.get("schema_version") == 1, "implementation map schema must be 1")
    entries = data.get("implementations")
    require(isinstance(entries, list) and entries, "implementation map must contain entries")
    ids = [e["id"] for e in entries]
    require(len(ids) == len(set(ids)), "duplicate implementation id")
    for entry in entries:
        for key in ("id", "title", "usage_mode", "scope", "integration_instructions", "validation_scope"):
            text(entry.get(key), "implementation." + key)
        require(entry["usage_mode"] in {"runtime_component", "historical_component", "diagnostic_only", "analysis_tool"},
                "invalid implementation usage mode")
        for key in ("correctness_conditions", "performance_conditions"):
            require(isinstance(entry.get(key), dict), "implementation " + key + " required")
        for key in ("entrypoints", "integration"):
            require(isinstance(entry.get(key), list) and entry[key], key + " references required")
        require(isinstance(entry.get("tests"), list), "test references required")
        if not entry["tests"]:
            text(entry.get("test_gap"), "unlocated tests must be disclosed")
        text(entry.get("fallback", {}).get("action"), "fallback action")
        require(bool(entry["fallback"].get("references")), "fallback references required")
        for related in entry.get("related_implementations", []):
            require(related["id"] in ids and related["id"] != entry["id"], "unknown/self related implementation")
            text(related.get("note"), "related implementation scope")
        for ref in references(entry):
            validate_reference(ref)
            text(ref.get("symbol"), "code symbol")
            require(ref["path"].endswith(".py"), "code map currently supports Python definitions")
            require(type(ref.get("line_start")) is int and type(ref.get("line_end")) is int
                    and 0 < ref["line_start"] <= ref["line_end"], "invalid code span")
            start, end = ref["line_start"], ref["line_end"]
            if "focus_call" in ref:
                text(ref["focus_call"], "integration call")
                require(type(ref.get("focus_start")) is int and type(ref.get("focus_end")) is int
                        and start <= ref["focus_start"] <= ref["focus_end"] <= end, "callsite must stay inside its symbol")
                start, end = ref["focus_start"], ref["focus_end"]
            require(ref.get("anchor_url") == ref["url"] + f"#L{start}-L{end}",
                    "code anchor must bind the declared span")
    return {entry["id"]: entry for entry in entries}


def references(entry):
    return entry["entrypoints"] + entry["integration"] + entry["tests"] + entry["fallback"]["references"]


def validate_case(case, catalog):
    nav = case.get("navigation", {})
    require(nav.get("status") in STATUSES, "case navigation status required: " + case["id"])
    text(nav.get("reason"), "navigation reason")
    links = nav.get("implementations")
    require(isinstance(links, list), "case implementation links required")
    require((nav["status"] == "linked") == bool(links), "linked status must agree with code links")
    require(len({link["id"] for link in links}) == len(links), "duplicate case code link")
    for link in links:
        require(link["id"] in catalog, "unknown implementation link: " + link["id"])
        require(link.get("relationship") in RELATIONS, "declare scoped code relationship")
        text(link.get("note"), "code relationship note")
        if catalog[link["id"]]["usage_mode"] in {"diagnostic_only", "analysis_tool"}:
            require(link["relationship"] not in {"historical_implementation", "reusable_candidate"},
                    "analysis/diagnostic source is not a runtime implementation")


def summarize(case, catalog):
    nav = case["navigation"]
    return {**nav, "implementations": [
        {**link, **relation_summary(link), "title": catalog[link["id"]]["title"],
         "entrypoints": [{k: ref[k] for k in ("repository", "revision", "path", "symbol", "anchor_url")}
                         for ref in catalog[link["id"]]["entrypoints"]],
         "locate": "locate --implementation " + link["id"]}
        for link in nav["implementations"]]}


def inspect_reference(ref, repo=None, source_repos=None, show_code=False):
    from dmd_knowledge import DEFAULT_REPOSITORY, verify_references
    check = verify_references([ref], repo, source_repos)
    # Never carry a prior result's excerpt/local verification into a new check.
    derived = {"excerpt", "excerpt_truncated", "local_path", "read_revision", "symbol_status",
               "artifact_verification", "evidence_checks", "evidence_errors"}
    row = {**{k: v for k, v in ref.items() if k not in derived}, **check, "symbol_status": "NOT_LOCALLY_CHECKED"}
    if not check["evidence_checks"] or check["evidence_checks"][0]["status"] != "HASH_MATCH":
        return row
    roots = dict(source_repos or {})
    if repo is not None:
        roots[DEFAULT_REPOSITORY] = repo
    root = Path(roots[ref["repository"]]).resolve()
    origin = check["evidence_checks"][0]["checked_from"]
    if origin == "working_tree_bytes":
        source = (root / ref["path"]).read_bytes()
    else:
        source = subprocess.check_output(["git", "-C", str(root), "show", ref["revision"] + ":" + ref["path"]], timeout=15)
    # Recheck the bytes we actually inspect, including a concurrent working-tree edit.
    if hashlib.sha256(source).hexdigest() != ref["sha256"]:
        row.update(symbol_status="SOURCE_CHANGED_DURING_READ", artifact_verification="STALE_OR_UNAVAILABLE")
        return row
    source = source.decode("utf-8")
    span = symbol_span(source, ref["symbol"])
    expected = (ref["line_start"], ref["line_end"])
    row["symbol_status"] = "SYMBOL_AND_SPAN_CHECKED" if span == expected else "SYMBOL_OR_SPAN_MISMATCH"
    if span == expected and "focus_call" in ref and not callsite_matches(source, ref):
        row["symbol_status"] = "CALLSITE_MISMATCH"
    row["local_path"] = str(root / ref["path"])
    row["read_revision"] = ref["revision"] if origin == "pinned_git_blob" else None
    if row["symbol_status"] != "SYMBOL_AND_SPAN_CHECKED":
        row["artifact_verification"] = "STALE_OR_UNAVAILABLE"
    elif show_code:
        start, end = (ref["focus_start"], ref["focus_end"]) if "focus_call" in ref else span
        shown_end = min(end, start + 29)
        row["excerpt"] = "\n".join(f"{i + 1}: {line}" for i, line in
                                    enumerate(source.splitlines()[start - 1:shown_end], start - 1))
        row["excerpt_truncated"] = shown_end < end
    return row


def locate(implementation_id, context, repo=None, source_repos=None, show_code=False, catalog=None):
    from dmd_knowledge import condition_match
    catalog = validate_catalog() if catalog is None else catalog
    require(implementation_id in catalog, "unknown implementation: " + implementation_id)
    entry = catalog[implementation_id]
    hard = condition_match(entry["correctness_conditions"], context)
    benefit = condition_match(entry["performance_conditions"], context)
    result = {**entry, "correctness_comparison": hard, "performance_comparison": benefit,
              "next_action": "DO_NOT_APPLY_AS_IS" if hard["conflicting"] else
                             "CHECK_UNKNOWN_PRECONDITIONS" if hard["unknown"] else "ASSESS_CURRENT_BENEFIT",
              "automatic_application": False, "gpu_validation": "NOT_RUN",
              "notice": "Pinned source navigation only. Related current code is not the historical measured implementation; "
                        "read the case relationship. A symbol/hash match proves neither portability nor training correctness."}
    checked = {}
    def inspect(ref):
        key = (ref["repository"], ref["revision"], ref["path"], ref["symbol"], ref.get("focus_call"))
        if key not in checked:
            checked[key] = inspect_reference(ref, repo, source_repos, show_code)
        return checked[key]
    for key in ("entrypoints", "integration", "tests"):
        result[key] = [inspect(ref) for ref in entry[key]]
    result["fallback"] = {**entry["fallback"], "references": [inspect(ref) for ref in entry["fallback"]["references"]]}
    return result
