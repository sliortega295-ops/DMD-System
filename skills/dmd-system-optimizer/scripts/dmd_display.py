"""Bounded reading indexes. Full observations and history remain in case/recall."""
from collections import Counter
from dmd_perf import require


def _short(value, depth=0):
    if isinstance(value, str):
        return value if len(value) <= 240 else value[:237] + "..."
    if isinstance(value, (dict, list)) and depth >= 4:
        return {"detail": "expand with case --id", "items": len(value)}
    if isinstance(value, list):
        rows = [_short(x, depth + 1) for x in value[:4]]
        return rows + ([{"omitted": len(value) - 4}] if len(value) > 4 else [])
    if isinstance(value, dict):
        rows = {k: _short(v, depth + 1) for k, v in list(value.items())[:12]}
        if len(value) > 12:
            rows["_omitted_fields"] = len(value) - 12
        return rows
    return value


def _correction(value):
    value = value or {}
    ids = value.get("current_successors", [])
    successors = value.get("successors", [])
    return {"status": value.get("status", "ACTIVE"), "history_preserved": True,
            "current_successors": ids[:3], "current_successors_total": len(ids),
            "successors": [{k: x[k] for k in ("id", "kind", "applies", "evidence_status") if k in x}
                           for x in successors[:3]], "successors_total": len(successors),
            "truncated": len(ids) > 3 or len(successors) > 3,
            **({"warning": _short(value["warning"])} if value.get("warning") else {})}


def _case(case):
    row = {"id": case["id"], "title": _short(case["title"]), "outcome": case["outcome"],
           "evidence_status": case["evidence_status"], "next_action": case["next_action"],
           "relevance_score": case.get("relevance_score", 0),
           "correctness_conflicts": case["correctness_comparison"]["conflicting"],
           "code_status": case["navigation"]["status"], "correction": _correction(case.get("correction")),
           "code_relationships": [{k: i[k] for k in ("id", "relationship", "relationship_action", "direct_reuse_verified")}
                                  for i in case["code_navigation"]["implementations"][:3]]}
    for name in ("facts", "interpretation", "observation_comparisons"):
        if case.get(name):
            row[name] = _short(case[name])
    if "store" in case:
        row.update(store=case["store"], evidence_qualification=case["evidence_qualification"],
                   reused_implementation_ids=[i["id"] for i in case["navigation"]["implementations"][:3]],
                   target_source=_short(case["record"].get("source")),
                   decision={k: _short((case.get("decision") or {}).get(k)) for k in
                       ("retention", "performance", "confirmation", "evidence_assurance", "measurement_kind", "validation_scope")})
    return row


def brief_recall(result, case_limit=3, case_offset=0, all_cases=False):
    """Show a few relevant records per outcome, with counts and explicit paging."""
    require(type(case_limit) is int and 1 <= case_limit <= 20, "case limit must be 1..20")
    require(type(case_offset) is int and case_offset >= 0, "case offset must be nonnegative")
    packets = []
    for packet in result["matches"]:
        cases, counts, omitted = {}, {}, {}
        for name, group in packet["cases"].items():
            selected = group if all_cases else group[case_offset:case_offset + case_limit]
            cases[name] = [_case(c) for c in selected]
            counts[name] = len(group)
            omitted[name] = len(group) - len(selected)
        coverage = Counter(c.get("correction", {}).get("status", "ACTIVE")
                           for group in packet["cases"].values() for c in group)
        p = packet["pattern"]
        has_more = not all_cases and any(n > case_offset + case_limit for n in counts.values())
        packets.append({"pattern": {k: p[k] for k in ("id", "title", "probe", "details")},
            "score": packet["score"], "matched_symptoms": packet["matched_symptoms"],
            "correctness_conditions": packet["correctness_conditions"],
            "performance_conditions": packet["performance_conditions"], "next_action": packet["next_action"],
            "cases": cases, "case_counts": counts, "omitted_cases": omitted,
            "correction_counts": dict(coverage), "all_linked_case_ids_included": not any(omitted.values()),
            "has_more": has_more, "next_case_offset": case_offset + case_limit if has_more else None,
            "detail_level": "index_only"})
    warnings = result.get("memory_warnings", [])
    return {**result, "matches": packets, "memory_warnings": warnings[:5],
            "memory_warning_count": len(warnings), "memory_warnings_truncated": len(warnings) > 5,
            "brief_policy": {"per_outcome_limit": case_limit, "offset": case_offset, "all_cases": all_cases},
            "read_next": "case --id CASE_ID with the same authorized stores returns full evidence and corrections. Use recall --brief --case-offset N to page each outcome, --all-cases for every ID, or omit --brief for full records.",
            "locate_next": "locate --case CASE_ID [--repo CHECKOUT] [--store PRIVATE_STORE] expands implementation guards, tests and fallback; preserve the same stores."}
