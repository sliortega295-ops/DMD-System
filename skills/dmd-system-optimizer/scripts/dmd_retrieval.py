"""Small, inspectable symptom routing over lexical retrieval; no inferred facts."""
import json
import math
import re
from pathlib import Path
from dmd_perf import require, text

ROUTES = Path(__file__).resolve().parents[1] / "assets/query-routes.json"
STOP = {"的", "了", "是", "和", "与", "在", "把", "有", "要", "我", "你", "我们", "这个", "那个",
        "一个", "一些", "怎么", "可以", "进行", "当前", "需要", "已经", "没有", "问题", "方法", "东西",
        "时候", "但是", "以及", "通过", "可能", "不同", "有些", "什么", "如何", "一下", "同样", "样的",
        "的东", "了好", "有点", "训练", "运行", "优化", "加速", "想要", "怎样", "能否",
        "the", "a", "an", "is", "are", "to", "of", "and", "for", "with", "my"}


def tokens(query):
    words = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]+", query.lower())
    return {t for t in words + [w[i:i+2] for w in words if re.fullmatch(r"[\u4e00-\u9fff]+", w)
                                for i in range(len(w) - 1)] if t not in STOP}


def validate_routes(pattern_ids, path=ROUTES):
    data = json.loads(Path(path).read_text())
    require(data.get("schema_version") == 1 and isinstance(data.get("routes"), list), "invalid query routes")
    rules = data["routes"]
    require(len({r["id"] for r in rules}) == len(rules), "duplicate query route")
    for rule in rules:
        text(rule.get("id"), "query route id")
        for field in ("symptoms", "patterns", "inspect_next"):
            require(isinstance(rule.get(field), list) and rule[field], "route " + field + " required")
            for value in rule[field]:
                text(value, "route " + field)
        require(set(rule["patterns"]) <= pattern_ids, "query route points to missing pattern")
        for alias in rule["symptoms"]:
            re.compile(alias)
    return rules


def rank(data, query, routes_path=ROUTES):
    """Routes suggest things to inspect; context is never constructed from a query."""
    rules = validate_routes({p["id"] for p in data["patterns"]}, routes_path)
    matched = [r for r in rules if any(re.search(alias, query, re.I) for alias in r["symptoms"])]
    corpora, fields = {}, {}
    for p in data["patterns"]:
        fields[p["id"]] = [(p["title"], 4), (" ".join(p["tags"]), 4), (p["mechanism"], 1),
            (" ".join(c["title"] + " " + c["observation"] for c in data["cases"] if c["pattern"] == p["id"]), 0.5)]
        corpora[p["id"]] = " ".join(v for v, _ in fields[p["id"]]).lower()
    ts = tokens(query)
    idf = {t: math.log(1 + len(corpora) / (1 + sum(t in v for v in corpora.values()))) for t in ts}
    rows = []
    for p in data["patterns"]:
        lexical = sum(weight * sum(idf[t] for t in ts if t in value.lower()) for value, weight in fields[p["id"]])
        hits = [r for r in matched if p["id"] in r["patterns"]]
        # The strongest symptom route wins; synonyms do not multiply the score.
        boost = max((80 - 16 * r["patterns"].index(p["id"]) for r in hits), default=0)
        score = round(lexical + boost, 3)
        if score >= 3 or boost:
            rows.append({"score": score, "pattern": p, "matched_symptoms": [r["id"] for r in hits]})
    rows.sort(key=lambda x: (-x["score"], x["pattern"]["id"]))
    broad = bool(matched) and all(r.get("ambiguous", False) for r in matched)
    status = ("NO_MATCH" if not rows else "NEEDS_DIAGNOSIS" if broad or not matched and rows[0]["score"] < 12
              else "CANDIDATE_MECHANISMS")
    questions = list(dict.fromkeys(q for r in matched for q in r["inspect_next"]))
    if status == "NEEDS_DIAGNOSIS" and not questions:
        questions = ["先取得慢阶段、调用次数、等待/计算/搬运占比；目前关键词不足以区分机制。"]
    return rows, {"status": status, "matched_symptoms": [r["id"] for r in matched],
                  "inspect_next": questions, "query_does_not_supply_context_facts": True,
                  "notice": "Symptom routes rank hypotheses, not root causes or applicable optimizations."}
