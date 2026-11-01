"""Derived current recipe navigation; parse data/AST, never import a workload.

The bundled index is generated, not a second manually maintained model registry.
An explicit checkout always wins. No checker or benchmark is executed here.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import re
import subprocess

from dmd_perf import load, require, sha_file
from dmd_retrieval import tokens

INDEX = Path(__file__).resolve().parents[1] / "assets/current-navigation.json"
REGISTRY = "src/distillschedule/training/catalog.py"
PROFILES = "src/distillschedule/training/official_profiles.py"
WORKLOADS = "src/distillschedule/workloads/catalog.json"
STRATEGIES = "src/distillschedule/training/strategy_catalog.py"
REPOSITORY = "sliortega295-ops/DMD-System"


def _path(root, relative):
    path = (root / relative).resolve()
    require(path.is_relative_to(root), "navigation source escapes checkout")
    return path


def _value(node, env=None):
    """Only literal strings/data and concatenation. Never eval/exec calls."""
    if isinstance(node, ast.Name):
        return (env or {}).get(node.id)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        a, b = _value(node.left, env), _value(node.right, env)
        return a + b if isinstance(a, str) and isinstance(b, str) else None
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError):
        return None


def _registrations(tree):
    result = {"model": {}, "algorithm": {}}
    def collect(nodes, env):
        for node in nodes:
            if isinstance(node, ast.For) and isinstance(node.iter, (ast.Tuple, ast.List)):
                if not isinstance(node.target, (ast.Tuple, ast.List)):
                    continue
                names = [x.id if isinstance(x, ast.Name) else None for x in node.target.elts]
                for row in node.iter.elts:
                    if isinstance(row, (ast.Tuple, ast.List)) and len(row.elts) == len(names):
                        collect(node.body, {**env, **{k: _value(v, env) for k, v in zip(names, row.elts) if k}})
            if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
                continue
            call = node.value
            kind = {"register_model": "model", "register_algorithm": "algorithm"}.get(getattr(call.func, "id", ""))
            if kind and call.args:
                name = _value(call.args[0], env)
                factory = next((_value(k.value, env) for k in call.keywords if k.arg == "factory"), None)
                if isinstance(name, str) and isinstance(factory, str):
                    result[kind][name] = factory
    collect(tree.body, {})
    return result


def _assignments(tree):
    return {target.id: _value(node.value) for node in tree.body if isinstance(node, ast.Assign)
            for target in node.targets if isinstance(target, ast.Name)}


def build_index(repo, revision=None):
    root = Path(repo).resolve()
    inputs = {}
    def read(relative):
        path = _path(root, relative)
        inputs[relative] = sha_file(path)
        return path.read_text(encoding="utf-8")
    registered = _registrations(ast.parse(read(REGISTRY)))
    official = _assignments(ast.parse(read(PROFILES)))
    workloads = {w["id"]: w for w in json.loads(read(WORKLOADS))["workloads"]}
    strategy_tree = ast.parse(read(STRATEGIES))
    capabilities = []
    for node in strategy_tree.body:
        if not isinstance(node, ast.Assign) or not any(getattr(x, "id", "") == "_CAPABILITIES" for x in node.targets):
            continue
        if isinstance(node.value, ast.Dict):
            for key, call in zip(node.value.keys, node.value.values):
                if isinstance(call, ast.Call) and getattr(call.func, "id", "") == "_cap" and len(call.args) == 5:
                    values = [_value(arg) for arg in call.args]
                    capabilities.append(dict(id=_value(key), category=values[0], implementation=values[1],
                        applicability=values[2], effects=values[3], evidence=values[4]))
    test_sources = {path.relative_to(root).as_posix(): _path(root, path.relative_to(root)).read_text(encoding="utf-8")
                    for path in sorted((root / "tests").glob("test_*.py"))}
    rows = []
    for folder in sorted((root / "recipes").iterdir()):
        if not folder.is_dir() or folder.is_symlink():
            continue
        preferred = set()
        row = {"id": folder.name, "models": [], "algorithms": [], "recipes": {"reference": [], "optimized": [], "other": []},
               "implementations": [], "profiles": [], "checks": [], "tests_run": False}
        for path in sorted(folder.glob("training*.json")):
            relative = path.relative_to(root).as_posix()
            config = json.loads(read(relative))
            if config.get("schema_version") != "distillschedule.training.v1":
                continue
            model, algorithm, execution = (config.get(k, {}) for k in ("model", "algorithm", "execution"))
            profile = model.get("parameters", {}).get("optimization_profile")
            mode = "optimized" if profile or "optimized" in path.name else "reference" if (
                execution.get("mode") == "reference" or "reference" in path.name) else "other"
            row["recipes"][mode].append(relative)
            for kind, value, key in (("model", model, "models"), ("algorithm", algorithm, "algorithms")):
                name = value.get("name") or value.get("workload") or value.get("factory")
                if name:
                    row[key].append(name)
                factory = value.get("factory") or registered[kind].get(name)
                if factory and re.fullmatch(r"[\w.]+:\w+", factory):
                    module = "src/" + factory.partition(":")[0].replace(".", "/") + ".py"
                    if _path(root, module).is_file():
                        row["implementations"].append(module)
                        if kind == "model":
                            preferred.add(module)
            native = workloads.get(model.get("workload"), {})
            row["implementations"].extend(p for p in native.get("code", []) if _path(root, p).is_file())
            if profile:
                metadata = (official.get("_PROFILES") or {}).get(profile, {})
                row["profiles"].append({"id": profile, "mechanisms": metadata.get("mechanisms", []),
                    "contract": metadata.get("contract", {}),
                    "numerical_status": metadata.get("origin", {}).get("numerical_status", "UNKNOWN")})
                row["implementations"].extend("src/distillschedule/training/" + p
                    for p in (official.get("_RUNTIME_FILES") or {}).get(profile, ())
                    if _path(root, "src/distillschedule/training/" + p).is_file())
        if not any(row["recipes"].values()):
            continue
        for key in ("models", "algorithms", "implementations"):
            row[key] = sorted(set(row[key]))
        row["profiles"] = list({p["id"]: p for p in row["profiles"]}.values())
        guide = folder / "README.md"
        row["guide"] = guide.relative_to(root).as_posix() if guide.is_file() else "docs/guides/support_matrix.md"
        row["title"] = read(row["guide"]).splitlines()[0].lstrip("# ")[:160]
        row["strategy_catalog"] = STRATEGIES
        row["has_optimized_recipe"] = bool(row["recipes"]["optimized"])
        # Exact module boundaries avoid matching dmd2 tests as dmd tests.
        checks = []
        for path, source in test_sources.items():
            score = 3 * ("recipes/" + folder.name + "/" in source)
            for module in row["implementations"]:
                if not module.startswith("src/distillschedule/"):
                    continue
                name = module[4:-3].replace("/", ".")
                if re.search(re.escape(name) + r"(?![\w])", source):
                    score += 2 if module in preferred else 1
            if score:
                checks.append((score, path))
        checks.sort(key=lambda x: (-x[0], x[1]))
        row["check_count"] = len(checks)
        row["checks"] = [path for _, path in checks[:4]]
        rows.append(row)
    if revision is None:
        try:
            result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
            revision = result.stdout.strip() if result.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            pass
    require(revision is None or re.fullmatch(r"[0-9a-f]{40}", revision), "invalid source revision")
    return {"schema_version": 1, "repository": REPOSITORY, "source_revision": revision,
            "source_hashes": inputs, "workflows": rows, "capabilities": capabilities, "automatic_application": False,
            "notice": "Derived recipe/code/test navigation, not a speed guarantee or new validation. Check current guards and fill asset placeholders."}


def lookup(query, repo=None, limit=4, index=INDEX):
    require(1 <= limit <= 20, "navigation limit must be 1..20")
    # Only inspect an explicitly provided checkout or this Skill's own checkout.
    root = Path(repo).resolve() if repo is not None else Path(__file__).resolve().parents[3]
    try:
        if repo is not None and _path(root, REGISTRY).is_file():
            data, source = build_index(root), "CHECKOUT_METADATA_NOT_RUNTIME_VALIDATED"
        elif repo is not None:
            return {"status": "NOT_A_DMD_CHECKOUT", "matches": [], "automatic_application": False}
        else:
            data, source = load(index), "BUNDLED_SNAPSHOT_NOT_CURRENT_TREE"
        require(data.get("schema_version") == 1, "unsupported navigation index")
        normalized_query = re.sub(r"(?<=[a-zA-Z])(?=[0-9])|(?<=[0-9])(?=[a-zA-Z])", " ", query)
        query_terms = {t for t in tokens(normalized_query.replace("-", " ").replace("_", " ").replace(".", " ")) if len(t) > 1 and not t.isdigit()}
        matches = []
        for row in data["workflows"]:
            identity = " ".join([row["id"], row.get("title", ""), *row["models"]]).lower().replace("_", " ").replace("-", " ")
            corpus = json.dumps({k: row[k] for k in ("id", "models", "algorithms", "profiles", "recipes", "implementations")}, ensure_ascii=False).lower().replace("_", " ").replace("-", " ")
            identity_hits = sum(t in identity for t in query_terms)
            version_names = re.findall(r"[a-z]+[0-9]+(?:[.][0-9]+)*", query.lower())
            compact_identity = re.sub(r"[^a-z0-9]", "", identity)
            exact_model = any(re.sub(r"[^a-z0-9]", "", name) in compact_identity for name in version_names)
            score = 32 * exact_model + 8 * identity_hits + sum(t in corpus for t in query_terms)
            if not query_terms or identity_hits or score >= 3:
                matches.append({**row, "score": score})
        matches.sort(key=lambda r: (-r["score"], r["id"]))
        caps = sorted(data.get("capabilities", []), key=lambda c: -sum(t in json.dumps(c).lower().replace("_", " ").replace("-", " ") for t in query_terms))
        caps = [c for c in caps if query_terms and any(t in json.dumps(c).lower().replace("_", " ").replace("-", " ") for t in query_terms)][:3]
        shown = []
        for row in matches[:limit]:
            shown.append({**row, "checks": row["checks"][:4], "check_count": row.get("check_count", len(row["checks"])),
                "recipe_counts": {k: len(v) for k, v in row["recipes"].items()},
                "recipes": {k: v[:4] for k, v in row["recipes"].items()}})
        return {"status": source, "source_revision": data.get("source_revision"),
                "matches": shown, "related_capabilities_require_recipe_check": caps, "total_matches": len(matches), "truncated": len(matches) > limit,
                "automatic_application": False, "notice": data["notice"]}
    except (OSError, ValueError, TypeError, KeyError, SyntaxError) as exc:
        return {"status": "NAVIGATION_UNAVAILABLE", "matches": [], "warning": str(exc),
                "automatic_application": False, "read_next": "Read current recipes/ and docs/guides/support_matrix.md; historical recall remains usable."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision")
    args = parser.parse_args()
    args.output.write_text(json.dumps(build_index(args.repo, args.revision), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
