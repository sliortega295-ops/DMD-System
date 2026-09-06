"""Strict best-known profile resolution over the existing workload workers.

Profiles are immutable measured configurations, not a new training loop. Assets
are supplied independently; a resolved receipt never promotes historical timing
to a measurement of this checkout. This module needs only the Python stdlib.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

SCHEMA = "distillschedule.best_known.v1"
GPU_NOT_RUN = "NOT_RUN_ON_INTEGRATION_BRANCH"
CANONICAL_IDS = frozenset({
    "wan13b_nogan_resident_best", "helios14b_nogan_best", "minimax_h3_grid1000_best",
    "helios14b_gan_best", "livewan_crossscale_pipeline_best",
})
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
HELIOS_SHA = "5d3ffe37cdea3c04a156106f6246c82f91010a1eaf0dd61529cd74083065a886"


def repository_root() -> Path:
    return Path(__file__).resolve().parents[4]


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _object(path: Path) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key} in {path}")
            result[key] = value
        return result
    value = json.loads(path.read_text(), object_pairs_hook=pairs)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_profile(workload: str, *, root: Path | None = None) -> dict[str, Any]:
    root = root or repository_root()
    _require(workload in CANONICAL_IDS, f"unknown canonical workload: {workload}")
    folder = root / "configs/distillschedule/best_known"
    locks = _object(folder / "registry.lock.json")["profiles"]
    _require(set(locks) == CANONICAL_IDS, "registry must contain exactly five canonical profiles")
    value = _object(folder / f"{workload}.json")
    _require(fingerprint(value) == locks[workload], f"canonical profile changed: {workload}")
    _require(value.get("schema_version") == SCHEMA and value.get("id") == workload,
             "profile schema/id mismatch")
    for field in ("contract", "topology", "memory", "runtime", "required_assets", "source", "historical", "validation"):
        _require(isinstance(value.get(field), dict), f"missing profile field {field}")
    _require(value["validation"]["server_revalidation"] == GPU_NOT_RUN, "historical result promoted to new GPU validation")
    _require(value["historical"]["formal_speedup_eligible"] is False, "historical timing is not an integration speedup")
    env = value["runtime"]["environment"]
    _require(all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()), "environment values must be explicit strings")
    _require("${" not in json.dumps(value), "canonical profile contains unresolved environment defaults")
    if workload == "minimax_h3_grid1000_best":
        _require(value["semantics"] == "approximate_grid1000", "Grid1000 must remain approximate")
        _require(env["H3_ACTIVATION_POLICY"] == "none" and env["H3_FA3_REPLAY_CACHE_STORAGE"] == "cpu_staged",
                 "H3 boundary/replay storage policies were conflated")
    if workload == "helios14b_gan_best":
        _require(value["contract"]["winner"] == "C25", "unvalidated GAN candidate promoted")
    if workload == "livewan_crossscale_pipeline_best":
        _require(value["contract"]["actual_global_batch"] == 16 and value["contract"]["timing_normalized_from_B32"] is False,
                 "B32 normalization is not actual-B16 timing")
    return value


def list_profiles(*, root: Path | None = None) -> list[dict[str, Any]]:
    return [load_profile(k, root=root) for k in sorted(CANONICAL_IDS)]


def git_head(root: Path) -> str:
    value = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    _require(bool(re.fullmatch(r"[0-9a-f]{40}", value)), "integration checkout has no exact Git HEAD")
    return value


def source_layout(root: Path, profile: Mapping[str, Any], *, verify_upstreams: bool = False) -> dict[str, Any]:
    missing = [p for p in (profile["runtime"]["launcher"], "scripts/distillctl.py") if not (root / p).is_file()]
    _require(not missing, f"missing integrated source: {missing}")
    if profile["model_family"] in {"helios", "wan2.1"}:
        path = root / "third_party/helios_compat/source/helios/modules/transformer_helios.py"
        _require(path.is_file() and file_sha256(path) == HELIOS_SHA, "bundled Helios Transformer does not match measured pin")
    rows = []
    for dep in profile["upstreams"]:
        path = root / dep["path"]
        # The repository tree itself is the canonical dependency pin. This is
        # checked even on CPU CI where submodules are intentionally not cloned.
        tree_line = subprocess.check_output(
            ["git", "-C", str(root), "ls-tree", "HEAD", "--", dep["path"]],
            text=True,
        ).strip()
        parts = tree_line.split(None, 3)
        _require(len(parts) == 4 and parts[0] == "160000" and parts[1] == "commit",
                 f"upstream is not a pinned gitlink: {dep['path']}")
        _require(parts[2] == dep["commit"], f"upstream gitlink revision mismatch: {dep['path']}")
        status = "PASS_GITLINK_PIN"
        if verify_upstreams:
            _require(path.is_dir(), f"upstream not initialized: {dep['path']}; run git submodule update --init --recursive")
            _require(git_head(path) == dep["commit"], f"upstream revision mismatch: {dep['path']}")
            dirty = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain"], text=True)
            _require(not dirty.strip(), f"upstream source is dirty: {dep['path']}")
            status = "PASS_CHECKED_OUT_CLEAN"
        rows.append({**dep, "verification": status})
    return {"status": "PASS", "scope": "PACKAGED_SOURCE_LAYOUT", "upstreams": rows,
            "real_model_import": "NOT_RUN", "gpu_revalidation": GPU_NOT_RUN}


def load_assets(path: Path, profile: Mapping[str, Any]) -> dict[str, Any]:
    value = _object(path)
    allowed = {"schema_version", "nodes", "master_addr", "master_port", "run_root", "node_local_cache_root", "assets", "correctness", "timeout_seconds"}
    _require(not (set(value) - allowed), f"unknown assets fields (performance overrides forbidden): {sorted(set(value)-allowed)}")
    _require(value.get("schema_version") == 1, "assets schema_version must be 1")
    nodes = value.get("nodes")
    _require(isinstance(nodes, list) and len(nodes) == 2, "exactly two nodes are required")
    _require(nodes[0].get("host") != nodes[1].get("host"), "nodes must use different SSH hosts")
    for n in nodes:
        _require(set(n) == {"host", "repo_root", "python_env"}, "node must explicitly specify host, repo_root, python_env")
        _require(bool(re.fullmatch(r"[A-Za-z0-9_.@:-]+", str(n["host"]))) and not n["host"].startswith("-"), "unsafe SSH host")
        for k in ["repo_root", "python_env"]:
            _require(isinstance(n[k], str) and Path(n[k]).is_absolute(), f"node {k} must be absolute")
    for k in ["run_root", "node_local_cache_root"]:
        _require(isinstance(value.get(k), str) and Path(value[k]).is_absolute(), f"{k} must be explicit and absolute")
    for node in nodes:
        repo = Path(node["repo_root"]).resolve()
        output = Path(value["run_root"]).resolve()
        _require(output != repo and repo not in output.parents, "run outputs must be outside the source checkout")
    cache = Path(value["node_local_cache_root"])
    _require(cache == Path('/tmp') or Path('/tmp') in cache.parents or cache == Path('/var/tmp') or Path('/var/tmp') in cache.parents,
             "Triton cache must be under node-local /tmp or /var/tmp, not a shared filesystem")
    _require(isinstance(value.get("master_addr"), str) and bool(value["master_addr"]), "master_addr is required")
    _require(type(value.get("master_port")) is int and 1024 <= value["master_port"] <= 65530, "invalid rendezvous port")
    _require(type(value.get("timeout_seconds", 18000)) is int and value.get("timeout_seconds", 18000) > 0, "invalid timeout")
    assets = value.get("assets")
    _require(isinstance(assets, dict) and set(assets) == set(profile["required_assets"]), "assets must exactly match required_assets for this workload")
    for name, spec in assets.items():
        _require(isinstance(spec, dict) and not (set(spec)-{"path", "sha256", "identity_file"}), f"unsupported asset fields: {name}")
        _require(isinstance(spec.get("path"), str) and Path(spec["path"]).is_absolute(), f"asset {name} path must be explicit and absolute")
        _require(bool(SHA256_RE.fullmatch(str(spec.get("sha256", "")))), f"asset {name} needs a SHA256 fingerprint")
        if profile["required_assets"][name]["kind"] == "directory":
            identity = Path(spec.get("identity_file", profile["required_assets"][name].get("identity_file", "")))
            _require(str(identity) not in {"", "."} and not identity.is_absolute() and '..' not in identity.parts, f"directory asset {name} needs a relative identity_file")
    return value


def verify_assets(profile: Mapping[str, Any], assets: Mapping[str, Any]) -> list[dict[str, Any]]:
    result = []
    for name, item in assets["assets"].items():
        definition = profile["required_assets"][name]
        path = Path(item["path"])
        if definition["kind"] == "directory":
            _require(path.is_dir(), f"missing directory asset {name}: {path}")
            identity = path / item.get("identity_file", definition.get("identity_file", ""))
        else:
            _require(path.is_file(), f"missing file asset {name}: {path}")
            identity = path
        _require(identity.is_file() and file_sha256(identity) == item["sha256"], f"asset fingerprint mismatch: {name}")
        result.append({"name": name, "path": str(path), "identity_file": str(identity), "sha256": item["sha256"],
                       "scope": "IDENTITY_FILE_ONLY_PAYLOADS_CHECKED_BY_NATIVE_WORKER"})
    return result


def _substitute(value: Any, replacements: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        for k, v in replacements.items():
            value = value.replace('{'+k+'}', v)
        _require(not re.search(r"\{(?:asset:|node_rank|run_id|head|master_|output_dir)", value), f"unresolved template: {value}")
        return value
    if isinstance(value, list):
        return [_substitute(v, replacements) for v in value]
    if isinstance(value, dict):
        return {k: _substitute(v, replacements) for k, v in value.items()}
    return value


def resolve_profile(workload: str, *, root: Path | None = None, assets_path: Path | None = None,
                    run_id: str = "source-inspection", mode: str = "correctness") -> dict[str, Any]:
    root = root or repository_root()
    profile = load_profile(workload, root=root)
    _require(bool(SAFE_ID.fullmatch(run_id)) and run_id not in {".", ".."}, "unsafe run id")
    _require(mode in profile["allowed_run_modes"], f"mode {mode} is not authorized for {workload}")
    head = git_head(root)
    native_plan = None
    if "typed_plan" in profile["runtime"]:
        from distillschedule.core.config import _load_with_includes
        files: list[Path] = []
        raw_plan = _load_with_includes(root / profile["runtime"]["typed_plan"], stack=(), sources=files)
        native_plan = {"template": raw_plan, "digest": fingerprint(raw_plan), "source_files": [str(f.relative_to(root)) for f in dict.fromkeys(files)]}
    receipt = {
        "schema_version": "distillschedule.best_known.receipt.v1", "integration_head": head,
        "profile_id": workload, "profile_version": profile["profile_version"],
        "profile_sha256": fingerprint(profile), "canonical_status": "CANONICAL", "canonical_scope": "CONFIGURATION_ONLY_ASSETS_SEPARATELY_FINGERPRINTED",
        "workload_semantics_hash": fingerprint({"semantics": profile["semantics"], "contract": profile["contract"]}),
        "optimization_fingerprint": fingerprint({"topology": profile["topology"], "memory": profile["memory"], "runtime": profile["runtime"]}),
        "source_provenance": profile["source"], "source_layout": source_layout(root, profile),
        "historical_evidence": {**profile["historical"], "valid_for_integration_head": False},
        "validation": {"historical_source_validated": True, "integration_ci_validated": "CHECK_EXACT_HEAD_ACTIONS",
                       "server_revalidation": GPU_NOT_RUN, "asset_verification": "NOT_RUN"},
        "native_plan": native_plan, "run_id": run_id, "mode": mode, "fully_resolved_profile": profile, "nodes": [],
        "asset_fingerprint": None,
    }
    if assets_path is None:
        receipt["scope"] = "SOURCE_ONLY_NOT_RUNNABLE"
        return receipt
    assets = load_assets(assets_path, profile)
    receipt["asset_fingerprint"] = fingerprint(assets["assets"])
    receipt["asset_config"] = assets
    receipt["scope"] = "RESOLVED_NOT_EXECUTED"
    correctness = assets.get("correctness")
    if mode in {"timing", "diagnostic-timing"}:
        _require(isinstance(correctness, dict) and set(correctness) == {"path", "sha256"},
                 "timing requires an explicit correctness receipt path and SHA256")
        _require(Path(correctness["path"]).is_absolute() and bool(SHA256_RE.fullmatch(correctness["sha256"])),
                 "invalid correctness receipt binding")
    output = str(Path(assets["run_root"]) / run_id)
    for i, n in enumerate(assets["nodes"]):
        rr = Path(n["repo_root"])
        env = dict(profile["runtime"]["environment"])
        env.update({d["environment"]: assets["assets"][k]["path"] for k, d in profile["required_assets"].items()})
        env.update(DMD_SYSTEM_ROOT=str(rr), DISTILLSCHEDULE_PYENV=n["python_env"],
                   DISTILLSCHEDULE_RUN_ROOT=output, DISTILLSCHEDULE_OUTPUT_DIR=output,
                   DISTILLSCHEDULE_RUN_ID=run_id, DISTILLSCHEDULE_RUN_KIND=mode,
                   DISTILLSCHEDULE_PYTHON=str(Path(n["python_env"])/"bin/python"),
                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                   PYTHONDONTWRITEBYTECODE="1", DISTILLSCHEDULE_NODE_LOCK="/tmp/distillschedule_gpu_node.lock",
                   TRITON_CACHE_DIR=str(Path(assets["node_local_cache_root"])/workload/run_id/f"node{i}"/"triton"))
        if profile["model_family"] in {"helios", "wan2.1"}:
            env["HELIOS_RUNTIME_ROOT"] = str(rr/"third_party/helios_compat/source")
            env["DISTILLSCHEDULE_ASSETS"] = str(Path(env["HF_HOME"]).parent)
            env["DISTILLSCHEDULE_GAN_CONFIG"] = str(rr/"configs/distillschedule/templates/helios_stage3_gan_384x640_systems.yaml")
        if workload == "minimax_h3_grid1000_best":
            env["LIGHTX2V_ROOT"] = str(rr/"third_party/lightx2v")
            env["H3_OUTPUT_DIR"] = str(Path(output)/"h3")
            env["H3_CONFIG"] = str(Path(output)/f"h3-config-node{i}.json")
        if workload == "livewan_crossscale_pipeline_best":
            # Source overlay is generated from pinned submodules, never an old experimental worktree.
            source_receipt = _object(Path(env["LIVEWAN_SOURCE_RECEIPT"]))
            _require(source_receipt.get("status") == "PASS", "LiveWan source receipt is not PASS")
            env["LIVEWAN_WAN21_SOURCE"] = str(source_receipt["source_root"])
            _require(Path(env["LIVEWAN_WAN21_SOURCE"]).is_absolute(), "prepared Wan source must be absolute")
        if correctness:
            bindings = ("DISTILLSCHEDULE_WAN13B_FAIR_CORRECTNESS_RECEIPT", "DISTILLSCHEDULE_WAN13B_FAIR_CORRECTNESS_SHA256") if workload.startswith("wan13b") else (
                ("LIVEWAN_PAIR_CORRECTNESS_RECEIPT", "LIVEWAN_PAIR_CORRECTNESS_SHA256") if workload.startswith("livewan") else
                ("DISTILLSCHEDULE_FSDP2_CORRECTNESS_SUMMARY", "DISTILLSCHEDULE_FSDP2_CORRECTNESS_SHA256"))
            env[bindings[0]], env[bindings[1]] = correctness["path"], correctness["sha256"]
        pp = [str(rr/"src"), str(rr)]
        if profile["model_family"] in {"helios", "wan2.1"}:
            pp.append(env["HELIOS_RUNTIME_ROOT"])
        if workload == "minimax_h3_grid1000_best":
            pp += [env["LIGHTX2V_ROOT"], str(Path(env["LIGHTX2V_ROOT"])/"lightx2v_train")]
        if workload == "livewan_crossscale_pipeline_best":
            pp = [env["LIVEWAN_PYTHON_OVERLAY"], *pp, str(rr/"third_party/livewan"), env["LIVEWAN_WAN21_SOURCE"]]
        env["PYTHONPATH"] = os.pathsep.join(pp)
        repl = {"node_rank":str(i), "run_id":run_id, "mode":mode, "head":head,
                "master_addr":assets["master_addr"], "master_port":str(assets["master_port"]),
                "output_dir":env.get("H3_OUTPUT_DIR",output),
                **{"asset:"+k:v["path"] for k,v in assets["assets"].items()}}
        args = _substitute(profile["runtime"]["arguments"], repl)
        node = {"node_rank": i, "host": n["host"], "repo_root":str(rr), "python":env["DISTILLSCHEDULE_PYTHON"],
                "environment":env, "command":["bash",str(rr/profile["runtime"]["launcher"]),*args]}
        if "model_config" in profile["runtime"]:
            node["resolved_model_config"] = _substitute(profile["runtime"]["model_config"],repl)
        if native_plan is not None:
            _require(native_plan["digest"] == profile["runtime"]["typed_plan_template_sha256"], "LiveWan typed plan changed without canonical update")
            def expand_native(value: Any) -> Any:
                if isinstance(value, str):
                    def subst(match: Any) -> str:
                        name = match.group(1)
                        _require(name in env, f"native plan variable not explicitly resolved: {name}")
                        return env[name]
                    return re.sub(r"\$\{ENV:([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}", subst, value)
                if isinstance(value, list): return [expand_native(x) for x in value]
                if isinstance(value, dict): return {k:expand_native(v) for k,v in value.items()}
                return value
            from distillschedule.core.plan import ExecutionPlan
            node["resolved_native_plan"] = ExecutionPlan.from_mapping(expand_native(native_plan["template"])).to_dict()
        receipt["nodes"].append(node)
    return receipt


def write_new_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Do not overwrite prior run receipts or resolved launch descriptions.
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
