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
# The H3 child is launched from an allowlisted environment. GCC may be found
# by Python's fallback search while its linker still requires an exported PATH.
# Do not inherit caller/virtualenv/CUDA paths into this system-toolchain search.
H3_SYSTEM_TOOLCHAIN_PATH = "/usr/bin:/bin"


def repository_root() -> Path:
    root = Path(__file__).resolve().parents[4]
    if not (root / "src/distillschedule").is_dir() or not (root / "pyproject.toml").is_file():
        raise ValueError(
            "Native/legacy operations require an editable DMD-System source checkout; "
            "install with pip install -e . there. Installed-wheel discovery is available "
            "through distillctl workloads list/describe."
        )
    return root


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
        _require(value["contract"]["continuous_cycles"] == 2,
                 "LiveWan canonical timing requires same-process warmup and measured cycle2")
        _require(value["runtime"]["native_run_modes"] == {
            "correctness": "trajectory",
            "diagnostic-timing": "two-cycle-timing"}, "LiveWan canonical native mode binding changed")
        _require("timing" not in value["allowed_run_modes"],
                 "numerical REVIEW cannot silently authorize strict timing")
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
    return validate_asset_config(_object(path), profile)


def validate_asset_config(value: dict[str, Any], profile: Mapping[str, Any]) -> dict[str, Any]:
    """Validate before an RPC persists an assets config; no filesystem writes."""
    allowed = {"schema_version", "nodes", "master_addr", "master_port", "run_root", "node_local_cache_root", "assets", "correctness", "trajectory_comparison", "timeout_seconds"}
    _require(not (set(value) - allowed), f"unknown assets fields (performance overrides forbidden): {sorted(set(value)-allowed)}")
    _require(value.get("schema_version") == 1, "assets schema_version must be 1")
    nodes = value.get("nodes")
    _require(isinstance(nodes, list) and len(nodes) == 2, "exactly two nodes are required")
    _require(nodes[0].get("host") != nodes[1].get("host"), "nodes must use different SSH hosts")
    for n in nodes:
        _require({"host", "repo_root", "python_env"}.issubset(n)
                 and not set(n) - {"host", "repo_root", "python_env", "ssh_port", "address_family", "python_path_overlays", "rpc_transport"},
                 "node must specify host, repo_root, python_env; optional transport and dependency overlays only")
        _require(type(n.get("ssh_port", 22)) is int and 1 <= n.get("ssh_port", 22) <= 65535, "invalid SSH port")
        _require(n.get("address_family", "auto") in {"auto", "ipv4", "ipv6"}, "invalid SSH address family")
        _require(n.get("rpc_transport", "stdin") in {"stdin", "argv-json-tty-v1"}, "invalid metadata RPC transport")
        _require(bool(re.fullmatch(r"[A-Za-z0-9_.@:-]+", str(n["host"]))) and not n["host"].startswith("-"), "unsafe SSH host")
        for k in ["repo_root", "python_env"]:
            _require(isinstance(n[k], str) and Path(n[k]).is_absolute(), f"node {k} must be absolute")
        overlays = n.get("python_path_overlays", [])
        _require(isinstance(overlays, list), "python_path_overlays must be an explicit ordered list")
        _require(not overlays or profile["id"] == "minimax_h3_grid1000_best",
                 "dependency overlays currently supported only by the H3 native launcher")
        paths = []
        for overlay in overlays:
            _require(isinstance(overlay, dict) and set(overlay) == {"path", "identity_file", "sha256"},
                     "dependency overlay needs exactly path, identity_file, sha256")
            path = overlay["path"]
            _require(isinstance(path, str) and Path(path).is_absolute() and Path(path) != Path('/')
                     and '..' not in Path(path).parts and os.pathsep not in path and '\0' not in path,
                     "dependency overlay path must be an absolute Python path entry")
            name = overlay["identity_file"]
            _require(isinstance(name, str) and name not in {'', '.'} and not Path(name).is_absolute()
                     and '..' not in Path(name).parts and '\0' not in name,
                     "dependency overlay identity_file must be relative inside the overlay")
            _require(bool(SHA256_RE.fullmatch(str(overlay["sha256"]))), "dependency overlay needs SHA256")
            paths.append(str(Path(path)))
        _require(len(paths) == len(set(paths)), "duplicate dependency overlay path")
    for k in ["run_root", "node_local_cache_root"]:
        _require(isinstance(value.get(k), str) and Path(value[k]).is_absolute()
                 and Path(value[k]) not in (Path('/'), Path('/home/lyy')),
                 f"{k} must be an explicit project subtree, not a filesystem/home root")
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


def asset_identity(assets: Mapping[str, Any]) -> Any:
    """Keep legacy fingerprints; bind opt-in dependency choices into all costs.

    Node order and overlay order are significant. Transport/output paths stay
    out of the identity, as before. No overlay is inferred from the caller's
    environment, and no historical cost can be reused after one is added.
    """
    overlays = [n.get("python_path_overlays", []) for n in assets.get("nodes", [])]
    return ({"assets": assets["assets"], "python_path_overlays_by_node": overlays}
            if any(overlays) else assets["assets"])


def verify_python_path_overlays(node: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Verify the local node only: another node's /tmp need not be mounted."""
    result = []
    for overlay in node.get("python_path_overlays", []):
        path = Path(overlay["path"])
        identity = path / overlay["identity_file"]
        _require(path.is_dir() and identity.is_file() and file_sha256(identity) == overlay["sha256"],
                 f"dependency overlay fingerprint mismatch: {path}")
        result.append({**overlay, "resolved_identity_file": str(identity.resolve()),
                       "scope": "IDENTITY_FILE_ONLY_NOT_ALL_PACKAGE_BYTES"})
    return result


def _python_path(profile: Mapping[str, Any], node: Mapping[str, Any], env: Mapping[str, str]) -> str:
    root = Path(node["repo_root"])
    pp = [str(root / "src"), str(root)]
    if profile["model_family"] in {"helios", "wan2.1"}:
        pp.append(env["HELIOS_RUNTIME_ROOT"])
    if profile["id"] == "minimax_h3_grid1000_best":
        pp += [env["LIGHTX2V_ROOT"], str(Path(env["LIGHTX2V_ROOT"]) / "lightx2v_train")]
        if env.get("H3_DIFFUSERS_SOURCE"):
            pp.insert(0, str(Path(env["H3_DIFFUSERS_SOURCE"]) / "src"))
    if profile["id"] == "livewan_crossscale_pipeline_best":
        pp = [env["LIVEWAN_PYTHON_OVERLAY"], *pp, str(root / "third_party/livewan"), env["LIVEWAN_WAN21_SOURCE"]]
    return os.pathsep.join([o["path"] for o in node.get("python_path_overlays", [])] + pp)


def validate_resolved_asset_binding(receipt: Mapping[str, Any], profile: Mapping[str, Any]) -> None:
    """Portable consistency only; physical admission belongs to the asset host."""
    assets = validate_asset_config(receipt["asset_config"], profile)
    _require(receipt["asset_fingerprint"] == fingerprint(asset_identity(assets)), "resolved asset/dependency identity changed")
    _require(len(receipt["nodes"]) == len(assets["nodes"]), "resolved node count changed")
    bindings = _portable_request_bindings(receipt, profile, assets)
    helios_trace = bindings.get("original_helios_trace_binding")
    for node, spec in zip(receipt["nodes"], assets["nodes"]):
        _require(node["repo_root"] == spec["repo_root"]
                 and node["python"] == str(Path(spec["python_env"]) / "bin/python"), "resolved source/python binding changed")
        _require(node["environment"]["PYTHONPATH"] == _python_path(profile, spec, node["environment"]),
                 "resolved dependency PYTHONPATH changed")
        if helios_trace is not None:
            key = _original_helios_trace_hash_environment(profile)
            _require(node["environment"].get(key) == helios_trace["bundle_hash"],
                     "resolved ORIGINAL Helios expected trace hash changed")
        if profile["id"] == "minimax_h3_grid1000_best":
            _require(node["environment"].get("PATH") == H3_SYSTEM_TOOLCHAIN_PATH,
                     "resolved H3 system toolchain PATH changed")


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


def _original_helios_trace_hash_environment(profile):
    return ("DISTILLSCHEDULE_GAN_TRACE_HASH" if profile["contract"]["variant"] == "gan"
            else "DISTILLSCHEDULE_TRACE_BUNDLE_HASH")


def _original_helios_trace_binding(profile, assets):
    """Only explicit ORIGINAL requests replace canonical trace hashes.

    The bundle file SHA is the existing asset fingerprint, not a caller env
    override. Its embedded content hash and every request-qualified rank
    manifest are independently checked before constructing a launch.
    """
    from distillschedule.search.helios_request import TRACE_KEY, OriginalHeliosRequest
    if profile["id"] not in {"helios14b_nogan_best", "helios14b_gan_best"} or TRACE_KEY not in profile["contract"]:
        return None
    from runtime.static_world16_trace import StaticWorld16Trace
    request = OriginalHeliosRequest.from_dict(profile["contract"][TRACE_KEY])
    spec = assets["assets"]["trace"]
    _require(spec.get("identity_file") == "bundle.json", "ORIGINAL Helios trace identity must be bundle.json")
    identity = Path(spec["path"]) / "bundle.json"
    _require(identity.is_file() and file_sha256(identity) == spec["sha256"],
             "ORIGINAL Helios trace bundle file SHA256 mismatch")
    trace = StaticWorld16Trace(spec["path"], variant=request.variant, request=request)
    return dict(bundle_hash=trace.bundle_hash, bundle_file_sha256=spec["sha256"], request_sha256=request.digest)


def _livewan_request_trace_binding(profile, assets, *, root):
    """Always reread physical request assets on RPC resolution/local admission."""
    from distillschedule.search.livewan_request import LiveWanRequest, routing_trace_metadata
    from distillschedule.runtime.livewan_crossscale_worker import FormalTrace, Bootstrap, validate_source_receipt
    request = LiveWanRequest.from_dict(profile["contract"]["request"])
    spec = assets["assets"]["trace"]
    _require(file_sha256(Path(spec["path"])) == spec["sha256"], "LiveWan request trace file SHA mismatch")
    trace = FormalTrace.load(spec["path"], expected_cycles=2, expected_global_batch=16, request=request)
    boot_spec, source_spec = assets["assets"]["bootstrap"], assets["assets"]["source_receipt"]
    _require(file_sha256(Path(boot_spec["path"])) == boot_spec["sha256"], "LiveWan request bootstrap file SHA mismatch")
    boot = Bootstrap.load(boot_spec["path"], trace=trace)
    _require(boot.value["trace"]["path"] == str(trace.path)
             and boot.value["source_receipt"]["path"] == source_spec["path"],
             "LiveWan request bootstrap/trace/source asset paths differ")
    _require(file_sha256(Path(source_spec["path"])) == source_spec["sha256"], "LiveWan request source file SHA mismatch")
    source = validate_source_receipt(boot, repo_root=root)
    _require(Path(source["source_root"]).is_absolute(), "prepared Wan source must be absolute")
    return dict(request=request.to_dict(), request_sha256=fingerprint(request.to_dict()),
        trace_sha256=trace.sha256, trace_file_sha256=spec["sha256"],
        routing_metadata=routing_trace_metadata(trace, request),
        bootstrap=dict(path=boot_spec["path"], file_sha256=boot_spec["sha256"], sha256=boot.sha256,
                       trace=dict(boot.value["trace"]), source_receipt=dict(boot.value["source_receipt"])),
        source_receipt=dict(path=source_spec["path"], file_sha256=source_spec["sha256"],
                            sha256=source["source_receipt_sha256"], source_root=source["source_root"]))


def _portable_request_bindings(receipt, profile, assets):
    """Check serialized facts, not caller claims of successful file validation.

    Digests and the routing derivation are bound to the recipe/asset inventory.
    Only the asset host can prove these facts against the original file bytes.
    """
    from distillschedule.search.helios_request import TRACE_KEY, OriginalHeliosRequest
    bindings = {}
    if profile["id"] in {"helios14b_nogan_best", "helios14b_gan_best"} and TRACE_KEY in profile["contract"]:
        request = OriginalHeliosRequest.from_dict(profile["contract"][TRACE_KEY])
        spec = assets["assets"]["trace"]
        trace = receipt.get("original_helios_trace_binding")
        _require(isinstance(trace, dict) and set(trace) == {"bundle_hash", "bundle_file_sha256", "request_sha256"}
                 and bool(SHA256_RE.fullmatch(str(trace.get("bundle_hash", ""))))
                 and trace["request_sha256"] == request.digest and trace["bundle_file_sha256"] == spec["sha256"]
                 and spec.get("identity_file") == "bundle.json", "resolved ORIGINAL Helios trace binding changed")
        bindings["original_helios_trace_binding"] = trace
    if profile["id"] == "livewan_crossscale_pipeline_best" and "request" in profile["contract"]:
        from distillschedule.search.livewan_request import LiveWanRequest, validate_routing_trace_metadata
        request = LiveWanRequest.from_dict(profile["contract"]["request"])
        trace = receipt.get("livewan_request_trace_binding")
        _require(isinstance(trace, dict) and set(trace) == {"request", "request_sha256", "trace_sha256",
            "trace_file_sha256", "routing_metadata", "bootstrap", "source_receipt"}, "LiveWan request trace binding fields changed")
        a = assets["assets"]
        _require(trace["request"] == request.to_dict() and trace["request_sha256"] == fingerprint(request.to_dict())
                 and trace["trace_file_sha256"] == a["trace"]["sha256"]
                 and bool(SHA256_RE.fullmatch(str(trace["trace_sha256"]))), "LiveWan request trace binding changed")
        validate_routing_trace_metadata(trace["routing_metadata"], request, trace["trace_sha256"])
        boot, source = trace["bootstrap"], trace["source_receipt"]
        _require(isinstance(boot, dict) and set(boot) == {"path", "file_sha256", "sha256", "trace", "source_receipt"}
                 and isinstance(source, dict) and set(source) == {"path", "file_sha256", "sha256", "source_root"},
                 "LiveWan bootstrap/source metadata fields changed")
        for name, meta in (("bootstrap", boot), ("source_receipt", source)):
            _require(meta["path"] == a[name]["path"] and meta["file_sha256"] == a[name]["sha256"]
                     and bool(SHA256_RE.fullmatch(str(meta["sha256"]))), "LiveWan bootstrap/source asset binding changed")
        _require(boot["trace"] == dict(path=a["trace"]["path"], sha256=trace["trace_sha256"])
                 and boot["source_receipt"] == dict(path=source["path"], sha256=source["sha256"])
                 and isinstance(source["source_root"], str) and Path(source["source_root"]).is_absolute(),
                 "LiveWan bootstrap/trace/source metadata differ")
        bindings["livewan_request_trace_binding"] = trace
    return bindings


def validate_request_configuration(receipt, profile, *, root, verify_physical=False):
    """Rebuild the complete request launch, with filesystem proof only locally."""
    bindings = _portable_request_bindings(receipt, profile, receipt["asset_config"])
    if not bindings:
        return
    assets = validate_asset_config(receipt["asset_config"], profile)
    rebuilt = (resolve_configuration(profile, root=root, asset_config=assets,
                    run_id=receipt["run_id"], mode=receipt["mode"]) if verify_physical else
               _resolve_configuration(profile, root=root, asset_config=assets,
                    run_id=receipt["run_id"], mode=receipt["mode"], request_bindings=bindings))
    # Search adds these descriptive fields/semantics hash after compilation;
    # validated_profile independently checks the candidate's exact semantics.
    for key, value in rebuilt.items():
        if key not in {"canonical_status", "canonical_scope", "workload_semantics_hash"}:
            _require(receipt.get(key) == value, f"request launch configuration changed: {key}")


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
    return resolve_configuration(profile, root=root, assets_path=assets_path, run_id=run_id, mode=mode)


def resolve_configuration(profile: dict[str, Any], *, root: Path, assets_path: Path | None = None,
                          asset_config: dict[str, Any] | None = None,
                          run_id: str = "source-inspection", mode: str = "correctness") -> dict[str, Any]:
    """Asset-host resolver: request asset verification is never optional."""
    _require(assets_path is None or asset_config is None, "provide file or in-memory assets, not both")
    assets = (load_assets(assets_path, profile) if assets_path is not None else
              validate_asset_config(asset_config, profile) if asset_config is not None else None)
    bindings = {}
    if assets is not None:
        helios_trace = _original_helios_trace_binding(profile, assets)
        if helios_trace is not None:
            bindings["original_helios_trace_binding"] = helios_trace
        if profile["id"] == "livewan_crossscale_pipeline_best" and "request" in profile["contract"]:
            bindings["livewan_request_trace_binding"] = _livewan_request_trace_binding(profile, assets, root=root)
    return _resolve_configuration(profile, root=root, asset_config=assets, run_id=run_id, mode=mode,
                                  request_bindings=bindings)


def _resolve_configuration(profile, *, root, asset_config, run_id, mode, request_bindings):
    """Shared compiler; dispatcher reconstructs the canonical/registered recipe."""
    workload = profile["id"]
    _require(bool(SAFE_ID.fullmatch(run_id)) and run_id not in {".", ".."}, "unsafe run id")
    _require(mode in profile["allowed_run_modes"], f"mode {mode} is not authorized for {workload}")
    native_mode = profile["runtime"].get("native_run_modes", {}).get(mode, mode)
    head = git_head(root)
    native_plan = None
    if "typed_plan" in profile["runtime"]:
        from distillschedule.core.config import _load_with_includes
        files: list[Path] = []
        raw_plan = _load_with_includes(root / profile["runtime"]["typed_plan"], stack=(), sources=files)
        template_digest = fingerprint(raw_plan)
        if workload == "livewan_crossscale_pipeline_best" and "request" in profile["contract"]:
            from distillschedule.search.livewan_request import LiveWanRequest, lower_native_plan
            raw_plan = lower_native_plan(raw_plan, LiveWanRequest.from_dict(profile["contract"]["request"]))
        native_plan = {"template": raw_plan, "digest": fingerprint(raw_plan), "template_digest": template_digest,
                       "template_file_sha256": file_sha256(root / profile["runtime"]["typed_plan"]),
                       "source_files": [str(f.relative_to(root)) for f in dict.fromkeys(files)]}
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
        "native_plan": native_plan, "run_id": run_id, "mode": mode, "native_mode": native_mode,
        "fully_resolved_profile": profile, "nodes": [],
        "asset_fingerprint": None,
    }
    if asset_config is None:
        receipt["scope"] = "SOURCE_ONLY_NOT_RUNNABLE"
        return receipt
    assets = validate_asset_config(asset_config, profile)
    helios_trace = request_bindings.get("original_helios_trace_binding")
    if helios_trace is not None:
        receipt["original_helios_trace_binding"] = helios_trace
    receipt["asset_fingerprint"] = fingerprint(asset_identity(assets))
    receipt["asset_config"] = assets
    receipt["scope"] = "RESOLVED_NOT_EXECUTED"
    if workload == "livewan_crossscale_pipeline_best" and "request" in profile["contract"]:
        from distillschedule.search.livewan_request import LiveWanRequest, lower_native_plan
        request = LiveWanRequest.from_dict(profile["contract"]["request"])
        trace_binding = request_bindings["livewan_request_trace_binding"]
        native_plan["template"] = lower_native_plan(native_plan["template"], request,
            routing_metadata=trace_binding["routing_metadata"])
        native_plan["digest"] = fingerprint(native_plan["template"])
        receipt["livewan_request_trace_binding"] = trace_binding
    correctness = assets.get("correctness")
    comparison = assets.get("trajectory_comparison")
    if workload.startswith("wan13b") and native_mode in {"two-cycle-timing", "two-cycle-calibration"}:
        _require(isinstance(comparison, dict) and set(comparison) == {"path", "sha256"}
                 and Path(comparison["path"]).is_absolute()
                 and bool(SHA256_RE.fullmatch(comparison["sha256"])),
                 "Wan two-cycle timing requires a hash-bound native trajectory comparison")
    elif comparison is not None:
        _require(False, "trajectory_comparison is only used by Wan native two-cycle timing")
    if mode in {"timing", "diagnostic-timing", "calibration", "native"}:
        _require(isinstance(correctness, dict) and set(correctness) == {"path", "sha256"},
                 "timing requires an explicit correctness receipt path and SHA256")
        _require(Path(correctness["path"]).is_absolute() and bool(SHA256_RE.fullmatch(correctness["sha256"])),
                 "invalid correctness receipt binding")
    output = str(Path(assets["run_root"]) / run_id)
    for i, n in enumerate(assets["nodes"]):
        rr = Path(n["repo_root"])
        env = dict(profile["runtime"]["environment"])
        if helios_trace is not None:
            env[_original_helios_trace_hash_environment(profile)] = helios_trace["bundle_hash"]
        if profile["id"].startswith("helios14b_") and "planner_implementation" in profile["runtime"]:
            _require(env.get("DISTILLSCHEDULE_HELIOS_PLANNER_MODE") == "off", "Helios planner must default off")
            env["DISTILLSCHEDULE_HELIOS_PLANNER_MODE"] = "native" if mode == "diagnostic-timing" else mode
        if workload == "livewan_crossscale_pipeline_best" and mode in {"calibration", "action-observation"}:
            _require(env.get("DISTILLSCHEDULE_LIVEWAN_OPERATION_CALIBRATION") == "0",
                     "only source-registered LiveWan candidates support operation calibration")
            env["DISTILLSCHEDULE_LIVEWAN_OPERATION_CALIBRATION"] = "1"
        if workload == "minimax_h3_grid1000_best" and mode == "calibration":
            _require(env.get("H3_PLANNER_CALIBRATION") == "0", "H3 calibration requires generated native adapter")
            env["H3_PLANNER_CALIBRATION"] = "1"
        env.update({d["environment"]: assets["assets"][k]["path"] for k, d in profile["required_assets"].items()})
        env.update(DMD_SYSTEM_ROOT=str(rr), DISTILLSCHEDULE_PYENV=n["python_env"],
                   DISTILLSCHEDULE_RUN_ROOT=output, DISTILLSCHEDULE_OUTPUT_DIR=output,
                   DISTILLSCHEDULE_RUN_ID=run_id, DISTILLSCHEDULE_RUN_KIND=native_mode,
                   DISTILLSCHEDULE_PYTHON=str(Path(n["python_env"])/"bin/python"),
                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                   PYTHONDONTWRITEBYTECODE="1", DISTILLSCHEDULE_NODE_LOCK="/tmp/distillschedule_gpu_node.lock",
                   TRITON_CACHE_DIR=str(Path(assets["node_local_cache_root"])/workload/run_id/f"node{i}"/"triton"))
        if profile["model_family"] in {"helios", "wan2.1"}:
            env["HELIOS_RUNTIME_ROOT"] = str(rr/"third_party/helios_compat/source")
            env["DISTILLSCHEDULE_ASSETS"] = str(Path(env["HF_HOME"]).parent)
            env["DISTILLSCHEDULE_GAN_CONFIG"] = str(rr/"configs/distillschedule/templates/helios_stage3_gan_384x640_systems.yaml")
        if workload == "minimax_h3_grid1000_best":
            env["PATH"] = H3_SYSTEM_TOOLCHAIN_PATH
            env["LIGHTX2V_ROOT"] = str(rr/"third_party/lightx2v")
            env["H3_OUTPUT_DIR"] = str(Path(output)/"h3")
            env["H3_CONFIG"] = str(Path(output)/f"h3-config-node{i}.json")
            if "diffusers_source" in assets["assets"]:
                env["H3_DIFFUSERS_SOURCE"] = assets["assets"]["diffusers_source"]["path"]
                env["LD_LIBRARY_PATH"] = os.pathsep.join((
                    str(Path(n["python_env"])/"lib/python3.10/site-packages/torch/lib"),
                    "/usr/lib/x86_64-linux-gnu", "/usr/local/cuda/targets/x86_64-linux/lib"))
                env.update(USE_TF="0", TRANSFORMERS_NO_TF="1", PYTHONNOUSERSITE="1")
        if workload == "livewan_crossscale_pipeline_best":
            # Source overlay is generated from pinned submodules, never an old experimental worktree.
            source_receipt = (request_bindings["livewan_request_trace_binding"]["source_receipt"]
                              if "request" in profile["contract"] else _object(Path(env["LIVEWAN_SOURCE_RECEIPT"])))
            if "request" not in profile["contract"]:
                _require(source_receipt.get("status") == "PASS", "LiveWan source receipt is not PASS")
            env["LIVEWAN_WAN21_SOURCE"] = str(source_receipt["source_root"])
            _require(Path(env["LIVEWAN_WAN21_SOURCE"]).is_absolute(), "prepared Wan source must be absolute")
        if correctness:
            bindings = ("DISTILLSCHEDULE_WAN13B_FAIR_CORRECTNESS_RECEIPT", "DISTILLSCHEDULE_WAN13B_FAIR_CORRECTNESS_SHA256") if workload.startswith("wan13b") else (
                ("LIVEWAN_PAIR_CORRECTNESS_RECEIPT", "LIVEWAN_PAIR_CORRECTNESS_SHA256") if workload.startswith("livewan") else
                ("DISTILLSCHEDULE_FSDP2_CORRECTNESS_SUMMARY", "DISTILLSCHEDULE_FSDP2_CORRECTNESS_SHA256"))
            env[bindings[0]], env[bindings[1]] = correctness["path"], correctness["sha256"]
        if comparison:
            env["DISTILLSCHEDULE_WAN13B_TRAJECTORY_COMPARISON_RECEIPT"] = comparison["path"]
            env["DISTILLSCHEDULE_WAN13B_TRAJECTORY_COMPARISON_SHA256"] = comparison["sha256"]
        env["PYTHONPATH"] = _python_path(profile, n, env)
        repl = {"node_rank":str(i), "run_id":run_id, "mode":native_mode, "head":head,
                "master_addr":assets["master_addr"], "master_port":str(assets["master_port"]),
                "output_dir":env.get("H3_OUTPUT_DIR",output),
                **{"asset:"+k:v["path"] for k,v in assets["assets"].items()}}
        args = _substitute(profile["runtime"]["arguments"], repl)
        node = {"node_rank": i, "host": n["host"], "repo_root":str(rr), "python":env["DISTILLSCHEDULE_PYTHON"],
                "environment":env, "command":["bash",str(rr/profile["runtime"]["launcher"]),*args]}
        for option in ("ssh_port", "address_family"):
            if option in n:
                node[option] = n[option]
        if "model_config" in profile["runtime"]:
            node["resolved_model_config"] = _substitute(profile["runtime"]["model_config"],repl)
        if native_plan is not None:
            _require(native_plan["template_digest"] == profile["runtime"]["typed_plan_template_sha256"], "LiveWan typed plan changed without canonical update")
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
            node["resolved_native_plan_sha256"] = fingerprint(node["resolved_native_plan"])
            if workload == "livewan_crossscale_pipeline_best" and "request" in profile["contract"]:
                env["DISTILLSCHEDULE_LIVEWAN_LAUNCH_PLAN"] = str(Path(output) / f"launch-native-plan.node{i}.json")
                env["DISTILLSCHEDULE_LIVEWAN_LAUNCH_PLAN_SHA256"] = fingerprint(node["resolved_native_plan"])
                env["DISTILLSCHEDULE_LIVEWAN_REQUEST_ID"] = request.id
        receipt["nodes"].append(node)
    return receipt


def write_new_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Do not overwrite prior run receipts or resolved launch descriptions.
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
