"""Direct-training assets, independent of historical benchmark launch manifests.

This is an asset/admission contract, not a new checkpoint format or optimizer.
The original Step10 package and original model/optimizer/EMA loaders remain the
owners. File hashes are operator-supplied; no experiment checkpoint is inferred.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from distillschedule.core.contracts import require
from distillschedule.workloads.best_known.registry import SHA256_RE, file_sha256

SCHEMA = "distillschedule.helios_reference_assets.v1"
IDENTITIES = {"generator_model": "transformer_ode/config.json", "score_model": "transformer/config.json",
    "generator_precast": "distillschedule_precast_manifest.json",
    "score_precast": "distillschedule_precast_manifest.json",
    "logical_checkpoint": "manifest.json", "negative_prompt": "manifest.json"}


def _require(value, message):
    require(value, message, path="helios.reference.assets")


def _file(root, relative):
    _require(isinstance(relative, str) and relative not in ("", ".") and not Path(relative).is_absolute()
             and ".." not in Path(relative).parts, "asset payload must stay inside its package")
    path = (root / relative).resolve()
    _require(root.resolve() in path.parents, "asset payload escaped its package")
    return path


def asset_config(value):
    _require(isinstance(value, dict) and set(value) == {"schema_version", "assets"}
             and value["schema_version"] == SCHEMA, "explicit versioned direct-reference assets required")
    rows = value["assets"]
    _require(isinstance(rows, dict) and set(rows) == set(IDENTITIES), "direct-reference assets must contain only the six native roles")
    for role, identity in IDENTITIES.items():
        row = rows[role]
        _require(isinstance(row, dict) and set(row) == ({"path", "identity_file", "sha256", "ema_manifest"}
                 if role == "logical_checkpoint" else {"path", "identity_file", "sha256"}), f"unsupported {role} asset fields")
        _require(isinstance(row["path"], str) and Path(row["path"]).is_absolute()
                 and ".." not in Path(row["path"]).parts and "\0" not in row["path"]
                 and row["identity_file"] == identity
                 and bool(SHA256_RE.fullmatch(str(row["sha256"]))), f"explicit {role} path/identity/SHA256 required")
    ema = rows["logical_checkpoint"]["ema_manifest"]
    _require(isinstance(ema, dict) and set(ema) == {"identity_file", "sha256"}
             and ema["identity_file"] in ("ema/sparse_ema_manifest.json", "model_ema/sparse_ema_manifest.json")
             and bool(SHA256_RE.fullmatch(str(ema["sha256"]))), "explicit original sparse EMA manifest pin required")
    return value


def verify_pins(value):
    rows = asset_config(value)["assets"]
    observations = []
    for role, row in rows.items():
        root = Path(row["path"])
        path = _file(root, row["identity_file"])
        _require(root.is_dir() and path.is_file() and file_sha256(path) == row["sha256"], f"asset fingerprint mismatch: {role}: {path}")
        observations.append({"name":role, "path":str(root), "identity_file":str(path), "sha256":row["sha256"],
                             "scope":"IDENTITY_FILE_ONLY_PAYLOADS_CHECKED_BY_NATIVE_WORKER"})
    row = rows["logical_checkpoint"]
    path = _file(Path(row["path"]), row["ema_manifest"]["identity_file"])
    _require(path.is_file() and file_sha256(path) == row["ema_manifest"]["sha256"], "sparse EMA manifest fingerprint mismatch")
    return observations


def load_bootstrap(value, *, verify_payloads=True):
    """Admit actual original G2/C10/EMA10 bytes before constructing models.

    Metadata-only public validation never claims payload admission. The worker
    repeats file pins and verifies every used master/moment/EMA payload under
    the all-rank metadata plane. No old input trace identity is imposed.
    """
    from runtime.critic_pool_checkpoint import LogicalCheckpointPackage
    row = asset_config(value)["assets"]["logical_checkpoint"]
    root = Path(row["path"])
    _require(file_sha256(root / "manifest.json") == row["sha256"], "logical manifest fingerprint mismatch")
    package = LogicalCheckpointPackage.load(root)
    _require(type(package.manifest.get("global_step")) is int and package.manifest["global_step"] == 10,
             "original bootstrap global clock must remain Step10")
    identity = {"manifest_hash":package.manifest_hash, "manifest_sha256":row["sha256"], "payloads":{},
                "ema_manifest_sha256":row["ema_manifest"]["sha256"], "payload_hashes_verified":bool(verify_payloads)}
    for role, step in (("generator", 2), ("critic", 10)):
        component = package.component(role)
        _require(type(component.get("optimizer_step")) is int and component["optimizer_step"] == step
                 and type(component.get("parameter_version")) is int and component["parameter_version"] == step
                 and type(component.get("scheduler_state", {}).get("last_epoch")) is int
                 and component["scheduler_state"]["last_epoch"] == step,
                 f"original {role} parameter/optimizer/scheduler clocks must remain {step}")
        names, records = component.get("parameter_names"), component.get("records")
        _require(isinstance(names, list) and names and all(isinstance(n, str) for n in names)
                 and len(names) == len(set(names)) and isinstance(records, dict) and set(records) == set(names),
                 f"logical {role} trainable inventory changed")
        ranges = []
        for name in names:
            record = records[name]; shape = record["shape"]
            _require(isinstance(shape, list) and all(type(d) is int and d > 0 for d in shape)
                     and type(record["numel"]) is int and record["numel"] == math.prod(shape)
                     and type(record["offset_elements"]) is int and record["offset_elements"] >= 0,
                     f"logical {role} tensor record changed")
            ranges.append((record["offset_elements"], record["offset_elements"] + record["numel"]))
        ranges.sort()
        _require(all(a[1] <= b[0] for a,b in zip(ranges, ranges[1:]))
                 and sum(end-start for start,end in ranges) == component.get("logical_numel"),
                 f"logical {role} tensor coverage changed")
        identity["payloads"][role] = {}
        for name in ("master", "first_moment", "second_moment"):
            payload = component["payloads"][name]
            _require(payload.get("dtype") == "torch.float32" and type(payload.get("bytes")) is int
                     and payload["bytes"] > 0 and payload["bytes"] % 4 == 0
                     and ranges[-1][1] <= payload["bytes"] // 4
                     and bool(SHA256_RE.fullmatch(str(payload.get("sha256", "")))), f"logical {role}/{name} FP32 payload changed")
            path = _file(root, payload["path"])
            if verify_payloads:
                _require(path.stat().st_size == payload["bytes"] and file_sha256(path) == payload["sha256"],
                         f"logical payload size/hash mismatch: {role}/{name}")
            identity["payloads"][role][name] = payload["sha256"]
    ema_file = _file(root, row["ema_manifest"]["identity_file"])
    # Match the unchanged original loader's first-existing-directory rule.
    selected = next((p / "sparse_ema_manifest.json" for p in (root/"ema", root/"model_ema")
                     if (p / "sparse_ema_manifest.json").is_file()), None)
    _require(selected is not None and selected.resolve() == ema_file
             and file_sha256(ema_file) == row["ema_manifest"]["sha256"], "original sparse EMA source/pin changed")
    ema = json.loads(ema_file.read_text())
    _require(ema.get("schema_version") == "distillschedule.sparse_ema.v1"
             and type(ema.get("optimization_step")) is int and ema["optimization_step"] == 10,
             "original sparse EMA must remain EMA10")
    names = ema["tracked_names"]
    normalized = [n[7:] if n.startswith("module.") else n for n in names]
    records = package.component("generator")["records"]
    _require(len(normalized) == len(set(normalized)) and set(normalized) == set(records), "logical sparse EMA trainable inventory changed")
    identity["ema_shards"] = {}
    loaded = []
    for shard in ema["shards"]:
        path = _file(ema_file.parent, shard["file"])
        _require(shard["file"] not in identity["ema_shards"] and bool(SHA256_RE.fullmatch(str(shard.get("sha256", "")))),
                 "logical sparse EMA shard inventory changed")
        identity["ema_shards"][shard["file"]] = shard["sha256"]
        loaded.extend(shard["names"])
        if verify_payloads:
            _validate_ema_shard(path, shard, records)
    _require(len(loaded) == len(set(loaded)) and set(loaded) == set(names), "logical sparse EMA shard coverage changed")
    return package, identity


def _validate_ema_shard(path, shard, records):
    # Bound the lifetime of both the shard dictionary and its last tensor to
    # one call, before the next shard is deserialized on every rank.
    import torch
    _require(file_sha256(path) == shard["sha256"], "logical EMA shard hash mismatch")
    tensors = torch.load(path, map_location="cpu", weights_only=True)
    _require(list(tensors) == shard["names"], "logical EMA shard names changed")
    for name, tensor in tensors.items():
        normalized_name = name[7:] if name.startswith("module.") else name
        _require(isinstance(tensor, torch.Tensor) and list(tensor.shape) == records[normalized_name]["shape"],
                 "logical EMA tensor shape changed")
