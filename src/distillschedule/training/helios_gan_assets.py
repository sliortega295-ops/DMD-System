"""Explicit local assets for the original systems-only W3 bootstrap.

No historical path/run identity is required; the original loader still rejects
an official-checkpoint-1000 label for physical G2/C10 + EMA800 systems state.
"""
import json
import math
from pathlib import Path

from distillschedule.core.contracts import require
from distillschedule.workloads.best_known.registry import SHA256_RE, file_sha256
from . import helios_reference_assets as reference

SCHEMA = "distillschedule.helios_gan_stream_assets.v1"


def _reference(value):
    return {"schema_version": reference.SCHEMA,
            "assets": {k: v for k, v in value["assets"].items() if k not in {"gan_config", "fa3_kernel"}}}


def asset_config(value):
    require(isinstance(value, dict) and set(value) == {"schema_version", "assets"}
            and value["schema_version"] == SCHEMA
            and isinstance(value["assets"], dict)
            and set(value["assets"]) == set(reference.IDENTITIES) | {"gan_config", "fa3_kernel"},
            "explicit eight-role native GAN stream assets required", path="helios.gan.assets")
    reference.asset_config(_reference(value))
    row = value["assets"]["gan_config"]
    require(isinstance(row, dict) and set(row) == {"path", "sha256"}
            and isinstance(row["path"], str) and Path(row["path"]).is_absolute()
            and ".." not in Path(row["path"]).parts and "\0" not in row["path"]
            and bool(SHA256_RE.fullmatch(str(row["sha256"]))),
            "explicit GAN config file/SHA256 required", path="helios.gan.assets")
    from distillschedule.runtime.helios_gan_manual.backend_library import FA3_PINS
    kernel = value["assets"]["fa3_kernel"]
    require(isinstance(kernel, dict) and set(kernel) == {"path", "files_sha256"}
            and isinstance(kernel["path"], str) and Path(kernel["path"]).is_absolute()
            and ".." not in Path(kernel["path"]).parts and "\0" not in kernel["path"]
            and kernel["files_sha256"] == FA3_PINS, "explicit exact pinned native FA3 package required",
            path="helios.gan.assets")
    return value


def verify_pins(value):
    asset_config(value)
    observed = reference.verify_pins(_reference(value))
    row = value["assets"]["gan_config"]
    path = Path(row["path"])
    require(path.is_file() and file_sha256(path) == row["sha256"], "GAN config identity changed", path="helios.gan.assets")
    from .helios_gan_inputs import read_config
    read_config(path)
    kernel = value["assets"]["fa3_kernel"]
    require(Path(kernel["path"]).is_dir(), "explicit local FA3 directory missing", path="helios.gan.assets")
    return observed + [{"name": "gan_config", "path": str(path), "sha256": row["sha256"]},
        {"name": "fa3_kernel", **kernel, "scope": "DECLARED_PINS_ONLY; NATIVE_WORKER_VERIFIES_BYTES_PUBLIC_ABI_AND_MAPPED_LIBRARY"}]


def load_bootstrap(value, *, verify_payloads=True):
    from runtime.critic_pool_checkpoint import LogicalCheckpointPackage
    verify_pins(value)
    row = value["assets"]["logical_checkpoint"]
    root = Path(row["path"])
    package = LogicalCheckpointPackage.load_systems_only_gan(root)
    identity = {"classification": "systems_only_gan_bootstrap_NOT_OFFICIAL_CHECKPOINT1000",
                "manifest_hash": package.manifest_hash, "manifest_sha256": row["sha256"],
                "ema_manifest_sha256": row["ema_manifest"]["sha256"], "payload_hashes_verified": bool(verify_payloads),
                "physical_initial": {"generator": 2, "critic": 10, "ema": 800},
                "curriculum_initial": {"generator": 200, "critic": 1000, "ema": 800}, "payloads": {}}
    for role in ("generator", "critic"):
        component = package.component(role)
        names, records = component.get("parameter_names"), component.get("records")
        require(isinstance(names, list) and names and len(names) == len(set(names))
                and all(isinstance(n, str) and n for n in names) and isinstance(records, dict) and set(names) == set(records),
                "original GAN trainable inventory changed", path="helios.gan.assets")
        ranges = []
        for name in names:
            record = records[name]; shape = record["shape"]
            require(isinstance(shape, list) and all(type(d) is int and d > 0 for d in shape)
                    and type(record["numel"]) is int and record["numel"] == math.prod(shape)
                    and type(record["offset_elements"]) is int and record["offset_elements"] >= 0,
                    "original GAN tensor records changed", path="helios.gan.assets")
            ranges.append((record["offset_elements"], record["offset_elements"] + record["numel"]))
        ranges.sort()
        require(all(a[1] <= b[0] for a,b in zip(ranges, ranges[1:]))
                and sum(b-a for a,b in ranges) == component["logical_numel"],
                "original GAN tensor coverage changed", path="helios.gan.assets")
        identity["payloads"][role] = {}
        for name in ("master", "first_moment", "second_moment"):
            payload = component["payloads"][name]
            require(payload.get("dtype") == "torch.float32" and type(payload.get("bytes")) is int
                    and payload["bytes"] > 0 and payload["bytes"] % 4 == 0 and ranges[-1][1] <= payload["bytes"] // 4
                    and bool(SHA256_RE.fullmatch(str(payload.get("sha256", "")))),
                    "original GAN master/moment descriptor changed", path="helios.gan.assets")
            path = reference._file(root, payload["path"])
            if verify_payloads:
                require(path.stat().st_size == payload["bytes"] and file_sha256(path) == payload["sha256"],
                        "original GAN master/moment payload changed", path="helios.gan.assets")
            identity["payloads"][role][name] = payload["sha256"]
    ema_file = reference._file(root, row["ema_manifest"]["identity_file"])
    first = next((p / "sparse_ema_manifest.json" for p in (root/"ema", root/"model_ema")
                  if (p / "sparse_ema_manifest.json").is_file()), None)
    require(first is not None and first.resolve() == ema_file, "native EMA source selection changed", path="helios.gan.assets")
    ema = json.loads(ema_file.read_text())
    require(ema.get("schema_version") == "distillschedule.sparse_ema.v1"
            and type(ema.get("optimization_step")) is int and ema["optimization_step"] == 800,
            "native GAN bootstrap must remain EMA800", path="helios.gan.assets")
    names = ema["tracked_names"]
    records = package.component("generator")["records"]
    require(len(names) == len(set(names)) and {n.removeprefix("module.") for n in names} == set(records),
            "GAN sparse EMA trainable inventory changed", path="helios.gan.assets")
    loaded = []
    identity["ema_shards"] = {}
    for shard in ema["shards"]:
        path = reference._file(ema_file.parent, shard["file"])
        require(shard["file"] not in identity["ema_shards"] and bool(SHA256_RE.fullmatch(str(shard.get("sha256", "")))),
                "GAN EMA shard inventory changed", path="helios.gan.assets")
        loaded.extend(shard["names"]); identity["ema_shards"][shard["file"]] = shard["sha256"]
        if verify_payloads:
            reference._validate_ema_shard(path, shard, records)
    require(len(loaded) == len(set(loaded)) and set(loaded) == set(names), "incomplete GAN EMA shards", path="helios.gan.assets")
    return package, identity


def immutable_assets(value):
    # The W2 helper only hashes precast/negative payloads for non-reference
    # schemas; the GAN bootstrap is verified by its original GAN loader below.
    from .helios_worker import immutable_assets as frozen_payloads
    verify_pins(value)
    observed = frozen_payloads(value)
    from runtime.precast_checkpoint import DEFAULT_INDEX_NAME, PRECAST_MANIFEST_NAME
    for role, folder in (("generator", "transformer_ode"), ("score", "transformer")):
        rows = value["assets"]
        manifest = json.loads((Path(rows[role + "_precast"]["path"]) / PRECAST_MANIFEST_NAME).read_text())
        source = Path(rows[role + "_model"]["path"]) / folder
        require(file_sha256(source / "config.json") == manifest["source"]["config_sha256"]
                and file_sha256(source / DEFAULT_INDEX_NAME) == manifest["source"]["index_sha256"],
                "GAN precast source config/index identity changed", path="helios.gan.assets")
    _, observed["logical_checkpoint"] = load_bootstrap(value)
    observed["gan_config"] = value["assets"]["gan_config"]["sha256"]
    from distillschedule.runtime.helios_gan_manual.backend_library import verify_explicit_files
    observed["fa3_kernel"] = verify_explicit_files(value["assets"]["fa3_kernel"]["path"])
    return observed
