"""Small shared validators and identities for offline data manifests."""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import re

from distillschedule.core.config import load_document
from distillschedule.workloads.best_known.registry import file_sha256, fingerprint

RAW_SCHEMA = "distillschedule.data.raw.v1"
PREENCODED_SCHEMA = "distillschedule.data.preencoded.v1"
PREPARED_SCHEMA = "distillschedule.data.prepared.v1"
FRAME_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".ppm"})
SHA256 = re.compile(r"[0-9a-f]{64}")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def json_value(value):
    """Use the registry's canonical, finite JSON contract for parameters/metadata."""
    try:
        fingerprint(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("parameters and metadata must be finite JSON values") from exc
    return deepcopy(value)


def nonempty(value, name):
    require(isinstance(value, str) and bool(value.strip()), f"{name} must be a nonempty string")
    return value


def sha256(value):
    require(isinstance(value, str) and SHA256.fullmatch(value) is not None, "invalid SHA256")
    return value


def records(value, *, allowed):
    require(isinstance(value, list) and bool(value), "records must be a nonempty list")
    seen = set()
    for record in value:
        require(isinstance(record, dict), "each record must be an object")
        require(set(record) <= allowed, "unknown record fields")
        identifier = nonempty(record.get("id"), "record id")
        require(identifier not in seen, f"duplicate record id: {identifier}")
        seen.add(identifier)
        require(isinstance(record.get("metadata", {}), dict), "record metadata must be an object")
        json_value(record.get("metadata", {}))
    return value


def load_input(path, schema, *, allowed):
    path = Path(path).expanduser().resolve()
    value, sources = load_document(path)
    require(value.get("schema_version") == schema, f"expected schema_version {schema}")
    require(set(value) <= allowed | {"schema_version", "records", "metadata"}, "unknown manifest fields")
    require(isinstance(value.get("metadata", {}), dict), "manifest metadata must be an object")
    json_value(value.get("metadata", {}))
    documents = [source_identity(source) for source in sources]
    return path, value, documents


def resolve_path(value, base_dir):
    nonempty(value, "source path")
    # Media decoding is offline only: URL-like references are never fetched.
    require("://" not in value, "source paths must be local, not URLs")
    return (Path(base_dir) / Path(value).expanduser()).resolve()


def frame_paths(path):
    require(path.is_dir(), f"source frame directory does not exist: {path}")
    frames = sorted((item for item in path.iterdir() if item.suffix.lower() in FRAME_SUFFIXES),
                    key=lambda item: item.name)
    require(bool(frames), f"source frame directory has no supported images: {path}")
    require(all(item.is_file() and not item.is_symlink() for item in frames),
            "frame entries must be regular files, not symlinks")
    return frames


def source_identity(path, *, frames=False):
    path = Path(path).resolve()
    if frames:
        files = [{"path": item.name, "sha256": file_sha256(item)} for item in frame_paths(path)]
        return {"kind": "frames", "path": str(path), "files": files, "sha256": fingerprint(files)}
    require(path.is_file(), f"source file does not exist: {path}")
    return {"kind": "file", "path": str(path), "sha256": file_sha256(path)}


def raw_record(record, base_dir):
    """Resolve references and hash all media before a trusted plugin is imported."""
    require(any(key in record for key in ("video", "audio", "text")), "raw record needs video, audio or text")
    resolved, identities = deepcopy(record), {}
    for name in ("video", "audio"):
        if name not in record:
            continue
        reference = record[name]
        if isinstance(reference, dict):
            require(set(reference) == {"path"}, f"{name} reference must contain only path")
            reference = reference["path"]
        path = resolve_path(reference, base_dir)
        identities[name] = source_identity(path, frames=name == "video" and path.is_dir())
        resolved[name] = str(path)
    if "text" in record:
        reference = record["text"]
        require(isinstance(reference, dict) and set(reference) in ({"path"}, {"value"}),
                "text must be an explicit {path: ...} or {value: ...} reference")
        if "path" in reference:
            path = resolve_path(reference["path"], base_dir)
            identities["text"] = source_identity(path)
            resolved["text"] = {"path": str(path)}
        else:
            require(isinstance(reference["value"], str), "inline text value must be a string")
            identities["text"] = {"kind": "inline_text", "sha256": hashlib.sha256(reference["value"].encode()).hexdigest()}
    return resolved, {"id": record["id"], "media": identities}


def validate_source(identity):
    source_schema(identity)
    if identity.get("kind") == "inline_text":
        return
    path = Path(identity["path"])
    try:
        current = source_identity(path, frames=identity["kind"] == "frames")
    except (OSError, ValueError) as exc:
        raise ValueError(f"source changed or is missing: {path}") from exc
    require(current == identity, f"source hash changed: {path}")


def source_schema(identity):
    """Validate recorded identities without requiring the original media to exist."""
    require(isinstance(identity, dict), "malformed source identity")
    sha256(identity.get("sha256"))
    kind = identity.get("kind")
    if kind == "inline_text":
        require(set(identity) == {"kind", "sha256"}, "malformed inline text identity")
        return
    require(kind in ("file", "frames"), "malformed source kind")
    fields = {"kind", "path", "sha256"} | ({"files"} if kind == "frames" else set())
    require(set(identity) == fields, "malformed source identity fields")
    require(Path(nonempty(identity["path"], "source identity path")).is_absolute(),
            "source identity paths must be absolute")
    if kind == "frames":
        files = identity["files"]
        require(isinstance(files, list) and bool(files), "frame source identity must list its files")
        names = []
        for item in files:
            require(isinstance(item, dict) and set(item) == {"path", "sha256"}, "malformed frame source identity")
            name = nonempty(item["path"], "frame source path")
            require(Path(name).name == name and name not in {".", ".."}, "frame source paths must be filenames")
            sha256(item["sha256"])
            names.append(name)
        require(names == sorted(set(names)), "frame sources must be unique and sorted")
        require(fingerprint(files) == identity["sha256"], "frame source hash metadata mismatch")


def provenance_schema(value, rows):
    require(isinstance(value, dict) and set(value) == {"mode", "documents", "sources", "encoder"},
            "malformed source provenance")
    require(value["mode"] in ("raw", "preencoded"), "malformed source provenance mode")
    require(isinstance(value["documents"], list) and bool(value["documents"]), "source documents are required")
    for item in value["documents"]:
        source_schema(item)
        require(item["kind"] == "file", "source documents must be files")
    require(isinstance(value["sources"], list) and len(value["sources"]) == len(rows),
            "source provenance must cover every record")
    for item, row in zip(value["sources"], rows):
        require(isinstance(item, dict) and set(item) == {"id", "media"} and item["id"] == row["id"],
                "source identities must match record order and ids")
        require(isinstance(item["media"], dict) and bool(item["media"]), "record source media are required")
        allowed = {"video", "audio", "text"} if value["mode"] == "raw" else {"features"}
        require(set(item["media"]) <= allowed, "unknown source media identity")
        for identity in item["media"].values():
            source_schema(identity)
    encoder = value["encoder"]
    if value["mode"] == "preencoded":
        require(encoder is None, "preencoded import must not claim encoder execution")
        return
    require(isinstance(encoder, dict) and set(encoder) == {
        "reference", "sha256", "source_path", "parameters", "parameters_sha256"}, "raw encoder identity is required")
    nonempty(encoder["reference"], "encoder reference")
    source_schema({"kind": "file", "path": encoder["source_path"], "sha256": encoder["sha256"]})
    require(isinstance(encoder["parameters"], dict), "encoder parameters must be an object")
    sha256(encoder["parameters_sha256"])
    require(fingerprint(encoder["parameters"]) == encoder["parameters_sha256"], "encoder parameters hash mismatch")


def feature_path(base_dir, reference):
    nonempty(reference, "feature path")
    path = Path(reference)
    require(not path.is_absolute(), "prepared feature paths must be relative")
    resolved = (base_dir / path).resolve()
    require(resolved.is_relative_to(base_dir), "feature path must stay inside the prepared dataset")
    return resolved


def tensor_schema(value, *, depth=0):
    """Validate metadata without importing Torch or deserializing a feature file."""
    require(depth <= 64 and isinstance(value, dict), "malformed tensor metadata")
    kind = value.get("kind")
    if kind == "tensor":
        require(set(value) == {"kind", "shape", "dtype"}, "malformed tensor metadata")
        require(isinstance(value["shape"], list) and all(type(n) is int and n >= 0 for n in value["shape"]),
                "tensor shape must be a list of nonnegative integers")
        require(isinstance(value["dtype"], str) and re.fullmatch(r"torch\.[a-zA-Z0-9_]+", value["dtype"]),
                "malformed tensor dtype")
    elif kind == "dict":
        require(set(value) == {"kind", "items"} and isinstance(value["items"], dict) and bool(value["items"]),
                "tensor dict metadata must be nonempty")
        for name, item in value["items"].items():
            nonempty(name, "tensor field")
            tensor_schema(item, depth=depth + 1)
    elif kind in ("list", "tuple"):
        require(set(value) == {"kind", "items"} and isinstance(value["items"], list) and bool(value["items"]),
                "tensor sequence metadata must be nonempty")
        for item in value["items"]:
            tensor_schema(item, depth=depth + 1)
    else:
        raise ValueError("malformed tensor metadata kind")
    return value
