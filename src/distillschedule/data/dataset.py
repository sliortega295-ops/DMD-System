"""An immutable-order prepared dataset; no sampler, epoch or recycling policy."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import stat

from distillschedule.workloads.best_known.registry import fingerprint
from .manifest import (PREPARED_SCHEMA, feature_path, provenance_schema, records, require,
                       sha256, tensor_schema, validate_source)
from .tensors import load_tensors


def _file_token(info, path):
    require(stat.S_ISREG(info.st_mode), f"prepared file must be regular: {path}")
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _path_token(path):
    try:
        return _file_token(path.stat(), path)
    except OSError as exc:
        raise ValueError(f"prepared file does not exist or cannot be read: {path}") from exc


def _open_file(path):
    try:
        # A replacement with a FIFO must not block before the descriptor check.
        return open(path, "rb", opener=lambda name, flags: os.open(name, flags | getattr(os, "O_NONBLOCK", 0)))
    except OSError as exc:
        raise ValueError(f"prepared file does not exist or cannot be read: {path}") from exc


def _hash_stream(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def _read_manifest(path):
    def unique_pairs(pairs):
        value = {}
        for key, item in pairs:
            require(key not in value, f"duplicate manifest key: {key}")
            value[key] = item
        return value

    with _open_file(path) as stream:
        token = _file_token(os.fstat(stream.fileno()), path)
        content = stream.read()
        digest = hashlib.sha256(content).hexdigest()
        require(_file_token(os.fstat(stream.fileno()), path) == token and _path_token(path) == token,
                "prepared manifest changed during read")
    value = json.loads(content, object_pairs_hook=unique_pairs)
    require(isinstance(value, dict), "prepared manifest must be an object")
    require(set(value) == {"schema_version", "status", "records", "provenance", "metadata"},
            "malformed prepared manifest fields")
    require(value["schema_version"] == PREPARED_SCHEMA and value["status"] == "complete",
            "prepared dataset is not a complete supported manifest")
    require(isinstance(value["metadata"], dict) and isinstance(value["provenance"], dict),
            "prepared metadata/provenance must be objects")
    fingerprint(value)  # reject NaN and noncanonical JSON values
    records(value["records"], allowed={"id", "features", "metadata"})
    provenance_schema(value["provenance"], value["records"])
    for record in value["records"]:
        spec = record.get("features")
        require(isinstance(spec, dict) and set(spec) == {"path", "sha256", "tensors"}, "malformed feature reference")
        sha256(spec["sha256"])
        tensor_schema(spec["tensors"])
        require(spec["tensors"]["kind"] == "dict", "feature metadata root must be a tensor dict")
    return value, digest, token


class PreparedDataset:
    """Verify immutable feature bytes lazily; safe-load a fresh CPU tree per sample.

    Cached verification hashes on first access and filesystem metadata drift.
    Strict verification hashes every feature at open and again on every read.
    Opening needs no Torch and never executes the recorded encoder.
    Traversal is exactly manifest order; callers own epochs and batching.
    """

    def __init__(self, manifest_path: str | Path, *, verification="cached"):
        require(verification in ("cached", "strict"), "verification must be 'cached' or 'strict'")
        self.path = Path(manifest_path).expanduser().resolve()
        self._verification = verification
        self._manifest, self._manifest_hash, self._manifest_token = _read_manifest(self.path)
        self._feature_tokens = {}
        for index in range(len(self)):
            if verification == "strict":
                self._read_feature(index)
            else:
                self._feature_path(index)  # Fail early on missing, nonregular or escaping paths.
        self._unchanged()
        hashes = [record["features"]["sha256"] for record in self._manifest["records"]]
        self.identity = fingerprint({"manifest_sha256": self._manifest_hash, "features": hashes})

    @property
    def manifest(self):
        return deepcopy(self._manifest)

    def __len__(self):
        return len(self._manifest["records"])

    def _index(self, index):
        if type(index) is not int or not 0 <= index < len(self):
            raise IndexError(f"sample index must be in [0, {len(self)})")
        return self._manifest["records"][index]

    def _unchanged(self, *, force=False):
        token = _path_token(self.path)
        if not force and self._verification == "cached" and token == self._manifest_token:
            return
        with _open_file(self.path) as stream:
            require(_file_token(os.fstat(stream.fileno()), self.path) == token,
                    "prepared manifest changed during read")
            require(_hash_stream(stream) == self._manifest_hash, "prepared manifest hash changed")
            require(_file_token(os.fstat(stream.fileno()), self.path) == token and _path_token(self.path) == token,
                    "prepared manifest changed during read")
        self._manifest_token = token

    def _feature_path(self, index):
        spec = self._index(index)["features"]
        path = feature_path(self.path.parent, spec["path"])
        return path, _path_token(path)

    def _read_feature(self, index, *, tensors=False, force=False):
        spec = self._index(index)["features"]
        path, token = self._feature_path(index)
        value = None
        with _open_file(path) as stream:
            require(_file_token(os.fstat(stream.fileno()), path) == token, f"feature changed during read: {path}")
            if force or self._verification == "strict" or self._feature_tokens.get(index) != token:
                require(_hash_stream(stream) == spec["sha256"], f"feature hash changed: {path}")
                require(_file_token(os.fstat(stream.fileno()), path) == token,
                        f"feature changed during read: {path}")
                stream.seek(0)
            if tensors:
                value, metadata = load_tensors(stream)
                require(metadata == spec["tensors"], f"tensor metadata mismatch: {path}")
            require(_file_token(os.fstat(stream.fileno()), path) == token, f"feature changed during read: {path}")
            # Resolve the original reference again: its directories/symlinks may
            # have changed even if the opened file itself is still immutable.
            require(self._feature_path(index) == (path, token), f"feature changed during read: {path}")
        self._feature_tokens[index] = token
        return value

    def sample_id(self, index):
        self._unchanged()
        return self._index(index)["id"]

    def sample(self, index) -> dict:
        return self._sample(index)

    def _sample(self, index, *, force=False):
        self._unchanged()
        value = self._read_feature(index, tensors=True, force=force)
        self._unchanged()
        return value

    def validate(self, *, sources=False):
        self._unchanged(force=True)
        for index in range(len(self)):
            self._sample(index, force=True)
        self._unchanged(force=True)
        if sources:
            provenance = self._manifest["provenance"]
            for identity in provenance.get("documents", []):
                validate_source(identity)
            for record in provenance.get("sources", []):
                for identity in record["media"].values():
                    validate_source(identity)
            encoder = provenance.get("encoder")
            if encoder:
                validate_source({"kind": "file", "path": encoder["source_path"], "sha256": encoder["sha256"]})
        return {"status": "PASS", "samples": len(self), "identity": self.identity,
                "source_validation": "PASS" if sources else "NOT_REQUESTED"}


def inspect_prepared(path):
    dataset = PreparedDataset(path, verification="strict")
    manifest = dataset.manifest
    return {"manifest": str(dataset.path), "identity": dataset.identity, "samples": len(dataset),
            "records": manifest["records"], "metadata": manifest["metadata"],
            "validation": "HASHES_ONLY_TENSORS_NOT_LOADED"}
