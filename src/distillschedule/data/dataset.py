"""An immutable-order prepared dataset; no sampler, epoch or recycling policy."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from distillschedule.workloads.best_known.registry import file_sha256, fingerprint
from .manifest import (PREPARED_SCHEMA, feature_path, provenance_schema, records, require,
                       sha256, tensor_schema, validate_source)
from .tensors import load_tensors


def _read_manifest(path):
    def unique_pairs(pairs):
        value = {}
        for key, item in pairs:
            require(key not in value, f"duplicate manifest key: {key}")
            value[key] = item
        return value

    content = path.read_bytes()
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
    return value, hashlib.sha256(content).hexdigest()


class PreparedDataset:
    """Hash-check bytes at open; deserialize one CPU-safe tensor tree per sample.

    Opening/inspection needs no Torch and never executes the recorded encoder.
    Traversal is exactly manifest order; callers own epochs and batching.
    """

    def __init__(self, manifest_path: str | Path):
        self.path = Path(manifest_path).expanduser().resolve()
        self._manifest, self._manifest_hash = _read_manifest(self.path)
        hashes = [self._feature(index)[1] for index in range(len(self))]
        self._unchanged()
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

    def _unchanged(self):
        require(file_sha256(self.path) == self._manifest_hash, "prepared manifest hash changed")

    def _feature(self, index):
        spec = self._index(index)["features"]
        path = feature_path(self.path.parent, spec["path"])
        require(path.is_file(), f"feature file does not exist: {path}")
        actual = file_sha256(path)
        require(actual == spec["sha256"], f"feature hash changed: {path}")
        return path, actual

    def sample_id(self, index):
        self._unchanged()
        return self._index(index)["id"]

    def sample(self, index) -> dict:
        self._unchanged()
        path, before = self._feature(index)
        value, metadata = load_tensors(path)
        require(file_sha256(path) == before, f"feature hash changed during read: {path}")
        require(metadata == self._index(index)["features"]["tensors"], f"tensor metadata mismatch: {path}")
        self._unchanged()
        return value

    def validate(self, *, sources=False):
        for index in range(len(self)):
            self.sample(index)
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
    dataset = PreparedDataset(path)
    manifest = dataset.manifest
    return {"manifest": str(dataset.path), "identity": dataset.identity, "samples": len(dataset),
            "records": manifest["records"], "metadata": manifest["metadata"],
            "validation": "HASHES_ONLY_TENSORS_NOT_LOADED"}
