"""Transactional preparation from raw media or already-native tensor files."""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

from distillschedule.training.plugins import load_callable, resolve_reference
from distillschedule.workloads.best_known.registry import file_sha256, fingerprint
from .dataset import PreparedDataset
from .decoding import decode_record
from .manifest import (PREENCODED_SCHEMA, PREPARED_SCHEMA, RAW_SCHEMA, json_value, load_input,
                       raw_record, records, require, resolve_path, sha256, source_identity,
                       nonempty, tensor_schema, validate_source)
from .tensors import load_tensors, tensor_tree


def _new_output(output_dir):
    path = Path(output_dir).expanduser()
    if path.is_symlink() or path.exists():
        raise FileExistsError(f"data output already exists: {path}")
    return path.resolve()


def _publish(output_dir, *, rows, provenance, metadata, produce):
    """Rename a fully verified sibling directory; no complete output is replaced."""
    output = _new_output(output_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.parent / f".{output.name}.lock"
    # An exclusive sibling lock also rejects two cooperating publishers racing.
    with lock.open("x", encoding="utf-8"):
        stage = None
        try:
            if output.exists():
                raise FileExistsError(f"data output already exists: {output}")
            stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
            (stage / "features").mkdir()
            prepared = []
            for index, record in enumerate(rows):
                value, tensors = produce(record)
                relative = f"features/{index:08d}.pt"
                target = stage / relative
                import torch
                with target.open("xb") as stream:
                    torch.save(value, stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                item = {"id": record["id"], "features": {
                    "path": relative, "sha256": file_sha256(target), "tensors": tensors}}
                if "metadata" in record:
                    item["metadata"] = deepcopy(record["metadata"])
                prepared.append(item)
            # Detect changing raw documents/media or encoder code before publishing.
            for identity in provenance["documents"]:
                validate_source(identity)
            for record in provenance["sources"]:
                for identity in record["media"].values():
                    validate_source(identity)
            if provenance.get("encoder"):
                encoder = provenance["encoder"]
                validate_source({"kind": "file", "path": encoder["source_path"], "sha256": encoder["sha256"]})
            manifest = {"schema_version": PREPARED_SCHEMA, "status": "complete", "records": prepared,
                        "provenance": provenance, "metadata": metadata}
            manifest_path = stage / "manifest.json"
            with manifest_path.open("x", encoding="utf-8") as stream:
                json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            PreparedDataset(manifest_path).validate()
            if output.exists():
                raise FileExistsError(f"data output already exists: {output}")
            os.rename(stage, output)
            stage = None
            return output / "manifest.json"
        finally:
            if stage is not None:
                shutil.rmtree(stage)
            lock.unlink(missing_ok=True)


def prepare(raw_manifest: str | Path, output_dir: str | Path) -> Path:
    """Decode real media and call an explicit trusted model-specific encoder.

    The encoder signature is ``encoder(decoded_record: dict, parameters: dict)
    -> dict[str, Tensor tree]``. Its native tensor shapes and dtypes are preserved.
    """
    _new_output(output_dir)  # refuse existing data before even importing a plugin
    path, value, documents = load_input(raw_manifest, RAW_SCHEMA, allowed={"encoder"})
    rows = records(value.get("records"), allowed={"id", "image", "video", "audio", "text", "metadata"})
    encoder = value.get("encoder")
    require(isinstance(encoder, dict) and set(encoder) <= {"factory", "parameters"}, "encoder needs factory and parameters")
    parameters = encoder.get("parameters", {})
    require(isinstance(parameters, dict), "encoder parameters must be an object")
    parameters = json_value(parameters)
    resolved = [raw_record(record, path.parent) for record in rows]
    reference = resolve_reference(nonempty(encoder.get("factory"), "encoder factory"), path.parent)
    function, identity = load_callable(reference)
    module = reference.rpartition(":")[0]
    source = (Path(module) if module.endswith(".py") else Path(sys.modules[module].__file__)).resolve()
    identity = {**identity, "source_path": str(source), "parameters": parameters,
                "parameters_sha256": fingerprint(parameters)}
    provenance = {"mode": "raw", "documents": documents,
                  "sources": [item[1] for item in resolved], "encoder": identity}

    def encode(record):
        decoded = decode_record(record)
        return tensor_tree(function(decoded, deepcopy(parameters)), cpu=True)

    return _publish(output_dir, rows=[item[0] for item in resolved], provenance=provenance,
                    metadata=json_value(value.get("metadata", {})), produce=encode)


def import_preencoded(source_manifest: str | Path, output_dir: str | Path) -> Path:
    """Import existing native features without any geometry or format conversion."""
    _new_output(output_dir)
    path, value, documents = load_input(source_manifest, PREENCODED_SCHEMA, allowed=set())
    rows = records(value.get("records"), allowed={"id", "features", "metadata"})
    sources = []
    for record in rows:
        spec = record.get("features")
        require(isinstance(spec, dict) and {"path", "sha256"} <= set(spec) <= {"path", "sha256", "tensors"},
                "preencoded features need path, sha256 and optional tensors")
        sha256(spec["sha256"])
        identity = source_identity(resolve_path(spec["path"], path.parent))
        require(identity["sha256"] == spec["sha256"], f"feature hash mismatch: {identity['path']}")
        if "tensors" in spec:
            tensor_schema(spec["tensors"])
        sources.append({"id": record["id"], "media": {"features": identity}})

    def read(record):
        spec = record["features"]
        source = resolve_path(spec["path"], path.parent)
        require(file_sha256(source) == spec["sha256"], f"feature hash changed: {source}")
        tensors, metadata = load_tensors(source)
        require(file_sha256(source) == spec["sha256"], f"feature hash changed during read: {source}")
        if "tensors" in spec:
            require(metadata == spec["tensors"], "preencoded tensor metadata mismatch")
        return tensors, metadata

    return _publish(output_dir, rows=rows, provenance={"mode": "preencoded", "documents": documents,
                    "sources": sources, "encoder": None}, metadata=json_value(value.get("metadata", {})), produce=read)
