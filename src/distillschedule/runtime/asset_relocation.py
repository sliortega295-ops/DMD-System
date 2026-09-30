"""Resolve relocated I/O paths without rewriting immutable workload metadata.

This is an explicit hardware-port compatibility opt-in. Content identity is
still checked by the existing loader/source/trace gates; a path map is not a
weight or source receipt. It does not create A100-path aliases on the server.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path


RECEIPT_ENV = "DISTILLSCHEDULE_ASSET_RELOCATION_RECEIPT"
SHA_ENV = "DISTILLSCHEDULE_ASSET_RELOCATION_SHA256"


@lru_cache(maxsize=4)
def _bindings(receipt_path: str, expected_sha: str):
    if len(expected_sha) != 64:
        raise ValueError("asset relocation requires a SHA-256-bound receipt")
    path = Path(receipt_path).expanduser().resolve()
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha:
        raise ValueError("asset relocation receipt SHA mismatch")
    value = json.loads(data)
    if value.get("schema_version") != "distillschedule.asset_relocation.v1":
        raise ValueError("unsupported asset relocation receipt")
    if value.get("hardware_profile") != "h100-8-b16":
        raise ValueError("asset relocation is not bound to this hardware profile")
    root = Path(value["owned_destination_root"]).expanduser().resolve()
    if not root.is_absolute() or len(root.parts) < 4:
        raise ValueError("asset relocation requires a specific owned project root")
    bindings = []
    sources = set()
    for row in value["bindings"]:
        source = Path(row["source"])
        target = Path(row["destination"]).expanduser().resolve()
        if not source.is_absolute() or ".." in source.parts or source in sources:
            raise ValueError("asset relocation has an invalid or duplicate source")
        if not target.is_relative_to(root):
            raise ValueError("asset relocation target escapes the owned root")
        if row.get("kind") not in {"file", "directory"}:
            raise ValueError("asset relocation binding kind must be explicit")
        sources.add(source)
        bindings.append((source, target, row["kind"]))
    if not bindings:
        raise ValueError("asset relocation has no bindings")
    return root, sorted(bindings, key=lambda row: len(row[0].parts), reverse=True)


def relocate_asset_path(value: str | Path) -> Path:
    receipt = os.environ.get(RECEIPT_ENV)
    if not receipt:
        if os.environ.get(SHA_ENV):
            raise ValueError("asset relocation SHA is set without a receipt")
        return Path(value).expanduser().resolve()
    if os.environ.get("DISTILLSCHEDULE_HARDWARE_PROFILE") != "h100-8-b16":
        raise ValueError("asset relocation requires explicit h100-8-b16")
    root, bindings = _bindings(receipt, os.environ.get(SHA_ENV, ""))
    requested = Path(value).expanduser()
    if not requested.is_absolute() or ".." in requested.parts:
        raise ValueError("relocated asset address must be absolute and canonical")
    if requested.is_relative_to(root):
        resolved = requested.resolve()
        if not resolved.is_relative_to(root):
            raise ValueError("relocated asset symlink escapes the owned root")
        return resolved
    for source, destination, kind in bindings:
        if requested == source or (kind == "directory" and requested.is_relative_to(source)):
            resolved = (destination / requested.relative_to(source)).resolve()
            if not resolved.is_relative_to(root):
                raise ValueError("relocated asset target escapes the owned root")
            return resolved
    raise ValueError(f"asset address has no explicit relocation binding: {requested}")


def relocation_identity() -> dict[str, str] | None:
    """Small receipt metadata only; never claim this proves payload identity."""
    receipt = os.environ.get(RECEIPT_ENV)
    if not receipt:
        return None
    sha = os.environ.get(SHA_ENV, "")
    _bindings(receipt, sha)
    return {"path": str(Path(receipt).expanduser().resolve()), "sha256": sha,
            "payload_identity": "REQUIRES_EXISTING_CONTENT_GATES"}
