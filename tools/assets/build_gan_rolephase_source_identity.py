#!/usr/bin/env python3
"""Build or verify a bounded source identity for staged role-phase runs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


SUFFIXES = {".py", ".sh", ".json", ".yaml", ".yml", ".toml"}
EXCLUDED_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    "artifacts",
    "research_artifacts",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def source_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in SUFFIXES:
            continue
        relative = path.relative_to(root)
        if EXCLUDED_PARTS.intersection(relative.parts):
            continue
        yield path


def tree_identity(root: Path) -> dict[str, Any]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    entries = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in source_files(root)
    ]
    return {
        "root": str(root.resolve()),
        "file_count": len(entries),
        "total_bytes": sum(item["bytes"] for item in entries),
        "tree_sha256": hashlib.sha256(canonical(entries)).hexdigest(),
    }


def build(*, dmd_root: Path, helios_root: Path, git_head: str) -> dict[str, Any]:
    payload = {
        "schema_version": "distillschedule.source_identity.v1",
        "status": "PASS",
        "git_head": git_head,
        "runtime_git_head": "external_helios_evidence_tree_non_git",
        "trees": {
            "dmd_system": tree_identity(dmd_root),
            "helios_runtime": tree_identity(helios_root),
        },
    }
    payload["identity_hash"] = hashlib.sha256(canonical(payload)).hexdigest()
    return payload


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--dmd-root", type=Path, required=True)
    value.add_argument("--helios-root", type=Path, required=True)
    value.add_argument("--git-head", required=True)
    value.add_argument("--output", type=Path)
    value.add_argument("--verify", type=Path)
    return value


def main() -> int:
    args = parser().parse_args()
    if (args.output is None) == (args.verify is None):
        raise ValueError("choose exactly one of --output or --verify")
    observed = build(
        dmd_root=args.dmd_root,
        helios_root=args.helios_root,
        git_head=args.git_head,
    )
    if args.verify is not None:
        expected = json.loads(args.verify.read_text())
        if expected != observed:
            raise ValueError("staged source identity changed")
    else:
        atomic_json(args.output, observed)
    print(json.dumps(observed, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
