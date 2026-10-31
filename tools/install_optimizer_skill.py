#!/usr/bin/env python3
"""Install the repository's self-contained Codex skill without replacing an existing copy."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys


def files(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts
            and p.suffix not in {".pyc", ".pyo"}}


def install(source, destination):
    source, destination = source.resolve(), destination.expanduser().absolute()
    if not (source / "SKILL.md").is_file():
        raise ValueError("skill source is missing")
    if destination.is_relative_to(source):
        raise ValueError("destination must be outside the source skill")
    expected = files(source)
    if destination.exists() or destination.is_symlink():
        if destination.is_dir() and not destination.is_symlink() and files(destination) == expected:
            return {"status": "ALREADY_INSTALLED", "path": str(destination), "files": len(expected)}
        raise ValueError("destination already exists; use a different directory or manage the existing version explicitly")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()  # exclusive; never copy into another writer's directory
    try:
        shutil.copytree(source, destination, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))
        if files(destination) != expected:
            raise ValueError("installed files differ from source snapshot")
    except BaseException:
        shutil.rmtree(destination)  # only the directory this call just created
        raise
    return {"status": "INSTALLED", "path": str(destination), "files": len(expected)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--destination", type=Path, default=Path(os.environ.get("CODEX_HOME", str(Path.home()/".codex"))) / "skills/dmd-system-optimizer")
    a = p.parse_args()
    try:
        print(json.dumps(install(Path(__file__).resolve().parents[1]/"skills/dmd-system-optimizer", a.destination)))
        return 0
    except (OSError, ValueError) as exc:
        print(json.dumps({"status": "INVALID", "error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
