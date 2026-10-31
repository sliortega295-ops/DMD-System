#!/usr/bin/env python3
"""Install the standalone skill; explicit updates preserve the previous copy."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile


def files(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts
            and p.suffix not in {".pyc", ".pyo"}}


def install(source, destination, *, update=False):
    source = source.resolve()
    destination = destination.expanduser().absolute()
    if destination.is_symlink():
        raise ValueError("destination is a symlink; update its source or choose another directory")
    destination = destination.resolve()
    if not (source / "SKILL.md").is_file():
        raise ValueError("skill source is missing")
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("source and destination must not overlap")
    expected = files(source)
    previous = None
    if destination.exists():
        if not destination.is_dir() or not (destination / "SKILL.md").is_file():
            raise ValueError("destination is not an installed skill; choose another directory")
        previous = files(destination)
        if previous == expected:
            return {"status": "ALREADY_INSTALLED", "path": str(destination), "files": len(expected)}
        if not update:
            raise ValueError("destination already exists; use --update to preserve it as a backup and install this version")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".dmd-install-", dir=destination.parent))
    backup = None
    try:
        shutil.copytree(source, stage, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))
        if files(stage) != expected:
            raise ValueError("installed files differ from source snapshot")
        if (files(destination) if destination.is_dir() else None) != previous:
            raise ValueError("destination changed during installation; retry after inspecting it")
        if destination.exists():
            # Outside a skills/ discovery root: the old skill must not load twice.
            backups = (destination.parent.parent / "skill-backups" if destination.parent.name == "skills"
                       else destination.parent / ".skill-backups")
            backups.mkdir(parents=True, exist_ok=True)
            backup = Path(tempfile.mkdtemp(prefix=destination.name + "-", dir=backups)) / destination.name
            destination.rename(backup)
        stage.rename(destination)
    except BaseException:
        if backup is not None and backup.exists() and not destination.exists():
            backup.rename(destination)
        raise
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    result = {"status": "UPDATED" if backup else "INSTALLED", "path": str(destination), "files": len(expected)}
    if backup:
        result["backup"] = str(backup)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--destination", type=Path, default=Path.home() / ".agents/skills/dmd-system-optimizer")
    p.add_argument("--update", action="store_true", help="back up an existing skill before replacing it; private files stay in the backup")
    a = p.parse_args()
    try:
        print(json.dumps(install(Path(__file__).resolve().parents[1] / "skills/dmd-system-optimizer",
                                 a.destination, update=a.update)))
        return 0
    except (OSError, ValueError) as exc:
        print(json.dumps({"status": "INVALID", "error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
