#!/usr/bin/env python3
"""Verify the recovered Helios source without importing Torch or loading models."""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
from typing import Any


EXPECTED_TRANSFORMER_SHA256 = (
    "5d3ffe37cdea3c04a156106f6246c82f91010a1eaf0dd61529cd74083065a886"
)
SCHEMA = "distillschedule.helios_compat_source.v1"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _module_name(path: str) -> str:
    parts = list(PurePosixPath(path).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _module_file(root: Path, name: str) -> Path | None:
    stem = root.joinpath(*name.split("."))
    for candidate in (stem.with_suffix(".py"), stem / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def _defined_names(path: Path) -> set[str]:
    """Static export check, not a substitute for a real dependency import test."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_bytes(), filename=str(path))):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
    return names


def _audit_imports(
    paths: list[Path], *, source_root: Path, runtime_root: Path, file_root: Path
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    roots = {"helios": source_root, "runtime": runtime_root}
    for path in paths:
        module = _module_name(path.relative_to(file_root).as_posix())
        package = module if path.name == "__init__.py" else module.rpartition(".")[0]
        tree = ast.parse(path.read_bytes(), filename=str(path))
        for node in ast.walk(tree):
            imported_names: list[str] = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                name = node.module or ""
                if node.level:
                    name = importlib.util.resolve_name("." * node.level + name, package)
                modules = [name]
                imported_names = [alias.name for alias in node.names]
            else:
                continue
            for name in modules:
                owner = name.split(".")[0]
                if owner not in roots:
                    continue
                target = _module_file(roots[owner], name)
                if target is None:
                    raise ValueError(f"missing local import: {path}:{node.lineno} -> {name}")
                # Core Helios must use DMD-System's public compatibility helpers.
                if owner == "runtime" and path.is_relative_to(source_root):
                    missing = set(imported_names) - _defined_names(target) - {"*"}
                    if missing:
                        raise ValueError(f"missing runtime exports: {name}: {sorted(missing)}")
                for item in imported_names:
                    if item == "*":
                        continue
                    # A from-package import can name a sibling module.
                    child = _module_file(roots[owner], name + "." + item)
                    if child is not None:
                        compile(child.read_bytes(), str(child), "exec")
                rows.append({
                    "file": path.relative_to(file_root).as_posix(),
                    "line": node.lineno,
                    "module": name,
                    "names": imported_names,
                    "provider": target.relative_to(roots[owner]).as_posix(),
                    "provider_sha256": _sha(target.read_bytes()),
                })
    return rows


def verify_source(repo_root: Path, snapshot_root: Path | None = None) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    root = (snapshot_root or repo_root / "third_party/helios_compat").resolve()
    manifest = json.loads((root / "source_manifest.json").read_text())
    if manifest.get("schema_version") != SCHEMA:
        raise ValueError("unsupported Helios source manifest schema")
    if manifest.get("expected_transformer_sha256") != EXPECTED_TRANSFORMER_SHA256:
        raise ValueError("historical Transformer identity must not be relaxed")
    files = manifest["files"]
    tree_sha = _sha(json.dumps(
        sorted(files, key=lambda row: row["path"]), sort_keys=True, separators=(",", ":")
    ).encode())
    if tree_sha != manifest["content_tree_sha256"]:
        raise ValueError("source manifest file inventory hash mismatch")
    listed: set[str] = set()
    total = 0
    for row in files:
        rel = PurePosixPath(row["path"])
        if rel.is_absolute() or ".." in rel.parts or row["path"] in listed:
            raise ValueError(f"invalid or duplicate source path: {row['path']}")
        path = root / rel
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f"source symlink/escape rejected: {rel}")
        data = path.read_bytes()
        if len(data) != row["bytes"] or _sha(data) != row["sha256"]:
            raise ValueError(f"source identity mismatch: {rel}")
        # Git object identity is independent of the filesystem's newline settings.
        blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        if blob != row["source_git_blob"]:
            raise ValueError(f"source Git blob mismatch: {rel}")
        if path.suffix == ".py":
            compile(data, str(rel), "exec")
        total += len(data)
        listed.add(row["path"])
    actual = {
        path.relative_to(root).as_posix()
        for namespace in ("source", "reference")
        for path in (root / namespace).rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    } | {"LICENSE.txt"}
    if actual != listed:
        raise ValueError(f"unregistered/missing source files: {sorted(actual ^ listed)}")
    if len(files) != manifest["file_count"] or total != manifest["total_bytes"]:
        raise ValueError("source totals do not reconcile")
    transformer = root / "source/helios/modules/transformer_helios.py"
    if _sha(transformer.read_bytes()) != EXPECTED_TRANSFORMER_SHA256:
        raise ValueError("recovered Transformer does not match the winner's source pin")
    source_root = root / "source"
    core_imports = _audit_imports(
        sorted(source_root.rglob("*.py")), source_root=source_root,
        runtime_root=repo_root, file_root=source_root,
    )
    reference_imports = _audit_imports(
        sorted((root / "reference").rglob("*.py")), source_root=source_root,
        runtime_root=root / "reference", file_root=root / "reference",
    )
    return {
        "schema_version": "distillschedule.helios_compat_source_receipt.v1",
        "status": "PASS",
        "scope": "SOURCE_BYTES_SYNTAX_STATIC_LOCAL_IMPORTS_ONLY",
        "source_git_commit": manifest["source_git_commit"],
        "manifest_sha256": _sha((root / "source_manifest.json").read_bytes()),
        "content_tree_sha256": tree_sha,
        "transformer_sha256": EXPECTED_TRANSFORMER_SHA256,
        "files_checked": len(files), "bytes_checked": total,
        "core_local_imports": core_imports,
        "historical_reference_local_imports": reference_imports,
        "real_model_import": "NOT_RUN",
        "gpu_execution": "NOT_RUN",
        "historical_performance_reproduction": "NOT_RUN",
        "environment_installation": "NOT_RUN",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--source-root", type=Path, help="Snapshot root, for source-audit canaries")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        receipt = verify_source(args.repo_root, args.source_root)
    except (ValueError, KeyError, OSError, SyntaxError) as error:
        receipt = {"status": "FAIL", "error": str(error), "gpu_execution": "NOT_RUN"}
    text = json.dumps(receipt, indent=2) + "\n"
    if args.output:
        if args.output.exists():
            raise SystemExit(f"refusing to overwrite receipt: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(text)
    print(text, end="")
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
