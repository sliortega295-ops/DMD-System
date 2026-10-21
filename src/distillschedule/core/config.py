"""Composable JSON/YAML plan loading with deterministic includes and env expansion."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .contracts import ContractViolation, require_mapping
from .plan import ExecutionPlan


_ENV_PATTERN = re.compile(r"\$\{ENV:([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


def _load_document(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        value = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        try:
            from omegaconf import OmegaConf
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise ContractViolation(
                "YAML plans require the optional 'config' dependencies",
                path=str(path),
            ) from exc
        value = OmegaConf.to_container(OmegaConf.create(text), resolve=False)
    else:
        raise ContractViolation(
            "plan files must use .json, .yaml, or .yml",
            path=str(path),
        )
    return dict(require_mapping(value, path=str(path)))


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = dict(base)
    for key, value in override.items():
        existing = result.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            result[key] = _deep_merge(existing, value)
        else:
            result[key] = value
    return result


def _expand_string(value: str, *, path: str) -> str:
    def replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        if name in os.environ:
            return os.environ[name]
        if default is not None:
            return default
        raise ContractViolation(
            f"environment variable {name!r} is not set",
            path=path,
        )

    return _ENV_PATTERN.sub(replace, value)


def _expand_env(value: Any, *, path: str) -> Any:
    if isinstance(value, str):
        return _expand_string(value, path=path)
    if isinstance(value, list):
        return [_expand_env(item, path=f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, Mapping):
        return {key: _expand_env(item, path=f"{path}.{key}") for key, item in value.items()}
    return value


def _load_with_includes(
    path: Path,
    *,
    stack: tuple[Path, ...],
    sources: list[Path],
) -> dict[str, Any]:
    resolved_path = path.expanduser().resolve()
    if resolved_path in stack:
        cycle = " -> ".join(str(item) for item in (*stack, resolved_path))
        raise ContractViolation(f"include cycle: {cycle}", path=str(path))
    if not resolved_path.is_file():
        raise ContractViolation("plan or include file does not exist", path=str(resolved_path))
    document = _load_document(resolved_path)
    include_value = document.pop("include", ())
    if isinstance(include_value, str):
        includes = [include_value]
    elif include_value:
        if not isinstance(include_value, list):
            raise ContractViolation("include must be a string or list of strings", path=f"{resolved_path}.include")
        includes = include_value
    else:
        includes = []

    merged: dict[str, Any] = {}
    for index, include in enumerate(includes):
        if not isinstance(include, str) or not include.strip():
            raise ContractViolation("include path must be a non-empty string", path=f"{resolved_path}.include[{index}]")
        include_path = (resolved_path.parent / include).resolve()
        merged = _deep_merge(
            merged,
            _load_with_includes(include_path, stack=(*stack, resolved_path), sources=sources),
        )
    sources.append(resolved_path)
    return _deep_merge(merged, document)


@dataclass(frozen=True)
class LoadedPlan:
    plan: ExecutionPlan
    source_path: Path
    source_files: tuple[Path, ...]
    resolved_mapping: Mapping[str, Any]
    digest: str

    @property
    def base_dir(self) -> Path:
        return self.source_path.parent

    def write_resolved(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.plan.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return output


def load_document(path: str | Path) -> tuple[dict[str, Any], tuple[Path, ...]]:
    """Resolve a public JSON/YAML document without importing a model backend.

    Training and data configuration share the plan loader's includes and
    environment rules; their own typed validators own the document schema.
    """
    sources: list[Path] = []
    raw = _load_with_includes(Path(path), stack=(), sources=sources)
    return _expand_env(raw, path="config"), tuple(dict.fromkeys(sources))


def load_plan(path: str | Path) -> LoadedPlan:
    source_path = Path(path).expanduser().resolve()
    sources: list[Path] = []
    raw = _load_with_includes(source_path, stack=(), sources=sources)
    expanded = _expand_env(raw, path="plan")
    plan = ExecutionPlan.from_mapping(expanded)
    canonical = json.dumps(plan.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()
    return LoadedPlan(
        plan=plan,
        source_path=source_path,
        source_files=tuple(dict.fromkeys(sources)),
        resolved_mapping=expanded,
        digest=digest,
    )
