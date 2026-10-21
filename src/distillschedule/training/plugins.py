"""Load explicitly configured, trusted Python adapters only at execution time."""
from __future__ import annotations
import hashlib
import importlib
import importlib.util
from pathlib import Path
import sys


def resolve_reference(reference: str, base_dir: Path) -> str:
    path, sep, symbol = reference.rpartition(":")
    if not sep or not symbol.isidentifier() or not path:
        raise ValueError("adapter must be a module:callable or local.py:callable")
    if path.endswith(".py"):
        return str((base_dir / path).expanduser().resolve()) + ":" + symbol
    if not all(part.isidentifier() for part in path.split(".")):
        raise ValueError("invalid adapter module name")
    return reference


def load_callable(reference: str):
    path, _, symbol = reference.rpartition(":")
    if path.endswith(".py"):
        source = Path(path)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        name = "dmd_user_" + digest
        spec = importlib.util.spec_from_file_location(name, source)
        if spec is None or spec.loader is None:
            raise ValueError("cannot load adapter source")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise ValueError("adapter source changed during import")
    else:
        module = importlib.import_module(path)
        origin = getattr(module, "__file__", None)
        if origin is None:
            raise ValueError("adapter must have inspectable Python source")
        digest = hashlib.sha256(Path(origin).read_bytes()).hexdigest()
    function = getattr(module, symbol, None)
    if not callable(function):
        raise ValueError(f"adapter is not callable: {reference}")
    return function, {"reference": reference, "sha256": digest}
