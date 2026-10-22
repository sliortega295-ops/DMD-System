"""Load explicitly configured, trusted Python adapters only at execution time."""
from __future__ import annotations
import hashlib
import importlib
import importlib.util
from importlib.machinery import SourceFileLoader
from pathlib import Path
import sys


def resolve_reference(reference: str, base_dir: Path) -> str:
    path, sep, symbol = reference.rpartition(":")
    if not sep or not symbol.isidentifier() or not path:
        raise ValueError("adapter must be a module:callable or local.py:callable")
    if path.endswith(".py"):
        return str((base_dir / Path(path).expanduser()).resolve()) + ":" + symbol
    if not all(part.isidentifier() for part in path.split(".")):
        raise ValueError("invalid adapter module name")
    return reference


class _SourceOnlyLoader(SourceFileLoader):
    """Keep ordinary module semantics while executing the recorded source."""

    def __init__(self, name, source, digest):
        super().__init__(name, str(source))
        self.source_sha256 = digest

    def get_code(self, fullname):
        data = self.get_data(self.path)
        if fullname != self.name or hashlib.sha256(data).hexdigest() != self.source_sha256:
            raise ValueError("adapter source changed during import")
        return self.source_to_code(data, self.path)


def _local_callable(source, symbol):
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    name = "dmd_user_" + digest
    spec = importlib.util.spec_from_file_location(name, source,
        loader=_SourceOnlyLoader(name, source, digest))
    if spec is None or spec.loader is None:
        raise ValueError("cannot load adapter source")
    module = importlib.util.module_from_spec(spec)
    absent = object()
    previous = sys.modules.get(name, absent)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise ValueError("adapter source changed during import")
        function = getattr(module, symbol, None)
        if not callable(function):
            raise ValueError(f"adapter is not callable: {source}:{symbol}")
        return function, digest
    except BaseException:
        if sys.modules.get(name) is module:
            if previous is absent:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
        raise


def load_callable(reference: str):
    path, _, symbol = reference.rpartition(":")
    if path.endswith(".py"):
        function, digest = _local_callable(Path(path), symbol)
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
