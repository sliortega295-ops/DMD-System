"""Import measured helpers against one pinned upstream namespace, never ``wan``.

The helpers' original definitions are source locked. Only absolute upstream
imports are rewritten with the existing PinnedSource AST transformer; sibling
helper imports keep their relative meaning. Inspection/Triton still see the
real packaged filename. This is not a benchmark runner or a model downloader.
"""
from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.abc
import importlib.util
import json
from pathlib import Path
import sys

from ..official_profiles import verify_sources
from ..upstream import PinnedSource, _Imports


class SourceBoundHelpers(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    PROFILE = "self-forcing-wan-fsdp2-r6-v1"

    def __init__(self, source):
        if not isinstance(source, PinnedSource):
            raise ValueError("official Self-Forcing helpers require an actual PinnedSource")
        self.source = source
        self.files = {Path(row["target"]).stem: row for row in verify_sources(self.PROFILE)}
        self.root = Path(__file__).resolve().parent / "_native"
        identity = json.dumps(dict(upstream=source.namespace, files={
            key: row["integrated_sha256"] for key, row in sorted(self.files.items())}), sort_keys=True)
        self.namespace = "_distillschedule_sf_helpers_" + hashlib.sha256(identity.encode()).hexdigest()[:24]
        self.validate()

    def validate(self):
        self.source.validate()
        rows = {Path(row["target"]).stem: row for row in verify_sources(self.PROFILE)}
        if rows != self.files:
            raise ValueError("official helper source binding changed after construction")
        return dict(namespace=self.namespace, upstream_namespace=self.source.namespace,
                    files={key: row["integrated_sha256"] for key, row in sorted(rows.items())})

    def find_spec(self, fullname, path=None, target=None):
        if fullname == self.namespace:
            return importlib.util.spec_from_loader(fullname, self, is_package=True)
        prefix = self.namespace + "."
        if not fullname.startswith(prefix):
            return None
        name = fullname[len(prefix):]
        if name not in self.files:
            raise ModuleNotFoundError("module is outside the official helper source closure: " + name)
        return importlib.util.spec_from_loader(fullname, self, is_package=False)

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        self.validate()
        if module.__name__ == self.namespace:
            module.__path__ = []
            return
        name = module.__name__.removeprefix(self.namespace + ".")
        path = self.root / (name + ".py")
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != self.files[name]["integrated_sha256"]:
            raise ValueError("official helper changed before compilation: " + name)
        module.__file__ = str(path)
        tree = ast.parse(payload, filename=str(path))
        tree = ast.fix_missing_locations(_Imports(self.source.namespace, self.source.import_roots).visit(tree))
        exec(compile(tree, str(path), "exec"), module.__dict__)
        self.validate()

    def module(self, name):
        if not isinstance(name, str) or name not in self.files:
            raise ValueError("unknown helper outside the reviewed source closure: " + str(name))
        self.validate()
        for finder in (self.source, self):
            if not any(getattr(item, "namespace", None) == finder.namespace for item in sys.meta_path):
                sys.meta_path.append(finder)
        return importlib.import_module(self.namespace + "." + name)
