"""Hash-locked, opt-in imports of externally licensed upstream source.

No source is fetched, copied into the distribution, or added to ``sys.path``.
Only import names are rewritten; selected definitions retain their original AST
and line numbers. The caller owns the source lock and any definition globals.
"""
from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.abc
import importlib.util
import json
from pathlib import Path, PurePosixPath
import sys


class _Imports(ast.NodeTransformer):
    def __init__(self, namespace, roots):
        self.namespace, self.roots = namespace, roots

    def _local(self, name):
        return name and name.split('.')[0] in self.roots

    def visit_ImportFrom(self, node):
        if node.level == 0 and self._local(node.module):
            node.module = self.namespace + '.' + node.module
        return node

    def visit_Import(self, node):
        rows = []
        for alias in node.names:
            if self._local(alias.name):
                if alias.asname:
                    rows.append(ast.Import(names=[ast.alias(
                        name=self.namespace + '.' + alias.name, asname=alias.asname)]))
                else:
                    # ``import a.b`` binds a, not b. Retain both semantics.
                    rows.append(ast.Import(names=[ast.alias(
                        name=self.namespace + '.' + alias.name, asname=None)]))
                    rows.append(ast.ImportFrom(module=self.namespace, names=[ast.alias(
                        name=alias.name.split('.')[0], asname=None)], level=0))
            else:
                rows.append(ast.Import(names=[alias]))
        return [ast.copy_location(row, node) for row in rows]


class PinnedSource(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """An explicit source root with an exact relative-path SHA256 allowlist.

    ``definitions`` optionally restricts a module to named top-level functions
    and classes; ``definition_globals`` supplies their explicit import globals.
    This is import compatibility, not a sandbox: the trusted source executes.
    """

    def __init__(self, source_root, files, *, import_roots=(), definitions=None,
                 definition_globals=None, python_roots=('',)):
        self.root = Path(source_root).expanduser().resolve(strict=True)
        if not self.root.is_dir() or not isinstance(files, dict) or not files:
            raise ValueError('pinned source requires a directory and file hashes')
        self.files = dict(files)
        for name, sha in self.files.items():
            path = PurePosixPath(name)
            if (path.is_absolute() or path.as_posix() != name or '..' in path.parts
                    or not isinstance(sha, str) or len(sha) != 64
                    or any(c not in '0123456789abcdef' for c in sha)):
                raise ValueError('invalid pinned source path/hash')
        self.import_roots = tuple(import_roots)
        if any(not root.isidentifier() for root in self.import_roots):
            raise ValueError('invalid upstream import root')
        self.python_roots = tuple(python_roots)
        if (not self.python_roots or len(set(self.python_roots)) != len(self.python_roots)
                or any(not isinstance(root, str) or root and (
                    PurePosixPath(root).is_absolute() or '..' in PurePosixPath(root).parts
                    or PurePosixPath(root).as_posix() != root) for root in self.python_roots)):
            raise ValueError('invalid upstream python root')
        self.definitions = {key: tuple(value) for key, value in (definitions or {}).items()}
        self.definition_globals = dict(definition_globals or {})
        if set(self.definition_globals) - set(self.definitions):
            raise ValueError('definition globals require an explicit definition selection')
        self.identity = {'files': dict(sorted(self.files.items())),
                         'import_roots': list(self.import_roots),
                         'definitions': {key: list(value) for key, value in sorted(self.definitions.items())}}
        if self.python_roots != ('',):
            self.identity['python_roots'] = list(self.python_roots)
        lock = json.dumps(self.identity, sort_keys=True).encode()
        digest = hashlib.sha256(str(self.root).encode() + b'\0' + lock).hexdigest()[:24]
        self.namespace = '_distillschedule_upstream_' + digest
        self.validate()

    def validate(self):
        for name, digest in self.files.items():
            path = self.root / name
            if not path.resolve().is_relative_to(self.root) or not path.is_file():
                raise ValueError('pinned source file is missing or escapes root: ' + name)
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError('pinned source SHA256 mismatch: ' + name)
        return self.identity

    def __deepcopy__(self, memo):
        # Imported utility globals are not mutable training state. The owner
        # still independently deep-copies its model and optimizer graph.
        self.validate()
        return self

    def _entry(self, fullname):
        if fullname == self.namespace:
            return None, True
        relative = fullname.removeprefix(self.namespace + '.').replace('.', '/')
        candidates = []
        namespace = False
        for root in self.python_roots:
            path = (root + '/' if root else '') + relative
            for entry, package in ((path + '.py', False), (path + '/__init__.py', True)):
                if entry in self.files:
                    candidates.append((entry, package))
            namespace = namespace or any(name.startswith(path + '/') for name in self.files)
        if len(candidates) > 1:
            raise ValueError('ambiguous upstream python roots for ' + relative)
        if candidates:
            return candidates[0]
        if namespace:
            return None, True
        return None, False

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self.namespace and not fullname.startswith(self.namespace + '.'):
            return None
        entry, package = self._entry(fullname)
        if entry is None and not package:
            raise ModuleNotFoundError('module is outside pinned source closure: ' + fullname)
        return importlib.util.spec_from_loader(fullname, self, is_package=package)

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        self.validate()
        entry, package = self._entry(module.__name__)
        if package:
            module.__path__ = []
        if entry is None:
            return
        path = self.root / entry
        module.__file__ = str(path)
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != self.files[entry]:
            raise ValueError('pinned source changed before compilation: ' + entry)
        tree = ast.parse(payload, filename=str(path))
        relative = module.__name__[len(self.namespace) + 1:]
        if relative in self.definitions:
            names = self.definitions[relative]
            selected = [node for node in tree.body if isinstance(node,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name in names]
            if len(selected) != len(names) or {node.name for node in selected} != set(names):
                raise ValueError('selected upstream definitions are missing or duplicated')
            tree.body = selected
            module.__dict__.update(self.definition_globals.get(relative, {}))
        tree = ast.fix_missing_locations(_Imports(self.namespace, self.import_roots).visit(tree))
        exec(compile(tree, str(path), 'exec'), module.__dict__)
        self.validate()

    def import_module(self, relative_module):
        if (not isinstance(relative_module, str) or not relative_module
                or any(not part.isidentifier() for part in relative_module.split('.'))):
            raise ValueError('invalid upstream module name')
        self.validate()  # Also check cached imports, not only the first import.
        if not any(getattr(finder, 'namespace', None) == self.namespace for finder in sys.meta_path):
            sys.meta_path.append(self)
        return importlib.import_module(self.namespace + '.' + relative_module)
