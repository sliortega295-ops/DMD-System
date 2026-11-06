"""Source-derived restore identities for mathematics and state ownership.

Computed at construction/admission, never in the cycle hot path. This follows
inspectable Python definitions and static imports, including function-local and
relative imports. Native adapters retain their upstream pins in model metadata;
opaque dynamic code is not inferred from a model name or an epoch alone.
"""
from __future__ import annotations

import ast
from copy import deepcopy
from functools import lru_cache
import importlib.metadata
import inspect
from pathlib import Path
import sys
import sysconfig

from distillschedule.workloads.best_known.registry import fingerprint as _digest

SCHEMA = 'distillschedule.resume_semantics.v1'
EPOCHS = {'mathematics': 1, 'updates': 1, 'state_encoding': 1, 'clocks': 1, 'execution': 1}

# Select the semantic portion of mixed lifecycle modules. Publication, logging,
# retention and planner cost/search budgets are deliberately not restore math.
_SCOPES = {
    'distillschedule.training.session': ('_seed', 'build_cycle'),
    'distillschedule.training.checkpoint': ('_class', '_parameter_names', 'pack_state',
        '_validate_layout', '_validate_optimizer_layout', '_stage_state', 'unpack_state',
        'rng_state', 'validate_rng', 'restore_rng'),
    'distillschedule.training.policy': ('cycle_structure',),
    'distillschedule.training.config': ('digest',),
    'distillschedule.workloads.best_known.registry': ('fingerprint',),
    'distillschedule.open_workload.torch_backend': ('random_scope', 'Context', 'Session._run'),
}
_EXCLUDED = {'distillschedule.open_workload.planning', 'distillschedule.training.semantics',
    'distillschedule.training.compatibility', 'distillschedule.training.inference',
    'distillschedule.training.validation', 'distillschedule.training.catalog',
    'distillschedule.data.preparation', 'distillschedule.data.export_inputs'}
# Type/admission allowlists name optional owners; their bodies are bound, but
# inspecting an unrelated candidate must not require all model dependencies.
_LEAVES = {('distillschedule.open_workload.collective_owner', 'admit'),
           ('distillschedule.open_workload.collective_owner', '_boundary'),
           # Bind clock/dispatch callsites without pulling publication,
           # retention or search implementations into training mathematics.
           ('distillschedule.training.session', '_train_owned'),
           ('distillschedule.training.policy', 'CyclePolicy._run')}
_COMMON = ('distillschedule.training.dmd', 'distillschedule.training.prediction',
    'distillschedule.training.low_rank', 'distillschedule.training.sampling',
    'distillschedule.training.data_streams', 'distillschedule.training.tensor_state',
    'distillschedule.training.checkpoint', 'distillschedule.training.session',
    'distillschedule.open_workload.dmd', 'distillschedule.open_workload.torch_backend',
    'distillschedule.workloads.program', 'distillschedule.runtime.role_runner',
    'distillschedule.runtime.version_sync')


class _WithoutDocstrings(ast.NodeTransformer):
    def generic_visit(self, node):
        super().generic_visit(node)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body.pop(0)
        return node


def _normalized(node):
    return ast.dump(_WithoutDocstrings().visit(deepcopy(node)), include_attributes=False)


class _Initialization(ast.NodeTransformer):
    """Keep import-time execution, not dormant function bodies.

    Defaults, decorators and class bodies execute while importing a module.
    Their source therefore matters even if the function is never called.
    """
    def visit_FunctionDef(self, node):
        node.body = [ast.Pass()]
        return node

    visit_AsyncFunctionDef = visit_FunctionDef


def _loaded_names(node):
    return {item.id for item in ast.walk(node)
            if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load)}


class _InitializationNames(ast.NodeVisitor):
    def __init__(self):
        self.names = set()

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            self.names.add(node.id)

    def visit_Lambda(self, node):
        # Creating an unused callback does not execute its body. A referenced
        # callback definition is inspected normally if math actually uses it.
        self.visit(node.args)

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Lambda):
            self.visit(node.func.body)


def _initialization(tree, scopes, names):
    projection = _Initialization().visit(deepcopy(tree))
    if scopes is None:
        return projection
    # Mixed lifecycle modules have an explicit semantic boundary. Expand only
    # its local definition dependencies, then include initialization affecting
    # those globals (including aliases and control-flow statements). This keeps
    # unrelated retention/catalog definitions outside the math identity.
    needed = set()
    pending = []
    for symbol in scopes:
        node = names.get(symbol.split('.')[0])
        for part in symbol.split('.')[1:]:
            node = _bindings(getattr(node, 'body', ())).get(part)
        if node is not None:
            pending.append(node)
    expanded = set()
    while pending:
        node = pending.pop()
        if node in expanded:
            continue
        expanded.add(node)
        references = _loaded_names(node)
        needed.update(references)
        pending.extend(names[reference] for reference in references
                       if reference in names and names[reference] not in expanded
                       and not isinstance(names[reference], (ast.Import, ast.ImportFrom)))
    candidates = []
    for node in projection.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                             ast.Import, ast.ImportFrom)):
            continue
        candidates.append(node)
    selected = set()
    while True:
        before = set(needed)
        for node in candidates:
            touched = {item.id for item in ast.walk(node) if isinstance(item, ast.Name)}
            # Standalone calls may mutate a relevant global without passing it
            # explicitly. Bind them conservatively rather than infer purity.
            if needed.intersection(touched) or isinstance(node, ast.Expr):
                selected.add(node)
                needed.update(touched)
        if before == needed:
            break
    projection.body = [node for node in candidates if node in selected]
    return projection


@lru_cache(maxsize=1)
def _distributions():
    return importlib.metadata.packages_distributions()


def _source_path(name):
    """Resolve source without importing packages (or initializing a backend)."""
    loaded = sys.modules.get(name)
    path = getattr(loaded, '__file__', None)
    if path and str(path).endswith('.py') and Path(path).is_file():
        return Path(path).resolve()
    parts = name.split('.')
    for root in sys.path:
        base = Path(root or '.').joinpath(*parts)
        for candidate in (base.with_suffix('.py'), base / '__init__.py'):
            if candidate.is_file():
                return candidate.resolve()
    return None


def _bindings(body):
    result = {}
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            result[node.name] = node
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                result[alias.asname or alias.name.split('.')[0]] = node
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                for name in ast.walk(target):
                    if isinstance(name, ast.Name):
                        result[name.id] = node
        elif isinstance(node, (ast.If, ast.Try)):
            # Keep the predicate and every alternative. Flattening to the last
            # definition would miss edits in the branch actually executing.
            bound = {}
            for child in ('body', 'orelse', 'finalbody'):
                bound.update(_bindings(getattr(node, child, ())))
            for handler in getattr(node, 'handlers', ()):
                bound.update(_bindings(handler.body))
            result.update(dict.fromkeys(bound, node))
    return result


class _Dependencies:
    def __init__(self):
        self.dependencies = {}
        self.external = {'python': f'{sys.version_info.major}.{sys.version_info.minor}'}
        self.external_prefixes = set()
        self.parsed = {}
        self.attributes = {}
        self.initialized = set()
        self.visited = set()
        self.owned = {"distillschedule", "runtime"}

    def external_module(self, name):
        top = name.split('.')[0]
        if top in sys.stdlib_module_names or top == '__future__':
            return
        if top in self.external_prefixes:
            return
        owners = _distributions().get(top, ())
        if not owners:
            raise ValueError('resume semantics cannot identify external dependency: ' + name)
        for owner in owners:
            self.external[owner] = importlib.metadata.version(owner)
        self.external_prefixes.add(top)

    def module(self, name, symbol=None, *, path=None):
        if name in _EXCLUDED:
            return
        if name in _SCOPES and symbol is None:
            for selected in _SCOPES[name]:
                self._definition(name, selected, path)
            return
        self._definition(name, symbol, path)

    def _definition(self, name, symbol, path):
        key = (name, symbol)
        if key in self.visited:
            return
        self.visited.add(key)
        if len(self.visited) > 2048:
            raise ValueError('resume semantics dependency graph exceeds its bounded inspection scope')
        if name.split('.')[0] in sys.stdlib_module_names or name == '__future__':
            return
        top = name.split('.')[0]
        # Validated environments may mount Torch/model dependencies outside
        # purelib. Distribution ownership, not one directory spelling, defines
        # that external version boundary. Our/user adapter roots override it.
        if top not in self.owned and _distributions().get(top):
            self.external_module(name)
            return
        path = path or _source_path(name)
        # Installed dependencies own their own implementation/version boundary.
        # Never crawl all of Torch/diffusers just to admit a checkpoint.
        sites = [Path(sysconfig.get_paths()['purelib']).resolve(),
                 *(Path(root).resolve() for root in sys.path
                   if Path(root).name in ('site-packages', 'dist-packages'))]
        if path is None or (any(path.is_relative_to(site) for site in sites)
                            and name.split('.')[0] not in self.owned):
            self.external_module(name)
            return
        if name not in self.parsed:
            tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
            self.parsed[name] = (path, tree, _bindings(tree.body))
        path, tree, names = self.parsed[name]
        if symbol is None:
            nodes = [tree]
        else:
            node = names.get(symbol.split('.')[0])
            for part in symbol.split('.')[1:]:
                node = _bindings(getattr(node, 'body', ())).get(part)
            if node is None:
                child = name + '.' + symbol
                if _source_path(child) is not None:
                    self.module(child)
                # Module attributes supplied by runtime branches are already
                # covered when the owning module is selected as a whole.
                else:
                    self.module(name)
                return
            nodes = [node]
        label = name + (':' + symbol if symbol else ':*')
        self.dependencies[label] = _digest([_normalized(node) for node in nodes])
        if symbol is not None and name not in self.initialized:
            self.initialized.add(name)
            initial = _initialization(tree, _SCOPES.get(name), names)
            self.dependencies[name + ':<module-initialization>'] = _digest(_normalized(initial))
            # Import declarations alone must not crawl unused optional model
            # packages. Calls/defaults executed during initialization do follow
            # their real imported/local helpers below.
            self._references(name, [initial], names, path, local_imports=False)
        if (name, symbol) in _LEAVES:
            return
        self._references(name, nodes, names, path)

    def _references(self, name, nodes, names, path, *, local_imports=True):
        package = name if path.name == '__init__.py' else name.rpartition('.')[0]
        for node in nodes:
            if local_imports:
                references = _loaded_names(node)
            else:
                initialization = _InitializationNames()
                initialization.visit(node)
                references = initialization.names
            # Each local import has its own scope: two methods may reuse the
            # same alias for different helpers. Never flatten one over another.
            imports = ([item for item in ast.walk(node)
                        if isinstance(item, (ast.Import, ast.ImportFrom))] if local_imports else [])
            for imported in imports:
                for alias in imported.names:
                    self._import(imported, alias.asname or alias.name.split('.')[0], node, package)
            for reference in references:
                definition = names.get(reference)
                if isinstance(definition, (ast.Import, ast.ImportFrom)):
                    self._import(definition, reference, node, package)
                elif definition is not None and definition is not node:
                    self._definition(name, reference, path)

    def _import(self, node, reference, scope, package):
        alias = next(item for item in node.names
                     if (item.asname or item.name.split('.')[0]) == reference)
        if scope not in self.attributes:
            index = {}
            for item in ast.walk(scope):
                if not isinstance(item, ast.Attribute):
                    continue
                parts, root = [], item
                while isinstance(root, ast.Attribute):
                    parts.insert(0, root.attr)
                    root = root.value
                if isinstance(root, ast.Name):
                    index.setdefault(root.id, []).append(parts)
            self.attributes[scope] = index
        attributes = self.attributes[scope].get(reference, [])
        if isinstance(node, ast.ImportFrom):
            if node.level:
                parents = package.split('.')
                if node.level > len(parents):
                    raise ValueError('resume semantics cannot resolve relative import')
                base = '.'.join(parents[:len(parents) - node.level + 1])
                target = '.'.join(filter(None, (base, node.module)))
            else:
                target = node.module
            if alias.name == '*':
                self.module(target)
                return
            child = target + '.' + alias.name
            if attributes and _source_path(child) is not None:
                for parts in attributes:
                    self.module(child, parts[0])
            else:
                self.module(target, alias.name)
        else:
            target = alias.name if alias.asname else alias.name.split('.')[0]
            if not attributes:
                self.module(target)
            for parts in attributes:
                dotted = target + '.' + '.'.join(parts)
                while _source_path(dotted) is None and '.' in dotted:
                    dotted = dotted.rpartition('.')[0]
                if _source_path(dotted) is None:
                    self.external_module(target)
                else:
                    symbol = (target + '.' + '.'.join(parts))[len(dotted):].lstrip('.')
                    self.module(dotted, symbol or None)

    def callable(self, value, *, owned=False):
        value = inspect.unwrap(getattr(value, '__func__', value))
        if not (inspect.isfunction(value) or inspect.isclass(value)):
            value = type(value)
        name, symbol = value.__module__, value.__qualname__
        if owned:
            self.owned.add(name.split(".")[0])
        if '<locals>' in symbol:
            # Its enclosing definition contains the closure code and imports.
            symbol = symbol.split('.<locals>', 1)[0]
        try:
            source = inspect.getsourcefile(value)
        except TypeError:
            source = None
        self.module(name, symbol, path=Path(source).resolve() if source else None)

    def state(self, state):
        import torch
        from torch.distributed.fsdp import FSDPModule
        from distillschedule.open_workload.torch_backend import unwrap_module

        for value in state.values():
            if hasattr(value, 'state_dict'):
                if isinstance(value, torch.nn.Module):
                    value = unwrap_module(value)
                else:
                    self.callable(type(value))
                if callable(getattr(value, 'modules', None)):
                    # DDP adds an outer wrapper; fully_shard replaces each
                    # root/child's class. The selected physical lifecycle is
                    # bound separately, not mistaken for model source code.
                    for module in value.modules():
                        cls = (next(cls for cls in type(module).__mro__
                                    if issubclass(cls, torch.nn.Module)
                                    and not issubclass(cls, FSDPModule))
                               if isinstance(module, FSDPModule) else type(module))
                        self.callable(cls)

    def descriptor(self):
        body = {'schema_version': SCHEMA, 'epochs': dict(EPOCHS),
                'dependencies': dict(sorted(self.dependencies.items())),
                'external': dict(sorted(self.external.items()))}
        return {**body, 'sha256': _digest(body)}


def valid(value):
    if not isinstance(value, dict) or set(value) != {'schema_version', 'epochs', 'dependencies', 'external', 'sha256'}:
        return False
    if (value['schema_version'] != SCHEMA or value['epochs'] != EPOCHS
            or not isinstance(value['epochs'], dict)
            or any(type(epoch) is not int for epoch in value['epochs'].values())):
        return False
    dependencies, external = value['dependencies'], value['external']
    if (not isinstance(dependencies, dict) or not dependencies or not isinstance(external, dict)
            or not all(isinstance(key, str) and key and isinstance(item, str) and item
                       for key, item in (*dependencies.items(), *external.items()))
            or not all(len(item) == 64 and all(c in '0123456789abcdef' for c in item)
                       for item in dependencies.values())):
        return False
    body = {key: item for key, item in value.items() if key != 'sha256'}
    return value['sha256'] == _digest(body)


def training_semantics(factory, algorithm, state):
    collector = _Dependencies()
    collector.owned.update(value.__module__.split('.')[0] for value in (factory, algorithm))
    collector.callable(factory, owned=True)
    collector.callable(algorithm, owned=True)
    for name in _COMMON:
        collector.module(name)
    collector.state(state)
    return collector.descriptor()


def execution_semantics(workload, implementation):
    collector = _Dependencies()
    collector.module('distillschedule.training.session', '_train_owned')
    collector.module('distillschedule.training.policy', 'CyclePolicy._run')
    collector.owned.update(value.__module__.split('.')[0] for value in workload.bindings.values())
    for binding in workload.bindings.values():
        collector.callable(binding, owned=True)
    for name in _COMMON:
        collector.module(name)
    collector.state(workload.state)
    # Physical lifecycle changes can alter gradients/ownership independently of
    # the algorithm entry. Bind only the actually selected backend family.
    reduction = implementation.get('reduction', '')
    if reduction == 'per_module':
        collector.module('distillschedule.open_workload.mixed_layout', 'MixedLifecycle')
        collector.module('distillschedule.open_workload.mixed_layout', 'graph')
    elif reduction.startswith(('fsdp2_', 'hsdp2_')):
        collector.module('distillschedule.open_workload.sharded', 'ShardedLifecycle')
    elif reduction:
        collector.module('distillschedule.open_workload.replicated', 'ReplicaLifecycle')
    if implementation.get('backend') == 'torch-callable-native-owner-v1':
        collector.module('distillschedule.open_workload.collective_owner', 'NativeOwnerLifecycle')
    if implementation.get('owner_execution'):
        collector.module('distillschedule.open_workload.owner_execution')
    return collector.descriptor()
