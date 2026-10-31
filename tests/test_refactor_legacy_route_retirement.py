"""Removed routes cannot remain import dependencies of current execution."""
import ast
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def manifest():
    return json.loads((ROOT / 'docs/architecture/legacy_route_retirement.json').read_text())


def test_retired_entrypoints_are_absent_and_shared_dependencies_remain():
    data = manifest()
    assert len(data['recovery_commit']) == 40
    paths = [r['path'] for r in data['files']]
    assert len(paths) == len(set(paths))
    for row in data['files']:
        assert not (ROOT / row['path']).exists(), row['path']
        assert len(row['sha256']) == 64
    for path in data['retained_shared_dependencies']:
        assert (ROOT / path).exists(), path


def test_current_python_import_closure_excludes_removed_modules():
    retired = {r['path'][:-3].removeprefix('src/').replace('/', '.')
               for r in manifest()['files'] if r['path'].endswith('.py')}
    for directory in ('src', 'runtime', 'scripts', 'tools', 'tests'):
        for path in (ROOT / directory).rglob('*.py'):
            if '__pycache__' in path.parts:
                continue
            relative = path.relative_to(ROOT).as_posix()
            module = relative.removeprefix('src/')[:-3].replace('/', '.')
            package = module.rsplit('.', 1)[0]
            for node in ast.walk(ast.parse(path.read_text())):
                candidates = []
                if isinstance(node, ast.Import):
                    candidates = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    name = node.module or ''
                    if node.level:
                        name = importlib.util.resolve_name('.' * node.level + name, package)
                    candidates = [name] + [name + '.' + a.name for a in node.names]
                assert not retired.intersection(candidates), (relative, candidates)


def test_retirement_does_not_delete_imported_or_publication_sources():
    retired = {r['path'] for r in manifest()['files']}
    imported = json.loads((ROOT / 'docs/workloads/source_import_map.json').read_text())['files']
    assert not retired.intersection(r['target'] for r in imported)
    assert not any(p.startswith(('third_party/', 'artifacts/publications/')) for p in retired)
