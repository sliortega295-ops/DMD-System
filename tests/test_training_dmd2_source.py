"""Pinned external importer mechanics; not a substitute numerical backend."""
import ast
import hashlib
import inspect
from pathlib import Path
import sys

import pytest

from distillschedule.training.upstream import PinnedSource


def write(root, name, content):
    path = root / name; path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_private_whole_file_imports_and_exact_selected_ast(tmp_path):
    hashes = {
        'main/helper.py': write(tmp_path, 'main/helper.py',
            'from unavailable_plotting_library import nope\ndef exact(value):\n    return value * factor\n'),
        'main/model.py': write(tmp_path, 'main/model.py',
            'from main.helper import exact\nimport main.helper\ndef call(value):\n    return main.helper.exact(exact(value))\n'),
    }
    before = {key: sys.modules.get(key) for key in ('main', 'main.helper', 'main.model')}
    source = PinnedSource(tmp_path, hashes, import_roots=('main',),
        definitions={'main.helper': ('exact',)}, definition_globals={'main.helper': {'factor': 3}})
    model = source.import_module('main.model')
    assert model.call(2) == 18
    assert model.__name__.startswith(source.namespace + '.')
    assert {key: sys.modules.get(key) for key in before} == before
    assert 'unavailable_plotting_library' not in sys.modules
    function = source.import_module('main.helper').exact
    actual = ast.dump(ast.parse(inspect.getsource(function)))
    expected = ast.dump(ast.Module(body=[ast.parse((tmp_path / 'main/helper.py').read_text()).body[1]], type_ignores=[]))
    assert actual == expected, 'numerical definition AST must remain exactly original'
    with pytest.raises(ModuleNotFoundError, match='outside pinned source closure'):
        source.import_module('main.unlisted')


@pytest.mark.parametrize('bad', ('changed', 'missing', 'escape'))
def test_hash_checked_before_cached_import_or_code_execution(tmp_path, bad):
    root = tmp_path / 'source'; root.mkdir()
    digest = write(root, 'model.py', 'value = 1\n')
    source = PinnedSource(root, {'model.py': digest})
    assert source.import_module('model').value == 1
    if bad == 'changed':
        (root / 'model.py').write_text('raise AssertionError("unverified code must not execute")\n')
    elif bad == 'missing':
        (root / 'model.py').unlink()
    else:
        (root / 'model.py').unlink()
        outside = tmp_path / 'outside.py'; outside.write_text('value = 1\n')
        (root / 'model.py').symlink_to(outside)
    with pytest.raises(ValueError, match='pinned source'):
        source.import_module('model')


def test_distinct_upstream_roots_can_coexist_without_global_aliases(tmp_path):
    mounts = []
    for index, rootname in enumerate(('utils', 'ltx_trainer')):
        root = tmp_path / str(index)
        hashes = {f'{rootname}/helper.py': write(root, f'{rootname}/helper.py', f'VALUE = {index}\n'),
                  'entry.py': write(root, 'entry.py', f'from {rootname}.helper import VALUE\n')}
        mounts.append(PinnedSource(root, hashes, import_roots=(rootname,)))
    assert [mount.import_module('entry').VALUE for mount in mounts] == [0, 1]
    assert mounts[0].namespace != mounts[1].namespace


@pytest.mark.parametrize('files', ({'../escape.py': '0' * 64}, {'/absolute.py': '0' * 64},
                                  {'main.py': 'not-a-sha'}, {'main//file.py': '0' * 64}))
def test_invalid_source_locks_fail_closed(tmp_path, files):
    with pytest.raises(ValueError):
        PinnedSource(tmp_path, files)


def test_pinned_source_handles_project_python_roots_without_moving_source_files(tmp_path):
    name = 'packages/provider-core/src/provider/model.py'
    hashes = {name: write(tmp_path, name, 'VALUE = 17\n')}
    source = PinnedSource(tmp_path, hashes, import_roots=('provider',),
                          python_roots=('packages/provider-core/src',))
    assert source.import_module('provider.model').VALUE == 17
    assert source.identity['python_roots'] == ['packages/provider-core/src']
    with pytest.raises(ValueError, match='python root'):
        PinnedSource(tmp_path, hashes, python_roots=('../escape',))
