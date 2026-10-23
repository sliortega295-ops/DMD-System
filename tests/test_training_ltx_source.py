"""LTX discovery and local source admission must not import Torch or CUDA."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_source_lock_and_catalog_are_lazy():
    script = '''
import sys
from distillschedule.training.catalog import resolve
from distillschedule.training.ltx_source import REVISION, SOURCE_FILES
assert resolve('model', 'ltx25')['factory'].endswith('ltx:build_model')
assert REVISION == 'e98f6d927d83c92ef2be6aec76c8440b19306e39'
assert len(SOURCE_FILES) == 114
assert 'torch' not in sys.modules and 'transformers' not in sys.modules
'''
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_source_changed_or_incomplete_fails_closed(tmp_path):
    from distillschedule.training.ltx_source import original
    with pytest.raises(ValueError, match='missing'):
        original(tmp_path)


def test_native_source_identity_when_explicitly_available():
    from distillschedule.training.ltx_source import original, REVISION
    root = os.environ.get('LTX_SOURCE_ROOT')
    if not root:
        pytest.skip('set LTX_SOURCE_ROOT for the optional upstream source acceptance')
    api = original(Path(root))
    assert api.model.LTXModel.__module__.startswith(api.source.namespace)
    assert api.identity['revision'] == REVISION
    assert api.source.python_roots == ('packages/ltx-core/src',)


def test_changed_locked_source_cannot_import(tmp_path):
    import shutil
    root = os.environ.get('LTX_SOURCE_ROOT')
    if not root:
        pytest.skip('set LTX_SOURCE_ROOT for the optional source-integrity test')
    from distillschedule.training.ltx_source import original
    copied = tmp_path / 'source'; shutil.copytree(root, copied)
    path = copied / 'packages/ltx-core/src/ltx_core/model/transformer/model.py'
    path.write_text(path.read_text() + '\n# changed source\n')
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        original(copied)


def test_native_local_states_reject_empty_nonfinite_meta_wrong_layout_and_hash(tmp_path):
    import torch
    from distillschedule.training.ltx_media import strict_load, file_sha256
    module = torch.nn.Linear(2, 3)
    values = [{}, {'weight': torch.full((3, 2), float('nan')), 'bias': torch.zeros(3)},
        {'weight': torch.empty(3, 2, device='meta'), 'bias': torch.empty(3, device='meta')},
        {'weight': torch.zeros(3, 2, dtype=torch.float64), 'bias': torch.zeros(3)},
        {'weight': torch.zeros(2, 3), 'bias': torch.zeros(3)}, {'bias': torch.zeros(3)}]
    for index, value in enumerate(values):
        path = tmp_path / f'bad-{index}.pt'; torch.save(value, path)
        with pytest.raises(ValueError):
            strict_load(module, path, file_sha256(path))
    path = tmp_path / 'valid.pt'; torch.save(module.state_dict(), path)
    with pytest.raises(ValueError, match='hash changed'):
        strict_load(module, path, '0' * 64)
