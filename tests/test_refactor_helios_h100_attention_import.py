"""CPU import regression, not CUDA/model/attention numerical evidence."""
from __future__ import annotations

import hashlib
import importlib
import sys
from pathlib import Path
from types import ModuleType

import pytest

from distillschedule.runtime import helios_attention_import as subject


DISPATCH = """import torch
from kernels import get_kernel
try:
    major, _ = torch.cuda.get_device_capability()
    if major < 9:
        raise RuntimeError('not Hopper')
    flash_attn3 = get_kernel('kernels-community/flash-attn3')
    flash_attn_func = flash_attn3.flash_attn_func
    flash_attn_varlen_func = flash_attn3.flash_attn_varlen_func
except (ImportError, RuntimeError):
    try:
        flash_attn2 = get_kernel('kernels-community/flash-attn2')
        flash_attn_func = flash_attn2.flash_attn_func
        flash_attn_varlen_func = flash_attn2.flash_attn_varlen_func
    except ImportError:
        flash_attn_func = flash_attn_varlen_func = None
sageattn = sageattn_varlen = xformers_attn_func = None
"""


@pytest.fixture
def hopper(tmp_path, monkeypatch):
    package = tmp_path / 'helios/modules/helios_kernels'
    package.mkdir(parents=True)
    for parent in (tmp_path / 'helios', tmp_path / 'helios/modules', package):
        (parent / '__init__.py').write_text('')
    source = package / 'attention_dispatch.py'
    source.write_text(DISPATCH)
    monkeypatch.syspath_prepend(str(tmp_path))
    for key in list(sys.modules):
        if key == 'helios' or key.startswith('helios.'):
            monkeypatch.delitem(sys.modules, key)
    torch = ModuleType('torch')
    torch.cuda = ModuleType('cuda')
    torch.cuda.get_device_capability = lambda: (9, 0)
    monkeypatch.setitem(sys.modules, 'torch', torch)
    kernels = ModuleType('kernels')
    calls = []
    def offline_loader(repo, *args, **kwargs):
        calls.append(repo)
        if repo == subject.FA2 and kwargs.get('fixture_fail') is not True:
            def function(*args, **kwargs):
                return 'fixture-fa2'
            function.__module__ = 'flash_attn2_fixture'
            result = ModuleType('flash_attn2_fixture')
            result.flash_attn_func = result.flash_attn_varlen_func = function
            return result
        raise LookupError('offline hub failure must never be suppressed')
    kernels.get_kernel = offline_loader
    monkeypatch.setitem(sys.modules, 'kernels', kernels)
    monkeypatch.setenv('LOCAL_KERNELS', subject.FA2 + '=' + str(tmp_path))
    monkeypatch.setattr(subject, 'DISPATCH_SHA256', hashlib.sha256(source.read_bytes()).hexdigest())
    yield source, kernels, calls
    for key in list(sys.modules):
        if key == 'helios' or key.startswith('helios.'):
            sys.modules.pop(key, None)


def test_native_hopper_import_reproduces_offline_lookup_failure(hopper):
    _, _, calls = hopper
    with pytest.raises(LookupError, match='offline hub'):
        importlib.import_module(subject.DISPATCH_MODULE)
    assert calls == [subject.FA3]


def test_sdpa_hopper_does_not_lookup_any_external_kernel(hopper, monkeypatch):
    _, kernels, calls = hopper
    monkeypatch.delenv('LOCAL_KERNELS')
    original = kernels.get_kernel
    dispatcher, receipt = subject.import_registered_attention_backend('sdpa')
    assert dispatcher.flash_attn_func is dispatcher.flash_attn_varlen_func is None
    assert calls == []
    assert receipt['declined_repositories'] == [subject.FA3, subject.FA2]
    assert kernels.get_kernel is dispatcher.get_kernel is original
    assert receipt['loader_restored'] and not receipt['frozen_source_modified']


def test_fa2_hopper_uses_only_existing_local_fa2_not_fa3(hopper):
    _, kernels, calls = hopper
    original = kernels.get_kernel
    dispatcher, receipt = subject.import_registered_attention_backend('local-fa2')
    assert calls == [subject.FA2]
    assert receipt['declined_repositories'] == [subject.FA3]
    assert dispatcher.flash_attn_func.__module__ == 'flash_attn2_fixture'
    assert kernels.get_kernel is dispatcher.get_kernel is original


def test_source_mismatch_fails_before_import_or_loader_mutation(hopper, monkeypatch):
    source, kernels, calls = hopper
    monkeypatch.delenv('LOCAL_KERNELS')
    source.write_text(DISPATCH + '\n# drift\n')
    original = kernels.get_kernel
    with pytest.raises(RuntimeError, match='source SHA'):
        subject.import_registered_attention_backend('sdpa')
    assert kernels.get_kernel is original and calls == []
    assert subject.DISPATCH_MODULE not in sys.modules


def test_real_fa2_import_error_is_not_silently_promoted(hopper):
    _, kernels, _ = hopper
    def fail(repo, *args, **kwargs):
        raise LookupError('actual FA2 ABI failure')
    kernels.get_kernel = fail
    with pytest.raises(LookupError, match='FA2 ABI'):
        subject.import_registered_attention_backend('local-fa2')
    assert kernels.get_kernel is fail


def test_missing_fa2_cannot_fall_back_to_sdpa_and_claim_pass(hopper):
    _, kernels, _ = hopper
    def missing(repo, *args, **kwargs):
        raise ImportError('required local FA2 is absent')
    kernels.get_kernel = missing
    with pytest.raises(RuntimeError, match='both pinned FA2'):
        subject.import_registered_attention_backend('local-fa2')
    assert kernels.get_kernel is missing


def test_preloaded_dispatcher_is_rejected_instead_of_switching_live_backend(hopper):
    _, _, _ = hopper
    sys.modules[subject.DISPATCH_MODULE] = ModuleType(subject.DISPATCH_MODULE)
    with pytest.raises(RuntimeError, match='already imported'):
        subject.import_registered_attention_backend('local-fa2')


def test_sdpa_rejects_fa2_override_and_fa2_requires_explicit_override(hopper, monkeypatch):
    with pytest.raises(ValueError, match='LOCAL_KERNELS'):
        subject.import_registered_attention_backend('sdpa')
    monkeypatch.delenv('LOCAL_KERNELS')
    with pytest.raises(ValueError, match='LOCAL_KERNELS'):
        subject.import_registered_attention_backend('local-fa2')


def test_other_backend_cannot_be_silently_selected(hopper):
    with pytest.raises(ValueError, match='unsupported'):
        subject.import_registered_attention_backend('fa3')


def test_non_h100_worker_keeps_original_import_path():
    import ast
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / 'scripts/distillschedule/probe_static_fsdp2_p0_world16.py').read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_attention_backend_receipt')
    guard = next(node for node in ast.walk(function) if isinstance(node, ast.If) and 'DISTILLSCHEDULE_HARDWARE_PROFILE' in ast.unparse(node.test))
    assert 'h100-8-b16' in ast.unparse(guard.test)
    assert 'import_registered_attention_backend' in ast.unparse(guard)


def test_registered_backend_is_bound_before_gan_hooks_can_import_helios():
    import ast
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / 'scripts/distillschedule/probe_static_fsdp2_p0_world16.py').read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    names = {'_attention_backend_receipt', 'install_critic_scalar_log_elision', 'prepare_generator_free_memory_receipt'}
    calls = sorted((node.lineno, node.func.id) for node in ast.walk(main)
                   if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in names)
    assert [name for _, name in calls][0] == '_attention_backend_receipt'
