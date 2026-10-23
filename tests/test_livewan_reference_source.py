"""Offline source admission of the private full-original Wan/LiveWan provider."""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import time

import pytest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src/distillschedule/reference/livewan_wan21"


def fresh(code, *args, timeout=120):
    child = subprocess.Popen(
        [sys.executable, "-B", "-c", textwrap.dedent(code), *map(str, args)],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
                 HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"),
    )
    deadline = time.monotonic() + timeout
    try:
        while True:
            status = Path(f"/proc/{child.pid}/status")
            if status.exists():
                rss = next((line.split()[1] for line in status.read_text().splitlines()
                            if line.startswith("VmRSS:")), "0")
                assert int(rss) * 1024 <= 2 * 1024**3, "probe exceeded 2GiB RSS"
            assert time.monotonic() < deadline, "probe exceeded wall budget"
            try:
                stdout, stderr = child.communicate(timeout=0.1)
                break
            except subprocess.TimeoutExpired:
                continue
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate()
    assert child.returncode == 0, stdout + stderr
    return stdout


def test_lazy_identity_preserves_native_namespaces_and_does_not_import_torch():
    identity = json.loads(fresh('''
        import json, sys
        from distillschedule.reference import livewan_wan21 as provider
        identity = provider.source_identity()
        assert identity['scope'] == 'FULL_ORIGINAL_WAN21_LIVEWAN_CPU_FP32_REFERENCE'
        assert identity['backend'] == 'torch-sdpa-cpu'
        assert identity['precision'] == 'float32'
        assert not any(n == 'torch' or n.startswith('torch.') for n in sys.modules)
        assert not any(n == 'wan' or n.startswith('wan.') or n == 'wanstreamer'
                       or n.startswith('wanstreamer.') for n in sys.modules)
        print(json.dumps(identity))
    '''))
    assert identity["files_checked"] >= 12


@pytest.mark.parametrize("backend,precision", [(None, "float32"), ("auto", "float32"),
    ("cuda", "float32"), ("torch-sdpa-cpu", None), ("torch-sdpa-cpu", "bfloat16")])
def test_backend_and_precision_are_explicit_before_torch(backend, precision):
    fresh('''
        import json, sys
        from distillschedule.reference import livewan_wan21 as provider
        try:
            provider.load_cpu(backend=json.loads(sys.argv[1]), precision=json.loads(sys.argv[2]))
        except ValueError as exc:
            assert 'explicit' in str(exc)
        else:
            raise AssertionError('implicit backend/precision accepted')
        assert 'torch' not in sys.modules
    ''', json.dumps(backend), json.dumps(precision))


def test_backend_override_and_direct_numerical_import_fail_before_torch():
    fresh('''
        import importlib, os, sys
        from distillschedule.reference import livewan_wan21 as provider
        os.environ['WAN_SDPA_BACKEND'] = 'FLASH_ATTENTION'
        try:
            provider.load_cpu(backend='torch-sdpa-cpu', precision='float32')
        except ValueError as exc:
            assert 'override' in str(exc)
        else:
            raise AssertionError('GPU backend override accepted')
        del os.environ['WAN_SDPA_BACKEND']
        for path in ('wan.modules.model', 'wanstreamer.stream'):
            try:
                importlib.import_module(provider.__name__ + '.' + path)
            except ValueError as exc:
                assert 'load_cpu' in str(exc)
            else:
                raise AssertionError('direct numerical import bypassed admission')
        assert 'torch' not in sys.modules
    ''')


@pytest.mark.parametrize("target", ["wan/modules/model.py", "wanstreamer/blockcausal.py",
    "wan/modules/attention.py", "source_manifest.json", "licenses/LiveWan-LICENSE.txt"])
def test_changed_source_manifest_or_license_rejected_before_torch(tmp_path, target):
    fresh('''
        from pathlib import Path
        import shutil, sys
        from distillschedule.reference import livewan_wan21 as provider
        copy = Path(sys.argv[1]) / 'copy'
        shutil.copytree(provider._ROOT, copy, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        provider._ROOT = copy
        path = copy / sys.argv[2]
        path.write_bytes(path.read_bytes() + b'\\n# altered\\n')
        try:
            provider.load_cpu(backend='torch-sdpa-cpu', precision='float32')
        except ValueError as exc:
            assert 'identity' in str(exc)
        else:
            raise AssertionError('altered source/license admitted')
        assert 'torch' not in sys.modules
    ''', tmp_path, target)


@pytest.mark.parametrize("mode", ["file", "symlink", "cache"])
def test_foreign_inventory_symlink_and_preloaded_namespace_are_rejected(tmp_path, mode):
    fresh('''
        from pathlib import Path
        import shutil, sys, types
        from distillschedule.reference import livewan_wan21 as provider
        copy = Path(sys.argv[1]) / 'copy'
        shutil.copytree(provider._ROOT, copy, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        provider._ROOT = copy
        mode = sys.argv[2]
        if mode == 'file':
            (copy / 'wanstreamer/foreign.py').write_text('UNADMITTED = True')
        elif mode == 'symlink':
            source = copy / 'wan/modules/model.py'
            target = copy.parent / 'model.py'
            source.rename(target)
            source.symlink_to(target)
        else:
            name = provider.__name__ + '.foreign'
            sys.modules[name] = types.ModuleType(name)
        try:
            provider.source_identity()
        except ValueError as exc:
            assert any(term in str(exc) for term in ('identity', 'cache'))
        else:
            raise AssertionError('foreign inventory/cache admitted')
        assert 'torch' not in sys.modules
    ''', tmp_path, mode)


def _ast_digest(node):
    for child in ast.walk(node):
        if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and 'type_params' not in child._fields:
            child._fields = (*child._fields, 'type_params')
            child.type_params = []
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def test_whole_original_model_numerical_definitions_and_both_licenses_are_pinned():
    from distillschedule.reference import livewan_wan21 as provider
    manifest = json.loads((PACKAGE / "source_manifest.json").read_text())
    rows = {r["path"]: r for r in manifest["files"]}
    model = rows["wan/modules/model.py"]
    assert model["kind"] == "unchanged_whole_module"
    assert model["origin"]["sha256"] == model["sha256"] == "c1572ade3bf7345bd4c00d4a19535788fab62f9c08b5aeb15f6d01f4435bb47a"
    for row in rows.values():
        definitions = {n.name: n for n in ast.parse((PACKAGE / row["path"]).read_text()).body
                       if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))}
        for symbol in row.get("symbols", []):
            assert _ast_digest(definitions[symbol["symbol"]]) == symbol["ast_sha256"]
        if row["kind"] == "unchanged_whole_module":
            assert row["sha256"] == row["origin"]["sha256"]
    assert rows["wanstreamer/blockcausal.py"]["kind"] == "static_namespace_relocation"
    assert rows["wan/modules/sdpa_original.py"]["kind"] == "selected_definition_derivative"
    assert {x["symbol"] for x in rows["wan/modules/sdpa_original.py"]["symbols"]} == {"_needs_mask", "sdpa_attention"}
    for name in ("Wan2.1-LICENSE.txt", "LiveWan-LICENSE.txt"):
        content = (PACKAGE / "licenses" / name).read_bytes()
        assert hashlib.sha256(content).hexdigest() == "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
        assert content == (ROOT / "licenses/livewan-wan21-reference" / name).read_bytes()
    assert not any("/home/" in json.dumps(row) or "/share/" in json.dumps(row) for row in rows.values())
    assert provider.source_identity()["files_checked"] == len(rows)


def test_actual_import_constructs_nothing_preserves_rng_and_native_namespaces():
    fresh('''
        import sys, types, torch
        from distillschedule.reference import livewan_wan21 as provider
        sentinels = {name: types.ModuleType(name) for name in ('wan', 'wan.modules', 'wanstreamer')}
        sys.modules.update(sentinels)
        before = torch.get_rng_state().clone()
        api = provider.load_cpu(backend='torch-sdpa-cpu', precision='float32')
        assert api is provider.load_cpu(backend='torch-sdpa-cpu', precision='float32')
        assert api.model.WanModel.__module__ == provider.__name__ + '.wan.modules.model'
        assert api.stream.FewStepStreamer.__module__ == provider.__name__ + '.wanstreamer.stream'
        assert api.model.flash_attention is api.attention.flash_attention is api.blockcausal.flash_attention
        assert api.dmd.bc is api.blockcausal and api.stream.bc is api.blockcausal
        assert all(sys.modules[name] is module for name, module in sentinels.items())
        assert not any(n == 'flash_attn' or n.startswith('flash_attn.') or n == 'flash_attn_interface' for n in sys.modules)
        assert torch.equal(before, torch.get_rng_state()) and not torch.cuda.is_initialized()
        assert not any(type(x).__module__ == provider.__name__ for x in sys.meta_path)
    ''')


def test_failed_dependency_import_cleans_private_cache_and_retries_without_stale_hooks():
    fresh('''
        import sys
        from distillschedule.reference import livewan_wan21 as provider
        # Import-failure sentinel only, never a replacement numerical model.
        sys.modules['diffusers.configuration_utils'] = None
        try:
            provider.load_cpu(backend='torch-sdpa-cpu', precision='float32')
        except ModuleNotFoundError:
            pass
        else:
            raise AssertionError('missing dependency was ignored')
        assert not provider._cached_modules()
        assert provider._loaded_modules is None
        assert not any(type(x).__module__ == provider.__name__ for x in sys.meta_path)
        del sys.modules['diffusers.configuration_utils']
        api = provider.load_cpu(backend='torch-sdpa-cpu', precision='float32')
        assert api.model.WanModel.__module__.startswith(provider.__name__)
        assert not sys.modules['torch'].cuda.is_initialized()
    ''')


def test_live_module_api_parent_code_method_dispatch_and_backend_poison_is_rejected():
    fresh('''
        import sys, types
        from distillschedule.reference import livewan_wan21 as provider
        api = provider.load_cpu(backend='torch-sdpa-cpu', precision='float32')
        def rejected():
            try:
                provider.load_cpu(backend='torch-sdpa-cpu', precision='float32')
            except ValueError as exc:
                assert 'binding' in str(exc)
            else:
                raise AssertionError('poisoned numerical namespace reused')
        mutations = [(api, 'model', None), (api.dmd, 'dmd_surrogate', lambda: None),
                     (api.model.WanModel, 'forward', lambda: None),
                     (api.core, 'flash_attention', lambda: None),
                     (api.model.__spec__, 'origin', 'foreign.py'),
                     (sys.modules[provider.__name__ + '.wan.modules'], 'model', None),
                     (api.attention.F, 'scaled_dot_product_attention', lambda: None),
                     (api.attention.sdpa_original, 'sdpa_backend_ctx', lambda: None)]
        for owner, name, replacement in mutations:
            old = getattr(owner, name)
            setattr(owner, name, replacement)
            rejected()
            setattr(owner, name, old)
            assert provider.load_cpu(backend='torch-sdpa-cpu', precision='float32') is api
        old = api.dmd.dmd_surrogate.__code__
        api.dmd.dmd_surrogate.__code__ = (lambda x0_hat, grad, scale=1.: None).__code__
        rejected()
        api.dmd.dmd_surrogate.__code__ = old
        dispatch = api.model.WAN_CROSSATTENTION_CLASSES
        old = dispatch['t2v_cross_attn']
        dispatch['t2v_cross_attn'] = lambda: None
        rejected()
        dispatch['t2v_cross_attn'] = old
        name = provider.__name__ + '.foreign'
        sys.modules[name] = types.ModuleType(name)
        rejected()
        del sys.modules[name]
        assert provider.load_cpu(backend='torch-sdpa-cpu', precision='float32') is api
    ''')


@pytest.mark.parametrize("invalidation", ["timestamp", "unchecked-hash"])
def test_source_only_import_does_not_execute_poisoned_bytecode(tmp_path, invalidation):
    fresh(r'''
        from pathlib import Path
        from importlib._bootstrap_external import _code_to_timestamp_pyc, _code_to_hash_pyc
        import importlib.util, shutil, sys
        from distillschedule.reference import livewan_wan21 as provider
        copy = Path(sys.argv[1]) / 'copy'
        shutil.copytree(provider._ROOT, copy, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        provider._ROOT = copy
        provider.__path__[:] = [str(copy)]
        source = copy / 'wan/modules/model.py'
        data, stat = source.read_bytes(), source.stat()
        code = compile(data + b'\nUNADMITTED_BYTECODE = True\n', str(source), 'exec')
        payload = (_code_to_timestamp_pyc(code, int(stat.st_mtime), len(data))
                   if sys.argv[2] == 'timestamp' else
                   _code_to_hash_pyc(code, importlib.util.source_hash(data), checked=False))
        cache = Path(importlib.util.cache_from_source(str(source)))
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(payload)
        api = provider.load_cpu(backend='torch-sdpa-cpu', precision='float32')
        assert not hasattr(api.model, 'UNADMITTED_BYTECODE')
        assert cache.read_bytes() == payload
        provider.source_identity()
    ''', tmp_path, invalidation)
