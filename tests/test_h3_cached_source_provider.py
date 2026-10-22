"""Offline source admission for a maintained cached-only DiffSynth derivative."""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src/distillschedule/reference/diffsynth_h3_cached"
BACKEND = "torch-dense-cpu"


def fresh(code, *args):
    result = subprocess.run(
        [sys.executable, "-B", "-c", textwrap.dedent(code), *map(str, args)],
        cwd=ROOT, capture_output=True, text=True,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
                 HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_lazy_identity_has_no_torch_or_native_diffsynth_imports():
    fresh('''
        import json, sys
        from distillschedule.reference import diffsynth_h3_cached as provider
        identity = provider.source_identity()
        assert identity['scope'] == 'MAINTAINED_CACHED_AV_DERIVATIVE_CPU_SOURCE_ONLY'
        assert identity['canonical_gan_start_step'] == 500
        assert identity['qualified_world_size'] == 1
        assert not any(x == 'torch' or x.startswith('torch.') for x in sys.modules)
        assert not any(x == 'diffsynth' or x.startswith('diffsynth.') for x in sys.modules)
        assert not any(x.endswith('.minimax_h3_dit') for x in sys.modules)
        print(json.dumps(identity, sort_keys=True))
    ''')


@pytest.mark.parametrize("backend", [None, "torch", "auto", "cuda", "flash_attention_2"])
def test_execution_needs_explicit_cpu_dense_backend_before_torch(backend):
    fresh('''
        import json, sys
        from distillschedule.reference import diffsynth_h3_cached as provider
        try:
            provider.load_cpu(backend=json.loads(sys.argv[1]))
        except ValueError as exc:
            assert 'explicit' in str(exc)
        else:
            raise AssertionError('unsupported backend was accepted')
        assert 'torch' not in sys.modules
    ''', json.dumps(backend))


@pytest.mark.parametrize("key,value", [
    ("DIFFSYNTH_ATTENTION_IMPLEMENTATION", "sage_attention"),
    ("DIFFSYNTH_FLASH_ATTN_KERNEL_REPO_ID", "not-a-real-repository"),
    ("DIFFSYNTH_FLASH_ATTN_KERNEL_VERSION", "1"),
])
def test_backend_environment_rejected_before_network_capable_attention(key, value):
    fresh('''
        import os, sys
        from distillschedule.reference import diffsynth_h3_cached as provider
        os.environ[sys.argv[1]] = sys.argv[2]
        try:
            provider.load_cpu(backend='torch-dense-cpu')
        except ValueError as exc:
            assert 'attention' in str(exc).lower()
        else:
            raise AssertionError('unsafe attention setting was accepted')
        assert 'torch' not in sys.modules and 'kernels' not in sys.modules
    ''', key, value)


def test_direct_numerical_import_requires_provider_admission():
    fresh('''
        import importlib, sys
        try:
            importlib.import_module('distillschedule.reference.diffsynth_h3_cached.models.minimax_h3_dit')
        except ValueError as exc:
            assert 'load_cpu' in str(exc)
        else:
            raise AssertionError('numerical import bypassed backend/source admission')
        assert 'torch' not in sys.modules
    ''')


def test_installed_optimized_attention_is_rejected_without_importing_it(tmp_path):
    fresh('''
        from pathlib import Path
        import sys
        from distillschedule.reference import diffsynth_h3_cached as provider
        root = Path(sys.argv[1])
        module = root / 'flash_attn'
        module.mkdir()
        marker = root / 'attention-imported'
        (module / '__init__.py').write_text('from pathlib import Path\\nPath(' + repr(str(marker)) + ').touch()\\n')
        sys.path.insert(0, str(root))
        try:
            provider.load_cpu(backend='torch-dense-cpu')
        except ValueError as exc:
            assert 'optimized attention installation' in str(exc)
        else:
            raise AssertionError('optimized attention probe executed')
        assert not marker.exists() and 'torch' not in sys.modules and 'flash_attn' not in sys.modules
    ''', tmp_path)


@pytest.mark.parametrize("target", ["models/minimax_h3_dit.py", "source_manifest.json"])
def test_changed_source_rejected_before_torch_or_model(tmp_path, target):
    fresh('''
        from pathlib import Path
        import shutil, sys
        from distillschedule.reference import diffsynth_h3_cached as provider
        copy = Path(sys.argv[1]) / 'copy'
        shutil.copytree(provider._ROOT, copy, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        provider._ROOT = copy
        file = copy / sys.argv[2]
        file.write_bytes(file.read_bytes() + b'\\n# changed source\\n')
        try:
            provider.load_cpu(backend='torch-dense-cpu')
        except ValueError as exc:
            assert 'identity' in str(exc).lower()
        else:
            raise AssertionError('modified source accepted')
        assert 'torch' not in sys.modules
    ''', tmp_path, target)


def test_manifest_pins_original_numerical_bytes_and_every_selected_ast():
    from distillschedule.reference import diffsynth_h3_cached as provider
    identity = provider.source_identity()
    manifest = json.loads((PACKAGE / "source_manifest.json").read_text())
    originals = {
        "models/minimax_h3_dit.py": "bd617a4312c70405e916707e1908100b484b2259dfbeba4af7de5bb145206d80",
        "core/attention/attention.py": "1126b2e19184299cbfc611dcf66e46d0fd1a7bfa1d616f2ed38bc60ba19679ae",
        "core/gradient/gradient_checkpoint.py": "00f33981e4ac155d1ae757de2765703cd4881775b9cf8360627c709d631c2b21",
        "diffusion/flow_match.py": "b4c162413008db4a3986ad8b3638166d7c9b5a4820580e4524e3891ec5c710ba",
        "dmd2_zero3.py": "c8b7beb4375c26daae9e8e0dd863fb8a9079e644c17e690d712d45f14c3bab7b",
    }
    rows = {r["path"]: r for r in manifest["files"]}
    for path, expected in originals.items():
        assert rows[path]["kind"] == "unchanged_whole_module"
        assert rows[path]["sha256"] == expected == hashlib.sha256((PACKAGE / path).read_bytes()).hexdigest()
    for row in manifest["files"]:
        if not row.get("symbols"):
            continue
        tree = ast.parse((PACKAGE / row["path"]).read_text())
        definitions = {}
        def collect(nodes, prefix=""):
            for node in nodes:
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    name = prefix + node.name
                    definitions[name] = node
                    if isinstance(node, ast.ClassDef):
                        collect(node.body, name + ".")
        collect(tree.body)
        for symbol in row["symbols"]:
            node = definitions[symbol.get("output_symbol", symbol["symbol"])]
            # Sole allowed in-body change: the original optional ZeRO3 import is
            # explicitly relative to this maintained namespace, not sys.path.
            if symbol["symbol"] == "save_checkpoint":
                for child in ast.walk(node):
                    if isinstance(child, ast.ImportFrom) and child.module == "dmd2_zero3":
                        assert child.level == 1
                        child.level = 0
            digest = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
            assert digest == symbol["ast_sha256"], (row["path"], symbol["symbol"])
    assert rows["h3_dmd2.py"]["symbols"][-1]["ast_sha256"] == "c88b4eac3a03b3570b2dc55b4c0a8b669f2024dd65ec06369300df1ba085f7a8"
    assert identity["files_checked"] == len(rows)
    assert not any("/home/" in json.dumps(row) or "/share/" in json.dumps(row) for row in manifest["files"])


def test_real_imports_are_normal_namespaced_source_and_native_cache_is_untouched():
    fresh('''
        import os, sys
        from distillschedule.reference import diffsynth_h3_cached as provider
        native = {n: m for n, m in sys.modules.items() if n == 'diffsynth' or n.startswith('diffsynth.')}
        before = os.environ.get('DIFFSYNTH_ATTENTION_IMPLEMENTATION')
        api = provider.load_cpu(backend='torch-dense-cpu')
        assert api is provider.load_cpu(backend='torch-dense-cpu')
        assert api.model.MiniMaxH3DiT.__module__.endswith('.models.minimax_h3_dit')
        assert api.dmd2.DMD2Trainer.__module__.endswith('.h3_dmd2')
        assert api.attention.ATTENTION_IMPLEMENTATION == 'torch'
        assert api.dmd2.model_fn_minimax_h3 is api.packing.model_fn_minimax_h3
        assert api.training.prepare_batch is api.dmd2.prepare_batch
        assert api.training.CachedTensorMover is api.support.CachedTensorMover
        assert not hasattr(api.support, 'DiffusionTrainingModule')
        assert not hasattr(api.packing, 'MiniMaxH3Pipeline')
        assert os.environ.get('DIFFSYNTH_ATTENTION_IMPLEMENTATION') == before
        assert native == {n: m for n, m in sys.modules.items() if n == 'diffsynth' or n.startswith('diffsynth.')}
        assert not sys.modules['torch'].cuda.is_initialized()
    ''')


def test_mutated_live_binding_is_rejected_on_reuse():
    fresh('''
        from distillschedule.reference import diffsynth_h3_cached as provider
        api = provider.load_cpu(backend='torch-dense-cpu')
        original = api.dmd2.model_fn_minimax_h3
        api.dmd2.model_fn_minimax_h3 = lambda *args, **kwargs: None
        try:
            provider.load_cpu(backend='torch-dense-cpu')
        except ValueError as exc:
            assert 'binding' in str(exc).lower()
        else:
            raise AssertionError('dirty numerical binding accepted')
        api.dmd2.model_fn_minimax_h3 = original
        assert provider.load_cpu(backend='torch-dense-cpu') is api
    ''')


@pytest.mark.parametrize("invalidation", ["timestamp", "unchecked-hash"])
def test_admitted_import_never_executes_different_bytecode(tmp_path, invalidation):
    fresh(r'''
        from pathlib import Path
        from importlib._bootstrap_external import _code_to_timestamp_pyc, _code_to_hash_pyc
        import importlib.util, shutil, sys
        from distillschedule.reference import diffsynth_h3_cached as provider
        copy = Path(sys.argv[1]) / 'copy'
        shutil.copytree(provider._ROOT, copy, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        provider._ROOT = copy
        provider.__path__[:] = [str(copy)]
        source = copy / 'models/minimax_h3_dit.py'
        data, stat = source.read_bytes(), source.stat()
        code = compile(data + b'\nUNADMITTED_BYTECODE = True\n', str(source), 'exec')
        payload = (_code_to_timestamp_pyc(code, int(stat.st_mtime), len(data))
                   if sys.argv[2] == 'timestamp' else
                   _code_to_hash_pyc(code, importlib.util.source_hash(data), checked=False))
        cache = Path(importlib.util.cache_from_source(str(source)))
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(payload)
        api = provider.load_cpu(backend='torch-dense-cpu')
        assert not hasattr(api.model, 'UNADMITTED_BYTECODE')
        assert cache.read_bytes() == payload
        assert not any(type(x).__module__ == provider.__name__ for x in sys.meta_path)
    ''', tmp_path, invalidation)
