"""Separate Diffusers 0.37 full-model CPU proof, not native W3 qualification.

Deliberately outside test_training_*.py's original Diffusers 0.34 helper group.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("target", ["transformer", "backend"])
@pytest.mark.parametrize("invalidation", ["timestamp", "unchecked-hash"])
def test_reference_import_uses_verified_source_not_stale_bytecode(tmp_path, target, invalidation):
    _fresh(r'''
        from pathlib import Path
        import hashlib
        import importlib.abc
        import importlib.util
        from importlib._bootstrap_external import _code_to_timestamp_pyc, _code_to_hash_pyc
        import shutil
        import sys
        from distillschedule.adapters import helios
        from distillschedule.adapters.helios import reference_model as provider
        from distillschedule.workloads.best_known import registry

        original = registry.repository_root()
        root, target, invalidation = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
        shutil.copytree(original / 'third_party/helios_compat', root / 'third_party/helios_compat',
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        private = root / 'adapter/_reference'
        shutil.copytree(provider._REFERENCE_PATH, private,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        # This process-local relocation avoids seeding any canonical cache.
        helios.__path__.insert(0, str(private.parent))
        registry.repository_root = provider.repository_root = lambda: root
        provider._REFERENCE_PATH = private
        provider._BACKEND_PATH = private / 'modules/helios_kernels.py'
        source = (root / 'third_party/helios_compat/source/helios/modules/transformer_helios.py'
                  if target == 'transformer' else provider._BACKEND_PATH)
        content, stats = source.read_bytes(), source.stat()
        code = compile(content + b'\nDMD_STALE_BYTECODE_SENTINEL = True\n', str(source), 'exec')
        payload = (_code_to_timestamp_pyc(code, int(stats.st_mtime), len(content))
                   if invalidation == 'timestamp' else
                   _code_to_hash_pyc(code, importlib.util.source_hash(content), checked=False))
        cache = Path(importlib.util.cache_from_source(str(source)))
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(payload)
        class UnrelatedFinder(importlib.abc.MetaPathFinder):
            def find_spec(self, *args, **kwargs):
                return None
        unrelated = UnrelatedFinder()
        sys.meta_path.insert(0, unrelated)
        before = provider.reference_model_identity(backend='torch-sdpa')
        cls = provider.load_reference_model_class(backend='torch-sdpa')
        module_name = cls.__module__ if target == 'transformer' else before['backend']['module']
        assert not getattr(sys.modules[module_name], 'DMD_STALE_BYTECODE_SENTINEL', False), \
            'source hashes passed but a different compiled payload executed'
        assert before['source']['files_checked'] == 83 and content == source.read_bytes()
        assert cache.read_bytes() == payload, 'provider must neither use nor rewrite caller bytecode'
        assert sys.meta_path.count(unrelated) == 1
        assert not any(type(finder).__module__ == provider.__name__ for finder in sys.meta_path)
        assert provider.reference_model_identity(backend='torch-sdpa') == before
        assert not sys.modules['torch'].cuda.is_initialized()
    ''', tmp_path, target, invalidation)


@pytest.mark.parametrize("failure", ["source", "namespace"])
def test_post_import_validation_failure_cleans_private_namespace_and_can_retry(failure):
    _fresh(r'''
        import sys
        from distillschedule.adapters import helios
        from distillschedule.adapters.helios import reference_model as provider
        native = {name: module for name, module in sys.modules.items()
                  if name == 'helios' or name.startswith('helios.')}
        original_source, original_validate = provider._source_identity, provider._validate_modules
        if sys.argv[1] == 'source':
            calls = 0
            def reject_source():
                global calls
                calls += 1
                source = original_source()
                if calls == 2:
                    source['content_tree_sha256'] = '0' * 64
                return source
            provider._source_identity = reject_source
        else:
            def reject_namespace(*args):
                original_validate(*args)
                raise ValueError('explicit post-import namespace rejection')
            provider._validate_modules = reject_namespace
        try:
            provider.load_reference_model_class(backend='torch-sdpa')
        except ValueError:
            pass
        else:
            raise AssertionError('post-import validation was not reached')
        assert not provider._cached_modules() and provider._loaded_modules is None
        assert provider._loaded_objects is None and not hasattr(helios, '_reference')
        assert not any(type(finder).__module__ == provider.__name__ for finder in sys.meta_path)
        provider._source_identity, provider._validate_modules = original_source, original_validate
        cls = provider.load_reference_model_class(backend='torch-sdpa')
        assert cls is provider.load_reference_model_class(backend='torch-sdpa')
        assert native == {name: module for name, module in sys.modules.items()
                          if name == 'helios' or name.startswith('helios.')}
        assert not sys.modules['torch'].cuda.is_initialized()
    ''', failure)


def _fresh(code, *arguments):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    result = subprocess.run([sys.executable, "-c", textwrap.dedent(code), *map(str, arguments)],
                            cwd=ROOT, env=env, text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_reference_model_source_discovery_needs_no_torch_or_model_import():
    _fresh('''
        import importlib.abc
        import sys
        class NoTorch(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, *args):
                if fullname == "torch" or fullname.startswith("torch."):
                    raise AssertionError("source discovery tried to import Torch")
        sys.meta_path.insert(0, NoTorch())
        from distillschedule.adapters.helios.reference_model import reference_model_identity
        identity = reference_model_identity(backend="torch-sdpa")
        assert identity["source"]["transformer_sha256"] == "5d3ffe37cdea3c04a156106f6246c82f91010a1eaf0dd61529cd74083065a886"
        assert identity["source"]["files_checked"] == 83
        assert identity["backend"]["name"] == "torch-sdpa"
        assert identity["backend"]["varlen"] is False
        assert len(identity["backend"]["sha256"]) == 64
        assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
        assert not any(name.startswith(identity["namespace"]) for name in sys.modules)
    ''')


def test_backend_must_be_explicit_and_never_falls_back_from_optimized_request():
    _fresh('''
        import sys
        from distillschedule.adapters.helios.reference_model import (
            load_reference_model_class, reference_model_identity,
        )
        for api in (load_reference_model_class, reference_model_identity):
            try:
                api()
            except TypeError:
                pass
            else:
                raise AssertionError("backend was implicit")
            for backend in ("sdpa", "flash-attn", "native", None):
                try:
                    api(backend=backend)
                except ValueError as error:
                    assert "torch-sdpa" in str(error)
                else:
                    raise AssertionError("requested backend silently fell back")
        assert "torch" not in sys.modules
    ''')


@pytest.mark.parametrize("suffix", ["", ".modules", ".modules.helios_kernels", ".modules.transformer_helios"])
def test_unadmitted_private_import_cache_is_rejected_before_torch(suffix):
    _fresh('''
        import sys
        from types import ModuleType
        from distillschedule.adapters.helios.reference_model import (
            load_reference_model_class, reference_model_identity,
        )
        namespace = reference_model_identity(backend="torch-sdpa")["namespace"]
        name = namespace + sys.argv[1]
        foreign = ModuleType(name)
        foreign.__file__ = "/foreign/transformer_helios.py"
        sys.modules[name] = foreign
        for api in (reference_model_identity, load_reference_model_class):
            try:
                api(backend="torch-sdpa")
            except ValueError as error:
                assert "reference" in str(error) and "cache" in str(error)
            else:
                raise AssertionError("foreign cached module admitted")
        assert sys.modules[name] is foreign and "torch" not in sys.modules
    ''', suffix)


def _provider():
    from distillschedule.adapters.helios import reference_model
    return reference_model


def _tiny_model(*, text_dim=8, patch_size=(1, 2, 2)):
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(131)
    cls = _provider().load_reference_model_class(backend="torch-sdpa")
    model = cls(patch_size=patch_size, num_attention_heads=2, attention_head_dim=16,
                in_channels=16, out_channels=16, text_dim=text_dim, freq_dim=16,
                ffn_dim=64, num_layers=1, rope_dim=(4, 6, 6), is_use_gan=True,
                is_use_gan_final=True, is_use_gan_hooks=False, gan_cond_map_dim=32)
    # The original constructor zeroes proj_out. This deterministic nonzero
    # frozen-output test fixture is explicit, not supplied pretrained weights.
    with torch.no_grad():
        model.proj_out.weight.normal_(mean=0., std=.02)
    return model


def _inputs(batch=1):
    import torch
    return dict(hidden_states=torch.randn(batch, 16, 9, 16, 16),
                timestep=torch.full((batch,), 500.),
                encoder_hidden_states=torch.randn(batch, 3, 8), return_dict=False)


def _loader_execution_metadata(model):
    """Observe CPU layout/backend differences without normalizing either side."""
    import torch
    def layout(value):
        return dict(dtype=str(value.dtype), shape=list(value.shape), stride=list(value.stride()),
                    storage_offset=value.storage_offset(), alignment_mod64=value.data_ptr() % 64)
    return dict(
        python=sys.version, torch=str(torch.__version__),
        cpu_capability=torch.backends.cpu.get_cpu_capability(),
        num_threads=torch.get_num_threads(), num_interop_threads=torch.get_num_interop_threads(),
        default_dtype=str(torch.get_default_dtype()),
        matmul_precision=torch.get_float32_matmul_precision(),
        mkldnn_enabled=torch.backends.mkldnn.enabled,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        parameters={name: layout(value) for name, value in model.named_parameters()},
        buffers={name: dict(layout(value), sha256=hashlib.sha256(
            value.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())
            for name, value in model.named_buffers()},
    )


def _install_loader_gan_observer(model):
    """Observe the original call once; hooks retain references, never compute."""
    records, handles = [], []
    for name, layer in model.gan_final_head.head3d.named_children():
        def observe(module, args, output, name=name):
            records.append((name, module, args, output))
        handles.append(layer.register_forward_hook(observe))
    return records, handles


def _loader_tensor_summary(value):
    """Post-forward CPU evidence, not a replacement input or reference."""
    import torch
    assert isinstance(value, torch.Tensor) and value.device.type == 'cpu'
    flat = value.detach().contiguous().reshape(-1)
    result = dict(dtype=str(value.dtype), shape=list(value.shape), stride=list(value.stride()),
        storage_offset=value.storage_offset(), alignment_mod64=value.data_ptr() % 64,
        sha256=hashlib.sha256(flat.view(torch.uint8).numpy().tobytes()).hexdigest())
    if value.numel() <= 128:
        def finite_json(item):
            if isinstance(item, list):
                return [finite_json(child) for child in item]
            if isinstance(item, float) and not math.isfinite(item):
                return str(item)
            return item
        result['small_values'] = finite_json(value.detach().tolist())
    return result


def _loader_gan_trace(records):
    # All hashing/copies happen after the original forward and hook removal.
    return [dict(name=name, type=type(layer).__module__ + '.' + type(layer).__qualname__,
                 inputs=[_loader_tensor_summary(value) for value in args],
                 parameters={key: _loader_tensor_summary(value)
                             for key, value in layer.named_parameters(recurse=False)},
                 output=_loader_tensor_summary(output))
            for name, layer, args, output in records]


def _loader_gan_comparison(actual, reference):
    def values(value):
        return {key: value[key] for key in ('dtype', 'shape', 'sha256')}
    def layout(value):
        return {key: value[key] for key in ('stride', 'storage_offset', 'alignment_mod64')}
    def flat(value):
        return [child for row in value for child in flat(row)] if isinstance(value, list) else [value]
    census_equal = [(r['name'], r['type']) for r in actual] == [(r['name'], r['type']) for r in reference]
    result = dict(census_equal=census_equal, actual_census=[r['name'] for r in actual],
                  reference_census=[r['name'] for r in reference], layers=[], first_divergence=None)
    if not census_equal:
        result['first_divergence'] = dict(name=None, kind='census_difference')
    for current, expected in zip(actual, reference):
        inputs_equal = [values(v) for v in current['inputs']] == [values(v) for v in expected['inputs']]
        parameters_equal = ({k: values(v) for k, v in current['parameters'].items()} ==
                            {k: values(v) for k, v in expected['parameters'].items()})
        output_equal = values(current['output']) == values(expected['output'])
        row = dict(name=current['name'], type=current['type'], input_values_equal=inputs_equal,
                   parameter_values_equal=parameters_equal, output_values_equal=output_equal,
                   input_layouts_equal=([layout(v) for v in current['inputs']] ==
                                        [layout(v) for v in expected['inputs']]),
                   parameter_layout_differences=[k for k, v in current['parameters'].items()
                       if k not in expected['parameters'] or layout(v) != layout(expected['parameters'][k])],
                   output_layouts_equal=layout(current['output']) == layout(expected['output']))
        if 'small_values' in current['output'] and 'small_values' in expected['output']:
            a, b = flat(current['output']['small_values']), flat(expected['output']['small_values'])
            if len(a) == len(b):
                finite = [(x, y) for x, y in zip(a, b) if isinstance(x, (int, float)) and isinstance(y, (int, float))]
                row['output_difference'] = dict(unequal=sum(x != y for x, y in zip(a, b)),
                    max_abs=max((abs(x - y) for x, y in finite), default=None))
        result['layers'].append(row)
        if result['first_divergence'] is None and not (inputs_equal and parameters_equal and output_equal):
            kind = ('input_value_difference' if not inputs_equal else
                    'parameter_value_difference' if not parameters_equal else 'operator_output_difference')
            result['first_divergence'] = dict(row, kind=kind)
    return result


def _loader_gan_counterfactual(model, records, reference_trace, fixture, checkpoint):
    """Failure-only attribution; never replaces actual outputs or any assertion."""
    import random
    import torch
    rng, python_rng = torch.get_rng_state().clone(), random.getstate()
    numpy = sys.modules.get('numpy')
    numpy_rng = numpy.random.get_state() if numpy is not None else None
    def state():
        return [(kind, name, id(value), _loader_tensor_summary(value))
                for kind, pairs in [('parameter', model.named_parameters()), ('buffer', model.named_buffers())]
                for name, value in pairs]
    before = state()
    result = dict(diagnostic_only=True)
    try:
        # The source-qualified original custom loader is a diagnostic control,
        # not a fallback for the failing default-loader result.
        custom = type(model).from_pretrained(str(checkpoint), use_default_loader=False).eval()
        observed, handles = _install_loader_gan_observer(custom)
        try:
            with torch.no_grad():
                custom_outputs = custom(**fixture['inputs'], gan_mode=True)
        finally:
            for handle in handles:
                handle.remove()
        custom_trace = _loader_gan_trace(observed)
        result['custom_copy'] = dict(matches_reference=all(torch.equal(a, b)
            for a, b in zip(custom_outputs, fixture['outputs'], strict=True)),
            comparison=_loader_gan_comparison(custom_trace, reference_trace), trace=custom_trace)
        comparison = _loader_gan_comparison(_loader_gan_trace(records), reference_trace)
        first = comparison['first_divergence']
        if first is None or first['kind'] != 'operator_output_difference':
            result['first_layer'] = dict(not_run='no first operator mismatch with equal input/parameter values')
        else:
            index = next(i for i, record in enumerate(records) if record[0] == first['name'])
            name, layer, args, _ = records[index]
            parameters = dict(layer.named_parameters())
            buffers = dict(layer.named_buffers())
            # Only this first mismatching operator's parameters are copied.
            # No original Parameter, buffer, model binding or default is changed.
            owned = {key: value.detach().clone(memory_format=torch.preserve_format)
                     for key, value in parameters.items()}
            with torch.no_grad():
                replay = layer(*args)
                aligned = torch.func.functional_call(layer, (owned, buffers), args, strict=True)
            expected = reference_trace[index]['output']
            replay_summary, aligned_summary = _loader_tensor_summary(replay), _loader_tensor_summary(aligned)
            result['first_layer'] = dict(name=name, type=first['type'],
                unmodified_replay_matches_reference=replay_summary['sha256'] == expected['sha256'],
                owned_parameter_replay_matches_reference=aligned_summary['sha256'] == expected['sha256'],
                cloned_parameter_bytes=sum(v.numel() * v.element_size() for v in owned.values()),
                original_parameter_layouts={k: _loader_tensor_summary(v) for k, v in parameters.items()},
                owned_parameter_layouts={k: _loader_tensor_summary(v) for k, v in owned.items()},
                unmodified_replay=replay_summary, owned_parameter_replay=aligned_summary)
    except Exception as error:
        # A diagnostic failure cannot hide the original strict output failure.
        result['diagnostic_error'] = dict(type=type(error).__name__, message=str(error))
    finally:
        torch.set_rng_state(rng)
        random.setstate(python_rng)
        if numpy_rng is not None:
            numpy.random.set_state(numpy_rng)
    result['original_model_unchanged'] = state() == before
    return result


def _loader_output_failure(outputs, expected, model, reference_metadata, *,
                           reference_trace=None, actual_trace=None, counterfactual=None):
    """Keep strict failure, but preserve the failing process's actual evidence."""
    import torch
    differences = []
    for index, (actual, reference) in enumerate(zip(outputs, expected, strict=True)):
        if not torch.equal(actual, reference):
            max_abs = float((actual - reference).abs().max())
            differences.append(dict(index=index, shape=list(actual.shape), dtype=str(actual.dtype),
                unequal=int(torch.count_nonzero(actual != reference)),
                actual_nonfinite=int(torch.count_nonzero(~torch.isfinite(actual))),
                reference_nonfinite=int(torch.count_nonzero(~torch.isfinite(reference))),
                max_abs=max_abs if math.isfinite(max_abs) else None))
    details = dict(error='strict fresh-process output mismatch', differences=differences,
                   reference=reference_metadata, actual=_loader_execution_metadata(model))
    if reference_trace is not None and actual_trace is not None:
        details['gan'] = dict(reference=reference_trace, actual=actual_trace,
            comparison=_loader_gan_comparison(actual_trace, reference_trace), counterfactual=counterfactual)
    return json.dumps(details, sort_keys=True, allow_nan=False)


@pytest.mark.parametrize('nonfinite', [False, True])
def test_loader_failure_diagnostic_preserves_rng_tensors_and_strict_failure(nonfinite):
    """Synthetic tensor diagnostic, not an original-model qualification proof."""
    import torch
    model = torch.nn.Linear(2, 1)
    reference = torch.tensor([1., 2.])
    actual = reference.clone()
    actual[0] = float('nan') if nonfinite else torch.nextafter(actual[0], torch.tensor(float('inf')))
    actual_bits = actual.view(torch.int32).clone()
    metadata = _loader_execution_metadata(model)
    rng = torch.get_rng_state().clone()
    state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    with pytest.raises(AssertionError, match='strict fresh-process output mismatch') as error:
        assert torch.equal(actual, reference), _loader_output_failure(
            [actual], [reference], model, metadata)
    details = json.loads(str(error.value).split('\n', 1)[0])
    assert details['differences'] == [dict(index=0, shape=[2], dtype='torch.float32',
        unequal=1, actual_nonfinite=int(nonfinite), reference_nonfinite=0,
        max_abs=None if nonfinite else float(actual[0] - reference[0]))]
    assert details['actual'] == metadata == details['reference']
    assert torch.equal(torch.get_rng_state(), rng)
    assert torch.equal(reference, torch.tensor([1., 2.]))
    assert torch.equal(actual.view(torch.int32), actual_bits)
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in state.items())


def test_full_original_class_dense_flow_and_original_gan_head_backward():
    import torch
    native_before = {k: v for k, v in sys.modules.items() if k == "helios" or k.startswith("helios.")}
    model = _tiny_model()
    provider = _provider()
    identity = provider.reference_model_identity(backend="torch-sdpa")
    source = sys.modules[type(model).__module__]
    backend = sys.modules[identity["backend"]["module"]]
    assert type(model).__module__ == identity["model_module"]
    assert Path(source.__file__).resolve() == Path(identity["source"]["transformer_path"])
    assert Path(type(model).forward.__wrapped__.__code__.co_filename).resolve() == Path(source.__file__).resolve()
    assert source.attn_varlen_func is backend.attn_varlen_func
    assert source.create_navit_attention_masks is backend.create_navit_attention_masks
    assert type(model.gan_final_head).__module__ == type(model).__module__
    assert sum(p.numel() for p in model.parameters()) == 236577
    calls = []
    handles = [layer.register_forward_hook(lambda *args: calls.append(True))
               for layer in (model.blocks[0].attn1, model.blocks[0].attn2)]
    try:
        x = _inputs()
        flow = model(**x)[0]
        assert flow.shape == x["hidden_states"].shape and torch.isfinite(flow).all()
        flow.square().mean().backward()
        assert torch.count_nonzero(model.patch_embedding.weight.grad) > 0
        model.zero_grad(set_to_none=True)
        y, logits = model(**_inputs(3), gan_mode=True)
        (y.square().mean() + logits.square().mean()).backward()
        assert y.shape == (3, 16, 9, 16, 16) and logits.shape == (3, 1)
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in model.gan_final_head.parameters())
        assert torch.count_nonzero(model.gan_final_head.head3d[-1].weight.grad) > 0
        assert len(calls) == 4
    finally:
        for handle in handles:
            handle.remove()
    assert provider.load_reference_model_class(backend="torch-sdpa") is type(model)
    assert native_before == {k: v for k, v in sys.modules.items() if k == "helios" or k.startswith("helios.")}
    assert not torch.cuda.is_initialized()


def test_dense_backend_is_original_sdpa_math_and_rejects_varlen():
    import torch
    provider = _provider()
    provider.load_reference_model_class(backend="torch-sdpa")
    identity = provider.reference_model_identity(backend="torch-sdpa")
    backend = sys.modules[identity["backend"]["module"]]
    q, k, v = (torch.randn(2, length, 2, 16, requires_grad=True) for length in (5, 7, 7))
    actual = backend.attn_varlen_func(q, k, v)
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
    assert torch.equal(actual, expected)
    actual_grad = torch.autograd.grad(actual.square().mean(), (q, k, v), retain_graph=True)
    expected_grad = torch.autograd.grad(expected.square().mean(), (q, k, v))
    assert all(torch.equal(a, b) for a, b in zip(actual_grad, expected_grad))
    with pytest.raises(NotImplementedError, match="NaViT|varlen"):
        backend.attn_varlen_func(q, k, v, attention_mask=(None, None, 5, 7))
    with pytest.raises(NotImplementedError, match="NaViT|varlen"):
        backend.create_navit_attention_masks()


@pytest.mark.parametrize("change", ["transformer", "scheduler", "manifest", "unlisted"])
def test_repeated_source_admission_rejects_drift(tmp_path, monkeypatch, change):
    provider = _provider()
    provider.load_reference_model_class(backend="torch-sdpa")
    snapshot = tmp_path / "third_party/helios_compat"
    shutil.copytree(ROOT / "third_party/helios_compat", snapshot, ignore=shutil.ignore_patterns("__pycache__"))
    if change == "transformer":
        path = snapshot / "source/helios/modules/transformer_helios.py"
    elif change == "scheduler":
        path = snapshot / "source/helios/scheduler/scheduling_helios.py"
    elif change == "manifest":
        path = snapshot / "source_manifest.json"
    else:
        path = snapshot / "source/helios/modules/unlisted.py"
    path.write_bytes((path.read_bytes() if path.exists() else b"") + b"\n# drift\n")
    monkeypatch.setattr(provider, "repository_root", lambda: tmp_path)
    for api in (provider.reference_model_identity, provider.load_reference_model_class):
        with pytest.raises(ValueError, match="source|manifest"):
            api(backend="torch-sdpa")


def test_repeated_backend_byte_admission_rejects_drift(tmp_path, monkeypatch):
    provider = _provider()
    provider.load_reference_model_class(backend="torch-sdpa")
    altered = tmp_path / "helios_kernels.py"
    altered.write_bytes(provider._BACKEND_PATH.read_bytes() + b"\n# drift\n")
    monkeypatch.setattr(provider, "_BACKEND_PATH", altered)
    for api in (provider.reference_model_identity, provider.load_reference_model_class):
        with pytest.raises(ValueError, match="backend.*identity"):
            api(backend="torch-sdpa")


@pytest.mark.parametrize("change", ["file", "spec", "path", "module", "parent", "root_binding",
                                    "module_name", "spec_name", "removed_function", "binding", "function", "class"])
def test_repeated_cached_namespace_admission_rejects_conflicts(monkeypatch, change):
    provider = _provider()
    cls = provider.load_reference_model_class(backend="torch-sdpa")
    identity = provider.reference_model_identity(backend="torch-sdpa")
    model = sys.modules[identity["model_module"]]
    backend = sys.modules[identity["backend"]["module"]]
    package = sys.modules[identity["namespace"] + ".modules"]
    if change == "file":
        monkeypatch.setattr(model, "__file__", "/foreign/transformer_helios.py")
    elif change == "spec":
        monkeypatch.setattr(backend.__spec__, "origin", "/foreign/helios_kernels.py")
    elif change == "path":
        monkeypatch.setattr(package, "__path__", ["/foreign/modules"])
    elif change == "module":
        monkeypatch.setitem(sys.modules, identity["model_module"], SimpleNamespace(__file__=model.__file__))
    elif change == "parent":
        monkeypatch.setattr(package, "transformer_helios", object())
    elif change == "root_binding":
        monkeypatch.setattr(sys.modules["distillschedule.adapters.helios"], "_reference", object())
    elif change == "module_name":
        monkeypatch.setattr(backend, "__name__", "foreign.backend")
    elif change == "spec_name":
        monkeypatch.setattr(model.__spec__, "name", "foreign.transformer")
    elif change == "removed_function":
        monkeypatch.delattr(backend, "attn_varlen_func")
    elif change == "binding":
        monkeypatch.setattr(model, "attn_varlen_func", lambda *a, **k: None)
    elif change == "function":
        monkeypatch.setattr(backend, "attn_varlen_func", lambda *a, **k: None)
    else:
        monkeypatch.setattr(model, "HeliosTransformer3DModel", type("Foreign", (), {}))
    for api in (provider.reference_model_identity, provider.load_reference_model_class):
        with pytest.raises(ValueError, match="reference"):
            api(backend="torch-sdpa")
    assert cls.__module__ == identity["model_module"]


def test_import_failure_cleans_only_its_private_namespace_for_retry():
    _fresh('''
        import importlib
        import sys
        from distillschedule.adapters.helios import reference_model as provider
        namespace = provider.reference_model_identity(backend="torch-sdpa")["namespace"]
        original = importlib.import_module
        def interrupted(name, *args):
            if name == namespace + ".modules.transformer_helios":
                original(namespace + ".modules.helios_kernels")
                raise RuntimeError("intentional dependency-import interruption")
            return original(name, *args)
        importlib.import_module = interrupted
        try:
            provider.load_reference_model_class(backend="torch-sdpa")
        except RuntimeError as error:
            assert "intentional" in str(error)
        else:
            raise AssertionError("interrupted model import succeeded")
        finally:
            importlib.import_module = original
        assert not any(k == namespace or k.startswith(namespace + ".") for k in sys.modules)
        cls = provider.load_reference_model_class(backend="torch-sdpa")
        assert cls.__module__ == namespace + ".modules.transformer_helios"
    ''')


@pytest.mark.parametrize("default_loader", [False, True])
def test_normal_model_save_pretrained_and_fresh_process_from_pretrained(tmp_path, default_loader):
    import torch
    model = _tiny_model().eval()
    kwargs = _inputs(3)
    records, handles = _install_loader_gan_observer(model)
    try:
        with torch.no_grad():
            expected = model(**kwargs, gan_mode=True)
    finally:
        for handle in handles:
            handle.remove()
    reference_trace = _loader_gan_trace(records)
    model.save_pretrained(tmp_path / "model", safe_serialization=True)
    assert (tmp_path / "model/config.json").is_file()
    assert (tmp_path / "model/diffusion_pytorch_model.safetensors").is_file()
    torch.save({"inputs": kwargs, "outputs": expected, "state": model.state_dict(),
                "buffers": dict(model.named_buffers()), "gan_trace": reference_trace,
                "execution_metadata": _loader_execution_metadata(model)}, tmp_path / "expected.pt")
    _fresh('''
        from pathlib import Path
        import sys
        import torch
        from distillschedule.adapters.helios.reference_model import (
            load_reference_model_class, reference_model_identity,
        )
        torch.set_num_threads(1)
        root = Path(sys.argv[1])
        cls = load_reference_model_class(backend="torch-sdpa")
        model = cls.from_pretrained(str(root / "model"), use_default_loader=sys.argv[2] == "True").eval()
        fixture = torch.load(root / "expected.pt", map_location="cpu", weights_only=True)
        actual = model.state_dict()
        assert actual.keys() == fixture["state"].keys()
        assert all(torch.equal(v, actual[n]) for n, v in fixture["state"].items())
        # The observer itself only appends Python references: no tensor
        # summaries, clones or counterfactual forwards execute on this path.
        from test_helios_reference_model_cpu import _install_loader_gan_observer
        records, handles = _install_loader_gan_observer(model)
        try:
            with torch.no_grad():
                outputs = model(**fixture["inputs"], gan_mode=True)
        finally:
            for handle in handles:
                handle.remove()
        assert [row[0] for row in records] == [row['name'] for row in fixture['gan_trace']]
        def failure_details():
            # Numerical diagnostics run only after a genuine strict failure.
            # Neither custom-copy nor layer replay replaces the actual result.
            from test_helios_reference_model_cpu import (
                _loader_gan_trace, _loader_gan_counterfactual, _loader_output_failure,
            )
            return _loader_output_failure(outputs, fixture["outputs"], model, fixture["execution_metadata"],
                reference_trace=fixture['gan_trace'], actual_trace=_loader_gan_trace(records),
                counterfactual=_loader_gan_counterfactual(model, records, fixture['gan_trace'], fixture, root / 'model'))
        assert len(outputs) == len(fixture["outputs"])
        assert all(torch.equal(a, b) for a, b in zip(outputs, fixture["outputs"])), \
            failure_details()
        buffers = dict(model.named_buffers())
        assert buffers.keys() == fixture["buffers"].keys()
        assert all(torch.equal(v, buffers[n]) for n, v in fixture["buffers"].items())
        assert type(model).__module__ == reference_model_identity(backend="torch-sdpa")["model_module"]
        assert not torch.cuda.is_initialized()
    ''', tmp_path, default_loader)


def test_full_class_true_peft_original_critic_helper_adam_and_restart(tmp_path):
    import torch
    from peft import LoraConfig
    from distillschedule.runtime.fsdp2_static import FSDP2EngineAdapter
    from distillschedule.strategies.wan_capture import native_primitives
    from distillschedule.training.checkpoint import save_checkpoint, restore_checkpoint, pack_state, rng_state
    from test_training_helios_gan_state import original, _update, _state, _assert_moments
    from test_training_end_to_end import equal
    # Reuse source-qualified original trainer binding/reset and loss helpers.
    # The whole transformer and its GAN head come from the public provider,
    # never the tiny construction or extracted head in that older fixture.
    native_before = {k: v for k, v in sys.modules.items() if k == "helios" or k.startswith("helios.")}
    # This separate 0.37 group opts in before the older fixture's ordinary
    # imports, so subsequent public owners reuse source-admitted helpers.
    native_primitives(source_only=True)
    original = original.__wrapped__()
    native_after = {k: v for k, v in sys.modules.items() if k == "helios" or k.startswith("helios.")}
    assert all(native_after[k] is v for k, v in native_before.items())

    def owner(seed):
        model = _tiny_model(text_dim=4, patch_size=(1, 1, 1)).requires_grad_(False)
        torch.manual_seed(seed)
        model.add_adapter(LoraConfig(r=2, lora_alpha=2, lora_dropout=0., init_lora_weights="gaussian",
                                     target_modules=["to_q", "to_k", "to_v"]))
        for name, parameter in model.named_parameters():
            if name.startswith("gan_final_head."):
                parameter.requires_grad_(True)
        binding = {"args": SimpleNamespace(training_config=SimpleNamespace(is_use_gan=True)),
                   "real_score_model": model, "critic_trainable_modules": ["gan_final_head"]}
        exec(original.bind, binding)
        base, extra = binding["gan_base_critic_trainable_params"], binding["gan_extra_critic_trainable_params"]
        assert len(base) == 12 and all("lora_" in n for n in base)
        assert len(extra) == 26 and all(n.startswith("gan_final_head.") for n in extra)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        engine = FSDP2EngineAdapter(model, optimizer, torch.optim.lr_scheduler.StepLR(optimizer, 2, gamma=.7),
                                    physical_step=0, global_batch=1)
        engine._gradient_clip = 1e-5
        return SimpleNamespace(model=model, engine=engine, binding=binding, base=base, extra=extra,
                               frozen={n for n, p in model.named_parameters() if not p.requires_grad})

    control = owner(57)
    identity = {"scope": "FULL_ORIGINAL_CLASS_PEFT_TINY_CPU_NOT_PRETRAINED_OR_PUBLIC_W3"}
    state = _state(control, True)
    _update(original, control, True)
    # Actual Gaussian PEFT initializes B to zero: A's first gradient/moments
    # must therefore be zero. A receives a real nonzero update on the next step.
    named = dict(control.model.named_parameters())
    for name in control.base:
        slots = control.engine.optimizer.state[named[name]]
        assert bool(torch.count_nonzero(slots["exp_avg"])) is ("lora_B" in name)
    _update(original, control, True)
    _assert_moments(control, 2)
    checkpoint = save_checkpoint(tmp_path / "critic", state, identity=identity,
                                 cycle=2, cursor=2, versions={"critic": 2})
    expected_loss, expected_logs, _ = _update(original, control, True)
    expected, expected_rng = pack_state(state), rng_state()
    fresh = owner(999)
    fresh_state = _state(fresh, True)
    restore_checkpoint(checkpoint, fresh_state, identity=identity)
    loss, logs, _ = _update(original, fresh, True)
    _assert_moments(fresh, 3)
    assert equal(expected_loss, loss) and equal(expected_logs, logs)
    assert equal(expected, pack_state(fresh_state)) and equal(expected_rng, rng_state())
    assert native_after == {k: v for k, v in sys.modules.items() if k == "helios" or k.startswith("helios.")}
    assert not torch.cuda.is_initialized()


def test_loader_gan_trace_observes_actual_forward_without_mutation():
    """Actual original GAN trace; neither a hosted reproduction nor a fix."""
    import random
    import torch
    model = _tiny_model().eval()
    kwargs = _inputs(3)
    with torch.no_grad():
        expected = model(**kwargs, gan_mode=True)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    rng, python_rng = torch.get_rng_state().clone(), random.getstate()
    records, handles = _install_loader_gan_observer(model)
    try:
        with torch.no_grad():
            actual = model(**kwargs, gan_mode=True)
    finally:
        for handle in handles:
            handle.remove()
    trace = _loader_gan_trace(records)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))
    assert [row['name'] for row in trace] == [str(index) for index in range(21)]
    assert [row['type'] for row in trace] == [
        type(layer).__module__ + '.' + type(layer).__qualname__
        for layer in model.gan_final_head.head3d]
    assert trace[0]['inputs'][0]['shape'] == [3, 16, 9, 16, 16]
    assert 'small_values' not in trace[0]['output']
    assert trace[-1]['output']['small_values'] == actual[1].tolist()
    assert trace[-1]['parameters']['weight']['shape'] == [1, 32]
    assert all(not layer._forward_hooks for layer in model.gan_final_head.head3d)
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())
    assert all(torch.equal(value, dict(model.named_buffers())[name]) for name, value in buffers.items())
    assert torch.equal(rng, torch.get_rng_state()) and python_rng == random.getstate()
    assert _loader_gan_comparison(trace, trace)['first_divergence'] is None
    assert not torch.cuda.is_initialized()


def test_loader_gan_failure_trace_and_counterfactual_keep_real_strict_failure(tmp_path):
    """Synthetic one-ulp hook fault in the actual original class, CPU only."""
    import copy
    import random
    import torch
    model = _tiny_model().eval()
    kwargs = _inputs(3)
    reference_records, handles = _install_loader_gan_observer(model)
    try:
        with torch.no_grad():
            expected = model(**kwargs, gan_mode=True)
    finally:
        for handle in handles:
            handle.remove()
    reference_trace = _loader_gan_trace(reference_records)
    metadata = _loader_execution_metadata(model)
    model.save_pretrained(tmp_path / 'model', safe_serialization=True)
    fixture = dict(inputs=kwargs, outputs=expected, gan_trace=reference_trace)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    rng, python_rng = torch.get_rng_state().clone(), random.getstate()

    # Fault injection belongs only to this diagnostic canary. The strict
    # assertion below must continue to fail; no golden/tolerance is changed.
    def one_ulp_fault(module, args, output):
        changed = output.clone()
        changed[0, 0] = torch.nextafter(changed[0, 0], torch.full_like(changed[0, 0], float('inf')))
        return changed
    fault = model.gan_final_head.head3d[-1].register_forward_hook(one_ulp_fault)
    records, handles = _install_loader_gan_observer(model)
    try:
        with torch.no_grad():
            actual = model(**kwargs, gan_mode=True)
    finally:
        for handle in handles:
            handle.remove()
        fault.remove()
    actual_trace = _loader_gan_trace(records)
    comparison = _loader_gan_comparison(actual_trace, reference_trace)
    assert comparison['census_equal'] and comparison['first_divergence']['name'] == '20'
    assert comparison['first_divergence']['kind'] == 'operator_output_difference'
    assert comparison['first_divergence']['input_values_equal']
    assert comparison['first_divergence']['parameter_values_equal']
    assert comparison['first_divergence']['output_difference']['unequal'] == 1

    # Separately prove that summaries distinguish changed inputs/parameters
    # from the observed operator-output difference and from layout alone.
    wrong_input = copy.deepcopy(actual_trace)
    wrong_input[-1]['inputs'][0]['sha256'] = '0' * 64
    assert _loader_gan_comparison(wrong_input, reference_trace)['first_divergence']['kind'] == 'input_value_difference'
    wrong_parameter = copy.deepcopy(actual_trace)
    wrong_parameter[-1]['parameters']['weight']['sha256'] = '0' * 64
    assert _loader_gan_comparison(wrong_parameter, reference_trace)['first_divergence']['kind'] == 'parameter_value_difference'
    layout_only = copy.deepcopy(reference_trace)
    layout_only[-1]['parameters']['weight']['alignment_mod64'] = 20
    layout_result = _loader_gan_comparison(layout_only, reference_trace)
    assert layout_result['first_divergence'] is None and layout_result['layers'][-1]['parameter_layout_differences'] == ['weight']

    counterfactual = _loader_gan_counterfactual(model, records, reference_trace, fixture, tmp_path / 'model')
    assert counterfactual['diagnostic_only'] and counterfactual['custom_copy']['matches_reference']
    assert counterfactual['first_layer']['name'] == '20'
    assert counterfactual['first_layer']['unmodified_replay_matches_reference']
    assert counterfactual['first_layer']['owned_parameter_replay_matches_reference']
    assert counterfactual['first_layer']['cloned_parameter_bytes'] == 132
    assert counterfactual['original_model_unchanged']
    details = _loader_output_failure(actual, expected, model, metadata,
        reference_trace=reference_trace, actual_trace=actual_trace, counterfactual=counterfactual)
    with pytest.raises(AssertionError, match='strict fresh-process output mismatch'):
        assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True)), details
    decoded = json.loads(details)
    assert decoded['gan']['comparison'] == comparison
    assert decoded['gan']['counterfactual']['first_layer']['unmodified_replay_matches_reference']
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())
    assert all(torch.equal(value, dict(model.named_buffers())[name]) for name, value in buffers.items())
    assert torch.equal(rng, torch.get_rng_state()) and python_rng == random.getstate()
    assert not torch.cuda.is_initialized()


def _hosted_final_linear_case():
    """Exact tiny tensors captured by the failure-only hosted diagnostic."""
    import torch
    path = ROOT / 'tests/fixtures/helios_cpu_fp32_final_linear_hosted.json'
    record = json.loads(path.read_text())
    assert record['source_report_sha256'] == '8a351ee180d371d3abeb3b0292b30d0eaa864e7a0b46ef06a8114741b59a646b'
    tensors = {name: torch.tensor(record[name]['small_values'], dtype=torch.float32)
               for name in ('input', 'weight', 'bias', 'reference_output', 'unaligned_hosted_output')}
    for name, value in tensors.items():
        assert _loader_tensor_summary(value)['sha256'] == record[name]['sha256']
    return tensors


def _shift_cpu_storage(parameter, offset):
    """Preserve actual tensor values/Parameter identity, set a known alignment."""
    import torch
    owner = torch.empty(parameter.numel() + 16, dtype=parameter.dtype)
    view = owner[offset:offset + parameter.numel()].view(parameter.shape)
    with torch.no_grad():
        view.copy_(parameter)
        parameter.data = view
    assert parameter.data_ptr() % 64 == offset * parameter.element_size()


@pytest.mark.parametrize('default_loader', [False, True])
def test_real_hosted_final_linear_pretrained_owns_only_its_132_unaligned_bytes(tmp_path, monkeypatch, default_loader):
    import torch
    from diffusers.models.modeling_utils import ModelMixin
    case = _hosted_final_linear_case()
    source = _tiny_model().eval()
    with torch.no_grad():
        source.gan_final_head.head3d[-1].weight.copy_(case['weight'])
        source.gan_final_head.head3d[-1].bias.copy_(case['bias'])
    source.save_pretrained(tmp_path / 'model', safe_serialization=True)
    original = ModelMixin.from_pretrained.__func__
    produced = {}
    def snapshot(model):
        return {name: dict(parameter_id=id(value), pointer=value.data_ptr(), version=value._version,
            requires_grad=value.requires_grad, grad=value.grad, summary=_loader_tensor_summary(value))
            for name, value in model.named_parameters()}
    def observe(cls, *args, **kwargs):
        model = original(cls, *args, **kwargs)
        produced.update(snapshot(model))
        return model
    monkeypatch.setattr(ModelMixin, 'from_pretrained', classmethod(observe))
    loaded = type(source).from_pretrained(str(tmp_path / 'model'), use_default_loader=default_loader).eval()
    final = loaded.gan_final_head.head3d[-1]
    assert final.weight.data_ptr() % 64 == final.bias.data_ptr() % 64 == 0
    with torch.no_grad():
        # Match the local owned-storage reference, not a cross-CPU BLAS golden.
        expected = torch.nn.functional.linear(case['input'], case['weight'], case['bias'])
        assert torch.equal(final(case['input']), expected)
    if default_loader:
        current = snapshot(loaded)
        changed = {name for name in current if current[name]['pointer'] != produced[name]['pointer']}
        assert changed == {'gan_final_head.head3d.20.weight', 'gan_final_head.head3d.20.bias'}
        assert sum(dict(loaded.named_parameters())[name].numel() * 4 for name in changed) == 132
        for name, row in current.items():
            before = produced[name]
            assert all(row[key] == before[key] for key in ('parameter_id', 'version', 'requires_grad', 'grad'))
            assert row['summary']['sha256'] == before['summary']['sha256']
    else:
        assert not produced, 'custom loader must not use Diffusers default loading'
    assert not torch.cuda.is_initialized()


def test_real_hosted_final_linear_ownership_keeps_parameter_gradient_version_rng_and_state():
    import random
    import torch
    case = _hosted_final_linear_case()
    model = _tiny_model().eval()
    final = model.gan_final_head.head3d[-1]
    with torch.no_grad():
        final.weight.copy_(case['weight']); final.bias.copy_(case['bias'])
    _shift_cpu_storage(final.weight, 5); _shift_cpu_storage(final.bias, 4)
    final.weight.grad = torch.full_like(final.weight, .25)
    final.bias.grad = torch.full_like(final.bias, .5)
    optimizer = torch.optim.AdamW(final.parameters())
    for parameter in final.parameters():
        optimizer.state[parameter] = dict(step=torch.tensor(7.), exp_avg=torch.full_like(parameter, .1),
                                          exp_avg_sq=torch.full_like(parameter, .2))
    before = {name: (id(value), value.data_ptr(), value._version, value.requires_grad,
                    value.grad, value.detach().clone()) for name, value in model.named_parameters()}
    optimizer_before = {parameter: {key: (id(value), value.clone()) for key, value in state.items()}
                        for parameter, state in optimizer.state.items()}
    rng, python_rng = torch.get_rng_state().clone(), random.getstate()
    assert _provider()._own_cpu_pretrained_gan_linear(model, default_loader=True) is model
    assert final.weight.data_ptr() % 64 == final.bias.data_ptr() % 64 == 0
    for name, value in model.named_parameters():
        identity, pointer, version, requires_grad, grad, values = before[name]
        assert id(value) == identity and value._version == version and value.requires_grad == requires_grad
        assert value.grad is grad and torch.equal(value, values)
        if name not in ('gan_final_head.head3d.20.weight', 'gan_final_head.head3d.20.bias'):
            assert value.data_ptr() == pointer
    for parameter, state in optimizer.state.items():
        for key, value in state.items():
            assert id(value) == optimizer_before[parameter][key][0]
            assert torch.equal(value, optimizer_before[parameter][key][1])
    x = case['input'].requires_grad_()
    w, b = case['weight'].requires_grad_(), case['bias'].requires_grad_()
    expected = torch.nn.functional.linear(x, w, b)
    actual = final(x)
    assert torch.equal(actual, expected)
    grad = torch.autograd.grad(actual.square().sum(), (x, final.weight, final.bias))
    reference = torch.autograd.grad(expected.square().sum(), (x, w, b))
    assert all(torch.equal(a, b) for a, b in zip(grad, reference, strict=True))
    assert torch.equal(rng, torch.get_rng_state()) and python_rng == random.getstate()
    assert final.weight.grad is before['gan_final_head.head3d.20.weight'][4]
    assert final.bias.grad is before['gan_final_head.head3d.20.bias'][4]
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('damage', ['weight_stride', 'bias_stride', 'weight_shape'])
def test_cpu_final_linear_rejects_damaged_layout_before_any_parameter_change(damage):
    import torch
    model = _tiny_model().eval(); final = model.gan_final_head.head3d[-1]
    _shift_cpu_storage(final.weight, 5); _shift_cpu_storage(final.bias, 4)
    with torch.no_grad():
        if damage == 'weight_stride':
            final.weight.data = torch.zeros(64)[::2].view(1, 32)
        elif damage == 'bias_stride':
            final.bias.data = torch.zeros(2)[::2]
        else:
            final.weight.data = torch.zeros(1, 31)
    before = {name: (id(value), value.data_ptr(), value._version, value.detach().clone())
              for name, value in model.named_parameters()}
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match='final GAN Linear layout'):
        _provider()._own_cpu_pretrained_gan_linear(model, default_loader=True)
    for name, value in model.named_parameters():
        assert (id(value), value.data_ptr(), value._version) == before[name][:3]
        assert torch.equal(value, before[name][3])
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize('path', ['custom', 'nogan', 'bf16', 'mixed', 'meta', 'width64', 'aligned'])
def test_nonselected_pretrained_final_linear_storage_is_unchanged(path):
    import torch
    model = _tiny_model().eval()
    if path == 'nogan':
        cls = type(model)
        model = cls.from_config(model.config, is_use_gan=False, is_use_gan_final=False, is_use_gan_hooks=False)
        assert not hasattr(model, 'gan_final_head')
    elif path == 'bf16':
        model = model.to(dtype=torch.bfloat16)
    elif path == 'mixed':
        model.patch_embedding.weight.data = model.patch_embedding.weight.to(dtype=torch.bfloat16)
    elif path == 'meta':
        model = model.to(device='meta')
    elif path == 'width64':
        model = type(model).from_config(model.config, gan_cond_map_dim=64)
    if path == 'custom':
        _shift_cpu_storage(model.gan_final_head.head3d[-1].weight, 5)
        _shift_cpu_storage(model.gan_final_head.head3d[-1].bias, 4)
    before = {name: (id(value), value._version, value.device, value.dtype,
                    value.data_ptr() if value.device.type != 'meta' else None)
              for name, value in model.named_parameters()}
    assert _provider()._own_cpu_pretrained_gan_linear(model, default_loader=path != 'custom') is model
    assert before == {name: (id(value), value._version, value.device, value.dtype,
                            value.data_ptr() if value.device.type != 'meta' else None)
                      for name, value in model.named_parameters()}
    assert not torch.cuda.is_initialized()


def test_pretrained_cpu_storage_policy_is_explicit_and_live_bound(monkeypatch):
    provider = _provider()
    cls = provider.load_reference_model_class(backend='torch-sdpa')
    identity = provider.reference_model_identity(backend='torch-sdpa')
    policy = identity['pretrained_storage_policy']
    assert policy['name'] == 'cpu-fp32-final-gan-linear-owned64-v1'
    assert policy['scope'] == 'DEFAULT_LOADER_FINAL_GAN_LINEAR_32_TO_1_ONLY'
    assert policy['device'] == 'cpu' and policy['dtype'] == 'torch.float32'
    assert policy['alignment_bytes'] == 64 and policy['max_parameter_bytes'] == 132
    assert policy['module'] == provider.__name__ and 'path' not in policy
    assert Path(sys.modules[policy['module']].__spec__.origin).resolve() == Path(provider.__file__).resolve()
    assert policy['sha256'] == hashlib.sha256(Path(provider.__file__).read_bytes()).hexdigest()
    wrapper = cls.from_pretrained.__func__
    assert wrapper.__module__ == provider.__name__
    assert wrapper.__wrapped__.__module__ == identity['model_module']
    for change in ('loader', 'original', 'policy'):
        with monkeypatch.context() as scoped:
            if change == 'loader':
                scoped.setattr(cls, 'from_pretrained', classmethod(lambda *args, **kwargs: None))
            elif change == 'original':
                scoped.setattr(wrapper, '__wrapped__', lambda *args, **kwargs: None)
            else:
                scoped.setattr(provider, '_own_cpu_pretrained_gan_linear', lambda *args, **kwargs: None)
            for admission in (provider.reference_model_identity, provider.load_reference_model_class):
                with pytest.raises(ValueError, match='pretrained storage policy binding'):
                    admission(backend='torch-sdpa')
        assert provider.reference_model_identity(backend='torch-sdpa') == identity


@pytest.mark.parametrize('binding', ['native', 'policy', 'signature', 'code'])
def test_pretrained_wrapper_actual_closure_rejects_unadmitted_delegate_and_recovers(tmp_path, binding):
    import torch
    provider = _provider()
    source = _tiny_model().eval()
    source.save_pretrained(tmp_path / 'model', safe_serialization=True)
    cls = type(source)
    identity = provider.reference_model_identity(backend='torch-sdpa')
    wrapper = cls.from_pretrained.__func__
    code = wrapper.__code__
    cells = dict(zip(code.co_freevars, wrapper.__closure__, strict=True))
    original = cells[binding].cell_contents if binding != 'code' else code
    reported_original = wrapper.__wrapped__
    calls = []

    def unadmitted(*args, **kwargs):
        calls.append('SYNTHETIC_TEST_ONLY_UNADMITTED_DELEGATE')
        raise AssertionError('unadmitted pretrained delegate executed')

    class UnadmittedSignature:
        bind = staticmethod(unadmitted)

    rng = torch.get_rng_state().clone()
    try:
        if binding == 'code':
            wrapper.__code__ = code.replace(co_name='unadmitted_pretrained_wrapper')
        else:
            cells[binding].cell_contents = UnadmittedSignature() if binding == 'signature' else unadmitted
        # These are not legitimate source/model changes. The same wrapper and
        # reported original remain, but its actual callee is no longer bound.
        assert cls.from_pretrained.__func__ is wrapper and wrapper.__wrapped__ is reported_original
        assert hashlib.sha256(Path(provider.__file__).read_bytes()).hexdigest() == \
            identity['pretrained_storage_policy']['sha256']
        for admission in (provider.reference_model_identity, provider.load_reference_model_class):
            with pytest.raises(ValueError, match='pretrained storage policy binding'):
                admission(backend='torch-sdpa')
        assert calls == [] and torch.equal(rng, torch.get_rng_state())
    finally:
        if binding == 'code':
            wrapper.__code__ = original
        else:
            cells[binding].cell_contents = original
    assert provider.reference_model_identity(backend='torch-sdpa') == identity
    loaded = cls.from_pretrained(str(tmp_path / 'model'), use_default_loader=True).eval()
    assert all(torch.equal(value, loaded.state_dict()[name]) for name, value in source.state_dict().items())
    assert loaded.gan_final_head.head3d[-1].weight.data_ptr() % 64 == 0
    assert loaded.gan_final_head.head3d[-1].bias.data_ptr() % 64 == 0
    assert calls == [] and not torch.cuda.is_initialized()
