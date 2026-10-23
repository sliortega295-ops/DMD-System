"""Separate Diffusers 0.37 full-model CPU proof, not native W3 qualification.

Deliberately outside test_training_*.py's original Diffusers 0.34 helper group.
"""
from __future__ import annotations

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
    with torch.no_grad():
        expected = model(**kwargs, gan_mode=True)
    model.save_pretrained(tmp_path / "model", safe_serialization=True)
    assert (tmp_path / "model/config.json").is_file()
    assert (tmp_path / "model/diffusion_pytorch_model.safetensors").is_file()
    torch.save({"inputs": kwargs, "outputs": expected, "state": model.state_dict()}, tmp_path / "expected.pt")
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
        with torch.no_grad():
            outputs = model(**fixture["inputs"], gan_mode=True)
        assert all(torch.equal(a, b) for a, b in zip(outputs, fixture["outputs"]))
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
