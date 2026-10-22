"""Inference weights work without the original optimizer/data/run directory."""
import json
from pathlib import Path

import pytest
import torch

from distillschedule.data import prepare
from distillschedule.examples.tiny_av import build_fixture
from distillschedule.training.config import load_training_config
from distillschedule.training.session import export, train


def test_export_generation_uses_new_inputs_without_training_data(tmp_path):
    from distillschedule.training.session import generate_from_export
    old, new = tmp_path / "training-media", tmp_path / "new-inputs"
    for root, count in ((old, 4), (new, 2)):
        build_fixture(root, samples=count)
        prepare(root / "raw/manifest.json", root / "prepared")
    config = load_training_config(old / "train.json")
    completed = train(config)
    exported = tmp_path / "weights"
    export(config, completed["checkpoint"], exported)
    (old / "prepared").rename(old / "original-inputs-unavailable")
    (old / "run").rename(old / "original-run-unavailable")
    generated = generate_from_export(config, exported, tmp_path / "generated",
                                    prepared=new / "prepared/manifest.json", sample_index=1)
    assert generated["sample_id"] == "sample-000001"
    assert generated["input_identity"] != completed["identity"]["dataset"]
    assert generated["kind"] == "INFERENCE_FROM_EXPORTED_WEIGHTS"
    values = torch.load(generated["artifact"], weights_only=True)
    assert values["decoded"]["video"].dtype == torch.uint8
    import numpy as np
    from PIL import Image
    frames = sorted((tmp_path / "generated/frames").glob("*.png"))
    assert len(frames) == len(values["decoded"]["video"])
    for index, path in enumerate(frames):
        with Image.open(path) as image:
            assert torch.equal(torch.from_numpy(np.array(image)), values["decoded"]["video"][index])
    assert generated["quality_validation"] == "NOT_RUN"


def test_corrupt_export_is_rejected_before_loading_model_or_creating_output(tmp_path, monkeypatch):
    from distillschedule.training.session import generate_from_export
    root = tmp_path / "media"
    build_fixture(root, samples=4)
    prepare(root / "raw/manifest.json", root / "prepared")
    config = load_training_config(root / "train.json")
    trained = train(config)
    output = tmp_path / "export"
    export(config, trained["checkpoint"], output)
    with (output / "weights.pt").open("ab") as handle:
        handle.write(b"corruption")
    from distillschedule.training import session
    monkeypatch.setattr(session, "load_callable", lambda *a: pytest.fail("model imported before export admission"))
    with pytest.raises(ValueError, match="hash"):
        generate_from_export(config, output, tmp_path / "bad", prepared=root / "prepared/manifest.json")
    assert not (tmp_path / "bad").exists()


def test_cli_export_generation_is_not_resume(tmp_path):
    from test_training_end_to_end import run_cli
    root = tmp_path / "media"
    build_fixture(root, samples=4)
    prepare(root / "raw/manifest.json", root / "prepared")
    result = run_cli("train", root / "train.json")
    run_cli("export", root / "train.json", "--checkpoint", result["checkpoint"], "--output", root / "weights")
    result = run_cli("generate", root / "train.json", "--weights", root / "weights",
                     "--prepared", root / "prepared/manifest.json", "--output", root / "output")
    assert result["kind"] == "INFERENCE_FROM_EXPORTED_WEIGHTS"
    manifest = json.loads((root / "weights/export.json").read_text())
    assert manifest["kind"] == "INFERENCE_WEIGHTS_NOT_RESUMABLE_CHECKPOINT"
    assert manifest["files"]["weights.pt"]
    assert Path(result["artifact"]).is_file()


def _export_fixture(tmp_path):
    root = tmp_path / "media"
    build_fixture(root, samples=4)
    prepare(root / "raw/manifest.json", root / "prepared")
    config = load_training_config(root / "train.json")
    trained = train(config)
    output = tmp_path / "export"
    export(config, trained["checkpoint"], output)
    return root, config, output


def _rewrite_manifest(output, change):
    import hashlib
    path = output / "export.json"
    manifest = json.loads(path.read_text())
    change(manifest)
    path.write_text(json.dumps(manifest))
    (output / "_SUCCESS").write_text(hashlib.sha256(path.read_bytes()).hexdigest() + "\n")


def _write_pickle_sentinel(path):
    Path(path).write_text("forbidden pickle global executed\n")


def test_hash_valid_export_rejects_forbidden_pickle_global_without_execution(tmp_path):
    import hashlib
    import pickle

    from distillschedule.training.session import generate_from_export
    root, config, output = _export_fixture(tmp_path)
    sentinel = tmp_path / "pickle-side-effect"

    class ForbiddenWeights:
        def __reduce__(self):
            return _write_pickle_sentinel, (str(sentinel),)

    path = output / "weights.pt"
    torch.save(ForbiddenWeights(), path)
    _rewrite_manifest(output, lambda value: value["files"].update({"weights.pt": hashlib.sha256(path.read_bytes()).hexdigest()}))
    safe_globals = torch.serialization.get_safe_globals()
    assert _write_pickle_sentinel not in safe_globals
    assert not sentinel.exists()
    with pytest.raises(ValueError, match="invalid safe exported weights") as rejected:
        generate_from_export(config, output, tmp_path / "bad-pickle", prepared=root / "prepared/manifest.json")
    assert isinstance(rejected.value.__cause__, pickle.UnpicklingError)
    assert "Unsupported global" in str(rejected.value.__cause__)
    assert "_write_pickle_sentinel" in str(rejected.value.__cause__)
    assert torch.serialization.get_safe_globals() == safe_globals
    assert not sentinel.exists()
    assert not (tmp_path / "bad-pickle").exists()


@pytest.mark.parametrize("fault", ["incomplete", "manifest_hash", "schema", "escape", "symlink"])
def test_export_admission_rejects_before_model_import(tmp_path, monkeypatch, fault):
    from distillschedule.training.session import generate_from_export
    root, config, output = _export_fixture(tmp_path)
    if fault == "incomplete":
        (output / "_SUCCESS").unlink()
    elif fault == "manifest_hash":
        with (output / "export.json").open("a") as handle:
            handle.write(" ")
    elif fault == "schema":
        _rewrite_manifest(output, lambda value: value.update(schema_version="unsupported"))
    elif fault == "escape":
        _rewrite_manifest(output, lambda value: value["files"].update({"../outside.pt": "0" * 64}))
    else:
        (output / "weights.pt").rename(output / "outside.pt")
        (output / "weights.pt").symlink_to(output / "outside.pt")
    from distillschedule.training import session
    monkeypatch.setattr(session, "load_callable", lambda *a: pytest.fail("import before artifact admission"))
    with pytest.raises(ValueError):
        generate_from_export(config, output, tmp_path / "bad", prepared=root / "prepared/manifest.json")
    assert not (tmp_path / "bad").exists()


def test_export_binds_parameters_metadata_and_module_layout(tmp_path, monkeypatch):
    from distillschedule.training import session
    from distillschedule.training.config import TrainingConfig
    from distillschedule.training.session import generate_from_export
    root, config, output = _export_fixture(tmp_path)
    manifest = json.loads((output / "export.json").read_text())
    assert manifest["model"]["parameters"]["seed"] == config.trainer["seed"]
    assert "device" not in manifest["model"]["parameters"]
    assert manifest["model"]["metadata"] == manifest["identity"]["model_metadata"]
    raw = config.to_dict()
    raw["model"]["parameters"]["student_width"] += 1
    with pytest.raises(ValueError, match="parameters"):
        generate_from_export(TrainingConfig.from_dict(raw, source=config.source), output,
                             tmp_path / "bad-parameters", prepared=root / "prepared/manifest.json")
    original = session.load_callable
    for fault in ("metadata", "layout"):
        def load(reference, fault=fault):
            factory, source = original(reference)
            def build(parameters):
                components = factory(parameters)
                if fault == "metadata":
                    components.metadata = {**components.metadata, "pretrained": True}
                else:
                    components.state["student"] = torch.nn.Linear(1, 1)
                return components
            return build, source
        monkeypatch.setattr(session, "load_callable", load)
        with pytest.raises(ValueError, match=fault):
            generate_from_export(config, output, tmp_path / ("bad-" + fault),
                                 prepared=root / "prepared/manifest.json")
        assert not (tmp_path / ("bad-" + fault)).exists()


def test_changed_factory_source_rejects_before_factory_execution(tmp_path):
    from distillschedule.training.config import TrainingConfig
    from distillschedule.training.session import generate_from_export
    root = tmp_path / "media"
    build_fixture(root, samples=4)
    prepare(root / "raw/manifest.json", root / "prepared")
    source = tmp_path / "model.py"
    source.write_text("from distillschedule.examples.tiny_av import build_model\n")
    config = load_training_config(root / "train.json")
    raw = config.to_dict()
    raw["model"]["factory"] = str(source) + ":build_model"
    config = TrainingConfig.from_dict(raw, source=config.source)
    trained = train(config)
    output = tmp_path / "weights"
    export(config, trained["checkpoint"], output)
    source.write_text("def build_model(parameters):\n    raise AssertionError('changed factory executed')\n")
    with pytest.raises(ValueError, match="source"):
        generate_from_export(config, output, tmp_path / "bad", prepared=root / "prepared/manifest.json")
    assert not (tmp_path / "bad").exists()


def test_inference_calls_only_model_factory_and_uses_eval_decoder(tmp_path, monkeypatch):
    from distillschedule.training import session
    from distillschedule.training.session import generate_from_export
    root, config, output = _export_fixture(tmp_path)
    original = session.load_callable
    decoded = []
    def load(reference):
        assert reference == config.model["factory"], "inference must not load the algorithm"
        factory, source = original(reference)
        def build(parameters):
            components = factory(parameters)
            generate, decode = components.generate, components.decode
            def inference(state, feeds, parameters):
                assert not torch.is_grad_enabled()
                assert all(not value.training for value in state.values() if isinstance(value, torch.nn.Module))
                return generate(state, feeds, parameters)
            def render(value, parameters):
                assert not torch.is_grad_enabled()
                decoded.append(True)
                return decode(value, parameters)
            components.generate, components.decode = inference, render
            return components
        return build, source
    monkeypatch.setattr(session, "load_callable", load)
    result = generate_from_export(config, output, tmp_path / "output", prepared=root / "prepared/manifest.json")
    assert decoded == [True]
    assert result["video"]["layout"] == "THWC"
    assert result["generation_parameters"] == {}
    assert result["inference_seed"] == config.trainer["seed"]


@pytest.mark.parametrize("fault", ["role", "shape", "dtype"])
def test_hash_valid_but_incompatible_weight_layout_rejects(tmp_path, fault):
    import hashlib

    from distillschedule.training.session import generate_from_export
    root, config, output = _export_fixture(tmp_path)
    path = output / "weights.pt"
    weights = torch.load(path, weights_only=True)
    if fault == "role":
        weights.pop("teacher")
    else:
        key = next(iter(weights["student"]))
        weights["student"][key] = (weights["student"][key].double() if fault == "dtype" else torch.zeros(1))
    torch.save(weights, path)
    _rewrite_manifest(output, lambda value: value["files"].update({"weights.pt": hashlib.sha256(path.read_bytes()).hexdigest()}))
    with pytest.raises(ValueError, match="names|layout"):
        generate_from_export(config, output, tmp_path / "bad", prepared=root / "prepared/manifest.json")
    assert not (tmp_path / "bad").exists()


def test_missing_decoder_does_not_render_latents_as_video(tmp_path, monkeypatch):
    from distillschedule.training import session
    from distillschedule.training.session import evaluate, generate_from_export
    root, config, output = _export_fixture(tmp_path)
    original = session.load_callable
    def load(reference):
        factory, source = original(reference)
        if reference != config.model["factory"]:
            return factory, source
        def build(parameters):
            components = factory(parameters)
            components.decode = None
            return components
        return build, source
    monkeypatch.setattr(session, "load_callable", load)
    with pytest.raises(ValueError, match="actual decoder"):
        generate_from_export(config, output, tmp_path / "bad", prepared=root / "prepared/manifest.json")
    assert not (tmp_path / "bad").exists()
    legacy = evaluate(config, root / "run/checkpoints/cycle-000004", tmp_path / "legacy-latent")
    assert legacy["video"]["status"] == "NOT_AVAILABLE"
    assert not (tmp_path / "legacy-latent/frames").exists()
    assert Path(legacy["artifact"]).is_file()


def test_receipt_distinguishes_model_initialization_seed_from_inference_seed(tmp_path):
    from distillschedule.training.config import TrainingConfig
    from distillschedule.training.session import generate_from_export
    root, config, output = _export_fixture(tmp_path)
    raw = config.to_dict()
    raw["model"]["parameters"]["seed"] = config.trainer["seed"]
    raw["trainer"]["seed"] = 113
    result = generate_from_export(TrainingConfig.from_dict(raw, source=config.source), output, tmp_path / "generated",
                                 prepared=root / "prepared/manifest.json", parameters={"receipt_test": [1, 2]})
    assert result["model"]["parameters"]["seed"] == config.trainer["seed"]
    assert result["inference_seed"] == 113
    assert result["generation_parameters"] == {"receipt_test": [1, 2]}


def test_identical_local_factory_can_be_explicitly_relocated(tmp_path):
    from distillschedule.training.config import TrainingConfig
    from distillschedule.training.session import generate_from_export
    root = tmp_path / "media"
    build_fixture(root, samples=4)
    prepare(root / "raw/manifest.json", root / "prepared")
    source = tmp_path / "original-model.py"
    source.write_text('''
import torch
from distillschedule.examples.tiny_av import build_model as original
class Student(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
def build_model(parameters):
    components = original(parameters)
    components.state["student"] = Student(components.state["student"])
    return components
''')
    config = load_training_config(root / "train.json")
    raw = config.to_dict()
    raw["model"]["factory"] = str(source) + ":build_model"
    config = TrainingConfig.from_dict(raw, source=config.source)
    trained = train(config)
    output = tmp_path / "weights"
    export(config, trained["checkpoint"], output)
    relocated = tmp_path / "relocated-model.py"
    source.rename(relocated)
    raw["model"]["factory"] = str(relocated) + ":build_model"
    relocated_config = TrainingConfig.from_dict(raw, source=config.source)
    result = generate_from_export(relocated_config, output, tmp_path / "generated", prepared=root / "prepared/manifest.json")
    assert result["loaded_factory"]["reference"] == str(relocated) + ":build_model"
    raw["trainer"].update(resume_from=trained["checkpoint"], output_dir=str(tmp_path / "invalid-resume"))
    with pytest.raises(ValueError, match="identity"):
        train(TrainingConfig.from_dict(raw, source=config.source))
    assert not (tmp_path / "invalid-resume").exists()
    raw["model"]["factory"] = str(relocated) + ":wrong_symbol"
    with pytest.raises(ValueError, match="factory"):
        generate_from_export(TrainingConfig.from_dict(raw, source=config.source), output, tmp_path / "wrong-symbol",
                             prepared=root / "prepared/manifest.json")
    assert not (tmp_path / "wrong-symbol").exists()


def test_export_integrity_rechecked_before_output_publication(tmp_path, monkeypatch):
    from distillschedule.training import session
    from distillschedule.training.session import generate_from_export
    root, config, output = _export_fixture(tmp_path)
    original = session.load_callable
    def load(reference):
        factory, source = original(reference)
        def build(parameters):
            components = factory(parameters)
            generate = components.generate
            def corrupt_after_loading(state, feeds, parameters):
                value = generate(state, feeds, parameters)
                with (output / "weights.pt").open("ab") as handle:
                    handle.write(b"changed while generating")
                return value
            components.generate = corrupt_after_loading
            return components
        return build, source
    monkeypatch.setattr(session, "load_callable", load)
    with pytest.raises(ValueError, match="hash"):
        generate_from_export(config, output, tmp_path / "bad", prepared=root / "prepared/manifest.json")
    assert not (tmp_path / "bad").exists()


def test_custom_export_requires_and_uses_explicit_loader(tmp_path, monkeypatch):
    from test_training_end_to_end import equal

    from distillschedule.training import session
    from distillschedule.training.config import TrainingConfig
    from distillschedule.training.session import evaluate, generate_from_export
    root = tmp_path / "media"
    build_fixture(root, samples=4)
    prepare(root / "raw/manifest.json", root / "prepared")
    source = tmp_path / "custom_model.py"
    source.write_text('''
import torch
from distillschedule.examples.tiny_av import build_model as original
def save(state, root):
    torch.save({name: value.state_dict() for name, value in state.items()
                if isinstance(value, torch.nn.Module)}, root / "adapter.pt")
    return {"file": "adapter.pt", "contract": "explicit-test-loader-v1"}
def load(state, root, metadata):
    assert metadata == {"file": "adapter.pt", "contract": "explicit-test-loader-v1"}
    for name, weights in torch.load(root / metadata["file"], weights_only=True).items():
        state[name].load_state_dict(weights, strict=True)
def build_model(parameters):
    components = original(parameters)
    components.export, components.load_export = save, load
    return components
''')
    config = load_training_config(root / "train.json")
    raw = config.to_dict()
    raw["model"]["factory"] = str(source) + ":build_model"
    config = TrainingConfig.from_dict(raw, source=config.source)
    trained = train(config)
    output = tmp_path / "weights"
    export(config, trained["checkpoint"], output)
    generated = generate_from_export(config, output, tmp_path / "generated", prepared=root / "prepared/manifest.json")
    evaluated = evaluate(config, trained["checkpoint"], tmp_path / "evaluated")
    assert equal(torch.load(generated["artifact"], weights_only=True), torch.load(evaluated["artifact"], weights_only=True))
    original = session.load_callable
    def no_loader(reference):
        factory, identity = original(reference)
        if reference != config.model["factory"]:
            return factory, identity
        def build(parameters):
            components = factory(parameters)
            components.load_export = None
            return components
        return build, identity
    monkeypatch.setattr(session, "load_callable", no_loader)
    with pytest.raises(ValueError, match="load_export"):
        generate_from_export(config, output, tmp_path / "bad-load", prepared=root / "prepared/manifest.json")
    assert not (tmp_path / "bad-load").exists()
    with pytest.raises(ValueError, match="load_export"):
        export(config, trained["checkpoint"], tmp_path / "bad-export")
    assert not (tmp_path / "bad-export").exists()


def test_export_write_failure_cannot_publish_completion_marker(tmp_path, monkeypatch):
    root, config, output = _export_fixture(tmp_path)
    import shutil
    shutil.rmtree(output)
    def fail(*args, **kwargs):
        raise OSError("injected export write failure")
    monkeypatch.setattr(torch, "save", fail)
    checkpoint = root / "run/checkpoints/cycle-000004"
    with pytest.raises(OSError, match="write failure"):
        export(config, checkpoint, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".export.tmp-*"))


def test_projection_contract_is_explicit_and_finite():
    from distillschedule.training.adapters import InferenceProjection, ModelComponents
    original = ModelComponents(state={'training-only': object()})
    assert original.inference_projection is None
    projected = ModelComponents(state={'generator': torch.nn.Linear(2, 1)})
    assert InferenceProjection(projected, 'some.module:factory', {'seed': 3}).validate().components is projected
    with pytest.raises(ValueError, match='finite JSON'):
        InferenceProjection(projected, 'some.module:factory', {'bad': float('nan')}).validate()


def test_latent_output_requires_explicit_named_floating_schema():
    from distillschedule.training.inference import latent_outputs
    value = {'latents': torch.ones(1, 2, 3), 'latent_schema': {
        'latents': {'layout': 'BCT', 'shape': [1, 2, 3], 'dtype': 'torch.float32'}}}
    assert latent_outputs(value)['latents']['layout'] == 'BCT'
    for bad in ({}, {'latents': value['latents']},
                {**value, 'latents': torch.ones(1, 2, 3, dtype=torch.int64)},
                {**value, 'latents': torch.full((1, 2, 3), float('nan'))},
                {**value, 'latent_schema': {'latents': {'layout': '', 'shape': [1,2,3], 'dtype':'torch.float32'}}},
                {**value, 'latent_schema': {'latents': {'layout': 'BCT', 'shape': [True,2,3], 'dtype':'torch.float32'}}}):
        with pytest.raises(ValueError, match='latent'):
            latent_outputs(bad)


def test_invalid_optional_projection_refuses_before_publication(tmp_path, monkeypatch):
    from distillschedule.training import session
    from distillschedule.training.adapters import InferenceProjection
    root, config, _ = _export_fixture(tmp_path)
    original = session._load_for_output
    for fault in ('wrong-value', 'nonfinite-parameters'):
        def load(*args, **kwargs):
            result = original(*args, **kwargs)
            components = result[0]
            def project(current, parameters):
                assert current is components
                if fault == 'wrong-value':
                    return current
                return InferenceProjection(current, config.model['factory'], {'invalid': float('nan')})
            components.inference_projection = project
            return result
        monkeypatch.setattr(session, '_load_for_output', load)
        output = tmp_path / fault
        with pytest.raises(ValueError, match='InferenceProjection|finite JSON'):
            session.export(config, root / 'run/checkpoints/cycle-000004', output)
        assert not output.exists() and not list(tmp_path.glob('.' + fault + '.tmp-*'))
    with pytest.raises(ValueError, match='inference_projection must be callable'):
        from distillschedule.training.adapters import ModelComponents
        ModelComponents(state={'owner':object()}, inference_projection=True).validate()


def test_projection_uses_effective_session_seed_when_omitted(tmp_path, monkeypatch):
    from distillschedule.training import session
    from distillschedule.training.adapters import InferenceProjection
    root, config, _ = _export_fixture(tmp_path)
    original = session._load_for_output
    def load(*args, **kwargs):
        result = original(*args, **kwargs)
        components = result[0]
        def project(current, parameters):
            # As with ordinary factory parameters, an omitted seed uses the
            # session default. Explicit projection seeds remain authoritative.
            return InferenceProjection(current, config.model['factory'],
                                       {key:value for key,value in parameters.items() if key != 'seed'})
        components.inference_projection = project
        return result
    monkeypatch.setattr(session, '_load_for_output', load)
    projected = tmp_path / 'projected'
    manifest = session.export(config, root / 'run/checkpoints/cycle-000004', projected)
    assert manifest['model']['parameters']['seed'] == config.trainer['seed']
    generated = session.generate_from_export(config, projected, tmp_path / 'generated',
                                             prepared=root / 'prepared/manifest.json')
    assert generated['video']['layout'] == 'THWC'
