"""CPU_TEST_ONLY: restored projection providers and caller-owned Gloo lifetime.

Real tiny-media training/export evidence, not GPU, native-model or quality proof.
"""
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
import json
import os
from pathlib import Path
import time

import pytest
import torch
import torch.distributed as dist

from distillschedule.data import prepare
from distillschedule.examples.tiny_av import build_fixture
from distillschedule.training import inference, session
from distillschedule.training.adapters import InferenceProjection
from distillschedule.training.config import TrainingConfig
from distillschedule.workloads.best_known.registry import file_sha256
from test_training_construction_lifetime import (
    _assert_released, _HookProvider, _peer_finished, _probe_group,
)
from test_training_control_admission import _fixture
from test_training_policy_continuation import _run_workers


def _files(root):
    return {str(path.relative_to(root)): file_sha256(path)
            for path in Path(root).rglob("*") if path.is_file()}


@pytest.fixture
def trained(tmp_path):
    config, raw = _fixture(tmp_path / "media")
    raw["trainer"].pop("distributed")
    raw["data"].pop("sharding")
    raw["trainer"].update(cycles=1, checkpoint_every=1, output_dir=str(tmp_path / "training"))
    config = TrainingConfig.from_dict(raw, source=config.source)
    result = session.train(config)
    assert result["status"] == "COMPLETE" and result["completed_cycles"] == 1
    assert Path(result["checkpoint"], "_SUCCESS").is_file()
    return config, result


def _track(components, name, providers, events, *, close_error=None):
    provider = _HookProvider(components, close_error=close_error)
    providers[name] = provider

    def close():
        events.append(name)
        provider.close()

    components.close = close


def _install_projection(patch, config, providers, events, *, fault=None,
                        same_components=False, invalid_close=None, close_error=None):
    original = session._load_for_output

    def load(*args, **kwargs):
        result = original(*args, **kwargs)
        current = result[0]
        # Copy the actual restored state before adding instrumentation hooks;
        # otherwise deepcopy would also duplicate the training provider's hooks.
        projected_state = None if same_components else deepcopy(current.state)
        _track(current, "training", providers, events)

        def project(owner, parameters):
            assert owner is current and not events
            projected = (current if same_components else replace(
                current, state=projected_state, inference_projection=None, close=None))
            if not same_components:
                if invalid_close is not None:
                    # An invalid callback cannot own a usable cleanup hook.
                    projected.close = invalid_close
                else:
                    _track(projected, "projection", providers, events, close_error=close_error)
            factory = config.model["factory"]
            if fault == "validation":
                projected.prediction = "CPU_TEST_ONLY: invalid returned prediction"
            elif fault == "parameters":
                parameters = {"invalid": float("nan")}
            elif fault == "factory":
                factory = "./relative.py:build_model"
            elif fault == "source":
                factory = "distillschedule.examples.tiny_av:missing_projection_factory"
            return InferenceProjection(projected, factory, parameters)

        current.inference_projection = project
        return result

    patch.setattr(session, "_load_for_output", load)


def _observe_writer(patch, providers, events, writes, output):
    original = inference.publish_export

    def publish(components, *args, **kwargs):
        assert components is providers.get("projection", providers["training"]).components
        assert not events and all(provider.close_calls == 0 for provider in providers.values())
        result = original(components, *args, **kwargs)
        assert Path(output, "_SUCCESS").is_file()
        assert not events and all(provider.close_calls == 0 for provider in providers.values())
        writes.append(result["status"])
        return result

    patch.setattr(inference, "publish_export", publish)


def _assert_owners_released(providers, events, *, same_components=False):
    assert events == (["training"] if same_components else ["projection", "training"])
    assert set(providers) == ({"training"} if same_components else {"training", "projection"})
    for provider in providers.values():
        observed = provider.report()
        _assert_released(observed)
        assert observed["state_unchanged"] and observed["forwards"] == 0, observed
        assert not any(observed["updates"].values()), observed


_ERRORS = {
    "validation": "prediction must be an explicit PredictionSpec",
    "parameters": "inference projection parameters must be finite JSON without device",
    "factory": "inference projection needs an explicit resolved factory reference",
    "source": "adapter is not callable",
}


@pytest.mark.parametrize("fault", ["validation", "parameters", "factory", "source"])
def test_returned_projection_rejection_releases_both_real_providers(trained, tmp_path, monkeypatch, fault):
    config, result = trained
    providers, events = {}, []
    checkpoint_before = _files(Path(result["checkpoint"]))
    output = tmp_path / "never-output"
    _install_projection(monkeypatch, config, providers, events, fault=fault)
    with pytest.raises(ValueError, match=_ERRORS[fault]):
        session.export(config, result["checkpoint"], output)
    _assert_owners_released(providers, events)
    assert _files(Path(result["checkpoint"])) == checkpoint_before
    assert not output.exists() and not list(tmp_path.glob(".never-output.tmp-*"))


@pytest.mark.parametrize("same_components", [False, True])
def test_success_keeps_projection_live_through_real_export_and_new_input_generation(
        trained, tmp_path, monkeypatch, same_components):
    config, result = trained
    providers, events, writes = {}, [], []
    output = tmp_path / "weights"
    checkpoint_before = _files(Path(result["checkpoint"]))
    with monkeypatch.context() as patch:
        _install_projection(patch, config, providers, events, same_components=same_components)
        _observe_writer(patch, providers, events, writes, output)
        exported = session.export(config, result["checkpoint"], output)
    assert writes == ["EXPORTED"] and exported["checkpoint_cycle"] == 1
    _assert_owners_released(providers, events, same_components=same_components)
    assert _files(Path(result["checkpoint"])) == checkpoint_before
    _, manifest = inference.read_export(output)
    assert manifest["identity"]["training_provenance"] == result["identity"]
    weights = torch.load(output / "weights.pt", weights_only=True)
    for name, module in inference.modules(providers["training"].components.state).items():
        for key, tensor in module.state_dict().items():
            assert torch.equal(weights[name][key], tensor)

    new = tmp_path / "new-inputs"
    build_fixture(new, samples=2)
    prepare(new / "raw/manifest.json", new / "prepared")
    Path(config.data["prepared"]).parent.rename(tmp_path / "original-inputs-unavailable")
    Path(config.trainer["output_dir"]).rename(tmp_path / "original-run-unavailable")
    generated = session.generate_from_export(config, output, tmp_path / "generated",
                                             prepared=new / "prepared/manifest.json", sample_index=1)
    assert generated["kind"] == "INFERENCE_FROM_EXPORTED_WEIGHTS"
    assert generated["input_identity"] != result["identity"]["dataset"]
    assert generated["sample_id"] == "sample-000001"
    values = torch.load(generated["artifact"], weights_only=True)
    assert values["decoded"]["video"].dtype == torch.uint8
    assert len(list((tmp_path / "generated/frames").glob("*.png"))) == 2
    assert (tmp_path / "generated/_SUCCESS").is_file()


@pytest.mark.parametrize("close", [False, 17, "not-callable"])
def test_invalid_projected_close_preserves_validation_error(trained, tmp_path, monkeypatch, close):
    config, result = trained
    providers, events = {}, []
    _install_projection(monkeypatch, config, providers, events, invalid_close=close)
    with pytest.raises(ValueError, match="model close must be callable"):
        session.export(config, result["checkpoint"], tmp_path / "never-output")
    _assert_owners_released(providers, events, same_components=True)
    assert not (tmp_path / "never-output").exists()


@pytest.mark.parametrize("stage", ["validation", "source", "published"])
def test_projected_cleanup_callback_error_policy_and_outer_cleanup(trained, tmp_path, monkeypatch, stage):
    config, result = trained
    providers, events = {}, []
    close_error = OSError("CPU_TEST_ONLY: projection cleanup failed")
    output = tmp_path / "weights"
    _install_projection(monkeypatch, config, providers, events,
                        fault=None if stage == "published" else stage, close_error=close_error)
    with pytest.raises(OSError) as caught:
        session.export(config, result["checkpoint"], output)
    assert caught.value is close_error
    if stage != "published":
        assert isinstance(caught.value.__context__, ValueError)
        assert _ERRORS[stage] in str(caught.value.__context__)
        assert not output.exists() and not list(tmp_path.glob(".weights.tmp-*"))
    else:
        assert caught.value.__context__ is None
        assert inference.read_export(output)[1]["status"] == "EXPORTED"
    _assert_owners_released(providers, events)


def _projection_worker(rank, rendezvous, output, source, raw, fault):
    torch.set_num_threads(1)
    os.environ.update(WORLD_SIZE="2", RANK=str(rank), LOCAL_RANK=str(rank),
                      LOCAL_WORLD_SIZE="2", GROUP_RANK="0")
    dist.init_process_group("gloo", init_method="file://" + rendezvous,
                            rank=rank, world_size=2, timeout=timedelta(seconds=10))
    root, providers, events, writes = Path(output), {}, [], []
    raw = deepcopy(raw)
    raw["trainer"].update(cycles=1, checkpoint_every=1, output_dir=str(root / "training"))
    config = TrainingConfig.from_dict(raw, source=source)
    try:
        trained = session.train(config)
        checkpoint_before = _files(Path(trained["checkpoint"]))
        error = result = None
        exported = root / "weights"
        with pytest.MonkeyPatch.context() as patch:
            local_fault = fault if rank == 1 and fault != "success" else None
            _install_projection(patch, config, providers, events, fault=local_fault)
            _observe_writer(patch, providers, events, writes, exported)
            started = time.monotonic()
            try:
                result = session.export(config, trained["checkpoint"], exported)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            elapsed = time.monotonic() - started
        _peer_finished(root, rank, "export-finished")
        group = _probe_group()
        (root / f"rank-{rank}.json").write_text(json.dumps({
            "status": "CPU_TEST_ONLY", "error": error, "elapsed": elapsed,
            "result": result, "events": events, "writes": writes, "group": group,
            "providers": {name: provider.report() for name, provider in providers.items()},
            "trained": trained["status"], "cycles": trained["completed_cycles"],
            "checkpoint_unchanged": _files(Path(trained["checkpoint"])) == checkpoint_before,
            "output_exists": exported.exists(),
            "staging_exists": bool(list(root.glob(".weights.tmp-*")))}))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("fault", ["validation", "source", "success"])
def test_gloo_projection_transaction_releases_each_owner_and_preserves_caller_group(tmp_path, fault):
    config, raw = _fixture(tmp_path / "media")
    rows = _run_workers(tmp_path / "workers", _projection_worker, str(config.source), raw, fault)
    assert rows[0]["error"] == rows[1]["error"], rows
    for row in rows:
        assert row["trained"] == "COMPLETE" and row["cycles"] == 1, row
        assert row["group"] == {"initialized": True, "error": None, "sum": 3}, row
        assert row["elapsed"] < 8 and row["checkpoint_unchanged"], row
        assert row["events"] == ["projection", "training"], row
        assert set(row["providers"]) == {"training", "projection"}, row
        for provider in row["providers"].values():
            _assert_released(provider)
            assert provider["state_unchanged"] and provider["forwards"] == 0, provider
            assert not any(provider["updates"].values()), provider
        assert not row["staging_exists"], row
        if fault == "success":
            assert row["error"] is None and row["result"]["status"] == "EXPORTED", row
            assert row["writes"] == ["EXPORTED"] and row["output_exists"], row
        else:
            label = "inference projection admission" if fault == "validation" else "inference factory source admission"
            assert row["error"] and label in row["error"] and _ERRORS[fault] in row["error"], row
            assert row["result"] is None and not row["writes"] and not row["output_exists"], row
    if fault == "success":
        assert inference.read_export(tmp_path / "workers/weights")[1]["world_size"] == 2
