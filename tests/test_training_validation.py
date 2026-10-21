"""Optional held-out callbacks are read-only, RNG-isolated diagnostics, not quality gates."""
import json
import random
from copy import deepcopy

import numpy as np
import pytest
import torch
from test_training_end_to_end import equal, fixture

from distillschedule.examples.tiny_av import build_model
from distillschedule.training.config import TrainingConfig
from distillschedule.training.session import train


def stochastic_metric(state, feeds, parameters):
    assert not torch.is_grad_enabled()
    assert all(not child.training for module in state.values() if isinstance(module, torch.nn.Module)
               for child in module.modules())
    prediction = state["student"](feeds["latent"], feeds["condition"])
    # All three ambient RNGs must be isolated, even for a stochastic adapter.
    noise = random.random() + float(np.random.rand()) + float(torch.rand(()))
    return {"heldout_feature_mse": (prediction - feeds["latent"]).square().mean() + noise}


def with_metric(parameters):
    model = build_model(parameters)
    model.validate_batch = stochastic_metric
    return model


def wrong_signature(state):
    return {"metric": 0.}


def with_wrong_signature(parameters):
    model = build_model(parameters)
    model.validate_batch = wrong_signature
    return model


def mutate(state, feeds, parameters):
    state["student_optimizer"].param_groups[0]["lr"] *= .5
    return {"metric": 0.}


def with_mutation(parameters):
    model = build_model(parameters)
    model.validate_batch = mutate
    return model


def nonfinite(state, feeds, parameters):
    return {"metric": float("nan")}


def raising_metric(state, feeds, parameters):
    random.random()
    np.random.rand()
    torch.rand(())
    raise RuntimeError("adapter validation failed")


def mutate_declared_state(state, feeds, parameters):
    target = parameters["target"]
    if target == "parameter":
        next(state["student"].parameters()).add_(1)
    elif target == "gradient":
        parameter = next(state["student"].parameters())
        parameter.grad = torch.ones_like(parameter)
    elif target == "buffer":
        state["student"].validation_buffer.add_(1)
    elif target == "scheduler":
        state["scheduler"].last_epoch += 1
    elif target == "scheduler_owner":
        state["scheduler"].optimizer = state["fake_optimizer"]
    else:
        state[target].add_(1)
    return {"metric": 0.}


def with_nonfinite(parameters):
    model = build_model(parameters)
    model.validate_batch = nonfinite
    return model


def validation_request(config, output, *, factory="with_metric", samples=2):
    raw = config.to_dict()
    raw["model"]["factory"] = __file__ + ":" + factory
    raw["data"]["validation_prepared"] = raw["data"]["prepared"]
    raw["trainer"].update(cycles=3, output_dir=str(output), checkpoint_every=1,
                          validation={"every_cycles": 1, "samples": samples, "parameters": {}})
    return raw


def test_validation_preserves_all_training_state_rng_cursor_and_fixed_ids(tmp_path):
    root, config = fixture(tmp_path, samples=6)
    raw = validation_request(config, root / "enabled")
    enabled = train(TrainingConfig.from_dict(raw, source=config.source))
    raw["trainer"].pop("validation")
    raw["data"].pop("validation_prepared")
    raw["trainer"]["output_dir"] = str(root / "disabled")
    disabled = train(TrainingConfig.from_dict(raw, source=config.source))
    a = torch.load(enabled["checkpoint"] + "/state.pt", weights_only=True)
    b = torch.load(disabled["checkpoint"] + "/state.pt", weights_only=True)
    for key in ("state", "rng", "cursor", "versions"):
        assert equal(a[key], b[key]), key
    rows = [json.loads(line) for line in (root / "enabled/validation.jsonl").read_text().splitlines()]
    assert [row["cycle"] for row in rows] == [1, 2, 3]
    assert all(row["sample_ids"] == ["sample-000000", "sample-000001"] for row in rows)
    assert enabled["cursor"] == disabled["cursor"] == 3
    assert enabled["quality_validation"] == disabled["quality_validation"] == "NOT_RUN"
    assert enabled["validation"]["status"] == "COMPLETED_ADAPTER_METRICS"
    assert enabled["validation"]["last"]["ranks"] == [rows[-1]]
    assert disabled["validation"]["status"] == "DISABLED"
    assert not (root / "disabled/validation.jsonl").exists()


@pytest.mark.parametrize("factory,message", [("with_wrong_signature", "signature"),
    (None, "validate_batch"), ("with_metric", "validation samples")])
def test_invalid_validation_admission_has_no_updates_or_output(tmp_path, monkeypatch, factory, message):
    from distillschedule.training.policy import CyclePolicy
    root, config = fixture(tmp_path, samples=4)
    raw = validation_request(config, root / "invalid", factory=factory or "with_metric", samples=5 if factory == "with_metric" else 1)
    if factory is None:
        raw["model"]["factory"] = config.model["factory"]
    monkeypatch.setattr(CyclePolicy, "run", lambda *a, **kw: pytest.fail("training updated before admission"))
    with pytest.raises(ValueError, match=message):
        train(TrainingConfig.from_dict(raw, source=config.source))
    assert not (root / "invalid").exists()


@pytest.mark.parametrize("change", [
    lambda raw: raw["data"].pop("validation_prepared"),
    lambda raw: raw["trainer"].pop("validation"),
    lambda raw: raw["trainer"]["validation"].update(samples=True),
    lambda raw: raw["trainer"]["validation"].update(every_cycles=0),
    lambda raw: raw["trainer"]["validation"].update(ignored=True),
    lambda raw: raw["trainer"]["validation"].update(parameters={"x": float("inf")}),
    lambda raw: raw["trainer"]["validation"].update(parameters=[1]),
    lambda raw: raw["trainer"]["validation"].update(parameters={1: "not a JSON key"}),
])
def test_validation_config_rejects_unpaired_unknown_and_non_json_fields(tmp_path, change):
    _, config = fixture(tmp_path, samples=4)
    raw = validation_request(config, tmp_path / "invalid")
    change(raw)
    with pytest.raises(ValueError, match="validation"):
        TrainingConfig.from_dict(raw, source=config.source)


def test_callback_modes_and_all_rng_restore_on_failure_and_mutation_is_not_rolled_back(tmp_path):
    from distillschedule.training.checkpoint import rng_state
    from distillschedule.training.validation import Validation
    _, config = fixture(tmp_path, samples=4)
    raw = validation_request(config, tmp_path / "unused")
    config = TrainingConfig.from_dict(raw, source=config.source)
    model = with_metric({"seed": 7})
    model.state["student"].layers[0].eval()  # Mixed nested modes, not just root mode.
    modes = [(module, module.training) for key in ("student", "fake", "teacher")
             for module in model.state[key].modules()]
    validation = Validation.prepare(config, model)
    before = deepcopy(rng_state())
    first = validation.run(config, model.state, cycle=1)
    assert equal(before, rng_state())
    assert first["metrics"] == validation.run(config, model.state, cycle=2)["metrics"]
    assert all(module.training == original for module, original in modes)
    model.validate_batch = mutate
    validation = Validation.prepare(config, model)
    lr = model.state["student_optimizer"].param_groups[0]["lr"]
    with pytest.raises(ValueError, match="mutated.*state"):
        validation.run(config, model.state, cycle=1)
    assert model.state["student_optimizer"].param_groups[0]["lr"] == lr * .5
    assert equal(before, rng_state())
    assert all(module.training == original for module, original in modes)
    model.validate_batch = raising_metric
    validation = Validation.prepare(config, model)
    with pytest.raises(RuntimeError, match="adapter validation failed"):
        validation.run(config, model.state, cycle=1)
    assert equal(before, rng_state())
    assert all(module.training == original for module, original in modes)


def test_validation_rng_only_queries_and_restores_owned_cuda_device(tmp_path, monkeypatch):
    from distillschedule.training.validation import Validation
    _, config = fixture(tmp_path, samples=4)
    config = TrainingConfig.from_dict(validation_request(config, tmp_path / "unused"), source=config.source)
    model = with_metric({"seed": 7})
    validation = Validation.prepare(config, model)
    expected = torch.arange(8, dtype=torch.uint8)
    restored = []
    seeded = []
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda device: expected.clone() if device == 3 else pytest.fail("foreign CUDA RNG"))
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: pytest.fail("foreign CUDA contexts"))
    monkeypatch.setattr(torch.cuda, "manual_seed", lambda seed: seeded.append((3, seed)))
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda state, device: restored.append((device, state.clone())))
    validation.run(config, model.state, cycle=1)
    assert len(seeded) == 2 and all(device == 3 for device, _ in seeded)
    assert len(restored) == 1 and restored[0][0] == 3 and torch.equal(restored[0][1], expected)


@pytest.mark.parametrize("value", [{}, [], {"metric": True}, {"metric": complex(1, 0)},
    {"metric": torch.ones(2)}, {"metric": torch.tensor(float("inf"))}, {"metric": 10 ** 1000}])
def test_validation_metrics_must_be_named_finite_scalars(value):
    from distillschedule.training.validation import _metrics
    with pytest.raises(ValueError, match="finite scalar"):
        _metrics(value)


@pytest.mark.parametrize("target", ["parameter", "gradient", "buffer", "scheduler", "scheduler_owner", "ema", "history"])
def test_validation_guards_all_declared_persistent_state(tmp_path, target):
    from distillschedule.training.validation import Validation
    _, config = fixture(tmp_path, samples=4)
    raw = validation_request(config, tmp_path / "unused")
    raw["trainer"]["validation"]["parameters"] = {"target": target}
    config = TrainingConfig.from_dict(raw, source=config.source)
    model = with_metric({"seed": 7})
    model.state["student"].register_buffer("validation_buffer", torch.zeros(()))
    model.state.update(ema=torch.zeros(2), history=torch.zeros(2), scheduler=torch.optim.lr_scheduler.ExponentialLR(
        model.state["student_optimizer"], gamma=.9))
    model.validate_batch = mutate_declared_state
    validation = Validation.prepare(config, model)
    with pytest.raises(ValueError, match="mutated.*state"):
        validation.run(config, model.state, cycle=1)


@pytest.mark.parametrize("factory,message", [("with_mutation", "mutated.*state"), ("with_nonfinite", "finite scalar")])
def test_bad_callback_cannot_publish_a_complete_checkpoint(tmp_path, factory, message):
    root, config = fixture(tmp_path, samples=4)
    raw = validation_request(config, root / "failed", factory=factory)
    with pytest.raises(ValueError, match=message):
        train(TrainingConfig.from_dict(raw, source=config.source))
    assert not list((root / "failed").glob("checkpoints/**/_SUCCESS"))
    assert not (root / "failed/summary.json").exists()


def test_changed_validation_dataset_rejects_restart_before_output(tmp_path):
    root, config = fixture(tmp_path, samples=6)
    raw = validation_request(config, root / "first")
    manifest = root / "prepared/manifest.json"
    validation_manifest = manifest.with_name("validation.json")
    validation_manifest.write_bytes(manifest.read_bytes())
    raw["data"]["validation_prepared"] = str(validation_manifest)
    original_config_identity = TrainingConfig.from_dict(raw, source=config.source).identity
    first = train(TrainingConfig.from_dict(raw, source=config.source))
    value = json.loads(validation_manifest.read_text())
    value["metadata"]["validation_revision"] = 2
    # Same request/path; only held-out content changes, not training content.
    validation_manifest.write_text(json.dumps(value))
    raw["trainer"].update(cycles=4, resume_from=first["checkpoint"], output_dir=str(root / "rejected"))
    assert TrainingConfig.from_dict(raw, source=config.source).identity == original_config_identity
    with pytest.raises(ValueError, match="identity"):
        train(TrainingConfig.from_dict(raw, source=config.source))
    assert not (root / "rejected").exists()


def test_validation_interval_does_not_force_a_final_callback(tmp_path):
    root, config = fixture(tmp_path, samples=4)
    raw = validation_request(config, root / "not-due", factory="with_nonfinite")
    raw["trainer"]["validation"]["every_cycles"] = 5
    result = train(TrainingConfig.from_dict(raw, source=config.source))
    assert result["validation"] == {"status": "NOT_DUE", "runs_this_process": 0, "last": None}
    assert not (root / "not-due/validation.jsonl").exists()


def test_distributed_validation_fails_closed_before_model_construction(tmp_path, monkeypatch):
    from distillschedule.training import distributed, session
    _, config = fixture(tmp_path, samples=4)
    raw = validation_request(config, tmp_path / "distributed")
    monkeypatch.setattr(distributed, "rank_world", lambda: (0, 2))
    monkeypatch.setattr(distributed, "checked", lambda fn, label: fn())
    monkeypatch.setattr(session, "construct", lambda config: pytest.fail("model constructed"))
    with pytest.raises(ValueError, match="validation.*single.process"):
        session._train(TrainingConfig.from_dict(raw, source=config.source))
    assert not (tmp_path / "distributed").exists()


def _distributed_rejection_worker(rank, rendezvous, config_path):
    import os
    from datetime import timedelta

    import torch.distributed as dist

    os.environ.update(WORLD_SIZE="2", RANK=str(rank), LOCAL_RANK=str(rank), LOCAL_WORLD_SIZE="2", GROUP_RANK="0")
    dist.init_process_group("gloo", init_method="file://" + rendezvous,
                            rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        from distillschedule.training.config import load_training_config
        config = load_training_config(config_path)
        with pytest.raises(ValueError, match="validation.*single.process"):
            train(config)
    finally:
        dist.destroy_process_group()


def test_real_gloo_validation_rejection_is_collective_before_factory_import(tmp_path):
    import torch.multiprocessing as mp
    root, config = fixture(tmp_path, samples=4)
    raw = validation_request(config, root / "rejected-distributed")
    raw["data"]["sharding"] = "rank_strided"
    raw["model"]["factory"] = "must_never_import_missing_adapter:build_model"
    config_path = root / "distributed-validation.json"
    config_path.write_text(json.dumps(raw))
    mp.spawn(_distributed_rejection_worker, args=(str(root / "validation-rendezvous"), str(config_path)), nprocs=2, join=True)
    assert not (root / "rejected-distributed").exists()
