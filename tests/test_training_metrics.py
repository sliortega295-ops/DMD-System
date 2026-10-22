"""Read-only host loss collection for atomic original-native operations."""
from copy import deepcopy
from pathlib import Path
import json

import pytest
import torch

from distillschedule.training.adapters import ModelComponents


def test_optional_host_metrics_keep_existing_scalar_metrics():
    from distillschedule.training.session import _training_metrics
    outputs = {"loss": torch.tensor(2.0), "native": {"loss": 3.5}}
    plain = ModelComponents(state={"value": 0}).validate()
    assert _training_metrics(plain, outputs) == {"loss": 2.0}
    model = ModelComponents(state={"value": 0}, collect_metrics=lambda rows: {"native/loss": rows["native"]["loss"]})
    assert _training_metrics(model.validate(), outputs) == {"loss": 2.0, "native/loss": 3.5}


@pytest.mark.parametrize("extra", [None, [], {"loss": 3.0}, {"": 1.0}, {3: 1.0},
                                   {"native": True}, {"native": 1}, {"native": torch.tensor(1.)},
                                   {"native": float("nan")}, {"native": float("inf")}])
def test_invalid_host_metrics_reject_without_tensor_coercion(extra):
    from distillschedule.training.session import _training_metrics
    model = ModelComponents(state={"value": 0}, collect_metrics=lambda rows: extra)
    with pytest.raises(ValueError, match="metric"):
        _training_metrics(model, {"loss": torch.tensor(2.)})


def test_metric_callback_must_be_callable():
    with pytest.raises(ValueError, match="collect_metrics must be callable"):
        ModelComponents(state={"value": 0}, collect_metrics=True).validate()


def test_real_training_host_metrics_do_not_change_updates_rng_or_data(tmp_path, monkeypatch):
    from distillschedule.data import prepare
    from distillschedule.examples.tiny_av import build_fixture
    from distillschedule.training.config import load_training_config, TrainingConfig
    from distillschedule.training import session
    from test_training_end_to_end import equal

    root = tmp_path / "media"
    build_fixture(root, samples=2)
    prepare(root / "raw/manifest.json", root / "prepared")
    loaded = load_training_config(root / "train.json")
    raw = loaded.to_dict()
    raw["trainer"]["cycles"] = 2
    config = TrainingConfig.from_dict(raw, source=loaded.source)
    control = session.train(config)
    original = session._construct_model
    seen = []

    def factory(*args, **kwargs):
        components, identity = original(*args, **kwargs)

        def collect(outputs):
            losses = [value for value in outputs.values() if isinstance(value, torch.Tensor) and value.numel() == 1]
            assert losses
            value = float(losses[0].detach())
            seen.append(value)
            return {"adapter/first_loss": value}

        components.collect_metrics = collect
        return components, identity

    monkeypatch.setattr(session, "_construct_model", factory)
    raw = deepcopy(config.to_dict())
    raw["trainer"]["output_dir"] = str(tmp_path / "with-metrics")
    candidate = session.train(TrainingConfig.from_dict(raw, source=config.source))
    a = torch.load(Path(control["checkpoint"]) / "state.pt", weights_only=True)
    b = torch.load(Path(candidate["checkpoint"]) / "state.pt", weights_only=True)
    for key in ("state", "rng", "cursor", "cycle", "versions", "policy"):
        assert equal(a[key], b[key]), key
    rows = [json.loads(line) for line in (tmp_path / "with-metrics/metrics.jsonl").read_text().splitlines()]
    assert len(rows) == len(seen) == 2
    assert [row["metrics"]["adapter/first_loss"] for row in rows] == seen
    assert [row["sample_id"] for row in rows] == ["sample-000000", "sample-000001"]
    assert not torch.cuda.is_initialized()
