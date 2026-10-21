import json
from pathlib import Path

import pytest

from distillschedule.training.config import TrainingConfig, load_training_config


def request():
    return {"schema_version": "distillschedule.training.v1",
            "model": {"factory": "model.py:build"}, "data": {"prepared": "features/manifest.json"},
            "algorithm": {"factory": "workflow.py:build"}, "execution": {"mode": "reference"},
            "trainer": {"cycles": 3, "seed": 7, "output_dir": "run"}}


def test_training_config_paths_includes_and_resume_identity(tmp_path, monkeypatch):
    base = tmp_path / "base.json"
    base.write_text(json.dumps(request()))
    monkeypatch.setenv("DMD_TEST_CYCLES", "5")
    config = tmp_path / "request.json"
    config.write_text(json.dumps({"include": "base.json", "trainer": {"cycles": 5}}))
    loaded = load_training_config(config)
    assert loaded.trainer["cycles"] == 5
    assert loaded.data["prepared"] == str(tmp_path / "features/manifest.json")
    changed = loaded.to_dict()
    changed["trainer"].update(cycles=8, resume_from="checkpoint", output_dir="resumed")
    assert TrainingConfig.from_dict(changed, source=config).identity == loaded.identity
    changed["trainer"]["seed"] = 8
    assert TrainingConfig.from_dict(changed, source=config).identity != loaded.identity


@pytest.mark.parametrize("change", [
    lambda x: x.update(schema_version="wrong"),
    lambda x: x["execution"].update(mode="unsupported"),
    lambda x: x["execution"].update(mode="plan"),
    lambda x: x["trainer"].update(cycles=0),
    lambda x: x["data"].clear(),
])
def test_training_config_rejects_invalid_requests(tmp_path, change):
    raw = request()
    change(raw)
    with pytest.raises(ValueError):
        TrainingConfig.from_dict(raw, source=tmp_path / "run.json")
