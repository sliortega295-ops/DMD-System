import json

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
    lambda x: x["trainer"].update(cylces=4),
    lambda x: x["execution"].update(mode="reference", implementation={"ignored": True}),
    lambda x: x["trainer"].update(seed=-1),
])
def test_training_config_rejects_invalid_requests(tmp_path, change):
    raw = request()
    change(raw)
    with pytest.raises(ValueError):
        TrainingConfig.from_dict(raw, source=tmp_path / "run.json")


def test_user_home_paths_expand_before_config_relative_resolution(tmp_path, monkeypatch):
    home = tmp_path / "user"
    monkeypatch.setenv("HOME", str(home))
    raw = request()
    raw["model"]["factory"] = "~/models/model.py:build"
    raw["algorithm"]["factory"] = "~/workflows/algorithm.py:build"
    raw["data"]["prepared"] = "~/data/manifest.json"
    raw["trainer"].update(output_dir="~/runs/output", resume_from="~/runs/checkpoint")
    config = TrainingConfig.from_dict(raw, source=tmp_path / "config/run.json")
    assert config.model["factory"] == str(home / "models/model.py") + ":build"
    assert config.algorithm["factory"] == str(home / "workflows/algorithm.py") + ":build"
    assert config.data["prepared"] == str(home / "data/manifest.json")
    assert config.trainer["output_dir"] == str(home / "runs/output")
    assert config.trainer["resume_from"] == str(home / "runs/checkpoint")
    from distillschedule.data.manifest import resolve_path
    assert resolve_path("~/raw/video.mp4", tmp_path / "config") == home / "raw/video.mp4"
