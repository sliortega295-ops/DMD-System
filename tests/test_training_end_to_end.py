"""Fresh-process raw-media/train/restart/export and shared-planner acceptance."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from distillschedule.data import prepare
from distillschedule.examples.tiny_av import build_fixture
from distillschedule.training.config import TrainingConfig, load_training_config
from distillschedule.training.session import train


def run_cli(*args):
    result = subprocess.run([sys.executable, "-m", "distillschedule", *map(str, args)],
                            capture_output=True, text=True,
                            env={**os.environ, "OMP_NUM_THREADS": "1"})
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout)


def equal(x, y):
    if isinstance(x, torch.Tensor):
        return isinstance(y, torch.Tensor) and x.dtype == y.dtype and torch.equal(x, y)
    if isinstance(x, dict):
        return x.keys() == y.keys() and all(equal(value, y[key]) for key, value in x.items())
    if isinstance(x, (list, tuple)):
        return len(x) == len(y) and all(equal(a, b) for a, b in zip(x, y))
    return x == y


def fixture(tmp_path, samples=6):
    root = tmp_path / "media"
    build_fixture(root, samples=samples)
    prepare(root / "raw/manifest.json", root / "prepared")
    return root, load_training_config(root / "train.json")


def test_fresh_process_resume_matches_all_training_state_and_generates(tmp_path):
    root, config = fixture(tmp_path)
    assert run_cli("doctor", "--config", root / "train.json")["model_imported"] is False
    first = run_cli("train", root / "train.json")
    checkpoint = Path(first["checkpoint"])
    resumed = run_cli("train", root / "train.json", "--resume", checkpoint,
                      "--cycles", 6, "--output", root / "resumed")
    uninterrupted = run_cli("train", root / "train.json", "--cycles", 6,
                            "--output", root / "uninterrupted")
    a = torch.load(Path(resumed["checkpoint"]) / "state.pt", weights_only=True)
    b = torch.load(Path(uninterrupted["checkpoint"]) / "state.pt", weights_only=True)
    for key in ("state", "rng", "cycle", "cursor", "versions", "policy"):
        assert equal(a[key], b[key]), key
    assert resumed["resume_cycle"] == 4 and resumed["cycles_this_process"] == 2
    assert resumed["final_versions"]["student"] == 6
    assert resumed["final_versions"]["fake"] == 12
    assert resumed["final_versions"]["teacher"] == 0
    generated = run_cli("evaluate", root / "train.json", "--checkpoint", checkpoint,
                        "--output", root / "generated")
    artifact = torch.load(generated["artifact"], weights_only=True)
    assert artifact["decoded"]["video"].dtype == torch.uint8
    exported = run_cli("export", root / "train.json", "--checkpoint", checkpoint,
                       "--output", root / "exported")
    assert exported["kind"] == "INFERENCE_WEIGHTS_NOT_RESUMABLE_CHECKPOINT"
    assert torch.load(root / "exported/weights.pt", weights_only=True)["student"]
    rows = [json.loads(row) for row in (root / "resumed/metrics.jsonl").read_text().splitlines()]
    assert [row["sample_id"] for row in rows] == ["sample-000004", "sample-000005"]


def test_measured_auto_plan_is_executed_and_reused_across_real_updates(tmp_path):
    root, config = fixture(tmp_path, samples=4)
    raw = config.to_dict()
    raw["trainer"].update(cycles=3, output_dir=str(root / "auto"))
    raw["execution"] = {"mode": "auto", "memory_limit_bytes": 100_000_000,
                        "max_candidates": 3, "samples": 1, "max_program_runs": 8}
    automatic = train(TrainingConfig.from_dict(raw, source=config.source))
    assert automatic["policy"]["plan_sha256"]
    assert automatic["policy"]["initial_reference_gate"]["status"] == "PASS"
    assert automatic["final_versions"]["fake"] == 6
    plan_path = root / "auto/planning/selected-plan.json"
    document = json.loads(plan_path.read_text())
    rows = [json.loads(row) for row in (root / "auto/metrics.jsonl").read_text().splitlines()]
    assert rows[1]["execution"]["dispatcher"]["executed_operations"] == document["selected"]["order"]
    # A saved plan from this same initial state also drives a fresh real run.
    raw["trainer"]["output_dir"] = str(root / "saved-plan")
    raw["execution"] = {"mode": "plan", "plan": str(plan_path)}
    saved = train(TrainingConfig.from_dict(raw, source=config.source))
    assert saved["policy"]["implementation"] == document["selected"]["implementation"]
    assert saved["policy"]["plan_sha256"] == document["plan_sha256"]
    assert saved["final_versions"] == automatic["final_versions"]


def test_unknown_model_factory_without_solver_edits(tmp_path):
    root, config = fixture(tmp_path, samples=4)
    template = Path(__file__).resolve().parents[1] / "examples/framework/newmodel.py"
    raw = config.to_dict()
    raw["model"]["factory"] = str(template) + ":build_model"
    raw["trainer"].update(cycles=2, output_dir=str(root / "new-model"))
    result = train(TrainingConfig.from_dict(raw, source=config.source))
    assert result["completed_cycles"] == 2
    assert result["final_versions"]["student"] == 2
    assert result["identity"]["model"]["reference"].endswith("newmodel.py:build_model")


def test_invalid_resume_identity_rejects_without_creating_output(tmp_path):
    root, config = fixture(tmp_path, samples=4)
    completed = train(config)
    raw = config.to_dict()
    raw["trainer"].update(seed=99, cycles=5, resume_from=completed["checkpoint"],
                          output_dir=str(root / "wrong"))
    # Supply one extra sample bound to this modified request first, so failure
    # is checkpoint identity rather than input exhaustion.
    raw["trainer"]["cycles"] = 4
    with pytest.raises(ValueError, match="identity"):
        train(TrainingConfig.from_dict(raw, source=config.source))
    assert not (root / "wrong").exists()
