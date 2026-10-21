import json
import os
from pathlib import Path
import subprocess
import sys

from distillschedule.cli import main


def test_doctor_does_not_import_torch_or_initialize_cuda():
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    result = subprocess.run([sys.executable, "-c",
        "import sys; from distillschedule.cli import main; "
        "assert main(['doctor']) == 0; assert 'torch' not in sys.modules"],
        env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["model_imported"] is False


def test_training_config_invalid_before_plugin_import(tmp_path):
    raw = {"schema_version": "distillschedule.training.v1", "model": {"factory": "missing:build"},
           "algorithm": {"factory": "missing:build"}, "data": {"prepared": "missing.json"},
           "execution": {"mode": "plan"}, "trainer": {"cycles": 2, "output_dir": "run"}}
    path = tmp_path / "train.json"
    path.write_text(json.dumps(raw))
    assert main(["train", str(path), "--dry-run"]) == 2
    assert not (tmp_path / "run").exists()
