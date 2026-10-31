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
    # A real user plugin can configure stdout logging while it is imported.
    # Keep the actual model, algorithm, checkpoint and inference chain intact.
    plugin = root / 'noisy.py'
    plugin.write_text('''import logging
import sys
from distillschedule.examples import tiny_av
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.propagate = False
def diagnostic(stage):
    print(stage + " print")
    logger.warning(stage + " log")
    logging.warning(stage + " root log")
    logging.getLogger("preconfigured.child").warning(stage + " named log")
diagnostic("import")
def build_model(parameters):
    diagnostic("model")
    components = tiny_av.build_model(parameters)
    components.generate = generate
    components.decode = decode
    return components
def build_algorithm(*args):
    diagnostic("algorithm")
    return tiny_av.build_algorithm(*args)
def generate(*args):
    diagnostic("generate")
    return tiny_av.generate(*args)
def decode(*args):
    diagnostic("decode")
    return tiny_av.decode(*args)
''')
    raw = config.to_dict()
    raw['model']['factory'] = str(plugin) + ':build_model'
    raw['algorithm']['factory'] = str(plugin) + ':build_algorithm'
    (root / 'train.json').write_text(json.dumps(raw))

    def noisy_cli(*args):
        script = '''import io
import logging
import sys
from distillschedule.cli import main
stdout, stderr = sys.stdout, sys.stderr
handler = logging.StreamHandler(stdout)
logging.getLogger().addHandler(handler)
logger = logging.getLogger("preconfigured.child")
logger.propagate = False
named_handler = logging.StreamHandler(stdout)
logger.addHandler(named_handler)
unrelated = io.StringIO()
unrelated_handler = logging.StreamHandler(unrelated)
logger.addHandler(unrelated_handler)
code = main(sys.argv[1:])
assert sys.stdout is stdout and sys.stderr is stderr
assert handler.stream is stdout and named_handler.stream is stdout
assert unrelated_handler.stream is unrelated and not unrelated.closed
assert "model named log" in unrelated.getvalue()
raise SystemExit(code)
'''
        completed = subprocess.run([sys.executable, '-c', script, *map(str, args)],
                                   capture_output=True, text=True,
                                   env={**os.environ, 'OMP_NUM_THREADS': '1'})
        assert completed.returncode == 0, completed.stdout + completed.stderr
        result = json.loads(completed.stdout)
        stages = ['import', 'model']
        if args[0] in ('train', 'evaluate', 'export'):
            stages.append('algorithm')
        if args[0] in ('evaluate', 'generate'):
            stages.extend(('generate', 'decode'))
        assert all(stage + ' ' + kind in completed.stderr for stage in stages
                   for kind in ('print', 'log', 'root log', 'named log'))
        return result

    assert run_cli("doctor", "--config", root / "train.json")["model_imported"] is False
    first = noisy_cli("train", root / "train.json")
    checkpoint = Path(first["checkpoint"])
    resumed = noisy_cli("train", root / "train.json", "--resume", checkpoint,
                      "--cycles", 6, "--output", root / "resumed")
    uninterrupted = noisy_cli("train", root / "train.json", "--cycles", 6,
                            "--output", root / "uninterrupted")
    a = torch.load(Path(resumed["checkpoint"]) / "state.pt", weights_only=True)
    b = torch.load(Path(uninterrupted["checkpoint"]) / "state.pt", weights_only=True)
    for key in ("state", "rng", "cycle", "cursor", "versions", "policy"):
        assert equal(a[key], b[key]), key
    assert resumed["resume_cycle"] == 4 and resumed["cycles_this_process"] == 2
    assert resumed["final_versions"]["student"] == 6
    assert resumed["final_versions"]["fake"] == 12
    assert resumed["final_versions"]["teacher"] == 0
    generated = noisy_cli("evaluate", root / "train.json", "--checkpoint", checkpoint,
                        "--output", root / "generated")
    artifact = torch.load(generated["artifact"], weights_only=True)
    assert artifact["decoded"]["video"].dtype == torch.uint8
    exported = noisy_cli("export", root / "train.json", "--checkpoint", checkpoint,
                       "--output", root / "exported")
    assert exported["kind"] == "INFERENCE_WEIGHTS_NOT_RESUMABLE_CHECKPOINT"
    assert torch.load(root / "exported/weights.pt", weights_only=True)["student"]
    portable = noisy_cli('generate', root / 'train.json', '--weights', root / 'exported',
                        '--prepared', root / 'prepared/manifest.json', '--output', root / 'portable')
    assert torch.load(portable['artifact'], weights_only=True)['decoded']['video'].dtype == torch.uint8
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
