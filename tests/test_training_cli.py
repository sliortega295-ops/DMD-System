import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from distillschedule.cli import main


@pytest.mark.parametrize('command', ['doctor', 'models', 'algorithms', 'strategies', 'capabilities'])
def test_doctor_does_not_import_torch_or_initialize_cuda(command):
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    result = subprocess.run([sys.executable, "-c",
        "import sys; from distillschedule.cli import main; "
        f"assert main([{command!r}]) == 0; assert 'torch' not in sys.modules; "
        "assert 'diffusers' not in sys.modules"],
        env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    parsed = json.loads(result.stdout)
    if command == 'doctor':
        assert parsed["model_imported"] is False


def test_doctor_reports_optional_tensorflow_import_advice_without_loading_it(monkeypatch):
    import importlib.metadata
    from distillschedule.training.cli import doctor

    def version(name):
        if name == "tensorflow":
            return "SYNTHETIC_METADATA_ONLY"
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", version)
    monkeypatch.delenv("USE_TF", raising=False)
    monkeypatch.delenv("USE_FLAX", raising=False)
    result = doctor()
    assert result["environment_controls"] == {"USE_TF": None, "USE_FLAX": None}
    assert result["import_warnings"][0]["code"] == "OPTIONAL_TENSORFLOW_IMPORTS"
    assert "USE_TF=0" in result["import_warnings"][0]["message"]
    assert "tensorflow" not in sys.modules and "keras" not in sys.modules
    monkeypatch.setenv("USE_TF", "0")
    monkeypatch.setenv("USE_FLAX", "0")
    configured = doctor()
    assert configured["environment_controls"] == {"USE_TF": "0", "USE_FLAX": "0"}
    assert configured["import_warnings"] == []
    assert configured["gpu_execution"] == "NOT_RUN"


def test_training_config_invalid_before_plugin_import(tmp_path):
    raw = {"schema_version": "distillschedule.training.v1", "model": {"factory": "missing:build"},
           "algorithm": {"factory": "missing:build"}, "data": {"prepared": "missing.json"},
           "execution": {"mode": "plan"}, "trainer": {"cycles": 2, "output_dir": "run"}}
    path = tmp_path / "train.json"
    path.write_text(json.dumps(raw))
    assert main(["train", str(path), "--dry-run"]) == 2
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize('args', [['train', '--help'], ['config', 'show'],
                                  ['config', 'validate'], ['train'], ['doctor', '--config']])
def test_inspection_stays_on_stdout_without_importing_plugin(tmp_path, args):
    plugin = tmp_path / 'must_not_import.py'
    plugin.write_text('raise AssertionError("inspection imported the model")\n')
    prepared = tmp_path / 'prepared.json'
    prepared.write_text('{}')  # Inspection checks existence, not model/tensor validity.
    raw = {'schema_version': 'distillschedule.training.v1',
           'model': {'factory': str(plugin) + ':build'},
           'algorithm': {'factory': str(plugin) + ':build'}, 'data': {'prepared': str(prepared)},
           'execution': {'mode': 'reference'}, 'trainer': {'cycles': 2, 'output_dir': 'run'}}
    path = tmp_path / 'train.json'
    path.write_text(json.dumps(raw))
    argv = list(args)
    if '--help' not in argv:
        argv.append(str(path))
        if argv[0] == 'train':
            argv.append('--dry-run')
    script = '''import sys
from distillschedule.cli import main
try:
    code = main(sys.argv[1:])
except SystemExit as exc:
    assert '--help' in sys.argv and exc.code == 0
else:
    assert code == 0
assert not any(name in sys.modules for name in ('torch', 'diffusers', 'deepspeed'))
'''
    completed = subprocess.run([sys.executable, '-c', script, *argv], text=True, capture_output=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    if '--help' in argv:
        assert 'usage: distillctl train' in completed.stdout
    else:
        assert isinstance(json.loads(completed.stdout), dict)
    assert completed.stderr == '' and not (tmp_path / 'run').exists()


@pytest.mark.parametrize('stage', ['import', 'call'])
@pytest.mark.parametrize('error', ['RuntimeError', 'KeyboardInterrupt'])
@pytest.mark.parametrize('command', ['train', 'evaluate', 'export', 'generate'])
def test_model_failure_restores_streams_without_swallowing_interrupt(tmp_path, stage, error, command):
    from test_training_end_to_end import fixture
    from distillschedule.training.config import TrainingConfig
    from distillschedule.training.session import export, train
    root, config = fixture(tmp_path)
    plugin = root / 'broken.py'
    plugin.write_text(f'''import logging
import os
from distillschedule.examples.tiny_av import build_model
def fail():
    print("model diagnostic")
    logging.warning("model root log")
    logging.getLogger("preconfigured.child").warning("model named log")
    raise {error}("model failed")
if os.environ.get("DMD_TEST_MODEL_FAILURE_STAGE") == "import":
    fail()
def build(parameters):
    if os.environ.get("DMD_TEST_MODEL_FAILURE_STAGE") == "call":
        fail()
    return build_model(parameters)
''')
    raw = config.to_dict()
    raw['model']['factory'] = str(plugin) + ':build'
    path = root / 'broken.json'
    path.write_text(json.dumps(raw))
    argv = [command, str(path)]
    if command != 'train':
        # Create genuine inputs while the immutable plugin's failure switch is
        # off; enable it only in the fresh CLI process below.
        config = TrainingConfig.from_dict(raw, source=config.source)
        trained = train(config)
        argv += ['--output', str(root / 'failed-output')]
        if command == 'generate':
            export(config, trained['checkpoint'], root / 'weights')
            argv += ['--weights', str(root / 'weights'), '--prepared', str(root / 'prepared/manifest.json')]
        else:
            argv += ['--checkpoint', trained['checkpoint']]
    script = f'''import io
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
try:
    code = main(sys.argv[1:])
except KeyboardInterrupt as exc:
    assert {error!r} == "KeyboardInterrupt" and str(exc) == "model failed"
else:
    assert {error!r} == "RuntimeError" and code == 2
assert sys.stdout is stdout and sys.stderr is stderr and handler.stream is stdout
assert named_handler.stream is stdout and unrelated_handler.stream is unrelated
assert unrelated.getvalue() == "model named log\\n" and not unrelated.closed
print("STREAMS_RESTORED")
'''
    completed = subprocess.run([sys.executable, '-c', script, *argv], text=True, capture_output=True,
                               env={**os.environ, 'DMD_TEST_MODEL_FAILURE_STAGE': stage, 'OMP_NUM_THREADS': '1'})
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stdout == 'STREAMS_RESTORED\n'
    assert 'model diagnostic' in completed.stderr
    assert 'model root log' in completed.stderr and 'model named log' in completed.stderr
    assert ('ERROR: model failed' in completed.stderr) is (error == 'RuntimeError')
    assert not (root / ('run/checkpoints' if command == 'train' else 'failed-output')).exists()


@pytest.mark.parametrize('return_code', [0, 19])
def test_native_frontdoor_keeps_its_stdout_and_return_code(tmp_path, return_code):
    from test_training_native import config
    path = tmp_path / 'native.json'
    path.write_text(json.dumps(config(tmp_path).to_dict()))
    # Boundary characterization only: real public and native routing, with the
    # GPU-owning front door replaced by an explicit CPU callback, not a GPU receipt.
    script = f'''import logging
import sys
from distillschedule.cli import main
from distillschedule.training import native
from distillschedule.strategies import h3_ar_cli
stdout, stderr = sys.stdout, sys.stderr
handler = logging.StreamHandler(stdout)
logging.getLogger().addHandler(handler)
logger = logging.getLogger("preconfigured")
logger.propagate = False
named_handler = logging.StreamHandler(stdout)
logger.addHandler(named_handler)
native._admit = lambda config: ("W6", {{"identity": {{"dirty": False}}}}, {{}}, {{"argv": ["cpu-stream-test"]}})
def frontdoor(argv):
    assert argv == ["cpu-stream-test"]
    assert sys.stdout is stdout and sys.stderr is stderr
    assert handler.stream is stdout and named_handler.stream is stdout
    print("NATIVE_FRONTDOOR_CPU_CALLBACK")
    logging.warning("native root log")
    logger.warning("native named log")
    print("native diagnostic", file=sys.stderr)
    return {return_code}
h3_ar_cli.main = frontdoor
code = main(sys.argv[1:])
assert code == {return_code} and sys.stdout is stdout and sys.stderr is stderr
assert handler.stream is stdout and named_handler.stream is stdout
raise SystemExit(code)
'''
    completed = subprocess.run([sys.executable, '-c', script, 'train', str(path)],
                               text=True, capture_output=True)
    assert completed.returncode == return_code, completed.stdout + completed.stderr
    assert completed.stdout == 'NATIVE_FRONTDOOR_CPU_CALLBACK\nnative root log\nnative named log\n'
    assert completed.stderr == 'native diagnostic\n'
