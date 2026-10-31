from __future__ import annotations

import json
import subprocess
import sys

import pytest

from distillschedule.data.cli import main
from test_training_h3_grid import cached_data as cached_data


def test_inspect_and_imports_do_not_load_torch(tmp_path):
    import hashlib
    import torch
    from distillschedule.data import import_preencoded

    feature = tmp_path / "tensor.pt"
    torch.save({"latent": torch.ones(1, 2, 3)}, feature)
    source = tmp_path / "source.json"
    source.write_text(json.dumps({
        "schema_version": "distillschedule.data.preencoded.v1",
        "records": [{"id": "one", "features": {"path": feature.name,
                    "sha256": hashlib.sha256(feature.read_bytes()).hexdigest()}}],
    }))
    manifest = import_preencoded(source, tmp_path / "prepared")
    script = f"from distillschedule.data.cli import main; import sys; assert 'torch' not in sys.modules; assert main(['inspect', {str(manifest)!r}]) == 0; assert 'torch' not in sys.modules"
    completed = subprocess.run([sys.executable, "-c", script], text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["validation"] == "HASHES_ONLY_TENSORS_NOT_LOADED"


def test_invalid_data_cli_returns_error_not_traceback(tmp_path, capsys):
    assert main(["inspect", str(tmp_path / "missing.json")]) == 2
    captured = capsys.readouterr()
    assert "ERROR:" in captured.err
    assert "Traceback" not in captured.err


def test_native_preparation_commands_are_discoverable_without_model_imports():
    script = """
import sys
from distillschedule.cli import main
try:
    main(['data', '--help'])
except SystemExit as exc:
    assert exc.code == 0
assert not any(name in sys.modules for name in ('torch', 'diffusers', 'lightx2v_train', 'peft'))
"""
    completed = subprocess.run([sys.executable, "-c", script], text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    assert "prepare-helios-stream" in completed.stdout
    assert "prepare-helios-gan-reference" in completed.stdout
    assert "prepare-wan-reference" in completed.stdout
    assert "prepare-h3-cached-reference" in completed.stdout
    assert "prepare-h3-grid" in completed.stdout
    assert "prepare-livewan-inputs" in completed.stdout


def test_original_grid_inventory_cli_does_not_decode_or_change_native_data(tmp_path, cached_data, capsys):
    import torch
    # The ordinary helper fixture uses tiny Tensor files. This command hashes
    # supplied data; semantic native tensor admission is still worker-owned.
    config, native, _ = cached_data
    output = tmp_path / "inventory.json"
    assert main(["prepare-h3-grid", "--native-config", str(config), "--output", str(output)]) == 0
    row = json.loads(capsys.readouterr().out)
    assert row == json.loads(output.read_text())
    assert row["schema_version"] == "original_h3_grid_latent_dataset.v1"
    assert len(row["rows"]) == 288 and row["cycles"] == native["training"]["max_train_iters"]
    assert torch.cuda.is_initialized() is False


def test_native_preparation_missing_input_is_a_normal_cli_error(tmp_path, capsys):
    assert main(["prepare-h3-grid", "--native-config", str(tmp_path / "missing.json"),
                 "--output", str(tmp_path / "output.json")]) == 2
    assert "ERROR:" in capsys.readouterr().err
    assert not (tmp_path / "output.json").exists()


@pytest.mark.parametrize('explicit_preparer', [False, True])
def test_explicit_native_preparer_runs_through_same_data_frontdoor(tmp_path, explicit_preparer):
    from PIL import Image
    Image.new('RGB', (4, 4), (1, 2, 3)).save(tmp_path / 'image.png')
    diagnostics = '''import logging
import sys
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.propagate = False
def diagnostic(stage):
    print(stage + " print")
    logger.warning(stage + " log")
    logging.warning(stage + " root log")
    logging.getLogger("preconfigured.child").warning(stage + " named log")
diagnostic("import")
'''
    (tmp_path / 'encoder.py').write_text(diagnostics + '''
def encode(record, parameters):
    diagnostic("encoder")
    print("encoder stderr", file=sys.stderr)
    return {"pixels": record["image"]}
''')
    source = tmp_path / 'raw.json'
    source.write_text(json.dumps({'schema_version': 'distillschedule.data.raw.v1',
        'encoder': {'factory': 'encoder.py:encode'}, 'records': [{'id': 'a', 'image': 'image.png'}]}))
    plugin = tmp_path / 'preparer.py'
    plugin.write_text(diagnostics + '''
from distillschedule.data import prepare
def prepare_native(source, output):
    diagnostic("preparer")
    return prepare(source, output)
''')
    args = ['data', 'prepare', str(source), '--output', str(tmp_path / 'prepared')]
    if explicit_preparer:
        args += ['--preparer', str(plugin) + ':prepare_native']
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
assert "encoder named log" in unrelated.getvalue()
raise SystemExit(code)
'''
    completed = subprocess.run([sys.executable, '-c', script, *args], text=True, capture_output=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(completed.stdout)
    stages = ['import', 'encoder', *(['preparer'] if explicit_preparer else [])]
    assert all(stage + ' ' + kind in completed.stderr for stage in stages
               for kind in ('print', 'log', 'root log', 'named log'))
    assert 'encoder stderr' in completed.stderr
    if explicit_preparer:
        assert result['preparer_source']['sha256']
    from distillschedule.data import PreparedDataset
    assert PreparedDataset(tmp_path / 'prepared/manifest.json').sample(0)['pixels'].shape == (4, 4, 3)


@pytest.mark.parametrize('stage', ['import', 'call'])
@pytest.mark.parametrize('error', ['RuntimeError', 'KeyboardInterrupt'])
def test_preparer_failure_restores_streams_without_swallowing_interrupt(tmp_path, stage, error):
    plugin = tmp_path / 'broken.py'
    diagnostic = ('print("preparer diagnostic"); logging.warning("preparer root log"); '
                  'logging.getLogger("preconfigured.child").warning("preparer named log"); '
                  'raise ' + error + '("preparer failed")')
    plugin.write_text('import logging\n' + (diagnostic + '\n' if stage == 'import' else
                      'def prepare(source, output):\n    ' + diagnostic + '\n'))
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
    class NoRestoreFlush(logging.StreamHandler):
        def flush(self):
            assert self.stream is stdout, "restoration must not flush the diagnostic stream"
    restore_guard = NoRestoreFlush(stdout)
    restore_guard.setLevel(logging.CRITICAL)  # The plugin emits only warnings.
    logger.addHandler(restore_guard)
    code = main(sys.argv[1:])
except KeyboardInterrupt as exc:
    assert {error!r} == "KeyboardInterrupt" and str(exc) == "preparer failed"
else:
    assert {error!r} == "RuntimeError" and code == 2
assert sys.stdout is stdout and sys.stderr is stderr and handler.stream is stdout
assert named_handler.stream is stdout and unrelated_handler.stream is unrelated
assert restore_guard.stream is stdout
assert unrelated.getvalue() == "preparer named log\\n" and not unrelated.closed
print("STREAMS_RESTORED")
'''
    output = tmp_path / 'prepared'
    completed = subprocess.run([sys.executable, '-c', script, 'data', 'prepare',
        str(tmp_path / 'unused.json'), '--output', str(output), '--preparer', str(plugin) + ':prepare'],
        text=True, capture_output=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stdout == 'STREAMS_RESTORED\n'
    assert 'preparer diagnostic' in completed.stderr
    assert 'preparer root log' in completed.stderr and 'preparer named log' in completed.stderr
    assert ('ERROR: preparer failed' in completed.stderr) is (error == 'RuntimeError')
    assert not output.exists()


@pytest.mark.parametrize('error', ['RuntimeError', 'KeyboardInterrupt'])
def test_handler_flush_failure_on_entry_restores_already_redirected_handlers(tmp_path, error):
    plugin = tmp_path / 'unreachable.py'
    plugin.write_text('raise AssertionError("preparer must not run after redirect entry fails")\n')
    script = f'''import logging
import sys
from distillschedule.cli import main
stdout, stderr = sys.stdout, sys.stderr
first = logging.StreamHandler(stdout)
logging.getLogger().addHandler(first)
class BrokenFlush(logging.StreamHandler):
    def flush(self):
        raise {error}("handler flush failed")
second = BrokenFlush(stdout)
logger = logging.getLogger("preconfigured.child")
logger.addHandler(second)
try:
    code = main(sys.argv[1:])
except KeyboardInterrupt as exc:
    assert {error!r} == "KeyboardInterrupt" and str(exc) == "handler flush failed"
else:
    assert {error!r} == "RuntimeError" and code == 2
assert sys.stdout is stdout and sys.stderr is stderr
assert first.stream is stdout and second.stream is stdout
# Do not let logging's interpreter-shutdown flush obscure the test outcome.
second.flush = lambda: None
print("STREAMS_RESTORED")
'''
    completed = subprocess.run([sys.executable, '-c', script, 'data', 'prepare',
        str(tmp_path / 'unused.json'), '--output', str(tmp_path / 'output'),
        '--preparer', str(plugin) + ':prepare'], text=True, capture_output=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stdout == 'STREAMS_RESTORED\n'
    assert ('ERROR: handler flush failed' in completed.stderr) is (error == 'RuntimeError')
    assert 'preparer must not run' not in completed.stderr
