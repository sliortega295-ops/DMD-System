"""The checkout command and installed command resolve the same package roots."""
import os
from pathlib import Path
import subprocess
import sys


def test_checkout_entrypoint_outside_checkout_keeps_discovery_model_free(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = '''
import importlib.util, runpy, sys
from pathlib import Path
root = Path(sys.argv[1])
sys.argv = [str(root/'scripts/distillctl.py'), 'doctor']
try:
    runpy.run_path(sys.argv[0], run_name='__main__')
except SystemExit as exc:
    assert exc.code == 0
import distillschedule
assert Path(distillschedule.__file__).resolve() == root/'src/distillschedule/__init__.py'
assert Path(importlib.util.find_spec('runtime').origin).resolve() == root/'runtime/__init__.py'
assert 'torch' not in sys.modules
print('PASS_CHECKOUT_ENTRYPOINT_NO_MODEL_IMPORT')
'''
    result = subprocess.run([sys.executable, "-c", source, str(root)], cwd=tmp_path,
                            env={**os.environ, "PYTHONPATH": str(root / "src"), "CUDA_VISIBLE_DEVICES": ""},
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS_CHECKOUT_ENTRYPOINT_NO_MODEL_IMPORT" in result.stdout
