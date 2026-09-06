"""C33 source-guarded no-grad canary, independent of model/asset imports."""
import inspect
import json
from pathlib import Path
import subprocess
import sys
import pytest

from distillschedule.runtime.livewan_generator_gate import build, canary_scope


def test_default_does_nothing():
    with canary_scope(None, None, None):
        pass


def test_grad_rejected_before_import_or_device_access():
    class Torch:
        @staticmethod
        def is_grad_enabled(): return True
    with pytest.raises(RuntimeError, match='requires no_grad'):
        with canary_scope(Torch, None, None, enabled=True):
            pass


def test_exact_rounding_strides_and_no_cpu_sync():
    s = inspect.getsource(build)
    assert s.count('mul.rn.f32') == s.count('add.rn.f32') == 1
    assert 'enable_fp_fusion=False' in s and 'torch.empty_like' in s
    for token in ('tl.sum', '.cpu(', '.item(', '.tolist(', 'synchronize(', 'fma.rn'):
        assert token not in s
    for token in ('XS=x.stride(1)', 'YS=y.stride(1)', 'OS=out.stride(1)'):
        assert token in s
    s = inspect.getsource(canary_scope)
    assert 'finally:' in s and 'bc._gate = original' in s


def test_plan_only_no_assets_or_cuda(tmp_path):
    root = Path(__file__).resolve().parents[1]
    out = tmp_path/'unused.json'
    p = subprocess.run([sys.executable,
        str(root/'scripts/distillschedule/canary_livewan_generator_gate.py'),
        '--trace','/unread','--bootstrap','/unread','--output',str(out),'--plan-only'],
        text=True,capture_output=True,check=True,timeout=20)
    v = json.loads(p.stdout)
    assert v['status'] == v['world16'] == 'NOT_RUN'
    assert v['parent'] == 'C32' and not v['production_installed']
    assert 'BITWISE' in v['correctness'] and not out.exists()


def test_no_production_install_yet():
    root = Path(__file__).resolve().parents[1]
    worker = (root/'src/distillschedule/runtime/livewan_crossscale_worker.py').read_text()
    assert 'livewan_generator_gate' not in worker
