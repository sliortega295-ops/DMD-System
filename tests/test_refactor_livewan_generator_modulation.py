"""C32 pure instrumentation/opt-in canary guard; no production activation."""
import inspect
import json
from pathlib import Path
import subprocess
import sys
import pytest

from distillschedule.runtime.livewan_generator_modulation import build, canary_scope


def test_default_does_nothing():
    with canary_scope(None, None, None): pass


def test_no_grad_required_before_import_or_device_access():
    class Torch:
        @staticmethod
        def is_grad_enabled(): return True
    with pytest.raises(RuntimeError, match='requires no_grad'):
        with canary_scope(Torch, None, None, enabled=True): pass


def test_exact_fp32_rounding_and_no_norm_replacement():
    s=inspect.getsource(build)
    assert s.count('add.rn.f32')==2 and 'mul.rn.f32' in s
    assert 'enable_fp_fusion=False' in s and 'torch.float32' in s
    for token in ('tl.sum', '.cpu(', '.item(', '.tolist(', 'synchronize(', '.backward(', 'fma.rn'):
        assert token not in s
    scope=inspect.getsource(canary_scope)
    assert 'normalized = norm(x)' in scope
    assert 'finally:' in scope and 'bc._modulate = original' in scope


def test_only_installed_on_explicit_validated_g_scope():
    root=Path(__file__).resolve().parents[1]
    worker=(root/'src/distillschedule/runtime/livewan_crossscale_worker.py').read_text()
    assert 'if _generator_modulation_enabled(plan, mode=mode) and rank < 8:' in worker
    assert 'generator_modulation_scope(torch, student)' in worker


def test_plan_only_no_assets_or_cuda(tmp_path):
    root=Path(__file__).resolve().parents[1]
    out=tmp_path/'unused.json'
    p=subprocess.run([sys.executable,str(root/'scripts/distillschedule/canary_livewan_generator_modulation.py'),
        '--trace','/unread','--bootstrap','/unread','--output',str(out),'--plan-only'],
        text=True,capture_output=True,check=True,timeout=20)
    v=json.loads(p.stdout)
    assert v['status']==v['world16']=='NOT_RUN'
    assert v['parent']=='C31' and not v['production_installed']
    assert 'BITWISE' in v['correctness'] and not out.exists()
