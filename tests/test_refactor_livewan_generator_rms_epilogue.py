import ast
import inspect
import json
from pathlib import Path
import subprocess
import sys

import pytest

from distillschedule.runtime.livewan_generator_rms_epilogue import (
    model_epilogue_canary_scope, validate_inputs, build_nograd_generator_rms_epilogue)


def test_default_scope_does_not_touch_models_or_torch():
    with model_epilogue_canary_scope(None, None, None):
        pass


def test_reject_grad_before_device_access():
    class Torch:
        @staticmethod
        def is_grad_enabled(): return True
    with pytest.raises(RuntimeError, match='requires no_grad'):
        validate_inputs(Torch, None, None, 1e-6)
    with pytest.raises(RuntimeError, match='requires no_grad'):
        with model_epilogue_canary_scope(Torch, None, None, enabled=True):
            pass


def test_no_production_integration_and_preserve_native_reduction():
    from distillschedule.runtime import livewan_crossscale_worker
    assert 'livewan_generator_rms_epilogue' not in inspect.getsource(livewan_crossscale_worker)
    source = inspect.getsource(build_nograd_generator_rms_epilogue)
    tree = ast.parse(source)
    assert 'x.float().pow(2).mean(dim=-1, keepdim=True)' in ast.unparse(tree)
    assert 'tl.sum' not in source
    assert 'enable_fp_fusion=False' in source
    for token in ('synchronize(', '.cpu(', '.item(', '.tolist(', '.step('):
        assert token not in source


def test_plan_only_has_no_torch_or_asset_dependency(tmp_path):
    root=Path(__file__).resolve().parents[1]
    r=subprocess.run([sys.executable,str(root/'scripts/distillschedule/canary_livewan_generator_rms_epilogue.py'),
        '--trace','/unread','--bootstrap','/unread','--output',str(tmp_path/'out.json'),'--plan-only'],
        check=True,text=True,capture_output=True)
    assert json.loads(r.stdout)['status']=='NOT_RUN'
    assert not (tmp_path/'out.json').exists()
