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


def test_preserve_native_reduction():
    source = inspect.getsource(build_nograd_generator_rms_epilogue)
    tree = ast.parse(source)
    assert 'x.float().pow(2).mean(dim=-1, keepdim=True)' in ast.unparse(tree)
    assert 'tl.sum' not in source
    assert 'enable_fp_fusion=False' in source
    for token in ('synchronize(', '.cpu(', '.item(', '.tolist(', '.step('):
        assert token not in source


def test_plan_only_has_no_torch_or_asset_dependency(tmp_path):
    root=Path(__file__).resolve().parents[1]
    r=subprocess.run([sys.executable,str(root/'benchmarks/kernels/canary_livewan_generator_rms_epilogue.py'),
        '--trace','/unread','--bootstrap','/unread','--output',str(tmp_path/'out.json'),'--plan-only'],
        check=True,text=True,capture_output=True)
    assert json.loads(r.stdout)['status']=='NOT_RUN'
    assert not (tmp_path/'out.json').exists()


def test_runtime_default_off_and_actual_receipt_gate():
    from types import SimpleNamespace
    from distillschedule.runtime.livewan_generator_rms_epilogue import (
        model_runtime_scope, runtime_receipt_errors, PLAN_CONTRACT)
    with model_runtime_scope(None, SimpleNamespace()): pass
    r=dict(PLAN_CONTRACT, execution_status='PASS', scope_entries=20, scope_exits=20,
        scope_failures=0, contract_failures=0, active=False,
        scope_forward_calls=[1920]*20, calls=1920*20)
    assert not runtime_receipt_errors(r, rank=0, cycles=2)
    for key,value in [('scope_entries',19),('scope_exits',19),('scope_failures',1),
                      ('contract_failures',1),('active',True),('output_dtype','bfloat16'),
                      ('calls',1),('scope_forward_calls',[1900]*20),('grad_path_changed',True)]:
        assert runtime_receipt_errors(dict(r,**{key:value}),rank=0,cycles=2)
    assert runtime_receipt_errors(r,rank=8,cycles=2)
    assert not runtime_receipt_errors({'execution_status':'NOT_APPLICABLE'},rank=8,cycles=2)
    from distillschedule.search.livewan_request import LiveWanRequest
    new = dict(r, scope_forward_calls=[1470]*20, calls=1470*20)
    assert runtime_receipt_errors(new,rank=0,cycles=2)
    assert not runtime_receipt_errors(new,rank=0,cycles=2,request=LiveWanRequest(frames=69))
    assert runtime_receipt_errors(dict(r,scope_forward_calls=[2820]*20,calls=2820*20),
                                  rank=0,cycles=2,request=LiveWanRequest(frames=69))


def test_c29_plan_changes_only_epilogue(monkeypatch):
    import copy
    from distillschedule.core.config import load_plan
    from distillschedule.runtime.livewan_generator_rms_epilogue import PLAN_CONTRACT
    from distillschedule.runtime.livewan_crossscale_worker import _generator_rms_epilogue_enabled
    root=Path(__file__).resolve().parents[1]/'configs/distillschedule/plans'
    parent=root/'livewan_wan13b_14b_sf_dmd_c28_continuous_b16_rollout_cast_cache.json'
    candidate=root/'livewan_wan13b_14b_sf_dmd_c29_continuous_b16_generator_rms_epilogue.json'
    a,b=json.loads(parent.read_text()),json.loads(candidate.read_text())
    assert b['metadata'].pop('generator_rms_epilogue')==PLAN_CONTRACT
    assert a==b
    for key in ('LIVEWAN_STUDENT_CHECKPOINT','LIVEWAN_CRITIC_CHECKPOINT','LIVEWAN_TEACHER_CHECKPOINT',
                'LIVEWAN_OPERATION_TRACE','LIVEWAN_SYSTEMS_BOOTSTRAP','DISTILLSCHEDULE_OUTPUT_DIR'):
        monkeypatch.setenv(key,'/unread/'+key)
    plan=load_plan(candidate).plan.to_dict()
    assert _generator_rms_epilogue_enabled(plan,mode='pipeline')
    assert not _generator_rms_epilogue_enabled(load_plan(parent).plan.to_dict(),mode='pipeline')
    with pytest.raises(RuntimeError): _generator_rms_epilogue_enabled(plan,mode='baseline')
    for key,value in [('native_pow_mean_preserved',False),('output_dtype','bfloat16'),
                      ('grad_path_changed',True),('input_dtype','float32')]:
        d=copy.deepcopy(plan);d['metadata']['generator_rms_epilogue'][key]=value
        with pytest.raises(RuntimeError): _generator_rms_epilogue_enabled(d,mode='pipeline')


def test_cpu_temporary_scope_restores_grad_math_even_on_exception(monkeypatch):
    torch=pytest.importorskip('torch')
    # Upstream's unrelated T5 class evaluates current_device() as a default
    # argument during package import. This CPU test constructs no T5/CUDA model.
    with monkeypatch.context() as metadata_import:
        metadata_import.setattr(torch.cuda, 'current_device', lambda: 0)
        model=pytest.importorskip('wan.modules.model')
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norms=torch.nn.ModuleList([model.WanRMSNorm(1536,eps=1e-6) for _ in range(4)])
    class Student(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks=torch.nn.ModuleList([Block() for _ in range(30)])
    student=Student();m=student.blocks[0].norms[0]
    x=torch.randn(1,2,1536).bfloat16()
    calls=[]
    def original_op(x,w,eps):
        calls.append(1)
        xf=x.float()
        return (xf*torch.rsqrt(xf.pow(2).mean(-1,keepdim=True)+eps)).type_as(x)*w
    m(x).sum().backward();before=m.weight.grad.clone();m.weight.grad=None
    with torch.no_grad():
        ref=m(x)
        with model_epilogue_canary_scope(torch,student,original_op,enabled=True):
            assert torch.equal(m(x),ref)
        assert calls==[1]
        with pytest.raises(ValueError,match='intentional'):
            with model_epilogue_canary_scope(torch,student,original_op,enabled=True):
                raise ValueError('intentional')
    assert all('forward' not in n.__dict__ for n in student.modules() if type(n) is model.WanRMSNorm)
    m(x).sum().backward();assert torch.equal(m.weight.grad,before)


def test_only_prepare_rollout_is_scoped():
    from distillschedule.runtime.livewan_crossscale_worker import prepare_rollout,generator_backward_for_sample
    tree=ast.parse(inspect.getsource(prepare_rollout))
    scopes=[n for n in ast.walk(tree) if isinstance(n,ast.With)]
    assert len(scopes)==1
    assert ast.unparse(scopes[0].items[0].context_expr)=='torch.no_grad()'
    assert ast.unparse(scopes[0].items[2].context_expr)=='generator_rms_scope(torch, student)'
    assert 'generator_rms_scope' not in inspect.getsource(generator_backward_for_sample)
