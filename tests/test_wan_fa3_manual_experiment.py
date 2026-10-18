"""SYNTHETIC_TEST_ONLY; real CUDA operator evidence is in R24, not here."""
from types import SimpleNamespace
import pytest
import torch
import torch.nn.functional as F
from distillschedule.runtime.wan13b_fa3_manual_experiment import LargeSelfAttention,dispatch_scope,explicit_worker

def receipt():return dict(fa3_calls=0,fa3_grad_calls=0,fa3_nograd_calls=0,native_fallback_calls=0,fa3_lengths={})

@pytest.mark.parametrize('option',[{},dict(dropout=0.1),dict(causal=True),dict(gqa=True),dict(scale=1.)])
def test_cpu_and_unreviewed_inputs_never_enter_FA3(option):
    x=torch.zeros((1,12,2160,128),dtype=torch.bfloat16)
    args=dict(mask=None,dropout=0.,causal=False,scale=None,gqa=False);args.update(option)
    assert not LargeSelfAttention.eligible(x,x,x,**args)

def test_scoped_dispatch_real_math_gradients_rng_and_exception_restore(monkeypatch):
    native=F.scaled_dot_product_attention;record=receipt()
    class Backend:
        def flash_attn_func(self,q,k,v,**kwargs):
            assert kwargs==dict(softmax_scale=None,causal=False,deterministic=False,return_attn_probs=False)
            return native(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2)).transpose(1,2)
    g=torch.Generator().manual_seed(128)
    xs=[torch.randn((1,3,7,4),generator=g,dtype=torch.float64,requires_grad=True)for _ in range(3)]
    ref=native(*xs);grads=torch.autograd.grad(ref.square().sum(),xs)
    before=torch.random.get_rng_state().clone()
    monkeypatch.setattr(LargeSelfAttention,'eligible',staticmethod(lambda *a:True))
    with pytest.raises(RuntimeError,match='test-only'):
        with dispatch_scope(Backend(),record):
            y=F.scaled_dot_product_attention(*xs);assert torch.equal(y,ref)
            got=torch.autograd.grad(y.square().sum(),xs)
            assert all(torch.equal(a,b)for a,b in zip(got,grads))
            with torch.no_grad():F.scaled_dot_product_attention(*xs)
            raise RuntimeError('test-only')
    assert F.scaled_dot_product_attention is native and torch.equal(before,torch.random.get_rng_state())
    assert record['fa3_calls']==2 and record['fa3_grad_calls']==1 and record['fa3_nograd_calls']==1

def test_native_control_build_does_not_import_kernel_or_patch_SDPA():
    native=F.scaled_dot_product_attention
    class Session:
        def __init__(self,**kwargs):self.build={role:dict(metadata={})for role in('generator','critic')}
    worker=SimpleNamespace(OriginalWanSession=Session)
    packet=dict(source_head='a'*40,arm='native',kernel_files={'SYNTHETIC':'only'})
    with explicit_worker(worker,packet,'test-sha'):
        s=worker.OriginalWanSession()
        assert not s.build['critic']['metadata']['self_attention_backend_experiment']['enabled']
        assert F.scaled_dot_product_attention is native
    assert worker.OriginalWanSession is Session and not torch.cuda.is_initialized()

def test_shape_guard_supports_only_probed_large_geometry():
    class Fake:
        ndim=4;shape=(1,12,2160,128);is_cuda=True;dtype=torch.bfloat16
        def stride(self,i):return 1
    x=Fake();assert LargeSelfAttention.eligible(x,x,x,None,0.,False,None,False)
    for shape in((1,12,540,128),(1,12,1000,128),(2,12,2160,128),(1,8,2160,128)):
        x=Fake();x.shape=shape;assert not LargeSelfAttention.eligible(x,x,x,None,0.,False,None,False)


def test_packet_binds_exact_source_strategy_assets_and_kernel(tmp_path):
    import json
    from distillschedule.runtime.wan13b_fa3_manual_experiment import SCHEMA,sha,validate_packet
    from distillschedule.strategies.wan_strategy import WanStrategy
    from distillschedule.strategies.wan_task import OriginalWanTask
    from distillschedule.search.wan_candidate_bundle import INCUMBENT
    task,assets,impl=(tmp_path/n for n in('task','assets','impl'))
    task.write_text(json.dumps(OriginalWanTask().to_dict()));assets.write_text('{}')
    impl.write_text(json.dumps(WanStrategy('resident',2048,True,5,candidate_bundle=INCUMBENT,
        retained_critic_blocks=(26,27,28,29)).to_dict()))
    pins={}
    for name in('kernel-one','kernel-two','kernel-three'):
        p=tmp_path/name;p.write_text('SYNTHETIC_TEST_ONLY');pins[str(p)]=sha(p)
    packet=dict(schema_version=SCHEMA,source_head='a'*40,arm='fa3-large-v1',run_id='synthetic-test-only',
        cycles=2,task_sha256=sha(task),assets_sha256=sha(assets),strategy_sha256=sha(impl),
        kernel_path=str(tmp_path),kernel_files=pins,kernel_import_root=str(tmp_path))
    kw=dict(head='a'*40,task=task,assets=assets,implementation=impl)
    validate_packet(packet,**kw)
    for key,val in(('source_head','b'*40),('cycles',1),('arm','all-shapes'),('assets_sha256','x')):
        with pytest.raises(Exception):validate_packet(dict(packet,**{key:val}),**kw)
    (tmp_path/'kernel-two').write_text('DRIFT')
    with pytest.raises(Exception):validate_packet(packet,**kw)


def test_child_loader_after_selected_device_before_constructor_no_persistent_overlay(monkeypatch):
    from distillschedule.runtime import wan13b_fa3_manual_experiment as experiment
    native=F.scaled_dot_product_attention;events=[]
    backend=SimpleNamespace(__file__='/synthetic/backend.py')
    def load(packet):events.append('load');return backend
    monkeypatch.setattr(experiment,'load_backend',load)
    monkeypatch.setattr(torch.cuda,'current_device',lambda:3)
    class Session:
        def __init__(self,**kw):
            events.append('original-construction')
            assert isinstance(F.scaled_dot_product_attention,LargeSelfAttention)
            self.build={r:dict(metadata={})for r in('generator','critic')}
    worker=SimpleNamespace(OriginalWanSession=Session)
    packet=dict(source_head='a'*40,arm='fa3-large-v1',kernel_files={})
    with explicit_worker(worker,packet,'test-sha'):
        s=worker.OriginalWanSession(device=SimpleNamespace(index=3))
        assert s.build['critic']['metadata']['self_attention_backend_experiment']['enabled']
    assert events==['load','original-construction']and F.scaled_dot_product_attention is native
    assert worker.OriginalWanSession is Session and not torch.cuda.is_initialized()
