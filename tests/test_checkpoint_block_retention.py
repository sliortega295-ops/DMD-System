"""CPU exact differential contract; synthetic tensors are NOT GPU evidence."""
import copy
import torch
import pytest
from torch.utils.checkpoint import checkpoint


def models():
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.linear=torch.nn.Linear(4,4);self.calls=0
        def forward(self,x):
            self.calls+=1
            return torch.tanh(self.linear(x))
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.blocks=torch.nn.ModuleList([Block() for _ in range(4)])
            self.gradient_checkpointing=True
            self._gradient_checkpointing_func=lambda mod,*a,**kw:checkpoint(mod,*a,use_reentrant=False,preserve_rng_state=True,**kw)
        def forward(self,x):
            for b in self.blocks:x=self._gradient_checkpointing_func(b,x)
            return x
    torch.manual_seed(7);a=Model();return a,copy.deepcopy(a)


def test_default_off_keeps_original_dispatch_object():
    from distillschedule.runtime.checkpoint_block_retention import install
    a,_=models();original=a._gradient_checkpointing_func
    r=install(a,indices=(),role='critic',expected_block_count=4)
    assert not r['enabled'] and a._gradient_checkpointing_func is original


def test_original_forward_backward_adam_rng_and_selected_recompute():
    from distillschedule.runtime.checkpoint_block_retention import install
    a,b=models();r=install(b,indices=(2,3),role='critic',expected_block_count=4)
    x=torch.arange(12,dtype=torch.float32).view(3,4)/8
    torch.manual_seed(13);before=torch.get_rng_state()
    ya=a(x);ya.square().sum().backward();rng_a=torch.get_rng_state()
    torch.set_rng_state(before);yb=b(x);yb.square().sum().backward();rng_b=torch.get_rng_state()
    assert torch.equal(ya,yb) and torch.equal(rng_a,rng_b)
    for p,q in zip(a.parameters(),b.parameters()):assert torch.equal(p.grad,q.grad)
    for m in (a,b):torch.optim.Adam(m.parameters(),lr=1e-4).step()
    for p,q in zip(a.parameters(),b.parameters()):assert torch.equal(p,q)
    assert [z.calls for z in a.blocks]==[2,2,2,2]
    assert [z.calls for z in b.blocks]==[2,2,1,1]
    assert r['retained_forward_calls']==2 and r['checkpoint_forward_calls']==2
    assert r['block_forward_counts']=={'0':0,'1':0,'2':1,'3':1}


def test_scope_wrapper_preserved_on_direct_call():
    from distillschedule.runtime.checkpoint_block_retention import install
    a,_=models();events=[]
    def wrap(fn):
        def scoped(*args,**kw):events.append('enter');return fn(*args,**kw)
        return scoped
    a._distillschedule_lora_checkpoint_function_wrapper=wrap
    install(a,indices=(3,),role='critic',expected_block_count=4)
    a(torch.ones(2,4)).sum().backward()
    assert events==['enter']


@pytest.mark.parametrize('indices',[(4,),(-1,),(1,1),(2,1),(True,),[1]])
def test_invalid_selection_fails_before_dispatch(indices):
    from distillschedule.runtime.checkpoint_block_retention import install
    a,_=models()
    with pytest.raises((ValueError,TypeError)):install(a,indices=indices,role='critic',expected_block_count=4)


def test_repeated_install_and_unknown_module_fail_closed():
    from distillschedule.runtime.checkpoint_block_retention import install
    a,_=models();install(a,indices=(3,),role='critic',expected_block_count=4)
    with pytest.raises(ValueError):install(a,indices=(3,),role='critic',expected_block_count=4)
    with pytest.raises(ValueError):a._gradient_checkpointing_func(torch.nn.Identity(),torch.ones(2,4))


def test_typed_optin_roundtrip_and_default_unchanged():
    from distillschedule.strategies.wan_strategy import WanStrategy
    from distillschedule.search.wan_candidate_bundle import INCUMBENT
    a=WanStrategy('resident',2048,True,5,INCUMBENT)
    assert 'retained_critic_blocks' not in a.to_dict()
    b=WanStrategy('resident',2048,True,5,INCUMBENT,retained_critic_blocks=(26,27,28,29))
    assert WanStrategy.from_dict(b.to_dict())==b and b.id!=a.id
    assert b.environment()['DISTILLSCHEDULE_WAN13B_RETAIN_CRITIC_BLOCKS']=='26,27,28,29'
    with pytest.raises(Exception):WanStrategy('fsdp',512,False,retained_critic_blocks=(29,))


def test_uncalibrated_retention_is_rejected_by_calibration_before_cuda(monkeypatch,tmp_path):
    from distillschedule.strategies.wan_worker import run_worker,OriginalWanAssets
    from distillschedule.strategies.wan_task import OriginalWanTask
    from distillschedule.strategies.wan_strategy import WanStrategy
    from distillschedule.search.wan_candidate_bundle import INCUMBENT
    for key,value in {'WORLD_SIZE':'16','RANK':'0','LOCAL_RANK':'0','LOCAL_WORLD_SIZE':'8'}.items():monkeypatch.setenv(key,value)
    task=OriginalWanTask();strategy=WanStrategy('resident',2048,True,1,INCUMBENT,retained_critic_blocks=(29,))
    assets=OriginalWanAssets('/dummy/model','/dummy/bootstrap','a'*64,'/dummy/trace','b'*64,'/dummy/negative')
    with pytest.raises(Exception,match='explicit manual candidate'):
        run_worker(task,assets,output=tmp_path/'never-created',mode='calibrate',strategy=strategy)
    assert not (tmp_path/'never-created').exists()
