"""Bounded capacity bracket after tail8 OOM; no historical policy widening."""
from dataclasses import replace
import pytest
from distillschedule.strategies.wan_strategy import WanStrategy
from distillschedule.search.wan_candidate_bundle import INCUMBENT


def control():
    return WanStrategy('resident', 2048, True, 5, INCUMBENT, retained_critic_blocks=(26,27,28,29))


def test_tail6_roundtrip_and_original_native_backend_unchanged():
    base=control()
    child=replace(base,retained_critic_blocks=tuple(range(24,30)),retention_policy='tail6.v1')
    assert WanStrategy.from_dict(child.to_dict())==child
    assert child.id.endswith('-retainC24_25_26_27_28_29-tail6v1')
    assert child.backend==base.backend
    assert child.environment()['DISTILLSCHEDULE_WAN13B_RETAIN_CRITIC_BLOCKS']=='24,25,26,27,28,29'
    assert 'retention_policy' not in base.to_dict()
    tail8=replace(base,retained_critic_blocks=tuple(range(22,30)),retention_policy='tail8.v1')
    assert tail8.id.endswith('-tail8v1')


@pytest.mark.parametrize('blocks',[(),(29,),tuple(range(25,30)),tuple(range(22,30)),tuple(range(23,29))])
def test_tail6_only_exact_scope_and_legacy_rejects_more_than_four(blocks):
    with pytest.raises(Exception):replace(control(),retained_critic_blocks=blocks,retention_policy='tail6.v1')
    with pytest.raises(Exception):replace(control(),retained_critic_blocks=tuple(range(24,30)))


def test_tail6_not_hidden_from_manual_only_calibration_gate(monkeypatch,tmp_path):
    from distillschedule.strategies.wan_worker import run_worker,OriginalWanAssets
    from distillschedule.strategies.wan_task import OriginalWanTask
    # Window1 is mandatory for calibrate; reach the retained-policy rejection
    # rather than the earlier generic calibration-window rejection.
    child=replace(control(),rollout_window=1,retained_critic_blocks=tuple(range(24,30)),retention_policy='tail6.v1')
    for k,v in {'WORLD_SIZE':'16','RANK':'0','LOCAL_RANK':'0','LOCAL_WORLD_SIZE':'8'}.items():monkeypatch.setenv(k,v)
    assets=OriginalWanAssets('/dummy/model','/dummy/bootstrap','a'*64,'/dummy/trace','b'*64,'/dummy/negative')
    with pytest.raises(Exception,match='explicit manual candidate'):
        run_worker(OriginalWanTask(),assets,output=tmp_path/'no_cuda',mode='calibrate',strategy=child)
    assert not (tmp_path/'no_cuda').exists()


def test_actual_nonreentrant_checkpoint_dropout_output_grad_adam_rng_equal():
    import copy
    import torch
    from torch import nn
    from distillschedule.runtime.fsdp2_static import fsdp2_metadata_tolerant_checkpoint
    from distillschedule.runtime.checkpoint_block_retention import install

    class Block(nn.Module):
        def __init__(self):
            super().__init__(); self.proj=nn.Linear(4,4); self.drop=nn.Dropout(.1)
        def forward(self,x):
            return x + .05*torch.tanh(self.proj(self.drop(x)))
    class Model(nn.Module):
        def __init__(self):
            super().__init__(); self.blocks=nn.ModuleList([Block() for _ in range(30)])
            self.gradient_checkpointing=True; self._gradient_checkpointing_func=fsdp2_metadata_tolerant_checkpoint
        def forward(self,x):
            for block in self.blocks:x=self._gradient_checkpointing_func(block,x)
            return x
    torch.manual_seed(41);a=Model();b=copy.deepcopy(a)
    receipts=[install(m,role='critic',indices=indices,expected_block_count=30)
              for m,indices in ((a,(26,27,28,29)),(b,tuple(range(24,30))))]
    def run(m):
        torch.manual_seed(42);x=torch.randn(3,4,requires_grad=True)
        opt=torch.optim.AdamW(m.parameters(),lr=1e-4)
        y=m(x);y.square().sum().backward()
        grads=[p.grad.clone() for p in m.parameters()];opt.step()
        return y,x.grad,grads,copy.deepcopy(m.state_dict()),copy.deepcopy(opt.state_dict()),torch.get_rng_state()
    left,right=run(a),run(b)
    def equal(x,y):
        if isinstance(x,torch.Tensor):assert torch.equal(x,y)
        elif isinstance(x,dict):
            assert x.keys()==y.keys()
            for k in x:equal(x[k],y[k])
        elif isinstance(x,(list,tuple)):
            assert len(x)==len(y)
            for p,q in zip(x,y):equal(p,q)
        else:assert x==y
    equal(left,right)
    assert [r['retained_forward_calls'] for r in receipts]==[4,6]
    assert not torch.cuda.is_initialized()
