"""SYNTHETIC_TEST_ONLY manual routing; not a GPU qualification."""
from contextvars import ContextVar
from types import SimpleNamespace
import inspect
import pytest
import torch
from torch import nn
from distillschedule.strategies.h3_ar_manual_attention import (
    ManualNoGradAttention, eligible, tensor_token, validate_backend_root, POLICY_ID, _ROOT, _MAIN)

@pytest.mark.parametrize('grad,root,call,main,wanted',[
 (False,('student',False),None,(0,10),True),
 (False,('student',False),'SELECTED_NG',(0,10),False),
 (True,('student',True),'STUDENT_GRAD',(0,10),False),
 (True,None,None,(0,10),False),
 (False,('teacher',False),None,(0,10),False),
 (False,('fake_score',False),None,(0,10),False),
 (False,('student',False),None,None,False)])
def test_only_uncached_no_grad_student_main_attention_admitted(grad,root,call,main,wanted):
 assert eligible(grad_enabled=grad,root=root,call=call,main=main)is wanted

def test_backend_pin_failure_before_model_or_cuda(tmp_path):
 with pytest.raises(ValueError,match='PINNED'):validate_backend_root(tmp_path)
 assert not torch.cuda.is_initialized()

def test_new_manual_id_rejected_by_automatic_planner():
 from distillschedule.strategies.h3_ar_task import H3ARImplementation
 from distillschedule.strategies.h3_ar_manual_frozen_bank import validate_manual_policy
 with pytest.raises(ValueError):H3ARImplementation(POLICY_ID)
 for mode in('calibrate','run'):
  with pytest.raises(ValueError):validate_manual_policy(POLICY_ID,mode=mode,implementation=SimpleNamespace(id='zero3-ap'),plan_path=None)
 validate_manual_policy(POLICY_ID,mode='manual',implementation=SimpleNamespace(id='zero3-ap'),plan_path=None)

def test_actual_function_global_binding_preserves_grad_and_refiner_fallbacks():
 # Real CPU output/grad/RNG/Adam, no backend CUDA emulation or threshold change.
 import copy
 from collections import Counter
 import torch.utils.checkpoint as cp
 def reference(q,k,v,cu,scale):return torch.nn.functional.scaled_dot_product_attention(q,k,v,scale=scale)
 env=dict(_sdpa_varlen_attention=reference)
 exec('def forward(self,x,*,max_seqlen):\n return _sdpa_varlen_attention(x,x,x,self.cu,128**-.5)*self.weight\n',env)
 class Attention(nn.Module):
  def __init__(self):
   super().__init__();self.weight=nn.Parameter(torch.tensor(1.3,dtype=torch.float64));self.num_heads=56;self.head_dim=128;self.cu=torch.tensor([0,4,8],dtype=torch.int32)
 Attention.forward=env['forward']
 class Model(nn.Module):
  def __init__(self):super().__init__();self.blocks=nn.ModuleList([nn.Module()]);self.blocks[0].attn=Attention()
  def forward(self,x):return self.blocks[0].attn(x,max_seqlen=4)
 class Adapter(nn.Module):
  def __init__(self):super().__init__();self.dit=Model();self.layers=[('fixture',SimpleNamespace(role='student'))]
  def predict(self,x):return self.dit(x)
 torch.manual_seed(176);base=Adapter();x=torch.randn(1,56,8,128,dtype=torch.float64)
 def run(optin):
  adapter=copy.deepcopy(base)
  # A fresh function globals table avoids contaminating reference test instances.
  local=dict(_sdpa_varlen_attention=reference)
  exec('def forward(self,x,*,max_seqlen):\n return _sdpa_varlen_attention(x,x,x,self.cu,128**-.5)*self.weight\n',local)
  adapter.dit.blocks[0].attn.forward=local['forward'].__get__(adapter.dit.blocks[0].attn,Attention)
  if optin:
   p=ManualNoGradAttention.__new__(ManualNoGradAttention);p.bounds=[0,4,8];p.cu_token=None;p.counts=Counter();p.previous=Counter();p.call_context=ContextVar('fixture_call',default=None)
   p.backend=SimpleNamespace(flash_attn_varlen_func=lambda *a,**k:(_ for _ in()).throw(AssertionError('grad/refiner touched FA3')))
   p.install_dispatch(SimpleNamespace(_sdpa_varlen_attention=reference),adapter,torch)
  opt=torch.optim.AdamW(adapter.parameters(),lr=.001);torch.manual_seed(198)
  y=cp.checkpoint(adapter.predict,x.clone().requires_grad_(),use_reentrant=False);y.square().mean().backward()
  grad=adapter.dit.blocks[0].attn.weight.grad.clone();opt.step()
  return y.detach(),grad,adapter.dit.blocks[0].attn.weight.detach(),copy.deepcopy(opt.state_dict()),torch.random.get_rng_state()
 a,b=run(False),run(True)
 assert all(torch.equal(x,y)for x,y in zip(a[:3],b[:3]))and torch.equal(a[4],b[4])
 for x,y in zip(a[3]['state'].values(),b[3]['state'].values()):
  assert all(torch.equal(x[k],y[k])for k in x)
 assert _ROOT.get()is None and _MAIN.get()is None

def test_router_census_fails_closed_and_no_combined_bank_default():
 from collections import Counter
 p=ManualNoGradAttention.__new__(ManualNoGradAttention);p.counts=Counter(actual_FA3_varlen_calls=349);p.previous=Counter()
 with pytest.raises(RuntimeError,match='CENSUS'):p.finish_cycle(1)
 from distillschedule.strategies import h3_ar_session,h3_ar_worker
 assert 'manual_attention_policy' in inspect.signature(h3_ar_session.H3ARSession).parameters
 text=inspect.getsource(h3_ar_worker.run_worker)
 assert text.index('validate_backend_root(manual_backend_root)')<text.index('host_admission')
 assert 'manual_backend_root is None' in text
