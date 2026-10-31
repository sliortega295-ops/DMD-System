"""CPU mechanism/contract only; GPU primitive and native C2 remain required."""
from pathlib import Path
import importlib
import hashlib
import os
import sys
import pytest
import torch
from torch.utils.checkpoint import checkpoint
from distillschedule.strategies.h3_ar_task import H3ARImplementation
from distillschedule.strategies.h3_ar_hybrid_cache import pool_class

P=Path(os.environ.get('DMD_H3_AR_TEST_NATIVE_HELPER_ROOT','/home/lyy/Helios/research_artifacts/DMD-System/h3_diffsynth_dmd2_world16_20261010/zero3_200s_r2'))
ROOTS=[P/'selected_cache_h2d_prefetch_260s_r1/scripts',P/'selected_reuse_block_fifo_260s_r1/scripts',P/'selected_rollout_exact_reuse_260s_r1/scripts',P/'continuation_h3_experience_20261011_r1/scripts']

def parent_module():
 from distillschedule.strategies.h3_ar_assets import HELPER_PINS
 for name in ('native_selected_h2d_prefetch_r1','native_selected_block_fifo_r1','native_selected_rollout_reuse_r1','selected_attention_pool_r1','native_sdpa_cpu_replay_r1'):
  files=[root/(name+'.py') for root in ROOTS if (root/(name+'.py')).is_file()]
  assert len(files)==1 and hashlib.sha256(files[0].read_bytes()).hexdigest()==HELPER_PINS[name]
 for p in ROOTS:sys.path.insert(0,str(p))
 return importlib.import_module('native_selected_h2d_prefetch_r1')

def toy(hybrid):
 parent=parent_module();cls=pool_class(parent) if hybrid else parent.PrefetchPool
 pool=cls(enabled=False);pool.enabled=True;pool.cpu_test=True;pool.target=torch.ops.aten.mm.default;pool.blocks={i:i for i in range(50)}
 torch.manual_seed(59)
 w=[(torch.randn(4,4)*.1,torch.randn(4,4)*.1) for _ in range(50)]
 student=torch.nn.Parameter(torch.randn(2,4));fake=torch.nn.Parameter(torch.randn(2,4))
 optg=torch.optim.AdamW([student],lr=1e-3);optf=torch.optim.AdamW([fake],lr=1e-3)
 state=torch.get_rng_state().clone()
 def block(z,index):
  a,b=w[index];return z*.8+.1*(z@a).sin()+.1*(z@b).sin()
 # Exact native ordering: selected NG source, disjoint Fake graph/commit,
 # Student grad and replay with the same original Student version.
 x=student.detach()
 with torch.no_grad():
  for i in range(50):
   with pool.capture(('studentG0',i),uses=2):x=block(x,i)
 outputs=[];x=fake
 for i in range(50):
  x=checkpoint(lambda z,i=i:block(z,i),x,use_reentrant=False,
      context_fn=lambda i=i:pool.fake_contexts(('fakeG0',i),lambda:(__import__('contextlib').nullcontext(),__import__('contextlib').nullcontext())))
  outputs.append(x)
 lossf=sum(z.square().sum() for z in outputs);lossf.backward();gf=fake.grad.clone();optf.step()
 outputs=[];x=student
 for i in range(50):
  x=checkpoint(lambda z,i=i:block(z,i),x,use_reentrant=False,
      context_fn=lambda i=i:pool.contexts_from_source(('studentG0',i)))
  outputs.append(x)
 lossg=sum(z.square().sum() for z in outputs);lossg.backward();gg=student.grad.clone();optg.step()
 result=pool.finish_selected()
 return dict(parameters=(student.detach(),fake.detach()),gradients=(gg,gf),losses=(lossg.detach(),lossf.detach()),
             optimizer=(optg.state[student],optf.state[fake]),rng=torch.get_rng_state(),initial_rng=state,receipt=result)


def test_full_original_consumer_order_cpu_loss_grad_rng_adam_and_actual_retention():
 control,candidate=toy(False),toy(True)
 for key in ('parameters','gradients','losses'):
  assert all(torch.equal(a,b) for a,b in zip(control[key],candidate[key]))
 for before,after in zip(control['optimizer'],candidate['optimizer']):
  assert before.keys()==after.keys() and all(torch.equal(before[k],after[k]) for k in before)
 assert torch.equal(control['rng'],candidate['rng']) and torch.equal(candidate['rng'],candidate['initial_rng'])
 counts=candidate['receipt']['counters'];base=control['receipt']['counters']
 assert counts['GPU_source_calls']==4 and counts['GPU_consumer_calls']==6 and counts['GPU_release_calls']==4 and counts['GPU_live_bytes']==0
 assert counts['source_calls']==base['source_calls']==200 and counts['consumer_calls']==base['consumer_calls']==300
 assert counts['D2H_logical_bytes']<base['D2H_logical_bytes'] and counts['H2D_logical_bytes']<base['H2D_logical_bytes']
 assert counts['D2H_block_groups']==98 and counts['GPU_source_block_groups']==2
 assert not torch.cuda.is_initialized()


def test_opt_in_identity_and_cost_storage_no_default_change():
 from distillschedule.strategies.h3_ar_planning import component
 from distillschedule.strategies.h3_ar_task import H3ARTask
 old=H3ARImplementation('zero3-ap');new=H3ARImplementation('zero3-ap-gpu-tail1')
 assert old.optimized and not old.hybrid and old.to_dict()['attention_replay_cache_storage']=='cpu_staged'
 assert new.optimized and new.hybrid and new.to_dict()['GPU_retained_blocks']==[49]
 assert H3ARImplementation.from_dict(new.to_dict())==new
 assert component(H3ARTask(),old).digest!=component(H3ARTask(),new).digest
 for wrong in ('zero3-ap-gpu-tail2','zero3-ap-grid1000'):
  with pytest.raises(ValueError):H3ARImplementation(wrong)


def test_real_installation_factory_is_reachable_and_restored_on_success_failure():
 from types import SimpleNamespace
 from distillschedule.strategies.h3_ar_hybrid_cache import install
 parent=parent_module();original=parent.PrefetchPool;calls=[]
 def epoch_install(*args):
  calls.append(args);pool=parent.PrefetchPool(enabled=False);pool.enabled=True
  return pool,'real-view'
 assets=SimpleNamespace(helper=lambda name:parent)
 epoch=SimpleNamespace(install=epoch_install)
 e,pool,view=install(epoch,assets,'native','module','adapter','trainer')
 assert e is epoch and view=='real-view' and type(pool) is not original and parent.PrefetchPool is original
 assert calls==[('native','module','adapter','trainer')]
 def broken(*args):raise RuntimeError('synthetic-owner-failure')
 with pytest.raises(RuntimeError,match='synthetic-owner-failure'):
  install(SimpleNamespace(install=broken),assets,'native','module','adapter','trainer')
 assert parent.PrefetchPool is original


def test_collector_requires_actual_gpu_storage_and_full_consumer_release():
 from copy import deepcopy
 from distillschedule.strategies.h3_ar_hybrid_cache import validate_receipt
 # Deliberately SYNTHETIC_TEST_ONLY counter fixture, never native qualification.
 valid=dict(storage='hybrid_gpu_tail1_pinned_cpu',GPU_retained_blocks=[49],GPU_cache_no_value_conversion=True,
            GPU_cache_consumed_and_released=True,counters=dict(GPU_source_calls=4,GPU_consumer_calls=6,GPU_release_calls=4,
            GPU_live_bytes=0,GPU_live_unique_storage_bytes=0,GPU_peak_unique_storage_bytes=64,GPU_saved_bytes=128,GPU_avoided_H2D_bytes=192))
 validate_receipt(valid)
 for key,value in (('GPU_consumer_calls',5),('GPU_release_calls',3),('GPU_peak_unique_storage_bytes',0),('GPU_live_unique_storage_bytes',64)):
  bad=deepcopy(valid);bad['counters'][key]=value
  with pytest.raises(ValueError):validate_receipt(bad)
 bad=deepcopy(valid);bad['storage']='cpu_staged'
 with pytest.raises(ValueError):validate_receipt(bad)
