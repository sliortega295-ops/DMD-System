"""CPU mechanism/identity tests; not native GPU correctness or performance."""
from dataclasses import replace
import pytest
import torch
from torch.utils.checkpoint import checkpoint,create_selective_checkpoint_contexts,CheckpointPolicy


def test_versioned_bundle_identity_does_not_unlock_old_tuple():
 from distillschedule.search.helios_model_adapter import gan_manual_incumbent
 from distillschedule.search.helios_gan_bundle import HeliosGANBundle,GAN_BUNDLE_ENV
 from distillschedule.search.helios_implementation import HeliosImplementation,current_implementation,MODE_ENV
 old=gan_manual_incumbent();new=replace(old,candidate_bundle=HeliosGANBundle.with_up_retention())
 assert 'up2' not in old.id and 'up2' in new.id
 assert HeliosImplementation.parse(new.id)==new
 assert HeliosImplementation.from_assignment(new.assignment)==new
 env=new.environment();env[MODE_ENV]='native';assert current_implementation(env)==new
 assert env[GAN_BUNDLE_ENV]==new.candidate_bundle.id
 assert old.lower_profile()['contract']==new.lower_profile()['contract']
 with pytest.raises(ValueError):HeliosGANBundle(True,up_blocks=(0,1))


def test_frozen_up_operand_contract_and_no_trainable_cache():
 from distillschedule.runtime.helios_gan_manual.down_contract import ScopeKey,TensorMeta,Leaf,select_frozen_up
 key=ScopeKey(rank=3,cycle=1,index=0,logical_step=1000,version=10,graph='B1_DENOISE',block=38)
 device='cuda:3';rows=11040
 def m(shape,stride,sid,grad=False):return TensorMeta(shape,stride,'torch.bfloat16',device,grad,sid,0)
 weight=m((13824,5120),(5120,1),10);bias=m((13824,),(1,),11);x=m((1,rows,5120),(rows*5120,5120,1),12,True)
 leaf=Leaf('critic.blocks.38.ffn.net.0.proj.base_layer',x,weight,bias)
 args=(bias,m((rows,5120),(5120,1),12,True),m((5120,13824),(1,5120),10))
 assert select_frozen_up(key,leaf,args,{},rows=rows)
 for bad in [replace(leaf,module='critic.blocks.38.ffn.net.2.base_layer'),replace(leaf,weight=replace(weight,requires_grad=True))]:
  with pytest.raises(ValueError):select_frozen_up(key,bad,args,{},rows=rows)
 with pytest.raises(ValueError):select_frozen_up(key,leaf,(bias,args[1],replace(args[2],storage_id=77)),{},rows=rows)


def test_real_selective_addmm_replay_preserves_finite_difference_r1_gradients_rng():
 # Native source SHA binds finite-difference R1, not analytic second backward.
 torch.manual_seed(19);w1=torch.randn(8,4);b1=torch.randn(8);w2=torch.randn(4,8);b2=torch.randn(4)
 x=torch.randn(9,4,requires_grad=True);y=x.detach().clone().requires_grad_();calls={'original':0,'replay':0};entries=[]
 def body(z):
  entries.append(True)
  return torch.tanh(torch.addmm(b2,torch.nn.functional.gelu(torch.addmm(b1,z,w1.t())),w2.t()))
 def contexts():
  def policy(ctx,op,*a,**k):
   if op==torch.ops.aten.addmm.default:
    calls['replay' if ctx.is_recompute else 'original']+=1;return CheckpointPolicy.MUST_SAVE
   return CheckpointPolicy.PREFER_RECOMPUTE
  return create_selective_checkpoint_contexts(policy,allow_cache_entry_mutation=False)
 def loss(out):
  fake,real,perturbed=out.chunk(3,dim=0)
  return fake.square().mean()+real.square().mean()+((perturbed-real)/.01).square().mean()
 state=torch.get_rng_state()
 a=checkpoint(body,x,use_reentrant=False);ga=torch.autograd.grad(loss(a),x)[0]
 torch.set_rng_state(state)
 b=checkpoint(body,y,use_reentrant=False,context_fn=contexts);gb=torch.autograd.grad(loss(b),y)[0]
 assert torch.equal(a,b) and torch.equal(ga,gb)
 assert len(entries)==4 and calls['original']==2 and torch.equal(state,torch.get_rng_state())
 # Torch2.14 stores decisions at forward and no longer re-enters policy_fn.
 # The production worker is pinned to2.10 and must observe both cache reads.
 if torch.__version__.startswith('2.10.'):
  assert calls['replay']==2
 else:
  assert torch.__version__.startswith('2.14.') and calls['replay']==0


def test_new_bundle_requires_actual_up_census_and_keeps_down40_checks():
 import json
 from copy import deepcopy
 from pathlib import Path
 from distillschedule.search.helios_gan_bundle import HeliosGANBundle
 fixture=json.loads((Path(__file__).parent/'fixtures/helios_gan_manual_counters.json').read_text())
 build=deepcopy(fixture['build']);state=build['gan_manual_runtime']['down40']
 state.update(up_blocks=[38,39],extra_up={'original':40,'replay':40})
 bundle=HeliosGANBundle.with_up_retention();after=bundle.effect_snapshot(build)
 result=bundle.effects_between(dict.fromkeys(after,0),after)
 assert result['call_deltas']['up_original']==40 and result['call_deltas']['down_original']==800
 for key in ('original','replay'):
  bad=deepcopy(build);bad['gan_manual_runtime']['down40']['extra_up'][key]=0
  wrong=bundle.effect_snapshot(bad)
  with pytest.raises(ValueError):bundle.effects_between(dict.fromkeys(wrong,0),wrong)
