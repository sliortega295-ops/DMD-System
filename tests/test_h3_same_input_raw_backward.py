"""CPU-only real probe machinery, not FA3/GPU numerical evidence."""
import importlib.util
from pathlib import Path
import pytest
import torch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('raw_repeat_probe',ROOT/'src/distillschedule/backends/training/minimax_h3/raw_backward_probe.py')
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)


def test_probe_keeps_actual_inputs_training_grads_rng_and_defers_host_reads(monkeypatch):
 torch.manual_seed(77)
 q,k,v=[torch.randn(2,4) for _ in range(3)];dout=torch.randn_like(q);out=torch.randn_like(q);lse=torch.randn(2)
 inputs=(dout,q,k,v,out,lse);primary=tuple(x.clone() for x in (q,k,v));before=[x.clone() for x in primary]
 state=torch.get_rng_state();calls=[]
 def raw(*args):
  assert all(args[i] is inputs[i] for i in range(6))
  deterministic=args[20];calls.append(deterministic)
  for x,target in zip((q,k,v),args[12:15]):target.copy_(x if deterministic else x+.01*len(calls))
 with monkeypatch.context() as mp:
  mp.setattr(torch.Tensor,'cpu',lambda *a,**kw: (_ for _ in ()).throw(AssertionError('host-visible tensor read inside cycle')))
  result=probe.run(torch,raw,inputs,primary,.5)
 assert calls==[False,False,False,True,True]
 assert torch.equal(state,torch.get_rng_state())
 assert all(torch.equal(a,b) for a,b in zip(before,primary))
 value=probe.finalize(result)
 assert all(x['unequal_elements']>0 for r in value['rows'][:3] for x in r['gradients'])
 assert all(x['bitwise_equal'] for x in value['rows'][-1]['gradients'])
 assert value['outputs_are_training_gradients'] is False and not torch.cuda.is_initialized()


def test_full_chunked_metrics_distinguish_zero_nonfinite_and_shape():
 a=torch.arange(30,dtype=torch.float32).reshape(5,6);b=a.clone();b[2,3]+=.25
 value=probe.finalize(dict(rows=[dict(gradients=[probe.metrics(torch,a.t(),b.t(),chunk=4)])]))
 m=value['rows'][0]['gradients'][0]
 assert m['unequal_elements']==1 and m['squared_error']==.25**2 and m['nonfinite_elements']==0
 with pytest.raises(RuntimeError):probe.metrics(torch,a,b.flatten())


def test_probe_rejects_input_mutation():
 q=torch.ones(2);inputs=(q,q,q,q,q,q);primary=(q.clone(),q.clone(),q.clone())
 def broken(*args):
  args[1].add_(1)
  for target in args[12:15]:target.zero_()
 with pytest.raises(RuntimeError,match='INPUT_VERSION_MUTATED'):probe.run(torch,broken,inputs,primary,1.)


def test_real_plan_binding_keeps_default_and_rejects_orphan_raw_flag():
 from copy import deepcopy
 from distillschedule.backends.training.minimax_h3 import divergence_probe as d
 base=dict(integration_head='a'*40,profile_id='minimax_h3_grid1000_best',mode='correctness',native_mode='correctness',run_id='SYNTHETIC_TEST_ONLY',asset_config=dict(run_root='/unread'),search_candidate=dict(id='joint-h3v1-d2-n128'))
 env=dict(EXPECTED_INTEGRATION_HEAD='a'*40,DISTILLSCHEDULE_SEARCH_CANDIDATE_ID='joint-h3v1-d2-n128',DISTILLSCHEDULE_RUN_KIND='correctness',H3_PLANNER_TWO_CYCLE='1',H3_PLANNER_CLOCK='host_fenced',DISTILLSCHEDULE_RUN_ID=base['run_id'],DISTILLSCHEDULE_RUN_ROOT='/unread/'+base['run_id'])
 base['nodes']=[dict(node_rank=i,environment=dict(env)) for i in (0,1)]
 default=d.bind_plan(base,'d2A');assert 'raw_repeat' not in default['h3_divergence'] and all('H3_PLANNER_RAW_BACKWARD_REPEAT' not in n['environment'] for n in default['nodes'])
 child=d.bind_plan(base,'d2A',raw_repeat=True)
 assert child['h3_divergence']['raw_repeat']==probe.CONTRACT and d.is_diagnostic_plan(child)
 for mutation in (lambda p:p['h3_divergence']['raw_repeat'].__setitem__('rank',1),lambda p:p['nodes'][1]['environment'].__setitem__('H3_PLANNER_RAW_BACKWARD_REPEAT','0')):
  bad=deepcopy(child);mutation(bad)
  with pytest.raises(ValueError):d.is_diagnostic_plan(bad)
 bad=deepcopy(base);bad['nodes'][0]['environment']['H3_PLANNER_RAW_BACKWARD_REPEAT']='1'
 with pytest.raises(ValueError):d.is_diagnostic_plan(bad)
 with pytest.raises(ValueError):d.bind_plan(base,'d2A',raw_repeat=1)
 assert not torch.cuda.is_initialized()


def test_new_block10_source_is_distinct_and_only_dispatches_actual_selected_block(monkeypatch):
 from types import SimpleNamespace
 from distillschedule.backends.training.minimax_h3 import divergence_probe as d
 assert probe.CONTRACT['block']==10 and probe.CONTRACT['schema']=='h3.same_inputs_raw_backward.block10.v1'
 assert probe.CONTRACT['training_gradients']=='ORIGINAL_UNMODIFIED' and not probe.CONTRACT['timing_eligible']
 seen=[]
 owner=SimpleNamespace(raw_repeat_enabled=True,active=True,rank=0,raw_repeat_result=None)
 monkeypatch.setattr(d,'_ACTIVE',owner)
 from distillschedule.backends.training.minimax_h3 import raw_backward_probe as real
 monkeypatch.setattr(real,'run',lambda *args:seen.append(args) or {'SYNTHETIC_TEST_ONLY':True})
 for block in (49,48,11):d.capture_raw_backward(None,None,block,(),(),1.)
 assert not seen and owner.raw_repeat_result is None
 runtime=SimpleNamespace(raw_backward=object())
 d.capture_raw_backward(None,runtime,10,(),(),1.)
 assert len(seen)==1 and owner.raw_repeat_result=={'SYNTHETIC_TEST_ONLY':True}
 with pytest.raises(ValueError,match='beyond one actual graph'):d.capture_raw_backward(None,runtime,10,(),(),1.)
