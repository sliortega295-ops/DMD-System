"""Explicit reference-output noise protocol; real upstream CPU class, no GPU."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from test_training_helios_public_cpu import prepared_config, equal
from distillschedule.training.config import TrainingConfig
from distillschedule.workloads.best_known.registry import repository_root

POLICY='fresh_projection_v1'


def test_explicit_reference_noise_policy_is_admitted_without_changing_default(tmp_path):
    from distillschedule.training.helios_public_assets import admit_assets
    config=prepared_config(tmp_path,capacity=5,cycles=1)
    parameters=deepcopy(config.model['parameters'])
    default=admit_assets(parameters)
    parameters['inference_noise_policy']=POLICY
    current=admit_assets(parameters)
    assert current==default
    parameters['inference_noise_policy']='unbound-foreign-policy'
    with pytest.raises(ValueError,match='inference noise policy'):
        admit_assets(parameters)


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_private_projection_inputs_match_existing_primitive_and_do_not_consume_rng(dtype):
    import random
    from runtime.official_baseline import stable_block_noise
    from distillschedule.strategies.wan_capture import input_stream
    from distillschedule.training.reference_inference_noise import protocol,prepare,SCHEMA
    from distillschedule.training.wan_native_inference import NOISE_PREFIX
    p=dict(seed=819,geometry=dict(noise_shape=[1,16,3,16,16]),
        native=dict(rollout=dict(stage2_num_inference_steps_list=[1,2,3])),noise_protocol=protocol(str(dtype)))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(819)  # The public runtime seed, not p['seed'].
        before=torch.get_rng_state().clone(),random.getstate()
        inputs=prepare(p,dtype)
        assert torch.equal(before[0],torch.get_rng_state()) and before[1]==random.getstate()
        for name in ('entry_noise','rollout_noise'):
            with input_stream(819,SCHEMA,name):expected=torch.randn([1,16,3,16,16],dtype=dtype)
            assert torch.equal(inputs.tensor(NOISE_PREFIX+'/'+name),expected)
        for stage,h in ((1,8),(2,16)):
            name='block_noise/'+str(stage)
            with input_stream(819,SCHEMA,name):
                expected=stable_block_noise(16*3*(h//2)**2,device='cpu',dtype=torch.float32)
            expected=expected.view(1,16,3,h//2,h//2,2,2).permute(0,1,2,3,5,4,6).reshape(1,16,3,h,h)
            assert torch.equal(inputs.tensor(NOISE_PREFIX+'/'+name),expected)
        schedule=inputs.operation(NOISE_PREFIX+'/rollout')['control']['pyramid_stage_call_sequence']
        assert [v['call_count'] for v in schedule]==[1,2,3]
        inputs.finish()
        assert torch.equal(before[0],torch.get_rng_state()) and before[1]==random.getstate()
        assert not torch.cuda.is_initialized()



@pytest.fixture(scope='module',params=['helios','wan'])
def roundtrip(request,tmp_path_factory):
    """Real C1, fresh-process C2, v2 export, then completely new inference process."""
    from distillschedule.training import session
    family=request.param
    root=tmp_path_factory.mktemp('reference-output-'+family)
    if family=='helios':
        config=prepared_config(root/'old',capacity=10,cycles=1)
        from test_training_helios_public_inference_cpu import new_inputs
    else:
        from test_training_wan_public_cpu import fixture_files,config_for
        files=fixture_files(root/'old/local',cohorts=2)
        config=config_for(files,root/'old/first',1)
        from test_training_wan_inference_cpu import new_inputs
    raw=config.to_dict();raw['model']['parameters']['inference_noise_policy']=POLICY
    config=TrainingConfig.from_dict(raw,source=root/'first.json')
    saved=os.environ.get('DISTILLSCHEDULE_STABLE_BLOCK_NOISE')
    try:
        # Existing explicit CPU training fixture policy only; never leaked into
        # export/generation or interpreted as the versioned output policy.
        os.environ['DISTILLSCHEDULE_STABLE_BLOCK_NOISE']='1'
        first=session.train(config)
    finally:
        if saved is None:os.environ.pop('DISTILLSCHEDULE_STABLE_BLOCK_NOISE',None)
        else:os.environ['DISTILLSCHEDULE_STABLE_BLOCK_NOISE']=saved
    raw['trainer'].update(cycles=2,resume_from=first['checkpoint'],output_dir=str(root/'old/resumed'))
    (root/'resume.json').write_text(json.dumps(raw))
    prepared=new_inputs(root/'new-prompt')
    script=r'''
import json,os,sys,torch
from pathlib import Path
from distillschedule.training import session
from distillschedule.training.config import load_training_config
root=Path(sys.argv[1]);config=load_training_config(root/'resume.json')
assert 'DISTILLSCHEDULE_STABLE_BLOCK_NOISE' not in os.environ
try:
 os.environ['DISTILLSCHEDULE_STABLE_BLOCK_NOISE']='1'
 result=session.train(config)
finally:os.environ.pop('DISTILLSCHEDULE_STABLE_BLOCK_NOISE',None)
assert result['resume_cycle']==1 and result['completed_cycles']==2
manifest=session.export(config,result['checkpoint'],root/'export')
p=manifest['model']['parameters']
assert p['schema_version'].endswith('.v2') and p['noise_protocol']['policy']=='fresh_projection_v1'
assert manifest['identity']['training_provenance']==result['identity']
(root/'resumed-result.json').write_text(json.dumps(result))
print('FRESH_PROCESS_C2_VERSIONED_OUTPUT_EXPORT_NO_ENV_PASS')
'''
    env=dict(os.environ);env.pop('DISTILLSCHEDULE_STABLE_BLOCK_NOISE',None)
    result=subprocess.run([sys.executable,'-c',script,str(root)],cwd=repository_root(),env=env,
        capture_output=True,text=True,timeout=180)
    assert result.returncode==0,result.stdout+result.stderr
    manifest=json.loads((root/'export/export.json').read_text())
    value=dict(schema_version='distillschedule.training.v1',model={k:manifest['model'][k] for k in ('factory','parameters')},
        algorithm=dict(factory='distillschedule.training.helios_inference:refuse_training'),data=dict(prepared=str(prepared)),
        execution=dict(adapter='program',mode='reference'),trainer=dict(cycles=1,seed=729,device='cpu',output_dir=str(root/'never-train')))
    (root/'inference.json').write_text(json.dumps(value))
    return family,root,config,first,manifest


def test_fresh_process_v2_export_and_generate_uses_real_operation_inputs_without_env(roundtrip):
    family,root,config,first,manifest=roundtrip
    script=r'''
import json,os,sys,torch,random
from pathlib import Path
from distillschedule.training import session
from distillschedule.training.config import load_training_config,TrainingConfig
from distillschedule.strategies.wan_capture import native_primitives
from distillschedule.training import reference_inference_noise as noise
root=Path(sys.argv[1]);assert 'DISTILLSCHEDULE_STABLE_BLOCK_NOISE' not in os.environ
native=native_primitives(source_only=True)
original=noise.prepare;sampler=native.post.sample_block_noise;seen=[];cholesky=[];rng=[]
def observe(frame,event,value):
 if event=='call' and frame.f_code is sampler.__code__:cholesky.append(True)
 if frame.f_code is original.__code__:
  if event=='call':rng.append((torch.get_rng_state().clone(),random.getstate()))
  elif event=='return':
   assert torch.equal(rng[-1][0],torch.get_rng_state()) and rng[-1][1]==random.getstate()
   seen.append((frame.f_locals['parameters']['noise_protocol']['policy'],str(frame.f_locals['dtype']),value,torch.initial_seed()))
config=load_training_config(root/'inference.json')
try:
 sys.setprofile(observe)
 a=session.generate_from_export(config,root/'export',root/'actual-a',prepared=config.data['prepared'],output_kind='latents')
 b=session.generate_from_export(config,root/'export',root/'actual-b',prepared=config.data['prepared'],output_kind='latents')
 raw=config.to_dict();raw['trainer']['seed']=730
 changed=TrainingConfig.from_dict(raw,source=root/'runtime730.json')
 c=session.generate_from_export(changed,root/'export',root/'actual-c',prepared=changed.data['prepared'],output_kind='latents')
finally:sys.setprofile(None)
assert not cholesky, 'v2 fell back to original Cholesky sampler'
x=torch.load(a['artifact'],weights_only=True)['generated']['latents']
y=torch.load(b['artifact'],weights_only=True)['generated']['latents']
assert torch.equal(x,y) and x.dtype==torch.float32 and bool(x.isfinite().all())
assert len(seen)==3 and all(v[0]=='fresh_projection_v1' for v in seen)
assert [v[3] for v in seen]==[729,729,730]
assert all(torch.equal(seen[0][2]._values[k],seen[1][2]._values[k]) for k in seen[0][2]._values)
assert all(not torch.equal(seen[0][2]._values[k],seen[2][2]._values[k]) for k in seen[0][2]._values)
for _,_,inputs,_ in seen:inputs.finish()
z=torch.load(c['artifact'],weights_only=True)['generated']['latents']
assert z.dtype==torch.float32 and bool(z.isfinite().all())
assert not torch.equal(x,z), 'different runtime seed must change the real generated output'
(root/'runtime-seed-proof.json').write_text(json.dumps(dict(runtime_seeds=[729,729,730],factory_seed=config.model['parameters']['seed'],
 all_four_inputs_same_for_same_runtime_seed=True,all_four_inputs_changed_for_new_runtime_seed=True,
 outputs_same_for_same_runtime_seed=torch.equal(x,y),outputs_different_for_new_runtime_seed=not torch.equal(x,z),
 original_rollout_input_census_completed=3,caller_rng_preserved_at_prepare=True)))
assert 'DISTILLSCHEDULE_STABLE_BLOCK_NOISE' not in os.environ and not torch.cuda.is_initialized()
print('NEW_PROCESS_REAL_V2_OPERATION_INPUTS_FINISH_EXACT_REPEAT_NO_CHOLESKY_NO_ENV_PASS')
'''
    env=dict(os.environ);env.pop('DISTILLSCHEDULE_STABLE_BLOCK_NOISE',None)
    result=subprocess.run([sys.executable,'-c',script,str(root)],cwd=repository_root(),env=env,
        capture_output=True,text=True,timeout=180)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'NO_CHOLESKY_NO_ENV_PASS' in result.stdout


@pytest.mark.parametrize('fault',['old-schema','foreign-schema','missing-protocol','gamma','policy','source'])
def test_v2_inference_rejects_wrong_protocol_version_and_source(roundtrip,fault):
    family,root,_,_,manifest=roundtrip
    from distillschedule.training.plugins import load_callable
    factory,_=load_callable(manifest['model']['factory'])
    p=deepcopy(manifest['model']['parameters'])
    if fault=='old-schema':p['schema_version']=p['schema_version'].replace('.v2','.v1')
    elif fault=='foreign-schema':p['schema_version']='foreign.v2'
    elif fault=='missing-protocol':p.pop('noise_protocol')
    elif fault=='source':p['noise_protocol']['source']['primitive']='0'*64
    elif fault=='gamma':p['noise_protocol']['gamma']=.25
    else:p['noise_protocol']['policy']='foreign'
    with pytest.raises(ValueError):factory(p)


def test_resume_identity_binds_explicit_output_policy(roundtrip):
    from distillschedule.training import session
    family,root,config,first,_=roundtrip
    raw=json.loads((root/'resume.json').read_text())
    raw['model']['parameters'].pop('inference_noise_policy')
    raw['trainer']['output_dir']=str(root/'wrong-policy')
    invalid=TrainingConfig.from_dict(raw,source=root/'wrong.json')
    with pytest.raises(ValueError,match='identity'):
        session.train(invalid)


@pytest.mark.parametrize('family',['helios','wan'])
def test_unselected_v1_keeps_original_call_and_environment_rejection(monkeypatch,family):
    """Inspect the actual v1 call boundary without requiring platform Cholesky PD."""
    from distillschedule.training import helios_inference,wan_inference
    from distillschedule.strategies.wan_capture import native_primitives
    module=helios_inference if family=='helios' else wan_inference
    dtype=torch.float32 if family=='helios' else torch.bfloat16
    with torch.random.fork_rng(devices=[]):g=torch.nn.Linear(1,1)
    g.requires_grad_(False).eval();g.gradient_checkpointing=False
    p=dict(schema_version=module.SCHEMA,geometry=dict(noise_shape=[1,16,3,16,16],prompt_shape=[1,2,4]),
        native=dict(args={},scheduler={},rollout={}))
    state=dict(generator=g,inference=p);feeds=dict(prompt_embeds=torch.zeros(1,2,4,dtype=dtype))
    monkeypatch.delenv('DISTILLSCHEDULE_STABLE_BLOCK_NOISE',raising=False)
    native=native_primitives(source_only=True);calls=[]
    class OriginalBoundaryReached(RuntimeError):pass
    def profile(frame,event,value):
        if event=='call' and frame.f_code is native.post.run_generator.__code__:
            calls.append(dict(operation_trace=frame.f_locals['operation_trace'],trace_prefix=frame.f_locals['trace_prefix'],
                noise=frame.f_locals['noise'].clone()))
            raise OriginalBoundaryReached
    try:
        sys.setprofile(profile)
        with pytest.raises(OriginalBoundaryReached):module.generate(state,feeds,{})
    finally:sys.setprofile(None)
    assert len(calls)==1 and calls[0]['operation_trace'] is None and calls[0]['trace_prefix'] is None
    assert calls[0]['noise'].dtype==dtype and torch.count_nonzero(calls[0]['noise'])==0
    monkeypatch.setenv('DISTILLSCHEDULE_STABLE_BLOCK_NOISE','1')
    with pytest.raises(ValueError,match='environment.selected compatibility'):
        module.generate(state,feeds,{})


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_runtime_inference_seed_not_export_seed_controls_private_noise(dtype):
    from distillschedule.training.reference_inference_noise import protocol,prepare
    p=dict(seed=819,geometry=dict(noise_shape=[1,16,3,16,16]),
        native=dict(rollout=dict(stage2_num_inference_steps_list=[1,2,3])),noise_protocol=protocol(str(dtype)))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(729);before=torch.get_rng_state().clone();a=prepare(p,dtype)
        assert torch.equal(before,torch.get_rng_state())
        torch.manual_seed(729);same=prepare(p,dtype)
        torch.manual_seed(730);before=torch.get_rng_state().clone();b=prepare(p,dtype)
        assert torch.equal(before,torch.get_rng_state())
    assert all(torch.equal(a._values[k],same._values[k]) for k in a._values)
    assert all(not torch.equal(a._values[k],b._values[k]) for k in a._values)


@pytest.mark.parametrize('binding',['primitive','stream','wrapper-globals'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_live_noise_helpers_reject_rebinding_before_any_draw(monkeypatch,binding,dtype):
    from contextlib import contextmanager, nullcontext
    from types import FunctionType
    import random
    import runtime.official_baseline as official
    from distillschedule.strategies import wan_capture
    from distillschedule.training.reference_inference_noise import protocol,prepare
    p=dict(seed=819,geometry=dict(noise_shape=[1,16,3,16,16]),
        native=dict(rollout=dict(stage2_num_inference_steps_list=[1,2,3])),noise_protocol=protocol(str(dtype)))
    calls=[]
    def zeros(blocks,**kwargs):
        calls.append('primitive');return torch.zeros(blocks,4,dtype=torch.float32)
    @contextmanager
    def leaking(seed,*identity):
        calls.append('stream');yield
    if binding=='primitive':monkeypatch.setattr(official,'stable_block_noise',zeros)
    elif binding=='stream':monkeypatch.setattr(wan_capture,'input_stream',leaking)
    else:
        original=wan_capture.input_stream
        altered=dict(original.__globals__)
        def empty_context(*args,**kwargs):
            calls.append('factory');return nullcontext()
        altered['_GeneratorContextManager']=empty_context
        clone=FunctionType(original.__code__,altered,original.__name__,original.__defaults__,original.__closure__)
        clone.__dict__.update(original.__dict__)
        clone.__module__=original.__module__;clone.__qualname__=original.__qualname__
        assert clone.__code__ is original.__code__ and clone.__closure__ is original.__closure__
        assert clone.__wrapped__ is original.__wrapped__ and clone.__globals__ is not original.__globals__
        monkeypatch.setattr(wan_capture,'input_stream',clone)
    original_randn=torch.randn
    def observed_randn(*args,**kwargs):
        calls.append('draw');return original_randn(*args,**kwargs)
    monkeypatch.setattr(torch,'randn',observed_randn)
    python_before=random.getstate()
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(729);before=torch.get_rng_state().clone()
            with pytest.raises(ValueError,match='live inference noise helper'):
                prepare(p,dtype)
            assert torch.equal(before,torch.get_rng_state()) and python_before==random.getstate()
            assert calls==[]
    finally:random.setstate(python_before)
