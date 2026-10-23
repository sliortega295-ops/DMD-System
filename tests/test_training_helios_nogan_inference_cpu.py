"""W2 current full unmerged G to independent native NEW-prompt latents."""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import torch

from test_training_helios_public_cpu import STACK_AVAILABLE, FIXTURE, write_json
from test_training_helios_nogan_public_cpu import fixture_files, config_for, _stack
from distillschedule.training import session
from distillschedule.training.config import TrainingConfig
from distillschedule.workloads.best_known.registry import repository_root, file_sha256

if os.environ.get('HELIOS_NOGAN_PUBLIC_REQUIRE_STACK') == '1' and not STACK_AVAILABLE:
    raise RuntimeError('required full W2 CPU stack missing: Diffusers0.37.1/PEFT0.17.1')
pytestmark = pytest.mark.skipif(not STACK_AVAILABLE,
    reason='separate full original Diffusers0.37.1/PEFT0.17.1 W2 CPU proof')


def test_independent_w2_inference_factory_and_projection_exist():
    from distillschedule.training.helios_nogan_inference import build_inference_model, project_current_generator
    assert callable(build_inference_model) and callable(project_current_generator)


def test_inference_rejects_non_fp32_default_before_model_construction():
    from distillschedule.training.helios_nogan_inference import _cpu1
    before = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        with pytest.raises(ValueError, match='FP32 runtime default'):
            _cpu1()
    finally:
        torch.set_default_dtype(before)
    assert not torch.cuda.is_initialized()


def _constructor_admission_parameters():
    """Admission-only config, not a warm execution receipt or training owner."""
    import inspect
    from distillschedule.adapters.helios.reference_model import load_reference_model_class
    from distillschedule.training import helios_nogan_inference as inference
    from distillschedule.training.helios_inference import _json,content_runtime
    from distillschedule.training.helios_inputs import HeliosCohortTask
    _stack()
    source = inference.inference_source()
    from runtime.critic_pool_helios import TRANSFORMER_ADDITIONAL_KWARGS,_all_linear_targets
    cls = load_reference_model_class(backend='torch-sdpa')
    architecture = _json({k:p.default for k,p in inspect.signature(cls.__init__).parameters.items() if k != 'self'})
    architecture.update(num_attention_heads=1,attention_head_dim=16,freq_dim=16,ffn_dim=32,
                        num_layers=1,rope_dim=[4,6,6],**TRANSFORMER_ADDITIONAL_KWARGS)
    with torch.device('meta'):
        probe = cls(**architecture)
    targets = sorted(set(_all_linear_targets(probe)+['patch_short','patch_mid','patch_long']))
    task = HeliosCohortTask(0)
    return dict(schema_version=inference.SCHEMA,scope=inference.SCOPE,precision=inference.PRECISION,
        source=source,runtime=content_runtime(),task=task.to_dict(),
        boundary=dict(cycle=1,versions=dict(generator=3,critic=15,ema=15),
            warm_kind='ACTUAL_ORIGINAL_W2_CPU_TEST_WARM_NOT_HISTORICAL_STEP10'),architecture=architecture,
        peft=dict(r=256,lora_alpha=256.,lora_dropout=0.,init_lora_weights='gaussian',
            target_modules=targets,exclude_modules=['down','up']),
        native=dict(args=dict(base_seq_len=256,max_seq_len=4096,base_shift=.5,max_shift=1.15),
            scheduler=dict(shift=1.,stages=3,stage_range=[0,1/3,2/3,1],gamma=1/3),rollout=inference._rollout(task)),
        geometry=dict(noise_shape=list(task.latent_shape),prompt_shape=[1,512,4096]),
        budget=dict(max_parameters=20000000,max_loaded_bytes=400000000),evidence_kind=FIXTURE,seed=773)


@pytest.mark.parametrize('theta',[10000,10000.0])
def test_fixed_float_constructor_accepts_exact_json_number_representations(theta):
    from distillschedule.training.helios_nogan_inference import _admit
    p = _constructor_admission_parameters(); p['architecture']['rope_theta'] = theta
    admitted,_ = _admit(p)
    assert type(admitted['architecture']['rope_theta']) is type(theta)
    assert admitted['architecture']['rope_theta'] == theta


@pytest.mark.parametrize('theta',[10001,10000.00000001,True,False,'10000',None])
def test_fixed_float_constructor_rejects_wrong_value_bool_and_nonnumeric(theta):
    from distillschedule.training.helios_nogan_inference import _admit
    p = _constructor_admission_parameters(); p['architecture']['rope_theta'] = theta
    with pytest.raises(ValueError,match='architecture/conditioning/pyramid'):
        _admit(p)


def byte_equal(left, right):
    """Bytewise tensors, including signed zero; no tolerant numerical test."""
    if isinstance(left, torch.Tensor):
        return (isinstance(right,torch.Tensor) and left.shape == right.shape and left.dtype == right.dtype
                and torch.equal(left.contiguous().reshape(-1).view(torch.uint8),
                                right.contiguous().reshape(-1).view(torch.uint8)))
    if type(left) is not type(right):
        return False
    if isinstance(left,dict):
        return left.keys() == right.keys() and all(byte_equal(v,right[k]) for k,v in left.items())
    if isinstance(left,(tuple,list)):
        return len(left) == len(right) and all(byte_equal(a,b) for a,b in zip(left,right,strict=True))
    return left == right


def snapshot(owner):
    from distillschedule.training.checkpoint import cpu_tree, rng_state
    return dict(native=owner.state_dict(),outer_rng=cpu_tree(rng_state()),adapters=deepcopy(owner._adapters()),
        ownership={r:{n:id(p) for n,p in e.module.named_parameters()} for r,e in owner._engines.items()},
        ema_ids=[id(t) for t in owner.ema.shadows],
        models={r:dict(full=cpu_tree(dict(e.module.state_dict())),buffers=cpu_tree(dict(e.module.named_buffers())),
            modes={n:m.training for n,m in e.module.named_modules()},
            flags={n:p.requires_grad for n,p in e.module.named_parameters()},
            grads={n:cpu_tree(p.grad) for n,p in e.module.named_parameters()}) for r,e in owner._engines.items()})


def new_inputs(root, *, shape=(1,512,4096), dtype=torch.bfloat16, nonfinite=False):
    from distillschedule.data import import_preencoded
    root.mkdir(parents=True)
    prompt = torch.randn(shape,generator=torch.Generator().manual_seed(983)).to(dtype)
    if nonfinite:
        prompt.flatten()[0] = float('nan')
    torch.save(dict(prompt_embeds=prompt),root/'native.pt')
    source = write_json(root/'inputs.json',dict(schema_version='distillschedule.data.preencoded.v1',
        records=[dict(id='NEW-W2-prompt-not-warm-or-training',
            features=dict(path='native.pt',sha256=file_sha256(root/'native.pt')))],metadata=dict(evidence_kind=FIXTURE)))
    return import_preencoded(source,root/'prepared')


def inference_config(root,manifest,prepared):
    return TrainingConfig.from_dict(dict(schema_version='distillschedule.training.v1',
        model={k:manifest['model'][k] for k in ('factory','parameters')},
        algorithm=dict(factory='distillschedule.training.helios_nogan_inference:refuse_training'),
        data=dict(prepared=str(prepared)),execution=dict(adapter='program',mode='reference'),
        trainer=dict(cycles=1,seed=773,device='cpu',output_dir=str(root/'not-training'))),source=root/'inference.json')


def process(script,root):
    result = subprocess.run([sys.executable,'-c',script,str(root)],cwd=repository_root(),env=dict(os.environ),
        text=True,capture_output=True,timeout=900)
    assert result.returncode == 0,result.stdout+result.stderr
    return result.stdout


def evidence(name,text):
    if os.environ.get('HELIOS_NOGAN_INFERENCE_EVIDENCE'):
        root = Path(os.environ['HELIOS_NOGAN_INFERENCE_EVIDENCE'])/'artifacts'
        root.mkdir(parents=True,exist_ok=True)
        (root/name).write_text(text)


EXPORT_PROCESS = r'''
from copy import deepcopy
from pathlib import Path
import json,sys,torch
from distillschedule.training import session
from distillschedule.training.config import load_training_config
from distillschedule.training.checkpoint import cpu_tree,rng_state,restore_rng
from distillschedule.training.inference import restore_modules
from distillschedule.training.plugins import load_callable
from distillschedule.data.helios_nogan_reference_inputs import decode,descriptor,HeliosCohortTask
from test_training_helios_nogan_inference_cpu import byte_equal,snapshot,inference_config
root=Path(sys.argv[1]);config=load_training_config(root/'resume.json')
resumed=session.train(config)
assert resumed['resume_cycle']==1 and resumed['cursor']==2
# The real output loader constructs warm owners, then strictly restores ALL
# owners. Capture its genuine warm G and restored current owner, not a stand-in.
construct,load=session.construct,session._load_for_output
warm,captured,before=[],[],[]
def capture_construct(*args,**kw):
 result=construct(*args,**kw)
 warm.append(cpu_tree(dict(result[0].state['native_cycle'].generator.module.state_dict())))
 return result
def capture_load(*args,**kw):
 result=load(*args,**kw);captured.append(result);before.append(snapshot(result[0].state['native_cycle']));return result
session.construct=capture_construct;session._load_for_output=capture_load
try:manifest=session.export(config,resumed['checkpoint'],root/'export')
finally:session.construct=construct;session._load_for_output=load
components,dataset,payload,identity,prepared_session=captured[0]
owner=components.state['native_cycle'];g=owner.generator.module
assert owner.versions==dict(generator=4,critic=20,ema=20) and owner.cycle==2
assert byte_equal(before[0],snapshot(owner)), 'standard export mutated native/outer RNG, full G/C, EMA, flags, adapters, gradients or ownership'
projection=components.inference_projection(components,dict(seed=config.trainer['seed']))
assert projection.components.state=={'generator':g} and projection.components.state['generator'] is g
assert projection.components.generate is projection.components.export is projection.components.load_export is None
assert projection.factory=='distillschedule.training.helios_nogan_inference:build_inference_model'
assert byte_equal(before[0],snapshot(owner)), 'projection mutated owner'
weights=torch.load(root/'export/weights.pt',weights_only=True)
assert set(weights)=={'generator'} and set(weights['generator'])==set(g.state_dict())
assert all(byte_equal(weights['generator'][n],t) for n,t in g.state_dict().items())
assert any(not byte_equal(warm[0][n],p) for n,p in g.named_parameters() if p.requires_grad)
assert all(byte_equal(warm[0][n],p) for n,p in g.named_parameters() if not p.requires_grad)
assert any(v.dtype==torch.bfloat16 for v in weights['generator'].values())
assert all(p.dtype==torch.float32 for n,p in g.named_parameters() if 'lora_' in n)
assert all(not getattr(m,'merged',False) for m in g.modules())
assert all(any(n.startswith(p+'.lora_') for n in weights['generator']) for p in ('patch_short','patch_mid','patch_long'))
assert all(not (n.startswith(('critic.','optimizer.','ema.')) or 'exp_avg' in n) for n in weights['generator'])
# At native EMA20 the decay is ZERO: equality is required, not evidence of
# selection. Object/storage identity proves CURRENT full G, not EMA substitution.
parameters=dict(g.named_parameters());shadows=owner.ema.shadows
assert owner.ema.cur_decay_value==0. and all(byte_equal(parameters[n],s) for n,s in zip(owner.ema.names,shadows,strict=True))
assert all(parameters[n].data_ptr()!=s.data_ptr() for n,s in zip(owner.ema.names,shadows,strict=True))
assert not set(t.data_ptr() for t in g.parameters()) & set(t.data_ptr() for t in owner.critic.module.parameters())
assert any(not byte_equal(weights['generator'][n],owner.critic.module.state_dict()[n])
           for n in weights['generator'] if n in owner.critic.module.state_dict())
# Exactly the NEXT original update, with and without projection, from this same
# restored native boundary. No callback replacement, new solver or fake math.
safe=owner.state_dict();outer=cpu_tree(rng_state())
trace=decode(dataset.sample(2),HeliosCohortTask(2));row=descriptor(trace.reference_document)
a=owner.run(trace,row);state_a=snapshot(owner)
owner.load_state_dict(safe);restore_rng(outer)
projection=components.inference_projection(components,dict(seed=config.trainer['seed']))
assert byte_equal(before[0],snapshot(owner))
b=owner.run(trace,row);state_b=snapshot(owner)
assert byte_equal(a,b) and byte_equal(state_a,state_b), 'projection changed EXACT next original G1/C5/EMA5 update'
owner.load_state_dict(safe);restore_rng(outer)
assert byte_equal(before[0],snapshot(owner))
factory,_=load_callable(manifest['model']['factory'])
inf=factory({**manifest['model']['parameters'],'device':'cpu'})
restore_modules(inf,root/'export',manifest)
assert all(byte_equal(inf.state['generator'].state_dict()[n],t) for n,t in g.state_dict().items())
assert byte_equal(dict(inf.state['generator'].named_buffers()),dict(g.named_buffers()))
assert not inf.state['generator'].training and not inf.state['generator'].gradient_checkpointing
assert all(not p.requires_grad for p in inf.state['generator'].parameters())
assert manifest['identity']['training_provenance']==resumed['identity']
cfg=inference_config(root,manifest,root/'NEW-inputs/prepared/manifest.json')
(root/'inference.json').write_text(json.dumps(cfg.to_dict()))
result=session.generate_from_export(cfg,root/'export',root/'expected',prepared=cfg.data['prepared'],output_kind='latents')
assert result['latents']['tensors']['latents']==dict(layout='BCTHW',shape=[1,16,9,48,80],dtype='torch.float32')
assert not torch.cuda.is_initialized()
(root/'export-proof.json').write_text(json.dumps(dict(status='PASS_CURRENT_FULL_UNMERGED_W2_NOT_PRETRAINED',
 current_keys=len(weights['generator']),current_full_generator_exact_export_and_strict_reload=True,
 nonmutating_projection_and_standard_export_bytewise=True,exact_next_original_update_with_without_projection=True,
 current_generator_changed_from_genuine_warm=True,ema_selection_proof='CURRENT_G_OBJECT_AND_NONALIASING_NOT_NUMERICAL_DIFFERENCE',
 ema_values_equal_current_g=True,ema_decay=owner.ema.cur_decay_value,
 resume_cycle=resumed['resume_cycle'],cursor=resumed['cursor'],versions=owner.versions,
 quality_validation='NOT_RUN',distributed_execution='NOT_RUN',claimable_speedup=False,cuda_initialized=False,
 source=manifest['model']['parameters']['source'],runtime=manifest['model']['parameters']['runtime'])))
print('W2_FRESH_PROCESS_RESTORED_CURRENT_FULL_UNMERGED_G_EXPORT_AND_EXACT_NEXT_UPDATE_PASS')
'''


@pytest.fixture(scope='module')
def portable(tmp_path_factory):
    _stack()
    root = tmp_path_factory.mktemp('helios-nogan-portable')
    old = root/'training-payloads'
    # OWN fixture only. Never delete, relocate, repin or change the L1a fixture.
    files = fixture_files(old/'local')
    trained = session.train(config_for(files,old/'partial',1))
    write_json(root/'resume.json',config_for(files,old/'resumed',2,resume=trained['checkpoint']).to_dict())
    prepared = new_inputs(root/'NEW-inputs')
    stdout = process(EXPORT_PROCESS,root)
    proof = (root/'export-proof.json').read_text()
    evidence('current-generator-export.json',proof)
    evidence('export-process.stdout',stdout)
    # Leave only export, source/runtime, NEW prepared prompts and a test oracle.
    # The independent generator is not allowed to read the oracle until AFTER
    # its real generation calls. No old base/config/logical/feature/run survives.
    shutil.rmtree(old)
    (root/'resume.json').unlink()
    assert not old.exists()
    return root,json.loads((root/'export/export.json').read_text()),prepared,json.loads(proof)


def test_actual_restored_full_current_generator_projection_and_export(portable):
    root,manifest,prepared,proof = portable
    assert proof['nonmutating_projection_and_standard_export_bytewise']
    assert proof['exact_next_original_update_with_without_projection']
    assert proof['current_generator_changed_from_genuine_warm']
    assert proof['current_keys'] == len(manifest['model']['modules']['generator']['state']['items'])
    assert str(root/'training-payloads') not in json.dumps(manifest['model']['parameters'])
    assert manifest['model']['metadata']['scope'].endswith('NOT_WORLD16')
    assert manifest['model']['metadata']['generator_selection'] == 'CURRENT_FULL_UNMERGED_NOT_EMA'
    assert manifest['model']['factory'] == 'distillschedule.training.helios_nogan_inference:build_inference_model'
    assert not (root/'training-payloads').exists() and Path(prepared).is_file()


INFERENCE_PROCESS = r'''
import json,sys,inspect,os,torch
from pathlib import Path
from distillschedule.training import session
from distillschedule.training.config import load_training_config
root=Path(sys.argv[1]);forbidden=str(root/'training-payloads')
active=True
read_paths=[]
def audit(event,args):
 if event=='open' and isinstance(args[0],(str,bytes)):
  path=os.fsdecode(args[0]);read_paths.append(path)
  assert forbidden not in path, 'old training path read'
  assert not (active and path.startswith(str(root/'expected'))), 'oracle read during inference'
sys.addaudithook(audit)
class NoTraining:
 def find_spec(self,fullname,path=None,target=None):
  assert fullname not in ('distillschedule.training.helios_nogan_public','distillschedule.training.helios_nogan_bootstrap',
   'distillschedule.training.wan_public','distillschedule.training.helios_step_owner',
   'distillschedule.training.helios_reference_assets'), fullname
sys.meta_path.insert(0,NoTraining())
def no_optimizer(*args,**kwargs):raise AssertionError('Adam constructed during inference')
torch.optim.Adam.__init__=torch.optim.AdamW.__init__=no_optimizer
from distillschedule.adapters.helios.reference_model import load_reference_model_class
from distillschedule.strategies.wan_capture import native_primitives
cls=load_reference_model_class(backend='torch-sdpa');native=native_primitives(source_only=True)
constructor=inspect.unwrap(cls.__init__).__code__;models=[];calls=[];forwards=[];native_results=[]
def profile(frame,event,arg):
 if frame.f_code is constructor and event=='return':
  assert not frame.f_locals['is_use_gan'] and not frame.f_locals['is_use_gan_final']
  models.append(frame.f_locals['self'].patch_embedding.weight.device.type)
 if frame.f_code is cls.forward.__code__ and event=='call':forwards.append(True)
 if frame.f_code is native.post.run_generator.__code__ and event=='return':
  calls.append(True);native_results.append(arg[0].detach().clone())
config=load_training_config(root/'inference.json')
try:
 sys.setprofile(profile)
 a=session.generate_from_export(config,root/'export',root/'actual-a',prepared=config.data['prepared'],output_kind='latents')
 b=session.generate_from_export(config,root/'export',root/'actual-b',prepared=config.data['prepared'],output_kind='latents')
finally:sys.setprofile(None)
active=False
assert models==['meta','cpu','meta','cpu'], 'one admitted meta probe plus ONE actual full G per run; never C'
assert len(calls)==2 and len(forwards)==12, (calls,len(forwards))
read=lambda name:torch.load(root/name/'generation.pt',weights_only=True)['generated']['latents']
x,y,z=read('expected'),read('actual-a'),read('actual-b')
assert list(z.shape)==[1,16,9,48,80] and z.dtype==torch.float32 and torch.equal(x,y) and torch.equal(y,z) and bool(z.isfinite().all())
assert torch.equal(y,native_results[0]) and torch.equal(z,native_results[1]), 'adapter changed original native result'
assert not torch.cuda.is_initialized()
assert not (root/'training-payloads').exists()
(root/'generation-proof.json').write_text(json.dumps(dict(status='PASS_ONLY_EXPORT_SOURCE_RUNTIME_NEW_PROMPT',
 original_full_generator_forwards=len(forwards),original_native_run_generator_calls=len(calls),constructor_devices=models,
 latents_shape=list(z.shape),latents_dtype=str(z.dtype),finite=True,exact_repeat=True,native_result_exact=True,
 cuda_initialized=False,old_payload_trees_deleted=True,old_path_reads_blocked=True,training_owner_imports_blocked=True,
 adam_construction_blocked=True,quality_validation='NOT_RUN',claimable_speedup=False)))
print('W2_ONLY_EXPORT_SOURCE_RUNTIME_NEW_PROMPT_ORIGINAL_FULL_MODEL_NATIVE_RUN_GENERATOR_EXACT_REPEAT_PASS')
'''


def test_fresh_process_new_prompt_without_old_paths_owners_or_optimizer(portable):
    root,_,_,_ = portable
    stdout = process(INFERENCE_PROCESS,root)
    evidence('generation-process.stdout',stdout)
    evidence('generation-proof.json',(root/'generation-proof.json').read_text())
    assert 'EXACT_REPEAT_PASS' in stdout


@pytest.mark.parametrize('kind', ['w1','w3','step0','scope','precision','runtime','source','layout','lora',
    'missing_pyramid','architecture','horizon','flow_shift','cache','history','boundary','budget'])
def test_foreign_task_scope_native_identity_and_policy_reject_before_forward(portable,kind):
    from distillschedule.training.helios_nogan_inference import build_inference_model
    from distillschedule.adapters.helios.reference_model import load_reference_model_class
    from distillschedule.strategies.wan_task import OriginalWanTask
    p = deepcopy(portable[1]['model']['parameters'])
    if kind == 'w1': p['task'] = OriginalWanTask().to_dict()
    elif kind == 'w3': p['task']['request']['variant'] = 'gan'
    elif kind == 'step0': p['task']['source_steps'][0] = 0
    elif kind == 'scope': p['scope'] = 'CPU1_RANK0_PER_SAMPLE_ORIGINAL_W1_NOT_WORLD16'
    elif kind == 'precision': p['precision'] = 'FP32'
    elif kind == 'runtime': p['runtime']['torch']['version'] = '0.0'
    elif kind == 'source': p['source']['portable']['inference'] = '0'*64
    elif kind == 'layout': p['source']['layout'][next(iter(p['source']['layout']))] = '0'*64
    elif kind == 'lora': p['peft']['r'] = 16
    elif kind == 'missing_pyramid': p['peft']['target_modules'].remove('patch_mid')
    elif kind == 'architecture': p['architecture']['guidance_cross_attn'] = False
    elif kind == 'horizon': p['native']['rollout']['stage2_num_inference_steps_list'] = [1,1,1]
    elif kind == 'flow_shift': p['native']['args']['base_shift'] = .6
    elif kind == 'cache': p['native']['rollout']['use_kv_cache'] = True
    elif kind == 'history': p['native']['rollout']['is_use_gt_history'] = True
    elif kind == 'boundary': p['boundary']['versions']['generator'] = 2
    else: p['budget']['max_parameters'] = 1
    cls = load_reference_model_class(backend='torch-sdpa');forwards = []
    def profile(frame,event,arg):
        if event == 'call' and frame.f_code is cls.forward.__code__: forwards.append(True)
    try:
        sys.setprofile(profile)
        with pytest.raises(ValueError): build_inference_model(p)
    finally:
        sys.setprofile(None)
    assert not forwards and not torch.cuda.is_initialized()


def test_projection_rejects_foreign_owner_even_with_same_state_name():
    from distillschedule.training.adapters import ModelComponents
    from distillschedule.training.helios_nogan_inference import project_current_generator
    with pytest.raises(ValueError,match='independent original W2 owner'):
        project_current_generator(ModelComponents(state={'native_cycle':object()}),dict(seed=773))
    with pytest.raises(ValueError,match='independent original W2 owner'):
        project_current_generator(ModelComponents(state={'native_step':object()}),dict(seed=773))


@pytest.mark.parametrize('tamper',['helper','model','namespace','environment'])
def test_existing_source_only_live_helper_and_cache_admission_stays_fail_closed(portable,monkeypatch,tamper):
    from distillschedule.training.helios_nogan_inference import build_inference_model
    from distillschedule.adapters.helios.reference_model import load_reference_model_class
    from distillschedule.strategies.wan_capture import native_primitives
    cls = load_reference_model_class(backend='torch-sdpa')
    native = native_primitives(source_only=True)
    called = []
    def forbidden(*args,**kwargs):
        called.append(True)
        raise AssertionError('redirected numerical callback executed')
    if tamper == 'helper': monkeypatch.setattr(native.post,'run_generator',forbidden)
    elif tamper == 'model': monkeypatch.setattr(cls,'forward',forbidden)
    elif tamper == 'namespace': monkeypatch.setattr(native.post,'__file__','/unadmitted-helper.py')
    else: monkeypatch.setenv('DISTILLSCHEDULE_STABLE_BLOCK_NOISE','1')
    with pytest.raises(ValueError):
        build_inference_model(deepcopy(portable[1]['model']['parameters']))
    assert not called


def test_nonfinite_extra_input_and_live_cache_refuse_before_native_forward(portable,monkeypatch):
    from distillschedule.training.helios_nogan_inference import build_inference_model
    from distillschedule.training.inference import restore_modules
    root,manifest,_,_ = portable
    inf = build_inference_model(manifest['model']['parameters'])
    restore_modules(inf,root/'export',manifest)
    with pytest.raises(ValueError,match='inference-only'):
        inf.admit_training()
    prompt = torch.zeros(1,512,4096,dtype=torch.bfloat16)
    prompt.flatten()[0] = float('nan')
    with pytest.raises(ValueError,match='prompt'):
        inf.generate(inf.state,dict(prompt_embeds=prompt),{})
    prompt.zero_()
    with pytest.raises(ValueError,match='prompt'):
        inf.generate(inf.state,dict(prompt_embeds=prompt,history_short=prompt),{})
    processor = inf.state['generator'].blocks[0].attn1.processor
    monkeypatch.setattr(processor,'cache_enabled',True)
    with pytest.raises(ValueError,match='cache-free'):
        inf.generate(inf.state,dict(prompt_embeds=prompt),{})


def test_safe_weight_hash_module_layout_and_missing_keys_reject_without_forward(portable,tmp_path):
    from distillschedule.adapters.helios.reference_model import load_reference_model_class
    root,manifest,prepared,_ = portable
    cfg = inference_config(root,manifest,prepared)
    cls = load_reference_model_class(backend='torch-sdpa');forwards = []
    def profile(frame,event,arg):
        if event == 'call' and frame.f_code is cls.forward.__code__: forwards.append(True)
    try:
        sys.setprofile(profile)
        for kind in ('hash','module_layout','missing_key','precision'):
            broken = tmp_path/kind;shutil.copytree(root/'export',broken)
            value = json.loads((broken/'export.json').read_text())
            if kind == 'hash':
                with (broken/'weights.pt').open('ab') as stream: stream.write(b'corrupt')
            elif kind == 'module_layout':
                value['model']['modules']['generator']['state']['items'].pop('patch_mid.base_layer.weight')
            else:
                weights = torch.load(broken/'weights.pt',weights_only=True)
                if kind == 'missing_key': weights['generator'].pop('patch_mid.base_layer.weight')
                else: weights['generator']['patch_mid.base_layer.weight'] = weights['generator']['patch_mid.base_layer.weight'].float()
                torch.save(weights,broken/'weights.pt')
                value['files']['weights.pt'] = file_sha256(broken/'weights.pt')
            if kind != 'hash':
                write_json(broken/'export.json',value)
                (broken/'_SUCCESS').write_text(file_sha256(broken/'export.json')+'\n')
            output = tmp_path/(kind+'-output')
            with pytest.raises(ValueError,match='hash|layout'):
                session.generate_from_export(cfg,broken,output,prepared=prepared,output_kind='latents')
            assert not output.exists()
    finally:
        sys.setprofile(None)
    assert not forwards


def test_new_prompt_admission_default_video_refusal_and_explicit_latent_cli(portable,tmp_path,monkeypatch,capsys):
    from distillschedule.training.cli import main
    from distillschedule.adapters.helios.reference_model import load_reference_model_class
    root,manifest,prepared,_ = portable
    cfg = inference_config(root,manifest,prepared)
    cls = load_reference_model_class(backend='torch-sdpa');forwards = []
    def profile(frame,event,arg):
        if event == 'call' and frame.f_code is cls.forward.__code__: forwards.append(True)
    try:
        sys.setprofile(profile)
        for kind in ('shape','dtype'):
            inputs = new_inputs(tmp_path/kind,shape=(1,511,4096) if kind=='shape' else (1,512,4096),
                dtype=torch.float32 if kind=='dtype' else torch.bfloat16)
            output = tmp_path/(kind+'-rejected')
            with pytest.raises(ValueError,match='prompt'):
                session.generate_from_export(cfg,root/'export',output,prepared=inputs,output_kind='latents')
            assert not output.exists()
        with pytest.raises(ValueError,match='prompt'):
            session.generate_from_export(cfg,root/'export',tmp_path/'override',prepared=prepared,
                parameters={'use_kv_cache':True},output_kind='latents')
    finally:
        sys.setprofile(None)
    assert not forwards
    # Generic output handling intentionally has no fabricated decoder.
    config_path = write_json(tmp_path/'inference.json',cfg.to_dict())
    args = ['generate',str(config_path),'--weights',str(root/'export'),'--prepared',str(prepared)]
    assert main([*args,'--output',str(tmp_path/'video')]) == 2
    assert 'decoder' in capsys.readouterr().err and not (tmp_path/'video').exists()
    assert main([*args,'--output',str(tmp_path/'latents'),'--output-kind','latents']) == 0
    result = json.loads((tmp_path/'latents/generation.json').read_text())
    assert result['output_kind'] == 'latents' and result['video']['status'] == 'NOT_AVAILABLE'
    assert result['quality_validation'] == 'NOT_RUN' and (tmp_path/'latents/_SUCCESS').is_file()
    monkeypatch.setenv('LOCAL_RANK','1')
    with pytest.raises(ValueError,match='CPU1/rank0'):
        session.generate_from_export(cfg,root/'export',tmp_path/'topology',prepared=prepared,output_kind='latents')
