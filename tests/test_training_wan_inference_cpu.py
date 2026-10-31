"""W1 current-full mixed G -> independent NEW-input native latent proof."""
from copy import deepcopy
import json
import os
import shutil
import subprocess
import sys

import pytest
import torch

from test_training_wan_public_cpu import (STACK_AVAILABLE,FIXTURE,fixture_files,config_for,write_json)
from distillschedule.training import session
from distillschedule.training.config import TrainingConfig
from distillschedule.workloads.best_known.registry import repository_root,file_sha256

pytestmark = pytest.mark.skipif(not STACK_AVAILABLE,reason='actual full original mixed-precision CPU stack required')


def new_inputs(root, *, shape=(1,2,4096), dtype=torch.bfloat16):
    from distillschedule.data import import_preencoded
    root.mkdir(parents=True)
    torch.save(dict(prompt_embeds=torch.randn(shape,generator=torch.Generator().manual_seed(983)).to(dtype)),root/'native.pt')
    source = write_json(root/'inputs.json',dict(schema_version='distillschedule.data.preencoded.v1',
        records=[dict(id='NEW-W1-prompt-not-training',features=dict(path='native.pt',sha256=file_sha256(root/'native.pt')))],
        metadata=dict(evidence_kind=FIXTURE)))
    return import_preencoded(source,root/'prepared')


def inference_config(root,manifest,prepared):
    return TrainingConfig.from_dict(dict(schema_version='distillschedule.training.v1',
        model={k:manifest['model'][k] for k in ('factory','parameters')},
        algorithm=dict(factory='distillschedule.training.wan_inference:refuse_training'),
        data=dict(prepared=str(prepared)),execution=dict(adapter='program',mode='reference'),
        trainer=dict(cycles=1,seed=729,device='cpu',output_dir=str(root/'not-training'))),source=root/'inference.json')


def process(script,root):
    env=dict(os.environ);env.pop('DISTILLSCHEDULE_STABLE_BLOCK_NOISE',None)
    result = subprocess.run([sys.executable,'-c',script,str(root)],cwd=repository_root(),env=env,
        text=True,capture_output=True,timeout=300)
    assert result.returncode == 0,result.stdout+result.stderr
    return result.stdout


@pytest.fixture(scope='module')
def portable(tmp_path_factory):
    root = tmp_path_factory.mktemp('wan-portable')
    old = root/'training-payloads'
    files = fixture_files(old/'local',cohorts=2)
    first_config=config_for(files,old/'first',1)
    first_raw=first_config.to_dict();first_raw['model']['parameters']['inference_noise_policy']='fresh_projection_v1'
    first_config=TrainingConfig.from_dict(first_raw,source=first_config.source)
    with pytest.MonkeyPatch.context() as scope:
        scope.setenv('DISTILLSCHEDULE_STABLE_BLOCK_NOISE','1')
        trained = session.train(first_config)
    resumed_config = config_for(files,old/'resumed',2,resume=trained['checkpoint'])
    resumed_raw=resumed_config.to_dict();resumed_raw['model']['parameters']['inference_noise_policy']='fresh_projection_v1'
    resumed_config=TrainingConfig.from_dict(resumed_raw,source=resumed_config.source)
    write_json(root/'resume.json',resumed_config.to_dict())
    prepared = new_inputs(root/'NEW-inputs')
    script = r'''
from pathlib import Path
import json,os,sys,torch
from distillschedule.training import session
from distillschedule.training.config import load_training_config
from distillschedule.training.inference import restore_modules
from distillschedule.training.plugins import load_callable
from test_training_wan_inference_cpu import inference_config
root=Path(sys.argv[1]);config=load_training_config(root/'resume.json')
try:
 os.environ['DISTILLSCHEDULE_STABLE_BLOCK_NOISE']='1'
 resumed=session.train(config)
finally:os.environ.pop('DISTILLSCHEDULE_STABLE_BLOCK_NOISE',None)
assert resumed['resume_cycle']==1 and resumed['cursor']==2
original=session._load_for_output;captured=[]
def load(*args,**kw):
 result=original(*args,**kw);captured.append(result[0]);return result
session._load_for_output=load
manifest=session.export(config,resumed['checkpoint'],root/'export')
assert manifest['model']['parameters']['schema_version'].endswith('.v2')
assert manifest['model']['parameters']['noise_protocol']['policy']=='fresh_projection_v1'
assert 'DISTILLSCHEDULE_STABLE_BLOCK_NOISE' not in os.environ
owner=captured[0].state['native_cycle'];g=owner.generator.module
assert owner.versions==dict(generator=2,critic=10,ema=10)
weights=torch.load(root/'export/weights.pt',weights_only=True)
assert set(weights)=={'generator'} and set(weights['generator'])==set(g.state_dict())
assert all(torch.equal(weights['generator'][name],value) for name,value in g.state_dict().items())
assert any('lora_B' in k and torch.count_nonzero(v)>0 for k,v in weights['generator'].items())
assert any(v.dtype==torch.bfloat16 for v in weights['generator'].values())
assert any(v.dtype==torch.float32 for v in weights['generator'].values())
assert any(not torch.equal(dict(g.named_parameters())[name],shadow)
           for name,shadow in zip(owner.ema.names,owner.ema.shadows))
assert all(not getattr(m,'merged',False) for m in g.modules())
factory,_=load_callable(manifest['model']['factory'])
inf=factory({**manifest['model']['parameters'],'device':'cpu'})
restore_modules(inf,root/'export',manifest)
assert all(torch.equal(inf.state['generator'].state_dict()[k],v) for k,v in g.state_dict().items())
assert manifest['identity']['training_provenance']==resumed['identity']
cfg=inference_config(root,manifest,root/'NEW-inputs/prepared/manifest.json')
(root/'inference.json').write_text(json.dumps(cfg.to_dict()))
result=session.generate_from_export(cfg,root/'export',root/'expected',prepared=cfg.data['prepared'],output_kind='latents')
assert result['latents']['tensors']['latents']['dtype']=='torch.float32'
assert not torch.cuda.is_initialized()
(root/'export-proof.json').write_text(json.dumps(dict(status='PASS_CURRENT_FULL_UNMERGED_W1_NOT_PRETRAINED',
 current_keys=len(weights['generator']),current_full_generator_exact_export_and_strict_reload=True,
 resume_cycle=resumed['resume_cycle'],cursor=resumed['cursor'],versions=owner.versions,
 quality_validation='NOT_RUN',claimable_speedup=False,source=manifest['model']['parameters']['source'],
 runtime=manifest['model']['parameters']['runtime'])))
print('W1_FRESH_PROCESS_EXACT_RESTART_CURRENT_FULL_UNMERGED_G_EXPORT_PASS')
'''
    stdout = process(script,root)
    if os.environ.get('WAN_PUBLIC_EVIDENCE'):
        from pathlib import Path
        destination = Path(os.environ['WAN_PUBLIC_EVIDENCE'])/'artifacts'
        destination.mkdir(parents=True,exist_ok=True)
        shutil.copy2(root/'export-proof.json',destination/'current-generator-export.json')
        (destination/'export-process.stdout').write_text(stdout)
    shutil.rmtree(old)
    (root/'resume.json').unlink()
    return root,json.loads((root/'export/export.json').read_text()),prepared,stdout


def test_fresh_process_new_prompt_only_no_old_assets_training_or_optimizer(portable):
    root,manifest,prepared,_ = portable
    script = r'''
import json,sys,inspect,torch
from pathlib import Path
from distillschedule.training import session
from distillschedule.training.config import load_training_config
root=Path(sys.argv[1]);forbidden=str(root/'training-payloads')
def audit(event,args):
 if event=='open' and isinstance(args[0],(str,bytes)):
  assert forbidden not in str(args[0]), 'old training path read'
sys.addaudithook(audit)
class NoTraining:
 def find_spec(self,fullname,path=None,target=None):
  assert fullname not in ('distillschedule.training.wan_public','distillschedule.training.helios_step_owner'), fullname
sys.meta_path.insert(0,NoTraining())
def no_optimizer(*args,**kwargs):raise AssertionError('optimizer constructed during inference')
torch.optim.Adam.__init__=torch.optim.AdamW.__init__=no_optimizer
from distillschedule.adapters.helios.reference_model import load_reference_model_class
from distillschedule.strategies.wan_capture import native_primitives
cls=load_reference_model_class(backend='torch-sdpa');native=native_primitives(source_only=True)
constructor=inspect.unwrap(cls.__init__).__code__;models=[];calls=[];forwards=[];native_results=[]
def profile(frame,event,arg):
 if frame.f_code is constructor and event=='return':
  assert not frame.f_locals['is_use_gan'] and not frame.f_locals['is_use_gan_final']
  models.append(True)
 if frame.f_code is cls.forward.__code__ and event=='call':forwards.append(True)
 if frame.f_code is native.post.run_generator.__code__ and event=='return':
  calls.append(True);native_results.append(arg[0].detach().clone())
config=load_training_config(root/'inference.json')
try:
 sys.setprofile(profile)
 a=session.generate_from_export(config,root/'export',root/'actual-a',prepared=config.data['prepared'],output_kind='latents')
 b=session.generate_from_export(config,root/'export',root/'actual-b',prepared=config.data['prepared'],output_kind='latents')
finally:sys.setprofile(None)
assert len(models)==4, 'one admitted meta probe plus ONE actual full G per run; never C'
assert len(calls)==2 and len(forwards)>=6
read=lambda name:torch.load(root/name/'generation.pt',weights_only=True)['generated']['latents']
x,y,z=read('expected'),read('actual-a'),read('actual-b')
assert x.dtype==torch.float32 and torch.equal(x,y) and torch.equal(y,z) and bool(z.isfinite().all())
assert torch.equal(y,native_results[0]) and torch.equal(z,native_results[1]), 'adapter changed original native result'
assert not torch.cuda.is_initialized()
print('W1_ONLY_EXPORT_SOURCE_RUNTIME_NEW_PROMPT_ORIGINAL_FULL_MODEL_NATIVE_RUN_GENERATOR_EXACT_REPEAT_PASS')
'''
    stdout = process(script,root)
    assert 'EXACT_REPEAT_PASS' in stdout
    if os.environ.get('WAN_PUBLIC_EVIDENCE'):
        from pathlib import Path
        destination = Path(os.environ['WAN_PUBLIC_EVIDENCE'])/'artifacts'
        (destination/'generation-process.stdout').write_text(stdout)


def test_export_precision_source_layout_hash_prompt_and_default_video_reject(portable,tmp_path,monkeypatch):
    root,manifest,prepared,_ = portable
    from distillschedule.training.wan_inference import build_inference_model
    cfg = inference_config(root,manifest,prepared)
    for key in ('precision','source','task'):
        p = deepcopy(manifest['model']['parameters'])
        if key == 'precision':p[key] = 'FP32'
        elif key == 'source':p[key]['portable']['input_admission'] = '0'*64
        else:p[key]['first_step'] = 10
        with pytest.raises(ValueError):build_inference_model(p)
    for kind in ('shape','dtype'):
        inputs = new_inputs(tmp_path/kind,shape=(1,3,4096) if kind=='shape' else (1,2,4096),
            dtype=torch.float32 if kind=='dtype' else torch.bfloat16)
        output = tmp_path/f'rejected-{kind}'
        with pytest.raises(ValueError,match='prompt'):
            session.generate_from_export(cfg,root/'export',output,prepared=inputs,output_kind='latents')
        assert not output.exists()
    with pytest.raises(ValueError,match='decoder'):
        session.generate_from_export(cfg,root/'export',tmp_path/'video',prepared=prepared)
    assert not (tmp_path/'video').exists()
    broken = tmp_path/'broken';shutil.copytree(root/'export',broken)
    with (broken/'weights.pt').open('ab') as stream:stream.write(b'corrupt')
    with pytest.raises(ValueError,match='hash'):
        session.generate_from_export(cfg,broken,tmp_path/'hash-output',prepared=prepared,output_kind='latents')
    assert not (tmp_path/'hash-output').exists()
    broken2 = tmp_path/'layout';shutil.copytree(root/'export',broken2)
    value = json.loads((broken2/'export.json').read_text())
    value['model']['modules']['generator']['state']['items'].pop('patch_mid.base_layer.weight')
    write_json(broken2/'export.json',value)
    (broken2/'_SUCCESS').write_text(file_sha256(broken2/'export.json')+'\n')
    with pytest.raises(ValueError,match='layout'):
        session.generate_from_export(cfg,broken2,tmp_path/'layout-output',prepared=prepared,output_kind='latents')
    assert not (tmp_path/'layout-output').exists()
    monkeypatch.setenv('WORLD_SIZE','2')
    with pytest.raises(ValueError,match='single-process'):
        session.generate_from_export(cfg,root/'export',tmp_path/'topology',prepared=prepared,output_kind='latents')
