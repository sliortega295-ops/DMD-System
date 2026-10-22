"""Portable CURRENT full original G, native LATENT DIAGNOSTIC, never media quality."""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import torch

from test_training_helios_public_cpu import (
    STACK_AVAILABLE, FIXTURE, prepared_config, write_json, equal,
)
from distillschedule.data import import_preencoded
from distillschedule.training import session
from distillschedule.training.config import TrainingConfig
from distillschedule.workloads.best_known.registry import file_sha256, repository_root

pytestmark = pytest.mark.skipif(not STACK_AVAILABLE, reason='requires actual original optional Helios stack')


def new_inputs(root):
    root.mkdir(parents=True)
    prompt = torch.randn(1, 2, 4, generator=torch.Generator().manual_seed(983))
    torch.save({'prompt_embeds': prompt}, root / 'native.pt')
    source = write_json(root / 'inputs.json', {
        'schema_version': 'distillschedule.data.preencoded.v1',
        'records': [{'id': 'NEW-native-prompt', 'features': {
            'path': 'native.pt', 'sha256': file_sha256(root / 'native.pt')}}],
        'metadata': {'evidence_kind': FIXTURE}})
    return import_preencoded(source, root / 'prepared')


def inference_config(root, manifest, prepared):
    return TrainingConfig.from_dict({
        'schema_version': 'distillschedule.training.v1',
        'model': {'factory': manifest['model']['factory'], 'parameters': manifest['model']['parameters']},
        'algorithm': {'factory': 'distillschedule.training.helios_inference:refuse_training'},
        'data': {'prepared': str(prepared)}, 'execution': {'adapter': 'program', 'mode': 'reference'},
        'trainer': {'cycles': 1, 'seed': 729, 'device': 'cpu', 'output_dir': str(root / 'not-training')},
    }, source=root / 'inference.json')


def _process(script, root):
    result = subprocess.run([sys.executable, '-c', script, str(root)], cwd=repository_root(),
                            env=dict(os.environ), text=True, capture_output=True, timeout=240)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.fixture(scope='module')
def portable(tmp_path_factory):
    root = tmp_path_factory.mktemp('portable-original-current-g')
    old = root / 'training-payloads'
    config = prepared_config(old, capacity=10, cycles=1)
    trained = session.train(config)
    raw = config.to_dict()
    raw['trainer'].update(cycles=2, resume_from=trained['checkpoint'], output_dir=str(old / 'resumed'))
    write_json(root / 'resume.json', raw)
    prepared = new_inputs(root / 'NEW-inputs')
    script = r'''
import hashlib,json,sys
from pathlib import Path
import torch
from distillschedule.training import session
from distillschedule.training.config import load_training_config
from distillschedule.training.inference import read_export,restore_modules
from distillschedule.training.plugins import load_callable
from test_training_helios_public_inference_cpu import inference_config
root=Path(sys.argv[1]);config=load_training_config(root/'resume.json')
resumed=session.train(config)
assert resumed['resume_cycle']==1 and resumed['completed_cycles']==2
original=session._load_for_output; captured=[]
def load(*args,**kwargs):
    result=original(*args,**kwargs);captured.append(result[0]);return result
session._load_for_output=load
manifest=session.export(config,resumed['checkpoint'],root/'export')
components=captured[0];g=components.state['native_step'].generator.module
weights=torch.load(root/'export/weights.pt',weights_only=True)
assert set(weights)=={'generator'}
assert set(weights['generator'])==set(g.state_dict())
assert all(torch.equal(weights['generator'][k],v) for k,v in g.state_dict().items())
assert any('lora_B' in k and torch.count_nonzero(v)>0 for k,v in weights['generator'].items())
assert manifest['identity']['training_provenance']==resumed['identity']
assert manifest['identity']['model']==manifest['model']['source']
assert manifest['identity']['model_metadata']==manifest['model']['metadata']
assert manifest['export']['format']=='torch_state_dicts'
full_g={name:hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        for name,t in g.state_dict().items()}
(root/'full-generator.json').write_text(json.dumps(full_g,sort_keys=True))
factory,_=load_callable(manifest['model']['factory'])
inf=factory({**manifest['model']['parameters'],'device':'cpu'})
restore_modules(inf,root/'export',manifest)
assert all(torch.equal(inf.state['generator'].state_dict()[k],v) for k,v in g.state_dict().items())
config=inference_config(root,manifest,root/'NEW-inputs/prepared/manifest.json')
(root/'inference.json').write_text(json.dumps(config.to_dict()))
result=session.generate_from_export(config,root/'export',root/'expected',prepared=config.data['prepared'],output_kind='latents')
assert result['latents']['interpretation']=='LATENT_DIAGNOSTIC_NOT_MEDIA_QUALITY'
assert not torch.cuda.is_initialized()
from distillschedule.workloads.best_known.registry import file_sha256
(root/'export-proof.json').write_text(json.dumps(dict(
    status='PASS_CURRENT_FULL_ORIGINAL_TINY_G_NOT_PRETRAINED_OR_QUALITY',
    full_generator_sha256=hashlib.sha256(json.dumps(full_g,sort_keys=True).encode()).hexdigest(),
    full_generator_state_keys=len(full_g), full_weights_sha256=file_sha256(root/'export/weights.pt'),
    live_current_weights_equal_export_and_strict_reload=True, fresh_process_resume_cycle=resumed['resume_cycle'],
    completed_cycles=resumed['completed_cycles'], source=manifest['model']['parameters']['source'],
    runtime=manifest['model']['parameters']['runtime'], evidence_kind=manifest['model']['parameters']['evidence_kind'],
    training_identity_only_separate_provenance=True, quality_validation='NOT_RUN', claimable_speedup=False)))
print('FRESH_PROCESS_RESUME_CURRENT_FULL_G_EXACT_EXPORT_PASS')
'''
    export_process = _process(script, root)
    # Not a rename with a readable backup: every original source/base/data/run is gone.
    shutil.rmtree(old)
    (root / 'resume.json').unlink()
    evidence=os.environ.get('HELIOS_PUBLIC_U2_EVIDENCE')
    if evidence:
        destination=Path(evidence);destination.mkdir(parents=True,exist_ok=True)
        shutil.copy2(root/'export-proof.json',destination/'export-proof.json')
        (destination/'export-process.stdout').write_text(export_process.stdout)
    return root, json.loads((root / 'export/export.json').read_text()), prepared, export_process.stdout


def test_full_chain_fresh_process_uses_only_export_source_runtime_and_NEW_prompt(portable):
    root, manifest, prepared, _ = portable
    script = r'''
import hashlib,json,sys
from pathlib import Path
import torch
from distillschedule.training.config import load_training_config
from distillschedule.training import session
root=Path(sys.argv[1]);forbidden=str(root/'training-payloads')
reads=[]
def audit(event,args):
    if event=='open' and isinstance(args[0],(str,bytes)):
        path=str(args[0]);reads.append(path)
        assert forbidden not in path, 'old training payload read: '+path
sys.addaudithook(audit)
class NoAlgorithm:
    def find_spec(self,fullname,path=None,target=None):
        assert fullname not in ('distillschedule.training.helios_public','distillschedule.training.helios_step_owner'), fullname
sys.meta_path.insert(0,NoAlgorithm())
def no_optimizer(*a,**kw): raise AssertionError('optimizer constructed during inference')
torch.optim.Adam.__init__=torch.optim.AdamW.__init__=no_optimizer
from distillschedule.adapters.helios.reference_model import load_reference_model_class
cls=load_reference_model_class(backend='torch-sdpa')
import inspect
constructor=inspect.unwrap(cls.__init__).__code__;models=[];calls=[];dit=[]
config=load_training_config(root/'inference.json')
from distillschedule.strategies.wan_capture import native_primitives
native=native_primitives(source_only=True);original=native.post.run_generator
load=session.load_callable

def only_model(reference):
    assert reference!=config.algorithm['factory'], 'training algorithm loaded during generation'
    return load(reference)
session.load_callable=only_model

def profile(frame,event,arg):
    if event=='return' and frame.f_code is constructor:
        assert not frame.f_locals['is_use_gan'] and not frame.f_locals['is_use_gan_final'], 'critic constructed'
        models.append(frame.f_locals['self'])
    if event=='call' and frame.f_code is cls.forward.__code__:
        dit.append(True)
    if event=='return' and frame.f_code is original.__code__:
        g=frame.f_locals['transformer']
        assert all(not p.requires_grad for p in g.parameters()) and not g.gradient_checkpointing
        assert not arg[0].requires_grad
        calls.append(True)
sys.setprofile(profile)
try:
    result=session.generate_from_export(config,root/'export',root/'fresh',prepared=config.data['prepared'],output_kind='latents')
finally:
    sys.setprofile(None)
assert len(models)==2 and len(dit)==3, (len(models),len(dit))
assert set(sys.modules).isdisjoint({'distillschedule.training.helios_public','distillschedule.training.helios_step_owner'})
g=models[-1]
full_g={name:hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        for name,t in g.state_dict().items()}
assert full_g==json.loads((root/'full-generator.json').read_text())
actual=torch.load(result['artifact'],weights_only=True)['generated']['latents']
expected=torch.load(root/'expected/generation.pt',weights_only=True)['generated']['latents']
assert torch.equal(actual,expected) and list(actual.shape)==[1,16,9,16,16] and actual.isfinite().all()
assert calls==[True] and not torch.cuda.is_initialized()
assert result['video']['status']=='NOT_AVAILABLE' and not (root/'fresh/frames').exists()
assert result['quality_validation']=='NOT_RUN' and result['output_kind']=='latents'
assert not any(forbidden in p for p in reads)
(root/'fresh-proof.json').write_text(json.dumps(dict(
    status='PASS_NEW_OS_PROCESS_ORIGINAL_NATIVE_LATENT_DIAGNOSTIC_NOT_QUALITY',
    full_generator_sha256=hashlib.sha256(json.dumps(full_g,sort_keys=True).encode()).hexdigest(),
    latent_sha256=hashlib.sha256(actual.contiguous().numpy().tobytes()).hexdigest(),
    shape=list(actual.shape),dtype=str(actual.dtype),finite=True,exact_seed_repeat=True,
    actual_original_dit_calls=len(dit),generator_constructions=dict(meta=1,cpu=1),
    critic_constructions=0,optimizer_constructions=0,configured_training_algorithm_loaded=False,
    forbidden_training_payload_reads=0,training_payloads_removed=True,cuda_initialized=False,
    torch=str(torch.__version__),quality_validation='NOT_RUN',claimable_speedup=False)))
print('NEW_OS_PROCESS_ORIGINAL_THREE_DIT_CALLS_EXACT_LATENT_NO_C_ADAM_OLD_READS_CUDA_PASS')
'''
    result = _process(script, root)
    assert 'NO_C_ADAM_OLD_READS_CUDA_PASS' in result.stdout
    evidence=os.environ.get('HELIOS_PUBLIC_U2_EVIDENCE')
    if evidence:
        shutil.copy2(root/'fresh-proof.json',Path(evidence)/'fresh-proof.json')
        (Path(evidence)/'inference-process.stdout').write_text(result.stdout)
    # Portable factory parameters/metadata contain content pins, not old absolute paths.
    portable_text = json.dumps({key: manifest['model'][key] for key in ('parameters', 'metadata')})
    assert str(root) not in portable_text and '/home/' not in portable_text and '/tmp/' not in portable_text


def test_projection_references_current_unmerged_G_without_mutating_training_flags(tmp_path):
    from distillschedule.training.helios_public import build_model
    from distillschedule.training.checkpoint import rng_state
    config = prepared_config(tmp_path, capacity=5, cycles=1)
    components = build_model(config.model['parameters']); owner = components.state['native_step']
    g = owner.generator.module
    before = ({k:v.clone() for k,v in g.state_dict().items()}, [(m.training) for m in g.modules()],
              [p.requires_grad for p in g.parameters()], rng_state())
    projection = components.inference_projection(components, {'seed':773})
    assert projection.components.state['generator'] is g
    assert projection.factory == 'distillschedule.training.helios_inference:build_inference_model'
    assert all(torch.equal(before[0][k],v) for k,v in g.state_dict().items())
    assert before[1]==[m.training for m in g.modules()] and before[2]==[p.requires_grad for p in g.parameters()]
    assert equal(before[3],rng_state())
    assert all(not getattr(m,'merged',False) for m in g.modules())


def test_explicit_latents_preserves_video_default_and_training_refusal(portable,tmp_path):
    root, manifest, prepared, _ = portable
    config = inference_config(tmp_path, manifest, prepared)
    with pytest.raises(ValueError,match='actual decoder'):
        session.generate_from_export(config,root/'export',tmp_path/'video',prepared=prepared)
    assert not (tmp_path/'video').exists()
    with pytest.raises(ValueError,match='inference.only'):
        session.train(config)
    assert not Path(config.trainer['output_dir']).exists()
    with pytest.raises(ValueError,match='inference.only'):
        session.evaluate(config,root/'missing',tmp_path/'evaluation')
    assert not (tmp_path/'evaluation').exists()


@pytest.mark.parametrize('fault',['source','backend','peft','config','geometry','budget','layers','rope','device'])
def test_inference_factory_binding_rejects_changed_contract_before_real_allocation(portable,monkeypatch,fault):
    from distillschedule.training.helios_inference import build_inference_model
    root, manifest, _, _ = portable
    params=deepcopy(manifest['model']['parameters'])
    if fault=='source': params['source']['model']['source']['content_tree_sha256']='0'*64
    elif fault=='backend': params['source']['model']['backend']['name']='foreign'
    elif fault=='peft': params['peft']['r']=0
    elif fault=='config': params['architecture']['text_dim']=9
    elif fault=='geometry': params['geometry']['noise_shape'][-1]=17
    elif fault=='budget': params['budget']['max_parameters']=1
    elif fault=='layers': params['architecture']['num_layers']=10**9
    elif fault=='rope': params['architecture']['rope_dim']=[2,2,2]
    elif fault=='device': params['device']='cuda'
    with pytest.raises(ValueError): build_inference_model(params)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('fault',['missing','extra','dtype','shape','nonfinite'])
def test_full_weight_restore_stages_all_values_before_mutation(portable,tmp_path,fault):
    from distillschedule.training.helios_inference import build_inference_model
    from distillschedule.training.inference import read_export,restore_modules
    root, manifest, _, _ = portable
    target=tmp_path/'export';shutil.copytree(root/'export',target)
    weights=torch.load(target/'weights.pt',weights_only=True)
    names=list(weights['generator']);key=names[-1]
    if fault=='missing': weights['generator'].pop(key)
    elif fault=='extra': weights['generator']['extra']=torch.ones(1)
    elif fault=='dtype': weights['generator'][key]=weights['generator'][key].double()
    elif fault=='shape': weights['generator'][key]=torch.zeros(1)
    elif fault=='nonfinite': weights['generator'][key].fill_(float('nan'))
    torch.save(weights,target/'weights.pt')
    manifest=deepcopy(manifest);manifest['files']['weights.pt']=file_sha256(target/'weights.pt')
    write_json(target/'export.json',manifest);(target/'_SUCCESS').write_text(file_sha256(target/'export.json')+'\n')
    components=build_inference_model(manifest['model']['parameters'])
    before={k:v.clone() for k,v in components.state['generator'].state_dict().items()}
    _,admitted=read_export(target)
    with pytest.raises(ValueError,match='layout|nonfinite'):
        restore_modules(components,target,admitted)
    assert all(torch.equal(before[k],v) for k,v in components.state['generator'].state_dict().items())


@pytest.mark.parametrize('fault',['layout','dtype','nonfinite','extra'])
def test_native_prompt_is_exactly_declared_not_training_rows(portable,tmp_path,fault):
    from distillschedule.training.helios_inference import build_inference_model
    _, manifest, _, _ = portable
    components=build_inference_model(manifest['model']['parameters'])
    feeds={'prompt_embeds':torch.ones(1,2,4)}
    if fault=='layout': feeds['prompt_embeds']=torch.ones(2,4)
    elif fault=='dtype': feeds['prompt_embeds']=feeds['prompt_embeds'].double()
    elif fault=='nonfinite': feeds['prompt_embeds'].fill_(float('inf'))
    else: feeds['unused_training_rows']=torch.ones(1)
    with pytest.raises(ValueError,match='prompt'):
        components.generate(components.state,feeds,{})


def test_public_CLI_explicit_latent_diagnostic(portable,tmp_path,capsys):
    from distillschedule.training.cli import main
    root, manifest, prepared, _ = portable
    config=inference_config(tmp_path,manifest,prepared)
    path=write_json(tmp_path/'inference.json',config.to_dict())
    assert main(['generate',str(path),'--weights',str(root/'export'),'--prepared',str(prepared),
                 '--output-kind','latents','--output',str(tmp_path/'cli-latents')])==0
    receipt=json.loads(capsys.readouterr().out)
    assert receipt['output_kind']=='latents' and receipt['video']['status']=='NOT_AVAILABLE'
    assert receipt['latents']['tensors']['latents']['layout']=='BCTHW'
    assert not (tmp_path/'cli-latents/frames').exists()


def test_inference_original_constructor_initializes_RoPE_and_never_reseeds(portable,monkeypatch):
    from distillschedule.training.helios_inference import build_inference_model
    _, manifest, _, _ = portable
    def forbidden_seed(*args,**kwargs): raise AssertionError('factory reseeded')
    monkeypatch.setattr(torch,'manual_seed',forbidden_seed)
    components=build_inference_model(manifest['model']['parameters'])
    g=components.state['generator']
    original=type(g.rope)(rope_dim=g.config.rope_dim,theta=g.config.rope_theta)
    assert all(torch.equal(t,dict(original.named_buffers())[name]) for name,t in g.rope.named_buffers())
    assert all(t.device.type=='cpu' and t.dtype==torch.float32 and bool(t.isfinite().all())
               for t in [*g.parameters(),*g.buffers()])
    assert all(not p.requires_grad for p in g.parameters()) and not g.gradient_checkpointing
    assert components.export is None and components.load_export is None


def test_projected_export_keeps_strict_factory_metadata_identity(portable,tmp_path):
    from distillschedule.training.inference import read_export
    root, manifest, _, _ = portable
    target=tmp_path/'identity-mismatch';shutil.copytree(root/'export',target)
    manifest=deepcopy(manifest)
    manifest['model']['metadata']['scope']='WRONG_SCOPE'
    write_json(target/'export.json',manifest)
    (target/'_SUCCESS').write_text(file_sha256(target/'export.json')+'\n')
    with pytest.raises(ValueError,match='disagrees with identity'):
        read_export(target)


def test_inference_CPU_target_is_explicit_and_non_FP32_default_refuses(portable):
    from distillschedule.training.helios_inference import build_inference_model
    _, manifest, _, _ = portable
    with torch.device('meta'):
        components=build_inference_model(manifest['model']['parameters'])
    assert all(t.device.type=='cpu' for t in [*components.state['generator'].parameters(),
                                             *components.state['generator'].buffers()])
    original=torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        with pytest.raises(ValueError,match='FP32'):
            build_inference_model(manifest['model']['parameters'])
    finally:
        torch.set_default_dtype(original)
    assert not torch.cuda.is_initialized()
