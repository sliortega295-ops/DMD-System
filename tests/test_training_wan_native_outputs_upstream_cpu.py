"""Real tiny upstream G + native sampler + real VAE, with synthetic rank transport.

No training cycle is rerun. A complete marker checkpoint at cycle100 is restored
by the actual OriginalWanSession/worker/checkpoint owners, through the public
export CLI. CUDA, asset/source admission and 16-rank transport are injected CPU
test seams, NOT GPU/pretrained/world16 qualification. Generation is independent.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest
import torch

from test_training_helios_public_cpu import STACK_AVAILABLE
from test_training_wan_native_outputs import (codec_request, pinned, output_cpu_worker)  # noqa: F401
from test_strategy_wan_bundle_stream import native_cpu_bundle  # noqa: F401
from test_strategy_wan_bundle_worker import cpu_worker  # noqa: F401
from test_training_end_to_end import equal
from distillschedule.workloads.best_known.registry import repository_root, fingerprint

pytestmark = pytest.mark.skipif(not STACK_AVAILABLE,reason='actual original Diffusers0.37.1/PEFT0.17.1 CPU stack required')


def original_g(task):
    from peft import LoraConfig
    from distillschedule.adapters.helios.reference_model import load_reference_model_class
    from runtime.critic_pool_helios import TRANSFORMER_ADDITIONAL_KWARGS,_all_linear_targets
    cls = load_reference_model_class(backend='torch-sdpa')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(283)
        g = cls(patch_size=(1,2,2),num_attention_heads=2,attention_head_dim=16,in_channels=16,out_channels=16,
            text_dim=4096,freq_dim=16,ffn_dim=64,num_layers=1,rope_dim=(4,6,6),
            is_use_gan=False,is_use_gan_final=False,is_use_gan_hooks=False,**TRANSFORMER_ADDITIONAL_KWARGS)
        with torch.no_grad():g.proj_out.weight.normal_(0,.02)
        for name,t in g.named_parameters():
            t.data = t.data.to(torch.float32 if any(k in name for k in cls._keep_in_fp32_modules) else torch.bfloat16)
        targets = sorted(set(_all_linear_targets(g)+['patch_short','patch_mid','patch_long']))
        g.requires_grad_(False)
        g.add_adapter(LoraConfig(r=task.lora_rank,lora_alpha=float(task.lora_rank),lora_dropout=0.,
            init_lora_weights='gaussian',target_modules=targets,exclude_modules=['down','up']))
        for name,t in g.named_parameters():
            if 'lora_' in name:t.data = t.data.float()
    g.enable_gradient_checkpointing();g.train()
    return g


def engines(task, *, current=False):
    from distillschedule.runtime.fsdp2_static import FSDP2EngineAdapter
    models = (original_g(task),torch.nn.Linear(1,1,bias=False))
    result = []
    for index,model in enumerate(models):
        trainables = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainables,lr=.002)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=2,gamma=.9)
        step = (100 if index == 0 else 500) if current else 0
        engine = FSDP2EngineAdapter(model,optimizer,scheduler,physical_step=step,global_batch=1)
        if current:
            scheduler.last_epoch = step;scheduler._step_count = step+1
            with torch.no_grad():
                for name,p in model.named_parameters():
                    if 'lora_B' in name or index == 1:p.fill_(.031)
                for p in trainables:
                    optimizer.state[p] = dict(step=torch.tensor(float(step)),exp_avg=torch.full_like(p,.017),exp_avg_sq=torch.full_like(p,.021))
        result.append(engine)
    g,c = result
    named = {name:p for name,p in g.module.named_parameters() if p.requires_grad}
    ema = NS(_names=list(named),_parameters=list(named.values()),
        _shadows=[p.detach().clone()+.07 for p in named.values()],optimization_step=500 if current else 0,
        decay=.9,min_decay=0.,update_after_step=0,cur_decay_value=None,device='cpu')
    return g,c,ema


def test_public_native_current_g_export_then_independent_new_prompt_real_video(output_cpu_worker,monkeypatch,capsys):
    from distillschedule.strategies import wan_worker,wan_stream
    from distillschedule.strategies.wan_task import OriginalWanTask
    from distillschedule.strategies.wan_strategy import WanStrategy,strategy_environment
    from distillschedule.strategies.wan_planning import source_fingerprint
    from distillschedule.training import native,cli,wan_native_inference as output
    from distillschedule.training.checkpoint import save_checkpoint,pack_state,rng_state
    from distillschedule.training.native_state import engine_state
    from distillschedule.open_workload.replicated import control_plane
    from distillschedule.search.wan_implementation import verify_execution_precision
    from distillschedule.training.config import load_training_config
    import distillschedule.strategies.wan_evidence as evidence
    h = output_cpu_worker;root = h.root
    task = OriginalWanTask(video_height=128,video_width=128,video_frames=33,first_step=0,lora_rank=1,stage2_inference_steps=(1,1,1))
    strategy = WanStrategy('resident',2048,False)
    stream = dict(stream_sha256='8'*64,cohorts=[dict(index=i) for i in range(100)])
    asset_identity = dict(model_files={'config.json':'7'*64},bootstrap_hash='c'*64,
        trace_bundle_hash='d'*64,trace_binding_hash='e'*64,negative_manifest_hash='f'*64)
    assets = wan_worker.OriginalWanAssets('SYNTHETIC_ADMISSION_ONLY','unused','c'*64,'unused','d'*64,'unused')
    bootstrap = NS(manifest=dict(lora_rank=1,model_identity_sha256='9'*64),component=lambda role:dict(seed=123))
    trace = NS(variant='no-gan',logical_steps=task.source_steps)
    negative = NS(embedding=NS(to=lambda **kw:torch.zeros(1)))
    constructed = []
    def backend(root):
        def build(**kw):
            g,c,ema = engines(task);constructed.append(NS(g=g,c=c,ema=ema))
            return g,c,ema,dict(scope='ACTUAL_TINY_G_SYNTHETIC_NATIVE_BUILD_METADATA')
        return NS(base=NS(_build_engines=build))
    monkeypatch.setattr(wan_worker,'load_assets',lambda *a,**kw:(bootstrap,trace,negative,asset_identity))
    monkeypatch.setattr(wan_worker,'load_original_backend',backend)
    monkeypatch.setattr(wan_stream,'load_stream',lambda *a,**kw:deepcopy(stream))
    monkeypatch.setattr(evidence,'validate_build',lambda *a,**kw:dict(status='SYNTHETIC_METADATA_ONLY'))
    monkeypatch.setattr(wan_worker,'execute_cycles',lambda *a,**kw:pytest.fail('native output executed training'))
    monkeypatch.setattr(wan_stream,'restore_stream_cursor',lambda *a,**kw:pytest.fail('native output advanced cursor'))
    old = root/'old-training';old.mkdir()
    checkpoint = old/'checkpoints/cycle-000100'
    h.transport.output = old
    with strategy_environment(strategy):
        verify_execution_precision(torch)
        g,c,ema = engines(task,current=True)
        current = deepcopy(g.module.state_dict())
        native_identity = dict(source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repository_root(),text=True).strip(),
            source_fingerprint=source_fingerprint(),task_sha256=task.digest,assets=asset_identity)
        runtime = {k:v for k,v in wan_worker.runtime_info(torch).items() if k != 'rank_seed'}
        identity = dict(native=native_identity,strategy=strategy.to_dict(),stream_sha256=stream['stream_sha256'],rank_runtimes=[runtime]*16)
        policy = dict(schema_version='distillschedule.wan_stream_checkpoint_policy.v1',strategy_sha256=fingerprint(strategy.to_dict()),
            stream_sha256=stream['stream_sha256'],selected_plan_sha256=None)
        h.transport.init_process_group('nccl');metadata = h.transport.new_group(backend='gloo')
        try:
            with control_plane(metadata):
                save_checkpoint(checkpoint,engine_state({'generator':g,'critic':c},ema=ema),identity=identity,
                    cycle=100,cursor=100,versions=dict(generator=100,critic=500,ema=500),policy=policy)
        finally:wan_worker.destroy_process_groups(torch.distributed,metadata)
    request = codec_request(old/'codec')
    config_path = root/'native.json'
    config_path.write_text(json.dumps(dict(schema_version='distillschedule.training.v1',model=dict(workload='W1'),algorithm={},
        data=dict(input_kind='original_wan_cohort_stream.v1',prepared=str(old/'stream.json')),
        execution=dict(adapter='native',mode='manual'),trainer=dict(cycles=100,output_dir=str(old),export=request))))
    def admit(config):
        # Only source/assets/topology transport are synthetic. The actual
        # public export dispatcher and native restore/output owner run below.
        return 'W1',dict(identity=dict(dirty=False)),dict(output=Path(config.trainer['output_dir']),plan=None),dict(
            task=task.to_dict(),assets=assets.__dict__,selected=strategy.to_dict())
    monkeypatch.setattr(native,'_admit',admit)
    snapshots = []
    publish = output.NativeOutputRequest.publish
    def observed(self,session,**kw):
        state = engine_state({'generator':session.g,'critic':session.c},ema=session.ema)
        before = pack_state(state);rng = rng_state()
        modes = {n:m.training for n,m in session.g.module.named_modules()}
        owners = [id(p) for p in session.g.module.parameters()]
        result = publish(self,session,**kw)
        assert equal(before,pack_state(state)) and equal(rng,rng_state())
        assert modes == {n:m.training for n,m in session.g.module.named_modules()}
        assert owners == [id(p) for p in session.g.module.parameters()]
        assert not hasattr(session,'_stream_binding') and session.runtime_cache == {}
        snapshots.append(True)
        return result
    monkeypatch.setattr(output.NativeOutputRequest,'publish',observed)
    weights = root/'portable'
    assert cli.main(['export',str(config_path),'--checkpoint',str(checkpoint),'--output',str(weights)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['status'] == 'EXPORTED' and result['checkpoint_cycle'] == 100 and snapshots == [True]
    saved = torch.load(weights/'weights.pt',weights_only=True)
    assert set(saved) == {'generator','codec'}
    assert all(torch.equal(saved['generator']['model.'+name],value) for name,value in current.items())
    assert any(v.dtype == torch.bfloat16 for v in saved['generator'].values())
    assert all(v.dtype == torch.float32 for k,v in saved['generator'].items() if 'lora_' in k)
    assert not any(getattr(m,'merged',False) for m in constructed[-1].g.module.modules())
    assert constructed[-1].g.global_steps == 100 and constructed[-1].c.global_steps == 500
    assert constructed[-1].ema.optimization_step == 500
    config = load_training_config(result['inference_config'])
    assert config.execution['adapter'] == 'program' and str(old) not in json.dumps(config.to_dict())
    p = result['model']['parameters']
    for fault,match in (('precision','ABI'),('legacy_schema','ABI'),('noise','noise protocol'),('source','source'),('codec_source','runtime'),
                        ('adapter','LoRA'),('capacity','capacity')):
        changed = deepcopy(p)
        if fault == 'precision':changed['precision'] = 'FP32'
        elif fault == 'legacy_schema':changed['schema_version'] = 'distillschedule.wan_native_generator_inference.v1'
        elif fault == 'noise':changed['noise_protocol']['schema_version'] = 'OLD_CHOLESKY'
        elif fault == 'source':changed['source']['providers']['helpers'] = '0'*64
        elif fault == 'codec_source':changed['codec']['runtime']['helpers']['decode_hook'] = '0'*64
        elif fault == 'adapter':changed['peft']['target_modules'].pop()
        else:changed['budget']['max_loaded_bytes'] = 64
        with pytest.raises(ValueError,match=match):output._admit(changed)
    from distillschedule.data import import_preencoded
    new = root/'NEW-input';new.mkdir()
    torch.save(dict(prompt_embeds=torch.randn((1,2,4096),generator=torch.Generator().manual_seed(981)).to(torch.bfloat16)),new/'prompt.pt')
    (new/'inputs.json').write_text(json.dumps(dict(schema_version='distillschedule.data.preencoded.v1',
        records=[dict(id='NEW-native-W1-prompt',features=dict(path='prompt.pt',sha256=pinned(new/'prompt.pt')['sha256']))],
        metadata=dict(evidence_kind=request['evidence_kind']))))
    prepared = import_preencoded(new/'inputs.json',new/'prepared')
    shutil.rmtree(old);config_path.unlink()
    script = r'''
import json,sys,torch
from pathlib import Path
root=Path(sys.argv[1]);old=str(root/'old-training')
def audit(event,args):
 if event=='open' and isinstance(args[0],(str,bytes)):assert old not in str(args[0]),'old training path read'
sys.addaudithook(audit)
def no_optimizer(*a,**kw):raise AssertionError('inference constructed optimizer')
torch.optim.Adam.__init__=torch.optim.AdamW.__init__=no_optimizer
class NoTraining:
 def find_spec(self,name,path=None,target=None):
  assert name not in ('distillschedule.training.native','distillschedule.training.wan_public','distillschedule.strategies.wan_worker'),name
sys.meta_path.insert(0,NoTraining())
torch.set_num_threads(1)
from distillschedule.training.cli import main
from distillschedule.training.wan_native_inference import NOISE_SCHEMA,noise_protocol
from distillschedule.strategies.wan_capture import native_primitives
from distillschedule.adapters.helios.reference_model import load_reference_model_class
from runtime.official_baseline import stable_block_noise
native=native_primitives(source_only=True);cls=load_reference_model_class(backend='torch-sdpa')
bindings=(native.post.sample_block_noise,native.post.run_generator,native.post.inference_with_trajectory_stage2)
counts=dict(native=0,pyramid=0,default_cholesky_sampler=0,stable_psd=0,forward=0)
codes={native.post.run_generator.__code__:'native',native.post.inference_with_trajectory_stage2.__code__:'pyramid',
 native.post.sample_block_noise.__code__:'default_cholesky_sampler',stable_block_noise.__code__:'stable_psd',cls.forward.__code__:'forward'}
def profile(frame,event,arg):
 if event=='call' and frame.f_code in codes:counts[codes[frame.f_code]]+=1
config=root/'portable.inference.json';prepared=root/'NEW-input/prepared/manifest.json'
parameters=json.loads(config.read_text())['model']['parameters']
assert parameters['schema_version']=='distillschedule.wan_native_generator_inference.v2'
assert parameters['noise_protocol']==noise_protocol() and parameters['noise_protocol']['schema_version']==NOISE_SCHEMA
try:
 sys.setprofile(profile)
 for suffix in ('video-a','video-b'):
  assert main(['generate',str(config),'--weights',str(root/'portable'),'--prepared',str(prepared),
    '--output-kind','video','--output',str(root/suffix)]) == 0
finally:sys.setprofile(None)
assert bindings==(native.post.sample_block_noise,native.post.run_generator,native.post.inference_with_trajectory_stage2)
assert counts['native']==counts['pyramid']==2 and counts['forward']==6
assert counts['default_cholesky_sampler']==0 and counts['stable_psd']==4
values=[torch.load(root/s/'generation.pt',weights_only=True) for s in ('video-a','video-b')]
assert torch.equal(values[0]['generated']['latents'],values[1]['generated']['latents'])
assert torch.equal(values[0]['decoded']['video'],values[1]['decoded']['video'])
assert values[0]['decoded']['video'].shape == (33,128,128,3)
assert values[0]['decoded']['decoder']['latent_clamp'] is False
assert not torch.cuda.is_initialized()
print('ACTUAL_TINY_ORIGINAL_G_NATIVE_SAMPLER_WAN21_VAE_NEW_INPUT_PORTABLE_PASS_NOT_GPU_OR_QUALITY')
'''
    env = dict(os.environ,WORLD_SIZE='1',RANK='0',LOCAL_RANK='0',LOCAL_WORLD_SIZE='1',GROUP_RANK='0',CUDA_VISIBLE_DEVICES='')
    process = subprocess.run([sys.executable,'-c',script,str(root)],cwd=repository_root(),env=env,text=True,capture_output=True,timeout=300)
    assert process.returncode == 0,process.stdout+process.stderr
    assert 'NEW_INPUT_PORTABLE_PASS' in process.stdout
    assert len(list((root/'video-a/frames').glob('*.png'))) == 33
    result = json.loads((root/'video-a/generation.json').read_text())
    assert result['sample_id'] == 'NEW-native-W1-prompt' and result['quality_validation'] == 'NOT_RUN'
    assert (root/'video-a/_SUCCESS').is_file()
