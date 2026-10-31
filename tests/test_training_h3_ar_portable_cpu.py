"""New native-layout inference consumer; no training/GPU/quality is claimed."""
from copy import deepcopy
import inspect
import json
import os
import subprocess
import sys

import pytest
import torch


def test_portable_h3_native_discovery_remains_model_free():
    script = "import sys;from distillschedule.training.h3_ar_portable import build_inference_model;assert callable(build_inference_model);assert 'torch' not in sys.modules;assert 'deepspeed' not in sys.modules"
    run = subprocess.run([sys.executable, '-B', '-c', script], capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stdout + run.stderr


@pytest.fixture(scope='module')
def original_native_layout_case(tmp_path_factory):
    from test_h3_cached_original_cpu import original, tiny_architecture, sample
    from test_training_h3_cached_inference_cpu import inference_config, new_inputs
    from distillschedule.training import h3_ar_portable as output
    from distillschedule.training.h3_cached_inference import _drop_fake
    from distillschedule.training.plugins import load_callable
    from distillschedule.training.inference import publish_export
    torch.set_num_threads(1)
    api = original()
    root = tmp_path_factory.mktemp('native-h3-portable')
    defaults = {k: v.default for k, v in inspect.signature(api.model.MiniMaxH3DiT.__init__).parameters.items()
                if k not in ('self', 'kwargs')}
    arch = json.loads(json.dumps({**defaults, **tiny_architecture()}))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(829)
        adapter = api.dmd2.H3Adapter(api.model.MiniMaxH3DiT(**arch), output.original_config())
        _drop_fake(adapter)
        for name, p in adapter.named_parameters():
            if '.adapters.student.' in name:
                with torch.no_grad(): p.uniform_(-.01, .01)
            else: p.data = p.data.bfloat16()
        raw = sample(7)
        fresh = new_inputs(root / 'fresh', raw=raw)
    geometry = dict(video_shape=[1,24,2,4,4], audio_shape=[2,32,3], prompt_shape=[3,5120],
                    reference_video_shape=[1,24,3,4,4], reference_audio_shape=[2,32,5])
    weights = {k: t.detach().clone() for k, t in adapter.state_dict().items()}
    layout = {k: dict(shape=list(t.shape), dtype=str(t.dtype)) for k, t in weights.items()}
    p = output.parameters(architecture=arch, geometry=geometry, weight_layout=layout, seed=729,
        evidence_kind=output.FIXTURE, budget=dict(max_parameters=1_000_000, max_loaded_bytes=256*2**20,
            max_input_elements=100_000, max_attention_elements=1_000_000, host_reserve_bytes=2**20))
    projection = output.export_components(p, weights=weights, model_class=api.model.MiniMaxH3DiT,
                                          adapter_class=api.dmd2.H3Adapter)
    _, source = load_callable(output.FACTORY)
    metadata = dict(projection.metadata)
    manifest = publish_export(projection, root/'export', model=dict(factory=output.FACTORY,source=source,
        parameters=p,metadata=metadata), identity=dict(model=source,model_metadata=metadata,
        training_provenance='SYNTHETIC_TEST_ONLY_NATIVE_LAYOUT_NOT_ZERO3_CHECKPOINT'), cycle=3)
    config = inference_config(root, manifest, fresh)
    raw_config = config.to_dict();raw_config['algorithm']['factory']='distillschedule.training.h3_ar_portable:refuse_training'
    (root/'inference.json').write_text(json.dumps(raw_config))
    return root, p, weights, fresh, raw, api, config


def test_original_mixed_storage_current_student_strict_load_and_same_rollout(original_native_layout_case):
    root, p, weights, fresh, raw, api, config = original_native_layout_case
    from distillschedule.training import h3_ar_portable as output
    from distillschedule.training.inference import read_export, restore_modules
    from distillschedule.training.checkpoint import rng_state, restore_rng
    from distillschedule.data import PreparedDataset
    from test_training_end_to_end import equal
    before = rng_state(device='cpu')
    comp = output.build_inference_model(p)
    assert equal(before, rng_state(device='cpu'))
    feeds = PreparedDataset(fresh).sample(0)
    with pytest.raises(ValueError, match='restored'): comp.generate(comp.state, feeds, {})
    _, manifest = read_export(root/'export')
    restore_modules(comp, root/'export', manifest)
    adapter = comp.state['student'].adapter
    assert all(torch.equal(t,weights[k]) and t.dtype==weights[k].dtype for k,t in adapter.state_dict().items())
    assert any(t.dtype==torch.bfloat16 for t in adapter.parameters())
    assert all(p.dtype==torch.float32 for _,layer in adapter.layers for p in layer.adapters['student'].parameters())
    assert all(set(layer.adapters)=={'student'} for _,layer in adapter.layers)
    inputs, negative = api.dmd2.prepare_batch(raw,1.0)
    assert negative is None
    random = rng_state(device='cpu')
    with torch.no_grad(),torch.autocast('cpu',dtype=torch.bfloat16,cache_enabled=False):
        expected,_ = api.dmd2.rollout(adapter,(inputs['input_latents'],inputs['audio_input_latents']),inputs,
                                     api.dmd2.make_schedule(output.original_config(),device='cpu'))
    restore_rng(random)
    actual=comp.generate(comp.state,feeds,{})
    assert torch.equal(expected[0],actual['video_latents']) and torch.equal(expected[1],actual['audio_latents'])
    assert not torch.cuda.is_initialized()


def test_new_process_public_generate_needs_no_native_training_assets(original_native_layout_case):
    root, p, weights, fresh, raw, api, config = original_native_layout_case
    script = r'''
import sys,torch,json
from pathlib import Path
torch.set_num_threads(1)
root=Path(sys.argv[1])
class NoNativeOwner:
 def find_spec(self,name,path=None,target=None):
  # Exercise the unchanged upstream optional-dependency branch as in a minimal
  # CPU install, not a replacement numeric method or a fake backend module.
  if name=='deepspeed' or name.startswith('deepspeed.'):
   raise ModuleNotFoundError('TEST_ONLY optional DeepSpeed package absent')
  assert name not in ('distillschedule.training.h3_ar_training',
   'distillschedule.strategies.h3_ar_session','distillschedule.training.h3_cached_owner'),name
sys.meta_path.insert(0,NoNativeOwner())
from distillschedule.cli import main
assert main(['generate',str(root/'inference.json'),'--weights',str(root/'export'),
 '--prepared',str(root/'fresh/prepared/manifest.json'),'--output',str(root/'generated'),
 '--output-kind','latents'])==0
meta=json.loads((root/'generated/generation.json').read_text())
assert meta['checkpoint_cycle']==3 and meta['sample_id']=='NEW-genuine-cached-AV-condition'
assert (root/'generated/_SUCCESS').is_file() and not torch.cuda.is_initialized()
assert 'deepspeed' not in sys.modules
print('ACTUAL_ORIGINAL_MIXED_STORAGE_H3_NEW_PROCESS_LATENT_OUTPUT_PASS')
'''
    run = subprocess.run([sys.executable,'-B','-c',script,str(root)],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),
                         capture_output=True,text=True,timeout=180)
    assert run.returncode==0,run.stdout+run.stderr
    assert 'ACTUAL_ORIGINAL_MIXED_STORAGE_H3_NEW_PROCESS_LATENT_OUTPUT_PASS' in run.stdout


@pytest.mark.parametrize('fault',['missing_key','dtype','lora_dtype','shape','budget','schedule','source','native_tiny','geometry'])
def test_native_portable_refusals_before_concrete_model(original_native_layout_case,monkeypatch,fault):
    root,p,*_=original_native_layout_case
    from distillschedule.training import h3_ar_portable as output
    p=deepcopy(p)
    key=next(iter(p['weight_layout']))
    if fault=='missing_key':p['weight_layout'].pop(key)
    elif fault=='dtype':p['weight_layout'][key]['dtype']='torch.float16'
    elif fault=='lora_dtype':
        name=next(n for n in p['weight_layout'] if '.adapters.student.' in n)
        p['weight_layout'][name]['dtype']='torch.bfloat16'
    elif fault=='shape':p['weight_layout'][key]['shape']=[1]
    elif fault=='budget':p['budget']['max_parameters']=1
    elif fault=='schedule':p['schedule']['num_inference_steps']=4
    elif fault=='source':p['source']={}
    elif fault=='native_tiny':p['evidence_kind']=output.NATIVE
    else:p['geometry']['audio_shape'][0]=1
    monkeypatch.setattr(output,'_materialize',lambda *a:pytest.fail('invalid native inference allocated concrete model'))
    with pytest.raises(ValueError):output.build_inference_model(p)
