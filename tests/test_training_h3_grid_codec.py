"""Exact original packed-token/codec ABI; no W6 sampler or GPU training."""
from copy import deepcopy
from pathlib import Path
import json
import os
import subprocess
import sys

import pytest


def test_grid_codec_discovery_is_model_free():
    result=subprocess.run([sys.executable,'-B','-c',
        'from distillschedule.training.h3_grid_codec import build_decoder;import sys;'
        'assert not any(n in sys.modules for n in ("torch","diffusers","diffsynth"))'],
        text=True,capture_output=True,timeout=30)
    assert result.returncode==0,result.stderr


def test_exact_original_packing_inverse_preserves_channel_time_and_spatial_order():
    import torch
    from distillschedule.training.h3_grid_codec import packing_api,unpack_generated
    api,lock=packing_api()
    v=torch.arange(24*7*4*6,dtype=torch.float32).reshape(1,24,7,4,6)/1031
    a=torch.arange(2*32*37,dtype=torch.float32).reshape(2,32,37)/887
    shape=dict(latent_frames=7,latent_height=4,latent_width=6,audio_latents=37,
        video_tokens=(1,42,96),audio_tokens=(1,74,32))
    tokens=dict(video_tokens=api.patchify_video(v).unsqueeze(0),audio_tokens=api.pack_audio(a).unsqueeze(0))
    result=unpack_generated(tokens,shape)
    torch.testing.assert_close(result['video_latents'],v,rtol=0,atol=0)
    torch.testing.assert_close(result['audio_latents'],a,rtol=0,atol=0)
    lock.validate()
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('fault',['dtype','shape','batch','audio_order','nonfinite','grad','mixed','schema'])
def test_invalid_packed_tensor_is_not_silently_cast_or_reshaped(fault):
    import torch
    from distillschedule.training.h3_grid_codec import unpack_generated
    shape=dict(latent_frames=7,latent_height=2,latent_width=2,audio_latents=37,
        video_tokens=(1,7,96),audio_tokens=(1,74,32))
    value=dict(video_tokens=torch.zeros(1,7,96),audio_tokens=torch.zeros(1,74,32))
    if fault=='dtype':value['video_tokens']=value['video_tokens'].bfloat16()
    elif fault=='shape':value['video_tokens']=value['video_tokens'][:,:,:95]
    elif fault=='batch':value['video_tokens']=value['video_tokens'].repeat(2,1,1)
    elif fault=='audio_order':value['audio_tokens']=value['audio_tokens'].permute(0,2,1)
    elif fault=='nonfinite':value['audio_tokens'][0,0,0]=float('nan')
    elif fault=='grad':value['video_tokens'].requires_grad_()
    elif fault=='mixed':value['video_latents']=torch.zeros(1,24,7,2,2)
    else:value['latent_schema']={'video_tokens':{'layout':'BCTHW'}}
    with pytest.raises(ValueError):unpack_generated(value,shape)


def test_optional_codec_does_not_allow_sampler_or_precision_overrides():
    from types import SimpleNamespace
    from distillschedule.training.h3_grid_inference import generate
    for options in ({'num_steps':1},{'precision':'float32'},{'decoder_manifest':{},'sigma':.3}):
        with pytest.raises(ValueError,match='original schedule; no overrides'):
            generate(dict(inference={},student=SimpleNamespace(restored=True,selected=None)),{},options)


from test_training_h3_grid_outputs import actual_grid_case


def pin(path):
    from distillschedule.workloads.best_known.registry import file_sha256
    path=Path(path).resolve()
    return dict(path=str(path),bytes=path.stat().st_size,sha256=file_sha256(path))


@pytest.fixture(scope='module')
def reused_codec():
    # Reuse existing R40 actual-source random codec payloads, never create a
    # replacement codec or claim these test weights are pretrained assets.
    path=Path(os.environ['DMD_H3_GRID_CODEC_TEST_MANIFEST'])
    assert path.is_file()
    return pin(path)


@pytest.mark.parametrize('fault',['frames','fps','audio_length'])
def test_incompatible_codec_domain_rejected_before_loading_models(actual_grid_case,reused_codec,tmp_path,monkeypatch,fault):
    from distillschedule.training import h3_grid_codec as bridge,h3_av_codec
    p=json.loads(Path(reused_codec['path']).read_text())
    if fault=='frames':p['geometry']['video_shape'][0]=23
    elif fault=='fps':p['geometry']['fps']=25
    else:
        p['geometry']['audio_latent_shape'][-1]=2;p['geometry']['audio_shape'][0]=1600
    file=tmp_path/'bad.json';file.write_text(json.dumps(p))
    monkeypatch.setattr(h3_av_codec,'build_decoder',lambda *a:pytest.fail('mismatch allocated codecs'))
    with pytest.raises(ValueError,match='original packed/codec'):
        bridge.build_decoder(dict(manifest=pin(file),grid_source=actual_grid_case.parameters['source'],
                                  geometry=actual_grid_case.parameters['geometry']))


def test_original_full_grid_sampler_current_student_decodes_in_new_public_process(
        actual_grid_case,reused_codec,tmp_path):
    import torch
    from distillschedule.training import h3_grid_inference as grid,h3_grid_codec as bridge
    from distillschedule.training.inference import publish_export
    from distillschedule.training.plugins import load_callable
    from distillschedule.data import import_preencoded
    from distillschedule.training.checkpoint import rng_state
    from test_training_end_to_end import equal
    case=actual_grid_case
    before=rng_state(device='cpu')
    expected_tokens=dict(video_tokens=case.oracle[0],audio_tokens=case.oracle[1])
    comp=bridge.build_decoder(dict(manifest=reused_codec,grid_source=case.parameters['source'],geometry=case.parameters['geometry']))
    from distillschedule.training.h3_av_codec import decode as original_decode
    from distillschedule.training.h3_grid_codec import unpack_generated
    shape,_=grid.geometry(case.api,case.parameters['geometry'],text_dim=5120)
    raw=unpack_generated(expected_tokens,shape)
    expected=original_decode(comp.state,raw,{})
    actual=comp.decode(expected_tokens,{})
    assert torch.equal(expected['video'],actual['video'])
    assert torch.equal(expected['audio']['samples'],actual['audio']['samples'])
    assert actual['video'].shape==(22,32,32,3)
    assert actual['audio']['samples'].shape==(29600,2)
    assert equal(before,rng_state(device='cpu'))
    torch.save({'video':expected['video'],'audio':expected['audio']['samples'],**expected_tokens},tmp_path/'expected.pt')
    del comp
    components=grid.export_components(case.parameters,weights=case.snapshot)
    _,source=load_callable(grid.FACTORY)
    manifest=publish_export(components,tmp_path/'export',model=dict(factory=grid.FACTORY,source=source,
        parameters=case.parameters,metadata=dict(components.metadata)),
        identity=dict(model=source,model_metadata=dict(components.metadata)),cycle=1)
    torch.save(case.feeds,tmp_path/'input.pt')
    file=tmp_path/'input.json';file.write_text(json.dumps(dict(schema_version='distillschedule.data.preencoded.v1',
        records=[dict(id='NEW-GRID-TEXT-CONDITION',features=dict(path=str(tmp_path/'input.pt'),sha256=pin(tmp_path/'input.pt')['sha256']))])))
    prepared=import_preencoded(file,tmp_path/'prepared')
    conf=dict(schema_version='distillschedule.training.v1',model={k:manifest['model'][k] for k in ('factory','parameters')},
        algorithm=dict(factory='distillschedule.training.h3_grid_inference:refuse_training'),
        data=dict(prepared=str(prepared)),execution=dict(adapter='program',mode='reference'),
        trainer=dict(cycles=1,seed=93,device='cpu',output_dir=str(tmp_path/'NOT_TRAINING')))
    (tmp_path/'inference.json').write_text(json.dumps(conf))
    (tmp_path/'parameters.json').write_text(json.dumps(dict(decoder_manifest=reused_codec)))
    script=r'''
import json,sys,wave,torch
from pathlib import Path
torch.set_num_threads(1);root=Path(sys.argv[1])
class NoTraining:
 def find_spec(self,name,path=None,target=None):
  assert name not in ('distillschedule.training.native','distillschedule.training.h3_ar_training',
   'distillschedule.training.h3_ar_portable','distillschedule.training.h3_cached_inference'),name
sys.meta_path.insert(0,NoTraining())
def forbidden(*a,**kw):raise AssertionError('generation constructed optimizer')
torch.optim.Adam.__init__=torch.optim.AdamW.__init__=forbidden
from distillschedule.cli import main
assert main(['generate',str(root/'inference.json'),'--weights',str(root/'export'),'--prepared',
 str(root/'prepared/manifest.json'),'--output',str(root/'media'),'--output-kind','audio_video',
 '--parameters',str(root/'parameters.json')])==0
actual=torch.load(root/'media/generation.pt',weights_only=True)
expected=torch.load(root/'expected.pt',weights_only=True)
for key in ('video_tokens','audio_tokens'):assert torch.equal(actual['generated'][key],expected[key])
assert torch.equal(actual['decoded']['video'],expected['video'])
assert torch.equal(actual['decoded']['audio']['samples'],expected['audio'])
assert len(list((root/'media/frames').glob('*.png')))==22
with wave.open(str(root/'media/audio.wav'),'rb') as f:assert (f.getnchannels(),f.getframerate(),f.getnframes())==(2,32000,29600)
assert (root/'media/_SUCCESS').is_file() and not torch.cuda.is_initialized()
print('PASS_ORIGINAL_GRID_FOURSTEP_CURRENT_STUDENT_PACKED_ABI_REAL_CODECS_FRESH_PROCESS_CPU')
'''
    result=subprocess.run([sys.executable,'-B','-c',script,str(tmp_path)],text=True,capture_output=True,timeout=240)
    assert result.returncode==0,result.stdout+result.stderr
    assert equal(before,rng_state(device='cpu'))
