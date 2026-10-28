"""Original H3 encoder source/ABI + raw data, not pretrained quality or CUDA."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import wave

import numpy as np
import pytest
import torch

from distillschedule.workloads.best_known.registry import file_sha256, repository_root
from test_h3_av_codec_target_cpu import SOURCE, pin


def manifest(root):
    from distillschedule.training import h3_av_encoder as enc
    from distillschedule.training import h3_av_codec as codec
    weight = dict(path=str(root/'weights.safetensors'), bytes=16, sha256='a'*64)
    return dict(schema_version=enc.SCHEMA, precision=enc.PRECISION, storage_dtype='float32',
        evidence_kind=codec.FIXTURE, source=dict(root=str(SOURCE), files=deepcopy(codec.SOURCE_FILES),
            provenance=dict(origin='operator supplied exact H3 snapshot', license='external source used in place', terms_accepted=True)),
        video_encoder={**deepcopy(enc.VIDEO_NATIVE), 'ch':32, 'num_res_blocks':1},
        audio_encoder={**deepcopy(enc.AUDIO_NATIVE), 'encoder_dim':8},
        weights=dict(video=[deepcopy(weight)],audio=[deepcopy(weight)]),
        budget=dict(max_parameters=100_000_000, max_loaded_bytes=2**31,
            max_input_elements=10_000_000, max_activation_elements=100_000_000,
            max_attention_elements=10_000_000, host_reserve_bytes=2**20))


def test_encoder_contract_is_available_without_importing_original_or_cuda(tmp_path):
    from distillschedule.training import h3_av_encoder as enc
    p=enc.validate_manifest(manifest(tmp_path))
    assert p['precision']=='CPU_FP32_ORIGINAL_H3_ENCODE'
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('fault',['unknown','schema','precision','source','terms','production_tiny','stride','nested_bool','channels','audio_rate','audio_hop','budget','weight'])
def test_encoder_invalid_manifest_fails_before_source_import(tmp_path,monkeypatch,fault):
    from distillschedule.training import h3_av_encoder as enc
    from distillschedule.training import h3_av_codec as codec
    p=manifest(tmp_path)
    if fault=='unknown':p['silent_resize']=True
    elif fault=='schema':p['schema_version']='bad'
    elif fault=='precision':p['precision']='CUDA_BF16'
    elif fault=='source':p['source']['files'][next(iter(p['source']['files']))]='0'*64
    elif fault=='terms':p['source']['provenance']['terms_accepted']=False
    elif fault=='production_tiny':p['evidence_kind']=codec.LOCAL
    elif fault=='stride':p['video_encoder']['space_down']=[2,2,2,1,1,1]
    elif fault=='nested_bool':p['video_encoder']['time_down'][0]=True
    elif fault=='channels':p['video_encoder']['embed_dim']=16
    elif fault=='audio_rate':p['audio_encoder']['sample_rate']=16000
    elif fault=='audio_hop':p['audio_encoder']['encoder_rates']=[2,4,8,8]
    elif fault=='budget':p['budget']['max_parameters']=True
    else:p['weights']['video'][0]['sha256']='bad'
    monkeypatch.setattr(codec,'source_api',lambda *_:pytest.fail('invalid manifest imported original codec'))
    with pytest.raises(ValueError):enc.validate_manifest(p)


@pytest.fixture(scope='module')
def original_encoder_bundle(tmp_path_factory):
    from safetensors.torch import save_file
    from distillschedule.training import h3_av_encoder as enc, h3_av_codec as codec
    if not SOURCE.is_dir():pytest.skip('operator original codec source is absent locally')
    root=tmp_path_factory.mktemp('actual-original-h3-encoder');torch.set_num_threads(1)
    reused=os.environ.get('DMD_H3_RAW_ENCODER_TEST_MANIFEST')
    if reused:
        from distillschedule.training.h3_cached_assets import read_json,checked_pin
        p=enc.validate_manifest(read_json(reused))
        for pins in p['weights'].values():
            for ref in pins:checked_pin(ref)
        (root/'encoder.json').write_text(json.dumps(p,indent=2))
        api=codec.source_api(p['source']);models,_=enc.meta_encoders(p,api)
        counts={role:sum(t.numel() for t in model.state_dict().values()) for role,model in models.items()}
        del models
        return root,p,api,counts
    p=manifest(root);api=codec.source_api(p['source']);models,_=enc.meta_encoders(p,api)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1147)
        for role,model in models.items():
            model.to_empty(device='cpu')
            with torch.no_grad():
                for n,t in model.state_dict().items():
                    t.uniform_(-.02,.02)
                    # WeightNorm uses weight_v's norm; do not create all-zero directions.
            file=root/(role+'.safetensors');save_file(model.state_dict(),file);p['weights'][role]=[pin(file)]
    (root/'encoder.json').write_text(json.dumps(p,indent=2))
    counts={role:sum(t.numel() for t in model.state_dict().values()) for role,model in models.items()}
    del models
    return root,p,api,counts


def media_record(root, stem='fresh-current', *, phase=0):
    from PIL import Image
    frames=root/(stem+'-frames');frames.mkdir()
    for i in range(22):
        y,x=np.indices((32,32));image=np.stack(((x+i*3+phase)%256,(y*2+i+phase)%256,(x+y+phase)%256),axis=-1).astype(np.uint8)
        Image.fromarray(image).save(frames/f'{i:04d}.png')
    path=root/(stem+'.wav')
    samples=np.stack((np.arange(2401)%101-50,np.arange(2401)%53-26),axis=1).astype('<i2')
    with wave.open(str(path),'wb') as f:
        f.setnchannels(2);f.setsampwidth(2);f.setframerate(32000);f.writeframes(samples.tobytes())
    from distillschedule.data.manifest import source_identity
    return dict(video=dict(path=str(frames),sha256=source_identity(frames,frames=True)['sha256']), audio=pin(path),fps=24)


def test_actual_original_encoder_means_padding_stereo_and_rng(original_encoder_bundle):
    from distillschedule.training import h3_av_encoder as enc
    from distillschedule.training.checkpoint import rng_state
    from test_training_end_to_end import equal
    root,p,api,counts=original_encoder_bundle
    before=rng_state(device='cpu');bundle=enc.build_encoder(dict(manifest=pin(root/'encoder.json')))
    assert equal(before,rng_state(device='cpu'))
    assert bundle['video'].__class__.__name__=='MiniMaxH3VideoVAE'
    assert bundle['audio'].__class__.__name__=='MiniMaxH3AudioVAE'
    assert not hasattr(bundle['video'],'decoder') and not hasattr(bundle['audio'],'decoder')
    assert not hasattr(bundle['audio'],'logs_proj')
    assert all(t.device.type=='cpu' and not t.requires_grad for m in (bundle['video'],bundle['audio']) for t in m.parameters())
    gen=torch.Generator().manual_seed(118)
    pixels=torch.randint(0,256,(22,32,32,3),dtype=torch.uint8,generator=gen)
    audio=dict(samples=torch.randn((2401,2),generator=gen)*.1,sample_rate=32000)
    original_pixels,original_audio=pixels.clone(),audio['samples'].clone()
    with torch.no_grad():
        v=bundle['video'].encode_video(pixels.permute(3,0,1,2).unsqueeze(0).float()/255,dtype=torch.float32)
        a=bundle['audio'].encode_audio(audio['samples'].t().contiguous(),dtype=torch.float32)
    actual=enc.encode(bundle,pixels,audio,fps=24)
    assert torch.equal(v,actual['video_latents']) and torch.equal(a,actual['audio_latents'])
    assert v.shape==(1,24,7,2,2) and a.shape==(2,32,4)
    assert actual['encoding']['audio_right_pad_samples']==799
    assert actual['encoding']['video_native_tail_repeat_frames']==12
    assert actual['encoding']['normalization']=='ORIGINAL_ENCODE_VIDEO_AND_ENCODE_AUDIO_MEANS_ONCE'
    assert torch.equal(pixels,original_pixels) and torch.equal(audio['samples'],original_audio)
    assert equal(before,rng_state(device='cpu')) and not torch.cuda.is_initialized()
    (root/'actual-encoder-counts.json').write_text(json.dumps(counts))


@pytest.mark.parametrize('fault',['mono','rate','range','dtype','frames','spatial','fps','grad','nonfinite'])
def test_encoding_rejects_implicit_media_changes(original_encoder_bundle,fault):
    from distillschedule.training import h3_av_encoder as enc
    root,p,_,_=original_encoder_bundle
    # No parameter load is needed to prove raw-input refusal before original encode.
    bundle=dict(spec=p)
    video=torch.zeros((22,32,32,3),dtype=torch.uint8)
    audio=dict(samples=torch.zeros((2401,2)),sample_rate=32000);fps=24
    if fault=='mono':audio['samples']=audio['samples'][:,:1]
    elif fault=='rate':audio['sample_rate']=16000
    elif fault=='range':audio['samples'][0,0]=2
    elif fault=='dtype':video=video.float()
    elif fault=='frames':video=video[:21]
    elif fault=='spatial':video=video[:,:,:31]
    elif fault=='fps':fps=30
    elif fault=='grad':audio['samples'].requires_grad_(True)
    else:audio['samples'][0,0]=float('nan')
    with pytest.raises(ValueError):enc.encode(bundle,video,audio,fps=fps)


def test_raw_previous_current_public_prepare_and_actual_update(original_encoder_bundle):
    from safetensors.torch import save_file
    from distillschedule.training import h3_av_encoder as enc
    from distillschedule.data import PreparedDataset
    from distillschedule.data.h3_cached_inputs import decode_row
    from test_training_h3_cached_public_cpu import make_case,request,json_file
    from test_h3_cached_original_cpu import original
    from distillschedule.training.session import train
    from distillschedule.training.checkpoint import rng_state
    from test_training_end_to_end import equal
    root,p,_,_=original_encoder_bundle
    folder=root/'public-raw';folder.mkdir()
    current=media_record(folder,'current',phase=19);previous=media_record(folder,'previous',phase=71)
    gen=torch.Generator().manual_seed(531)
    prompt=folder/'prompt.safetensors';save_file(dict(prompt_embeds=torch.randn((3,5120),generator=gen),text_token_tags=torch.ones(3,dtype=torch.int64)),prompt)
    raw=dict(schema_version='distillschedule.original_h3_raw_av_inputs.v1',encoder=dict(manifest=pin(root/'encoder.json')),
             records=[dict(id='new-raw-av-pair',current=current,previous=previous,prompt=pin(prompt))])
    rawpath=json_file(folder/'raw.json',raw)
    script='''import json,sys,torch
from pathlib import Path
from distillschedule.cli import main
torch.set_num_threads(1)
p=Path(sys.argv[1]);before=torch.random.get_rng_state().clone()
assert main(['data','prepare',str(p/'raw.json'),'--preparer','distillschedule.data.h3_raw_inputs:prepare_inputs','--output',str(p/'prepared-raw')])==0
assert torch.equal(before,torch.random.get_rng_state())
assert not torch.cuda.is_initialized()
'''
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',WORLD_SIZE='1',RANK='0',LOCAL_RANK='0',LOCAL_WORLD_SIZE='1')
    result=subprocess.run([sys.executable,'-B','-c',script,str(folder)],cwd=repository_root(),env=env,text=True,capture_output=True,timeout=250)
    assert result.returncode==0,result.stdout+result.stderr
    prepared=folder/'prepared-raw/prepared/manifest.json';data=PreparedDataset(prepared)
    assert data.validate(sources=True)['status']=='PASS'
    geometry=data.manifest['metadata']['geometry'];assert geometry['video_shape']==[1,24,7,2,2]
    row=decode_row(data.sample(0),absolute_step=0,geometry=geometry)
    inputs,negative=original().dmd2.prepare_batch(row['raw'],1.0)
    assert negative is None and inputs['prompt_embeds'].shape==(3,5120)
    assert inputs['ref_visual_anchor'].shape==(7,96) and inputs['ref_audio_anchor'].shape==(8,32)
    receipt=json.loads((folder/'prepared-raw/encoding.json').read_text())
    assert receipt['schema_version']=='distillschedule.original_h3_raw_encoding_receipt.v1'
    assert receipt['records'][0]['current']['audio_right_pad_samples']==799
    assert receipt['records'][0]['previous']['audio_right_pad_samples']==799
    assert receipt['GPU_started']==0 and (folder/'prepared-raw/_SUCCESS').is_file()
    assets,_=make_case(folder/'model',count=1,gan_start=500)
    a=json.loads(assets.read_text());a['geometry']=geometry;a['budget']['max_input_elements']=200_000;json_file(assets,a)
    config=request(folder/'training',assets,prepared,1)
    outcome=train(config)
    assert outcome['completed_cycles']==1 and outcome['final_versions']=={'native_update':1}
    metrics=json.loads((folder/'training/metrics.jsonl').read_text())['metrics']
    assert all(np.isfinite(v) for v in metrics.values())
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('fault',['missing_previous','unknown_condition','changed_frames','mono','fps','prompt_dtype','prompt_tags','mixed_geometry'])
def test_raw_contract_errors_do_not_materialize_encoder_or_publish(original_encoder_bundle,tmp_path,monkeypatch,fault):
    from safetensors.torch import save_file
    from distillschedule.training import h3_av_encoder as enc
    from distillschedule.data import h3_raw_inputs as raw
    root,p,_,_=original_encoder_bundle
    current=media_record(tmp_path,'current',phase=1);previous=media_record(tmp_path,'previous',phase=2)
    prompt=tmp_path/'prompt.safetensors'
    values=dict(prompt_embeds=torch.zeros((3,5120)),text_token_tags=torch.ones(3,dtype=torch.int64))
    if fault=='prompt_dtype':values['prompt_embeds']=values['prompt_embeds'].half()
    if fault=='prompt_tags':values['text_token_tags'][0]=0
    save_file(values,prompt)
    row=dict(id='raw-refusal',current=current,previous=previous,prompt=pin(prompt))
    if fault=='missing_previous':row.pop('previous')
    elif fault=='unknown_condition':row['invented_kv_cache']=True
    elif fault=='changed_frames':row['current']['video']['sha256']='0'*64
    elif fault=='fps':row['current']['fps']=30
    elif fault=='mono':
        wav=Path(row['current']['audio']['path'])
        with wave.open(str(wav),'wb') as f:
            f.setnchannels(1);f.setsampwidth(2);f.setframerate(32000);f.writeframes(np.zeros(2401,dtype='<i2').tobytes())
        row['current']['audio']=pin(wav)
    records=[row]
    if fault=='mixed_geometry':
        other=deepcopy(row);other['id']='other-prompt-length'
        second=tmp_path/'other-prompt.safetensors';save_file(dict(prompt_embeds=torch.zeros((4,5120)),text_token_tags=torch.ones(4,dtype=torch.int64)),second)
        other['prompt']=pin(second);records.append(other)
    path=tmp_path/'raw.json';path.write_text(json.dumps(dict(schema_version=raw.SCHEMA,encoder=dict(manifest=pin(root/'encoder.json')),records=records)))
    monkeypatch.setattr(enc,'build_encoder',lambda *_:pytest.fail('invalid raw contract loaded codec weights'))
    with pytest.raises(ValueError):raw.prepare_inputs(path,tmp_path/'refused')
    assert not (tmp_path/'refused').exists()


def test_encoder_discovery_imports_no_model_or_torch():
    script='''import sys
class NoModels:
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('torch','torchvision','diffsynth','transformers'):
   raise AssertionError('discovery imported '+name)
sys.meta_path.insert(0,NoModels())
from distillschedule.training import h3_av_encoder
from distillschedule.data import h3_raw_inputs
assert h3_av_encoder.PRECISION=='CPU_FP32_ORIGINAL_H3_ENCODE'
assert 'torch' not in sys.modules
'''
    result=subprocess.run([sys.executable,'-B','-c',script],cwd=repository_root(),env=os.environ,text=True,capture_output=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr


def test_short_previous_chunk_uses_native_temporal_padding(original_encoder_bundle):
    from distillschedule.training import h3_av_encoder as enc
    root,_,_,_=original_encoder_bundle
    bundle=enc.build_encoder(dict(manifest=pin(root/'encoder.json')))
    pixels=torch.arange(5*32*32*3,dtype=torch.int32).remainder(256).to(torch.uint8).reshape(5,32,32,3)
    audio=dict(samples=torch.zeros((1600,2)),sample_rate=32000)
    result=enc.encode(bundle,pixels,audio,fps=24)
    assert result['video_latents'].shape==(1,24,2,2,2)
    assert result['audio_latents'].shape==(2,32,2)
    assert result['encoding']['video_native_tail_repeat_frames']==12
    assert result['encoding']['audio_right_pad_samples']==0
