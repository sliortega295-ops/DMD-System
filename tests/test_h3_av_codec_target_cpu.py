"""New real H3 decoder proof, not Student training/native GPU/quality evidence."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from distillschedule.workloads.best_known.registry import file_sha256, repository_root
from test_h3_av_codec import manifest

SOURCE = Path(os.environ.get('DISTILLSCHEDULE_H3_AV_CODEC_SOURCE',
    '/share/project/shuchenweng/yongyan_liu/research_artifacts/DMD-System/h3_diffsynth_dmd2_world16_20261010/assets/exact_reference_source_r2'))
pytestmark = pytest.mark.skipif(not SOURCE.is_dir(), reason='actual external original H3 codec source is not present locally')


def pin(path):
    return dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=file_sha256(path))


@pytest.fixture(scope='module')
def original_codec_bundle(tmp_path_factory):
    from safetensors.torch import save_file
    from distillschedule.training import h3_av_codec as codec
    root = tmp_path_factory.mktemp('actual-original-h3-codec')
    torch.set_num_threads(1)
    assert str(torch.__version__).split('+')[0] == '2.10.0'
    p = manifest(root)
    p['source']['root'] = str(SOURCE)
    p['source']['provenance'] = dict(origin='operator-provided exact DiffSynth snapshot from native assets',
        license='external source, used in-place only, not redistributed by this test', terms_accepted=True)
    api = codec.source_api(p['source'])
    models, _ = codec.meta_decoders(p, api)
    # Only fixture *weights* are random. The original actual source classes,
    # normalization, overlap, SDPA and full BigVGAN decoder stay unchanged.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(509)
        for role, model in models.items():
            model.to_empty(device='cpu')
            with torch.no_grad():
                for tensor in model.state_dict().values():
                    tensor.uniform_(-.01, .01)
            model.requires_grad_(False).eval()
            path = root / (role + '.safetensors')
            save_file(model.state_dict(), path)
            p['weights'][role] = [pin(path)]
    path = root / 'codec.json'; path.write_text(json.dumps(p, indent=2))
    assert sum(t.numel() for t in models['audio'].state_dict().values()) == 64_939_353
    del models
    return root, p, path


def test_original_h3_decoders_match_direct_methods_and_public_new_process(original_codec_bundle):
    from distillschedule.training import h3_av_codec as codec
    from distillschedule.training.checkpoint import rng_state
    from test_training_end_to_end import equal
    from distillschedule.data import import_preencoded
    root, p, path = original_codec_bundle
    before = rng_state(device='cpu')
    comp = codec.build_decoder(dict(manifest=pin(path)))
    assert equal(before, rng_state(device='cpu'))
    assert comp.state['video_codec'].__class__.__name__ == 'MiniMaxH3VideoVAE'
    assert comp.state['video_codec'].decoder.__class__.__name__ == 'ViT3DDecoder'
    assert comp.state['audio_codec'].decoder.__class__.__name__ == 'BigVGAN'
    assert all(not t.requires_grad and t.device.type == 'cpu' for role in ('video_codec','audio_codec')
               for t in comp.state[role].parameters())
    generator = torch.Generator().manual_seed(986)
    inputs = dict(video_latents=torch.randn(p['geometry']['video_latent_shape'], generator=generator)*.1,
                  audio_latents=torch.randn(p['geometry']['audio_latent_shape'], generator=generator)*.1)
    originals = (inputs['video_latents'].clone(), inputs['audio_latents'].clone())
    v, a = comp.state['video_codec'], comp.state['audio_codec']
    methods = (type(v).decode_video.__code__, type(a).decode_audio.__code__)
    with torch.no_grad():
        direct_video = v.decode_video(inputs['video_latents'], dtype=torch.float32)
        direct_audio = a.decode_audio(inputs['audio_latents'], dtype=torch.float32)
        actual = comp.decode(inputs, {})
    expected = (direct_video[0].permute(1,2,3,0).clamp(0,1)*255).round().to(torch.uint8)
    assert torch.equal(actual['video'], expected)
    assert torch.equal(actual['audio']['samples'], direct_audio[0].t())
    assert actual['video'].shape == (22,32,32,3)
    assert actual['audio']['samples'].shape == (1600,2)
    assert actual['decoder']['latent_clamp'] is False
    assert all(torch.equal(x, y) for x,y in zip(originals, inputs.values()))
    assert methods == (type(v).decode_video.__code__, type(a).decode_audio.__code__)
    assert equal(before, rng_state(device='cpu'))
    torch.save(inputs, root / 'latents.pt')
    (root / 'input.json').write_text(json.dumps(dict(schema_version='distillschedule.data.preencoded.v1',
        records=[dict(id='NEW-H3-codec-condition', features=dict(path='latents.pt', sha256=file_sha256(root/'latents.pt')))],
        metadata=dict(evidence_kind=codec.FIXTURE, scope='CODEC_ONLY_NO_STUDENT_TRAINING'))))
    import_preencoded(root/'input.json', root/'prepared')
    config = dict(schema_version='distillschedule.decode_prepared.v1',
        decoder=dict(factory='distillschedule.training.h3_av_codec:build_decoder',parameters=dict(manifest=pin(path))),
        data=dict(prepared='prepared/manifest.json',sample_index=0),output_kind='audio_video')
    (root/'decode.json').write_text(json.dumps(config))
    del comp, v, a
    script = r'''
import json,sys,torch,wave
from pathlib import Path
root=Path(sys.argv[1]);torch.set_num_threads(1)
class NoTraining:
 def find_spec(self,name,path=None,target=None):
  assert name not in ('distillschedule.training.native','distillschedule.training.h3_ar_training',
   'distillschedule.strategies.h3_ar_session','distillschedule.training.h3_cached_owner'),name
sys.meta_path.insert(0,NoTraining())
def no_optimizer(*a,**kw):raise AssertionError('codec constructed an optimizer')
torch.optim.Adam.__init__=torch.optim.AdamW.__init__=no_optimizer
from distillschedule.cli import main
assert main(['decode',str(root/'decode.json'),'--output',str(root/'media')])==0
meta=json.loads((root/'media/generation.json').read_text())
assert meta['status']=='DECODED' and meta['sample_id']=='NEW-H3-codec-condition'
assert len(list((root/'media/frames').glob('*.png')))==22
with wave.open(str(root/'media/audio.wav'),'rb') as f:
 assert (f.getframerate(),f.getnchannels(),f.getnframes())==(32000,2,1600)
assert (root/'media/_SUCCESS').is_file()
assert not torch.cuda.is_initialized()
print('ACTUAL_ORIGINAL_H3_VIDEO_AND_FULL_BIGVGAN_PUBLIC_DECODE_PASS_NO_TRAINING_GPU_OR_QUALITY')
'''
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='',WORLD_SIZE='1',LOCAL_WORLD_SIZE='1',RANK='0',LOCAL_RANK='0')
    result = subprocess.run([sys.executable,'-B','-c',script,str(root)],cwd=repository_root(),env=env,
                            text=True,capture_output=True,timeout=300)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'ACTUAL_ORIGINAL_H3_VIDEO_AND_FULL_BIGVGAN_PUBLIC_DECODE_PASS' in result.stdout
    assert not torch.cuda.is_initialized()


def test_native_codec_capacity_stops_before_weight_materialization(original_codec_bundle, monkeypatch):
    from distillschedule.training import h3_av_codec as codec
    root, p, _ = original_codec_bundle
    p = deepcopy(p);p['budget']['max_parameters']=1
    path=root/'too-small.json';path.write_text(json.dumps(p))
    monkeypatch.setattr(codec,'_load_weights',lambda *a:pytest.fail('capacity failure allocated actual weights'))
    with pytest.raises(ValueError,match='capacity exceeded'):
        codec.build_decoder(dict(manifest=pin(path)))


@pytest.fixture(scope='module')
def original_fl2va_bundle(original_codec_bundle):
    from safetensors.torch import load_file,save_file
    from distillschedule.training import h3_av_weight_import as native
    root,p,canonical_path=original_codec_bundle
    values=load_file(p['weights']['video'][0]['path'])
    raw={k.removesuffix('.weight') if k=='decoder.register_tokens.weight'
        or k.endswith(('.scale1.weight','.scale2.weight')) else k:v for k,v in values.items()}
    raw['decoder.mask_token']=torch.full((1,1,values['decoder.register_tokens.weight'].shape[-1]),.375)
    raw_path=root/'fl2va-video.safetensors';save_file(raw,raw_path)
    raw_codec=deepcopy(p);raw_codec['weights']['video']=[pin(raw_path)]
    codec_path=root/'fl2va-codec.json';codec_path.write_text(json.dumps(raw_codec))
    source=Path(os.environ.get('DISTILLSCHEDULE_H3_AV_CODEC_CONVERTER_SOURCE',str(SOURCE)))
    spec=dict(schema_version=native.SCHEMA,weight_format=native.FORMAT,codec_manifest=pin(codec_path),
        converter_source=dict(root=str(source),files=deepcopy(native.CONVERTER_FILES),
            provenance=deepcopy(p['source']['provenance'])))
    path=root/'fl2va-import.json';path.write_text(json.dumps(spec))
    return root,p,canonical_path,raw_codec,spec,path,raw


def test_original_raw_fl2va_strict_load_and_decode_equal_canonical_without_changing_default(original_fl2va_bundle):
    from distillschedule.training import h3_av_codec as codec
    from distillschedule.training.checkpoint import rng_state
    from test_training_end_to_end import equal
    root,p,canonical_path,raw_codec,spec,path,raw=original_fl2va_bundle
    before=rng_state(device='cpu')
    with pytest.raises(ValueError,match='safe header/key/shape/storage mismatch'):
        codec.build_decoder(dict(manifest=pin(Path(spec['codec_manifest']['path']))))
    assert equal(before,rng_state(device='cpu'))
    a=codec.build_decoder(dict(manifest=pin(canonical_path)))
    b=codec.build_fl2va_decoder(dict(manifest=pin(path)))
    for role in ('video_codec','audio_codec'):
        assert type(a.state[role]) is type(b.state[role])
        actual_state=b.state[role].state_dict()
        assert all(torch.equal(t,actual_state[n]) for n,t in a.state[role].state_dict().items())
    assert sum(t.numel() for t in b.state['audio_codec'].state_dict().values())==64_939_353
    gen=torch.Generator().manual_seed(986)
    inputs=dict(video_latents=torch.randn(p['geometry']['video_latent_shape'],generator=gen)*.1,
                audio_latents=torch.randn(p['geometry']['audio_latent_shape'],generator=gen)*.1)
    x,y=a.decode(inputs,{}),b.decode(inputs,{})
    assert torch.equal(x['video'],y['video']) and torch.equal(x['audio']['samples'],y['audio']['samples'])
    m=b.metadata['weight_import']
    assert m['format']==spec['weight_format'] and m['converter_source']==spec['converter_source']
    assert m['roles'][0]['raw_keys']==len(raw) and len(m['roles'][0]['renamed'])==3
    assert m['roles'][0]['excluded']==['decoder.mask_token'] and m['full_weight_payload_sha_verified'] is True
    assert equal(before,rng_state(device='cpu')) and not torch.cuda.is_initialized()


def test_raw_fl2va_public_decode_new_process_and_exact_output_contract(original_fl2va_bundle):
    from distillschedule.data import import_preencoded
    root,p,_,_,_,path,_=original_fl2va_bundle
    inputs=dict(video_latents=torch.zeros(p['geometry']['video_latent_shape']),
                audio_latents=torch.zeros(p['geometry']['audio_latent_shape']))
    payload=root/'fl2va-latents.pt';torch.save(inputs,payload)
    source=root/'fl2va-input.json'
    source.write_text(json.dumps(dict(schema_version='distillschedule.data.preencoded.v1',
        records=[dict(id='SYNTHETIC_ORIGINAL_FL2VA_CODEC_ONLY',features=dict(path=payload.name,sha256=file_sha256(payload)))],
        metadata=dict(evidence_kind='SYNTHETIC_TEST_ONLY',purpose='NO_STUDENT_OR_QUALITY'))))
    import_preencoded(source,root/'fl2va-prepared')
    request=root/'fl2va-decode.json';request.write_text(json.dumps(dict(schema_version='distillschedule.decode_prepared.v1',
        decoder=dict(factory='distillschedule.training.h3_av_codec:build_fl2va_decoder',parameters=dict(manifest=pin(path))),
        data=dict(prepared='fl2va-prepared/manifest.json',sample_index=0),output_kind='audio_video')))
    code='''
import json,sys,torch,wave
from pathlib import Path
torch.set_num_threads(1)
from distillschedule.cli import main
root=Path(sys.argv[1])
assert main(['decode',str(root/'fl2va-decode.json'),'--output',str(root/'fl2va-media')])==0
result=json.loads((root/'fl2va-media/generation.json').read_text())
assert result['status']=='DECODED' and result['decoder_source']['reference'].endswith(':build_fl2va_decoder')
assert result['decoder_metadata']['weight_import']['full_weight_payload_sha_verified'] is True
assert result['decoder_metadata']['evidence_kind']=='RANDOM_ORIGINAL_H3_CODEC_NOT_PRETRAINED'
assert len(list((root/'fl2va-media/frames').glob('*.png')))==22
with wave.open(str(root/'fl2va-media/audio.wav'),'rb') as f:
 assert (f.getframerate(),f.getnchannels(),f.getnframes())==(32000,2,1600)
assert not torch.cuda.is_initialized()
print('PUBLIC_ORIGINAL_FL2VA_SYNTHETIC_CODEC_PASS_NOT_PRETRAINED_OR_QUALITY')
'''
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',WORLD_SIZE='1',LOCAL_WORLD_SIZE='1',RANK='0',LOCAL_RANK='0')
    result=subprocess.run([sys.executable,'-B','-c',code,str(root)],env=env,cwd=Path(__file__).resolve().parents[1],
        text=True,capture_output=True,timeout=300)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'PUBLIC_ORIGINAL_FL2VA_SYNTHETIC_CODEC_PASS' in result.stdout
    assert (root/'fl2va-media/_SUCCESS').read_text().strip()==file_sha256(root/'fl2va-media/generation.json')


@pytest.mark.parametrize('fault',['missing','extra','shape','dtype','nonfinite','hash','duplicate','mask_shape'])
def test_raw_fl2va_real_decoder_refusals(original_fl2va_bundle,tmp_path,fault):
    from safetensors.torch import save_file
    from distillschedule.training import h3_av_codec as codec
    from distillschedule.training.checkpoint import rng_state
    from test_training_end_to_end import equal
    _,_,_,raw_codec,spec,_,raw=original_fl2va_bundle
    values={k:v.clone() for k,v in raw.items()};key='decoder.register_tokens'
    if fault=='missing':del values[key]
    elif fault=='extra':values['unrecognized']=torch.zeros(1)
    elif fault=='shape':values[key]=torch.zeros(1,4,17)
    elif fault=='dtype':values[key]=values[key].half()
    elif fault=='nonfinite':values[key][0,0,0]=float('nan')
    elif fault=='mask_shape':values['decoder.mask_token']=torch.zeros(1,2,16)
    weights=tmp_path/'bad.safetensors';save_file(values,weights)
    value=deepcopy(raw_codec);ref=pin(weights)
    if fault=='hash':ref['sha256']='0'*64
    value['weights']['video']=[ref,deepcopy(ref)] if fault=='duplicate' else [ref]
    config=tmp_path/'codec.json';config.write_text(json.dumps(value))
    view=deepcopy(spec);view['codec_manifest']=pin(config)
    manifest_path=tmp_path/'import.json';manifest_path.write_text(json.dumps(view))
    before=rng_state(device='cpu')
    with pytest.raises(ValueError):codec.build_fl2va_decoder(dict(manifest=pin(manifest_path)))
    assert equal(before,rng_state(device='cpu'))


def test_raw_fl2va_three_audio_biases_use_original_attention(original_fl2va_bundle,tmp_path):
    from safetensors.torch import save_file
    from distillschedule.training import h3_av_codec as codec,h3_av_weight_import as native
    _,p,_,_,spec,_,_=original_fl2va_bundle
    api=codec.source_api(p['source'])
    def model():
        result=torch.nn.Module();result.pre_block=torch.nn.Module()
        with torch.device('meta'):result.pre_block.attn=api.audio.CausalAttention(16,16,2)
        return result
    a,b=model(),model();gen=torch.Generator().manual_seed(691)
    values={k:torch.empty(v.shape,dtype=v.dtype).uniform_(-.01,.01,generator=gen) for k,v in a.state_dict().items()}
    names=('pre_block.attn.q_bias.weight','pre_block.attn.v_bias.weight','pre_block.attn.zero_k_bias.weight')
    raw={k.removesuffix('.weight') if k in names else k:v for k,v in values.items()}
    canonical_file,raw_file=tmp_path/'canonical.safetensors',tmp_path/'raw.safetensors'
    save_file(values,canonical_file);save_file(raw,raw_file)
    layout={k:(tuple(t.shape),t.dtype) for k,t in a.state_dict().items()}
    codec._load_weights(a,[pin(canonical_file)],layout,'float32')
    result=native._load_weights('audio',b,[pin(raw_file)],layout,'float32',native.converter_api(spec['converter_source']))
    assert len(result['renamed'])==3 and result['excluded']==[]
    x=torch.randn(1,3,16,generator=gen)
    assert torch.equal(a.pre_block.attn(x),b.pre_block.attn(x))


def test_raw_fl2va_header_preflight_never_hashes_or_materializes_weight_payload(original_fl2va_bundle,monkeypatch):
    import safetensors
    from distillschedule.training import h3_cached_assets as assets,h3_av_weight_import as native
    from distillschedule.training.checkpoint import rng_state
    from test_training_end_to_end import equal
    _,_,_,_,_,path,_=original_fl2va_bundle
    real_open=safetensors.safe_open
    class HeaderOnly:
        def __init__(self,*a,**kw):self.inner=real_open(*a,**kw)
        def __enter__(self):self.inner.__enter__();return self
        def __exit__(self,*a):return self.inner.__exit__(*a)
        def keys(self):return self.inner.keys()
        def get_slice(self,name):return self.inner.get_slice(name)
        def get_tensor(self,name):pytest.fail('header preflight read weight tensor')
    check=assets.checked_pin
    def small_only(ref):
        assert not ref['path'].endswith('.safetensors'),'header preflight rehashed full weight payload'
        return check(ref)
    monkeypatch.setattr(safetensors,'safe_open',HeaderOnly);monkeypatch.setattr(assets,'checked_pin',small_only)
    before=rng_state(device='cpu');result=native.inspect_headers(dict(manifest=pin(path)))
    assert result['status']=='HEADERS_ONLY_NOT_WEIGHT_HASH_LOAD_OR_DECODE'
    assert result['full_weight_payload_sha_verified'] is result['concrete_model_weights_loaded'] is False
    assert equal(before,rng_state(device='cpu'))


@pytest.mark.parametrize('fault',['missing','changed','symlink'])
def test_raw_fl2va_converter_source_drift_refused_before_model_construction(original_fl2va_bundle,tmp_path,monkeypatch,fault):
    import shutil
    from distillschedule.training import h3_av_codec as codec,h3_av_weight_import as native
    _,_,_,_,spec,_,_=original_fl2va_bundle
    source=Path(spec['converter_source']['root']);copy=tmp_path/'source'
    for name in native.CONVERTER_FILES:
        target=copy/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/name,target)
    changed=copy/next(iter(native.CONVERTER_FILES))
    if fault=='missing':changed.unlink()
    elif fault=='changed':changed.write_text(changed.read_text()+'\n')
    else:
        moved=tmp_path/'linked.py';changed.rename(moved);changed.symlink_to(moved)
    view=deepcopy(spec);view['converter_source']['root']=str(copy)
    path=tmp_path/'import.json';path.write_text(json.dumps(view))
    monkeypatch.setattr(codec,'meta_decoders',lambda *a:pytest.fail('source drift constructed model metadata'))
    with pytest.raises(ValueError):codec.build_fl2va_decoder(dict(manifest=pin(path)))


def test_raw_fl2va_explicit_bf16_storage_lifts_exactly_to_original_fp32(original_fl2va_bundle,tmp_path):
    from safetensors.torch import save_file
    from distillschedule.training import h3_av_codec as codec,h3_av_weight_import as native
    _,p,_,_,spec,_,raw=original_fl2va_bundle
    api=codec.source_api(p['source']);models,original=codec.meta_decoders(p,api)
    model=models['video']
    values={k:v.bfloat16() for k,v in raw.items()}
    path=tmp_path/'bf16.safetensors';save_file(values,path)
    native._load_weights('video',model,[pin(path)],original['video'],'bfloat16_to_float32',native.converter_api(spec['converter_source']))
    for name,tensor in model.state_dict().items():
        key=name.removesuffix('.weight') if name=='decoder.register_tokens.weight' or name.endswith(('.scale1.weight','.scale2.weight')) else name
        assert tensor.dtype==torch.float32 and torch.equal(tensor,values[key].float())
    assert all(not t.requires_grad for t in model.parameters())
