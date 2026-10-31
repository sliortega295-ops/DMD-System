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
