"""Explicit operator-source Grid/codec acceptance; missing inputs fail.

Run this file with DMD_H3_TEST_UPSTREAM and DMD_H3_GRID_CODEC_TEST_MANIFEST.
The public source-free packing/contracts remain in test_training_h3_grid_codec.
"""
from copy import deepcopy
from pathlib import Path
import json
import os
import subprocess
import sys

import pytest

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
