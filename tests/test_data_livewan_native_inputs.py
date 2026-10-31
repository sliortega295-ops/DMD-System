"""Real codec/prepared-input/current-Student chain; tiny RANDOM CPU fixtures."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from test_training_livewan_generator_import import fixture_request, pin, same, write_json


SCHEMA='distillschedule.livewan_native_new_inputs.v1'


def prepare(request,output):
    if importlib.util.find_spec('distillschedule.data.livewan_native_inputs') is None:
        # Actual old public input producer rejection, not missing API/import.
        from distillschedule.data.preparation import import_preencoded
        return import_preencoded(request,output)
    from distillschedule.data.livewan_native_inputs import prepare_inputs
    return prepare_inputs(request,output)


@pytest.fixture(scope='module')
def case(tmp_path_factory):
    from PIL import Image
    import numpy as np
    from distillschedule.data.preparation import import_preencoded
    from distillschedule.data.manifest import PREENCODED_SCHEMA
    from distillschedule.training import livewan_native_inference as native
    root=tmp_path_factory.mktemp('native-livewan-inputs-SYNTHETIC_ONLY')
    case=fixture_request(root/'origin',qualified_runtime=True)
    generator=torch.Generator().manual_seed(163)
    context=torch.randn(1,512,4096,generator=generator)
    torch.save({'prompt_embeds':context},root/'positive.pt')
    p=pin(root/'positive.pt');p.pop('bytes')
    source=write_json(root/'positive.json',dict(schema_version=PREENCODED_SCHEMA,
        records=[dict(id='NEW_POSITIVE_SYNTHETIC_ONLY',features=p)]))
    positive=import_preencoded(source,root/'positive-prepared')
    frames=[]
    for i in range(9):
        a=np.arange(32*32*3,dtype=np.uint16).reshape(32,32,3)
        image=((a+i*17)%256).astype(np.uint8)
        path=root/f'frame-{i:03d}.png';Image.fromarray(image).save(path);frames.append(pin(path))
    request=dict(schema_version=SCHEMA,evidence_kind=native.FIXTURE,
        geometry=dict(resolution=[32,32],frames=81,fps=16),budget=case['request']['parameters']['budget'],
        codec=case['request']['codec'],positive=dict(manifest=pin(positive),sample_id='NEW_POSITIVE_SYNTHETIC_ONLY'),
        opening=dict(world_frames=3,frames=frames))
    return {**case,'base':root,'context':context,'request':request,'path':write_json(root/'inputs.json',request)}


def test_exact_pinned_rgb_opening_and_positive_producer_to_current_consumer(case,tmp_path):
    from PIL import Image
    import numpy as np
    from distillschedule.data import PreparedDataset
    from distillschedule.training import livewan_generator_import as importer,session
    from distillschedule.training.checkpoint import rng_state
    from distillschedule.training.config import load_training_config
    before=rng_state()
    prepared=prepare(case['path'],tmp_path/'prepared')
    assert same(before,rng_state())
    dataset=PreparedDataset(prepared);dataset.validate(sources=True)
    row=dataset.sample(0)
    assert set(row)=={'context','world'} and torch.equal(row['context'],case['context'])
    assert list(row['world'].shape)==[1,16,3,4,4] and row['world'].dtype==torch.float32
    pixels=torch.from_numpy(np.stack([np.array(Image.open(p['path']),copy=True) for p in case['request']['opening']['frames']]))
    pixels=pixels.permute(3,0,1,2).unsqueeze(0).float()/127.5-1
    codec=case['codec']
    with torch.no_grad():z=codec.encode(pixels).latent_dist.mode()
    mean=torch.tensor(codec.config.latents_mean).view(1,16,1,1,1)
    std=torch.tensor(codec.config.latents_std).view(1,16,1,1,1)
    assert torch.equal(row['world'],(z-mean)/std)
    manifest=importer.import_generator_weights(case['path'].parent/'origin/request.json',tmp_path/'portable')
    cfg=load_training_config(tmp_path/'portable.inference.json')
    result=session.generate_from_export(cfg,tmp_path/'portable',tmp_path/'generated',prepared=prepared,output_kind='video')
    value=torch.load(result['artifact'],weights_only=True)
    assert torch.equal(value['generated']['latents'][:,:,:3],row['world'])
    assert list(value['generated']['latents'].shape)==[1,16,21,4,4]
    assert list(value['decoded']['video'].shape)==[81,32,32,3]
    assert len(list((tmp_path/'generated/frames').glob('*.png')))==81
    assert manifest['identity']['weights_initialization']['origin_numerical_status']=='FAIL_EXTREME_NUMERICAL_DRIFT'
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('fault',['count','world21','duplicate','geometry','changed_pin','sample','extra'])
def test_invalid_source_refused_before_codec_construction(case,tmp_path,monkeypatch,fault):
    from distillschedule.training import wan_native_inference as wan
    request=deepcopy(case['request'])
    if fault=='count':request['opening']['frames'].pop()
    elif fault=='world21':request['opening']['world_frames']=21
    elif fault=='duplicate':request['opening']['frames'][1]=request['opening']['frames'][0]
    elif fault=='geometry':request['geometry']['resolution']=[368,640]
    elif fault=='changed_pin':request['opening']['frames'][0]['sha256']='0'*64
    elif fault=='sample':request['positive']['sample_id']='ABSENT'
    else:request['opening']['implicit_crop']=True
    monkeypatch.setattr(wan,'_load_codec',lambda *a,**k:pytest.fail('invalid source reached codec construction'))
    with pytest.raises(ValueError):prepare(write_json(tmp_path/'bad.json',request),tmp_path/'not-published')
    assert not (tmp_path/'not-published').exists()


def test_existing_output_refused_before_model_and_duplicate_json_keys(case,tmp_path,monkeypatch):
    from distillschedule.training import wan_native_inference as wan
    output=tmp_path/'USER_OWNED';output.mkdir()
    monkeypatch.setattr(wan,'_load_codec',lambda *a,**k:pytest.fail('collision reached model construction'))
    with pytest.raises(FileExistsError):prepare(case['path'],output)
    path=tmp_path/'duplicate.json';path.write_text('{"schema_version":"bad","schema_version":"bad"}')
    with pytest.raises(ValueError):prepare(path,tmp_path/'no-output')


def test_exact_public_cli_preparer_contract_in_fresh_cpu_process(case,tmp_path):
    from distillschedule.data import PreparedDataset
    result=subprocess.run([sys.executable,'-B','-m','distillschedule','data','prepare',str(case['path']),
        '--preparer','distillschedule.data.livewan_native_inputs:prepare_inputs','--output',str(tmp_path/'prepared')],
        env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'},capture_output=True,text=True,timeout=120)
    assert result.returncode==0,result.stdout+result.stderr
    dataset=PreparedDataset(tmp_path/'prepared/manifest.json');dataset.validate(sources=True)
    assert torch.equal(dataset.sample(0)['context'],case['context'])
