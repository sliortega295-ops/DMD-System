"""Original history19 primary/secondary mapping; synthetic fixture evidence only."""
from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest
import torch

from test_training_helios_public_cpu import fixture_files, pin, write_json, FIXTURE, pytestmark as pytestmark
from test_helios_step_owner_cpu import equal
from distillschedule.data import PreparedDataset
from distillschedule.data.helios_native_inputs import prepare_inputs,decode_row


def test_original_history19_native_producer_mapping_and_tensor_only_delivery(tmp_path):
    assets,sources = fixture_files(tmp_path/'local',capacity=5)
    output = prepare_inputs(sources,tmp_path/'prepared')
    dataset = PreparedDataset(output)
    assert len(dataset)==1 and dataset.validate(sources=True)['status']=='PASS'
    source = json.loads(sources.read_text())
    rows = dataset.sample(0)['native_rows']
    meta = dataset.manifest['records'][0]['metadata']['native_rows']
    for i,(row,metadata) in enumerate(zip(rows,meta)):
        raw = torch.load(source['records'][i]['features']['path'],weights_only=True)
        decoded = decode_row(row,absolute_step=i)
        assert decoded['generator_history']['gt_history_latents'].shape==(1,16,19,16,16)
        assert equal(decoded['gan_prompt_embeds'],raw['prompt_embed'].unsqueeze(0))
        assert equal(decoded['prompt_embeds'],decoded['gan_prompt_embeds'])
        flat = raw['vae_latent'].permute(1,0,2,3,4).reshape(16,-1,16,16)
        padded = torch.cat([torch.zeros(16,19,16,16),flat],dim=1)
        for role,section in [('generator',metadata['primary_section']),('critic',metadata['secondary_section'])]:
            start = section*9
            history = decoded[role+'_history']
            assert equal(history['gt_history_latents'],padded[:,start:start+19].unsqueeze(0))
            assert equal(history['gt_target_latents'],padded[:,start+19:start+28].unsqueeze(0))
        assert equal(decoded['gan_vae_latents'],decoded['generator_history']['gt_target_latents'])
    assert dataset.manifest['metadata']['evidence_kind']==FIXTURE
    assert dataset.manifest['metadata']['quality_validation']=='NOT_RUN'
    with pytest.raises(FileExistsError): prepare_inputs(sources,tmp_path/'prepared')


@pytest.mark.parametrize('change',['path','bytes','content','index','label','shape','nonfinite','fields','history'])
def test_native_source_fail_closed_without_partial_publication(tmp_path,change):
    assets,sources = fixture_files(tmp_path/'local',capacity=5)
    value = json.loads(sources.read_text())
    if change=='path': value['records'][4]['features']=deepcopy(value['records'][0]['features'])
    elif change=='bytes':
        original = Path(value['records'][0]['features']['path'])
        copied = tmp_path/'copied.pt';shutil.copyfile(original,copied)
        value['records'][4]['features']=pin(copied)
    elif change=='content':
        raw = torch.load(value['records'][0]['features']['path'],weights_only=True)
        copied = tmp_path/'different-container-same-native.pt'
        raw['prompt_raw']='new label is not fresh data'
        torch.save(raw,copied);value['records'][4]['features']=pin(copied)
    elif change=='index': value['records'][4]['native_index']=0
    elif change=='label': value['evidence_kind']='LOCAL_PRETRAINED_NATIVE_FEATURES_UNVERIFIED_QUALITY'
    elif change in ('shape','nonfinite','fields'):
        path = Path(value['records'][4]['features']['path']);raw=torch.load(path,weights_only=True)
        if change=='shape': raw['vae_latent']=raw['vae_latent'][:,:,:1]
        elif change=='nonfinite': raw['prompt_embed'].fill_(float('nan'))
        else: raw['precomputed_rollout']=torch.zeros(1)
        torch.save(raw,path);value['records'][4]['features']=pin(path)
    elif change=='history':
        path = Path(value['args']['path']);args=json.loads(path.read_text())
        args['training_config']['history_sizes']=[4,2,1]
        write_json(path,args);value['args']=pin(path)
    write_json(sources,value)
    with pytest.raises(ValueError): prepare_inputs(sources,tmp_path/'rejected')
    assert not (tmp_path/'rejected').exists()
    assert not list(tmp_path.glob('.rejected.*'))


def test_preparation_retains_at_most_two_native_cycles_not_whole_inventory(tmp_path, monkeypatch):
    """Real producer tensors must be released between publication windows."""
    import weakref
    from distillschedule.training import helios_step_owner

    _, sources = fixture_files(tmp_path / 'local', capacity=25)
    original = helios_step_owner._content
    refs = []
    alive = []
    def observe(value):
        if isinstance(value, dict) and 'generator_history' in value and 'critic_history' in value:
            refs.append(weakref.ref(value['generator_history']['gt_history_latents']))
            alive.append(sum(ref() is not None for ref in refs))
        return original(value)
    monkeypatch.setattr(helios_step_owner, '_content', observe)
    before = torch.get_rng_state().clone()
    manifest = prepare_inputs(sources, tmp_path / 'prepared')
    assert torch.equal(before, torch.get_rng_state())
    assert len(PreparedDataset(manifest)) == 5
    # One five-row cycle being published and the publisher's prior cycle can
    # coexist. No tensors from the rest of the finite inventory may stay live.
    assert refs and max(alive) <= 10, f'entire native tensor inventory retained: peak={max(alive)}'


def test_source_change_between_validation_and_stream_publication_is_atomic(tmp_path, monkeypatch):
    from distillschedule.data import preparation

    _, sources = fixture_files(tmp_path / 'local', capacity=5)
    source = json.loads(sources.read_text())
    feature = Path(source['records'][0]['features']['path'])
    original = preparation._publish
    def change_then_publish(*args, **kwargs):
        feature.write_bytes(feature.read_bytes() + b'changed after source admission')
        return original(*args, **kwargs)
    monkeypatch.setattr(preparation, '_publish', change_then_publish)
    output = tmp_path / 'prepared'
    with pytest.raises(ValueError, match='pin changed|content changed'):
        prepare_inputs(sources, output)
    assert not output.exists()
    assert not (tmp_path / '.prepared.lock').exists()
    assert not list(tmp_path.glob('.prepared.*'))
