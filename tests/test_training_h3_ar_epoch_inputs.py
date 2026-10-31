"""CPU/SYNTHETIC metadata and original checkpoint protocol, not GPU/native quality."""
from copy import deepcopy
import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from distillschedule.training import h3_ar_inputs as inputs
from distillschedule.strategies.h3_ar_task import H3ARTask
from distillschedule.workloads.best_known.registry import file_sha256, fingerprint
from test_training_h3_ar_continuation import stream, write, public_config

POLICY = 'distillschedule.h3_ar_record_epochs.v1'
KIND = 'original_h3_ar_epoch_stream.v1'


def pool(tmp_path, *, per_rank=1):
    path, value = stream(tmp_path, cycles=max(2, per_rank))
    records = [{k:v for k,v in row.items() if k not in ('cycle','metadata')}
               for row in value['records'][:16*per_rank]]
    source = write(tmp_path/'pool.json', records)
    return source, records


def manifest(tmp_path, *, cycles=50, per_rank=1, monkeypatch):
    source, records = pool(tmp_path, per_rank=per_rank)
    calls=[]
    def admit(self, task, cache_manifest, native, rank, torch, **kwargs):
        row=self.record(self.completed+1)
        calls.append((rank,row['record_id']))
        self.pending=self.completed+1
        return None,None,inputs.source_identity(row),dict(input_shapes=task.input_shapes(rank))
    monkeypatch.setattr(inputs.H3ARInputStream,'prepare_rank_input',admit)
    output=tmp_path/'epochs.json'
    result=inputs.prepare_stream(source,output,native=None,torch=torch,cycles=cycles,seed=17,input_policy=POLICY)
    return output,json.loads(output.read_text()),records,calls,result


def test_explicit_epoch_pool_is_not_800_fresh_sources(tmp_path,monkeypatch):
    random.seed(811); torch.manual_seed(812)
    py,th=random.getstate(),torch.get_rng_state().clone()
    path,value,records,calls,result=manifest(tmp_path,monkeypatch=monkeypatch)
    assert value['input_kind']==KIND and value['schema_version']==inputs.EPOCH_SCHEMA
    assert len(value['source_records'])==16 and len(value['draws'])==800
    assert len(calls)==16  # original parser admission once/source, not 50 aliases
    assert len({d['draw_id'] for d in value['draws']})==800
    assert [d['draw_index'] for d in value['draws']]==list(range(800))
    assert value['input_policy']['schema_version']==POLICY
    assert result['source_record_count']==16 and result['logical_draw_count']==800
    assert result['source_reuse_across_cycles'] is True
    assert inputs.input_metadata(path,cycles=50,seed=17,verify_files=True)==value
    assert random.getstate()==py and torch.equal(torch.get_rng_state(),th)
    assert not torch.cuda.is_initialized()


def test_rank_geometry_and_shared_draw_identity(tmp_path,monkeypatch):
    from distillschedule.training.h3_ar_training import validate_joint
    path,value,_,_,_=manifest(tmp_path,cycles=5,per_rank=2,monkeypatch=monkeypatch)
    joint=[]
    for rank in range(16):
        cursor=inputs.H3ARInputStream(path,rank=rank,cycles=5,seed=17,completed=3)
        row=cursor.record(4)
        assert row['rank']==rank and row['cycle']==4
        assert row['metadata']['input_shapes']['audio']==[2,32,94 if rank==6 else 93]
        assert row['draw_index']==48+rank and row['epoch_index']==1
        joint.append(dict(rank=rank,cycle=4,versions=dict(student=4,fake=4,ema=0),
                          native_global_steps=8,input_identity=inputs.source_identity(row)))
    validate_joint(joint,cycle=4,inputs=value)
    bad=deepcopy(joint); bad[6]['input_identity']['draw_id']='0'*64
    with pytest.raises(ValueError): validate_joint(bad,cycle=4,inputs=value)
    # Resume addresses same draw; does not advance RNG or select a fresh alias.
    assert inputs.H3ARInputStream(path,rank=6,cycles=5,seed=17,completed=3).record(4)==inputs.record(value,4,6)


@pytest.mark.parametrize('bad',['duplicate_id','duplicate_path','duplicate_hash','missing_rank','rank6','source','draw','seed','unknown_policy','alias'])
def test_epoch_rejections(tmp_path,monkeypatch,bad):
    path,value,_,_,_=manifest(tmp_path,cycles=3,monkeypatch=monkeypatch)
    rows=value['source_records']
    if bad=='duplicate_id': rows[1]['record_id']=rows[0]['record_id']
    elif bad=='duplicate_path': rows[1]['path']=rows[0]['path']
    elif bad=='duplicate_hash': rows[1]['sha256']=rows[0]['sha256']
    elif bad=='missing_rank': rows.pop()
    elif bad=='rank6': rows[6]['metadata']['input_shapes']['audio'][-1]=93
    elif bad=='source': rows[0]['sources']['original_cached_record']['sha256']='f'*64
    elif bad=='draw': value['draws'][16]['source_index']=1
    elif bad=='seed': value['seed']=18
    elif bad=='alias':
        alias=tmp_path/'alias.pt'; alias.symlink_to(rows[0]['path']); rows[1]['path']=str(alias)
    else: value['input_policy']['schema_version']='invented'
    value['input_sha256']=fingerprint({k:v for k,v in value.items() if k!='input_sha256'})
    write(path,value)
    with pytest.raises(ValueError): inputs.input_metadata(path,cycles=3,seed=17,verify_files=True)


def test_v1_refuses_reuse_and_epoch_requires_opt_in(tmp_path):
    source,records=pool(tmp_path)
    with pytest.raises(ValueError):
        inputs.prepare_stream(source,tmp_path/'refused.json',native=None,torch=torch,cycles=50,seed=17)
    assert not (tmp_path/'refused.json').exists()
    assert not (tmp_path/'.refused.json.preparing').exists()


@pytest.mark.parametrize('reference',[True,False])
def test_public_config_explicit_kind_lowering(tmp_path,monkeypatch,reference):
    from distillschedule.training.config import TrainingConfig
    from distillschedule.training.native import validate_native, _admit
    from distillschedule.training.h3_ar_training import training_identity
    config=public_config(tmp_path/'config',monkeypatch,reference=reference)
    path,value,_,_,_=manifest(tmp_path/'epoch',cycles=100,monkeypatch=monkeypatch)
    raw=config.to_dict(); raw['data'].update(prepared=str(path),input_kind=KIND)
    assets=json.loads((tmp_path/'config/assets.json').read_text())
    assets['cache_manifest']=dict(path=str(path),sha256=file_sha256(path))
    write(tmp_path/'config/assets.json',assets)
    config=TrainingConfig.from_dict(raw,source=config.source)
    report=validate_native(config)
    admitted=_admit(config)[3]
    assert admitted['inputs']==value and report['native_worker'].endswith(':run_training')
    identity=training_identity('a'*40,admitted,config)
    assert identity['input_sha256']==value['input_sha256']
    raw['data']['input_kind']=inputs.KIND
    with pytest.raises(ValueError): validate_native(TrainingConfig.from_dict(raw,source=config.source))


def test_public_cli_requires_named_opt_in(tmp_path,monkeypatch,capsys):
    from distillschedule.data.cli import main
    import distillschedule.data.h3_ar_stream as data
    calls=[]
    monkeypatch.setattr(data,'prepare_inputs',lambda *a,**kw:calls.append(kw) or {'status':'CPU_TEST'})
    args=['prepare-h3-ar-stream','--sources',str(tmp_path/'sources.json'),'--assets',str(tmp_path/'assets.json'),
          '--cycles','50','--seed','17','--output',str(tmp_path/'prepared.json')]
    assert main(args+['--input-policy',POLICY])==0
    assert calls[0]['input_policy']==POLICY


@pytest.mark.parametrize("output_only", [False, True])
def test_epoch_stage_uses_one_actual_restricted_load_and_same_parser_hook(tmp_path,monkeypatch,output_only):
    original=inputs.H3ARInputStream.prepare_rank_input
    path,value,_,_,_=manifest(tmp_path,cycles=3,monkeypatch=monkeypatch)
    monkeypatch.setattr(inputs.H3ARInputStream,'prepare_rank_input',original)
    row=value['source_records'][6]; metadata=row['metadata']
    def meta(value):
        if isinstance(value,dict) and set(value)=={'shape','dtype'}:
            return torch.empty(value['shape'],dtype=getattr(torch,value['dtype'].split('.')[-1]),device='meta')
        if isinstance(value,dict): return {k:meta(v) for k,v in value.items()}
        if isinstance(value,list): return [meta(v) for v in value]
        return value
    torch.save(meta(metadata['raw_tensors']),row['path'])
    row['sha256']=file_sha256(Path(row['path']))
    value=inputs._epoch_document(value['source_records'],cycles=3,seed=17)
    write(path,value)
    positive={**meta(metadata['positive_tensors']),**metadata['positive_scalars']}
    calls=[]; loads=[]; load=torch.load
    def counted_load(*a,**kw): loads.append((a,kw)); return load(*a,**kw)
    native=SimpleNamespace(prepare_batch=lambda raw,cfg:calls.append('SYNTHETIC_TEST_ONLY_META_PARSER') or (positive,None))
    monkeypatch.setattr(torch,'load',counted_load)
    monkeypatch.setattr(inputs,'validate_parser',lambda _:None)
    if output_only:
        from distillschedule.training.h3_ar_export import OutputInputStream
        cursor=OutputInputStream(path,rank=6,cycles=3,seed=17,completed=2)
    else:
        cursor=inputs.H3ARInputStream(path,rank=6,cycles=3,seed=17,completed=1)
    py,th=random.getstate(),torch.get_rng_state().clone()
    witness=cursor.stage_initial(H3ARTask(),cursor.reference,native,6,torch)
    prepared=cursor.prepare_rank_input(H3ARTask(),cursor.reference,native,6,torch)
    assert len(loads)==len(calls)==1 and loads[0][1]==dict(map_location='cpu',weights_only=True)
    assert prepared[0] is positive and prepared[3]==witness
    assert prepared[2]==inputs.source_identity(cursor.record(2))
    assert prepared[2]['epoch_index']==1 and prepared[2]['input_policy']==POLICY
    if output_only:
        assert cursor.completed==2 and cursor.pending is None and cursor.construction.pending==2
        with pytest.raises(ValueError): cursor.commit(3)
    else:
        assert cursor.pending==2 and cursor.completed==1
    assert random.getstate()==py and torch.equal(torch.get_rng_state(),th)


def test_epoch_checkpoint_binds_schedule_and_complete_cursor_without_new_state_abi(tmp_path,monkeypatch):
    from test_training_h3_ar_continuation import session, complete_cycle
    from distillschedule.training.h3_ar_state import NativeCheckpoint, Progress
    from distillschedule.training import checkpoint as cp
    path,value,_,_,_=manifest(tmp_path/'input',cycles=3,monkeypatch=monkeypatch)
    cursor=inputs.H3ARInputStream(path,rank=0,cycles=3,seed=17)
    s=session(); s.input_provider=cursor
    owner=NativeCheckpoint(s); progress=Progress(s,cursor,owner)
    complete_cycle(s,cursor,1)
    identity=dict(input_sha256=value['input_sha256'],seed=17,horizon=3)
    cp.save_checkpoint(tmp_path/'ckpt',{'native_progress':progress},identity=identity,cycle=1,cursor=1,
                       versions=s.versions(),native_checkpoint=owner)
    saved=progress.state_dict()
    assert saved['cursor']==1 and saved['input_sha256']==value['input_sha256']
    for cycle in (2,3): complete_cycle(s,cursor,cycle)
    expected=(deepcopy(s.engine.module.state_dict()),deepcopy(s.optimizer.state_dict()),torch.get_rng_state())
    restored=session(); target=inputs.H3ARInputStream(path,rank=0,cycles=3,seed=17,completed=1)
    restored.input_provider=target; native=NativeCheckpoint(restored); target_progress=Progress(restored,target,native)
    cp.restore_checkpoint(tmp_path/'ckpt',{'native_progress':target_progress},identity=identity,native_checkpoint=native)
    assert target.record(2)==inputs.record(value,2,0)
    for cycle in (2,3): complete_cycle(restored,target,cycle)
    def equal(a,b):
        if isinstance(a,torch.Tensor): assert torch.equal(a,b)
        elif isinstance(a,dict):
            assert a.keys()==b.keys()
            for k in a: equal(a[k],b[k])
        elif isinstance(a,(list,tuple)):
            assert len(a)==len(b)
            for x,y in zip(a,b): equal(x,y)
        else: assert a==b
    equal(expected,(restored.engine.module.state_dict(),restored.optimizer.state_dict(),torch.get_rng_state()))
    assert target.completed==3 and restored.engine.global_steps==6
    bad=deepcopy(saved);bad['input_sha256']='f'*64
    with pytest.raises(ValueError): target_progress.validate_state_dict(bad)


def test_epoch_public_export_and_terminal_readonly_cursor(tmp_path,monkeypatch):
    from distillschedule.training import native,h3_ar_training as worker
    from distillschedule.training.h3_ar_export import OutputInputStream
    from distillschedule.training.config import TrainingConfig
    path,value,_,_,_=manifest(tmp_path/'epoch',cycles=3,monkeypatch=monkeypatch)
    cursor=OutputInputStream(path,rank=6,cycles=3,seed=17,completed=3)
    assert cursor.completed==3 and cursor.pending is None
    assert cursor.construction.record(3)==inputs.record(value,3,6)
    conf=public_config(tmp_path/'config',monkeypatch,reference=True)
    raw=conf.to_dict();raw['data']['input_kind']=KIND
    raw['trainer']['export']={'schema_version':'test-only-request'}
    conf=TrainingConfig.from_dict(raw,source=conf.source)
    def run(clean,*,admit,native_output):
        assert clean.data['input_kind']==KIND and clean.trainer['cycles']==100
        assert clean.trainer['checkpoint_every']==50 and clean.trainer['seed']==17
        assert 'stop_after_cycle' not in clean.trainer and 'export' not in clean.trainer
        native_output.result={'status':'EXPORTED','scope':'SYNTHETIC_TEST_ONLY_DISPATCH'}
        return 0
    monkeypatch.setattr(worker,'run_training',run)
    monkeypatch.setattr(native,'_admit',lambda clean:('W6',{}, {}, {}))
    assert native.export_native(conf,tmp_path/'checkpoint',tmp_path/'out')['status']=='EXPORTED'
    raw['data']['input_kind']='invented_epoch_kind'
    with pytest.raises(ValueError,match='fresh W6'):
        native.export_native(TrainingConfig.from_dict(raw,source=conf.source),tmp_path/'checkpoint',tmp_path/'bad')
