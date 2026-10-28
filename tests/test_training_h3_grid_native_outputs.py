"""W5 original owner output boundary; never a W6/optimized-gate bypass."""
from copy import deepcopy
import pytest


def test_grid_output_terminal_boundary_does_not_relax_normal_restart():
    from distillschedule.training.h3_grid_training import validate_progress, versions
    identity={'source':'SYNTHETIC_TEST_ONLY'};policy={'policy':'SYNTHETIC_TEST_ONLY'}
    header=dict(cycle=3,cursor=18,versions=versions(3),policy=policy,identity=identity)
    with pytest.raises(ValueError):validate_progress(header,identity=identity,policy=policy,target=3)
    validate_progress(header,identity=identity,policy=policy,target=3,output_only=True)
    for change in (dict(cycle=4,cursor=24),dict(cycle=0,cursor=0),dict(cursor=17),dict(versions=versions(2)),dict(policy={})):
        with pytest.raises(ValueError):validate_progress({**header,**change},identity=identity,policy=policy,target=3,output_only=True)
    for bad in (1,None,'True'):
        with pytest.raises(ValueError):validate_progress(header,identity=identity,policy=policy,target=3,output_only=bad)

import json
import os
from pathlib import Path
from types import SimpleNamespace as NS
from test_training_h3_grid import original_source, cached_data, real_loader, training
from test_training_h3_grid_outputs import actual_grid_case


def test_output_discovery_is_model_free():
    import subprocess,sys
    result=subprocess.run([sys.executable,'-B','-c',
        "from distillschedule.training.h3_grid_export import export_native;import sys;"
        "assert not any(n in sys.modules for n in ('torch','diffusers','lightx2v_train'))"],capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr


@pytest.mark.parametrize('cycle',[1,3])
def test_real_native_output_cursor_never_creates_iterator_or_reads_samples(tmp_path,original_source,cached_data,monkeypatch,cycle):
    import torch
    from distillschedule.training.h3_grid_inputs import prepare_inputs,NativeLatentCursor
    native_path,native,_=cached_data
    inventory=prepare_inputs(native_path,tmp_path/'inputs.json')
    loader=real_loader(native,None)
    monkeypatch.setattr(type(loader),'__iter__',lambda self:(_ for _ in ()).throw(AssertionError('output consumed data')))
    before=torch.get_rng_state().clone()
    cursor=NativeLatentCursor(loader,inventory,cycle=cycle,resume=True,output_only=True)
    assert cursor.cursor==6*cycle and cursor.iterator is None
    cursor.validate_cycle(cycle)
    assert torch.equal(before,torch.get_rng_state())
    for function in (lambda:next(cursor),cursor._next_local):
        with pytest.raises(ValueError,match='read-only'):function()
    if cycle==3:
        with pytest.raises(ValueError):NativeLatentCursor(loader,inventory,cycle=cycle,resume=True)


def test_output_runtime_config_moves_only_metadata_and_keeps_default_identity(tmp_path):
    from distillschedule.training.h3_grid_training import GridTrainingLifecycle
    conf=training(tmp_path)
    raw=dict(training=dict(output_dir='original',seed=17,save_every_iters=50,other={'keep':True}),model={'keep':True})
    normal=GridTrainingLifecycle(conf,{}, {})
    assert normal.runtime_config(raw) is raw
    callback=NS(owner_output=tmp_path/'separate-owner')
    output=GridTrainingLifecycle(conf,{}, {},native_output=callback)
    effective=output.runtime_config(raw)
    assert effective['training']['output_dir']==str(callback.owner_output)
    assert raw['training']['output_dir']=='original'
    effective['training']['output_dir']='original'
    assert effective==raw


def policy(case):
    from distillschedule.training.h3_grid_export import SCHEMA
    return dict(schema_version=SCHEMA,seed=93,geometry=case.parameters['geometry'],
        source_terms=case.parameters['source']['terms'],
        snapshot_budget=dict(max_parameters=40_000_000_000,max_export_bytes=256*2**30,max_table_bytes=256*2**20,
                             host_reserve_bytes=2**20,device_reserve_bytes=2**20),
        inference_budget=case.parameters['budget'])


def native_owner(case):
    from contextvars import ContextVar
    case.controller.stats=lambda:NS(scope='SYNTHETIC_TEST_ONLY_UNCHANGED_CACHE_COUNTERS')
    shared=NS(denoiser_module=lambda:case.model,adaln_cache=lambda:case.controller,device=__import__('torch').device('cpu'),
              _role=ContextVar('test_role',default='student'))
    clocks=NS(state_dict=lambda:dict(cycle=1,score_serial=6,physical_role='student'))
    life=NS(completed=1,cursor=NS(cursor=6),identity={'scope':'SYNTHETIC_TEST_ONLY_CPU_CALLBACK_OWNER'},state={'clocks':clocks})
    return NS(shared_model=shared),life


def test_real_current_grid_callback_common_export_and_source_guard(actual_grid_case,tmp_path,monkeypatch):
    """Actual source/model/snapshot/common publication; only tiny admission injected.

    No native CUDA/FSDP/process-group owner is represented by this CPU fixture.
    Production _parameters still requires the full50block native architecture.
    """
    import torch
    from distillschedule.training import h3_grid_inference as grid
    from distillschedule.training.h3_grid_export import NativeOutputRequest,_owner
    from distillschedule.training.inference import read_export,restore_modules
    from distillschedule.training.checkpoint import rng_state
    case=actual_grid_case
    request=NativeOutputRequest(policy(case),tmp_path/'export')
    admitted=dict(assets={'lightx2v':{'path':case.parameters['source']['root']}},native_config={'training':{'output_dir':str(tmp_path/'original')}})
    request.admit(admitted,{'test_only':True})
    trainer,life=native_owner(case)
    before=_owner(trainer,life);random=rng_state(device='cpu')
    weights,table=grid.current_student_view(case.model,case.controller,case.api,max_table_bytes=1_000_000)
    production=request._parameters(case.model,weights)
    assert production['evidence_kind']==grid.NATIVE
    with pytest.raises(ValueError,match='full50'):grid.export_components(production,weights=weights)
    # Only the explicit fixture architecture/evidence admission differs.
    monkeypatch.setattr(request,'_parameters',lambda module,values:deepcopy(case.parameters))
    result=request.publish(trainer,life)
    assert _owner(trainer,life)==before and torch.equal(random['torch'],rng_state(device='cpu')['torch'])
    assert result['output_only']==dict(cycles=0,forward=0,backward=0,optimizer=0,ema=0,samples_consumed=0,merged_lora=False)
    assert result['capacity']['device']['scope']=='CPU_TEST_NO_CUDA_CAPACITY_QUALIFICATION'
    assert Path(result['inference_config']).is_file()
    root,manifest=read_export(tmp_path/'export')
    separate=grid.build_inference_model(case.parameters)
    restore_modules(separate,root,manifest)
    actual=separate.generate(separate.state,case.feeds,{})
    torch.testing.assert_close(actual['video_tokens'],case.oracle[0],rtol=0,atol=0)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('fault',['budget','existing','rank_disagreement','config_failure','owner_mutation'])
def test_native_grid_output_failures_leave_no_valid_new_artifact(actual_grid_case,tmp_path,monkeypatch,fault):
    from distillschedule.training import h3_grid_export as output
    from distillschedule.training.h3_grid_export import NativeOutputRequest
    case=actual_grid_case;value=policy(case)
    if fault=='budget':value['snapshot_budget']['max_export_bytes']=0
    request=NativeOutputRequest(value,tmp_path/'export')
    admitted=dict(assets={'lightx2v':{'path':case.parameters['source']['root']}},native_config={'training':{'output_dir':str(tmp_path/'original')}})
    if fault=='existing':request.owner_output.mkdir()
    elif fault=='rank_disagreement':monkeypatch.setattr(output.distributed,'gather',lambda row:[row,{}])
    if fault in ('budget','existing','rank_disagreement'):
        with pytest.raises(ValueError):request.admit(admitted,{'test_only':True})
        assert not (request.output/'_SUCCESS').exists()
        return
    request.admit(admitted,{'test_only':True})
    trainer,life=native_owner(case)
    monkeypatch.setattr(request,'_parameters',lambda module,values:deepcopy(case.parameters))
    if fault=='config_failure':request.config_path.write_text('do not replace')
    else:
        # _owner is the production post-gather preservation check. Perturb only
        # its test observation after publication, not a tolerance or source gate.
        original=output._owner;seen=[]
        def observe(*args):
            row=original(*args);seen.append(True)
            if len(seen)>1:row['cursor']+=1
            return row
        monkeypatch.setattr(output,'_owner',observe)
    with pytest.raises((ValueError,FileExistsError)):request.publish(trainer,life)
    assert not (request.output/'_SUCCESS').exists()
    if fault=='config_failure':assert request.config_path.read_text()=='do not replace'


def test_public_native_grid_export_preserves_identity_and_uses_output_checkpoint_gate(tmp_path,monkeypatch):
    from distillschedule.training import native,h3_grid_export as output,h3_grid_training as owner
    from distillschedule.training.config import TrainingConfig
    raw=training(tmp_path).to_dict();raw['trainer']['export']={'test_policy':'SYNTHETIC_TEST_ONLY_DISPATCH'}
    raw['trainer']['resume_from']=str(tmp_path/'wrong_old')
    raw['trainer']['stop_after_cycle']=1
    config=TrainingConfig.from_dict(raw,source=tmp_path/'train.json')
    seen=[]
    def admit(clean):
        assert not {'export','resume_from','stop_after_cycle'} & set(clean.trainer)
        assert clean.trainer['output_dir']==config.trainer['output_dir'] and clean.trainer['checkpoint_every']==1
        return 'W5',dict(identity=dict(head='SYNTHETIC_TEST_ONLY',dirty=False)),{},dict(
            assets=dict(native_config={'sha256':'SYNTHETIC_TEST_ONLY'},dependency_versions={'torch':'2.10.0'}),
            inputs={'inventory_sha256':'SYNTHETIC_TEST_ONLY'})
    monkeypatch.setattr(native,'_admit',admit)
    def metadata(path,**kwargs):
        assert path==str((tmp_path/'actual').resolve()) and kwargs['output_only'] is True
        seen.append('real_output_metadata_call')
        return {'cycle':3}
    monkeypatch.setattr(owner,'resume_metadata',metadata)
    def run(conf,*,admit,native_output):
        assert conf.trainer['resume_from']==str((tmp_path/'actual').resolve())
        assert 'stop_after_cycle' not in conf.trainer
        assert conf.trainer['output_dir']==config.trainer['output_dir']
        assert conf.execution==config.execution and conf.data==config.data
        admit();native_output.result={'scope':'SYNTHETIC_TEST_ONLY_DISPATCH'}
    monkeypatch.setattr(owner,'run_reference',run)
    assert native.describe_native(config)['capabilities']['export']['supported']
    result=native.export_native(config,tmp_path/'actual',tmp_path/'export')
    assert result['scope']=='SYNTHETIC_TEST_ONLY_DISPATCH' and seen==['real_output_metadata_call']
    bad=deepcopy(raw);bad['execution']['mode']='manual'
    with pytest.raises(ValueError,match='original fresh'):native.export_native(TrainingConfig.from_dict(bad,source=config.source),tmp_path/'actual',tmp_path/'bad')


def test_original_training_update_loop_ast_only_adds_post_checkpoint_stop():
    import ast,subprocess
    from distillschedule.workloads.best_known.registry import repository_root
    path='src/distillschedule/backends/training/minimax_h3/trainer_loop.py'
    root=repository_root()
    old=subprocess.check_output(['git','show','2a7708f5064fbb6df91810145d90fd3f3b3e33b4:'+path],cwd=root,text=True)
    def loop(source):
        train=next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef) and n.name=='train')
        return next(n for n in train.body if isinstance(n,ast.While))
    original,current=loop(old),loop((root/path).read_text())
    # Permit only the explicit lifecycle return/break seam, not changed model,
    # loss, scheduler, input, observer, census, logging or benchmark branches.
    added=current.body[-1].body
    expected=ast.parse('if lifecycle.after_cycle(self, current_iter):\n    break').body
    assert ast.dump(ast.Module(body=added,type_ignores=[]),include_attributes=False)==ast.dump(
        ast.Module(body=expected,type_ignores=[]),include_attributes=False)
    current.body[-1].body=[ast.Expr(value=added[0].test)]
    assert ast.dump(original,include_attributes=False)==ast.dump(current,include_attributes=False)


def test_actual_checkpoint_restore_precedes_output_and_performs_zero_updates(tmp_path,original_source,cached_data):
    """Real original scalar-edge loop/checkpoint, not pretrained/FSDP evidence."""
    import torch
    from test_training_h3_grid import tiny_config,tiny_original_trainer
    from distillschedule.training.checkpoint import pack_state,rng_state
    from test_training_end_to_end import equal
    native,inventory,public=tiny_config(tmp_path,cached_data)
    trainer,life=tiny_original_trainer(tmp_path,native,inventory,public)
    trainer.train()
    expected,random=pack_state(life.state),rng_state()
    checkpoint=tmp_path/'run/checkpoints/cycle-000003'
    plain,normal=tiny_original_trainer(tmp_path,native,inventory,public,resume=checkpoint)
    with pytest.raises(ValueError,match='incomplete-target'):plain.train()
    resumed,output=tiny_original_trainer(tmp_path,native,inventory,public,resume=checkpoint)
    calls=[]
    def publish(owner,lifecycle):
        assert lifecycle.completed==3 and lifecycle.cursor.cursor==18 and lifecycle.cursor.iterator is None
        assert equal(expected,pack_state(lifecycle.state))
        assert equal(random,rng_state())
        calls.append('REAL_COMPLETE_STATE_RESTORED_BEFORE_OUTPUT_NO_NEXT_SAMPLE')
    output.native_output=NS(owner_output=tmp_path/'separate-output-owner',publish=publish)
    resumed.output_train_dir=str(output.native_output.owner_output)
    resumed.forward_student_loss=lambda *args,**kwargs:(_ for _ in ()).throw(AssertionError('output performed training'))
    resumed.train()
    assert calls==['REAL_COMPLETE_STATE_RESTORED_BEFORE_OUTPUT_NO_NEXT_SAMPLE']
    assert equal(expected,pack_state(output.state)) and equal(random,rng_state())
    assert not torch.cuda.is_initialized()


def _grid_dtensor_export_rank(rank,root,parameters):
    """Real CPU/Gloo DTensor shards through the native export callback.

    Constructor-free cluster owner and tiny admission only. No FSDP/CUDA/world16
    capacity or long-training equivalence is inferred from this test.
    """
    from datetime import timedelta
    from contextvars import ContextVar
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import distribute_tensor,Shard
    from distillschedule.training import h3_grid_inference as grid
    from distillschedule.training.h3_grid_export import NativeOutputRequest
    root=Path(root);torch.set_num_threads(1)
    parts=grid.build_inference_model(parameters)
    snapshot=torch.load(root/'expected.pt',weights_only=True)
    parts.state['student'].load_state_dict(snapshot,strict=True)
    model=parts.state['student'].transformer
    dist.init_process_group('gloo',init_method=(root/'rendezvous').as_uri(),rank=rank,world_size=2,timeout=timedelta(seconds=90))
    try:
        mesh=init_device_mesh('cpu',(2,))
        for name,param in list(model.named_parameters()):
            parent,_,leaf=name.rpartition('.')
            model.get_submodule(parent)._parameters[leaf]=torch.nn.Parameter(
                distribute_tensor(param.detach().clone(),mesh,[Shard(0)]),requires_grad=('.student.' in name))
        table=torch.zeros(1004,1,6,6,32,dtype=torch.bfloat16)
        table[1000:]=snapshot['rollout_modulations']
        controller=NS(payload=dict(schema_version='h3_a100.adaln_grid.v1',grid_size=1000,num_entries=1004,
            num_blocks=1,hidden_size=32,rows_per_entry=6,modulation_chunks=6,video_shift=6.0,audio_shift=3.0),
            _entry_by_timestep_bits={tuple(b):1000+i for i,b in enumerate(grid.rollout_bits(grid.source_api(parameters['source'])))},
            _table=table,stats=lambda:NS(scope='SYNTHETIC_TEST_ONLY_GRID_TABLE_OWNER'))
        shared=NS(denoiser_module=lambda:model,adaln_cache=lambda:controller,device=torch.device('cpu'),_role=ContextVar('role',default='student'))
        life=NS(completed=1,cursor=NS(cursor=6),identity={'scope':'SYNTHETIC_TEST_ONLY_GLOO2_DTENSOR_NOT_FSDP16'},
                state={'clocks':NS(state_dict=lambda:dict(cycle=1,score_serial=6))})
        case=NS(parameters=parameters)
        request=NativeOutputRequest(policy(case),root/'distributed-export')
        request.admit(dict(assets={'lightx2v':{'path':parameters['source']['root']}},
                           native_config={'training':{'output_dir':str(root/'unused-training')}}),life.identity)
        request._parameters=lambda module,weights:deepcopy(parameters)  # explicit tiny-class admission only
        request.publish(NS(shared_model=shared),life)
        (root/f'rank-{rank}.json').write_text(json.dumps(dict(status='PASS_CPU_GLOO_DTENSOR_ONLY',cuda_initialized=torch.cuda.is_initialized())))
    finally:
        dist.destroy_process_group()


def test_real_gloo_dtensor_export_gathers_all_current_weight_shards(actual_grid_case,tmp_path):
    import torch
    import torch.multiprocessing as mp
    case=actual_grid_case
    torch.save(case.snapshot,tmp_path/'expected.pt')
    mp.spawn(_grid_dtensor_export_rank,args=(str(tmp_path),case.parameters),nprocs=2,join=True)
    saved=torch.load(tmp_path/'distributed-export/weights.pt',weights_only=True)['student']
    assert saved.keys()==case.snapshot.keys()
    assert all(torch.equal(value,saved[name]) for name,value in case.snapshot.items())
    for rank in range(2):
        assert json.loads((tmp_path/f'rank-{rank}.json').read_text())==dict(status='PASS_CPU_GLOO_DTENSOR_ONLY',cuda_initialized=False)


@pytest.mark.parametrize('free',['enough','insufficient'])
def test_output_only_cuda_capacity_metadata_releases_unused_cache_before_check(actual_grid_case,tmp_path,monkeypatch,free):
    """SYNTHETIC_TEST_ONLY CUDA metadata APIs; never initializes CUDA."""
    import torch
    from distillschedule.training.h3_grid_export import NativeOutputRequest
    from distillschedule.training import host_memory
    request=NativeOutputRequest(policy(actual_grid_case),tmp_path/'out')
    calls=[]
    monkeypatch.setattr(torch.cuda,'empty_cache',lambda:calls.append('release_unused'))
    def observe(device):
        calls.append('fresh_driver_free')
        return (4*2**20 if free=='enough' else 1,8*2**20)
    monkeypatch.setattr(torch.cuda,'mem_get_info',observe)
    monkeypatch.setattr(host_memory,'admit_host_memory',lambda required,reserve:dict(required=required,reserve=reserve,scope='SYNTHETIC_TEST_ONLY'))
    tensor=NS(numel=lambda:1024,element_size=lambda:4,device=NS(type='cuda'))
    if free=='enough':
        result=request._capacity({'logical':tensor},0,torch.device('cuda:0'))
        assert result['device']['largest_materialized_tensor_bytes']==4096
        assert result['device']['timing_scope']=='OUTPUT_ONLY_NOT_TRAINING'
    else:
        with pytest.raises(ValueError,match='device reserve'):request._capacity({'logical':tensor},0,torch.device('cuda:0'))
    assert calls==['release_unused','fresh_driver_free'] and not torch.cuda.is_initialized()


def test_native_output_rejects_artifacts_inside_source_checkout(actual_grid_case):
    from distillschedule.training.h3_grid_export import NativeOutputRequest
    from distillschedule.workloads.best_known.registry import repository_root
    request=NativeOutputRequest(policy(actual_grid_case),repository_root()/'__grid_export_must_not_exist__')
    with pytest.raises(ValueError,match='outside the source'):
        request.admit({}, {'scope':'SYNTHETIC_TEST_ONLY_PATH_REJECTION'})
    assert not request.output.exists()


def test_rng_capture_failure_is_collective_before_snapshot(actual_grid_case,tmp_path,monkeypatch):
    from distillschedule.training import checkpoint,h3_grid_export as output
    request=output.NativeOutputRequest(policy(actual_grid_case),tmp_path/'out')
    trainer,life=native_owner(actual_grid_case)
    calls=[];checked=output.distributed.checked
    def capture(function,label):
        calls.append(label);return checked(function,label)
    def fail(**kwargs):raise ValueError('SYNTHETIC_TEST_ONLY_RNG_CAPTURE_FAILURE')
    monkeypatch.setattr(output.distributed,'checked',capture)
    monkeypatch.setattr(checkpoint,'rng_state',fail)
    with pytest.raises(ValueError,match='RNG_CAPTURE_FAILURE'):request.publish(trainer,life)
    assert calls==['W5 pre-export rank RNG capture'] and not request.output.exists()
