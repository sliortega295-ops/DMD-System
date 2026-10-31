"""Native video post-backward B1 replication; tiny proofs are SYNTHETIC_TEST_ONLY."""
from copy import deepcopy
from dataclasses import replace
import importlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from test_training_flow_image_device import CUDA_MANUAL, REPLICATED_MANUAL
from test_training_flow_video_device import FAMILIES


@pytest.mark.parametrize('family,module', FAMILIES)
@pytest.mark.parametrize('device,backend', [('cpu', 'gloo'), ('cuda', 'nccl')])
def test_video_replication_catalog_is_exact_and_lazy(family, module, device, backend, tmp_path, monkeypatch):
    from distillschedule.training.config import TrainingConfig
    from distillschedule.training import session
    value = dict(schema_version='distillschedule.training.v1', model=dict(name=family, parameters={}),
        algorithm=dict(name='dmd'), data=dict(prepared='MISSING', sharding='rank_strided'),
        execution=deepcopy(REPLICATED_MANUAL),
        trainer=dict(device=device, distributed=dict(backend=backend), cycles=1, output_dir=str(tmp_path/'out')))
    config = TrainingConfig.from_dict(value, source=tmp_path/'request.json')
    def forbidden(*args, **kwargs):
        pytest.fail('replica policy refusal must precede data, model and RNG')
    for key in ('training_data', '_seed', '_construct_model'):
        monkeypatch.setattr(session, key, forbidden)
    invalid = [dict(mode='auto'), dict(mode='plan', plan='missing'), CUDA_MANUAL,
        dict(REPLICATED_MANUAL, order=[]), dict(mode='reference', implementation={}),
        dict(mode='reference', adapter='native'), dict(mode='reference', memory_limit_bytes=1),
        *[dict(REPLICATED_MANUAL, implementation=dict(REPLICATED_MANUAL['implementation'], reduction=kind))
          for kind in ('ddp_overlap', 'fsdp2_reshard', 'hsdp2_reshard', 'per_module')],
        dict(REPLICATED_MANUAL, implementation=dict(REPLICATED_MANUAL['implementation'], bucket_cap_mb=False)),
        dict(REPLICATED_MANUAL, implementation=dict(REPLICATED_MANUAL['implementation'],
            local=dict(CUDA_MANUAL['implementation'], share_immutable=1)))]
    for execution in invalid:
        with pytest.raises(ValueError, match='REPLICATED execution requires'):
            session.construct(replace(config, execution=execution))
    value['execution'] = dict(mode='reference')
    TrainingConfig.from_dict(value, source=tmp_path/'reference.json')
    value['trainer']['distributed']['backend'] = 'nccl' if backend == 'gloo' else 'gloo'
    with pytest.raises(ValueError, match='backend'):
        TrainingConfig.from_dict(value, source=tmp_path/'wrong-backend.json')
    assert not torch.cuda.is_initialized()


def _owned_group(monkeypatch):
    """Admission-only witness; actual Gloo/NCCL tests below never mock ranks."""
    for key, value in dict(WORLD_SIZE=2, RANK=1, LOCAL_WORLD_SIZE=2, LOCAL_RANK=1, GROUP_RANK=0).items():
        monkeypatch.setenv(key, str(value))
    for name, value in dict(is_initialized=True, get_world_size=2, get_rank=1, get_backend='gloo').items():
        monkeypatch.setattr(torch.distributed, name, lambda *args, _value=value, **kwargs: _value)


@pytest.mark.parametrize('family,module', FAMILIES)
def test_video_rank_owner_rejected_before_assets(family, module, monkeypatch):
    api = importlib.import_module('distillschedule.training.' + module)
    _owned_group(monkeypatch)
    parameters = dict(assets='MISSING', device='cpu', trainable=dict(mode='lora'), student_lr=.003,
        fake_lr=.004, scheduler_gamma=.95, ema_decay=.9, generation_sigmas=[1.,.75,.5,.25,0.])
    class ReachedAssets(Exception): pass
    def assets(*args, **kwargs):
        assert kwargs == dict(device=torch.device('cpu'))
        raise ReachedAssets
    monkeypatch.setattr(api, 'load_media_bundle', assets)
    with pytest.raises(ReachedAssets):
        api.build_model(parameters)
    for key, value in [('WORLD_SIZE','3'), ('RANK','0'), ('LOCAL_WORLD_SIZE','3'),
                       ('LOCAL_RANK','0'), ('GROUP_RANK','1'), ('LOCAL_WORLD_SIZE',None)]:
        with monkeypatch.context() as patch:
            patch.delenv(key) if value is None else patch.setenv(key,value)
            with pytest.raises(ValueError, match='torchrun|ownership'):
                api.build_model(parameters)
    for name, value in [('is_initialized',False), ('get_world_size',1), ('get_backend','nccl')]:
        with monkeypatch.context() as patch:
            patch.setattr(torch.distributed,name,lambda *args, _value=value, **kwargs: _value)
            with pytest.raises(ValueError, match='torchrun|ownership|Gloo'):
                api.build_model(parameters)
    with torch.autocast('cpu',dtype=torch.bfloat16), pytest.raises(ValueError,match='FP32'):
        api.build_model(parameters)
    # The training-only opt-in never admits distributed portable inference.
    with pytest.raises(ValueError,match='distributed'):
        api.build_inference_model({key:{} for key in api._DESCRIPTOR})
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('family,module', FAMILIES)
def test_video_rank_data_policy_and_family_binding_remain_exact(family,module,monkeypatch):
    from types import SimpleNamespace
    from distillschedule.training.checkpoint import rng_state
    from test_training_end_to_end import equal
    api = importlib.import_module('distillschedule.training.' + module)
    _owned_group(monkeypatch)
    schedule = [1.,.75,.5,.25,0.]
    state = {name:SimpleNamespace(device=torch.device('cpu'),binding='exact',generation_sigmas=schedule)
             for name in ('student','fake','teacher','media')}
    request = SimpleNamespace(trainer=dict(device='cpu',distributed=dict(backend='gloo')),
        data=dict(sharding='rank_strided'),execution=dict(mode='reference'),
        algorithm=dict(factory='distillschedule.training.dmd:build_algorithm',parameters=dict(
            teacher_cfg=1,fake_cfg=1,fake_updates=5,rollout_sigmas=schedule)))
    metadata = dict(schema_version=api.INPUT_SCHEMA,family=family,binding='exact')
    before = rng_state(device='cpu')
    for execution in (dict(mode='reference'),deepcopy(REPLICATED_MANUAL)):
        request.execution = execution
        api.admit_training(state=state,request=request,prepared_identity='test',prepared_manifest=dict(metadata=metadata))
    assert equal(before,rng_state(device='cpu'))
    for key,value in [('schema_version','other'),('family','other'),('binding','stale')]:
        with pytest.raises(ValueError,match='binding|family|stale'):
            api.admit_training(state=state,request=request,prepared_identity='test',
                prepared_manifest=dict(metadata={**metadata,key:value}))
    for field,value in [('data',{}),('data',dict(sharding='rank_strided',streams={})),
                        ('trainer',dict(device='cpu',distributed=dict(backend='nccl'))),
                        ('execution',CUDA_MANUAL),('execution',dict(mode='auto'))]:
        bad = deepcopy(request); setattr(bad,field,value)
        with pytest.raises(ValueError,match='rank_strided|backend|private flow|family'):
            api.admit_training(state=state,request=bad,prepared_identity='test',prepared_manifest=dict(metadata=metadata))
    state['fake'].device = torch.device('cuda:0')
    with pytest.raises(ValueError,match='ownership'):
        api.admit_training(state=state,request=request,prepared_identity='test',prepared_manifest=dict(metadata=metadata))


@pytest.mark.parametrize('family,module', FAMILIES)
def test_video_replication_does_not_broaden_native_b1_features(family,module,monkeypatch):
    from test_training_flow_video_device import _row
    api = importlib.import_module('distillschedule.training.' + module)
    _owned_group(monkeypatch)
    row,config,empty = _row(module)
    role = api.Role(torch.nn.Linear(2,2),config=config,media_config={},binding='00'*32,
        empty=empty,trainable={},schedule=[1.,.75,.5,.25,0.])
    assert role.validate_feeds(row) is row
    for fault in ('batch','shape','cache','i2v','binding','null','nonfinite'):
        bad = deepcopy(row)
        if fault == 'batch': bad['latent_shape'][0] = 2
        elif fault == 'shape': bad['latents']['video'] = torch.zeros(1,4,1,2,2)
        elif fault == 'cache': bad['kv_cache'] = torch.zeros(1)
        elif fault == 'i2v': bad['image'] = torch.zeros(1)
        elif fault == 'binding': bad['encoder_identity'][0] ^= 1
        elif fault == 'null': bad['null_context'][0,0,0] += 1
        else: bad['latents']['video'][0,0,0,0,0] = float('nan')
        with pytest.raises(ValueError): role.validate_feeds(bad)


@pytest.mark.parametrize('family,module', FAMILIES)
def test_video_replica_metadata_preserves_single_device_and_inference(family,module,monkeypatch):
    from distillschedule.training import distributed
    from test_training_flow_video_device import _row
    from distillschedule.training.checkpoint import rng_state
    from test_training_end_to_end import equal
    api = importlib.import_module('distillschedule.training.' + module)
    row,config,empty = _row(module)
    class MetadataOnlyMedia(torch.nn.Linear):
        def decode(self,*args): raise AssertionError('metadata-only fixture cannot decode')
    state = {name:api.Role(torch.nn.Linear(2,2).requires_grad_(name!='teacher'),config=config,
        media_config={},binding='00'*32,empty=empty,trainable={},schedule=[1.,.75,.5,.25,0.])
        for name in ('student','fake','teacher')}
    state['media'] = MetadataOnlyMedia(2,2).requires_grad_(False)
    descriptor = dict(evidence_kind='SYNTHETIC_TEST_ONLY')
    before = rng_state(device='cpu')
    monkeypatch.setattr(distributed,'rank_world',lambda:(0,1))
    single = api._components(state,descriptor,training=True)
    assert single.metadata['precision'] == 'FP32_NATIVE_SINGLE_DEVICE_B1' and 'replication' not in single.metadata
    assert single.capabilities == ('video','flow','lora','dmd_flow_v1')
    monkeypatch.setattr(distributed,'rank_world',lambda:(1,2))
    replicas = api._components(state,descriptor,training=True)
    assert replicas.capabilities == (*single.capabilities,'dmd_flow_private_replicated_b1_v1')
    assert replicas.metadata['replication'] == 'EXPLICIT_EQUAL_RANK_LOCAL_B1_POST_BACKWARD_MEAN'
    assert replicas.metadata['precision'] == 'FP32_NATIVE_RANK_LOCAL_REPLICATED'
    inference = api._components({key:state[key] for key in ('student','media')},descriptor,training=False)
    assert inference.metadata == single.metadata and inference.capabilities == single.capabilities
    assert equal(before,rng_state(device='cpu'))


@pytest.mark.parametrize('family,module', FAMILIES)
def test_video_fixture_explicit_samples_keep_legacy_defaults(family,module,tmp_path):
    from distillschedule.training.plugins import load_callable
    recipe = Path(__file__).resolve().parents[1]/f'recipes/{module}_dmd/create_cpu_fixture.py'
    build,_ = load_callable(str(recipe)+':build_fixture')
    parameters = inspect.signature(build).parameters
    assert parameters['cycles'].default == 3 and parameters['samples'].default is None
    for values in (dict(samples=0),dict(samples=True),dict(cycles=0),dict(cycles=True)):
        with pytest.raises(ValueError,match='positive explicit'):
            build(tmp_path/'must-not-exist',**values)
    assert not (tmp_path/'must-not-exist').exists()


def test_portable_replica_cases_keep_image_proofs_and_native_video_geometry():
    from native_flow_replicated import _portable_case
    assert _portable_case('flux2-klein-base') == _portable_case('qwen-image') == (
        'image',[1,4,4,4],(8,8,3),'a blue boat on a lake')
    assert _portable_case('wan22-ti2v5b-t2v')[:3] == ('video',[1,4,2,2,2],(5,32,32,3))
    assert _portable_case('hunyuan15-t2v')[:3] == ('video',[1,4,2,4,4],(5,16,16,3))
    with pytest.raises(AssertionError): _portable_case('ltx25')


def native_video_replica_case(tmp_path,module,device):
    from native_flow_replicated import request, lifecycle
    repo = Path(__file__).resolve().parents[1]
    recipe = repo/f'recipes/{module}_dmd/create_cpu_fixture.py'
    result = subprocess.run([sys.executable,str(recipe),str(tmp_path/'fixture'),'--samples','6'],
        capture_output=True,text=True,timeout=300,
        env={**os.environ,'OMP_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':''})
    (tmp_path/'fixture.log').write_text(result.stdout+'\n'+result.stderr)
    assert result.returncode == 0,result.stdout+result.stderr
    files = json.loads(result.stdout)
    config = request(files['config'],tmp_path/'replicated.json',device)
    command = [sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node','2',
        str(repo/'tests/native_flow_replicated.py'),'oracle','--config',str(config),'--output',str(tmp_path/'oracle')]
    result = subprocess.run(command,capture_output=True,text=True,timeout=600,
        env={**os.environ,'OMP_NUM_THREADS':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8'})
    (tmp_path/'oracle.log').write_text(result.stdout+'\n'+result.stderr)
    assert result.returncode == 0,result.stdout+result.stderr
    proof = json.loads((tmp_path/'oracle/proof.json').read_text())
    assert len(proof['ranks']) == 2 and all(row['average_before_clip_adam'] for row in proof['ranks'])
    assert all(row['oracle_cycles'] == 3 and row['later_student_gradient_nonzero'] for row in proof['ranks'])
    assert all(len(record['cycles']) == 3 for row in proof['ranks'] for record in row['records'])
    lifecycle(config,tmp_path/'lifecycle',ranks=2)


@pytest.mark.skipif(os.environ.get('DMD_MODERN_FLOW_CPU') != '1' or os.environ.get('DMD_NATIVE_FLOW_REPLICATED') != '1',
    reason='native video Gloo2 NOT_RUN: requires pinned stack and DMD_MODERN_FLOW_CPU=1 DMD_NATIVE_FLOW_REPLICATED=1')
@pytest.mark.parametrize('module',['wan22','hunyuan_video'])
def test_actual_native_video_gloo2_independent_oracle_and_fresh_cli(tmp_path,module):
    native_video_replica_case(tmp_path,module,'cpu')
