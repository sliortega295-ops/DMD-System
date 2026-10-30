"""W2 full SafeBundle/native history CPU contract proofs, never GPU acceptance."""
from copy import deepcopy
from contextlib import contextmanager
import json
from types import SimpleNamespace as NS

import pytest

from distillschedule.training.config import TrainingConfig
from test_training_helios_native_history import KIND, POLICY, task, controls, metadata_stream, phase_protocol
from test_training_helios_stream import config

PRESET = 'helios_nogan.native_history_safe_ed8.v1'


def native_safe_config(tmp_path, *, cycles=2):
    raw = config(tmp_path, cycles=cycles, mode='manual').to_dict()
    raw['data']['input_kind'] = KIND
    raw['execution']['preset'] = PRESET
    return TrainingConfig.from_dict(raw, source=tmp_path/'native-safe.json')


def test_new_preset_routes_full_bundle_without_widening_old_native_or_safe_defaults(tmp_path):
    from distillschedule.training import native, helios_native_safe_stream as combined
    from distillschedule.training.native_presets import describe_preset, preset_implementation
    from distillschedule.training.helios_safe_stream import implementation
    conf = native_safe_config(tmp_path)
    assert native._options(conf, 'W2') == 'manual'
    result = native.describe_native(conf)
    assert result['preset']['id'] == PRESET
    assert result['capabilities']['native_history'] == combined.history_descriptor()
    assert result['capabilities']['physical_policy'].startswith('node-HSDP/C512/p0/window5/full SafeBundle/stride0')
    assert preset_implementation('W2', PRESET) == implementation()
    assert describe_preset('W2', PRESET)['continuous_training'] is True
    for patch in ({'preset':'helios_nogan.safe_stream_ed8.v1'}, {'mode':'reference'}, {'plan':'old.json'}):
        raw = conf.to_dict(); raw['execution'].update(patch)
        with pytest.raises(ValueError): native._options(TrainingConfig.from_dict(raw, source=conf.source), 'W2')
    old = config(tmp_path, mode='manual').to_dict(); old['execution']['preset']='helios_nogan.safe_stream_ed8.v1'
    assert native._options(TrainingConfig.from_dict(old, source=conf.source), 'W2') == 'manual'


@pytest.mark.parametrize('mutation', ['normal_input','mode','transport','implementation','capacity_policy','shape_task'])
def test_combined_candidate_rejects_unbound_contracts(tmp_path, mutation):
    from distillschedule.training import native
    raw = native_safe_config(tmp_path).to_dict()
    if mutation == 'normal_input': raw['data']['input_kind']='original_helios_cohort_stream.v1'
    elif mutation == 'mode': raw['execution']['mode']='reference'
    elif mutation == 'shape_task': raw['execution']['preset']='helios_nogan.safe_ed8.v1'
    else: raw['execution'][mutation]='unbound'
    with pytest.raises(ValueError): native._options(TrainingConfig.from_dict(raw, source=tmp_path/'bad.json'), 'W2')


def test_real_native_stream_producer_consumer_lowering_and_separate_checkpoint_identity(metadata_stream,tmp_path,monkeypatch):
    from distillschedule.training import helios_worker as worker, helios_native_safe_stream as combined
    from distillschedule.training.helios_reference_assets import SCHEMA as ASSETS
    from distillschedule.search.helios_request import OriginalHeliosRequest
    from distillschedule.training.helios_lifecycle import _checkpoint_policy
    root, stream = metadata_stream
    assets=dict(schema_version=ASSETS, assets={name:dict(path=f'/assets/{name}') for name in (
        'generator_model','score_model','generator_precast','score_precast','logical_checkpoint','negative_prompt')})
    path=tmp_path/'assets.json'; path.write_text(json.dumps(assets))
    monkeypatch.setattr(worker,'reference_assets',lambda value:value)
    monkeypatch.setattr(worker,'verify_reference_pins',lambda value:dict(evidence_kind='SYNTHETIC_TEST_ONLY'))
    monkeypatch.setattr(worker,'load_reference_bootstrap',lambda *a,**k:(NS(manifest_hash='a'*64),{}))
    raw=native_safe_config(tmp_path).to_dict(); raw['data']['prepared']=str(root/'stream.json')
    conf=TrainingConfig.from_dict(raw,source=tmp_path/'native-safe.json')
    admitted=worker.admit_native_safe_stream(conf,OriginalHeliosRequest().to_dict(),assets,dict(assets=path,output=tmp_path/'run'))
    assert admitted['selected']==combined.descriptor()
    assert admitted['stream']==stream and admitted['selected']['native_history']==combined.history_descriptor()
    argv=admitted['argv']; assert argv[argv.index('--mesh-policy')+1]=='node-hsdp'
    assert argv[argv.index('--attention-backend')+1]=='local-fa2'
    assert '--lora-checkpoint-replay-scope' in argv and '--lora-b-gemm-epilogue' in argv
    assert '--no-generator-nograd-persistent-unshard' in argv
    assert argv[argv.index('--selective-flash-checkpoint-stride')+1]=='0'
    identity={'schema_version':combined.IDENTITY,'native_history':combined.history_descriptor(),
              'execution_policy':combined.descriptor()}
    policy=_checkpoint_policy(identity, optimized=True)
    assert policy['schema_version']==combined.CHECKPOINT
    with pytest.raises(ValueError): _checkpoint_policy(identity,optimized=False)
    wrong=deepcopy(identity); wrong['execution_policy']['preset']='helios_nogan.safe_stream_ed8.v1'
    with pytest.raises(ValueError): _checkpoint_policy(wrong,optimized=True)
    with pytest.raises(ValueError): worker.admit_safe_stream(conf,OriginalHeliosRequest().to_dict(),assets,dict(assets=path,output=tmp_path/'old'))


@pytest.mark.parametrize('failure',[False,True])
def test_actual_planner_retains_window5_and_complete_block_phase_with_full_bundle(phase_protocol,tmp_path,monkeypatch,failure):
    from distillschedule.runtime.helios_planner_cycle import _prepare_cycle
    from distillschedule.runtime.role_runner import OperationContext
    from distillschedule.runtime.version_sync import VersionLedger
    from distillschedule.runtime.fsdp2_p0 import StaticFSDP2P0Policy
    from distillschedule.training import helios_native_safe_stream as combined, helios_lifecycle as lifecycle
    from distillschedule.runtime import helios_native_safe_history as history
    engine,_,_=phase_protocol
    # Physical CUDA layout admission is deliberately isolated here. The exact
    # production owner still rejects CPU DTensors; real CPU HSDP runs separately.
    monkeypatch.setattr(history,'admit_engine',lambda value:dict(checkpoint_blocks=40,fsdp_units=41))
    class Observer:
        mode,rank,device='native',0,'cpu'
        impl=combined.implementation()
        generator,critic=engine,object()
        def __init__(self):self.rows=[]
        def versions(self):return dict(generator=engine.global_steps,critic=0,ema=0)
        @contextmanager
        def action(self,op,*,source_steps):
            row=dict(action_id=op.operation_id,source_steps=source_steps,memory_after={});self.rows.append(row);yield row
        def produced(self,step,output): engine.events.append(('produced',step))
    observer=Observer();token=object()
    def original_g(**kwargs):
        view=kwargs['generator_engine'];assert view is not engine
        assert all(unit.group.is_unsharded for unit in engine.module.units)
        for i in range(8):view(i)
        if failure:raise RuntimeError('original helper error')
        view.backward('loss');view.step();return token
    def original_rollout(**kwargs):
        assert all(unit.group.is_unsharded for unit in engine.module.units)
        kwargs['generator_engine'](9)
        return {10:token},[{}],dict(enabled=False,restored=True)
    p0=NS(_run_nogan_generator_update_p0=original_g,_precompute_critic_rollouts=original_rollout)
    policy=StaticFSDP2P0Policy(mesh_policy='node-hsdp',**{k:v for k,v in combined.effective_flags().items()
        if k in StaticFSDP2P0Policy.__dataclass_fields__})
    binding=dict(config=native_safe_config(tmp_path),implementation=combined.implementation(),
        identity=dict(schema_version=combined.IDENTITY,native_history=combined.history_descriptor(),execution_policy=combined.descriptor()),
        effective_configuration=combined.descriptor())
    with lifecycle.original_training_scope(binding):
        prepared,graph,order=_prepare_cycle(observer=observer,p0=p0,args=NS(timing_mode=True,profile_mode=False,profile_diagnostics=False),
            policy=policy,steps=task().source_steps,trace=NS(task=task()),negative=token,cfg=None,diagnostics=token,generator_free_memory={})
        assert order.index('r4')<order.index('c0')
        from distillschedule.core.enums import OperationKind as Kind
        assert [sum(op.kind==kind for op in graph.operations) for kind in (
            Kind.GENERATOR_UPDATE,Kind.CRITIC_ROLLOUT,Kind.CRITIC_UPDATE,Kind.EMA_UPDATE)]==[1,5,5,5]
        op,callback=prepared._bound[0];ctx=OperationContext(op,VersionLedger(observer.versions()))
        if failure:
            with pytest.raises(RuntimeError,match='original helper'):callback(ctx)
        else:
            assert callback(ctx) is token
            op,callback=next((op,fn) for op,fn in prepared._bound if op.operation_id=='r0')
            assert callback(OperationContext(op,VersionLedger(observer.versions())))=={10:token}
    for row in observer.rows:
        receipt=row['native_history_phase'];assert receipt['schema_version']==combined.history_descriptor()['schema_version']
        assert receipt['unshard_count']==receipt['reshard_count']==41 and receipt['restored']
        assert receipt['commit_count']==int(row['action_id']=='g' and not failure)
    assert all(not unit.group.is_unsharded for unit in engine.module.units)


def test_worker_identity_construction_and_checkpoint_bind_new_policy_without_model_construction(metadata_stream,tmp_path,monkeypatch):
    from distillschedule.training import helios_worker as worker, helios_lifecycle as lifecycle, helios_native_safe_stream as combined
    from distillschedule.training.helios_reference_assets import SCHEMA as ASSETS
    from distillschedule.runtime.fsdp2_p0 import StaticFSDP2P0Policy
    _,stream=metadata_stream
    assets=dict(schema_version=ASSETS,assets={})
    conf=native_safe_config(tmp_path)
    selected=combined.descriptor()
    admitted=dict(stream=stream,assets=assets,implementation=combined.implementation(),selected=selected,
        argv=[],bootstrap=dict(logical_manifest_hash='a'*64))
    source=dict(identity=dict(head='SYNTHETIC_TEST_ONLY_SOURCE',dirty=False))
    policy=StaticFSDP2P0Policy(mesh_policy='node-hsdp')
    for key,value in {'WORLD_SIZE':'16','LOCAL_WORLD_SIZE':'8','RANK':'0','LOCAL_RANK':'0'}.items():monkeypatch.setenv(key,value)
    monkeypatch.setattr(combined,'validate_environment',lambda **kwargs:None)
    monkeypatch.setattr(lifecycle,'admit_original_output',lambda **kwargs:None)
    seen=[]
    def original_main():
        binding=lifecycle.current_training()
        assert binding['identity']['schema_version']==combined.IDENTITY
        assert binding['identity']['execution_policy']==selected
        assert binding['identity']['native_history']==combined.history_descriptor()
        assert 'reference_policy' not in binding['identity']
        binding['identity']['verified_immutable_assets']=dict(scope='SYNTHETIC_TEST_ONLY')
        lifecycle._bind_native_history_construction(policy=policy)
        assert lifecycle.native_history_cpu_first_construction(policy=policy)
        assert lifecycle._checkpoint_policy(binding['identity'],optimized=True)['schema_version']==combined.CHECKPOINT
        binding['identity']['stream_sha256']='foreign'
        with pytest.raises(ValueError,match='construction identity'):lifecycle.native_history_cpu_first_construction(policy=policy)
        seen.append(True);return 17
    assert worker._run_admitted(conf,admitted,source,backend=NS(main=original_main),optimized=True)==17
    assert seen==[True] and lifecycle.current_training() is None


def test_new_phase_census_cannot_be_satisfied_by_reference_or_wrong_mesh():
    from distillschedule.runtime import helios_native_safe_history as history, helios_native_history as original
    value=task();control=controls(value)[0].operation('step/10/generator/rollout')['control']
    entry=dict(native_history_policy=POLICY,native_history_semantics=original.SEMANTICS,source_rank=0,
        all_history_zero=True,stage_schedule=control['pyramid_stage_call_sequence'],native_task_sha256=value.digest)
    phase=dict(schema_version=history.SCHEMA,execution_requirement=history.REQUIREMENT,action_id='g',task_sha256=value.digest,
        with_grad=True,unshard_count=41,reshard_count=41,backward_count=1,commit_count=1,restored=True,mesh_policy='node-hsdp',mesh_shape=[2,8])
    row=dict(action_id='g',source_rank=0,trace_entries=[entry],native_history_phase=phase)
    history.validate_action(row)
    for mutation in ({'schema_version':original.SCHEMA},{'mesh_shape':[1,16]},{'mesh_policy':'world16-fsdp'},
                     {'task_sha256':'foreign'},{'commit_count':0},{'unshard_count':40},{'restored':False}):
        bad=deepcopy(row);bad['native_history_phase'].update(mutation)
        with pytest.raises(ValueError):history.validate_action(bad)
    assert row['native_history_phase']==phase


def test_unbound_native_safe_phase_rejects_before_collectives(phase_protocol,monkeypatch):
    from distillschedule.runtime import helios_native_safe_history as history
    engine,_,_=phase_protocol
    monkeypatch.setattr(history,'admit_engine',lambda value:pytest.fail('must reject before physical admission or collectives'))
    with pytest.raises(ValueError,match='admitted public lifecycle'):
        with history.generator_phase(engine,task=task(),with_grad=False,action_id='r0'):pytest.fail('unbound entry')


def test_planner_consumer_cannot_treat_full_safe_action_as_old_reference_phase():
    from distillschedule.runtime import helios_native_history as old
    from distillschedule.runtime.helios_planner_cycle import _validate_action
    from distillschedule.training.helios_native_safe_stream import implementation
    value=task();control=controls(value)[0].operation('step/10/generator/rollout')['control']
    entry=dict(native_history_policy=POLICY,native_history_semantics=old.SEMANTICS,source_rank=0,
        all_history_zero=True,stage_schedule=control['pyramid_stage_call_sequence'],native_task_sha256=value.digest)
    row=dict(action_id='g',source_rank=0,trace_entries=[entry],native_history_phase=dict(schema_version=old.SCHEMA,
        action_id='g',execution_requirement=old.REQUIREMENT,task_sha256=value.digest,with_grad=True,
        unshard_count=41,reshard_count=41,backward_count=1,commit_count=1,restored=True))
    with pytest.raises(ValueError,match='SafeBundle actual complete-block'):_validate_action(row,implementation())


def test_actual_checkpoint_publication_and_old_policy_refusal_for_new_identity(tmp_path):
    import torch
    from distillschedule.training import helios_native_safe_stream as combined, helios_lifecycle as lifecycle
    from distillschedule.training.checkpoint import save_checkpoint,read_manifest,restore_checkpoint,rng_state
    from distillschedule.training.native_state import engine_state
    from test_training_native_state import _engine,_update
    from test_training_end_to_end import equal
    # Component proof on actual tiny native engine/Adam, NOT a HSDP owner or
    # an assertion of original DMD loss/EMA/full-model numerical parity.
    engine=_engine();_update(engine)
    identity=dict(schema_version=combined.IDENTITY,native_history=combined.history_descriptor(),
        execution_policy=combined.descriptor(),source='CPU_SYNTHETIC_TEST_ONLY')
    policy=lifecycle._checkpoint_policy(identity,optimized=True)
    path=save_checkpoint(tmp_path/'checkpoint',engine_state({'generator':engine}),identity=identity,
        cycle=1,cursor=1,versions=lifecycle.expected_versions(1),policy=policy)
    assert (path/'_SUCCESS').is_file()
    header=read_manifest(path,identity=identity)
    binding=dict(identity=identity,stream=dict(cohorts=[{},{}]),config=native_safe_config(tmp_path),checkpoint_policy=policy)
    lifecycle.validate_progress(header,binding)
    new=_engine();ids=[id(p) for p in new.module.parameters()]
    restore_checkpoint(path,engine_state({'generator':new}),identity=identity)
    assert ids==[id(p) for p in new.module.parameters()]
    assert equal(engine.optimizer.state_dict(),new.optimizer.state_dict())
    _update(engine);_update(new)
    assert equal(engine.module.state_dict(),new.module.state_dict())
    wrong=deepcopy(identity);wrong['schema_version']='distillschedule.helios_safe_stream_identity.v1'
    before=deepcopy(new.module.state_dict());before_rng=rng_state()
    with pytest.raises(ValueError):restore_checkpoint(path,engine_state({'generator':new}),identity=wrong)
    assert equal(before,new.module.state_dict()) and equal(before_rng,rng_state())


def test_native_safe_recipe_cold_discovery_avoids_torch_cuda_and_model_assets():
    import os,subprocess,sys
    from distillschedule.workloads.best_known.registry import repository_root
    root=repository_root()
    code='''
import json,sys
from pathlib import Path
from distillschedule.training.config import TrainingConfig
from distillschedule.training.native import _options,describe_native
root=Path(sys.argv[1]);p=root/'recipes/helios_nogan/training.native-history-safe.json'
c=TrainingConfig.from_dict(json.loads(p.read_text()),source=p)
assert _options(c,'W2')=='manual'
v=describe_native(c)
assert v['capabilities']['native_history']['mesh_shape']==[2,8]
assert v['preset']['id']=='helios_nogan.native_history_safe_ed8.v1'
assert 'torch' not in sys.modules
print('PASS_METADATA_NO_TORCH_NO_ASSETS')
'''
    env=dict(os.environ,PYTHONPATH=os.pathsep.join((str(root/'src'),str(root/'tests'))),CUDA_VISIBLE_DEVICES='')
    result=subprocess.run([sys.executable,'-c',code,str(root)],env=env,text=True,capture_output=True,check=True,timeout=20)
    assert result.stdout.strip()=='PASS_METADATA_NO_TORCH_NO_ASSETS'
