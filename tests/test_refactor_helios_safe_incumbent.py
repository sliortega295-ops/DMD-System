"""CPU protocol fixtures, never current GPU/cost/numerical evidence."""
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace as NS
import argparse
import ast
import functools
import json
import sys
import types

import pytest

from distillschedule.search.helios_implementation import HeliosImplementation, NO_GAN, GAN
from distillschedule.search.helios_safe_bundle import (
    BUNDLE_ID, B_GEMM_IMPLEMENTATION, FLAG_ENV, HeliosSafeBundle, LORA_COUNTERS, NORM_COUNTERS,
)
from distillschedule.search.helios_model_adapter import safe_incumbent
from distillschedule.workloads.best_known.registry import fingerprint, load_profile

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('variant,mesh,chunk,persistent,environment,profile', [
    ('no-gan','world16-fsdp',512,False,
     'b6fd3e36dc725c121b752dc134816513b6325afc2fb15fb06affd3072bd8da09',
     'dcf1680c1856128ce63fcc3dd2d59d7646017ac0483269958681cdb057df8627'),
    ('no-gan','node-hsdp',512,False,
     '9c90ca3cadcc924c72314df8dab9711d5259807f41817c44e15281f49fa256ea',
     'c17971715360035ea835e1f1794b332c7c7c5cf86b09353f6045f9a4cba7f783'),
    ('gan','world16-fsdp',2048,True,
     'fed709fb6877ab2dc35419aff2b2fb88ab5d364bd51ceb3b149e5c26abcd65c2',
     '2dd0e41663ae9ce76e2cc2759ef1766eda5784f405be4ee93250acc09252bd3a'),
])
def test_old_ids_environment_and_lowered_profiles_are_byte_identity_unchanged(
        variant,mesh,chunk,persistent,environment,profile):
    i=HeliosImplementation(variant,mesh,chunk,5,persistent)
    assert '-k' not in i.id and 'candidate_bundle' not in i.assignment
    assert fingerprint(i.environment())==environment
    assert fingerprint(i.lower_profile())==profile


def test_typed_bundle_roundtrip_exact_scope_and_no_gan_admission():
    i=safe_incumbent()
    assert i.id=='joint-helios1-nogan-hsdp-c512-p0-w5-ksafe-nogan-ed8-v1'
    assert HeliosImplementation.parse(i.id)==HeliosImplementation.from_assignment(i.assignment)==i
    assert all(asdict(i.candidate_bundle).values())
    assert all(i.environment()[k]=='on' for k in FLAG_ENV.values())
    assert i.environment()['DISTILLSCHEDULE_HELIOS_SELECTIVE_FLASH_CHECKPOINT_STRIDE']=='0'
    assert i.lower_profile()['historical']['formal_speedup_eligible'] is False
    with pytest.raises(ValueError,match='all-or-none'):HeliosSafeBundle(lora_b_gemm_epilogue=True)
    with pytest.raises(ValueError,match='booleans'):HeliosSafeBundle(lora_b_gemm_epilogue=1)
    with pytest.raises(ValueError,match='typed'):replace(i,candidate_bundle=BUNDLE_ID)
    for changed in ({'variant':'gan','mesh':'world16-fsdp'}, {'mesh':'world16-fsdp'},
                    {'chunk_tokens':2048},{'rollout_window':4}):
        with pytest.raises(ValueError,match='exactly|GAN'):replace(i,**changed)
    with pytest.raises(ValueError):HeliosImplementation.parse(i.id.replace(BUNDLE_ID,'invented'))


def test_inventory_optin_adds_one_tuple_only_and_no_precision_override():
    from distillschedule.search.helios_model_adapter import implementations,domains,capability_report
    from distillschedule.search.workflow_cli import candidate_inventory
    old=implementations(NO_GAN);new=implementations(NO_GAN,include_safe_incumbent=True)
    assert len(old)==20 and new==(*old,safe_incumbent())
    assert 'candidate_bundle' not in domains(NO_GAN)
    assert domains(NO_GAN,include_safe_incumbent=True)['candidate_bundle']==('none',BUNDLE_ID)
    assert len(candidate_inventory(NO_GAN)['candidates'])==20
    assert len(candidate_inventory(NO_GAN,include_helios_safe_incumbent=True)['candidates'])==21
    for fn in (implementations,capability_report,domains):
        with pytest.raises(ValueError,match='No-GAN'):fn(GAN,include_safe_incumbent=True)
    baseline=HeliosImplementation('no-gan','node-hsdp',512,5,False).environment()
    assert not any(k.startswith('NVIDIA') for k in safe_incumbent().environment())
    assert safe_incumbent().environment()['OMP_NUM_THREADS']==baseline['OMP_NUM_THREADS']


def test_every_action_including_ema_binds_full_fields_and_actual_sources():
    from distillschedule.search.helios_action_costs import component_contract,action_graph
    i=safe_incumbent();off=replace(i,candidate_bundle=HeliosSafeBundle())
    for op in action_graph(i).operations:
        work={'physical_batch':1,'logical_batch':16}
        on=component_contract(op,i,work);old=component_contract(op,off,work)
        bundle=on.implementation['candidate_bundle']
        assert bundle['fields']==asdict(i.candidate_bundle)
        assert bundle['source']==i.candidate_bundle.source_identity()
        assert on.digest!=old.digest and 'candidate_bundle' not in old.implementation
    assert i.candidate_bundle.source_identity()['files'][
        'src/distillschedule/runtime/helios_lora_b_gemm_kernel.py']==(
        '700054fa857437023974352c0c589b8bc232e7a45443e5271bd102ba07539bc5')


def installed_build():
    result={}
    for role in ('generator','critic'):
        lora=dict(enabled=True,scope='ALL_ROLE_SCALE1_MODEL_FORWARD',checkpoint_replay_scope=True,
            lora_b_gemm_epilogue_requested=True,lora_b_gemm_epilogue_enabled=True,
            checkpoint_missing_active_scope_count=0,
            lora_b_gemm_runtime_identity={'implementation':B_GEMM_IMPLEMENTATION},
            **{key:0 for key in LORA_COUNTERS})
        result[role]=dict(all_role_lora_scale1_add=lora,
            checkpoint_policy=dict(lora_checkpoint_replay_scope=True,selective_flash_stride=0))
    result['generator']['exact_modulated_norm_epilogue']=dict(
        **{k:True for k in ('enabled','requested','materialization_requested','backward_requested',
                           'layernorm_requested','backward_kernel_requested')},
        reference_fallback_count=0,backward_kernel_reference_fallback_count=0,
        **{key:0 for key in NORM_COUNTERS})
    return result


def increment_build(build):
    for role in ('generator','critic'):
        for key in LORA_COUNTERS:build[role]['all_role_lora_scale1_add'][key]+=1
    for key in NORM_COUNTERS:build['generator']['exact_modulated_norm_epilogue'][key]+=1


@pytest.mark.parametrize('failure',['zero_calls','missing_scope','fake_kernel','norm_fallback','bad_stride'])
def test_actual_effect_receipt_rejects_metadata_only_missing_or_fallback(failure):
    b=safe_incumbent().candidate_bundle;build=installed_build();before=b.effect_snapshot(build)
    increment_build(build)
    if failure=='zero_calls':build['critic']['all_role_lora_scale1_add']['model_call_count']=0
    if failure=='missing_scope':build['critic']['all_role_lora_scale1_add']['checkpoint_missing_active_scope_count']=1
    if failure=='fake_kernel':build['critic']['all_role_lora_scale1_add']['lora_b_gemm_runtime_identity']['implementation']='TEST_INJECTED_NOT_GPU_PRODUCTION'
    if failure=='norm_fallback':build['generator']['exact_modulated_norm_epilogue']['reference_fallback_count']=1
    if failure=='bad_stride':build['critic']['checkpoint_policy']['selective_flash_stride']=2
    with pytest.raises(ValueError):b.executed_effects(build,before)


def safe_rows(monkeypatch):
    from test_refactor_helios_planner_native import _fixture
    from distillschedule.runtime.helios_planner_cycle import run_two_cycles
    f=_fixture(monkeypatch,window=5,mode='calibration');i=safe_incumbent()
    for key,value in (i.environment()|{'DISTILLSCHEDULE_HELIOS_PLANNER_MODE':'calibration'}).items():
        monkeypatch.setenv(key,value)
    f.kwargs['policy'].mesh_policy=i.mesh
    for name in FLAG_ENV:
        setattr(f.kwargs['args'] if name=='lora_checkpoint_replay_scope' else f.kwargs['policy'],name,True)
    f.result['build']=installed_build()
    original=f.kwargs['p0']._run_nogan_generator_update_p0
    def observed(**kwargs):
        increment_build(f.result['build'])  # Explicit synthetic fixture, not actual kernel evidence.
        row=original(**kwargs)
        row['rollout'].update(synchronize_after_rollout=False,empty_cache_before_rollout=False)
        return row
    f.kwargs['p0']._run_nogan_generator_update_p0=observed
    original_rollout=f.kwargs['p0']._precompute_critic_rollouts
    def observed_rollout(**kwargs):
        outputs,records,persistence=original_rollout(**kwargs)
        for record in records:record['synchronize_after_rollout']=False
        return outputs,records,persistence
    f.kwargs['p0']._precompute_critic_rollouts=observed_rollout
    run_two_cycles(**f.kwargs)
    rows=[]
    for rank in range(16):
        r=deepcopy(f.result);r.update(rank=rank,run_id='CPU_FIXTURE_ONLY')
        for c in r['cycles']:
            for a in c['actions']:
                a['source_rank']=rank
                if 'helper_record' in a:a['helper_record']['source_rank']=rank
                for trace in a['trace_entries']:trace['source_rank']=rank
        rows.append(r)
    return rows,i


def test_native_runtime_and_collector_bind_both_cycle_effects_and_source(monkeypatch):
    from distillschedule.search.helios_native import summarize
    rows,i=safe_rows(monkeypatch)
    report=summarize(json.loads(json.dumps(rows)),expected_head=rows[0]['source_head'],candidate_id=i.id)
    assert report['status']=='PASS' and report['numerical_status']=='NOT_RUN'
    for field in ('candidate_bundle_effects','build','planner_binding'):
        bad=deepcopy(rows);bad[7][field]={}
        with pytest.raises(ValueError):summarize(bad,expected_head=rows[0]['source_head'],candidate_id=i.id)
    bad=deepcopy(rows);bad[7]['cycles'][1]['candidate_bundle_effects']['before']['critic']['model_call_count']=0
    with pytest.raises(ValueError,match='counter chain'):summarize(bad,expected_head=rows[0]['source_head'],candidate_id=i.id)
    bad=deepcopy(rows);bad[7]['cycles'][1]['actions'][0]['helper_record']['rollout']['synchronize_after_rollout']=True
    with pytest.raises(ValueError,match='fence request'):summarize(bad,expected_head=rows[0]['source_head'],candidate_id=i.id)


def test_effective_args_and_pre_model_launcher_reject_missing_flags():
    from distillschedule.search.helios_native import validate_launcher_admission
    i=safe_incumbent();b=i.candidate_bundle
    args=NS(lora_checkpoint_replay_scope=True)
    policy=NS(**{name:True for name in FLAG_ENV if name!='lora_checkpoint_replay_scope'},
              selective_flash_checkpoint_stride=0)
    assert b.bind_effective(args=args,policy=policy)==b.cost_identity()
    del policy.lora_b_gemm_epilogue
    with pytest.raises(ValueError,match='effective'):b.bind_effective(args=args,policy=policy)
    env=i.environment()|{'DISTILLSCHEDULE_HELIOS_PLANNER_MODE':'correctness'}
    assert validate_launcher_admission(env)['status']=='PASS_CORRECTNESS_BOOTSTRAP_ONLY'
    env['DISTILLSCHEDULE_HELIOS_SELECTIVE_FLASH_CHECKPOINT_STRIDE']='2'
    with pytest.raises(ValueError,match='configuration'):validate_launcher_admission(env)


def test_bundle_off_control_rejects_unbound_new_lora_flags_and_effective_policy(monkeypatch):
    from distillschedule.search.helios_implementation import current_implementation
    from test_refactor_helios_planner_native import _fixture
    f=_fixture(monkeypatch,window=5)
    env=f.impl.environment()|{'DISTILLSCHEDULE_HELIOS_PLANNER_MODE':'correctness'}
    for key in ('DISTILLSCHEDULE_HELIOS_LORA_B_GEMM_EPILOGUE',
                'DISTILLSCHEDULE_HELIOS_LORA_CHECKPOINT_REPLAY_SCOPE'):
        with pytest.raises(ValueError,match='bundle-off'):current_implementation(env|{key:'on'})
    f.kwargs['args'].lora_checkpoint_replay_scope=True
    with pytest.raises(ValueError,match='bundle-off'):
        f.impl.receipt_binding(args=f.kwargs['args'],policy=f.kwargs['policy'])
    f.kwargs['args'].lora_checkpoint_replay_scope=False
    f.kwargs['policy'].lora_b_gemm_epilogue=True
    with pytest.raises(ValueError,match='bundle-off'):
        f.impl.receipt_binding(args=f.kwargs['args'],policy=f.kwargs['policy'])


@pytest.mark.parametrize('changed', [dict(variant='gan'),dict(rollout_batch_size=2),
    dict(rollout_batch_size=True),dict(pack_real_score_pair=True),dict(attention_backend='sdpa'),
    dict(checkpoint_replay_scope=False),dict(all_role_scale1_add=False),dict(enabled=1)])
def test_actual_model_independent_port_gate_rejects_unqualified_variants(changed):
    from distillschedule.runtime.fsdp2_p0 import validate_nogan_b_gemm_request
    config=dict(enabled=True,checkpoint_replay_scope=True,all_role_scale1_add=True,
                variant='no-gan',rollout_batch_size=1,pack_real_score_pair=False,attention_backend='local-fa2')
    validate_nogan_b_gemm_request(**config)
    with pytest.raises(ValueError):validate_nogan_b_gemm_request(**(config|changed))


def test_newer_worker_and_sharding_hooks_survive_the_bounded_port():
    worker=(ROOT/'scripts/distillschedule/probe_static_fsdp2_p0_world16.py').read_text()
    checkpoint=(ROOT/'src/distillschedule/runtime/fsdp2_static.py').read_text()
    policy=(ROOT/'src/distillschedule/runtime/fsdp2_p0.py').read_text()
    assert 'expected_block_count' in policy and 'validate_exact_rollout_call_inventory' in policy
    assert '_trajectory_capture_grad_norm' in checkpoint
    assert 'deferred_finite_checks' in worker and 'runtime_cache' in worker
    assert worker.index('validate_nogan_b_gemm_request(')<worker.index('dist.init_process_group(')
    assert worker.index('lora_fused_epilogue = install_generator_nograd_lora_fused_epilogue(')<worker.index('checkpoint_policy = configure_fsdp2_checkpointing(')


def test_two_arm_optin_problem_search_lower_and_mandatory_seed_validation(tmp_path,monkeypatch):
    from test_refactor_helios_search_bridge import native_rows
    from test_refactor_model_planning import BINDING
    from distillschedule.search import helios_model_adapter as m,execution
    from distillschedule.search.cost_model import CostModel,OperationCost
    from distillschedule.search.model_search import search_plans
    from distillschedule.search.model_validation import ModelValidationSession
    from distillschedule.search.helios_action_costs import PRECISION,anchors
    from distillschedule.search.contracts import semantic_contract
    from distillschedule.search.simulator import simulate
    baseline=anchors(NO_GAN)[0];incumbent=safe_incumbent()
    rows,_=native_rows(monkeypatch,window=5)
    identity={**BINDING,'source_head':rows[0]['source_head'],'precision_fingerprint':PRECISION,
              'semantic_fingerprint':fingerprint(semantic_contract(load_profile(NO_GAN)))}
    for r in rows:
        for mem in [r['post_setup_memory'],*[a[k] for c in r['cycles'] for a in c['actions'] for k in ('memory_before','memory_after')]]:
            mem['driver_total_bytes']=40*2**30;mem['driver_free_bytes']=35*2**30
    pair=(baseline,incumbent)
    monkeypatch.setattr(m,'_read_fragment',lambda ref,root: ({'binding':identity},pair[ref['index']],deepcopy(rows)))
    # Only native source/receipt acquisition above is synthetic. Exercise real
    # assembly, hash-verified pooling, seed selection and saved-plan validation.
    # None of these temp receipts pass outside this explicitly injected reader.
    results=[]
    for index,i in enumerate(pair):
        folder=tmp_path/f'arm{index}';folder.mkdir()
        fragment=folder/'fragment.json';fragment.write_text(json.dumps({'CPU_FIXTURE_ONLY':True}))
        fragment_ref=m.ref(fragment)|{'index':index}
        measured_records=[]
        for key,row in m._component_samples(i,rows).items():
            raw=dict(schema_version='distillschedule.operation_calibration.v1',binding=identity,key=key,
                samples_ns=row['samples_ns'],workspace_bytes_per_rank=row['workspace_bytes_per_rank'],
                status='PASS',profiled=False,kind='native_operation',operation_contract_sha256=key,
                component_contract=row['contract'].to_dict(),operation_ids=row['operation_ids'],
                helios_native_fragment=fragment_ref)
            path=folder/f'{key}.json';path.write_text(json.dumps(raw))
            evidence={k:raw[k] for k in ('status','profiled','kind','operation_contract_sha256','component_contract')}
            evidence.update(source_receipt=str(path),source_receipt_sha256=m.file_sha256(path))
            measured_records.append(OperationCost(key,tuple(row['samples_ns']),row['workspace_bytes_per_rank'],evidence))
        model=CostModel(identity,tuple(measured_records));model.verify_receipts()
        path=folder/'cost-model.json';path.write_text(json.dumps(model.to_dict()))
        result=folder/'result.json';result.write_text(json.dumps(dict(status='PASS_NATIVE_COMPONENT_CALIBRATION',
            source_head=identity['source_head'],implementation_id=i.id,layout_fragment=fragment_ref,
            operation_costs=m.ref(path))))
        results.append(result)
    assembled,pooled=m.assemble_calibrations(results,output=tmp_path/'assembled',
        expected_head=identity['source_head'],include_safe_incumbent=True)
    actual_prediction=search_plans(assembled,pooled,max_proposals=80,exhaustive=True)
    assert all(simulate(assembled,s,pooled)['status']=='PREDICTED_FEASIBLE' for s in assembled.seeds)
    actual_session=ModelValidationSession(problem=assembled,costs=pooled,prediction=actual_prediction,
        output=tmp_path/'actual-session')
    assert actual_session.seeds==(baseline.id,incumbent.id)
    actual_chosen=dict(actual_prediction);actual_chosen['predicted_best']=simulate(assembled,assembled.seeds[1],pooled)
    actual_plan=m.lower_prediction(actual_chosen,assembled,pooled,run_id='CPU_FIXTURE_ONLY')
    assert execution.validated_profile(actual_plan,root=ROOT)
    layout=tmp_path/'layout.json'
    data=dict(schema_version=m.SCHEMA,fragments=[{'index':0},{'index':1}],include_safe_incumbent_bundle=True)
    layout.write_text(json.dumps(data));measured=m.build_problem(layout)
    assert len(measured.seeds)==2
    assert [HeliosImplementation.from_assignment(s).id for s in measured.seeds]==[baseline.id,incumbent.id]
    assert measured.seeds[0]['candidate_bundle']=='none'
    p=replace(measured,evidence_scope='CPU_FIXTURE_ONLY')
    records={}
    for i in pair:
        for key,row in m._component_samples(i,rows).items():
            records[key]=OperationCost(key,tuple(row['samples_ns']),row['workspace_bytes_per_rank'],
                                      {'component_contract':row['contract'].to_dict()})
    costs=CostModel(identity,tuple(records.values()),scope='CPU_FIXTURE_ONLY')
    prediction=search_plans(p,costs,max_proposals=80,exhaustive=True)
    assert all(simulate(p,s,costs)['status']=='PREDICTED_FEASIBLE' for s in p.seeds)
    assert any(r['assignment']['candidate_bundle']==BUNDLE_ID for r in prediction['ranked'])
    for changed in ({'mesh':'world16-fsdp'},{'chunk_tokens':'2048'},{'rollout_window':'4'}):
        with pytest.raises(ValueError,match='forbidden'):p.check_assignment(p.seeds[1]|changed)
    # Fixture-only bypass is explicit at the measured-evidence boundary; all
    # assignment/seed/lowering/profile gates under test remain production code.
    monkeypatch.setattr(m,'validate_native_problem',lambda problem:None)
    session=ModelValidationSession(problem=p,costs=costs,prediction=prediction,output=tmp_path/'session')
    assert session.seeds==(baseline.id,incumbent.id)
    chosen=dict(prediction);chosen['predicted_best']=simulate(p,p.seeds[1],costs)
    plan=m.lower_prediction(chosen,p,costs,run_id='CPU_FIXTURE_ONLY')
    assert plan['search_candidate']['id']==incumbent.id
    assert execution.validated_profile(plan,root=ROOT)
    plan['model_plan']['search_result']['predicted_best']['assignment']['candidate_bundle']='none'
    with pytest.raises(ValueError):execution.validated_profile(plan,root=ROOT)
    data['include_safe_incumbent_bundle']=False;layout.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='explicit layout'):m.build_problem(layout)
    data['include_safe_incumbent_bundle']=True;data['fragments']=[{'index':1}];layout.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='requires measured'):m.build_problem(layout)


def actual_function(path,name,namespace):
    node=next(n for n in ast.parse((ROOT/path).read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
    tree=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),node],type_ignores=[])
    ast.fix_missing_locations(tree);exec(compile(tree,str(path),'exec'),namespace)
    return namespace[name]


def test_actual_worker_parser_and_checkpoint_saved_callable_are_default_off(monkeypatch):
    parser=actual_function('scripts/distillschedule/probe_static_fsdp2_p0_world16.py','_parser',
        {'base':NS(parser=argparse.ArgumentParser),'DEFAULT_TARGET_BYTES':128,'__doc__':'CPU fixture'})()
    args=parser.parse_args([])
    assert args.lora_b_gemm_epilogue is args.lora_checkpoint_replay_scope is False
    args=parser.parse_args(['--lora-b-gemm-epilogue','--lora-checkpoint-replay-scope'])
    assert args.lora_b_gemm_epilogue is args.lora_checkpoint_replay_scope is True
    calls=[];checkpoint=types.ModuleType('torch.utils.checkpoint')
    checkpoint.checkpoint=lambda fn,*a,**kw:calls.append((fn,kw)) or fn(*a)
    monkeypatch.setitem(sys.modules,'torch.utils.checkpoint',checkpoint)
    f=actual_function('src/distillschedule/runtime/fsdp2_static.py','fsdp2_metadata_tolerant_checkpoint',{'functools':functools})
    model=NS(__call__=lambda value:value+1)
    assert f(model,2)==3
    assert calls[0][1]==dict(use_reentrant=False,determinism_check='none',preserve_rng_state=True)
    wrapped=[]
    assert f(model,3,function_wrapper=lambda fn:wrapped.append(fn) or (lambda *a:fn(*a)))==4
    assert len(wrapped)==1 and calls[1][0] is not wrapped[0]
