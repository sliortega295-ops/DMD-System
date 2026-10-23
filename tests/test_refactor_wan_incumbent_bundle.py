"""CPU metadata/source protocol tests; these are NOT CUDA/numerical evidence."""
from __future__ import annotations
import ast
import copy
from dataclasses import replace
from contextlib import contextmanager, nullcontext
import json
from pathlib import Path
import os
from types import SimpleNamespace as NS
import pytest

from distillschedule.search.wan_candidate_bundle import (
    ASSETS, CAPABILITIES, FORBIDDEN, INCUMBENT, OFF, WanCandidateBundle)
from distillschedule.search.wan_implementation import WanImplementation, PRECISION, current_implementation
from distillschedule.search.registry import get_candidate, candidate_profile
from distillschedule.strategies.wan_task import OriginalWanTask
from distillschedule.strategies.wan_strategy import WanStrategy, implementations, strategy_environment
from distillschedule.strategies.wan_calibration import component
from distillschedule.runtime.wan_incumbent_evidence import validate_bundle_build

ROOT = Path(__file__).resolve().parents[1]

def incumbent(window=1):
    return WanStrategy('resident', 2048, True, window, INCUMBENT)


def test_old_ids_defaults_and_capabilities_are_byte_shape_compatible():
    old = WanStrategy('resident', 2048, True)
    assert old.to_dict() == dict(placement='resident', chunk_tokens=2048, lora_epilogue=True, rollout_window=1)
    assert old.id == 'joint-wan1-resident-c2048-e1-original-w1'
    assert old.backend.assignment == dict(placement='resident', chunk_tokens='2048', lora_epilogue='on')
    assert len(implementations(OriginalWanTask(), triton_available=True)) == 7
    assert len(implementations(OriginalWanTask(), triton_available=False)) == 4
    assert OFF.to_dict() == dict(schema_version='distillschedule.wan_candidate_bundle.v1',
                                **{k: False for k in CAPABILITIES})
    assert all(old.environment()[k] == '0' for k in [*CAPABILITIES.values(), *FORBIDDEN])


def test_opt_in_registry_and_actual_strategy_round_trip():
    task = OriginalWanTask()
    choices = implementations(task, triton_available=True, incumbent_bundle_available=True)
    assert len(choices) == 8 and choices[-1] == incumbent()
    s = choices[-1]
    assert WanStrategy.from_dict(s.to_dict()) == s
    assert WanImplementation.from_assignment(s.backend.assignment) == s.backend
    assert WanImplementation.parse(s.backend.id) == s.backend
    scheduled = replace(s.backend, rollout_window=5)
    assert WanImplementation.parse(scheduled.id) == scheduled
    assert get_candidate('wan13b_nogan_resident_best', s.backend.id) == s.backend.candidate()
    with pytest.raises(ValueError, match='task-bound'):
        candidate_profile(s.backend.candidate())
    with pytest.raises(ValueError, match='Triton'):
        implementations(task, triton_available=False, incumbent_bundle_available=True)


@pytest.mark.parametrize('field', list(CAPABILITIES))
def test_partial_or_forged_bundle_rejected(field):
    raw = INCUMBENT.to_dict(); raw[field] = False
    with pytest.raises(ValueError, match='coupled'):
        WanCandidateBundle.from_dict(raw)
    raw = INCUMBENT.to_dict(); raw[field] = 1
    with pytest.raises(ValueError, match='boolean'):
        WanCandidateBundle.from_dict(raw)


@pytest.mark.parametrize('placement,chunk,epilogue', [('fsdp',512,False),('resident',512,True),('resident',2048,False)])
def test_coupled_backend_gate(placement,chunk,epilogue):
    with pytest.raises(ValueError, match='resident/chunk2048'):
        WanStrategy(placement,chunk,epilogue,candidate_bundle=INCUMBENT)


@pytest.mark.parametrize('changes', [dict(video_height=512),dict(video_width=512),dict(video_frames=49),
    dict(lora_rank=64),dict(fake_updates=3,first_step=6),dict(first_step=15),dict(ema_every=0),
    dict(stage2_inference_steps=(2,2,3))])
def test_exact_immutable_task_is_not_generalized(changes):
    task = replace(OriginalWanTask(),**changes)
    with pytest.raises(ValueError, match='exact original'):
        incumbent().validate_for(task)
    # Prior independent paths remain available for the new task.
    assert implementations(task,triton_available=True)


@pytest.mark.parametrize('field', list(ASSETS))
def test_every_semantic_asset_is_pinned(field):
    INCUMBENT.validate_assets(ASSETS)
    wrong = dict(ASSETS); wrong[field] = '0'*64
    with pytest.raises(ValueError, match='immutable semantic assets'):
        INCUMBENT.validate_assets(wrong)
    wrong = dict(ASSETS); wrong.pop(field)
    with pytest.raises(ValueError): INCUMBENT.validate_assets(wrong)


def test_actual_model_bytes_must_match_manifest_not_only_its_claimed_identity(monkeypatch):
    from distillschedule.workloads.best_known.registry import fingerprint
    rows=[dict(path='transformer/config.json',bytes=10,sha256='a'*64),
          dict(path='transformer/diffusion_pytorch_model.safetensors',bytes=20,sha256='b'*64)]
    manifest=dict(model_files=rows, components={'generator':{'optimizer_step':2},'critic':{'optimizer_step':10}},
                  ema={'optimization_step':10})
    monkeypatch.setitem(ASSETS,'model_identity_sha256',fingerprint(rows))
    observed={Path(r['path']).name:r['sha256'] for r in rows}
    INCUMBENT.validate_model_files(manifest,observed)
    wrong=dict(observed);wrong['diffusion_pytorch_model.safetensors']='c'*64
    with pytest.raises(ValueError,match='actual model content'):
        INCUMBENT.validate_model_files(manifest,wrong)
    wrong=copy.deepcopy(manifest);wrong['components']['generator']['optimizer_step']=0
    with pytest.raises(ValueError,match='G2'):
        INCUMBENT.validate_model_files(wrong,observed)
    wrong=copy.deepcopy(manifest);wrong['model_files'][0]['sha256']='d'*64
    with pytest.raises(ValueError,match='historical content'):
        INCUMBENT.validate_model_files(wrong,observed)


def test_environment_is_identity_bound_controls_off_and_current_precision_equal(monkeypatch):
    reference=WanStrategy('fsdp',512,False);candidate=incumbent()
    assert reference.environment()['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] == candidate.environment()['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] == '1'
    assert reference.environment()['DISTILLSCHEDULE_WAN13B_PLANNER_PRECISION'] == candidate.environment()['DISTILLSCHEDULE_WAN13B_PLANNER_PRECISION'] == PRECISION.digest
    assert 'NVIDIA_TF32_OVERRIDE' not in candidate.environment()
    for key in CAPABILITIES.values(): monkeypatch.setenv(key,'old')
    with strategy_environment(candidate):
        assert current_implementation() == candidate.backend
        assert all(os.environ[k]=='1' for k in CAPABILITIES.values())
        INCUMBENT.verify_environment(os.environ)
        for field in FORBIDDEN:
            assert os.environ[field]=='0'
        wrong=dict(os.environ);wrong[next(iter(CAPABILITIES.values()))]='0'
        with pytest.raises(ValueError,match='mechanism flags'):INCUMBENT.verify_environment(wrong)
    assert all(os.environ[k]=='old' for k in CAPABILITIES.values())
    with strategy_environment(reference):
        assert all(os.environ[k]=='0' for k in CAPABILITIES.values())


def test_all_component_keys_bind_bundle_capabilities_and_actual_source_bytes():
    task=OriginalWanTask(); old=WanStrategy('resident',2048,True); new=incumbent()
    work={action:{'fixture':'CPU_METADATA_ONLY'} for action in ('g','r0','c0','e0')}
    for action in work:
        a=component(task,action,old,cycle=2,work=work);b=component(task,action,new,cycle=2,work=work)
        assert a.digest != b.digest
        assert b.implementation['candidate_bundle']==INCUMBENT.to_dict()
        assert set(b.implementation['bundle_sources']['runtime_sha256']) >= {'wan13b_lora_cast','wan13b_critic_grad_graph'}
    bad=copy.deepcopy(INCUMBENT.to_dict());bad['schema_version']='unregistered'
    with pytest.raises(ValueError):WanCandidateBundle.from_dict(bad)


@pytest.mark.parametrize('faster', [False, True])
def test_bundle_reaches_common_search_and_is_not_a_default_winner(tmp_path,monkeypatch,faster):
    from test_strategy_wan_planning import collection
    import distillschedule.strategies.wan_planning as planning
    from distillschedule.search.service import select as common_select
    from distillschedule.workloads.best_known.registry import write_new_json
    task=OriginalWanTask();calls=[]
    def observed(candidates,**kwargs):
        calls.append(candidates)
        return common_select(candidates,**kwargs)
    monkeypatch.setattr(planning,'select',observed)
    source=collection(tmp_path,task,[(WanStrategy('fsdp',512,False),100,50,10,200),
                                   (incumbent(),50 if faster else 150,60,10,200)])
    result=planning.plan(task,[source])
    assert len(calls)==1 and len(calls[0])==2
    assert all(c.executable_backend=='original-wan-world16' for c in calls[0])
    assert ('candidate_bundle' in result['selected']['strategy']) is faster
    bundled=next(r for r in result['ranked'] if 'candidate_bundle' in r['strategy'])
    assert [k for k in bundled['order'] if k.startswith('r')]==['r0','r1','r2','r3','r4']
    assert WanStrategy.from_dict(bundled['strategy']).candidate_bundle==INCUMBENT
    target=tmp_path/'saved-plan.json';write_new_json(target,result)
    with pytest.raises(ValueError,match='fixture plans'):
        planning.validate_plan(task,target)


def test_literal_order_and_versions_fail_closed():
    INCUMBENT.validate_order(('g','r0','r1','r2','r3','r4'))
    with pytest.raises(ValueError,match='ascending'):
        INCUMBENT.validate_order(('g','r1','r0','r2','r3','r4'))
    for versions in [dict(generator=0,critic=10,ema=10),dict(generator=2,critic=0,ema=10)]:
        with pytest.raises(ValueError,match='G2'):INCUMBENT.validate_versions(versions)


def _build_receipts():
    result={}
    for role in ('generator','critic'):
        m={'candidate_bundle':INCUMBENT.to_dict(),'candidate_bundle_sources':INCUMBENT.source_identity()}
        for name in CAPABILITIES:
            enabled = name in ('grad_lora_fused_epilogue','modulated_norm_fusion') or (
                role=='generator' and name in ('generator_nograd_lora_cast_cache','generator_rollout_cuda_graph')) or (
                role=='critic' and name=='critic_grad_cuda_graph')
            m[name]={'requested':True,'enabled':enabled}
        result[role]={'metadata':m}
    return result


def test_missing_unrequested_or_different_source_receipts_rejected():
    build=_build_receipts()
    assert not validate_bundle_build(INCUMBENT,build)['actual_compute_observed']
    for field in CAPABILITIES:
        bad=copy.deepcopy(build);bad['generator']['metadata'].pop(field)
        with pytest.raises(ValueError):validate_bundle_build(INCUMBENT,bad)
    bad=copy.deepcopy(build);bad['critic']['metadata']['candidate_bundle_sources']['historical_source_head']='forged'
    with pytest.raises(ValueError,match='source/capability'):validate_bundle_build(INCUMBENT,bad)
    with pytest.raises(ValueError,match='unrequested'):validate_bundle_build(OFF,build)
    with pytest.raises(ValueError,match='two original'):validate_bundle_build(INCUMBENT,build,executed=True,cycles=3)
    with pytest.raises(AssertionError,match='never executed'):validate_bundle_build(INCUMBENT,build,executed=True,cycles=2)


@pytest.mark.parametrize('failure',[False,True])
def test_actual_native_rollout_dispatch_routes_controller_and_always_ends_scope(failure):
    """Execute the real dispatcher body using CPU host-only handler facades."""
    path=ROOT/'scripts/distillschedule/probe_static_fsdp2_p0_world16.py'
    node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_precompute_critic_rollouts')
    code=compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),node],type_ignores=[])),str(path),'exec')
    events=[]
    class Graph:
        active=False
        def begin_rollout(self,**kw):
            assert kw==dict(version=3,logical_steps=(10,));self.active=True;events.append('begin')
        def end_rollout(self):self.active=False;events.append('end')
    graph=Graph();engine=NS(module=NS(_distillschedule_rollout_cuda_graph=graph),global_steps=3)
    @contextmanager
    def persistence(*args,**kwargs):yield {'rollout_batch_count':0}
    def run(**kwargs):
        assert kwargs['model'] is graph and graph.active;events.append('run')
        if failure:raise RuntimeError('injected rollout failure')
        return ('opaque detached B1',),{}
    ns=dict(PACKED_SCORE_MEAN_REL_TOLERANCE=1e-5,chunked=lambda steps,n:[steps], generator_nograd_persistent_unshard_scope=persistence,
        # This is only the dispatch/finally facade, never a CUDA admission or
        # allocation proof. Model work and telemetry are intentionally absent.
        run_generator_trace_rollout_batch=run, torch=NS(device=lambda value:NS(type='cuda'),cuda=NS(**{name:lambda device:0 for name in
            ('memory_allocated','memory_reserved','max_memory_allocated','max_memory_reserved')})))
    exec(code,ns)
    args=dict(rank=0,steps=(10,),device='CPU_HANDLER_ONLY',generator_engine=engine,
        trace=NS(load_critic_entry=lambda **kw:NS(pool_entry=NS(logical_step=10))),
        policy=NS(rollout_batch_size=1,generator_nograd_persistent_unshard=False,
            generator_nograd_persistent_unshard_flush_allocator_cache=False,
            generator_nograd_cross_kv_cache=False,generator_nograd_static_context_cache=False,
            elide_explicit_rollout_sync=False),collect_diagnostics=False,
        diagnostics=NS(scope=lambda label:nullcontext(),add_tensor_tree=lambda *a,**k:None),rollout_parity_check=False,
        clean_runtime=True,runtime_cache={})
    if failure:
        with pytest.raises(RuntimeError,match='injected'):ns[node.name](**args)
    else:
        outputs,records,scope=ns[node.name](**args)
        assert outputs=={10:'opaque detached B1'} and len(records)==1 and scope['rollout_batch_count']==1
    assert events==['begin','run','end'] and not graph.active


def test_native_hooks_not_a_mock_and_no_rejected_mechanism_source():
    hook=(ROOT/'scripts/distillschedule/probe_wan13b_fair_world16.py').read_text()
    for name in ('install_grad_lora_fused_epilogue','install_norm_policy','install_rollout_graph','install_critic_graph'):
        assert name+'(' in hook
    worker=(ROOT/'src/distillschedule/strategies/wan_worker.py').read_text()
    assert 'OriginalWanSession(' in worker and 'validate_build(strategy, session.build, executed=True, cycles=cycles)' in worker
    assert 'validate_model_files' in worker and 'stream_path is None' in worker
    for name in ('wan13b_rollout_shadow.py','wan13b_input_lora_gemm.py'):
        assert not (ROOT/'src/distillschedule/runtime'/name).exists()
    cast=(ROOT/'src/distillschedule/runtime/wan13b_lora_cast.py').read_text()
    assert 'update = lora_b(lora_a(dropout(cast_x)))' in cast and 'fused_input_lora_a' not in cast
    graph=(ROOT/'src/distillschedule/runtime/wan13b_rollout_cuda_graph.py').read_text()
    assert 'version != (3 if self._version is None else self._version + 1)' in graph
    assert 'new CUDA Graph signature after warmup cycle is forbidden' in graph
    assert '27 * cycles' in graph and 'range(10, 15)' in graph


def test_selected_strategy_executes_same_original_session_and_prepared_handlers(tmp_path):
    """CPU handler wiring ONLY; selected JSON is not a fabricated GPU plan."""
    from test_strategy_wan_planning import collection
    from distillschedule.strategies.wan_planning import plan
    from distillschedule.strategies.wan_bridge import OriginalWanSession
    task=OriginalWanTask()
    source=collection(tmp_path,task,[(WanStrategy('fsdp',512,False),100,50,10,200),
                                   (incumbent(),50,60,10,200)])
    selected=plan(task,[source])['selected']
    strategy=WanStrategy.from_dict(json.loads(json.dumps(selected['strategy'])))
    g,c,ema=NS(global_steps=2),NS(global_steps=10),NS(optimization_step=10)
    calls=[]
    def gupdate(**kw):calls.append('g');g.global_steps+=1;return {}
    def rollout(**kw):
        step=kw['steps'][0];calls.append('r'+str(step-10))
        kw['entry_outputs'][step]=NS(tensors={})
        return {step:('opaque_b1',g.global_steps)},[{}],{}
    def critic(**kw):
        assert kw['x0']==('opaque_b1',3)
        calls.append('c'+str(kw['step']-10));c.global_steps+=1;return {}
    def estep():calls.append('e');ema.optimization_step+=1
    ema.step=estep
    def build(**kwargs):
        assert kwargs['args'].original_wan_task==task
        assert current_implementation()==strategy.backend
        return g,c,ema,{}
    fair=NS(base=NS(_build_engines=build,p0=NS(_run_nogan_generator_update_p0=gupdate,
        _precompute_critic_rollouts=rollout,_run_nogan_critic_precomputed=critic)))
    with strategy_environment(strategy):
        session=OriginalWanSession(task=task,strategy=strategy,fair=fair,
            bootstrap=NS(manifest={'lora_rank':256}),model_root='CPU_HANDLER_FIXTURE',device='CPU_HANDLER_ONLY',rank=0,
            trace=NS(variant='no-gan',logical_steps=task.source_steps),negative=None,diagnostics=None,asset_identity=ASSETS)
        prepared=session.prepare(operation_order=tuple(selected['order']))
        assert not calls
        prepared.run()
        assert session.versions()==dict(generator=3,critic=15,ema=15)
        assert [k for k in calls if k.startswith('r')]==['r0','r1','r2','r3','r4']
        assert prepared.receipt()['executed_operations']==list(selected['order'])


def test_public_cli_route_is_existing_measure_search_save_execute_not_old_recipe():
    cli=(ROOT/'src/distillschedule/strategies/wan_cli.py').read_text()
    planning=(ROOT/'src/distillschedule/strategies/wan_planning.py').read_text()
    worker=(ROOT/'src/distillschedule/strategies/wan_worker.py').read_text()
    assert "'--include-incumbent-bundle'" in cli
    assert 'assemble(task, args.rank_receipts' in cli
    assert 'plan(task, args.calibration' in cli and 'write_new_json(args.output, result)' in cli
    assert 'searched=select(bound' in planning
    assert 'document = validate_plan(task, plan_path)' in worker
    assert 'OriginalWanSession(task=task, strategy=strategy' in worker
    assert 'result = execute_cycles(session' in worker
    assert 'subprocess.Popen' not in worker and 'run_best_known_workload' not in worker


def _executed_receipts():
    """Fabricated host metadata for negative gates, never GPU execution proof."""
    from test_refactor_wan13b_rollout_cuda_graph import _receipt
    build=_build_receipts()
    g,c=(build[r]['metadata'] for r in ('generator','critic'))
    for metadata in (g,c):
        metadata['grad_lora_fused_epilogue']['fused_invocation_count']=1
        metadata['modulated_norm_fusion'].update(registered_width=1536,fused_call_count=1,
            layernorm_fused_call_count=1,backward_kernel_call_count=1,reference_fallback_count=0,
            backward_kernel_reference_fallback_count=0)
    g['generator_nograd_lora_cast_cache'].update(cache_capacity=1,cache_hit_count=1,cache_miss_count=1,
        avoided_destination_write_bytes=1024)
    g['generator_rollout_cuda_graph'].update(_receipt())
    c['generator_rollout_cuda_graph']['disabled_reason']='GENERATOR_ONLY'
    c['critic_grad_cuda_graph'].update(replay_calls=10,backward_replay_calls=10,reference_calls=0,
        weights_merged=False,optimizer_reducer_ema_unchanged=True,
        captured_parameter_gradients_return_to_original_accumulators=True,capture_entries=1,
        cold_original_forward_calls=4,cold_original_backward_calls=4)
    return build


def test_full_coupled_receipt_gate_needs_every_executed_mechanism_not_just_flags():
    build=_executed_receipts()
    result=validate_bundle_build(INCUMBENT,build,executed=True,cycles=2)
    assert result['actual_compute_observed'] and result['historical_time_used_as_cost'] is False
    assert result['norm']['counter_scope']=='PROCESS_SHARED; DO_NOT_ADD_DUPLICATE_ROLE_RECEIPTS'
    mutations=[('generator','grad_lora_fused_epilogue','fused_invocation_count',0),
        ('critic','modulated_norm_fusion','backward_kernel_reference_fallback_count',1),
        ('generator','generator_nograd_lora_cast_cache','cache_hit_count',0),
        ('generator','generator_rollout_cuda_graph','graph_replay_model_call_count',53),
        ('critic','critic_grad_cuda_graph','backward_replay_calls',9),
        ('critic','critic_grad_cuda_graph','capture_entries',4)]
    for role,mechanism,key,value in mutations:
        bad=copy.deepcopy(build);bad[role]['metadata'][mechanism][key]=value
        with pytest.raises((ValueError,AssertionError)):
            validate_bundle_build(INCUMBENT,bad,executed=True,cycles=2)
