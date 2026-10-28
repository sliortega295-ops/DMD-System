"""R33 new 100-cohort boundaries; all model/collective standins SYNTHETIC_TEST_ONLY.

Real native scheduler, schemas, dispatch, scalar Adam/cache controllers and
checkpoint IO are exercised, NOT pretrained models, CUDA capture or quality.
"""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from distillschedule.training.config import TrainingConfig
from distillschedule.training import native
from distillschedule.strategies.wan_task import OriginalWanTask
from distillschedule.strategies.wan_stream import WanCohortSources, task_at, order_for_cohort
from distillschedule.strategies.wan_inputs import WanInputSources
from distillschedule.training.helios_inputs import HeliosCohortTask, HeliosCohortSources
from distillschedule.workloads.best_known.registry import fingerprint


def configuration(tmp_path, workload, optimized=False, cycles=100):
    preset = {'W1':'wan_ar.incumbent21a_stream.v2', 'W2':'helios_nogan.safe_stream_ed8.v1'}
    kind = {'W1':'original_wan_cohort_stream.v1', 'W2':'original_helios_cohort_stream.v1'}
    return TrainingConfig.from_dict(dict(schema_version='distillschedule.training.v1',
        model=dict(workload=workload), algorithm={}, data=dict(input_kind=kind[workload], prepared='stream.json'),
        execution=dict(adapter='native', mode='manual' if optimized else 'reference',
            task='task.json', assets='assets.json', **(dict(preset=preset[workload]) if optimized else {})),
        trainer=dict(cycles=cycles, checkpoint_every=50, output_dir=str(tmp_path/'run'))),
        source=tmp_path/'training.json')


@pytest.mark.parametrize('workload,optimized', [('W1',False),('W1',True),('W2',False),('W2',True)])
def test_public_fresh_horizon_contract_is_100(tmp_path, workload, optimized):
    config = configuration(tmp_path, workload, optimized)
    assert native._options(config, workload) == ('manual' if optimized else 'reference')
    report = native.describe_native(config)
    assert report['capabilities']['cycles']['max'] == 100
    assert report['gpu_execution'] == 'NOT_RUN'
    if optimized:
        assert report['preset']['cycles']['max'] == 100
        assert report['preset']['numerical_status'] == 'REVIEW'


@pytest.mark.parametrize('workload', ['W1','W2'])
@pytest.mark.parametrize('bad', [101, True, 100.0])
def test_public_horizon_rejects_101_bool_float(tmp_path, workload, bad):
    with pytest.raises(ValueError):
        native._options(configuration(tmp_path, workload, cycles=bad), workload)


def test_100_one_seed_identities_and_absolute_source_ema_clock(tmp_path):
    task = OriginalWanTask(ema_every=3)
    sources = WanCohortSources(tuple(WanInputSources(
        tuple(str(tmp_path/f'cohort-{i}/rank-{r}.pt') for r in range(16)), seed=42, cohort=i) for i in range(100)))
    assert len({(s.seed,s.cohort) for s in sources.cohorts}) == 100
    assert WanCohortSources.from_dict(sources.to_dict()) == sources
    assert [task_at(task,i).first_step for i in (63,64,99)] == [325,330,505]
    assert [HeliosCohortTask(i).source_steps for i in (63,64,99)] == [
        tuple(range(s,s+5)) for s in (325,330,505)]
    order = ('g', *(f'r{i}' for i in range(5)), *(f'c{i}' for i in range(5)))
    observed = []
    for i in range(100):
        current = task_at(task,i)
        assert HeliosCohortTask.from_dict(HeliosCohortTask(i).to_dict()) == HeliosCohortTask(i)
        observed.extend(current.source_steps[int(key[1:])] for key in order_for_cohort(order,current) if key.startswith('e'))
    assert observed == [step for step in range(10,510) if (step+1)%3 == 0]
    for bad in (100,101,True,99.0):
        with pytest.raises(ValueError): HeliosCohortTask(bad)
        with pytest.raises(ValueError): task_at(task,bad)
    with pytest.raises(ValueError): WanCohortSources(sources.cohorts+(replace(sources.cohorts[0],cohort=100),))
    with pytest.raises(ValueError,match='distinct logical RNG'):
        WanCohortSources(sources.cohorts[:-1]+(sources.cohorts[0],))


def test_actual_native_scheduler_at_63_64_99_keeps_shape_budget_shift_and_clamps():
    import torch
    pytest.importorskip('ftfy',reason='actual native helper closure unavailable; no optional dependency install in R33')
    from distillschedule.strategies.wan_capture import native_primitives, stage_control, input_stream
    primitives = native_primitives(source_only=True)
    task = OriginalWanTask()
    with torch.random.fork_rng(devices=[]):
        for index in (63,64,99):
            current = task_at(task,index)
            control = stage_control(primitives,current,42,index,current.first_step,'generator')
            stages = control['pyramid_stage_call_sequence']
            assert [s['denoising_step_budget'] for s in stages] == [2,2,2]
            assert [s['latent_shape'] for s in stages] == [[1,16,9,*shape] for shape in task.stage_spatial_shapes]
            assert all(1 <= s['call_count'] <= 2 for s in stages)
            for step in current.source_steps:
                with input_stream(42,index,step,'horizon-test'):
                    result = primitives.post.sample_dynamic_timestep(B=16,global_step=step,dynamic_step=500,
                        dynamic_alpha=4.,dynamic_beta=1.5,dynamic_sample_type='beta',device='cpu')
                assert bool(((result >= 20)&(result <= 980)).all())
        # Curriculum saturates, rather than rejecting or replaying step0.
        torch.manual_seed(773)
        at500 = primitives.post.sample_dynamic_timestep(B=128,global_step=500,dynamic_step=500,
            dynamic_sample_type='beta',device='cpu')
        torch.manual_seed(773)
        at509 = primitives.post.sample_dynamic_timestep(B=128,global_step=509,dynamic_step=500,
            dynamic_sample_type='beta',device='cpu')
        assert torch.equal(at500,at509)
    assert not torch.cuda.is_initialized()


def test_full_pinned_native_scheduler_class_has_no_cohort64_horizon(monkeypatch):
    """Whole source-class CPU execution, independent of optional text/model helpers."""
    import importlib.util
    import sys
    import torch
    from distillschedule.adapters.helios.reference_model import reference_model_identity, _SourceOnlyLoader
    source = reference_model_identity(backend='torch-sdpa')['source']
    manifest_path = Path(source['manifest_path'])
    relative = 'source/helios/scheduler/scheduling_helios.py'
    digest = next(row['sha256'] for row in json.loads(manifest_path.read_text())['files'] if row['path']==relative)
    name = '_r33_whole_pinned_scheduler_test'
    path = manifest_path.parent/relative
    spec = importlib.util.spec_from_file_location(name,path,loader=_SourceOnlyLoader(name,path,digest))
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules,name,module)
    spec.loader.exec_module(module)  # Intact pinned module; no source extraction or substitute equations.
    for index in (63,64,99):
        task = HeliosCohortTask(index)
        scheduler = module.HeliosScheduler(shift=1.,stages=3,stage_range=[0,1/3,2/3,1],gamma=1/3)
        for stage,budget in enumerate(task.stage2_inference_steps):
            scheduler.set_timesteps(budget+1,stage,device='cpu')
            assert len(scheduler.timesteps[:-1]) == budget == 2
            assert bool(torch.isfinite(scheduler.timesteps).all())
            assert bool(torch.isfinite(scheduler.sigmas).all())
    assert not torch.cuda.is_initialized()


def test_helios_actual_stream_metadata_loader_admits_100_and_rejects_changed_64(tmp_path, monkeypatch):
    from distillschedule.training import helios_inputs as inputs
    sources, inventories, entries = [],[],[]
    source_pin = inputs.source_fingerprint()
    for i in range(100):
        task = HeliosCohortTask(i)
        source = HeliosCohortSources(42,tuple(tuple(str(tmp_path/f'features/{i}-{s}-{r}.pt')
            for r in range(16)) for s in range(5)))
        sources.append(source.to_dict())
        inventories.append([dict(step=step,rank=rank,path=source.samples[offset][rank],
            sha256=fingerprint([i,offset,rank])) for offset,step in enumerate(task.source_steps) for rank in range(16)])
        entries.append(dict(index=i,task=task.to_dict(),trace_bundle=str(tmp_path/f'cohort-{i:04d}'),trace_hash=fingerprint(i)))
    document = dict(schema_version=inputs.SCHEMA,status='READY_FRESH_COHORT_STREAM',
        identity=dict(request=HeliosCohortTask(0).request.to_dict(),sources=sources,inventory=inventories,
            source_fingerprint=source_pin,training_config_sha256=fingerprint('SYNTHETIC_TEST_ONLY')),cohorts=entries)
    def load(task,path,expected_hash):
        i = task.index
        assert expected_hash == entries[i]['trace_hash']
        controls = {rank:NS(manifest=dict(source=dict(source_fingerprint=source_pin,
            training_config_sha256=document['identity']['training_config_sha256'],rank=rank,
            samples=[r for r in inventories[i] if r['rank']==rank],seed=42,cohort=i,
            rng_scope='new_native_CPU_streams_not_prior_training_rng_continuation'))) for rank in range(16)}
        return NS(_control=controls)
    monkeypatch.setattr(inputs,'load_cohort',load)  # SYNTHETIC_TEST_ONLY trace IO; schema/provenance admission is real.
    def publish(value):
        unsigned = deepcopy(value); unsigned.pop('stream_sha256',None)
        value['stream_sha256'] = fingerprint(unsigned)
        path = tmp_path/'stream.json'; path.write_text(json.dumps(value)); return path
    path = publish(document)
    assert inputs.load_stream(path,cycles=100) == document
    for mutation in ('shape','source','identity'):
        bad = deepcopy(document)
        if mutation == 'shape': bad['cohorts'][64]['task']['request']['video_frames'] = 17
        elif mutation == 'source': bad['identity']['inventory'][64][0]['sha256'] = inventories[63][0]['sha256']
        else: bad['cohorts'][64]['index'] = 63
        with pytest.raises(ValueError): inputs.load_stream(publish(bad),cycles=100)


@pytest.mark.parametrize('optimized',[False,True])
def test_public_wan_100_lowering_reaches_actual_worker_horizon(tmp_path, monkeypatch, optimized):
    from test_training_native import wan_reference_fixture, write
    from distillschedule.strategies import wan_planning, wan_stream, wan_worker
    config = wan_reference_fixture(tmp_path)
    task = OriginalWanTask.from_dict(json.loads((tmp_path/'task.json').read_text()))
    sources = WanCohortSources(tuple(WanInputSources(tuple(str(tmp_path/f'c{i}-r{r}.pt') for r in range(16)),
        cohort=i) for i in range(100)))
    document = dict(schema_version=wan_stream.SCHEMA,status='READY_FRESH_COHORT_STREAM',task=task.to_dict(),
        identity=dict(task=task.to_dict(),sources=sources.to_dict(),source_fingerprint=wan_planning.source_fingerprint(),
            inventory=[{}]*100),cohorts=[dict(index=i,task=task_at(task,i).to_dict()) for i in range(100)])
    document['stream_sha256'] = fingerprint(document); write(tmp_path/'stream.json',document)
    raw = config.to_dict(); raw['data'] = dict(input_kind='original_wan_cohort_stream.v1',prepared='stream.json')
    raw['trainer'].update(cycles=100,checkpoint_every=50)
    if optimized:
        raw['execution'].pop('implementation'); raw['execution'].update(mode='manual',preset='wan_ar.incumbent21a_stream.v2')
    config = TrainingConfig.from_dict(raw,source=config.source)
    report = native.validate_native(config)
    seen = []
    monkeypatch.setattr(wan_stream,'load_stream',lambda *a:document)
    monkeypatch.setattr(wan_worker,'run_worker',lambda *a,**kw:seen.append(kw) or {'evidence':'SYNTHETIC_TEST_ONLY'})
    observed = native._source(config); observed['identity']['dirty'] = False
    monkeypatch.setattr(native,'_source',lambda _:observed)
    assert native.run_native(config) in ({'evidence':'SYNTHETIC_TEST_ONLY'},0)
    assert seen[0]['cycles'] == 100 and seen[0]['mode'] == 'train' and seen[0]['checkpoint_every'] == 50
    assert report['capabilities']['cycles']['max'] == 100


def test_actual_incumbent_dispatch_100_then_restore50_to100_with_bounded_caches(tmp_path,monkeypatch):
    import torch
    from test_strategy_wan_bundle_stream import native_cpu_bundle
    from distillschedule.strategies.wan_strategy import strategy_environment
    from distillschedule.strategies.wan_stream import bind_cohort, restore_stream_cursor
    from distillschedule.strategies.wan_worker import execute_cycles
    from distillschedule.strategies.wan_evidence import validate_cycle
    from distillschedule.runtime.wan_bundle_lifecycle import validate_stream_receipts
    from distillschedule.training.native_state import engine_state
    from distillschedule.training.checkpoint import save_checkpoint,restore_checkpoint,pack_state,rng_state
    from test_training_end_to_end import equal
    f = native_cpu_bundle.__wrapped__(monkeypatch)
    f.stream['cohorts'] = [dict(task=task_at(f.task,i).to_dict(),trace_bundle=str(i),trace_hash=f'{i+1:064x}') for i in range(100)]
    order = ('g',*(f'r{i}' for i in range(5)),*(k for i in range(5) for k in (f'c{i}',f'e{i}')))
    identity = dict(scope='SYNTHETIC_TEST_ONLY',stream=f.stream)
    def state(s): return engine_state({'generator':s.g,'critic':s.c},ema=s.ema)
    def run(s,start=0,after=None):
        return execute_cycles(s,cycles=100,start_cycle=start,mode='run',
            torch=NS(cuda=NS(synchronize=lambda _:None)),distributed=NS(barrier=lambda:None),
            before_cycle=lambda cycle,current:bind_cohort(current,f.task,f.stream,cycle-1,order),after_cycle=after,
            validate_result=lambda result:validate_cycle(s.task,s.trace,result))
    with strategy_environment(f.strategy):
        session,seen = f.build()
        owners = [(id(e),id(e.optimizer),id(e.lr_scheduler)) for e in (session.g,session.c)]
        def save(cycle,current,record):
            if cycle == 50: save_checkpoint(tmp_path/'cycle50',state(current),identity=identity,
                cycle=50,cursor=50,versions=current.versions())
        result = run(session,after=save)
        expected,rng = pack_state(state(session)),rng_state()
        assert owners == [(id(e),id(e.optimizer),id(e.lr_scheduler)) for e in (session.g,session.c)]
        assert result['final_versions'] == dict(generator=100,critic=500,ema=500)
        assert [step for role,step in seen if role=='c'] == list(range(500))
        assert [row['cohort']['cohort_index'] for row in result['cycles']] == list(range(100))
        validate_stream_receipts(session.build,cycles=100)
        restored,resumed_seen = f.build()
        progress = restore_checkpoint(tmp_path/'cycle50',state(restored),identity=identity)
        assert progress['cycle'] == progress['cursor'] == 50
        restore_stream_cursor(restored,f.task,f.stream,completed_cycles=50)
        continued = run(restored,start=50)
        assert [row['cycle'] for row in continued['cycles']] == list(range(51,101))
        assert [step for role,step in resumed_seen if role=='c'] == list(range(250,500))
        assert equal(expected,pack_state(state(restored))) and equal(rng,rng_state())
        validate_stream_receipts(restored.build,cycles=50)
        assert restored.bundle_lifecycle.rollout.receipt['cache_entry_count'] == 3
        assert restored.bundle_lifecycle.critic.receipt['capture_entries'] == 3
        assert session.g.lr_scheduler.last_epoch == 100 and session.c.lr_scheduler.last_epoch == 500
        controller = restored.bundle_lifecycle.rollout
        with pytest.raises(ValueError,match='version/sample'):
            controller.begin_rollout(version=99,logical_steps=(495,))
        with torch.no_grad():
            controller.begin_rollout(version=100,logical_steps=(495,))
            try:
                with pytest.raises(ValueError,match='signature after'):
                    controller(hidden_states=torch.ones(1,1,13,20))
            finally:
                controller.end_rollout()
        assert controller.receipt['cache_entry_count'] == 3
        with pytest.raises(ValueError,match='source'):
            controller.bind_stream_cycle(version=101,logical_steps=tuple(range(500,505)),source='b'*64)
        with pytest.raises(ValueError):
            f.strategy.validate_for(replace(task_at(f.task,99),video_height=512))


@pytest.mark.parametrize('start',[0,50])
def test_actual_helios_loop_horizon_and_restored_cursor_are_not_reset(tmp_path,monkeypatch,start):
    """Real W2 loop/control, SYNTHETIC_TEST_ONLY model, checkpoint and collectives."""
    from distillschedule.training import helios_lifecycle as lifecycle, native_state, checkpoint
    from distillschedule.runtime import helios_planner_cycle as cycle
    versions = lifecycle.expected_versions(0)
    generator,critic,ema = NS(global_steps=2),NS(global_steps=10),NS(optimization_step=10)
    binding = dict(config=NS(trainer=dict(cycles=100,output_dir=str(tmp_path),**(dict(resume_from='SYNTHETIC_TEST_ONLY') if start else {}))),
        identity={},checkpoint_policy={},stream=dict(stream_sha256='SYNTHETIC_TEST_ONLY',cohorts=[
            dict(trace_bundle=str(i),trace_hash=fingerprint(i)) for i in range(100)]))
    monkeypatch.setattr(native_state,'engine_state',lambda *a,**kw:{})
    monkeypatch.setattr(lifecycle,'load_cohort',lambda task,*a,**kw:task)
    header = dict(cycle=50,cursor=50,policy={},versions=lifecycle.expected_versions(50))
    monkeypatch.setattr(checkpoint,'read_manifest',lambda *a,**kw:header)
    def restore(*a,**kw):
        generator.global_steps,critic.global_steps,ema.optimization_step = header['versions'].values()
        return header
    monkeypatch.setattr(checkpoint,'restore_checkpoint',restore)
    from contextlib import nullcontext
    class Observer:
        def __init__(self,**kw): self.actions=[]
        def installed(self): return nullcontext()
        def finish(self): return dict(evidence='SYNTHETIC_TEST_ONLY')
    monkeypatch.setattr(cycle,'_Observer',Observer)
    seen = []
    def prepare(**kw):
        seen.append(kw['steps']); index = (kw['steps'][0]-10)//5
        def run():
            generator.global_steps += 1;critic.global_steps += 5;ema.optimization_step += 5
        return NS(run=run,receipt=lambda:{'evidence':'SYNTHETIC_TEST_ONLY'}),NS(metadata=dict(
            final_versions=lifecycle.expected_versions(index+1))),('g','r0','c0')
    monkeypatch.setattr(cycle,'_prepare_cycle',prepare)
    result = lifecycle.execute_cycles(p0=NS(torch=None),args=None,policy=None,generator=generator,critic=critic,
        ema=ema,diagnostics=None,negative=None,rank=0,device='cpu',binding=binding,result={})
    assert versions == lifecycle.expected_versions(0)
    assert result['resume_cycle'] == start and result['completed_cycles'] == 100
    assert result['final_versions'] == dict(generator=102,critic=510,ema=510)
    assert seen == [tuple(range(10+5*i,15+5*i)) for i in range(start,100)]
    assert [row['cycle'] for row in result['cycles']] == list(range(start+1,101))


@pytest.mark.parametrize('mode,observer',[('run',None),('calibrate',object())])
def test_shared_wan_loop_never_expands_legacy_fixed_trace_limit(mode,observer):
    from distillschedule.strategies.wan_worker import execute_cycles
    for count in (65,100,101,True,100.0):
        with pytest.raises(ValueError):
            execute_cycles(None,cycles=count,mode=mode,observer=observer,torch=None,distributed=None)


def test_legacy_three_cycle_diagnostic_still_reaches_prepare_without_running_math():
    from distillschedule.strategies.wan_worker import execute_cycles
    class PreparedBoundaryReached(Exception): pass
    def prepare(**kw): raise PreparedBoundaryReached
    session = NS(versions=lambda:dict(generator=0,critic=0,ema=0),prepare=prepare)
    with pytest.raises(PreparedBoundaryReached):
        execute_cycles(session,cycles=3,mode='run',torch=None,distributed=None)


@pytest.mark.parametrize('mode',['manual','calibrate'])
def test_public_worker_fixed_trace_modes_still_require_two_before_construction(tmp_path,monkeypatch,mode):
    from distillschedule.strategies.wan_worker import run_worker, OriginalWanAssets
    from distillschedule.strategies.wan_strategy import WanStrategy
    for key,value in dict(WORLD_SIZE=16,LOCAL_WORLD_SIZE=8,RANK=0,LOCAL_RANK=0).items():
        monkeypatch.setenv(key,str(value))
    assets = OriginalWanAssets('unused','unused','a'*64,'unused','b'*64,'unused')
    for count in (3,64,65,100):
        with pytest.raises(ValueError):
            run_worker(OriginalWanTask(),assets,output=tmp_path/'UNREACHED',mode=mode,
                strategy=WanStrategy('fsdp',512,False),cycles=count)
    assert not (tmp_path/'UNREACHED').exists()


def test_helios_preparation100_reaches_existing_capture_admission_but101_does_not(tmp_path,monkeypatch):
    from distillschedule.training import helios_inputs as inputs
    from distillschedule.strategies import wan_capture
    sources = [HeliosCohortSources(42,tuple(tuple(str(tmp_path/f'{i}-{s}-{r}.pt') for r in range(16))
        for s in range(5))) for i in range(101)]
    class InputConfigBoundaryReached(Exception): pass
    def read(path): raise InputConfigBoundaryReached
    monkeypatch.setattr(wan_capture,'read_training_config',read)
    with pytest.raises(InputConfigBoundaryReached):
        inputs.prepare_stream(HeliosCohortTask(0).request,sources[:100],'SYNTHETIC_TEST_ONLY',tmp_path/'UNREACHED')
    with pytest.raises(ValueError,match='typed cohorts'):
        inputs.prepare_stream(HeliosCohortTask(0).request,sources,'SYNTHETIC_TEST_ONLY',tmp_path/'UNREACHED')
    assert not (tmp_path/'UNREACHED').exists()


@pytest.mark.parametrize('optimized',[False,True])
def test_w2_public_100_metadata_lowering_keeps_reference_or_safe_policy(tmp_path,monkeypatch,optimized):
    from distillschedule.training import helios_worker as worker
    config = configuration(tmp_path,'W2',optimized)
    assets = dict(schema_version=worker.REFERENCE_ASSETS,assets={name:dict(path=str(tmp_path/name)) for name in (
        'generator_model','generator_precast','score_model','score_precast','logical_checkpoint','trace','negative_prompt')})
    (tmp_path/'assets.json').write_text(json.dumps(assets))
    (tmp_path/'task.json').write_text(json.dumps(HeliosCohortTask(0).request.to_dict()))
    monkeypatch.setattr(worker,'reference_assets',lambda value:value)
    monkeypatch.setattr(worker,'verify_reference_pins',lambda value:{'evidence':'SYNTHETIC_TEST_ONLY'})
    monkeypatch.setattr(worker,'load_reference_bootstrap',lambda *a,**kw:(NS(manifest_hash=fingerprint('logical')),None))
    stream = dict(identity=dict(request=HeliosCohortTask(0).request.to_dict()),cohorts=[
        dict(trace_bundle=str(tmp_path/f'c{i}'),trace_hash=fingerprint(i)) for i in range(100)])
    counts = []
    def load(path,*,cycles): counts.append(cycles); return stream
    monkeypatch.setattr(worker,'load_stream',load)  # SYNTHETIC_TEST_ONLY physical assets/trace IO, real lowerer.
    report = native.validate_native(config)
    assert report['capabilities']['cycles']['max'] == 100 and counts == [100]
    policy = 'node-hsdp' if optimized else 'world16-fsdp'
    argv = report['native_argv']
    assert argv[argv.index('--mesh-policy')+1] == policy
    def boundary(config,*,admit):
        assert len(admit()[3]['stream']['cohorts']) == 100
        return 'SYNTHETIC_TEST_ONLY_BEFORE_MODEL_CONSTRUCTION'
    monkeypatch.setattr(worker,'run_safe_stream' if optimized else 'run_reference',boundary)
    assert native.run_native(config) == 'SYNTHETIC_TEST_ONLY_BEFORE_MODEL_CONSTRUCTION'
    assert counts == [100,100]


def test_w2_current_g_export_boundary_accepts100_without_widening_scope(monkeypatch):
    from distillschedule.training import helios_nogan_inference as inference
    from distillschedule.adapters.helios import reference_model
    task = HeliosCohortTask(0)
    p = dict(schema_version=inference.SCHEMA,scope=inference.SCOPE,precision=inference.PRECISION,
        source='SYNTHETIC_TEST_ONLY',runtime='SYNTHETIC_TEST_ONLY',task=task.to_dict(),
        boundary=dict(cycle=100,versions=dict(generator=102,critic=510,ema=510),
            warm_kind='ACTUAL_ORIGINAL_W2_CPU_TEST_WARM_NOT_HISTORICAL_STEP10'),architecture={},peft={},native={},
        geometry=dict(noise_shape=list(task.latent_shape),prompt_shape=[1,512,4096]),
        budget=dict(max_parameters=1000000,max_loaded_bytes=400000000),
        evidence_kind='FULL_ORIGINAL_TINY_CPU_FIXTURE_NOT_PRETRAINED_OR_QUALITY',seed=42)
    monkeypatch.setattr(inference,'_cpu1',lambda:None)
    monkeypatch.setattr(inference,'inference_source',lambda:p['source'])
    monkeypatch.setattr(inference,'content_runtime',lambda:p['runtime'])
    class ConstructionBoundaryReached(Exception): pass
    def construct(**kw): raise ConstructionBoundaryReached
    monkeypatch.setattr(reference_model,'load_reference_model_class',construct)
    with pytest.raises(ConstructionBoundaryReached): inference._admit(p)
    for bad in (101,True,100.0):
        value = deepcopy(p); value['boundary']['cycle'] = bad
        with pytest.raises(ValueError,match='completed current warm-owner boundary'): inference._admit(value)
    value = deepcopy(p); value['scope'] = 'CUDA_OR_DECODED_VIDEO'
    with pytest.raises(ValueError,match='inference ABI'): inference._admit(value)


def test_w2_cpu_reference100_still_obeys_original_capacity_and_algorithm_cursor(monkeypatch):
    """Guard-only SYNTHETIC_TEST_ONLY owner; no full model or old warm proof rerun."""
    from distillschedule.training import helios_nogan_public as public
    owner = public.HeliosNoGANCycleOwner.__new__(public.HeliosNoGANCycleOwner)
    owner._validate_live = lambda:None
    owner.args = NS(training_config=NS(max_train_steps=510))
    owner.warm = NS(identity=dict(evidence_kind=public.bootstrap.ORIGINAL,
        assets=dict(base_evidence_kind='SYNTHETIC_TEST_ONLY')))
    owner.identity = {}; owner._stream = None
    monkeypatch.setattr(public,'admit_assets',lambda *a:(None,{'input_config':'SYNTHETIC_TEST_ONLY'},None))
    monkeypatch.setattr(public,'source_fingerprint',lambda:'source')
    monkeypatch.setattr(public,'layout_identity',lambda:'layout')
    monkeypatch.setattr(public,'conditioning_fingerprint',fingerprint)
    request = NS(execution=dict(adapter='program',mode='reference'),trainer=dict(cycles=100),data={},
        algorithm=dict(factory='distillschedule.training.helios_nogan_public:build_algorithm'),model={})
    records = [dict(metadata=dict(cycle=i,trace=dict(control_hash=fingerprint(i),content=dict(cohort=i),
        sources=[fingerprint([i,j]) for j in range(80)]))) for i in range(100)]
    manifest = dict(metadata=dict(schema_version=public.INPUTS,scope=public.SCOPE,precision=public.PRECISION,
        task=HeliosCohortTask(0).to_dict(),source='source',stream='SYNTHETIC_TEST_ONLY',
        input_config='SYNTHETIC_TEST_ONLY',evidence_kind='SYNTHETIC_TEST_ONLY',layout='layout'),
        records=records,provenance=dict(documents=[]))
    owner.args.training_config.max_train_steps = 509
    with pytest.raises(ValueError,match='inventory exhausted'):
        public.admit_training(state={'native_cycle':owner},request=request,prepared_identity='a'*64,prepared_manifest=manifest)
    assert owner._stream is None
    owner.args.training_config.max_train_steps = 510
    public.admit_training(state={'native_cycle':owner},request=request,prepared_identity='a'*64,prepared_manifest=manifest)
    assert owner.capacity == 100 and len(owner._stream) == 100
    monkeypatch.setattr(public,'checked_pin',lambda value:None)
    class DecodeBoundaryReached(Exception): pass
    def decode(feeds,task): raise DecodeBoundaryReached(task.index)
    monkeypatch.setattr(public,'decode',decode)
    components = NS(state={'native_cycle':owner},metadata=dict(asset_manifest=None,
        assets=dict(bootstrap=None,args=None,input_config=None)))
    for i in (63,64,99):
        owner.cycle = i
        with pytest.raises(DecodeBoundaryReached) as hit: public.build_algorithm(components,{},None,i,42)
        assert hit.value.args == (i,)
    for i in (100,101,True,99.0):
        with pytest.raises(ValueError): public.build_algorithm(components,{},None,i,42)


def _composed_stream_receipts(cycles):
    """SYNTHETIC_TEST_ONLY receipt data, not new model/capture execution.

    Reuse the existing coupled kernel receipt seam while exercising the real
    outer worker validator AND its inner lifecycle validator, without replaying
    the already verified scalar100/model/restore loops.
    """
    from test_refactor_wan_incumbent_bundle import _executed_receipts
    from test_strategy_wan_bundle_stream import stream_bundle
    from distillschedule.runtime.wan_bundle_lifecycle import SCHEMA
    from distillschedule.strategies.wan_strategy import WanStrategy
    bundle = stream_bundle()
    strategy = WanStrategy('resident',2048,True,5,bundle)
    build = _executed_receipts()
    rows, entries = [],[]
    for i in range(cycles):
        steps = list(range(5*i,5*i+5))
        current_entries = [dict(version=i+1,logical_steps=[s]) for s in steps]
        rows.append(dict(cohort_index=i,stream_sha256=fingerprint('SYNTHETIC_TEST_ONLY'),trace_hash=fingerprint(i),
            before=dict(generator=i,critic=5*i,ema=5*i),after=dict(generator=i+1,critic=5*(i+1),ema=5*(i+1)),
            source_steps=steps,inventories=[dict(call_count=3) for _ in range(5)],rollout_calls=15,
            rollout_entries=current_entries,critic_versions=steps))
        entries.extend(current_entries)
    lifecycle = dict(schema_version=SCHEMA,active=False,source=fingerprint('SYNTHETIC_SOURCE'),cycles=rows)
    for role in ('generator','critic'):
        m = build[role]['metadata']
        m.update(candidate_bundle=bundle.to_dict(),candidate_bundle_sources=bundle.source_identity(),bundle_lifecycle=lifecycle,
            modulation_chunk_runtime=dict(effective_tokens=[2048]),generator_nograd_lora_fused_epilogue=dict(
                requested=True,enabled=role=='generator',fused_invocation_count=1,fused_element_count=1))
        build[role].update(engine=dict(gradient_sync='WORLD16_GRAD_READY_ASYNC_SUM_THEN_DIVIDE_ALL_REDUCE'),
            placement=dict(param_compute_dtype_policy='PRESERVE_ORIGINAL_PER_PARAMETER',
                trainable_parameter_dtypes=['torch.float32'],frozen_parameter_dtypes=['torch.bfloat16']))
    g = build['generator']['metadata']['generator_rollout_cuda_graph']
    g.update(active=False,model_call_count=15*cycles,graph_replay_model_call_count=15*cycles,
        rollout_entry_count=5*cycles,entries=entries,versions=list(range(1,cycles+1)),
        diagnostic_reference_model_call_count=0,diagnostic_reference_entry_count=0,
        weights_merged=False,optimizer_reducer_ema_unchanged=True,cache_entry_count=3,
        capture_model_call_count=3,warmup_model_call_count=9,
        cache_entries=[dict(latent_spatial_shape=list(shape)) for shape in ((12,20),(24,40),(48,80))],
        stream_lifecycle=dict(source=lifecycle['source'],first_version=1,rng_checked_captures=3))
    c = build['critic']['metadata']['critic_grad_cuda_graph']
    c.update(replay_calls=5*cycles,backward_replay_calls=5*cycles,capture_entries=3,
        cold_original_forward_calls=12,cold_original_backward_calls=12,
        stream_lifecycle=dict(source=lifecycle['source'],first_version=0,rng_checked_captures=3,
            versions=list(range(5*cycles))))
    return strategy,build


@pytest.mark.parametrize('cycles',[65,100])
def test_review_real_composed_worker_receipt_gate_accepts_65_and100(cycles):
    from distillschedule.runtime.wan_incumbent_evidence import validate_bundle_build
    from distillschedule.strategies.wan_evidence import validate_build
    strategy,build = _composed_stream_receipts(cycles)
    outer = validate_bundle_build(strategy.candidate_bundle,build,executed=True,cycles=cycles)
    assert outer['actual_compute_observed'] and len(outer['lifecycle']['cycles']) == cycles
    # This is the same composed public helper run_worker invokes before final
    # state publication, not merely validate_stream_receipts.
    result = validate_build(strategy,build,executed=True,cycles=cycles)
    assert result['status'] == 'PASS_BACKEND_EFFECTS'
    assert len(result['candidate_bundle_effects']['lifecycle']['cycles']) == cycles


def test_review_composed_receipt_gate_rejects101_and_literal_stays_exactly2():
    from distillschedule.runtime.wan_incumbent_evidence import validate_bundle_build
    from distillschedule.strategies.wan_evidence import validate_build
    from distillschedule.search.wan_candidate_bundle import INCUMBENT
    from test_refactor_wan_incumbent_bundle import _executed_receipts
    strategy,build = _composed_stream_receipts(101)
    for validate,args in ((validate_bundle_build,(strategy.candidate_bundle,build)),(validate_build,(strategy,build))):
        with pytest.raises(ValueError): validate(*args,executed=True,cycles=101)
    literal = _executed_receipts()
    assert validate_bundle_build(INCUMBENT,literal,executed=True,cycles=2)['actual_compute_observed']
    for cycles in (65,100,101):
        with pytest.raises(ValueError): validate_bundle_build(INCUMBENT,literal,executed=True,cycles=cycles)


@pytest.fixture
def review_w2_preconstruction_assets(tmp_path):
    """Real Args/parser/pins/input/layout admission; NO model/warm-cycle rerun.

    Minimal SYNTHETIC_TEST_ONLY provenance documents reach this preconstruction
    guard; they are not a valid materialized bootstrap or executed warm proof.
    No admit_assets, optimizer/algorithm or topology function is patched.
    """
    import yaml
    from omegaconf import OmegaConf
    from distillschedule.strategies.wan_capture import native_primitives
    from distillschedule.training import helios_nogan_public as public
    from distillschedule.workloads.best_known.registry import repository_root
    native_primitives(source_only=True)
    from helios.utils.train_config import Args
    template = yaml.safe_load((repository_root()/'configs/distillschedule/templates/helios_stage3_384x640.yaml').read_text())
    args = OmegaConf.merge(OmegaConf.structured(Args),template)
    args.training_config.max_grad_norm = args.training_config.max_grad_norm_critic = 1.
    args.validation_config.use_kv_cache = False
    config = tmp_path/'input-config.yaml'; config.write_text(yaml.safe_dump(template))
    def pin(path):
        data = path.read_bytes()
        import hashlib
        return dict(path=str(path),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    def write(name,value):
        path = tmp_path/name; path.write_text(json.dumps(value)); return pin(path)
    proof = write('SYNTHETIC_ADMISSION_ONLY_proof.json',dict(evidence_kind=public.bootstrap.CPU_WARM,
        update_source='ORIGINAL_W2_HELPERS',evidence_scope='SYNTHETIC_TEST_ONLY_PRECONSTRUCTION_DOCUMENTS'))
    warm = write('SYNTHETIC_ADMISSION_ONLY_warm.json',dict(provenance=proof))
    envelope = dict(schema_version=public.SCHEMA,scope=public.SCOPE,precision=public.PRECISION,
        bootstrap=warm,input_config=pin(config))
    def asset_manifest(horizon,mutation=None):
        values = OmegaConf.to_container(args,resolve=True)
        values['training_config']['max_train_steps'] = horizon
        if mutation:
            key,value = mutation; values['training_config'][key] = value
        current = dict(envelope,args=write('args.json',values))
        path = tmp_path/'assets.json'; path.write_text(json.dumps(current))
        return dict(asset_manifest=str(path),device='cpu'),current
    return asset_manifest


@pytest.mark.parametrize('horizon,accepted',[(330,True),(335,True),(510,True),(515,False),(True,False),(330.5,False)])
def test_review_real_w2_admit_assets_source_clock_ceiling(review_w2_preconstruction_assets,horizon,accepted):
    from distillschedule.training.helios_nogan_public import admit_assets
    parameters,document = review_w2_preconstruction_assets(horizon)
    if accepted:
        path,actual,args = admit_assets(parameters)
        assert path == Path(parameters['asset_manifest']) and actual == document
        assert type(args.training_config.max_train_steps) is int and args.training_config.max_train_steps == horizon
    else:
        with pytest.raises(ValueError): admit_assets(parameters)


@pytest.mark.parametrize('mutation',[('mixed_precision','no'),('learning_rate',1e-5),('use_ema',False)])
def test_review_w2_real_asset_horizon_does_not_relax_other_guards(review_w2_preconstruction_assets,mutation):
    from distillschedule.training.helios_nogan_public import admit_assets
    parameters,_ = review_w2_preconstruction_assets(330,mutation)
    with pytest.raises(ValueError,match='optimizer/precision/batch/EMA/horizon'): admit_assets(parameters)
