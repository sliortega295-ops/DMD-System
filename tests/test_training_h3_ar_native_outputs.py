"""Public native H3 output binding; CPU doubles are never CUDA qualification."""
from copy import deepcopy
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest


def test_native_h3_export_discovery_is_model_free():
    code = "from distillschedule.training.h3_ar_export import export_native;import sys;assert callable(export_native);assert 'torch' not in sys.modules;assert 'deepspeed' not in sys.modules"
    result = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_output_terminal_boundary_does_not_weaken_training_restart():
    from distillschedule.training.h3_ar_training import checkpoint_policy, validate_restart
    identity = dict(horizon=100, source='SYNTHETIC_TEST_ONLY')
    header = dict(cycle=100, cursor=100, versions=dict(student=100, fake=100, ema=0),
                  policy=checkpoint_policy(identity), native_checkpoint={})
    with pytest.raises(ValueError, match='nonterminal'):
        validate_restart(header, identity)
    validate_restart(header, identity, output_only=True)
    for change in (dict(cycle=101, cursor=101), dict(cycle=0, cursor=0), dict(cursor=99),
                   dict(versions=dict(student=100, fake=99, ema=0)), dict(policy={})):
        with pytest.raises(ValueError):
            validate_restart({**header, **change}, identity, output_only=True)
    for invalid in (1, 'True', None):
        with pytest.raises(ValueError):
            validate_restart(header, identity, output_only=invalid)


@pytest.mark.parametrize('completed', [1, 3])
def test_output_cursor_parses_actual_saved_record_without_consuming_next(tmp_path, completed):
    from distillschedule.training.h3_ar_export import OutputInputStream
    from distillschedule.training.h3_ar_inputs import H3ARInputStream
    from test_training_h3_ar_continuation import stream
    path, _ = stream(tmp_path, cycles=3)
    cursor = OutputInputStream(path, rank=6, cycles=3, seed=17, completed=completed)
    assert isinstance(cursor, H3ARInputStream)
    assert cursor.completed == completed and cursor.pending is None
    assert cursor.construction.completed == completed-1
    assert cursor.construction.record(completed)['record_id'] == f'record-{(completed-1)*16+6}'
    for method in (lambda: cursor.commit(completed+1), lambda: cursor.graph(None, completed+1)):
        with pytest.raises(ValueError, match='read-only'):
            method()


def test_read_only_cursor_real_delegation_is_once_and_keeps_true_progress(tmp_path, monkeypatch):
    from distillschedule.training.h3_ar_export import OutputInputStream
    from distillschedule.training.h3_ar_inputs import H3ARInputStream
    from test_training_h3_ar_continuation import stream
    path, _ = stream(tmp_path, cycles=3)
    seen = []
    def parse(self, *args, **kwargs):
        seen.append(self.record(self.completed+1)['cycle'])
        self.pending = self.completed+1
        return 'SYNTHETIC_TEST_ONLY_POSITIVE', {}, {'cycle':self.pending}, {'cycle':self.pending}
    monkeypatch.setattr(H3ARInputStream,'prepare_rank_input',parse)
    cursor = OutputInputStream(path,rank=0,cycles=3,seed=17,completed=3)
    cursor.stage_initial(None,None,None,0,None)
    cursor.prepare_rank_input(None,None,None,0,None,teacher_cfg_scale=1.0)
    assert seen==[3,3] and cursor.completed==3 and cursor.pending is None
    with pytest.raises(ValueError,match='already served'):
        cursor.prepare_rank_input(None,None,None,0,None)


@pytest.mark.parametrize('reference',[True,False])
def test_public_export_preserves_training_identity_and_routes_to_real_owner(tmp_path,monkeypatch,reference):
    from distillschedule.training import native,h3_ar_training as worker
    from distillschedule.training.config import TrainingConfig
    from test_training_h3_ar_continuation import public_config
    conf = public_config(tmp_path,monkeypatch,reference=reference)
    raw = conf.to_dict();raw['trainer']['export']={'schema_version':'test-only-request'}
    raw['trainer']['resume_from']=str(tmp_path/'stale')
    conf = TrainingConfig.from_dict(raw,source=conf.source)
    def run(clean,*,admit,native_output):
        assert clean.trainer['checkpoint_every']==50 and clean.trainer['cycles']==100
        assert clean.trainer['resume_from']==str(tmp_path/'actual') and 'stop_after_cycle' not in clean.trainer
        assert clean.execution==conf.execution and clean.data==conf.data
        assert 'export' not in clean.trainer and clean.trainer['seed']==17
        assert admit()[0]=='W6'
        native_output.result={'status':'EXPORTED','scope':'SYNTHETIC_TEST_ONLY_DISPATCH'}
        return 0
    monkeypatch.setattr(worker,'run_training',run)
    monkeypatch.setattr(native,'_admit',lambda clean:('W6',{'identity':{'dirty':False}}, {}, {}))
    assert native.describe_native(conf)['capabilities']['export']['supported'] is True
    assert native.export_native(conf,tmp_path/'actual',tmp_path/'out')['status']=='EXPORTED'
    raw['data']={};raw['execution'].pop('preset',None)
    with pytest.raises(ValueError,match='fresh W6'):
        native.export_native(TrainingConfig.from_dict(raw,source=conf.source),tmp_path/'actual',tmp_path/'bad')


def output_policy():
    from distillschedule.training.h3_ar_export import SCHEMA
    return dict(schema_version=SCHEMA,seed=93,
        geometry=dict(video_shape=[1,24,2,4,4],audio_shape=[2,32,3],prompt_shape=[3,5120],
            reference_video_shape=[1,24,3,4,4],reference_audio_shape=[2,32,5]),
        consolidation_budget=dict(max_parameters=40_000_000_000,max_consolidated_bytes=256*2**30,
            host_reserve_bytes=2**20,device_reserve_bytes=2**20),
        inference_budget=dict(max_parameters=40_000_000_000,max_loaded_bytes=512*2**30,
            max_input_elements=64_000_000,max_attention_elements=100_000_000_000,host_reserve_bytes=2**20))


@pytest.mark.parametrize('fault',['none','reserve','budget','shape','rank_disagreement','existing'])
def test_output_policy_preserves_original_capacity_and_allrank_binding(tmp_path,monkeypatch,fault):
    from distillschedule.training.h3_ar_export import NativeOutputRequest,distributed
    value=output_policy()
    policy=dict(gpu_limit_bytes=100_000_000,gpu_reserve_bytes=2**20,host_limit_bytes=200_000_000,host_reserve_bytes=2**20)
    if fault=='reserve':value['consolidation_budget']['device_reserve_bytes']=1
    elif fault=='budget':value['inference_budget']['max_loaded_bytes']=0
    elif fault=='shape':value['geometry']['audio_shape']=[1,32,3]
    elif fault=='existing':(tmp_path/'out').mkdir()
    elif fault=='rank_disagreement':monkeypatch.setattr(distributed,'gather',lambda row:[row,{}])
    callback=NativeOutputRequest(value,tmp_path/'out')
    if fault=='none':callback.admit(policy)
    else:
        with pytest.raises(ValueError):callback.admit(policy)
    assert not (tmp_path/'out/_SUCCESS').exists()


@pytest.fixture
def restored_native_output_case(tmp_path,monkeypatch,request):
    """Actual checkpoint/Progress/consumer; constructor-free CPU engine only.

    No old training cycles are rerun. Clock and Adam payloads are explicitly
    synthetic serialization markers, not evidence of training or CUDA ZeRO3.
    """
    import inspect,json,torch
    from test_training_h3_ar_continuation import session,stream
    from test_h3_cached_original_cpu import original,tiny_architecture
    from distillschedule.training import h3_ar_training as worker,h3_ar_portable as portable
    from distillschedule.training.h3_ar_state import NativeCheckpoint,Progress
    from distillschedule.training.h3_ar_inputs import H3ARInputStream
    from distillschedule.training.h3_ar_export import NativeOutputRequest,OutputInputStream
    from distillschedule.training import h3_ar_inference as snapshot,checkpoint as cp
    torch.set_num_threads(1)
    api=original()
    arch=json.loads(json.dumps({**{k:v.default for k,v in inspect.signature(api.model.MiniMaxH3DiT.__init__).parameters.items()
        if k not in ('self','kwargs')},**tiny_architecture()}))
    path,_=stream(tmp_path/'inputs',cycles=3)
    def owner(current):
        s=session(seed=19)
        s.adapter=api.dmd2.H3Adapter(api.model.MiniMaxH3DiT(**arch),portable.original_config())
        s.discriminator=torch.nn.Linear(2,1).requires_grad_(False)
        if getattr(request,'param','FP32')=='BF16':s.adapter.bfloat16()
        s.adapter.select('student',trainable=True)
        # Optimizer.__init__ instruments class.step at runtime. Use the R32
        # exact-original AST classes, not the independently source-locked CPU
        # inference provider's classes; its source identity must stay intact.
        s.engine.module=s.engine.native_api.DMD2Module(s.adapter,s.discriminator)
        s.optimizer=s.engine.native_api.AlternatingAdamW([
            dict(params=s.adapter.role_parameters('student'),role='student',lr=.01),
            dict(params=s.adapter.role_parameters('fake_score'),role='critic',lr=.01),
            dict(params=list(s.discriminator.parameters()),role='critic',head=True,lr=.01)])
        s.optimizer.active_role,s.optimizer.train_heads='student',False
        s.engine.optimizer=s.optimizer
        s.native=api.dmd2;s.dit=s.adapter.dit
        s.trainer=NS(generator_step=3 if current else 0,critic_step=3 if current else 0)
        s.engine.global_steps=6 if current else 0
        s.input_provider=(H3ARInputStream(path,rank=0,cycles=3,seed=17) if current else
            OutputInputStream(path,rank=0,cycles=3,seed=17,completed=3))
        if current:
            s.input_provider.completed=3
            with torch.no_grad():
                for p in s.adapter.role_parameters('student'):p.fill_(.0125)
                for p in s.adapter.role_parameters('fake_score'):p.fill_(-.0375)
            for group in s.optimizer.param_groups:
                if group.get('head'):continue
                for p in group['params']:
                    s.optimizer.state[p]=dict(step=torch.tensor(3.),exp_avg=torch.full_like(p,.01),exp_avg_sq=torch.full_like(p,.02))
        return s
    source=owner(True); native=NativeCheckpoint(source)
    state={'native_progress':Progress(source,source.input_provider,native)}
    identity=dict(checkpoint_every=1,scope='SYNTHETIC_TEST_ONLY_NO_TRAINING',horizon=3)
    checkpoint=cp.save_checkpoint(tmp_path/'checkpoint',state,identity=identity,cycle=3,cursor=3,
        versions=source.versions(),native_checkpoint=native)
    saved_rng=cp.rng_state(device='cpu')
    target=owner(False);target.resume_from=str(checkpoint)
    expected=deepcopy(source.engine.module.state_dict()),deepcopy(source.optimizer.state_dict()),saved_rng
    value=output_policy();callback=NativeOutputRequest(value,tmp_path/'portable')
    # Production requires full50 native architecture; only this CPU test injects
    # explicitly labeled original-class tiny metadata and a dense snapshot seam.
    def params(s):
        layout={k:dict(shape=list(t.shape),dtype=str(t.dtype)) for k,t in s.adapter.state_dict().items()
                if '.adapters.fake_score.' not in k}
        return portable.parameters(architecture=arch,geometry=value['geometry'],weight_layout=layout,seed=value['seed'],
            evidence_kind=portable.FIXTURE,budget=value['inference_budget'])
    monkeypatch.setattr(callback,'_parameters',params)
    def cpu_snapshot(s,native,*,budget):
        native.validate_checkpoint_boundary()
        assert s.trainer.generator_step==3 and s.engine.global_steps==6 and s.local_completed==0
        return {k:t.detach().clone() for k,t in s.adapter.state_dict().items() if '.adapters.fake_score.' not in k},dict(
            scope='SYNTHETIC_TEST_ONLY_DENSE_NOT_ZERO3_GATHER',cycle=3,host_capacity={})
    monkeypatch.setattr(snapshot,'consolidate_current_student',cpu_snapshot)
    return worker,target,callback,identity,expected,cp


def test_terminal_real_restore_before_common_publication_without_any_update(restored_native_output_case,tmp_path):
    import torch
    from distillschedule.training.inference import read_export,restore_modules
    from distillschedule.training import h3_ar_portable
    from test_training_end_to_end import equal
    worker,s,callback,identity,expected,cp=restored_native_output_case
    def no_update(*a,**kw):raise AssertionError('output-only must not execute training')
    s.prepare=s.finish=no_update
    rows=worker.execute_segment(s,stop=3,output=tmp_path/'owner',identity=identity,capacity_check=lambda:{},native_output=callback)
    assert rows==[] and s.input_provider.completed==3 and s.input_provider.pending is None
    assert equal(expected[:2],(s.engine.module.state_dict(),s.optimizer.state_dict()))
    assert equal(expected[2],cp.rng_state(device='cpu'))
    assert s.census==s.updates==s.adam=={} and s.local_completed==0 and s.versions()==dict(student=3,fake=3,ema=0)
    root,manifest=read_export(tmp_path/'portable')
    assert manifest['checkpoint_cycle']==3 and manifest['identity']['training_provenance']==identity
    assert callback.result['output_only']==dict(cycles=0,forward=0,backward=0,optimizer=0,ema=0,merged_lora=False)
    p=manifest['model']['parameters'];comp=h3_ar_portable.build_inference_model(p)
    restore_modules(comp,root,manifest)
    assert all(torch.equal(v,comp.state['student'].adapter.state_dict()[k])
        for k,v in s.adapter.state_dict().items() if '.adapters.fake_score.' not in k)
    assert not any('fake_score' in k for k in comp.state['student'].state_dict())
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('fault',['restore','snapshot','capacity','config'])
def test_output_failure_cannot_leave_success_marker(restored_native_output_case,tmp_path,monkeypatch,fault):
    worker,s,callback,identity,expected,cp=restored_native_output_case
    def fail(*a,**kw):raise RuntimeError('injected '+fault)
    capacity=lambda:{}
    if fault=='restore':monkeypatch.setattr(cp,'restore_checkpoint',fail)
    elif fault=='snapshot':
        from distillschedule.training import h3_ar_inference
        monkeypatch.setattr(h3_ar_inference,'consolidate_current_student',fail)
    elif fault=='capacity':capacity=fail
    else:
        import distillschedule.training.h3_ar_export as output
        monkeypatch.setattr(output.os,'link',fail)
    with pytest.raises(RuntimeError,match='injected'):
        worker.execute_segment(s,stop=3,output=tmp_path/'owner',identity=identity,capacity_check=capacity,native_output=callback)
    assert not (tmp_path/'portable/_SUCCESS').exists()
    assert s.census==s.updates==s.adam=={}


@pytest.mark.parametrize('fault',['tiny_config','constructed_architecture','config_hash'])
def test_original_checkpoint_architecture_and_hash_are_required_before_snapshot(tmp_path,fault):
    import inspect,json,torch
    from safetensors.torch import save_file
    from test_h3_cached_original_cpu import original,tiny_architecture
    from distillschedule.strategies.h3_ar_assets import admit_checkpoint_once
    from distillschedule.workloads.best_known.registry import file_sha256
    from distillschedule.training.h3_ar_export import NativeOutputRequest
    api=original()
    architecture=json.loads(json.dumps({k:v.default for k,v in inspect.signature(api.model.MiniMaxH3DiT.__init__).parameters.items()
                                       if k not in ('self','kwargs')}))
    if fault=='tiny_config':architecture.update(tiny_architecture())
    root=tmp_path/'base';root.mkdir()
    (root/'config.json').write_text(json.dumps(architecture))
    # This is only an asset-metadata/hash fixture, not a 33B checkpoint.
    save_file({'SYNTHETIC_TEST_ONLY':torch.ones(1)},str(root/'model.safetensors'))
    audit=tmp_path/'audit.json';files=admit_checkpoint_once(root,audit)
    assets=NS(checkpoint_root=str(root),checkpoint_files=files,
        checkpoint_audit=dict(path=str(audit),sha256=file_sha256(audit)),source_module=lambda name:api.weights)
    s=NS(assets=assets,dit=api.model.MiniMaxH3DiT.__new__(api.model.MiniMaxH3DiT),
         build={'architecture':deepcopy(architecture)})
    if fault=='constructed_architecture':s.build['architecture']['norm_eps']=1e-3
    elif fault=='config_hash':(root/'config.json').write_text(json.dumps({**architecture,'norm_eps':1e-3}))
    with pytest.raises(ValueError):NativeOutputRequest(output_policy(),tmp_path/'out')._parameters(s)
    assert not (tmp_path/'out/_SUCCESS').exists()


def test_post_gather_capacity_policy_failure_does_not_publish(restored_native_output_case,tmp_path):
    worker,s,callback,identity,expected,cp=restored_native_output_case
    calls=[]
    def capacity():
        calls.append(True)
        if len(calls)==2:raise RuntimeError('post-gather capacity')
        return {}
    with pytest.raises(RuntimeError,match='post-gather capacity'):
        worker.execute_segment(s,stop=3,output=tmp_path/'owner',identity=identity,capacity_check=capacity,native_output=callback)
    assert len(calls)==2 and not (tmp_path/'portable/_SUCCESS').exists()


@pytest.mark.parametrize('restored_native_output_case',['BF16'],indirect=True)
def test_actual_zero3_bf16_student_storage_is_exported_without_upcast(restored_native_output_case,tmp_path):
    import torch
    from distillschedule.training import h3_ar_portable as portable
    from distillschedule.training.inference import read_export,restore_modules
    worker,s,callback,identity,expected,cp=restored_native_output_case
    worker.execute_segment(s,stop=3,output=tmp_path/'owner',identity=identity,capacity_check=lambda:{},native_output=callback)
    root,manifest=read_export(tmp_path/'portable')
    p=manifest['model']['parameters']
    assert all(spec['dtype']=='torch.bfloat16' for spec in p['weight_layout'].values())
    comp=portable.build_inference_model(p);restore_modules(comp,root,manifest)
    actual=comp.state['student'].adapter.state_dict()
    assert all(actual[k].dtype==v.dtype==torch.bfloat16 and torch.equal(actual[k],v)
               for k,v in s.adapter.state_dict().items() if '.adapters.fake_score.' not in k)
    from test_h3_cached_original_cpu import original,sample
    from test_training_h3_cached_inference_cpu import new_inputs
    from distillschedule.data import PreparedDataset
    api=original();raw=sample(9);fresh=new_inputs(tmp_path/'fresh',raw=raw)
    feeds=PreparedDataset(fresh).sample(0)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(p['seed'])
        observed=comp.generate(comp.state,feeds,{})
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(p['seed'])
        inputs,_=api.dmd2.prepare_batch(raw,1.0)
        with torch.no_grad(),torch.autocast('cpu',dtype=torch.bfloat16,cache_enabled=False):
            expected,_=api.dmd2.rollout(comp.state['student'].adapter,
                (inputs['input_latents'],inputs['audio_input_latents']),inputs,
                api.dmd2.make_schedule(portable.original_config(),device='cpu'))
    assert torch.equal(observed['video_latents'],expected[0]) and torch.equal(observed['audio_latents'],expected[1])
    assert not torch.cuda.is_initialized()


def test_storage_precision_labels_and_unsupported_dtypes_remain_strict():
    from distillschedule.training import h3_ar_portable as portable
    import torch
    torch.set_num_threads(1)
    value=output_policy()
    for dtype,precision in (('torch.float32',portable.PRECISION),('torch.bfloat16',portable.NATIVE_BF16_PRECISION)):
        p=portable.parameters(architecture={},geometry=value['geometry'],
            weight_layout={'dit.x.adapters.student.A.weight':dict(shape=[2,2],dtype=dtype)},
            seed=1,evidence_kind=portable.FIXTURE,budget=value['inference_budget'])
        assert p['precision']==precision and portable._request(p)==p
        wrong=deepcopy(p);wrong['precision']=(portable.NATIVE_BF16_PRECISION if precision==portable.PRECISION else portable.PRECISION)
        with pytest.raises(ValueError,match='precision declaration'):portable._request(wrong)
        wrong=deepcopy(p);wrong['weight_layout']['dit.x.adapters.student.A.weight']['dtype']='torch.float16'
        with pytest.raises(ValueError,match='storage layout'):portable._request(wrong)


def test_owner_mutation_is_checked_on_control_plane_before_publication(restored_native_output_case,tmp_path,monkeypatch):
    worker,s,callback,identity,expected,cp=restored_native_output_case
    from distillschedule.training import h3_ar_inference as snapshot,distributed
    original=snapshot.consolidate_current_student
    labels=[];checked=distributed.checked
    def observe(fn,label):
        labels.append(label)
        return checked(fn,label)
    def mutate(*a,**kw):
        result=original(*a,**kw)
        s.census['SYNTHETIC_TEST_ONLY_FORBIDDEN_CALL']=1
        return result
    monkeypatch.setattr(snapshot,'consolidate_current_student',mutate)
    monkeypatch.setattr(distributed,'checked',observe)
    with pytest.raises(ValueError,match='owner or call census'):
        worker.execute_segment(s,stop=3,output=tmp_path/'owner',identity=identity,capacity_check=lambda:{},native_output=callback)
    assert 'W6 output owner preservation' in labels
    assert not (tmp_path/'portable/_SUCCESS').exists()


def test_late_owner_failure_invalidates_only_its_new_export(restored_native_output_case,tmp_path):
    worker,s,callback,identity,expected,cp=restored_native_output_case
    old=tmp_path/'old-unrelated';old.mkdir();(old/'_SUCCESS').write_text('old evidence remains')
    worker.execute_segment(s,stop=3,output=tmp_path/'owner',identity=identity,capacity_check=lambda:{},native_output=callback)
    assert (tmp_path/'portable/_SUCCESS').is_file()
    callback.invalidate()
    assert not (tmp_path/'portable/_SUCCESS').exists()
    assert (tmp_path/'portable/export.json').is_file()  # failed evidence retained
    assert (old/'_SUCCESS').read_text()=='old evidence remains'
