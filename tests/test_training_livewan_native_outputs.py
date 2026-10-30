"""C51 current-G output; CPU fixtures are not native world16 qualification."""
from copy import deepcopy
from pathlib import Path
import ast
import json
import os
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest


def test_native_c51_output_discovery_does_not_import_models():
    result = subprocess.run([sys.executable, '-B', '-c',
        'from distillschedule.training.livewan_native_export import export_native; '
        'from distillschedule.training.livewan_native_inference import FACTORY; import sys; '
        'assert not any(n in sys.modules for n in ("torch","diffusers","wan","wanstreamer"))'],
        text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_native_c51_terminal_output_does_not_relax_training_restart():
    from distillschedule.training.livewan_c51_training import validate_progress, checkpoint_policy
    identity = dict(horizon=3, scope='SYNTHETIC_TEST_ONLY')
    header = dict(cycle=3, cursor=15, versions=dict(generator=3, critic=15, ema=0),
                  policy=checkpoint_policy(identity))
    with pytest.raises(ValueError):
        validate_progress(header, identity=identity)
    validate_progress(header, identity=identity, output_only=True)
    for change in (dict(cycle=4), dict(cycle=0), dict(cursor=14), dict(versions={}), dict(policy={})):
        with pytest.raises(ValueError):
            validate_progress({**header, **change}, identity=identity, output_only=True)
    for flag in (1, None, 'True'):
        with pytest.raises(ValueError):
            validate_progress(header, identity=identity, output_only=flag)


def test_primary_replica_publishes_without_duplicating_role_weights(tmp_path):
    import torch
    from distillschedule.training.inference import publish_export, read_export
    from distillschedule.training.adapters import ModelComponents
    from distillschedule.training.plugins import load_callable
    factory='distillschedule.examples.tiny_av:build_model'
    _, source=load_callable(factory)
    net=torch.nn.Linear(2, 3)
    components=ModelComponents(state={'generator':net}, metadata={'scope':'SYNTHETIC_TEST_ONLY'})
    model=dict(factory=factory,source=source,parameters={},metadata=dict(components.metadata))
    result=publish_export(components,tmp_path/'export',model=model,
        identity=dict(model=source,model_metadata=model['metadata']),cycle=2,primary_replica=True)
    root,manifest=read_export(tmp_path/'export')
    assert manifest==result and manifest['world_size']==1
    saved=torch.load(root/'weights.pt',weights_only=True)
    assert torch.equal(saved['generator']['weight'],net.weight)
    assert not torch.cuda.is_initialized()


def test_native_export_dispatch_preserves_complete_source_identity(tmp_path,monkeypatch):
    from distillschedule.training import native,livewan_c51_training as train
    from distillschedule.training.config import TrainingConfig
    request=dict(schema_version='distillschedule.livewan_native_output_request.v1',test_only=True)
    raw=dict(schema_version='distillschedule.training.v1',model=dict(workload='W4'),
        algorithm=dict(name='original_self_forcing_c51'),
        data=dict(input_kind='original_livewan_c51_teacher_stream.v1',prepared='/explicit/original.json'),
        execution=dict(adapter='native',mode='manual',task='/explicit/task.json',assets='/explicit/assets.json',
            capacity_policy='/explicit/capacity.json',preset='self_forcing.c51_stream.v1'),
        trainer=dict(cycles=100,seed=31415,checkpoint_every=50,output_dir=str(tmp_path/'old'),export=request))
    conf=TrainingConfig.from_dict(raw,source=tmp_path/'training.json')
    seen=[]
    def run(config,*,admit,native_output=None):
        seen.append(config.to_dict())
        assert config.trainer['cycles']==100 and config.trainer['checkpoint_every']==50
        assert config.trainer['resume_from']==str(tmp_path/'checkpoint')
        assert config.trainer['output_dir']==str(tmp_path/'portable.native-owner')
        assert native_output.value==request
        native_output.result={'status':'SYNTHETIC_DISPATCH_ONLY'}
        return 0
    monkeypatch.setattr(train,'run_training',run)
    assert native.export_native(conf,tmp_path/'checkpoint',tmp_path/'portable')['status']=='SYNTHETIC_DISPATCH_ONLY'
    assert len(seen)==1 and conf.to_dict()==raw
    bad=deepcopy(raw);bad['execution']['preset']='self_forcing.c51.v1'
    with pytest.raises(ValueError,match='full fresh C51'):
        native.export_native(TrainingConfig.from_dict(bad,source=conf.source),tmp_path/'checkpoint',tmp_path/'no')


def _reviewed_continuous_worker(parent):
    """Apply only U3 input receipts and U7 opt-in phase receipts to the old AST.

    56ba6b7bcfb5639c9e3eeed56f1c1f610ff67a15 and
    2c94ca49464323f674ae5a229e276bc53c6b65fa added these exact statements.
    The 422e5582 baseline and every other statement/call remain authoritative.
    """
    expected = deepcopy(parent)
    dump = lambda node: ast.dump(node, include_attributes=False)
    expr = lambda text: ast.parse(text, mode='eval').body
    loops = [node for node in expected.body if isinstance(node, ast.For)
             and isinstance(node.target, ast.Name) and node.target.id == 'cycle'
             and dump(node.iter) == dump(expr('range(start_cycle, end_cycle)'))]
    assert len(loops) == 1
    loop = loops[0]
    assert dump(loop.body[1]) == dump(ast.parse('cycle_started = time.perf_counter()').body[0])
    completed = [node for node in loop.body if isinstance(node, ast.If)
                 and dump(node.test) == dump(expr('continuation is not None'))]
    assert len(completed) == 1 and len(completed[0].body) == 1
    assert dump(completed[0].body[0].value.func) == dump(expr('continuation.cycle_complete'))
    loop.body.insert(1, ast.parse('''
if continuation is not None and getattr(continuation, 'phase_diagnostics', None) is not None:
    continuation.phase_receipt(torch=torch, rank=rank, device=device,
        phase='full5_entry', cycle=cycle + 1, versions=versions)
''').body[0])
    completed[0].body.insert(0, ast.parse('''
if getattr(continuation, 'phase_diagnostics', None) is not None:
    continuation.phase_receipt(torch=torch, rank=rank, device=device,
        phase='full5_exit', cycle=cycle + 1, versions=versions)
''').body[0])
    receipts = [node.value for node in expected.body if isinstance(node, ast.Assign)
                and [dump(target) for target in node.targets] == [dump(
                    ast.parse('evidence.continuous_pipeline = {}').body[0].targets[0])]]
    assert len(receipts) == 1 and isinstance(receipts[0], ast.Dict)
    receipt = receipts[0]
    positions = [i for i, key in enumerate(receipt.keys)
                 if isinstance(key, ast.Constant) and key.value == 'source_reuse_across_cycles']
    assert len(positions) == 1
    index = positions[0]
    assert dump(receipt.values[index]) == dump(expr('False if continuation is not None else trace.source_reuse_across_cycles'))
    receipt.values[index] = expr('trace.source_reuse_across_cycles')
    final = expected.body[-1]
    assert isinstance(final, ast.If) and dump(final.test) == dump(expr('continuation is not None'))
    assert len(final.body) == 1 and isinstance(final.body[0], ast.Expr)
    update = final.body[0].value
    assert isinstance(update, ast.Call) and dump(update.func) == dump(expr('evidence.continuous_pipeline.update'))
    flags = [keyword for keyword in update.keywords if keyword.arg == 'source_reuse_across_cycles']
    assert len(flags) == 1 and dump(flags[0].value) == dump(expr('False'))
    flags[0].arg, flags[0].value = None, expr('input_marker(trace)')
    final.body.insert(0, ast.parse('from distillschedule.training.livewan_c51_training import input_marker').body[0])
    return expected


def _assert_worker_lineage(parent, current):
    def functions(text):
        return {node.name: node for node in ast.parse(text).body if isinstance(node, ast.FunctionDef)}
    old, new = functions(parent), functions(current)
    names = ('_run_pipeline_8x8', 'generator_backward_for_sample', 'critic_backward_for_batch')
    for name in names:
        assert ast.dump(old[name], include_attributes=False) == ast.dump(new[name], include_attributes=False), name
    name = '_run_pipeline_8x8_continuous'
    assert ast.dump(_reviewed_continuous_worker(old[name]), include_attributes=False) == ast.dump(new[name], include_attributes=False), name


def _worker_lineage_sources():
    from distillschedule.runtime import livewan_crossscale_worker as worker
    root=Path(worker.__file__).parents[3]
    parent=subprocess.check_output(['git','show','422e5582be57c5b060738aea8be01004c8468f83:'+str(
        Path(worker.__file__).relative_to(root))],cwd=root,text=True)
    return parent, Path(worker.__file__).read_text()


def test_real_worker_output_seam_precedes_any_context_or_transport_and_math_is_unchanged():
    from distillschedule.runtime import livewan_crossscale_worker as worker
    import inspect
    source=inspect.getsource(worker.run_worker)
    assert source.index('continuation.export_current(')<source.index('generator_streamers = []')
    assert source.index('continuation.export_current(')<source.index('build_score_context(')
    assert source.index('continuation.export_current(')<source.index('transport_factory =')
    _assert_worker_lineage(*_worker_lineage_sources())


@pytest.mark.parametrize('mutation', ['entry-name', 'exit-cycle', 'versions', 'phase-location',
                                    'input-marker', 'reuse-flag', 'loop-bound', 'math-call'])
def test_worker_source_review_rejects_unreviewed_receipt_or_math_changes(mutation):
    parent, source = _worker_lineage_sources()
    tree = ast.parse(source)
    worker = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == '_run_pipeline_8x8_continuous')
    phases = [node for node in ast.walk(worker) if isinstance(node, ast.Call)
              and isinstance(node.func, ast.Attribute) and node.func.attr == 'phase_receipt']
    assert len(phases) == 2
    phase = {next(keyword.value.value for keyword in call.keywords if keyword.arg == 'phase'): call
             for call in phases}
    if mutation in ('entry-name', 'exit-cycle', 'versions'):
        call = phase['full5_exit' if mutation == 'exit-cycle' else 'full5_entry']
        key = {'entry-name': 'phase', 'exit-cycle': 'cycle', 'versions': 'versions'}[mutation]
        next(keyword for keyword in call.keywords if keyword.arg == key).value = ast.Constant('UNREVIEWED')
    elif mutation == 'phase-location':
        loop = next(node for node in worker.body if isinstance(node, ast.For))
        loop.body[1], loop.body[2] = loop.body[2], loop.body[1]
    elif mutation == 'input-marker':
        marker = next(node for node in ast.walk(worker) if isinstance(node, ast.Call)
                      and isinstance(node.func, ast.Name) and node.func.id == 'input_marker')
        marker.args = [ast.Name(id='other_trace', ctx=ast.Load())]
    elif mutation == 'reuse-flag':
        receipt = next(node.value for node in worker.body if isinstance(node, ast.Assign)
                       and isinstance(node.value, ast.Dict) and any(isinstance(key, ast.Constant)
                           and key.value == 'source_reuse_across_cycles' for key in node.value.keys))
        index = next(i for i, key in enumerate(receipt.keys) if isinstance(key, ast.Constant)
                     and key.value == 'source_reuse_across_cycles')
        receipt.values[index] = ast.Constant(False)
    elif mutation == 'loop-bound':
        loop = next(node for node in worker.body if isinstance(node, ast.For))
        loop.iter.args[-1] = ast.Constant(2)
    else:
        call = next(node for node in ast.walk(worker) if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name) and node.func.id == '_run_pipeline_8x8')
        next(keyword for keyword in call.keywords if keyword.arg == 'step_offset').value = ast.Constant(0)
    with pytest.raises(AssertionError):
        _assert_worker_lineage(parent, ast.unparse(ast.fix_missing_locations(tree)))


def _primary_worker(rank,rendezvous,root,fault):
    import torch
    import torch.distributed as dist
    from datetime import timedelta
    from distillschedule.training.adapters import ModelComponents
    from distillschedule.training.inference import publish_export
    from distillschedule.training.plugins import load_callable
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2,timeout=timedelta(seconds=30))
    try:
        # Genuine role partition: the Score-like rank has no Generator tensor.
        components=ModelComponents(state={'generator':torch.nn.Linear(2,3)},metadata={'scope':'SYNTHETIC_TEST_ONLY'}) if rank==0 else None
        if fault=='foreign_owner' and rank==1:
            components=ModelComponents(state={'wrong_role':torch.nn.Linear(3,2)})
        if fault=='dtensor':
            from torch.distributed.tensor import distribute_tensor,Replicate
            from torch.distributed.device_mesh import init_device_mesh
            # Both ranks construct only this tiny rejection fixture, not a
            # production Generator or a new primary-replica staging path.
            mesh=init_device_mesh('cpu',(2,))
            value=distribute_tensor(torch.ones(3,2),device_mesh=mesh,placements=[Replicate()])
            if rank==0:components.state['generator'].weight=torch.nn.Parameter(value)
        _,origin=load_callable('distillschedule.examples.tiny_av:build_model')
        model=dict(factory=origin['reference'],source=origin,parameters={},metadata={'scope':'SYNTHETIC_TEST_ONLY'})
        if fault=='metadata' and rank==1:model['metadata']={'wrong':True}
        try:
            value=publish_export(components,Path(root)/'export',model=model,
                identity=dict(model=origin,model_metadata=model['metadata']),cycle=3,primary_replica=True)
            assert fault is None and value['checkpoint_cycle']==3 and value['world_size']==2
            result={'status':'PASS_REAL_GLOO2_COMPLETE_PRIMARY_REPLICA','cuda_initialized':torch.cuda.is_initialized()}
        except (ValueError,RuntimeError) as exc:
            assert fault is not None
            result={'status':'EXPECTED_COLLECTIVE_REJECTION','error':str(exc)}
        (Path(root)/f'rank-{rank}.json').write_text(json.dumps(result))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('fault',[None,'foreign_owner','metadata','dtensor'])
def test_real_gloo_primary_replica_never_allocates_generator_on_score(tmp_path,fault):
    import torch.multiprocessing as mp
    mp.spawn(_primary_worker,args=(str(tmp_path/'rdzv'),str(tmp_path),fault),nprocs=2,join=True)
    rows=[json.loads((tmp_path/f'rank-{rank}.json').read_text()) for rank in range(2)]
    if fault is None:
        assert all(row['status']=='PASS_REAL_GLOO2_COMPLETE_PRIMARY_REPLICA' and not row['cuda_initialized'] for row in rows)
    else:
        assert all(row['status']=='EXPECTED_COLLECTIVE_REJECTION' for row in rows)
        assert not (tmp_path/'export/_SUCCESS').exists()


@pytest.mark.parametrize('fault',['mutually_exclusive','meta','not_boolean','custom'])
def test_primary_replica_cannot_hide_incomplete_state(tmp_path,fault):
    import torch
    from distillschedule.training.adapters import ModelComponents
    from distillschedule.training.inference import publish_export
    model={'test_only':True};components=ModelComponents(state={'student':torch.nn.Linear(2,3,device='meta' if fault=='meta' else 'cpu')})
    options=dict(primary_replica=True)
    if fault=='mutually_exclusive':options['primary_only']=True
    if fault=='not_boolean':options['primary_replica']=1
    if fault=='custom':components.export=lambda *a:(_ for _ in ()).throw(AssertionError('custom exporter ran'))
    with pytest.raises(ValueError):
        publish_export(components,tmp_path/'export',model=model,identity={},cycle=1,**options)
    assert not (tmp_path/'export').exists()


@pytest.fixture(scope='module')
def original_current_case(tmp_path_factory):
    """Actual original class/block gradient/Adam, NOT native C51 training."""
    import torch
    from distillschedule.reference import livewan_wan21 as provider
    from distillschedule.training import livewan_native_inference as inference
    from distillschedule.training.wan_native_inference import _load_codec
    from test_livewan_reference_cpu import fixture_model
    from test_training_wan_native_outputs import codec_request
    root=tmp_path_factory.mktemp('real-original-current-C51-Student')
    torch.set_num_threads(1)
    api=provider.load_cpu(backend='torch-sdpa-cpu',precision='float32')
    student=fixture_model(687).train()
    with torch.no_grad():
        student.freqs[1,0]*=complex(.99,.03)  # prove actual nonpersistent state is captured, not guessed
    generator=torch.Generator().manual_seed(952)
    context=torch.randn(1,512,4096,generator=generator)
    z=torch.randn(1,16,3,4,4,generator=generator)
    rope=api.core.make_rope_table(student,2,2,278,'cpu')
    optimizer=torch.optim.AdamW(student.parameters(),lr=2e-6,betas=(0.,.99),weight_decay=0.,eps=1e-8,foreach=True)
    before=student.head.head.weight.detach().clone()
    result=api.blockcausal.block_forward(student,z,.4,0,rope,student.text_embedding(context),None,dtype=torch.float32)
    result.square().mean().backward();optimizer.step()
    assert not torch.equal(student.head.head.weight,before)
    spec=codec_request(root/'old-codec')
    codec,portable=_load_codec(spec['codec'],spec['budget'],inference.FIXTURE)
    p=inference.plain(dict(schema_version=inference.SCHEMA,source=inference.inference_source(),
        runtime=inference.runtime_identity(),architecture=inference.architecture(student),
        geometry=dict(resolution=[32,32],frames=81,fps=16),precision=inference.PRECISION,seed=822,
        evidence_kind=inference.FIXTURE,budget=dict(max_parameters=2_000_000,max_loaded_bytes=2**30,
            max_primary_export_bytes=2**30,max_attention_elements=8_000_000,max_input_elements=1_000_000,
            host_reserve_bytes=2**20),codec=portable))
    feeds=dict(world=torch.randn(1,16,6,4,4,generator=generator),context=context)
    return NS(root=root,student=student,optimizer=optimizer,codec=codec,parameters=p,feeds=feeds,api=api,codec_spec=spec['codec'])


def _cpu_owner(case,output):
    # Only the physical world16/cursor shell is synthetic. Current model,
    # actual gradients/Adam, codec, publication and independent sampler are real.
    versions=dict(generator=2,critic=0,ema=0)
    progress=NS(actual=versions,state_dict=lambda:dict(cycle=2,versions=versions,scope='SYNTHETIC_TEST_ONLY'))
    lifecycle=NS(header={'cycle':2},loader=NS(cycle=2,pending=[]),progress=progress,
        state={},identity={'scope':'SYNTHETIC_TEST_ONLY_CPU_OUTPUT_CALLBACK'},capacity_check=lambda:None,output=output)
    bundle=NS(student=case.student,critic=None,critic_shadow=None,teacher=None)
    return lifecycle,bundle


def _native_callback(case,root,monkeypatch):
    from distillschedule.training.livewan_native_export import NativeOutputRequest,SCHEMA
    p=case.parameters
    request=NativeOutputRequest(dict(schema_version=SCHEMA,seed=p['seed'],geometry=p['geometry'],budget=p['budget'],
                                    codec=case.codec_spec),root)
    request.codec=case.codec
    # Native production parameters MUST require 368x640/full1.3B. Only this
    # explicit fixture substitutes its bounded original class/geometry label.
    monkeypatch.setattr(request,'_parameters',lambda student:deepcopy(p))
    return request


@pytest.fixture(scope='module')
def portable_current_case(original_current_case,tmp_path_factory):
    import torch
    from distillschedule.training.checkpoint import cpu_tree,rng_state
    from distillschedule.training.inference import read_export,restore_modules
    from distillschedule.training import livewan_native_inference as inference
    from test_training_end_to_end import equal
    from _pytest.monkeypatch import MonkeyPatch
    case=original_current_case
    root=tmp_path_factory.mktemp('C51-standalone-output')
    lifecycle,bundle=_cpu_owner(case,root/'owner')
    before=dict(weights=cpu_tree(case.student.state_dict()),optimizer=cpu_tree(case.optimizer.state_dict()),rng=rng_state())
    with MonkeyPatch.context() as patch:
        callback=_native_callback(case,root/'export',patch)
        report=callback.publish(lifecycle,bundle,rank=0)
    assert report['output_only']==dict(cycles=0,forward=0,backward=0,optimizer=0,ema=0,samples_consumed=0,merged_lora=False)
    assert equal(before,dict(weights=cpu_tree(case.student.state_dict()),optimizer=cpu_tree(case.optimizer.state_dict()),rng=rng_state()))
    artifact,manifest=read_export(root/'export')
    separate=inference.build_inference_model(case.parameters)
    with pytest.raises(ValueError,match='strict restored'):
        separate.generate(separate.state,case.feeds,{})
    restore_modules(separate,artifact,manifest)
    assert equal(separate.state['generator'].model.state_dict(),case.student.state_dict())
    # Independent native sampler oracle with the same current Student. No W1
    # pyramid math and no precomputed training rollout is used.
    with torch.no_grad():
        stream=case.api.stream.FewStepStreamer(case.student,width=32,height=32,max_frames=278,device='cpu',
            dtype=torch.float32,window_frames=None,time_scale=1000.,cache_frames=22,block_frames=3,num_steps=4,cuda_graphs=False)
        stream.set_text(case.student.text_embedding(case.feeds['context']));stream.set_world(case.feeds['world'])
        generator=torch.Generator().manual_seed(case.parameters['seed'])
        expected=torch.cat([case.feeds['world'],*[stream.generate_block(generator=generator) for _ in range(5)]],dim=2)
    generated=separate.generate(separate.state,case.feeds,{})
    torch.testing.assert_close(generated['latents'],expected,rtol=0,atol=0)
    assert equal(before,dict(weights=cpu_tree(case.student.state_dict()),optimizer=cpu_tree(case.optimizer.state_dict()),rng=rng_state()))
    torch.save(case.feeds,root/'new-inputs.pt');torch.save(generated,root/'expected.pt')
    from distillschedule.data.preparation import import_preencoded
    from distillschedule.data.manifest import PREENCODED_SCHEMA
    from distillschedule.workloads.best_known.registry import file_sha256
    source=root/'inputs.json';source.write_text(json.dumps(dict(schema_version=PREENCODED_SCHEMA,
        records=[dict(id='NEW-CLEAN-WORLD-POSITIVE-TEXT',features=dict(path=str(root/'new-inputs.pt'),
            sha256=file_sha256(root/'new-inputs.pt')))])))
    prepared=import_preencoded(source,root/'prepared')
    return NS(root=root,manifest=manifest,prepared=prepared,config=Path(report['inference_config']),case=case)


def test_real_current_student_callback_and_original_streamer_exact(portable_current_case):
    assert (portable_current_case.root/'export/_SUCCESS').is_file()


def test_fresh_process_generates_and_decodes_without_training_assets(portable_current_case):
    case=portable_current_case;old=case.case.root/'old-codec'
    import shutil
    shutil.rmtree(old)
    script=r'''
import json,sys
from pathlib import Path
import torch
from distillschedule.training.config import load_training_config
from distillschedule.training.session import generate_from_export
from test_training_end_to_end import equal
torch.set_num_threads(1)
root=Path(sys.argv[1]);config=load_training_config(root/'export.inference.json')
result=generate_from_export(config,root/'export',root/'generated',prepared=root/'prepared/manifest.json',output_kind='video')
actual=torch.load(result['artifact'],weights_only=True)
expected=torch.load(root/'expected.pt',weights_only=True)
assert equal(actual['generated'],expected)
assert len(list((root/'generated/frames').glob('*.png')))==81
assert not torch.cuda.is_initialized()
print(json.dumps(dict(status='PASS_REAL_ORIGINAL_CURRENT_STUDENT_FRESH_PROCESS_CPU',video_frames=81)))
'''
    result=subprocess.run([sys.executable,'-B','-c',script,str(case.root)],
        env=dict(os.environ, PYTHONPATH=os.pathsep.join((str(Path(__file__).resolve().parent),
                                                       os.environ.get('PYTHONPATH', '')))),
        text=True,capture_output=True,timeout=150)
    assert result.returncode==0,result.stdout+result.stderr


@pytest.mark.parametrize('fault',['shape','dtype','history','negative','override','source','budget','unrestored','native-tiny'])
def test_native_consumer_refuses_invalid_inputs_and_identity(original_current_case,fault):
    import torch
    from distillschedule.training import livewan_native_inference as inference
    from distillschedule.training.checkpoint import rng_state
    from test_training_end_to_end import equal
    case=original_current_case;p=deepcopy(case.parameters)
    if fault=='source':p['source']['consumer']='0'*64
    if fault=='budget':p['budget']['max_loaded_bytes']=64
    if fault=='native-tiny':p.update(evidence_kind=inference.NATIVE,geometry=dict(resolution=[368,640],frames=81,fps=16))
    if fault in ('source','budget','native-tiny'):
        message={'source':'source/runtime/precision','budget':'capacity','native-tiny':'current full native'}[fault]
        with pytest.raises(ValueError,match=message):inference.build_inference_model(p)
        return
    model=inference.build_inference_model(p)
    if fault!='unrestored':model.state['generator'].model.load_state_dict(case.student.state_dict(),strict=True)
    feeds={k:v.clone() for k,v in case.feeds.items()};options={}
    if fault=='shape':feeds['context']=feeds['context'][:,:2]
    if fault=='dtype':feeds['world']=feeds['world'].bfloat16()
    if fault=='history':feeds['world']=feeds['world'][:,:,:4]
    if fault=='negative':feeds['negative']=feeds['context']
    if fault=='override':options={'num_steps':1}
    before=rng_state()
    with pytest.raises(ValueError):model.generate(model.state,feeds,options)
    assert equal(before,rng_state())


@pytest.mark.parametrize('fault',['config_failure','owner_mutation','capacity','merged'])
def test_native_publication_failures_leave_no_valid_artifact(original_current_case,tmp_path,monkeypatch,fault):
    from distillschedule.training import livewan_native_export as module
    case=original_current_case
    callback=_native_callback(case,tmp_path/'export',monkeypatch)
    lifecycle,bundle=_cpu_owner(case,tmp_path/'owner')
    if fault=='config_failure':callback.config_path.write_text('DO NOT REPLACE')
    if fault=='owner_mutation':
        observe=module._owner;seen=[]
        def changed(*args):
            row=observe(*args);seen.append(True)
            if len(seen)>1:row['pending']=['SYNTHETIC_TEST_ONLY_MUTATION']
            return row
        monkeypatch.setattr(module,'_owner',changed)
    if fault=='capacity':
        monkeypatch.setattr(module.NativeOutputRequest,'_project',lambda *a:(_ for _ in ()).throw(ValueError('capacity')))
    if fault=='merged':monkeypatch.setattr(case.student,'merged',True,raising=False)
    match={'config_failure':'File exists','owner_mutation':'changed native','capacity':'capacity','merged':'merged'}[fault]
    with pytest.raises((ValueError,FileExistsError),match=match):callback.publish(lifecycle,bundle,rank=0)
    assert not (callback.output/'_SUCCESS').exists()


def test_native_view_preserves_actual_nonpersistent_positional_storage(original_current_case):
    from distillschedule.training.livewan_native_inference import current_view,architecture
    student=original_current_case.student
    freqs=student.freqs
    keys=set(student.state_dict())
    view=current_view(student)
    assert student.freqs is freqs and view.rope_frequencies is freqs
    assert 'rope_frequencies' in view.state_dict() and set(student.state_dict())==keys
    assert 'patch_size' not in student.config and architecture(student)['patch_size']==[1,2,2]


from test_training_livewan_state import original_source,original_lora
from test_training_livewan_c51_continuation import stream


@pytest.mark.parametrize('role',['generator','critic'])
def test_actual_complete_checkpoint_restores_before_output_without_next_sample(
        stream,original_source,original_lora,tmp_path,monkeypatch,role):
    """Real named Adam/LoRA/data/RNG checkpoint; scalar model edge, NOT Wan math."""
    import torch
    from test_training_livewan_state import owner,update,equal
    from test_training_livewan_c51_continuation import pool,consume
    from distillschedule.training.livewan_c51_state import owner_state,Progress,physical_versions
    from distillschedule.training.livewan_c51_training import Continuation,checkpoint_policy
    from distillschedule.training.checkpoint import save_checkpoint,read_manifest,pack_state,cpu_tree,rng_state
    rank=0 if role=='generator' else 8
    cursor=pool(stream,rank)
    current=owner(original_lora,role)
    actual=physical_versions(rank,0)
    progress=Progress(cursor,actual)
    state=owner_state(current.bundle,source_root=original_source,progress=progress)
    for cycle in range(4):
        for _ in range(1 if rank==0 else 5):update(current);actual[role]+=1
        consume(cursor,cycle)
    identity=dict(horizon=4,scope='SYNTHETIC_PHYSICAL_CPU_SCALAR_EDGE_WITH_REAL_CHECKPOINT')
    saved=tmp_path/'checkpoint'
    save_checkpoint(saved,state,identity=identity,cycle=4,cursor=20,
        versions=dict(generator=4,critic=20,ema=0),policy=checkpoint_policy(identity))
    expected=cpu_tree(pack_state(state));random=rng_state()
    fresh=owner(original_lora,role)
    loader=pool(stream,rank).loader
    def no_read(*a,**kw):raise AssertionError('output read a training sample')
    monkeypatch.setattr(type(loader),'load',no_read)
    monkeypatch.setattr(type(loader),'context_from_trace',no_read)
    config=NS(trainer=dict(cycles=4,checkpoint_every=2,resume_from=str(saved)))
    class Output:
        def publish(self,cont,bundle,*,rank):
            assert cont.loader.cycle==4 and not cont.loader.pending and cont.cycles_executed==0
            assert equal(cpu_tree(pack_state(cont.state)),expected)
            assert equal(rng_state(),random)
            return {'status':'PASS_CPU_ACTUAL_RESTORE_NO_DATA_READ_NO_NEW_UPDATE'}
    cont=Continuation(config=config,admitted={'assets':{'native':{'sources':{'livewan':{'root':str(original_source)}}}}},
        identity=identity,output=tmp_path/'separate-output',bootstrap=None,loader=loader,
        header=read_manifest(saved,identity=identity),capacity_check=lambda:None,native_output=Output())
    restored=physical_versions(rank,0)
    cont.bind(torch=torch,rank=rank,bundle=fresh.bundle,versions=restored,pairs=cursor.pairs)
    assert cont.export_current(bundle=fresh.bundle,rank=rank)['status'].startswith('PASS_CPU_ACTUAL_RESTORE')
    assert restored==physical_versions(rank,4) and equal(rng_state(),random)


def test_explicit_inference_capacity_keeps_reference_limits_and_original_math():
    import torch
    from distillschedule.reference import livewan_wan21 as provider
    torch.set_num_threads(1)
    api=provider.load_cpu(backend='torch-sdpa-cpu',precision='float32')
    # Small inputs but just above the ORIGINAL 16M attention-element guard.
    # The extra workspace is real CPU math, never a native 1.3B qualification.
    generator=torch.Generator().manual_seed(985)
    q,k,v=[torch.randn(1,4097,1,16,generator=generator) for _ in range(3)]
    with torch.no_grad():
        with pytest.raises(ValueError,match='attention-element budget'):
            api.attention.flash_attention(q,k,v)
        with api.attention.portable_inference_capacity(max_input_elements=100_000,max_attention_elements=17_000_000):
            actual=api.attention.flash_attention(q,k,v)
        expected=api.attention.sdpa_original.sdpa_attention(q,k,v,dtype=torch.float32)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        with pytest.raises(ValueError,match='attention-element budget'):
            api.attention.flash_attention(q,k,v)
    with pytest.raises(ValueError,match='inference-only'):
        with api.attention.portable_inference_capacity(max_input_elements=100_000,max_attention_elements=17_000_000):pass
    assert api.attention.MAX_ATTENTION_ELEMENTS==16_777_216
    assert api.attention.MAX_INPUT_ELEMENTS==8_388_608
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('bounds',[(True,100),(1,True),(64_000_001,100),(100,20_000_000_001)])
def test_portable_inference_capacity_refuses_unbounded_or_noninteger_limits(bounds):
    import torch
    from distillschedule.reference import livewan_wan21 as provider
    api=provider.load_cpu(backend='torch-sdpa-cpu',precision='float32')
    with torch.no_grad(),pytest.raises(ValueError,match='bounded explicit'):
        with api.attention.portable_inference_capacity(max_input_elements=bounds[0],max_attention_elements=bounds[1]):pass
    assert api.attention._INFERENCE_CAPACITY.get() is None


@pytest.mark.parametrize('fault',['budget','seed','geometry','existing','missing_shards'])
def test_output_request_rejects_bad_policy_before_codec_or_training_construction(tmp_path,monkeypatch,fault):
    from distillschedule.training import livewan_native_export as module
    request=dict(schema_version=module.SCHEMA,seed=822,geometry=dict(resolution=[368,640],frames=81,fps=16),
        budget=dict(max_parameters=2_000_000_000,max_loaded_bytes=128*2**30,max_primary_export_bytes=32*2**30,
                    max_attention_elements=10_000_000_000,max_input_elements=64_000_000,host_reserve_bytes=2**30),
        codec={'weights':[{'bytes':1}]})
    if fault=='budget':request['budget']['max_loaded_bytes']=0
    if fault=='seed':request['seed']=-1
    if fault=='geometry':request['geometry']['resolution']=[32,32]
    if fault=='missing_shards':request['codec']['weights']=[]
    callback=module.NativeOutputRequest(request,tmp_path/'export')
    if fault=='existing':callback.owner_output.mkdir()
    monkeypatch.setattr(module.distributed,'primary',lambda *a:(_ for _ in ()).throw(AssertionError('codec construction reached')))
    with pytest.raises(ValueError):callback.admit({}, {'scope':'SYNTHETIC_TEST_ONLY'})
    assert not (callback.output/'_SUCCESS').exists()
