"""Bounded ORIGINAL LiveWan geometry; CPU metadata is not GPU admission."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from distillschedule.search.livewan_implementation import LIVEWAN, LiveWanImplementation
from distillschedule.workloads.best_known.registry import fingerprint, load_profile


def test_request_identity_roundtrip_and_default_unchanged():
    from distillschedule.search.livewan_request import LiveWanRequest
    from distillschedule.search.contracts import semantic_contract
    default, new = LiveWanRequest(), LiveWanRequest(frames=69)
    assert default.profile() == load_profile(LIVEWAN)
    assert new.latent_shape == (1, 16, 18, 46, 80)
    assert new.attention_tokens == 16560
    assert [new.new_blocks(w) for w in (3, 6, 9)] == [5, 4, 3]
    assert [new.source_nograd_calls(w) for w in (3, 6, 9)] == [26, 21, 16]
    assert fingerprint(semantic_contract(new.profile())) != fingerprint(semantic_contract(default.profile()))
    for req in (default, new):
        assert LiveWanRequest.from_dict(req.to_dict()) == req
        impl = LiveWanImplementation('pipeline', True, True, True, request=req)
        assert LiveWanImplementation.parse(impl.id) == impl
        assert LiveWanImplementation.from_assignment(impl.assignment) == impl
    assert LiveWanImplementation('pipeline').id == 'joint-livewan1-pipeline-m0-g0-r0'
    with pytest.raises(ValueError):
        LiveWanRequest.from_dict({**new.to_dict(), 'steps_per_block': 2})


@pytest.mark.parametrize('value', [1, 8, 18, 65, 73, True, 69.0])
def test_only_bounded_pixel_frames_are_admitted(value):
    from distillschedule.search.livewan_request import LiveWanRequest
    with pytest.raises(ValueError):
        LiveWanRequest(frames=value)


def test_candidate_precision_cost_and_cli_bind_request():
    from distillschedule.search.livewan_request import LiveWanRequest
    from distillschedule.search.native_macro_adapter import native_graph, cost_key, precision_contract
    from distillschedule.search.workflow_cli import candidate_inventory
    old = LiveWanImplementation('pipeline')
    new = LiveWanImplementation('pipeline', request=LiveWanRequest(frames=69))
    op = native_graph(LIVEWAN).operations[0]
    assert old.id != new.id and cost_key(LIVEWAN, op, old) != cost_key(LIVEWAN, op, new)
    assert precision_contract(LIVEWAN) != precision_contract(LIVEWAN, livewan_request=new.request)
    profile = new.lower_profile()
    assert profile['contract']['request'] == new.request.to_dict()
    inventory = candidate_inventory(LIVEWAN, request=new.request)
    assert len(inventory['candidates']) == 5
    assert len({r['semantic_fingerprint'] for r in inventory['candidates']}) == 1
    assert all(LiveWanImplementation.parse(r['id']).request == new.request for r in inventory['candidates'])


def test_unified_cli_request_inventory_and_source_only_calibration(tmp_path,monkeypatch,capsys):
    from distillschedule.search.livewan_request import LiveWanRequest
    from distillschedule.search.workflow_cli import main
    from distillschedule.search.execution import NativeTrialRunner
    from distillschedule.workloads.best_known.registry import git_head,repository_root
    monkeypatch.setattr(NativeTrialRunner,'_rpc',lambda *a,**kw:pytest.fail('CPU intent must not launch or SSH'))
    req=LiveWanRequest(frames=69);path=tmp_path/'request.json';path.write_text(json.dumps(req.to_dict()))
    assert main(['candidates','--workload',LIVEWAN,'--request',str(path)])==0
    inventory=json.loads(capsys.readouterr().out)
    assert inventory['workload_request']==req.to_dict()
    baseline=LiveWanImplementation('fsdp',request=req)
    peer=LiveWanImplementation('pipeline',True,True,True,request=req)
    args=['calibrate','--workload',LIVEWAN,'--expected-head',git_head(repository_root()),'--run-id','cpu-intent']
    out=tmp_path/'intent'
    assert main([*args,'--candidate',baseline.id,peer.id,'--output',str(out)])==0
    result=json.loads((out/'intent.json').read_text())
    assert result['gpu_execution']=='NOT_RUN'
    assert all(p['fully_resolved_profile']['contract']['request']==req.to_dict() for p in result['source_only_plans'])
    bad=tmp_path/'wrong'
    assert main([*args,'--candidate',baseline.id,LiveWanImplementation('pipeline').id,'--output',str(bad)])==2
    assert not bad.exists()


def test_common_native_calibration_assembly_search_and_saved_lowering(tmp_path,monkeypatch):
    """Real common chain over explicitly synthetic CPU native receipt envelopes."""
    import test_refactor_native_macro_calibration as fixtures
    from distillschedule.search.livewan_request import LiveWanRequest
    from distillschedule.search.native_macro_calibration import assemble_calibrations
    from distillschedule.search.native_macro_adapter import lower_prediction,validate_lowered_plan
    from distillschedule.search.model_search import search_plans
    from distillschedule.search.cost_model import CostModel
    from distillschedule.workloads.best_known.registry import git_head,repository_root
    head=git_head(repository_root());monkeypatch.setattr(fixtures,'HEAD',head)
    req=LiveWanRequest(frames=69);paths=[]
    anchors=(LiveWanImplementation('fsdp',request=req),LiveWanImplementation('pipeline',True,True,True,request=req))
    for index,impl in enumerate(anchors):
        folder=tmp_path/f'fixture{index}';folder.mkdir()
        f=fixtures._fixture(folder,monkeypatch,LIVEWAN,impl)
        result=fixtures._assemble(f);path=folder/'result.json';path.write_text(json.dumps(result));paths.append(path)
    problem,costs=assemble_calibrations(paths,output=tmp_path/'assembled',expected_head=head)
    result=search_plans(problem,costs,exhaustive=True,max_proposals=32)
    receipt=lower_prediction(result,problem,costs,run_id='cpu-selected')
    serialized=json.loads(json.dumps(receipt));validate_lowered_plan(serialized)
    assert LiveWanImplementation.parse(serialized['search_candidate']['id']).request==req
    assert serialized['native_plan']['template']['workload']['shape']['latent']['frames']==18
    assert {r['assignment']['request'] for r in result['ranked']}=={'f69'}
    old_binding={**costs.binding,'semantic_fingerprint':fingerprint(__import__(
        'distillschedule.search.contracts',fromlist=['semantic_contract']).semantic_contract(load_profile(LIVEWAN)))}
    with pytest.raises(ValueError,match='mismatch'):
        CostModel(old_binding,costs.records).bind(problem.binding,problem_scope='MEASURED')


def test_native_lowering_is_request_derived_not_template_mutation():
    from distillschedule.search.livewan_request import LiveWanRequest, lower_native_plan
    from distillschedule.search.execution import resolve_candidate, validated_profile
    from distillschedule.workloads.best_known.registry import repository_root
    old = resolve_candidate(LIVEWAN, LiveWanImplementation('pipeline').id)
    new = resolve_candidate(LIVEWAN, LiveWanImplementation('pipeline', request=LiveWanRequest(frames=69)).id)
    assert new['native_plan']['template_digest'] == old['native_plan']['template_digest']
    assert new['native_plan']['digest'] != old['native_plan']['digest']
    raw = new['native_plan']['template']
    assert raw['workload']['shape']['frames'] == 18
    assert raw['metadata']['critic_attention_checkpoint']['query_tokens'] == 16560
    assert raw['workload']['metadata']['steps_per_block'] == 4
    validated_profile(new, root=repository_root())
    tampered = copy.deepcopy(new)
    tampered['fully_resolved_profile']['contract']['request']['frames'] = 81
    with pytest.raises(ValueError):
        validated_profile(tampered, root=repository_root())
    assert lower_native_plan(copy.deepcopy(raw), LiveWanRequest(frames=69)) == raw


def test_immutable_geometry_install_and_thread_reads():
    code = '''
from concurrent.futures import ThreadPoolExecutor
from distillschedule.search.livewan_request import LiveWanRequest
from distillschedule.runtime.livewan_geometry import install_geometry, geometry
r = LiveWanRequest(frames=69)
install_geometry(r, plan_sha256='a'*64)
with ThreadPoolExecutor(8) as pool:
    assert list(pool.map(lambda _: geometry().latent_shape, range(32))) == [r.latent_shape]*32
try:
    install_geometry(r, plan_sha256='a'*64)
except RuntimeError:
    pass
else:
    raise AssertionError('reinstallation must fail even for same request')
'''
    subprocess.run([sys.executable, '-c', code], check=True)


@pytest.mark.parametrize('placement,flags',[('fsdp',(False,False,False)),('pipeline',(False,False,False)),
    ('pipeline',(True,False,False)),('pipeline',(True,True,False)),('pipeline',(True,True,True))])
def test_actual_native_contract_helpers_accept_only_fully_lowered_geometry(placement,flags):
    from distillschedule.search.livewan_request import LiveWanRequest
    cid=LiveWanImplementation(placement,*flags,request=LiveWanRequest(frames=69)).id
    code='''
from distillschedule.search.execution import resolve_candidate
from distillschedule.search.livewan_implementation import LIVEWAN
from distillschedule.search.livewan_request import request_from_plan
from distillschedule.workloads.best_known.registry import fingerprint
from distillschedule.runtime.livewan_geometry import install_geometry
from distillschedule.runtime.livewan_crossscale_worker import _validate_plan_identity,_apply_trajectory_contract,_generator_runahead_transport_enabled
from distillschedule.core.plan import ExecutionPlan
import sys
def expand(v):
    if isinstance(v,str) and '${ENV:' in v:return '/cpu/unused'
    if isinstance(v,list):return [expand(x) for x in v]
    if isinstance(v,dict):return {k:expand(x) for k,x in v.items()}
    return v
p=ExecutionPlan.from_mapping(expand(resolve_candidate(LIVEWAN,sys.argv[1])['native_plan']['template'])).to_dict()
install_geometry(request_from_plan(p),plan_sha256=fingerprint(p))
mode=p['metadata']['execution_mode']
_apply_trajectory_contract(p,cycles=2,mode=mode)
_validate_plan_identity(p,mode=mode)
if mode=='pipeline':assert _generator_runahead_transport_enabled(p,mode=mode)
'''
    subprocess.run([sys.executable,'-c',code,cid],check=True)


def test_exact_new_input_not_prefix_crop(tmp_path):
    torch = pytest.importorskip('torch')
    from distillschedule.search.livewan_request import LiveWanRequest
    from distillschedule.runtime.livewan_crossscale_worker import InputLoader, FormalTrace, OperationTraceRow
    req = LiveWanRequest(frames=69)
    prompts = tmp_path/'prompts.pt'
    torch.save({'pos': torch.zeros(1, 512, 4096), 'neg': torch.zeros(1, 512, 4096)}, prompts)
    sample = tmp_path/'sample.pt'
    row = OperationTraceRow(0, 0, 0, str(sample), None, 'test', 9, 0, {}, True, True, request=req)
    trace = FormalTrace(tmp_path/'trace.json', 'a'*64, prompts, 'b'*64, (row,), request=req)
    loader = InputLoader(torch=torch, trace=trace)
    for frames, metadata, ok in ((18,69,True),(21,81,False),(21,69,False),(18,81,False),(18,None,False)):
        value = dict(latents=torch.zeros(16,frames,46,80), prompt_idx=0, steps=40, size='640*368')
        if metadata is not None: value['frames'] = metadata
        torch.save(value, sample)
        from dataclasses import replace
        from distillschedule.workloads.best_known.registry import file_sha256
        checked_row = replace(row,sample_sha256=file_sha256(sample))
        if ok:
            assert tuple(loader.load(checked_row)[0].shape) == (16,18,46,80)
        else:
            with pytest.raises(RuntimeError, match='shape|frames'):
                loader.load(checked_row)


def request_trace(tmp_path):
    """Existing real trace builders over hash-only CPU fixture assets."""
    from distillschedule.search.livewan_request import LiveWanRequest
    root=Path(__file__).resolve().parents[1]
    data=tmp_path/'data';data.mkdir()
    for n in range(80): (data/f'{n:03d}.pt').write_bytes(f'CPU_HASH_ONLY_FIXTURE_{n}'.encode())
    prompts=tmp_path/'prompts.bin';prompts.write_bytes(b'CPU_HASH_ONLY_PROMPTS')
    source=tmp_path/'source.json';target=tmp_path/'trace.json'
    subprocess.run([sys.executable,str(root/'scripts/distillschedule/build_livewan_crossscale_trace.py'),
        '--data',str(data),'--prompts',str(prompts),'--output',str(source)],check=True,capture_output=True)
    subprocess.run([sys.executable,str(root/'scripts/distillschedule/build_livewan_continuous_trace.py'),
        '--source-trace',str(source),'--output',str(target),'--cycles','2'],check=True,capture_output=True)
    value=json.loads(target.read_text());req=LiveWanRequest(frames=69)
    value.update(workload_request=req.to_dict(),workload_request_sha256=fingerprint(req.to_dict()),
        sample_generation=dict(method='gen_teacher',livewan_commit='1eabf43c777bc5b2816926ca998cd4e4c2dfa3d5',frames=69,steps=40,size='1.3B'))
    value.pop('trace_sha256');value['trace_sha256']=fingerprint(value)
    target.write_text(json.dumps(value))
    return target,value


def test_wrong_trace_request_and_source_reuse_fail_closed(tmp_path):
    from distillschedule.search.livewan_request import LiveWanRequest
    from distillschedule.runtime.livewan_crossscale_worker import FormalTrace
    path,value=request_trace(tmp_path);req=LiveWanRequest(frames=69)
    trace=FormalTrace.load(path,expected_cycles=2,request=req)
    assert len(trace.rows)==160 and trace.request==req
    assert {r.new_blocks for r in trace.rows}=={3,4,5}
    with pytest.raises(RuntimeError,match='request/shape'):
        FormalTrace.load(path,expected_cycles=2,request=LiveWanRequest())
    for kind in ('request','source','frames','crop','step','reuse'):
        bad=copy.deepcopy(value)
        if kind=='request': bad.pop('workload_request')
        if kind=='source': bad['sample_generation']['method']='fresh_noise'
        if kind=='frames': bad['sample_generation']['frames']=81
        if kind=='crop': bad['workload_request_sha256']='0'*64
        if kind=='step': bad['sample_generation']['steps']=2
        if kind=='reuse': bad['rows'][80]['world_frames']=9 if bad['rows'][80]['world_frames']!=9 else 3
        bad.pop('trace_sha256');bad['trace_sha256']=fingerprint(bad);path.write_text(json.dumps(bad))
        with pytest.raises((RuntimeError,ValueError)):
            FormalTrace.load(path,expected_cycles=2,request=req)


def test_real_resolver_binds_trace_bootstrap_expansion_and_launcher(tmp_path):
    from distillschedule.search.livewan_request import LiveWanRequest
    from distillschedule.search.execution import resolve_candidate,validated_profile
    from distillschedule.runtime.livewan_launch_plan import checked_launch_plan
    from distillschedule.runtime.livewan_crossscale_worker import FormalTrace,_validate_trace_balanced_pairing
    from distillschedule.workloads.best_known.registry import file_sha256,repository_root
    path,trace=request_trace(tmp_path);req=LiveWanRequest(frames=69)
    source=tmp_path/'source-receipt.json'
    sv=dict(status='PASS',source_root='/cpu/unused/wan21',source_receipt_sha256='a'*64)
    source.write_text(json.dumps(sv))
    boot=tmp_path/'bootstrap.json'
    bv=dict(kind='systems_only_livewan_crossscale_bootstrap',trace=dict(path=str(path),sha256=trace['trace_sha256']),
        source_receipt=dict(path=str(source),sha256=sv['source_receipt_sha256']),critic_lora=dict(seed=2026083001))
    bv['bootstrap_sha256']=fingerprint(bv);boot.write_text(json.dumps(bv))
    profile=LiveWanImplementation('pipeline',request=req).lower_profile()
    assets=dict(schema_version=1,nodes=[dict(host=f'cpu{n}',repo_root=str(repository_root()),python_env='/cpu/venv') for n in (0,1)],
        master_addr='127.0.0.1',master_port=29930,run_root=str(tmp_path/'runs'),node_local_cache_root='/tmp/cpu/cache',
        assets={k:dict(path=f'/cpu/unused/{k}',sha256='b'*64,
            **({'identity_file':'identity.json'} if d['kind']=='directory' else {})) for k,d in profile['required_assets'].items()})
    for k,p in (('trace',path),('source_receipt',source),('bootstrap',boot)):
        assets['assets'][k]=dict(path=str(p),sha256=file_sha256(p))
    config=tmp_path/'assets.json';config.write_text(json.dumps(assets))
    receipt=resolve_candidate(LIVEWAN,LiveWanImplementation('pipeline',request=req).id,assets_path=config,run_id='request-test')
    validated_profile(receipt,root=repository_root())
    native=receipt['nodes'][0]['resolved_native_plan']
    _validate_trace_balanced_pairing(native,FormalTrace.load(path,expected_cycles=2,request=req))
    assert native['workload']['shape']['latent']['frames']==18
    assert max(native['metadata']['current_step_max_new_blocks'])<=10
    assert native['metadata']['request_trace_sha256']==trace['trace_sha256']
    assert receipt['native_plan']['template_file_sha256']!=receipt['nodes'][0]['resolved_native_plan_sha256']
    launch=tmp_path/'launch.json';launch.write_text(json.dumps(native))
    assert checked_launch_plan(launch,sha256=fingerprint(native),request=req)==native
    # Execute the actual launcher selection/render fragment without GPUs or
    # topology probes; even a valid old template cannot replace a saved plan.
    import os
    script=(repository_root()/'scripts/distillschedule/launch_livewan_crossscale_world16_node.sh').read_text()
    fragment=script[script.index('RESOLVED="${RUN_ROOT}/resolved_plan.node'):script.index('# Fail before 14B construction')]
    env={**os.environ,'PYTHON':sys.executable,'RUN_ROOT':str(tmp_path),'NODE_RANK':'0','NODE_LOG':str(tmp_path),
         'DISTILLSCHEDULE_LIVEWAN_REQUEST_ID':req.id,'DISTILLSCHEDULE_LIVEWAN_LAUNCH_PLAN':str(launch),
         'DISTILLSCHEDULE_LIVEWAN_LAUNCH_PLAN_SHA256':fingerprint(native),
         'PLAN':str(repository_root()/profile['runtime']['typed_plan'])}
    result=subprocess.run(['bash','-c','set -eu\n'+fragment],env=env,capture_output=True,text=True)
    assert result.returncode==0,(result.stdout,result.stderr)
    rendered=tmp_path/'resolved_plan.node0.json'
    assert json.loads(rendered.read_text())['workload']['shape']['frames']==18
    rendered.unlink()
    wrong=copy.deepcopy(native);wrong['workload']['shape']['frames']=21
    wrong['workload']['shape']['latent']['frames']=21;launch.write_text(json.dumps(wrong))
    with pytest.raises(ValueError,match='shape'):
        checked_launch_plan(launch,sha256=fingerprint(wrong),request=req)
    env['DISTILLSCHEDULE_LIVEWAN_LAUNCH_PLAN_SHA256']=fingerprint(wrong)
    assert subprocess.run(['bash','-c','set -eu\n'+fragment],env=env,capture_output=True).returncode!=0
    assert not rendered.exists()
    for field in ('native_plan','nodes'):
        stale=copy.deepcopy(receipt)
        if field=='native_plan': stale[field]['template']['workload']['shape']['frames']=21
        else: stale[field][0]['resolved_native_plan']['workload']['shape']['latent']['frames']=21
        with pytest.raises(ValueError):validated_profile(stale,root=repository_root())


def test_actual_pinned_original_geometry_and_sampling_helpers_cpu():
    """Execute exact pure helper bodies; no replacement trainer or fake model."""
    import ast
    from distillschedule.search.livewan_request import LiveWanRequest
    candidates=[Path(__file__).resolve().parents[1]/'third_party/livewan',
        Path('/home/lyy/Helios/DMD-System/worktrees/livewan-crossscale-pipeline-20260830/third_party/livewan')]
    source=next((p for p in candidates if (p/'wanstreamer/core.py').is_file()),None)
    if source is None:pytest.skip('pinned LiveWan checkout is not populated locally')
    assert subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()=='1eabf43c777bc5b2816926ca998cd4e4c2dfa3d5'
    def original_helper(file,name):
        tree=ast.parse((source/file).read_text())
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        namespace={};exec(compile(ast.Module(body=[node],type_ignores=[]),str(source/file),'exec'),namespace)
        return namespace[name]
    geometry=original_helper('wanstreamer/core.py','latent_geometry')
    schedule=original_helper('wanstreamer/stream.py','few_step_schedule')
    req=LiveWanRequest(frames=69)
    assert geometry(640,368)==(46,80,23,40)
    assert schedule(4)==[1.0,.75,.5,.25]
    assert req.attention_tokens==18*geometry(640,368)[2]*geometry(640,368)[3]
    # The real source's clean KV forward is additional to its four steps.
    tree=ast.parse((source/'wanstreamer/stream.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='FewStepStreamer')
    generate=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='generate_block')
    assert 'if not self.noisy_context' in ast.unparse(generate)
    assert [req.source_nograd_calls(w) for w in (3,6,9)]==[26,21,16]
