"""Synthetic CPU R9 receipts / mocked transport; never GPU or timing evidence."""
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import subprocess

import pytest

from distillschedule.search import r9_contract as contract, r9_native, r9_transport, execution
from distillschedule.search.h3_implementation import H3, H3Implementation
from distillschedule.search.h3_request import H3Request
from distillschedule.workloads.best_known import registry, dispatch

LOW = H3Request(480, 832)
CPU = H3Implementation(request=LOW, cache_policy='cpu_staged_trim')
CUDA = replace(CPU, cache_policy='cuda_no_trim')


def write(path, value):
    registry.write_new_json(path, value)
    return r9_native.binding(path)


def fixture(tmp_path, monkeypatch, *, impl=CUDA, mode='correctness', run_id='CPU_FIXTURE_ONLY'):
    profile = impl.lower_profile()
    source = registry.repository_root()
    rows = []
    nodes = []
    for node in (0, 1):
        overlays = []
        for package in ('torch', 'transformers'):
            path = tmp_path / f'n{node}' / (package + '-overlay')
            target = path / package
            target.mkdir(parents=True)
            (target / '__init__.py').write_text('CPU_FIXTURE_NOT_AN_IMPORTABLE_MODEL')
            overlays.append(dict(path=str(path), identity_file=package + '/__init__.py',
                sha256=registry.file_sha256(target / '__init__.py'),
                files=[dict(name=package, symlink=False, target=str(target))]))
        rows.append(dict(node=node, overlays=overlays))
        nodes.append(dict(host=f'fixture-host{node}', repo_root=str(source), python_env=sys.prefix,
            python_path_overlays=[{k:r[k] for k in ('path','identity_file','sha256')} for r in overlays]))
    assets = {}
    for name, definition in profile['required_assets'].items():
        path = tmp_path / 'assets' / name
        if definition['kind'] == 'directory':
            path.mkdir(parents=True)
            anchor = path / 'CPU_ONLY_IDENTITY'; anchor.write_text(name)
            assets[name] = dict(path=str(path), identity_file=anchor.name, sha256=registry.file_sha256(anchor))
        else:
            path.parent.mkdir(parents=True, exist_ok=True); path.write_text(name)
            assets[name] = dict(path=str(path), sha256=registry.file_sha256(path))
    a = dict(schema_version=1, nodes=nodes, master_addr='127.0.0.1', master_port=30589,
             run_root=str(tmp_path / 'runs'), node_local_cache_root='/tmp/distillschedule-cpu-fixture', assets=assets)
    if mode != 'correctness':
        a['correctness'] = write(tmp_path / 'prior-gate.json', dict(fixture='CPU_ONLY'))
    assets_path = tmp_path / 'assets.json'; write(assets_path, a)
    plan = execution.resolve_candidate(H3, impl.id, assets_path=assets_path, run_id=run_id, mode=mode)
    # Missing native upstream/model assets are the only preflight stub. All
    # real lowering/schema, wrapper/source/plan/env/dependency hashes run.
    monkeypatch.setattr(dispatch, 'validate_local', lambda *a, **k: {'fixture': 'CPU_ONLY'})
    owner = tmp_path / 'owner.py'; owner.write_text('CPU_FIXTURE_OWNER_NEVER_EXECUTED')
    # Cannot redistribute a site owner's absolute paths into library fixtures.
    # Gate this exact inert fixture owner only within this CPU test context.
    digest = registry.file_sha256(owner)
    monkeypatch.setattr(contract, 'OWNER_SHA256', digest)
    monkeypatch.setattr(r9_native, 'OWNER_SHA256', digest)
    monkeypatch.setattr(r9_transport, 'OWNER_SHA256', digest)
    helper_pins = {}
    for name in ('native_worker_oom_guard_r1.py','verified_zombie_fd_rescan_r1.py','historical_ssh_read_only_r2.py'):
        path = tmp_path / 'safety-fixture' / name; path.parent.mkdir(exist_ok=True)
        path.write_text('CPU_FIXTURE_NOT_A_SAFETY_IMPLEMENTATION')
        helper_pins[str(path)] = registry.file_sha256(path)
    donor = dict(source_root='/old/source', packet_root='/old/packet', file_sha256=helper_pins,
        owned_worker_oom_fail_fast=True, verified_zombie_fd_rescan=True, cycles=2, profile=False,
        numerical_status='GRID1000_REVIEW_POLICY_ONLY', expected_final_versions=dict(generator=2,critic=10,ema=0),
        hosts=['fixture0','fixture1'], ordered_UUIDs_by_node=[[f'CPU-FIXTURE-{n}-{d}' for d in range(8)] for n in (0,1)],
        work_seconds=1200, lease_seconds=1500,
        expected_runtime={k:'CPU_FIXTURE_ONLY' for k in ('torch','cuda','diffusers','peft','transformers')})
    spec = dict(schema_version=contract.SCHEMA, owner=r9_native.binding(owner),
        template=write(tmp_path/'template.json', donor), dependency_provenance=write(tmp_path/'dependencies.json', rows))
    plan['execution_transport'] = dict(schema_version=contract.SCHEMA, deployment_sha256=registry.fingerprint(spec))
    prepared = r9_native.prepare(plan, spec)
    sealed = seal_cpu_fixture(prepared['draft'])
    return plan, sealed['configuration'], sealed['manifest'], spec


def seal_cpu_fixture(draft):
    cfg = r9_native.read(draft)
    for node in (0,1):
        write(Path(cfg['qualification'][f'source_cpu_{node}']['path']), dict(
            status='PASS_H3_SELECTED_PLAN_CPU_NO_GPU', node=node, source_head=cfg['source_head'],
            candidate_id=cfg['candidate_id'], selected_plan_sha256=cfg['selected_plan']['sha256'],
            qualified_files_sha256=cfg['file_sha256'], fixture_scope='CPU_SYNTHETIC_NOT_QUALIFICATION',
            toolchain=dict(status='PASS_CPU_COMPILE_LINK_EXECUTE_NO_CUDA',path='/usr/bin:/bin',
                tools={'gcc':{},'ld':{}},compile_returncode=0,execute_returncode=0)))
    sealed = r9_native.seal(draft)
    return sealed


def populate(plan, cfg, manifest, *, calibration=False):
    from test_refactor_h3_search_native import _receipts, _with_rng
    from test_refactor_native_macro_calibration import _calibration
    root = Path(cfg['run_root'])
    for node in (0,1):
        env = dict(cfg['environment_by_node'][node], DISTILLSCHEDULE_EXPECTED_SOURCE_RECEIPT=cfg['source_head'],
            DMD_SELECTED_PLAN_MANIFEST_SHA256=manifest['sha256'],
            CUDA_VISIBLE_DEVICES=','.join(cfg['ordered_UUIDs_by_node'][node]))
        argv = [cfg['python'],'-m','torch.distributed.run','--nnodes=2','--nproc-per-node=8','--max-restarts=0',
            f'--node-rank={node}','--master-addr='+cfg['master_addr'],'--master-port='+str(cfg['port']),cfg['worker_script']]
        write(root/f'node_{node}/lifecycle.json',dict(status='PASS_LOCAL_OWNED_RUN_PENDING_WORLD_GATE',
            cleanup_pass=True,lock_held=True,node=node,run_id=cfg['run_id'],source_head=cfg['source_head'],
            manifest_sha256=manifest['sha256'],oom_delta=dict(oom=0,oom_kill=0), held_seconds=10,
            initial_FD_audit=dict(handles=[]),terminal_FD_audit=dict(handles=[]),
            phases=[dict(argv=argv,worker_rc=0)],environment=env,fixture_scope='CPU_ONLY'))
        samples=[dict(cgroup_events=dict(oom=0,oom_kill=0),cgroup_current_bytes=1000,
            gpus=[dict(device=d,uuid=cfg['ordered_UUIDs_by_node'][node][d],total_bytes=4096,free_bytes=3000)
                  for d in range(8)]) for _ in range(2)]
        write(root/f'search-resources.node{node}.json',dict(status='PASS',node_rank=node,
            run_id=cfg['run_id'],samples=samples,fixture_scope='CPU_ONLY'))
    impl = H3Implementation.parse(cfg['candidate_id'])
    for row in _with_rng(_receipts(impl)):
        row['rng_source_binding']['integration_head']=cfg['source_head']
        rank=row['rank'];row.update(integration_head=cfg['source_head'],run_kind=plan['mode'],calibration_only=calibration)
        row['source_environment']=dict(plan['nodes'][rank//8]['environment'])
        for cycle in row['cycles']:
            for key in ('memory_before','memory_after'):
                cycle[key]=dict(hbm=dict(peak_allocated_bytes=500,peak_reserved_bytes=600))
        if calibration:
            row['stage_calibration']=_calibration(H3,impl,rank)
            row['stage_calibration']['source_head']=cfg['source_head']
        write(root/f'execution/ranks/rank_{rank:03d}.json',row)
        write(root/f'selected-plan-binding/rank_{rank:03d}.json',dict(rank=rank,source_head=cfg['source_head'],
            selected_plan=cfg['selected_plan'],candidate_id=cfg['candidate_id'],manifest_sha256=manifest['sha256'],
            mode=plan['mode'],selection=cfg['selection'],entrypoint='ORIGINAL_INTEGRATED_ENTRYPOINT'))


@pytest.mark.parametrize('impl', [CPU,CUDA])
@pytest.mark.parametrize('mode', ['correctness','calibration','diagnostic-timing'])
def test_real_plan_prepare_seal_and_actual_r9_collection(tmp_path,monkeypatch,impl,mode):
    plan,cfg,manifest,_=fixture(tmp_path,monkeypatch,impl=impl,mode=mode)
    populate(plan,cfg,manifest,calibration=mode=='calibration')
    result=r9_native.collect(manifest)
    assert result['status']=='PASS_NATIVE_R9_EXECUTION'
    assert result['selection']=='REGISTERED_CANDIDATE'
    sources=r9_native.verify_execution(plan)
    assert sources and not list(Path(cfg['run_root']).rglob('launch_manifest.json'))
    assert not list(Path(cfg['run_root']).rglob('teardown_status.txt'))
    from distillschedule.search.h3_native_bridge import _checked_native
    assert _checked_native(plan)[1]['calibration_only']==(mode=='calibration')


@pytest.mark.parametrize('change',['argv','oom','lock','fd','cleanup','source','manifest','env','env_extra',
    'rank_missing','rank_candidate','rank_cache','rank_mode','resources_uuid','resources_oom','resources_missing'])
def test_r9_actual_evidence_fail_closed(tmp_path,monkeypatch,change):
    plan,cfg,manifest,_=fixture(tmp_path,monkeypatch)
    populate(plan,cfg,manifest)
    root=Path(cfg['run_root']);path=root/'node_1/lifecycle.json';row=json.loads(path.read_text())
    if change=='argv':row['phases'][0]['argv'].insert(0,'nsys')
    if change=='oom':row['oom_delta']['oom']=1
    if change=='lock':row['lock_held']=False
    if change=='fd':row['terminal_FD_audit']['handles']=[{'foreign':'CPU_FIXTURE'}]
    if change=='cleanup':row['cleanup_pass']=False
    if change=='source':row['source_head']='a'*40
    if change=='manifest':row['manifest_sha256']='a'*64
    if change=='env':row['environment']['PATH']=''
    if change=='env_extra':row['environment']['LD_PRELOAD']='/injected'
    path.write_text(json.dumps(row))
    if change.startswith('rank'):
        path=root/'execution/ranks/rank_015.json';row=json.loads(path.read_text())
        if change=='rank_missing':path.unlink()
        else:
            if change=='rank_candidate':row['candidate_id']=CPU.id
            if change=='rank_cache':row['cycles'][1]['fa3_replay_cache_registration']['storage']='cpu_staged'
            if change=='rank_mode':row['run_kind']='calibration'
            path.write_text(json.dumps(row))
    if change.startswith('resources'):
        path=root/'search-resources.node1.json';row=json.loads(path.read_text())
        if change=='resources_missing':path.unlink()
        else:
            if change=='resources_uuid':row['samples'][1]['gpus'][0]['uuid']='FOREIGN'
            if change=='resources_oom':row['samples'][1]['cgroup_events']['oom']=1
            path.write_text(json.dumps(row))
    with pytest.raises((ValueError,OSError)):
        r9_native.collect(manifest)
    assert not (root/'r9-execution.json').exists()


def test_rehashed_or_dropped_execution_evidence_is_rechecked(tmp_path,monkeypatch):
    plan,cfg,manifest,_=fixture(tmp_path,monkeypatch)
    populate(plan,cfg,manifest);r9_native.collect(manifest)
    path=Path(cfg['run_root'])/'r9-execution.json';row=json.loads(path.read_text())
    row['evidence']=row['evidence'][:1];path.write_text(json.dumps(row))
    with pytest.raises(ValueError,match='differs from actual'):
        r9_native.verify_execution(plan)


def test_no_legacy_fallback_for_missing_r9_execution_proof(tmp_path,monkeypatch):
    plan,cfg,manifest,_=fixture(tmp_path,monkeypatch)
    populate(plan,cfg,manifest);r9_native.collect(manifest)
    (Path(cfg['run_root'])/'r9-execution.json').unlink()
    from distillschedule.search.h3_native_bridge import _checked_native
    with pytest.raises(FileNotFoundError):_checked_native(plan)


@pytest.mark.parametrize('change',['wrapper','source','mode','policy','shape','toolchain'])
def test_selected_runtime_contract_not_overridden(tmp_path,monkeypatch,change):
    plan,cfg,manifest,_=fixture(tmp_path,monkeypatch)
    if change=='wrapper':Path(cfg['worker_script']).write_text('raise SystemExit(0)')
    if change=='source':cfg['source_head']='a'*40
    if change=='mode':cfg['environment_by_node'][0]['DISTILLSCHEDULE_RUN_KIND']='unknown'
    if change=='policy':cfg['environment_by_node'][0]['H3_FA3_REPLAY_CACHE_STORAGE']='cpu_staged'
    if change=='shape':cfg['environment_by_node'][0]['H3_DMD_HEIGHT']='768'
    if change=='toolchain':cfg['environment_by_node'][0]['PATH']=''
    with pytest.raises(ValueError):contract.validate_config(cfg)


def test_single_execution_numbered_admission_and_no_concurrent_rpc(tmp_path,monkeypatch):
    plan,cfg,manifest,_=fixture(tmp_path,monkeypatch)
    calls=[]
    class Child:
        def wait(self,**kwargs):calls.append('EOF');return 0
    def popen(command,**kwargs):calls.append(command);return Child()
    monkeypatch.setattr(r9_transport, 'git_head', lambda _: cfg['source_head'])
    monkeypatch.setattr(r9_transport.subprocess,'Popen',popen)
    output=tmp_path/'controller'
    r9_transport.launch_owners(cfg,manifest,output=output,execute=False,attempt=1)
    assert [x if x=='EOF' else 'SSH' for x in calls]==['SSH','SSH','EOF','EOF']
    r9_transport.launch_owners(cfg,manifest,output=output,execute=True,attempt=1)
    r9_transport.launch_owners(cfg,manifest,output=output,execute=False,attempt=2)
    with pytest.raises(FileExistsError):
        r9_transport.launch_owners(cfg,manifest,output=output,execute=True,attempt=2)
    assert (output/'owner-plan-attempt001-eof.json').exists()
    assert (output/'owner-plan-attempt002-eof.json').exists()


def test_failed_admission_preserved_and_never_starts_gpu(tmp_path,monkeypatch):
    _,cfg,manifest,_=fixture(tmp_path,monkeypatch)
    class Child:
        def wait(self,**kwargs):return 2
    monkeypatch.setattr(r9_transport, 'git_head', lambda _: cfg['source_head'])
    monkeypatch.setattr(r9_transport.subprocess,'Popen',lambda *a,**k:Child())
    output=tmp_path/'controller'
    with pytest.raises(ValueError,match='no automatic retry'):
        r9_transport.launch_owners(cfg,manifest,output=output,execute=False,attempt=1)
    with pytest.raises(ValueError,match='successful admission'):
        r9_transport.launch_owners(cfg,manifest,output=output,execute=True,attempt=1)
    assert not (output/'owner-execute-started.json').exists()


def test_interrupted_owner_unknown_terminal_state_forbids_new_sessions(tmp_path,monkeypatch):
    _,cfg,manifest,_=fixture(tmp_path,monkeypatch)
    output=tmp_path/'controller'
    output.mkdir()
    write(output/'owner-execute-started.json',dict(status='STARTED_NOT_RESULT'))
    monkeypatch.setattr(r9_transport.subprocess,'Popen',lambda *a,**k:pytest.fail('no extra SSH during unknown lease'))
    monkeypatch.setattr(r9_transport,'git_head',lambda _:cfg['source_head'])
    with pytest.raises(ValueError,match='terminal state unknown'):
        r9_transport.launch_owners(cfg,manifest,output=output,execute=False,attempt=2)


def test_synthetic_r9_correctness_calibration_search_selected_execution_chain(tmp_path,monkeypatch):
    """Real CPU parsers/assembler/search/lowering; all GPU rows are synthetic."""
    from distillschedule.search.h3_native_bridge import prepare_pair
    from distillschedule.search.native_macro_calibration import assemble_native, assemble_calibrations
    from distillschedule.search.native_macro_adapter import precision_contract, lower_prediction
    from distillschedule.search.runtime_identity import observed_binding
    from distillschedule.search.model_search import search_plans
    from test_refactor_native_macro_calibration import _observations
    cpu,cfg,manifest,spec=fixture(tmp_path,monkeypatch,impl=CPU,run_id='CPU_SYNTHETIC_GATE')
    populate(cpu,cfg,manifest);r9_native.collect(manifest)
    marker=cpu['execution_transport']
    def run_synthetic(impl,mode,run_id,assets):
        path=tmp_path/(run_id+'-assets.json');write(path,assets)
        plan=execution.resolve_candidate(H3,impl.id,assets_path=path,run_id=run_id,mode=mode)
        plan['execution_transport']=marker
        sealed=seal_cpu_fixture(r9_native.prepare(plan,spec)['draft'])
        populate(plan,sealed['configuration'],sealed['manifest'],calibration=mode=='calibration')
        if mode=='calibration' and impl==CPU:
            for path in Path(sealed['configuration']['run_root']).glob('execution/ranks/*.json'):
                row=json.loads(path.read_text())
                for stage in row['stage_calibration']['rows']:
                    for key in ('start_ns','stop_ns','duration_ns'):stage[key]*=2
                path.write_text(json.dumps(row))
        r9_native.collect(sealed['manifest'])
        return plan,sealed
    gpu,_=run_synthetic(CUDA,'correctness','CUDA_SYNTHETIC_GATE',cpu['asset_config'])
    pair=prepare_pair(cpu,gpu)
    observations=_observations()
    for row in observations:row['source_head']=cpu['integration_head']
    identity=dict(source_head=cpu['integration_head'],asset_fingerprint=cpu['asset_fingerprint'],
        semantic_fingerprint=cpu['workload_semantics_hash'],
        precision_fingerprint=registry.fingerprint(precision_contract(H3,h3_request=LOW)),
        **observed_binding(observations,source_head=cpu['integration_head']))
    results=[]
    for impl in (CPU,CUDA):
        plan,_=run_synthetic(impl,'calibration','CAL_'+impl.storage,pair[impl.id]['assets'])
        result=assemble_native(plan,pair[impl.id],identity,dict(before=observations,after=observations))
        path=tmp_path/(impl.storage+'-result.json');write(path,result);results.append(path)
    problem,costs=assemble_calibrations(results,output=tmp_path/'joined',expected_head=cpu['integration_head'])
    prediction=search_plans(problem,costs,exhaustive=True,max_proposals=16)
    assert len(prediction['ranked'])==2 and prediction['rejections']
    assets=tmp_path/'selected-assets.json';write(assets,pair[CUDA.id]['assets'])
    selected=lower_prediction(prediction,problem,costs,run_id='CPU_SYNTHETIC_SEARCH_SELECTED',assets_path=assets)
    assert selected['search_candidate']['id']==CUDA.id and 'model_plan' in selected
    selected['execution_transport']=marker
    # Round trip the real saved searched plan; no operator-selected substitute.
    selected=json.loads(json.dumps(selected))
    # Controller sees an intact selected model object, but cannot read remote
    # raw costs. Full validation/rebinding happens on the receipt host.
    from distillschedule.search import native
    requests=[]
    def resolve_on_receipt_host(payload):
        requests.append(payload)
        assert payload['action']=='resolve'
        path=tmp_path/'receipt-host-assets.json';write(path,payload['assets'])
        return execution.resolve_candidate(H3,payload['candidate_id'],assets_path=path,
            run_id=payload['run_id'],mode=payload['mode'])
    monkeypatch.setattr(native,'handle_request',resolve_on_receipt_host)
    rebound=r9_native.rebind(selected,run_id='CPU_SYNTHETIC_REBOUND',assets=selected['asset_config'])
    assert rebound['model_plan']==selected['model_plan']
    assert rebound['execution_transport']==marker and len(requests)==1
    assert registry.fingerprint(rebound['model_plan'])==registry.fingerprint(selected['model_plan'])
    sealed=seal_cpu_fixture(r9_native.prepare(selected,spec)['draft'])
    assert sealed['configuration']['selection']=='MEASURED_MODEL_SELECTED'
    populate(selected,sealed['configuration'],sealed['manifest'])
    result=r9_native.collect(sealed['manifest'])
    assert result['selection']=='MEASURED_MODEL_SELECTED'
    r9_native.verify_execution(selected)
    assert not list(tmp_path.rglob('launch_manifest.json'))
    assert not list(tmp_path.rglob('teardown_status.txt'))
    # Invalid remote cost bytes are not silently accepted by portable dry-run.
    source=Path(results[0]);source.write_text(source.read_text()+'\n')
    contract.validate_controller_envelope(rebound)
    with pytest.raises(ValueError):r9_native.prepare(rebound,spec)
    with pytest.raises(ValueError):
        r9_native.rebind(rebound,run_id='CPU_SYNTHETIC_BAD_COST',assets=rebound['asset_config'])
    assert len(requests)==1
    assert not (Path(rebound['asset_config']['run_root'])/rebound['run_id']).exists()


def test_controller_calls_rpc_only_before_or_after_both_owner_phases(tmp_path,monkeypatch):
    plan,cfg,manifest,spec=fixture(tmp_path,monkeypatch)
    plan.pop('execution_transport')
    calls=[]
    def rpc(**payload):
        calls.append(payload['action'])
        if payload['action']=='r9_prepare':return {'draft':{'path':'CPU_ONLY','sha256':'a'*64}}
        if payload['action']=='r9_seal':return dict(configuration=cfg,manifest=manifest)
        return {'status':'CPU_FIXTURE_NOT_EXECUTED'}
    monkeypatch.setattr(execution,'native_rpc',rpc)
    monkeypatch.setattr(r9_transport,'launch_owners',lambda *a,execute,**k:calls.append('OWNER_EXECUTE_EOF' if execute else 'OWNER_PLAN_EOF'))
    assert r9_transport.run_r9_plan(plan,spec,output=tmp_path/'mock-controller')==0
    assert calls==['r9_prepare','r9_qualify','r9_qualify','r9_seal','OWNER_PLAN_EOF','OWNER_EXECUTE_EOF','r9_collect']
    assert plan['execution_transport']['deployment_sha256']==registry.fingerprint(spec)


@pytest.mark.parametrize('change',['source','runtime','profile','marker','extra_deployment','owner'])
def test_r9_rpc_envelope_and_deployment_negative(tmp_path,monkeypatch,change):
    plan,cfg,manifest,spec=fixture(tmp_path,monkeypatch)
    if change=='source':plan['integration_head']='a'*40
    if change=='runtime':plan['nodes'][0]['environment']['H3_PLANNER_CANDIDATE_ID']=CPU.id
    if change=='profile':plan['fully_resolved_profile']['runtime']['CPU_TAMPER']=True
    if change=='marker':plan['execution_transport']['deployment_sha256']='a'*64
    if change=='extra_deployment':spec['inherited_pythonpath']='/experiment/scripts'
    if change=='owner':spec['owner']['sha256']='a'*64
    monkeypatch.setattr(execution,'native_rpc',lambda **kwargs:pytest.fail('must reject before RPC'))
    with pytest.raises((ValueError,KeyError)):
        execution.run_plan(plan,transport_config=spec,coordinator_output=tmp_path/'should-not-exist')
    assert not (tmp_path/'should-not-exist').exists()


def test_explicit_transport_never_falls_back_to_shell_dispatch(tmp_path,monkeypatch):
    plan,_,_,_=fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(dispatch,'run_resolved',lambda *a,**k:pytest.fail('legacy dispatch forbidden'))
    with pytest.raises(ValueError,match='fall back'):
        execution.run_plan(plan)
    with pytest.raises(ValueError,match='matching replay'):
        execution.rebind_plan(plan,run_id='new',coordinator_output=tmp_path/'legacy')


def test_rpc_transport_assets_and_resolve_payload_assets_are_distinct(tmp_path,monkeypatch):
    """Exercise the real NativeTrialRunner RPC envelope, not a stubbed _rpc."""
    from types import SimpleNamespace
    plan,_,_,_=fixture(tmp_path,monkeypatch)
    runner=execution.NativeTrialRunner.__new__(execution.NativeTrialRunner)
    runner.assets=plan['asset_config'];runner.identity={'source_head':plan['integration_head']};runner.output=tmp_path
    payload_assets=copy.deepcopy(runner.assets);payload_assets['run_root']+='/payload'
    def run(command,**kwargs):
        payload=json.loads(kwargs['input'])
        assert command[0]=='ssh' and runner.assets['nodes'][0]['host'] in command
        assert payload['assets']==payload_assets and payload['action']=='resolve'
        return SimpleNamespace(returncode=0,stdout='{"CPU_FIXTURE_ONLY": true}',stderr='')
    monkeypatch.setattr(subprocess,'run',run)
    assert runner._rpc('resolve',assets=payload_assets)=={'CPU_FIXTURE_ONLY':True}


def test_rebind_controller_preserves_selected_json_and_rejects_rpc_mutation(tmp_path,monkeypatch):
    plan,_,_,_=fixture(tmp_path,monkeypatch)
    # Deliberately unavailable receipt-host data: controller must not attempt
    # local cost reads or claim the object has passed cost validation.
    plan['model_plan']={'remote_cost_source':'CPU_FIXTURE_UNAVAILABLE'}
    result=execution.resolve_candidate(H3,CUDA.id,assets_path=tmp_path/'assets.json',run_id='NEW_CPU_ONLY')
    result.update(model_plan=copy.deepcopy(plan['model_plan']),execution_transport=plan['execution_transport'])
    def rpc(**kwargs):
        assert kwargs['action']=='r9_rebind' and kwargs['receipt']==plan
        return result
    monkeypatch.setattr(execution,'native_rpc',rpc)
    rebound=r9_transport.rebind_r9_plan(plan,run_id=result['run_id'],coordinator_output=tmp_path/'replay')
    assert rebound['model_plan']==plan['model_plan']
    assert json.loads((tmp_path/'replay/execution-plan.json').read_text())==rebound
    result['model_plan']={'changed':True}
    with pytest.raises(ValueError,match='changed the selected'):
        r9_transport.rebind_r9_plan(plan,run_id=result['run_id'],coordinator_output=tmp_path/'bad')
