"""Public metadata/exec contracts. Synthetic owners are NOT native/GPU proof."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_cold_public_help_has_explicit_owner_reproduction_without_torch():
    code = """import sys
from distillschedule.cli import main
try: main(['native-output','--help'])
except SystemExit as exc: assert exc.code==0
assert 'torch' not in sys.modules
"""
    result = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--execute' in result.stdout and '--node' in result.stdout


def test_public_command_is_discoverable_without_changing_old_generate(capsys):
    from distillschedule.cli import main
    with pytest.raises(SystemExit) as stopped:
        main(['--help'])
    assert stopped.value.code == 0
    assert 'native-output' in capsys.readouterr().out


def test_documented_template_is_fail_closed_not_an_execution_recipe(capsys):
    from distillschedule.cli import main
    template = Path(__file__).resolve().parents[1] / 'recipes/h3_ar/native-output.reproduce.template.json'
    assert main(['native-output', str(template)]) == 2
    assert 'file pin required' in capsys.readouterr().err


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pin(path):
    return dict(path=str(path), sha256=sha(path))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))
    return pin(path)


@pytest.fixture
def packet(tmp_path, monkeypatch):
    from distillschedule.training import h3_native_output_reproduction as api
    root = tmp_path / 'SYNTHETIC_TEST_ONLY'
    scripts = root / 'packet/scripts'
    scripts.mkdir(parents=True)
    # Test-only owned child: real subprocess, no model, no physical admission.
    child = scripts / 'native_output_node_owner.py'
    child.write_text("import json,os,sys\nprint(json.dumps(dict(SYNTHETIC_TEST_ONLY=True,argv=sys.argv[1:],"
                     "old_runtime=os.environ['PYTHONPATH'],leak=os.environ.get('LD_PRELOAD'),"
                     "world=os.environ.get('WORLD_SIZE'),GPU_started=0)))\n")
    fixed = {'native_output_node_owner.py': sha(child)}
    for name in ('native_output_packet.py', 'native_fd_binding.py', 'rank_native_output.py'):
        path = scripts / name
        path.write_text('# SYNTHETIC_TEST_ONLY ' + name + '\n')
        fixed[name] = sha(path)
    # Only this new adapter's file-policy constants are fixture-bound. No v2
    # source, owner/training check, golden or production execute flag is changed.
    monkeypatch.setattr(api, 'PACKET_FILES', fixed)
    source = root / 'old-source'
    source.mkdir()
    python = pin(Path(sys.executable).absolute())
    nodes = [dict(node_rank=i, hostname='synthetic-node'+str(i), gpu_uuids=['GPU-'+str(i)+'-'+str(k) for k in range(8)],
                  environment=dict(PATH=os.environ['PATH'], PYTHONPATH=str(source/'src')+':'+str(source),
                                   OMP_NUM_THREADS='4', CUDA_VISIBLE_DEVICES=','.join('GPU-'+str(i)+'-'+str(k) for k in range(8))))
             for i in (0, 1)]
    gates = {str(i): write(root/f'cpu{i}.json', dict(source_head=api.HEAD,returncode=0,
              CUDA_initialized=False, hostname=nodes[i]['hostname'],native_continuation_tests_executed=True)) for i in (0, 1)}
    extensions = {}
    for role in ('scope','generation'):
        path=root/(role+'.py');path.write_text('# SYNTHETIC_TEST_ONLY '+role)
        extensions[role]=pin(path)
    monkeypatch.setattr(api, 'EXTENSIONS', {k:v['sha256']for k,v in extensions.items()})
    prepared=dict(path=str(root/'prepared/manifest.json'),sha256=api.INPUT_SHA)
    full=dict(path=str(root/'raw.pt'),sha256='a'*64)
    composition=write(root/'composition.json',dict(schema_version=api.COMPOSITION,status='COMPLETE_FUNCTIONAL_INPUT_NOT_GENERATED_MEDIA',
        sample_id=api.SAMPLE,scope='FUNCTIONAL_CURRENT_C50_OUTPUT_NOT_HELDOUT_QUALITY',prepared=prepared,retained_full_raw=full,
        original_history=dict(sha256=api.HISTORY_SHA),new_text_features=dict(sha256=api.TEXT_SHA),
        transport=dict(parsed_BF16_content_sha256='b'*64)))
    (root/'_SUCCESS').write_text(composition['sha256']+'\n')
    conditions=write(root/'conditions.json',dict(schema_version='distillschedule.h3_native_parsed_conditions.v1',
        prepared=prepared,full_raw_sidecar=full,conversion_witness=composition,native_parsed_sha256='b'*64))
    scope_gate=write(root/'scope-gate.json',dict(status='PASS_SCOPE_V2_REVIEW_AND_ACTUAL_INPUT_CPU',source_pins=extensions,
        prepared_sha256=api.INPUT_SHA,CUDA_initialized=False,model_constructed=False,GPU_started=0))
    original=dict(source_root=str(source),source_head=api.HEAD,python=python,nodes=nodes,target_cpu_gates=gates,
        owner_helpers={'node_stage.py':write(root/'node_stage.py',dict(SYNTHETIC_TEST_ONLY=True))})
    original_ref=write(root/'packet/configs/original-launch.json',original)
    monkeypatch.setattr(api, 'ORIGINAL_LAUNCH_SHA', original_ref['sha256'])
    run=root/'run';checkpoint=str(root/'checkpoint')
    request=write(root/'native-request.json',dict(schema_version='distillschedule.h3_native_generation_request.v1',kind=api.KIND,
        cycle=50,seed=73,output=str(run/'latents'),historical_training_source_head=api.HEAD,
        extension_source=extensions,checkpoint_manifest=dict(path=checkpoint+'/manifest.json',sha256=api.C50_SHA)))
    component=dict(schema_version=api.PACKET_SCHEMA,kind=api.KIND,status='ROOT_ADMITTED_NATIVE_OUTPUT_ONLY',capture_c0=False,
        reexport=False,checkpoint=checkpoint,checkpoint_manifest_sha256=api.C50_SHA,scope_v2_pins=extensions,
        scope_v2_gate=scope_gate,conditions=conditions,native_request=request,original_launch=original_ref,
        rank_entry=pin(scripts/'rank_native_output.py'),bound_packet_files=[pin(scripts/n) for n in fixed],
        output=str(run/'latents'),owner_output=str(run/'owner'))
    launch={**copy.deepcopy(original), 'run_root':str(run),'master_addr':'192.0.2.1','master_port':29551,'w6_native_output':component}
    launch_ref=write(root/'launch.json',launch)
    public=dict(schema_version=api.SCHEMA,scope=api.SCOPE,launch=launch_ref,
        runtime=dict(python=python,source_root=str(source),source_head=api.HEAD),
        owner=dict(entry=pin(child),original_launch=original_ref,admission=scope_gate,run_root=str(run),
                   nodes=[{k:n[k]for k in ('node_rank','hostname','gpu_uuids')}|dict(target_cpu_gate=gates[str(i)])for i,n in enumerate(nodes)]),
        checkpoint_manifest=dict(path=checkpoint+'/manifest.json',sha256=api.C50_SHA),extensions=extensions,
        conditions=conditions,input_witness=composition)
    public_path=root/'public.json';write(public_path,public)
    return dict(api=api,path=public_path,value=public,launch=launch,root=root)


def test_public_cli_inspects_real_request_without_runtime_import_or_execution(packet,capsys):
    from distillschedule.cli import main
    assert main(['native-output',str(packet['path'])])==0
    value=json.loads(capsys.readouterr().out)
    assert value['status']=='INSPECTED_REPRODUCTION_NOT_OWNER_ADMISSION'
    assert value['public_front_door']['sha256'] == sha(packet['api'].__file__)
    assert len(value['owner_commands'])==2 and value['GPU_started']==0
    assert all('native_output_node_owner.py'in row['argv'][2] for row in value['owner_commands'])
    assert all('--execute'in row['argv'] for row in value['owner_commands'])


@pytest.mark.parametrize('problem',['schema','old_kind','wrong_head','wrong_source','missing_owner','rank_entry_as_owner',
    'missing_peer','wrong_role','missing_cpu_gate','missing_scope_gate','missing_witness','changed_witness','updated_training','bad_pin',
    'wrong_checkpoint','changed_extension','missing_conditions','bad_owner_admission','unsafe_cache_path'])
def test_request_refusal_matrix(packet,problem):
    api=packet['api'];v=copy.deepcopy(packet['value']);cfg=copy.deepcopy(packet['launch'])
    if problem=='schema':v['schema_version']='UNKNOWN'
    if problem=='wrong_head':v['runtime']['source_head']='0'*40
    if problem=='wrong_source':v['runtime']['source_root']=str(packet['root']/'another-source')
    if problem=='missing_owner':v.pop('owner')
    if problem=='rank_entry_as_owner':v['owner']['entry']=cfg['w6_native_output']['rank_entry']
    if problem=='missing_peer':v['owner']['nodes'].pop()
    if problem=='wrong_role':v['owner']['nodes'][1]['node_rank']=0
    if problem=='missing_witness':v.pop('input_witness')
    if problem=='bad_pin':v['launch']['sha256']='0'*64
    if problem=='wrong_checkpoint':v['checkpoint_manifest']['sha256']='0'*64
    if problem=='changed_extension':v['extensions']['generation']['sha256']='0'*64
    if problem=='missing_conditions':v.pop('conditions')
    if problem=='bad_owner_admission':v['owner']['admission']=v['owner']['nodes'][0]['target_cpu_gate']
    if problem=='unsafe_cache_path':cfg['nodes'][0]['environment']['TMPDIR']='/unreviewed/cache'
    if problem=='changed_witness':Path(v['input_witness']['path']).write_text('{}')
    if problem=='old_kind':cfg['w6_native_output']['kind']='INFERENCE_FROM_EXPORTED_WEIGHTS'
    if problem=='updated_training':cfg['w6_native_output']['capture_c0']=True
    if problem=='missing_cpu_gate':cfg['target_cpu_gates'].pop('1')
    if problem=='missing_scope_gate':cfg['w6_native_output']['scope_v2_gate']=None
    if problem in ('old_kind','updated_training','missing_cpu_gate','missing_scope_gate','unsafe_cache_path'):
        v['launch']=write(Path(v['launch']['path']),cfg)
    write(packet['path'],v)
    with pytest.raises((ValueError,KeyError)):api.inspect_request(packet['path'])


def test_execution_requires_node_owner_not_an_arbitrary_torchrun_rank(packet,monkeypatch):
    api=packet['api'];monkeypatch.setenv('WORLD_SIZE','16')
    with pytest.raises(ValueError,match='owner|torchrun'):api.dispatch(packet['path'],execute=True,node=0)
    monkeypatch.delenv('WORLD_SIZE')
    with pytest.raises(ValueError,match='explicit node'):api.dispatch(packet['path'],execute=True)


def test_exec_plan_keeps_old_runtime_and_cannot_target_rank_directly(packet,monkeypatch):
    api=packet['api'];monkeypatch.setattr(api.socket,'gethostname',lambda:'synthetic-node1')
    monkeypatch.setattr(api,'_source_check',lambda root:None) # Synthetic source has no real historical git HEAD.
    monkeypatch.setenv('PYTHONPATH','CURRENT_PACKAGE_MUST_NOT_LEAK')
    monkeypatch.setenv('LD_PRELOAD','CURRENT_PRELOAD_MUST_NOT_LEAK')
    calls=[]
    def capture(executable,argv,environment):calls.append((executable,argv,environment));raise RuntimeError('STOP_BEFORE_OWNER_EXEC')
    monkeypatch.setattr(api.os,'execve',capture)
    with pytest.raises(RuntimeError,match='STOP_BEFORE_OWNER_EXEC'):api.dispatch(packet['path'],execute=True,node=1)
    executable,argv,env=calls[0]
    assert executable==packet['value']['runtime']['python']['path']
    assert Path(argv[2]).name=='native_output_node_owner.py' and argv[-1]=='--execute'
    assert argv[argv.index('--node')+1]=='1' and env['OMP_NUM_THREADS']=='4'
    assert env['PYTHONPATH'].startswith(packet['value']['runtime']['source_root']+'/src:')
    assert 'LD_PRELOAD'not in env and 'WORLD_SIZE'not in env


def test_public_cli_execute_reaches_original_owner_handoff_only(packet,monkeypatch,capsys):
    from distillschedule.cli import main
    api=packet['api'];monkeypatch.setattr(api.socket,'gethostname',lambda:'synthetic-node0')
    monkeypatch.setattr(api,'_source_check',lambda root:None) # Explicit synthetic source, no C50 admission.
    calls=[]
    def stop(executable,argv,environment):
        calls.append(argv)
        raise RuntimeError('SYNTHETIC_STOP_BEFORE_ORIGINAL_OWNER_EXEC')
    monkeypatch.setattr(api.os,'execve',stop)
    assert main(['native-output',str(packet['path']),'--node','0','--execute'])==2
    assert len(calls)==1 and Path(calls[0][2]).name=='native_output_node_owner.py'
    diagnostics=capsys.readouterr().err
    assert 'HANDOFF_NOT_OWNER_SUCCESS' in diagnostics
    assert 'SYNTHETIC_STOP_BEFORE_ORIGINAL_OWNER_EXEC' in diagnostics


def test_real_process_boundary_runs_only_synthetic_owner_not_current_package(packet,monkeypatch):
    api=packet['api'];plan=api.inspect_request(packet['path'])
    # A child subprocess actually consumes the emitted ORIGINAL-OWNER-shaped argv.
    # The child here is explicitly a no-model synthetic owner, not GPU admission.
    env=api.owner_environment(packet['launch'],0,{})
    result=subprocess.run(plan['owner_commands'][0]['argv'],env=env,cwd=packet['root'],text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    child=json.loads(result.stdout)
    assert child['SYNTHETIC_TEST_ONLY']is True and child['GPU_started']==0
    assert child['old_runtime']==env['PYTHONPATH'] and child['world']is None and child['leak']is None


def test_old_public_commands_keep_their_existing_argument_contracts(capsys):
    from distillschedule.cli import main
    for command in ('train','export','generate','decode'):
        with pytest.raises(SystemExit)as stopped:main([command,'--help'])
        assert stopped.value.code==0
        help_text=capsys.readouterr().out
        assert 'native-output'not in help_text
        if command=='generate':assert '--weights'in help_text and '--prepared'in help_text


def test_changed_declared_owner_environment_is_not_a_new_resource_policy(packet):
    cfg=copy.deepcopy(packet['launch'])
    cfg['nodes'][0]['environment']['UNREVIEWED_RUNTIME_SWITCH']='1'
    request=copy.deepcopy(packet['value'])
    request['launch']=write(Path(request['launch']['path']),cfg)
    write(packet['path'],request)
    with pytest.raises(ValueError,match='environment|original node'):
        packet['api'].inspect_request(packet['path'])


def test_owner_safe_inheritance_matches_original_h3_common_environment(packet):
    caller=dict(HOME='/synthetic/home', USER='synthetic', LOGNAME='synthetic', LANG='C.UTF-8',
                LC_ALL='CALLER_NOT_BOUND', TZ='CALLER_NOT_BOUND', PYTHONPATH='CALLER_NOT_BOUND',
                LD_PRELOAD='CALLER_NOT_BOUND', WORLD_SIZE='16')
    result=packet['api'].owner_environment(packet['launch'],0,caller)
    expected={key:caller[key]for key in ('HOME','USER','LOGNAME','LANG')}
    expected.update(packet['launch']['nodes'][0]['environment'])
    assert result==expected


def test_wrong_actual_git_source_head_refuses_before_exec(packet,monkeypatch):
    api=packet['api'];monkeypatch.setattr(api.socket,'gethostname',lambda:'synthetic-node0')
    monkeypatch.setattr(api.os,'execve',lambda *args:pytest.fail('must not exec owner with a fake historical tree'))
    with pytest.raises(ValueError,match='historical|source'):
        api.dispatch(packet['path'],execute=True,node=0)
