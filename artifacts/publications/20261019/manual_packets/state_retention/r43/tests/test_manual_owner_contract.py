"""SYNTHETIC_TEST_ONLY producer receipts; no native speed or qualification."""
from pathlib import Path
import importlib.util,json,sys
import pytest
R=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('manual_external_controller',R/'scripts/native_r2/h3_ar_match_controller.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
from test_strategy_h3_ar import calibration_rows,POLICY


def fixture(tmp_path,monkeypatch,phase):
    stage=tmp_path/phase;stage.mkdir()
    originals=calibration_rows(stage,'zero3-ap',100)
    task=tmp_path/'task.json';policy=tmp_path/'policy.json';cfgpath=tmp_path/'cfg.json'
    cfgpath.write_text('{"test_scope":"SYNTHETIC_TEST_ONLY"}')
    cfg=dict(run_root=str(tmp_path),task=c.ref(task)if task.exists()else None,
             capacity_policy=None,nodes=[dict(hostname='cpu0'),dict(hostname='cpu1')],
             _config_path=str(cfgpath))
    for rank,path in enumerate(originals):
        row=json.loads(path.read_text());row.update(schema_version='distillschedule.manual_h3_ar_execution.v1',
            mode='manual',manual_policy_id='off'if phase=='manual-control'else'uncached-rollout-fa3-varlen-v1',
            weights_merged=False,automatic_planning_eligible=False,plan_sha256=None,selected_order=['native_update'])
        row['runtime']=dict(torch='2.10.0+cu128',cuda='12.8')
        row['hardware']['hostname']='cpu'+str(rank//8)
        for cycle in row['cycles']:cycle['leaf_runtime']=dict(leaf_blocks=[dict(logical_parameter_numel=657499392)for _ in range(50)])
        if phase=='manual-child':
            for i,cycle in enumerate(row['cycles'],1):
                cycle['backend_effects']['manual_uncached_rollout_attention']=dict(policy_id='uncached-rollout-fa3-varlen-v1',
                    per_cycle_counts=dict(main_forward_calls=700,application_predict_calls=12,actual_FA3_varlen_calls=350,
                        main_native_fallback=350,refiner_native_fallback=24),heads=56,head_dim=128,
                    grad_and_selected_cache_paths_unchanged=True,params_relocated=False,weights_merged=False,
                    hot_path_host_value_reads=0,bounds=[0,34876,34880])
        target=stage/f'rank_{rank:03d}.json';target.write_text(json.dumps(row))
        path.unlink()
    task.write_text(json.dumps(row['task']));policy.write_text(json.dumps(POLICY))
    cfg.update(task=c.ref(task),capacity_policy=c.ref(policy))
    monkeypatch.setattr(c,'gate_admission',lambda cfg,digest:dict(identity=row['identity']))
    return cfg


@pytest.mark.parametrize('phase',['manual-control','manual-child'])
def test_actual_rank_consumer_accepts_only_separate_complete_manual_receipts(tmp_path,monkeypatch,phase):
    cfg=fixture(tmp_path,monkeypatch,phase)
    assert len(c.validate_ranks(cfg,phase))==16
    assert [r[1]['rank']for r in c.validate_ranks(cfg,phase,node=1)]==list(range(8,16))


@pytest.mark.parametrize('change',['wrong_schema','wrong_policy','merged','missing_cycle','wrong_census',
    'bad_FA3_count','missing_backend','relocated_parameters','changed_grad_path','planner_flag'])
def test_actual_consumer_rejects_bad_manual_or_bank_evidence(tmp_path,monkeypatch,change):
    cfg=fixture(tmp_path,monkeypatch,'manual-child');path=Path(cfg['run_root'])/'manual-child/rank_000.json'
    row=json.loads(path.read_text())
    if change=='wrong_schema':row['schema_version']='distillschedule.original_h3_ar_execution.v1'
    if change=='wrong_policy':row['manual_policy_id']='off'
    if change=='merged':row['weights_merged']=True
    if change=='missing_cycle':row['cycles'].pop()
    if change=='wrong_census':row['cycles'][1]['census']['student']=8
    if change=='bad_FA3_count':row['cycles'][1]['backend_effects']['manual_uncached_rollout_attention']['per_cycle_counts']['actual_FA3_varlen_calls']=349
    if change=='missing_backend':del row['cycles'][1]['backend_effects']['manual_uncached_rollout_attention']
    if change=='relocated_parameters':row['cycles'][1]['backend_effects']['manual_uncached_rollout_attention']['params_relocated']=True
    if change=='changed_grad_path':row['cycles'][1]['backend_effects']['manual_uncached_rollout_attention']['grad_and_selected_cache_paths_unchanged']=False
    if change=='planner_flag':row['automatic_planning_eligible']=True
    path.write_text(json.dumps(row))
    with pytest.raises((RuntimeError,ValueError,KeyError)):c.validate_ranks(cfg,'manual-child')


def test_real_command_routes_two_manual_arms_to_same_native_implementation():
    cfg=dict(source='/not-used/SYNTHETIC_TEST_ONLY',run_root='/not-used/SYNTHETIC_TEST_ONLY',
        nodes=[dict(python=dict(path='/validated/python'))for _ in range(2)],master_addr='127.0.0.1',
        master_ports={'manual-control':33711,'manual-child':33712},task=dict(path='/test-task'),capacity_policy=dict(path='/test-policy'))
    for phase,policy in [('manual-control','off'),('manual-child','uncached-rollout-fa3-varlen-v1')]:
        command=c.gpu_command(cfg,0,phase)
        assert command[0]=='/validated/python'
        assert '--execute'in command and ('--fa3-kernel-root'in command)==(phase=='manual-child')
        assert command[command.index('--manual-policy')+1]==policy
        assert c.implementation(cfg,phase)=='zero3-ap'
        assert '--plan'not in command and '--implementation'not in command
    import inspect
    assert 'manual_policy_id=a.manual_policy' in(R/'scripts/native_r2/manual_bank_worker.py').read_text()
    assert 'len(used)<2'in inspect.getsource(c.check_sequence)
