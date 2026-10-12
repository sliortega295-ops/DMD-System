from dataclasses import replace
from types import SimpleNamespace as NS
import json
import pytest
from distillschedule.strategies.wan import OriginalWanTask as Task,OriginalWanStrategy as Strategy
from distillschedule.strategies.wan_native import WanAssets,apply_environment,build_engines,validate_build,reserve_trial


def assets():
    return WanAssets('model','bootstrap','a'*64,'trace','b'*64,'negative')

@pytest.mark.parametrize('s',[Strategy(),Strategy('resident',512,False),Strategy('resident',2048,True)])
def test_original_factory_receives_selected_placement_before_construction(monkeypatch,s):
    env={};monkeypatch.setattr('distillschedule.strategies.wan_native.os.environ',env)
    t=Task(fake_updates=2);calls=[]
    def factory(**kw):
        assert env['DISTILLSCHEDULE_SEARCH_CANDIDATE_ID']==s.implementation.id
        assert kw['args'].placement==s.implementation.native_placement
        assert kw['bootstrap']=='bootstrap-instance'
        assert kw['device']=='native-device'
        calls.append(kw)
        return ('original-G','original-C','original-EMA',{'original':True})
    apply_environment(t,s)
    result=build_engines(t,s,assets(),'bootstrap-instance','native-device',worker=NS(base=NS(_build_engines=factory)))
    assert len(calls)==1 and result[:3]==('original-G','original-C','original-EMA')
    assert env.get('NCCL_ALGO')==('Tree' if s.placement=='resident' else None)
    assert 'DISTILLSCHEDULE_WAN_ROLLOUT_WINDOW' not in env


def test_no_native_construction_if_strategy_is_not_bound(monkeypatch):
    monkeypatch.setattr('distillschedule.strategies.wan_native.os.environ',{})
    with pytest.raises(ValueError,match='before model construction'):
        build_engines(Task(),Strategy(),assets(),None,None,worker=NS())


def test_unrelated_optimization_override_rejected(monkeypatch):
    monkeypatch.setattr('distillschedule.strategies.wan_native.os.environ',{'DISTILLSCHEDULE_WAN13B_UNREGISTERED_OPT':'1'})
    with pytest.raises(ValueError,match='undeclared'):apply_environment(Task(),Strategy())


def test_shared_budget_counts_failed_or_duplicate_trials(tmp_path):
    for i in range(8):assert reserve_trial(tmp_path,'task-hardware-source',f'run{i}')==i+1
    with pytest.raises(ValueError,match='exhausted'):reserve_trial(tmp_path,'task-hardware-source','run8')
    with pytest.raises(ValueError,match='identity'):reserve_trial(tmp_path,'different','run8')
    value=json.loads((tmp_path/'budget.json').read_text())
    assert value['trials']==[f'run{i}' for i in range(8)]


def test_native_build_receipt_cannot_claim_wrong_engine_or_chunk():
    from distillschedule.strategies.wan import PRECISION
    s=Strategy('resident',2048,True)
    def row(role):
        return {'metadata':{'modulation_chunk_runtime':dict(effective_tokens=[2048],all_blocks_match=True,block_count=30),
                'planner_execution_precision':{'precision_fingerprint':PRECISION.digest},
                'generator_nograd_lora_fused_epilogue':{'enabled':role=='generator'}},
                'placement':{'gradient_reduction':'SUM_THEN_DIVIDE_WORLD_SIZE'}}
    build={role:row(role) for role in ('generator','critic')}
    engine_type=type('GradReadyAsyncResidentEngine',(),{})
    assert validate_build(build,s,(engine_type(),engine_type()))['status']=='PASS_BUILD_BINDING'
    with pytest.raises(ValueError,match='engine'):validate_build(build,s,(NS(),NS()))
    build['critic']['metadata']['modulation_chunk_runtime']['effective_tokens']=[512]
    with pytest.raises(ValueError,match='chunk'):validate_build(build,s,(engine_type(),engine_type()))
