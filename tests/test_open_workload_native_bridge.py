"""The old native optimizer is a client of the SAME service, not a second solver."""
from distillschedule.search import service
from distillschedule.strategies import wan_planning
from distillschedule.strategies.wan_task import OriginalWanTask
from distillschedule.strategies.wan_strategy import WanStrategy
from test_strategy_wan_planning import collection


def test_original_wan_uses_shared_service_and_retains_native_strategy(tmp_path,monkeypatch):
    task=OriginalWanTask(fake_updates=3,first_step=6)
    data=collection(tmp_path,task,[(WanStrategy('fsdp',512,False),100,50,20,300),
                                  (WanStrategy('resident',2048,True),50,100,20,300)])
    calls=[]
    def observed(candidates,**kwargs):
        calls.extend(candidates)
        return service.select(candidates,**kwargs)
    monkeypatch.setattr(wan_planning,'select',observed)
    result=wan_planning.plan(task,[data])
    assert len(calls)==2
    assert all(c.executable_backend=='original-wan-world16' for c in calls)
    assert result['selected']['strategy']['placement']=='resident'
    assert result['selected']['strategy']['chunk_tokens']==2048
    assert result['selected']['strategy']['lora_epilogue'] is True
    assert result['evidence_scope']=='CPU_FIXTURE_ONLY'
    assert result['gpu_execution']=='NOT_RUN_BY_PLANNER'
