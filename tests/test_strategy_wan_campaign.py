from pathlib import Path
import json
import pytest
from distillschedule.strategies.wan_task import OriginalWanTask
from distillschedule.strategies.wan_worker import OriginalWanAssets
from distillschedule.strategies.wan_campaign import prepare,verify,run


def assets():
    return OriginalWanAssets(model_root='local-model',bootstrap='local-bootstrap',expected_bootstrap_hash='a'*64,
        trace_bundle='local-trace',expected_trace_hash='b'*64,negative_prompt='local-negative')


def test_campaign_covers_old_optimized_arms_and_counts_confirmation(tmp_path,monkeypatch):
    c=prepare(OriginalWanTask(),assets(),tmp_path/'campaign')
    assert c['planned_native_launches']==c['max_native_launches']==8
    assert len(c['jobs'])==7
    choices=[json.loads(Path(j['implementation']).read_text()) for j in c['jobs']]
    assert choices[0]['placement']=='fsdp'
    assert dict(placement='resident',chunk_tokens=2048,lora_epilogue=True,rollout_window=1) in choices
    import subprocess
    def reject(*args,**kwargs):raise AssertionError('no implicit GPU or process launch')
    monkeypatch.setattr(subprocess,'run',reject)
    r=run(tmp_path/'campaign/campaign.json',node_rank=0,master_addr='explicit-host',master_port=29401)
    assert r['gpu_execution']=='NOT_RUN_REQUIRES_EXPLICIT_EXECUTE'
    assert not (tmp_path/'campaign/node-0').exists()


def test_small_budget_does_not_hide_missing_optimized_candidates(tmp_path):
    with pytest.raises(ValueError,match='budget cannot cover'):
        prepare(OriginalWanTask(),assets(),tmp_path/'out',max_launches=7)
    assert not (tmp_path/'out').exists()
    c=prepare(OriginalWanTask(),assets(),tmp_path/'out',include_fusion=False,max_launches=5)
    assert c['planned_native_launches']==5


def test_previous_campaign_not_overwritten(tmp_path):
    prepare(OriginalWanTask(),assets(),tmp_path/'out')
    with pytest.raises(FileExistsError):prepare(OriginalWanTask(),assets(),tmp_path/'out')


def test_changed_implementation_invalidates_campaign(tmp_path):
    c=prepare(OriginalWanTask(),assets(),tmp_path/'out')
    Path(c['jobs'][0]['implementation']).write_text('{}')
    with pytest.raises(ValueError,match='strategy changed'):verify(tmp_path/'out/campaign.json')


def test_two_node_controller_flow_and_resume_with_injected_workers(tmp_path,monkeypatch):
    """Controller-only test: injected workers do NOT perform tensor/GPU work."""
    from concurrent.futures import ThreadPoolExecutor
    import threading
    import distillschedule.strategies.wan_campaign as module
    import distillschedule.strategies.wan_evidence as evidence
    c=prepare(OriginalWanTask(),assets(),tmp_path/'out')
    spawned=[];guard=threading.Lock()
    class Process:
        def __init__(self,command,**kwargs):
            rank=int(next(v.split('=')[1] for v in command if v.startswith('--node-rank=')))
            dest=Path(command[command.index('--output')+1]);dest.mkdir(parents=True,exist_ok=True)
            with guard:spawned.append((rank,command))
            for r in range(8*rank,8*rank+8):
                (dest/f'rank_{r:02d}.json').write_text(json.dumps({'fixture':'INJECTED_WORKER_NO_GPU','rank':r}))
        def wait(self,timeout=None):return 0
    monkeypatch.setattr(module.subprocess,'Popen',Process)
    def assemble_fixture(task,paths,output):
        assert len(paths)==16 and all(p.is_file() for p in paths)
        Path(output).mkdir();(Path(output)/'collection.json').write_text('{"fixture":true}')
    monkeypatch.setattr(module,'assemble',assemble_fixture)
    monkeypatch.setattr(module,'plan',lambda task,paths:{'fixture':True,'observed_collections':len(paths)})
    monkeypatch.setattr(evidence,'compare_runs',lambda *args:{'status':'PASS','fixture':'NO_NUMERICAL_DATA'})
    def one(rank,resume=False):return run(tmp_path/'out/campaign.json',node_rank=rank,master_addr='fixture-host',
                                        master_port=29401,execute=True,resume=resume)
    with ThreadPoolExecutor(max_workers=2) as pool:
        result=list(pool.map(one,(0,1)))
    assert len(spawned)==16  # eight distributed launches, two local torchrun parents per launch
    assert all(r['native_launches']==8 for r in result)
    assert all(r['status']=='COMPLETED_WITH_REFERENCE_PARITY' for r in result)
    assert json.loads((tmp_path/'out/selected-plan.json').read_text())['observed_collections']==7
    with ThreadPoolExecutor(max_workers=2) as pool:
        again=list(pool.map(lambda r:one(r,True),(0,1)))
    assert len(spawned)==16 and all(r['native_launches']==8 for r in again)
    with pytest.raises(ValueError,match='already attempted'):one(0)
