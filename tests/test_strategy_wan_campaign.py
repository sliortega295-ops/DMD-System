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
