"""Preparation orchestration tests; injected capture/assets are NOT GPU evidence."""
from dataclasses import replace
from pathlib import Path
import json
import pytest

from distillschedule.strategies.wan_task import OriginalWanTask
from distillschedule.strategies.wan_inputs import WanInputSources
from distillschedule.strategies.wan_prepare import prepare
from distillschedule.workloads.best_known.registry import write_new_json


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    import distillschedule.strategies.wan_prepare as module
    import distillschedule.strategies.wan_capture as capture_module
    import distillschedule.strategies.wan_trace as trace_module
    task=OriginalWanTask(fake_updates=1, first_step=2)
    sources=WanInputSources(tuple(str(tmp_path/f'{r}.pt') for r in range(16)))
    model={'model_root':'/fixture/model', 'bootstrap':'/fixture/bootstrap', 'negative_prompt':'/fixture/negative'}
    enriched={**model, 'expected_bootstrap_hash':'a'*64, 'model_files':{'config.json':'1'*64},
              'negative_manifest_hash':'2'*64}
    config=tmp_path/'train.yaml';config.write_text('training_config: {}\n')
    inventory={'digest':'3'*64,'files':[],'sources':sources.to_dict(),'task_sha256':task.digest}
    monkeypatch.setattr(module,'validate_model_assets',lambda *a:enriched)
    monkeypatch.setattr(module,'preflight_sources',lambda *a:inventory)
    monkeypatch.setattr(WanInputSources,'inventory',lambda self:inventory)
    monkeypatch.setattr(capture_module,'read_training_config',lambda p:{})
    monkeypatch.setattr(trace_module,'load_original_trace',lambda *a,**kw:None)
    calls=[]
    def fake_capture(t,s,c,out):
        from distillschedule.strategies.wan_planning import source_fingerprint
        from distillschedule.workloads.best_known.registry import file_sha256
        out=Path(out);out.mkdir()
        result={'status':'PASS_CAPTURED_ORIGINAL_INPUTS','task':t.to_dict(),'input_inventory':inventory,
                'training_config_sha256':file_sha256(c),'source_fingerprint':source_fingerprint(),
                'trace_bundle':str(out),'expected_trace_hash':'b'*64}
        write_new_json(out/'capture.json',result);calls.append(str(out))
        return result
    monkeypatch.setattr(capture_module,'capture',fake_capture)
    return task,sources,model,config,calls


def test_preparation_routes_actual_campaign_and_resume_without_reset(tmp_path, inputs):
    task,sources,model,config,calls=inputs
    root=tmp_path/'prepared'
    result=prepare(task,sources,model,config,root,execute=True)
    assert result['status']=='READY_FOR_ORIGINAL_WAN_CAMPAIGN'
    campaign=json.loads(Path(result['campaign']).read_text())
    assert campaign['planned_native_launches']==8
    choices=[json.loads(Path(j['implementation']).read_text()) for j in campaign['jobs']]
    assert len(choices)==7 and {v['placement'] for v in choices}=={'fsdp','resident'}
    assert any(v['lora_epilogue'] and v['chunk_tokens']==2048 for v in choices)
    assert len(calls)==1 and not (root/'prepare.lock').exists()
    marker=root/'campaign'/'node-0';marker.mkdir();(marker/'journal.json').write_text('unknown-outcome')
    assert prepare(task,sources,model,config,root,execute=True,resume=True)==result
    assert len(calls)==1 and (marker/'journal.json').read_text()=='unknown-outcome'
    with pytest.raises(ValueError,match='resume required'):
        prepare(task,sources,model,config,root,execute=True)
    with pytest.raises(ValueError,match='preparation changed'):
        prepare(task,sources,model,config,root,execute=True,resume=True,include_fusion=False)


def test_preparation_dry_run_never_writes_or_captures(tmp_path, inputs):
    task,sources,model,config,calls=inputs
    result=prepare(task,sources,model,config,tmp_path/'out')
    assert result['gpu_execution']=='NOT_RUN'
    assert not calls and not (tmp_path/'out').exists()


def test_budget_rejected_before_any_capture(tmp_path, inputs):
    task,sources,model,config,calls=inputs
    with pytest.raises(ValueError,match='budget'):
        prepare(task,sources,model,config,tmp_path/'out',execute=True,max_launches=2)
    assert not calls and not (tmp_path/'out').exists()


def test_incomplete_capture_not_overwritten(tmp_path, inputs):
    task,sources,model,config,calls=inputs
    root=tmp_path/'out'
    prepare(task,sources,model,config,root,execute=True)
    (root/'trace'/'capture.json').unlink()
    with pytest.raises(ValueError,match='incomplete capture'):
        prepare(task,sources,model,config,root,execute=True,resume=True)
    assert len(calls)==1 and not (root/'prepare.lock').exists()


def test_prepare_task_uses_public_original_protocol_route(tmp_path, monkeypatch, capsys):
    from distillschedule.cli import main
    import distillschedule.strategies.wan_prepare as module
    task=tmp_path/'task.json';task.write_text(json.dumps(OriginalWanTask().to_dict()))
    src=tmp_path/'sources.json';src.write_text(json.dumps(WanInputSources(tuple(f'{r}.pt' for r in range(16))).to_dict()))
    assets=tmp_path/'assets.json';assets.write_text('{}')
    seen=[]
    def fake(*args,**kwargs):
        seen.append((args,kwargs));return {'status':'CPU_ROUTING_FIXTURE'}
    monkeypatch.setattr(module,'prepare',fake)
    args=['workloads','native','prepare-task','--task',str(task),'--sources',str(src),
          '--model-assets',str(assets),'--training-config',str(tmp_path/'config.yaml'),
          '--output',str(tmp_path/'out'),'--without-fusion','--max-launches','5']
    assert main(args)==0
    assert seen[0][1]==dict(execute=False,resume=False,include_fusion=False,max_launches=5)
    assert main(args+['--execute','--resume'])==0
    assert seen[1][1]['execute'] and seen[1][1]['resume']
    assert 'CPU_ROUTING_FIXTURE' in capsys.readouterr().out
