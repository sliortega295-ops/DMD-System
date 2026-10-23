"""Fresh-data public route tests; no implicit GPU or actual big-model launch."""
from dataclasses import replace
import json
import pytest
from distillschedule.strategies.wan_task import OriginalWanTask
from distillschedule.strategies.wan_inputs import WanInputSources
from distillschedule.strategies.wan_worker import OriginalWanAssets
from distillschedule.cli import main


def test_public_prepare_stream_preserves_original_protocol(tmp_path, monkeypatch, capsys):
    import distillschedule.strategies.wan_stream as module
    task = tmp_path/'task.json'; task.write_text(json.dumps(OriginalWanTask().to_dict()))
    source = WanInputSources(tuple(f'{r}.pt' for r in range(16)))
    sources = tmp_path/'sources.json'
    sources.write_text(json.dumps({'cohorts': [source.to_dict(), replace(source,cohort=1).to_dict()]}))
    seen = []
    monkeypatch.setattr(module,'prepare_stream',lambda *a,**kw: seen.append((a,kw)) or {'status':'CPU_ROUTE_FIXTURE'})
    args=['workloads','native','prepare-stream','--task',str(task),'--sources',str(sources),
          '--training-config','missing.yaml','--output',str(tmp_path/'out')]
    assert main(args)==0 and not seen[0][1]['execute']
    assert main(args+['--execute','--resume'])==0 and seen[1][1]=={'execute':True,'resume':True}
    assert len(seen[1][0][1].cohorts)==2 and not (tmp_path/'out').exists()


def test_public_train_stream_requires_execute_and_validated_complete_sequence(tmp_path, monkeypatch, capsys):
    import distillschedule.strategies.wan_stream as stream
    import distillschedule.strategies.wan_worker as worker
    task = tmp_path/'task.json';task.write_text(json.dumps(OriginalWanTask().to_dict()))
    assets = tmp_path/'assets.json'
    assets.write_text(json.dumps(OriginalWanAssets('model','bootstrap','a'*64,'trace','b'*64,'negative').__dict__))
    seen=[]; loads=[]
    monkeypatch.setattr(stream,'load_stream',lambda *args:loads.append(args) or {'cohorts':[{}, {}, {}]})
    monkeypatch.setattr(worker,'run_worker',lambda *a,**kw:seen.append((a,kw)) or {'status':'CPU_WORKER_ROUTE_FIXTURE'})
    args=['workloads','native','train-stream','--task',str(task),'--assets',str(assets),
          '--plan','selected.json','--stream','stream.json','--output',str(tmp_path/'out')]
    assert main(args)==0 and not loads and not seen
    assert main(args+['--execute'])==0 and len(seen)==1
    assert seen[0][1]['mode']=='run' and seen[0][1]['cycles']==3
    assert str(seen[0][1]['stream_path'])=='stream.json'


def test_unmatched_streams_rejected_before_numerical_comparison(tmp_path):
    from distillschedule.strategies.wan_evidence import compare_runs
    for name, stream in [('a','stream-a'),('b','stream-b')]:
        root=tmp_path/name;root.mkdir()
        for rank in range(16):
            (root/f'rank_{rank:02d}.json').write_text(json.dumps(dict(rank=rank,
                status='COMPLETED_ORIGINAL_WAN_WORK',identity={},task={},initial_versions={},final_versions={},
                measurement_scope='FRESH_COHORT_SEQUENCE_PERSISTENT_OPTIMIZERS',stream_sha256=stream,cycles=[])))
    with pytest.raises(ValueError,match='different cohort'):
        compare_runs(tmp_path/'a',tmp_path/'b')
