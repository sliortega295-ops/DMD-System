"""Unified original-Wan CLI and peer-publication contracts; no native GPU work."""
import json
import pytest
from distillschedule.strategies import wan_autoplan as module


def test_public_auto_run_routes_to_original_protocol(tmp_path,monkeypatch,capsys):
    from distillschedule.cli import main
    from distillschedule.strategies.wan_task import OriginalWanTask
    from distillschedule.strategies.wan_inputs import WanInputSources
    task=tmp_path/'task.json';task.write_text(json.dumps(OriginalWanTask(fake_updates=1,first_step=2).to_dict()))
    sources=tmp_path/'sources.json';sources.write_text(json.dumps(WanInputSources(tuple(f'{i}.pt' for i in range(16))).to_dict()))
    assets=tmp_path/'model.json';assets.write_text('{}')
    calls=[]
    def fake(*args,**kw):calls.append((args,kw));return {'status':'INJECTED_ORIGINAL_PROTOCOL_ROUTE'}
    monkeypatch.setattr(module,'run',fake)
    args=['workloads','native','auto-run','--task',str(task),'--sources',str(sources),
          '--model-assets',str(assets),'--training-config',str(tmp_path/'training.yaml'),
          '--output',str(tmp_path/'out'),'--node-rank','1','--master-addr','allocated-node',
          '--master-port','29401','--without-fusion','--max-launches','5']
    assert main(args)==0 and len(calls)==1
    assert not calls[0][1]['execute'] and calls[0][0][0].fake_updates==1
    assert main(args+['--execute','--resume'])==0 and calls[1][1]['resume']
    assert calls[1][1]['node_rank']==1 and calls[1][1]['execute']
    assert 'INJECTED_ORIGINAL_PROTOCOL_ROUTE' in capsys.readouterr().out


def test_atomic_peer_record_refuses_overwrite_and_cleans_temporary(tmp_path):
    from distillschedule.strategies.wan_prepare import publish_new_json
    p=tmp_path/'ready.json'
    publish_new_json(p, {'good': True})
    with pytest.raises(FileExistsError):
        publish_new_json(p, {'different': True})
    assert json.loads(p.read_text()) == {'good': True}
    assert list(tmp_path.iterdir()) == [p]
