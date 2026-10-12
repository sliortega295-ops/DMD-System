"""Two-node controller tests with injected preparation; no GPU performance claims."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import json
import pytest
from distillschedule.strategies import wan_autoplan as module
from distillschedule.workloads.best_known.registry import write_new_json


@pytest.fixture
def setup(tmp_path, monkeypatch):
    calls=[]; launches=[]; task=SimpleNamespace(digest='t'*64)
    def prepare(task,src,assets,cfg,out,**kw):
        if not kw.get('execute'):
            return {'preparation_sha256':'a'*64, 'gpu_execution':'NOT_RUN'}
        calls.append(kw);root=Path(out);root.mkdir(exist_ok=True)
        ready={'status':'READY_FOR_ORIGINAL_WAN_CAMPAIGN','preparation_sha256':'a'*64,
               'campaign':str(root/'campaign/campaign.json'),'campaign_sha256':'c'*64,'trace_hash':'b'*64}
        if not (root/'ready.json').exists():write_new_json(root/'ready.json',ready)
        return ready
    def campaign(*args, **kwargs):
        launches.append(kwargs)
        return {'status':'COMPLETED_WITH_NUMERICAL_REVIEW', 'gpu_execution':'INJECTED_NO_GPU',
                'native_launches':8, 'claimable_speedup':False}
    monkeypatch.setattr(module,'prepare',prepare)
    monkeypatch.setattr(module,'verify_campaign',lambda p:{'campaign_sha256':'c'*64,'task_sha256':task.digest})
    monkeypatch.setattr(module,'run_campaign',campaign)
    def run(node,**kw):
        return module.run(task,None,None,None,tmp_path/'out',node_rank=node,
                          master_addr='fixture-host',master_port=29401,**kw)
    return run,calls,launches


def test_two_nodes_prepare_once_then_delegate_unchanged_budget_and_review(setup):
    run,calls,launches=setup
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(lambda node:run(node,execute=True),(1,0)))
    assert len(calls)==1 and {r['node_rank'] for r in launches}=={0,1}
    assert all(r['status']=='COMPLETED_WITH_NUMERICAL_REVIEW' for r in results)
    assert all(r['timeout_seconds']==7200 and not r['resume'] for r in launches)


def test_dryrun_no_preparation_or_native_launch(setup,tmp_path):
    run,calls,launches=setup
    assert run(0)['status']=='VALIDATED_AUTO_RUN_NOT_EXECUTED'
    assert not calls and not launches and not (tmp_path/'out').exists()


def test_peer_wrong_task_never_launches(setup,tmp_path):
    run,calls,launches=setup
    root=tmp_path/'out';root.mkdir();(root/'ready.json').write_text(json.dumps(
        {'status':'READY_FOR_ORIGINAL_WAN_CAMPAIGN','preparation_sha256':'wrong'}))
    with pytest.raises(ValueError,match='different task'):
        run(1,execute=True)
    assert not calls and not launches


def test_coordinator_failure_reaches_peer_without_launch(setup,tmp_path,monkeypatch):
    run,calls,launches=setup
    old=module.prepare
    def fail(*args,**kwargs):
        if kwargs.get('execute'):
            Path(args[4]).mkdir(exist_ok=True)
            raise ValueError('unaligned native source')
        return old(*args,**kwargs)
    monkeypatch.setattr(module,'prepare',fail)
    with pytest.raises(ValueError,match='unaligned'):
        run(0,execute=True)
    with pytest.raises(ValueError,match='failed previously'):
        run(1,execute=True)
    assert not launches


def test_timeout_does_not_consume_launch_budget(setup,monkeypatch):
    run,calls,launches=setup
    ticks=iter([0,2])
    monkeypatch.setattr(module.time,'monotonic',lambda:next(ticks))
    with pytest.raises(TimeoutError,match='no GPU launch'):
        run(1,execute=True,preparation_timeout=1)
    assert not calls and not launches


def test_duplicate_controller_does_not_poison_active_preparation(setup,tmp_path,monkeypatch):
    run,calls,launches=setup
    root=tmp_path/'out';root.mkdir();(root/'prepare.lock').write_text('active')
    old=module.prepare
    def busy(*args,**kw):
        if kw.get('execute'):raise FileExistsError('another controller holds preparation lock')
        return old(*args,**kw)
    monkeypatch.setattr(module,'prepare',busy)
    with pytest.raises(FileExistsError):run(0,execute=True)
    assert not list(root.glob('prepare-failure*')) and not launches
