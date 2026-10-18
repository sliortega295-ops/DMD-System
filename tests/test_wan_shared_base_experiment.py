"""Synthetic CPU contracts; never GPU/timing evidence."""
import copy
import json
from types import SimpleNamespace
import pytest
import torch
from distillschedule.runtime.wan13b_shared_base_experiment import (
    ARMS, SCHEMA, cold_controller, install_explicit_session, sha, validate_packet)
from test_shared_frozen_base import roles


def test_packet_source_workload_and_arms_fail_closed(tmp_path):
    task, assets, strategy = (tmp_path / n for n in ('task', 'assets', 'strategy'))
    task.write_text('{}'); assets.write_text('{}')
    strategy.write_text(json.dumps(dict(placement='resident', retained_critic_blocks=[26,27,28,29])))
    packet=dict(schema_version=SCHEMA, source_head='a'*40, arm='private-retain4',
                run_id='explicit-test-only', share_frozen_base=False, cycles=2,
                task_sha256=sha(task), assets_sha256=sha(assets), strategy_sha256=sha(strategy),
                checkpoint_identity=str(tmp_path.resolve()))
    kw=dict(head='a'*40, task=task, assets=assets, strategy=strategy, model_root=tmp_path)
    validate_packet(packet,**kw)
    for key,value in [('source_head','b'*40),('cycles',1),('share_frozen_base',True),
                      ('arm','shared-retain6'),('task_sha256','bad'),('checkpoint_identity','wrong')]:
        with pytest.raises(Exception):validate_packet(dict(packet,**{key:value}),**kw)
    with pytest.raises(Exception):validate_packet(dict(packet,hidden_override=True),**kw)


class Cold:
    def __init__(self,model):
        self.model=model;self.cache={}
        self.receipt={k:0 for k in('capture_entries','replay_calls','backward_replay_calls','reference_calls')}
        self.pointer_contract=self.parameter_identity()
    def parameter_identity(self):return tuple(p.data_ptr()for p in self.model.parameters())


def test_cold_controller_rejects_any_execution_or_address_drift():
    a,b=roles();c=Cold(b);b._distillschedule_critic_grad_cuda_graph=c
    assert cold_controller(b)is c
    for key in c.receipt:
        c.receipt[key]=1
        with pytest.raises(Exception):cold_controller(b)
        c.receipt[key]=0
    c.cache[1]=object()
    with pytest.raises(Exception):cold_controller(b)
    c.cache.clear();c.pointer_contract=()
    with pytest.raises(Exception):cold_controller(b)


@pytest.mark.parametrize('enabled',[False,True])
def test_real_setup_branch_control_unchanged_alias_before_capture_restore(monkeypatch,enabled):
    a,b=roles();c=Cold(b);b._distillschedule_critic_grad_cuda_graph=c
    ref_a,ref_b=copy.deepcopy(a),copy.deepcopy(b)
    snapshots=[];monkeypatch.setattr(torch.cuda,'memory_allocated',lambda device:123)
    class Session:
        def __init__(self,**kwargs):
            self.g=SimpleNamespace(module=a);self.c=SimpleNamespace(module=b);self.device='cpu'
            self.build={k:{'metadata':{}}for k in('generator','critic')}
        def versions(self):return dict(generator=2,critic=10,ema=10)
        def prepare(self,**kwargs):return 'ORIGINAL_PREPARED'
    worker=SimpleNamespace(OriginalWanSession=Session)
    packet=dict(arm='shared-retain6'if enabled else'private-retain4',source_head='a'*40,
                run_id='test-only',share_frozen_base=enabled,checkpoint_identity='/same')
    pointer=c.pointer_contract;rng=torch.random.get_rng_state().clone()
    with install_explicit_session(worker,packet,'packet-sha'):
        s=worker.OriginalWanSession(model_root='/same')
        assert s.prepare()=='ORIGINAL_PREPARED' and s.versions()['generator']==2
        assert (a.base_layer.weight.data_ptr()==b.base_layer.weight.data_ptr())is enabled
        assert c.pointer_contract==c.parameter_identity()
        assert bool(c.pointer_contract!=pointer)is enabled
        assert c.cache=={}and c.receipt['capture_entries']==0
        for left,right in((a,ref_a),(b,ref_b)):
            x=torch.arange(16,dtype=torch.float64).reshape(2,8)/20
            y,z=left(x),right(x);assert torch.equal(y,z)
            y.square().mean().backward();z.square().mean().backward()
            for p,q in zip(left.parameters(),right.parameters()):
                if p.requires_grad:assert torch.equal(p.grad,q.grad)
        assert torch.equal(rng,torch.random.get_rng_state())
    assert worker.OriginalWanSession is Session
    assert not torch.cuda.is_initialized()


def test_scoped_worker_restored_after_failure():
    original=object();worker=SimpleNamespace(OriginalWanSession=original)
    # Class construction itself requires an actual original class.
    class Session:pass
    worker.OriginalWanSession=Session
    with pytest.raises(RuntimeError):
        with install_explicit_session(worker,{},'x'):raise RuntimeError('native failure')
    assert worker.OriginalWanSession is Session


def test_explicit_entrypoint_binds_packet_and_uses_original_two_cycle_worker(tmp_path,monkeypatch):
    from distillschedule.runtime import wan13b_shared_base_experiment as experiment
    from distillschedule.strategies import wan_worker as worker
    from distillschedule.strategies.wan_task import OriginalWanTask
    from distillschedule.strategies.wan_strategy import WanStrategy
    from distillschedule.search.wan_candidate_bundle import INCUMBENT
    task,assets,impl,packet_path=(tmp_path/n for n in('task','assets','impl','packet'))
    task.write_text(json.dumps(OriginalWanTask().to_dict()))
    assets.write_text(json.dumps(dict(schema_version='distillschedule.original_wan_assets.v1',
        model_root='/synthetic/model',bootstrap='/synthetic/bootstrap',
        expected_bootstrap_hash='a'*64,trace_bundle='/synthetic/trace',
        expected_trace_hash='b'*64,negative_prompt='/synthetic/negative')))
    impl.write_text(json.dumps(WanStrategy('resident',2048,True,5,candidate_bundle=INCUMBENT,
        retained_critic_blocks=(26,27,28,29)).to_dict()))
    packet=dict(schema_version=SCHEMA,source_head='a'*40,arm='private-retain4',
        run_id='synthetic-test-only',share_frozen_base=False,cycles=2,
        task_sha256=sha(task),assets_sha256=sha(assets),strategy_sha256=sha(impl),
        checkpoint_identity='/synthetic/model')
    packet_path.write_text(json.dumps(packet));original=worker.OriginalWanSession;calls=[]
    monkeypatch.setattr(experiment.subprocess,'check_output',lambda *a,**k:'a'*40+'\n')
    def run(task,assets,**kw):
        calls.append(kw)
        assert issubclass(worker.OriginalWanSession,original)
        assert kw==dict(output=tmp_path/'output',mode='manual',strategy=WanStrategy.from_dict(
            json.loads(impl.read_text())),cycles=2,record_memory_highwater=True)
        return dict(status='SYNTHETIC_TEST_ONLY',rank=0)
    monkeypatch.setattr(worker,'run_worker',run)
    argv=[]
    for key,value in dict(task=task,assets=assets,implementation=impl,output=tmp_path/'output',
                          packet=packet_path,**{'packet-sha256':sha(packet_path)}).items():
        argv+=['--'+key,str(value)]
    with pytest.raises(Exception):experiment.main(argv)
    assert calls==[]
    experiment.main(argv+['--execute'])
    assert len(calls)==1 and worker.OriginalWanSession is original
    assert not torch.cuda.is_initialized()
