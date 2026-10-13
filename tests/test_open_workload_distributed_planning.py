"""Actual multiprocess calibration -> shared DP -> selected execution, CPU only."""
from datetime import timedelta
import json
from pathlib import Path
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from distillschedule.open_workload.replica_contract import ReplicaSemantics
from distillschedule.open_workload.execution import ExecutionBackend
from distillschedule.open_workload.planning import calibrate, plan, run
from test_open_workload_execution import build


def planning_worker(rank, rendezvous, root):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2,timeout=timedelta(seconds=40))
    try:
        w=build(n=2,width=4,length=5);w.feeds={'x':w.feeds['x']+rank*.3}
        w.replica_semantics=ReplicaSemantics({'forward':'learner'},('b0','b1'),5)
        backend=ExecutionBackend(w);root=Path(root)
        c=calibrate(w,root/'cal',max_candidates=12,samples=1,max_program_runs=14)
        assert c['replicas']==2 and len(c['variants'])==12
        assert c['inventory']['exhaustive']
        p=backend.select_plan(lambda:plan(root/'cal/calibration.json',memory_limit_bytes=10**7))
        backend.publish(root/'plan.json',p)
        assert len(p['selected']['memory'])==2
        assert len(p['selected']['timeline'][0]['resources'])==2
        result,session,outputs=run(w,root/'plan.json',root/'exec')
        assert result['status']=='PASS_SELECTED_PROGRAM'
        assert result['replicas']==2 and result['gpu_execution']=='NOT_RUN'
        assert all(r['effects']['gradient_collectives']>0 for r in result['rank_executions'])
        assert len(c['identity']['replicas']['ranks'])==2
        # Only one replica changes data: all replicas must reject the stale plan,
        # not let the other rank enter model collectives with incompatible inputs.
        w.feeds={'x':w.feeds['x']+(1. if rank else 0.)}
        try: run(w,root/'plan.json',root/'must-not-execute')
        except ValueError as exc: assert 'changed' in str(exc)
        else: raise AssertionError('stale rank input accepted')
        assert not (root/'must-not-execute').exists()
    finally:dist.destroy_process_group()


def test_unseen_workload_replicated_auto_plan(tmp_path):
    mp.spawn(planning_worker,args=(str(tmp_path/'rdzv'),str(tmp_path/'campaign')),nprocs=2,join=True)
    result=json.loads((tmp_path/'campaign/exec/execution.json').read_text())
    assert result['reference_gate']['status']=='PASS'
    assert result['replicas']==2
