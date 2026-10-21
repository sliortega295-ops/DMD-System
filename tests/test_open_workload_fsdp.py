"""Actual CPU FSDP2 on an unknown model, including already-initialized Adam."""
from datetime import timedelta
from pathlib import Path
import json
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from distillschedule.open_workload.sharded import ShardedLifecycle,MODES
from distillschedule.open_workload.torch_backend import Session,candidates,compare
from test_open_workload_multirole_replica import build_workload

pytestmark=pytest.mark.skipif(tuple(int(x) for x in torch.__version__.split('.')[:2])<(2,10),
                             reason='CPU FSDP2 validation uses Torch >=2.10; not a CUDA pass')


def worker(rank,rendezvous,output,frequency):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2,
                            timeout=timedelta(seconds=40))
    try:
        from distillschedule.open_workload.replicated import make_session,inventory
        w=build_workload(frequency,4,3,rank=rank)
        # First train once using averaged replicated gradients, then keep Adam
        # moments for the next round. FSDP must redistribute these exact moments.
        warm=make_session(w,inventory(w,limit=1)['candidates'][0]);warm.run()
        w.state=warm.state
        global_w=build_workload(frequency,4,3,global_batch=True)
        global_w.state=warm.state
        ref=Session(global_w,candidates(global_w,limit=1)['candidates'][0])
        expected=ref.result_state(ref.run())['state']
        options=candidates(w,limit=16)['candidates']
        chosen=[options[0],next(o for o in options if not o['share_immutable'] and len(o['checkpoint_bindings'])==3)]
        for mode in MODES:
            for local in chosen:
                impl={'backend':'torch-callable-replicated-v1','local':local,'reduction':mode,'bucket_cap_mb':0}
                s=make_session(w,impl)
                result=s.run();state=s.result_state(result)
                checked=compare(expected,state['state'],rtol=4e-5,atol=4e-7)
                assert checked['status']=='PASS',checked
                from torch.distributed.tensor import DTensor
                assert all(isinstance(p,DTensor) for key in ('role_a','role_b') for p in s.state[key].parameters())
                assert s.effects['optimizer_moments_migrated']==4
                assert s.effects['fsdp_sharded_parameters']==6
                assert all(p.to_local().numel()<p.numel() for key in ('role_a','role_b') for p in s.state[key].parameters())
        Path(output,f'rank{rank}.json').write_text(json.dumps({'variants':4,'gpu_execution':'NOT_RUN'}))
    finally:dist.destroy_process_group()


@pytest.mark.parametrize('frequency',[1,3])
def test_real_fsdp2_preserves_multirole_adam_and_input_gradients(tmp_path,frequency):
    mp.spawn(worker,args=(str(tmp_path/'rdzv'),str(tmp_path),frequency),nprocs=2,join=True)
    assert json.loads((tmp_path/'rank0.json').read_text())['variants']==4


def auto_worker(rank,rendezvous,output):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2,
                            timeout=timedelta(seconds=40))
    try:
        from distillschedule.open_workload.execution import ExecutionBackend
        from distillschedule.open_workload.planning import calibrate,plan,run
        from distillschedule.workloads.best_known.registry import fingerprint
        w=build_workload(1,4,3,rank=rank);backend=ExecutionBackend(w);root=Path(output)
        manifest=calibrate(w,root/'cal',samples=1,max_candidates=5,max_program_runs=6)
        assert {v['implementation']['reduction'] for v in manifest['variants']}=={
            'post_backward','ddp_overlap',*MODES}
        selected=backend.select_plan(lambda:plan(root/'cal/calibration.json',memory_limit_bytes=10**7))
        backend.publish(root/'plan.json',selected)
        result,_,_=run(w,root/'plan.json',root/'run')
        assert result['status']=='PASS_SELECTED_PROGRAM'
        # Independent execution of a measured sharded candidate verifies lowering
        # even when actual observed CPU costs prefer replication on this input.
        candidate=next(x for x in selected['search']['ranked'] if x['implementation']['reduction']=='fsdp2_reshard')
        selected['selected']=candidate
        selected.pop('plan_sha256');selected['plan_sha256']=fingerprint(selected)
        backend.publish(root/'shard-plan.json',selected)
        result,_,_=run(w,root/'shard-plan.json',root/'shard-run')
        assert result['status']=='PASS_SELECTED_PROGRAM'
        assert all(r['effects']['fsdp_sharded_parameters']>0 for r in result['rank_executions'])
    finally:dist.destroy_process_group()


def test_shared_auto_search_includes_actual_fsdp2_layout(tmp_path):
    mp.spawn(auto_worker,args=(str(tmp_path/'rdzv'),str(tmp_path/'auto')),nprocs=2,join=True)
    manifest=json.loads((tmp_path/'auto/cal/calibration.json').read_text())
    assert len(manifest['variants'])==5
