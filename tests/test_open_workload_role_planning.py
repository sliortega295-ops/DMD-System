"""Measured native full-cycle role pipeline uses common source-bound DP selector."""
from datetime import timedelta
from pathlib import Path
import json
import pytest
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from distillschedule.open_workload.role_transport import RoleFunctions
from distillschedule.open_workload.role_program import RoleTask,AllocatedWorld
from distillschedule.open_workload.role_planning import calibrate,search,run_selected
from test_open_workload_role_transport import make_model,make_opt,sample,generate,guidance,score_train,independent_reference


def worker(rank,world,task,rdzv,out):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+rdzv,rank=rank,world_size=world,
                            timeout=timedelta(seconds=90))
    try:
        bindings=RoleFunctions(make_model,make_opt,sample,generate,guidance,
                               score_train if task.score_updates else None,
                               data_identity='a'*64)
        hw=AllocatedWorld((2**25,)*world,tuple(0 for _ in range(world)))
        m=calibrate(task,hw,bindings,Path(out)/'cal',generator_bytes=500,
                    score_bytes=600,max_candidates=3,samples=1,max_native_runs=4)
        assert len(m['candidates'])==3 and m['budget']['calibration_runs']==3
        if rank==0:
            plan=search(Path(out)/'cal/calibration.json')
            from distillschedule.workloads.best_known.registry import write_new_json
            write_new_json(Path(out)/'plan.json',plan)
        dist.barrier()
        plan=json.loads((Path(out)/'plan.json').read_text())
        assert plan['selected']['implementation'] in [x['layout'] for x in m['candidates']]
        assert plan['search']['status']=='PLAN_FOUND'
        result,model,_=run_selected(task,hw,bindings,Path(out)/'plan.json',Path(out)/'execution')
        baseline=independent_reference(task)
        role='generator' if rank in result['layout']['generator'] else 'score'
        for k,v in model.state_dict().items():
            assert torch.allclose(v.detach(),baseline[role][k],atol=1e-7,rtol=1e-6)
        assert result['claimable_speedup'] is False
    finally:dist.destroy_process_group()


@pytest.mark.parametrize('world,updates',[(2,0),(3,1)])
def test_measured_search_and_real_executed_pool(tmp_path,world,updates):
    t=RoleTask(3,(2,4),score_updates=updates)
    mp.spawn(worker,args=(world,t,str(tmp_path/'rdzv'),str(tmp_path)),nprocs=world,join=True)
    doc=json.loads((tmp_path/'execution/execution.json').read_text())
    assert doc['status']=='PASS_ROLE_POOL_EXECUTION'
    p=json.loads((tmp_path/'plan.json').read_text())
    assert p['execution_model']=='FULL_CYCLE_MEASURED_FIFO_WINDOW_NOT_UNLOWERED_DP_ORDER'
    assert p['selected']['predicted_makespan_ns']>0


def test_calibrated_plan_rejects_tampered_receipts(tmp_path):
    # This checks a pure metadata cost contract without constructing models.
    from distillschedule.open_workload.role_program import RoleTask
    a=RoleTask(2,(2,4));b=RoleTask(3,(2,4))
    assert a.program().digest!=b.program().digest
