"""Cohort-dependent storage and truly frozen guidance, not trainable stand-ins."""
from datetime import timedelta
from pathlib import Path
import json
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from distillschedule.open_workload.role_program import (
    RoleTask,AllocatedWorld,PoolLayout,transport_bytes_by_rank,enumerate_layouts)
from distillschedule.open_workload.role_transport import RoleFunctions,RolePoolExecutor
from test_open_workload_role_transport import make_model,make_opt,sample,generate,guidance,independent_reference


def test_score_transport_scales_with_retained_samples_and_updates():
    layout=PoolLayout((0,2),(1,3),2)
    small=transport_bytes_by_rank(RoleTask(2,(2,4)),layout)
    many=transport_bytes_by_rank(RoleTask(20,(2,4)),layout)
    train=transport_bytes_by_rank(RoleTask(20,(2,4),score_updates=1),layout)
    assert many[1]>small[1] and train[1]>many[1]
    assert train[0]==many[0]  # same generator window
    offload=PoolLayout(layout.generator,layout.score,2,'cpu_offload_prefetch')
    assert transport_bytes_by_rank(RoleTask(20,(2,4)),offload)==many


def test_previously_underestimated_score_cohort_is_rejected():
    task=RoleTask(32,(1,1024),score_updates=1)
    hw=AllocatedWorld((2**25,)*2)
    layout=PoolLayout((0,),(1,),1)
    bounds=transport_bytes_by_rank(task,layout)
    assert bounds[1] > 3*task.bytes_per_latent
    too_small=AllocatedWorld((2**25,1000+3*task.bytes_per_latent))
    with pytest.raises(ValueError,match='no feasible'):
        enumerate_layouts(task,too_small,generator_bytes=1000,score_bytes=1000)


def frozen_model(role,device):
    model=make_model(role,device)
    if role=='score':model.requires_grad_(False)
    return model


def optimizer_only_for_generator(role,model):
    assert role=='generator','frozen Score must not construct optimizer'
    return make_opt(role,model)


def frozen_worker(rank,rdzv,out,offload):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+rdzv,rank=rank,world_size=2,
                            timeout=timedelta(seconds=60))
    try:
        task=RoleTask(5,(2,4))
        layout=PoolLayout((0,),(1,),2,'cpu_offload_prefetch' if offload else 'resident')
        hw=AllocatedWorld((2**25,)*2,host_capacities=(2**25,)*2)
        f=RoleFunctions(frozen_model,optimizer_only_for_generator,sample,generate,guidance)
        result,model,opt=RolePoolExecutor(task,layout,hw,f,device='cpu').run()
        role='generator' if rank==0 else 'score';reference=independent_reference(task)
        for key,t in model.state_dict().items():
            torch.testing.assert_close(t,reference[role][key],rtol=1e-6,atol=1e-7)
        if role=='score':
            assert opt is None and all(not p.requires_grad and p.grad is None for p in model.parameters())
        assert all(r['transport_peak_bytes']<=r['transport_bound_bytes'] for r in result['ranks'])
        assert result['ranks'][rank]['cuda_peak_allocated_bytes'] is None
        Path(out,f'{rank}.json').write_text(json.dumps({'passed':True}))
    finally:dist.destroy_process_group()


@pytest.mark.parametrize('offload',[False,True])
def test_true_frozen_score_returns_input_gradient(tmp_path,offload):
    mp.spawn(frozen_worker,args=(str(tmp_path/'rdzv'),str(tmp_path),offload),nprocs=2,join=True)
    assert all(json.loads((tmp_path/f'{i}.json').read_text())['passed'] for i in range(2))
