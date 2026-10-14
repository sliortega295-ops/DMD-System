"""Actual FSDP2 Generator shards + score P2P latent gradients (CPU/Gloo).

These checks never substitute single-process mock communication for FSDP.
CUDA/A100/H100 validation is a separate gate.
"""
from datetime import timedelta
from pathlib import Path
import json
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from distillschedule.open_workload.role_program import PoolLayout,RoleTask,AllocatedWorld,enumerate_layouts
from distillschedule.open_workload.role_transport import RolePoolExecutor,RoleFunctions
from test_open_workload_role_transport import make_model,make_opt,sample,generate,guidance,score_train,independent_reference


@pytest.mark.parametrize('n,window,legal',[(4,2,True),(4,1,False),(5,3,False)])
def test_sharded_pool_requires_aligned_collective_order(n,window,legal):
    task=RoleTask(n,(2,4),score_updates=1)
    layout=PoolLayout((0,2),(1,3),window,'resident','fsdp2_keep_forward')
    hw=AllocatedWorld((2**26,)*4,(0,0,1,1))
    if legal: assert layout.validate(task,hw) is layout
    else:
        with pytest.raises(ValueError,match='equal Generator work'):layout.validate(task,hw)
    assert PoolLayout.from_dict(layout.to_dict())==layout


def test_candidate_matching_does_not_assume_sharded_pools_always_win():
    task=RoleTask(4,(2,4),score_updates=1)
    hw=AllocatedWorld((2**26,)*4,(0,0,1,1))
    choices=enumerate_layouts(task,hw,generator_bytes=5000,score_bytes=5000,
                              max_candidates=256,allow_generator_fsdp=True)
    sharded=[p for p in choices['candidates'] if p.generator_parallelism=='fsdp2_keep_forward']
    assert sharded and any(p.generator==(0,2) and p.window==2 for p in sharded)
    assert any(p.generator_parallelism=='replicated' for p in choices['candidates'])
    assert any(r['reason']=='FSDP_COLLECTIVE_ORDER' for r in choices['rejected'])


def _worker(rank,rdzv,folder):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+rdzv,rank=rank,world_size=4,
                            timeout=timedelta(seconds=60))
    runner=None
    try:
        task=RoleTask(4,(2,4),score_updates=1)
        layout=PoolLayout((0,2),(1,3),2,'resident','fsdp2_keep_forward')
        hw=AllocatedWorld((2**26,)*4,(0,0,1,1))
        f=RoleFunctions(make_model,make_opt,sample,generate,guidance,score_train)
        runner=RolePoolExecutor(task,layout,hw,f,device='cpu')
        receipt,model,opt=runner.run()
        assert receipt['status']=='PASS_ROLE_POOL_EXECUTION'
        if rank in layout.generator:
            from torch.distributed.tensor import DTensor
            assert isinstance(next(model.parameters()),DTensor)
            assert receipt['ranks'][rank]['sharding']['role']=='generator_sharded'
            weights={k:(v.full_tensor() if isinstance(v,DTensor) else v).detach().cpu()
                     for k,v in model.state_dict().items()}
            assert opt.state and any(isinstance(p,DTensor) for p in opt.state)
        else:
            weights={k:v.detach().cpu() for k,v in model.state_dict().items()}
        expected=independent_reference(task)[runner.role]
        assert all(torch.allclose(v,expected[k],rtol=2e-5,atol=2e-6)
                   for k,v in weights.items())
        Path(folder,f'rank-{rank}.json').write_text(json.dumps({'pass':True}))
    finally:
        if runner is not None: runner.close()
        dist.destroy_process_group()


def test_four_process_fsdp2_g_score_vjp_matches_global_adam(tmp_path):
    mp.spawn(_worker,args=(str(tmp_path/'rdzv'),str(tmp_path)),nprocs=4,join=True)
    assert all(json.loads((tmp_path/f'rank-{r}.json').read_text())['pass'] for r in range(4))
