"""Prove supported combinations; reject unsupported ones before entering P2P.

Real four-process CPU/Gloo tests. They do not validate CUDA copy overlap.
"""
from datetime import timedelta
from pathlib import Path
import json
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from distillschedule.open_workload.role_program import RoleTask, PoolLayout, AllocatedWorld
from distillschedule.open_workload.role_transport import RoleFunctions, RolePoolExecutor
from test_open_workload_role_trainable_score import (
    make_model, warm_opt, first_data, generate, constructed, data,
    score_train, update_input, reference, full,
)


def _composition_worker(rank,rdzv,folder,invalid_scalar):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+rdzv,rank=rank,world_size=4,
                            timeout=timedelta(seconds=65))
    runner=None
    try:
        task=RoleTask(4,(2,4),score_updates=1 if invalid_scalar else 3)
        layout=PoolLayout((0,2),(1,3),2,'cpu_offload_prefetch','replicated','fsdp2_train')
        hw=AllocatedWorld((2**26,)*4,(0,0,1,1),(2**24,)*4)
        fn=RoleFunctions(make_model,warm_opt,first_data,generate,
                         score_train if invalid_scalar else None,score_train,
                         score_input_gradient=None if invalid_scalar else constructed,
                         sample_at=data,score_update_input=update_input)
        runner=RolePoolExecutor(task,layout,hw,fn,device='cpu')
        if invalid_scalar:
            # Do not reinterpret a scalar input-gradient protocol as no-grad
            # DMD to admit a candidate. All peers must exit before any send.
            with pytest.raises(ValueError,match='no-grad constructed gradient semantics'):
                runner.run()
            assert not runner.events
            runner.close()
            flags=[None]*4
            dist.all_gather_object(flags,True)
            assert flags==[True]*4
        else:
            for cycle in range(2):
                receipt,model,optimizer=runner.run()
                expected,expected_optimizer=reference(task,cycle+1,False)[runner.role]
                torch.testing.assert_close(full(model.state_dict()),full(expected.state_dict()),
                                           rtol=3e-5,atol=3e-6)
                torch.testing.assert_close(full(optimizer.state_dict()),full(expected_optimizer.state_dict()),
                                           rtol=5e-5,atol=5e-6)
                for row in receipt['ranks']:
                    assert row['host_leases_closed']
                    if row['role']=='generator':
                        assert row['offload']['packed_tensors']>0
                        assert row['offload']['unpacked_tensors']>0
                        assert row['offload']['prefetch_tensors']>0
                        assert row['offload']['peak_host_bytes']<=hw.host_capacities[row['rank']]
                    else:
                        assert row['sharding']['role']=='score_trainable_sharded'
                assert receipt['final_versions']['score']==3*(cycle+1)
        Path(folder,f'{rank}.json').write_text(json.dumps({'ok':True}))
    finally:
        if runner is not None:runner.close()
        dist.destroy_process_group()


@pytest.mark.parametrize('invalid_scalar',[False,True],ids=['offload-and-trainable-shards','reject-incompatible-gradient-protocol'])
def test_score_sharding_composition_is_native_and_collective_safe(tmp_path,invalid_scalar):
    mp.spawn(_composition_worker,args=(str(tmp_path/'rdzv'),str(tmp_path),invalid_scalar),nprocs=4,join=True)
    assert all(json.loads((tmp_path/f'{r}.json').read_text())['ok'] for r in range(4))
