"""Actual CPU Gloo/DTensor owners for the new native output publication callback.

Only projection/model math is injected here. This is not Wan math or world16;
the separate upstream test owns actual G/sampler/VAE coverage.
"""
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def owner(rank, rendezvous, directory, failed):
    from datetime import timedelta
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import distribute_tensor,DTensor,Shard
    from distillschedule.training import wan_native_inference as output
    from distillschedule.training.adapters import ModelComponents,InferenceProjection
    from distillschedule.open_workload.replicated import control_plane
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2,timeout=timedelta(seconds=60))
    root = Path(directory);calls=[]
    original_full = DTensor.full_tensor
    def full(tensor,*a,**kw):
        calls.append('full_tensor');return original_full(tensor,*a,**kw)
    DTensor.full_tensor = full
    original_save = torch.save
    def save(*a,**kw):
        assert rank == 0,'nonwriter attempted portable tensor publication'
        calls.append('writer')
        if failed:raise OSError('injected native publication disk failure')
        return original_save(*a,**kw)
    torch.save = save
    try:
        mesh = init_device_mesh('cpu',(2,))
        model = torch.nn.Linear(4,4,bias=False,dtype=torch.bfloat16)
        expected = torch.arange(16,dtype=torch.bfloat16).reshape(4,4)
        model.weight = torch.nn.Parameter(distribute_tensor(expected,mesh,[Shard(0)]),requires_grad=False)
        codec = torch.nn.Linear(1,1,bias=False)
        with torch.no_grad():codec.weight.fill_(.125)
        components = ModelComponents(state=dict(generator=output.CurrentGenerator(model),codec=codec),
            metadata=dict(scope='INJECTED_MODEL_MATH_REAL_CPU_DTENSOR_CALLBACK_ONLY'))
        def project(session, decoder, request, *, world_size):
            assert world_size == 2
            calls.append('projection')
            return InferenceProjection(components,output.FACTORY,dict(seed=721))
        output.project_current_generator = project
        request = output.NativeOutputRequest(dict(seed=721),root/'portable')
        with control_plane(dist.group.WORLD):
            if failed:
                with pytest.raises(ValueError,match='publication disk failure'):
                    request.publish(NS(),manifest=dict(cycle=100),identity=dict(scope='SYNTHETIC_CHECKPOINT_IDENTITY'))
            else:
                result = request.publish(NS(),manifest=dict(cycle=100),identity=dict(scope='SYNTHETIC_CHECKPOINT_IDENTITY'))
                assert result['world_size'] == 2 and result['checkpoint_cycle'] == 100
        assert calls.count('full_tensor') == 1 and calls.count('projection') == 1
        assert calls.count('writer') == (1 if rank == 0 else 0)
        (root/f'owner-{rank}.json').write_text(json.dumps(dict(rank=rank,calls=calls)))
        dist.barrier()
    finally:
        torch.save = original_save;DTensor.full_tensor = original_full
        dist.destroy_process_group()


@pytest.mark.parametrize('failed',[False,True],ids=['all_owners_rank0_publication','publication_failure_no_false_completion'])
def test_native_callback_all_owners_materialize_full_tensors_only_primary_publishes(tmp_path,failed):
    mp.spawn(owner,args=(str(tmp_path/'rendezvous'),str(tmp_path),failed),nprocs=2,join=True)
    if failed:
        assert not (tmp_path/'portable').exists() and not (tmp_path/'portable.inference.json').exists()
        stages = list(tmp_path.glob('.portable.tmp-*'))
        assert len(stages) == 1 and not (stages[0]/'_SUCCESS').exists()
        # Shared distributed publication intentionally preserves an explicitly
        # unpublished staging directory for failure audit, never a false PASS.
    else:
        from distillschedule.training.inference import read_export
        root,manifest = read_export(tmp_path/'portable')
        weights = torch.load(root/'weights.pt',weights_only=True)
        assert torch.equal(weights['generator']['model.weight'],torch.arange(16,dtype=torch.bfloat16).reshape(4,4))
        assert manifest['export']['roles'] == ['codec','generator']
        assert (tmp_path/'portable.inference.json').is_file()
    assert all((tmp_path/f'owner-{rank}.json').is_file() for rank in (0,1))
