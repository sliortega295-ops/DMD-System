"""Actual PyTorch CPU PP2 forward/backward fixture, never a speed surrogate."""
from datetime import timedelta
import copy
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _cpu_pp(rank,rendezvous):
    from distillschedule.runtime.static_pipeline import build_static_gpipe
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2,
        timeout=timedelta(seconds=90))
    try:
        torch.manual_seed(921)
        dense=torch.nn.Sequential(torch.nn.Linear(5,8),torch.nn.Tanh(),torch.nn.Linear(8,3))
        x=torch.randn(8,5)
        target=torch.randn(8,3)
        expected=dense(x)
        loss=torch.nn.functional.mse_loss(expected,target)
        loss.backward()
        module=copy.deepcopy(dense[:2] if rank==0 else dense[2:])
        module.zero_grad(set_to_none=True)
        inputs=x[:1] if rank==0 else dense[:2](x[:1]).detach()
        outputs=dense[:2](x[:1]).detach() if rank==0 else expected[:1].detach()
        _,schedule=build_static_gpipe(module,stage_index=rank,group=dist.group.WORLD,
            device=torch.device('cpu'),input_args=(inputs,),output_args=(outputs,),
            loss_fn=torch.nn.functional.mse_loss)
        if rank==0:
            schedule.step(x)
        else:
            actual=schedule.step(target=target)
            torch.testing.assert_close(actual,expected,rtol=2e-6,atol=2e-7)
        reference=list((dense[:2] if rank==0 else dense[2:]).parameters())
        for p,r in zip(module.parameters(),reference):
            torch.testing.assert_close(p.grad,r.grad,rtol=3e-6,atol=2e-7)
        assert not torch.cuda.is_initialized()
    finally:
        dist.destroy_process_group()


def test_real_cpu_pp2_gpipe_eight_microbatch_backward(tmp_path):
    mp.spawn(_cpu_pp,args=(str(tmp_path/'gloo_pp'),),nprocs=2,join=True)
