"""Real16-rank CPU DP/TP/PP norm and one-commit state reference."""
from datetime import timedelta
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _rank_state(rank,rendezvous):
    from distillschedule.runtime.static_3d_layout import Static3DLayout
    from distillschedule.runtime.static_3d_state import StaticDataParallelState,StaticParameterPlacement
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=16,
        timeout=timedelta(seconds=150))
    try:
        layout=Static3DLayout()
        pp,dp,tp=layout.coordinate(rank)
        dp_group=None;model_group=None
        # Every rank creates every subgroup in the same order.
        for members in layout.groups()['dp']:
            group=dist.new_group(members)
            if rank in members:dp_group=group
        for d in (0,1):
            members=[layout.physical_rank(p,d,t) for p in (0,1) for t in range(4)]
            group=dist.new_group(members)
            if rank in members:model_group=group
        replicated=torch.nn.Parameter(torch.ones(2,dtype=torch.float32))
        partitioned=torch.nn.Parameter(torch.ones(2,dtype=torch.float32))
        items=[StaticParameterPlacement('replicated_A',replicated,'replicated',None),
               StaticParameterPlacement('partitioned_B',partitioned,'output',0)]
        state=StaticDataParallelState(items,dp_group=dp_group,model_group=model_group,max_norm=1.25)
        global_template=torch.arange(1,9,dtype=torch.float32)*(pp+1)
        local_template=global_template.chunk(4)[tp]
        rep_template=torch.tensor([0.5,-1.0],dtype=torch.float32)*(pp+1)
        for micro in range(8):
            logical_rank=layout.logical_sample_rank(dp,micro)
            loss=(replicated.dot(rep_template)+partitioned.dot(local_template))*(logical_rank+1)
            state.mean_microbatch_loss(loss).backward()
        assert state.accumulations==8
        norm=state.synchronize_and_clip()
        expected_rep=rep_template*8.5
        expected_part=global_template*8.5
        squared=sum(((torch.arange(1,9,dtype=torch.float32)*(p+1)*8.5).square().sum()
            +(torch.tensor([0.5,-1.0])*(p+1)*8.5).square().sum()) for p in (0,1))
        expected_norm=squared.sqrt()
        coefficient=(1.25/(expected_norm+1e-6)).clamp(max=1)
        torch.testing.assert_close(norm,expected_norm,rtol=3e-6,atol=3e-6)
        torch.testing.assert_close(replicated.grad,expected_rep*coefficient,rtol=3e-6,atol=3e-7)
        torch.testing.assert_close(partitioned.grad,expected_part.chunk(4)[tp]*coefficient,rtol=3e-6,atol=3e-7)
        reference_rep=torch.nn.Parameter(torch.ones_like(replicated))
        reference_part=torch.nn.Parameter(torch.ones(8,dtype=torch.float32))
        reference_rep.grad=expected_rep*coefficient
        reference_part.grad=expected_part*coefficient
        actual_optimizer=torch.optim.AdamW([replicated,partitioned],lr=1e-3,betas=(0.0,0.999),weight_decay=0.01)
        reference_optimizer=torch.optim.AdamW([reference_rep,reference_part],lr=1e-3,betas=(0.0,0.999),weight_decay=0.01)
        actual_optimizer.step();reference_optimizer.step()
        torch.testing.assert_close(replicated,reference_rep,rtol=2e-6,atol=2e-7)
        torch.testing.assert_close(partitioned,reference_part.chunk(4)[tp],rtol=2e-6,atol=2e-7)
        assert state.receipt()['global_batch']==16 and state.commits==1 and state.accumulations==0
        assert not torch.cuda.is_initialized()
    finally:
        dist.destroy_process_group()


def test_actual_cpu_world16_dp_mean_global_norm_and_single_commit(tmp_path):
    mp.spawn(_rank_state,args=(str(tmp_path/'gloo_state'),),nprocs=16,join=True)


def test_explicit_tp_bootstrap_slice_and_unplaced_trainable_rejection():
    from distillschedule.runtime.static_3d_state import (
        StaticParameterPlacement,placed_trainables,shard_bootstrap_tensor,
    )
    import pytest
    parameter=torch.nn.Parameter(torch.ones(2,3))
    placement=StaticParameterPlacement('B',parameter,'output',0)
    full=torch.arange(24,dtype=torch.float32).reshape(8,3)
    torch.testing.assert_close(shard_bootstrap_tensor(full,placement,2),full[4:6])
    with pytest.raises(ValueError,match='explicit binding'):
        placed_trainables(torch.nn.Linear(3,2))
    with pytest.raises(ValueError,match='partition dimension'):
        StaticParameterPlacement('bad',parameter,'output',1)


def test_native_root_lora_requires_exact_explicit_replication_names():
    from distillschedule.runtime.static_3d_state import placed_trainables
    import pytest
    root=torch.nn.Linear(3,2,dtype=torch.float32)
    placements=placed_trainables(root,replicated_root_names=('weight','bias'))
    assert {p.name for p in placements}=={'weight','bias'}
    assert all(p.tp_placement=='replicated' and not p.pp_replicated for p in placements)
    with pytest.raises(ValueError,match='duplicate'):
        placed_trainables(root,replicated_root_names=('weight','weight'))
    with pytest.raises(ValueError,match='missing'):
        placed_trainables(root,replicated_root_names=('weight','unknown'))
