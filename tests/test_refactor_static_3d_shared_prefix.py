"""Actual16-process CPU proof of shared trainable prefix across PP2 halves."""
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(rank, rendezvous):
    from distillschedule.runtime.static_3d_layout import Static3DLayout
    from distillschedule.runtime.static_3d_state import (
        StaticDataParallelState, StaticParameterPlacement,
    )
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + rendezvous,
        rank=rank, world_size=16, timeout=timedelta(seconds=150))
    try:
        layout = Static3DLayout()
        pp, dp, tp = layout.coordinate(rank)
        pp_group = dp_group = model_group = None
        for members in layout.groups()['pp']:
            group = dist.new_group(members)
            if rank in members:
                pp_group = group
        for members in layout.groups()['dp']:
            group = dist.new_group(members)
            if rank in members:
                dp_group = group
        for d in (0, 1):
            members = [layout.physical_rank(p, d, t) for p in (0, 1) for t in range(4)]
            group = dist.new_group(members)
            if rank in members:
                model_group = group
        prefix = torch.nn.Parameter(torch.ones(2))
        local = torch.nn.Parameter(torch.ones(2))
        placements = [StaticParameterPlacement('condition', prefix, 'replicated', None, pp_replicated=True),
                      StaticParameterPlacement('block', local, 'output', 0)]
        state = StaticDataParallelState(placements, dp_group=dp_group,
            model_group=model_group, pp_group=pp_group, max_norm=1.0)
        coefficient = torch.tensor([0.3, -0.7]) * (pp + 1)
        template = torch.arange(1, 9, dtype=torch.float32) * (pp + 1)
        for micro in range(8):
            logical = layout.logical_sample_rank(dp, micro)
            loss = (prefix.dot(coefficient) + local.dot(template.chunk(4)[tp])) * (logical + 1)
            state.mean_microbatch_loss(loss).backward()
        norm = state.synchronize_and_clip()
        expected_prefix = torch.tensor([0.3, -0.7]) * 3 * 8.5
        unique_norm_squared = sum((torch.arange(1, 9, dtype=torch.float32) * (p + 1) * 8.5).square().sum()
                                  for p in (0, 1))
        expected_norm = (expected_prefix.square().sum() + unique_norm_squared).sqrt()
        factor = (1.0 / (expected_norm + 1e-6)).clamp(max=1)
        torch.testing.assert_close(norm, expected_norm, rtol=3e-6, atol=3e-6)
        torch.testing.assert_close(prefix.grad, expected_prefix * factor, rtol=3e-6, atol=3e-7)
        torch.testing.assert_close(local.grad, template.chunk(4)[tp] * 8.5 * factor, rtol=3e-6, atol=3e-7)
        assert state.receipt()['pp_shared_trainables'] == 1
        assert state.receipt()['commits'] == 1
        assert not torch.cuda.is_initialized()
    finally:
        dist.destroy_process_group()


def test_actual16rank_shared_prefix_sum_and_global_clip(tmp_path):
    mp.spawn(_worker, args=(str(tmp_path / 'shared_prefix'),), nprocs=16, join=True)
