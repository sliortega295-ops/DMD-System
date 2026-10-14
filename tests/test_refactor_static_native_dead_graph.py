"""Native no-grad denoising can sever an earlier selected pyramid graph."""
import copy
from datetime import timedelta
import gc

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from test_refactor_static_gpipe_autograd import _Half


def _run(rank, rendezvous):
    from distillschedule.runtime.static_gpipe_external_loss import build_external_loss_gpipe
    from distillschedule.runtime.static_gpipe_autograd import NativeBatchGPipeAutograd

    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + rendezvous,
        rank=rank, world_size=2, timeout=timedelta(seconds=80))
    try:
        torch.manual_seed(719)
        original = torch.nn.Sequential(_Half(5), _Half(5))
        local = copy.deepcopy(original[rank])
        value = torch.randn(8, 5, dtype=torch.float64)
        first = original(value)
        # Exactly the native no-grad solver's ownership rule: the following
        # update loses the prior selected-forward autograd dependency.
        with torch.no_grad():
            intermediate = first * .7 + value * .3
        del first
        middle = original(intermediate)
        expected = original(middle * .6 + intermediate * .4)
        expected.square().mean().backward()

        calls = []
        for _ in range(3):
            with torch.no_grad():
                hidden = original[0](value[:1])
            schedule = build_external_loss_gpipe(local, stage_index=rank,
                group=dist.group.WORLD, device='cpu',
                input_args=(value[:1] if rank == 0 else hidden,),
                output_args=(hidden if rank == 0 else expected[:1].detach(),),
                final_output_args=(expected[:1].detach(),))
            calls.append(NativeBatchGPipeAutograd(schedule,
                stage_index=rank, group=dist.group.WORLD, device='cpu'))

        discarded = calls[0](value)
        assert calls[0].schedule.pending
        with torch.no_grad():
            intermediate = discarded * .7 + value * .3
        del discarded
        gc.collect()
        middle = calls[1](intermediate)
        actual = calls[2](middle * .6 + intermediate * .4)
        actual.square().mean().backward()
        pending = [call.schedule.pending for call in calls]
        print('OBSERVED_NATIVE_SELECTED3_PENDING_AFTER_BACKWARD', rank, pending, flush=True)
        assert pending == [True, False, False]
        torch.testing.assert_close(actual.detach(), expected.detach(), rtol=1e-12, atol=1e-12)
        for param, ref in zip(local.parameters(), original[rank].parameters(), strict=True):
            torch.testing.assert_close(param.grad, ref.grad, rtol=1e-10, atol=1e-12)
        # The fix must retire only a proved-unreachable graph, with no fake
        # backward and no extra model evaluation or parameter gradient.
        assert calls[0].schedule.retire_unreachable_graph()
        assert not calls[0].schedule.pending
        receipt = calls[0].schedule.receipt()
        assert receipt['completed_forward_backward_batches'] == 0
        assert receipt['discarded_unreachable_forward_batches'] == 1
        assert not calls[0].schedule.retire_unreachable_graph()

        # Never retire a live graph to make the cycle gate pass.
        live = calls[0](value)
        assert not calls[0].schedule.retire_unreachable_graph()
        assert calls[0].schedule.pending
        live.square().mean().backward()
        assert not calls[0].schedule.pending
        assert not torch.cuda.is_initialized()
    finally:
        dist.destroy_process_group()


def test_native_no_grad_cut_retires_only_unreachable_pp_graph(tmp_path):
    mp.spawn(_run, args=(str(tmp_path / 'dead_native_graph'),), nprocs=2, join=True)
