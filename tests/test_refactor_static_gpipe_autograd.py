"""GPipe8 native batch-loss scaling and chained pyramid backward, CPU only."""
import copy
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.checkpoint import checkpoint


class _Half(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.linear = torch.nn.Linear(width, width, dtype=torch.float64)

    def forward(self, value):
        return checkpoint(lambda x: self.linear(x).tanh(), value, use_reentrant=False)


def _run(rank, rendezvous):
    from distillschedule.runtime.static_gpipe_external_loss import build_external_loss_gpipe
    from distillschedule.runtime.static_gpipe_autograd import NativeBatchGPipeAutograd
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank,
        world_size=2, timeout=timedelta(seconds=80))
    try:
        torch.manual_seed(195)
        original = torch.nn.Sequential(_Half(5), _Half(5))
        local = copy.deepcopy(original[rank])
        value = torch.randn(8, 5, dtype=torch.float64, requires_grad=True)
        x = value.detach().clone().requires_grad_(True)
        expected_a = original(value)
        expected = original(expected_a * .7 + value * .3)
        external = expected.detach().cos() * .2
        ((expected - (expected.detach() - external)).square()).mean().backward()
        native_calls = []
        for _ in range(2):
            hidden = original[0](value[:1]).detach()
            schedule = build_external_loss_gpipe(local, stage_index=rank,
                group=dist.group.WORLD, device='cpu',
                input_args=(value[:1].detach() if rank == 0 else hidden,),
                output_args=(hidden if rank == 0 else expected[:1].detach(),),
                final_output_args=(expected[:1].detach(),))
            native_calls.append(NativeBatchGPipeAutograd(schedule, stage_index=rank,
                group=dist.group.WORLD, device='cpu'))
        actual_a = native_calls[0](x)
        actual = native_calls[1](actual_a * .7 + x * .3)
        torch.testing.assert_close(actual.detach(), expected.detach(), rtol=1e-12, atol=1e-12)
        # Identical native MSE over batch8, not eight separately normalized losses.
        ((actual - (actual.detach() - external)).square()).mean().backward()
        torch.testing.assert_close(x.grad, value.grad, rtol=1e-10, atol=1e-12)
        for p, ref in zip(local.parameters(), original[rank].parameters(), strict=True):
            torch.testing.assert_close(p.grad, ref.grad, rtol=1e-10, atol=1e-12)
        assert not torch.cuda.is_initialized()
    finally:
        dist.destroy_process_group()


def test_gpipe_native_mean8_chained_input_gradients(tmp_path):
    mp.spawn(_run, args=(str(tmp_path/'gloo_native_chained'),), nprocs=2, join=True)
