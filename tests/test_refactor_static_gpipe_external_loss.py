"""Real GPipe8 external DMD-loss seam; CPU is not a performance result."""
import copy
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.checkpoint import checkpoint


class _CheckpointHalf(torch.nn.Module):
    def __init__(self, width_in, width_out):
        super().__init__()
        self.linear = torch.nn.Linear(width_in, width_out, dtype=torch.float64)

    def forward(self, x):
        return checkpoint(lambda y: self.linear(y).tanh(), x, use_reentrant=False)


def _external_loss_worker(rank, rendezvous):
    from distillschedule.runtime.static_gpipe_external_loss import build_external_loss_gpipe
    from distillschedule.runtime.static_pipeline import build_static_gpipe
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + rendezvous,
        rank=rank, world_size=2, timeout=timedelta(seconds=80))
    try:
        torch.manual_seed(918)
        generator = torch.nn.Sequential(_CheckpointHalf(5, 8), _CheckpointHalf(8, 3))
        fake = torch.nn.Sequential(torch.nn.Linear(3, 6, dtype=torch.float64),
                                   torch.nn.Linear(6, 3, dtype=torch.float64))
        teacher = copy.deepcopy(fake)
        teacher[1].weight.data.add_(.02)
        x = torch.randn(8, 5, dtype=torch.float64)
        g_reference = generator(x)
        with torch.no_grad():
            gradient = fake(g_reference.detach()) - teacher(g_reference.detach())
        ((g_reference * gradient).sum(1)).mean().backward()
        g_half = copy.deepcopy(generator[rank]); g_half.zero_grad(set_to_none=True)
        hidden = generator[0](x[:1]).detach()
        gp = build_external_loss_gpipe(g_half, stage_index=rank, group=dist.group.WORLD,
            device='cpu', input_args=(x[:1] if rank == 0 else hidden,),
            output_args=(hidden if rank == 0 else g_reference[:1].detach(),),
            final_output_args=(g_reference[:1].detach(),))
        # Generator's eight graphs stay live while *other whole PP roles* run.
        g_actual = gp.forward_batch(x)
        torch.testing.assert_close(g_actual, g_reference.detach(), rtol=1e-12, atol=1e-12)
        score_outputs = []
        for role in (fake, teacher):
            module = copy.deepcopy(role[rank])
            hidden_score = role[0](g_actual[:1]).detach()
            _, score = build_static_gpipe(module, stage_index=rank, group=dist.group.WORLD,
                device='cpu', input_args=(g_actual[:1] if rank == 0 else hidden_score,),
                output_args=(hidden_score if rank == 0 else role(g_actual[:1]).detach(),))
            with torch.no_grad():
                output = score.step(g_actual.detach())
                transported = output.detach().clone() if rank == 1 else torch.empty(8, 3, dtype=torch.float64)
                dist.broadcast(transported, src=1)
            score_outputs.append(transported)
        external_gradient = score_outputs[0] - score_outputs[1]
        losses = ([(output * external_gradient[i:i+1]).sum()
                   for i, output in enumerate(gp.last_stage_microbatch_outputs())]
                  if rank == 1 else None)
        gp.backward_losses(losses)
        for actual, expected in zip(g_half.parameters(), generator[rank].parameters()):
            torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-10, atol=1e-12)
        assert gp.receipt()['completed_forward_backward_batches'] == 1
        try:
            gp.backward_losses(losses)
        except RuntimeError as e:
            assert 'pending' in str(e)
        else:
            raise AssertionError('second backward must fail closed')
        assert not torch.cuda.is_initialized()
    finally:
        dist.destroy_process_group()


def test_actual_gpipe8_other_roles_before_generator_checkpoint_backward(tmp_path):
    mp.spawn(_external_loss_worker, args=(str(tmp_path / 'gloo_external_gpipe'),), nprocs=2, join=True)
