"""Actual CPU PP2 external-loss/checkpoint backward vs one dense reference.

Small mathematical fixtures validate transport only; never timing surrogates.
"""
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.checkpoint import checkpoint


class CheckpointedPart(torch.nn.Module):
    def __init__(self, part):
        super().__init__()
        self.part = part

    def forward(self, value):
        return checkpoint(self.part, value, use_reentrant=False)


def _make_role(seed):
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.Tanh(),
                                   torch.nn.Linear(5, 3)).double()


def _worker(rank, rendezvous):
    from distillschedule.runtime.static_pipeline import (
        StaticExternalGradientPP2, StaticPipelineSignature,
    )
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + rendezvous,
        rank=rank, world_size=2, timeout=timedelta(seconds=90))
    try:
        roles = {}
        references = {}
        for name, seed in (('student', 13), ('fake', 23), ('teacher', 31)):
            full = _make_role(seed)
            references[name] = _make_role(seed)
            part = full[:2] if rank == 0 else full[2]
            roles[name] = StaticExternalGradientPP2(CheckpointedPart(part),
                stage_index=rank, group=dist.group.WORLD, device='cpu',
                signatures={'B1': StaticPipelineSignature(hidden_shape=(1, 5),
                    hidden_dtype=torch.float64, output_shape=(1, 3), output_dtype=torch.float64)})
            if name == 'teacher':
                roles[name].requires_grad_(False)
                references[name].requires_grad_(False)
        x = torch.tensor([[0.25, -0.5, 0.125]], dtype=torch.float64)
        generated = roles['student'](x, signature_key='B1')
        expected = references['student'](x)
        with torch.no_grad():
            fake = roles['fake'](generated.detach(), signature_key='B1')
            teacher = roles['teacher'](generated.detach(), signature_key='B1')
            ref_fake = references['fake'](expected.detach())
            ref_teacher = references['teacher'](expected.detach())
            target = (generated - (fake - teacher)).detach()
            ref_target = (expected - (ref_fake - ref_teacher)).detach()
        loss = (generated - target).square().mean()
        ref_loss = (expected - ref_target).square().mean()
        loss.backward()
        ref_loss.backward()
        torch.testing.assert_close(generated, expected, rtol=2e-12, atol=2e-12)
        reference_part = references['student'][:2] if rank == 0 else references['student'][2]
        for actual, ref in zip(roles['student'].stage_module.part.parameters(), reference_part.parameters()):
            torch.testing.assert_close(actual.grad, ref.grad, rtol=2e-12, atol=2e-12)
        assert all(p.grad is None for p in roles['fake'].parameters())
        # Five independent Fake graphs, retaining their original update order.
        for index in range(5):
            values = (x + index / 10).detach()
            actual = roles['fake'](values, signature_key='B1')
            expected = references['fake'](values)
            actual.square().mean().backward()
            expected.square().mean().backward()
            reference_part = references['fake'][:2] if rank == 0 else references['fake'][2]
            for actual_param, reference_param in zip(roles['fake'].stage_module.part.parameters(), reference_part.parameters()):
                torch.testing.assert_close(actual_param.grad, reference_param.grad, rtol=2e-12, atol=2e-12)
            for model in (roles['fake'], references['fake']):
                with torch.no_grad():
                    for parameter in model.parameters():
                        parameter.add_(parameter.grad, alpha=-1e-3)
                model.zero_grad(set_to_none=True)
        assert roles['student'].grad_forwards == 1
        assert roles['fake'].application_forwards == 6 and roles['fake'].grad_forwards == 5
        assert roles['teacher'].application_forwards == 1 and roles['teacher'].grad_forwards == 0
        assert not torch.cuda.is_initialized()
    finally:
        dist.destroy_process_group()


def test_cpu_pp2_multirole_external_loss_and_checkpoint_recompute(tmp_path):
    mp.spawn(_worker, args=(str(tmp_path / 'external_pp2'),), nprocs=2, join=True)
