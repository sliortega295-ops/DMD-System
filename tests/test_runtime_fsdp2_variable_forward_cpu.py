"""Actual CPU4 HSDP/PEFT update proof, not a large Helios/GPU qualification."""
from copy import deepcopy
from datetime import timedelta
import json
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('peft')
import torch.distributed as dist
import torch.multiprocessing as mp

from distillschedule.runtime.fsdp2_static import FSDP2EngineAdapter
from distillschedule.runtime.fsdp2_variable_forward import fsdp2_variable_forward_phase


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from peft.tuners.lora.layer import Linear
        self.projection = Linear(torch.nn.Linear(8, 8, bias=False, dtype=torch.bfloat16),
            adapter_name='default', r=2, lora_alpha=2, lora_dropout=0., init_lora_weights='gaussian')
        self.projection.base_layer.requires_grad_(False)
        for parameter in self.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
        with torch.no_grad():
            self.projection.lora_B['default'].weight.normal_(0, .1)

    def forward(self, x):
        y = self.projection(x)
        return x + torch.nn.functional.dropout(torch.tanh(y), .15, True)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # The FSDP unit is the complete checkpoint block, as in Helios and
        # the previously validated R172 topology; not an interior PEFT Linear.
        self.blocks = torch.nn.ModuleList([Block(), Block()])

    def forward(self, x):
        from torch.utils.checkpoint import checkpoint
        for block in self.blocks:
            def fn(value, block=block):
                parameters = getattr(block, 'phase_parameters', None)
                return block(value) if parameters is None else torch.func.functional_call(block, parameters, (value,))
            x = checkpoint(fn, x, use_reentrant=False, preserve_rng_state=True) if torch.is_grad_enabled() else fn(x)
        return x


def forward_prefix(engine, x, exits):
    calls = []
    for stage, exit_index in enumerate(exits):
        x = x * .8 + (stage + 1) * .01
        for index in range(exit_index + 1):
            with torch.set_grad_enabled(index == exit_index):
                x = engine(x)
                calls.append((stage, index, torch.is_grad_enabled()))
    return x, calls


def full(value):
    return value.full_tensor() if hasattr(value, 'full_tensor') else value


def worker(rank, rendezvous, output):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=4,
                            timeout=timedelta(seconds=40))
    rows = []
    try:
        torch.manual_seed(111)
        reference = Model()
        sharded = deepcopy(reference)
        mesh = init_device_mesh('cpu', (2, 2), mesh_dim_names=('replicate', 'shard'))
        policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        bad = deepcopy(reference)
        for block in bad.blocks:
            fully_shard(block.projection, mesh=mesh, mp_policy=policy, reshard_after_forward=True)
        fully_shard(bad, mesh=mesh, mp_policy=policy, reshard_after_forward=True)
        bad_optimizer = torch.optim.AdamW([p for p in bad.parameters() if p.requires_grad], lr=.003)
        bad_engine = FSDP2EngineAdapter(bad, bad_optimizer, None, physical_step=0)
        with pytest.raises(ValueError, match='checkpoint blocks'):
            with fsdp2_variable_forward_phase(bad_engine, enabled=True, with_grad=True,
                                             expected_units=3, checkpoint_block_type=Block):
                pytest.fail('interior PEFT sharding must fail before forward/materialization')
        del bad_engine, bad_optimizer, bad
        for block in sharded.blocks:
            fully_shard(block, mesh=mesh, mp_policy=policy, reshard_after_forward=True)
        fully_shard(sharded, mesh=mesh, mp_policy=policy, reshard_after_forward=True)
        reference_parameters = [p for p in reference.parameters() if p.requires_grad]
        parameters = [p for p in sharded.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=.003)
        reference_optimizer = torch.optim.AdamW(reference_parameters, lr=.003)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
        reference_scheduler = torch.optim.lr_scheduler.LambdaLR(reference_optimizer, lambda _: 1.)
        engine = FSDP2EngineAdapter(sharded, optimizer, scheduler, physical_step=0,
                                  global_batch=4, dp_world_size=4, gradient_clip=.1)
        engine._trajectory_capture_grad_norm = True
        exits = (((0, 1, 0), (2, 0, 3), (1, 2, 0), (3, 0, 1)),
                 ((3, 0, 2), (0, 2, 0), (0, 1, 3), (2, 0, 0)))
        for cycle in range(2):
            torch.manual_seed(700 + rank + cycle * 17)
            x = torch.randn(2, 8, dtype=torch.bfloat16)
            rng = torch.get_rng_state().clone()
            for block in reference.blocks:
                # One differentiable BF16 compute copy per phase, as in the
                # FSDP mixed-precision reference, not per-call FP32 accumulation.
                block.phase_parameters = {name: p.to(torch.bfloat16) for name, p in block.named_parameters()}
            y, calls = forward_prefix(reference, x.clone(), exits[cycle][rank])
            loss = y.float().square().mean()
            loss.backward()
            after = torch.get_rng_state().clone()
            for p in reference_parameters:
                dist.all_reduce(p.grad)
                p.grad.div_(4)
            gradients = [p.grad.clone() for p in reference_parameters]
            norm = torch.nn.utils.clip_grad_norm_(reference_parameters, .1)
            reference_optimizer.step(); reference_scheduler.step()

            torch.set_rng_state(rng)
            with fsdp2_variable_forward_phase(engine, enabled=True, with_grad=True,
                    expected_units=3, checkpoint_block_type=Block) as (view, receipt):
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
                    actual, actual_calls = forward_prefix(view, x.clone(), exits[cycle][rank])
                    actual_loss = actual.float().square().mean()
                view.backward(actual_loss)
                assert calls == actual_calls and torch.equal(after, torch.get_rng_state())
                # Inspect after phase.release, at the ORIGINAL engine entry;
                # those are the gradients the real clip/Adam actually consume.
                original_step = engine.step
                def observed_step():
                    for p, expected in zip(parameters, gradients, strict=True):
                        torch.testing.assert_close(full(p.grad), expected, rtol=1e-6, atol=1e-7)
                    original_step()
                engine.step = observed_step
                try:
                    view.step()
                finally:
                    engine.step = original_step
            torch.testing.assert_close(actual, y, rtol=1e-6, atol=1e-7)
            torch.testing.assert_close(actual_loss, loss, rtol=1e-6, atol=1e-7)
            torch.testing.assert_close(torch.tensor(engine.last_gradient_norm), norm, rtol=1e-6, atol=1e-7)
            for p, expected in zip(sharded.parameters(), reference.parameters(), strict=True):
                torch.testing.assert_close(full(p), expected, rtol=1e-6, atol=1e-7)
            for p, expected in zip(parameters, reference_parameters, strict=True):
                for key in ('step', 'exp_avg', 'exp_avg_sq'):
                    torch.testing.assert_close(full(optimizer.state[p][key]), reference_optimizer.state[expected][key],
                                               rtol=1e-6, atol=1e-7)
            assert scheduler.state_dict() == reference_scheduler.state_dict()
            assert engine.global_steps == cycle + 1 and all(p.grad is None for p in parameters)
            events = {e.key: e.count for e in profile.key_averages() if 'all_gather' in e.key.lower() or 'allgather' in e.key.lower()}
            assert events == {}, events
            rows.append(dict(cycle=cycle + 1, calls=len(calls), selected_calls=3, rng_equal=True,
                             parameter_gathers_in_variable_forward=events, phase=receipt))
            reference_optimizer.zero_grad(set_to_none=True)
        assert not torch.cuda.is_initialized()
    finally:
        dist.destroy_process_group()
        Path(output, f'rank{rank}.json').write_text(json.dumps(dict(scope='SYNTHETIC_CPU4_REAL_PEFT_HSDP_TEST_ONLY',
            rows=rows, cuda_initialized=torch.cuda.is_initialized(), process_group_destroyed=not dist.is_initialized())))


def test_actual_hsdp_peft_unequal_prefix_original_adam_and_rng(tmp_path):
    assert not torch.cuda.is_initialized()
    mp.spawn(worker, args=(str(tmp_path / 'gloo'), str(tmp_path)), nprocs=4, join=True)
    reports = [json.loads((tmp_path / f'rank{rank}.json').read_text()) for rank in range(4)]
    assert all(len(row['rows']) == 2 and row['process_group_destroyed'] and not row['cuda_initialized'] for row in reports)
