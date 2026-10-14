"""Eight B1 samples/DP in four GPipe2 groups: no workload reduction."""
from __future__ import annotations

import copy
from datetime import timedelta
import hashlib
import importlib.util
import inspect
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _run(rank, rendezvous):
    from distillschedule.runtime.static_native_forward import create_admitted_forward_split
    from distillschedule.runtime.static_native_pp_stage import NativeSourcePPStage
    from distillschedule.runtime.static_native_proxy import NativeStaticPPProxy
    path = Path(__file__).with_name('test_refactor_static_native_proxy.py')
    spec = importlib.util.spec_from_file_location('static_seq_fixture', path)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + rendezvous,
        rank=rank, world_size=2, timeout=timedelta(seconds=100))
    try:
        torch.manual_seed(846)
        reference = fixture._Native()
        native = copy.deepcopy(reference)
        x = torch.randn(8, 2, 5, dtype=torch.bfloat16)
        text = torch.randn_like(x)
        scalar = torch.zeros(8, 1)
        split = create_admitted_forward_split(native,
            expected_source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            expected_blocks=4)
        with torch.no_grad():
            context = split.prefix(native, x[:1], scalar[:1], text[:1], return_dict=False)
        context['return_dict'] = False
        stage = NativeSourcePPStage(native, split=split, stage_index=rank,
            context_template=context, remove_root_modules=('proj_out',) if rank == 0
                else ('patch_embedding', 'condition_embedder'))
        proxy = NativeStaticPPProxy(stage, pp_group=dist.group.WORLD, device='cpu',
                                   role='generator', pipeline_microbatches=2)
        for index in range(0, 8, 2):
            row = dict(hidden_states=x[index:index+2], timestep=scalar[index:index+2],
                encoder_hidden_states=text[index:index+2], return_dict=False)
            for name in proxy.INPUT_NAMES:
                if name not in row:
                    row[name] = scalar[index:index+2]
            expected = torch.cat([reference(x[i:i+1], scalar[i:i+1], text[i:i+1])[0]
                                  for i in range(index, index + 2)])
            actual = proxy(**row)[0]
            assert torch.equal(actual, expected)
            (expected.float().square().mean() / 4).backward()
            (actual.float().square().mean() / 4).backward()
            assert not any(getattr(s, 'pending', False) for s, _ in proxy._schedules.values())
        for name, parameter in native.named_parameters():
            torch.testing.assert_close(parameter.grad, dict(reference.named_parameters())[name].grad,
                                       rtol=.03, atol=.0005)
        receipt = proxy.receipt()
        assert receipt['actual_stage_B1_forward_count'] == 8
        assert receipt['application_forward_per_logical_sample'] == 1
        assert receipt['grad_forward_per_logical_sample'] == 1
        assert receipt['pending_backward_schedules'] == 0
        assert receipt['pipeline_microbatches_per_invocation'] == 2
        assert receipt['logical_microbatches_per_DP'] == 8
        assert sum(r['completed_forward_backward_batches'] for r in receipt['gradient_transport']) == 4
        assert not torch.cuda.is_initialized()
    finally:
        dist.destroy_process_group()


def test_two_live_graphs_preserve_eight_independent_B1_losses_and_gradients(tmp_path):
    mp.spawn(_run, args=(str(tmp_path / 'pp2'),), nprocs=2, join=True)


def test_explicit_pipeline_microbatch_selector_keeps_default_eight():
    from distillschedule.runtime.static_native_proxy import NativeStaticPPProxy
    signature = inspect.signature(NativeStaticPPProxy)
    assert signature.parameters['pipeline_microbatches'].default == 8


def test_dmd_per_sample_normalization_and_mean8_gradient_survive_four_pairs():
    from runtime.dmd_boundary import dmd_input_gradient, detached_dmd_surrogate_loss
    torch.manual_seed(591)
    x0 = torch.randn(8, 2, 1, 2, 3, dtype=torch.float64, requires_grad=True)
    noisy = torch.randn_like(x0)
    flows = [torch.randn_like(x0) for _ in range(3)]
    timestep = torch.arange(8, dtype=torch.float64)
    sigmas = torch.linspace(.1, .8, 8, dtype=torch.float64)

    def gradient(sl):
        return dmd_input_gradient(estimated_clean=x0[sl], noisy_sample=noisy[sl],
            timestep=timestep[sl], sigmas=sigmas, timesteps=timestep,
            fake_flow_cond=flows[0][sl], real_flow_cond=flows[1][sl],
            real_flow_uncond=flows[2][sl])[0].detach()

    whole = gradient(slice(None))
    detached_dmd_surrogate_loss(x0, whole).backward()
    expected = x0.grad.clone()
    x0.grad = None
    for i in range(0, 8, 2):
        sl = slice(i, i + 2)
        pair = gradient(sl)
        assert torch.equal(pair, whole[sl])
        (detached_dmd_surrogate_loss(x0[sl], pair) / 4).backward()
    torch.testing.assert_close(x0.grad, expected, rtol=0, atol=0)
    assert not torch.cuda.is_initialized()


def test_group_mean_scaling_requires_all_eight_samples_before_commit():
    from distillschedule.runtime.static_3d_state import StaticDataParallelState
    state = object.__new__(StaticDataParallelState)
    state.accumulations = 0
    parameter = torch.nn.Parameter(torch.tensor(3.))
    for start in range(0, 8, 2):
        pair_mean = parameter * torch.tensor([start + 1., start + 2.]).mean()
        state.mean_microbatch_group_loss(pair_mean, group_size=2).backward()
    assert state.accumulations == 8
    assert parameter.grad == torch.tensor(4.5)
    with pytest.raises(ValueError, match='eight'):
        state.mean_microbatch_group_loss(parameter, group_size=2)
    for count in (1, 4, True):
        state.accumulations = 0
        with pytest.raises(ValueError, match='two or eight'):
            state.mean_microbatch_group_loss(parameter, group_size=count)


def test_pair_dispatch_preserves_rank_local_calls_and_rng():
    from distillschedule.runtime.static_native_cooperative import run_native_B1_cohort
    source = torch.arange(10., dtype=torch.float32).reshape(2, 5).requires_grad_()
    def model(**kwargs):
        assert kwargs['hidden_states'].shape[0] == 2
        return (kwargs['hidden_states'] * kwargs['timestep'].reshape(-1, 1),)
    def sample(view, index):
        return view(hidden_states=source[index:index+1],
            timestep=torch.tensor([index + 2.]), return_dict=False)[0]
    rng = torch.get_rng_state().clone()
    outputs, receipt = run_native_B1_cohort(model, sample, device='cpu', cohort_size=2)
    assert torch.equal(rng, torch.get_rng_state())
    assert torch.equal(torch.cat(outputs), source * torch.tensor([[2.], [3.]]))
    torch.cat(outputs).sum().backward()
    assert torch.equal(source.grad, torch.tensor([[2.] * 5, [3.] * 5]))
    assert receipt['logical_samples'] == 2
    assert receipt['physical_model_batch'] == 1
    assert receipt['per_sample_call_counts'] == [1, 1]

