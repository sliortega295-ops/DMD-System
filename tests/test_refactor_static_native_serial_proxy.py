"""Actual Gloo PP2, native checkpoint gradients and mean8, no GPU claim."""
import copy
from datetime import timedelta
import hashlib
import importlib.util
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _run(rank, rendezvous):
    from distillschedule.runtime.static_native_forward import create_admitted_forward_split
    from distillschedule.runtime.static_native_pp_stage import NativeSourcePPStage
    from distillschedule.runtime.static_native_serial_proxy import NativeSerialPPProxy
    path = Path(__file__).with_name('test_refactor_static_native_proxy.py')
    spec = importlib.util.spec_from_file_location('serial_fixture', path)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank,
                            world_size=2, timeout=timedelta(seconds=100))
    try:
        torch.manual_seed(846)
        x = torch.randn(8, 2, 5, dtype=torch.bfloat16)
        text = torch.randn_like(x)
        scalar = torch.zeros(8, 1)
        roles = {}
        references = {}
        for role in ('generator', 'critic'):
            reference = fixture._Native()
            native = copy.deepcopy(reference)
            split = create_admitted_forward_split(native,
                expected_source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), expected_blocks=4)
            with torch.no_grad():
                context = split.prefix(native, x[:1], scalar[:1], text[:1], return_dict=False)
            context['return_dict'] = False
            stage = NativeSourcePPStage(native, split=split, stage_index=rank,
                context_template=context, remove_root_modules=('proj_out',) if rank == 0
                    else ('patch_embedding', 'condition_embedder'))
            roles[role] = NativeSerialPPProxy(stage, pp_group=dist.group.WORLD, device='cpu', role=role)
            references[role] = reference

        def call(role, value, i):
            row = dict(hidden_states=value, timestep=scalar[i:i+1], encoder_hidden_states=text[i:i+1],
                       return_dict=False)
            for name in roles[role].INPUT_NAMES:
                if name not in row:
                    row[name] = scalar[i:i+1]
            return roles[role](**row)[0]

        for i in range(8):
            # Two differentiable pyramid calls, unrelated score forward,
            # then backward: preserve both earlier native input gradients.
            a = x[i:i+1].clone().requires_grad_()
            b = x[i:i+1].clone().requires_grad_()
            expected = references['generator'](a, scalar[i:i+1], text[i:i+1])[0]
            expected = references['generator'](expected, scalar[i:i+1], text[i:i+1])[0]
            actual = call('generator', b, i)
            actual = call('generator', actual, i)
            assert torch.equal(actual, expected)
            with torch.no_grad():
                score_ref = references['critic'](expected, scalar[i:i+1], text[i:i+1])[0]
                score = call('critic', actual, i)
                assert torch.equal(score, score_ref)
            ((expected.float() - score_ref.float()).square().mean() / 8).backward()
            ((actual.float() - score.float()).square().mean() / 8).backward()
            torch.testing.assert_close(a.grad, b.grad, rtol=.03, atol=.0005)
            assert roles['generator'].pending == 0

        for i in range(8):
            expected = references['critic'](x[i:i+1], scalar[i:i+1], text[i:i+1])[0]
            actual = call('critic', x[i:i+1], i)
            assert torch.equal(actual, expected)
            (expected.float().square().mean() / 8).backward()
            (actual.float().square().mean() / 8).backward()

        for role, proxy in roles.items():
            for name, parameter in proxy.native_model.named_parameters():
                torch.testing.assert_close(parameter.grad,
                    dict(references[role].named_parameters())[name].grad, rtol=.03, atol=.0005)
            receipt = proxy.receipt()
            assert receipt['actual_stage_B1_forward_count'] == 16
            assert receipt['pending_backward_schedules'] == 0
            assert receipt['pipeline_microbatches_per_invocation'] == 1
            assert receipt['logical_microbatches_per_DP'] == 8
        assert not torch.cuda.is_initialized()
    finally:
        dist.destroy_process_group()


def test_serial_B1_keeps_pyramid_and_multi_role_checkpoint_gradients(tmp_path):
    mp.spawn(_run, args=(str(tmp_path / 'serial_pp2'),), nprocs=2, join=True)


def test_eight_single_losses_preserve_original_mean8():
    from distillschedule.runtime.static_3d_state import StaticDataParallelState
    state = object.__new__(StaticDataParallelState)
    state.accumulations = 0
    p = torch.nn.Parameter(torch.tensor(3., dtype=torch.float64))
    for i in range(8):
        state.mean_microbatch_group_loss(p * (i + 1), group_size=1).backward()
    assert p.grad == 4.5 and state.accumulations == 8
    with pytest.raises(ValueError, match='eight'):
        state.mean_microbatch_group_loss(p, group_size=1)


def test_single_trace_dispatch_preserves_RNG_and_B1_identity():
    from distillschedule.runtime.static_native_cooperative import run_native_B1_cohort
    source = torch.randn(1, 2, requires_grad=True)
    rng = torch.get_rng_state().clone()
    result, receipt = run_native_B1_cohort(lambda **kw: (kw['hidden_states'] * 2,),
        lambda view, index: view(hidden_states=source, return_dict=False)[0],
        device='cpu', cohort_size=1)
    result[0].sum().backward()
    assert torch.equal(source.grad, torch.full_like(source, 2))
    assert torch.equal(torch.get_rng_state(), rng)
    assert receipt['logical_samples'] == 1
    assert receipt['model_boundary'] == 'SERIAL_B1_PP2_NOT_GPIPE1'
