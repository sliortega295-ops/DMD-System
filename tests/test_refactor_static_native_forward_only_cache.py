"""Forward-only PP caches must not keep discarded native pyramid graphs alive."""
import copy
from datetime import timedelta
import gc
import hashlib
import inspect
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import pytest

from test_refactor_static_native_proxy import _Native


def _run(rank, rendezvous, collect_garbage):
    from distillschedule.runtime.static_native_cooperative import run_native_B1_cohort
    from distillschedule.runtime.static_native_forward import create_admitted_forward_split
    from distillschedule.runtime.static_native_pp_stage import NativeSourcePPStage
    from distillschedule.runtime.static_native_proxy import NativeStaticPPProxy

    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + rendezvous,
        rank=rank, world_size=2, timeout=timedelta(seconds=90))
    try:
        torch.manual_seed(831)
        reference = _Native()
        native = copy.deepcopy(reference)
        x = torch.randn(8, 2, 5, dtype=torch.bfloat16)
        text = torch.randn_like(x)
        scalar = torch.zeros(8, 1)
        split = create_admitted_forward_split(native,
            expected_source_sha256=hashlib.sha256(Path(inspect.getsourcefile(_Native)).read_bytes()).hexdigest(),
            expected_blocks=4)
        with torch.no_grad():
            context = split.prefix(native, x[:1], scalar[:1], text[:1], return_dict=False)
        context['return_dict'] = False
        stage = NativeSourcePPStage(native, split=split, stage_index=rank,
            context_template=context, remove_root_modules=('proj_out',) if rank == 0
            else ('patch_embedding', 'condition_embedder'))
        proxy = NativeStaticPPProxy(stage, pp_group=dist.group.WORLD,
            device='cpu', role='generator')

        def sample(model, i):
            row = dict(hidden_states=x[i:i+1], timestep=scalar[i:i+1],
                encoder_hidden_states=text[i:i+1], indices_hidden_states=scalar[i:i+1],
                indices_latents_history_short=scalar[i:i+1],
                indices_latents_history_mid=scalar[i:i+1],
                indices_latents_history_long=scalar[i:i+1],
                latents_history_short=scalar[i:i+1],
                latents_history_mid=scalar[i:i+1], latents_history_long=scalar[i:i+1],
                return_dict=False, is_first_denoising_step=True)
            first = model(**row)[0]
            row.update(hidden_states=first * .7 + x[i:i+1] * .3,
                       is_first_denoising_step=False)
            # Native next-scale no-grad call receives a differentiable input,
            # but the resulting solver update cuts that earlier graph.
            assert row['hidden_states'].requires_grad
            with torch.no_grad():
                row['hidden_states'] = model(**row)[0] * .7 + x[i:i+1] * .3
            middle = model(**row)[0]
            row['hidden_states'] = middle * .6 + x[i:i+1] * .4
            return model(**row)[0]

        expected = torch.cat([sample(reference, i) for i in range(8)])
        expected.float().square().mean().backward()
        results, dispatch = run_native_B1_cohort(proxy, sample, device='cpu')
        actual = torch.cat(results)
        assert torch.equal(actual.detach(), expected.detach())
        actual.float().square().mean().backward()
        for name, p in native.named_parameters():
            torch.testing.assert_close(p.grad, dict(reference.named_parameters())[name].grad,
                                       rtol=.03, atol=.0005)
        if collect_garbage:
            gc.collect()
        receipt = proxy.receipt()
        print('OBSERVED_PP_FORWARD_ONLY_CACHE_PENDING', rank,
              receipt['pending_backward_schedules'], flush=True)
        assert dispatch['per_sample_call_counts'] == [4] * 8
        assert receipt['application_forward_per_logical_sample'] == 4
        assert receipt['grad_forward_per_logical_sample'] == 3
        assert receipt['pending_backward_schedules'] == 0
        assert sum(r['discarded_unreachable_forward_batches']
                   for r in receipt['gradient_transport']) == 1
        assert sum(r['completed_forward_backward_batches']
                   for r in receipt['gradient_transport']) == 2
        assert not torch.cuda.is_initialized()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('collect_garbage', [False, True])
def test_cooperative_no_grad_pp_does_not_retain_earlier_gradient_graph(tmp_path, collect_garbage):
    mp.spawn(_run, args=(str(tmp_path / 'native_forward_only_cache'), collect_garbage),
             nprocs=2, join=True)
