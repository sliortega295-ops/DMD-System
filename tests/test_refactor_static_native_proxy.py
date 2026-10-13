"""Native B1 PP dispatcher integration, not a GPU/model performance result."""
import copy
from datetime import timedelta
import hashlib
import inspect
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.checkpoint import checkpoint


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 5, dtype=torch.bfloat16)

    def forward(self, hidden, encoder, timestep_proj, rotary):
        # RoPE metadata has no upstream differentiable owner. In particular,
        # a kernel may return no derivative for its fixed frequencies. PP
        # must not require such a derivative or suppress genuine inputs.
        return (self.linear(hidden) + encoder + timestep_proj * rotary.detach()).tanh()


class _Native(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embedding = torch.nn.Linear(5, 5, dtype=torch.bfloat16)
        self.condition_embedder = torch.nn.Linear(5, 5, dtype=torch.bfloat16)
        self.proj_out = torch.nn.Linear(5, 5, dtype=torch.bfloat16)
        self.blocks = torch.nn.ModuleList([_Block() for _ in range(4)])
        self.gradient_checkpointing = True

    def _gradient_checkpointing_func(self, function, *args):
        return checkpoint(function, *args, use_reentrant=False)

    def enable_adapters(self):
        pass

    def disable_adapters(self):
        pass

    def forward(self, hidden_states, timestep, encoder_hidden_states,
                indices_hidden_states=None, indices_latents_history_short=None,
                indices_latents_history_mid=None, indices_latents_history_long=None,
                latents_history_short=None, latents_history_mid=None, latents_history_long=None,
                *, return_dict=False, is_first_denoising_step=False):
        hidden_states = self.patch_embedding(hidden_states)
        encoder_hidden_states = self.condition_embedder(encoder_hidden_states)
        timestep_proj = encoder_hidden_states * 2
        temb = encoder_hidden_states * .5
        rotary_emb = torch.ones_like(encoder_hidden_states)
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for iidx, block in enumerate(self.blocks):
                hidden_states = self._gradient_checkpointing_func(block, hidden_states,
                    encoder_hidden_states, timestep_proj, rotary_emb)
        else:
            for iidx, block in enumerate(self.blocks):
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)
        return (self.proj_out(hidden_states + temb),)


def _run(rank, rendezvous, cooperative=False):
    from distillschedule.runtime.static_native_pp_stage import NativeSourcePPStage
    from distillschedule.runtime.static_native_forward import create_admitted_forward_split
    from distillschedule.runtime.static_native_proxy import NativeStaticPPProxy
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + rendezvous,
        world_size=2, rank=rank, timeout=timedelta(seconds=80))
    try:
        torch.manual_seed(846)
        reference = _Native()
        native = copy.deepcopy(reference)
        x = torch.randn(8, 2, 5, dtype=torch.bfloat16)
        text = torch.randn_like(x)
        scalar = torch.zeros(8, 1)
        kwargs = dict(hidden_states=x, timestep=scalar, encoder_hidden_states=text,
            indices_hidden_states=scalar, indices_latents_history_short=scalar,
            indices_latents_history_mid=scalar, indices_latents_history_long=scalar,
            latents_history_short=scalar, latents_history_mid=scalar, latents_history_long=scalar,
            return_dict=False)
        split = create_admitted_forward_split(native,
            expected_source_sha256=hashlib.sha256(Path(inspect.getsourcefile(_Native)).read_bytes()).hexdigest(),
            expected_blocks=4)
        with torch.no_grad():
            context = split.prefix(native, x[:1], scalar[:1], text[:1], return_dict=False)
        context['return_dict'] = False
        stage = NativeSourcePPStage(native, split=split, stage_index=rank, context_template=context,
            remove_root_modules=('proj_out',) if rank == 0 else ('patch_embedding','condition_embedder'))
        proxy = NativeStaticPPProxy(stage, pp_group=dist.group.WORLD, device='cpu', role='generator')
        for grad in (False, True):
            with torch.set_grad_enabled(grad):
                expected_parts = [reference(x[i:i+1], scalar[i:i+1], text[i:i+1])[0] for i in range(8)]
                if cooperative:
                    from distillschedule.runtime.static_native_cooperative import run_native_B1_cohort
                    def sample(view, index):
                        row = {k: v[index:index+1] if isinstance(v, torch.Tensor) else v
                               for k,v in kwargs.items()}
                        return view(**row)[0]
                    outputs, receipt = run_native_B1_cohort(proxy, sample, device='cpu')
                    assert receipt['per_sample_call_counts'] == [1]*8
                    actual = torch.cat(outputs)
                else:
                    actual = proxy(**kwargs)[0]
                assert torch.equal(actual.detach(), torch.cat(expected_parts).detach())
                if grad:
                    torch.cat(expected_parts).float().square().mean().backward()
                    actual.float().square().mean().backward()
                    for name, parameter in native.named_parameters():
                        torch.testing.assert_close(parameter.grad, dict(reference.named_parameters())[name].grad,
                                                   rtol=.03, atol=.0005)
        # Reuse the admitted no-grad schedule with a shared B1 negative text
        # condition. Its expand(8, ...) view has a zero batch stride; PP still
        # needs the same canonical microbatch layout as distinct B1 texts.
        with torch.no_grad():
            shared = dict(kwargs, encoder_hidden_states=text[:1])
            expected = torch.cat([reference(x[i:i+1], scalar[i:i+1], text[:1])[0]
                                  for i in range(8)])
            actual = proxy(**shared)[0]
            assert torch.equal(actual, expected)
        assert proxy.receipt()['actual_stage_B1_forward_count'] == 24
        assert proxy.receipt()['pending_backward_schedules'] == 0
        transport = proxy.receipt()['gradient_transport']
        assert len(transport) == 1
        assert transport[0]['constant_boundary_inputs'] == ['rotary_emb']
        assert transport[0]['transport_only_zero_grad_count'] == (8 if rank == 1 else 0)
        assert not torch.cuda.is_initialized()
    finally:
        dist.destroy_process_group()


def test_native_proxy_runs_B1_on_both_stages_for_grad_and_nograd(tmp_path):
    mp.spawn(_run, args=(str(tmp_path/'native_proxy'),), nprocs=2, join=True)


def test_independent_B1_sampler_threads_feed_real_PP2_GPipe8(tmp_path):
    mp.spawn(_run, args=(str(tmp_path/'cooperative_proxy'),True), nprocs=2, join=True)


def test_shared_B1_condition_transport_is_contiguous_and_preserves_gradient():
    from distillschedule.runtime.static_native_proxy import NativeStaticPPProxy
    proxy = object.__new__(NativeStaticPPProxy)
    values = {name: torch.ones(8, 2, 5) for name in proxy.INPUT_NAMES}
    shared = torch.arange(10., requires_grad=True).reshape(1, 2, 5)
    shared.retain_grad()
    values['encoder_hidden_states'] = shared
    inputs = proxy._cohort_inputs(values)
    transported = inputs[proxy.INPUT_NAMES.index('encoder_hidden_states')]
    assert transported.is_contiguous()
    assert torch.equal(transported, shared.expand(8, -1, -1))
    transported.sum().backward()
    assert torch.equal(shared.grad, torch.full_like(shared, 8.))
    assert not torch.cuda.is_initialized()
