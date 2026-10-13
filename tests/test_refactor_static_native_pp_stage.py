"""Source-derived PP2 prefix ownership and multi-tensor autograd transport."""
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
        self.linear = torch.nn.Linear(5, 5, dtype=torch.float64)

    def forward(self, hidden, encoder, timestep_proj, rotary):
        return (self.linear(hidden) + encoder + timestep_proj * rotary).tanh()


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embedding = torch.nn.Linear(5, 5, dtype=torch.float64)
        self.condition_embedder = torch.nn.Linear(5, 5, dtype=torch.float64)
        self.proj_out = torch.nn.Linear(5, 3, dtype=torch.float64)
        self.blocks = torch.nn.ModuleList([_Block() for _ in range(4)])
        self.gradient_checkpointing = True

    def _gradient_checkpointing_func(self, function, *args):
        return checkpoint(function, *args, use_reentrant=False)

    def forward(self, hidden_states, encoder_hidden_states, *, return_dict=False):
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
        output = self.proj_out(hidden_states + temb)
        return (output,)


def _run_stage(rank, rendezvous):
    from distillschedule.runtime.static_native_pp_stage import NativeSourcePPStage
    from distillschedule.runtime.static_native_forward import create_admitted_forward_split
    from distillschedule.runtime.static_gpipe_external_loss import build_external_loss_gpipe
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + rendezvous,
        rank=rank, world_size=2, timeout=timedelta(seconds=80))
    try:
        torch.manual_seed(615)
        reference = _Model()
        native = copy.deepcopy(reference)
        x = torch.randn(8, 5, dtype=torch.float64)
        condition = torch.randn(8, 5, dtype=torch.float64)
        expected = reference(x, condition)[0]
        with torch.no_grad():
            score_gradient = expected.square() * .1
        (expected * score_gradient).sum(1).mean().backward()
        split = create_admitted_forward_split(native,
            expected_source_sha256=hashlib.sha256(Path(inspect.getsourcefile(_Model)).read_bytes()).hexdigest(),
            expected_blocks=4)
        with torch.no_grad():
            context = split.prefix(native, x[:1], condition[:1], return_dict=False)
            boundary_context = split.blocks(native, context, 0, 2)
        # This fixture's suffix returns a tuple unconditionally, so the AST
        # splitter has no need to retain return_dict. State the PP output
        # contract explicitly rather than weakening the production guard.
        context['return_dict'] = False
        fields = NativeSourcePPStage.TRANSPORT_FIELDS
        boundary_examples = tuple(boundary_context[k] for k in fields)
        stage = NativeSourcePPStage(native, split=split, stage_index=rank,
            context_template=context,
            remove_root_modules=('proj_out',) if rank == 0 else ('condition_embedder', 'patch_embedding'))
        gp = build_external_loss_gpipe(stage, stage_index=rank, group=dist.group.WORLD,
            device='cpu', input_args=(x[:1], condition[:1]) if rank == 0 else boundary_examples,
            output_args=boundary_examples if rank == 0 else (expected[:1].detach(),),
            final_output_args=(expected[:1].detach(),))
        actual = gp.forward_batch(x, condition)
        torch.testing.assert_close(actual.detach(), expected.detach(), rtol=1e-12, atol=1e-12)
        losses = ([(v * score_gradient[i:i+1]).sum()
                   for i, v in enumerate(gp.last_stage_microbatch_outputs())] if rank == 1 else None)
        gp.backward_losses(losses)
        reference_params = dict(reference.named_parameters())
        for name, p in native.named_parameters():
            torch.testing.assert_close(p.grad, reference_params[name].grad, rtol=1e-10, atol=1e-12)
        for i in range(4):
            if (i < 2) == (rank == 0):
                assert list(native.blocks[i].parameters())
            else:
                assert not list(native.blocks[i].parameters())
        assert stage.receipt()['pp_root_parameter_replication'] is False
        assert not torch.cuda.is_initialized()
    finally:
        dist.destroy_process_group()


def test_native_source_pp2_five_tensor_context_and_root_gradient_ownership(tmp_path):
    mp.spawn(_run_stage, args=(str(tmp_path/'gloo_native_boundary'),), nprocs=2, join=True)
