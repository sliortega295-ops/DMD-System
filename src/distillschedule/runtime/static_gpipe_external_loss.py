"""Eight-microbatch GPipe with a later, externally composed DMD loss.

Keep the Generator's native checkpoint graphs while both PP stages execute
Fake/Teacher calls. Only then run its normal GPipe backward. This is a checked
PyTorch 2.10 schedule seam, not a native model binding or a timing simulator.
"""
from __future__ import annotations

import hashlib
import inspect

import torch
import torch.distributed as dist
from torch.distributed.pipelining import PipelineStage, ScheduleGPipe
from torch.distributed.pipelining.schedules import _sorted_batch_p2p, _wait_batch_p2p


def _external_loss_only(*_args):
    raise RuntimeError('DMD loss must be supplied after the separate Score forwards')


class ExternalLossGPipe(ScheduleGPipe):
    """Retains eight B1 graphs; exactly one mean8 backward/update boundary.

    Outputs are available on both stages (detached on stage0), so native DMD
    code may construct the same detached score inputs. No optimizer step, DP
    synchronization, caching, role sharing or sample selection happens here.
    """
    def __init__(self, stage, final_output_example):
        if torch.__version__ != '2.10.0+cu128':
            raise ValueError('private GPipe seam is admitted only for Torch 2.10.0+cu128')
        super().__init__(stage, n_microbatches=8, loss_fn=_external_loss_only, scale_grads=True)
        self.pending = False
        self.completed_batches = 0
        self.final_output_shape = tuple(final_output_example.shape)
        self.final_output_dtype = final_output_example.dtype

    def forward_batch(self, *args, **kwargs):
        if self.pending:
            raise RuntimeError('previous external-loss backward is still pending')
        if not torch.is_grad_enabled():
            raise RuntimeError('external-loss Generator forward requires autograd')
        # ScheduleSingle.step performs the ordinary argument splitting and
        # clears runtime state, but our override deliberately stops after F.
        output = super().step(*args, **kwargs)
        self.pending = True
        shape = self.final_output_shape
        if self._stage.is_last and (tuple(output.shape) != (8 * shape[0], *shape[1:])
                                   or output.dtype != self.final_output_dtype):
            raise ValueError('actual last-stage output violates its admitted final signature')
        transported = (output.detach().contiguous().clone() if self._stage.is_last
            else torch.empty((8 * shape[0], *shape[1:]),
                             dtype=self.final_output_dtype, device=self._stage.device))
        dist.broadcast(transported, src=dist.get_global_rank(self._stage.group, 1),
                       group=self._stage.group)
        return output if self._stage.is_last else transported

    def _step_microbatches(self, arg_mbs=None, kwarg_mbs=None, target_mbs=None,
                          losses=None, return_outputs=True):
        if target_mbs is not None or losses is not None or not return_outputs:
            raise ValueError('forward-only external-loss pass must retain output graphs')
        arg_mbs, kwarg_mbs = self._check_inputs(arg_mbs, kwarg_mbs, None, None)
        self._initialize_stage(arg_mbs[0], kwarg_mbs[0])
        sends = []
        for i in range(8):
            for work in _sorted_batch_p2p(self._stage.get_fwd_recv_ops(i), desc='fwd_recv').values():
                _wait_batch_p2p(work)
            self._stage.forward_one_chunk(i, arg_mbs[i], kwarg_mbs[i], save_forward_output=True)
            sends.extend(_sorted_batch_p2p(self._stage.get_fwd_send_ops(i), desc='fwd_send').values())
        for work in sends:
            _wait_batch_p2p(work)

    def backward_losses(self, losses=None):
        if not self.pending:
            raise RuntimeError('no pending external-loss GPipe forward')
        if self._stage.is_last:
            if (not isinstance(losses, (tuple, list)) or len(losses) != 8
                    or any(not isinstance(loss, torch.Tensor) or loss.ndim != 0
                           or not loss.requires_grad for loss in losses)):
                raise ValueError('last stage requires eight independent scalar B1 losses')
        elif losses is not None:
            raise ValueError('only the last PP stage owns the scalar loss graphs')
        sends = []
        for i in range(8):
            for work in _sorted_batch_p2p(self._stage.get_bwd_recv_ops(i), desc='bwd_recv').values():
                _wait_batch_p2p(work)
            self._stage.backward_one_chunk(i, loss=losses[i] if self._stage.is_last else None,
                                           last_backward=i == 7)
            sends.extend(_sorted_batch_p2p(self._stage.get_bwd_send_ops(i), desc='bwd_send').values())
        for work in sends:
            _wait_batch_p2p(work)
        self._stage.perform_reduce_grad(8)
        self.pending = False
        self.completed_batches += 1

    def last_stage_microbatch_outputs(self):
        if not self.pending or not self._stage.is_last:
            raise RuntimeError('independent loss graphs exist only on the pending last stage')
        # A loss sliced out of the merged cat would retain *all eight* output
        # graphs, causing replay/free of other microbatches during backward0.
        # Losses must attach to their original B1 output graph instead.
        return tuple(self._stage.output_chunks)

    def receipt(self):
        return dict(torch_version=torch.__version__, microbatch_size=1, microbatches=8,
            schedule='GPipe_FORWARD8_SCORE_ROLES_THEN_BACKWARD8',
            gradient_scaling='MICROBATCH_MEAN8_BEFORE_DP2_MEAN',
            completed_forward_backward_batches=self.completed_batches,
            pending_external_backward=self.pending,
            upstream_gpipe_source_sha256=hashlib.sha256(inspect.getsource(ScheduleGPipe).encode()).hexdigest(),
            native_model_binding='NOT_RUN_COMPONENT_ONLY', native_timing='NOT_RUN')


def build_external_loss_gpipe(stage_module, *, stage_index, group, device,
                              input_args, output_args, final_output_args):
    if not dist.is_initialized() or dist.get_world_size(group) != 2:
        raise ValueError('requires explicit PP2')
    if stage_index not in (0, 1) or dist.get_rank(group) != stage_index:
        raise ValueError('PP coordinate mismatch')
    if (not isinstance(stage_module, torch.nn.Module) or not output_args
            or any(not isinstance(v, torch.Tensor) or v.shape[0] != 1 for v in output_args)
            or len(final_output_args) != 1
            or final_output_args[0].shape[0] != 1):
        raise ValueError('requires one actual tensor output per B1 stage')
    stage = PipelineStage(stage_module, stage_index, 2, torch.device(device),
        input_args=input_args, output_args=output_args, group=group)
    return ExternalLossGPipe(stage, final_output_args[0])
