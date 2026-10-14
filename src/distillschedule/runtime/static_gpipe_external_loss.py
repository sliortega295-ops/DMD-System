"""Eight-microbatch GPipe with a later, externally composed DMD loss.

Keep the Generator's native checkpoint graphs while both PP stages execute
Fake/Teacher calls. Only then run its normal GPipe backward. This is a checked
PyTorch 2.10 schedule seam, not a native model binding or a timing simulator.
"""
from __future__ import annotations

import hashlib
import inspect
import weakref

import torch
import torch.distributed as dist
from torch.distributed.pipelining import PipelineStage, ScheduleGPipe
from torch.distributed.pipelining.schedules import _sorted_batch_p2p, _wait_batch_p2p


def _external_loss_only(*_args):
    raise RuntimeError('DMD loss must be supplied after the separate Score forwards')


def _complete_boundary_gradients(gradients, buffers, *, constant_inputs, names):
    """Supply wire zeros only for admitted inputs with no upstream owner.

    Torch 2.10 PP mirrors every forward receive with a backward send, even
    when the original tensor is a constant. A wire zero satisfies that fixed
    transport protocol; stage0 autograd discards it for its constant output.
    A missing differentiable input must remain an error, not a zero gradient.
    """
    if not (len(gradients) == len(buffers) == len(names)):
        raise ValueError('native PP boundary gradient arity changed')
    completed, zeros = [], 0
    for index, (gradient, buffer, name) in enumerate(zip(gradients, buffers, names, strict=True)):
        if index in constant_inputs:
            if buffer.requires_grad or gradient is not None:
                raise ValueError('declared constant acquired a gradient owner: '+name)
            gradient = torch.zeros_like(buffer)
            zeros += 1
        elif gradient is None:
            raise ValueError('missing differentiable native PP boundary gradient: '+name)
        if (gradient.shape != buffer.shape or gradient.dtype != buffer.dtype
                or gradient.device != buffer.device):
            raise ValueError('native PP boundary gradient signature changed: '+name)
        completed.append(gradient)
    return tuple(completed), zeros


class ExternalLossGPipe(ScheduleGPipe):
    """Retains eight B1 graphs; exactly one mean8 backward/update boundary.

    Outputs are available on both stages (detached on stage0), so native DMD
    code may construct the same detached score inputs. No optimizer step, DP
    synchronization, caching, role sharing or sample selection happens here.
    """
    def __init__(self, stage, final_output_example, *, constant_boundary_inputs=(),
                 boundary_input_names=(), pipeline_microbatches=8):
        if torch.__version__ != '2.10.0+cu128':
            raise ValueError('private GPipe seam is admitted only for Torch 2.10.0+cu128')
        if type(pipeline_microbatches) is not int or pipeline_microbatches not in (2, 8):
            raise ValueError('PP2 invocation admits two or eight B1 microbatches')
        self.native_microbatches = pipeline_microbatches
        super().__init__(stage, n_microbatches=pipeline_microbatches, loss_fn=_external_loss_only, scale_grads=True)
        self.pending = False
        self.completed_batches = 0
        self.discarded_unreachable_forward_batches = 0
        self._external_graph_ref = None
        self.final_output_shape = tuple(final_output_example.shape)
        self.final_output_dtype = final_output_example.dtype
        self.constant_boundary_inputs = frozenset(constant_boundary_inputs)
        self.boundary_input_names = tuple(boundary_input_names)
        self.transport_only_zero_grad_count = 0

    def forward_batch(self, *args, **kwargs):
        self.retire_unreachable_graph()
        if self.pending:
            raise RuntimeError('previous external-loss backward is still pending')
        if not torch.is_grad_enabled():
            raise RuntimeError('external-loss Generator forward requires autograd')
        # ScheduleSingle.step performs the ordinary argument splitting and
        # clears runtime state, but our override deliberately stops after F.
        output = super().step(*args, **kwargs)
        self.pending = True
        shape = self.final_output_shape
        if self._stage.is_last and (tuple(output.shape) != (self.native_microbatches * shape[0], *shape[1:])
                                   or output.dtype != self.final_output_dtype):
            raise ValueError('actual last-stage output violates its admitted final signature')
        transported = (output.detach().contiguous().clone() if self._stage.is_last
            else torch.empty((self.native_microbatches * shape[0], *shape[1:]),
                             dtype=self.final_output_dtype, device=self._stage.device))
        dist.broadcast(transported, src=dist.get_global_rank(self._stage.group, 1),
                       group=self._stage.group)
        return output if self._stage.is_last else transported

    def bind_external_graph(self, context):
        """Observe, but never retain, the native outer autograd node.

        A later native no-grad denoising call can sever the selected earlier
        pyramid graph. Ordinary autograd then releases it without backward;
        the explicit PP schedule must release its corresponding inner cache.
        """
        if not self.pending or self._external_graph_ref is not None:
            raise RuntimeError('external graph registration requires one pending forward')
        self._external_graph_ref = weakref.ref(context)

    def retire_unreachable_graph(self):
        """Release only a graph that native autograd has already discarded.

        No backward, zeros, collective, evaluation or update is synthesized.
        A live/unknown owner remains pending and the normal gate rejects it.
        """
        if (not self.pending or self._external_graph_ref is None
                or self._external_graph_ref() is not None):
            return False
        self._stage.clear_runtime_states()
        self.pending = False
        self._external_graph_ref = None
        self.discarded_unreachable_forward_batches += 1
        return True

    def _step_microbatches(self, arg_mbs=None, kwarg_mbs=None, target_mbs=None,
                          losses=None, return_outputs=True):
        if target_mbs is not None or losses is not None or not return_outputs:
            raise ValueError('forward-only external-loss pass must retain output graphs')
        arg_mbs, kwarg_mbs = self._check_inputs(arg_mbs, kwarg_mbs, None, None)
        self._initialize_stage(arg_mbs[0], kwarg_mbs[0])
        if self._stage.is_last:
            for received in self._stage.args_recv_info.values():
                for index in self.constant_boundary_inputs:
                    received[index].buffer.requires_grad_(False)
        sends = []
        for i in range(self.native_microbatches):
            if self._stage.is_first:
                # Chunking creates non-leaf views. PyTorch's stage backward
                # reads their .grad to return native upstream input gradients.
                for tensor in arg_mbs[i]:
                    if isinstance(tensor, torch.Tensor) and tensor.requires_grad:
                        tensor.retain_grad()
            for work in _sorted_batch_p2p(self._stage.get_fwd_recv_ops(i), desc='fwd_recv').values():
                _wait_batch_p2p(work)
            self._stage.forward_one_chunk(i, arg_mbs[i], kwarg_mbs[i], save_forward_output=True)
            sends.extend(_sorted_batch_p2p(self._stage.get_fwd_send_ops(i), desc='fwd_send').values())
        for work in sends:
            _wait_batch_p2p(work)

    def backward_losses(self, losses=None):
        self._backward_losses(losses, gradient_divisor=self.native_microbatches)

    def _backward_losses(self, losses, *, gradient_divisor):
        if not self.pending:
            raise RuntimeError('no pending external-loss GPipe forward')
        if self._stage.is_last:
            if (not isinstance(losses, (tuple, list)) or len(losses) != self.native_microbatches
                    or any(not isinstance(loss, torch.Tensor) or loss.ndim != 0
                           or not loss.requires_grad for loss in losses)):
                raise ValueError('last stage requires the admitted independent scalar B1 losses')
        elif losses is not None:
            raise ValueError('only the last PP stage owns the scalar loss graphs')
        sends = []
        for i in range(self.native_microbatches):
            for work in _sorted_batch_p2p(self._stage.get_bwd_recv_ops(i), desc='bwd_recv').values():
                _wait_batch_p2p(work)
            self._stage.backward_one_chunk(i, loss=losses[i] if self._stage.is_last else None,
                                           last_backward=i == self.native_microbatches - 1)
            if self._stage.is_last:
                received = self._stage.args_recv_info[i]
                completed, count = _complete_boundary_gradients(self._stage.bwd_cache[i],
                    tuple(r.buffer for r in received), constant_inputs=self.constant_boundary_inputs,
                    names=self.boundary_input_names)
                self._stage.bwd_cache[i] = completed
                self.transport_only_zero_grad_count += count
            sends.extend(_sorted_batch_p2p(self._stage.get_bwd_send_ops(i), desc='bwd_send').values())
        for work in sends:
            _wait_batch_p2p(work)
        self._stage.perform_reduce_grad(gradient_divisor)
        self.pending = False
        self._external_graph_ref = None
        self.completed_batches += 1

    def backward_output_gradient(self, gradient):
        """The native batch loss already averages eight samples: no /8 again."""
        if (not isinstance(gradient, torch.Tensor)
                or tuple(gradient.shape) != (self.native_microbatches * self.final_output_shape[0],
                                              *self.final_output_shape[1:])):
            raise ValueError('external output gradient violates the final batch signature')
        losses = None
        if self._stage.is_last:
            with torch.enable_grad():
                losses = [(output * part.detach()).sum() for output, part in zip(
                    self.last_stage_microbatch_outputs(), gradient.chunk(self.native_microbatches), strict=True)]
        self._backward_losses(losses, gradient_divisor=1)

    def first_stage_input_gradients(self, input_count):
        if self.pending or not self._stage.is_first:
            raise RuntimeError('input gradients exist only after first-stage backward')
        result = []
        for index in range(input_count):
            parts = [self._stage.bwd_cache[i][index] for i in range(self.native_microbatches)]
            if all(p is None for p in parts):
                result.append(None)
            elif any(p is None for p in parts):
                raise ValueError('inconsistent native input gradient across microbatches')
            else:
                result.append(torch.cat(parts))
        return result

    def last_stage_microbatch_outputs(self):
        if not self.pending or not self._stage.is_last:
            raise RuntimeError('independent loss graphs exist only on the pending last stage')
        # A loss sliced out of the merged cat would retain *all eight* output
        # graphs, causing replay/free of other microbatches during backward0.
        # Losses must attach to their original B1 output graph instead.
        return tuple(self._stage.output_chunks)

    def receipt(self):
        return dict(torch_version=torch.__version__, microbatch_size=1, microbatches=self.native_microbatches,
            schedule=('GPipe_FORWARD8_SCORE_ROLES_THEN_BACKWARD8' if self.native_microbatches == 8
                      else 'GPipe_FORWARD2_SCORE_ROLES_BACKWARD2_OUTER_ACCUMULATE8'),
            gradient_scaling=('MICROBATCH_MEAN8_BEFORE_DP2_MEAN' if self.native_microbatches == 8
                              else 'OUTER_EXPLICIT_MEAN8_BEFORE_DP2_MEAN'),
            completed_forward_backward_batches=self.completed_batches,
            discarded_unreachable_forward_batches=self.discarded_unreachable_forward_batches,
            pending_external_backward=self.pending,
            constant_boundary_inputs=[self.boundary_input_names[i]
                                      for i in sorted(self.constant_boundary_inputs)],
            transport_only_zero_grad_count=self.transport_only_zero_grad_count,
            upstream_gpipe_source_sha256=hashlib.sha256(inspect.getsource(ScheduleGPipe).encode()).hexdigest(),
            native_model_binding='NOT_RUN_COMPONENT_ONLY', native_timing='NOT_RUN')


def build_external_loss_gpipe(stage_module, *, stage_index, group, device,
                              input_args, output_args, final_output_args,
                              constant_boundary_inputs=(), boundary_input_names=None,
                              pipeline_microbatches=8):
    if not dist.is_initialized() or dist.get_world_size(group) != 2:
        raise ValueError('requires explicit PP2')
    if stage_index not in (0, 1) or dist.get_rank(group) != stage_index:
        raise ValueError('PP coordinate mismatch')
    if (not isinstance(stage_module, torch.nn.Module) or not output_args
            or any(not isinstance(v, torch.Tensor) or v.shape[0] != 1 for v in output_args)
            or len(final_output_args) != 1
            or final_output_args[0].shape[0] != 1):
        raise ValueError('requires one actual tensor output per B1 stage')
    boundary = output_args if stage_index == 0 else input_args
    if (not isinstance(constant_boundary_inputs, tuple)
            or any(type(i) is not int or not 0 <= i < len(boundary)
                   for i in constant_boundary_inputs)
            or len(set(constant_boundary_inputs)) != len(constant_boundary_inputs)):
        raise ValueError('constant boundary inputs require explicit unique field indices')
    names = (tuple('input_'+str(i) for i in range(len(boundary)))
             if boundary_input_names is None else boundary_input_names)
    if (not isinstance(names, tuple) or len(names) != len(boundary)
            or any(not isinstance(n, str) or not n for n in names)):
        raise ValueError('all native boundary fields must be named')
    stage = PipelineStage(stage_module, stage_index, 2, torch.device(device),
        input_args=input_args, output_args=output_args, group=group)
    return ExternalLossGPipe(stage, final_output_args[0],
        constant_boundary_inputs=constant_boundary_inputs, boundary_input_names=names,
        pipeline_microbatches=pipeline_microbatches)
