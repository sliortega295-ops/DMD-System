"""Checked PyTorch2.10 PP2/GPipe construction, not a native DMD task binding.

References: https://docs.pytorch.org/docs/2.10/distributed.pipelining.html
The model adapter must supply actual stage modules and static shape/dtype
examples. Never reinterpret TP/PP collaborators as independent samples.
"""
from __future__ import annotations
import torch
import torch.distributed as dist
from torch.distributed.pipelining import PipelineStage, ScheduleGPipe


def build_static_gpipe(stage_module, *, stage_index, group, device, input_args,
                       output_args, loss_fn=None, pipeline_microbatches=8):
    if not dist.is_initialized() or dist.get_world_size(group)!=2:
        raise ValueError('explicit PP2 group required')
    if stage_index not in (0,1) or dist.get_rank(group)!=stage_index:
        raise ValueError('PP rank/stage mismatch')
    if not isinstance(stage_module,torch.nn.Module):
        raise ValueError('an actual native stage module is required')
    if type(pipeline_microbatches) is not int or pipeline_microbatches not in (2, 8):
        raise ValueError('explicit PP2 invocation has two or eight B1 microbatches')
    stage=PipelineStage(stage_module,stage_index,2,device,
        input_args=input_args,output_args=output_args,group=group)
    schedule=ScheduleGPipe(stage,n_microbatches=pipeline_microbatches,loss_fn=loss_fn,scale_grads=True)
    return stage,schedule


class _Stage0ExternalBoundary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, transport, signature):
        ctx.transport, ctx.signature = transport, signature
        transport._send(hidden.detach())
        return transport._receive(signature.output_shape, signature.output_dtype)

    @staticmethod
    def backward(ctx, output_gradient):
        # Both stages evaluate the same scalar loss, but only stage1 owns the
        # output graph. Its backward sends dL/dhidden; do NOT send output_gradient
        # a second time or accumulate two copies of the logical sample.
        signature = ctx.signature
        gradient = ctx.transport._receive(signature.hidden_shape, signature.hidden_dtype)
        return gradient, None, None


class _Stage1ExternalBoundary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dummy, transport, signature):
        ctx.transport, ctx.signature = transport, signature
        return transport._receive(signature.hidden_shape, signature.hidden_dtype)

    @staticmethod
    def backward(ctx, gradient):
        ctx.transport._send(gradient.detach())
        return None, None, None


class StaticPipelineSignature:
    """Explicit static activation/output metadata, not a hot-path CUDA read.

    A native binding must supply one signature for each admitted pyramid
    shape/history layout. There is no guessed shape or model-specific fallback.
    """
    def __init__(self, *, hidden_shape, hidden_dtype, output_shape, output_dtype):
        for shape in (hidden_shape, output_shape):
            if not shape or any(type(x) is not int or x <= 0 for x in shape):
                raise ValueError('static PP tensor dimensions must be positive integers')
        allowed = (torch.bfloat16, torch.float32, torch.float64)
        if hidden_dtype not in allowed or output_dtype not in allowed:
            raise ValueError('unsupported static PP activation dtype')
        self.hidden_shape, self.output_shape = tuple(hidden_shape), tuple(output_shape)
        self.hidden_dtype, self.output_dtype = hidden_dtype, output_dtype


class StaticExternalGradientPP2(torch.nn.Module):
    """Two-stage autograd boundary supporting a subsequent multi-role DMD loss.

    Vanilla GPipe's last-stage loss callback cannot launch a *whole other role*
    while its first stage waits for the generator backward. This boundary keeps
    the generator graph, executes Fake/Teacher forwards on both stages, then
    propagates the external DMD gradient through PP2. It does not invent a DMD
    loss, select samples, synchronize DP gradients, or imply native task timing.

    The transport is deliberately synchronous and unoptimized. A native binding
    still needs real half-model modules, exact prefix/postfix math and eight
    independent B1 microbatches per DP replica before a single optimizer commit.
    """
    def __init__(self, stage_module, *, stage_index, group, device, signatures):
        super().__init__()
        if not dist.is_initialized() or dist.get_world_size(group) != 2:
            raise ValueError('explicit PP2 group required')
        if stage_index not in (0, 1) or dist.get_rank(group) != stage_index:
            raise ValueError('PP rank/stage mismatch')
        if not isinstance(stage_module, torch.nn.Module) or not signatures:
            raise ValueError('requires an actual stage module and admitted signatures')
        if any(not isinstance(s, StaticPipelineSignature) for s in signatures.values()):
            raise ValueError('every shape key needs an explicit static signature')
        self.stage_module = stage_module
        self.stage_index, self.group, self.device = stage_index, group, torch.device(device)
        self.signatures = dict(signatures)
        self.peer_rank = dist.get_global_rank(group, 1 - stage_index)
        self.application_forwards = self.grad_forwards = 0

    def _send(self, value):
        dist.send(value.contiguous(), dst=self.peer_rank, group=self.group)

    def _receive(self, shape, dtype):
        value = torch.empty(shape, dtype=dtype, device=self.device)
        dist.recv(value, src=self.peer_rank, group=self.group)
        return value

    def forward(self, value, *, signature_key, **stage_kwargs):
        if signature_key not in self.signatures:
            raise ValueError('shape/history layout not admitted by native stage binding')
        signature = self.signatures[signature_key]
        gradient_enabled = torch.is_grad_enabled()
        if self.stage_index == 0:
            hidden = self.stage_module(value, **stage_kwargs)
            if hidden.shape != signature.hidden_shape or hidden.dtype != signature.hidden_dtype:
                raise ValueError('actual first-stage activation violates its PP signature')
            if gradient_enabled and not hidden.requires_grad:
                raise ValueError('grad path has no first-stage trainable graph')
            if gradient_enabled:
                output = _Stage0ExternalBoundary.apply(hidden, self, signature)
            else:
                self._send(hidden)
                output = self._receive(signature.output_shape, signature.output_dtype)
        else:
            if gradient_enabled:
                dummy = torch.empty((), device=self.device, requires_grad=True)
                hidden = _Stage1ExternalBoundary.apply(dummy, self, signature)
            else:
                hidden = self._receive(signature.hidden_shape, signature.hidden_dtype)
            output = self.stage_module(hidden, **stage_kwargs)
            if output.shape != signature.output_shape or output.dtype != signature.output_dtype:
                raise ValueError('actual last-stage output violates its PP signature')
            if gradient_enabled and not output.requires_grad:
                raise ValueError('grad path has no last-stage trainable graph')
            self._send(output.detach())
        self.application_forwards += 1
        self.grad_forwards += int(gradient_enabled)
        return output
