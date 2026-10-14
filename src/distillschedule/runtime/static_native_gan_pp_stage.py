"""Exact native GAN tap transport for an already partitioned Helios PP stage.

This is a transport component, not a completed GAN worker. It reuses the
source-pinned prefix/block/suffix functions and original head arithmetic.
The caller must join each pipeline invocation before switching GAN/adapter
scopes and must still bind structured outputs, input gradients and optimizer
ownership. No target-GPU acceptance is implied by constructing this module.
"""
from __future__ import annotations

import copy

import torch
from torch import nn

from .static_native_pp_stage import NativeSourcePPStage


class NativeSourceGANPPStage(nn.Module):
    """GAN-only view of the same physical parameters as a native score stage.

    Stage0 sends the original five boundary tensors followed by its ordered
    native GAN taps. Stage1 appends its own taps, then executes the untouched
    native suffix returning ``(sample, logits)``. Separate tensors avoid
    packing, recoding or changing their numeric values.
    """

    def __init__(self, native_stage: NativeSourcePPStage):
        super().__init__()
        if not isinstance(native_stage, NativeSourcePPStage):
            raise TypeError("requires the existing admitted native PP stage")
        model = native_stage.model
        if not bool(getattr(model, "is_use_gan", False)):
            raise ValueError("native GAN was not enabled in the source model")
        names = set(native_stage.split.receipt["context_names"])
        if not {"gan_mode", "logits_hidden"}.issubset(names):
            raise ValueError("source split lacks native GAN context fields")
        if native_stage.context_metadata.get("enable_navit", False):
            raise ValueError("NaViT requires another transport contract")
        hooks = tuple(getattr(model, "gan_hooks", ())) if model.is_use_gan_hooks else ()
        if any(type(i) is not int or not 0 <= i < native_stage.split.block_count for i in hooks):
            raise ValueError("native GAN hook inventory is malformed")
        if tuple(sorted(set(hooks))) != hooks:
            raise ValueError("native hook/head order must be explicitly increasing")
        self.hooks = hooks
        self.first_stage_hooks = tuple(i for i in hooks if i < native_stage.split.block_count // 2)
        self.local_hooks = tuple(i for i in hooks if native_stage.start <= i < native_stage.stop)
        self.model = model  # same parameter objects, not a clone or a shadow
        self.split = native_stage.split
        self.stage_index = native_stage.stage_index
        self.start, self.stop = native_stage.start, native_stage.stop
        self.context_metadata = copy.deepcopy(native_stage.context_metadata)
        self.context_metadata["gan_mode"] = True
        self.context_metadata["logits_hidden"] = []
        self.native_fields = native_stage.TRANSPORT_FIELDS
        self.owned_parameter_ids = frozenset(id(p) for p in native_stage.parameters())
        if frozenset(id(p) for p in self.parameters()) != self.owned_parameter_ids:
            raise ValueError("GAN view created or omitted physical parameters")
        if self.stage_index == 0:
            if model.is_use_gan_hooks and getattr(model, "gan_heads", None) is not None:
                raise ValueError("hook heads must be owned only by the suffix stage")
            if model.is_use_gan_final and getattr(model, "gan_final_head", None) is not None:
                raise ValueError("final GAN head must be owned only by the suffix stage")
        elif model.is_use_gan_hooks:
            if not isinstance(getattr(model, "gan_heads", None), nn.ModuleDict):
                raise ValueError("last stage is missing native hook heads")
            if len(model.gan_heads) != len(hooks):
                raise ValueError("native head/tap inventory changed")

    def _check_taps(self, values, count):
        if not isinstance(values, list) or len(values) != count:
            raise ValueError("native GAN tap count/order changed")
        if any(not isinstance(v, torch.Tensor) or v.ndim != 3 for v in values):
            raise ValueError("native hook activations must be BxTokensxHidden tensors")

    def forward(self, *args, **kwargs):
        if self.stage_index == 0:
            if kwargs.get("gan_mode") is not True or kwargs.get("attention_kwargs"):
                raise ValueError("GAN-only view requires explicit unscaled gan_mode=True")
            if kwargs.get("return_dict", False):
                raise ValueError("pipeline output contract requires the native tuple")
            context = self.split.prefix(self.model, *args, **dict(kwargs, return_dict=False))
            if context.get("enable_navit", False) or context.get("gan_mode") is not True:
                raise ValueError("unexpected native GAN/NaViT branch")
            context = self.split.blocks(self.model, context, self.start, self.stop)
            taps = context["logits_hidden"]
            self._check_taps(taps, len(self.local_hooks))
            values = tuple(context[name] for name in self.native_fields) + tuple(taps)
            if any(not isinstance(v, torch.Tensor) for v in values):
                raise ValueError("native GAN boundary contains a non-tensor field")
            return tuple(v.contiguous() for v in values)
        expected = len(self.native_fields) + len(self.first_stage_hooks)
        if kwargs or len(args) != expected:
            raise ValueError("GAN suffix requires every ordered native field and stage0 tap")
        context = dict(self.context_metadata)
        context["logits_hidden"] = list(args[len(self.native_fields):])
        self._check_taps(context["logits_hidden"], len(self.first_stage_hooks))
        context.update(zip(self.native_fields, args[:len(self.native_fields)], strict=True))
        context = self.split.blocks(self.model, context, self.start, self.stop)
        self._check_taps(context["logits_hidden"], len(self.hooks))
        output = self.split.suffix(self.model, context)
        if not isinstance(output, tuple) or len(output) != 2:
            raise ValueError("native GAN suffix output must remain (sample, logits)")
        if any(not isinstance(v, torch.Tensor) for v in output):
            raise ValueError("native GAN outputs must contain real sample and logits tensors")
        return output

    def receipt(self):
        return dict(
            component="native_gan_pp_tap_transport.v1",
            source_sha256=self.split.receipt["source_sha256"],
            stage_index=self.stage_index,
            block_range=[self.start, self.stop],
            native_fields=list(self.native_fields),
            native_stage0_hook_indices=list(self.first_stage_hooks),
            local_hook_indices=list(self.local_hooks),
            output="ORIGINAL_NATIVE_SAMPLE_AND_LOGITS",
            checkpoint_dispatch="ORIGINAL_SOURCE_AST",
            shared_parameter_objects=True,
            optimizations=False,
            target_GPU="NOT_RUN_COMPONENT_ONLY",
            complete_GAN_worker="NOT_BOUND",
        )
