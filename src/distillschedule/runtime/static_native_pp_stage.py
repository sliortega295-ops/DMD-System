"""Source-derived PP halves with explicit tensor and parameter ownership.

The original prefix belongs to stage0, original postfix to stage1. All five
conditioning/activation tensors cross the autograd boundary, so prefix LoRA
gradients include both halves without duplicating root parameters. No model
math is reimplemented here; no native GPU/cycle result is implied.
"""
from __future__ import annotations

import copy
import torch
from torch import nn

from .static_native_forward import AdmittedForwardSplit


class _UnownedBlock(nn.Module):
    def forward(self, *_args, **_kwargs):
        raise RuntimeError('attempt to execute a block owned by the other PP stage')


class NativeSourcePPStage(nn.Module):
    TRANSPORT_FIELDS = ('hidden_states', 'encoder_hidden_states', 'timestep_proj',
                        'temb', 'rotary_emb')

    def __init__(self, model, *, split, stage_index, context_template,
                 remove_root_modules):
        super().__init__()
        if stage_index not in (0, 1) or not isinstance(split, AdmittedForwardSplit):
            raise ValueError('requires an admitted source split and PP2 coordinate')
        if len(model.blocks) != split.block_count or hasattr(model, '_static_pp_stage_index'):
            raise ValueError('unexpected/already-pruned PP model')
        if context_template.get('gan_mode', False) or context_template.get('enable_navit', False):
            raise ValueError('GAN/NaViT needs its own explicit PP transport contract')
        if context_template.get('return_dict', True):
            raise ValueError('native pipeline postfix must return the original tuple')
        for name in self.TRANSPORT_FIELDS:
            if not isinstance(context_template.get(name), torch.Tensor):
                raise ValueError('PP transport needs actual native tensor field: ' + name)
        metadata = {k: v for k, v in context_template.items() if k not in self.TRANSPORT_FIELDS}
        if any(isinstance(v, torch.Tensor) for v in metadata.values()):
            raise ValueError('unregistered prefix tensor cannot be silently omitted from PP')
        if (not isinstance(remove_root_modules, tuple)
                or any(not isinstance(getattr(model, n, None), nn.Module) or n == 'blocks'
                       for n in remove_root_modules)):
            raise ValueError('root ownership must name existing whole modules explicitly')
        self.model = model
        self.split = split
        self.stage_index = stage_index
        self.start = stage_index * (split.block_count // 2)
        self.stop = self.start + split.block_count // 2
        self.context_metadata = copy.deepcopy(metadata)
        self.removed_roots = tuple(remove_root_modules)
        for name in remove_root_modules:
            setattr(model, name, None)
        for i in range(split.block_count):
            if not self.start <= i < self.stop:
                model.blocks[i] = _UnownedBlock()
        model._static_pp_stage_index = stage_index

    def forward(self, *args, **kwargs):
        if self.stage_index == 0:
            if kwargs.get('gan_mode', False) or kwargs.get('attention_kwargs'):
                raise ValueError('unsupported GAN/scaled-adapter pipeline call')
            if kwargs.get('return_dict', False):
                raise ValueError('native pipeline requires tuple outputs')
            kwargs = dict(kwargs, return_dict=False)
            context = self.split.prefix(self.model, *args, **kwargs)
            if context.get('enable_navit', False):
                raise ValueError('NaViT was not admitted by this transport')
            context = self.split.blocks(self.model, context, self.start, self.stop)
            fields = tuple(context[name].contiguous() for name in self.TRANSPORT_FIELDS)
            if any(not isinstance(v, torch.Tensor) for v in fields):
                raise ValueError('actual native conditioning violated tensor transport')
            return fields
        if kwargs or len(args) != len(self.TRANSPORT_FIELDS):
            raise ValueError('last stage takes precisely its five native boundary tensors')
        context = dict(self.context_metadata)
        context.update(zip(self.TRANSPORT_FIELDS, args, strict=True))
        context = self.split.blocks(self.model, context, self.start, self.stop)
        output = self.split.suffix(self.model, context)
        if not isinstance(output, tuple) or not output or not isinstance(output[0], torch.Tensor):
            raise ValueError('last stage did not return its native tensor output')
        return output[0]

    def receipt(self):
        return dict(stage_index=self.stage_index, block_range=[self.start, self.stop],
            transported_native_fields=list(self.TRANSPORT_FIELDS),
            removed_root_modules=list(self.removed_roots),
            local_parameter_numel=sum(p.numel() for p in self.model.parameters()),
            pp_root_parameter_replication=False, prefix_and_postfix_math='ADMITTED_SOURCE_AST',
            native_source=self.split.receipt, GPU='NOT_RUN_COMPONENT_ONLY', native_timing='NOT_RUN')
