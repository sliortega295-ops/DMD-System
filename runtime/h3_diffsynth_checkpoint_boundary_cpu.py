"""Opt-in native W6 checkpoint-input staging, not whole-graph offload.

Wrap only the source-pinned native checkpoint CALL. The context closes when
that call returns, so backward's checkpoint function never enters this wrapper
again. Original native tensor expressions, checkpoint functions and order are
untouched. Disabled mode leaves the function object unchanged.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import inspect
from pathlib import Path

import torch


PINNED_NATIVE_CHECKPOINT_FILE_SHA256 = (
    "00f33981e4ac155d1ae757de2765703cd4881775b9cf8360627c709d631c2b21"
)


class NativeCheckpointBoundaryCPU:
    def __init__(self, native_module, blocks, role, *, enabled=False):
        self.enabled = enabled
        self.native_module = native_module
        self.original = native_module.gradient_checkpoint_forward
        self.blocks = {id(block): index for index, block in enumerate(blocks)}
        self.role = role
        self.counts = Counter()
        self.by_role = Counter()
        self.sizes = Counter()
        if not enabled:
            return
        path = inspect.getsourcefile(self.original)
        if not path or hashlib.sha256(Path(path).read_bytes()).hexdigest() != PINNED_NATIVE_CHECKPOINT_FILE_SHA256:
            raise ValueError("W6 checkpoint source identity changed")
        if not self.blocks:
            raise ValueError("native main block boundaries required")
        native_module.gradient_checkpoint_forward = self._forward

    def _forward(self, model, checkpoint, native_offload, *args, **kwargs):
        if id(model) not in self.blocks or not checkpoint or not torch.is_grad_enabled():
            return self.original(model, checkpoint, native_offload, *args, **kwargs)
        if native_offload:
            raise RuntimeError("do not nest two checkpoint offload policies")
        namespace = self.original.__globals__
        if namespace.get("_HAS_DEEPSPEED") and namespace["deepspeed"].checkpointing.is_configured():
            raise RuntimeError("native torch checkpoint path required")
        role = self.role()
        if role not in ("student", "fake_score"):
            raise RuntimeError("only native grad Student/Fake may stage boundaries")
        self.counts["checkpoint_calls"] += 1
        self.by_role[role] += 1
        # Reuse PyTorch's pack/unpack implementation, adding metadata-only
        # counters. No additional tensor copy or synchronization is introduced.
        cpu = torch.autograd.graph.save_on_cpu(pin_memory=True)

        def pack(tensor):
            result = cpu.pack_hook(tensor)
            nbytes = tensor.numel() * tensor.element_size()
            if nbytes:
                self.counts["saved_input_count"] += 1
            if tensor.device.type == "cuda" and nbytes:
                if result[1].device.type != "cpu" or not result[1].is_pinned():
                    raise RuntimeError("checkpoint boundary must use pinned CPU")
                self.counts["D2H_count"] += 1
                self.counts["D2H_bytes"] += nbytes
                self.sizes[nbytes] += 1
            return result

        def unpack(packed):
            device, tensor = packed
            nbytes = tensor.numel() * tensor.element_size()
            if device.type == "cuda" and nbytes:
                self.counts["H2D_count"] += 1
                self.counts["H2D_bytes"] += nbytes
            return cpu.unpack_hook(packed)

        with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
            return self.original(model, checkpoint, False, *args, **kwargs)

    def snapshot(self):
        return {
            "enabled": self.enabled,
            "policy": "native_per_checkpoint_input_pinned_cpu" if self.enabled else "none",
            "checkpoint_blocks": len(self.blocks),
            "counters": dict(self.counts),
            "calls_by_role": dict(self.by_role),
            "copy_size_histogram": {str(k): v for k, v in sorted(self.sizes.items())},
            "recompute_hook_scope": "outside_offload_wrapper",
        }

    def finish_cycle(self):
        result = self.snapshot()
        if self.enabled:
            expected = 2 * len(self.blocks)  # two independent logical B1 rows
            if self.by_role != {"student": expected, "fake_score": expected}:
                raise RuntimeError("FAIL_W6_NATIVE_BOUNDARY_CALL_CENSUS")
            if self.counts["D2H_count"] < expected * 2:
                raise RuntimeError("FAIL_W6_ACTUAL_PINNED_BOUNDARY_COPY_CENSUS")
            if self.counts["H2D_count"] != self.counts["D2H_count"] or self.counts["H2D_bytes"] != self.counts["D2H_bytes"]:
                raise RuntimeError("FAIL_W6_BOUNDARY_CONSUMPTION_RECONCILIATION")
        self.counts.clear(); self.by_role.clear(); self.sizes.clear()
        return result

    def close(self):
        if self.enabled:
            self.native_module.gradient_checkpoint_forward = self.original
