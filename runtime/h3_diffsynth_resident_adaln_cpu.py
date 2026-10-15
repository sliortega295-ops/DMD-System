"""Opt-in frozen AdaLN-only pinned-CPU parameter staging for Resident capacity.
Not a no-offload baseline. The native nn.Linear module/hooks and F.linear
expression remain; CPU canonical BF16 parameters are immutable and temporary
CUDA copies follow the current stream. Never apply to trainable timestep paths.
"""
from __future__ import annotations
from collections import Counter
import inspect
import hashlib
from pathlib import Path
from types import MethodType
import torch

PINNED_NATIVE_FILE_SHA256 = "bd617a4312c70405e916707e1908100b484b2259dfbeba4af7de5bb145206d80"

class FrozenAdaLNCPUStaging:
    def __init__(self, blocks, role, *, count=0):
        self.enabled = count > 0
        self.role = role
        self.layers = []
        self.counts = Counter()
        self.by_role = Counter()
        self.closed = False
        self.initial_D2H_bytes = 0
        if not self.enabled:
            return
        if count > len(blocks) or count < 1:
            raise ValueError("selected AdaLN block count exceeds actual model")
        selected = [block.adaln_proj for block in list(blocks)[:count]]
        for projection in selected:
            source = inspect.getsourcefile(type(projection))
            if (type(projection).__name__ != "MiniMaxH3AdalnProj" or not source
                    or hashlib.sha256(Path(source).read_bytes()).hexdigest() != PINNED_NATIVE_FILE_SHA256):
                raise ValueError("native AdaLN source identity changed")
            linear = projection.linear
            if type(linear) is not torch.nn.Linear:
                raise ValueError("native frozen AdaLN Linear required")
            if any(p.requires_grad or p.dtype != torch.bfloat16 or not p.is_contiguous()
                   for p in linear.parameters()):
                raise ValueError("only frozen contiguous BF16 AdaLN parameters may stage")
        for index, projection in enumerate(selected):
            linear = projection.linear
            # Do not retain an alias to the original CUDA parameters.
            linear.weight = torch.nn.Parameter(self._copy_master(linear.weight), requires_grad=False)
            linear.bias = torch.nn.Parameter(self._copy_master(linear.bias), requires_grad=False)
            nbytes = sum(p.numel() * p.element_size() for p in linear.parameters())
            self.initial_D2H_bytes += nbytes
            self.layers.append((index, linear, linear.weight._version, linear.bias._version, nbytes))
            owner = self
            def forward(module, input, owner=owner):
                return owner._forward(module, input)
            linear.forward = MethodType(forward, linear)

    def _copy_master(self, parameter):
        if parameter.device.type != "cuda":
            raise ValueError("production staging requires an already loaded CUDA model")
        result = parameter.detach().to("cpu").pin_memory()
        if not result.is_pinned():
            raise RuntimeError("canonical AdaLN CPU parameters must be pinned")
        return result

    def _check_input(self, value):
        if (value.device.type != "cuda" or value.requires_grad or
                not torch.is_autocast_enabled("cuda") or
                torch.get_autocast_dtype("cuda") != torch.bfloat16):
            raise RuntimeError("frozen timestep input and native CUDA BF16 autocast required")

    def _forward(self, linear, value):
        if self.closed:
            raise RuntimeError("staged model cannot be reused after teardown")
        self._check_input(value)
        role = self.role()
        if role not in ("student", "fake_score", "teacher"):
            raise RuntimeError("unknown native AdaLN role")
        if any(p.device.type != "cpu" or p.dtype != torch.bfloat16 or p.requires_grad
               for p in linear.parameters()):
            raise RuntimeError("canonical frozen AdaLN placement/dtype changed")
        weight = linear.weight.to(value.device, non_blocking=True)
        bias = linear.bias.to(value.device, non_blocking=True)
        self.counts["linear_calls"] += 1
        self.counts["H2D_count"] += 2
        self.counts["H2D_bytes"] += weight.numel() * weight.element_size() + bias.numel() * bias.element_size()
        self.by_role[role] += 1
        # Same native nn.Linear expression, dtype and module invocation.
        result = torch.nn.functional.linear(value, weight, bias)
        if result.requires_grad:
            raise RuntimeError("unexpected autograd graph through frozen timestep projection")
        return result

    def snapshot(self):
        return dict(enabled=self.enabled, policy="selected_frozen_AdaLN_pinned_CPU_parameters",
                    selected_blocks=len(self.layers), initial_D2H_bytes_excluded_from_training=self.initial_D2H_bytes,
                    canonical_CPU_bytes=self.initial_D2H_bytes, counters=dict(self.counts),
                    calls_by_role=dict(self.by_role), optimizer_or_loss_changed=False,
                    parameter_values_or_dtype_changed=False,
                    parameter_offload=True if self.enabled else False,
                    no_offload_Resident_baseline=False if self.enabled else None)

    def finish_cycle(self):
        result = self.snapshot()
        if self.enabled:
            k = len(self.layers)
            expected = {"student": 20*k, "fake_score": 6*k, "teacher": 2*k}
            if self.by_role != expected or self.counts["linear_calls"] != 28*k:
                raise RuntimeError("FAIL_FROZEN_ADALN_APPLICATION_PLUS_REPLAY_CENSUS")
            nbytes = sum(row[4] for row in self.layers)
            if self.counts["H2D_bytes"] != 28*nbytes or self.counts["H2D_count"] != 56*k:
                raise RuntimeError("FAIL_FROZEN_ADALN_H2D_BYTES_RECONCILIATION")
            for _, layer, wv, bv, _ in self.layers:
                if layer.weight._version != wv or layer.bias._version != bv:
                    raise RuntimeError("canonical frozen AdaLN parameters mutated")
        self.counts.clear()
        self.by_role.clear()
        return result

    def close(self):
        # Parameters intentionally stay CPU through teardown. Never rematerialize
        # weights after an OOM just to restore the default forward.
        self.closed = True
