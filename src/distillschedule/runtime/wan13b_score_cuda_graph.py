"""Default-off CUDA graphs for the three detached pre-G Critic/Teacher scores.

Resident only; no weight merging, parameter copy, new random operation, or
training/backward graph. Adapter-on and adapter-off have separate captures.
Extra warmup/capture/reference calls are diagnostics, not application calls.
"""
from __future__ import annotations

import functools
import os
from typing import Any

from .wan13b_rollout_cuda_graph import clone_tree, copy_tree, signature

ENV = "DISTILLSCHEDULE_WAN13B_SCORE_CUDA_GRAPH"
PLACEMENT = "resident-async128-tree-chunk2048"
MAX_ENTRIES = 4


def requested() -> bool:
    value = os.environ.get(ENV, "0")
    if value not in {"0", "1"}:
        raise ValueError(f"{ENV} must be 0 or 1")
    return value == "1"


def check_admission(placement: str, *, enabled: bool, role: str) -> bool:
    if not enabled:
        return False
    if placement != PLACEMENT:
        raise ValueError("Score CUDA graphs require full Resident placement")
    return role == "critic"


class ScoreCudaGraph:
    def __init__(self, source: Any, *, reference_checks: bool) -> None:
        self.source = source
        self.original_forward = source.forward
        self.reference_checks = reference_checks
        self.cache: dict[Any, Any] = {}
        self.pointer_contract = self.parameter_identity()
        self.state = getattr(source, "_distillschedule_resident_clean_state", None)
        if self.state is None:
            raise ValueError("Score CUDA graphs require the existing adapter state machine")
        self._audit_source()
        self.receipt: dict[str, Any] = {
            "requested": True, "enabled": True, "role": "critic",
            "scope": "THREE_DETACHED_SCORE_CALLS_BEFORE_GENERATOR_BACKWARD",
            "weights_merged": False, "training_backward_unchanged": True,
            "rng_ops": "NONE_ZERO_DROPOUT_PINNED_MODEL",
            "adapter_modes_separate": True, "pointer_identity_checks": 0,
            "replay_calls": 0, "adapter_on_calls": 0, "adapter_off_calls": 0,
            "warmup_calls": 0, "capture_calls": 0, "cache_entries": [],
            "reference_calls": 0, "reference_parity": [],
            "extra_calls_scope": "COLD_C1_AND_CORRECTNESS_ONLY_NOT_APPLICATION_CENSUS",
        }

    def _audit_source(self) -> None:
        import torch
        from peft.tuners.lora.layer import LoraLayer
        source = self.source
        for name, parameter in source.named_parameters():
            if parameter.device.type != "cuda" or hasattr(parameter, "device_mesh"):
                raise ValueError(f"{name}: Resident CUDA parameters required")
            if parameter.requires_grad and parameter.dtype != torch.float32:
                raise ValueError(f"{name}: FP32 LoRA master required")
        for name, child in source.named_modules():
            if isinstance(child, torch.nn.modules.dropout._DropoutNd) and child.p:
                raise ValueError(f"{name}: stochastic dropout cannot be captured")
            if isinstance(child, LoraLayer) and (len(child.active_adapters) != 1 or child.merged):
                raise ValueError(f"{name}: vanilla unmerged single adapter required")

    def parameter_identity(self) -> Any:
        return tuple((name, value.data_ptr(), tuple(value.shape), str(value.dtype),
                      str(value.device)) for name, value in self.source.named_parameters())

    def _capture(self, args: Any, kwargs: Any) -> Any:
        import torch
        static_args, static_kwargs = clone_tree(args), clone_tree(kwargs)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.original_forward(*static_args, **static_kwargs)
                self.receipt["warmup_calls"] += 1
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()  # FIRST CYCLE COLD CAPTURE ONLY.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = self.original_forward(*static_args, **static_kwargs)
        self.receipt["capture_calls"] += 1
        return static_args, static_kwargs, graph, output

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        import torch

        if torch.is_grad_enabled():
            return self.original_forward(*args, **kwargs)
        calls = self.receipt["replay_calls"]
        if calls % 3 == 0:
            if self.parameter_identity() != self.pointer_contract:
                raise ValueError("Resident Score parameter address changed")
            self.receipt["pointer_identity_checks"] += 1
        if not self.state.training or not self.source.training:
            raise ValueError("Score graphs must preserve train mode")
        adapter_enabled = bool(self.state.adapter_enabled)
        autocast = torch.is_autocast_enabled("cuda")
        key = (adapter_enabled, autocast, str(torch.get_autocast_dtype("cuda")),
               signature((args, kwargs)))
        if key not in self.cache:
            if calls >= 3 or len(self.cache) >= MAX_ENTRIES:
                raise ValueError("New Score graph metadata after cold cycle")
            self.cache[key] = self._capture(args, kwargs)
            self.receipt["cache_entries"].append({"adapter_enabled": adapter_enabled,
                                                  "signature": repr(key)})
        static_args, static_kwargs, graph, output = self.cache[key]
        copy_tree(static_args, args)
        copy_tree(static_kwargs, kwargs)
        graph.replay()
        result = clone_tree(output)
        self.receipt["replay_calls"] += 1
        self.receipt["adapter_on_calls" if adapter_enabled else "adapter_off_calls"] += 1
        if self.reference_checks:
            # Only trajectory/correctness runs use host-visible comparison.
            # Never included in native C2 performance experiments.
            from distillschedule.runtime.p0_parity import require_tensor_parity

            reference = self.original_forward(*args, **kwargs)
            a, b = reference[0], result[0]
            metrics = require_tensor_parity(a, b, label="Detached Resident Score CUDA graph")
            self.receipt["reference_calls"] += 1
            self.receipt["reference_parity"].append(metrics)
        return result


def install(source: Any, *, role: str, placement: str,
            reference_checks: bool) -> dict[str, Any]:
    enabled = requested()
    if not check_admission(placement, enabled=enabled, role=role):
        return {"requested": enabled, "enabled": False, "role": role}
    controller = ScoreCudaGraph(source, reference_checks=reference_checks)
    @functools.wraps(source.forward)
    def forward(*args: Any, **kwargs: Any) -> Any:
        return controller.forward(*args, **kwargs)

    source.forward = forward
    object.__setattr__(source, "_distillschedule_score_cuda_graph", controller)
    return controller.receipt


def validate_receipt(receipt: dict[str, Any], *, enabled: bool, cycles: int,
                     reference_checks: bool) -> None:
    if not enabled:
        if receipt.get("enabled"):
            raise AssertionError("Unrequested Score graph executed")
        return
    expected = {"requested": True, "enabled": True, "replay_calls": 3 * cycles,
                "adapter_on_calls": cycles, "adapter_off_calls": 2 * cycles,
                "pointer_identity_checks": cycles,
                "reference_calls": 3 * cycles if reference_checks else 0}
    for name, value in expected.items():
        if receipt.get(name) != value:
            raise AssertionError(f"Score graph census mismatch {name}")
    entries = len(receipt.get("cache_entries", []))
    if not 2 <= entries <= MAX_ENTRIES:
        raise AssertionError("Missing independent adapter-on/off Score graph")
    if receipt["capture_calls"] != entries or receipt["warmup_calls"] != 3 * entries:
        raise AssertionError("Score graph cold extra call inventory mismatch")
    if reference_checks and (len(receipt["reference_parity"]) != 3 * cycles or
                             not all(x.get("status") == "PASS" for x in receipt["reference_parity"])):
        raise AssertionError("Missing trained detached Score graph reference parity")


def validate_build(build: dict[str, Any], *, cycles: int,
                   reference_checks: bool) -> dict[str, Any]:
    generator = build.get("generator", {}).get("metadata", {}).get("score_cuda_graph", {})
    critic = build.get("critic", {}).get("metadata", {}).get("score_cuda_graph", {})
    if generator.get("enabled"):
        raise AssertionError("Score CUDA graph entered Generator")
    validate_receipt(critic, enabled=requested(), cycles=cycles,
                     reference_checks=reference_checks)
    return {"generator": generator, "critic": critic}
