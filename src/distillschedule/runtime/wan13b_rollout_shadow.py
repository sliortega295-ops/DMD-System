"""Default-off, version-bound merged execution weights for post-G B1 rollouts.

The training model, gradient reducer, optimizer and EMA are never modified.
Only the five Critic-input rollouts may use the independent frozen shadow.
BF16 weight merging is floating-point reassociation, NOT bitwise equivalence.
"""
from __future__ import annotations

import copy
import os
import time
from typing import Any

ENV = "DISTILLSCHEDULE_WAN13B_GENERATOR_ROLLOUT_MERGED_SHADOW"
PLACEMENT = "resident-async128-tree-chunk2048"


def requested() -> bool:
    value = os.environ.get(ENV, "0")
    if value not in {"0", "1"}:
        raise ValueError(f"{ENV} must be 0 or 1")
    return value == "1"


class MergedRolloutShadow:
    """An unregistered inference replica; refresh and GPU work are timed by caller."""

    def __init__(self, source: Any, *, linear_type: type[Any] | None = None) -> None:
        import torch

        if linear_type is None:
            from peft.tuners.lora.layer import Linear

            linear_type = Linear
        self.source = source
        layers = [(name, child) for name, child in source.named_modules()
                  if isinstance(child, linear_type)]
        if not layers:
            raise ValueError("merged rollout shadow found no LoRA Linear modules")
        adapters: dict[str, str] = {}
        for name, layer in layers:
            active = tuple(layer.active_adapters)
            if len(active) != 1 or layer.disable_adapters or layer.merged:
                raise ValueError(f"{name}: require one enabled unmerged adapter")
            adapter = active[0]
            if adapter not in layer.lora_A or layer.lora_variant:
                raise ValueError(f"{name}: unsupported LoRA adapter/variant")
            if getattr(layer.lora_dropout[adapter], "p", 0.0) != 0.0:
                raise ValueError(f"{name}: nonzero LoRA dropout cannot be merged")
            if getattr(layer, "lora_bias", {}).get(adapter, False):
                raise ValueError(f"{name}: LoRA bias is outside this canary")
            for weight in (layer.lora_A[adapter].weight, layer.lora_B[adapter].weight):
                if weight.dtype != torch.float32:
                    raise ValueError(f"{name}: source LoRA must remain FP32")
            if hasattr(layer.base_layer.weight, "device_mesh"):
                raise ValueError("merged shadow requires original Resident tensors")
            adapters[name] = adapter
        allowed = {id(parameter) for _, layer in layers
                   for container in (layer.lora_A, layer.lora_B)
                   for parameter in container.parameters()}
        if any(parameter.requires_grad and id(parameter) not in allowed
               for parameter in source.parameters()):
            raise ValueError("trainable non-LoRA parameter cannot use static shadow")

        # Called before any instance-local forward/state/audit wrappers are installed.
        self.model = copy.deepcopy(source)
        for name, _ in layers:
            child = self.model.get_submodule(name)
            parent_name, _, leaf = name.rpartition(".")
            parent = self.model.get_submodule(parent_name) if parent_name else self.model
            setattr(parent, leaf, child.base_layer)
        self.model.requires_grad_(False)
        self.model.train(source.training)
        self.layers = [(name, layer, self.model.get_submodule(name), adapters[name])
                       for name, layer in layers]
        self.current_source_version: int | None = None
        self.receipt: dict[str, Any] = {
            "requested": True,
            "enabled": True,
            "scope": "FIVE_POST_GENERATOR_CRITIC_INPUT_B1_ROLLOUTS_ONLY",
            "numerical_class": "BF16_EXECUTION_WEIGHT_REASSOCIATION_REQUIRES_PARITY",
            "training_master_unchanged": True,
            "optimizer_reducer_ema_unchanged": True,
            "lora_layer_count": len(layers),
            "source_lora_dtype": "torch.float32",
            "shadow_parameter_bytes": sum(p.numel() * p.element_size()
                                          for p in self.model.parameters()),
            "shadow_lora_parameter_count": sum("lora_" in name for name, _ in
                                               self.model.named_parameters()),
            "refresh_count": 0,
            "refresh_host_enqueue_seconds": [],
            "refresh_versions": [],
            "rollout_entry_count": 0,
            "model_call_count": 0,
            "entries": [],
            "active": False,
        }
        if self.receipt["shadow_lora_parameter_count"] != 0:
            raise ValueError("shadow retained registered LoRA state")

    def begin_rollout(self, *, version: int, logical_steps: tuple[int, ...]) -> None:
        import torch

        version = int(version)
        if self.receipt["active"]:
            raise RuntimeError("nested merged-shadow rollout scope")
        if len(logical_steps) != 1:
            raise ValueError("merged-shadow canary freezes physical rollout B1")
        if self.current_source_version != version:
            if self.current_source_version is not None and version != self.current_source_version + 1:
                raise ValueError("Generator version must advance exactly once per cycle")
            started = time.perf_counter()
            with torch.no_grad():
                for name, source, destination, adapter in self.layers:
                    if tuple(source.active_adapters) != (adapter,) or source.disable_adapters or source.merged:
                        raise ValueError(f"{name}: Generator adapter state changed")
                    # Always rebuild from the immutable original base, never unmerge.
                    # get_delta_weight preserves PEFT scaling and fan_in/fan_out.
                    delta = source.get_delta_weight(adapter)
                    destination.weight.copy_(source.base_layer.weight.float().add(delta))
            self.current_source_version = version
            self.receipt["refresh_count"] += 1
            self.receipt["refresh_versions"].append(version)
            self.receipt["refresh_host_enqueue_seconds"].append(time.perf_counter() - started)
        self.receipt["rollout_entry_count"] += 1
        self.receipt["entries"].append({"version": version, "logical_steps": list(logical_steps)})
        self.receipt["active"] = True

    def end_rollout(self) -> None:
        if not self.receipt["active"]:
            raise RuntimeError("merged-shadow scope was not active")
        self.receipt["active"] = False

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        import torch

        if torch.is_grad_enabled() or not self.receipt["active"]:
            raise RuntimeError("merged shadow may execute only in post-G no-grad scope")
        self.receipt["model_call_count"] += 1
        return self.model(*args, **kwargs)

    def enable_adapters(self) -> None:
        # The original adapter was validated; its update is folded into shadow weights.
        if not self.receipt["active"]:
            raise RuntimeError("shadow adapter request outside post-G scope")

    def train(self, mode: bool = True) -> Any:
        if bool(mode) != bool(self.model.training):
            raise RuntimeError("shadow cannot change model mode during rollout")
        return self


def install(source: Any, *, role: str, placement: str) -> dict[str, Any]:
    enabled = requested()
    receipt: dict[str, Any] = {"requested": enabled, "enabled": False, "role": role}
    if not enabled:
        receipt["disabled_reason"] = "NOT_REQUESTED"
        return receipt
    if placement != PLACEMENT:
        raise ValueError("merged rollout shadow requires the existing Resident winner")
    if role != "generator":
        receipt["disabled_reason"] = "GENERATOR_ONLY"
        return receipt
    controller = MergedRolloutShadow(source)
    object.__setattr__(source, "_distillschedule_merged_rollout_shadow", controller)
    controller.receipt["role"] = role
    return controller.receipt


def validate(build: dict[str, Any], *, enabled: bool, cycles: int) -> dict[str, Any]:
    generator = build.get("generator", {}).get("metadata", {}).get("generator_rollout_merged_shadow", {})
    critic = build.get("critic", {}).get("metadata", {}).get("generator_rollout_merged_shadow", {})
    if enabled:
        expected = {"requested": True, "enabled": True, "refresh_count": cycles,
                    "rollout_entry_count": 5 * cycles, "model_call_count": 27 * cycles,
                    "shadow_lora_parameter_count": 0, "active": False,
                    "training_master_unchanged": True, "optimizer_reducer_ema_unchanged": True}
        for key, value in expected.items():
            if generator.get(key) != value:
                raise AssertionError(f"merged shadow census changed {key}: {generator.get(key)} != {value}")
        if generator.get("refresh_versions") != list(range(3, 3 + cycles)):
            raise AssertionError("merged shadow must bind G3, G4, ... commits")
        entries = generator.get("entries", [])
        wanted = [{"version": 3 + cycle, "logical_steps": [step]}
                  for cycle in range(cycles) for step in range(10, 15)]
        if entries != wanted:
            raise AssertionError("merged shadow sample/step routing changed")
        if critic.get("enabled") or critic.get("disabled_reason") != "GENERATOR_ONLY":
            raise AssertionError("Critic must not use merged execution weights")
    elif generator.get("enabled") or critic.get("enabled"):
        raise AssertionError("unrequested merged shadow executed")
    return {"enabled": enabled, "generator": generator, "critic": critic}


def expected_epilogue_counts(*, cycles: int, shadow_enabled: bool) -> dict[str, int]:
    # Original generator no-grad census is 28 calls/cycle; exactly 27 are the
    # five post-commit rollouts. Only the remaining call keeps PEFT epilogues.
    calls = (1 if shadow_enabled else 28) * cycles
    return {"model_call_count": calls, "fused_invocation_count": 305 * calls,
            "reference_fallback_count": 4 * calls}
