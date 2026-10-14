"""Official whole-role CPU/GPU movement with native BF16 optimizer bindings.

Only model parameter storage moves. FP32 masters, Adam, accumulated gradients
and EMA remain GPU-resident. Restoring LoRA views is an integration requirement:
Module.to() otherwise detaches them from DeepSpeed's BF16 flat update buffer.
"""
from __future__ import annotations

from typing import Any


def rebind_native_bf16_trainables(engine: Any) -> int:
    zero = engine.optimizer
    if type(zero).__name__ != "BF16_Optimizer":
        raise TypeError("official-swap baseline requires the reviewed native BF16 ZeRO-1 API")
    seen = 0
    for parameters, flat in zip(zero.bf16_groups, zero.bf16_groups_flat, strict=True):
        offset = 0
        for parameter in parameters:
            count = parameter.numel()
            view = flat.narrow(0, offset, count).view(parameter.shape)
            if parameter.dtype != view.dtype or view.device.type != "cuda":
                raise AssertionError("LoRA BF16 GPU compute/storage contract changed")
            parameter.data = view
            if parameter.data_ptr() != view.data_ptr():
                raise AssertionError("LoRA update-buffer alias restoration failed")
            offset += count
            seen += 1
        if offset > flat.numel():
            raise AssertionError("native optimizer flat-buffer layout is incomplete")
    if not seen:
        raise AssertionError("empty native BF16 optimizer trainable inventory")
    return seen


class OfficialRoleSwap:
    def __init__(self):
        from runtime.helios_official_low_vram import OptimizedLowVRAMManager
        self.manager = OptimizedLowVRAMManager()
        self.engines: dict[int, Any] = {}
        self.state: dict[int, str] = {}
        self.roles: dict[int, str] = {}
        self.transfers = {"H2D": {"bytes": 0, "events": 0},
                          "D2H": {"bytes": 0, "events": 0}}
        self.alias_checks = 0

    def register(self, engine: Any, *, role: str) -> None:
        module = engine.module
        if role not in ("generator", "critic") or role in self.roles.values():
            raise ValueError("role registration changed")
        self.engines[id(module)] = engine
        self.state[id(module)] = "cuda"
        self.roles[id(module)] = role
        self.alias_checks += rebind_native_bf16_trainables(engine)

    def _registered(self, model: Any) -> tuple[Any, Any]:
        module = model.module if hasattr(model, "module") else model
        if id(module) not in self.engines:
            raise ValueError("unregistered role; never move an unknown model")
        return module, self.engines[id(module)]

    def _bytes(self, module: Any) -> int:
        return sum(p.numel() * p.element_size() for p in module.parameters())

    def move_to_cpu(self, model: Any, non_blocking=True, offload_grad=False) -> None:
        if offload_grad:
            raise ValueError("official baseline retains dmd_is_offload_grad=false")
        module, _engine = self._registered(model)
        if self.state[id(module)] == "cpu":
            return
        identities = [id(p) for p in module.parameters()]
        size = self._bytes(module)
        self.manager.move_to_cpu(module, non_blocking=non_blocking, offload_grad=False)
        if identities != [id(p) for p in module.parameters()]:
            raise AssertionError("Module.to changed Parameter objects; optimizer hooks are invalid")
        if any(p.device.type != "cpu" for p in module.parameters()):
            raise AssertionError("official inactive role did not reach CPU")
        self.state[id(module)] = "cpu"
        self.transfers["D2H"]["bytes"] += size
        self.transfers["D2H"]["events"] += 1

    def move_to_gpu(self, model: Any, device: Any, non_blocking=True, load_grad=False) -> None:
        if load_grad:
            raise ValueError("official baseline retains dmd_is_offload_grad=false")
        module, engine = self._registered(model)
        if self.state[id(module)] == "cuda":
            return
        identities = [id(p) for p in module.parameters()]
        size = self._bytes(module)
        self.manager.move_to_gpu(module, device, non_blocking=non_blocking, load_grad=False)
        if identities != [id(p) for p in module.parameters()]:
            raise AssertionError("Module.to changed Parameter objects; optimizer hooks are invalid")
        self.alias_checks += rebind_native_bf16_trainables(engine)
        if any(p.device.type != "cuda" for p in module.parameters()):
            raise AssertionError("official active role did not reach GPU")
        self.state[id(module)] = "cuda"
        self.transfers["H2D"]["bytes"] += size
        self.transfers["H2D"]["events"] += 1

    def activate(self, role: str, device: Any) -> None:
        if role not in self.roles.values():
            raise ValueError("unknown role")
        for key, current in self.roles.items():
            if current != role:
                self.move_to_cpu(self.engines[key])
        key = next(key for key, value in self.roles.items() if value == role)
        self.move_to_gpu(self.engines[key], device)

    def receipt(self) -> dict[str, Any]:
        return {"backend": "OFFICIAL_HELIOS_WHOLE_ROLE_SWAP_GPU_ZERO1",
                "official_head": "babed9811266e4b5b111c9c1e0977a07899066ab",
                "official_manager_fragment_sha256": "a47f363141b284977e4c7424f9a53970c2df829010d80f397fecc74e3a87cc5b",
                "parameter_offload": True, "parameter_streaming": False,
                "activation_offload": False, "optimizer_cpu_offload": False,
                "dmd_is_offload_grad": False, "trainable_alias_checks": self.alias_checks,
                "application_parameter_copy_bytes": self.transfers,
                "final_role_devices": {self.roles[k]: value for k, value in self.state.items()},
                "byte_scope": "application tensor sizes; not measured CUPTI memcpy traffic"}
