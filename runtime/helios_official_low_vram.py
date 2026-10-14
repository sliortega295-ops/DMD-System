"""Pinned official Helios training role mover; no loss/model implementation.

The class body below is byte-identical to the upstream source fragment at
PKU-YuanGroup/Helios@babed9811266e4b5b111c9c1e0977a07899066ab.
Class fragment SHA256: a47f363141b284977e4c7424f9a53970c2df829010d80f397fecc74e3a87cc5b.
The GPU ZeRO-1 binding adapter is kept separately in the typed runtime.
"""
import torch


def free_memory():
    from helios.utils.utils_helios_post import free_memory as implementation
    return implementation()


class OptimizedLowVRAMManager:
    def __init__(self):
        self.pinned_models = set()
        self.grad_cache = {}

    def move_to_cpu(self, model, non_blocking=True, offload_grad=False):
        model_to_move = model.module if hasattr(model, "module") else model
        model_to_move.to("cpu", non_blocking=non_blocking)

        if id(model) not in self.pinned_models:
            for buffer in model_to_move.buffers():
                if buffer.device.type == "cpu" and not buffer.is_pinned():
                    buffer.data = buffer.data.pin_memory()
            self.pinned_models.add(id(model))

        if offload_grad:
            model_id = id(model)

            if model_id not in self.grad_cache:
                self.grad_cache[model_id] = {}

            for i, param in enumerate(model_to_move.parameters()):
                if param.grad is not None:
                    if i not in self.grad_cache[model_id]:
                        self.grad_cache[model_id][i] = torch.empty_like(param.grad, device="cpu", pin_memory=True)

                    self.grad_cache[model_id][i].copy_(param.grad, non_blocking=non_blocking)
                    param.grad = None

        free_memory()

    def move_to_gpu(self, model, device, non_blocking=True, load_grad=False):
        model_to_move = model.module if hasattr(model, "module") else model
        model_to_move.to(device, non_blocking=non_blocking)

        if load_grad:
            model_id = id(model)
            if model_id in self.grad_cache:
                for i, param in enumerate(model_to_move.parameters()):
                    if i in self.grad_cache[model_id]:
                        if param.grad is None:
                            param.grad = self.grad_cache[model_id][i].to(device, non_blocking=non_blocking)
                        else:
                            param.grad.copy_(self.grad_cache[model_id][i], non_blocking=non_blocking)
