"""Native Torch and optional Wan model adapters, with shape-aware LoRA binding.

The Torch network is an explicitly random, test-only flow network. Wan uses the
real diffusers WanTransformer3DModel and local checkpoints, never the old
Helios-pyramid Wan or LightX2V Grid1000 semantics. No network/model downloads.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .task import NativeTask, ensure


def tensor_hash(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    return hashlib.sha256(str((str(value.dtype), tuple(value.shape))).encode() +
                          value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def tensors_digest(values) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(values):
        h.update(name.encode()); h.update(tensor_hash(tensor).encode())
    return h.hexdigest()


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, task: NativeTask) -> None:
        super().__init__()
        self.base = base.requires_grad_(False)
        self.scale = float(task.lora_alpha / task.lora_rank)
        self.a = nn.Parameter(torch.empty(task.lora_rank, base.in_features, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(base.out_features, task.lora_rank, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.a, a=5**0.5)
        self.use_fused = False
        self.fused_calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base(x)
        update = F.linear(F.linear(x.float(), self.a), self.b)
        if self.use_fused and not torch.is_grad_enabled():
            from distillschedule.runtime.wan13b_lora_epilogue import fused_lora_epilogue
            self.fused_calls += 1
            return fused_lora_epilogue(base, update, self.scale)
        return (base.float() + update * self.scale).to(base.dtype)


def install_lora(model: nn.Module, task: NativeTask) -> list[str]:
    model.requires_grad_(False)
    found = [(name, mod) for name, mod in model.named_modules() if isinstance(mod, nn.Linear)
             and any(name == key or name.endswith("." + key) for key in task.target_modules)]
    ensure(found, "no matching LoRA targets in native model; cannot silently train zero parameters")
    for key in task.target_modules:
        ensure(any(n == key or n.endswith("." + key) for n, _ in found), f"unmatched LoRA target: {key}")
    for name, mod in found:
        parent, _, leaf = name.rpartition(".")
        setattr(model.get_submodule(parent) if parent else model, leaf, LoRALinear(mod, task))
    return [n for n, _ in found]


class _TinyFlow(nn.Module):
    def __init__(self, task: NativeTask) -> None:
        super().__init__()
        self.projection = nn.Linear(task.channels, task.channels)
        self.condition = nn.Linear(task.text_dim, task.channels, bias=False)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        y = self.projection(x.movedim(1, -1)).movedim(-1, 1)
        bias = self.condition(condition.mean(1))[:, :, None, None, None]
        return torch.tanh(y + bias + sigma[:, None, None, None, None].to(x.dtype))


def load_models(task: NativeTask) -> tuple[nn.ModuleDict, dict[str, Any]]:
    """Construct on CPU and validate architecture before any CUDA allocation."""
    ensure(task.fixture or task.backend == "diffusers-wan", "random reference model is fixture-only")
    models = {}
    sources = {}
    for i, role in enumerate(("generator", "critic", "teacher")):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(task.seed + i)
            if task.backend == "torch-reference":
                model = _TinyFlow(task)
            else:
                try:
                    import diffusers
                    from diffusers import WanTransformer3DModel
                except ImportError as exc:
                    raise RuntimeError("Wan adapter requires the optional native-wan dependencies") from exc
                if task.fixture:
                    ensure(task.model_config is not None, "Wan random fixture requires an explicit small model_config")
                    model = WanTransformer3DModel(**dict(task.model_config))
                else:
                    root = Path(task.checkpoints[role]).resolve()
                    ensure(root.is_dir(), f"local checkpoint directory missing for {role}")
                    payloads = sorted(root.glob("*.safetensors"))
                    ensure(payloads and (root / "config.json").is_file(), "local diffusers config/safetensors required")
                    # Hash real input files, not only user-provided identity strings.
                    from distillschedule.workloads.best_known.registry import file_sha256
                    sources[role] = {p.name: file_sha256(p) for p in [root / "config.json", *payloads]}
                    # Wan's native FP32-preservation contract requires the low-memory loader.
                    # Keep source precision until the one-shot native precision mapping below.
                    model, info = WanTransformer3DModel.from_pretrained(
                        str(root), local_files_only=True, use_safetensors=True,
                        low_cpu_mem_usage=True, torch_dtype=torch.float32, output_loading_info=True)
                    ensure(all(not info.get(key) for key in
                               ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")),
                           f"incomplete native checkpoint for {role}: {info}")
                cfg = model.config
                ensure(cfg.in_channels == task.channels and cfg.out_channels == task.channels,
                       "Wan latent channels mismatch; I2V/concatenated-history input is not admitted")
                ensure(cfg.text_dim == task.text_dim and cfg.added_kv_proj_dim is None,
                       "Wan adapter requires matching T2V text/no-image conditioning")
                ensure(all(s % p == 0 for s, p in zip(task.latent_shape[2:], cfg.patch_size)),
                       "latent shape must be divisible by the native Wan patch lattice")
                sources["diffusers_version"] = diffusers.__version__
            model.eval().requires_grad_(False)
            if role != "teacher":
                sources[role + "_lora_targets"] = install_lora(model, task)
            models[role] = model
    return nn.ModuleDict(models), sources


class ModelAdapter:
    """Separate model interface binding from DMD protocol and execution order."""
    def __init__(self, task: NativeTask) -> None:
        self.task = task
        self.models, self.sources = load_models(task)
        self.device = torch.device(task.device)
        self.dtype = getattr(torch, task.dtype)
        if self.device.type == "cuda":
            ensure(torch.cuda.is_available(), "CUDA explicitly requested but unavailable")
        keep_fp32 = set()
        for model in self.models.values():
            patterns = tuple(getattr(model, "_keep_in_fp32_modules", ()) or ())
            keep_fp32.update(id(p) for name, p in model.named_parameters()
                             if any(pattern in name.split(".") for pattern in patterns))
        if self.device.type == "cuda":
            frozen = sum(p.numel() * (4 if self.dtype == torch.float32 or id(p) in keep_fp32 else 2)
                         for p in self.models.parameters() if not p.requires_grad)
            trainable = sum(p.numel() * 4 for p in self.models.parameters() if p.requires_grad)
            free, _ = torch.cuda.mem_get_info(self.device)
            ensure(frozen + 5 * trainable + task.gpu_reserve_bytes <= free,
                   "native resident parameter/optimizer lower bound exceeds current GPU capacity")
        # Move base weights at declared precision; LoRA master and compute remain FP32.
        def move(value):
            dtype = torch.float32 if value.requires_grad or id(value) in keep_fp32 else self.dtype
            return value.to(device=self.device, dtype=dtype) if value.is_floating_point() else value.to(self.device)
        # Module.to(dtype=BF16) followed by float() would already round LoRA masters.
        # Apply the mixed-precision rule once, before optimizers or graphs exist.
        self.models._apply(move)
        self.initial_model_sha256 = tensors_digest(self.models.state_dict().items())
        self.base_sha256 = self.base_digest()
        self.checkpoint_enabled = False
        self.epilogue = "reference"
        self.calls = {k: 0 for k in self.models}
        ids = [{id(p) for p in self.models[r].parameters() if p.requires_grad} for r in ("generator", "critic")]
        ensure(ids[0] and ids[1] and not ids[0] & ids[1], "adapter/optimizer trainable storage must be disjoint")

    def trainables(self, role: str):
        return {n: p for n, p in self.models[role].named_parameters() if p.requires_grad}

    def base_digest(self) -> str:
        return tensors_digest((n, p) for n, p in self.models.named_parameters() if not p.requires_grad)

    def set_implementation(self, *, checkpoint_enabled: bool, epilogue: str) -> None:
        ensure(type(checkpoint_enabled) is bool and epilogue in {"reference", "wan-fused"}, "invalid implementation")
        if epilogue == "wan-fused":
            ensure(self.device.type == "cuda" and self.dtype == torch.bfloat16, "fused epilogue requires CUDA/BF16")
            try:
                import triton  # noqa: F401
            except ImportError as exc:
                raise ValueError("fused epilogue requires Triton; reference remains available") from exc
        self.checkpoint_enabled, self.epilogue = checkpoint_enabled, epilogue
        for model in self.models.values():
            for mod in model.modules():
                if isinstance(mod, LoRALinear):
                    mod.use_fused = epilogue == "wan-fused"

    def predict(self, role: str, x: torch.Tensor, sigma: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        self.calls[role] += 1
        model = self.models[role]
        def call(value, time, cond):
            if self.task.backend == "torch-reference":
                return model(value, time, cond)
            return model(hidden_states=value, timestep=time * 1000,
                         encoder_hidden_states=cond, return_dict=False)[0]
        if self.checkpoint_enabled and torch.is_grad_enabled() and role != "teacher":
            out = checkpoint(call, x, sigma, condition, use_reentrant=False, preserve_rng_state=True)
        else:
            out = call(x, sigma, condition)
        ensure(out.shape == x.shape and out.dtype == x.dtype, "native model output violates latent contract")
        return out

    def static_bytes(self) -> int:
        seen = {}
        for t in [*self.models.parameters(), *self.models.buffers()]:
            key = (str(t.device), t.untyped_storage().data_ptr())
            seen[key] = t.untyped_storage().nbytes()
        return sum(seen.values())
