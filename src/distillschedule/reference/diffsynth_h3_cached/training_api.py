"""Original selected training/checkpoint helpers, not the CUDA train entry.

WARNING: load_checkpoint is the exact legacy trusted-pickle API. Call it only
on checkpoints you created and trust. It is not a safe public untrusted loader.
CacheDataset uses the original weights_only=True tensor-cache parser.
"""
import hashlib
import json
import random
import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import save_file
from .h3_dmd2 import (DMD2Config as DMD2Config, DMD2Trainer as DMD2Trainer,
    H3Adapter as H3Adapter, VelocityDiscriminator as VelocityDiscriminator,
    make_schedule as make_schedule, prepare_batch)
from .cache import NodeLocalSampler as NodeLocalSampler
from .support import CachedTensorMover as CachedTensorMover
from .weights import checkpoint_files, load_full_weights, model_config
from .models.minimax_h3_dit import MiniMaxH3DiT


def rank():
    return dist.get_rank() if dist.is_initialized() else 0


def world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def barrier():
    if dist.is_initialized():
        dist.barrier()


def gather_objects(value):
    if not dist.is_initialized():
        return [value]
    result = [None] * world_size()
    dist.all_gather_object(result, value)
    return result


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_node_caches(info, local_world):
    if local_world <= 0 or len(info) % local_world or any(x["local_world"] != local_world for x in info):
        raise ValueError("All nodes must use the same local process count: " +
                         repr([x["local_world"] for x in info]))
    for start in range(0, len(info), local_world):
        node = info[start:start + local_world]
        ranks = [x["local_rank"] for x in node]
        if ranks != list(range(local_world)):
            raise ValueError(f"Invalid local rank mapping for global ranks {start}:{start + local_world}: "
                             f"got {ranks}, expected {list(range(local_world))}")
        if any(x["data"] != node[0]["data"] for x in node):
            details = [{"global_rank": start + i, "local_rank": x["local_rank"], **x["data"]}
                       for i, x in enumerate(node)]
            raise ValueError("Node-local cache manifests differ (root/count/file size/mtime). "
                             "Finish copying or generating caches before launching training; "
                             "all local ranks must read the same stable directory. Per-rank manifests: " +
                             json.dumps(details, ensure_ascii=False))


def load_teacher(path, device, dtype=torch.bfloat16, ds_config=None):
    files, directory = checkpoint_files(path)
    config = model_config(directory)
    # Materialize exactly once, directly in the chosen dtype/device. No random
    # full-size CPU model and no model-directory autodetection or adapter fusion.
    if ds_config is not None:
        import deepspeed
        # Same partition-at-construction strategy as the full SFT entry.
        with deepspeed.zero.Init(config_dict_or_path=ds_config, remote_device=device, dtype=dtype):
            dit = MiniMaxH3DiT(**config)
    else:
        with torch.device("meta"):
            dit = MiniMaxH3DiT(**config)
        dit.to(dtype=dtype).to_empty(device=device)
    load_full_weights(dit, files)
    identity = [{"path": str(p.resolve()), "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in files]
    return dit, config, identity


class CacheDataset:
    def __init__(self, root, cfg_scale):
        self.root = root.expanduser().resolve()
        self.files = sorted(self.root.rglob("*.pth")) if self.root.is_dir() else []
        if not self.files:
            raise ValueError(f"No .pth caches in {self.root}")
        self.cfg_scale = cfg_scale

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        try:
            result = prepare_batch(torch.load(path, map_location="cpu", weights_only=True), self.cfg_scale)
            for condition in result:
                if condition is not None:
                    for name, value in condition.items():
                        if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                            raise ValueError(f"Nonfinite {name}")
            return result
        except Exception as exc:
            raise ValueError(f"Invalid H3 DMD2 cache {path}: {exc}") from exc

    def identity(self):
        # Metadata fingerprint detects normal cache edits without reading TB of
        # data; it is deliberately not advertised as a content hash.
        digest = hashlib.sha256()
        for path in self.files:
            stat = path.stat()
            digest.update(f"{path.relative_to(self.root)}:{stat.st_size}:{stat.st_mtime_ns}\n".encode())
        return {"root": str(self.root), "count": len(self), "manifest_sha256": digest.hexdigest()}


def make_optimizers(adapter, discriminator, config):
    student = adapter.role_parameters("student")
    fake = adapter.role_parameters("fake_score")
    heads = list(discriminator.parameters())
    optimizers = {
        "student": torch.optim.AdamW(student, lr=config.student_learning_rate, weight_decay=config.weight_decay),
        "critic": torch.optim.AdamW([
            {"params": fake, "lr": config.fake_score_learning_rate},
            {"params": heads, "lr": config.discriminator_learning_rate},
        ], weight_decay=config.weight_decay),
    }
    return optimizers, {"student": student, "critic": fake + heads}


def optimizer_step(role, loss, optimizers, parameters, config, accelerator=None):
    finite = torch.isfinite(loss.detach()).int()
    if dist.is_initialized():
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    if not finite.item():
        raise FloatingPointError(f"Nonfinite {role} loss")
    if accelerator is None:
        loss.backward()
    else:
        accelerator.backward(loss)
    params = parameters[role]
    # Explicit averaging of the small trainable tensors avoids DDP reducer state
    # across alternating adapters. The frozen backbone is replicated per rank.
    for param in params:
        if not param.requires_grad:
            continue
        if param.grad is None:
            param.grad = torch.zeros_like(param)
        if dist.is_initialized():
            dist.all_reduce(param.grad)
            param.grad.div_(world_size())
    norm = torch.nn.utils.clip_grad_norm_(params, config.max_grad_norm if role == "student" else config.critic_max_grad_norm)
    if not torch.isfinite(norm):
        raise FloatingPointError(f"Nonfinite {role} gradient norm")
    optimizers[role].step()
    optimizers[role].zero_grad(set_to_none=True)


def adapter_state(adapter):
    return {name: param.detach().cpu().clone() for name, param in adapter.named_parameters() if ".adapters." in name}


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"])


def save_checkpoint(destination, trainer, optimizers, metadata, cursor):
    """All ranks call at a complete N:1 cycle; publish only after all state exists."""
    temporary = destination.with_name("." + destination.name + ".tmp")
    exists = destination.exists() or temporary.exists()
    if any(gather_objects(exists)):
        raise FileExistsError(f"Checkpoint or incomplete checkpoint already exists: {destination}")
    if rank() == 0:
        temporary.mkdir(parents=True)
    barrier()
    states = gather_objects({"rng": rng_state(), "cursor": cursor})
    engine = getattr(trainer, "zero3_engine", None)
    if engine is not None:
        from .dmd2_zero3 import export_states
        state = {"generator_step": trainer.generator_step, "critic_step": trainer.critic_step,
                 "ranks": states, "metadata": metadata}
        # requires_grad changes during alternating phases. Save both LoRAs and
        # heads, even when currently frozen, while excluding the immutable DiT.
        trainable = trainer.adapter.role_parameters("student") + trainer.adapter.role_parameters("fake_score")
        trainable += list(trainer.discriminator.parameters())
        flags = [p.requires_grad for p in trainable]
        try:
            for p in trainable:
                p.requires_grad_(True)
            engine.save_checkpoint(str(temporary / "zero3"), tag="state", client_state=state,
                                   exclude_frozen_parameters=True)
        finally:
            for p, flag in zip(trainable, flags):
                p.requires_grad_(flag)
        exports = export_states(trainer.adapter, trainer.discriminator)
        if rank() == 0:
            torch.save(state, temporary / "training_state.pt")
            for role, weights in exports.items():
                save_file(weights, str(temporary / f"{role}.safetensors"))
            (temporary / "metadata.json").write_text(json.dumps({**metadata,
                "generator_step": trainer.generator_step, "critic_step": trainer.critic_step}, indent=2) + "\n")
            temporary.rename(destination)
        barrier()
        return
    if rank() == 0:
        state = {
            "adapters": adapter_state(trainer.adapter), "discriminator": trainer.discriminator.state_dict(),
            "optimizers": {name: opt.state_dict() for name, opt in optimizers.items()},
            "generator_step": trainer.generator_step, "critic_step": trainer.critic_step,
            "ranks": states, "metadata": metadata,
        }
        torch.save(state, temporary / "training_state.pt")
        for role in ("student", "fake_score"):
            save_file(trainer.adapter.export_lora(role), str(temporary / f"{role}.safetensors"),
                      metadata={"role": role, "step": str(trainer.generator_step), "alpha_baked_into_B": "true"})
        save_file({k: v.detach().cpu().contiguous() for k, v in trainer.discriminator.state_dict().items()},
                  str(temporary / "discriminator.safetensors"))
        info = {**metadata, "generator_step": trainer.generator_step, "critic_step": trainer.critic_step}
        (temporary / "metadata.json").write_text(json.dumps(info, indent=2) + "\n")
        temporary.rename(destination)
    barrier()


def load_checkpoint(path, trainer, optimizers, metadata):
    if not (path / "metadata.json").is_file():
        raise ValueError("Resume requires a complete DMD2 step directory")
    # Full optimizer/RNG state uses pickle: resume only checkpoints you trust.
    state = torch.load(path / "training_state.pt", map_location="cpu", weights_only=False)
    for key in ("format_version", "dmd", "teacher", "architecture", "world_size", "local_world_size", "data", "seed", "attention", "torch_version"):
        if state["metadata"][key] != metadata[key]:
            raise ValueError(f"Resume metadata mismatch: {key}")
    if state["metadata"].get("distributed_backend", "replicated") != metadata.get("distributed_backend", "replicated"):
        raise ValueError("Resume metadata mismatch: distributed_backend")
    engine = getattr(trainer, "zero3_engine", None)
    if engine is not None:
        for key in ("deepspeed_version", "deepspeed_config"):
            if state["metadata"].get(key) != metadata.get(key):
                raise ValueError(f"Resume metadata mismatch: {key}")
        loaded, _ = engine.load_checkpoint(str(path / "zero3"), tag="state", load_module_strict=False)
        if loaded is None:
            raise ValueError("Missing ZeRO-3 training checkpoint")
        trainer.generator_step, trainer.critic_step = state["generator_step"], state["critic_step"]
        if trainer.critic_step != trainer.generator_step * trainer.config.critic_updates:
            raise ValueError("Resume critic/student counters are inconsistent")
        local = state["ranks"][rank()]
        restore_rng(local["rng"])
        return local["cursor"]
    expected = {name: param for name, param in trainer.adapter.named_parameters() if ".adapters." in name}
    if expected.keys() != state["adapters"].keys():
        raise ValueError("Resume adapter keys do not match")
    with torch.no_grad():
        for name, param in expected.items():
            param.copy_(state["adapters"][name])
    trainer.discriminator.load_state_dict(state["discriminator"], strict=True)
    for name, opt in optimizers.items():
        opt.load_state_dict(state["optimizers"][name])
    trainer.generator_step, trainer.critic_step = state["generator_step"], state["critic_step"]
    if trainer.critic_step != trainer.generator_step * trainer.config.critic_updates:
        raise ValueError("Resume critic/student counters are inconsistent")
    local = state["ranks"][rank()]
    restore_rng(local["rng"])
    return local["cursor"]
