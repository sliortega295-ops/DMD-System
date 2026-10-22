"""ZeRO-3 backend for the shared-backbone, alternating H3 DMD2 trainer."""

import torch


class AlternatingAdamW(torch.optim.AdamW):
    """Skip inactive groups completely, including Adam clocks and weight decay.

    ZeRO supplies zero-filled flat gradients for unused groups. A zero gradient
    alone does NOT freeze AdamW after a group has accumulated momentum.
    Group metadata survives ZeRO's replacement of parameters by FP32 shards.
    """

    active_role = None
    train_heads = True

    @torch.no_grad()
    def step(self, closure=None):
        skipped = []
        for group in self.param_groups:
            active = self.active_role is None or group["role"] == self.active_role
            if group.get("head", False) and not self.train_heads:
                active = False
            if not active:
                for param in group["params"]:
                    skipped.append((param, param.grad))
                    param.grad = None
        try:
            return super().step(closure)
        finally:
            for param, grad in skipped:
                param.grad = grad


class DMD2Module(torch.nn.Module):
    def __init__(self, adapter, discriminator):
        super().__init__()
        self.adapter = adapter
        self.discriminator = discriminator

    def forward(self, latents, sigmas, inputs, checkpoint=False):
        return self.adapter(latents, sigmas, inputs, checkpoint)


def configure(accelerator, config):
    """Resolve the launcher's DeepSpeedPlugin before constructing any weights."""
    ds_config = accelerator.state.deepspeed_plugin.deepspeed_config
    zero = ds_config.get("zero_optimization", {})
    if (zero.get("stage") != 3 or ds_config.get("gradient_accumulation_steps", "auto") not in (1, "auto")
            or accelerator.gradient_accumulation_steps != 1):
        raise ValueError("DMD2 requires ZeRO stage 3 and gradient_accumulation_steps=1")
    if not ds_config.get("bf16", {}).get("enabled"):
        raise ValueError("DMD2 ZeRO-3 requires BF16")
    if any(zero.get(key, {}).get("device", "none") != "none"
           for key in ("offload_optimizer", "offload_param")):
        raise ValueError("DMD2 ZeRO-3 does not support offload")
    if "optimizer" in ds_config or "scheduler" in ds_config:
        raise ValueError("DMD2 supplies its own alternating optimizer and fixed learning rates")
    # As in the SFT entry, no dataloader is passed to prepare(): it would shard
    # node-local caches a second time. Supply the batch semantics explicitly.
    ds_config.update(train_micro_batch_size_per_gpu=1, gradient_accumulation_steps=1,
                     train_batch_size=accelerator.num_processes, gradient_clipping=config.max_grad_norm)
    # Bounded defaults for H3; explicit DeepSpeed plugin/file values take precedence.
    for key, value in {"overlap_comm": False, "contiguous_gradients": True,
                       "reduce_bucket_size": 50000000, "stage3_prefetch_bucket_size": 50000000,
                       "stage3_param_persistence_threshold": 0}.items():
        zero.setdefault(key, value)
    return ds_config


def initialize(accelerator, adapter, discriminator, config):

    groups = [
        {"params": adapter.role_parameters("student"), "role": "student",
         "lr": config.student_learning_rate},
        {"params": adapter.role_parameters("fake_score"), "role": "critic",
         "lr": config.fake_score_learning_rate},
        {"params": list(discriminator.parameters()), "role": "critic", "head": True,
         "lr": config.discriminator_learning_rate},
    ]
    # Both branches must be registered before ZeRO builds its master shards and
    # gradient hooks. select() will freeze the inactive branch during each phase.
    for group in groups:
        for param in group["params"]:
            param.requires_grad_(True)
    optimizer = AlternatingAdamW(groups, weight_decay=config.weight_decay)
    engine, _ = accelerator.prepare(DMD2Module(adapter, discriminator), optimizer)
    # A callable avoids registering the engine as an adapter child (a cycle).
    adapter._distributed_predict = lambda *args: engine(*args)
    adapter.select("teacher")
    return engine, optimizer


def step(accelerator, engine, optimizer, discriminator, role, loss, config):
    finite = torch.isfinite(loss.detach()).int()
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(finite, op=torch.distributed.ReduceOp.MIN)
    if not finite.item():
        raise FloatingPointError(f"Nonfinite {role} loss")
    optimizer.active_role = role
    optimizer.train_heads = any(p.requires_grad for p in discriminator.parameters())
    engine.optimizer.clip_grad = config.max_grad_norm if role == "student" else config.critic_max_grad_norm
    # Accelerate's DeepSpeed backward wrapper performs engine.step() as well.
    # Calling it again here would advance Adam twice per critic/student update.
    accelerator.backward(loss)


def export_states(adapter, discriminator):
    """All ranks gather trainable weights only; only rank 0 stages CPU copies."""
    import deepspeed

    main = torch.distributed.get_rank() == 0
    exports = {"student": {}, "fake_score": {}, "discriminator": {}}
    for name, layer in adapter.layers:
        for role in ("student", "fake_score"):
            for letter in ("A", "B"):
                param = layer.adapters[role][letter].weight
                with deepspeed.zero.GatheredParameters([param]):
                    if main:
                        value = param.detach().float().cpu().clone()
                        if letter == "B":
                            value.mul_(layer.scale)
                        exports[role][name + f".lora_{letter}.weight"] = value.contiguous()
    for name, param in discriminator.named_parameters():
        with deepspeed.zero.GatheredParameters([param]):
            if main:
                exports["discriminator"][name] = param.detach().float().cpu().clone().contiguous()
    return exports
