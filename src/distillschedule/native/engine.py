"""Real forward/backward handlers for parameterized post-G flow-DMD cycles.

One process/device, separate role models. No FSDP/DP/pool capability is inferred
from a world-size field. The sequential reference loop is independent of the
planner, and full trainable/Adam/EMA states are compared outside timing.
"""
from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
import time
from typing import Any, Callable
import torch
from torch.nn import functional as F

from distillschedule.core.enums import OperationKind as Kind
from distillschedule.runtime.parameterized_cycle import build_parameterized_cycle_graph, parameterized_window_order
from distillschedule.runtime.role_runner import StaticRoleRunner
from distillschedule.runtime.version_sync import VersionLedger
from distillschedule.workloads.best_known.registry import file_sha256
from runtime.dmd_boundary import detached_dmd_surrogate_loss, critic_denoising_loss
from .models import ModelAdapter, tensor_hash
from .task import NativeTask, ensure


class Conditions:
    def __init__(self, task: NativeTask):
        self.task = task
        self.tensors = None
        self.identity = "SYNTHETIC_CONDITIONS_EXPLICIT_FIXTURE"
        if not task.fixture:
            path = Path(task.conditions_path)
            ensure(path.is_file() and path.suffix == ".safetensors", "conditions must be local safetensors")
            from safetensors.torch import load_file
            self.identity = file_sha256(path)
            self.tensors = load_file(str(path), device="cpu")
            required = {"prompt_embeds"} | ({"negative_prompt_embeds"} if task.teacher_guidance else set())
            ensure(required <= set(self.tensors), "missing conditional/unconditional embeddings")
            ensure(set(self.tensors) <= {"prompt_embeds", "negative_prompt_embeds"}, "unknown condition: history not supported")
            shape = self.tensors["prompt_embeds"].shape
            ensure(len(shape) == 3 and tuple(shape[1:]) == (task.text_tokens, task.text_dim), "condition shape mismatch")
            ensure(all(tuple(t.shape) == tuple(shape) and t.is_floating_point() and torch.isfinite(t).all()
                       for t in self.tensors.values()), "invalid condition tensor")

    def get(self, cycle: int, slot: int, micro: int, generator: torch.Generator, device, dtype):
        t = self.task
        if self.tensors is None:
            cond = torch.randn((t.microbatch, t.text_tokens, t.text_dim), generator=generator, dtype=torch.float32)
            neg = torch.zeros_like(cond)
        else:
            index = (cycle * (t.fake_updates + 1) + slot) * t.batch + micro * t.microbatch
            ensure(index + t.microbatch <= len(self.tensors["prompt_embeds"]), "condition cohort exhausted; no silent recycling")
            cond = self.tensors["prompt_embeds"][index:index + t.microbatch]
            neg = self.tensors.get("negative_prompt_embeds")
            neg = neg[index:index + t.microbatch] if neg is not None else torch.zeros_like(cond)
        return cond.to(device=device, dtype=dtype), neg.to(device=device, dtype=dtype)


def _cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(v) for v in value)
    return copy.deepcopy(value)


class NativeEngine:
    def __init__(self, task: NativeTask) -> None:
        ensure(int(os.environ.get("WORLD_SIZE", "1")) == 1 and
               not (torch.distributed.is_available() and torch.distributed.is_initialized()
                    and torch.distributed.get_world_size() != 1),
               "parameterized native backend currently supports one process/device, not implicit DDP/FSDP")
        self.task = task
        self.adapter = ModelAdapter(task)
        self.conditions = Conditions(task)
        self.device, self.dtype = self.adapter.device, self.adapter.dtype
        self.optimizers = {r: torch.optim.AdamW(list(self.adapter.trainables(r).values()),
                            lr=task.learning_rate, betas=task.adam_betas, eps=task.adam_epsilon,
                            weight_decay=task.weight_decay, foreach=False)
                           for r in ("generator", "critic")}
        # Materialize Adam moments outside timing, preserving the actual step=0.
        for opt in self.optimizers.values():
            for group in opt.param_groups:
                for p in group["params"]:
                    opt.state[p] = {"step": torch.tensor(0.0), "exp_avg": torch.zeros_like(p),
                                    "exp_avg_sq": torch.zeros_like(p)}
        self.ema = {n: p.detach().clone() for n, p in self.adapter.trainables("generator").items()}
        self.versions = {"generator": 0, "critic": 0, "ema": 0}
        self.next_cycle = 0
        self.losses: dict[str, torch.Tensor] = {}
        self.last_gradients: dict[str, dict[str, torch.Tensor]] = {}
        self.cycles = []

    def fence(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def snapshot(self) -> dict[str, Any]:
        return {"params": {r: {n: p.detach().cpu().clone() for n, p in self.adapter.trainables(r).items()}
                           for r in self.optimizers},
                "optimizers": _cpu_tree({r: o.state_dict() for r, o in self.optimizers.items()}),
                "ema": {n: p.detach().cpu().clone() for n, p in self.ema.items()},
                "versions": dict(self.versions), "next_cycle": self.next_cycle}

    def restore(self, state: dict[str, Any]) -> None:
        for r, params in state["params"].items():
            for n, value in params.items():
                self.adapter.trainables(r)[n].data.copy_(value)
            self.optimizers[r].load_state_dict(copy.deepcopy(state["optimizers"][r]))
            self.optimizers[r].zero_grad(set_to_none=True)
        for n, value in state["ema"].items():
            self.ema[n].copy_(value)
        self.versions, self.next_cycle = dict(state["versions"]), state["next_cycle"]
        self.losses.clear(); self.last_gradients.clear(); self.cycles.clear()
        self.adapter.calls = {k: 0 for k in self.adapter.calls}

    def signature_tensors(self) -> dict[str, torch.Tensor]:
        out = {f"{r}/parameter/{n}": p.detach().cpu().clone()
               for r in self.optimizers for n, p in self.adapter.trainables(r).items()}
        for r, opt in self.optimizers.items():
            for name, p in self.adapter.trainables(r).items():
                for key, value in opt.state[p].items():
                    if isinstance(value, torch.Tensor):
                        out[f"{r}/adam/{name}/{key}"] = value.detach().cpu().clone()
        out.update({"ema/" + n: p.detach().cpu().clone() for n, p in self.ema.items()})
        return out

    def static_bytes(self) -> int:
        return (self.adapter.static_bytes() + sum(v.numel() * v.element_size() for v in self.ema.values())
                + sum(v.numel() * v.element_size() for o in self.optimizers.values()
                      for state in o.state.values() for v in state.values()
                      if isinstance(v, torch.Tensor) and v.device == self.device))

    def inputs(self, cycle: int, slot: int, micro: int):
        # Binding is independent of physical dispatch order and global Torch RNG.
        payload = f"{self.task.seed}:{cycle}:{slot}:{micro}".encode()
        seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63)
        g = torch.Generator(device="cpu").manual_seed(seed)
        condition, negative = self.conditions.get(cycle, slot, micro, g, self.device, self.dtype)
        initial = torch.randn(self.task.latent_shape, generator=g, dtype=torch.float32).to(self.device, self.dtype)
        score_noise = torch.randn(self.task.latent_shape, generator=g, dtype=torch.float32).to(self.device, self.dtype)
        score_sigma = (self.task.score_sigma_min + torch.rand(self.task.microbatch, generator=g) *
                       (self.task.score_sigma_max - self.task.score_sigma_min)).to(self.device)
        return initial, condition, negative, score_noise, score_sigma

    def rollout(self, noise, condition, *, grad_last: bool):
        steps, shift = self.task.denoising_steps, self.task.sigma_shift
        schedule = [shift * (1-i/steps) / (1 + (shift-1)*(1-i/steps)) for i in range(steps + 1)]
        x = noise
        for i in range(steps):
            sigma = torch.full((self.task.microbatch,), schedule[i], device=self.device, dtype=torch.float32)
            with torch.set_grad_enabled(grad_last and i == steps-1):
                v = self.adapter.predict("generator", x, sigma, condition)
                x = (x.float() + (schedule[i+1] - schedule[i]) * v.float()).to(self.dtype)
            if i < steps-1:
                x = x.detach()
        return x

    def _commit(self, role: str) -> None:
        params = self.adapter.trainables(role)
        ensure(all(p.grad is not None for p in params.values()), "a trainable parameter missed backward")
        torch.nn.utils.clip_grad_norm_(list(params.values()), self.task.max_grad_norm, error_if_nonfinite=True)
        self.optimizers[role].step()
        self.optimizers[role].zero_grad(set_to_none=True)
        self.versions[role] += 1

    def generator_update(self, cycle: int) -> None:
        self.optimizers["generator"].zero_grad(set_to_none=True)
        metrics = []
        for micro in range(self.task.accumulation):
            noise, cond, neg, score_noise, sigma = self.inputs(cycle, 0, micro)
            generated = self.rollout(noise, cond, grad_last=True)
            with torch.no_grad():
                s = sigma[:, None, None, None, None]
                noisy = ((1-s) * generated.detach().float() + s * score_noise.float()).to(self.dtype)
                fake = self.adapter.predict("critic", noisy, sigma, cond)
                real = self.adapter.predict("teacher", noisy, sigma, cond)
                if self.task.teacher_guidance:
                    uncond = self.adapter.predict("teacher", noisy, sigma, neg)
                    real = real + self.task.teacher_guidance * (real - uncond)
                fake_x0, real_x0 = noisy.float() - s * fake.float(), noisy.float() - s * real.float()
                norm = (generated.detach().float() - real_x0).abs().mean((1,2,3,4), keepdim=True)
                gradient = (fake_x0 - real_x0) / norm.clamp_min(self.task.normalization_floor)
            loss = detached_dmd_surrogate_loss(generated, gradient)
            (loss / self.task.accumulation).backward()
            metrics.append(loss.detach())
        self._commit("generator")
        self.losses["g"] = torch.stack(metrics).mean()

    def produce(self, cycle: int, index: int) -> tuple[torch.Tensor, ...]:
        with torch.no_grad():
            return tuple(self.rollout(*self.inputs(cycle, index+1, micro)[:2], grad_last=False).detach()
                         for micro in range(self.task.accumulation))

    def critic_update(self, cycle: int, index: int, generated) -> None:
        ensure(len(generated) == self.task.accumulation, "rollout accumulation changed")
        self.optimizers["critic"].zero_grad(set_to_none=True)
        metrics = []
        for micro, x0 in enumerate(generated):
            ensure(not x0.requires_grad and tuple(x0.shape) == self.task.latent_shape, "invalid retained rollout")
            _, cond, _, noise, sigma = self.inputs(cycle, index+1, micro)
            s = sigma[:, None, None, None, None]
            noisy = ((1-s) * x0.float() + s * noise.float()).to(self.dtype)
            prediction = self.adapter.predict("critic", noisy, sigma, cond)
            loss = critic_denoising_loss(prediction, noise, x0)
            (loss / self.task.accumulation).backward()
            metrics.append(loss.detach())
        self._commit("critic")
        self.losses[f"c{index}"] = torch.stack(metrics).mean()

    def update_ema(self) -> None:
        with torch.no_grad():
            for n, p in self.adapter.trainables("generator").items():
                self.ema[n].lerp_(p, 1-self.task.ema_decay)
        self.versions["ema"] += 1

    def graph(self):
        return build_parameterized_cycle_graph(source_steps=tuple(range(self.task.fake_updates)),
                    initial_versions=self.versions, ema_after=self.task.ema_after)

    def run_reference_cycle(self):
        """Independent serial reference, never delegated to the planner's ordering."""
        cycle = self.next_cycle
        self.losses = {}
        self.generator_update(cycle)
        for i in range(self.task.fake_updates):
            outputs = self.produce(cycle, i)
            self.critic_update(cycle, i, outputs)
            del outputs
            if i in self.task.ema_after:
                self.update_ema()
        self.next_cycle += 1
        self.fence()
        return self._loss_values()

    def _loss_values(self):
        values = {k: float(v.cpu()) for k, v in self.losses.items()}
        import math
        ensure(all(math.isfinite(x) for x in values.values()), "nonfinite native loss")
        return values

    def run_cycle(self, order: tuple[str, ...] | None = None, *, calibration: bool = False):
        graph, cycle = self.graph(), self.next_cycle
        if order is None:
            order = parameterized_window_order(graph, max_live_rollouts=1)
        self.losses = {}
        start_calls = dict(self.adapter.calls)
        observations = []
        initial = dict(self.versions)
        def wrapped(ctx):
            key = ctx.operation.operation_id
            saved_bytes = 0
            def pack(value):
                nonlocal saved_bytes
                saved_bytes += value.numel() * value.element_size()
                return value
            if calibration:
                self.fence()
                if self.device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(self.device)
                    entry_allocated = torch.cuda.memory_allocated(self.device)
            started = time.perf_counter_ns()
            def execute():
                if key == "g":
                    return self.generator_update(cycle)
                index = int(key[1:])
                if key[0] == "r":
                    return self.produce(cycle, index)
                if key[0] == "c":
                    return self.critic_update(cycle, index, ctx.values[f"r{index}"])
                return self.update_ema()
            if calibration and self.device.type == "cpu":
                with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
                    result = execute()
            else:
                result = execute()
            if calibration:
                self.fence()
                duration = time.perf_counter_ns() - started
                output_bytes = sum(x.numel()*x.element_size() for x in result) if key.startswith("r") else 0
                # CPU accounting is a conservative tensor estimate, NOT RSS/HBM observation.
                workspace = (max(0, torch.cuda.max_memory_allocated(self.device)-entry_allocated)
                             if self.device.type == "cuda" else saved_bytes + 4*self.task.rollout_bytes)
                observations.append({"action": key, "duration_ns": max(1,duration), "workspace_bytes": workspace,
                                     "output_bytes": output_bytes})
            expected = dict(ctx.versions.snapshot().values)
            expected.update(ctx.operation.produced_versions)
            ensure(self.versions == expected, "actual optimizer/EMA commits disagree with selected operation")
            return result
        kinds = (Kind.GENERATOR_UPDATE, Kind.CRITIC_ROLLOUT, Kind.CRITIC_UPDATE, Kind.EMA_UPDATE)
        runner = StaticRoleRunner("native_runtime", {k.value: wrapped for k in kinds}, versions=VersionLedger(initial))
        prepared = runner.prepare(graph, order=tuple(order))
        # Admission before any state-changing model call; input-byte cap is exact for this protocol.
        live, peak = 0, 0
        for key in order:
            if key.startswith("r"): live += self.task.rollout_bytes
            elif key.startswith("c"): live -= self.task.rollout_bytes
            peak = max(peak, live)
        ensure(peak <= self.task.max_retained_bytes, "selected order exceeds retained-output budget")
        self.fence(); started = time.perf_counter_ns()
        prepared.run()
        self.fence(); duration = time.perf_counter_ns()-started
        ensure(self.versions == graph.metadata["final_versions"], "native final versions mismatch")
        delta = {k: self.adapter.calls[k]-start_calls[k] for k in start_calls}
        expected_calls = {"generator": (self.task.fake_updates+1)*self.task.denoising_steps*self.task.accumulation,
                          "critic": (self.task.fake_updates+1)*self.task.accumulation,
                          "teacher": (1+int(self.task.teacher_guidance != 0))*self.task.accumulation}
        ensure(delta == expected_calls, "native forward census changed")
        self.next_cycle += 1
        result = {"status": "PASS_NATIVE_TENSOR_EXECUTION", "cycle": cycle, "order": list(order),
                  "versions": dict(self.versions), "calls": delta, "losses": self._loss_values(),
                  "duration_ns": duration, "calibration": calibration,
                  "observations": observations, "retained_peak_bytes": peak,
                  "execution": prepared.receipt(), "device": str(self.device), "fixture": self.task.fixture,
                  "memory_scope": "CUDA_ALLOCATOR" if self.device.type == "cuda" else "TENSOR_ACCOUNTING_NOT_RSS"}
        self.cycles.append(result)
        return result


def compare_states(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor], *, rtol=1e-5, atol=1e-7):
    ensure(set(left) == set(right), "state inventories differ")
    failures = []
    maximum = 0.0
    for key in left:
        a, b = left[key], right[key]
        ensure(a.shape == b.shape and a.dtype == b.dtype, "state tensor contract changed")
        if not torch.isfinite(a).all() or not torch.isfinite(b).all() or not torch.allclose(a,b,rtol=rtol,atol=atol):
            failures.append(key)
        maximum = max(maximum, float((a-b).abs().max()) if a.numel() else 0.0)
    return {"status": "PASS" if not failures else "FAIL", "rtol": rtol, "atol": atol,
            "max_absolute_error": maximum, "compared_tensors": len(left), "failures": failures,
            "scope": "ALL_TRAINABLE_ADAM_EMA_TENSORS_NOT_LONG_HORIZON_QUALITY"}
