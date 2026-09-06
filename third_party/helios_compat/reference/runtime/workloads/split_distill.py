"""Two-role execution primitives for a Generator/Score split schedule."""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterator, Mapping, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .mini_distill import (
    GeneratorEMA,
    MiniBatch,
    MiniDistillConfig,
    MiniGenerator,
    ReplayRecord,
    SharedScore,
)


def _gradient_snapshot(module: nn.Module) -> Dict[str, Tensor]:
    return {
        name: parameter.grad.detach().cpu().clone()
        for name, parameter in module.named_parameters()
        if parameter.requires_grad and parameter.grad is not None
    }


def _parameter_snapshot(module: nn.Module) -> Dict[str, Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }


def tensor_dict_checksum(values: Mapping[str, Tensor]) -> float:
    return sum(float(value.detach().double().sum().cpu()) for value in values.values())


def tensor_dict_norm(values: Mapping[str, Tensor]) -> float:
    return math.sqrt(
        sum(float(value.detach().float().pow(2).sum().cpu()) for value in values.values())
    )


@dataclass
class GeneratorRoleResult:
    loss: float
    replay_max_abs_error: float
    fake_checksum: float
    gradients: Dict[str, Tensor]
    parameters: Dict[str, Tensor]
    ema_parameters: Dict[str, Tensor]

    def summary(self) -> dict:
        return {
            "loss": self.loss,
            "replay_max_abs_error": self.replay_max_abs_error,
            "fake_checksum": self.fake_checksum,
            "gradient_norm": tensor_dict_norm(self.gradients),
            "parameter_checksum": tensor_dict_checksum(self.parameters),
            "ema_checksum": tensor_dict_checksum(self.ema_parameters),
        }


@dataclass
class ScoreGradientResult:
    input_gradient: Tensor
    generator_loss: float
    dmd_surrogate: float
    gan_loss: float
    dmd_gradient_checksum: float


@dataclass(frozen=True)
class GeneratorMicrobatchResult:
    """Generator replay result before the shared optimizer-version barrier."""

    loss: float
    replay_max_abs_error: float
    fake_checksum: float


@dataclass
class CriticRoleResult:
    loss: float
    gradients: Dict[str, Tensor]
    parameters: Dict[str, Tensor]

    def summary(self) -> dict:
        return {
            "loss": self.loss,
            "gradient_norm": tensor_dict_norm(self.gradients),
            "parameter_checksum": tensor_dict_checksum(self.parameters),
        }


@dataclass(frozen=True)
class CriticMicrobatchResult:
    """Critic loss whose scaled backward has run, but whose step is deferred."""

    loss: float


class SplitGeneratorRole:
    """Generator/EMA state owned by rank 0 in a split schedule."""

    def __init__(
        self,
        config: MiniDistillConfig,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed)
            self.generator = MiniGenerator(config)
        self.generator.to(device=self.device, dtype=dtype)
        self.optimizer = torch.optim.SGD(
            [parameter for parameter in self.generator.parameters() if parameter.requires_grad],
            lr=config.generator_lr,
        )
        self.ema = GeneratorEMA(self.generator)
        self.version = 0
        self.ema_version = 0
        self.counters: Dict[str, int] = {}

    def _increment(self, name: str, amount: int = 1) -> None:
        self.counters[name] = self.counters.get(name, 0) + amount

    def rollout(
        self, batch: MiniBatch, *, save_replay: bool
    ) -> Tuple[Tensor, Optional[ReplayRecord]]:
        latent = batch.latent.detach()
        history = torch.zeros_like(latent)
        record: Optional[ReplayRecord] = None
        for section in range(self.config.sections):
            for denoise in range(self.config.denoise_steps):
                input_latent = latent
                input_history = history
                with torch.no_grad():
                    latent = self.generator.forward_step(
                        input_latent,
                        input_history,
                        batch.conditioning,
                        section,
                        denoise,
                    )
                self._increment("generator_forward_no_grad")
                if (
                    save_replay
                    and section == self.config.resolved_selected_section
                    and denoise == self.config.resolved_selected_denoise
                ):
                    record = ReplayRecord(
                        latent=input_latent.detach(),
                        history=input_history.detach(),
                        conditioning=batch.conditioning.detach(),
                        section=section,
                        denoise=denoise,
                        expected_output=latent.detach(),
                    )
            with torch.no_grad():
                history = 0.7 * history + 0.3 * latent
                self._increment("history_update")
        return latent, record

    def begin_accumulation(self) -> None:
        """Open one Generator parameter version for microbatch accumulation."""

        self.optimizer.zero_grad(set_to_none=True)
        self._increment("generator_accumulation_begin")

    def backward_input_gradient(
        self,
        input_gradient: Tensor,
        replay_record: ReplayRecord,
        generator_loss: float,
        *,
        gradient_scale: float = 1.0,
    ) -> GeneratorMicrobatchResult:
        """Replay one selected step and accumulate its externally supplied gradient."""

        if gradient_scale <= 0.0:
            raise ValueError("gradient_scale must be positive")
        with torch.enable_grad():
            replayed = self.generator.forward_step(
                replay_record.latent,
                replay_record.history,
                replay_record.conditioning,
                replay_record.section,
                replay_record.denoise,
            )
        self._increment("generator_forward_replay")
        replay_error = float(
            (replayed.detach() - replay_record.expected_output).abs().max().cpu()
        )
        replayed.backward(input_gradient * gradient_scale)
        self._increment("generator_backward")
        return GeneratorMicrobatchResult(
            loss=generator_loss,
            replay_max_abs_error=replay_error,
            fake_checksum=float(replay_record.expected_output.double().sum().cpu()),
        )

    def finish_accumulation(
        self, microbatches: Tuple[GeneratorMicrobatchResult, ...]
    ) -> GeneratorRoleResult:
        """Cross the version barrier once after every replay backward finishes."""

        if not microbatches:
            raise ValueError("at least one Generator microbatch is required")
        gradients = _gradient_snapshot(self.generator)
        self.optimizer.step()
        self.version += 1
        self.ema.update(self.generator, self.config.ema_decay)
        self.ema_version += 1
        self._increment("generator_optimizer_step")
        self._increment("ema_update")
        return GeneratorRoleResult(
            loss=sum(item.loss for item in microbatches) / len(microbatches),
            replay_max_abs_error=max(
                item.replay_max_abs_error for item in microbatches
            ),
            fake_checksum=sum(item.fake_checksum for item in microbatches),
            gradients=gradients,
            parameters=_parameter_snapshot(self.generator),
            ema_parameters={
                name: value.detach().cpu().clone() for name, value in self.ema.values.items()
            },
        )

    def apply_input_gradient(
        self,
        input_gradient: Tensor,
        replay_record: ReplayRecord,
        generator_loss: float,
    ) -> GeneratorRoleResult:
        """Backward-compatible one-microbatch update."""

        self.begin_accumulation()
        microbatch = self.backward_input_gradient(
            input_gradient,
            replay_record,
            generator_loss,
        )
        return self.finish_accumulation((microbatch,))


class SplitScoreRole:
    """Real/Fake Score and trainable Critic state owned by rank 1."""

    def __init__(
        self,
        config: MiniDistillConfig,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        # Consume Generator initialization exactly as MiniDistillWorkload does,
        # then instantiate Score. The temporary never moves to the accelerator.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed)
            _generator_rng_prefix = MiniGenerator(config)
            self.score = SharedScore(config)
            del _generator_rng_prefix
        self.score.to(device=self.device, dtype=dtype)
        self.optimizer = torch.optim.SGD(
            [parameter for parameter in self.score.parameters() if parameter.requires_grad],
            lr=config.critic_lr,
        )
        self.version = 0
        self.counters: Dict[str, int] = {}

    def _increment(self, name: str, amount: int = 1) -> None:
        self.counters[name] = self.counters.get(name, 0) + amount

    @contextmanager
    def _freeze_trainables(self) -> Iterator[None]:
        parameters = [
            parameter for parameter in self.score.parameters() if parameter.requires_grad
        ]
        previous = [parameter.requires_grad for parameter in parameters]
        for parameter in parameters:
            parameter.requires_grad_(False)
        try:
            yield
        finally:
            for parameter, requires_grad in zip(parameters, previous):
                parameter.requires_grad_(requires_grad)

    def compute_input_gradient(self, fake_sample: Tensor) -> ScoreGradientResult:
        self.optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            real_prediction = self.score.predict(fake_sample.detach(), "real")
            fake_prediction = self.score.predict(fake_sample.detach(), "fake")
            self._increment("score_real_forward")
            self._increment("score_fake_forward")
            scale = real_prediction.detach().abs().mean().clamp_min(1.0e-3)
            dmd_gradient = (fake_prediction - real_prediction) / scale
        dmd_input_gradient = dmd_gradient / fake_sample.numel()
        dmd_surrogate = float(
            ((fake_sample.detach() * dmd_gradient).sum() / fake_sample.numel()).cpu()
        )
        gan_loss_value = 0.0
        total_input_gradient = dmd_input_gradient
        if self.config.use_gan:
            boundary = fake_sample.detach().requires_grad_(True)
            with self._freeze_trainables():
                gan_loss = F.softplus(
                    -self.score.discriminate(boundary, "fake")
                ).mean()
                gan_gradient = torch.autograd.grad(gan_loss, boundary)[0]
            gan_loss_value = float(gan_loss.detach().cpu())
            total_input_gradient = (
                total_input_gradient + self.config.gan_weight * gan_gradient.detach()
            )
            self._increment("gan_generator_forward")
        if any(
            parameter.grad is not None
            for parameter in self.score.parameters()
            if parameter.requires_grad
        ):
            raise RuntimeError("input-gradient phase polluted Critic parameter gradients")
        return ScoreGradientResult(
            input_gradient=total_input_gradient.detach(),
            generator_loss=dmd_surrogate + self.config.gan_weight * gan_loss_value,
            dmd_surrogate=dmd_surrogate,
            gan_loss=gan_loss_value,
            dmd_gradient_checksum=float(dmd_gradient.double().sum().cpu()),
        )

    def begin_critic_accumulation(self) -> None:
        """Open one Critic parameter version for microbatch accumulation."""

        self.optimizer.zero_grad(set_to_none=True)
        self._increment("critic_accumulation_begin")

    def backward_critic(
        self,
        fake_sample: Tensor,
        real_sample: Tensor,
        *,
        gradient_scale: float = 1.0,
    ) -> CriticMicrobatchResult:
        """Accumulate one Critic microbatch without changing its parameter version."""

        if gradient_scale <= 0.0:
            raise ValueError("gradient_scale must be positive")
        with torch.no_grad():
            target = self.score.predict(fake_sample.detach(), "real")
            self._increment("score_real_forward")
        prediction = self.score.predict(fake_sample.detach(), "fake")
        self._increment("score_fake_forward")
        loss = F.mse_loss(prediction, target.detach())
        if self.config.use_gan:
            real_logits = self.score.discriminate(real_sample, "real")
            fake_logits = self.score.discriminate(fake_sample.detach(), "fake")
            discriminator_loss = (
                F.softplus(-real_logits).mean() + F.softplus(fake_logits).mean()
            )
            loss = loss + self.config.critic_gan_weight * discriminator_loss
            self._increment("gan_critic_forward", 2)
        (loss * gradient_scale).backward()
        self._increment("critic_backward")
        return CriticMicrobatchResult(loss=float(loss.detach().cpu()))

    def finish_critic_accumulation(
        self, microbatches: Tuple[CriticMicrobatchResult, ...]
    ) -> CriticRoleResult:
        """Step Critic exactly once after all microbatches observe one version."""

        if not microbatches:
            raise ValueError("at least one Critic microbatch is required")
        gradients = _gradient_snapshot(self.score)
        self.optimizer.step()
        self.version += 1
        self._increment("critic_optimizer_step")
        return CriticRoleResult(
            loss=sum(item.loss for item in microbatches) / len(microbatches),
            gradients=gradients,
            parameters=_parameter_snapshot(self.score),
        )

    def critic_update(self, fake_sample: Tensor, real_sample: Tensor) -> CriticRoleResult:
        """Backward-compatible one-microbatch Critic update."""

        self.begin_critic_accumulation()
        microbatch = self.backward_critic(fake_sample, real_sample)
        return self.finish_critic_accumulation((microbatch,))
