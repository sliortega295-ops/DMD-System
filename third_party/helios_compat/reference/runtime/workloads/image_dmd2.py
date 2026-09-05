"""Image-DMD2-shaped Level-3 workload for runtime generalization tests.

This is not the full DMD2 model or a quality experiment.  It preserves the
execution semantics found in tianweiy/DMD2 commit
8d8fa55633d47cfb81bbc7a892e7248f9518763f: a one-step trainable Generator,
independent frozen-real and trainable-fake guidance networks, a detached DMD
input gradient, optional differentiable GAN input gradient, periodic Generator
updates, and a fake-guidance update on the cached pre-update generated image.
Unlike the Helios-shaped workload it has no video section/history axis, no
shared Real/Fake backbone, and no EMA state.
"""

from __future__ import annotations

import copy
import math
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Dict, Iterator, Mapping, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from runtime.ir import (
    Edge,
    EdgeType,
    ExecutionGraph,
    GradMode,
    OperationNode,
    OperationType,
    PhysicalState,
)
from .split_distill import (
    CriticMicrobatchResult,
    CriticRoleResult,
    GeneratorMicrobatchResult,
    GeneratorRoleResult,
    ScoreGradientResult,
)


DMD2_SOURCE_COMMIT = "8d8fa55633d47cfb81bbc7a892e7248f9518763f"


@dataclass(frozen=True)
class ImageDMD2Config:
    batch_size: int = 2
    channels: int = 256
    image_size: int = 32
    generator_depth: int = 8
    guidance_depth: int = 8
    generator_update_frequency: int = 2
    guidance_update_frequency: int = 1
    generator_lr: float = 2.0e-3
    guidance_lr: float = 1.0e-3
    gan_weight: float = 0.2
    guidance_gan_weight: float = 0.1
    use_gan: bool = True
    seed: int = 20260722

    def __post_init__(self) -> None:
        for name in (
            "batch_size",
            "channels",
            "image_size",
            "generator_depth",
            "guidance_depth",
            "generator_update_frequency",
            "guidance_update_frequency",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.channels < 8:
            raise ValueError("channels must be at least 8")


@dataclass(frozen=True)
class ImageDMD2Batch:
    latent: Tensor
    conditioning: Tensor
    real_sample: Tensor

    @classmethod
    def deterministic(
        cls,
        config: ImageDMD2Config,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        seed_offset: int = 17,
    ) -> "ImageDMD2Batch":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(config.seed + seed_offset)
        shape = (
            config.batch_size,
            config.channels,
            config.image_size,
            config.image_size,
        )
        latent = torch.randn(shape, generator=generator, dtype=torch.float32)
        # Class/text conditioning is represented as a broadcast channel vector.
        conditioning = torch.randn(
            config.batch_size,
            config.channels,
            1,
            1,
            generator=generator,
            dtype=torch.float32,
        ) * 0.15
        real_sample = torch.randn(shape, generator=generator, dtype=torch.float32)
        return cls(
            latent=latent.to(device=device, dtype=dtype),
            conditioning=conditioning.to(device=device, dtype=dtype),
            real_sample=real_sample.to(device=device, dtype=dtype),
        )


class ImageResidualNetwork(nn.Module):
    """Small convolutional network standing in for an image diffusion UNet."""

    def __init__(self, channels: int, depth: int) -> None:
        super().__init__()
        groups = math.gcd(channels, 32)
        self.layers = nn.ModuleList(
            nn.Conv2d(channels, channels, kernel_size=1, bias=True)
            for _ in range(depth)
        )
        self.norms = nn.ModuleList(
            nn.GroupNorm(groups, channels) for _ in range(depth)
        )
        self.output = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def features(self, value: Tensor) -> Tensor:
        hidden = value
        for layer, norm in zip(self.layers, self.norms):
            hidden = hidden + 0.25 * torch.tanh(layer(norm(hidden)))
        return hidden

    def forward(self, value: Tensor) -> Tensor:
        return self.output(self.features(value))


class ImageDMD2Generator(nn.Module):
    def __init__(self, config: ImageDMD2Config) -> None:
        super().__init__()
        self.network = ImageResidualNetwork(config.channels, config.generator_depth)
        self.condition_scale = nn.Parameter(torch.full((config.channels,), 0.02))

    def forward(self, noise: Tensor, conditioning: Tensor) -> Tensor:
        conditioned = noise + conditioning * self.condition_scale.view(1, -1, 1, 1)
        return 0.65 * noise + 0.35 * torch.tanh(self.network(conditioned))


class ImageDMD2Guidance(nn.Module):
    """Independent frozen real teacher plus trainable fake guidance network."""

    def __init__(self, config: ImageDMD2Config) -> None:
        super().__init__()
        self.real_unet = ImageResidualNetwork(config.channels, config.guidance_depth)
        self.fake_unet = copy.deepcopy(self.real_unet)
        self.real_unet.requires_grad_(False)
        self.fake_unet.requires_grad_(True)
        self.gan_head = nn.Conv2d(config.channels, 1, kernel_size=1)

    def real_prediction(self, value: Tensor) -> Tensor:
        return self.real_unet(value)

    def fake_prediction(self, value: Tensor) -> Tensor:
        return self.fake_unet(value)

    def discriminate(self, value: Tensor) -> Tensor:
        features = self.fake_unet.features(value)
        return self.gan_head(features).mean(dim=(2, 3)).squeeze(1)


@dataclass(frozen=True)
class ImageReplayRecord:
    latent: Tensor
    conditioning: Tensor
    expected_output: Tensor


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


def _module_parameter_bytes(module: nn.Module) -> int:
    return sum(parameter.numel() * parameter.element_size() for parameter in module.parameters())


def _module_trainable_bytes(module: nn.Module) -> int:
    return sum(
        parameter.numel() * parameter.element_size()
        for parameter in module.parameters()
        if parameter.requires_grad
    )


class ImageDMD2GeneratorRole:
    """One-step Generator role; deliberately has no EMA state."""

    has_ema = False
    reuses_boundary_fake_for_guidance = True

    def __init__(
        self,
        config: ImageDMD2Config,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed)
            self.generator = ImageDMD2Generator(config)
        self.generator.to(device=self.device, dtype=dtype)
        self.optimizer = torch.optim.SGD(self.generator.parameters(), lr=config.generator_lr)
        self.version = 0
        # Compatibility with the profile schema: absence is represented by version 0
        # and an empty state, not by a fabricated EMA copy.
        self.ema_version = 0
        self.counters: Dict[str, int] = {}

    def _increment(self, name: str, amount: int = 1) -> None:
        self.counters[name] = self.counters.get(name, 0) + amount

    def rollout(
        self, batch: ImageDMD2Batch, *, save_replay: bool
    ) -> Tuple[Tensor, Optional[ImageReplayRecord]]:
        with torch.no_grad():
            image = self.generator(batch.latent, batch.conditioning)
        self._increment("generator_forward_no_grad")
        record = None
        if save_replay:
            record = ImageReplayRecord(
                latent=batch.latent.detach(),
                conditioning=batch.conditioning.detach(),
                expected_output=image.detach(),
            )
        return image, record

    def begin_accumulation(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        self._increment("generator_accumulation_begin")

    def backward_input_gradient(
        self,
        input_gradient: Tensor,
        replay_record: ImageReplayRecord,
        generator_loss: float,
        *,
        gradient_scale: float = 1.0,
    ) -> GeneratorMicrobatchResult:
        if gradient_scale <= 0.0:
            raise ValueError("gradient_scale must be positive")
        with torch.enable_grad():
            replayed = self.generator(replay_record.latent, replay_record.conditioning)
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
        if not microbatches:
            raise ValueError("at least one Generator microbatch is required")
        gradients = _gradient_snapshot(self.generator)
        self.optimizer.step()
        self.version += 1
        self._increment("generator_optimizer_step")
        return GeneratorRoleResult(
            loss=sum(item.loss for item in microbatches) / len(microbatches),
            replay_max_abs_error=max(item.replay_max_abs_error for item in microbatches),
            fake_checksum=sum(item.fake_checksum for item in microbatches),
            gradients=gradients,
            parameters=_parameter_snapshot(self.generator),
            ema_parameters={},
        )


class ImageDMD2GuidanceRole:
    """Frozen-real/trainable-fake guidance and optional GAN owned by rank 1."""

    def __init__(
        self,
        config: ImageDMD2Config,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed + 1)
            self.score = ImageDMD2Guidance(config)
        self.score.to(device=self.device, dtype=dtype)
        trainables = [parameter for parameter in self.score.parameters() if parameter.requires_grad]
        self.optimizer = torch.optim.SGD(trainables, lr=config.guidance_lr)
        self.version = 0
        self.counters: Dict[str, int] = {}

    def _increment(self, name: str, amount: int = 1) -> None:
        self.counters[name] = self.counters.get(name, 0) + amount

    @contextmanager
    def _freeze_trainables(self) -> Iterator[None]:
        parameters = [parameter for parameter in self.score.parameters() if parameter.requires_grad]
        for parameter in parameters:
            parameter.requires_grad_(False)
        try:
            yield
        finally:
            for parameter in parameters:
                parameter.requires_grad_(True)

    def compute_input_gradient(self, fake_sample: Tensor) -> ScoreGradientResult:
        self.optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            real_prediction = self.score.real_prediction(fake_sample.detach())
            fake_prediction = self.score.fake_prediction(fake_sample.detach())
            self._increment("real_teacher_forward")
            self._increment("fake_guidance_dmd_forward")
            p_real = fake_sample.detach() - real_prediction
            p_fake = fake_sample.detach() - fake_prediction
            scale = p_real.abs().mean().clamp_min(1.0e-3)
            dmd_gradient = (p_real - p_fake) / scale
        total_input_gradient = dmd_gradient / fake_sample.numel()
        dmd_surrogate = float(
            ((fake_sample.detach() * dmd_gradient).sum() / fake_sample.numel()).cpu()
        )
        gan_loss_value = 0.0
        if self.config.use_gan:
            boundary = fake_sample.detach().requires_grad_(True)
            with self._freeze_trainables():
                gan_loss = F.softplus(-self.score.discriminate(boundary)).mean()
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
            raise RuntimeError("input-gradient phase polluted guidance gradients")
        return ScoreGradientResult(
            input_gradient=total_input_gradient.detach(),
            generator_loss=dmd_surrogate + self.config.gan_weight * gan_loss_value,
            dmd_surrogate=dmd_surrogate,
            gan_loss=gan_loss_value,
            dmd_gradient_checksum=float(dmd_gradient.double().sum().cpu()),
        )

    def begin_critic_accumulation(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        self._increment("guidance_accumulation_begin")

    def backward_critic(
        self,
        fake_sample: Tensor,
        real_sample: Tensor,
        *,
        gradient_scale: float = 1.0,
    ) -> CriticMicrobatchResult:
        if gradient_scale <= 0.0:
            raise ValueError("gradient_scale must be positive")
        target = fake_sample.detach()
        # Deterministic stand-in for DMD2's sampled diffusion timestep/noise.
        noisy_fake = target + 0.15 * torch.tanh(real_sample.detach())
        prediction = self.score.fake_prediction(noisy_fake)
        loss = F.mse_loss(prediction, target)
        self._increment("fake_guidance_train_forward")
        if self.config.use_gan:
            real_logits = self.score.discriminate(real_sample.detach())
            fake_logits = self.score.discriminate(target)
            gan_loss = F.softplus(-real_logits).mean() + F.softplus(fake_logits).mean()
            loss = loss + self.config.guidance_gan_weight * gan_loss
            self._increment("gan_guidance_forward", 2)
        (loss * gradient_scale).backward()
        self._increment("guidance_backward")
        return CriticMicrobatchResult(loss=float(loss.detach().cpu()))

    def finish_critic_accumulation(
        self, microbatches: Tuple[CriticMicrobatchResult, ...]
    ) -> CriticRoleResult:
        if not microbatches:
            raise ValueError("at least one guidance microbatch is required")
        gradients = _gradient_snapshot(self.score)
        self.optimizer.step()
        self.version += 1
        self._increment("guidance_optimizer_step")
        return CriticRoleResult(
            loss=sum(item.loss for item in microbatches) / len(microbatches),
            gradients=gradients,
            parameters=_parameter_snapshot(self.score),
        )


def build_image_dmd2_execution_graph(
    config: ImageDMD2Config,
) -> ExecutionGraph:
    """Build the generator-update branch of the source-grounded DMD2 graph."""

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        generator = ImageDMD2Generator(config)
        torch.manual_seed(config.seed + 1)
        guidance = ImageDMD2Guidance(config)
    graph = ExecutionGraph(
        metadata={
            "workload": "image_dmd2_structure_equivalent",
            "level": 3,
            "nested_axes": ["iteration", "microbatch", "component"],
            "continuous_history": False,
            "external_source": {
                "repository": "https://github.com/tianweiy/DMD2",
                "commit": DMD2_SOURCE_COMMIT,
                "files": [
                    "main/edm/edm_guidance.py",
                    "main/edm/edm_unified_model.py",
                    "main/edm/train_edm.py",
                ],
            },
            "config": asdict(config),
        }
    )
    generator_bytes = _module_parameter_bytes(generator)
    graph.add_physical_state(
        PhysicalState(
            "generator",
            parameter_bytes=generator_bytes,
            trainable_bytes=generator_bytes,
            optimizer_bytes=generator_bytes,
            modes=("one_step",),
            metadata={"full_model_trainable": True, "ema": False},
        )
    )
    teacher_bytes = _module_parameter_bytes(guidance.real_unet)
    fake_bytes = _module_parameter_bytes(guidance.fake_unet) + _module_parameter_bytes(guidance.gan_head)
    graph.add_physical_state(
        PhysicalState(
            "real_teacher",
            parameter_bytes=teacher_bytes,
            trainable_bytes=0,
            modes=("real",),
            metadata={"frozen": True, "shared_with_fake": False},
        )
    )
    graph.add_physical_state(
        PhysicalState(
            "fake_guidance",
            parameter_bytes=fake_bytes,
            trainable_bytes=fake_bytes,
            optimizer_bytes=fake_bytes,
            modes=("dmd", "gan", "train"),
            metadata={"frozen": False, "shared_with_real": False},
        )
    )

    nodes = [
        OperationNode(
            "generator_forward",
            "Generator",
            "generator",
            OperationType.FORWARD,
            GradMode.NO_GRAD,
            adapter_mode="one_step",
            update_frequency=config.guidance_update_frequency,
            nested_position=(0,),
            metadata={"one_step_image_generation": True, "replayable": True},
        ),
        OperationNode(
            "real_teacher_dmd",
            "RealTeacher",
            "real_teacher",
            OperationType.FORWARD,
            GradMode.NO_GRAD,
            adapter_mode="real",
            update_frequency=config.generator_update_frequency,
            condition="generator_update_due",
        ),
        OperationNode(
            "fake_guidance_dmd",
            "FakeGuidance",
            "fake_guidance",
            OperationType.FORWARD,
            GradMode.NO_GRAD,
            adapter_mode="dmd",
            update_frequency=config.generator_update_frequency,
            condition="generator_update_due",
        ),
    ]
    if config.use_gan:
        nodes.append(
            OperationNode(
                "gan_generator",
                "GAN",
                "fake_guidance",
                OperationType.FORWARD,
                GradMode.GRAD,
                adapter_mode="gan",
                update_frequency=config.generator_update_frequency,
                condition="generator_update_due and gan_enabled",
            )
        )
    nodes.extend(
        [
            OperationNode(
                "generator_backward",
                "Generator",
                "generator",
                OperationType.BACKWARD,
                GradMode.EXTERNAL_GRAD,
                adapter_mode="one_step",
                update_frequency=config.generator_update_frequency,
                condition="generator_update_due",
                metadata={"gradient_boundary": True},
            ),
            OperationNode(
                "generator_optimizer_step",
                "Generator",
                "generator",
                OperationType.OPTIMIZER_STEP,
                GradMode.NO_GRAD,
                adapter_mode="one_step",
                update_frequency=config.generator_update_frequency,
                condition="generator_update_due",
                parameter_version="generator:v+1",
            ),
            OperationNode(
                "fake_guidance_train",
                "FakeGuidance",
                "fake_guidance",
                OperationType.FORWARD,
                GradMode.GRAD,
                adapter_mode="train",
                update_frequency=config.guidance_update_frequency,
                metadata={"input": "cached_pre_update_generator_sample"},
            ),
            OperationNode(
                "fake_guidance_backward",
                "FakeGuidance",
                "fake_guidance",
                OperationType.BACKWARD,
                GradMode.GRAD,
                adapter_mode="train",
                update_frequency=config.guidance_update_frequency,
            ),
            OperationNode(
                "fake_guidance_optimizer_step",
                "FakeGuidance",
                "fake_guidance",
                OperationType.OPTIMIZER_STEP,
                GradMode.NO_GRAD,
                adapter_mode="train",
                update_frequency=config.guidance_update_frequency,
                parameter_version="fake_guidance:v+1",
            ),
        ]
    )
    for node in nodes:
        graph.add_node(node)
    for target in ("real_teacher_dmd", "fake_guidance_dmd"):
        graph.add_edge(Edge("generator_forward", target, EdgeType.DATA, name="x_fake"))
    graph.add_edge(Edge("real_teacher_dmd", "generator_backward", EdgeType.GRADIENT, name="dmd_dx"))
    graph.add_edge(Edge("fake_guidance_dmd", "generator_backward", EdgeType.GRADIENT, name="dmd_dx"))
    if config.use_gan:
        graph.add_edge(Edge("generator_forward", "gan_generator", EdgeType.DATA, name="x_fake"))
        graph.add_edge(Edge("gan_generator", "generator_backward", EdgeType.GRADIENT, name="gan_dx"))
    graph.add_edge(Edge("generator_backward", "generator_optimizer_step", EdgeType.VERSION))
    # Source DMD2 updates Generator first, then trains fake guidance on the cached
    # detached sample produced before that update.
    graph.add_edge(Edge("generator_forward", "fake_guidance_train", EdgeType.STATE, name="cached_pre_update_fake"))
    graph.add_edge(Edge("generator_optimizer_step", "fake_guidance_train", EdgeType.CONTROL))
    graph.add_edge(Edge("fake_guidance_train", "fake_guidance_backward", EdgeType.GRADIENT))
    graph.add_edge(Edge("fake_guidance_dmd", "fake_guidance_backward", EdgeType.VERSION, name="same_fake_guidance_version"))
    graph.add_edge(Edge("fake_guidance_backward", "fake_guidance_optimizer_step", EdgeType.VERSION))
    graph.add_edge(
        Edge(
            "generator_forward",
            "generator_backward",
            EdgeType.FREQUENCY,
            condition=f"iteration % {config.generator_update_frequency} == 0",
        )
    )
    graph.validate()
    return graph
