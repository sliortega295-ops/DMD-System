"""A small but structurally faithful multi-component distillation workload.

This is a Level-3 workload, not a numerical approximation of Helios. It keeps
Helios-like scheduling semantics that matter to a runtime: nested section and
denoising axes, a mostly-no-grad Generator rollout, continuous latent/history
state, one physical Score backbone serving Real/Fake/Critic modes, a gradient
edge from Score/GAN to Generator, distinct update frequencies, and EMA state.
"""

from __future__ import annotations

import copy
import math
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterator, List, Mapping, MutableMapping, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from runtime.ir import (
    Edge,
    EdgeType,
    ExecutionGraph,
    GradMode,
    GradientStrategy,
    OperationNode,
    OperationType,
    PhysicalState,
)
from runtime.residency import ModuleResidencyBackend, ResidencyPolicy
from runtime.trace import PhaseTracer


@dataclass(frozen=True)
class MiniDistillConfig:
    batch_size: int = 2
    tokens: int = 32
    hidden_size: int = 64
    generator_depth: int = 3
    score_depth: int = 3
    adapter_rank: int = 8
    sections: int = 3
    denoise_steps: int = 4
    selected_section: int = -1
    selected_denoise: int = -1
    generator_update_frequency: int = 2
    critic_update_frequency: int = 1
    ema_update_frequency: int = 1
    generator_lr: float = 2.0e-2
    critic_lr: float = 1.0e-2
    ema_decay: float = 0.95
    gan_weight: float = 0.2
    critic_gan_weight: float = 0.1
    use_gan: bool = True
    use_reward: bool = False
    score_layout: str = "shared"
    seed: int = 20260722

    def __post_init__(self) -> None:
        positive_ints = (
            "batch_size",
            "tokens",
            "hidden_size",
            "generator_depth",
            "score_depth",
            "adapter_rank",
            "sections",
            "denoise_steps",
            "generator_update_frequency",
            "critic_update_frequency",
            "ema_update_frequency",
        )
        for name in positive_ints:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.score_layout not in {"shared", "duplicated"}:
            raise ValueError("score_layout must be 'shared' or 'duplicated'")
        selected_section = self.resolved_selected_section
        selected_denoise = self.resolved_selected_denoise
        if not 0 <= selected_section < self.sections:
            raise ValueError("selected_section is outside the section axis")
        if not 0 <= selected_denoise < self.denoise_steps:
            raise ValueError("selected_denoise is outside the denoising axis")
        # A single-op replay is semantics preserving only when no differentiable
        # state transition follows the chosen operation.
        if selected_section != self.sections - 1 or selected_denoise != self.denoise_steps - 1:
            raise ValueError(
                "the MVP replay boundary must be the final section/final denoise step"
            )

    @property
    def resolved_selected_section(self) -> int:
        return self.sections - 1 if self.selected_section < 0 else self.selected_section

    @property
    def resolved_selected_denoise(self) -> int:
        return self.denoise_steps - 1 if self.selected_denoise < 0 else self.selected_denoise

    @property
    def rollout_forwards(self) -> int:
        return self.sections * self.denoise_steps


@dataclass(frozen=True)
class MiniBatch:
    latent: Tensor
    conditioning: Tensor
    real_sample: Tensor

    @classmethod
    def deterministic(
        cls,
        config: MiniDistillConfig,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        seed_offset: int = 17,
    ) -> "MiniBatch":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(config.seed + seed_offset)
        shape = (config.batch_size, config.tokens, config.hidden_size)
        latent = torch.randn(shape, generator=generator, dtype=torch.float32)
        conditioning = torch.randn(shape, generator=generator, dtype=torch.float32) * 0.25
        real_sample = torch.randn(shape, generator=generator, dtype=torch.float32)
        return cls(
            latent=latent.to(device=device, dtype=dtype),
            conditioning=conditioning.to(device=device, dtype=dtype),
            real_sample=real_sample.to(device=device, dtype=dtype),
        )


class LowRankAdapter(nn.Module):
    def __init__(self, hidden_size: int, rank: int) -> None:
        super().__init__()
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        self.scale = 1.0 / math.sqrt(rank)
        nn.init.normal_(self.down.weight, std=0.04)
        nn.init.normal_(self.up.weight, std=0.04)

    def forward(self, value: Tensor) -> Tensor:
        return self.up(self.down(value)) * self.scale


class FrozenResidualBackbone(nn.Module):
    def __init__(self, hidden_size: int, depth: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            nn.Linear(hidden_size, hidden_size) for _ in range(depth)
        )
        self.norm = nn.LayerNorm(hidden_size)
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, value: Tensor) -> Tensor:
        hidden = value
        for layer in self.layers:
            hidden = hidden + 0.35 * torch.tanh(layer(self.norm(hidden)))
        return self.norm(hidden)


class MiniGenerator(nn.Module):
    def __init__(self, config: MiniDistillConfig) -> None:
        super().__init__()
        self.backbone = FrozenResidualBackbone(
            config.hidden_size, config.generator_depth
        )
        self.adapter = LowRankAdapter(config.hidden_size, config.adapter_rank)
        self.condition_scale = nn.Parameter(torch.full((config.hidden_size,), 0.02))

    def forward_step(
        self,
        latent: Tensor,
        history: Tensor,
        conditioning: Tensor,
        section: int,
        denoise: int,
    ) -> Tensor:
        position = (section + 1) * 0.013 + (denoise + 1) * 0.007
        model_input = latent + 0.12 * history + conditioning * self.condition_scale + position
        prediction = self.backbone(model_input) + self.adapter(model_input)
        return 0.78 * latent + 0.22 * torch.tanh(prediction)

    def trainable_named_parameters(self) -> Iterator[Tuple[str, nn.Parameter]]:
        for name, parameter in self.named_parameters():
            if parameter.requires_grad:
                yield name, parameter


class SharedScore(nn.Module):
    """One physical backbone with mutually-exclusive Real/Fake logical modes."""

    VALID_MODES = ("real", "fake")

    def __init__(self, config: MiniDistillConfig) -> None:
        super().__init__()
        self.backbone = FrozenResidualBackbone(config.hidden_size, config.score_depth)
        self.adapter = LowRankAdapter(config.hidden_size, config.adapter_rank)
        self.score_head = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.gan_head = nn.Linear(config.hidden_size, 1)
        for parameter in self.score_head.parameters():
            parameter.requires_grad_(False)
        self._active_mode: Optional[str] = None
        self.mode_transitions: List[str] = []

    @contextmanager
    def mode(self, mode: str) -> Iterator[None]:
        if mode not in self.VALID_MODES:
            raise ValueError(f"unknown Score adapter mode: {mode}")
        if self._active_mode is not None:
            raise RuntimeError(
                f"Score mode {self._active_mode!r} is active; concurrent mode "
                f"{mode!r} is not representable by this physical state"
            )
        self._active_mode = mode
        self.mode_transitions.append(mode)
        try:
            yield
        finally:
            self._active_mode = None

    def features(self, value: Tensor) -> Tensor:
        if self._active_mode is None:
            raise RuntimeError("Score forward requires an explicit adapter mode")
        hidden = self.backbone(value)
        if self._active_mode == "fake":
            hidden = hidden + self.adapter(hidden)
        return hidden

    def predict(self, value: Tensor, mode: str) -> Tensor:
        with self.mode(mode):
            return self.score_head(self.features(value))

    def discriminate(self, value: Tensor, mode: str) -> Tensor:
        with self.mode(mode):
            return self.gan_head(self.features(value)).mean(dim=1).squeeze(-1)

    def trainable_named_parameters(self) -> Iterator[Tuple[str, nn.Parameter]]:
        for name, parameter in self.named_parameters():
            if parameter.requires_grad:
                yield name, parameter


class FrozenReward(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.head = nn.Linear(hidden_size, 1, bias=False)
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, value: Tensor) -> Tensor:
        return self.head(value).mean()


class GeneratorEMA:
    def __init__(self, generator: MiniGenerator) -> None:
        self.values = {
            name: parameter.detach().clone()
            for name, parameter in generator.trainable_named_parameters()
        }
        self.version = 0

    @torch.no_grad()
    def update(self, generator: MiniGenerator, decay: float) -> None:
        for name, parameter in generator.trainable_named_parameters():
            self.values[name].mul_(decay).add_(parameter.detach(), alpha=1.0 - decay)
        self.version += 1


@dataclass(frozen=True)
class ReplayRecord:
    latent: Tensor
    history: Tensor
    conditioning: Tensor
    section: int
    denoise: int
    expected_output: Tensor


@dataclass
class IterationResult:
    iteration: int
    strategy: str
    generator_updated: bool
    critic_updated: bool
    ema_updated: bool
    generator_loss: Optional[float]
    dmd_surrogate: Optional[float]
    generator_gan_loss: Optional[float]
    critic_loss: Optional[float]
    reward: Optional[float]
    replay_max_abs_error: Optional[float]
    fake_checksum: Optional[float]
    dmd_gradient_checksum: Optional[float]
    counters: Dict[str, int]
    score_mode_transitions: List[str]
    versions: Dict[str, int]
    generator_gradients: Dict[str, Tensor] = field(default_factory=dict)
    critic_gradients: Dict[str, Tensor] = field(default_factory=dict)
    generator_parameters: Dict[str, Tensor] = field(default_factory=dict)
    score_parameters: Dict[str, Tensor] = field(default_factory=dict)
    ema_parameters: Dict[str, Tensor] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        result = {
            key: value
            for key, value in self.__dict__.items()
            if key
            not in {
                "generator_gradients",
                "critic_gradients",
                "generator_parameters",
                "score_parameters",
                "ema_parameters",
            }
        }
        result["gradient_norms"] = {
            "generator": _tensor_dict_norm(self.generator_gradients),
            "critic": _tensor_dict_norm(self.critic_gradients),
        }
        result["parameter_checksums"] = {
            "generator": _tensor_dict_checksum(self.generator_parameters),
            "score": _tensor_dict_checksum(self.score_parameters),
            "ema": _tensor_dict_checksum(self.ema_parameters),
        }
        return result


def _tensor_dict_norm(values: Mapping[str, Tensor]) -> float:
    if not values:
        return 0.0
    return math.sqrt(
        sum(float(tensor.detach().float().pow(2).sum().cpu()) for tensor in values.values())
    )


def _tensor_dict_checksum(values: Mapping[str, Tensor]) -> float:
    return sum(float(tensor.detach().double().sum().cpu()) for tensor in values.values())


def _snapshot_named(named_parameters: Iterator[Tuple[str, nn.Parameter]]) -> Dict[str, Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in named_parameters
    }


def _gradient_snapshot(named_parameters: Iterator[Tuple[str, nn.Parameter]]) -> Dict[str, Tensor]:
    result: Dict[str, Tensor] = {}
    for name, parameter in named_parameters:
        if parameter.grad is not None:
            result[name] = parameter.grad.detach().cpu().clone()
    return result


class MiniDistillWorkload:
    def __init__(
        self,
        config: MiniDistillConfig,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        tracer: Optional[PhaseTracer] = None,
        residency_policy: ResidencyPolicy | str = ResidencyPolicy.RESIDENT,
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.dtype = dtype
        self.tracer = tracer
        # Make separately-created strategy runners byte-identical.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed)
            self.generator = MiniGenerator(config)
            self.score = SharedScore(config)
            if config.score_layout == "shared":
                self.real_score = self.score
                self.fake_score = self.score
                self.gan_score: Optional[SharedScore] = (
                    self.score if config.use_gan else None
                )
                self._physical_score_modules = {"score": self.score}
            else:
                # This deliberately models a logical-model-centric runtime:
                # Real Score, Fake Score, GAN, and the trainable Critic each
                # own a full physical copy. Frozen replicas are synchronized
                # from the Critic after its optimizer step.
                self.real_score = copy.deepcopy(self.score)
                self.fake_score = copy.deepcopy(self.score)
                self.gan_score = (
                    copy.deepcopy(self.score) if config.use_gan else None
                )
                self._physical_score_modules = {
                    "score_real": self.real_score,
                    "score_fake": self.fake_score,
                    "score_critic": self.score,
                }
                if self.gan_score is not None:
                    self._physical_score_modules["score_gan"] = self.gan_score
                for name, module in self._physical_score_modules.items():
                    if name == "score_critic":
                        continue
                    for parameter in module.parameters():
                        parameter.requires_grad_(False)
            self.reward = FrozenReward(config.hidden_size) if config.use_reward else None
        self.generator.to(device=self.device, dtype=dtype)
        for module in self._physical_score_modules.values():
            module.to(device=self.device, dtype=dtype)
        if self.reward is not None:
            self.reward.to(device=self.device, dtype=dtype)
        self.generator_optimizer = torch.optim.SGD(
            [parameter for _, parameter in self.generator.trainable_named_parameters()],
            lr=config.generator_lr,
        )
        self.critic_optimizer = torch.optim.SGD(
            [parameter for _, parameter in self.score.trainable_named_parameters()],
            lr=config.critic_lr,
        )
        self.ema = GeneratorEMA(self.generator)
        self.versions = {"generator": 0, "score": 0, "ema": 0}
        self.counters: Dict[str, int] = {}
        residency_policy = ResidencyPolicy(residency_policy)
        if residency_policy == ResidencyPolicy.TIME_MULTIPLEX:
            if self.device.type != "cuda":
                raise ValueError("time-multiplex residency requires a CUDA device")
            if config.score_layout != "shared":
                raise ValueError(
                    "the time-multiplex MVP currently requires shared Score state"
                )
        self.residency_backend = ModuleResidencyBackend(
            accelerator_device=str(self.device),
            host_device="cpu",
            policy=residency_policy,
        )
        self.residency_backend.register_state(
            "generator",
            self.generator,
            aliases=("Generator",),
            modes=("default",),
            initial_device=str(self.device),
            version=0,
            before_offload=lambda: self.generator_optimizer.zero_grad(
                set_to_none=True
            ),
        )
        if config.score_layout == "shared":
            self.residency_backend.register_state(
                "score",
                self.score,
                aliases=("Score", "RealScore", "FakeScore", "Critic", "GAN"),
                modes=("real", "fake"),
                initial_device=str(self.device),
                version=0,
                before_offload=lambda: self.critic_optimizer.zero_grad(
                    set_to_none=True
                ),
            )
        else:
            self.residency_backend.register_state(
                "score_real",
                self.real_score,
                aliases=("RealScore",),
                modes=("real",),
                initial_device=str(self.device),
                version=0,
            )
            self.residency_backend.register_state(
                "score_fake",
                self.fake_score,
                aliases=("FakeScore",),
                modes=("fake",),
                initial_device=str(self.device),
                version=0,
            )
            self.residency_backend.register_state(
                "score_critic",
                self.score,
                aliases=("Score", "Critic"),
                modes=("real", "fake"),
                initial_device=str(self.device),
                version=0,
            )
            if self.gan_score is not None:
                self.residency_backend.register_state(
                    "score_gan",
                    self.gan_score,
                    aliases=("GAN",),
                    modes=("fake",),
                    initial_device=str(self.device),
                    version=0,
                )
        self.residency_backend.initialize("Generator")

    def _expected_residency_version(self, alias: str) -> int:
        if alias == "Generator":
            return self.versions["generator"]
        if self.config.score_layout == "shared":
            return self.versions["score"]
        if alias == "RealScore":
            return 0
        return self.versions["score"]

    def _resident(self, alias: str, *, mode: Optional[str] = None):
        return self.residency_backend.acquire(
            alias,
            mode=mode,
            expected_version=self._expected_residency_version(alias),
        )

    def physical_score_modules(self) -> Mapping[str, SharedScore]:
        return dict(self._physical_score_modules)

    def _clear_score_mode_transitions(self) -> None:
        for module in self._physical_score_modules.values():
            module.mode_transitions.clear()

    def _score_mode_transitions(self) -> List[str]:
        if self.config.score_layout == "shared":
            return list(self.score.mode_transitions)
        result: List[str] = []
        for name, module in self._physical_score_modules.items():
            result.extend(f"{name}:{mode}" for mode in module.mode_transitions)
        return result

    @torch.no_grad()
    def _sync_score_replicas(self) -> None:
        if self.config.score_layout == "shared":
            return
        source = dict(self.score.named_parameters())
        copied_bytes = 0
        replica_count = 0
        replicas = [self.fake_score]
        if self.gan_score is not None:
            replicas.append(self.gan_score)
        for replica in replicas:
            replica_count += 1
            for name, target in replica.named_parameters():
                if not (name.startswith("adapter.") or name.startswith("gan_head.")):
                    continue
                target.copy_(source[name])
                copied_bytes += target.numel() * target.element_size()
        self._increment("score_replica_sync", replica_count)
        self._increment("score_replica_sync_bytes", copied_bytes)

    def _increment(self, name: str, amount: int = 1) -> None:
        self.counters[name] = self.counters.get(name, 0) + amount

    def _phase(self, phase: str, **kwargs: Any):
        if self.tracer is None:
            return nullcontext()
        return self.tracer.phase(phase, **kwargs)

    def _generator_step(
        self,
        latent: Tensor,
        history: Tensor,
        conditioning: Tensor,
        section: int,
        denoise: int,
        *,
        label: str,
    ) -> Tensor:
        grad_enabled = torch.is_grad_enabled()
        self._increment(
            "generator_forward_grad" if grad_enabled else "generator_forward_no_grad"
        )
        self._increment(f"generator_forward_{label}")
        with self._phase(
            "generator",
            op="forward",
            section=section,
            denoise=denoise,
            grad_enabled=grad_enabled,
            metadata={"label": label},
        ):
            return self.generator.forward_step(
                latent, history, conditioning, section, denoise
            )

    def _rollout(
        self,
        batch: MiniBatch,
        *,
        preserve_selected_graph: bool,
        label: str,
        save_replay: bool,
    ) -> Tuple[Tensor, Optional[ReplayRecord]]:
        with self._resident("Generator", mode="default"):
            return self._rollout_impl(
                batch,
                preserve_selected_graph=preserve_selected_graph,
                label=label,
                save_replay=save_replay,
            )

    def _rollout_impl(
        self,
        batch: MiniBatch,
        *,
        preserve_selected_graph: bool,
        label: str,
        save_replay: bool,
    ) -> Tuple[Tensor, Optional[ReplayRecord]]:
        latent = batch.latent.detach()
        history = torch.zeros_like(latent)
        replay_record: Optional[ReplayRecord] = None
        selected_section = self.config.resolved_selected_section
        selected_denoise = self.config.resolved_selected_denoise
        for section in range(self.config.sections):
            for denoise in range(self.config.denoise_steps):
                selected = section == selected_section and denoise == selected_denoise
                use_grad = preserve_selected_graph and selected
                step_context = torch.enable_grad() if use_grad else torch.no_grad()
                input_latent = latent
                input_history = history
                with step_context:
                    next_latent = self._generator_step(
                        input_latent,
                        input_history,
                        batch.conditioning,
                        section,
                        denoise,
                        label=label,
                    )
                if selected and save_replay:
                    # The prefix tensors are immutable (state updates are
                    # out-of-place), so references are sufficient. Cloning here
                    # would charge replay for redundant copies that a real
                    # runtime does not need. Keeping the references still
                    # accounts for their true replay-state lifetime.
                    replay_record = ReplayRecord(
                        latent=input_latent.detach(),
                        history=input_history.detach(),
                        conditioning=batch.conditioning.detach(),
                        section=section,
                        denoise=denoise,
                        expected_output=next_latent.detach(),
                    )
                latent = next_latent
            with torch.no_grad():
                history = 0.7 * history + 0.3 * latent
                self._increment("history_update")
        return latent, replay_record

    @contextmanager
    def _freeze_score_trainables(self, module: SharedScore) -> Iterator[None]:
        parameters = [
            parameter for _, parameter in module.trainable_named_parameters()
        ]
        previous = [parameter.requires_grad for parameter in parameters]
        for parameter in parameters:
            parameter.requires_grad_(False)
        try:
            yield
        finally:
            for parameter, requires_grad in zip(parameters, previous):
                parameter.requires_grad_(requires_grad)

    def _dmd_gradient(self, fake_sample: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        with torch.no_grad():
            with self._resident("RealScore", mode="real"):
                with self._phase("real_score", op="forward", grad_enabled=False, adapter="real"):
                    real_prediction = self.real_score.predict(fake_sample.detach(), "real")
                    self._increment("score_real_forward")
            with self._resident("FakeScore", mode="fake"):
                with self._phase("fake_score", op="forward", grad_enabled=False, adapter="fake"):
                    fake_prediction = self.fake_score.predict(fake_sample.detach(), "fake")
                    self._increment("score_fake_forward")
            scale = real_prediction.detach().abs().mean().clamp_min(1.0e-3)
            gradient = (fake_prediction - real_prediction) / scale
        return gradient.detach(), real_prediction.detach(), fake_prediction.detach()

    def _generator_objective(
        self,
        batch: MiniBatch,
        strategy: GradientStrategy,
    ) -> Tuple[float, float, float, float, float, Optional[float]]:
        # Full/cache retain selected Generator state until backward. Replay
        # exposes a safe boundary at which the backend may offload Generator.
        if strategy != GradientStrategy.GENERATOR_REPLAY:
            with self._resident("Generator", mode="default"):
                return self._generator_objective_impl(batch, strategy)
        return self._generator_objective_impl(batch, strategy)

    def _generator_objective_impl(
        self,
        batch: MiniBatch,
        strategy: GradientStrategy,
    ) -> Tuple[float, float, float, float, float, Optional[float]]:
        preserve_graph = strategy != GradientStrategy.GENERATOR_REPLAY
        fake_sample, replay_record = self._rollout(
            batch,
            preserve_selected_graph=preserve_graph,
            label="generator_update",
            save_replay=strategy == GradientStrategy.GENERATOR_REPLAY,
        )
        dmd_gradient, _, _ = self._dmd_gradient(fake_sample)
        dmd_input_gradient = dmd_gradient / fake_sample.numel()
        gan_loss_value = 0.0
        replay_error: Optional[float] = None

        if strategy == GradientStrategy.FULL_GRAPH:
            if not fake_sample.requires_grad:
                raise RuntimeError("full-graph rollout did not retain the selected step")
            surrogate = (fake_sample * dmd_gradient).sum() / fake_sample.numel()
            generator_loss = surrogate
            if self.config.use_gan:
                assert self.gan_score is not None
                with self._resident("GAN", mode="fake"):
                    with self._freeze_score_trainables(self.gan_score):
                        with self._phase(
                            "gan_generator",
                            op="forward",
                            grad_enabled=True,
                            adapter="fake",
                        ):
                            gan_loss = F.softplus(
                                -self.gan_score.discriminate(fake_sample, "fake")
                            ).mean()
                            self._increment("gan_generator_forward")
                gan_loss_value = float(gan_loss.detach().cpu())
                generator_loss = generator_loss + self.config.gan_weight * gan_loss
            with self._phase("generator", op="backward", grad_enabled=True):
                generator_loss.backward()
                self._increment("generator_backward")
            generator_loss_value = float(generator_loss.detach().cpu())
            surrogate_value = float(surrogate.detach().cpu())

        else:
            boundary = fake_sample.detach().requires_grad_(self.config.use_gan)
            total_input_gradient = dmd_input_gradient
            if self.config.use_gan:
                assert self.gan_score is not None
                with self._resident("GAN", mode="fake"):
                    with self._freeze_score_trainables(self.gan_score):
                        with self._phase(
                            "gan_generator",
                            op="input_gradient",
                            grad_enabled=True,
                            adapter="fake",
                        ):
                            gan_loss = F.softplus(
                                -self.gan_score.discriminate(boundary, "fake")
                            ).mean()
                            gan_gradient = torch.autograd.grad(gan_loss, boundary)[0]
                            self._increment("gan_generator_forward")
                            self._increment("input_gradient_cache")
                gan_loss_value = float(gan_loss.detach().cpu())
                total_input_gradient = (
                    total_input_gradient + self.config.gan_weight * gan_gradient.detach()
                )
            if strategy == GradientStrategy.CACHE_INPUT_GRADIENT:
                if not fake_sample.requires_grad:
                    raise RuntimeError("cache strategy requires selected Generator activation")
                with self._phase("gradient_boundary", op="backward_cached_gradient"):
                    fake_sample.backward(total_input_gradient)
                    self._increment("generator_backward")
            elif strategy == GradientStrategy.GENERATOR_REPLAY:
                if replay_record is None:
                    raise RuntimeError("replay strategy did not save its boundary state")
                with self._resident("Generator", mode="default"):
                    with torch.enable_grad():
                        replayed = self._generator_step(
                            replay_record.latent,
                            replay_record.history,
                            replay_record.conditioning,
                            replay_record.section,
                            replay_record.denoise,
                            label="generator_replay",
                        )
                    replay_error = float(
                        (replayed.detach() - replay_record.expected_output)
                        .abs()
                        .max()
                        .cpu()
                    )
                    with self._phase("gradient_boundary", op="backward_replay"):
                        replayed.backward(total_input_gradient)
                        self._increment("generator_backward")
                        self._increment("generator_replay")
            else:  # pragma: no cover - enum makes this defensive.
                raise ValueError(f"unsupported gradient strategy: {strategy}")
            surrogate_value = float(
                ((fake_sample.detach() * dmd_gradient).sum() / fake_sample.numel()).cpu()
            )
            generator_loss_value = surrogate_value + self.config.gan_weight * gan_loss_value

        if self.config.use_reward:
            assert self.reward is not None
            with torch.no_grad():
                reward_value = float(self.reward(fake_sample.detach()).cpu())
                self._increment("reward_forward")
        else:
            reward_value = 0.0

        return (
            generator_loss_value,
            surrogate_value,
            gan_loss_value,
            float(fake_sample.detach().double().sum().cpu()),
            float(dmd_gradient.detach().double().sum().cpu()),
            replay_error,
        )

    def _critic_objective(self, batch: MiniBatch) -> float:
        fake_sample, _ = self._rollout(
            batch,
            preserve_selected_graph=False,
            label="critic_rollout",
            save_replay=False,
        )
        with self._resident("Critic"):
            with torch.no_grad():
                with self._phase("critic_real_score", op="forward", adapter="real"):
                    target = self.score.predict(fake_sample.detach(), "real")
                    self._increment("score_real_forward")
            with self._phase("critic_fake_score", op="forward", adapter="fake"):
                prediction = self.score.predict(fake_sample.detach(), "fake")
                self._increment("score_fake_forward")
                critic_loss = F.mse_loss(prediction, target.detach())
            if self.config.use_gan:
                with self._phase("gan_critic", op="forward", adapter="real/fake"):
                    real_logits = self.score.discriminate(batch.real_sample, "real")
                    fake_logits = self.score.discriminate(fake_sample.detach(), "fake")
                    discriminator_loss = (
                        F.softplus(-real_logits).mean() + F.softplus(fake_logits).mean()
                    )
                    critic_loss = (
                        critic_loss + self.config.critic_gan_weight * discriminator_loss
                    )
                    self._increment("gan_critic_forward", 2)
            with self._phase("critic", op="backward"):
                critic_loss.backward()
                self._increment("critic_backward")
        return float(critic_loss.detach().cpu())

    def run_iteration(
        self,
        strategy: GradientStrategy | str,
        batch: MiniBatch,
        *,
        iteration: int = 0,
        capture_tensors: bool = True,
    ) -> IterationResult:
        strategy = GradientStrategy(strategy)
        self.counters = {}
        self._clear_score_mode_transitions()
        generator_due = iteration % self.config.generator_update_frequency == 0
        critic_due = iteration % self.config.critic_update_frequency == 0
        ema_due = generator_due and iteration % self.config.ema_update_frequency == 0

        generator_loss: Optional[float] = None
        surrogate: Optional[float] = None
        generator_gan_loss: Optional[float] = None
        fake_checksum: Optional[float] = None
        dmd_checksum: Optional[float] = None
        replay_error: Optional[float] = None
        generator_gradients: Dict[str, Tensor] = {}
        critic_gradients: Dict[str, Tensor] = {}

        if generator_due:
            self.generator_optimizer.zero_grad(set_to_none=True)
            self.critic_optimizer.zero_grad(set_to_none=True)
            (
                generator_loss,
                surrogate,
                generator_gan_loss,
                fake_checksum,
                dmd_checksum,
                replay_error,
            ) = self._generator_objective(batch, strategy)
            if capture_tensors:
                generator_gradients = _gradient_snapshot(
                    self.generator.trainable_named_parameters()
                )
            # Score trainables are frozen/no-grad for the Generator objective.
            if any(
                parameter.grad is not None
                for _, parameter in self.score.trainable_named_parameters()
            ):
                raise RuntimeError("Generator objective polluted Critic parameter gradients")
            with self._phase("generator", op="optimizer_step", iteration=iteration):
                self.generator_optimizer.step()
                self.versions["generator"] += 1
                self.residency_backend.mark_updated(
                    "Generator", self.versions["generator"]
                )
                self._increment("generator_optimizer_step")
            if ema_due:
                with self._phase("ema", op="update", iteration=iteration):
                    self.ema.update(self.generator, self.config.ema_decay)
                    self.versions["ema"] += 1
                    self._increment("ema_update")

        critic_loss: Optional[float] = None
        if critic_due:
            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_loss = self._critic_objective(batch)
            if capture_tensors:
                critic_gradients = _gradient_snapshot(
                    self.score.trainable_named_parameters()
                )
            # Critic rollout is detached from Generator by construction.
            if any(
                parameter.grad is not None
                for _, parameter in self.generator.trainable_named_parameters()
            ):
                # Generator gradients from the earlier update are expected, but
                # must not change. They are cleared after capture to make this
                # invariant observable.
                for _, parameter in self.generator.trainable_named_parameters():
                    parameter.grad = None
            with self._phase("critic", op="optimizer_step", iteration=iteration):
                self.critic_optimizer.step()
                self.versions["score"] += 1
                self.residency_backend.mark_updated(
                    "Critic", self.versions["score"]
                )
                self._increment("critic_optimizer_step")
            if self.config.score_layout == "duplicated":
                with self._phase(
                    "score_replica_sync",
                    op="parameter_version_copy",
                    iteration=iteration,
                ):
                    self._sync_score_replicas()
                    self.residency_backend.mark_updated(
                        "FakeScore", self.versions["score"]
                    )
                    if self.gan_score is not None:
                        self.residency_backend.mark_updated(
                            "GAN", self.versions["score"]
                        )

        return IterationResult(
            iteration=iteration,
            strategy=strategy.value,
            generator_updated=generator_due,
            critic_updated=critic_due,
            ema_updated=ema_due,
            generator_loss=generator_loss,
            dmd_surrogate=surrogate,
            generator_gan_loss=generator_gan_loss,
            critic_loss=critic_loss,
            reward=(0.0 if self.config.use_reward and generator_due else None),
            replay_max_abs_error=replay_error,
            fake_checksum=fake_checksum,
            dmd_gradient_checksum=dmd_checksum,
            counters=dict(self.counters),
            score_mode_transitions=self._score_mode_transitions(),
            versions=dict(self.versions),
            generator_gradients=generator_gradients,
            critic_gradients=critic_gradients,
            generator_parameters=(
                _snapshot_named(self.generator.trainable_named_parameters())
                if capture_tensors
                else {}
            ),
            score_parameters=(
                _snapshot_named(self.score.trainable_named_parameters())
                if capture_tensors
                else {}
            ),
            ema_parameters=(
                {name: value.detach().cpu().clone() for name, value in self.ema.values.items()}
                if capture_tensors
                else {}
            ),
        )


def _module_parameter_bytes(module: nn.Module) -> int:
    return sum(parameter.numel() * parameter.element_size() for parameter in module.parameters())


def _module_trainable_bytes(module: nn.Module) -> int:
    return sum(
        parameter.numel() * parameter.element_size()
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def build_mini_execution_graph(
    config: MiniDistillConfig,
    workload: Optional[MiniDistillWorkload] = None,
) -> ExecutionGraph:
    """Build the iteration-0 branch of the logical and physical execution graph."""

    if workload is None:
        workload = MiniDistillWorkload(config)
    graph = ExecutionGraph(
        metadata={
            "workload": "structure_equivalent_mini_distill",
            "level": 3,
            "nested_axes": ["iteration", "section", "denoise", "component"],
            "config": asdict(config),
        }
    )
    generator_parameter_bytes = _module_parameter_bytes(workload.generator)
    generator_trainable_bytes = _module_trainable_bytes(workload.generator)
    graph.add_physical_state(
        PhysicalState(
            name="generator",
            parameter_bytes=generator_parameter_bytes,
            trainable_bytes=generator_trainable_bytes,
            optimizer_bytes=generator_trainable_bytes,
            modes=("default",),
            metadata={"frozen_backbone": True, "adapter": "generator_lora"},
        )
    )
    if config.score_layout == "shared":
        score_parameter_bytes = _module_parameter_bytes(workload.score)
        score_trainable_bytes = _module_trainable_bytes(workload.score)
        graph.add_physical_state(
            PhysicalState(
                name="score",
                parameter_bytes=score_parameter_bytes,
                trainable_bytes=score_trainable_bytes,
                optimizer_bytes=score_trainable_bytes,
                modes=("real", "fake"),
                metadata={
                    "frozen_backbone": True,
                    "shared_by": ["RealScore", "FakeScore", "Critic", "GAN"],
                    "adapter": "critic_lora",
                },
            )
        )
        real_score_state = fake_score_state = critic_score_state = "score"
        gan_score_state: Optional[str] = "score" if config.use_gan else None
    else:
        for state_name, module in workload.physical_score_modules().items():
            trainable_bytes = _module_trainable_bytes(module)
            graph.add_physical_state(
                PhysicalState(
                    name=state_name,
                    parameter_bytes=_module_parameter_bytes(module),
                    trainable_bytes=trainable_bytes,
                    optimizer_bytes=(
                        trainable_bytes if state_name == "score_critic" else 0
                    ),
                    modes=("real", "fake"),
                    metadata={
                        "frozen_backbone": True,
                        "logical_model_copy": True,
                        "optimizer_owner": state_name == "score_critic",
                    },
                )
            )
        real_score_state = "score_real"
        fake_score_state = "score_fake"
        critic_score_state = "score_critic"
        gan_score_state = "score_gan" if config.use_gan else None
    graph.add_physical_state(
        PhysicalState(
            name="generator_ema",
            parameter_bytes=generator_trainable_bytes,
            ema_bytes=generator_trainable_bytes,
            modes=("sharded",),
            metadata={"owner": "Generator"},
        )
    )

    previous: Optional[str] = None
    for section in range(config.sections):
        for denoise in range(config.denoise_steps):
            selected = (
                section == config.resolved_selected_section
                and denoise == config.resolved_selected_denoise
            )
            name = f"generator_update_s{section}_d{denoise}"
            graph.add_node(
                OperationNode(
                    name=name,
                    logical_component="Generator",
                    physical_state="generator",
                    operation_type=OperationType.FORWARD,
                    grad_mode=GradMode.GRAD if selected else GradMode.NO_GRAD,
                    adapter_mode="default",
                    update_frequency=config.generator_update_frequency,
                    condition="generator_update_due",
                    nested_position=(section, denoise),
                    metadata={
                        "selected_gradient_position": selected,
                        "continuous_state": "latent",
                    },
                )
            )
            if previous is not None:
                graph.add_edge(
                    Edge(previous, name, EdgeType.STATE, name="continuous_latent")
                )
            previous = name
        history_name = f"history_update_s{section}"
        graph.add_node(
            OperationNode(
                name=history_name,
                logical_component="HistoryState",
                physical_state=None,
                operation_type=OperationType.FORWARD,
                grad_mode=GradMode.NO_GRAD,
                update_frequency=config.generator_update_frequency,
                condition="generator_update_due",
                nested_position=(section,),
                metadata={"continuous_state": "history"},
            )
        )
        assert previous is not None
        graph.add_edge(Edge(previous, history_name, EdgeType.STATE, name="history"))
        previous = history_name

    selected_node = (
        f"generator_update_s{config.resolved_selected_section}_"
        f"d{config.resolved_selected_denoise}"
    )
    score_nodes = [
        OperationNode(
            "real_score",
            "RealScore",
            real_score_state,
            OperationType.FORWARD,
            GradMode.NO_GRAD,
            adapter_mode="real",
            update_frequency=config.generator_update_frequency,
            condition="generator_update_due",
        ),
        OperationNode(
            "fake_score_dmd",
            "FakeScore",
            fake_score_state,
            OperationType.FORWARD,
            GradMode.NO_GRAD,
            adapter_mode="fake",
            update_frequency=config.generator_update_frequency,
            condition="generator_update_due",
        ),
    ]
    if config.use_gan:
        assert gan_score_state is not None
        score_nodes.append(
            OperationNode(
                "gan_generator",
                "GAN",
                gan_score_state,
                OperationType.FORWARD,
                GradMode.GRAD,
                adapter_mode="fake",
                update_frequency=config.generator_update_frequency,
                condition="generator_update_due and gan_enabled",
            )
        )
    for node in score_nodes:
        graph.add_node(node)
        graph.add_edge(
            Edge(selected_node, node.name, EdgeType.DATA, name="x_fake")
        )

    graph.add_node(
        OperationNode(
            "generator_backward",
            "Generator",
            "generator",
            OperationType.BACKWARD,
            GradMode.EXTERNAL_GRAD,
            adapter_mode="default",
            update_frequency=config.generator_update_frequency,
            condition="generator_update_due",
            metadata={"gradient_boundary": True},
        )
    )
    graph.add_edge(
        Edge("real_score", "generator_backward", EdgeType.GRADIENT, name="dmd_dx")
    )
    graph.add_edge(
        Edge("fake_score_dmd", "generator_backward", EdgeType.GRADIENT, name="dmd_dx")
    )
    if config.use_gan:
        graph.add_edge(
            Edge("gan_generator", "generator_backward", EdgeType.GRADIENT, name="gan_dx")
        )
    graph.add_node(
        OperationNode(
            "generator_optimizer_step",
            "Generator",
            "generator",
            OperationType.OPTIMIZER_STEP,
            GradMode.NO_GRAD,
            adapter_mode="default",
            update_frequency=config.generator_update_frequency,
            condition="generator_update_due",
            parameter_version="generator:v+1",
        )
    )
    graph.add_edge(
        Edge("generator_backward", "generator_optimizer_step", EdgeType.VERSION)
    )
    graph.add_node(
        OperationNode(
            "ema_update",
            "EMA",
            "generator_ema",
            OperationType.EMA_UPDATE,
            GradMode.NO_GRAD,
            adapter_mode="sharded",
            update_frequency=config.ema_update_frequency,
            condition="generator_update_due and ema_update_due",
            parameter_version="ema:v+1",
        )
    )
    graph.add_edge(
        Edge("generator_optimizer_step", "ema_update", EdgeType.VERSION)
    )

    # Critic sees a fresh no-grad rollout after the Generator parameter update.
    previous = "ema_update"
    for section in range(config.sections):
        for denoise in range(config.denoise_steps):
            name = f"critic_rollout_s{section}_d{denoise}"
            graph.add_node(
                OperationNode(
                    name,
                    "Generator",
                    "generator",
                    OperationType.FORWARD,
                    GradMode.NO_GRAD,
                    adapter_mode="default",
                    update_frequency=config.critic_update_frequency,
                    condition="critic_update_due",
                    nested_position=(section, denoise),
                    parameter_version="generator:v+1",
                )
            )
            graph.add_edge(Edge(previous, name, EdgeType.STATE, name="continuous_latent"))
            previous = name
    graph.add_node(
        OperationNode(
            "critic_real_score",
            "Critic",
            critic_score_state,
            OperationType.FORWARD,
            GradMode.NO_GRAD,
            adapter_mode="real",
            update_frequency=config.critic_update_frequency,
            condition="critic_update_due",
        )
    )
    graph.add_node(
        OperationNode(
            "critic_fake_score",
            "Critic",
            critic_score_state,
            OperationType.FORWARD,
            GradMode.GRAD,
            adapter_mode="fake",
            update_frequency=config.critic_update_frequency,
            condition="critic_update_due",
        )
    )
    graph.add_edge(Edge(previous, "critic_real_score", EdgeType.DATA, name="x_fake"))
    graph.add_edge(Edge(previous, "critic_fake_score", EdgeType.DATA, name="x_fake"))
    graph.add_node(
        OperationNode(
            "critic_backward",
            "Critic",
            critic_score_state,
            OperationType.BACKWARD,
            GradMode.GRAD,
            adapter_mode="fake",
            update_frequency=config.critic_update_frequency,
            condition="critic_update_due",
        )
    )
    graph.add_edge(
        Edge("critic_real_score", "critic_backward", EdgeType.DATA, name="target")
    )
    graph.add_edge(
        Edge("critic_fake_score", "critic_backward", EdgeType.GRADIENT, name="critic_loss")
    )
    graph.add_node(
        OperationNode(
            "critic_optimizer_step",
            "Critic",
            critic_score_state,
            OperationType.OPTIMIZER_STEP,
            GradMode.NO_GRAD,
            adapter_mode="fake",
            update_frequency=config.critic_update_frequency,
            condition="critic_update_due",
            parameter_version="score:v+1",
        )
    )
    graph.add_edge(Edge("critic_backward", "critic_optimizer_step", EdgeType.VERSION))
    if config.score_layout == "duplicated":
        sync_targets = [("score_fake", workload.fake_score)]
        if config.use_gan:
            assert workload.gan_score is not None
            sync_targets.append(("score_gan", workload.gan_score))
        for state_name, module in sync_targets:
            sync_bytes = sum(
                parameter.numel() * parameter.element_size()
                for name, parameter in module.named_parameters()
                if name.startswith("adapter.") or name.startswith("gan_head.")
            )
            node_name = f"sync_{state_name}"
            graph.add_node(
                OperationNode(
                    node_name,
                    "StateSync",
                    state_name,
                    OperationType.TRANSFER,
                    GradMode.NO_GRAD,
                    adapter_mode="fake",
                    update_frequency=config.critic_update_frequency,
                    condition="critic_update_due and duplicated_score_layout",
                    parameter_version="score:v+1",
                    metadata={"transfer_bytes": sync_bytes},
                )
            )
            graph.add_edge(
                Edge(
                    "critic_optimizer_step",
                    node_name,
                    EdgeType.VERSION,
                    tensor_bytes=sync_bytes,
                    name="adapter_parameter_sync",
                )
            )
    graph.validate()
    return graph
