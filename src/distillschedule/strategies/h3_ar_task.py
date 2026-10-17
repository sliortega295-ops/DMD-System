"""Bounded adapter for W6's ORIGINAL DiffSynth previous/current AV DMD2.

This is not W5 Grid1000, a full-sequence replacement, or the new Euler task.
The first adapter freezes the measured workload and only enumerates two native
implementations. A changed shape/update ratio needs a new contract and costs.
"""
from dataclasses import asdict, dataclass
from typing import Mapping

from distillschedule.core.contracts import require
from distillschedule.core.enums import GradientMode, OperationKind
from distillschedule.core.graph import ExecutionGraph, Operation
from distillschedule.workloads.best_known.registry import fingerprint

SCHEMA = 'distillschedule.original_h3_ar_task.v1'
SOURCE_RECEIPT = '9ca4ad8ca13b0a9e7397f6313e4eca225f10f52802e90423b06f7e786a58b989'
SOURCE_PINS = {
    'examples/minimax_h3/model_training/h3_dmd2.py': '77744f5ec391be813d42ebaf6a2692c18ea58f5ca7fb41398fd287d2a177ca62',
    'examples/minimax_h3/model_training/train_dmd2.py': '1bd881e755687218c982adea11cc4b430e2b858d60a59a35d8da42f25ce674c5',
    'examples/minimax_h3/model_training/dmd2_zero3.py': 'c8b7beb4375c26daae9e8e0dd863fb8a9079e644c17e690d712d45f14c3bab7b',
    'diffsynth/models/minimax_h3_dit.py': 'bd617a4312c70405e916707e1908100b484b2259dfbeba4af7de5bb145206d80',
}


@dataclass(frozen=True)
class H3ARTask:
    schema_version: str = SCHEMA
    protocol: str = 'diffsynth_native_previous_current_av_dmd2_v1'
    video_frames: int = 56
    previous_frames: int = 56
    video_height: int = 768
    video_width: int = 1344
    fps: int = 24
    denoising_steps: int = 8
    fake_updates: int = 1
    ema_updates: int = 0
    lora_rank: int = 64
    world_size: int = 16
    physical_batch: int = 1
    precision: str = 'native_bf16_autocast_cache_disabled'

    def __post_init__(self):
        defaults = {k: f.default for k, f in self.__dataclass_fields__.items()}
        require(asdict(self) == defaults and all(type(getattr(self, k)) is type(v) for k, v in defaults.items()),
                'W6 frozen native contract changed; no Grid/fullseq/precision/update substitution', path='h3_ar.task')

    @property
    def census(self):
        return dict(student=9, fake_score=2, teacher=1, student_grad=1,
                    fake_score_grad=1, student_backward=1, critic_backward=1)

    @property
    def input_shapes(self):
        # Original cache + native README, not Wan VAE compression inference.
        return dict(video=[1, 24, 17, 48, 84], audio=[2, 32, 93],
                    previous_video=[17136, 96], previous_audio=[186, 32])

    @property
    def digest(self):
        return fingerprint(self.to_dict())

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        require(isinstance(value, Mapping) and set(value) <= set(cls.__dataclass_fields__),
                'unknown H3 AR task field', path='h3_ar.task')
        return cls(**value)


@dataclass(frozen=True)
class H3ARImplementation:
    id: str

    def __post_init__(self):
        require(self.id in ('zero3-native', 'zero3-ap'), 'unadmitted H3 AR implementation', path='h3_ar.implementation')

    @property
    def optimized(self):
        return self.id == 'zero3-ap'

    def to_dict(self):
        return dict(id=self.id, zero_stage=3, parameter_offload=False, optimizer_offload=False,
                    checkpoint_boundary_offload=False, selected_cache_prefetch=self.optimized,
                    attention_replay_cache_storage='cpu_staged' if self.optimized else 'none',
                    attention_replay_cache_transfer='native_D2H_H2D_with_bounded_prefetch' if self.optimized else 'none',
                    unit_prefetch=self.optimized, pointwise_fusion=self.optimized,
                    lora_inplace=self.optimized, swiglu_fusion=self.optimized,
                    rms_fusion=self.optimized, cutlass_lora=self.optimized,
                    overlap_comm=self.optimized,
                    pinned_allocator='expandable_segments_host_register4' if self.optimized else 'native')

    @classmethod
    def from_dict(cls, value):
        require(isinstance(value, Mapping) and isinstance(value.get('id'), str),
                'invalid H3 AR implementation', path='h3_ar.implementation')
        result = cls(value['id'])
        require(value == result.to_dict(), 'partial/changed native bundle is not calibrated', path='h3_ar.implementation')
        return result


def native_graph(task, *, cycle=2):
    require(type(cycle) is int and cycle in (1, 2), 'only two native updates admitted', path='h3_ar.graph')
    before = dict(student=cycle-1, fake=cycle-1, ema=0)
    after = dict(student=cycle, fake=cycle, ema=0)
    op = Operation('native_update', cycle, 'native_runtime', OperationKind.CALL,
                   GradientMode.GRAD, required_versions=before,
                   produced_versions={k: v for k, v in after.items() if k != 'ema'},
                   metadata={'task_sha256': task.digest, 'call_census': task.census})
    return ExecutionGraph((op,), (), metadata=dict(initial_versions=before, final_versions=after,
        scope='WHOLE_ORIGINAL_NATIVE_UPDATE_NO_INTER_ROLE_OVERLAP_MODEL', task_sha256=task.digest))


def native_config():
    # Exact native JSON plus the source dataclass defaults are pinned separately.
    return dict(num_inference_steps=8, video_shift=2.22, audio_shift=2.22, critic_updates=1,
        student_learning_rate=1e-5, fake_score_learning_rate=1e-5, discriminator_learning_rate=4e-7,
        lora_rank=64, lora_alpha=64, lora_targets='attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2',
        audio_loss_weight=0.5, teacher_cfg_scale=1.0, adversarial_enabled=True, gan_start_step=500,
        generator_loss_weight=0.05, discriminator_loss_weight=0.01, gradient_checkpointing=True)
