"""Validated mechanism compatibility layer for the DistillSchedule refactor.

The stable public API lives under :mod:`distillschedule`.  These modules are
kept importable while real Helios operations migrate behind typed adapters.
"""

from .activation_offload import (
    ActivationOffloadRegistration,
    AuditedSavedTensorOffloader,
    install_activation_offload_hooks,
)
from .dmd_boundary import (
    critic_denoising_loss,
    detached_dmd_surrogate_loss,
    dmd_input_gradient,
    flow_prediction_to_x0,
)
from .ema import TrainableStateEMA
from .frequency import Stage3FrequencyStep, stage3_frequency_step
from .host_admission import (
    HostAdmission,
    MinimumSharePlan,
    SharedBlockCandidate,
    evaluate_host_admission,
    plan_minimum_shared_blocks,
)
from .residency import ModuleResidencyBackend, ResidencyPolicy

__all__ = [
    "ActivationOffloadRegistration",
    "AuditedSavedTensorOffloader",
    "HostAdmission",
    "MinimumSharePlan",
    "ModuleResidencyBackend",
    "ResidencyPolicy",
    "SharedBlockCandidate",
    "Stage3FrequencyStep",
    "TrainableStateEMA",
    "critic_denoising_loss",
    "detached_dmd_surrogate_loss",
    "dmd_input_gradient",
    "evaluate_host_admission",
    "flow_prediction_to_x0",
    "install_activation_offload_hooks",
    "plan_minimum_shared_blocks",
    "stage3_frequency_step",
]
