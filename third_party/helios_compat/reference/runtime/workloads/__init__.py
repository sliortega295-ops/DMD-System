"""Structure-equivalent workloads used to validate DistillSchedule."""

from .mini_distill import (
    MiniBatch,
    MiniDistillConfig,
    MiniDistillWorkload,
    build_mini_execution_graph,
)

from .split_distill import (
    SplitGeneratorRole,
    SplitScoreRole,
)
from .image_dmd2 import (
    DMD2_SOURCE_COMMIT,
    ImageDMD2Batch,
    ImageDMD2Config,
    ImageDMD2GeneratorRole,
    ImageDMD2GuidanceRole,
    build_image_dmd2_execution_graph,
)

__all__ = [
    "MiniBatch",
    "MiniDistillConfig",
    "MiniDistillWorkload",
    "SplitGeneratorRole",
    "SplitScoreRole",
    "DMD2_SOURCE_COMMIT",
    "ImageDMD2Batch",
    "ImageDMD2Config",
    "ImageDMD2GeneratorRole",
    "ImageDMD2GuidanceRole",
    "build_image_dmd2_execution_graph",
    "build_mini_execution_graph",
]
