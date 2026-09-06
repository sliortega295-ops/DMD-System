"""Cross-workload evidence schemas and offline comparison helpers."""

from .trajectory_coarse import (
    COARSE_50C_THRESHOLDS,
    SCHEMA_VERSION,
    AppendOnlyTrajectoryWriter,
    TrajectoryCycleReceipt,
    TrajectoryStateComparison,
    TrajectoryThresholds,
    compare_trajectories,
    load_trajectory_jsonl,
    write_trajectory_report,
)

__all__ = [
    "COARSE_50C_THRESHOLDS",
    "SCHEMA_VERSION",
    "AppendOnlyTrajectoryWriter",
    "TrajectoryCycleReceipt",
    "TrajectoryStateComparison",
    "TrajectoryThresholds",
    "compare_trajectories",
    "load_trajectory_jsonl",
    "write_trajectory_report",
]
