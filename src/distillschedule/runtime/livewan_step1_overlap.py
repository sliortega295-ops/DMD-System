"""Default-off C36-parent step1 scheduling; no kernel or optimizer changes."""
from typing import Any, Mapping

VARIANT = "c37-continuous-b16-step1-overlap"
SCHEDULE = "wave0_b1_backward_then_wave1_b1_backward_single_commit"
PLAN_CONTRACT = {
    "enabled": True,
    "name": "c36_parent_step1_early_b1_backward_v1",
    "parent": "c36-continuous-b16-score-layernorm",
    "scope": "score_step1_only",
    "global_batch": 16,
    "loss_weights": [0.5, 0.5],
    "physical_calls_by_step": [1, 2, 1, 1, 1],
    "single_reduction_and_commit_after_both_waves": True,
    "other_steps_and_all_parent_kernels_unchanged": True,
    "numerical_policy_unchanged": True,
}


def runtime_receipt_errors(
    continuous: Mapping[str, Any], *, rank: int, cycles: int,
    inventory: Mapping[str, int],
) -> list[str]:
    """Require actual application counters, not just planned physical calls."""
    errors = []
    if continuous.get("critic_step1_physical_schedule") != SCHEDULE:
        errors.append("C37 early-wave schedule not executed")
    if continuous.get("critic_step1_single_commit") is not True:
        errors.append("C37 lost its single commit boundary")
    if continuous.get("critic_physical_batch_size") != "mixed_step1_b1_other_b2":
        errors.append("C37 physical batch receipt mislabels mixed B1/B2")
    if continuous.get("critic_physical_backward_calls_per_step") != [1, 2, 1, 1, 1]:
        errors.append("C37 physical per-step schedule changed")
    expected = [[1, 2, 1, 1, 1] for _ in range(cycles)] if rank >= 8 else []
    if continuous.get("critic_physical_calls_by_cycle") != expected:
        errors.append("C37 observed per-step Critic forward census changed")
    actual = int(inventory.get("critic/grad/checkpoint", 0))
    if actual != (6 * cycles if rank >= 8 else 0):
        errors.append("C37 observed physical Critic total changed")
    return errors
