from __future__ import annotations

import pytest

from distillschedule.runtime.gan_rolephase_transition import (
    RolePhaseTransitionContract,
    transition_contract_summary,
)


def test_rolephase_v1_uses_proven_pinned_transition_not_gpu_live_reshard() -> None:
    contract = RolePhaseTransitionContract()
    assert contract.backend == "pr6_persistent_pinned_target_materialization"
    assert contract.target_preconstructed_before_timing is True
    assert contract.target_host_state_pinned_before_timing is True
    assert contract.steady_state_repeated_cpu_streaming is False
    assert contract.forward_handoff_timed is True
    assert contract.reverse_handoff_timed is True
    assert contract.uses_gpu_live_reshard is False
    assert contract.requires_locality_aware_reshard_before_first_gpu_run is False


def test_transition_contract_rejects_hidden_or_repeated_costs() -> None:
    with pytest.raises(ValueError, match="stable phases"):
        RolePhaseTransitionContract(steady_state_repeated_cpu_streaming=True)
    with pytest.raises(ValueError, match="both role transitions"):
        RolePhaseTransitionContract(reverse_handoff_timed=False)
    with pytest.raises(ValueError, match="target model construction"):
        RolePhaseTransitionContract(target_preconstructed_before_timing=False)


def test_transition_summary_is_claim_explicit() -> None:
    summary = transition_contract_summary()
    assert summary["uses_gpu_live_reshard"] is False
    assert summary["steady_state_repeated_cpu_streaming"] is False
