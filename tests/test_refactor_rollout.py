from __future__ import annotations

import pytest

from distillschedule.backends.rollout import (
    ModelVersion,
    NativeRolloutProvider,
    RolloutArtifact,
    RolloutRequest,
    WeightSyncRequest,
)
from distillschedule.core.contracts import ContractViolation


def test_native_rollout_requires_exact_synchronized_version() -> None:
    def rollout(request: RolloutRequest) -> RolloutArtifact:
        return RolloutArtifact(
            request_id=request.request_id,
            source_version=request.required_version,
            samples={"ok": True},
        )

    provider = NativeRolloutProvider(rollout, max_version_staleness=0)
    version = ModelVersion("generator", 3, "abc")
    with pytest.raises(ContractViolation, match="no synchronized model version"):
        provider.rollout(RolloutRequest("r0", version, payload={}))
    provider.synchronize(WeightSyncRequest(version=version))
    artifact = provider.rollout(RolloutRequest("r1", version, payload={}))
    assert artifact.samples == {"ok": True}


def test_stale_rollout_is_rejected_by_default() -> None:
    def rollout(request: RolloutRequest) -> RolloutArtifact:
        current = ModelVersion("generator", 2)
        return RolloutArtifact(request.request_id, current, samples=None)

    provider = NativeRolloutProvider(rollout)
    provider.synchronize(WeightSyncRequest(ModelVersion("generator", 2)))
    with pytest.raises(ContractViolation, match="stale by 1"):
        provider.rollout(RolloutRequest("r", ModelVersion("generator", 3), payload=None))
