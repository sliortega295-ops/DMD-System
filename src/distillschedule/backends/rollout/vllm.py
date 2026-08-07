"""vLLM rollout facade without importing vLLM into the core package."""

from __future__ import annotations

from .base import ExternalRolloutProvider, RemoteRolloutClient
from distillschedule.core.enums import RolloutEngineKind


class VLLMRolloutProvider(ExternalRolloutProvider):
    engine = RolloutEngineKind.VLLM

    def __init__(self, client: RemoteRolloutClient, *, max_version_staleness: int = 0) -> None:
        super().__init__(client, max_version_staleness=max_version_staleness)
