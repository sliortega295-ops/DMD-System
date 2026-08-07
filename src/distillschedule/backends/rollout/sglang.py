"""SGLang rollout facade without importing SGLang into the core package."""

from __future__ import annotations

from .base import ExternalRolloutProvider, RemoteRolloutClient
from distillschedule.core.enums import RolloutEngineKind


class SGLangRolloutProvider(ExternalRolloutProvider):
    engine = RolloutEngineKind.SGLANG

    def __init__(self, client: RemoteRolloutClient, *, max_version_staleness: int = 0) -> None:
        super().__init__(client, max_version_staleness=max_version_staleness)
