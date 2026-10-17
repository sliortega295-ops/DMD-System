"""Opt-in scratch sharing for independent, cloned-output no-grad graphs only."""
from __future__ import annotations

import os
import threading

ENV = "DISTILLSCHEDULE_WAN13B_GENERATOR_GRAPH_POOL"
ISOLATED = "isolated.v1"
SHARED = "shared-scratch.v1"


def requested_policy():
    policy = os.environ.get(ENV, ISOLATED)
    if policy not in (ISOLATED, SHARED):
        raise ValueError("unsupported Generator graph pool policy")
    return policy


class IndependentRolloutPool:
    def __init__(self):
        self.policy = requested_policy()
        self.stream = self.pool = self.replay_identity = None
        self.receipt = dict(policy=self.policy, verified_capture_count=0,
                            scope="GENERATOR_NOGRAD_SCRATCH_ONLY",
                            critic_grad_pool_shared=False,
                            outputs_freshly_cloned=True)

    def before_call(self, torch):
        if self.policy == ISOLATED:
            return
        if torch.is_grad_enabled() or torch.cuda.is_current_stream_capturing():
            raise ValueError("shared rollout pool requires noncapturing no-grad caller")
        stream = torch.cuda.current_stream()
        identity = (threading.get_ident(), str(stream.device), stream.cuda_stream)
        if self.replay_identity is None:
            self.replay_identity = identity
        if identity != self.replay_identity:
            raise ValueError("shared rollout graphs cannot replay on another stream/thread")

    def capture_options(self, torch):
        if self.policy == ISOLATED:
            return {}
        if self.replay_identity is None:
            raise ValueError("shared graph capture must follow admitted no-grad call")
        if self.pool is None:
            self.pool = torch.cuda.graph_pool_handle()
            self.stream = torch.cuda.Stream()
        return dict(pool=self.pool, stream=self.stream)

    def verify_capture(self, graph):
        if self.policy == SHARED:
            if self.pool is None or graph.pool() != self.pool:
                raise ValueError("Generator capture did not use admitted shared pool")
            self.receipt["verified_capture_count"] += 1

