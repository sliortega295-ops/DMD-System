"""Eight rank-qualified trace views per DP replica, not eight new samples.

Only external loss inputs are stacked. Native host samplers must consume
their individual traces through the cooperative B1 dispatcher; a single
control dictionary cannot represent rank-local history/schedule choices.
"""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

from .fsdp2_p0 import _is_shared_index_key, _share_identical_tensors, _stack_tensors


class StaticCohortOperationTrace:
    def __init__(self, members, *, cohort_size=8):
        if type(cohort_size) is not int or cohort_size not in (1, 2, 8) or len(members) != cohort_size:
            raise ValueError('static PP2 group needs its admitted independent trace members')
        self.members = tuple(members)
        self.manifest = dict(members[0].operation_trace.manifest)
        self.root = members[0].operation_trace.root
        self.trace_hash = hashlib.sha256(json.dumps(
            [m.operation_trace.trace_hash for m in members]).encode()).hexdigest()

    def operation(self, key):
        values = [member.operation_trace.operation(key) for member in self.members]
        if any(v != values[0] for v in values[1:]):
            raise ValueError('native host control is not identical across cohort: '+key)
        return values[0]

    def tensor(self, key, *, clone=True):
        values = [member.operation_trace.tensor(key, clone=clone) for member in self.members]
        return (_share_identical_tensors(values, label=key) if _is_shared_index_key(key)
                else _stack_tensors(values))


def cohort_entry(members, *, role, cohort_size=8):
    if (role not in ('generator', 'critic') or type(cohort_size) is not int
            or cohort_size not in (1, 2, 8) or len(members) != cohort_size):
        raise ValueError('requires the admitted exact same-operation rank-qualified entries')
    members = tuple(members)
    step = (members[0].logical_step if role == 'generator' else members[0].pool_entry.logical_step)
    ranks = [m.source_id if role == 'generator' else m.pool_entry.source_trace_rank for m in members]
    if len(set(ranks)) != cohort_size:
        raise ValueError('TP/PP collaborators must not duplicate logical samples')
    if any((m.logical_step if role == 'generator' else m.pool_entry.logical_step) != step for m in members):
        raise ValueError('different logical updates cannot be combined')
    tensor_names = tuple(members[0].tensors)
    if any(tuple(m.tensors) != tensor_names for m in members):
        raise ValueError('different tensor inventory across logical rank samples')
    tensors = {n: (_share_identical_tensors([m.tensors[n] for m in members], label=n)
                  if n.startswith('indices_') else _stack_tensors([m.tensors[n] for m in members]))
               for n in tensor_names}
    identity = hashlib.sha256(json.dumps([m.identity_sha256 if role == 'generator'
                else m.pool_entry.identity_sha256 for m in members]).encode()).hexdigest()
    trace = StaticCohortOperationTrace(members, cohort_size=cohort_size)
    # The native samplers have already consumed each member's original
    # control. Do not substitute the first member's rank-local decisions.
    return SimpleNamespace(tensors=tensors, operation_trace=trace, logical_step=step,
        source_id=ranks[0], identity={'source_rank': ranks[0], 'logical_source_ranks': ranks},
        identity_sha256=identity, rollout_operation=f'step/{step}/{role}/rollout', members=members,
        pool_entry=SimpleNamespace(logical_step=step, source_trace_rank=ranks[0],
            logical_source_ranks=ranks, identity_sha256=identity))
