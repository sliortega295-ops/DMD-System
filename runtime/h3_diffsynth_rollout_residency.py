"""Default-off, call-local parameter reuse for the pinned native AR8 rollout.

Only selected main-block FSDP groups change residency. Root/refiner groups,
checkpointing, adapter selection, numerical policy and native rollout stay intact.
Public FSDP APIs mutate placement; private state is read only for fail-closed
lifecycle checks. Collective interception is correctness-only instrumentation.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import functools


class NativeNoGradParameterReuse:
    def __init__(self, dmd, adapter, *, retained_main_blocks=0, logical_slot=None,
                 audit_collectives=False, module_type=None, torch_module=None):
        self.enabled = retained_main_blocks > 0
        self.records = []
        self.retained_main_blocks = retained_main_blocks
        self.audit_collectives = audit_collectives
        self._dmd = dmd
        self._original = None
        if not self.enabled and not audit_collectives:
            return  # Default policy must not import Torch or patch native code.
        if torch_module is None:
            import torch as torch_module
        self.torch = torch_module
        if module_type is None:
            from torch.distributed.fsdp import FSDPModule as module_type
        blocks = list(adapter.dit.blocks)
        if len(blocks) not in (2, 50) or not 0 <= retained_main_blocks <= min(40, len(blocks)):
            raise ValueError('BOUNDED_H3_MAIN_BLOCK_RETENTION_ONLY')
        if any(not isinstance(b, module_type) for b in blocks):
            raise ValueError('ALL_H3_MAIN_BLOCKS_MUST_REMAIN_FSDP')
        self.groups = [b._get_fsdp_state()._fsdp_param_group for b in blocks]
        if any(g is None for g in self.groups) or len(set(map(id, self.groups))) != len(blocks):
            raise ValueError('ONE_UNIQUE_FSDP_GROUP_PER_MAIN_BLOCK_REQUIRED')
        self.blocks = blocks[:retained_main_blocks]
        self.selected_groups = self.groups[:retained_main_blocks]
        self.parameter_bytes = sum(p.numel() * p.element_size()
                                   for b in self.blocks for p in b.parameters())
        original = dmd.rollout
        self._original = original

        @functools.wraps(original)
        def rollout(a, real, inputs, schedule, selected_step=None):
            if a is not adapter or len(schedule[0]) != 9 or selected_step is None:
                raise RuntimeError('UNCHANGED_NATIVE_STUDENT_AR8_ROLLOUT_REQUIRED')
            slot = logical_slot() if logical_slot is not None else None
            if slot is None:
                raise RuntimeError('EXPLICIT_LOGICAL_SLOT_REQUIRED')
            # The native function is itself @no_grad; extend that existing
            # scope only over the placement setup/cleanup, never a grad call.
            with self.torch.no_grad(), self.scope(slot):
                result = original(a, real, inputs, schedule, selected_step)
                # Native rollout itself selects Student, possibly from a
                # preceding Teacher/Fake role. Do not move that transition.
                if any(layer.role != 'student' for _, layer in adapter.layers):
                    raise RuntimeError('NATIVE_ROLLOUT_RETURNED_WRONG_ADAPTER_ROLE')
                return result
        self._wrapper = rollout
        dmd.rollout = rollout

    @contextmanager
    def _collective_audit(self, counts):
        if not self.audit_collectives:
            yield
            return
        import torch.distributed.fsdp._fully_shard._fsdp_param_group as implementation
        if not self.torch.__version__.startswith('2.10.'):
            raise RuntimeError('FSDP_DIAGNOSTIC_HOOK_REQUIRES_AUDITED_TORCH_2_10')
        original = implementation.foreach_all_gather
        group_keys = {frozenset(map(id, g.fsdp_params)): i for i, g in enumerate(self.groups)}
        @functools.wraps(original)
        def issued(params, *args, **kwargs):
            index = group_keys.get(frozenset(map(id, params)))
            if index is not None:
                counts[index] += 1
            return original(params, *args, **kwargs)
        implementation.foreach_all_gather = issued
        try:
            yield
        finally:
            if implementation.foreach_all_gather is not issued:
                raise RuntimeError('COLLECTIVE_AUDIT_HOOK_OWNERSHIP_LOST')
            implementation.foreach_all_gather = original

    @contextmanager
    def scope(self, slot):
        if self.torch.is_grad_enabled():
            raise RuntimeError('RETENTION_MUST_NOT_CROSS_GRAD_PATH')
        if any(g.is_unsharded or not g._reshard_after_forward for g in self.selected_groups):
            raise RuntimeError('SELECTED_MAIN_GROUP_ENTERED_WITH_UNEXPECTED_POLICY')
        row = {'logical_slot': slot, 'retained_main_blocks': len(self.blocks),
               'retained_full_parameter_bytes': self.parameter_bytes,
               'collective_audit_enabled': self.audit_collectives,
               'allgather_issuances_by_main_block': None, 'restored': False}
        counts = Counter()
        completed = False
        try:
            for b in self.blocks:
                b.set_reshard_after_forward(False, recurse=False)
            with self._collective_audit(counts):
                yield row
            completed = True
        finally:
            row['unsharded_before_exit'] = sum(g.is_unsharded for g in self.selected_groups)
            for b in self.blocks:
                b.set_reshard_after_forward(True, recurse=False)
            for b in reversed(self.blocks):
                b.reshard()
            row['unsharded_after_exit'] = sum(g.is_unsharded for g in self.selected_groups)
            row['restored'] = (row['unsharded_after_exit'] == 0 and
                               all(g._reshard_after_forward for g in self.selected_groups))
            if self.audit_collectives:
                row['allgather_issuances_by_main_block'] = [counts[i] for i in range(len(self.groups))]
            self.records.append(row)
            if not row['restored']:
                raise RuntimeError('SELECTED_MAIN_PARAMETERS_NOT_RESHARDED_BEFORE_TRAINING')
            if completed and row['unsharded_before_exit'] != len(self.blocks):
                raise RuntimeError('NATIVE_ROLLOUT_DID_NOT_MATERIALIZE_ALL_SELECTED_UNITS')

    def finish_cycle(self):
        if not self.enabled and not self.audit_collectives:
            return {'enabled': False, 'status': 'DEFAULT_OFF_NO_PATCH'}
        if len(self.records) != 2 or len({r['logical_slot'] for r in self.records}) != 2:
            raise RuntimeError('TWO_INDEPENDENT_LOGICAL_ROLLOUTS_REQUIRED')
        if not all(r['restored'] for r in self.records):
            raise RuntimeError('UNRESTORED_ROLLOUT_SCOPE')
        return {'enabled': self.enabled, 'records': list(self.records),
                'root_refiners_unchanged': True, 'early_unshard': False}

    def begin_next_cycle(self):
        self.records.clear()

    def close(self):
        if self._original is not None:
            if self._dmd.rollout is not self._wrapper:
                raise RuntimeError('ROLLOUT_WRAPPER_OWNERSHIP_LOST')
            self._dmd.rollout = self._original
