"""Source-generated parameter-grid admission, canonical trace defaults intact."""
from __future__ import annotations

import hashlib
import json

from distillschedule.runtime.wan13b_parameter_grid import WanParameterGridSpec
from .static_world16_trace import StaticWorld16Trace


class WanParameterGridTrace(StaticWorld16Trace):
    def __init__(self, root, *, workload, execution=None):
        if not isinstance(workload, WanParameterGridSpec):
            raise TypeError('explicit versioned grid workload required')
        self.workload = workload
        super().__init__(root, variant='no-gan', execution=execution)
        self.binding_hash = hashlib.sha256(json.dumps(
            dict(schema='dmd.wan_parameter_grid_trace.v1',
                 workload_fingerprint=workload.receipt()['fingerprint'],
                 source_trace_binding=self.binding_hash),
            sort_keys=True, separators=(',', ':')).encode()).hexdigest()

    def _validate(self):
        # Called by the canonical reader before its binding is finalized.
        # All native tensor loading/source/entry machinery is reused.
        self.logical_steps = self.workload.logical_steps
        for trace in self._control.values():
            self.workload.validate_rank_manifest(trace.manifest)
        for step in self.logical_steps:
            self._validate_same_physical_signature(f'step/{step}/critic/rollout')
        self._validate_same_physical_signature(f'step/{self.logical_steps[0]}/generator/rollout')

    def _validate_same_physical_signature(self, operation):
        values = [self._physical_signature(self._control[r].operation(operation))
                  for r in range(self.world_size)]
        if any(value != values[0] for value in values[1:]):
            raise ValueError('native source generated rank-divergent calls; not admissible to this FSDP timing worker')
