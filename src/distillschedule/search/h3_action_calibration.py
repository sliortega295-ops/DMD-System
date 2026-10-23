"""Default-off H3 action calibration using the shared fenced collector.

Actual application DiT metadata is recorded while the native helpers run.
Contracts are sealed *after* observation, not inferred from a config. No tensor
value reads. Calibration fences/counters are never native timing evidence.
"""
from contextlib import contextmanager
from dataclasses import fields, is_dataclass

from distillschedule.workloads.best_known.registry import fingerprint
from .contracts import require
from .h3_action_costs import ADAPTER, H3, PRECISION, action_graph, component_contract
from .h3_implementation import H3Implementation
from .wan_action_calibration import ActionCalibrator


def shallow_prepared(value):
    if is_dataclass(value) and not isinstance(value,type):
        return {f.name:shallow_prepared(getattr(value,f.name)) for f in fields(value)}
    if isinstance(value,dict): return {k:shallow_prepared(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [shallow_prepared(v) for v in value]
    return value  # Keep original tensor/storage; asdict/deepcopy is forbidden.


def tensor_layout(torch, value):
    if torch.is_tensor(value):
        return dict(shape=list(value.shape), dtype=str(value.dtype))
    if isinstance(value,dict): return {k:tensor_layout(torch,v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [tensor_layout(torch,v) for v in value]
    require(value is None or type(value) in (bool,int,float,str), "unsupported H3 physical layout metadata")
    return value


def validate_calls(action, calls):
    wanted = ([('student',False)]*3+[('student',True),('fake',False),('teacher',False)] if action=='g'
              else [('student',False)]*4 if action.startswith('r') else [('fake',True)])
    require([(c['role'],c['grad_enabled']) for c in calls] == wanted, "H3 actual application call inventory differs")
    require(all(set(c)=={'role','grad_enabled','latents','sigmas','condition','latent_shape'} for c in calls),
            "H3 forward tensor/condition metadata missing")


class H3ActionCalibrator(ActionCalibrator):
    SCHEMA = "distillschedule.h3_action_rank_calibration.v1"
    ADAPTER = ADAPTER
    WORKLOAD = H3
    IMPLEMENTATION_TYPE = H3Implementation
    PRECISION_FINGERPRINT = PRECISION

    def __init__(self, **kwargs):
        self._calls = None
        self._original_predict = None
        super().__init__(work={}, **kwargs)

    def operation_graph(self):
        return action_graph(self._implementation.rollout_window)

    def contract(self, op):
        # This placeholder is never admitted or emitted by a finished collector.
        return component_contract(op,self._implementation,self.work.get(op.operation_id,
            {"status":"PENDING_NATIVE_METADATA_NOT_ADMISSIBLE"})).to_dict()

    def note_forward(self, row):
        require(self._active is not None and self._calls is not None,"H3 application call outside calibration action")
        self._calls.append(row)

    def install(self, trainer):
        require(self._original_predict is None,"H3 metadata hook already installed")
        original=trainer._predict_velocity_role
        self._original_predict=(trainer,'_predict_velocity_role' in trainer.__dict__,trainer.__dict__.get('_predict_velocity_role'))
        def observed(role, latents, sigmas, condition, latent_shape):
            self.note_forward(dict(role=role,grad_enabled=bool(self.torch.is_grad_enabled()),
                **{k:tensor_layout(self.torch,v) for k,v in dict(latents=latents,sigmas=sigmas,
                    condition=condition,latent_shape=latent_shape).items()}))
            return original(role,latents,sigmas,condition,latent_shape)
        trainer._predict_velocity_role=observed

    def close(self):
        if self._original_predict is not None:
            trainer,had_local,old=self._original_predict
            if had_local: trainer._predict_velocity_role=old
            else: del trainer._predict_velocity_role
            self._original_predict=None

    @contextmanager
    def stage(self, cycle, stage_name):
        self._calls=[]
        try:
            with super().stage(cycle,stage_name) as sample:
                yield sample
                validate_calls(stage_name,self._calls)
            key=f"cycle{cycle}.{stage_name}"
            self.work[key]=dict(physical_batch=1,logical_batch=16,application_calls=self._calls)
            op=next(op for op in self._graph.operations if op.operation_id==key)
            contract=self.contract(op)
            self._stages[-1].update(key=fingerprint(contract),operation_contract=contract,
                operation_contract_sha256=fingerprint(contract),component_contract=contract)
        finally:
            self._calls=None

    def record_output(self, operation_id, value):
        return super().record_output(operation_id,shallow_prepared(value))

    def snapshot(self):
        return {**super().snapshot(),"allocation_limitations":
                "PERSISTENT_GROWTH_INCLUDING_ADAM_GRID_LAYOUT_CACHE_IS_IN_ENVELOPE_NOT_PURE_WORKSPACE; "
                "EXTERNAL_PERSISTENT_STORAGE_ALIASES_NOT_CLASSIFIED; "
                "LOGICAL_LAST_CONSUMER_RELEASE_IS_NOT_ALLOCATOR_FREE_PROOF"}
