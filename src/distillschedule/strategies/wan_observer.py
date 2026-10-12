"""Reuse the audited synchronized observer with the task's actual action graph."""
from distillschedule.search.wan_action_calibration import ActionCalibrator
from distillschedule.search.wan_stage_calibration import StageCalibrator
from distillschedule.search.wan_implementation import WanImplementation
from .wan_planning import action_graph,component_contract

class OriginalWanCalibrator(ActionCalibrator):
    SCHEMA='distillschedule.original_wan_action_rank.v1'
    ADAPTER='original_wan_parameterized'
    WORKLOAD='wan13b_original_parameterized'

    def __init__(self, *, task, strategy, initial_versions, **kwargs):
        self.task,self.strategy,self.initial_versions=task,strategy,dict(initial_versions)
        # The legacy observer uses this type only as an identity holder. Graph
        # and component contracts below use the real n/window/task instead.
        impl=WanImplementation(strategy.placement,strategy.chunk_tokens,strategy.lora_epilogue,1)
        super().__init__(work={},implementation=impl,**kwargs)

    def operation_graph(self):
        return action_graph(self.task,self.initial_versions,self.strategy.rollout_window)

    def contract(self,operation):
        return component_contract(self.task,operation,self.strategy).to_dict()

    def snapshot(self):
        value=super().snapshot()
        value.update(task=self.task.to_dict(),implementation=self.strategy.to_dict())
        return value
