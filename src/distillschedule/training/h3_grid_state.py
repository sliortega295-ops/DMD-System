"""Serialization views over the real shared-H3 two-adapter state owners.

Frozen model/table bytes are bound by the outer identity, not duplicated in
every optimizer shard. The common checkpoint writer retains local DTensors,
all-rank admission, safe trees and atomic publication. EMA is explicitly zero.
"""
from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

from .h3_grid_inputs import require
from .native_state import TrainableWeights, OptimizerState
from .checkpoint import _validate_layout


class RoleOptimizer(OptimizerState):
    def __init__(self, engine, lifecycle, ratio):
        super().__init__(engine)
        self.lifecycle, self.ratio = lifecycle, ratio

    def validate_state_dict(self, value):
        import torch
        super().validate_state_dict(value)
        expected = self.ratio * self.lifecycle.completed
        state = value["optimizer"]
        require(state["param_groups"] == self.engine.optimizer.state_dict()["param_groups"],
                "original Adam hyperparameters/parameter groups changed")
        parameter_ids = [index for group in state["param_groups"] for index in group["params"]]
        require(set(state["state"]) == (set(parameter_ids) if expected else set()),
                "original Adam state is incomplete at the declared update boundary")
        for group, saved in zip(self.engine.optimizer.param_groups, state["param_groups"], strict=True):
            for parameter, index in zip(group["params"], saved["params"], strict=True):
                if not expected:
                    continue
                slots = state["state"][index]
                require(set(slots) == {"step", "exp_avg", "exp_avg_sq"}, "original Adam slots changed")
                step = slots["step"]
                require(isinstance(step, torch.Tensor) and step.ndim == 0 and step.dtype == torch.float32
                        and float(step) == expected, "original Adam update clock changed")
                # Validate SOURCE slots, before native load_state_dict can cast
                # a malformed moment dtype or mask a source-layout mismatch.
                for name in ("exp_avg", "exp_avg_sq"):
                    _validate_layout(parameter, slots[name], "H3 original Adam " + name)


class ConstantScheduler:
    def __init__(self, scheduler, lifecycle, ratio):
        self.scheduler, self.lifecycle, self.ratio = scheduler, lifecycle, ratio

    def state_dict(self):
        return self.scheduler.state_dict()

    def validate_state_dict(self, value):
        current = self.state_dict()
        dynamic = {"last_epoch", "_step_count", "_last_lr"}
        expected = self.lifecycle.completed * self.ratio
        require(isinstance(value, dict) and set(value) == set(current)
                and type(value.get("last_epoch")) is int and value["last_epoch"] == expected
                and type(value.get("_step_count")) is int and value["_step_count"] == expected + 1
                and value["_last_lr"] == current["base_lrs"]
                and all(value[k] == current[k] for k in set(current) - dynamic),
                "original constant scheduler layout/policy/update clock changed")

    def load_state_dict(self, value):
        self.validate_state_dict(value)
        self.scheduler.load_state_dict(deepcopy(value))


class TrainingClocks:
    def __init__(self, trainer, lifecycle, optimizers, schedulers):
        self.trainer, self.lifecycle = trainer, lifecycle
        self.optimizers, self.schedulers = optimizers, schedulers

    def _sampler(self):
        loader = self.trainer.dataloader_train
        sampler = loader.sampler
        return {"epoch": 0, "cursor": self.lifecycle.completed * 6, "seed": sampler.seed,
                "rank": sampler.rank, "num_replicas": sampler.num_replicas, "dataset_length": len(loader.dataset)}

    def state_dict(self):
        model = self.trainer.shared_model
        return {"cycle": self.lifecycle.completed, "score_serial": self.trainer._score_serial,
                "sampler": self._sampler(), "physical_role": model._physical_role,
                "physical_training": model._physical_training_mode}

    def validate_state_dict(self, value):
        require(isinstance(value, dict) and set(value) == set(self.state_dict())
                and type(value["cycle"]) is int and value["cycle"] == self.lifecycle.completed
                and type(value["score_serial"]) is int and value["score_serial"] == 6 * self.lifecycle.completed
                and value["sampler"] == self._sampler() and all(type(v) is int for v in value["sampler"].values())
                and value["physical_role"] == "student"
                and (value["physical_training"] is True or self.lifecycle.completed == 0
                     and value["physical_training"] is None),
                "original trainer/sampler/score clock is not the declared complete boundary")

    def load_state_dict(self, value):
        self.validate_state_dict(value)
        model = self.trainer.shared_model
        self.trainer._score_serial = value["score_serial"]
        # Restore the real adapter/mode owner, not only its cache marker. Its
        # activation method also restores PEFT's actual active-adapter state.
        model._physical_role = None
        model._activate_role(value["physical_role"])
        model._physical_training_mode = None
        if value["physical_training"] is not None:
            model.set_transformer_training(value["physical_training"])

    def validate_checkpoint_boundary(self):
        import torch
        from distillschedule.open_workload.torch_backend import physical_tensor
        self.validate_state_dict(self.state_dict())
        require(self.trainer.shared_model._role.get() == "student", "original adapter scope is still active")
        cursor = self.lifecycle.cursor
        require(cursor is not None, "original native sampler cursor is missing")
        cursor.validate_cycle(self.lifecycle.completed)
        for optimizer in self.optimizers:
            optimizer.validate_state_dict(optimizer.state_dict())
            for group in optimizer.engine.optimizer.param_groups:
                for parameter in group["params"]:
                    require(parameter.grad is None, "original checkpoint has an unconsumed gradient")
                    require(bool(torch.isfinite(physical_tensor(parameter)).all()), "nonfinite original trainable")
                    for slot in optimizer.engine.optimizer.state.get(parameter, {}).values():
                        if isinstance(slot, torch.Tensor):
                            require(bool(torch.isfinite(physical_tensor(slot)).all()), "nonfinite original Adam slot")
        for scheduler in self.schedulers:
            scheduler.validate_state_dict(scheduler.state_dict())


def trainer_state(trainer, lifecycle):
    """Bind the two real optimizer-owned role sets on one physical module."""
    import torch
    model = trainer.shared_model
    module = model.denoiser_module()
    student, fake = model.role_parameters("student"), model.role_parameters("fake")
    require({id(p) for p in student} == {id(p) for p in trainer.trainable_params}
            and {id(p) for p in fake} == {id(p) for p in trainer.fake_trainable_params}
            and not {id(p) for p in student} & {id(p) for p in fake}
            and {id(p) for p in student + fake} == {id(p) for p in module.parameters() if p.requires_grad},
            "original shared model/role trainable ownership changed")
    state, optimizers, schedulers = {}, [], []
    for role, params, optimizer, scheduler, ratio in (
        ("student", student, trainer.optimizer, trainer.lr_scheduler, 1),
        ("fake", fake, trainer.fake_optimizer, trainer.fake_lr_scheduler, 5)):
        require(type(optimizer) is torch.optim.AdamW
                and [id(p) for group in optimizer.param_groups for p in group["params"]] == [id(p) for p in params],
                "actual original AdamW parameter order/ownership changed")
        engine = SimpleNamespace(module=module, optimizer=optimizer)
        view = RoleOptimizer(engine, lifecycle, ratio)
        schedule = ConstantScheduler(scheduler, lifecycle, ratio)
        state.update({role + ".weights": TrainableWeights(engine), role + ".optimizer": view,
                      role + ".scheduler": schedule})
        optimizers.append(view); schedulers.append(schedule)
    state["clocks"] = TrainingClocks(trainer, lifecycle, optimizers, schedulers)
    return state
