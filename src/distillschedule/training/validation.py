"""Bounded read-only held-out callbacks; generation and quality claims stay separate.

This initial capability is deliberately single-process. A distributed owner
needs a separate collective-forward/lifecycle contract, not a rank-zero call
through live DDP/FSDP wrappers or an implicit full-model copy.
"""
from __future__ import annotations

import inspect
import json
import math
import random
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real

from . import distributed
from .checkpoint import restore_rng, rng_state
from .config import digest


def admit_scope(config):
    if "validation" in config.trainer and distributed.rank_world()[1] != 1:
        raise ValueError("periodic validation is single-process only; distributed validation is not admitted")


def _modules(value):
    import torch
    if isinstance(value, torch.nn.Module):
        yield from value.modules()
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from _modules(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _modules(child)


def _guard(state):
    """Use the runtime's content signatures and retain live parameter ownership."""
    import torch

    from distillschedule.open_workload.torch_backend import signature
    modules = list(_modules(state))
    return {"contents": signature(state), "objects": {key: id(value) for key, value in state.items()},
            "modules": [(id(module), [(name, id(parameter), signature(parameter.grad))
                                      for name, parameter in module.named_parameters()],
                         [(name, id(buffer)) for name, buffer in module.named_buffers()]) for module in modules],
            "optimizers": {key: [[id(parameter) for parameter in group["params"]]
                                  for group in value.param_groups]
                           for key, value in state.items() if isinstance(value, torch.optim.Optimizer)},
            "scheduler_owners": {key: id(value.optimizer) for key, value in state.items()
                                 if isinstance(getattr(value, "optimizer", None), torch.optim.Optimizer)}}


def _metrics(value):
    import torch
    if not isinstance(value, Mapping) or not value:
        raise ValueError("validate_batch must return a nonempty mapping of named finite scalar metrics")
    result = {}
    for name, metric in value.items():
        if isinstance(metric, torch.Tensor) and metric.numel() == 1:
            metric = metric.detach().cpu().item()
        if (not isinstance(name, str) or not name or isinstance(metric, bool)
                or not isinstance(metric, Real)):
            raise ValueError("validate_batch must return named finite scalar metrics")
        try:
            scalar = float(metric)
        except (ValueError, OverflowError) as exc:
            raise ValueError("validate_batch must return named finite scalar metrics") from exc
        if not math.isfinite(scalar):
            raise ValueError("validate_batch must return named finite scalar metrics")
        result[name] = scalar
    return result


@dataclass(frozen=True)
class Validation:
    dataset: object
    callback: object
    settings: dict
    identity: dict

    @classmethod
    def prepare(cls, config, components):
        if "validation" not in config.trainer:
            return None
        admit_scope(config)
        import torch

        from distillschedule.data import PreparedDataset
        from distillschedule.open_workload.torch_backend import code_signature
        callback = components.validate_batch
        if not inspect.isfunction(callback):
            raise ValueError("validation requires an inspectable model validate_batch(state, feeds, parameters) function")
        try:
            inspect.signature(callback).bind({}, {}, {})
        except TypeError as exc:
            raise ValueError("validate_batch signature must accept (state, feeds, parameters)") from exc
        captured = inspect.getclosurevars(callback)
        if any(isinstance(value, (torch.nn.Module, torch.optim.Optimizer, torch.Tensor))
               for value in (*captured.nonlocals.values(), *captured.globals.values())):
            raise ValueError("validate_batch must use declared state/feeds, not captured models or tensors")
        dataset = PreparedDataset(config.data["validation_prepared"])
        settings = config.trainer["validation"]
        rank, world = distributed.rank_world()
        if settings["samples"] * world > len(dataset):
            raise ValueError("validation samples exceed the complete prepared validation cohort")
        # Reject malformed fixed records before creating output or updating a
        # model. The immutable dataset checks bytes again on every later read.
        for index in range(rank, settings["samples"] * world, world):
            dataset.sample(index)
        identity = {"dataset": dataset.identity, "callback": code_signature(callback),
                    "settings": settings, "scope": "SINGLE_PROCESS_ADAPTER_METRICS"}
        return cls(dataset, callback, settings, identity)

    def run(self, config, state, *, cycle):
        import torch
        from torch.utils._pytree import tree_map
        admit_scope(config)
        rank, world = distributed.rank_world()
        before_rng = rng_state()
        modes = {id(module): (module, module.training) for module in _modules(state)}
        before_state = _guard(state)
        rows = []
        ids = []
        try:
            # Restore exact mixed parent/child modes, not module.train(old),
            # whose recursive assignment would erase child-specific modes.
            for module, _ in modes.values():
                module.training = False
            eval_state = _guard(state)
            with torch.no_grad():
                for index in range(rank, self.settings["samples"] * world, world):
                    seed = int(digest({"seed": config.trainer.get("seed", 0),
                                       "validation": self.identity, "sample_index": index, "rank": rank})[:15], 16)
                    random.seed(seed)
                    torch.random.default_generator.manual_seed(seed)
                    if "cuda" in before_rng:
                        torch.cuda.manual_seed(seed)  # Current owned device only.
                    try:
                        import numpy as np
                        np.random.seed(seed % 2**32)
                    except ImportError:
                        pass
                    feeds = tree_map(lambda value: value.to(config.trainer.get("device", "cpu"))
                                     if isinstance(value, torch.Tensor) else value, self.dataset.sample(index))
                    # A callback cannot change the next callback's parameters.
                    parameters = json.loads(json.dumps(self.settings["parameters"]))
                    try:
                        metrics = _metrics(self.callback(state, feeds, parameters))
                    finally:
                        if _guard(state) != eval_state:
                            raise ValueError("validate_batch mutated declared persistent state; training must stop (no state rollback)")
                    if rows and set(metrics) != set(rows[0]):
                        raise ValueError("validation metric names changed within the fixed cohort")
                    rows.append(metrics)
                    ids.append(self.dataset.sample_id(index))
        finally:
            for module, training in modes.values():
                module.training = training
            try:
                if _guard(state) != before_state:
                    raise ValueError("validate_batch mutated declared persistent state; training must stop (no state rollback)")
            finally:
                restore_rng(before_rng)
        metrics = {name: math.fsum(row[name] / len(rows) for row in rows) for name in rows[0]}
        if not all(math.isfinite(value) for value in metrics.values()):
            raise ValueError("validation aggregation must produce finite scalar metrics")
        return {"cycle": cycle, "rank": rank, "samples": len(rows), "sample_ids": ids, "metrics": metrics,
                "aggregation": "equal_record_mean", "validation_identity": self.identity,
                "metric_scope": "ADAPTER_DEFINED_HELD_OUT_DIAGNOSTIC_NOT_QUALITY_GATE",
                "quality_validation": "NOT_RUN"}
