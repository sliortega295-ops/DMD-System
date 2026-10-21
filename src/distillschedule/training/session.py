"""Persistent public training, backed by the existing Program runtime.

Algorithms still own their mathematical forwards, losses and update hooks.
This layer owns data progress, complete-cycle persistence and user artifacts.
"""
from __future__ import annotations
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import random
import time

from .adapters import ModelComponents
from .checkpoint import cpu_tree, restore_checkpoint, save_checkpoint, read_checkpoint, unpack_state
from .config import TrainingConfig, digest
from .plugins import load_callable
from .policy import CyclePolicy


def _json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _seed(seed):
    import torch
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as np
        np.random.seed(seed % 2**32)
    except ImportError:
        pass


def _source():
    from distillschedule.open_workload.planning import source_identity
    root = Path(__file__).parent
    files = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(root.glob("*.py"))}
    data = root.parent / "data"
    files.update({"data/" + path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in sorted(data.glob("*.py"))})
    return digest({"runtime": source_identity(), "lifecycle": files})


def construct(config):
    import torch
    from distillschedule.data import PreparedDataset
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or (
            torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1):
        raise ValueError("persistent Program training is single-process; use a native distributed recipe")
    dataset = PreparedDataset(config.data["prepared"])
    _seed(config.trainer.get("seed", 0))
    factory, model_source = load_callable(config.model["factory"])
    parameters = dict(config.model.get("parameters", {}))
    parameters.setdefault("device", config.trainer.get("device", "cpu"))
    parameters.setdefault("seed", config.trainer.get("seed", 0))
    components = factory(parameters)
    if not isinstance(components, ModelComponents):
        raise ValueError("model factory must return ModelComponents")
    components.validate()
    algorithm, algorithm_source = load_callable(config.algorithm["factory"])
    identity = {"config": config.identity, "dataset": dataset.identity, "model": model_source,
                "algorithm": algorithm_source, "framework": _source(),
                "model_metadata": dict(components.metadata)}
    return components, dataset, algorithm, identity


def build_cycle(config, components, dataset, algorithm, *, cycle, cursor, versions=None):
    from distillschedule.open_workload.torch_backend import Workload, tensors
    import torch
    if cursor >= len(dataset):
        raise ValueError("prepared input exhausted; add samples instead of silently recycling")
    feeds = dataset.sample(cursor)
    device = config.trainer.get("device", "cpu")
    from torch.utils._pytree import tree_map
    feeds = tree_map(lambda value: value.to(device) if isinstance(value, torch.Tensor) else value, feeds)
    seed = int(digest({"seed": config.trainer.get("seed", 0), "cycle": cycle})[:15], 16)
    workload = algorithm(components, feeds, dict(config.algorithm.get("parameters", {})), cycle, seed)
    if not isinstance(workload, Workload):
        raise ValueError("algorithm factory must return an existing Workload for one cycle")
    workload.validate()
    for name, value in components.state.items():
        if name not in workload.state or workload.state[name] is not value:
            raise ValueError(f"algorithm must use the owned model state: {name}")
    if versions is not None:
        if set(versions) != set(workload.state):
            raise ValueError("algorithm changed logical state between cycles")
        workload.program = replace(workload.program, initial_versions=dict(versions))
    return workload


def _metrics(outputs):
    import torch
    return {str(name): float(value.detach().cpu()) for name, value in outputs.items()
            if isinstance(value, torch.Tensor) and value.numel() == 1}


def _finite(state, outputs):
    import torch
    from distillschedule.open_workload.torch_backend import tensors
    return all(bool(torch.isfinite(tensor).all()) for tensor in tensors((state, outputs))
               if tensor.is_floating_point() or tensor.is_complex())


def train(config: TrainingConfig):
    components, dataset, algorithm, identity = construct(config)
    if config.trainer["cycles"] > len(dataset):
        raise ValueError("trainer.cycles exceeds prepared samples; explicit dataset progression is required")
    start_cycle = cursor = 0
    versions = None
    policy = CyclePolicy(config.execution)
    if config.trainer.get("resume_from"):
        # Build the original declared state shape before restoring modules,
        # optimizer/scheduler state, history and the exact next data position.
        workload = build_cycle(config, components, dataset, algorithm, cycle=0, cursor=0)
        restored = restore_checkpoint(config.trainer["resume_from"], workload.state, identity=identity)
        start_cycle, cursor, versions = restored["cycle"], restored["cursor"], restored["versions"]
        components.state.update({key: workload.state[key] for key in components.state})
        policy.restore(restored["policy"])
    if start_cycle >= config.trainer["cycles"]:
        raise ValueError("resume target must exceed the checkpoint's completed cycles")
    root = Path(config.trainer["output_dir"])
    root.mkdir(parents=True, exist_ok=False)
    _json(root / "resolved-config.json", config.to_dict())
    _json(root / "identity.json", identity)
    rows = []
    latest = None
    try:
        for cycle in range(start_cycle, config.trainer["cycles"]):
            workload = build_cycle(config, components, dataset, algorithm, cycle=cycle, cursor=cursor,
                                   versions=versions)
            started = time.perf_counter()
            session, outputs = policy.run(workload, root)
            wall = time.perf_counter() - started
            if not _finite(session.state, outputs):
                raise ValueError(f"nonfinite training state/output at cycle {cycle + 1}")
            components.state.update({key: session.state[key] for key in components.state})
            versions = session.receipt["dispatcher"]["final_versions"]
            cursor += 1
            row = {"cycle": cycle + 1, "sample_id": dataset.sample_id(cursor - 1), "cursor": cursor,
                   "metrics": _metrics(outputs), "wall_s": wall,
                   "versions": versions, "finite": True,
                   "execution": session.receipt, "policy_sha256": digest(policy.bound),
                   "timing_scope": "TRAINING_DIAGNOSTIC_INCLUDES_INITIAL_PLAN_ADMISSION"}
            rows.append(row)
            with (root / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            every = config.trainer.get("checkpoint_every", 1)
            if every and ((cycle + 1) % every == 0 or cycle + 1 == config.trainer["cycles"]):
                latest = save_checkpoint(root / "checkpoints" / f"cycle-{cycle + 1:06d}", session.state,
                                         identity=identity, cycle=cycle + 1, cursor=cursor,
                                         versions=versions, policy=policy.bound)
            _json(root / "effective-policy.json", policy.bound)
        result = {"schema_version": "distillschedule.training_result.v1", "status": "COMPLETE",
                  "completed_cycles": config.trainer["cycles"], "cycles_this_process": len(rows),
                  "resume_cycle": start_cycle, "cursor": cursor, "final_versions": versions,
                  "finite": True, "numerical_status": policy.bound["initial_reference_gate"]["status"],
                  "quality_validation": "NOT_RUN", "claimable_speedup": False,
                  "checkpoint": str(latest) if latest else None, "identity": identity,
                  "policy": policy.bound, "gpu_execution": session.receipt["gpu_execution"]}
        _json(root / "summary.json", result)
        return result
    except BaseException as exc:
        _json(root / "failure.json", {"status": "FAILED", "error": str(exc),
                                     "completed_cycles_this_process": len(rows), "cursor": cursor,
                                     "last_complete_checkpoint": str(latest) if latest else None})
        raise


def _load_for_output(config, checkpoint):
    components, dataset, algorithm, identity = construct(config)
    workload = build_cycle(config, components, dataset, algorithm, cycle=0, cursor=0)
    payload = read_checkpoint(checkpoint, identity=identity)
    unpack_state(payload["state"], workload.state)
    components.state.update({key: workload.state[key] for key in components.state})
    return components, dataset, payload, identity


def evaluate(config, checkpoint, output, *, sample_index=0, parameters=None):
    import torch
    components, dataset, payload, identity = _load_for_output(config, checkpoint)
    if components.generate is None:
        raise ValueError("model adapter does not implement generation")
    if not 0 <= sample_index < len(dataset):
        raise ValueError("evaluation sample index out of range")
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    from torch.utils._pytree import tree_map
    feeds = tree_map(lambda value: value.to(config.trainer.get("device", "cpu"))
                     if isinstance(value, torch.Tensor) else value, dataset.sample(sample_index))
    with torch.no_grad():
        generated = components.generate(components.state, feeds, dict(parameters or {}))
        decoded = components.decode(generated, dict(parameters or {})) if components.decode else generated
    if not _finite({}, {"generated": generated, "decoded": decoded}):
        raise ValueError("generation produced nonfinite values")
    torch.save(cpu_tree({"generated": generated, "decoded": decoded}), root / "generation.pt")
    result = {"status": "GENERATED", "checkpoint_cycle": payload["cycle"],
              "sample_id": dataset.sample_id(sample_index), "identity": identity,
              "quality_validation": "NOT_RUN", "artifact": str(root / "generation.pt")}
    _json(root / "generation.json", result)
    return result


def export(config, checkpoint, output):
    import torch
    components, _, payload, identity = _load_for_output(config, checkpoint)
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    if components.export is not None:
        metadata = components.export(components.state, root)
    else:
        weights = {name: cpu_tree(model.state_dict()) for name, model in components.state.items()
                   if isinstance(model, torch.nn.Module)}
        if not weights:
            raise ValueError("model adapter has no exportable modules")
        torch.save(weights, root / "weights.pt")
        metadata = {"format": "torch_state_dicts", "roles": list(weights)}
    result = {"status": "EXPORTED", "kind": "INFERENCE_WEIGHTS_NOT_RESUMABLE_CHECKPOINT",
              "checkpoint_cycle": payload["cycle"], "identity": identity, "export": metadata}
    _json(root / "export.json", result)
    return result
