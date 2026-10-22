"""Persistent public training, backed by the existing Program runtime.

Algorithms still own their mathematical forwards, losses and update hooks.
This layer owns data progress, complete-cycle persistence and user artifacts.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import replace
from pathlib import Path

from distillschedule.workloads.best_known.registry import file_sha256

from . import distributed
from .adapters import InferenceProjection, ModelComponents
from .checkpoint import (
    cpu_tree,
    read_checkpoint,
    read_manifest,
    restore_checkpoint,
    save_checkpoint,
    unpack_state,
)
from .config import TrainingConfig, digest
from .plugins import load_callable
from .policy import CyclePolicy, cycle_structure
from .sampling import ProgramSampling
from .validation import Validation, admit_scope


def _json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _append_json(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


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


def _model_parameters(config):
    """Effective factory arguments; device alone is relocatable for inference."""
    import torch
    parameters = dict(config.model.get("parameters", {}))
    device = config.trainer.get("device", "cpu")
    if device == "cuda":
        device = "cuda:" + str(torch.cuda.current_device())
    parameters.setdefault("device", device)
    parameters.setdefault("seed", config.trainer.get("seed", 0))
    return parameters


def _construct_model(config, *, source=None):
    factory, model_source = load_callable(config.model["factory"])
    if source is not None and (model_source["sha256"] != source["sha256"]
                              or not _same_factory(model_source["reference"], source["reference"])):
        raise ValueError("export model factory source changed")
    parameters = _model_parameters(config)
    components = factory(parameters)
    if not isinstance(components, ModelComponents):
        raise ValueError("model factory must return ModelComponents")
    components.validate()
    return components, model_source


def _same_factory(current, original):
    if current == original:
        return True
    current_path, _, current_symbol = current.rpartition(":")
    original_path, _, original_symbol = original.rpartition(":")
    # Inference alone permits an explicit relocation of byte-identical .py
    # adapters. Checkpoint identity and package/module references stay exact.
    return current_path.endswith(".py") and original_path.endswith(".py") and current_symbol == original_symbol


def construct(config, *, include_validation=False):
    from distillschedule.data import PreparedDataset
    dataset = PreparedDataset(config.data["prepared"])
    _seed(config.trainer.get("seed", 0))
    components, model_source = _construct_model(config)
    algorithm, algorithm_source = load_callable(config.algorithm["factory"])
    if components.admit_training is not None:
        components.admit_training(state=components.state, request=config,
                                  prepared_identity=dataset.identity, prepared_manifest=dataset.manifest)
    validation = Validation.prepare(config, components)
    identity = {"config": config.identity, "dataset": dataset.identity, "model": model_source,
                "algorithm": algorithm_source, "framework": _source(),
                "model_metadata": dict(components.metadata)}
    if validation is not None:
        identity["validation"] = validation.identity
    if config.execution.get("mode") == "plan":
        identity["input_plan_sha256"] = hashlib.sha256(Path(config.execution["plan"]).read_bytes()).hexdigest()
    result = (components, dataset, algorithm, identity)
    return (*result, validation) if include_validation else result


def build_cycle(config, components, dataset, algorithm, *, cycle, cursor, versions=None):
    """Build from a physical manifest index; training resolves sampling first."""
    import torch

    from distillschedule.open_workload.torch_backend import Workload
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


def _training_metrics(components, outputs):
    """Collect original host losses without flattening control receipts or tensors.

    Adapter callbacks are trusted, read-only functions of the actual outputs.
    Explicit Python floats avoid hidden tensor coercion or new GPU readbacks.
    """
    import math

    result = _metrics(outputs)
    if components.collect_metrics is None:
        return result
    extra = components.collect_metrics(outputs)
    if not isinstance(extra, dict) or not all(
            isinstance(key, str) and key and type(value) is float and math.isfinite(value)
            for key, value in extra.items()):
        raise ValueError("adapter metric collection requires named finite Python floats")
    if result.keys() & extra.keys():
        raise ValueError("adapter metric names collide with existing scalar metrics")
    return {**result, **extra}


def _finite(state, outputs):
    import torch

    from distillschedule.open_workload.torch_backend import physical_tensor, tensors
    return all(bool(torch.isfinite(physical_tensor(tensor)).all()) for tensor in tensors((state, outputs))
               if tensor.is_floating_point() or tensor.is_complex())


def train(config: TrainingConfig):
    with distributed.context(config):
        return _train(config)


def _train(config: TrainingConfig):
    # Reject unsupported collective validation before constructing any model,
    # installing a wrapper, updating state or creating output.
    distributed.checked(lambda: admit_scope(config), "validation scope admission")
    distributed.admit_run_controls(config)
    components, dataset, algorithm, identity, validation = distributed.checked(
        lambda: construct(config, include_validation=True), "training construction")
    rank, world = distributed.rank_world()
    if any(row != identity for row in distributed.gather(identity)):
        raise ValueError("ranks disagree on training configuration, source or dataset identity")
    sampling = ProgramSampling(config.data.get("sampling"), size=len(dataset),
                               seed=config.trainer.get("seed", 0), world_size=world)
    sampling.admit_cycles(config.trainer["cycles"])
    start_cycle = cursor = 0
    versions = None
    policy = CyclePolicy(config.execution)
    if config.trainer.get("resume_from"):
        # Build the original declared state shape before restoring modules,
        # optimizer/scheduler state, history and the exact next data position.
        workload = distributed.checked(lambda: build_cycle(
            config, components, dataset, algorithm, cycle=0,
            cursor=sampling.position(rank).sample_index), "resume cycle construction")
        metadata = distributed.checked(lambda: read_manifest(config.trainer["resume_from"], identity=identity),
                                       "checkpoint reading")
        distributed.checked(lambda: sampling.validate_progress(metadata["cycle"], metadata["cursor"]),
                            "checkpoint sampling progress admission")
        policy.restore(metadata["policy"])
        del metadata
        session = policy.prepare_resume(workload)
        restored = restore_checkpoint(config.trainer["resume_from"], session.state, identity=identity)
        start_cycle, cursor, versions = restored["cycle"], restored["cursor"], restored["versions"]
        distributed.checked(lambda: sampling.validate_progress(start_cycle, cursor),
                            "restored sampling progress admission")
        components.state.update({key: session.state[key] for key in components.state})
        del restored
    if start_cycle >= config.trainer["cycles"]:
        raise ValueError("resume target must exceed the checkpoint's completed cycles")
    root = Path(config.trainer["output_dir"])
    distributed.primary(lambda: root.mkdir(parents=True, exist_ok=False), "training output creation")
    distributed.primary(lambda: _json(root / "resolved-config.json", config.to_dict()), "config publication")
    distributed.primary(lambda: _json(root / "identity.json", identity), "identity publication")
    rank_root = root if world == 1 else root / "ranks" / f"rank-{rank:05d}"
    if world > 1:
        distributed.checked(lambda: rank_root.mkdir(parents=True, exist_ok=False), "rank output creation")
    completed_cycles_this_process = 0
    validation_runs = 0
    last_validation = None
    latest = Path(config.trainer["resume_from"]).resolve() if config.trainer.get("resume_from") else None
    try:
        for cycle in range(start_cycle, config.trainer["cycles"]):
            position = sampling.position(cursor + rank)
            workload = distributed.checked(lambda: build_cycle(
                config, components, dataset, algorithm, cycle=cycle, cursor=position.sample_index,
                versions=versions), "cycle construction")
            started = time.perf_counter()
            session, outputs = policy.run(workload, root)
            wall = time.perf_counter() - started
            if not all(distributed.gather(_finite(session.state, outputs))):
                raise ValueError(f"nonfinite training state/output at cycle {cycle + 1}")
            components.state.update({key: session.state[key] for key in components.state})
            versions = session.receipt["dispatcher"]["final_versions"]
            if validation is not None and (cycle + 1) % validation.settings["every_cycles"] == 0:
                validation_row = validation.run(config, session.state, cycle=cycle + 1)
                distributed.checked(lambda: _append_json(rank_root / "validation.jsonl", validation_row),
                                    "validation metrics publication")
                rank_validation = distributed.gather(validation_row)
                last_validation = {"cycle": cycle + 1, "ranks": rank_validation,
                                   "metrics": validation_row["metrics"], "aggregation": "equal_record_mean",
                                   "metric_scope": validation_row["metric_scope"], "quality_validation": "NOT_RUN"}
                validation_runs += 1
                distributed.primary(lambda last_validation=last_validation: _json(root / "validation-summary.json", last_validation),
                                    "validation summary publication")
            metrics = distributed.checked(lambda: _training_metrics(components, outputs), "training metric collection")
            cursor += world
            row = {"cycle": cycle + 1, "rank": rank, "sample_id": dataset.sample_id(position.sample_index), "cursor": cursor,
                   "sample_index": position.sample_index, "epoch": position.epoch, "epoch_offset": position.epoch_offset,
                   "metrics": metrics, "wall_s": wall,
                   "versions": versions, "finite": True,
                   "execution": session.receipt, "policy_sha256": digest(policy.bound),
                   "timing_scope": "TRAINING_DIAGNOSTIC_INCLUDES_INITIAL_PLAN_ADMISSION"}
            completed_cycles_this_process += 1
            distributed.checked(lambda: _append_json(rank_root / "metrics.jsonl", row),
                                "training metrics publication")
            every = config.trainer.get("checkpoint_every", 1)
            if every and ((cycle + 1) % every == 0 or cycle + 1 == config.trainer["cycles"]):
                latest = save_checkpoint(root / "checkpoints" / f"cycle-{cycle + 1:06d}", session.state,
                                         identity=identity, cycle=cycle + 1, cursor=cursor,
                                         versions=versions, policy=policy.bound)
            distributed.primary(lambda: _json(root / "effective-policy.json", policy.bound), "policy publication")
        rank_results = distributed.gather({"rank": rank, "completed_cycles": config.trainer["cycles"],
                                           "cursor": cursor, "versions": versions})
        result = {"schema_version": "distillschedule.training_result.v1", "status": "COMPLETE",
                  "completed_cycles": config.trainer["cycles"], "cycles_this_process": completed_cycles_this_process,
                  "resume_cycle": start_cycle, "cursor": cursor, "final_versions": versions,
                  "finite": True, "numerical_status": policy.bound["initial_reference_gate"]["status"],
                  "quality_validation": "NOT_RUN", "claimable_speedup": False,
                  "validation": {"status": ("DISABLED" if validation is None else
                                             "COMPLETED_ADAPTER_METRICS" if validation_runs else "NOT_DUE"),
                                 "runs_this_process": validation_runs, "last": last_validation},
                  "checkpoint": str(latest) if latest else None, "identity": identity,
                  "sampling": sampling.describe(),
                  "world_size": world, "rank_count": len(rank_results), "ranks": rank_results,
                  "policy": policy.bound, "gpu_execution": session.receipt["gpu_execution"]}
        distributed.primary(lambda: _json(root / "summary.json", result), "training summary publication")
        return result
    except BaseException as exc:
        try:
            _json(rank_root / "failure.json", {"status": "FAILED", "error": str(exc),
                                         "completed_cycles_this_process": completed_cycles_this_process, "cursor": cursor,
                                         "last_complete_checkpoint": str(latest) if latest else None})
        except Exception:
            # A full/unwritable output directory must not mask the training error.
            pass
        raise


def _load_for_output(config, checkpoint):
    components, dataset, algorithm, identity = distributed.checked(lambda: construct(config), "output construction")
    if any(row != identity for row in distributed.gather(identity)):
        raise ValueError("ranks disagree on checkpoint output configuration, source or dataset")
    rank, world = distributed.rank_world()
    sampling = ProgramSampling(config.data.get("sampling"), size=len(dataset),
                               seed=config.trainer.get("seed", 0), world_size=world)
    workload = distributed.checked(lambda: build_cycle(config, components, dataset, algorithm, cycle=0,
                                                      cursor=sampling.position(rank).sample_index),
                                   "output workload construction")
    metadata = distributed.checked(lambda: read_manifest(checkpoint, identity=identity), "output checkpoint reading")
    if any(row != digest(metadata) for row in distributed.gather(digest(metadata))):
        raise ValueError("ranks disagree on output checkpoint manifest or progress")
    distributed.checked(lambda: sampling.validate_progress(metadata["cycle"], metadata["cursor"]),
                        "output sampling progress admission")
    policy = CyclePolicy(config.execution)
    distributed.checked(lambda: policy.restore(metadata["policy"]), "output policy admission")
    def structure_admission():
        if cycle_structure(workload) != policy.bound["structure"]:
            raise ValueError("checkpoint output workload structure changed")
    distributed.checked(structure_admission, "output workload structure admission")
    # The saved DDP/FSDP/HSDP owner must exist BEFORE decoding local shards.
    # Reuse the persistent Program lifecycle, without executing a train update.
    session = policy.prepare_resume(workload)
    payload = distributed.checked(lambda: read_checkpoint(checkpoint, identity=identity), "output checkpoint files admission")
    distributed.checked(lambda: sampling.validate_progress(payload["cycle"], payload["cursor"]),
                        "output shard sampling progress admission")
    unpack_state(payload["state"], session.state, progress={key:payload[key] for key in ("cycle", "cursor", "versions")})
    components.state.update({key: session.state[key] for key in components.state})
    return components, dataset, payload, identity, session


def _generation_parameters(parameters):
    try:
        # Pass the same canonical JSON value to every callback, not only an
        # equal hash of differing Python tuple/key representations.
        return json.loads(json.dumps(dict(parameters or {}), sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("generation parameters must be finite JSON values") from exc


def _generate_values(config, components, feeds, parameters, *, video_required=False, latents_required=False):
    import torch
    from torch.utils._pytree import tree_map

    from .inference import decoded_video, latent_outputs
    if components.generate is None:
        raise ValueError("model adapter does not implement generation")
    for value in components.state.values():
        if isinstance(value, torch.nn.Module):
            value.eval()
    feeds = tree_map(lambda value: value.to(config.trainer.get("device", "cpu"))
                     if isinstance(value, torch.Tensor) else value, feeds)
    # This runtime noise seed is independent of factory initialization/defaults
    # and of RNG consumed while assembling the checkpoint's training workload.
    _seed(config.trainer.get("seed", 0))
    with torch.no_grad():
        generated = components.generate(components.state, feeds, dict(parameters or {}))
        decoded = (components.decode(generated, dict(parameters or {}))
                   if components.decode and not latents_required else generated)
    if not _finite({}, {"generated": generated, "decoded": decoded}):
        raise ValueError("generation produced nonfinite values")
    if latents_required:
        latent_outputs(generated)
    else:
        decoded_video(decoded, required=video_required)
    return {"generated": generated, "decoded": decoded}


def evaluate(config, checkpoint, output, *, sample_index=0, parameters=None):
    with distributed.context(config):
        from .inference import write_generation
        root = Path(output).expanduser().resolve()
        def admit_request():
            effective = _generation_parameters(parameters)
            return effective, digest({"output": str(root), "sample_index": sample_index, "parameters": effective})
        parameters, request = distributed.checked(admit_request, "evaluation request admission")
        if any(row != request for row in distributed.gather(request)):
            raise ValueError("ranks disagree on shared evaluation request")
        components, dataset, payload, identity, _session = _load_for_output(config, checkpoint)
        rank, world = distributed.rank_world()
        def admit():
            if type(sample_index) is not int or sample_index < 0 or sample_index + world > len(dataset):
                raise ValueError("evaluation sample index cohort out of range")
            if components.generate is None:
                raise ValueError("model adapter does not implement generation")
            return dataset.sample(sample_index + rank), dataset.sample_id(sample_index + rank)
        feeds, sample_id = distributed.checked(admit, "evaluation input admission")
        # All ranks execute native forwards (and any FSDP all-gathers), not just
        # rank zero. Each receives its own rank-strided prepared sample.
        values = distributed.checked(lambda: _generate_values(config, components, feeds, parameters), "checkpoint generation")
        # Finish rank-local validation everywhere before any full_tensor gather.
        values = distributed.checked(lambda: cpu_tree(values), "generation full-output gathering")
        distributed.primary(lambda: root.mkdir(parents=True, exist_ok=False), "evaluation output creation")
        rank_root = root if world == 1 else root / "ranks" / f"rank-{rank:05d}"
        if world > 1:
            distributed.checked(lambda: rank_root.mkdir(parents=True, exist_ok=False), "evaluation rank output creation")
        result = distributed.checked(lambda: write_generation(rank_root, values, metadata={
            "status": "GENERATED", "kind": "INFERENCE_FROM_TRAINING_CHECKPOINT",
            "checkpoint_cycle": payload["cycle"], "rank": rank, "world_size": world,
            "sample_index": sample_index + rank, "sample_id": sample_id, "identity": identity,
            "input_identity": dataset.identity, "generation_parameters": parameters,
            "inference_seed": config.trainer.get("seed", 0)}), "evaluation artifact writing")
        results = distributed.gather(result)
        def publish():
            if world > 1:
                _json(root / "generation.json", {"status": "GENERATED", "world_size": world,
                    "checkpoint_cycle": payload["cycle"], "identity": identity, "ranks": results,
                    "generation_parameters": parameters, "inference_seed": config.trainer.get("seed", 0),
                    "quality_validation": "NOT_RUN"})
            (root / "_SUCCESS").write_text(file_sha256(root / "generation.json") + "\n")
        distributed.primary(publish, "evaluation summary publication")
        return result


def export(config, checkpoint, output):
    with distributed.context(config):
        from .inference import publish_export
        components, _, payload, identity, _session = _load_for_output(config, checkpoint)
        parameters = {key: value for key, value in _model_parameters(config).items() if key != "device"}
        factory, source = config.model["factory"], identity["model"]
        if components.inference_projection is not None:
            def project():
                value = components.inference_projection(components, dict(parameters))
                if not isinstance(value, InferenceProjection):
                    raise ValueError("inference_projection must return InferenceProjection")
                return value.validate()
            projection = distributed.checked(project, "inference projection admission")
            _, source = distributed.checked(lambda: load_callable(projection.factory), "inference factory source admission")
            components, factory = projection.components, projection.factory
            parameters = _generation_parameters(projection.parameters)
            parameters.setdefault("seed", config.trainer.get("seed", 0))
            identity = {"model": source, "model_metadata": dict(components.metadata), "training_provenance": identity}
        return publish_export(components, output, model={"factory": factory,
            "source": source, "parameters": parameters, "metadata": dict(components.metadata)},
            identity=identity, cycle=payload["cycle"])


def generate_from_export(config, weights_dir, output, *, prepared, sample_index=0, parameters=None, output_kind="video"):
    """Generate on new prepared inputs, without training data, algorithm or RNG state."""
    from distillschedule.data import PreparedDataset

    from .inference import read_export, restore_modules, write_generation
    if output_kind not in ("video", "latents"):
        raise ValueError("output_kind must be video or explicit diagnostic latents")
    if distributed.rank_world()[1] != 1 or int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("portable export generation is single-process; distributed checkpoint evaluation uses evaluate")
    if config.execution.get("adapter", "program") != "program":
        raise ValueError("portable inference requires a Program model adapter")
    # Integrity admission precedes model imports/construction and any output.
    root, manifest = read_export(weights_dir)
    parameters = _generation_parameters(parameters)
    model = manifest["model"]
    effective = {key: value for key, value in _model_parameters(config).items() if key != "device"}
    if not _same_factory(config.model["factory"], model["factory"]) or digest(effective) != digest(model["parameters"]):
        raise ValueError("export model factory or effective parameters changed")
    dataset = PreparedDataset(prepared)
    if type(sample_index) is not int or not 0 <= sample_index < len(dataset):
        raise ValueError("generation sample index out of range")
    feeds, sample_id = dataset.sample(sample_index), dataset.sample_id(sample_index)
    _seed(config.trainer.get("seed", 0))
    components, model_source = _construct_model(config, source=model["source"])
    if digest(dict(components.metadata)) != digest(model["metadata"]):
        raise ValueError("export model metadata changed")
    restore_modules(components, root, manifest)
    values = cpu_tree(_generate_values(config, components, feeds, parameters,
                                      video_required=output_kind == "video", latents_required=output_kind == "latents"))
    _, current = read_export(root)
    if digest(current) != digest(manifest):
        raise ValueError("export manifest changed during generation")
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = write_generation(output, values, metadata={
        "status": "GENERATED", "kind": "INFERENCE_FROM_EXPORTED_WEIGHTS", "sample_index": sample_index,
        "sample_id": sample_id, "input_identity": dataset.identity,
        "generation_parameters": parameters, "inference_seed": config.trainer.get("seed", 0),
        "loaded_factory": model_source,
        "export_manifest_sha256": file_sha256(root / "export.json"),
        "model": model, "checkpoint_cycle": manifest["checkpoint_cycle"]}, output_kind=output_kind)
    (output / "_SUCCESS").write_text(file_sha256(output / "generation.json") + "\n")
    return result
