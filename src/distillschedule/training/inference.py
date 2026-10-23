"""Portable, hash-bound inference artifacts, separate from restart state.

Only named module state dictionaries have a built-in loader. Custom formats
remain explicit trusted adapter contracts; no optimizer/RNG restore is inferred.
"""
from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path, PurePosixPath

from distillschedule.workloads.best_known.registry import file_sha256

from . import distributed
from .checkpoint import cpu_tree
from .config import digest

SCHEMA = "distillschedule.inference_export.v1"
KIND = "INFERENCE_WEIGHTS_NOT_RESUMABLE_CHECKPOINT"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _file(root, name):
    _require(isinstance(name, str) and name and "\\" not in name, "invalid export file path")
    path = PurePosixPath(name)
    _require(not path.is_absolute() and name == path.as_posix() and ".." not in path.parts
             and name not in {"export.json", "_SUCCESS"}, "export file path escapes artifact bounds")
    result = root / name
    _require(result.resolve().is_relative_to(root)
             and all(not parent.is_symlink() for parent in (result, *result.parents) if parent != root),
             "export file path escapes artifact bounds or contains a symlink")
    return result


def _files(root):
    files = {}
    for path in sorted(root.rglob("*")):
        _require(not path.is_symlink(), "export file path contains a symlink")
        if path.is_file():
            name = path.relative_to(root).as_posix()
            if name not in {"export.json", "_SUCCESS"}:
                files[name] = file_sha256(_file(root, name))
        else:
            _require(path.is_dir(), "export contains an unsupported file type")
    return files


def _tree_layout(value):
    import torch
    if isinstance(value, torch.Tensor):
        return {"kind": "tensor", "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, dict):
        _require(all(isinstance(key, str) for key in value), "export state keys must be strings")
        return {"kind": "dict", "items": {key: _tree_layout(item) for key, item in value.items()}}
    if isinstance(value, (list, tuple)):
        return {"kind": type(value).__name__, "items": [_tree_layout(item) for item in value]}
    _require(value is None or type(value) in (str, bool, int, float), "unsupported export extra state")
    return {"kind": "scalar", "type": type(value).__name__}


def modules(state):
    import torch

    from distillschedule.open_workload.torch_backend import unwrap_module
    return {name: unwrap_module(value) for name, value in sorted(state.items()) if isinstance(value, torch.nn.Module)}


def _finite_weights(weights):
    import torch
    from torch.utils._pytree import tree_leaves
    _require(all(bool(torch.isfinite(tensor).all()) for tensor in tree_leaves(weights)
                 if isinstance(tensor, torch.Tensor) and (tensor.is_floating_point() or tensor.is_complex())),
             "export contains nonfinite module weights")


def module_layout(state):
    import torch
    from torch.distributed.fsdp import FSDPModule
    result = {}
    for name, module in modules(state).items():
        # fully_shard adds a runtime subclass, not a new model architecture.
        cls = next(cls for cls in type(module).__mro__ if issubclass(cls, torch.nn.Module)
                   and not issubclass(cls, FSDPModule)) if isinstance(module, FSDPModule) else type(module)
        result[name] = {"class": cls.__module__ + "." + cls.__qualname__,
                        "state": _tree_layout(module.state_dict())}
    return result


def publish_export(components, output, *, model, identity, cycle):
    """All ranks materialize full tensors; only primary writes shared files."""
    import torch
    rank, world = distributed.rank_world()
    output = Path(output).expanduser().resolve()
    _require(all(row == str(output) for row in distributed.gather(str(output))),
             "ranks disagree on shared inference output path")
    def admit():
        if components.export is not None:
            _require(world == 1, "custom distributed export callbacks are not supported")
            _require(components.load_export is not None, "custom export requires an explicit load_export callback")
        else:
            _require(bool(modules(components.state)), "model adapter has no exportable modules")
        value = {**model, "modules": module_layout(components.state)}
        digest(value)
        return value
    model = distributed.checked(admit, "inference export admission")
    _require(all(row == model for row in distributed.gather(model)), "ranks disagree on exported model layout")
    # cpu_tree calls full_tensor on DTensors, which MUST run on all owners.
    weights = (distributed.checked(lambda: {name: cpu_tree(dict(sorted(module.state_dict().items())))
               for name, module in modules(components.state).items()}, "inference full-weight gathering")
               if components.export is None else None)
    token = distributed.gather(uuid.uuid4().hex if rank == 0 else None)[0]
    staging = output.with_name("." + output.name + ".tmp-" + token)
    def create():
        _require(not output.exists(), f"inference output already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        staging.mkdir()
    distributed.primary(create, "inference export staging")
    def publish():
        if components.export is None:
            _finite_weights(weights)
            torch.save(weights, staging / "weights.pt")
            metadata = {"format": "torch_state_dicts", "roles": sorted(weights)}
        else:
            metadata = {"format": "adapter", "metadata": components.export(components.state, staging)}
        _require(not (staging / "export.json").exists() and not (staging / "_SUCCESS").exists(),
                 "custom exporter wrote reserved manifest/completion files")
        files = _files(staging)
        _require(bool(files), "export callback produced no files")
        result = {"schema_version": SCHEMA, "status": "EXPORTED", "kind": KIND,
                  "checkpoint_cycle": cycle, "identity": identity, "model": model,
                  "world_size": world, "export": metadata, "files": files,
                  "quality_validation": "NOT_RUN"}
        _json(staging / "export.json", result)
        (staging / "_SUCCESS").write_text(file_sha256(staging / "export.json") + "\n")
        os.rename(staging, output)
        return result
    try:
        result = distributed.primary(publish, "inference export publication")
    except BaseException:
        if world == 1:
            shutil.rmtree(staging, ignore_errors=True)
        raise
    return distributed.gather(result)[0]


def read_export(path):
    """Admit completeness, schema, every file hash and bounds before imports."""
    root = Path(path).expanduser().resolve()
    manifest, marker = root / "export.json", root / "_SUCCESS"
    _require(manifest.is_file() and marker.is_file() and not manifest.is_symlink()
             and not marker.is_symlink(), "inference export is not complete")
    expected = marker.read_text().strip()
    _require(_sha(expected) and expected == file_sha256(manifest), "export manifest hash changed")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, f"duplicate export manifest key: {key}")
            result[key] = value
        return result
    value = json.loads(manifest.read_text(), object_pairs_hook=unique)
    _require(file_sha256(manifest) == expected, "export manifest hash changed during read")
    _require(isinstance(value, dict) and set(value) == {
        "schema_version", "status", "kind", "checkpoint_cycle", "identity", "model", "world_size",
        "export", "files", "quality_validation"}, "malformed inference export schema")
    _require(value["schema_version"] == SCHEMA and value["status"] == "EXPORTED"
             and value["kind"] == KIND and value["quality_validation"] == "NOT_RUN",
             "unsupported inference export schema or kind")
    _require(type(value["checkpoint_cycle"]) is int and value["checkpoint_cycle"] >= 0
             and type(value["world_size"]) is int and value["world_size"] >= 1
             and isinstance(value["identity"], dict), "invalid export provenance")
    model = value["model"]
    _require(isinstance(model, dict) and set(model) == {"factory", "source", "parameters", "metadata", "modules"}
             and isinstance(model["factory"], str)
             and all(isinstance(model[key], dict) for key in ("source", "parameters", "metadata", "modules")),
             "malformed exported model schema")
    _require(set(model["source"]) == {"reference", "sha256"}
             and model["source"]["reference"] == model["factory"] and _sha(model["source"]["sha256"])
             and "device" not in model["parameters"], "invalid exported model source or parameters")
    _require(value["identity"].get("model") == model["source"]
             and value["identity"].get("model_metadata") == model["metadata"], "exported model metadata disagrees with identity")
    metadata, files = value["export"], value["files"]
    _require(isinstance(metadata, dict) and isinstance(files, dict) and files, "malformed export files or format")
    if metadata.get("format") == "torch_state_dicts":
        _require(set(metadata) == {"format", "roles"} and metadata["roles"] == sorted(model["modules"])
                 and set(files) == {"weights.pt"} and bool(model["modules"]), "invalid named-module export format")
    else:
        _require(set(metadata) == {"format", "metadata"} and metadata["format"] == "adapter",
                 "unsupported custom export format")
    digest(value)
    for name, sha in files.items():
        item = _file(root, name)
        _require(_sha(sha) and item.is_file() and file_sha256(item) == sha, f"export file hash changed: {name}")
    _require(_files(root) == files, "export has unrecorded files or a file hash changed")
    return root, value


def restore_modules(components, root, manifest):
    """Validate all named state trees before mutating any module."""
    import torch
    _require(digest(module_layout(components.state)) == digest(manifest["model"]["modules"]),
             "export model module layout changed")
    if manifest["export"]["format"] == "adapter":
        _require(components.export is not None and components.load_export is not None,
                 "custom export loading requires an explicit load_export callback")
        components.load_export(components.state, root, manifest["export"]["metadata"])
        _require(digest(module_layout(components.state)) == digest(manifest["model"]["modules"]),
                 "custom loader changed model layout")
        return
    _require(components.export is None, "export format differs from current model adapter")
    path = _file(root, "weights.pt")
    expected = manifest["files"]["weights.pt"]
    _require(file_sha256(path) == expected, "export weights hash changed before loading")
    try:
        weights = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError("invalid safe exported weights") from exc
    _require(file_sha256(path) == expected, "export weights hash changed during loading")
    models = modules(components.state)
    _require(isinstance(weights, dict) and set(weights) == set(models), "export module names changed")
    for name, module in models.items():
        _require(_tree_layout(weights[name]) == _tree_layout(module.state_dict()), f"export module state layout changed: {name}")
    _finite_weights(weights)
    for name, module in models.items():
        module.load_state_dict(weights[name], strict=True)


def decoded_video(decoded, *, required=True):
    """No latent reshape, resizing, guessed normalization or invented decoder."""
    from collections.abc import Mapping

    if not required and (not isinstance(decoded, Mapping) or "video" not in decoded):
        return None
    import torch
    video = decoded.get("video") if isinstance(decoded, Mapping) else None
    _require(isinstance(video, torch.Tensor) and video.dtype == torch.uint8 and video.ndim == 4
             and min(video.shape) > 0 and video.shape[-1] in (3, 4),
             "generation needs an actual decoder returning explicit THWC uint8 RGB/RGBA video")
    return video


def latent_outputs(generated):
    """Explicit named floating tensors and exact declared layouts, not media.

    Adapters return ``{name: Tensor, ..., 'latent_schema': {name: spec, ...}}``.
    Each spec declares layout axes, shape and Torch dtype; no decoder is guessed.
    """
    import torch
    schema = generated.get("latent_schema") if isinstance(generated, dict) else None
    _require(isinstance(schema, dict) and schema and set(generated) == set(schema) | {"latent_schema"},
             "latent diagnostic needs nonempty explicitly named tensors and latent_schema")
    for name, spec in schema.items():
        tensor = generated[name]
        _require(isinstance(name, str) and bool(name.strip()) and name != "latent_schema"
                 and isinstance(tensor, torch.Tensor) and tensor.layout == torch.strided
                 and tensor.is_floating_point() and tensor.numel() > 0 and bool(torch.isfinite(tensor).all()),
                 "latent diagnostic needs finite nonempty named floating tensors")
        _require(isinstance(spec, dict) and set(spec) == {"layout", "shape", "dtype"}
                 and isinstance(spec["layout"], str) and spec["layout"].isascii()
                 and spec["layout"].isalpha() and spec["layout"].isupper()
                 and len(spec["layout"]) == tensor.ndim and len(set(spec["layout"])) == tensor.ndim
                 and isinstance(spec["shape"], list) and all(type(n) is int and n > 0 for n in spec["shape"])
                 and spec["shape"] == list(tensor.shape) and spec["dtype"] == str(tensor.dtype),
                 "latent diagnostic layout/shape/dtype declaration disagrees with tensors")
    return schema


def decoded_media(decoded, *, required_kind=None):
    """Validate explicit decoded media; layouts are model adapters' responsibility."""
    import torch
    from collections.abc import Mapping
    _require(isinstance(decoded, Mapping), 'generation needs an actual decoder returning an explicit media mapping')
    result = {}
    if 'video' in decoded:
        result['video'] = decoded_video(decoded)
    if 'image' in decoded:
        image = decoded['image']
        _require(isinstance(image, torch.Tensor) and image.dtype == torch.uint8
                 and image.ndim == 3 and min(image.shape) > 0 and image.shape[-1] in (3, 4),
                 'image decoder must return HWC uint8 RGB/RGBA')
        result['image'] = image
    if 'audio' in decoded:
        audio = decoded['audio']
        _require(isinstance(audio, Mapping) and set(audio) == {'samples', 'sample_rate'},
                 'audio decoder needs samples and sample_rate')
        samples, rate = audio['samples'], audio['sample_rate']
        _require(isinstance(samples, torch.Tensor) and samples.is_floating_point()
                 and samples.ndim == 2 and min(samples.shape) > 0 and type(rate) is int and rate > 0
                 and bool(torch.isfinite(samples).all()) and bool((samples.abs() <= 1).all()),
                 'audio decoder must return finite normalized TC floating samples and positive integer sample_rate')
        result['audio'] = audio
    required = {'audio_video': {'audio', 'video'}}.get(required_kind, {required_kind} if required_kind else set())
    _require(required <= result.keys(), f'generation needs an actual decoder returning required {required_kind} media')
    return result


def write_generation(root, values, *, metadata, output_kind=None):
    """Write one already-validated, CPU-materialized sample, without resampling."""
    import torch
    latents = latent_outputs(values["generated"]) if output_kind == "latents" else None
    media = ({} if latents is not None or (output_kind is None and not isinstance(values['decoded'], dict))
             else decoded_media(values['decoded'], required_kind=output_kind))
    video = media.get('video')
    if video is not None:
        from PIL import Image
        frames = root / "frames"
        frames.mkdir()
        for index, frame in enumerate(video):
            Image.fromarray(frame.contiguous().numpy()).save(frames / f"frame-{index:06d}.png")
        video_metadata = {"status": "DECODED_DIAGNOSTIC", "layout": "THWC", "dtype": "uint8", "shape": list(video.shape),
                          "frames": str(root / "frames"), "interpretation": "DECODED_DIAGNOSTIC_NOT_QUALITY_VALIDATION"}
    else:
        video_metadata = {"status": "NOT_AVAILABLE", "reason": "ADAPTER_RETURNED_NO_EXPLICIT_VIDEO"}
    torch.save(values, root / "generation.pt")
    result = {**metadata, "video": video_metadata,
              "artifact": str(root / "generation.pt"), "quality_validation": "NOT_RUN"}
    if 'image' in media:
        from PIL import Image
        image = media['image']
        Image.fromarray(image.contiguous().numpy()).save(root / 'image.png')
        result['image'] = {'path': str(root / 'image.png'), 'layout': 'HWC', 'shape': list(image.shape)}
    if 'audio' in media:
        import wave
        import numpy as np
        audio = media['audio']
        samples = audio['samples']
        with wave.open(str(root / 'audio.wav'), 'wb') as stream:
            stream.setnchannels(samples.shape[1])
            stream.setsampwidth(2)
            stream.setframerate(audio['sample_rate'])
            # NumPy cannot represent BF16, and FP16 rounds 32767 to 32768.
            # Quantize in FP32 for every admitted floating input dtype.
            pcm = samples.detach().float().cpu().numpy()
            stream.writeframes(np.rint(pcm * 32767).astype('<i2').tobytes())
        result['audio'] = {'path': str(root / 'audio.wav'), 'layout': 'TC', 'shape': list(samples.shape),
                           'sample_rate': audio['sample_rate'], 'encoding': 'PCM16'}
    if output_kind:
        result['output_kind'] = output_kind
    if latents is not None:
        result["output_kind"] = "latents"
        result["latents"] = {"schema_version": "distillschedule.latent_outputs.v1", "tensors": latents,
                             "interpretation": "LATENT_DIAGNOSTIC_NOT_MEDIA_QUALITY"}
    _json(root / "generation.json", result)
    return result
