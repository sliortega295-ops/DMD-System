"""Explicit, bounded local assets for the original cached-AV CPU reference.

No directory discovery, remote loading, random base, or precision conversion.
This module stays Torch-free until an execution/asset admission is requested.
"""
from __future__ import annotations

from dataclasses import asdict, fields
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import platform

from distillschedule.workloads.best_known.registry import file_sha256, fingerprint

SCHEMA = "distillschedule.h3_cached_cpu_assets.v1"
FIXTURE = "FULL_ORIGINAL_TINY_CPU_FIXTURE_NOT_PRETRAINED_OR_QUALITY"
REAL = "LOCAL_PRETRAINED_CACHED_AV_UNVERIFIED_QUALITY"
EVIDENCE_KINDS = (FIXTURE, REAL)
BACKEND = "torch-dense-cpu"
PRECISION = "CPU_FP32_NO_CAST"


def require(ok, message):
    if not ok:
        raise ValueError("H3 cached CPU reference: " + message)


def read_json(path):
    def pairs(rows):
        value = {}
        for key, item in rows:
            require(key not in value, "duplicate JSON key: " + key)
            value[key] = item
        return value
    value = json.loads(Path(path).read_bytes(), object_pairs_hook=pairs)
    fingerprint(value)
    return value


def local_file(value):
    require(isinstance(value, str) and "://" not in value and Path(value).is_absolute(),
            "explicit absolute local file required")
    path = Path(value)
    require(path.resolve() == path and path.is_file(), "missing, noncanonical or symlink file: " + value)
    return path


def checked_pin(value):
    require(isinstance(value, dict) and set(value) == {"path", "bytes", "sha256"}, "exact file pin required")
    path = local_file(value["path"])
    require(type(value["bytes"]) is int and value["bytes"] > 0 and path.stat().st_size == value["bytes"]
            and file_sha256(path) == value["sha256"], "file pin changed: " + str(path))
    return path


def tensor_content(value):
    """Content identities, not serialized-pickle identities or Tensor equality."""
    import torch
    if isinstance(value, torch.Tensor):
        require(value.device.type == "cpu" and value.layout == torch.strided,
                "dense CPU tensor required")
        t = value.detach().contiguous()
        require(not t.is_floating_point() or bool(t.isfinite().all()), "nonfinite tensor")
        return dict(shape=list(t.shape), dtype=str(t.dtype),
                    sha256=hashlib.sha256(t.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())
    if isinstance(value, dict):
        return {k: tensor_content(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return {"sequence": type(value).__name__, "items": [tensor_content(v) for v in value]}
    require(value is None or type(value) in (str, bool, int, float), "unsupported content identity leaf")
    return {"type": type(value).__name__, "value": value}


def content_digest(value):
    return fingerprint(tensor_content(value))


def runtime_identity():
    result = {"python": platform.python_version()}
    # This closure deliberately does not import Helios, PEFT, Diffusers or Hub.
    for name in ("torch", "numpy", "safetensors", "einops"):
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        result[name] = dict(version=importlib.metadata.version(name), imported_version=str(module.__version__),
                            path=str(path), entry_sha256=file_sha256(path))
    import torch
    result["torch_cpu"] = dict(threads=torch.get_num_threads(), default_dtype=str(torch.get_default_dtype()))
    return result


def validate_topology():
    import torch
    require(all(int(os.environ.get(key, default)) == expected for key, default, expected in
                (("WORLD_SIZE", "1", 1), ("LOCAL_WORLD_SIZE", "1", 1),
                 ("RANK", "0", 0), ("LOCAL_RANK", "0", 0))), "CPU1/world1 topology required")
    require(not torch.distributed.is_initialized() and not torch.cuda.is_initialized(),
            "distributed groups or initialized CUDA are unsupported")
    require(torch.get_default_dtype() == torch.float32 and torch.get_num_threads() == 1,
            "explicit single-thread FP32 CPU runtime required")


def validate_geometry(value):
    require(isinstance(value, dict) and set(value) == {"video_shape", "audio_shape", "prompt_shape",
            "reference_video_shape", "reference_audio_shape"}, "explicit current/previous AV geometry required")
    for key, shape in value.items():
        require(isinstance(shape, list) and all(type(n) is int and n > 0 for n in shape), "invalid " + key)
    for key in ("video_shape", "reference_video_shape"):
        s = value[key]
        require(len(s) == 5 and s[:2] == [1, 24] and s[-2] % 2 == s[-1] % 2 == 0,
                "native batch1/channel24/even video geometry required")
    for key in ("audio_shape", "reference_audio_shape"):
        s = value[key]
        require(len(s) == 3 and s[:2] == [2, 32], "native stereo/channel32 audio geometry required")
    require(len(value["prompt_shape"]) == 2 and value["prompt_shape"][-1] == 5120,
            "original prompt width5120 required")
    return value


def _configuration(api, value):
    cfg = value["dmd_config"]
    require(isinstance(cfg, dict) and set(cfg) == {f.name for f in fields(api.dmd2.DMD2Config)},
            "complete explicit original DMD2Config required; no implicit F5/defaults")
    config = api.dmd2.DMD2Config(**cfg)
    require(config.hidden_channels <= 1024 and config.num_layers <= 8, "CPU discriminator construction capacity exceeded")
    expected = asdict(api.dmd2.DMD2Config(critic_updates=1, hidden_channels=config.hidden_channels,
        num_layers=config.num_layers, gan_start_step=config.gan_start_step))
    require(cfg == expected, "fixed original 8-step/G1F1/LoRA64/sigma/CFG1 protocol changed")
    require(value["protocol"] == "CANONICAL_GAN500" and config.gan_start_step == 500 or
            value["protocol"] == "GAN_START_2_TEST_FIXTURE" and config.gan_start_step == 2
            and value["evidence_kind"] == FIXTURE, "GAN-start protocol/fixture label changed")
    return config


def _inspect_weights(spec, evidence_kind):
    from safetensors import safe_open, SafetensorError
    require(isinstance(spec, dict) and set(spec) == {"config", "weights"}
            and isinstance(spec["weights"], list) and spec["weights"], "explicit full base config/shards required")
    checked_pin(spec["config"])
    shapes, paths = {}, []
    for pin in spec["weights"]:
        path = checked_pin(pin)
        require(path.suffix == ".safetensors" and str(path) not in paths, "unique full safetensors shards required")
        try:
            with safe_open(path, framework="pt", device="cpu") as handle:
                require((handle.metadata() or {}).get("evidence_kind") == evidence_kind,
                        "payload evidence label differs; fixture cannot become pretrained")
                for name in handle.keys():
                    view = handle.get_slice(name)
                    require(name not in shapes and view.get_dtype() == "F32", "duplicate keys or non-FP32 full base")
                    shapes[name] = tuple(view.get_shape())
        except SafetensorError as exc:
            raise ValueError("invalid full base safetensors") from exc
        paths.append(str(path))
    return shapes


def admit_assets(parameters):
    """Inspect all pins/layout/capacity before allocating real model parameters."""
    from distillschedule.reference import diffsynth_h3_cached as provider
    require(isinstance(parameters, dict) and set(parameters) <= {"asset_manifest", "seed", "device"},
            "factory accepts only asset_manifest/device/seed, never data/tensors")
    require(parameters.get("device", "cpu") == "cpu", "CPU only; no CUDA/backend substitution")
    require(type(parameters.get("seed", 0)) is int and 0 <= parameters.get("seed", 0) < 2**63, "invalid seed")
    path = local_file(parameters.get("asset_manifest"))
    value = read_json(path)
    require(isinstance(value, dict) and set(value) == {"schema_version", "evidence_kind", "source", "runtime",
        "base", "dmd_config", "geometry", "budget", "backend", "precision", "protocol"}, "asset manifest fields changed")
    require(value["schema_version"] == SCHEMA and value["evidence_kind"] in EVIDENCE_KINDS
            and value["backend"] == BACKEND and value["precision"] == PRECISION, "asset schema/backend/precision changed")
    require(value["source"] == provider.source_identity(), "original source/provider identity changed")
    api = provider.load_cpu(backend=BACKEND)
    validate_topology()
    require(value["runtime"] == runtime_identity(), "dependency/runtime identity changed")
    config = _configuration(api, value)
    validate_geometry(value["geometry"])
    budget = value["budget"]
    limits = dict(max_parameters=10_000_000, max_loaded_bytes=512_000_000,
                  max_cycles=10_000, max_input_elements=10_000_000)
    require(isinstance(budget, dict) and set(budget) == set(limits)
            and all(type(budget[k]) is int and 0 < budget[k] <= limit for k, limit in limits.items()),
            "explicit bounded CPU parameter/byte/cycle/input capacity required")
    require(sum(math.prod(s) for s in value["geometry"].values()) <= budget["max_input_elements"],
            "input geometry exceeds CPU capacity")
    shapes = _inspect_weights(value["base"], value["evidence_kind"])
    require(sum(math.prod(s) for s in shapes.values()) <= budget["max_parameters"], "full base parameter capacity exceeded")
    require(sum(p["bytes"] for p in value["base"]["weights"]) <= budget["max_loaded_bytes"], "base payload capacity exceeded")
    architecture = read_json(checked_pin(value["base"]["config"]))
    signature = inspect.signature(api.model.MiniMaxH3DiT.__init__)
    require(isinstance(architecture, dict) and set(architecture) <= set(signature.parameters) - {"self", "kwargs"},
            "unknown original architecture field")
    effective = {k: architecture.get(k, p.default) for k, p in signature.parameters.items()
                 if k not in ("self", "kwargs")}
    require(effective["latents_dim"] == 24 and effective["audio_latents_dim"] == 32
            and list(effective["patch_size"]) == [1, 2, 2] and effective["text_dim"] == 5120,
            "original AV channel/patch/prompt architecture required")
    require(all(type(effective[k]) is int and 0 < effective[k] <= 16384 for k in (
        "hidden_size", "num_attention_heads", "attention_head_dim", "ffn_hidden_size", "timestep_input_dim",
        "time_embed_hidden_size", "time_embed_dim", "adaln_out_features", "final_adaln_out_features", "rope_inv_freq_len"))
        and all(type(effective[k]) is int and 0 < effective[k] <= 8 for k in ("num_layers", "token_refiner_num_layers")),
        "CPU architecture construction capacity exceeded")
    import torch
    from .checkpoint import rng_state, restore_rng
    before = rng_state()
    try:
        with torch.device("meta"):
            model = api.model.MiniMaxH3DiT(**architecture)
            head = api.dmd2.VelocityDiscriminator(config)
        expected = {k: tuple(t.shape) for k, t in model.state_dict().items()}
        require(shapes == expected, "missing/extra full base keys or architecture/layout changed; no random fallback")
        targets = config.lora_targets.split(",")
        lora_count = sum(2 * config.lora_rank * (m.in_features + m.out_features)
                         for name, m in model.named_modules() if isinstance(m, torch.nn.Linear)
                         and any(name == t or name.endswith("." + t) for t in targets))
        base_count = sum(p.numel() for p in model.parameters())
        mutable_count = lora_count + sum(p.numel() for p in head.parameters())
        require(base_count + mutable_count <= budget["max_parameters"]
                and 4 * (base_count + 3 * mutable_count) <= budget["max_loaded_bytes"],
                "model/LoRA/head/Adam capacity exceeded")
        require(not list(model.buffers()), "unadmitted original nonpersistent buffers")
    finally:
        restore_rng(before)
    return path, value, config, model


def load_base(model, assets):
    """Materialize the admitted actual meta model with every immutable base byte."""
    import torch
    from safetensors.torch import load_file
    weights = {}
    for pin in assets["base"]["weights"]:
        weights.update(load_file(str(checked_pin(pin)), device="cpu"))
    require(all(t.dtype == torch.float32 and bool(t.isfinite().all()) for t in weights.values()), "nonfinite full base")
    model.load_state_dict(weights, strict=True, assign=True)
    require(all(p.device.type == "cpu" and p.dtype == torch.float32 for p in model.parameters()), "unmaterialized base")
    for pin in [assets["base"]["config"], *assets["base"]["weights"]]:
        checked_pin(pin)
    return model
