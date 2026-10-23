"""Portable CURRENT full base + unmerged Student, original cached AV rollout.

CPU1/FP32 LATENT DIAGNOSTIC only. No training owner, discriminator, optimizer,
old assets, decoder, model downloads, LoRA merge, or alternative sampler.
Discovery stays Torch-free; execution explicitly admits the original source.
"""
from __future__ import annotations

from copy import copy, deepcopy
import inspect
import json
import math
from pathlib import Path
from types import SimpleNamespace

from distillschedule.workloads.best_known.registry import file_sha256
from .adapters import InferenceProjection, ModelComponents
from .h3_cached_assets import (BACKEND, PRECISION, EVIDENCE_KINDS, require,
                               read_json, checked_pin, runtime_identity,
                               validate_geometry, validate_topology)

SCHEMA = "distillschedule.h3_cached_current_student_inference.v1"
FACTORY = "distillschedule.training.h3_cached_inference:build_inference_model"
SCOPE = "CPU1_FP32_ORIGINAL_CACHED_AV_LATENT_DIAGNOSTIC"


def _json(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def content_identity():
    """Actual packaged source16 and current admission helpers, without paths."""
    from distillschedule.reference import diffsynth_h3_cached as provider
    from distillschedule.data import h3_cached_inputs
    from . import h3_cached_assets
    return dict(source16=provider.source_identity(),
        helpers={name: file_sha256(Path(module.__file__)) for name, module in (
            ("provider", provider), ("cached_inputs", h3_cached_inputs), ("assets", h3_cached_assets))})


def content_runtime():
    return {name: {k: v for k, v in value.items() if k != "path"} if isinstance(value, dict) else value
            for name, value in runtime_identity().items()}


def _latent_schema(geometry):
    return {name: dict(layout=layout, shape=deepcopy(geometry[shape]), dtype="torch.float32")
            for name, layout, shape in (("video_latents", "BCTHW", "video_shape"),
                                       ("audio_latents", "SCT", "audio_shape"))}


def _metadata(parameters):
    return dict(schema_version=SCHEMA, scope=SCOPE, parameters=deepcopy(parameters),
        generator_selection="CURRENT_FULL_BASE_UNMERGED_STUDENT_NOT_EMA",
        current_av_values="GEOMETRY_TEMPLATES_ONLY_ORIGINAL_ROLLOUT_SAMPLES_NOISE",
        conditioning="ACTUAL_CACHED_PROMPT_AND_CLEAN_PREVIOUS_AV",
        evidence_kind=parameters["evidence_kind"], quality_validation="NOT_RUN", claimable_speedup=False,
        interpretation="LATENT_DIAGNOSTIC_NOT_MEDIA_QUALITY", decoded_media="NOT_IMPLEMENTED_NO_DECODER",
        latent_schema=_latent_schema(parameters["geometry"]))


def _config(parameters):
    # H3Adapter and make_schedule need only these original scalar attributes.
    # No optimizer, discriminator, GAN clock, sigma-training or restart config.
    return SimpleNamespace(**parameters["lora"], **parameters["schedule"])


def _drop_fake(adapter):
    """Only a separately constructed inference object may be pruned in place."""
    for _, layer in adapter.layers:
        del layer.adapters["fake_score"]
    adapter.select("student")


def _admit(parameters):
    """Fail closed on source/runtime/modes/capacity before real allocation."""
    require(isinstance(parameters, dict) and set(parameters) <= {"schema_version", "source", "runtime",
        "architecture", "lora", "schedule", "geometry", "budget", "evidence_kind", "seed", "backend",
        "precision", "device"}, "unsupported inference factory parameters")
    require(parameters.get("device", "cpu") == "cpu", "inference CPU FP32 only; no CUDA substitution")
    p = _json({k: v for k, v in parameters.items() if k != "device"})
    require(set(p) == {"schema_version", "source", "runtime", "architecture", "lora", "schedule", "geometry",
        "budget", "evidence_kind", "seed", "backend", "precision"} and p["schema_version"] == SCHEMA,
        "complete explicit inference schema required")
    validate_topology()
    require(p["backend"] == BACKEND and p["precision"] == PRECISION
            and p["source"] == content_identity(), "inference original source/backend/precision changed")
    require(p["runtime"] == content_runtime(), "inference dependency/runtime identity changed")
    require(type(p["seed"]) is int and 0 <= p["seed"] < 2**63 and p["evidence_kind"] in EVIDENCE_KINDS,
            "inference seed/evidence kind changed")
    budget = p["budget"]
    limits = dict(max_parameters=10_000_000, max_loaded_bytes=512_000_000, max_input_elements=10_000_000)
    require(isinstance(budget, dict) and set(budget) == set(limits)
            and all(type(budget[k]) is int and 0 < budget[k] <= limit for k, limit in limits.items()),
            "explicit bounded inference parameter/byte/input capacity required")
    validate_geometry(p["geometry"])
    elements = sum(math.prod(s) for s in p["geometry"].values())
    require(elements <= budget["max_input_elements"] and 4 * elements <= budget["max_loaded_bytes"],
            "inference input geometry exceeds CPU capacity")
    # These are the original canonical inference/LoRA constants, not training
    # configuration overrides or the original scaled-B LoRA-only export.
    require(p["lora"] == dict(lora_rank=64, lora_alpha=64,
        lora_targets="attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2"), "original unmerged LoRA64 contract changed")
    require(type(p["lora"]["lora_rank"]) is int, "original integer LoRA rank required")
    require(p["schedule"] == dict(num_inference_steps=8, video_shift=2.22, audio_shift=2.22)
            and type(p["schedule"]["num_inference_steps"]) is int,
            "original eight-step continuous schedule changed")
    from distillschedule.reference import diffsynth_h3_cached as provider
    api = provider.load_cpu(backend=BACKEND)
    signature = inspect.signature(api.model.MiniMaxH3DiT.__init__)
    architecture = p["architecture"]
    require(isinstance(architecture, dict) and set(architecture) == set(signature.parameters) - {"self", "kwargs"},
            "complete original architecture required; no unknown/implicit fields")
    require(all(type(architecture[k]) is int and 0 < architecture[k] <= 16384 for k in (
        "hidden_size", "num_attention_heads", "attention_head_dim", "ffn_hidden_size", "timestep_input_dim",
        "time_embed_hidden_size", "time_embed_dim", "adaln_out_features", "final_adaln_out_features", "rope_inv_freq_len"))
        and all(type(architecture[k]) is int and 0 < architecture[k] <= 8
                for k in ("num_layers", "token_refiner_num_layers")), "unbounded inference architecture")
    require(all(type(architecture[k]) is int and architecture[k] == n for k, n in (
        ("latents_dim", 24), ("audio_latents_dim", 32), ("text_dim", 5120)))
        and architecture["patch_size"] == [1, 2, 2] and all(type(n) is int for n in architecture["patch_size"])
        and architecture["timestep_input_dim"] % 2 == 0
        and 6 * architecture["rope_inv_freq_len"] <= architecture["attention_head_dim"]
        and architecture["adaln_out_features"] == 18 * architecture["hidden_size"]
        and architecture["final_adaln_out_features"] == 2 * architecture["hidden_size"]
        and all(type(architecture[k]) in (float, int) and 0 < architecture[k] < 1
                for k in ("norm_eps", "qk_norm_eps", "final_norm_eps")), "unsupported native inference architecture")
    import torch
    from .checkpoint import rng_state, restore_rng
    before = rng_state()
    try:
        with torch.device("meta"):
            probe = api.dmd2.H3Adapter(api.model.MiniMaxH3DiT(**architecture), _config(p))
        transient = sum(t.numel() for t in probe.parameters())
        _drop_fake(probe)
        retained = sum(t.numel() for t in probe.parameters())
        require(not list(probe.buffers()) and retained > 0 and transient <= budget["max_parameters"]
            and 4 * (transient + retained + elements) <= budget["max_loaded_bytes"],
            "inference full base/Student/strict-load capacity exceeded")
    finally:
        restore_rng(before)
    return p, api


def _student_view(adapter, api):
    """Clone structure, NEVER tensors, modes, flags, role selection or forwards.

    Every module keeps its exact original class/methods/scalars. The view has
    independent registration dictionaries; Student/base Parameters and buffers
    are the actual restored owner's objects. No mutating module API is called.
    """
    memo = {}

    def clone(module):
        if id(module) not in memo:
            view = copy(module)
            memo[id(module)] = view
            view._parameters, view._buffers = dict(module._parameters), dict(module._buffers)
            view._modules = {n: clone(m) if m is not None else None for n, m in module._modules.items()}
            view._non_persistent_buffers_set = set(module._non_persistent_buffers_set)
            if isinstance(module, api.dmd2.DualLoRALinear):
                del view.adapters._modules["fake_score"]
        return memo[id(module)]

    view = clone(adapter)
    view.layers = [(name, view.dit.get_submodule(name)) for name, _ in adapter.layers]
    return view


def project_current_student(components, effective_training_parameters):
    """Export-only structure view of the restored CURRENT unmerged Student."""
    from .h3_cached_owner import H3CachedOwner
    require(set(components.state) == {"native_update"} and type(components.state["native_update"]) is H3CachedOwner,
            "explicit restored original H3CachedOwner required")
    owner = components.state["native_update"]
    owner.validate_checkpoint_boundary()
    require(type(effective_training_parameters.get("seed")) is int, "explicit effective factory seed required")
    architecture = read_json(checked_pin(owner.assets["base"]["config"]))
    signature = inspect.signature(owner.api.model.MiniMaxH3DiT.__init__)
    architecture = {k: architecture.get(k, v.default) for k, v in signature.parameters.items()
                    if k not in ("self", "kwargs")}
    p = _json(dict(schema_version=SCHEMA, source=content_identity(), runtime=content_runtime(),
        architecture=architecture,
        lora={k: getattr(owner.config, k) for k in ("lora_rank", "lora_alpha", "lora_targets")},
        schedule={k: getattr(owner.config, k) for k in ("num_inference_steps", "video_shift", "audio_shift")},
        geometry=owner.assets["geometry"], budget={k: owner.assets["budget"][k] for k in (
            "max_parameters", "max_loaded_bytes", "max_input_elements")}, backend=BACKEND, precision=PRECISION,
        evidence_kind=owner.assets["evidence_kind"], seed=effective_training_parameters["seed"]))
    _admit(p)
    student = _student_view(owner.adapter, owner.api)
    # Deliberately no generate/decode callback: common export only reads state.
    projected = ModelComponents(state={"student": student}, metadata=_metadata(p)).validate()
    return InferenceProjection(projected, FACTORY, p).validate()


def _weights_restored(module, incompatible_keys):
    """Completion guard after the common ALL-tensor staged strict restore.

    This serialization-only post hook does not replace any original numerical
    method and exists only on the separately constructed inference object.
    """
    import torch
    require(not incompatible_keys.missing_keys and not incompatible_keys.unexpected_keys
            and all(t.device.type == "cpu" and t.dtype == torch.float32 and bool(t.isfinite().all())
                    for t in module.state_dict().values()), "complete finite strict inference weights required")
    module._inference_weights_restored = True


def build_inference_model(parameters):
    """Construct an original strict-load target; random initialization is unusable."""
    import torch
    p, api = _admit(parameters)
    with torch.device("cpu"):
        student = api.dmd2.H3Adapter(api.model.MiniMaxH3DiT(**p["architecture"]), _config(p))
    _drop_fake(student)
    student.requires_grad_(False).eval()
    student._inference_weights_restored = False
    student.register_load_state_dict_post_hook(_weights_restored)
    return ModelComponents(state={"student": student, "inference": deepcopy(p)}, metadata=_metadata(p),
        generate=generate, admit_training=refuse_training).validate()


def refuse_training(*args, **kwargs):
    raise ValueError("H3 inference-only factory refuses training/checkpoint evaluation; use exported weights generation")


def generate(state, feeds, parameters):
    """Actual original parser + original make_schedule + unchanged eight-step rollout."""
    import torch
    from distillschedule.reference import diffsynth_h3_cached as provider
    from distillschedule.data.h3_cached_inputs import decode_row
    require(not parameters, "inference has no unbound sampler/conditioning overrides")
    validate_topology()
    p, student = state["inference"], state["student"]
    require(p["source"] == content_identity() and p["runtime"] == content_runtime(),
            "inference source/runtime changed")
    require(student._inference_weights_restored, "strict exported weights must be restored before any forward")
    require(not student.training and all(not t.requires_grad and t.device.type == "cpu" and t.dtype == torch.float32
            and bool(t.isfinite().all()) for t in student.parameters()), "separately frozen finite FP32 Student required")
    require(isinstance(feeds, dict) and isinstance(feeds.get("absolute_step"), torch.Tensor)
            and feeds["absolute_step"].shape == (1,), "actual prepared original cached AV tensor row required")
    row = decode_row(feeds, absolute_step=feeds["absolute_step"].item(), geometry=p["geometry"])
    api = provider.load_cpu(backend=BACKEND)
    inputs, negative = api.dmd2.prepare_batch(row["raw"], 1.0)
    require(negative is None, "original CFG1 conditioning required")
    schedule = api.dmd2.make_schedule(_config(p), device="cpu")
    # Current AV VALUES are geometry templates only. Original rollout samples
    # randn_like; only the actual prompt/previous AV enter native conditioning.
    with torch.no_grad():
        (video, audio), _ = api.dmd2.rollout(student,
            (inputs["input_latents"], inputs["audio_input_latents"]), inputs, schedule)
    require(list(video.shape) == p["geometry"]["video_shape"] and list(audio.shape) == p["geometry"]["audio_shape"]
            and all(t.dtype == torch.float32 and t.device.type == "cpu" and bool(t.isfinite().all())
                    for t in (video, audio)), "original native AV latent output changed/nonfinite")
    return dict(video_latents=video, audio_latents=audio, latent_schema=_latent_schema(p["geometry"]))
