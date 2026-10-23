"""CURRENT full original WanModel + fresh native FewStepStreamer, CPU1 FP32.

No Teacher, Critic, Adam, training RNG/cursor, cached rollout, weight download or
alternative sampler. Optional decoding is a separately pinned original codec.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from distillschedule.workloads.best_known.registry import file_sha256
from .adapters import InferenceProjection, ModelComponents
from .livewan_public import (BACKEND, PRECISION, SCOPE, GEOMETRY, EVIDENCE_KINDS,
    require, topology, runtime_identity, _architecture)

SCHEMA = "distillschedule.livewan_cpu1_current_generator_inference.v1"
FACTORY = "distillschedule.training.livewan_inference:build_inference_model"
STREAMER = dict(width=32, height=32, max_frames=278, cache_frames=22, block_frames=3,
    num_steps=4, shift=1.0, sampler="renoise", noisy_context=False, latent_norm=0.0,
    window_frames=None, time_scale=1000.0, cuda_graphs=False, rope_gap=0)


def _json(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def content_runtime():
    return {name: {k: v for k, v in value.items() if k != "path"} if isinstance(value, dict) else value
            for name, value in runtime_identity().items()}


def content_identity():
    from distillschedule.reference import livewan_wan21 as provider
    from distillschedule.data import livewan_inference_inputs, livewan_reference_inputs
    from . import livewan_public, wan21_codec, upstream
    return dict(source=provider.source_identity(), helpers={name: file_sha256(Path(module.__file__))
        for name, module in (("provider", provider), ("inputs", livewan_inference_inputs),
                             ("input_admission", livewan_reference_inputs), ("admission", livewan_public),
                             ("optional_codec", wan21_codec), ("pinned_source_importer", upstream))})


def latent_schema():
    return dict(latents=dict(layout="BCTHW", shape=list(GEOMETRY["latent_shape"]), dtype="torch.float32"))


def _metadata(p):
    return dict(schema_version=SCHEMA, scope=SCOPE, parameters=deepcopy(p),
        generator_selection="CURRENT_FULL_STUDENT_NOT_PRE_G_ROLLOUT_NOT_EMA",
        sampler="ORIGINAL_FEWSTEPSTREAMER_FOUR_STEP_RENOISE_FRESH_NOISE_CLEAN_T0_KV_COMMIT",
        conditioning="NEW_SUPPLIED_CLEAN_WORLD3_6_9_PLUS_RAW_POSITIVE_TEXT_NO_NEGATIVE",
        output="WORLD_PREFIX_PLUS_GENERATED_SUFFIX_LATENT21_81FRAMES_32PX",
        latent_schema=latent_schema(), evidence_kind=p["evidence_kind"], quality_validation="NOT_RUN",
        decoded_media="OPTIONAL_SEPARATELY_PINNED_ORIGINAL_WAN21_CODEC", native_world16="NOT_RUN",
        c51_restart="NOT_IMPLEMENTED", claimable_speedup=False)


def _admit(parameters):
    import torch
    require(isinstance(parameters, dict) and set(parameters) <= {"schema_version", "source", "runtime", "architecture",
        "geometry", "streamer", "budget", "evidence_kind", "seed", "backend", "precision", "device"},
        "unsupported inference factory parameters; no training asset paths")
    require(parameters.get("device", "cpu") == "cpu", "CPU FP32 inference only")
    p = _json({k: v for k, v in parameters.items() if k != "device"})
    require(set(p) == {"schema_version", "source", "runtime", "architecture", "geometry", "streamer", "budget",
        "evidence_kind", "seed", "backend", "precision"} and p["schema_version"] == SCHEMA,
        "complete explicit current-G inference schema required")
    topology()
    require(p["source"] == content_identity() and p["runtime"] == content_runtime()
            and p["backend"] == BACKEND and p["precision"] == PRECISION, "inference source/runtime/backend/precision changed")
    require(p["geometry"] == GEOMETRY and p["streamer"] == STREAMER, "original CPU geometry/streamer policy changed")
    require(type(p["seed"]) is int and 0 <= p["seed"] < 2**63 and p["evidence_kind"] in EVIDENCE_KINDS,
            "inference seed/evidence changed")
    architecture = _architecture(p["architecture"])
    budget = p["budget"]
    limits = dict(max_parameters=10_000_000, max_loaded_bytes=512_000_000, max_attention_elements=8_000_000)
    require(isinstance(budget, dict) and set(budget) == set(limits)
            and all(type(budget[k]) is int and 0 < budget[k] <= v for k, v in limits.items()),
            "explicit bounded inference capacity required")
    d, f, n = (architecture[k] for k in ("dim", "ffn_dim", "num_layers"))
    require(4096*d + n*(12*d*d + 2*d*f) <= budget["max_parameters"]
            and 2*architecture["num_heads"]*(84**2 + 84*512) <= budget["max_attention_elements"],
            "inference construction/attention capacity exceeded")
    from distillschedule.reference import livewan_wan21 as provider
    from .checkpoint import rng_state, restore_rng
    api = provider.load_cpu(backend=BACKEND, precision="float32")
    before = rng_state()
    try:
        with torch.device("meta"):
            probe = api.model.WanModel(**architecture)
        count = sum(t.numel() for t in probe.state_dict().values())
        require(0 < count <= budget["max_parameters"]
                and 4*(2*count + 512*4096 + 16*21*4*4) <= budget["max_loaded_bytes"],
                "full current Student strict-load/input capacity exceeded")
    finally:
        restore_rng(before)
    return p, api


def project_current_generator(components, effective_training_parameters):
    """Read-only live CURRENT Student view; no forwards, copies, modes or RNG."""
    from .livewan_public import LiveWanCPUOwner
    require(set(components.state) == {"native_cycle"} and type(components.state["native_cycle"]) is LiveWanCPUOwner,
            "explicit original complete LiveWan CPU owner required")
    owner = components.state["native_cycle"]
    owner.validate_checkpoint_boundary()
    require(type(effective_training_parameters.get("seed")) is int
            and 0 <= effective_training_parameters["seed"] < 2**63, "explicit effective inference factory seed required")
    # The architecture was admitted at construction and retained as CONTENTS.
    # No training config/weight/data payload is reopened to form this export.
    p = _json(dict(schema_version=SCHEMA, source=content_identity(), runtime=content_runtime(),
        architecture=owner.architecture, geometry=GEOMETRY, streamer=STREAMER,
        budget={k: owner.assets["budget"][k] for k in ("max_parameters", "max_loaded_bytes", "max_attention_elements")},
        evidence_kind=owner.assets["evidence_kind"], seed=effective_training_parameters["seed"],
        backend=BACKEND, precision=PRECISION))
    projected = ModelComponents(state={"student": owner.models["student"]}, metadata=_metadata(p)).validate()
    return InferenceProjection(projected, FACTORY, p).validate()


def _weights_restored(module, incompatible_keys):
    import torch
    require(not incompatible_keys.missing_keys and not incompatible_keys.unexpected_keys
            and all(t.device.type == "cpu" and t.dtype == torch.float32 and bool(t.isfinite().all())
                    for t in module.state_dict().values()), "complete finite strict inference weights required")
    module._inference_weights_restored = True


def build_inference_model(parameters):
    import torch
    p, api = _admit(parameters)
    with torch.device("cpu"):
        student = api.model.WanModel(**p["architecture"])
    student.requires_grad_(False).eval()
    student._inference_weights_restored = False
    student.register_load_state_dict_post_hook(_weights_restored)
    return ModelComponents(state=dict(student=student, inference=p), metadata=_metadata(p),
        generate=generate, decode=decode, admit_training=refuse_training).validate()


def refuse_training(*args, **kwargs):
    raise ValueError("LiveWan inference-only factory refuses training/evaluation; generate from complete exported weights")


def _parameters(parameters):
    require(isinstance(parameters, dict) and set(parameters) <= {"decoder_manifest"},
            "inference has no sampler/conditioning overrides; only an explicit decoder_manifest pin is supported")
    if "decoder_manifest" in parameters:
        from .wan21_codec import admit_manifest
        admit_manifest(parameters["decoder_manifest"])


def generate(state, feeds, parameters):
    import torch
    from distillschedule.data.livewan_inference_inputs import decode_row
    from distillschedule.reference import livewan_wan21 as provider
    topology()
    _parameters(parameters)
    p, student = state["inference"], state["student"]
    require(p["source"] == content_identity() and p["runtime"] == content_runtime(), "inference source/runtime changed")
    require(student._inference_weights_restored, "strict exported weights must be restored before any forward")
    require(not student.training and all(not t.requires_grad and t.device.type == "cpu" and t.dtype == torch.float32
            and bool(t.isfinite().all()) for t in student.parameters()), "separately frozen finite FP32 Student required")
    row = decode_row(feeds, evidence_kind=p["evidence_kind"])
    api = provider.load_cpu(backend=BACKEND, precision="float32")
    cfg = {k: v for k, v in STREAMER.items() if k not in {"rope_gap", "latent_norm"}}
    # Fresh cache and original full native sampler on EVERY call. Session seeds
    # immediately before this callback; do not restore factory/training RNG or
    # reseed individual blocks. Native renoise draws fresh eps at each step.
    streamer = api.stream.FewStepStreamer(student, device="cpu", dtype=torch.float32, **cfg)
    streamer.rope_gap, streamer.latent_norm = 0, 0.0
    with torch.no_grad():
        streamer.set_text(student.text_embedding(row["context"]))
        world = row["world"]
        streamer.set_world(world)
        blocks = [streamer.generate_block() for _ in range((21-world.shape[2])//3)]
        result = torch.cat([world, *blocks], dim=2)
    require(tuple(result.shape) == tuple(GEOMETRY["latent_shape"]) and result.dtype == torch.float32
            and result.device.type == "cpu" and bool(result.isfinite().all()) and streamer.n_frames == 21,
            "native full-step generated latent layout/census changed")
    return dict(latents=result, latent_schema=latent_schema())


def decode(generated, parameters):
    _parameters(parameters)
    require("decoder_manifest" in parameters,
            "no original Wan2.1 decoder admitted; use --output-kind latents or supply an explicit pinned decoder_manifest")
    from .wan21_codec import decode_video
    return decode_video(generated, parameters["decoder_manifest"])
