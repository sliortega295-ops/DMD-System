"""Stream original cached previous/current AV into tensor-only PreparedDataset.

The original parser owns packing and stereo order. Canonical raw records keep
only the previous-anchor grid needed to rebuild it, never parsed Python scalars,
precomputed training noise, timesteps, rollouts, or a fake decoder.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import pickle
import math

from distillschedule.workloads.best_known.registry import fingerprint
from distillschedule.training.h3_cached_assets import (require, read_json, local_file, checked_pin,
    validate_geometry, content_digest, BACKEND, PRECISION, EVIDENCE_KINDS)

SCHEMA = "distillschedule.h3_cached_inputs.v1"
ROW_SCHEMA = "distillschedule.h3_cached_tensor_row.v1"


def producer_identity():
    from distillschedule.reference import diffsynth_h3_cached as provider
    from distillschedule.workloads.best_known.registry import file_sha256
    return dict(source=provider.source_identity(), provider_sha256=file_sha256(Path(provider.__file__)),
                callable="training.prepare_batch", canonicalization=ROW_SCHEMA)


def _id_tensor(value):
    import torch
    require(isinstance(value, str) and len(value) == 64, "SHA256 required")
    try:
        return torch.tensor(list(bytes.fromhex(value)), dtype=torch.uint8)
    except ValueError as exc:
        raise ValueError("invalid SHA256") from exc


def _tensor_id(value):
    import torch
    require(isinstance(value, torch.Tensor) and value.device.type == "cpu" and value.dtype == torch.uint8
            and value.shape == (32,), "row identity must be32 CPU uint8 bytes")
    return bytes(value.tolist()).hex()


def row_identity(source_sha256, content_sha256, absolute_step):
    return fingerprint(dict(schema=ROW_SCHEMA, producer=producer_identity(), source_sha256=source_sha256,
                            content_sha256=content_sha256, absolute_step=absolute_step))


def _leaves(value, *, path=""):
    import torch
    if isinstance(value, torch.Tensor):
        require(value.device.type == "cpu" and value.layout == torch.strided and not value.requires_grad
                and value.grad_fn is None and not value.is_complex() and not value.is_quantized,
                "detached dense real CPU cache tensors required")
        if value.is_floating_point():
            # Original derived positional grid is float64, not a latent cast.
            expected = torch.float64 if path.endswith("packed.img_position_ids") else torch.float32
            require(value.dtype == expected and bool(value.isfinite().all()), "finite FP32 cache required: " + path)
        return
    if isinstance(value, dict):
        for k, v in value.items():
            require(isinstance(k, str), "cache keys must be strings")
            _leaves(v, path=path + "." + k)
    elif isinstance(value, (tuple, list)):
        for i, v in enumerate(value):
            _leaves(v, path=path + "." + str(i))
    else:
        require(value is None or type(value) in (int, float, bool), "unsupported raw cache leaf")
        if type(value) is float:
            import math
            require(math.isfinite(value), "nonfinite raw cache scalar")


def canonical_raw(raw, geometry):
    """Parse first, then prove exact parser equivalence of the tensor-only raw tree."""
    import torch
    from distillschedule.reference import diffsynth_h3_cached as provider
    api = provider.load_cpu(backend=BACKEND)
    parsed, negative = api.dmd2.prepare_batch(raw, 1.0)
    require(negative is None, "CFG1 reference only")
    if isinstance(raw, (tuple, list)):
        require(len(raw) == 3 and all(isinstance(v, dict) for v in raw), "official split-cache triple required")
        shared, positive, minus = raw
    else:
        require(isinstance(raw, dict), "flat tensor cache required")
        shared = positive = raw
        minus = {}
    shared_keys = {"input_latents", "target_video_latents", "audio_input_latents", "target_audio_latents",
        "ref_video_latents", "ref_audio_latents", "ref_visual_anchor", "ref_audio_anchor",
        "imgvid_cond_noise_aug", "audio_cond_noise_aug"}
    positive_keys = {"prompt_embeds", "text_token_tags", "packed"}
    require(set(shared) <= shared_keys | positive_keys and set(positive) <= shared_keys | positive_keys
            and set(minus) <= {"prompt_embeds", "text_token_tags"},
            "unsupported raw cache fields; no silent stripping of conditioning")
    require(not ({"input_latents", "target_video_latents"} <= set(shared))
            and not ({"audio_input_latents", "target_audio_latents"} <= set(shared)), "ambiguous target aliases")
    for part in (shared, positive, minus):
        if part.get("packed") is not None:
            require(isinstance(part["packed"], dict) and set(part["packed"]) <= {
                "img_pos", "audio_pos", "text_pos", "img_position_ids", "token_tags", "cu_seqlens", "seq_len"},
                "unsupported packed conditioning fields; no silent stripping")
    _leaves(raw)
    validate_geometry(geometry)
    require(list(parsed["input_latents"].shape) == geometry["video_shape"]
            and list(parsed["audio_input_latents"].shape) == geometry["audio_shape"]
            and list(parsed["prompt_embeds"].shape) == geometry["prompt_shape"], "current AV/prompt geometry changed")
    require("ref_visual_anchor" in parsed and "ref_audio_anchor" in parsed, "one clean previous AV chunk required")
    ref_shape = api.cache.reference_shape_from_packed(parsed["packed"], len(parsed["ref_visual_anchor"]))
    require(list(ref_shape) == geometry["reference_video_shape"][2:]
            and list(parsed["ref_audio_anchor"].shape) == [2 * geometry["reference_audio_shape"][-1], 32],
            "previous AV geometry changed")
    packed = parsed["packed"]
    canonical = dict(input_latents=parsed["input_latents"], audio_input_latents=parsed["audio_input_latents"],
        prompt_embeds=parsed["prompt_embeds"], ref_visual_anchor=parsed["ref_visual_anchor"],
        ref_audio_anchor=parsed["ref_audio_anchor"], imgvid_cond_noise_aug=torch.tensor(1.0),
        audio_cond_noise_aug=torch.tensor(1.0), text_token_tags=packed["token_tags"][packed["text_pos"]],
        packed={k: packed[k] for k in ("img_pos", "img_position_ids")})
    recovered = api.dmd2.prepare_batch(canonical, 1.0)
    require(content_digest((parsed, negative)) == content_digest(recovered),
            "canonical tensor-only cache changed original parser result")
    from .tensors import tensor_tree
    tensor_tree(canonical)
    return canonical


def _read(pin, geometry):
    import torch
    path = checked_pin(pin)
    require(pin["bytes"] <= 256_000_000, "bounded CPU cache payload required")
    try:
        raw = torch.load(path, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, RuntimeError, TypeError, ValueError, OSError, EOFError) as exc:
        raise ValueError("cannot safely read original cached AV: " + str(path)) from exc
    value = canonical_raw(raw, geometry)
    checked_pin(pin)
    return value


def decode_row(row, *, absolute_step, geometry):
    import torch
    require(isinstance(row, dict) and set(row) == {"identity", "absolute_step", "source_sha256", "content_sha256", "raw"},
            "malformed original cache tensor row")
    t = row["absolute_step"]
    require(isinstance(t, torch.Tensor) and t.device.type == "cpu" and t.dtype == torch.int64
            and t.shape == (1,) and t.item() == absolute_step, "cached row absolute cursor changed")
    canonical = canonical_raw(row["raw"], geometry)
    content = content_digest(canonical)
    require(content == _tensor_id(row["content_sha256"]) and content == content_digest(row["raw"]),
            "cached tensor-only content changed")
    source = _tensor_id(row["source_sha256"])
    identity = _tensor_id(row["identity"])
    require(identity == row_identity(source, content, absolute_step), "cached row source/identity changed")
    return dict(identity=identity, absolute_step=absolute_step, source_sha256=source,
                content_sha256=content, raw=canonical)


def prepare_inputs(source_manifest, output):
    import torch
    from distillschedule.reference import diffsynth_h3_cached as provider
    from .preparation import _new_output, _publish
    from .manifest import source_identity
    from .tensors import tensor_tree
    _new_output(output)
    path = local_file(str(Path(source_manifest).absolute()))
    source_pin = source_identity(path)
    value = read_json(path)
    require(isinstance(value, dict) and set(value) == {"schema_version", "evidence_kind", "source", "precision", "geometry", "records"}
            and value["schema_version"] == SCHEMA and value["evidence_kind"] in EVIDENCE_KINDS
            and value["source"] == provider.source_identity() and value["precision"] == PRECISION,
            "cached source/schema/evidence/precision changed")
    geometry = validate_geometry(value["geometry"])
    require(sum(math.prod(s) for s in geometry.values()) <= 10_000_000,
            "cached input geometry exceeds CPU preparation capacity")
    records = value["records"]
    require(isinstance(records, list) and 0 < len(records) <= 10_000, "explicit finite bounded cache inventory required")
    rows, seen_ids, seen_paths, seen_bytes, seen_content = [], set(), set(), set(), set()
    # First pass validates every source before publication. Retain only metadata,
    # not an entire cache cohort; the publisher reads one admitted row at a time.
    for index, record in enumerate(records):
        require(isinstance(record, dict) and set(record) == {"id", "cache"} and isinstance(record["id"], str)
                and record["id"] and record["id"] not in seen_ids, "unique cache IDs required")
        cache = checked_pin(record["cache"])
        raw = _read(record["cache"], geometry)
        content = content_digest(raw)
        require(str(cache) not in seen_paths and record["cache"]["sha256"] not in seen_bytes
                and content not in seen_content, "reused/copied cache content is not fresh input")
        seen_ids.add(record["id"]); seen_paths.add(str(cache)); seen_bytes.add(record["cache"]["sha256"]); seen_content.add(content)
        descriptor = dict(absolute_step=index, source=deepcopy(record["cache"]), content_sha256=content,
                          identity=row_identity(record["cache"]["sha256"], content, index))
        rows.append(dict(id=record["id"], metadata=descriptor))
        del raw
    metadata = dict(schema_version=SCHEMA, row_schema=ROW_SCHEMA, evidence_kind=value["evidence_kind"],
        producer=producer_identity(), precision=PRECISION, geometry=geometry,
        quality_validation="NOT_RUN", claimable_speedup=False, decoded_media="NOT_IMPLEMENTED_CACHED_LATENTS_ONLY")
    provenance = dict(mode="preencoded", documents=[source_pin],
        sources=[dict(id=r["id"], media={"features": source_identity(Path(r["metadata"]["source"]["path"]))}) for r in rows], encoder=None)

    def produce(record):
        desc = record["metadata"]
        raw = _read(desc["source"], geometry)
        require(content_digest(raw) == desc["content_sha256"], "cached content changed during preparation")
        return tensor_tree(dict(identity=_id_tensor(desc["identity"]),
            absolute_step=torch.tensor([desc["absolute_step"]], dtype=torch.int64),
            source_sha256=_id_tensor(desc["source"]["sha256"]), content_sha256=_id_tensor(desc["content_sha256"]), raw=raw))

    return _publish(output, rows=rows, provenance=provenance, metadata=metadata, produce=produce)
