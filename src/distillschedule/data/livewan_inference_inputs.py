"""Fresh positive text + supplied clean opening world, never a training cohort.

This CPU32/FP32 schema requires an explicit world3/6/9 BCTHW tensor. It never
crops a TeacherLatents clip, fabricates history, embeds text, or uses negatives.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from distillschedule.training.livewan_public import (BACKEND, PRECISION, SCOPE, GEOMETRY,
    EVIDENCE_KINDS, require, local_file, checked_pin, read_json, content_digest, _safe_shapes)
from .livewan_reference_inputs import _tensor, _canonical, hashlib_sha

SCHEMA = "distillschedule.livewan_cpu1_fresh_inference_inputs.v1"
FORMAT = "SUPPLIED_CLEAN_OPENING_WORLD_AND_RAW_POSITIVE_TEXT_FP32_32PX_V1"


def producer_identity():
    from distillschedule.reference import livewan_wan21 as provider
    return dict(source=provider.source_identity(), format=FORMAT,
                history="SUPPLIED_CLEAN_WORLD_ONLY_NO_TARGET_FRAMES", negative="NOT_USED")


def _read(pin, *, world_frames, evidence_kind):
    from safetensors.torch import load_file
    require(type(world_frames) is int and world_frames in (3, 6, 9), "fresh opening world3/6/9 required")
    expected = dict(world=(1, 16, world_frames, 4, 4), context=(1, 512, 4096))
    require(_safe_shapes(pin, evidence_kind=evidence_kind, keys=set(expected)) == expected
            and pin["bytes"] <= 9_000_000, "fresh inference source geometry/capacity changed")
    raw = load_file(str(checked_pin(pin)), device="cpu")
    checked_pin(pin)
    for key, shape in expected.items():
        _tensor(raw[key], shape, "fresh inference " + key)
    return raw


def decode_row(feeds, *, evidence_kind):
    import torch
    require(isinstance(feeds, dict) and set(feeds) == {"control", "control_length", "world", "context"},
            "fresh positive-text/opening-world envelope required; training cohorts cannot be replayed")
    code, length = feeds["control"], feeds["control_length"]
    require(type(code) is torch.Tensor and code.device.type == "cpu" and code.dtype == torch.uint8
            and code.shape == (8192,) and type(length) is torch.Tensor and length.device.type == "cpu"
            and length.dtype == torch.int64 and length.shape == (1,) and 0 < length.item() < code.numel(),
            "bounded fresh inference control required")
    n = int(length.item())
    require(not bool(code[n:].count_nonzero()), "fresh inference control padding changed")
    encoded = bytes(code[:n].tolist())
    control = json.loads(encoded)
    require(isinstance(control, dict) and _canonical(control) == encoded and set(control) == {
        "schema_version", "format", "scope", "precision", "geometry", "backend", "producer",
        "evidence_kind", "id", "source", "content_sha256", "world_frames"}, "fresh inference control schema changed")
    require(control["schema_version"] == SCHEMA and control["format"] == FORMAT and control["scope"] == SCOPE
            and control["precision"] == PRECISION and control["geometry"] == GEOMETRY
            and control["backend"] == BACKEND and control["producer"] == producer_identity()
            and control["evidence_kind"] == evidence_kind and evidence_kind in EVIDENCE_KINDS
            and isinstance(control["id"], str) and bool(control["id"].strip()),
            "fresh inference source/evidence/precision/geometry changed")
    world_frames = control["world_frames"]
    require(type(world_frames) is int and world_frames in (3, 6, 9), "fresh opening world3/6/9 required")
    _tensor(feeds["world"], (1, 16, world_frames, 4, 4), "fresh opening world")
    _tensor(feeds["context"], (1, 512, 4096), "fresh raw positive text")
    current = dict(world=feeds["world"], context=feeds["context"])
    require(content_digest(current) == control["content_sha256"], "fresh inference tensor content changed")
    actual = _read(control["source"], world_frames=world_frames, evidence_kind=evidence_kind)
    require(content_digest(actual) == control["content_sha256"], "fresh inference pinned source content changed")
    return dict(control=control, control_sha256=hashlib_sha(encoded), **current)


def prepare_inputs(source_manifest, output):
    """Publish complete safe tensors and source pins, with no runtime RNG use."""
    import torch
    from .manifest import source_identity
    from .preparation import _new_output, _publish
    from .tensors import tensor_tree
    _new_output(output)
    path = local_file(str(Path(source_manifest).absolute()))
    value = read_json(path)
    require(isinstance(value, dict) and set(value) == {"schema_version", "format", "scope", "precision", "geometry",
        "backend", "producer", "evidence_kind", "records"} and value["schema_version"] == SCHEMA
        and value["format"] == FORMAT and value["scope"] == SCOPE and value["precision"] == PRECISION
        and value["geometry"] == GEOMETRY and value["backend"] == BACKEND
        and value["producer"] == producer_identity() and value["evidence_kind"] in EVIDENCE_KINDS,
        "explicit fresh inference input schema required")
    require(type(value["records"]) is list and 0 < len(value["records"]) <= 64, "bounded fresh inference records required")
    rows, ids, hashes, contents = [], set(), set(), set()
    for record in value["records"]:
        require(isinstance(record, dict) and set(record) == {"id", "source", "world_frames"}
                and isinstance(record["id"], str) and bool(record["id"].strip()), "named fresh source/world length required")
        raw = _read(record["source"], world_frames=record["world_frames"], evidence_kind=value["evidence_kind"])
        content = content_digest(raw)
        for item, seen in ((record["id"], ids), (record["source"]["sha256"], hashes), (content, contents)):
            require(item not in seen, "duplicate fresh inference source/content")
            seen.add(item)
        control = {k: deepcopy(v) for k, v in value.items() if k != "records"}
        control.update(deepcopy(record), content_sha256=content)
        rows.append(dict(id=record["id"], control=control, metadata=dict(world_frames=record["world_frames"],
            generated_latent_frames=21-record["world_frames"], output_frames=81,
            conditioning_pixel_frames=(record["world_frames"]-1)*4+1, content_sha256=content)))

    def produce(record):
        control = record["control"]
        raw = _read(control["source"], world_frames=control["world_frames"], evidence_kind=value["evidence_kind"])
        encoded = _canonical(control)
        require(len(encoded) < 8192, "fresh inference control capacity exceeded")
        code = torch.zeros(8192, dtype=torch.uint8)
        code[:len(encoded)] = torch.tensor(list(encoded), dtype=torch.uint8)
        feeds = dict(control=code, control_length=torch.tensor([len(encoded)], dtype=torch.int64), **raw)
        decode_row(feeds, evidence_kind=value["evidence_kind"])
        return tensor_tree(feeds)

    provenance = dict(mode="preencoded", documents=[source_identity(path)] + [
        source_identity(checked_pin(r["source"])) for r in value["records"]],
        sources=[dict(id=r["id"], media={"features": source_identity(checked_pin(r["source"]))})
                 for r in value["records"]], encoder=None)
    return _publish(output, rows=rows, provenance=provenance,
        metadata={k: deepcopy(v) for k, v in value.items() if k != "records"}, produce=produce)
