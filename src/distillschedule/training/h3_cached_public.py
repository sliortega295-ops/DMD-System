"""Opt-in original cached-AV W6 reference through the public Program lifecycle.

One nonreplayable Call delegates one complete original update. A separate
Student-only projection exports full base + unmerged LoRA for cached-condition
latent diagnostics, not decoded media or the frozen distributed W6 contract.
"""
from __future__ import annotations

from copy import deepcopy
from .adapters import ModelComponents
from .h3_cached_assets import (require, admit_assets, load_base, checked_pin, content_digest,
                               runtime_identity, PRECISION)

MODEL = "distillschedule.training.h3_cached_public:build_model"
ALGORITHM = "distillschedule.training.h3_cached_public:build_algorithm"


def build_model(parameters):
    path, assets, config, meta_model = admit_assets(parameters)
    from distillschedule.workloads.best_known.registry import file_sha256
    from .h3_cached_owner import H3CachedOwner
    from .h3_cached_inference import project_current_student
    # The common session owns setup seeding. Preserve an already advanced
    # caller stream through admission/meta inspection, then let the original
    # adapter/head initialization consume it naturally, without a factory reset.
    model = load_base(meta_model, assets)
    asset_pin = dict(path=str(path), bytes=path.stat().st_size, sha256=file_sha256(path))
    owner = H3CachedOwner(model, config, assets, asset_pin)
    metadata = dict(schema_version="distillschedule.h3_cached_cpu_public.v1", assets=deepcopy(assets),
        asset_manifest=asset_pin, owner_identity=deepcopy(owner.identity),
        scope="CPU1_FP32_ORIGINAL_CACHED_AV_REFERENCE_NOT_NATIVE_WORLD16",
        evidence_kind=assets["evidence_kind"], quality_validation="NOT_RUN", claimable_speedup=False,
        portable_generator_export="CURRENT_FULL_BASE_UNMERGED_STUDENT_CPU1_FP32",
        inference_generation="ORIGINAL_8_STEP_CACHED_AV_LATENT_DIAGNOSTIC_NOT_MEDIA_QUALITY",
        decoded_media="NOT_IMPLEMENTED_CACHED_LATENTS_ONLY")
    return ModelComponents(state={"native_update": owner}, metadata=metadata, admit_training=admit_training,
                           collect_metrics=collect_metrics, inference_projection=project_current_student).validate()


def collect_metrics(outputs):
    """Expose actual original host losses, without recomputing or reading tensors."""
    require(set(outputs) == {"h3.cached.original_update"}, "one original update receipt required for metrics")
    row = outputs["h3.cached.original_update"]
    return {"generator/dmd_loss": row["dmd_loss"], "generator/gan_loss": row["gan_loss"],
            "critic/loss": row["critic_loss"]}


def admit_training(*, state, request, prepared_identity, prepared_manifest):
    from distillschedule.data import PreparedDataset
    from distillschedule.data.h3_cached_inputs import SCHEMA, ROW_SCHEMA, producer_identity, decode_row, _read
    from distillschedule.data.manifest import validate_source
    owner = state["native_update"]
    owner._validate_live()
    require(set(request.execution) <= {"adapter", "mode"}
            and request.execution.get("adapter", "program") == "program"
            and request.execution.get("mode", "reference") == "reference", "reference mode only; no auto/manual/plan")
    require(request.trainer.get("device", "cpu") == "cpu" and not request.trainer.get("distributed")
            and not request.trainer.get("validation") and request.data.get("sharding", "single") == "single",
            "CPU1/world1 only; distributed/validation/backend substitution unsupported")
    require(request.data.get("sampling", {"kind": "finite", "shuffle": False}) == {"kind": "finite", "shuffle": False},
            "finite sequential input only; no epochs/shuffle/GAS")
    require(request.model["factory"] == MODEL and request.algorithm["factory"] == ALGORITHM
            and not request.algorithm.get("parameters"), "original one-call algorithm with no overrides required")
    meta = prepared_manifest["metadata"]
    require(meta.get("schema_version") == SCHEMA and meta.get("row_schema") == ROW_SCHEMA
            and meta.get("producer") == producer_identity() and meta.get("precision") == PRECISION
            and meta.get("geometry") == owner.assets["geometry"]
            and meta.get("evidence_kind") == owner.assets["evidence_kind"], "prepared source/geometry/precision/evidence changed")
    require(owner.assets["runtime"] == runtime_identity(), "dependency runtime changed")
    for pin in prepared_manifest["provenance"]["documents"]:
        validate_source(pin)
    records = prepared_manifest["records"]
    require(0 < request.trainer["cycles"] <= len(records) <= owner.assets["budget"]["max_cycles"],
            "prepared inventory or requested target exceeds native finite capacity")
    dataset = PreparedDataset(request.data["prepared"])
    require(dataset.identity == prepared_identity and dataset.manifest == prepared_manifest, "prepared admission identity changed")
    rows, ids, paths, hashes, contents = [], set(), set(), set(), set()
    first_schema = records[0]["features"]["tensors"]
    # Stream validation before creating training output or running any Call.
    for i, record in enumerate(records):
        require(record["features"]["tensors"] == first_schema, "fixed reference tensor layout changed")
        desc = record.get("metadata", {})
        require(set(desc) == {"absolute_step", "source", "identity", "content_sha256"}
                and type(desc["absolute_step"]) is int and desc["absolute_step"] == i, "prepared row order/metadata changed")
        path = checked_pin(desc["source"])
        row = decode_row(dataset.sample(i), absolute_step=i, geometry=owner.assets["geometry"])
        require(row["identity"] == desc["identity"] and row["content_sha256"] == desc["content_sha256"]
                and row["source_sha256"] == desc["source"]["sha256"]
                and content_digest(_read(desc["source"], owner.assets["geometry"])) == row["content_sha256"],
                "prepared cache no longer matches actual original source")
        require(row["identity"] not in ids and str(path) not in paths and row["source_sha256"] not in hashes
                and row["content_sha256"] not in contents, "reused cached source/content")
        ids.add(row["identity"]); paths.add(str(path)); hashes.add(row["source_sha256"]); contents.add(row["content_sha256"])
        rows.append({k: row[k] for k in ("identity", "absolute_step", "source_sha256", "content_sha256")})
    owner.bind_inputs(prepared_identity, rows)


def native_update(ctx):
    from distillschedule.data.h3_cached_inputs import decode_row
    owner = ctx.state["native_update"]
    cycle = ctx.attributes["cycle"]
    row = decode_row(ctx.inputs, absolute_step=cycle, geometry=owner.assets["geometry"])
    return owner.update(row, cycle=cycle)


def build_algorithm(components, feeds, parameters, cycle, seed):
    from distillschedule.open_workload.torch_backend import Workload
    from distillschedule.workloads.program import Program, Call
    from distillschedule.data.h3_cached_inputs import decode_row
    require(not parameters and type(cycle) is int and cycle >= 0 and set(components.state) == {"native_update"},
            "original single-owner G1F1 algorithm required")
    owner = components.state["native_update"]
    owner._validate_live()
    require(owner.cycle == cycle, "native/outer cycle version changed")
    checked_pin(components.metadata["asset_manifest"])
    row = decode_row(feeds, absolute_step=cycle, geometry=owner.assets["geometry"])
    require({k: row[k] for k in ("identity", "absolute_step", "source_sha256", "content_sha256")}
            == owner._consumption(cycle), "prepared input/order not admitted")
    call = Call(id="h3.cached.original_update", binding="original_dmd2_update", inputs=tuple(feeds),
        reads=("native_update",), writes=("native_update",), grad="external_grad", replay_safe=False,
        attributes=dict(cycle=cycle))
    return Workload(program=Program((call,), initial_versions={"native_update": cycle}, feeds=tuple(feeds), outputs=(call.id,)),
        bindings={"original_dmd2_update": native_update}, state=components.state, feeds=feeds, seed=seed,
        identity_extra={"scope": "ORIGINAL_CACHED_AV_CPU_REFERENCE_NOT_QUALITY"})
