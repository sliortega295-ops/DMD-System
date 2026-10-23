"""Safe finite CPU1 five-record cohorts, NOT the native world16 archive schema.

Each PreparedDataset sample is a whole G1/C5/EMA0 cycle. The explicitly admitted
safetensors format is equivalent to TeacherLatents' uncropped latent/context
return, in the separately named32px/FP32 contract. No original pickle reader,
tokenizer, model-generated fallback, resizing, cropping, or fabricated data.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import random

from distillschedule.workloads.best_known.registry import fingerprint, file_sha256
from distillschedule.training.livewan_public import (require, checked_pin, read_json, local_file,
    content_digest, SCOPE, PRECISION, GEOMETRY, EVIDENCE_KINDS, CPUReferenceRequest, _safe_shapes)

SCHEMA = "distillschedule.livewan_cpu1_fp32_five_record_inputs.v1"
FORMAT = "SAFE_EQUIVALENT_UNCROPPED_TEACHER_LATENTS_RAW_TEXT_FP32_32PX_V1"
STREAMS = ("rollout", "generator_dmd_t", "generator_dmd_noise", "generator_grad_blocks", "critic_t", "critic_noise")


def producer_identity():
    from distillschedule.reference import livewan_wan21 as provider
    return dict(source=provider.source_identity(), provider_sha256=file_sha256(Path(provider.__file__)),
                format=FORMAT, equivalent="TeacherLatents.__getitem__: complete clean latents + raw indexed positive text; separate raw negative")


def _tensor(value, shape, label):
    import torch
    require(type(value) is torch.Tensor and value.device.type == "cpu" and value.layout == torch.strided
            and value.dtype == torch.float32 and tuple(value.shape) == tuple(shape)
            and not value.requires_grad and value.grad_fn is None and bool(value.isfinite().all()),
            "finite detached FP32 CPU " + label + " required; no reshape/cast/native prefix crop")


def _read(pin, *, negative=False, evidence_kind):
    from safetensors.torch import load_file
    expected = {"negative": (1, 512, 4096)} if negative else {"latents": (16, 21, 4, 4), "context": (512, 4096)}
    shapes = _safe_shapes(pin, evidence_kind=evidence_kind, keys=set(expected))
    require(shapes == expected and pin["bytes"] <= 9_000_000, "CPU source geometry/payload changed")
    value = load_file(str(checked_pin(pin)), device="cpu")
    checked_pin(pin)
    for name, shape in expected.items():
        _tensor(value[name], shape, name)
    return value


def _seed(seed, key, stream):
    # EXACT existing operation-key derivation, scoped to this CPU1 envelope.
    from .livewan_inputs import _derived_seed
    return _derived_seed(seed, key, stream)


def _operation(seed, cycle, step, entry):
    absolute = 5*cycle+step
    key = f"cycle={cycle}/step={step}/global_step={absolute}/slot=0"
    control = random.Random(_seed(seed, key, "control"))
    world = (3, 6, 9)[control.randrange(3)]
    gap = control.randint(1, 240) if control.random() < .7 else 0
    return dict(logical_step=absolute, logical_slot=0, dataset_index=entry["dataset_index"],
        sample_path=entry["source"]["path"], sample_sha256=entry["source"]["sha256"], operation_key=key,
        world_frames=world, rope_gap=gap, seeds={name: _seed(seed, key, name) for name in STREAMS},
        generator_update=step == 0, critic_update=True)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def descriptor(row):
    control = row["control"]
    return dict(cycle=control["cycle"], control_sha256=row["control_sha256"],
                entries=[dict(id=e["id"], source=deepcopy(e["source"]), content_sha256=e["content_sha256"], operation=deepcopy(e["operation"]))
                         for e in control["entries"]], negative=deepcopy(control["negative"]))


def decode_cycle(feeds, *, cycle):
    import torch
    require(isinstance(feeds, dict) and set(feeds) == {"control", "control_length", "entries", "negative"}, "complete CPU1 five-entry tensor envelope required")
    code = feeds["control"]
    require(type(code) is torch.Tensor and code.dtype == torch.uint8 and code.device.type == "cpu"
            and code.shape == (32768,), "fixed bounded CPU uint8 control required")
    length = feeds["control_length"]
    require(type(length) is torch.Tensor and length.device.type == "cpu" and length.dtype == torch.int64
            and length.shape == (1,) and 0 < length.item() < code.numel(), "bounded control length required")
    count = int(length.item())
    require(not bool(code[count:].count_nonzero()), "nonzero control padding")
    content = bytes(code[:count].tolist())
    # Canonical byte equality rejects duplicate JSON keys, nonfinite scalars and
    # alternate serialization; no side metadata silently overrides the tensors.
    control = json.loads(content)
    require(_canonical(control) == content and isinstance(control, dict) and set(control) == {
        "schema_version", "format", "scope", "precision", "geometry", "evidence_kind", "producer", "cycle", "seed", "entries", "negative"},
        "CPU control schema/canonical form changed")
    require(control["schema_version"] == SCHEMA and control["format"] == FORMAT and control["scope"] == SCOPE
            and control["precision"] == PRECISION and control["geometry"] == GEOMETRY
            and control["producer"] == producer_identity() and control["evidence_kind"] in EVIDENCE_KINDS
            and type(control["cycle"]) is int and control["cycle"] == cycle and type(cycle) is int and cycle >= 0
            and type(control["seed"]) is int and 0 <= control["seed"] < 2**63, "source/precision/geometry/cursor changed")
    require(type(feeds["entries"]) is list and type(control["entries"]) is list
            and len(feeds["entries"]) == len(control["entries"]) == 5, "exactly five distinct source rows per CPU cycle required")
    ids, paths, hashes, contents = set(), set(), set(), set()
    for step, (entry, tensors) in enumerate(zip(control["entries"], feeds["entries"], strict=True)):
        require(isinstance(entry, dict) and set(entry) == {"id", "source", "content_sha256", "dataset_index", "operation"}
                and isinstance(entry["id"], str) and bool(entry["id"].strip()) and type(entry["dataset_index"]) is int
                and entry["dataset_index"] >= 0, "source record identity fields changed")
        require(isinstance(tensors, dict) and set(tensors) == {"latents", "context"}, "missing clean history or raw text")
        _tensor(tensors["latents"], (16, 21, 4, 4), "whole teacher latent")
        _tensor(tensors["context"], (512, 4096), "raw positive context")
        require(content_digest(tensors) == entry["content_sha256"], "prepared source content changed")
        require(entry["operation"] == _operation(control["seed"], cycle, step, entry), "operation/control/independent seeds changed")
        for value, seen in ((entry["id"], ids), (entry["source"]["path"], paths),
                            (entry["source"]["sha256"], hashes), (entry["content_sha256"], contents)):
            require(value not in seen, "five-entry cohort replays source/content")
            seen.add(value)
    _tensor(feeds["negative"], (1, 512, 4096), "raw negative context")
    require(content_digest(dict(negative=feeds["negative"])) == control["negative"]["content_sha256"], "negative prompt content changed")
    return dict(control=control, control_sha256=hashlib_sha(content), entries=feeds["entries"], negative=feeds["negative"])


def hashlib_sha(content):
    import hashlib
    return hashlib.sha256(content).hexdigest()


def validate_source_cycle(row):
    control = row["control"]
    for entry in control["entries"]:
        raw = _read(entry["source"], evidence_kind=control["evidence_kind"])
        require(content_digest(raw) == entry["content_sha256"], "actual supplied teacher source changed")
    negative = _read(control["negative"]["source"], negative=True, evidence_kind=control["evidence_kind"])
    require(content_digest(negative) == control["negative"]["content_sha256"], "actual supplied negative prompt changed")


def prepare_inputs(source_manifest, output):
    import torch
    from .preparation import _new_output, _publish
    from .manifest import source_identity
    from .tensors import tensor_tree
    _new_output(output)
    path = local_file(str(Path(source_manifest).absolute()))
    source_pin = source_identity(path)
    value = read_json(path)
    require(isinstance(value, dict) and set(value) == {"schema_version", "format", "scope", "precision", "geometry",
        "evidence_kind", "producer", "seed", "cycles", "negative", "records"}
        and value["schema_version"] == SCHEMA and value["format"] == FORMAT and value["scope"] == SCOPE
        and value["precision"] == PRECISION and value["geometry"] == GEOMETRY
        and value["producer"] == producer_identity() and value["evidence_kind"] in EVIDENCE_KINDS,
        "separately named CPU1 source format/producer/precision/geometry required")
    require(type(value["cycles"]) is int and 0 < value["cycles"] <= 64
            and type(value["seed"]) is int and 0 <= value["seed"] < 2**63
            and isinstance(value["records"], list) and len(value["records"]) == 5*value["cycles"], "finite five-row cohort inventory required")
    negative = _read(value["negative"], negative=True, evidence_kind=value["evidence_kind"])
    negative_desc = dict(source=deepcopy(value["negative"]), content_sha256=content_digest(negative))
    inventory, ids, paths, hashes, contents = [], set(), set(), set(), set()
    for index, record in enumerate(value["records"]):
        require(isinstance(record, dict) and set(record) == {"id", "source"} and isinstance(record["id"], str)
                and bool(record["id"].strip()), "explicit named safe source required")
        raw = _read(record["source"], evidence_kind=value["evidence_kind"])
        digest = content_digest(raw)
        for item, seen in ((record["id"], ids), (record["source"]["path"], paths), (record["source"]["sha256"], hashes), (digest, contents)):
            require(item not in seen, "duplicate/copied source archives are not fresh cohorts")
            seen.add(item)
        inventory.append(dict(id=record["id"], source=deepcopy(record["source"]), content_sha256=digest, dataset_index=index))
    # Original-style local source ordering and independent control RNG; never
    # consume process Python/Torch/NumPy streams during preparation.
    random.Random(value["seed"]).shuffle(inventory)
    rows = []
    for cycle in range(value["cycles"]):
        entries = deepcopy(inventory[cycle*5:cycle*5+5])
        for step, entry in enumerate(entries):
            entry["operation"] = _operation(value["seed"], cycle, step, entry)
        control = {k: deepcopy(value[k]) for k in ("schema_version", "format", "scope", "precision", "geometry", "evidence_kind", "producer", "seed")}
        control.update(cycle=cycle, entries=entries, negative=deepcopy(negative_desc))
        desc = descriptor(dict(control=control, control_sha256=hashlib_sha(_canonical(control))))
        rows.append(dict(id=f"cpu1-complete-cycle-{cycle}", metadata=desc, control=control))
    metadata = dict(schema_version=SCHEMA, producer=producer_identity(), scope=SCOPE, precision=PRECISION,
                    geometry=deepcopy(GEOMETRY), evidence_kind=value["evidence_kind"], negative=deepcopy(value["negative"]))
    # The generic provenance has one features document per sample. The pinned
    # source envelope is that five-entry cohort document; every leaf archive
    # and negative prompt is additionally checked as a source document.
    documents = [source_pin, source_identity(checked_pin(value["negative"]))]
    documents.extend(source_identity(checked_pin(r["source"])) for r in inventory)
    provenance = dict(mode="preencoded", documents=documents,
        sources=[dict(id=r["id"], media={"features": deepcopy(source_pin)}) for r in rows], encoder=None)

    def produce(record):
        control = record["control"]
        entries = [_read(e["source"], evidence_kind=value["evidence_kind"]) for e in control["entries"]]
        negative = _read(value["negative"], negative=True, evidence_kind=value["evidence_kind"])["negative"]
        content = _canonical(control)
        require(len(content) < 32768, "bounded five-entry control capacity exceeded")
        code = torch.zeros(32768, dtype=torch.uint8)
        code[:len(content)] = torch.tensor(list(content), dtype=torch.uint8)
        feeds = dict(control=code, control_length=torch.tensor([len(content)], dtype=torch.int64), entries=entries, negative=negative)
        row = decode_cycle(feeds, cycle=control["cycle"])
        require(descriptor(row) == record["metadata"], "source changed during safe cohort publication")
        return tensor_tree(feeds)

    return _publish(output, rows=rows, provenance=provenance, metadata=metadata, produce=produce)


class CPUCycleInputs:
    """Worker-compatible, uncropped safe input/trace view of ONE complete cohort."""
    def __init__(self, row, *, reference_context):
        import torch
        from distillschedule.runtime.livewan_crossscale_worker import OperationTraceRow
        # A decoded row is mutable, not an admission token. Re-admit its CURRENT
        # tensors and canonical control before the Teacher sees the negative.
        # In particular, pinning the old file cannot qualify a replaced tensor.
        require(isinstance(row, dict) and set(row) == {"control", "control_sha256", "entries", "negative"},
                "complete decoded CPU control/negative envelope required")
        content = _canonical(row["control"])
        require(0 < len(content) < 32768 and hashlib_sha(content) == row["control_sha256"],
                "decoded CPU control identity changed")
        code = torch.zeros(32768, dtype=torch.uint8)
        code[:len(content)] = torch.tensor(list(content), dtype=torch.uint8)
        row = decode_cycle(dict(control=code, control_length=torch.tensor([len(content)], dtype=torch.int64),
            entries=row["entries"], negative=row["negative"]), cycle=row["control"]["cycle"])
        negative = _read(row["control"]["negative"]["source"], negative=True,
                         evidence_kind=row["control"]["evidence_kind"])
        require(content_digest(negative) == row["control"]["negative"]["content_sha256"],
                "pinned negative prompt content changed")
        self.value, self.reference_context = row, reference_context
        self.read_count, self.consumed = 0, []
        self.rows = [OperationTraceRow(**entry["operation"], request=CPUReferenceRequest()) for entry in row["control"]["entries"]]
        self._by_key = {r.operation_key: i for i, r in enumerate(self.rows)}
        self.negative = row["negative"]

    def row(self, step, rank):
        require(rank == 0 and step == self.value["control"]["cycle"]*5 + self.read_count, "CPU1 baseline source cursor reordered")
        require(self.read_count < 5, "CPU source cohort exhausted")
        return self.rows[self.read_count]

    def load(self, row):
        require(self.read_count < 5 and row is self.rows[self.read_count], "CPU source replay/reordering")
        entry = self.value["control"]["entries"][self.read_count]
        checked_pin(entry["source"])
        raw = self.value["entries"][self.read_count]
        _tensor(raw["latents"], (16, 21, 4, 4), "whole teacher latent")
        _tensor(raw["context"], (512, 4096), "raw positive context")
        require(content_digest(raw) == entry["content_sha256"], "safe source tensor content changed during native cycle")
        self.read_count += 1
        self.consumed.append(dict(operation_key=row.operation_key, record_id=entry["id"],
            source=deepcopy(entry["source"]), content_sha256=entry["content_sha256"], world_frames=row.world_frames,
            rope_gap=row.rope_gap, seeds=dict(row.seeds)))
        return raw["latents"], raw["context"], row.dataset_index
