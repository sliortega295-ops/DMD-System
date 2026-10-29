"""Finite W6 source/draw ownership; the original deterministic parser is authority.

Metadata discovery imports no Torch/model/CUDA code. This contract does not
change the historical path/SHA-qualified two-cycle input contract.
"""
from copy import deepcopy
import json
import random
from pathlib import Path

from distillschedule.core.contracts import require
from distillschedule.core.enums import GradientMode, OperationKind
from distillschedule.core.graph import ExecutionGraph, Operation
from distillschedule.strategies.h3_ar_assets import checked
from distillschedule.strategies.h3_ar_inputs import expected_input, tensor_metadata, validate_parser
from distillschedule.strategies.h3_ar_task import H3ARTask, SOURCE_RECEIPT
from distillschedule.workloads.best_known.registry import SHA256_RE, file_sha256, fingerprint, write_new_json

KIND = "original_h3_ar_fresh_stream.v1"
SCHEMA = "distillschedule.original_h3_ar_fresh_stream.v1"
EPOCH_KIND = "original_h3_ar_epoch_stream.v1"
EPOCH_SCHEMA = "distillschedule.original_h3_ar_epoch_stream.v1"
EPOCH_POLICY = "distillschedule.h3_ar_record_epochs.v1"


def _epoch_document(rows, *, cycles, seed):
    """Bind finite ORIGINAL rank-local pools separately from logical draws.

    A local seeded permutation repeats, including a partial final epoch. This
    does not use training RNG, edit tensor data or invent fresh source records.
    """
    require(type(cycles) is int and 2 <= cycles <= 100 and type(seed) is int
            and 0 <= seed < 2**63-16, "finite W6 epoch horizon/seed required", path="h3_ar.stream")
    task = H3ARTask()
    require(isinstance(rows, list) and 16 <= len(rows) <= 16*cycles,
            "finite rank-qualified W6 source pool required", path="h3_ar.stream")
    for row in rows:
        require(isinstance(row, dict) and set(row) == {"rank", "record_id", "path", "sha256", "sources", "metadata"}
                and type(row["rank"]) is int and 0 <= row["rank"] < 16
                and isinstance(row["record_id"], str) and bool(row["record_id"])
                and _reference({k: row[k] for k in ("path", "sha256")})
                and str(Path(row["path"]).resolve()) == row["path"]
                and isinstance(row["sources"], dict) and bool(row["sources"])
                and all(isinstance(k, str) and k and _reference(v) for k, v in row["sources"].items())
                and fingerprint(row["metadata"]) == fingerprint(_metadata(task, row["rank"])),
                "W6 epoch original rank/source/metadata changed", path="h3_ar.stream")
    require(all(len({row[key] for row in rows}) == len(rows) for key in ("record_id", "path", "sha256")),
            "W6 epoch source aliases are not distinct records", path="h3_ar.stream")
    orders = []
    for rank in range(16):
        order = [i for i, row in enumerate(rows) if row["rank"] == rank]
        require(1 <= len(order) <= cycles, "each rank needs a finite original source pool within horizon", path="h3_ar.stream")
        random.Random(seed+rank).shuffle(order)
        orders.append(order)
    inventory_sha = fingerprint(rows)
    draws = []
    for cycle in range(1, cycles+1):
        for rank, order in enumerate(orders):
            epoch, position = divmod(cycle-1, len(order))
            draw = dict(cycle=cycle, rank=rank, draw_index=(cycle-1)*16+rank,
                        source_index=order[position], epoch_index=epoch, epoch_offset=position)
            draw["draw_id"] = fingerprint(dict(schema_version=EPOCH_POLICY, seed=seed,
                source_inventory_sha256=inventory_sha, **draw))
            draws.append(draw)
    policy = dict(schema_version=EPOCH_POLICY, ordering="rank_local_seeded_permutation_repeat.v1",
        evidence_kind="ORIGINAL_CACHED_AV_UNVERIFIED_QUALITY", seed=seed,
        source_inventory_sha256=inventory_sha, source_record_count=len(rows),
        logical_draw_count=len(draws), draws_sha256=fingerprint(draws))
    value = dict(schema_version=EPOCH_SCHEMA, input_kind=EPOCH_KIND, cycles=cycles, seed=seed,
        native_source_receipt_sha256=SOURCE_RECEIPT, input_contract_sha256=task.input_contract_sha256,
        input_policy=policy, source_records=rows, draws=draws)
    value["input_sha256"] = fingerprint(value)
    return value


def record(document, cycle, rank):
    """Resolve the SAME admitted logical draw for parser, runtime and evidence."""
    require(type(cycle) is int and 1 <= cycle <= document["cycles"]
            and type(rank) is int and 0 <= rank < 16, "W6 stream exhausted", path="h3_ar.stream")
    index = (cycle-1)*16+rank
    if document["input_kind"] == EPOCH_KIND:
        draw = document["draws"][index]
        return deepcopy({**document["source_records"][draw["source_index"]], **draw})
    return deepcopy(document["records"][index])


def source_identity(row):
    keys = ("path", "sha256", "record_id", "cycle")
    identity = {k: row[k] for k in keys}
    if "draw_id" in row:
        identity.update({k: row[k] for k in ("draw_id", "draw_index", "epoch_index", "epoch_offset")})
        identity["input_policy"] = EPOCH_POLICY
    return identity



def _metadata(task, rank):
    return {k: v for k, v in expected_input(task, rank).items() if k not in ("rank", "input_identity")}


def _reference(value):
    return (isinstance(value, dict) and set(value) == {"path", "sha256"}
            and isinstance(value["path"], str) and Path(value["path"]).is_absolute()
            and isinstance(value["sha256"], str) and SHA256_RE.fullmatch(value["sha256"]))


def input_metadata(path, *, cycles, seed, verify_files=False):
    path = Path(path).resolve()
    before = file_sha256(path)
    value = json.loads(path.read_text())
    if isinstance(value, dict) and value.get("input_kind") == EPOCH_KIND:
        expected = _epoch_document(value.get("source_records"), cycles=cycles, seed=seed)
        require(fingerprint(value) == fingerprint(expected), "W6 epoch source/policy/schedule identity changed", path="h3_ar.stream")
        if verify_files:
            for row in value["source_records"]:
                checked({k: row[k] for k in ("path", "sha256")})
                for ref in row["sources"].values():
                    checked(ref)
        require(file_sha256(path) == before, "W6 stream changed during admission", path="h3_ar.stream")
        return value
    fields = {"schema_version", "input_kind", "cycles", "seed", "native_source_receipt_sha256",
              "input_contract_sha256", "records", "input_sha256"}
    task = H3ARTask()
    require(isinstance(value, dict) and set(value) == fields and value["schema_version"] == SCHEMA
            and value["input_kind"] == KIND and type(cycles) is int and 2 <= cycles <= 100
            and type(value["cycles"]) is int and value["cycles"] == cycles
            and type(seed) is int and 0 <= seed < 2**63-16 and type(value["seed"]) is int and value["seed"] == seed
            and value["native_source_receipt_sha256"] == SOURCE_RECEIPT
            and value["input_contract_sha256"] == task.input_contract_sha256
            and value["input_sha256"] == fingerprint({k: v for k, v in value.items() if k != "input_sha256"}),
            "fresh W6 schema/source/input/horizon/seed identity changed", path="h3_ar.stream")
    rows = value["records"]
    require(isinstance(rows, list) and len(rows) == 16*cycles, "all finite W6 rank records required", path="h3_ar.stream")
    for index, row in enumerate(rows):
        require(isinstance(row, dict) and set(row) == {"cycle", "rank", "record_id", "path", "sha256", "sources", "metadata"}
                and type(row["cycle"]) is int and type(row["rank"]) is int
                and (row["cycle"], row["rank"]) == (index//16+1, index % 16)
                and isinstance(row["record_id"], str) and bool(row["record_id"])
                and _reference({k: row[k] for k in ("path", "sha256")})
                and isinstance(row["sources"], dict) and bool(row["sources"])
                and all(isinstance(k, str) and k and _reference(v) for k, v in row["sources"].items())
                and fingerprint(row["metadata"]) == fingerprint(_metadata(task, row["rank"])),
                "fresh W6 source order/rank/shape/dtype/scalars changed", path="h3_ar.stream")
        if verify_files:
            checked({k: row[k] for k in ("path", "sha256")})
            for ref in row["sources"].values():
                checked(ref)
    require(all(len({row[key] for row in rows}) == len(rows) for key in ("record_id", "path", "sha256")),
            "fresh W6 stream repeats a source record", path="h3_ar.stream")
    require(file_sha256(path) == before, "W6 stream changed during admission", path="h3_ar.stream")
    return value


def stream_graph(task, *, cycle, horizon, input_sha256, input_policy=None):
    require(isinstance(task, H3ARTask) and type(horizon) is int and 2 <= horizon <= 100
            and type(cycle) is int and 1 <= cycle <= horizon,
            "finite W6 graph horizon exceeded", path="h3_ar.stream")
    require(input_policy in (None, EPOCH_POLICY), "unsupported W6 graph input policy", path="h3_ar.stream")
    scope = ("WHOLE_ORIGINAL_NATIVE_UPDATE_FRESH_FINITE_STREAM" if input_policy is None
             else "WHOLE_ORIGINAL_NATIVE_UPDATE_EXPLICIT_RECORD_EPOCHS")
    before, after = dict(student=cycle-1, fake=cycle-1, ema=0), dict(student=cycle, fake=cycle, ema=0)
    op = Operation("native_update", cycle, "native_runtime", OperationKind.CALL, GradientMode.GRAD,
        required_versions=before, produced_versions={k: v for k, v in after.items() if k != "ema"},
        metadata=dict(task_sha256=task.digest, call_census=task.census, input_sha256=input_sha256))
    return ExecutionGraph((op,), (), metadata=dict(initial_versions=before, final_versions=after,
        scope=scope, task_sha256=task.digest, input_sha256=input_sha256))


class H3ARInputStream:
    def __init__(self, path, *, rank, cycles, seed, completed=0):
        self.path = Path(path).resolve()
        self.document = input_metadata(self.path, cycles=cycles, seed=seed)
        require(type(rank) is int and 0 <= rank < 16 and type(completed) is int and 0 <= completed < cycles,
                "W6 rank/cursor outside finite input horizon", path="h3_ar.stream")
        self.rank, self.cycles, self.seed, self.completed = rank, cycles, seed, completed
        self.reference = dict(path=str(self.path), sha256=file_sha256(self.path))
        self.input_sha256, self.pending = self.document["input_sha256"], None
        self._prepared = None

    def record(self, cycle):
        return record(self.document, cycle, self.rank)

    def prepare_rank_input(self, task, cache_manifest, native, rank, torch, *, teacher_cfg_scale=1.0):
        require(rank == self.rank and cache_manifest == self.reference
                and type(teacher_cfg_scale) is float and teacher_cfg_scale == 1.0,
                "fresh W6 manifest/rank/CFG differs from admitted input provider", path="h3_ar.stream")
        checked(self.reference)
        cycle = self.completed+1
        require(self.pending in (None, cycle), "W6 pending input cannot be skipped", path="h3_ar.stream")
        row = self.record(cycle)
        for ref in row["sources"].values():
            checked(ref)
        validate_parser(native)
        identity = {k: row[k] for k in ("path", "sha256")}
        if self._prepared is not None:
            prepared = self._prepared
            require(prepared[2] == source_identity(row),
                    "W6 staged initial source changed", path="h3_ar.stream")
            checked(identity)
            self._prepared = None
            return prepared
        raw = torch.load(checked(identity), map_location="cpu", weights_only=True)
        positive, negative = native.prepare_batch(raw, teacher_cfg_scale)
        observed = dict(raw_tensors=tensor_metadata(raw, torch), positive_tensors=tensor_metadata(positive, torch),
            negative_tensors=tensor_metadata(negative, torch), input_shapes=task.input_shapes(rank),
            positive_scalars={k: v for k, v in positive.items() if not isinstance(v, (torch.Tensor, dict, list, tuple))})
        require(observed == _metadata(task, rank)
                and all(type(v) is float for v in observed["positive_scalars"].values()),
                "fresh W6 original parser shape/dtype/scalar/mask contract changed", path="h3_ar.stream")
        require(file_sha256(checked(identity)) == identity["sha256"], "W6 record changed while parsing", path="h3_ar.stream")
        self.pending = cycle
        witness = dict(schema_version=self.document["schema_version"], cycle=cycle, rank=rank, record_id=row["record_id"],
                       input_sha256=self.input_sha256, **observed)
        return positive, negative, source_identity(row), witness

    def stage_initial(self, task, cache_manifest, native, rank, torch):
        """All ranks validate their next original parser result before allocation.

        H3ARSession consumes this same CPU result, so there is no second parser
        implementation and no rank can reject its first input while peers enter
        collective construction of the 50-block model.
        """
        require(self._prepared is None, "W6 initial input already staged", path="h3_ar.stream")
        self._prepared = self.prepare_rank_input(task, cache_manifest, native, rank, torch)
        return self._prepared[3]

    def commit(self, cycle):
        require(cycle == self.completed+1 and self.pending == cycle, "W6 input commit is not consecutive", path="h3_ar.stream")
        self.completed, self.pending = cycle, None

    def graph(self, task, cycle):
        policy = EPOCH_POLICY if self.document["input_kind"] == EPOCH_KIND else None
        return stream_graph(task, cycle=cycle, horizon=self.cycles, input_sha256=self.input_sha256, input_policy=policy)


def prepare_stream(sources, output, *, native, torch, cycles, seed, pre_publication_guard=None, input_policy=None):
    """Admit existing original cached AV records; no encoding or tensor edits."""
    require(not torch.cuda.is_initialized(), "W6 data preparation must remain CPU", path="h3_ar.stream")
    require(input_policy in (None, EPOCH_POLICY), "unsupported explicit W6 input policy", path="h3_ar.stream")
    rows = deepcopy(json.loads(Path(sources).read_text()))
    require(isinstance(rows, list), "explicit cycle-major rank source list required", path="h3_ar.stream")
    task = H3ARTask()
    for index, row in enumerate(rows):
        require(isinstance(row, dict) and "metadata" not in row, "source rows must not supply their own parser proof", path="h3_ar.stream")
        rank = row.get("rank") if input_policy is not None else index % 16
        row["metadata"] = _metadata(task, rank)
    value = dict(schema_version=SCHEMA, input_kind=KIND, cycles=cycles, seed=seed,
        native_source_receipt_sha256=SOURCE_RECEIPT, input_contract_sha256=task.input_contract_sha256, records=rows)
    value["input_sha256"] = fingerprint(value)
    if input_policy is not None:
        value = _epoch_document(rows, cycles=cycles, seed=seed)
    output = Path(output).resolve()
    # Unpublished admission artifact: a parser failure never leaves a manifest
    # that the public training entry could mistake for a prepared stream.
    candidate = output.with_name("."+output.name+".preparing")
    write_new_json(candidate, value)
    try:
        input_metadata(candidate, cycles=cycles, seed=seed, verify_files=True)
        if input_policy is None:
            for rank in range(16):
                cursor = H3ARInputStream(candidate, rank=rank, cycles=cycles, seed=seed)
                for cycle in range(1, cycles+1):
                    cursor.prepare_rank_input(task, cursor.reference, native, rank, torch)
                    cursor.commit(cycle)
        else:
            # Every immutable source is parser-qualified once using its first
            # actual logical draw. Runtime consumes the same resolver/loader.
            seen = set()
            for draw in value["draws"]:
                if draw["source_index"] in seen:
                    continue
                cursor = H3ARInputStream(candidate, rank=draw["rank"], cycles=cycles,
                                        seed=seed, completed=draw["cycle"]-1)
                cursor.prepare_rank_input(task, cursor.reference, native, draw["rank"], torch)
                cursor.commit(draw["cycle"])
                seen.add(draw["source_index"])
        require(not torch.cuda.is_initialized(), "original W6 parser initialized CUDA", path="h3_ar.stream")
        if pre_publication_guard is not None:
            pre_publication_guard()
        write_new_json(output, value)
    finally:
        candidate.unlink(missing_ok=True)
    receipt = dict(status="ADMITTED_FINITE_NATIVE_INPUTS", prepared=str(output), sha256=file_sha256(output),
                   input_sha256=value["input_sha256"], cycles=cycles, seed=seed, GPU_started=0)
    if input_policy is not None:
        receipt.update(input_kind=EPOCH_KIND, input_policy=EPOCH_POLICY, source_record_count=len(rows),
                       logical_draw_count=16*cycles, source_reuse_across_cycles=len(rows) < 16*cycles)
    return receipt
