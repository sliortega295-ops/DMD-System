"""Default-OFF H3 correctness capture protocol. No launcher or permission grant.

Only an explicitly SHA-bound native observer can activate this module. No Torch
import, model construction, provider replacement, RNG or collective lives here.
The Torch adapter imports nothing itself: the existing native Torch is injected.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import functools
import json
import math
import os
from pathlib import Path
import re
import sys

SCHEMA = "distillschedule.h3_actual_attention_tuple_proposal.v1"
NAMES = ("dout", "q", "k", "v", "out", "softmax_lse",
         "original_dq", "original_dk", "original_dv")
ARGUMENT_NAMES = ("dout", "q", "k", "v", "out", "softmax_lse", "cu_seqlens_q", "cu_seqlens_k",
                  "sequed_q", "sequed_k", "max_seqlen_q", "max_seqlen_k", "dq", "dk", "dv",
                  "softmax_scale", "is_causal", "window_size_left", "window_size_right", "softcap",
                  "deterministic", "sm_margin")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def json_copy(value):
    # Reject tensors, callable objects and non-finite provenance instead of scanning them.
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def durable_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".incomplete")
    with temporary.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def storage_span(shape, stride, offset, element_size):
    """Exact prefix+positive-stride span; reject overlapping/unsupported layouts.

    This is metadata arithmetic, not a read of tensor values. The sorted-stride
    check is conservative; unsupported unusual nonoverlapping layouts fail closed.
    """
    require(len(shape) == len(stride) and len(shape) > 0, "unsupported shape/stride")
    require(type(offset) is int and offset >= 0 and type(element_size) is int and element_size > 0,
            "unsupported storage offset/element size")
    require(all(type(n) is int and n > 0 for n in shape), "empty or symbolic tensor shape")
    require(all(type(s) is int and s >= 0 for s in stride), "negative/symbolic tensor stride")
    extent = 1
    for step, size in sorted((s,n) for n,s in zip(shape,stride) if n > 1):
        require(step >= extent, "overlapping/unsupported tensor stride")
        extent += (size - 1) * step
    return (offset + sum((n-1)*s for n,s in zip(shape,stride)) + 1) * element_size


@dataclass(frozen=True)
class CaptureTarget:
    rank: int = 0
    cycle: int = 1
    fake_index: int = 0
    block_index: int = 10

    def __post_init__(self):
        require((self.rank,self.cycle,self.fake_index) == (0,1,0),
                "this narrow proposal supports only rank0/cycle1/first Fake")
        require(type(self.block_index) is int and 0 <= self.block_index < 50, "invalid block")


@dataclass(frozen=True)
class CaptureBudget:
    max_payload_bytes: int
    max_replay_span_bytes: int
    max_tensor_bytes: int
    max_file_bytes: int
    max_host_working_bytes: int

    def __post_init__(self):
        require(all(type(v) is int and v > 0 for v in self.__dict__.values()), "invalid capture budget")


class TupleCapture:
    """One forward-bound ctx, one original raw backward, one complete publication.

    The owner must already have admitted source/provider/input/capacity and attached
    this object explicitly before native setup. This object never launches anything
    and cannot authorize its own use or upgrade W5's numerical FAIL.
    """
    def __init__(self, *, target, rank, budget, provenance, io, output,
                 q_shape=(1,37760,56,128), lse_shape=(1,56,37760)):
        require(isinstance(target,CaptureTarget) and isinstance(budget,CaptureBudget), "invalid capture contract")
        require(type(rank) is int and rank >= 0, "invalid global rank")
        required = {"source_head", "provider_schema", "provider_files", "native_config_sha256",
                    "input_manifest_sha256", "owner_receipt_sha256", "capacity_receipt_sha256", "runtime_identity", "run_kind"}
        require(required <= set(provenance), "unbound source/provider/input/owner/capacity provenance")
        require(provenance["run_kind"] == "correctness", "attention capture is forbidden in performance/calibration modes")
        require(re.fullmatch(r"[0-9a-f]{40}",provenance["source_head"] or "") is not None, "invalid source HEAD")
        for key in ("native_config_sha256", "input_manifest_sha256", "owner_receipt_sha256", "capacity_receipt_sha256"):
            require(isinstance(provenance[key],str) and re.fullmatch(r"[0-9a-f]{64}",provenance[key]) is not None,
                    "missing exact " + key)
        require(bool(provenance["provider_schema"]) and bool(provenance["provider_files"])
                and bool(provenance["runtime_identity"]), "unbound live provider/runtime")
        self.target, self.rank, self.budget, self.provenance = target,rank,budget,json_copy(provenance)
        self.io, self.output = io,Path(output)
        self.q_shape, self.lse_shape = list(q_shape),list(lse_shape)
        self.prepared, self.active, self.trainer = {},None,None
        self.claimed, self.completed, self.nonce = False,False,0

    def record_prepared(self, trainer, item, *, sample_identity, fake_index, micro_index):
        if self.rank != self.target.rank or self.claimed or fake_index != self.target.fake_index:
            return
        owner = trainer.planner_cycle_observer
        require(owner.run_kind == "correctness", "native owner is not correctness-only")
        if owner.pending["cycle"] != self.target.cycle:
            return
        require(owner.active == 0 and micro_index == 0, "capture needs original cycle1/accum1")
        require(owner.source_head == self.provenance["source_head"], "native owner source mismatch")
        sample = dict(trainer.matched_cycle_census.samples[-1])
        require(sample == dict(stage="fake_0",identity=sample_identity), "actual prepared sample/census mismatch")
        expected = getattr(self, "native_expected_input", None)
        if expected is not None:
            require(dict(sample_identity=sample_identity, latent_shape=json_copy(item.latent_shape),
                         cache_key=json_copy(item.cache_key)) == expected, "actual prepared input binding mismatch")
        geometry = {}
        if getattr(self, "native_bind_prepared_geometry", False):
            actual = prepared_native_geometry(trainer, item)
            require(_exact_json(actual, self.native_geometry),
                    f"actual prepared geometry mismatch: expected={self.native_geometry}, actual={actual}")
            geometry["prepared_geometry"] = actual
        require(id(item) not in self.prepared, "prepared item bound twice")
        self.prepared[id(item)] = (item,dict(sample=sample, cache_key=json_copy(item.cache_key),
                                        latent_shape=json_copy(item.latent_shape), **geometry))

    @contextmanager
    def fake_scope(self, trainer, item):
        # A nonselected rank/action does not inspect model/tensor state.
        record = self.prepared.pop(id(item),None)
        if record is None or self.claimed:
            yield
            return
        require(record[0] is item and self.active is None, "invalid/nested actual prepared item scope")
        owner = trainer.planner_cycle_observer
        require(owner.active == 0 and owner.pending["cycle"] == 1, "native cycle scope mismatch")
        require(owner.updates == dict(student=1,fake=0,ema=0), "first Fake version contract mismatch")
        require(trainer.shared_model._physical_role == "fake"
                and trainer.shared_model._physical_training_mode is True, "actual Fake role/training mismatch")
        self.nonce += 1
        self.active = dict(**record[1],source_head=self.provenance["source_head"], rank=self.rank,
                           cycle=1, action="native.fake_update_0", fake_index=0, micro_index=0,
                           role="fake", versions=dict(owner.updates), scope_nonce=self.nonce)
        self.trainer = trainer
        try:
            yield
        finally:
            self.active,self.trainer = None,None

    def bind_forward(self, block_index):
        if self.active is None or self.claimed or block_index != self.target.block_index:
            return None
        self._validate_live_context()
        return json_copy(dict(self.active,block_index=int(block_index)))

    def _validate_live_context(self):
        require(self.active is not None and self.trainer is not None, "capture outside original action scope")
        owner = self.trainer.planner_cycle_observer
        require(owner.source_head == self.active["source_head"] and owner.active == 0
                and owner.pending["cycle"] == self.active["cycle"], "native source/cycle scope drift")
        require(owner.updates == self.active["versions"], "native version changed across forward/backward")
        require(self.trainer.shared_model._physical_role == "fake"
                and self.trainer.shared_model._physical_training_mode is True, "native Fake role drift")

    def claim_backward(self, token):
        if token is None or self.claimed:
            return None
        self._validate_live_context()
        require(token == dict(self.active,block_index=self.target.block_index),
                "saved forward ctx does not match actual backward scope")
        # Reserve before native call; an exception cannot trigger retry/double capture.
        self.claimed = True
        return json_copy(token)

    def wrap_raw(self, claim, original):
        """Local call wrapper only; do not modify provider or kernel registry."""
        @functools.wraps(original)
        def observed(*arguments, **keywords):
            require(not keywords, "compact raw backward changed to keyword arguments")
            raw_result = original(*arguments, **keywords)  # Exactly once, original objects/scalars.
            return self.publish_after_raw(claim, arguments, raw_result)
        return observed

    def publish_after_raw(self, claim, arguments, raw_result):
        require(self.claimed and not self.completed, "unclaimed/repeated capture")
        self._validate_live_context()
        require(claim == dict(self.active,block_index=self.target.block_index), "raw/backward claim mismatch")
        require(len(arguments) == 22, "raw22 argument arity changed")
        require(all(v is None for v in arguments[6:12]), "raw optional varlen contract changed")
        require(type(arguments[15]) is float and math.isfinite(arguments[15]), "raw scale contract changed")
        expected = (False,-1,-1,0.0,False,0)
        require(all(type(a) is type(b) and a == b for a,b in zip(arguments[16:],expected)), "raw scalar contract changed")
        require(arguments[19].hex() == (0.0).hex(), "raw softcap sign changed")
        values = (*arguments[:6],*arguments[12:15])
        rows, alias_groups = [],{}
        for name,value in zip(NAMES,values):
            metadata = json_copy(self.io.metadata(value))
            expected_shape = self.lse_shape if name == "softmax_lse" else self.q_shape
            expected_dtype = "torch.float32" if name == "softmax_lse" else "torch.bfloat16"
            require(metadata["shape"] == expected_shape and metadata["dtype"] == expected_dtype
                    and metadata["element_size"] == (4 if name == "softmax_lse" else 2)
                    and metadata["layout"] == "torch.strided",
                    f"actual tensor dtype/shape/layout mismatch for {name}: "
                    f"expected={expected_dtype}/{expected_shape}/torch.strided, "
                    f"actual={metadata['dtype']}/{metadata['shape']}/{metadata['layout']}")
            span = storage_span(metadata["shape"],metadata["stride"],metadata["storage_offset"],metadata["element_size"])
            require(span <= metadata["storage_nbytes"], "tensor span exceeds original storage")
            logical = math.prod(metadata["shape"]) * metadata["element_size"]
            key = (metadata["device"],metadata["storage_identity"])
            group = alias_groups.setdefault(key,len(alias_groups))
            rows.append(dict(name=name, file=name+".pt", original=metadata, alias_group=group,
                             logical_bytes=logical, replay_span_bytes=span))
        require(len({r["original"]["device"] for r in rows}) == 1, "tuple crosses devices")
        for group in range(len(alias_groups)):
            require(len({r["original"]["dtype"] for r in rows if r["alias_group"] == group}) == 1,
                    "mixed-dtype alias group unsupported")
        for output in rows[6:]:
            require(all(output["alias_group"] != other["alias_group"] for other in rows if other is not output),
                    "raw mutable output aliases another input/output")
        logical = sum(r["logical_bytes"] for r in rows)
        largest = max(r["logical_bytes"] for r in rows)
        spans = sum(max(r["replay_span_bytes"] for r in rows if r["alias_group"] == group)
                    for group in range(len(alias_groups)))
        host_estimate = 2*largest+(8 << 20)  # Explicit conservative accounting assumption, NOT a measurement.
        require(logical <= self.budget.max_payload_bytes and largest <= self.budget.max_tensor_bytes
                and spans <= self.budget.max_replay_span_bytes
                and host_estimate <= self.budget.max_host_working_bytes, "capture metadata exceeds declared budget")
        require(self.output.parent.is_dir() and not self.output.exists(), "capture path absent parent/already exists")
        # The original native callable has ALREADY returned. First fence/copy is here.
        self.io.fence(rows[1]["original"]["device"])
        self.output.mkdir()
        total_file_bytes = 0
        for row,value in zip(rows,values):
            path = self.output/row["file"]
            self.io.save(value,path)
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
            row["file_bytes"] = path.stat().st_size
            total_file_bytes += row["file_bytes"]
            require(total_file_bytes <= self.budget.max_file_bytes, "serialized files exceed declared budget")
            row["sha256"] = digest_file(path)
        bound_arguments = []
        for index,(name,value) in enumerate(zip(ARGUMENT_NAMES,arguments)):
            if index < 6 or 12 <= index < 15:
                bound_arguments.append(dict(argument=name,tensor=NAMES[index if index < 6 else index-6]))
            else:
                require(type(value) in (type(None),bool,int,float), "unsupported raw scalar")
                bound_arguments.append(dict(argument=name,type=type(value).__name__,value=value,
                                            **({"float_hex":value.hex()} if type(value) is float else {})))
        manifest = dict(schema_version=SCHEMA, complete=True, provenance=self.provenance,
                        binding=claim, tensors=rows, raw_arguments=bound_arguments,
                        payload_representation=self.io.representation,
                        transfer_chunk_logical_byte_limit=getattr(self.io,"max_transfer_chunk_bytes",None),
                        raw_result_contract="ORIGINAL_SOFTMAX_D_RETURN_IGNORED_BY_NATIVE_CALLER_NOT_A_GRADIENT",
                        snapshot_order="ORIGINAL_RAW_ONCE_THEN_CURRENT_STREAM_FENCE_THEN_SEQUENTIAL_DEDICATED_CPU_COPIES",
                        budgets=self.budget.__dict__, actual_payload_bytes=logical, actual_file_bytes=total_file_bytes,
                        replay_span_bytes=spans, host_working_estimate_bytes=host_estimate,
                        host_working_estimate_not_measured=True, byte_order=sys.byteorder,
                        dout_activity="NOT_EVALUATED_DEFERRED_POST_NATIVE_CYCLE_TIMER_CPU_CONSUMER",
                        same_tuple_diagnosis_eligibility="INCONCLUSIVE_UNTIL_COMPLETE_AND_FINITE_NONZERO_DOUT",
                        formal_timing_eligible=False, numerical_cross_reference="NOT_RUN_W5_FAIL_RETAINED")
        durable_json(self.output/"manifest.json",manifest)
        durable_json(self.output/"_SUCCESS",dict(manifest_sha256=digest_file(self.output/"manifest.json")))
        self.completed = True
        return raw_result  # Preserve original object; native caller still ignores it.

    def require_complete_after_native(self):
        """Call only after original train() returns through ALL original gates."""
        if self.rank == self.target.rank:
            require(self.completed and (self.output/"_SUCCESS").is_file(), "native run did not capture its declared tuple")
        return self.completed


def load_capture(path, *, max_file_bytes, expected_provenance=None):
    """Small integrity consumer. No model/Torch import and no forward replacement."""
    path = Path(path)
    marker = json.loads((path/"_SUCCESS").read_text())
    require(marker == dict(manifest_sha256=digest_file(path/"manifest.json")), "manifest checksum mismatch")
    manifest = json.loads((path/"manifest.json").read_text())
    require(manifest["schema_version"] == SCHEMA and manifest["complete"] is True
            and manifest["formal_timing_eligible"] is False, "incomplete/unsupported tuple manifest")
    require(tuple(r["name"] for r in manifest["tensors"]) == NAMES, "tuple tensor inventory mismatch")
    if expected_provenance is not None:
        require(manifest["provenance"] == expected_provenance, "source/provider/runtime/input provenance mismatch")
    total = 0
    for row in manifest["tensors"]:
        require(row["file"] == row["name"]+".pt", "unsafe tensor filename")
        file = path/row["file"]
        require(not file.is_symlink() and file.is_file(), "missing/linked tensor file")
        total += file.stat().st_size
        require(total <= max_file_bytes and file.stat().st_size == row["file_bytes"], "tensor file size/budget mismatch")
        require(digest_file(file) == row["sha256"], "tensor file checksum mismatch")
    require(total == manifest["actual_file_bytes"], "tuple total size mismatch")
    return manifest


class TorchSnapshotIO:
    """CPU serialization/restore tested; actual CUDA transfer remains NOT_RUN."""
    representation = "TORCH_SAVE_DEDICATED_CONTIGUOUS_CPU_TENSOR_ORIGINAL_LAYOUT_IN_MANIFEST"

    def __init__(self, torch, *, max_transfer_chunk_bytes=16 << 20):
        require(type(max_transfer_chunk_bytes) is int and max_transfer_chunk_bytes > 0,
                "invalid logical transfer chunk limit")
        self.torch = torch
        self.max_transfer_chunk_bytes = max_transfer_chunk_bytes
        self.copy_chunks = []

    def metadata(self, tensor):
        torch = self.torch
        require(type(tensor) is torch.Tensor and tensor.layout == torch.strided
                and not tensor.is_conj() and not tensor.is_neg(), "unsupported tensor subclass/layout/bits")
        storage = tensor.untyped_storage()
        return dict(shape=list(tensor.shape), stride=list(tensor.stride()), storage_offset=tensor.storage_offset(),
                    dtype=str(tensor.dtype), element_size=tensor.element_size(), device=str(tensor.device),
                    storage_identity=str(storage._cdata), storage_nbytes=storage.nbytes(),
                    data_ptr=tensor.data_ptr(), storage_data_ptr=storage.data_ptr(), layout=str(tensor.layout))

    def fence(self, device):
        require(str(device).startswith("cuda:"), "native capture requires actual CUDA tensor")
        require(not self.torch.cuda.is_current_stream_capturing(), "CUDA Graph capture unsupported")
        self.torch.cuda.current_stream(device).synchronize()

    def save(self, tensor, path):
        torch = self.torch
        # No explicit CUDA clone, random draw, cast, extra forward or provider invocation.
        # Strided D2H may need an internal contiguous CUDA temporary. Bound each
        # submitted logical copy to 16 MiB rather than asking for a 0.5 GiB copy.
        # Allocator/copy implementation overhead still needs target measurement.
        with torch.no_grad():
            host = torch.empty(tuple(tensor.shape),dtype=tensor.dtype,device="cpu")
            source = tensor.detach()
            axis = max(range(source.ndim),key=lambda i:source.shape[i])
            cross_section = source.numel()*source.element_size()//source.shape[axis]
            require(cross_section <= self.max_transfer_chunk_bytes,
                    "logical tensor cross-section exceeds bounded copy limit")
            width = max(1,self.max_transfer_chunk_bytes//cross_section)
            for start in range(0,source.shape[axis],width):
                count = min(width,source.shape[axis]-start)
                part = source.narrow(axis,start,count)
                self.copy_chunks.append(part.numel()*part.element_size())
                host.narrow(axis,start,count).copy_(part,non_blocking=False)
            with Path(path).open("xb") as stream:
                torch.save(host,stream)
            del host

    def restore_tensors(self, path, manifest, *, device, names=None):
        """Separate admitted consumer: exact logical values/strides/offset/aliases.

        Unaccessed storage gaps are zeroed, not represented as captured data. No
        native provider is called here. Original output payloads are references;
        a later raw repeat must allocate fresh outputs, not reuse saved originals.
        """
        torch = self.torch
        require(manifest["payload_representation"] == self.representation, "not a native Torch tuple")
        require(manifest["byte_order"] == sys.byteorder, "cross-endian replay unsupported")
        names = NAMES if names is None else tuple(names)
        require(names and len(names) == len(set(names)) and set(names) <= set(NAMES), "invalid restore tensor selection")
        rows = [r for r in manifest["tensors"] if r["name"] in names]
        loaded,result = {},{}
        for row in rows:
            tensor = torch.load(Path(path)/row["file"],map_location="cpu",weights_only=True)
            require(type(tensor) is torch.Tensor and list(tensor.shape) == row["original"]["shape"]
                    and str(tensor.dtype) == row["original"]["dtype"]
                    and tensor.is_contiguous() and tensor.storage_offset() == 0, "serialized tensor contract mismatch")
            loaded[row["name"]] = tensor
        for group in sorted({r["alias_group"] for r in rows}):
            members = [r for r in rows if r["alias_group"] == group]
            require(len({r["original"]["dtype"] for r in members}) == 1, "mixed-dtype alias group unsupported")
            first = members[0]
            count = max(r["replay_span_bytes"] for r in members)//first["original"]["element_size"]
            cpu = torch.zeros(count,dtype=loaded[first["name"]].dtype,device="cpu")
            for row in members:
                meta = row["original"]
                cpu.as_strided(meta["shape"],meta["stride"],meta["storage_offset"]).copy_(loaded[row["name"]])
            # Detect conflicting alias snapshots using complete bit comparisons on CPU.
            for row in members:
                meta = row["original"]
                view = cpu.as_strided(meta["shape"],meta["stride"],meta["storage_offset"])
                require(torch.equal(view.contiguous().view(torch.uint8),loaded[row["name"]].view(torch.uint8)),
                        "inconsistent logical values in shared storage snapshots")
            backing = cpu.to(device=device)
            for row in members:
                meta = row["original"]
                view = backing.as_strided(meta["shape"],meta["stride"],meta["storage_offset"])
                if str(device).startswith("cuda"):
                    require(view.data_ptr() % 256 == meta["data_ptr"] % 256,
                            "reconstructed CUDA pointer alignment differs from original")
                result[row["name"]] = view
        return result

    def restore_raw_arguments(self, path, manifest, *, device):
        """Prepare inputs + FRESH mutable output buffers, but NEVER call a provider.

        Original gradient files remain separate comparison evidence on CPU/disk.
        Caller must independently admit loaded provider/schema/capacity/ownership.
        """
        torch = self.torch
        inputs = self.restore_tensors(path,manifest,device=device,names=NAMES[:6])
        outputs = {}
        for row in manifest["tensors"][6:]:
            meta = row["original"]
            require(meta["dtype"] == "torch.bfloat16", "unsupported output dtype")
            storage = torch.empty(row["replay_span_bytes"]//meta["element_size"],dtype=torch.bfloat16,device=device)
            view = storage.as_strided(meta["shape"],meta["stride"],meta["storage_offset"])
            if str(device).startswith("cuda"):
                require(view.data_ptr() % 256 == meta["data_ptr"] % 256, "fresh CUDA output alignment mismatch")
            outputs[row["name"]] = view
        result = []
        for entry in manifest["raw_arguments"]:
            if "tensor" in entry:
                name = entry["tensor"]
                result.append(inputs[name] if name in inputs else outputs[name])
            else:
                value = entry["value"]
                require(type(value).__name__ == entry["type"], "saved raw scalar type mismatch")
                if type(value) is float:
                    require(value.hex() == entry["float_hex"], "saved raw float precision mismatch")
                result.append(value)
        require(len(result) == 22, "saved raw argument arity mismatch")
        return tuple(result)

    def summarize_dout_after_native_completion(self, path, *, max_file_bytes, native_receipt_sha256):
        """Full CPU-only scan AFTER native completion/timer stop, not in the hook.

        Caller binds the actual closed native receipt. Chunked FP64 accumulation
        avoids a second full-dout cast. This does not grant numerical acceptance.
        """
        require(isinstance(native_receipt_sha256,str)
                and re.fullmatch(r"[0-9a-f]{64}",native_receipt_sha256) is not None,
                "post-timer dout summary needs a completed native receipt hash")
        manifest = load_capture(path,max_file_bytes=max_file_bytes)
        require(manifest["payload_representation"] == self.representation, "not an actual Torch tuple")
        torch = self.torch
        dout = torch.load(Path(path)/"dout.pt",map_location="cpu",weights_only=True)
        require(type(dout) is torch.Tensor and dout.device.type == "cpu"
                and list(dout.shape) == manifest["tensors"][0]["original"]["shape"]
                and str(dout.dtype) == "torch.bfloat16" and dout.is_contiguous(), "invalid saved dout")
        nonzero,finite,squared_sums = 0,True,[]
        flat = dout.view(-1)
        for start in range(0,flat.numel(),1 << 20):
            chunk = flat[start:start+(1 << 20)].double()
            finite = finite and bool(torch.isfinite(chunk).all())
            nonzero += int(torch.count_nonzero(chunk))
            squared_sums.append(float(chunk.square().sum()))
        norm = math.sqrt(math.fsum(squared_sums)) if finite else None
        return dict(schema_version="distillschedule.h3_captured_dout_activity.v1",
                    native_completed_receipt_sha256=native_receipt_sha256,
                    manifest_sha256=digest_file(Path(path)/"manifest.json"), full_numel=dout.numel(),
                    full_nonzero_count=nonzero, all_finite=finite, l2_fp64_chunked=norm,
                    status=dout_activity_status(all_finite=finite,nonzero_count=nonzero),
                    scope="POST_NATIVE_TIMER_FULL_SAVED_CPU_DOUT_NOT_A_NUMERICAL_PASS")


def dout_activity_status(*, all_finite, nonzero_count):
    require(type(all_finite) is bool and type(nonzero_count) is int and nonzero_count >= 0,
            "invalid full-dout activity summary")
    if not all_finite:
        return "INCONCLUSIVE_NONFINITE_DOUT"
    if nonzero_count == 0:
        return "INCONCLUSIVE_ZERO_DOUT"
    return "NONZERO_DOUT_TUPLE_DIAGNOSIS_ONLY_NOT_NUMERICAL_ACCEPTANCE"


# Only the explicit native correctness hook below reads this protocol. No
# environment-driven activation or Torch import happens when this file loads.
NATIVE_Q_SHAPE = (1, 37760, 56, 128)
NATIVE_LSE_SHAPE = (1, 56, 37760)
REMOTE_PERSISTENT_ROOT = Path("/share/project")
NATIVE_SELECTOR = "h3.w5.c1.fake0.block10.actualtuple.v1"
PREPARED_GEOMETRY_SELECTOR = "h3.w5.c1.fake0.block10.actualtuple.v2"
RAW_BACKWARD_SCHEMA = "_flash_attn3_cuda_373a45c::_flash_attn_backward(Tensor dout, Tensor q, Tensor k, Tensor v, Tensor out, Tensor softmax_lse, Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_k=None, Tensor? sequed_q=None, Tensor? sequed_k=None, SymInt? max_seqlen_q=None, SymInt? max_seqlen_k=None, Tensor(a12!)? dq=None, Tensor(a13!)? dk=None, Tensor(a14!)? dv=None, float? softmax_scale=None, bool is_causal=False, SymInt window_size_left=-1, SymInt window_size_right=-1, float softcap=0., bool deterministic=False, SymInt sm_margin=0) -> Tensor"
_NATIVE_FILES = ("attention_tuple_capture.py", "integrated_entrypoint.py", "trainer_loop.py",
                 "grid_adaln.py", "fa3_replay_cache.py")


def _exact_json(actual, expected):
    # In particular True is not rank1/cycle1 and float1.0 is not integer1.
    return json.dumps(actual, sort_keys=True, allow_nan=False) == json.dumps(expected, sort_keys=True, allow_nan=False)


def native_capture_geometry(packet):
    """Shared producer/consumer geometry contract; not an admission decision."""
    bind_prepared = packet["schema_version"] == "distillschedule.h3_native_attention_capture_opt_in.v2"
    selector = PREPARED_GEOMETRY_SELECTOR if bind_prepared else NATIVE_SELECTOR
    require(packet["schema_version"] in ("distillschedule.h3_native_attention_capture_opt_in.v1",
                                         "distillschedule.h3_native_attention_capture_opt_in.v2")
            and packet["selector"] == selector, "capture schema/selector binding mismatch")
    q_shape, lse_shape = list(NATIVE_Q_SHAPE), list(NATIVE_LSE_SHAPE)
    if bind_prepared:
        # The exact declared sample must still fit the reviewed maximum.
        # record_prepared separately checks the real condition before Fake.
        q_shape = packet["geometry"]["q_shape"]
        require(isinstance(q_shape, list) and len(q_shape) == 4
                and all(type(n) is int and n > 0 for n in q_shape)
                and q_shape[0] == 1 and q_shape[2:] == list(NATIVE_Q_SHAPE[2:])
                and q_shape[1] <= NATIVE_Q_SHAPE[1], "capture shape exceeds reviewed geometry")
        lse_shape = [q_shape[0], q_shape[2], q_shape[1]]
    geometry = dict(q_shape=q_shape, lse_shape=lse_shape,
                    q_dtype="torch.bfloat16", lse_dtype="torch.float32", layout="BSHD")
    require(_exact_json(packet["geometry"], geometry), "capture shape/precision/layout metadata mismatch")
    return json_copy(geometry)


def prepared_native_geometry(trainer, item):
    """Read metadata only from the original padless [text | audio | video] input.

    Text length belongs to the selected sample, not the maximum cache capacity.
    No tensor values, additional layout construction, device copy or RNG is used.
    The native condition/packing gates still validate token contents themselves.
    """
    tags = item.condition["text_token_tags"]
    embeds = item.condition["prompt_embeds"]
    latent = item.latent_shape
    video, audio = latent["video_tokens"], latent["audio_tokens"]
    require(len(tags.shape) == 1 and len(embeds.shape) == 3
            and embeds.shape[0] == 1 and embeds.shape[1] == tags.shape[0]
            and type(latent["batch_size"]) is int and latent["batch_size"] == 1
            and len(video) == len(audio) == 3 and video[0] == audio[0] == 1
            and all(type(n) is int and n > 0 for n in (*video, *audio, *tags.shape, *embeds.shape)),
            "invalid actual prepared condition geometry")
    config = trainer.shared_model.denoiser_module().config
    heads, dim = config.num_attention_heads, config.attention_head_dim
    require(type(heads) is int and type(dim) is int and heads > 0 and dim > 0,
            "invalid actual prepared model geometry")
    sequence = tags.shape[0] + video[1] + audio[1]
    return dict(q_shape=[1, sequence, heads, dim], lse_shape=[1, heads, sequence],
                q_dtype="torch.bfloat16", lse_dtype="torch.float32", layout="BSHD")


def _checked_reference(reference):
    require(isinstance(reference, dict) and set(reference) == {"path", "sha256"}, "unbound file reference")
    require(isinstance(reference["path"], str) and Path(reference["path"]).is_absolute(), "absolute evidence path required")
    require(isinstance(reference["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", reference["sha256"]),
            "invalid evidence checksum")
    path = Path(reference["path"])
    require(path.is_file() and digest_file(path) == reference["sha256"], "evidence checksum mismatch: " + str(path))
    return path


def native_runtime_identity(torch):
    return dict(python_executable=sys.executable, python_prefix=sys.prefix,
                torch_version=torch.__version__, torch_cuda_version=torch.version.cuda)


class _NativeTorchSnapshotIO(TorchSnapshotIO):
    def __init__(self, torch, *, native_device):
        super().__init__(torch)
        self.native_device = native_device

    def metadata(self, tensor):
        row = super().metadata(tensor)
        require(row["device"] == self.native_device, "actual tensor device differs from admitted native rank")
        return row


def attach_native_capture(trainer, observer, *, torch, rank):
    """Bind a root-sealed diagnostic to the existing native owner, never launch.

    Root remains responsible for live ownership/capacity/source admission. Its
    exact receipt/input/provider files are hash-checked, not regenerated here.
    There is no new feature source, model allocation, RNG read/reset or fence.
    """
    env = os.environ
    require(env.get("DISTILLSCHEDULE_RUN_KIND") == observer.run_kind == "correctness"
            and not observer.calibration_enabled and env.get("H3_PLANNER_CALIBRATION", "0") == "0",
            "attention capture requires correctness-only mode, never profile/native timing/calibration")
    require(env.get("H3_PLANNER_TWO_CYCLE") == "1", "attention capture requires original two-cycle observer")
    require(getattr(trainer, "attention_tuple_capture", None) is None, "attention tuple observer already installed")
    # Reuse the original generated implementation/source/config gate, not a
    # parallel recipe validator. Original train.validate_start also still runs.
    from distillschedule.search.h3_implementation import current_implementation
    impl = current_implementation()
    require(impl is not None and impl.id == "joint-h3v1-d2-n128", "capture needs exact original generated candidate")
    impl.receipt_binding(trainer.config, env, mode="correctness")
    require(observer.source_head == env.get("EXPECTED_INTEGRATION_HEAD"), "native source admission mismatch")
    require(type(rank) is int and observer.rank == rank and torch.distributed.is_initialized()
            and torch.distributed.get_rank() == rank and 0 <= rank < 16
            and torch.distributed.get_world_size() == trainer.matched_expected_world_size == 16,
            "actual rank/world16 identity mismatch")
    require(torch.cuda.is_available() and torch.__version__.startswith("2.10.") and torch.version.cuda == "12.8",
            "capture requires admitted native Torch2.10/CUDA12.8 runtime")
    native_device = "cuda:" + str(torch.cuda.current_device())
    require(str(trainer.shared_model.device) == native_device, "native model/device identity mismatch")
    packet_path = _checked_reference(dict(path=env.get("H3_ATTENTION_TUPLE_CAPTURE_CONFIG"),
                                        sha256=env.get("H3_ATTENTION_TUPLE_CAPTURE_CONFIG_SHA256")))
    require(packet_path.stat().st_size <= 1 << 20, "capture metadata packet exceeds 1MiB")
    packet = json.loads(packet_path.read_text())
    bind_prepared = packet["schema_version"] == "distillschedule.h3_native_attention_capture_opt_in.v2"
    expected_geometry = native_capture_geometry(packet)
    require(packet["candidate_id"] == impl.id
            and packet["run_id"] == env.get("DISTILLSCHEDULE_RUN_ID"), "capture run/selector/candidate binding mismatch")
    require(packet.get("profile") is False and packet.get("formal_timing_eligible") is False,
            "capture is correctness-only; profile and timing eligibility must be explicitly false")
    expected_target = dict(rank=0, world_size=16, cycle=1, role="fake", fake_index=0, micro_index=0,
                           block_index=10, versions=dict(student=1, fake=0, ema=0))
    require(_exact_json(packet["target"], expected_target), "capture target rank/role/cycle/block/version metadata mismatch")
    q_shape, lse_shape = expected_geometry["q_shape"], expected_geometry["lse_shape"]
    expected_input = packet["expected_input"]
    require(isinstance(expected_input, dict) and set(expected_input) == {"sample_identity", "latent_shape", "cache_key"}
            and isinstance(expected_input["sample_identity"], str) and bool(json.loads(expected_input["sample_identity"]))
            and isinstance(expected_input["latent_shape"], dict) and bool(expected_input["latent_shape"])
            and isinstance(expected_input["cache_key"], list) and len(expected_input["cache_key"]) == 2
            and expected_input["cache_key"][0] == "score" and type(expected_input["cache_key"][1]) is int
            and expected_input["cache_key"][1] >= 0,
            "missing exact prepared input identity/latent/cache metadata")
    sample = json.loads(expected_input["sample_identity"])
    require(isinstance(sample, dict) and all(isinstance(value, str) for value in sample.values())
            and set(sample) <= {"id", "row_index", "condition_path", "prompt_path", "lmdb_path",
                                "video_latent_path", "audio_latent_path"}
            and json.dumps(sample, sort_keys=True, separators=(",", ":")) == expected_input["sample_identity"],
            "input identity must use the original canonical sample metadata")
    provenance = packet["provenance"]
    require(provenance["source_head"] == observer.source_head and provenance["run_kind"] == "correctness",
            "capture source/mode mismatch")
    require(provenance["runtime_identity"] == native_runtime_identity(torch), "capture runtime identity mismatch")
    require(provenance["provider_schema"] == RAW_BACKWARD_SCHEMA and bool(provenance["provider_files"]),
            "missing/wrong original raw22 provider metadata")
    for path, checksum in provenance["provider_files"].items():
        _checked_reference(dict(path=path, sha256=checksum))
    references = packet["references"]
    require(set(references) == {"native_config", "input_manifest", "owner_receipt", "capacity_receipt", "source_layout"},
            "missing original root admission references")
    for name, reference in references.items():
        path = _checked_reference(reference)
        require(provenance[name + "_sha256"] == reference["sha256"], "capture " + name + " binding mismatch")
        if name == "native_config":
            require(path.resolve() == Path(env["H3_CONFIG"]).resolve(), "native config path mismatch")
    directory = Path(__file__).resolve().parent
    require(Path(references["source_layout"]["path"]).resolve() == directory.parents[4] / "docs/workloads/source_layout.lock.json",
            "source-layout reference must name the original in-repo lock")
    require(set(packet["source_files"]) == {str(directory / name) for name in _NATIVE_FILES},
            "missing/wrong reviewed native source file inventory")
    for path, checksum in packet["source_files"].items():
        _checked_reference(dict(path=path, sha256=checksum))
    budget = CaptureBudget(**packet["budget"])
    ceilings = CaptureBudget(4339077120, 13017231360, 541327360, 6442450944, 2147483648)
    require(all(value <= ceilings.__dict__[name] for name, value in budget.__dict__.items()),
            "capture budget exceeds reviewed bounded proposal")
    largest = math.prod(q_shape) * 2
    payload = 8 * largest + math.prod(lse_shape) * 4
    require(budget.max_tensor_bytes >= largest and budget.max_payload_bytes >= payload
            and budget.max_replay_span_bytes >= payload and budget.max_file_bytes >= payload
            and budget.max_host_working_bytes >= 2 * largest + (8 << 20),
            "capture budget cannot hold the declared original tuple")
    run_root = Path(env["DISTILLSCHEDULE_RUN_ROOT"])
    require(run_root.is_absolute() and run_root.resolve().is_relative_to(REMOTE_PERSISTENT_ROOT.resolve()),
            "native tuple payload must remain in remote persistent project storage")
    require(packet["output_relative"] == "diagnostics/attention_tuple_rank000", "unsafe/wrong capture output binding")
    output = run_root / packet["output_relative"]
    require(not output.is_symlink() and not output.exists(), "capture output already exists")
    capture = TupleCapture(target=CaptureTarget(), rank=rank, budget=budget, provenance=provenance,
                           io=_NativeTorchSnapshotIO(torch, native_device=native_device), output=output,
                           q_shape=q_shape, lse_shape=lse_shape)
    capture.native_expected_input = json_copy(expected_input)
    capture.native_bind_prepared_geometry = bind_prepared
    capture.native_geometry = json_copy(expected_geometry)
    capture.native_selector = packet["selector"]
    capture.native_packet_sha256 = env["H3_ATTENTION_TUPLE_CAPTURE_CONFIG_SHA256"]
    if rank == capture.target.rank:
        output.parent.mkdir(exist_ok=True)
        require(output.parent.resolve().is_relative_to(run_root.resolve()), "capture output parent escapes run root")
    trainer.attention_tuple_capture = capture
    return capture


def finish_native_capture(capture, row):
    """Only call AFTER original train + final observer/state gates returned."""
    complete = capture.require_complete_after_native()
    diagnostic = dict(status="CAPTURED_AFTER_NATIVE_SEMANTIC_GATES" if complete else "NOT_SELECTED_NO_CAPTURE_IO",
                      selector=capture.native_selector, rank=capture.rank, target_rank=capture.target.rank,
                      packet_sha256=capture.native_packet_sha256, formal_timing_eligible=False,
                      numerical_cross_reference="NOT_RUN_W5_FAIL_RETAINED")
    if complete:
        # Integrity only. Full dout scan is a separate post-native CPU consumer.
        load_capture(capture.output, max_file_bytes=capture.budget.max_file_bytes,
                     expected_provenance=capture.provenance)
        diagnostic.update(path=str(capture.output), manifest_sha256=digest_file(capture.output / "manifest.json"))
    # The legacy native timing/cost/pair consumer admits only the unmodified
    # PASS_NATIVE_RUNTIME_CONTRACTS status. Do not let it silently aggregate an
    # instrumented correctness run merely because its old timer fields exist.
    row.update(native_runtime_contract_status=row["status"],
               status="PASS_NATIVE_RUNTIME_CONTRACTS_CAPTURE_DIAGNOSTIC_ONLY",
               original_observer_timing_scope=row.get("timing_scope"),
               timing_scope="CAPTURE_INSTRUMENTED_CORRECTNESS_ONLY_NOT_PERFORMANCE",
               attention_tuple_capture=diagnostic, performance_timing_eligible=False, formal_timing_eligible=False,
               timing_exclusion_reason="ATTENTION_TUPLE_CAPTURE_SYNCHRONIZATION_AND_IO",
               numerical_cross_reference="NOT_RUN_W5_FAIL_RETAINED")
    for cycle in row["cycles"]:
        cycle.update(performance_timing_eligible=False, formal_timing_eligible=False,
                     timing_exclusion_reason=row["timing_exclusion_reason"])


def close_native_capture(trainer, capture):
    """Release observer references on success/failure without a new import/fence."""
    if getattr(trainer, "attention_tuple_capture", None) is capture:
        del trainer.attention_tuple_capture
    module = sys.modules.get(__package__ + ".fa3_replay_cache")
    runtime = getattr(module, "_RUNTIME", None)
    if runtime is not None and getattr(runtime, "tuple_capture", None) is capture:
        runtime.tuple_capture = None
    capture.prepared.clear()
    capture.active, capture.trainer = None, None
