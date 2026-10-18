"""Default-off full LOCAL LoRA/Adam diagnostic, never a timing/admission path.

No frozen backbone, global reconstruction, model call, collective or RNG draw.
CPU copies and serialization must run only outside the native cycle envelope.
Large tensors remain under the remote run root; receipts contain metadata only.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

from distillschedule.evidence.trajectory_state_sketch import canonical_state_name


SCHEMA = "distillschedule.h3_full_local_state.v1"
MAX_BYTES = 2**31
TAGS = ("initial", "cycle1", "cycle2")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_contract():
    return dict(schema_version=SCHEMA, helper_sha256=_sha(__file__),
                max_tensor_bytes_per_rank_snapshot=MAX_BYTES, tags=list(TAGS),
                tensor_scope="ALL_LOCAL_TRAINABLE_LORA_AND_NATIVE_ADAM_STATE_ONLY",
                capture_boundary="BEFORE_CYCLE1_OR_AFTER_EXISTING_CYCLE_STOP_FENCE",
                training_changes="NONE", native_timing="NOT_RUN", numerical_admission=False)


def _local(value):
    detached = value.detach()
    return detached.to_local() if hasattr(detached, "to_local") else detached


def inventory(trainer, torch):
    """Metadata-only prepass: reject unknown/over-budget state before any D2H."""
    names = {}
    for raw, parameter in trainer.shared_model.denoiser_module().named_parameters():
        name = canonical_state_name(raw)
        _require(id(parameter) not in names, "duplicate full-state parameter identity")
        names[id(parameter)] = name
    rows, seen = [], set()
    for role, parameters, optimizer in (
        ("student", trainer.trainable_params, trainer.optimizer),
        ("fake", trainer.fake_trainable_params, trainer.fake_optimizer),
    ):
        ids = [id(parameter) for parameter in parameters]
        _require(ids and len(set(ids)) == len(ids), "empty/duplicate diagnostic trainables")
        _require({id(p) for g in optimizer.param_groups for p in g["params"]} == set(ids),
                 "full-state optimizer/trainable inventory mismatch")
        for parameter in parameters:
            name = names.get(id(parameter), "")
            _require((".lora_A." in name or ".lora_B." in name) and f".{role}." in name,
                     "full local diagnostic may export only the named role LoRA")
            values = [("parameter", parameter), *sorted(optimizer.state.get(parameter, {}).items())]
            for kind, value in values:
                _require(kind in ("parameter", "step", "exp_avg", "exp_avg_sq"),
                         "unexpected optimizer state; do not silently omit it")
                key = f"{role}/{name}/{kind}"
                _require(key not in seen, "duplicate full local state key")
                seen.add(key)
                row = dict(key=key, role=role, name=name, kind=kind)
                if torch.is_tensor(value):
                    local = _local(value)
                    row.update(dtype=str(local.dtype), local_shape=list(local.shape),
                               stored_shape=list(value.shape), local_stride=list(local.stride()),
                               bytes=int(local.numel() * local.element_size()),
                               representation="DTENSOR_LOCAL_SHARD" if hasattr(value, "to_local") else "LOCAL_TENSOR",
                               placements=[str(x) for x in getattr(value, "placements", ())])
                    if hasattr(value, "device_mesh"):
                        row["mesh_shape"] = list(value.device_mesh.shape)
                    rows.append((row, value))
                else:
                    _require(type(value) in (int, float) and math.isfinite(value) and kind == "step",
                             "unsupported non-tensor native state")
                    rows.append((dict(row, representation="PYTHON_SCALAR", value=value, bytes=0), value))
    total = sum(row["bytes"] for row, _ in rows)
    _require(0 < total <= MAX_BYTES, "full local state exceeds fixed 2GiB per-snapshot bound")
    return sorted(rows, key=lambda item: item[0]["key"]), total


class FullLocalStateProbe:
    def __init__(self, *, torch, rank, source_head, run_root, run_id, arm, rng_snapshot):
        _require(type(rank) is int and 0 <= rank < 16 and len(source_head) == 40,
                 "invalid full local state source/rank")
        self.torch, self.rank, self.source_head = torch, rank, source_head
        self.run_root, self.run_id, self.arm = Path(run_root), run_id, arm
        self.rng_snapshot = rng_snapshot
        self.snapshots = []

    def capture(self, trainer, tag, *, outside_cycle):
        _require(outside_cycle and len(self.snapshots) < len(TAGS)
                 and tag == TAGS[len(self.snapshots)], "full local state capture order/boundary")
        directory = self.run_root / "execution/full-local-state" / f"rank_{self.rank:03d}" / tag
        _require(not directory.exists(), "full state refuses to overwrite evidence")
        rows, total = inventory(trainer, self.torch)
        before = self.rng_snapshot()
        cpu = {}
        for row, value in rows:
            if self.torch.is_tensor(value):
                # copy=True prevents a CPU optimizer tensor aliasing live state.
                copied = _local(value).to(device="cpu", copy=True).contiguous()
                _require(bool(self.torch.isfinite(copied).all()), "nonfinite full local state")
                cpu[row["key"]] = copied
        after = self.rng_snapshot()
        _require(before == after, "full-state observer changed the training RNG")
        directory.mkdir(parents=True, exist_ok=False)
        temporary, path = directory / "state.pt.tmp", directory / "state.pt"
        with temporary.open("xb") as handle:
            self.torch.save(cpu, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
        payload = dict(schema_version=SCHEMA, status="CAPTURED_FULL_LOCAL_STATE_DIAGNOSTIC_ONLY",
                       source_head=self.source_head, rank=self.rank, run_id=self.run_id, arm=self.arm,
                       tag=tag, rows=[row for row, _ in rows], tensor_bytes=total,
                       tensor_count=len(cpu), rng_before=before, rng_after=after,
                       state_file=dict(path=str(path), sha256=_sha(path), size_bytes=path.stat().st_size),
                       contract=source_contract(), full_global_reconstruction="NOT_RUN",
                       numerical_admission=False, formal_timing_eligible=False)
        metadata = directory / "manifest.json"
        with metadata.open("x") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        self.snapshots.append(dict(tag=tag, path=str(metadata), sha256=_sha(metadata),
                                   tensor_bytes=total, tensor_count=len(cpu)))
        del cpu

    def receipt(self):
        _require([s["tag"] for s in self.snapshots] == list(TAGS), "full local state snapshots incomplete")
        return dict(status="FULL_LOCAL_SHARDS_CAPTURED_NOT_NUMERICAL_ADMISSION",
                    contract=source_contract(), snapshots=list(self.snapshots),
                    numerical_admission=False, formal_timing_eligible=False)


def verify_receipt(receipt, *, root, rank, source_head, run_id, arm):
    """Real producer files drive the consumer; no synthetic GPU qualification."""
    _require(isinstance(receipt, dict) and receipt.get("contract") == source_contract()
             and receipt.get("status") == "FULL_LOCAL_SHARDS_CAPTURED_NOT_NUMERICAL_ADMISSION"
             and receipt.get("numerical_admission") is False and receipt.get("formal_timing_eligible") is False,
             "full-state receipt missing or contract/admission changed")
    snapshots = receipt.get("snapshots", [])
    _require([x.get("tag") for x in snapshots] == list(TAGS), "full-state snapshot coverage/order changed")
    parameter_inventory = None
    for tag, ref in zip(TAGS, snapshots, strict=True):
        directory = Path(root) / "execution/full-local-state" / f"rank_{rank:03d}" / tag
        metadata = directory / "manifest.json"
        _require(Path(ref.get("path", "")).resolve() == metadata.resolve() and metadata.is_file()
                 and ref.get("sha256") == _sha(metadata), "missing or hash-drifted full-state manifest")
        value = json.loads(metadata.read_text())
        _require(value.get("schema_version") == SCHEMA and value.get("source_head") == source_head
                 and value.get("rank") == rank and value.get("run_id") == run_id and value.get("arm") == arm
                 and value.get("tag") == tag and value.get("contract") == source_contract()
                 and value.get("numerical_admission") is False and value.get("formal_timing_eligible") is False
                 and value.get("rng_before") == value.get("rng_after"), "full-state source/rank/RNG/boundary mismatch")
        rows = value.get("rows", [])
        keys = [row["key"] for row in rows]
        _require(keys and len(set(keys)) == len(keys) and keys == sorted(keys)
                 and {row["role"] for row in rows} == {"student", "fake"}, "full-state inventory incomplete")
        params = [{k: row.get(k) for k in ("key", "local_shape", "stored_shape", "dtype", "placements", "mesh_shape")}
                  for row in rows if row["kind"] == "parameter"]
        if parameter_inventory is None:
            parameter_inventory = params
        _require(params == parameter_inventory, "full-state local trainable layout changed across cycles")
        count = sum(row.get("representation") != "PYTHON_SCALAR" for row in rows)
        size = sum(row["bytes"] for row in rows)
        _require(0 < size <= MAX_BYTES and size == value.get("tensor_bytes") == ref.get("tensor_bytes")
                 and count == value.get("tensor_count") == ref.get("tensor_count"), "full-state byte/count mismatch")
        state = directory / "state.pt"
        saved = value.get("state_file", {})
        _require(Path(saved.get("path", "")).resolve() == state.resolve() and state.is_file()
                 and saved.get("size_bytes") == state.stat().st_size and saved.get("sha256") == _sha(state),
                 "missing or hash-drifted full-state tensor file")
    return True
