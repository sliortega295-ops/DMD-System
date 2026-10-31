"""Explicit finite W2/W3 record-pool epochs; never a trainer or trace replayer.

Immutable native features may repeat across cohorts, not within a cohort. The
unchanged native capture owns new history/noise at every logical operation.
Absent this policy, the original globally-fresh v1 input ABI is unchanged.
"""
from __future__ import annotations

from pathlib import Path
import random

from distillschedule.core.contracts import require
from distillschedule.workloads.best_known.registry import fingerprint

EPOCH_POLICY = "distillschedule.helios_record_epochs.v1"
SOURCE_SCHEMA = "distillschedule.helios_record_pool.v1"
EVIDENCE_KINDS = ("ENCODED_FEATURES_UNVERIFIED_QUALITY", "CPU_SYNTHETIC_TEST_ONLY")
_SOURCE_FIELDS = {"schema_version", "seed", "cycles", "evidence_kind", "records"}
_RECORD_FIELDS = {"id", "path", "sha256"}


def _source_records(value, task):
    from runtime.operation_trace import tensor_sha256
    from distillschedule.strategies.wan_inputs import load_feature
    require(isinstance(value, dict) and set(value) == _SOURCE_FIELDS
            and value["schema_version"] == SOURCE_SCHEMA,
            "complete explicit finite record pool required", path="helios.epochs")
    require(type(value["seed"]) is int and 0 <= value["seed"] < 2**63
            and type(value["cycles"]) is int and 2 <= value["cycles"] <= 100
            and value["evidence_kind"] in EVIDENCE_KINDS,
            "finite 2..100 cycles, 63-bit seed and explicit evidence kind required", path="helios.epochs")
    records = value["records"]
    require(isinstance(records, list) and len(records) >= 80,
            "at least 80 distinct original feature records required", path="helios.epochs")
    rows, ids, paths, hashes, contents = [], set(), set(), set(), set()
    for record in records:
        require(isinstance(record, dict) and set(record) == _RECORD_FIELDS
                and isinstance(record["id"], str) and record["id"] and record["id"] not in ids
                and isinstance(record["path"], str) and Path(record["path"]).is_absolute()
                and isinstance(record["sha256"], str) and len(record["sha256"]) == 64,
                "explicit unique record IDs and pinned absolute paths required", path="helios.epochs")
        path = Path(record["path"]).resolve()
        require(path.is_file() and path.suffix in {".pt", ".safetensors"}
                and str(path) not in paths and record["sha256"] not in hashes,
                "distinct original paths/payloads required; aliases are not records", path="helios.epochs")
        # The existing restricted loader enforces native geometry, padding,
        # finiteness and the exact declared file hash before/after reading.
        latent, text, mask = load_feature(task, dict(path=str(path), sha256=record["sha256"]))
        # Binary mask encodings have the same native meaning; changing only
        # bool/int/float storage must not manufacture another source record.
        # Keep the original tensors and payload hash for actual capture.
        content = fingerprint([tensor_sha256(t.detach().contiguous().clone())
                               for t in (latent, text, mask.bool())])
        require(content not in contents, "reserialized identical native features are not distinct records",
                path="helios.epochs")
        rows.append(dict(id=record["id"], path=str(path), sha256=record["sha256"],
                         content_sha256=content, latent_shape=list(latent.shape), text_shape=list(text.shape)))
        ids.add(record["id"]); paths.add(str(path)); hashes.add(record["sha256"]); contents.add(content)
    require(len({tuple(row["text_shape"]) for row in rows}) == 1,
            "rank/step conditioning geometry must match", path="helios.epochs")
    return rows


def build_cohorts(value, *, task_type, input_policy):
    """Read each immutable source once; logical ordering uses no training RNG.

    One local seeded permutation repeats to the finite horizon, including a
    partial final epoch. Any consecutive 80 draws are distinct for pool >=80,
    including a cohort crossing an epoch boundary. No source files are made.
    """
    from distillschedule.training.helios_inputs import HeliosCohortSources
    require(input_policy == EPOCH_POLICY, "unsupported explicit Helios input policy", path="helios.epochs")
    records = _source_records(value, task_type(0))
    order = list(range(len(records)))
    random.Random(value["seed"]).shuffle(order)
    inventory_sha = fingerprint(records)
    sources, inventories = [], []
    for index in range(value["cycles"]):
        task, rows = task_type(index), []
        for offset, step in enumerate(task.source_steps):
            for rank in range(16):
                draw_index = index * 80 + offset * 16 + rank
                epoch, position = divmod(draw_index, len(records))
                record = records[order[position]]
                draw = dict(schema_version=EPOCH_POLICY, seed=value["seed"], inventory_sha256=inventory_sha,
                            cohort=index, step=step, rank=rank, record_id=record["id"],
                            source_sha256=record["sha256"], draw_index=draw_index,
                            epoch_index=epoch, epoch_offset=position)
                rows.append(dict(rank=rank, step=step, path=record["path"], sha256=record["sha256"],
                    latent_shape=record["latent_shape"], text_shape=record["text_shape"], record_id=record["id"],
                    draw_index=draw_index, epoch_index=epoch, epoch_offset=position, draw_id=fingerprint(draw)))
        require(len({row["record_id"] for row in rows}) == 80,
                "80 distinct records per native cohort required", path="helios.epochs")
        inventories.append(rows)
        sources.append(HeliosCohortSources(value["seed"], tuple(
            tuple(row["path"] for row in rows[offset*16:(offset+1)*16]) for offset in range(5))))
    draw_count = 80 * value["cycles"]
    policy = dict(schema_version=EPOCH_POLICY, evidence_kind=value["evidence_kind"], seed=value["seed"],
        cycles=value["cycles"], records=records, record_order=[records[i]["id"] for i in order],
        record_inventory_sha256=inventory_sha, draw_count=draw_count,
        epoch_count=(draw_count + len(records)-1)//len(records),
        draws_sha256=fingerprint([[row["draw_id"] for row in rows] for rows in inventories]))
    policy["policy_sha256"] = fingerprint(policy)
    return sources, inventories, policy


def validate_identity(identity, *, task_type):
    """Recompute producer ordering and actual source identities, not just hashes."""
    policy = identity["input_policy"]
    require(isinstance(policy, dict) and policy.get("schema_version") == EPOCH_POLICY,
            "unsupported explicit Helios input policy", path="helios.epochs")
    source = dict(schema_version=SOURCE_SCHEMA, seed=policy["seed"], cycles=policy["cycles"],
                  evidence_kind=policy["evidence_kind"],
                  records=[{key: row[key] for key in _RECORD_FIELDS} for row in policy["records"]])
    sources, inventories, expected = build_cohorts(source, task_type=task_type, input_policy=EPOCH_POLICY)
    require(policy == expected and identity["sources"] == [s.to_dict() for s in sources]
            and identity["inventory"] == inventories,
            "epoch policy/source/draw order or identity changed", path="helios.epochs")


def cohort_identity(identity, index):
    if "input_policy" not in identity:
        return {}
    policy, rows = identity["input_policy"], identity["inventory"][index]
    return {"input_policy": dict(schema_version=EPOCH_POLICY, policy_sha256=policy["policy_sha256"],
        cohort_id=fingerprint(dict(policy_sha256=policy["policy_sha256"], index=index,
                                   draw_ids=[row["draw_id"] for row in rows])),
        first_draw=index*80, draw_count=80,
        record_ids_sha256=fingerprint([row["record_id"] for row in rows]),
        draw_ids_sha256=fingerprint([row["draw_id"] for row in rows]))}


def identity_binding(stream):
    """Explicit checkpoint pin; the existing stream SHA additionally binds all rows."""
    policy = stream["identity"].get("input_policy")
    if policy is None:
        return {}
    return {"input_policy": {key: policy[key] for key in ("schema_version", "policy_sha256",
        "record_inventory_sha256", "draws_sha256", "draw_count")}}


def receipt(stream, *, cycle):
    policy = stream["identity"].get("input_policy")
    if policy is None:
        return {}
    require(type(cycle) is int and 0 <= cycle <= policy["cycles"],
            "epoch cursor outside declared finite horizon", path="helios.epochs")
    return dict(**identity_binding(stream), source_cohort_policy=EPOCH_POLICY,
        source_record_count=len(policy["records"]), logical_draw_count=policy["draw_count"],
        source_reuse_across_cohorts=policy["draw_count"] > len(policy["records"]),
        input_draw_cursor=80*cycle, evidence_kind=policy["evidence_kind"])


def validate_cursor(header, stream):
    """Policy-only cursor tightening; frozen v1 boundary behavior is untouched."""
    policy = stream.get("identity", {}).get("input_policy")
    if policy is not None:
        require(type(header["cursor"]) is int and header["cursor"] == header["cycle"]
                and 0 <= header["cursor"] <= policy["cycles"],
                "epoch cursor must be an integer complete-cohort boundary", path="helios.epochs")
