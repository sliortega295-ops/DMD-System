"""Build GAN fairness/compute receipts from *observed* execution evidence.

The operation trace is the workload contract, not proof that a worker consumed
it.  This module joins immutable trace descriptors with append-only tensor
access receipts and frozen-block materialization events.  A receipt is emitted
only when every required stochastic/history value was actually accessed and
the aggregate physical DiT/Score call count agrees with the logical trace.

The builder is intentionally usable by both the sequential world16 baseline
and the persistent 8G+8Score candidate.  Physical rank placement is validated
before it is erased by the canonical fairness comparison.
"""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from .gan_compute_inventory import (
    CRITIC_FIXED_CALLS,
    GENERATOR_FIXED_CALLS,
    normalize_gan_compute_receipt,
    validate_compute_against_fairness,
)
from .gan_fairness import (
    FULL5_STEPS,
    GENERATOR_UPDATE_STEPS,
    LOGICAL_SAMPLE_SLOTS,
    GanLogicalSampleAddress,
    normalize_gan_fairness_receipt,
    object_digest,
)


PROVENANCE_SCHEMA = "distillschedule.gan_execution_provenance.v1"

_SOURCE_PRIMARY = ("primary/x0", "primary/history", "primary/target")
_SOURCE_SECONDARY = ("secondary/x0", "secondary/history", "secondary/target")
_HISTORY_SUFFIXES = (
    "continuous_history",
    "history_short",
    "history_mid",
    "history_long",
    "indices_hidden_states",
    "indices_history_short",
    "indices_history_mid",
    "indices_history_long",
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"malformed JSONL {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"non-object JSONL row {path}:{line_number}")
            rows.append(row)
    return rows


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _operation_digest(operation: Mapping[str, Any]) -> str:
    return object_digest(operation)


def _load_bundle(root: Path) -> tuple[str, dict[int, dict[str, Any]]]:
    bundle = _load_json(root / "bundle.json")
    if bundle.get("schema_version") != "distillschedule.operation_trace_bundle.v1":
        raise ValueError("unsupported operation-trace bundle schema")
    ranks_raw = bundle.get("ranks")
    if not isinstance(ranks_raw, list):
        raise ValueError("trace bundle ranks must be a list")
    ranks: dict[int, dict[str, Any]] = {}
    for item in ranks_raw:
        if not isinstance(item, Mapping):
            raise ValueError("trace bundle rank entry must be an object")
        rank = int(item.get("rank", -1))
        if rank in ranks:
            raise ValueError(f"duplicate trace rank {rank}")
        trace = _load_json(root / str(item["path"]) / "trace.json")
        if trace.get("trace_hash") != item.get("trace_hash"):
            raise ValueError(f"trace hash drift for rank {rank}")
        operations = trace.get("operations")
        tensors = trace.get("tensors")
        if not isinstance(operations, list) or not isinstance(tensors, Mapping):
            raise ValueError(f"trace rank {rank} lacks operations/tensors")
        op_by_key = {str(operation["key"]): operation for operation in operations}
        if len(op_by_key) != len(operations):
            raise ValueError(f"duplicate operation key in trace rank {rank}")
        ranks[rank] = {
            "trace_hash": str(trace["trace_hash"]),
            "operations": op_by_key,
            "tensors": dict(tensors),
        }
    if set(ranks) != set(LOGICAL_SAMPLE_SLOTS):
        raise ValueError(f"trace bundle rank inventory drift: {sorted(ranks)}")
    bundle_hash = str(bundle.get("bundle_hash", ""))
    if len(bundle_hash) != 64:
        raise ValueError("trace bundle hash must be SHA256")
    return bundle_hash, ranks


def _load_accesses(
    access_root: Path,
    traces: Mapping[int, Mapping[str, Any]],
) -> tuple[
    dict[tuple[int, str], list[dict[str, Any]]],
    dict[str, int],
]:
    slot_by_trace = {str(value["trace_hash"]): slot for slot, value in traces.items()}
    if len(slot_by_trace) != len(traces):
        raise ValueError("trace hashes are not rank-qualified")
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    file_counts: dict[str, int] = {}
    paths = sorted(access_root.glob("rank_*.jsonl"))
    if not paths:
        raise ValueError(f"no actual-consumption access logs under {access_root}")
    for path in paths:
        rows = _load_jsonl(path)
        file_counts[path.name] = len(rows)
        for row in rows:
            trace_hash = str(row.get("trace_hash", ""))
            if trace_hash not in slot_by_trace:
                raise ValueError(f"unknown consumed trace hash in {path}")
            slot = slot_by_trace[trace_hash]
            key = str(row.get("key", ""))
            descriptor = traces[slot]["tensors"].get(key)
            if not isinstance(descriptor, Mapping):
                raise ValueError(f"accessed unknown tensor key slot={slot}: {key}")
            if row.get("sha256") != descriptor.get("sha256"):
                raise ValueError(f"actual consumed tensor hash drift slot={slot}: {key}")
            rank = int(row.get("rank", -1))
            if rank not in LOGICAL_SAMPLE_SLOTS:
                raise ValueError(f"invalid access rank {rank}")
            grouped[(slot, key)].append(dict(row))
    return grouped, file_counts


def _required_rank_set(family: str, slot: int, key: str) -> set[int]:
    """Return the physical ranks that must consume one logical tensor.

    The split executor deliberately replays source/conditioning tensors on
    both role ranks.  Generator rollout randomness remains Generator-owned,
    while DMD/GAN/Critic-loss randomness is Score-owned.  Requiring equality
    (rather than merely accepting a subset) prevents a receipt from passing
    when one side silently skipped a required replay lookup.
    """

    if family == "baseline":
        return {slot}
    address = GanLogicalSampleAddress.for_slot(slot)
    generator_rank = address.pipeline_generator_rank
    score_rank = address.pipeline_score_rank
    if key.startswith("sample/"):
        return {generator_rank, score_rank}
    if "/generator/" in key:
        score_suffixes = ("dmd_timestep", "dmd_noise", "gan_g_noise", "gan_g_crop")
        if key.endswith(score_suffixes):
            return {score_rank}
        if key.endswith(_HISTORY_SUFFIXES):
            return {generator_rank, score_rank}
        return {generator_rank}
    if "/critic/" in key:
        score_suffixes = (
            "critic_timestep",
            "critic_noise",
            "gan_d_fake_crop",
            "gan_d_real_crop",
            "gan_d_r1_crop",
            "r1_epsilon",
        )
        if key.endswith(score_suffixes):
            return {score_rank}
        if key.endswith(_HISTORY_SUFFIXES) or key.endswith("entry_noise"):
            return {generator_rank, score_rank}
        return {generator_rank}
    raise ValueError(f"unclassified operation-trace tensor key: {key}")


def _observed_hash(
    *,
    family: str,
    slot: int,
    key: str,
    accesses: Mapping[tuple[int, str], list[dict[str, Any]]],
    traces: Mapping[int, Mapping[str, Any]],
) -> str:
    rows = accesses.get((slot, key), [])
    if not rows:
        raise ValueError(f"required tensor was not actually consumed: slot={slot} key={key}")
    required = _required_rank_set(family, slot, key)
    observed_ranks = {int(row["rank"]) for row in rows}
    if observed_ranks != required:
        raise ValueError(
            f"tensor physical-consumer set drift slot={slot} key={key}: "
            f"observed={sorted(observed_ranks)} required={sorted(required)}"
        )
    descriptor = traces[slot]["tensors"][key]
    return str(descriptor["sha256"])


def _packed_hash(
    *,
    family: str,
    slot: int,
    keys: Iterable[str],
    accesses: Mapping[tuple[int, str], list[dict[str, Any]]],
    traces: Mapping[int, Mapping[str, Any]],
) -> str:
    ordered = [
        {
            "key": key,
            "sha256": _observed_hash(
                family=family,
                slot=slot,
                key=key,
                accesses=accesses,
                traces=traces,
            ),
        }
        for key in keys
    ]
    return object_digest(ordered)


def _dit(operation: Mapping[str, Any]) -> dict[str, Any]:
    control = operation.get("control")
    if not isinstance(control, Mapping):
        raise ValueError("rollout operation lacks control")
    schedule = control.get("pyramid_stage_call_sequence")
    if not isinstance(schedule, list) or len(schedule) != 3:
        raise ValueError("rollout operation lacks three-stage schedule")
    schedule_by_stage = {int(item["stage"]): item for item in schedule}
    if set(schedule_by_stage) != {0, 1, 2}:
        raise ValueError("rollout stage inventory drift")
    exits = [int(schedule_by_stage[stage]["denoising_exit"]) for stage in range(3)]
    counts = [int(schedule_by_stage[stage]["call_count"]) for stage in range(3)]
    if counts != [value + 1 for value in exits]:
        raise ValueError("trace rollout call count disagrees with stage exits")
    if int(control.get("call_count", -1)) != sum(counts):
        raise ValueError("trace rollout total call count drift")
    return {
        "stage_schedule_scope": control.get("stage_schedule_scope"),
        "stage_exits": exits,
        "stage_call_counts": counts,
        "total_dit_forwards": sum(counts),
    }


def _source_roles(
    family: str,
    slot: int,
    accesses: Mapping[tuple[int, str], list[dict[str, Any]]],
    traces: Mapping[int, Mapping[str, Any]],
) -> dict[str, str]:
    # Each rank-qualified Helios trace stores its *local* batch item under a
    # local sample index (currently ``sample/0``).  The logical global slot is
    # already bound by the rank-qualified trace hash; it must not be reused as
    # a tensor-key index.  Accept either local or globally numbered trace
    # bundles, but require one unambiguous source prefix per logical slot.
    source_prefixes = {
        "/".join(str(key).split("/")[:2])
        for key in traces[slot]["tensors"]
        if str(key).startswith("sample/") and len(str(key).split("/")) >= 3
    }
    if len(source_prefixes) != 1:
        raise ValueError(
            f"logical slot {slot} has ambiguous source prefixes: "
            f"{sorted(source_prefixes)}"
        )
    prefix = next(iter(source_prefixes))
    return {
        "source_primary": _packed_hash(
            family=family,
            slot=slot,
            keys=(f"{prefix}/{suffix}" for suffix in _SOURCE_PRIMARY),
            accesses=accesses,
            traces=traces,
        ),
        "source_secondary": _packed_hash(
            family=family,
            slot=slot,
            keys=(f"{prefix}/{suffix}" for suffix in _SOURCE_SECONDARY),
            accesses=accesses,
            traces=traces,
        ),
        "text_embedding": _observed_hash(
            family=family,
            slot=slot,
            key=f"{prefix}/text_embedding",
            accesses=accesses,
            traces=traces,
        ),
    }


def _phase(
    *,
    family: str,
    slot: int,
    step: int,
    component: str,
    source: Mapping[str, str],
    accesses: Mapping[tuple[int, str], list[dict[str, Any]]],
    traces: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    prefix = f"step/{step}/{component}"
    operation = traces[slot]["operations"].get(f"{prefix}/rollout")
    if not isinstance(operation, Mapping):
        raise ValueError(f"missing rollout operation: slot={slot} {prefix}")
    history = _packed_hash(
        family=family,
        slot=slot,
        keys=(f"{prefix}/{suffix}" for suffix in _HISTORY_SUFFIXES),
        accesses=accesses,
        traces=traces,
    )
    if component == "generator":
        consumed = {
            **source,
            "processed_generator_history": history,
            "generator_rollout_entry_noise": _observed_hash(
                family=family, slot=slot, key=f"{prefix}/entry_noise", accesses=accesses, traces=traces
            ),
            "generator_rollout_renoise_sequence": _packed_hash(
                family=family, slot=slot, keys=(f"{prefix}/rollout_noise",), accesses=accesses, traces=traces
            ),
            "generator_block_noise_stage1": _observed_hash(
                family=family, slot=slot, key=f"{prefix}/block_noise/1", accesses=accesses, traces=traces
            ),
            "generator_block_noise_stage2": _observed_hash(
                family=family, slot=slot, key=f"{prefix}/block_noise/2", accesses=accesses, traces=traces
            ),
            "generator_dmd_timestep": _observed_hash(
                family=family, slot=slot, key=f"{prefix}/dmd_timestep", accesses=accesses, traces=traces
            ),
            "generator_dmd_noise": _observed_hash(
                family=family, slot=slot, key=f"{prefix}/dmd_noise", accesses=accesses, traces=traces
            ),
            "generator_gan_g_noise": _observed_hash(
                family=family, slot=slot, key=f"{prefix}/gan_g_noise", accesses=accesses, traces=traces
            ),
            "generator_gan_g_crop": _observed_hash(
                family=family, slot=slot, key=f"{prefix}/gan_g_crop", accesses=accesses, traces=traces
            ),
        }
    else:
        consumed = {
            **source,
            "processed_critic_history": history,
            "critic_rollout_entry_noise": _observed_hash(
                family=family, slot=slot, key=f"{prefix}/entry_noise", accesses=accesses, traces=traces
            ),
            "critic_rollout_renoise_sequence": _packed_hash(
                family=family, slot=slot, keys=(f"{prefix}/rollout_noise",), accesses=accesses, traces=traces
            ),
            "critic_block_noise_stage1": _observed_hash(
                family=family, slot=slot, key=f"{prefix}/block_noise/1", accesses=accesses, traces=traces
            ),
            "critic_block_noise_stage2": _observed_hash(
                family=family, slot=slot, key=f"{prefix}/block_noise/2", accesses=accesses, traces=traces
            ),
            "critic_timestep": _observed_hash(
                family=family, slot=slot, key=f"{prefix}/critic_timestep", accesses=accesses, traces=traces
            ),
            "critic_noise": _observed_hash(
                family=family, slot=slot, key=f"{prefix}/critic_noise", accesses=accesses, traces=traces
            ),
            "gan_d_fake_crop": _observed_hash(
                family=family, slot=slot, key=f"{prefix}/gan_d_fake_crop", accesses=accesses, traces=traces
            ),
            "gan_d_real_crop": _observed_hash(
                family=family, slot=slot, key=f"{prefix}/gan_d_real_crop", accesses=accesses, traces=traces
            ),
            "gan_d_r1_crop": _observed_hash(
                family=family, slot=slot, key=f"{prefix}/gan_d_r1_crop", accesses=accesses, traces=traces
            ),
            "r1_epsilon": _observed_hash(
                family=family, slot=slot, key=f"{prefix}/r1_epsilon", accesses=accesses, traces=traces
            ),
        }
    return {
        "trace_operation_sha256": _operation_digest(operation),
        "dit": _dit(operation),
        "consumed": consumed,
    }


def _physical(family: str, slot: int) -> dict[str, int]:
    address = GanLogicalSampleAddress.for_slot(slot)
    if family == "baseline":
        return {"rank": address.baseline_rank}
    return {
        "wave": address.pipeline_wave,
        "generator_rank": address.pipeline_generator_rank,
        "score_rank": address.pipeline_score_rank,
    }


def _logical_calls(trace: Mapping[str, Any], component: str, step: int) -> int:
    operation = trace["operations"].get(f"step/{step}/{component}/rollout")
    if not isinstance(operation, Mapping):
        raise ValueError(f"missing {component} rollout at step {step}")
    return int(_dit(operation)["total_dit_forwards"])


def _block_event_proof(
    *,
    family: str,
    run_root: Path,
    traces: Mapping[int, Mapping[str, Any]],
    expected_streamed_block_count: int,
) -> dict[str, Any]:
    if expected_streamed_block_count <= 0:
        raise ValueError("physical call proof requires at least one streamed block")
    expected_by_rank: dict[tuple[int, str], int] = {}
    if family == "baseline":
        for rank in LOGICAL_SAMPLE_SLOTS:
            expected_by_rank[(rank, "generator")] = (
                _logical_calls(traces[rank], "generator", 1000)
                + sum(
                    _logical_calls(traces[rank], "critic", step)
                    for step in FULL5_STEPS
                )
            )
            expected_by_rank[(rank, "score")] = len(GENERATOR_UPDATE_STEPS) * sum(
                GENERATOR_FIXED_CALLS[key]
                for key in (
                    "dmd_fake_cond_score_forwards",
                    "dmd_fake_uncond_score_forwards",
                    "dmd_real_cond_score_forwards",
                    "dmd_real_uncond_score_forwards",
                    "gan_g_score_forwards",
                )
            ) + len(FULL5_STEPS) * (
                CRITIC_FIXED_CALLS["critic_denoise_score_forwards"]
                + CRITIC_FIXED_CALLS["gan_d_score_forwards"]
            )
    else:
        for lane in range(8):
            slots = (lane, lane + 8)
            expected_by_rank[(lane, "generator")] = sum(
                _logical_calls(traces[slot], "generator", 1000)
                + sum(
                    _logical_calls(traces[slot], "critic", step)
                    for step in FULL5_STEPS
                )
                for slot in slots
            )
            expected_by_rank[(lane + 8, "score")] = len(slots) * (
                len(GENERATOR_UPDATE_STEPS) * 4 + len(FULL5_STEPS) * 2
            )

    block_roots = [
        candidate
        for candidate in (
            run_root / "block_streaming",
            run_root / "results" / "block_streaming",
        )
        if candidate.is_dir()
    ]
    if len(block_roots) != 1:
        raise ValueError(
            "expected exactly one block-streaming evidence root, observed "
            f"{[str(path) for path in block_roots]}"
        )
    block_root = block_roots[0]

    proof: dict[str, Any] = {}
    for (rank, component), expected in sorted(expected_by_rank.items()):
        path = block_root / f"rank_{rank:03d}_{component}.jsonl"
        if not path.is_file():
            raise ValueError(f"missing physical call evidence: {path}")
        rows = _load_jsonl(path)
        installed = [row for row in rows if row.get("event") == "streamer_installed"]
        if len(installed) != 1:
            raise ValueError(f"streamer install receipt drift: {path}")
        indices = [int(value) for value in installed[0].get("streamed_block_indices", [])]
        if len(indices) != expected_streamed_block_count:
            raise ValueError(
                f"streamed block policy drift in {path}: {len(indices)} != {expected_streamed_block_count}"
            )
        logical_counts: Counter[int] = Counter()
        recompute_counts: Counter[int] = Counter()
        release_counts: Counter[int] = Counter()
        for row in rows:
            event = row.get("event")
            if event == "materialize_block":
                block = int(row["block_index"])
                if row.get("generation") == "CHECKPOINT_RECOMPUTE":
                    recompute_counts[block] += 1
                else:
                    logical_counts[block] += 1
            elif event == "release_block":
                release_counts[int(row["block_index"])] += 1
        if set(logical_counts) != set(indices):
            raise ValueError(f"logical block evidence inventory drift: {path}")
        if any(logical_counts[index] != expected for index in indices):
            raise ValueError(
                f"physical logical call count mismatch rank={rank} component={component}: "
                f"observed={dict(logical_counts)} expected_each={expected}"
            )
        materialized_total = sum(logical_counts.values()) + sum(recompute_counts.values())
        if sum(release_counts.values()) != materialized_total:
            raise ValueError(f"materialize/release imbalance: {path}")
        proof[f"rank_{rank:03d}_{component}"] = {
            "path": str(path.resolve()),
            "streamed_block_indices": indices,
            "expected_logical_forwards": expected,
            "observed_logical_forwards_per_block": expected,
            "checkpoint_recompute_forwards_per_block": (
                recompute_counts[indices[0]] if indices else 0
            ),
            "materialize_release_balanced": True,
        }
    return proof


def build_gan_execution_receipts(
    *,
    family: str,
    run_root: str | Path,
    trace_bundle: str | Path,
    expected_streamed_block_count: int = 40,
) -> dict[str, Any]:
    if family not in {"baseline", "pipeline"}:
        raise ValueError("family must be baseline or pipeline")
    run_root = Path(run_root).resolve(strict=True)
    trace_bundle = Path(trace_bundle).resolve(strict=True)
    bundle_hash, traces = _load_bundle(trace_bundle)
    accesses, access_file_counts = _load_accesses(run_root / "trace_access", traces)

    fairness_entries: list[dict[str, Any]] = []
    compute_entries: list[dict[str, Any]] = []
    for step in FULL5_STEPS:
        for slot in LOGICAL_SAMPLE_SLOTS:
            source = _source_roles(family, slot, accesses, traces)
            generator = (
                _phase(
                    family=family,
                    slot=slot,
                    step=step,
                    component="generator",
                    source=source,
                    accesses=accesses,
                    traces=traces,
                )
                if step in GENERATOR_UPDATE_STEPS
                else None
            )
            critic = _phase(
                family=family,
                slot=slot,
                step=step,
                component="critic",
                source=source,
                accesses=accesses,
                traces=traces,
            )
            fairness_entries.append(
                {
                    "logical_step": step,
                    "logical_sample_slot": slot,
                    "physical": _physical(family, slot),
                    "generator": generator,
                    "critic": critic,
                }
            )
            compute_entries.append(
                {
                    "logical_step": step,
                    "logical_sample_slot": slot,
                    "generator": (
                        {
                            "rollout_generator_dit_forwards": generator["dit"]["total_dit_forwards"],
                            **GENERATOR_FIXED_CALLS,
                        }
                        if generator is not None
                        else None
                    ),
                    "critic": {
                        "rollout_generator_dit_forwards": critic["dit"]["total_dit_forwards"],
                        **CRITIC_FIXED_CALLS,
                    },
                }
            )

    fairness_raw = {
        "schema_version": "distillschedule.gan_fairness_receipt.v1",
        "family": family,
        "status": "PASS",
        "trace_bundle_hash": bundle_hash,
        "r2_enabled": False,
        "entries": fairness_entries,
    }
    compute_raw = {
        "schema_version": "distillschedule.gan_compute_receipt.v1",
        "family": family,
        "status": "PASS",
        "entries": compute_entries,
    }
    fairness = normalize_gan_fairness_receipt(fairness_raw)
    compute = normalize_gan_compute_receipt(compute_raw)
    cross_check = validate_compute_against_fairness(compute_raw, fairness_raw)
    block_proof = _block_event_proof(
        family=family,
        run_root=run_root,
        traces=traces,
        expected_streamed_block_count=expected_streamed_block_count,
    )
    access_counts = {
        f"slot_{slot:02d}/{key}": len(rows)
        for (slot, key), rows in sorted(accesses.items())
    }
    provenance = {
        "schema_version": PROVENANCE_SCHEMA,
        "status": "PASS",
        "family": family,
        "run_root": str(run_root),
        "trace_bundle": str(trace_bundle),
        "trace_bundle_hash": bundle_hash,
        "access_file_counts": access_file_counts,
        "actual_access_counts": access_counts,
        "physical_call_proof": block_proof,
        "fairness_contract_digest": fairness["fairness_contract_digest"],
        "compute_contract_digest": compute["compute_contract_digest"],
        "cross_check": cross_check,
        "claim_boundary": (
            "Fairness is based on actual trace tensor lookups. Aggregate physical "
            "logical forwards are independently reconciled from streamed-block events; "
            "checkpoint recomputation is reported separately."
        ),
    }
    return {"fairness": fairness_raw, "compute": compute_raw, "provenance": provenance}


def write_gan_execution_receipts(
    *,
    family: str,
    run_root: str | Path,
    trace_bundle: str | Path,
    output_dir: str | Path,
    expected_streamed_block_count: int = 40,
) -> dict[str, str]:
    built = build_gan_execution_receipts(
        family=family,
        run_root=run_root,
        trace_bundle=trace_bundle,
        expected_streamed_block_count=expected_streamed_block_count,
    )
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "gan_fairness_receipt": output_dir / "gan_fairness_receipt.json",
        "gan_compute_receipt": output_dir / "gan_compute_receipt.json",
        "gan_execution_provenance": output_dir / "gan_execution_provenance.json",
    }
    _atomic_json(paths["gan_fairness_receipt"], built["fairness"])
    _atomic_json(paths["gan_compute_receipt"], built["compute"])
    _atomic_json(paths["gan_execution_provenance"], built["provenance"])
    return {key: str(path) for key, path in paths.items()}


__all__ = [
    "PROVENANCE_SCHEMA",
    "build_gan_execution_receipts",
    "write_gan_execution_receipts",
]
