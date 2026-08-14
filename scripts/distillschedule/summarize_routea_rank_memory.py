#!/usr/bin/env python3
"""Summarize existing low-overhead Route-A per-rank memory evidence.

This is an offline attribution tool.  It never attaches to a CUDA process and
therefore cannot extend tensor lifetime or change admission.  Values not owned
by the block-stream and saved-tensor instrumentation remain explicitly
unclassified rather than being inferred.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from typing import Any, Iterable


GIB = 1 << 30
SCHEMA = "distillschedule.routea_rank_memory.v1"


def _rows(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSONL {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise RuntimeError(f"non-object JSONL record {path}:{line_number}")
            result.append(value)
    return result


def _rank_from_name(path: Path) -> int:
    stem = path.stem
    token = stem.split("_")[1]
    return int(token)


def _last_phase(rows: Iterable[dict[str, Any]], phase: str) -> dict[str, Any]:
    matches = [row for row in rows if row.get("phase") == phase]
    if not matches:
        raise RuntimeError(f"missing required phase {phase!r}")
    return matches[-1]


def summarize_rank(run_root: Path, rank: int) -> dict[str, Any]:
    phase_file = run_root / "phase_trace" / f"rank_{rank:03d}.jsonl"
    activation_file = (
        run_root
        / "activation_offload"
        / f"rank_{rank:03d}_generator_selected.jsonl"
    )
    block_file = (
        run_root / "block_streaming" / f"rank_{rank:03d}_generator.jsonl"
    )
    immutable_file = (
        run_root / "immutable_model_backing" / f"rank_{rank:03d}_generator.jsonl"
    )
    for path in (phase_file, activation_file, block_file, immutable_file):
        if not path.is_file():
            raise RuntimeError(f"missing rank-{rank} evidence: {path}")

    phases = _rows(phase_file)
    pre_backward = _last_phase(phases, "semantic_generator_pre_backward")
    step_start = _last_phase(phases, "training_step_start")
    generator_loads = [
        row
        for row in phases
        if row.get("phase") == "model_swap_end"
        and row.get("incoming_component") == "generator"
        and row.get("outgoing_component") == "score"
    ]
    if not generator_loads:
        raise RuntimeError(f"missing initial Generator load boundary for rank {rank}")
    initial_generator_loaded = generator_loads[0]
    cuda = pre_backward.get("cuda") or {}
    allocated = int(cuda.get("allocated_bytes", 0))
    reserved = int(cuda.get("reserved_bytes", 0))
    inactive = int(cuda.get("inactive_split_bytes", 0))

    activation_rows = _rows(activation_file)
    scope_ends = [row for row in activation_rows if row.get("event") == "scope_end"]
    if not scope_ends:
        raise RuntimeError(f"missing activation scope_end for rank {rank}")
    # Scope stats are cumulative.  Summing scope_end records would overcount.
    activation = dict(scope_ends[-1].get("stats") or {})
    # The runtime's cumulative counter is per saved-tensor pack. Aliased views
    # can name the same physical storage more than once, so it is diagnostic
    # traffic/pack volume rather than a physical-live-byte estimate. Deduplicate
    # the recorded storage identity before comparing against a weight block.
    kept_pack_bytes = int(activation.get("kept_below_threshold_storage_bytes", 0))
    unique_kept: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in activation_rows:
        if (
            row.get("event") == "pack"
            and row.get("policy") == "KEEP_BELOW_THRESHOLD"
            and str(row.get("device", "")).startswith("cuda:")
        ):
            storage_id = tuple(row.get("storage_id") or ())
            if storage_id:
                unique_kept[storage_id] = {
                    "storage_bytes": int(row.get("storage_bytes", 0)),
                    "shape": list(row.get("shape") or []),
                    "dtype": row.get("dtype"),
                }
    unique_kept_bytes = sum(
        int(row["storage_bytes"]) for row in unique_kept.values()
    )
    groups: dict[tuple[tuple[int, ...], str], dict[str, Any]] = {}
    for row in unique_kept.values():
        key = (tuple(row["shape"]), str(row["dtype"]))
        group = groups.setdefault(
            key,
            {
                "shape": row["shape"],
                "dtype": row["dtype"],
                "unique_storage_count": 0,
                "unique_storage_bytes": 0,
            },
        )
        group["unique_storage_count"] += 1
        group["unique_storage_bytes"] += int(row["storage_bytes"])
    top_groups = sorted(
        groups.values(),
        key=lambda row: (-int(row["unique_storage_bytes"]), str(row["shape"])),
    )[:10]

    block_rows = _rows(block_file)
    installed = next(
        (row for row in block_rows if row.get("event") == "streamer_installed"),
        None,
    )
    if installed is None:
        raise RuntimeError(f"missing streamer_installed for rank {rank}")
    materializations = [
        row for row in block_rows if row.get("event") == "materialize_block"
    ]
    unique_block_bytes: dict[int, int] = {}
    for row in materializations:
        unique_block_bytes[int(row["block_index"])] = int(row["bytes"])
    streamed_capacity = sum(unique_block_bytes.values())
    block_sizes = sorted(unique_block_bytes.values())
    median_block = int(statistics.median(block_sizes)) if block_sizes else 0
    allocator_slack = max(0, reserved - allocated)

    immutable_rows = _rows(immutable_file)
    immutable_install = next(
        (
            row
            for row in immutable_rows
            if row.get("event") == "immutable_model_backing_installed"
        ),
        None,
    )
    if immutable_install is None:
        raise RuntimeError(f"missing immutable backing install for rank {rank}")
    immutable_stats = dict(immutable_install.get("stats") or {})
    pre_backward_time = float(pre_backward.get("time_unix", float("inf")))
    block_events_before = [
        row
        for row in block_rows
        if row.get("event") in {"materialize_block", "release_block"}
        and float(row.get("time_unix", 0.0)) <= pre_backward_time
    ]
    streamed_resident = (
        int(block_events_before[-1].get("resident_bytes", 0))
        if block_events_before
        else 0
    )
    shared_immutable = int(immutable_stats.get("shared_immutable_bytes", 0))
    private_parameters = int(immutable_stats.get("private_parameter_bytes", 0))
    known_model_resident = shared_immutable + private_parameters + streamed_resident
    dynamic_minus_model = max(0, allocated - known_model_resident)
    residual_after_kept = max(0, dynamic_minus_model - unique_kept_bytes)
    runtime_floor = int((step_start.get("cuda") or {}).get("allocated_bytes", 0))
    generator_loaded_allocated = int(
        (initial_generator_loaded.get("cuda") or {}).get("allocated_bytes", 0)
    )
    forward_incremental = max(0, allocated - generator_loaded_allocated)
    forward_residual_after_kept = max(0, forward_incremental - unique_kept_bytes)

    return {
        "rank": rank,
        "pre_backward": {
            "allocated_bytes": allocated,
            "reserved_bytes": reserved,
            "allocator_reserved_unallocated_bytes": allocator_slack,
            "active_bytes": int(cuda.get("active_bytes", 0)),
            "inactive_split_bytes": inactive,
            "driver_free_bytes": int(cuda.get("driver_free_bytes", 0)),
            "num_alloc_retries": int(cuda.get("num_alloc_retries", 0)),
            "num_ooms": int(cuda.get("num_ooms", 0)),
        },
        "generator_saved_activation": {
            "scope_count": int(activation.get("scope_count", 0)),
            "pack_count": int(activation.get("pack_count", 0)),
            "kept_below_threshold_count": int(
                activation.get("kept_below_threshold_count", 0)
            ),
            "kept_below_threshold_pack_storage_bytes": kept_pack_bytes,
            "unique_kept_below_threshold_storage_count": len(unique_kept),
            "unique_kept_below_threshold_storage_bytes": unique_kept_bytes,
            "top_unique_kept_groups": top_groups,
            "alias_pack_amplification": (
                kept_pack_bytes / unique_kept_bytes if unique_kept_bytes else None
            ),
            "offloaded_storage_bytes": int(
                activation.get("offloaded_storage_bytes", 0)
            ),
            "peak_live_cpu_bytes": int(activation.get("peak_live_cpu_bytes", 0)),
            "parameter_or_buffer_exclusions": int(
                activation.get("kept_parameter_or_buffer_count", 0)
            ),
        },
        "generator_frozen_blocks": {
            "streamed_block_indices": list(installed.get("streamed_block_indices", [])),
            "streamed_block_count": int(installed.get("streamed_block_count", 0)),
            "unique_streamed_capacity_bytes": streamed_capacity,
            "median_block_bytes": median_block,
            "materialization_count": len(materializations),
            "issued_materialization_bytes": sum(
                int(row.get("bytes", 0)) for row in materializations
            ),
            "materialization_wall_sum_seconds": sum(
                float(row.get("elapsed_seconds", 0.0)) for row in materializations
            ),
            "last_materialization_generation": (
                materializations[-1].get("generation") if materializations else None
            ),
            "last_materialized_block": (
                materializations[-1].get("block_index") if materializations else None
            ),
        },
        "generator_model_residency": {
            "shared_immutable_bytes": shared_immutable,
            "private_parameter_bytes": private_parameters,
            "externally_managed_parameter_bytes": int(
                immutable_stats.get("externally_managed_parameter_bytes", 0)
            ),
            "streamed_resident_at_pre_backward_bytes": streamed_resident,
            "known_model_resident_at_pre_backward_bytes": known_model_resident,
            "allocated_minus_known_model_bytes": dynamic_minus_model,
            "unclassified_after_unique_kept_activation_bytes": residual_after_kept,
        },
        "phase_delta_attribution": {
            "persistent_runtime_floor_before_step_bytes": runtime_floor,
            "initial_generator_loaded_allocated_bytes": generator_loaded_allocated,
            "forward_incremental_to_pre_backward_bytes": forward_incremental,
            "forward_unclassified_after_unique_kept_activation_bytes": (
                forward_residual_after_kept
            ),
            "note": (
                "Phase deltas are simultaneous allocator observations and avoid "
                "adding peaks. The persistent floor is not classified as redundant."
            ),
        },
        "value_comparison": {
            "unique_kept_activation_equivalent_blocks": (
                unique_kept_bytes / median_block if median_block else None
            ),
            "allocator_slack_equivalent_blocks": (
                allocator_slack / median_block if median_block else None
            ),
            "lower_reuse_candidate": "kept_saved_activation_below_32_mib",
            "classification": (
                "candidate_only_not_proven_releasable_at_peak"
                if unique_kept_bytes
                else "no_candidate_observed"
            ),
        },
        "unclassified": {
            "note": (
                "Model residency and unique kept saved-activation storage are "
                "subtracted only at the pre-backward boundary. Remaining bytes can "
                "include buffers, checkpoint boundaries, outputs, temporaries, "
                "workspaces, continuous state, or accounting uncertainty. No "
                "residual is labelled redundant."
            )
        },
    }


def summarize(run_root: Path, expected_world_size: int) -> dict[str, Any]:
    phase_files = sorted((run_root / "phase_trace").glob("rank_*.jsonl"))
    ranks = [_rank_from_name(path) for path in phase_files]
    expected = list(range(expected_world_size))
    if ranks != expected:
        raise RuntimeError(f"rank evidence mismatch: expected {expected}, observed {ranks}")
    per_rank = [summarize_rank(run_root, rank) for rank in ranks]

    def median(path: tuple[str, str]) -> float:
        values = [int(row[path[0]][path[1]]) for row in per_rank]
        return float(statistics.median(values))

    return {
        "schema_version": SCHEMA,
        "run_root": str(run_root),
        "world_size": expected_world_size,
        "evidence_scope": "existing_low_overhead_events_offline_analysis",
        "per_rank": per_rank,
        "aggregate": {
            "median_pre_backward_allocated_bytes": median(
                ("pre_backward", "allocated_bytes")
            ),
            "median_pre_backward_reserved_bytes": median(
                ("pre_backward", "reserved_bytes")
            ),
            "median_pre_backward_driver_free_bytes": median(
                ("pre_backward", "driver_free_bytes")
            ),
            "median_kept_saved_activation_pack_bytes": median(
                (
                    "generator_saved_activation",
                    "kept_below_threshold_pack_storage_bytes",
                )
            ),
            "median_unique_kept_saved_activation_bytes": median(
                (
                    "generator_saved_activation",
                    "unique_kept_below_threshold_storage_bytes",
                )
            ),
            "median_offloaded_saved_activation_bytes": median(
                ("generator_saved_activation", "offloaded_storage_bytes")
            ),
            "median_streamed_block_capacity_bytes": median(
                ("generator_frozen_blocks", "unique_streamed_capacity_bytes")
            ),
            "median_dynamic_minus_known_model_bytes": median(
                ("generator_model_residency", "allocated_minus_known_model_bytes")
            ),
            "median_unclassified_after_unique_kept_activation_bytes": median(
                (
                    "generator_model_residency",
                    "unclassified_after_unique_kept_activation_bytes",
                )
            ),
            "median_persistent_runtime_floor_bytes": median(
                ("phase_delta_attribution", "persistent_runtime_floor_before_step_bytes")
            ),
            "median_forward_incremental_to_pre_backward_bytes": median(
                ("phase_delta_attribution", "forward_incremental_to_pre_backward_bytes")
            ),
            "median_forward_unclassified_after_unique_kept_activation_bytes": median(
                (
                    "phase_delta_attribution",
                    "forward_unclassified_after_unique_kept_activation_bytes",
                )
            ),
            "interpretation": (
                "Unique saved-activation storages retained only because each storage "
                "is below the 32-MiB coarse threshold are a lower-reuse candidate than "
                "repeatedly materialized frozen blocks. The separate pack-byte counter "
                "includes aliases and must not be treated as physical GPU capacity. "
                "Even the deduplicated volume is not proof that all bytes are "
                "simultaneously releasable at the OOM peak."
            ),
        },
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--run-root", type=Path, required=True)
    result.add_argument("--world-size", type=int, default=16)
    result.add_argument("--output", type=Path)
    return result


def main() -> int:
    args = parser().parse_args()
    payload = summarize(args.run_root.resolve(), args.world_size)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
