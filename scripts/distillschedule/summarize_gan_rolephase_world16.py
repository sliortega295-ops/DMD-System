#!/usr/bin/env python3
"""Fail-closed evidence summarizer for the real GAN role-phase Full5 run."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any


RANKS = tuple(range(16))
G_A = tuple(range(8))
S_A = tuple(range(8, 16))
G_B = tuple(range(4))
C_B = tuple(range(4, 16))
PHASE_B_MEMORY_POLICIES = (0, 4, 10)
TAIL10_BLOCK_INDICES = list(range(30, 40))


class EvidenceError(RuntimeError):
    pass


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise EvidenceError(f"{path}: expected JSON object")
    return value


def strict_kv(path: Path) -> dict[str, str]:
    value: dict[str, str] = {}
    for line_no, raw in enumerate(path.read_text(errors="replace").splitlines(), 1):
        if not raw.strip():
            continue
        if "=" not in raw:
            raise EvidenceError(f"{path}:{line_no}: expected key=value")
        key, item = raw.split("=", 1)
        if not key or key.strip() != key or key in value:
            raise EvidenceError(f"{path}:{line_no}: invalid/duplicate key")
        value[key] = item
    return value


def memory_events(text: str) -> dict[str, int]:
    tokens = text.replace(";", " ").split()
    return {
        tokens[index]: int(tokens[index + 1])
        for index in range(0, len(tokens) - 1, 2)
        if tokens[index + 1].lstrip("-").isdigit()
    }


def resource_peak(path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        raise EvidenceError(f"missing resource monitor: {path}")
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                gpu = [int(item) for item in row["gpu_used_mib"].split(":")]
                rows.append(
                    {
                        "cgroup_bytes": int(row["cgroup_bytes"]),
                        # awk may render large byte counts in scientific
                        # notation unless given an explicit printf format.
                        # Accept that valid numeric spelling for preserved
                        # runs; future launchers emit integral text.
                        "host_used_bytes": int(float(row["host_used_bytes"])),
                        "gpu_used_mib": gpu,
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
    if not rows:
        raise EvidenceError(f"resource monitor has no valid rows: {path}")
    return {
        "samples": len(rows),
        "cgroup_peak_bytes": max(item["cgroup_bytes"] for item in rows),
        "host_used_peak_bytes": max(item["host_used_bytes"] for item in rows),
        "per_gpu_peak_mib": [
            max(item["gpu_used_mib"][index] for item in rows)
            for index in range(8)
        ],
    }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def validate_phase_b_lifecycle(
    *, rank: int, value: dict[str, Any], streamed_blocks: int
) -> None:
    """Fail closed on the preregistered DP12 tail10+A128 lifecycle receipt."""

    activation = value.get("phase_b_activation_offload", {})
    _require(
        int(activation.get("live_cpu_bytes", -1)) == 0,
        f"rank{rank}: Phase-B activation leak",
    )
    if streamed_blocks != 10:
        return

    _require(
        int(activation.get("scope_count", -1)) == 10,
        f"rank{rank}: tail10+A128 must cover exactly ten Score forwards",
    )
    _require(
        int(activation.get("cpu_copy_count", 0)) > 0,
        f"rank{rank}: tail10+A128 selected no saved tensors",
    )
    residency = value.get("phase_b_parameter_residency", {})
    _require(
        residency.get("parameter_policy")
        == "SYNCHRONOUS_FROZEN_BLOCK_STREAMING",
        f"rank{rank}: tail10 streamer receipt missing",
    )
    _require(
        residency.get("host_policy") == "HOST_SHARED_IMMUTABLE",
        f"rank{rank}: tail10 host ownership drift",
    )
    _require(
        residency.get("window_size") == 1,
        f"rank{rank}: tail10 streaming window drift",
    )
    _require(
        residency.get("streamed_block_indices") == TAIL10_BLOCK_INDICES,
        f"rank{rank}: tail10 block set drift",
    )
    _require(
        residency.get("idle_at_phase_b_boundary") is True,
        f"rank{rank}: tail10 streamer not idle at Phase-B boundary",
    )
    stats = residency.get("stats", {})
    materialize_count = int(stats.get("materialize_count", 0))
    release_count = int(stats.get("release_count", -1))
    materialized_bytes = int(stats.get("materialized_bytes", 0))
    released_bytes = int(stats.get("released_bytes", -1))
    _require(
        int(stats.get("streamed_block_count", -1)) == 10,
        f"rank{rank}: tail10 streamer count drift",
    )
    _require(materialize_count > 0, f"rank{rank}: tail10 never materialized")
    _require(
        (release_count, released_bytes) == (materialize_count, materialized_bytes),
        f"rank{rank}: tail10 materialize/release reconciliation failed",
    )
    _require(
        int(stats.get("resident_bytes", -1)) == 0,
        f"rank{rank}: tail10 resident bytes leaked",
    )


def validate_only(ranks: dict[int, dict[str, Any]]) -> dict[str, Any]:
    for rank, value in ranks.items():
        _require(value.get("validation_only") is True, f"rank{rank}: not validation-only")
        expected = {
            "world": list(range(16)),
            "generator": list(range(4)),
            "critic": list(range(4, 16)),
            "singleton": None,
        }
        _require(value.get("phase_b_topology") == expected, f"rank{rank}: topology drift")
        _require(
            value.get("phase_b_communicators") == {
                "control": "GLOBAL_WORLD16_GLOO",
                "generator": list(range(4)),
                "critic": list(range(4, 16)),
                "p2p": "COLLECTIVELY_INITIALIZED_DEFAULT_WORLD16_NCCL",
                "pair_groups_created": 0,
                "singleton_groups_created": 0,
                "model_parallel_group": None,
            },
            f"rank{rank}: communicator inventory drift",
        )
        reverse = value.get("reverse_phase_a", {})
        _require(
            (reverse.get("world_size"), reverse.get("rank"), reverse.get("expected_local_rank"))
            == (8, rank % 8, rank % 8),
            f"rank{rank}: reverse Phase-A world drift",
        )
        _require(
            isinstance(reverse.get("master_port"), int),
            f"rank{rank}: reverse Phase-A rendezvous was not recorded",
        )
    return {"topology_lifecycle": "PASS"}


def validate_full(ranks: dict[int, dict[str, Any]], *, timing: bool) -> dict[str, Any]:
    phase_a_x0: dict[int, tuple[str, str, str]] = {}
    phase_a_grad: dict[int, tuple[str, str]] = {}
    phase_b_g: dict[tuple[int, int], tuple[str, str, str]] = {}
    phase_b_c: dict[tuple[int, int], tuple[str, str, str]] = {}
    for rank, value in ranks.items():
        _require(value.get("timing_mode") is timing, f"rank{rank}: timing mode drift")
        _require(value.get("final_versions") == {
            "generator": 201,
            "critic": 1005,
            "ema": 801,
            "next_boundary": "phase_a_ready_step1005_not_executed",
        }, f"rank{rank}: final versions drift")
        updates = value.get("updates")
        expected_updates = {
            "generator": 1 if rank in G_A else 0,
            "critic": 5 if rank in C_B else 0,
            "ema": 1 if rank == 0 else 0,
        }
        _require(updates == expected_updates, f"rank{rank}: update count drift {updates}")
        streamed_blocks = value.get("phase_b_streamed_blocks")
        _require(
            streamed_blocks in PHASE_B_MEMORY_POLICIES,
            f"rank{rank}: invalid memory policy",
        )

        phase_a = value.get("phase_a", {})
        samples_a = phase_a.get("samples", [])
        _require(len(samples_a) == 2, f"rank{rank}: Phase-A sample count")
        expected_sources = [rank % 8, rank % 8 + 8]
        _require([item.get("source_id") for item in samples_a] == expected_sources,
                 f"rank{rank}: Phase-A source mapping")
        if rank in G_A:
            _require(phase_a.get("role") == "generator", f"rank{rank}: Phase-A role")
            _require(phase_a.get("accumulation_boundaries") == 1, f"rank{rank}: G commit")
            for item in samples_a:
                source = int(item["source_id"])
                phase_a_x0[source] = (
                    str(item["x0_checksum_kind"]), str(item["x0_checksum"]),
                    str(item["trace_identity_sha256"]),
                )
                phase_a_grad[source] = (
                    str(item["gradient_checksum_kind"]), str(item["gradient_checksum"])
                )
        else:
            _require(phase_a.get("role") == "score_gradient_service", f"rank{rank}: Score role")
            _require(value.get("score_forward_calls_phase_a") == 8,
                     f"rank{rank}: Phase-A Score inventory")
            for item in samples_a:
                source = int(item["source_id"])
                expected = (
                    str(item["x0_checksum_kind"]), str(item["x0_checksum"]),
                    str(item["trace_identity_sha256"]),
                )
                _require(phase_a_x0.get(source) == expected,
                         f"source{source}: Phase-A x0/trace mismatch")
                expected_grad = (
                    str(item["gradient_checksum_kind"]), str(item["gradient_checksum"])
                )
                _require(phase_a_grad.get(source) == expected_grad,
                         f"source{source}: Phase-A gradient mismatch")

        phase_b = value.get("phase_b", {})
        samples_b = phase_b.get("samples", [])
        if rank in G_B:
            _require(phase_b.get("role") == "generator", f"rank{rank}: Phase-B G role")
            _require(len(samples_b) == 15, f"rank{rank}: Phase-B produced count")
            queue = phase_b.get("queue", {})
            _require(
                (queue.get("produced_global_batches"), queue.get("committed_global_batches"),
                 queue.get("sent_payloads"), queue.get("outstanding_transport"))
                == (5, 5, 15, 0),
                f"rank{rank}: Phase-B producer queue",
            )
            _require(int(queue.get("produced_minus_committed_max", 99)) <= 2,
                     f"rank{rank}: Q2 exceeded")
            for item in samples_b:
                key = (int(item["logical_step"]), int(item["logical_slot"]))
                phase_b_g[key] = (
                    str(item["x0_checksum"]), str(item["trace_identity_sha256"]),
                    str(item["trace_hash"]),
                )
            ema = phase_b.get("ema", [])
            _require(len(ema) == (1 if rank == 0 else 0), f"rank{rank}: EMA count")
            if rank == 0:
                _require((ema[0].get("before"), ema[0].get("after")) == (800, 801),
                         "rank0: EMA recurrence")
        elif rank in C_B:
            _require(phase_b.get("role") == "critic", f"rank{rank}: Phase-B C role")
            _require(len(samples_b) == 5, f"rank{rank}: Phase-B consumed count")
            _require(value.get("score_forward_calls_phase_b") == 10,
                     f"rank{rank}: Phase-B Score inventory")
            _require(phase_b.get("queue") == {
                "consumer_slot": "EMPTY", "outstanding_transport": 0
            }, f"rank{rank}: consumer queue")
            expected_slot = rank - 4
            for update, item in enumerate(samples_b):
                _require(
                    (item.get("logical_step"), item.get("logical_slot"),
                     item.get("critic_before"), item.get("critic_after"),
                     item.get("optimizer_step_before"), item.get("optimizer_step_after"),
                     item.get("physical_critic_loss_calls"))
                    == (1000 + update, expected_slot, 1000 + update, 1001 + update,
                        10 + update, 11 + update, 1),
                    f"rank{rank}: Critic update {update} drift",
                )
                key = (1000 + update, expected_slot)
                phase_b_c[key] = (
                    str(item["x0_checksum"]), str(item["trace_identity_sha256"]),
                    str(item["trace_hash"]),
                )

        activation_a = value.get("phase_a_activation_offload", {})
        _require(int(activation_a.get("live_cpu_bytes", 0)) == 0,
                 f"rank{rank}: Phase-A activation leak")
        if rank in C_B:
            validate_phase_b_lifecycle(
                rank=rank,
                value=value,
                streamed_blocks=int(streamed_blocks),
            )

    _require(set(phase_a_x0) == set(range(16)), "Phase-A x0 coverage")
    _require(set(phase_a_grad) == set(range(16)), "Phase-A gradient coverage")
    expected_b = {(step, slot) for step in range(1000, 1005) for slot in range(12)}
    _require(set(phase_b_g) == expected_b, "Phase-B producer coverage")
    _require(set(phase_b_c) == expected_b, "Phase-B consumer coverage")
    _require(phase_b_g == phase_b_c, "Phase-B P2P x0/trace mismatch")
    starts = [int(value["timing_start_unix_ns"]) for value in ranks.values()]
    stops = [int(value["timing_stop_unix_ns"]) for value in ranks.values()]
    return {
        "phase_a_exact_payloads": len(phase_a_x0),
        "phase_b_exact_payloads": len(phase_b_g),
        "full5_global_seconds": (max(stops) - min(starts)) / 1e9,
        "max_rank_full5_seconds": max(float(value["full5_seconds"]) for value in ranks.values()),
        "phase_max_seconds": {
            "phase_a": max(float(value["phase_a_wall_seconds"]) for value in ranks.values()),
            "handoff_to_b": max(float(value["handoff_to_b"]["wall_seconds"]) for value in ranks.values()),
            "phase_b": max(float(value["phase_b_wall_seconds"]) for value in ranks.values()),
            "handoff_to_a": max(float(value["handoff_to_a"]["wall_seconds"]) for value in ranks.values()),
        },
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--run-root", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    return value


def main() -> int:
    args = parser().parse_args()
    result: dict[str, Any] = {
        "schema_version": "distillschedule.gan_rolephase_world16_summary.v1",
        "run_root": str(args.run_root.resolve()),
        "status": "FAIL",
        "errors": [],
        "claimable_correctness": False,
        "claimable_timing": False,
    }
    try:
        manifests = {
            node: strict_kv(args.run_root / "launcher" / f"node_{node}" / "launch_manifest.txt")
            for node in (0, 1)
        }
        _require(manifests[0]["run_id"] == manifests[1]["run_id"], "run id mismatch")
        for node, manifest in manifests.items():
            _require(manifest.get("final_return_code") == "0", f"node{node}: launcher failed")
            before = memory_events(manifest.get("memory_events_before", ""))
            after = memory_events(manifest.get("memory_events_after", ""))
            _require(after.get("oom", 0) == before.get("oom", 0), f"node{node}: cgroup OOM")
            _require(after.get("oom_kill", 0) == before.get("oom_kill", 0), f"node{node}: cgroup OOM kill")
        modes = {value.get("mode") for value in manifests.values()}
        _require(len(modes) == 1, "node mode mismatch")
        mode = modes.pop()
        ranks = {
            rank: load_json(args.run_root / "execution" / "ranks" / f"rank_{rank:02d}.json")
            for rank in RANKS
        }
        for rank, value in ranks.items():
            _require(value.get("rank") == rank and value.get("status") == "PASS",
                     f"rank{rank}: missing/non-PASS receipt")
        common_identity = {
            key: {json.dumps(value.get("identities", {}).get(key), sort_keys=True)
                  for value in ranks.values()}
            for key in ("logical_manifest_hash", "trace_bundle_hash",
                        "trace_source_bundle_hash", "trace_binding_hash",
                        "rank_control_validation")
        }
        _require(all(len(values) == 1 for values in common_identity.values()),
                 "rank identity mismatch")
        if mode == "validate":
            evidence = validate_only(ranks)
        else:
            evidence = validate_full(ranks, timing=mode == "timing")
        policies = {int(value["phase_b_streamed_blocks"]) for value in ranks.values()}
        _require(len(policies) == 1, "rank memory-policy mismatch")
        resources = {
            node: resource_peak(args.run_root / "launcher" / f"node_{node}" / "resource.csv")
            for node in (0, 1)
        }
        result.update(
            status="PASS",
            run_id=manifests[0]["run_id"],
            mode=mode,
            phase_b_streamed_blocks=policies.pop(),
            evidence=evidence,
            resources=resources,
            identities={key: json.loads(next(iter(values))) for key, values in common_identity.items()},
            claimable_correctness=mode == "correctness",
            claimable_timing=mode == "timing",
            claim_boundary=(
                "systems_only_phase_local_B16_B12_pareto;"
                "not_official_checkpoint1000;not_matched_batch_speedup"
            ),
        )
    except BaseException as error:
        result["errors"].append(f"{type(error).__name__}: {error}")
    atomic_json(args.output, result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
