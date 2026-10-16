#!/usr/bin/env python3
"""Fail-closed summary for one matched sequential or C24 trajectory arm."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from distillschedule.evidence.trajectory_coarse import load_trajectory_jsonl


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--run-root", type=Path, required=True)
    value.add_argument("--cycles", type=int, required=True)
    value.add_argument("--expected-head", required=True)
    value.add_argument("--output", type=Path, required=True)
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def launch_manifest(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            rows[key] = value
    return rows


def main() -> int:
    args = parser().parse_args()
    if args.cycles not in {1, 2, 5, 50}:
        raise ValueError("trajectory cycles must be one of 1, 2, 5, or 50")
    errors: list[str] = []
    rank_rows: dict[int, list[Any]] = {}
    final_receipts: dict[int, dict[str, Any]] = {}
    launch_modes: set[str] = set()
    launch_arm_ids: set[str] = set()
    launch_variants: set[str] = set()
    launch_run_ids: set[str] = set()
    launch_heads: set[str] = set()
    launch_seeds: set[int] = set()
    for node in range(2):
        path = args.run_root / "launcher" / f"node_{node}" / "launch_manifest.txt"
        try:
            manifest = launch_manifest(path)
        except OSError as error:
            errors.append(f"node{node}: invalid launch manifest: {error}")
            continue
        if manifest.get("head") != args.expected_head:
            errors.append(f"node{node}: launch HEAD differs from expected HEAD")
        if manifest.get("trajectory_cycles") != str(args.cycles):
            errors.append(f"node{node}: launch trajectory cycle count changed")
        if manifest.get("claim_boundary") != "DIAGNOSTIC_ONLY":
            errors.append(f"node{node}: trajectory claim boundary changed")
        launch_modes.add(str(manifest.get("execution_mode", "")))
        launch_arm_ids.add(str(manifest.get("trajectory_arm_id", "")))
        launch_variants.add(str(manifest.get("pipeline_variant", "")))
        launch_run_ids.add(str(manifest.get("run_id", "")))
        launch_heads.add(str(manifest.get("head", "")))
        try:
            launch_seeds.add(int(manifest.get("trajectory_seed", "-1")))
        except ValueError:
            errors.append(f"node{node}: trajectory seed is invalid")
    if len(launch_modes) != 1 or next(iter(launch_modes), "") not in {
        "baseline",
        "pipeline",
    }:
        errors.append("two-node trajectory execution mode is missing or inconsistent")
    if len(launch_arm_ids) != 1 or not next(iter(launch_arm_ids), ""):
        errors.append("two-node trajectory arm identity is missing or inconsistent")
    execution_mode = next(iter(launch_modes), "")
    arm_id = next(iter(launch_arm_ids), "")
    pipeline_variant = next(iter(launch_variants), "")
    run_id = next(iter(launch_run_ids), "")
    seed = next(iter(launch_seeds), -1)
    if len(launch_run_ids) != 1 or not run_id:
        errors.append("two-node trajectory run identity is missing or inconsistent")
    if launch_heads != {args.expected_head}:
        errors.append("two-node trajectory source HEAD is missing or inconsistent")
    if len(launch_seeds) != 1 or seed < 0:
        errors.append("two-node trajectory seed is missing or inconsistent")
    allowed_variants = (
        {"b1"}
        if execution_mode == "baseline"
        else {
            "c24-continuous-b16-score-modulation",
            "c25-continuous-b16-score-rmsnorm", "c26-continuous-b16-critic-attention-checkpoint", "c27-continuous-b16-pinned-prompt-staging", "c28-continuous-b16-rollout-cast-cache", "c29-continuous-b16-generator-rms-epilogue", "c30-continuous-b16-generator-rms-critic-checkpoint", "c31-continuous-b16-generator-rms-cast-square", "c32-continuous-b16-generator-modulation", "c33-continuous-b16-generator-gate", "c34-continuous-b16-residual-layout", "c35-continuous-b16-score-gate", "c36-continuous-b16-score-layernorm", "c37-continuous-b16-step1-overlap",
            "c24-continuous-b16-score-modulation-b1-attribution", 'c38-continuous-b16-score-fa3', 'c49-continuous-b16-generator-readonly-splitkv', 'c51-continuous-b16-generator-fused-rms', 'c51-critic-fa3-recompute-v1', 'c39-continuous-b16-step0-delivery', 'c40-continuous-b16-score-shared-input', 'c44-continuous-b16-critic-ffn-down-cache',
        }
    )
    if len(launch_variants) != 1 or pipeline_variant not in allowed_variants:
        errors.append("two-node trajectory physical-batch variant is invalid")
    conditional_plan_shas: set[str] = set()
    if pipeline_variant == "c51-critic-fa3-recompute-v1":
        from distillschedule.runtime.livewan_replay_ablation import validate_effective_policy
        from distillschedule.runtime.livewan_crossscale_worker import _effective_plan_sha256
        if args.cycles != 2:
            errors.append("C51 replay ablation requires exactly two cycles")
        for node in range(2):
            try:
                plan = json.loads((args.run_root / f"effective_plan.node{node}.json").read_text())
                recorded = plan.pop("effective_plan_sha256", "")
                if recorded != _effective_plan_sha256(plan):
                    raise RuntimeError("effective replay plan SHA mismatch")
                validate_effective_policy(plan)
                conditional_plan_shas.add(recorded)
            except (OSError, ValueError, RuntimeError) as error:
                errors.append(f"node{node}: {error}")
        if len(conditional_plan_shas) != 1:
            errors.append("two-node effective replay plans differ")
    for rank in range(16):
        path = args.run_root / "trajectory" / f"rank_{rank:03d}.jsonl"
        try:
            rows = load_trajectory_jsonl(path)
        except (FileNotFoundError, ValueError, TypeError) as error:
            errors.append(f"rank{rank}: invalid trajectory JSONL: {error}")
            continue
        if len(rows) != args.cycles:
            errors.append(f"rank{rank}: observed {len(rows)} cycles, expected {args.cycles}")
        if [row.cycle for row in rows] != list(range(1, args.cycles + 1)):
            errors.append(f"rank{rank}: cycle sequence is not contiguous")
        if any(not row.finite or row.runtime_status != "PASS" for row in rows):
            errors.append(f"rank{rank}: non-finite or failed trajectory row")
        rank_rows[rank] = rows

        receipt_path = args.run_root / "receipts" / f"rank_{rank:03d}.json"
        try:
            final = json.loads(receipt_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError) as error:
            errors.append(f"rank{rank}: invalid final receipt: {error}")
            continue
        if final.get("status") != "PASS" or final.get("run_kind") != "trajectory":
            errors.append(f"rank{rank}: final trajectory receipt is not PASS")
        if pipeline_variant == "c40-continuous-b16-score-shared-input":
            from distillschedule.runtime.livewan_lora_shared_input import runtime_receipt_errors as shared_errors
            errors.extend(f"rank{rank}: {error}" for error in shared_errors(
                dict(final.get("critic_lora_kernel", {})), rank=rank))
        if pipeline_variant == "c39-continuous-b16-step0-delivery":
            from distillschedule.runtime.livewan_step0_delivery import (
                runtime_receipt_errors as step0_errors,
                transport_receipt_errors as step0_transport_errors,
            )
            errors.extend(step0_errors(final.get("step0_delivery", {}), rank=rank, cycles=args.cycles))
            errors.extend(step0_transport_errors(
                final.get("generator_runahead_transport", {}), rank=rank, cycles=args.cycles))
        if pipeline_variant in {"c35-continuous-b16-score-gate", "c36-continuous-b16-score-layernorm", "c37-continuous-b16-step1-overlap", 'c38-continuous-b16-score-fa3', 'c49-continuous-b16-generator-readonly-splitkv', 'c51-continuous-b16-generator-fused-rms', 'c51-critic-fa3-recompute-v1', 'c39-continuous-b16-step0-delivery', 'c40-continuous-b16-score-shared-input', 'c44-continuous-b16-critic-ffn-down-cache'}:
            from distillschedule.runtime.livewan_score_gate import runtime_receipt_errors as score_gate_errors
            errors.extend(f"rank{rank}: {error}" for error in score_gate_errors(
                dict(final.get("score_residual_gate", {})), rank=rank,
                inventory=dict(final.get("physical_call_inventory", {}))))
        if pipeline_variant in {"c36-continuous-b16-score-layernorm", "c37-continuous-b16-step1-overlap", 'c38-continuous-b16-score-fa3', 'c49-continuous-b16-generator-readonly-splitkv', 'c51-continuous-b16-generator-fused-rms', 'c51-critic-fa3-recompute-v1', 'c39-continuous-b16-step0-delivery', 'c40-continuous-b16-score-shared-input', 'c44-continuous-b16-critic-ffn-down-cache'}:
            from distillschedule.runtime.livewan_score_layernorm import runtime_receipt_errors as score_ln_errors
            errors.extend(f"rank{rank}: {error}" for error in score_ln_errors(
                dict(final.get("score_layernorm", {})), rank=rank,
                inventory=dict(final.get("physical_call_inventory", {}))))
        if pipeline_variant == "c37-continuous-b16-step1-overlap":
            from distillschedule.runtime.livewan_step1_overlap import runtime_receipt_errors as step1_errors
            errors.extend(f"rank{rank}: {error}" for error in step1_errors(
                dict(final.get("continuous_pipeline", {})), rank=rank, cycles=args.cycles,
                inventory=dict(final.get("physical_call_inventory", {}))))
        if pipeline_variant in {"c25-continuous-b16-score-rmsnorm", "c26-continuous-b16-critic-attention-checkpoint", "c27-continuous-b16-pinned-prompt-staging", "c28-continuous-b16-rollout-cast-cache", "c29-continuous-b16-generator-rms-epilogue", "c30-continuous-b16-generator-rms-critic-checkpoint", "c31-continuous-b16-generator-rms-cast-square", "c32-continuous-b16-generator-modulation", "c33-continuous-b16-generator-gate", "c34-continuous-b16-residual-layout", "c35-continuous-b16-score-gate", "c36-continuous-b16-score-layernorm", "c37-continuous-b16-step1-overlap", 'c38-continuous-b16-score-fa3', 'c49-continuous-b16-generator-readonly-splitkv', 'c51-continuous-b16-generator-fused-rms', 'c51-critic-fa3-recompute-v1', 'c39-continuous-b16-step0-delivery', 'c40-continuous-b16-score-shared-input', 'c44-continuous-b16-critic-ffn-down-cache'}:
            from distillschedule.runtime.livewan_score_rmsnorm import runtime_receipt_errors

            errors.extend(f"rank{rank}: {error}" for error in runtime_receipt_errors(
                dict(final.get("score_rmsnorm_kernel", {})), rank=rank))
        if pipeline_variant in {"c29-continuous-b16-generator-rms-epilogue", "c30-continuous-b16-generator-rms-critic-checkpoint", "c31-continuous-b16-generator-rms-cast-square", "c32-continuous-b16-generator-modulation", "c33-continuous-b16-generator-gate", "c34-continuous-b16-residual-layout", "c35-continuous-b16-score-gate", "c36-continuous-b16-score-layernorm", "c37-continuous-b16-step1-overlap", 'c38-continuous-b16-score-fa3', 'c49-continuous-b16-generator-readonly-splitkv', 'c51-continuous-b16-generator-fused-rms', 'c51-critic-fa3-recompute-v1', 'c39-continuous-b16-step0-delivery', 'c40-continuous-b16-score-shared-input', 'c44-continuous-b16-critic-ffn-down-cache'}:
            from distillschedule.runtime.livewan_generator_rms_epilogue import runtime_receipt_errors as rms_epilogue_errors

            errors.extend(f"rank{rank}: {error}" for error in rms_epilogue_errors(
                dict(final.get("generator_rms_epilogue", {})), rank=rank, cycles=args.cycles,
                fuse_cast_square=pipeline_variant in {"c31-continuous-b16-generator-rms-cast-square", "c32-continuous-b16-generator-modulation", "c33-continuous-b16-generator-gate", "c34-continuous-b16-residual-layout", "c35-continuous-b16-score-gate", "c36-continuous-b16-score-layernorm", "c37-continuous-b16-step1-overlap", 'c38-continuous-b16-score-fa3', 'c49-continuous-b16-generator-readonly-splitkv', 'c39-continuous-b16-step0-delivery', 'c40-continuous-b16-score-shared-input', 'c44-continuous-b16-critic-ffn-down-cache'}))
        if pipeline_variant in {"c28-continuous-b16-rollout-cast-cache", "c29-continuous-b16-generator-rms-epilogue", "c30-continuous-b16-generator-rms-critic-checkpoint", "c31-continuous-b16-generator-rms-cast-square", "c32-continuous-b16-generator-modulation", "c33-continuous-b16-generator-gate", "c34-continuous-b16-residual-layout", "c35-continuous-b16-score-gate", "c36-continuous-b16-score-layernorm", "c37-continuous-b16-step1-overlap", 'c38-continuous-b16-score-fa3', 'c49-continuous-b16-generator-readonly-splitkv', 'c51-continuous-b16-generator-fused-rms', 'c51-critic-fa3-recompute-v1', 'c39-continuous-b16-step0-delivery', 'c40-continuous-b16-score-shared-input', 'c44-continuous-b16-critic-ffn-down-cache'}:
            from distillschedule.runtime.livewan_rollout_cast_cache import runtime_receipt_errors as cast_errors

            errors.extend(f"rank{rank}: {error}" for error in cast_errors(
                dict(final.get("generator_rollout_cast_cache", {})), rank=rank, cycles=args.cycles))
        if pipeline_variant in {"c32-continuous-b16-generator-modulation", "c33-continuous-b16-generator-gate", "c34-continuous-b16-residual-layout", "c35-continuous-b16-score-gate", "c36-continuous-b16-score-layernorm", "c37-continuous-b16-step1-overlap", 'c38-continuous-b16-score-fa3', 'c49-continuous-b16-generator-readonly-splitkv', 'c51-continuous-b16-generator-fused-rms', 'c51-critic-fa3-recompute-v1', 'c39-continuous-b16-step0-delivery', 'c40-continuous-b16-score-shared-input', 'c44-continuous-b16-critic-ffn-down-cache'}:
            from distillschedule.runtime.livewan_generator_modulation import runtime_receipt_errors as modulation_errors

            errors.extend(f"rank{rank}: {error}" for error in modulation_errors(
                dict(final.get("generator_modulation", {})), rank=rank, cycles=args.cycles))
        if pipeline_variant in {"c33-continuous-b16-generator-gate", "c34-continuous-b16-residual-layout", "c35-continuous-b16-score-gate", "c36-continuous-b16-score-layernorm", "c37-continuous-b16-step1-overlap", 'c38-continuous-b16-score-fa3', 'c49-continuous-b16-generator-readonly-splitkv', 'c51-continuous-b16-generator-fused-rms', 'c51-critic-fa3-recompute-v1', 'c39-continuous-b16-step0-delivery', 'c40-continuous-b16-score-shared-input', 'c44-continuous-b16-critic-ffn-down-cache'}:
            from distillschedule.runtime.livewan_generator_gate import runtime_receipt_errors as gate_errors

            errors.extend(f"rank{rank}: {error}" for error in gate_errors(
                dict(final.get("generator_gate", {})), rank=rank, cycles=args.cycles,
                output_layout="row_major" if pipeline_variant in {"c34-continuous-b16-residual-layout", "c35-continuous-b16-score-gate", "c36-continuous-b16-score-layernorm", "c37-continuous-b16-step1-overlap", 'c38-continuous-b16-score-fa3', 'c49-continuous-b16-generator-readonly-splitkv', 'c51-continuous-b16-generator-fused-rms', 'c51-critic-fa3-recompute-v1', 'c39-continuous-b16-step0-delivery', 'c40-continuous-b16-score-shared-input', 'c44-continuous-b16-critic-ffn-down-cache'} else "native"))
        if pipeline_variant == "c27-continuous-b16-pinned-prompt-staging":
            from distillschedule.runtime.livewan_context_staging import runtime_receipt_errors as staging_errors

            errors.extend(f"rank{rank}: {error}" for error in staging_errors(
                dict(final.get("score_prompt_staging", {})), rank=rank, cycles=args.cycles))
        if pipeline_variant in {"c26-continuous-b16-critic-attention-checkpoint", "c30-continuous-b16-generator-rms-critic-checkpoint", "c31-continuous-b16-generator-rms-cast-square", "c32-continuous-b16-generator-modulation", "c33-continuous-b16-generator-gate", "c34-continuous-b16-residual-layout", "c35-continuous-b16-score-gate", "c36-continuous-b16-score-layernorm", "c37-continuous-b16-step1-overlap", 'c38-continuous-b16-score-fa3', 'c49-continuous-b16-generator-readonly-splitkv', 'c51-continuous-b16-generator-fused-rms', 'c51-critic-fa3-recompute-v1', 'c39-continuous-b16-step0-delivery', 'c40-continuous-b16-score-shared-input', 'c44-continuous-b16-critic-ffn-down-cache'}:
            if pipeline_variant in {"c38-continuous-b16-score-fa3", 'c49-continuous-b16-generator-readonly-splitkv', 'c51-continuous-b16-generator-fused-rms', 'c51-critic-fa3-recompute-v1', "c39-continuous-b16-step0-delivery", 'c40-continuous-b16-score-shared-input', 'c44-continuous-b16-critic-ffn-down-cache'}:
                from distillschedule.runtime.livewan_score_fa3 import runtime_receipt_errors as sac_errors, backend_receipt_errors
                errors.extend(f"rank{rank}: {error}" for error in backend_receipt_errors(
                    dict(final.get("score_attention_backend", {})), rank=rank))
            else:
                from distillschedule.runtime.livewan_attention_checkpoint import runtime_receipt_errors as sac_errors

            if pipeline_variant == "c51-critic-fa3-recompute-v1":
                from distillschedule.runtime.livewan_replay_ablation import runtime_receipt_errors as sac_errors
            errors.extend(f"rank{rank}: {error}" for error in sac_errors(
                dict(final.get("critic_attention_checkpoint", {})), rank=rank,
                # C37 has six physical forwards per cycle, not five B2 calls.
                # Its independent per-step census gate still rejects bad totals.
                expected_calls=int(dict(final.get("physical_call_inventory", {}))
                                   .get("critic/grad/checkpoint", -1)) * 30))
        if pipeline_variant in {'c49-continuous-b16-generator-readonly-splitkv', 'c51-continuous-b16-generator-fused-rms', 'c51-critic-fa3-recompute-v1'}:
            from distillschedule.runtime.livewan_generator_splitkv import runtime_receipt_errors as splitkv_errors
            attention = dict(final.get("generator_attention_backend", {}))
            errors.extend(f"rank{rank}: {error}" for error in splitkv_errors(
                dict(attention.get("readonly_splitkv", {})), rank=rank,
                total_calls=int(attention.get("invocation_count", 0))))
        if pipeline_variant == "c44-continuous-b16-critic-ffn-down-cache":
            from distillschedule.runtime.livewan_ffn_down_checkpoint import runtime_receipt_errors as ffn_errors
            errors.extend(f"rank{rank}: {error}" for error in ffn_errors(
                dict(final.get("critic_ffn_down_checkpoint", {})), rank=rank,
                expected_calls=int(dict(final.get("physical_call_inventory", {}))
                                   .get("critic/grad/checkpoint", -1)) * 30))
        if final.get("final_versions") != {
            "generator": args.cycles,
            "critic": 5 * args.cycles,
            "ema": 0,
        }:
            errors.append(f"rank{rank}: final G/C/EMA versions changed")
        expected_commits = (
            (args.cycles, 5 * args.cycles)
            if execution_mode == "baseline"
            else ((args.cycles, 0) if rank < 8 else (0, 5 * args.cycles))
        )
        if (
            int(final.get("physical_generator_commits", -1)),
            int(final.get("physical_critic_commits", -1)),
        ) != expected_commits:
            errors.append(f"rank{rank}: physical commit inventory changed")
        if final.get("trajectory", {}).get("formal_timing_eligible") is not False:
            errors.append(f"rank{rank}: trajectory incorrectly claims formal timing")
        if final.get("trajectory", {}).get("arm_id") != arm_id:
            errors.append(f"rank{rank}: trajectory arm identity changed")
        final_receipts[rank] = final
        if pipeline_variant == "c51-critic-fa3-recompute-v1" and final.get("effective_plan_sha256") not in conditional_plan_shas:
            errors.append(f"rank{rank}: effective replay plan binding changed")

    identities = {
        (
            row.get("trace_sha256"),
            row.get("bootstrap_sha256"),
            row.get("source_receipt_sha256"),
        )
        for row in final_receipts.values()
    }
    if len(identities) != 1 or None in next(iter(identities), (None,)):
        errors.append("rank trace/bootstrap/source-receipt identity differs")
        identity: dict[str, str] | None = None
    else:
        trace_sha256, bootstrap_sha256, source_receipt_sha256 = next(iter(identities))
        identity = {
            "trace_sha256": str(trace_sha256),
            "bootstrap_sha256": str(bootstrap_sha256),
            "source_receipt_sha256": str(source_receipt_sha256),
        }

    expected_anchor_cycles = sorted(
        {cycle for cycle in (1, 10, 25, 50, args.cycles) if cycle <= args.cycles}
    )
    anchors: list[dict[str, Any]] = []
    anchor_digests: dict[int, str] = {}
    for cycle in expected_anchor_cycles:
        path = args.run_root / "trajectory" / "anchors" / f"cycle_{cycle:02d}_global.json"
        try:
            value = json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError) as error:
            errors.append(f"cycle{cycle}: invalid numeric state anchor: {error}")
            continue
        recorded = value.get("anchor_sha256")
        canonical = dict(value)
        canonical.pop("anchor_sha256", None)
        actual = hashlib.sha256(
            json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if recorded != actual:
            errors.append(f"cycle{cycle}: state anchor SHA mismatch")
        components = dict(value.get("components", {}))
        expected_components = {
            "generator_trainable",
            "generator_adam",
            "critic_trainable",
            "critic_adam",
        }
        if set(components) != expected_components:
            errors.append(f"cycle{cycle}: numeric state component inventory changed")
        for component, sketch in components.items():
            if int(sketch.get("coordinate_count", 0)) <= 0:
                errors.append(f"cycle{cycle}: {component} has no numeric coordinates")
            if not sketch.get("coordinates"):
                errors.append(f"cycle{cycle}: {component} coordinate values are missing")
        anchors.append(
            {"cycle": cycle, "path": str(path), "sha256": recorded}
        )
        if isinstance(recorded, str):
            anchor_digests[cycle] = recorded

    if len(rank_rows) == 16:
        reference_contract = rank_rows[0][0].matched_contract
        for cycle in range(1, args.cycles + 1):
            expected_versions = {
                "generator": cycle,
                "critic": 5 * cycle,
                "ema": 0,
            }
            cycle_samples: list[str] = []
            cycle_rng: dict[str, str] = {}
            reference_call_census = rank_rows[0][cycle - 1].call_census
            for rank in range(16):
                row = rank_rows[rank][cycle - 1]
                if row.matched_contract != reference_contract:
                    errors.append(f"rank{rank}/cycle{cycle}: matched contract changed")
                if dict(row.versions) != expected_versions:
                    errors.append(f"rank{rank}/cycle{cycle}: logical versions changed")
                if dict(row.cycle_updates) != {"generator": 1, "critic": 5, "ema": 0}:
                    errors.append(f"rank{rank}/cycle{cycle}: G1/C5 update contract changed")
                if row.variant != arm_id:
                    errors.append(f"rank{rank}/cycle{cycle}: arm identity changed")
                if row.implementation.get("execution_mode") != execution_mode:
                    errors.append(f"rank{rank}/cycle{cycle}: execution mode changed")
                expected_physical = (
                    ("1", "5", "5")
                    if execution_mode == "baseline"
                    else (
                        ("mixed_step1_b1_other_b2", "0" if rank < 8 else "6", "0" if rank < 8 else "5")
                        if pipeline_variant == "c37-continuous-b16-step1-overlap"
                        else (
                        ("1", "0" if rank < 8 else "10", "0" if rank < 8 else "5")
                        if pipeline_variant
                        == "c24-continuous-b16-score-modulation-b1-attribution"
                        else ("2", "0" if rank < 8 else "5", "0" if rank < 8 else "5")
                        )
                    )
                )
                observed_physical = tuple(
                    row.implementation.get(key)
                    for key in (
                        "critic_physical_batch_size",
                        "critic_physical_backward_calls",
                        "critic_optimizer_commits",
                    )
                )
                if observed_physical != expected_physical:
                    errors.append(
                        f"rank{rank}/cycle{cycle}: physical Critic B/call/commit inventory changed"
                    )
                if row.call_census != reference_call_census:
                    errors.append(f"rank{rank}/cycle{cycle}: normalized call census changed")
                cycle_samples.extend(row.sample_identities)
                for key, digest in row.rng_identities.items():
                    previous = cycle_rng.setdefault(key, digest)
                    if previous != digest:
                        errors.append(f"cycle{cycle}: duplicate RNG identity disagrees for {key}")
                expected_state_keys = (
                    {
                        "generator_trainable",
                        "generator_adam",
                        "critic_trainable",
                        "critic_adam",
                        "anchor",
                    }
                    if cycle in expected_anchor_cycles
                    else set()
                )
                if set(row.state_fingerprints) != expected_state_keys:
                    errors.append(
                        f"rank{rank}/cycle{cycle}: state fingerprint inventory changed"
                    )
                if cycle in anchor_digests and row.state_fingerprints.get(
                    "anchor"
                ) != anchor_digests[cycle]:
                    errors.append(f"rank{rank}/cycle{cycle}: anchor digest changed")
            if len(cycle_samples) != 80 or len(set(cycle_samples)) != 80:
                errors.append(
                    f"cycle{cycle}: logical-rank sample bijection is not 80 unique rows"
                )
            if set(cycle_samples) != set(cycle_rng):
                errors.append(f"cycle{cycle}: sample and RNG operation keys differ")

    result = {
        "schema_version": "distillschedule.livewan_trajectory_summary.v1",
        "status": "PASS" if not errors else "FAIL",
        "run_root": str(args.run_root),
        "run_id": run_id,
        "expected_head": args.expected_head,
        "head": args.expected_head,
        "heads": {execution_mode: args.expected_head} if execution_mode else {},
        "execution_mode": execution_mode,
        "arm_id": arm_id,
        "pipeline_variant": pipeline_variant,
        "seed": seed,
        "identity": identity,
        "cycles": args.cycles,
        "rank_cycle_receipts": sum(len(rows) for rows in rank_rows.values()),
        "expected_rank_cycle_receipts": 16 * args.cycles,
        "final_versions": {
            "generator": args.cycles,
            "critic": 5 * args.cycles,
            "ema": 0,
        },
        "sample_contract": (
            "80 logical rank-qualified operation keys per cycle; source samples "
            "reused across cycles with fresh operation-keyed RNG"
        ),
        "numeric_state_anchors": anchors,
        "formal_timing_eligible": False,
        "ordinary_numerical_drift_early_stop": False,
        "errors": errors,
    }
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
