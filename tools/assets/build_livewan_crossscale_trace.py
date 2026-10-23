#!/usr/bin/env python3
"""Build the immutable rank-qualified trace shared by baseline and pipeline.

The trace changes only RNG *organization*: each logical sample owns independent
operation streams so the same values are consumed after rank/pool remapping.
The distributions and LiveWan source operations remain unchanged.  This is a
systems-only benchmark trace, not a continuation of an upstream training RNG
trajectory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
from typing import Any


LIVEWAN_COMMIT = "1eabf43c777bc5b2816926ca998cd4e4c2dfa3d5"
WAN21_COMMIT = "9737cba9c1c3c4d04b33fcad41c111989865d315"
WORLD_SIZE = 16
FULL5_STEPS = 5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def derived_seed(base_seed: int, operation_key: str, stream: str) -> int:
    payload = f"{base_seed}/{operation_key}/{stream}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--data", required=True, type=Path)
    value.add_argument("--prompts", required=True, type=Path)
    value.add_argument("--output", required=True, type=Path)
    value.add_argument("--seed", type=int, default=20260830)
    value.add_argument("--rope-gap-max", type=int, default=240)
    value.add_argument("--rope-gap-probability", type=float, default=0.7)
    value.add_argument("--hash-latents", choices=("all", "none"), default="all")
    return value


def main() -> int:
    args = parser().parse_args()
    files = sorted(args.data.expanduser().resolve().glob("*.pt"))
    required_samples = WORLD_SIZE * FULL5_STEPS
    if len(files) < required_samples:
        raise SystemExit(
            f"formal Full5 needs {required_samples} distinct latent samples, "
            f"found {len(files)}"
        )
    prompts = args.prompts.expanduser().resolve()
    if not prompts.is_file():
        raise SystemExit(f"prompt bundle does not exist: {prompts}")
    if not (0.0 <= args.rope_gap_probability <= 1.0):
        raise SystemExit("rope-gap probability must be in [0,1]")

    # One deterministic permutation provides exactly 80 distinct samples.  We
    # fail instead of cycling a smaller source because repeating an underlying
    # clip would weaken the rank-qualified global-B16 input contract.
    order = list(range(len(files)))
    random.Random(args.seed).shuffle(order)
    rows: list[dict[str, Any]] = []
    latent_hash_cache: dict[int, str | None] = {}
    worlds = (3, 6, 9)
    for step in range(FULL5_STEPS):
        for slot in range(WORLD_SIZE):
            ordinal = step * WORLD_SIZE + slot
            dataset_index = order[ordinal]
            operation_key = f"step={step}/slot={slot}"
            control_rng = random.Random(derived_seed(args.seed, operation_key, "control"))
            world_frames = worlds[control_rng.randrange(len(worlds))]
            rope_gap = 0
            if args.rope_gap_max and control_rng.random() < args.rope_gap_probability:
                rope_gap = control_rng.randint(1, args.rope_gap_max)
            path = files[dataset_index]
            if dataset_index not in latent_hash_cache:
                latent_hash_cache[dataset_index] = (
                    sha256_file(path) if args.hash_latents == "all" else None
                )
            rows.append(
                {
                    "logical_step": step,
                    "logical_slot": slot,
                    "dataset_index": dataset_index,
                    "sample_path": str(path),
                    "sample_sha256": latent_hash_cache[dataset_index],
                    "operation_key": operation_key,
                    "world_frames": world_frames,
                    "rope_gap": rope_gap,
                    "seeds": {
                        name: derived_seed(args.seed, operation_key, name)
                        for name in (
                            "rollout",
                            "generator_dmd_t",
                            "generator_dmd_noise",
                            "generator_grad_blocks",
                            "critic_t",
                            "critic_noise",
                        )
                    },
                    "generator_update": step == 0,
                    "critic_update": True,
                }
            )

    trace = {
        "schema_version": 1,
        "kind": "systems_only_operation_keyed_livewan_sfdmd",
        "claim_boundary": (
            "Same source distributions and exact consumption across baseline/pipeline; "
            "not an upstream sequential-RNG continuation trajectory."
        ),
        "livewan_commit": LIVEWAN_COMMIT,
        "wan21_commit": WAN21_COMMIT,
        "seed": args.seed,
        "world_size": WORLD_SIZE,
        "global_batch": WORLD_SIZE,
        "steps": FULL5_STEPS,
        "generator_updates": 1,
        "critic_updates": 5,
        "prompts": {"path": str(prompts), "sha256": sha256_file(prompts)},
        "source_sample_count": len(files),
        "unique_samples_consumed": len({row["dataset_index"] for row in rows}),
        "rows": rows,
    }
    canonical = json.dumps(trace, sort_keys=True, separators=(",", ":")).encode("utf-8")
    trace["trace_sha256"] = hashlib.sha256(canonical).hexdigest()
    atomic_json(args.output.expanduser().resolve(), trace)
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(args.output.expanduser().resolve()),
                "rows": len(rows),
                "unique_samples": trace["unique_samples_consumed"],
                "trace_sha256": trace["trace_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
