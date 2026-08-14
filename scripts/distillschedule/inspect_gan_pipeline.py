#!/usr/bin/env python3
"""Render the immutable operation/version schedule for the GAN pipeline prototype."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from distillschedule.runtime.gan_pipeline import (
    GanPipelinePlanner,
    VersionState,
    expected_final_versions,
    summarize_operations,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--start-step", type=int, default=1000)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--generator-version", type=int, default=200)
    p.add_argument("--critic-version", type=int, default=1000)
    p.add_argument("--ema-version", type=int, default=800)
    p.add_argument("--output", type=Path)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    initial = VersionState(
        generator=args.generator_version,
        critic=args.critic_version,
        ema=args.ema_version,
    )
    operations = GanPipelinePlanner().compile(
        start_step=args.start_step,
        steps=args.steps,
        initial=initial,
    )
    expected = expected_final_versions(
        start_step=args.start_step,
        steps=args.steps,
        initial=initial,
    )
    payload = {
        "schema_version": "distillschedule.gan_pipeline_schedule.v1",
        "status": "PASS",
        "layout": "persistent_8g_8score",
        "start_step": args.start_step,
        "steps": args.steps,
        "initial_versions": asdict(initial),
        "expected_final_versions": asdict(expected),
        "summary": summarize_operations(operations),
        "operations": [
            {
                "logical_step": op.logical_step,
                "phase": op.phase.value,
                "micro_wave": op.micro_wave,
                "versions_before": asdict(op.versions_before),
                "versions_after": asdict(op.versions_after),
                "queue_must_be_empty": op.queue_must_be_empty,
                "score_trainable": op.score_trainable,
                "optimizer_commit": op.optimizer_commit,
                "overlap_lane": op.overlap_lane,
            }
            for op in operations
        ],
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite {args.output}")
        args.output.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
